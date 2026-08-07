# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm block-FP8 GEMM/residual RMSNorm/GEMM rewrite."""

import operator
from dataclasses import dataclass

import torch
from torch import fx
from torch._ops import OpOverload

import vllm._aiter_ops  # noqa: F401
from vllm.config import VllmConfig
from vllm.utils.torch_utils import direct_register_custom_op

from ..fx_utils import find_getitem_maybe, is_func
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

PREFILL_GEMM_OP = torch.ops.vllm.rocm_aiter_triton_gemm_a8w8_blockscale.default
PREFILL_GEMM_OP_PACKET = torch.ops.vllm.rocm_aiter_triton_gemm_a8w8_blockscale
DECODE_GEMM_OP = torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale.default
DECODE_GEMM_OP_PACKET = torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale
RMSNORM_QUANT_OP = torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant.default
RMSNORM_QUANT_OP_PACKET = torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant
REFERENCE_OP_NAME = "rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference"
PASS_NAME = "rocm_aiter_blockscale_gemm_rmsnorm_gemm_fusion_pass"
QWEN3_EPSILON = 1e-6
QWEN3_GROUP_SIZE = 128


def rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference(
    input: torch.Tensor,
    gemm0_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gemm0_weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    epsilon: float,
    group_size: int,
    gemm1_weight: torch.Tensor,
    gemm1_weight_scale: torch.Tensor,
    retained_gemm0: torch.Tensor,
    gemm0_use_triton: bool,
    gemm0_output_dtype: torch.dtype,
    gemm1_output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the two-GEMM Qwen3-32B block-FP8 candidate.

    This temporary, deliberately simple implementation defines the contract
    for a future fused AITER kernel. It implements the delayed-rounding
    variant of the ordering evaluated by the numerical experiment in
    ``gemm.py``:

      FP32_GEMM1(quantize((GEMM0(input) + residual) * rms_weight))
          * inverse_rms

    In contrast, the original model applies ``inverse_rms`` before
    quantization and GEMM1. Moving that row-wise factor after GEMM1 is
    algebraically valid, but the two paths can round differently in BF16 and
    FP8. Keeping GEMM1's output and the inverse-RMS multiply in FP32 until the
    original BF16 model boundary is part of this contract.
    """
    # The original GEMM0 result is an intentional numerical-check value. It is
    # not used for the fused arithmetic, but accepting it as an input keeps the
    # original GEMM live while this reference recomputes the fused candidate.
    del retained_gemm0

    # GEMM0 produces the BF16 projection that starts the matched region. The
    # target model uses the Triton wrapper for some prefill shapes and the CK
    # wrapper for decode, so the graph match records the selected wrapper.
    gemm0_op = PREFILL_GEMM_OP if gemm0_use_triton else DECODE_GEMM_OP
    gemm0 = gemm0_op(
        input,
        gemm0_weight,
        input_scale,
        gemm0_weight_scale,
        gemm0_output_dtype,
    )

    # Form the updated residual in the model dtype. This intentionally
    # materializes the same BF16 value used by the gist-style transformation;
    # the inverse-RMS reduction itself is evaluated in FP32 below.
    updated_residual = gemm0 + residual

    # RMSNorm's inverse-RMS factor is constant across each row. That is the
    # factor that can move through GEMM1, unlike rms_weight, which varies by
    # hidden dimension and must remain on GEMM1's input side.
    inverse_rms = torch.rsqrt(
        updated_residual.float().pow(2).mean(dim=-1, keepdim=True) + epsilon
    )
    weighted = updated_residual * rms_weight

    # Quantize the unnormalized, weighted activation exactly as the numerical
    # experiment does. Keep this as explicit PyTorch arithmetic: this op is a
    # readable correctness model for a future fused kernel, not a fast path.
    quantized, quantized_scale = _quantize_fp8_tile_activation(weighted, group_size)

    # GEMM1 consumes the newly quantized activation and its original 128x128
    # block-scaled weight. Always use AITER's Triton wrapper here, including
    # for decode graphs that originally selected CK: Triton accumulates in
    # FP32 and can store that accumulator to an FP32 output, while CK currently
    # restricts block-scaled GEMM output to BF16/FP16.
    gemm1 = PREFILL_GEMM_OP(
        quantized,
        gemm1_weight,
        quantized_scale,
        gemm1_weight_scale,
        torch.float32,
    )

    # Apply inverse RMS while the complete GEMM result is still FP32, then
    # round once at the original BF16 model boundary. Return the pre-cast value
    # separately so numerical checks can inspect the implemented arithmetic.
    transformed_gemm1 = gemm1 * inverse_rms
    model_gemm1 = transformed_gemm1.to(gemm1_output_dtype)
    return updated_residual, transformed_gemm1, model_gemm1


def _quantize_fp8_tile_activation(
    input: torch.Tensor,
    tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the gist's 1xN symmetric FP8 activation quantization."""
    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)
    rows, columns = input.shape
    tile_columns = (columns + tile_size - 1) // tile_size

    # Pad only for the per-tile absolute-maximum reduction. The returned FP8
    # activation retains the original shape, so padding never reaches GEMM1.
    padded = torch.nn.functional.pad(
        input.float(), (0, tile_columns * tile_size - columns)
    )
    absolute_maximum = (
        padded.reshape(rows, tile_columns, tile_size)
        .abs()
        .amax(dim=-1)
        .clamp(min=1e-12)
    )

    # The GEMM consumes inverse scales, one FP32 value per 1xtile_size block.
    # Expand the forward scale only long enough to quantize each input value.
    scale = (fp8_info.max / absolute_maximum).repeat_interleave(tile_size, dim=1)[
        :, :columns
    ]
    quantized = (input.float() * scale).clamp(fp8_info.min, fp8_info.max).to(fp8_dtype)
    inverse_scale = (absolute_maximum / fp8_info.max).to(torch.float32)
    return quantized, inverse_scale


def rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference_fake(
    input: torch.Tensor,
    gemm0_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gemm0_weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    epsilon: float,
    group_size: int,
    gemm1_weight: torch.Tensor,
    gemm1_weight_scale: torch.Tensor,
    retained_gemm0: torch.Tensor,
    gemm0_use_triton: bool,
    gemm0_output_dtype: torch.dtype,
    gemm1_output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del gemm0_weight, input_scale, gemm0_weight_scale, rms_weight, retained_gemm0
    del epsilon, group_size, gemm1_weight_scale, gemm0_use_triton
    del gemm0_output_dtype
    return (
        torch.empty_like(residual),
        torch.empty(
            (input.shape[0], gemm1_weight.shape[0]),
            dtype=torch.float32,
            device=input.device,
        ),
        torch.empty(
            (input.shape[0], gemm1_weight.shape[0]),
            dtype=gemm1_output_dtype,
            device=input.device,
        ),
    )


direct_register_custom_op(
    op_name=REFERENCE_OP_NAME,
    op_func=rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference,
    fake_impl=rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference_fake,
)
REFERENCE_OP: OpOverload = getattr(torch.ops.vllm, REFERENCE_OP_NAME).default


@dataclass(frozen=True)
class _Qwen3BlockFP8FusionMatch:
    gemm0: fx.Node
    rmsnorm_quant: fx.Node
    quantized: fx.Node
    residual: fx.Node
    quantized_scale: fx.Node
    gemm1: fx.Node


class RocmAiterBlockscaleGemmRMSNormGemmFusionPass(VllmInductorPass):
    """Replace Qwen3-32B's two-GEMM block-FP8 RMSNorm path.

    The replacement owns both the updated residual and GEMM1's output. The
    original GEMM0 remains untouched for independent numerical-check users;
    the now-unobserved combined residual-RMSNorm/quant node is removed.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.pass_name = PASS_NAME
        self.matched_count = 0

    def uuid(self) -> str:
        return self.hash_source(
            self,
            f"{REFERENCE_OP_NAME}:{QWEN3_EPSILON}:{QWEN3_GROUP_SIZE}",
            rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference,
            rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference_fake,
            _match_qwen3_block_fp8_chain,
            _replace_outputs,
            _is_qwen3_blockscale_gemm,
            _is_qwen3_rmsnorm_quant,
        )

    @VllmInductorPass.time_and_log  # pyright: ignore[reportArgumentType]
    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, graph: fx.Graph
    ) -> None:
        self.matched_count = 0
        gemm_nodes = [node for node in graph.nodes if _is_qwen3_blockscale_gemm(node)]
        for gemm1 in gemm_nodes:
            match = _match_qwen3_block_fp8_chain(gemm1)
            if match is None:
                continue
            _replace_outputs(graph, match)
            self.matched_count += 1

        VllmPatternMatcherPass.match_table[self.pass_name] += self.matched_count
        graph.eliminate_dead_code()
        graph.lint()


def _match_qwen3_block_fp8_chain(
    gemm1: fx.Node,
) -> _Qwen3BlockFP8FusionMatch | None:
    """Walk backward from GEMM1 through Qwen3's combined RMSNorm/quant op."""
    quantized = gemm1.args[0]
    quantized_scale = gemm1.args[2]
    if not (
        isinstance(quantized, fx.Node)
        and is_func(quantized, operator.getitem)
        and quantized.args[1] == 0
        and isinstance(quantized_scale, fx.Node)
        and is_func(quantized_scale, operator.getitem)
        and quantized_scale.args[1] == 2
    ):
        return None

    rmsnorm_quant = quantized.args[0]
    if not (
        isinstance(rmsnorm_quant, fx.Node)
        and quantized_scale.args[0] is rmsnorm_quant
        and _is_qwen3_rmsnorm_quant(rmsnorm_quant)
    ):
        return None

    gemm0 = rmsnorm_quant.args[0]
    if not isinstance(gemm0, fx.Node) or not _is_qwen3_blockscale_gemm(gemm0):
        return None

    residual = find_getitem_maybe(rmsnorm_quant, 1)
    if residual is None or not residual.users:
        return None

    # GEMM1 must be the only observer of the quantized activation and scales;
    # otherwise replacing GEMM1 would leave part of the original normalized
    # quant path observable. The residual output is intentionally independent.
    if set(quantized.users) != {gemm1}:
        return None
    if set(quantized_scale.users) != {gemm1}:
        return None
    if set(rmsnorm_quant.users) != {quantized, residual, quantized_scale}:
        return None

    return _Qwen3BlockFP8FusionMatch(
        gemm0=gemm0,
        rmsnorm_quant=rmsnorm_quant,
        quantized=quantized,
        residual=residual,
        quantized_scale=quantized_scale,
        gemm1=gemm1,
    )


def _replace_outputs(
    graph: fx.Graph,
    match: _Qwen3BlockFP8FusionMatch,
) -> None:
    """Redirect the updated residual and GEMM1 result to the reference."""
    gemm0 = match.gemm0
    rmsnorm_quant = match.rmsnorm_quant
    gemm1 = match.gemm1

    with graph.inserting_before(gemm1):
        fused = graph.call_function(
            REFERENCE_OP,
            args=(
                gemm0.args[0],
                gemm0.args[1],
                gemm0.args[2],
                gemm0.args[3],
                rmsnorm_quant.args[1],
                rmsnorm_quant.args[2],
                rmsnorm_quant.args[3],
                rmsnorm_quant.args[4],
                gemm1.args[1],
                gemm1.args[3],
                gemm0,
                _uses_triton_gemm(gemm0),
                gemm0.args[4],
                gemm1.args[4],
            ),
        )
        updated_residual = graph.call_function(operator.getitem, args=(fused, 0))
        model_gemm1 = graph.call_function(
            operator.getitem,
            args=(fused, 2),
        )

    # The gist intentionally leaves its final multiply in FP32. Preserve that
    # value at the reference-op boundary for numerical inspection, while the
    # graph consumes the separately returned BF16 model value. In particular,
    # downstream AITER activation quantizers do not accept FP32. Manually
    # inserted nodes do not receive fake-value metadata, so propagate all three
    # tuple values explicitly.
    transformed_meta = dict(gemm1.meta)
    if "val" in transformed_meta:
        transformed_meta["val"] = transformed_meta["val"].float()
    if "val" in match.residual.meta and "val" in transformed_meta:
        fused.meta["val"] = (
            match.residual.meta["val"],
            transformed_meta["val"],
            gemm1.meta["val"],
        )
    updated_residual.meta.update(match.residual.meta)
    model_gemm1.meta.update(gemm1.meta)

    match.residual.replace_all_uses_with(updated_residual)
    gemm1.replace_all_uses_with(model_gemm1)


def _uses_triton_gemm(node: fx.Node) -> bool:
    return node.target in {PREFILL_GEMM_OP, PREFILL_GEMM_OP_PACKET}


def _is_qwen3_blockscale_gemm(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target
        in {
            PREFILL_GEMM_OP,
            PREFILL_GEMM_OP_PACKET,
            DECODE_GEMM_OP,
            DECODE_GEMM_OP_PACKET,
        }
        and len(node.args) == 5
        and node.args[4] == torch.bfloat16
    )


def _is_qwen3_rmsnorm_quant(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target in {RMSNORM_QUANT_OP, RMSNORM_QUANT_OP_PACKET}
        and len(node.args) == 5
        and isinstance(node.args[3], float)
        and node.args[3] == QWEN3_EPSILON
        and isinstance(node.args[4], int)
        and node.args[4] == QWEN3_GROUP_SIZE
    )
