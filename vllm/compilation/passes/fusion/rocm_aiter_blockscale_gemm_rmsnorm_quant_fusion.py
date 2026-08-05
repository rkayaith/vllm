# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm AITER block-scaled GEMM/residual RMSNorm/group-quant rewrite."""

import operator
from dataclasses import dataclass

import torch
from torch import fx
from torch._ops import OpOverload

import vllm._aiter_ops  # noqa: F401
import vllm.ir.ops  # noqa: F401
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from ..fx_utils import find_getitem_maybe, is_func
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

FP8_DTYPE = current_platform.fp8_dtype()
PREFILL_GEMM_OP = torch.ops.vllm.rocm_aiter_triton_gemm_a8w8_blockscale.default
PREFILL_GEMM_OP_PACKET = torch.ops.vllm.rocm_aiter_triton_gemm_a8w8_blockscale
DECODE_GEMM_OP = torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale.default
DECODE_GEMM_OP_PACKET = torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale
FUSED_ADD_RMS_NORM_OP = torch.ops.vllm_ir.fused_add_rms_norm.default
FUSED_ADD_RMS_NORM_OP_PACKET = torch.ops.vllm_ir.fused_add_rms_norm
GROUP_QUANT_OP = torch.ops.vllm.rocm_aiter_group_fp8_quant.default
GROUP_QUANT_OP_PACKET = torch.ops.vllm.rocm_aiter_group_fp8_quant

REFERENCE_OP_NAME = "rocm_aiter_blockscale_gemm_rmsnorm_quant_reference"
PASS_NAME = "rocm_aiter_blockscale_gemm_rmsnorm_quant_fusion_pass"
QWEN_EPSILON = 1e-6
QWEN_GROUP_SIZE = 128


def rocm_aiter_blockscale_gemm_rmsnorm_quant_reference(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    epsilon: float,
    group_size: int,
    use_triton_gemm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the candidate Qwen fused region with existing PyTorch ops.

    This is a correctness reference, not a fused AITER kernel. It deliberately
    spells out the three stages hidden behind the replacement node so the graph
    rewrite and output contract can be validated before a fused implementation
    exists.
    """
    # Stage 1: reproduce the matched block-scaled FP8 GEMM. Qwen prefill uses
    # the CKTile-backed wrapper, while decode uses the CK-backed wrapper.
    gemm_op = PREFILL_GEMM_OP if use_triton_gemm else DECODE_GEMM_OP
    gemm = gemm_op(
        input,
        weight,
        input_scale,
        weight_scale,
        torch.bfloat16,
    )

    # Stage 2: add the residual through the same vllm_ir operation as the
    # matched graph. Its worker-level priority selects the same implementation
    # as the retained original RMSNorm (AITER for the target Qwen setup). The
    # updated residual is intentionally discarded here because the graph keeps
    # the original path live for its router and residual consumers.
    normalized, _ = FUSED_ADD_RMS_NORM_OP(
        gemm,
        residual,
        rms_weight,
        epsilon,
    )

    # Stage 3: dynamically quantize each 1x128 activation group. Calling the
    # same PyTorch-visible quant op preserves its architecture-specific FP8
    # conversion and scale layout, including any layout expected by Qwen's
    # downstream MoE implementation.
    return GROUP_QUANT_OP(normalized, group_size)


def rocm_aiter_blockscale_gemm_rmsnorm_quant_reference_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    epsilon: float,
    group_size: int,
    use_triton_gemm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del input, weight, input_scale, weight_scale, rms_weight, epsilon
    del use_triton_gemm
    quantized = torch.empty_like(residual, dtype=FP8_DTYPE)
    scales = torch.empty(
        (
            *residual.shape[:-1],
            (residual.shape[-1] + group_size - 1) // group_size,
        ),
        dtype=torch.float32,
        device=residual.device,
    )
    return quantized, scales


direct_register_custom_op(
    op_name=REFERENCE_OP_NAME,
    op_func=rocm_aiter_blockscale_gemm_rmsnorm_quant_reference,
    fake_impl=rocm_aiter_blockscale_gemm_rmsnorm_quant_reference_fake,
)
REFERENCE_OP: OpOverload = getattr(torch.ops.vllm, REFERENCE_OP_NAME).default


@dataclass(frozen=True)
class _QwenFusionMatch:
    gemm: fx.Node
    rmsnorm: fx.Node
    quant: fx.Node
    quantized: fx.Node
    scales: fx.Node


class RocmAiterBlockscaleGemmRMSNormQuantFusionPass(VllmInductorPass):
    """Replace Qwen's block-scaled GEMM/residual RMSNorm/group quant path.

    Only the two outputs of ``rocm_aiter_group_fp8_quant`` are redirected to
    the reference fused node. The original GEMM and RMSNorm outputs are left
    untouched. This is important for Qwen3-235B-A22B-FP8: the normalized BF16
    value also feeds the router, and the updated residual continues through
    the decoder stack.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.pass_name = PASS_NAME
        self.matched_count = 0

    def uuid(self) -> str:
        # The pass delegates almost all semantics to module-level helpers and
        # the temporary reference implementation. Include them in the cache
        # key so changing any part of the rewrite invalidates compiled graphs.
        return self.hash_source(
            self,
            f"{REFERENCE_OP_NAME}:{QWEN_EPSILON}:{QWEN_GROUP_SIZE}",
            rocm_aiter_blockscale_gemm_rmsnorm_quant_reference,
            rocm_aiter_blockscale_gemm_rmsnorm_quant_reference_fake,
            _match_qwen_chain,
            _replace_quant_outputs,
            _is_qwen_blockscale_gemm,
            _is_qwen_rmsnorm,
            _is_group_quant,
        )

    @VllmInductorPass.time_and_log
    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, graph: fx.Graph
    ) -> None:
        self.matched_count = 0
        quant_nodes = [node for node in graph.nodes if _is_group_quant(node)]
        for quant in quant_nodes:
            match = _match_qwen_chain(quant)
            if match is None:
                continue
            _replace_quant_outputs(graph, match)
            self.matched_count += 1

        VllmPatternMatcherPass.match_table[self.pass_name] += self.matched_count
        graph.eliminate_dead_code()
        graph.lint()


def _match_qwen_chain(quant: fx.Node) -> _QwenFusionMatch | None:
    """Walk backward from group quant through the exact cleaned Qwen chain."""
    normalized = quant.args[0]
    if not (
        isinstance(normalized, fx.Node)
        and is_func(normalized, operator.getitem)
        and normalized.args[1] == 0
    ):
        return None

    rmsnorm = normalized.args[0]
    if not isinstance(rmsnorm, fx.Node) or not _is_qwen_rmsnorm(rmsnorm):
        return None

    gemm = rmsnorm.args[0]
    if not isinstance(gemm, fx.Node) or not _is_qwen_blockscale_gemm(gemm):
        return None

    quantized = find_getitem_maybe(quant, 0)
    scales = find_getitem_maybe(quant, 1)
    if quantized is None or scales is None:
        return None

    # Replacing both tuple outputs makes the standalone quant node dead. Do not
    # match a graph that observes the tuple itself or an unexpected element.
    if set(quant.users) != {quantized, scales}:
        return None

    return _QwenFusionMatch(
        gemm=gemm,
        rmsnorm=rmsnorm,
        quant=quant,
        quantized=quantized,
        scales=scales,
    )


def _replace_quant_outputs(graph: fx.Graph, match: _QwenFusionMatch) -> None:
    """Insert the reference node and redirect only the requantized outputs."""
    gemm = match.gemm
    rmsnorm = match.rmsnorm
    use_triton_gemm = gemm.target in {PREFILL_GEMM_OP, PREFILL_GEMM_OP_PACKET}

    with graph.inserting_before(match.quant):
        fused = graph.call_function(
            REFERENCE_OP,
            args=(
                gemm.args[0],
                gemm.args[1],
                gemm.args[2],
                gemm.args[3],
                rmsnorm.args[1],
                rmsnorm.args[2],
                rmsnorm.args[3],
                match.quant.args[1],
                use_triton_gemm,
            ),
        )
        quantized = graph.call_function(operator.getitem, args=(fused, 0))
        scales = graph.call_function(operator.getitem, args=(fused, 1))

    # Manually inserted nodes do not receive fake-value metadata. Preserve the
    # values of the exact tuple elements being replaced for later vLLM passes.
    if "val" in match.quantized.meta and "val" in match.scales.meta:
        fused.meta["val"] = (
            match.quantized.meta["val"],
            match.scales.meta["val"],
        )
    quantized.meta.update(match.quantized.meta)
    scales.meta.update(match.scales.meta)

    match.quantized.replace_all_uses_with(quantized)
    match.scales.replace_all_uses_with(scales)


def _is_qwen_blockscale_gemm(node: fx.Node) -> bool:
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


def _is_qwen_rmsnorm(node: fx.Node) -> bool:
    epsilon = node.args[3] if len(node.args) == 4 else None
    return (
        node.op == "call_function"
        and node.target in {FUSED_ADD_RMS_NORM_OP, FUSED_ADD_RMS_NORM_OP_PACKET}
        and len(node.args) == 4
        and isinstance(epsilon, float)
        and epsilon == QWEN_EPSILON
    )


def _is_group_quant(node: fx.Node) -> bool:
    group_size = node.args[1] if len(node.args) == 2 else None
    return (
        node.op == "call_function"
        and node.target in {GROUP_QUANT_OP, GROUP_QUANT_OP_PACKET}
        and len(node.args) == 2
        and isinstance(group_size, int)
        and group_size == QWEN_GROUP_SIZE
    )
