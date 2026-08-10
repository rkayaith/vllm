# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm CoDA two-GEMM residual RMSNorm rewrites."""

import operator
from dataclasses import dataclass

import torch
from torch import fx
from torch._ops import OpOverload
from torch.fx.node import Argument

import vllm._aiter_ops  # noqa: F401
import vllm.ir.ops  # noqa: F401
from vllm.config import VllmConfig
from vllm.model_executor.kernels.linear.mxfp8 import rocm_native  # noqa: F401
from vllm.utils.torch_utils import direct_register_custom_op

from ..fx_utils import find_getitem_maybe, is_func
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

BLOCK_FP8_PREFILL_GEMM_OP = (
    torch.ops.vllm.rocm_aiter_triton_gemm_a8w8_blockscale.default
)
BLOCK_FP8_PREFILL_GEMM_OP_PACKET = torch.ops.vllm.rocm_aiter_triton_gemm_a8w8_blockscale
BLOCK_FP8_DECODE_GEMM_OP = torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale.default
BLOCK_FP8_DECODE_GEMM_OP_PACKET = torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale
BLOCK_FP8_QUANT_OP = torch.ops.vllm.rocm_aiter_group_fp8_quant.default
BLOCK_FP8_QUANT_OP_PACKET = torch.ops.vllm.rocm_aiter_group_fp8_quant
FUSED_ADD_RMS_NORM_OP = torch.ops.vllm_ir.fused_add_rms_norm.default
MXFP8_GEMM_OP = torch.ops.vllm.rocm_mxfp8_dot_scaled_mm.default
MXFP8_GEMM_OP_PACKET = torch.ops.vllm.rocm_mxfp8_dot_scaled_mm
MXFP8_QUANT_OP = torch.ops.vllm.mxfp8_quantize.default
MXFP8_QUANT_OP_PACKET = torch.ops.vllm.mxfp8_quantize

BLOCK_FP8_REFERENCE_OP_NAME = "rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference"
MXFP8_REFERENCE_OP_NAME = "rocm_mxfp8_gemm_rmsnorm_gemm_reference"
PASS_NAME = "rocm_coda_gemm_rmsnorm_gemm_fusion_pass"
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
    gemm0_use_triton: bool,
    gemm0_output_dtype: torch.dtype,
    gemm1_output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the delayed-rounding Qwen3-32B block-FP8 CoDA candidate."""
    gemm0_op = (
        BLOCK_FP8_PREFILL_GEMM_OP if gemm0_use_triton else BLOCK_FP8_DECODE_GEMM_OP
    )
    gemm0 = gemm0_op(
        input,
        gemm0_weight,
        input_scale,
        gemm0_weight_scale,
        gemm0_output_dtype,
    )
    updated_residual = gemm0 + residual
    inverse_rms = torch.rsqrt(
        updated_residual.float().pow(2).mean(dim=-1, keepdim=True) + epsilon
    )
    weighted = updated_residual * rms_weight
    quantized, quantized_scale = _quantize_fp8_tile_activation(weighted, group_size)

    # The CK wrapper cannot return the FP32 value required at this boundary.
    # Use Triton for GEMM1 in both prefill and decode, then round only after the
    # FP32 inverse-RMS multiply.
    gemm1 = BLOCK_FP8_PREFILL_GEMM_OP(
        quantized,
        gemm1_weight,
        quantized_scale,
        gemm1_weight_scale,
        torch.float32,
    )
    transformed_gemm1 = gemm1 * inverse_rms
    model_gemm1 = transformed_gemm1.to(gemm1_output_dtype)
    return updated_residual, transformed_gemm1, model_gemm1


def _quantize_fp8_tile_activation(
    input: torch.Tensor,
    tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the CoDA 1xN symmetric FP8 activation quantization."""
    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)
    rows, columns = input.shape
    tile_columns = (columns + tile_size - 1) // tile_size
    padded = torch.nn.functional.pad(
        input.float(), (0, tile_columns * tile_size - columns)
    )
    absolute_maximum = (
        padded.reshape(rows, tile_columns, tile_size)
        .abs()
        .amax(dim=-1)
        .clamp(min=1e-12)
    )
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
    gemm0_use_triton: bool,
    gemm0_output_dtype: torch.dtype,
    gemm1_output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fake_reference_outputs(input, residual, gemm1_weight, gemm1_output_dtype)


direct_register_custom_op(
    op_name=BLOCK_FP8_REFERENCE_OP_NAME,
    op_func=rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference,
    fake_impl=rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference_fake,
)
BLOCK_FP8_REFERENCE_OP: OpOverload = getattr(
    torch.ops.vllm, BLOCK_FP8_REFERENCE_OP_NAME
).default


def rocm_mxfp8_gemm_rmsnorm_gemm_reference(
    input: torch.Tensor,
    gemm0_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gemm0_weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    epsilon: float,
    gemm1_weight: torch.Tensor,
    gemm1_weight_scale: torch.Tensor,
    gemm0_output_dtype: torch.dtype,
    gemm1_output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the delayed-rounding Qwen3-32B MXFP8 CoDA candidate."""
    gemm0 = MXFP8_GEMM_OP(
        input,
        input_scale,
        gemm0_weight,
        gemm0_weight_scale,
        gemm0_output_dtype,
    )
    updated_residual = gemm0 + residual
    inverse_rms = torch.rsqrt(
        updated_residual.float().pow(2).mean(dim=-1, keepdim=True) + epsilon
    )
    weighted = updated_residual * rms_weight
    quantized, quantized_scale = MXFP8_QUANT_OP(weighted, False, 0)
    gemm1 = MXFP8_GEMM_OP(
        quantized,
        quantized_scale,
        gemm1_weight,
        gemm1_weight_scale,
        torch.float32,
    )
    transformed_gemm1 = gemm1 * inverse_rms
    model_gemm1 = transformed_gemm1.to(gemm1_output_dtype)
    return updated_residual, transformed_gemm1, model_gemm1


def rocm_mxfp8_gemm_rmsnorm_gemm_reference_fake(
    input: torch.Tensor,
    gemm0_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gemm0_weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    epsilon: float,
    gemm1_weight: torch.Tensor,
    gemm1_weight_scale: torch.Tensor,
    gemm0_output_dtype: torch.dtype,
    gemm1_output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fake_reference_outputs(input, residual, gemm1_weight, gemm1_output_dtype)


direct_register_custom_op(
    op_name=MXFP8_REFERENCE_OP_NAME,
    op_func=rocm_mxfp8_gemm_rmsnorm_gemm_reference,
    fake_impl=rocm_mxfp8_gemm_rmsnorm_gemm_reference_fake,
)
MXFP8_REFERENCE_OP: OpOverload = getattr(
    torch.ops.vllm, MXFP8_REFERENCE_OP_NAME
).default


def _fake_reference_outputs(
    input: torch.Tensor,
    residual: torch.Tensor,
    gemm1_weight: torch.Tensor,
    gemm1_output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_shape = (input.shape[0], gemm1_weight.shape[0])
    return (
        torch.empty_like(residual),
        torch.empty(output_shape, dtype=torch.float32, device=input.device),
        torch.empty(output_shape, dtype=gemm1_output_dtype, device=input.device),
    )


@dataclass(frozen=True)
class _CodaFormat:
    name: str
    gemm_targets: frozenset[object]
    quant_targets: frozenset[object]
    input_scale_index: int
    weight_index: int
    weight_scale_index: int
    reference_op: OpOverload


BLOCK_FP8_FORMAT = _CodaFormat(
    name="block_fp8",
    gemm_targets=frozenset(
        {
            BLOCK_FP8_PREFILL_GEMM_OP,
            BLOCK_FP8_PREFILL_GEMM_OP_PACKET,
            BLOCK_FP8_DECODE_GEMM_OP,
            BLOCK_FP8_DECODE_GEMM_OP_PACKET,
        }
    ),
    quant_targets=frozenset({BLOCK_FP8_QUANT_OP, BLOCK_FP8_QUANT_OP_PACKET}),
    input_scale_index=2,
    weight_index=1,
    weight_scale_index=3,
    reference_op=BLOCK_FP8_REFERENCE_OP,
)
MXFP8_FORMAT = _CodaFormat(
    name="mxfp8",
    gemm_targets=frozenset({MXFP8_GEMM_OP, MXFP8_GEMM_OP_PACKET}),
    quant_targets=frozenset({MXFP8_QUANT_OP, MXFP8_QUANT_OP_PACKET}),
    input_scale_index=1,
    weight_index=2,
    weight_scale_index=3,
    reference_op=MXFP8_REFERENCE_OP,
)
CODA_FORMATS = (BLOCK_FP8_FORMAT, MXFP8_FORMAT)


@dataclass(frozen=True)
class _CodaFusionMatch:
    format: _CodaFormat
    gemm0: fx.Node
    rmsnorm: fx.Node
    normalized: fx.Node
    residual: fx.Node
    quant: fx.Node
    quantized: fx.Node
    quantized_scale: fx.Node
    gemm1: fx.Node


class RocmCodaGemmRMSNormGemmFusionPass(VllmInductorPass):
    """Replace Qwen3-32B block-FP8 and MXFP8 two-GEMM CoDA regions."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.pass_name = PASS_NAME
        self.matched_count = 0
        self.matched_count_by_format = {
            format_spec.name: 0 for format_spec in CODA_FORMATS
        }

    def uuid(self) -> str:
        return self.hash_source(
            self,
            f"{PASS_NAME}:{QWEN3_EPSILON}:{QWEN3_GROUP_SIZE}",
            rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference,
            rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference_fake,
            rocm_mxfp8_gemm_rmsnorm_gemm_reference,
            rocm_mxfp8_gemm_rmsnorm_gemm_reference_fake,
            _quantize_fp8_tile_activation,
            _match_qwen3_coda_chain,
            _replace_outputs,
            _format_for_gemm,
            _is_qwen3_gemm,
            _is_qwen3_rmsnorm,
            _is_qwen3_quant,
        )

    @VllmInductorPass.time_and_log  # pyright: ignore[reportArgumentType]
    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, graph: fx.Graph
    ) -> None:
        self.matched_count = 0
        self.matched_count_by_format = {
            format_spec.name: 0 for format_spec in CODA_FORMATS
        }
        gemm_nodes = [
            node for node in graph.nodes if _format_for_gemm(node) is not None
        ]
        for gemm1 in gemm_nodes:
            match = _match_qwen3_coda_chain(gemm1)
            if match is None:
                continue
            _replace_outputs(graph, match)
            self.matched_count += 1
            self.matched_count_by_format[match.format.name] += 1

        VllmPatternMatcherPass.match_table[self.pass_name] += self.matched_count
        graph.eliminate_dead_code()
        graph.lint()


def _match_qwen3_coda_chain(gemm1: fx.Node) -> _CodaFusionMatch | None:
    format_spec = _format_for_gemm(gemm1)
    if format_spec is None or not _is_qwen3_gemm(gemm1, format_spec):
        return None

    quantized = gemm1.args[0]
    quantized_scale = gemm1.args[format_spec.input_scale_index]
    if not (
        isinstance(quantized, fx.Node)
        and is_func(quantized, operator.getitem)
        and quantized.args[1] == 0
        and isinstance(quantized_scale, fx.Node)
        and is_func(quantized_scale, operator.getitem)
        and quantized_scale.args[1] == 1
    ):
        return None

    quant = quantized.args[0]
    if not (
        isinstance(quant, fx.Node)
        and quantized_scale.args[0] is quant
        and _is_qwen3_quant(quant, format_spec)
    ):
        return None

    normalized = quant.args[0]
    if not (
        isinstance(normalized, fx.Node)
        and is_func(normalized, operator.getitem)
        and normalized.args[1] == 0
    ):
        return None

    rmsnorm = normalized.args[0]
    if not isinstance(rmsnorm, fx.Node) or not _is_qwen3_rmsnorm(rmsnorm):
        return None

    gemm0 = rmsnorm.args[0]
    if not (
        isinstance(gemm0, fx.Node)
        and _format_for_gemm(gemm0) == format_spec
        and _is_qwen3_gemm(gemm0, format_spec)
    ):
        return None

    residual = find_getitem_maybe(rmsnorm, 1)
    if residual is None or not residual.users:
        return None

    if set(quantized.users) != {gemm1}:
        return None
    if set(quantized_scale.users) != {gemm1}:
        return None
    if set(quant.users) != {quantized, quantized_scale}:
        return None
    if set(normalized.users) != {quant}:
        return None
    if set(rmsnorm.users) != {normalized, residual}:
        return None

    return _CodaFusionMatch(
        format=format_spec,
        gemm0=gemm0,
        rmsnorm=rmsnorm,
        normalized=normalized,
        residual=residual,
        quant=quant,
        quantized=quantized,
        quantized_scale=quantized_scale,
        gemm1=gemm1,
    )


def _replace_outputs(graph: fx.Graph, match: _CodaFusionMatch) -> None:
    with graph.inserting_before(match.gemm1):
        fused = graph.call_function(
            match.format.reference_op,
            args=_reference_args(match),
        )
        updated_residual = graph.call_function(operator.getitem, args=(fused, 0))
        model_gemm1 = graph.call_function(operator.getitem, args=(fused, 2))

    transformed_meta = dict(match.gemm1.meta)
    if "val" in transformed_meta:
        transformed_meta["val"] = transformed_meta["val"].float()
    if "val" in match.residual.meta and "val" in transformed_meta:
        fused.meta["val"] = (
            match.residual.meta["val"],
            transformed_meta["val"],
            match.gemm1.meta["val"],
        )
    updated_residual.meta.update(match.residual.meta)
    model_gemm1.meta.update(match.gemm1.meta)

    match.residual.replace_all_uses_with(updated_residual)
    match.gemm1.replace_all_uses_with(model_gemm1)


def _reference_args(match: _CodaFusionMatch) -> tuple[Argument, ...]:
    format_spec = match.format
    gemm0 = match.gemm0
    gemm1 = match.gemm1
    common_args = (
        gemm0.args[0],
        gemm0.args[format_spec.weight_index],
        gemm0.args[format_spec.input_scale_index],
        gemm0.args[format_spec.weight_scale_index],
        match.rmsnorm.args[1],
        match.rmsnorm.args[2],
        match.rmsnorm.args[3],
    )
    gemm1_args = (
        gemm1.args[format_spec.weight_index],
        gemm1.args[format_spec.weight_scale_index],
    )
    output_args = (gemm0.args[4], gemm1.args[4])
    if format_spec == BLOCK_FP8_FORMAT:
        return (
            *common_args,
            match.quant.args[1],
            *gemm1_args,
            _uses_block_fp8_triton_gemm(gemm0),
            *output_args,
        )
    return (*common_args, *gemm1_args, *output_args)


def _format_for_gemm(node: fx.Node) -> _CodaFormat | None:
    if node.op != "call_function":
        return None
    for format_spec in CODA_FORMATS:
        if node.target in format_spec.gemm_targets:
            return format_spec
    return None


def _is_qwen3_gemm(node: fx.Node, format_spec: _CodaFormat) -> bool:
    return (
        node.op == "call_function"
        and node.target in format_spec.gemm_targets
        and len(node.args) == 5
        and node.args[4] == torch.bfloat16
    )


def _is_qwen3_rmsnorm(node: fx.Node) -> bool:
    if not (
        node.op == "call_function"
        and node.target == FUSED_ADD_RMS_NORM_OP
        and len(node.args) in {4, 5}
    ):
        return False
    epsilon = node.args[3]
    return (
        isinstance(epsilon, float)
        and epsilon == QWEN3_EPSILON
        and _rmsnorm_variance_size(node) is None
    )


def _is_qwen3_quant(node: fx.Node, format_spec: _CodaFormat) -> bool:
    if node.op != "call_function" or node.target not in format_spec.quant_targets:
        return False
    if format_spec == BLOCK_FP8_FORMAT:
        return (
            len(node.args) == 2
            and isinstance(node.args[1], int)
            and node.args[1] == QWEN3_GROUP_SIZE
        )
    is_sf_swizzled_layout = _quant_is_sf_swizzled_layout(node)
    alignment = _quant_alignment(node)
    return (
        len(node.args) in {1, 2, 3}
        and isinstance(is_sf_swizzled_layout, bool)
        and not is_sf_swizzled_layout
        and isinstance(alignment, int)
        and alignment == 0
    )


def _uses_block_fp8_triton_gemm(node: fx.Node) -> bool:
    return node.target in {
        BLOCK_FP8_PREFILL_GEMM_OP,
        BLOCK_FP8_PREFILL_GEMM_OP_PACKET,
    }


def _rmsnorm_variance_size(node: fx.Node) -> Argument:
    return node.args[4] if len(node.args) == 5 else None


def _quant_is_sf_swizzled_layout(node: fx.Node) -> Argument:
    return node.args[1] if len(node.args) >= 2 else False


def _quant_alignment(node: fx.Node) -> Argument:
    return node.args[2] if len(node.args) == 3 else 0
