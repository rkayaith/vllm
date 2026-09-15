# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm CoDA two-GEMM residual RMSNorm rewrites."""

import operator
from dataclasses import dataclass

import torch
from torch import fx
from torch._ops import OpOverload
from torch.fx.node import Argument

import vllm.ir.ops  # noqa: F401
import vllm.model_executor.layers.utils  # noqa: F401
from vllm._aiter_ops import _ensure_hipb_mm_extension_initialized
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
MXFP8_GEMM_OP = torch.ops.vllm.rocm_mxfp8_dot_scaled_mm_gfx950.default
MXFP8_GEMM_OP_PACKET = torch.ops.vllm.rocm_mxfp8_dot_scaled_mm_gfx950
MXFP8_QUANT_OP = torch.ops.vllm.rocm_aiter_mxfp8_quantize_gfx950.default
MXFP8_QUANT_OP_PACKET = torch.ops.vllm.rocm_aiter_mxfp8_quantize_gfx950
BF16_GEMM_OP = torch.ops.vllm.rocm_unquantized_gemm.default
BF16_GEMM_OP_PACKET = torch.ops.vllm.rocm_unquantized_gemm

BLOCK_FP8_REFERENCE_OP_NAME = "rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference"
MXFP8_FUSED_OP_NAME = "rocm_mxfp8_gemm_rmsnorm_gemm"
BF16_FUSED_OP_NAME = "rocm_aiter_hipb_mm_epilogue_mm_bf16"
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
) -> tuple[torch.Tensor, torch.Tensor]:
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
    model_gemm1 = (gemm1 * inverse_rms).to(gemm1_output_dtype)
    return updated_residual, model_gemm1


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
) -> tuple[torch.Tensor, torch.Tensor]:
    return _fake_outputs(input, residual, gemm1_weight, gemm1_output_dtype)


direct_register_custom_op(
    op_name=BLOCK_FP8_REFERENCE_OP_NAME,
    op_func=rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference,
    fake_impl=rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference_fake,
)
BLOCK_FP8_REFERENCE_OP: OpOverload = getattr(
    torch.ops.vllm, BLOCK_FP8_REFERENCE_OP_NAME
).default


def rocm_mxfp8_gemm_rmsnorm_gemm(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    gemm0_weight: torch.Tensor,
    gemm0_weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    epsilon: float,
    gemm1_weight: torch.Tensor,
    gemm1_weight_scale: torch.Tensor,
    gemm0_output_dtype: torch.dtype,
    gemm1_output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the MXFP8 GEMM -> residual -> RMSNorm -> GEMM sequence."""
    from aiter.ops.gradlib import (
        HipblasLtMatmulScaleMode,
        Requant,
        ResidualAdd,
        RMSNorm,
        hipb_mm_epilogue_mm,
    )

    del gemm0_output_dtype
    _ensure_hipb_mm_extension_initialized()
    residual_out = torch.empty_like(residual)
    scale_rows = (input.shape[0] + 31) // 32 * 32
    scale_columns = ((gemm0_weight.shape[0] + 31) // 32 + 7) // 8 * 8
    requant_scale_out = torch.empty(
        (scale_rows, scale_columns),
        dtype=torch.uint8,
        device=input.device,
    )
    gemm_out = torch.empty(
        (input.shape[0], gemm1_weight.shape[0]),
        dtype=gemm1_output_dtype,
        device=input.device,
    )
    scale_mode = HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT
    gemm_out = hipb_mm_epilogue_mm(
        input,
        gemm0_weight.mT,
        gemm1_weight.mT,
        stages=(
            ResidualAdd(residual=residual, residual_out=residual_out),
            RMSNorm(gamma=rms_weight, eps=epsilon),
            Requant(scale_out=requant_scale_out, scale_mode=scale_mode),
        ),
        intermediate_dtype=torch.float8_e4m3fn,
        dtype=gemm1_output_dtype,
        mat1_scale=input_scale,
        mat2_scale=gemm0_weight_scale,
        mat3_scale=gemm1_weight_scale,
        mat1_scale_mode=scale_mode,
        mat2_scale_mode=scale_mode,
        mat3_scale_mode=scale_mode,
        out=gemm_out,
    )
    return residual_out, gemm_out


def rocm_mxfp8_gemm_rmsnorm_gemm_fake(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    gemm0_weight: torch.Tensor,
    gemm0_weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    epsilon: float,
    gemm1_weight: torch.Tensor,
    gemm1_weight_scale: torch.Tensor,
    gemm0_output_dtype: torch.dtype,
    gemm1_output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    del input_scale
    return _fake_outputs(input, residual, gemm1_weight, gemm1_output_dtype)


direct_register_custom_op(
    op_name=MXFP8_FUSED_OP_NAME,
    op_func=rocm_mxfp8_gemm_rmsnorm_gemm,
    fake_impl=rocm_mxfp8_gemm_rmsnorm_gemm_fake,
)
MXFP8_FUSED_OP: OpOverload = getattr(torch.ops.vllm, MXFP8_FUSED_OP_NAME).default


def rocm_aiter_hipb_mm_epilogue_mm_bf16(
    input: torch.Tensor,
    gemm0_weight: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    epsilon: float,
    gemm1_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the BF16 GEMM -> residual -> RMSNorm -> GEMM sequence."""
    from aiter.ops.gradlib import (
        ResidualAdd,
        RMSNorm,
        hipb_mm_epilogue_mm,
    )

    _ensure_hipb_mm_extension_initialized()
    residual_out = torch.empty_like(residual)
    gemm_out = torch.empty(
        (input.shape[0], gemm1_weight.shape[0]),
        dtype=torch.bfloat16,
        device=input.device,
    )
    gemm_out = hipb_mm_epilogue_mm(
        input,
        gemm0_weight.mT,
        gemm1_weight.mT,
        stages=(
            ResidualAdd(residual=residual, residual_out=residual_out),
            RMSNorm(gamma=rms_weight, eps=epsilon),
        ),
        intermediate_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        out=gemm_out,
    )
    return residual_out, gemm_out


def rocm_aiter_hipb_mm_epilogue_mm_bf16_fake(
    input: torch.Tensor,
    gemm0_weight: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    epsilon: float,
    gemm1_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del gemm0_weight, rms_weight, epsilon
    output = torch.empty(
        (input.shape[0], gemm1_weight.shape[0]),
        dtype=torch.bfloat16,
        device=input.device,
    )
    return torch.empty_like(residual), output


direct_register_custom_op(
    op_name=BF16_FUSED_OP_NAME,
    op_func=rocm_aiter_hipb_mm_epilogue_mm_bf16,
    fake_impl=rocm_aiter_hipb_mm_epilogue_mm_bf16_fake,
)
BF16_FUSED_OP: OpOverload = getattr(torch.ops.vllm, BF16_FUSED_OP_NAME).default


def _fake_outputs(
    input: torch.Tensor,
    residual: torch.Tensor,
    gemm1_weight: torch.Tensor,
    gemm1_output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_shape = (input.shape[0], gemm1_weight.shape[0])
    return (
        torch.empty_like(residual),
        torch.empty(output_shape, dtype=gemm1_output_dtype, device=input.device),
    )


@dataclass(frozen=True)
class _CodaFormat:
    name: str
    gemm_targets: frozenset[object]
    quant_targets: frozenset[object]
    input_scale_index: int | None
    weight_index: int
    weight_scale_index: int | None
    replacement_op: OpOverload


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
    replacement_op=BLOCK_FP8_REFERENCE_OP,
)
MXFP8_FORMAT = _CodaFormat(
    name="mxfp8",
    gemm_targets=frozenset({MXFP8_GEMM_OP, MXFP8_GEMM_OP_PACKET}),
    quant_targets=frozenset({MXFP8_QUANT_OP, MXFP8_QUANT_OP_PACKET}),
    input_scale_index=1,
    weight_index=2,
    weight_scale_index=3,
    replacement_op=MXFP8_FUSED_OP,
)
BF16_FORMAT = _CodaFormat(
    name="bf16",
    gemm_targets=frozenset({BF16_GEMM_OP, BF16_GEMM_OP_PACKET}),
    quant_targets=frozenset(),
    input_scale_index=None,
    weight_index=1,
    weight_scale_index=None,
    replacement_op=BF16_FUSED_OP,
)
STAGED_CODA_FORMATS = (BLOCK_FP8_FORMAT, MXFP8_FORMAT)
CODA_FORMATS = (*STAGED_CODA_FORMATS, BF16_FORMAT)


@dataclass(frozen=True)
class _CodaFusionMatch:
    format: _CodaFormat
    gemm0: fx.Node
    rmsnorm: fx.Node
    normalized: fx.Node
    residual: fx.Node
    quant: fx.Node | None
    quantized: fx.Node | None
    quantized_scale: fx.Node | None
    gemm1: fx.Node
    input_quant: fx.Node | None = None
    input_quantized: fx.Node | None = None
    input_quantized_scale: fx.Node | None = None


class RocmCodaGemmRMSNormGemmFusionPass(VllmInductorPass):
    """Replace supported two-GEMM residual RMSNorm regions."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.pass_name = PASS_NAME
        self.enabled_formats = (
            CODA_FORMATS
            if config.kernel_config.use_hipbmm_fused_epilogue
            else STAGED_CODA_FORMATS
        )
        self.matched_count = 0
        self.matched_count_by_format = {
            format_spec.name: 0 for format_spec in CODA_FORMATS
        }

    def uuid(self) -> str:
        return self.hash_source(
            self,
            f"{PASS_NAME}:{QWEN3_EPSILON}:{QWEN3_GROUP_SIZE}:"
            + ",".join(format_spec.name for format_spec in self.enabled_formats),
            rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference,
            rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference_fake,
            rocm_mxfp8_gemm_rmsnorm_gemm,
            rocm_mxfp8_gemm_rmsnorm_gemm_fake,
            rocm_aiter_hipb_mm_epilogue_mm_bf16,
            rocm_aiter_hipb_mm_epilogue_mm_bf16_fake,
            _quantize_fp8_tile_activation,
            _match_qwen3_coda_chain,
            _replace_outputs,
            _format_for_gemm,
            _is_qwen3_gemm,
            _is_qwen3_rmsnorm,
            _is_qwen3_quant,
            _is_bf16_tensor,
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
            node
            for node in graph.nodes
            if _format_for_gemm(node, self.enabled_formats) is not None
        ]
        for gemm1 in gemm_nodes:
            match = _match_qwen3_coda_chain(gemm1, self.enabled_formats)
            if match is None:
                continue
            _replace_outputs(graph, match)
            self.matched_count += 1
            self.matched_count_by_format[match.format.name] += 1

        VllmPatternMatcherPass.match_table[self.pass_name] += self.matched_count
        graph.eliminate_dead_code()
        remaining_count_by_format = {
            format_spec.name: 0 for format_spec in CODA_FORMATS
        }
        for node in graph.nodes:
            match = _match_qwen3_coda_chain(node, self.enabled_formats)
            if match is not None:
                remaining_count_by_format[match.format.name] += 1
        if any(remaining_count_by_format.values()):
            raise AssertionError(
                f"{PASS_NAME} left eligible chains after rewriting: "
                f"{remaining_count_by_format}"
            )
        graph.lint()


def _match_qwen3_coda_chain(
    gemm1: fx.Node,
    formats: tuple[_CodaFormat, ...] = CODA_FORMATS,
) -> _CodaFusionMatch | None:
    format_spec = _format_for_gemm(gemm1, formats)
    if format_spec is None or not _is_qwen3_gemm(gemm1, format_spec):
        return None

    quant = None
    quantized = None
    quantized_scale = None
    if format_spec == BF16_FORMAT:
        normalized = gemm1.args[0]
    else:
        assert format_spec.input_scale_index is not None
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
    if _rmsnorm_variance_size(rmsnorm) is not None:
        return None
    if format_spec == BF16_FORMAT:
        if not _is_bf16_tensor(rmsnorm.args[1], ndim=2):
            return None
        if not _is_bf16_tensor(rmsnorm.args[2], ndim=1):
            return None
    elif format_spec == BLOCK_FP8_FORMAT and rmsnorm.args[3] != QWEN3_EPSILON:
        return None

    gemm0 = rmsnorm.args[0]
    if not (
        isinstance(gemm0, fx.Node)
        and _format_for_gemm(gemm0, formats) == format_spec
        and _is_qwen3_gemm(gemm0, format_spec)
    ):
        return None

    input_quant = None
    input_quantized = None
    input_quantized_scale = None
    if format_spec == MXFP8_FORMAT:
        input_quantized = gemm0.args[0]
        input_quantized_scale = gemm0.args[format_spec.input_scale_index]
        if not (
            isinstance(input_quantized, fx.Node)
            and is_func(input_quantized, operator.getitem)
            and input_quantized.args[1] == 0
            and isinstance(input_quantized_scale, fx.Node)
            and is_func(input_quantized_scale, operator.getitem)
            and input_quantized_scale.args[1] == 1
        ):
            return None

        input_quant = input_quantized.args[0]
        if not (
            isinstance(input_quant, fx.Node)
            and input_quantized_scale.args[0] is input_quant
            and _is_qwen3_quant(input_quant, format_spec)
        ):
            return None

    residual = find_getitem_maybe(rmsnorm, 1)
    if residual is None or not residual.users:
        return None

    if format_spec == BF16_FORMAT:
        if set(normalized.users) != {gemm1}:
            return None
    else:
        assert quant is not None
        assert quantized is not None
        assert quantized_scale is not None
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
        input_quant=input_quant,
        input_quantized=input_quantized,
        input_quantized_scale=input_quantized_scale,
    )


def _replace_outputs(graph: fx.Graph, match: _CodaFusionMatch) -> None:
    with graph.inserting_before(match.gemm1):
        fused = graph.call_function(
            match.format.replacement_op,
            args=_replacement_args(match),
        )
        updated_residual = graph.call_function(operator.getitem, args=(fused, 0))
        model_gemm1 = graph.call_function(operator.getitem, args=(fused, 1))

    if "val" in match.residual.meta and "val" in match.gemm1.meta:
        gemm1_value = match.gemm1.meta["val"]
        fused.meta["val"] = (
            match.residual.meta["val"],
            gemm1_value,
        )
    updated_residual.meta.update(match.residual.meta)
    model_gemm1.meta.update(match.gemm1.meta)
    if "val" in fused.meta:
        model_gemm1.meta["val"] = fused.meta["val"][1]

    match.residual.replace_all_uses_with(updated_residual)
    match.gemm1.replace_all_uses_with(model_gemm1)


def _replacement_args(match: _CodaFusionMatch) -> tuple[Argument, ...]:
    format_spec = match.format
    gemm0 = match.gemm0
    gemm1 = match.gemm1
    norm_args = (
        match.rmsnorm.args[1],
        match.rmsnorm.args[2],
        match.rmsnorm.args[3],
    )
    if format_spec == BF16_FORMAT:
        return (
            gemm0.args[0],
            gemm0.args[format_spec.weight_index],
            *norm_args,
            gemm1.args[format_spec.weight_index],
        )

    assert format_spec.weight_scale_index is not None
    gemm1_args = (
        gemm1.args[format_spec.weight_index],
        gemm1.args[format_spec.weight_scale_index],
    )
    output_args = (gemm0.args[4], gemm1.args[4])
    if format_spec == BLOCK_FP8_FORMAT:
        return (
            gemm0.args[0],
            gemm0.args[format_spec.weight_index],
            gemm0.args[format_spec.input_scale_index],
            gemm0.args[format_spec.weight_scale_index],
            *norm_args,
            match.quant.args[1],
            *gemm1_args,
            _uses_block_fp8_triton_gemm(gemm0),
            *output_args,
        )
    assert match.input_quant is not None
    assert match.input_quantized is not None
    assert match.input_quantized_scale is not None
    return (
        match.input_quantized,
        match.input_quantized_scale,
        gemm0.args[format_spec.weight_index],
        gemm0.args[format_spec.weight_scale_index],
        *norm_args,
        *gemm1_args,
        *output_args,
    )


def _format_for_gemm(
    node: fx.Node,
    formats: tuple[_CodaFormat, ...] = CODA_FORMATS,
) -> _CodaFormat | None:
    if node.op != "call_function":
        return None
    for format_spec in formats:
        if node.target in format_spec.gemm_targets:
            return format_spec
    return None


def _is_qwen3_gemm(node: fx.Node, format_spec: _CodaFormat) -> bool:
    if format_spec == BF16_FORMAT:
        return (
            node.op == "call_function"
            and node.target in format_spec.gemm_targets
            and len(node.args) in {2, 3}
            and (len(node.args) == 2 or node.args[2] is None)
            and _is_bf16_tensor(node.args[0], ndim=2)
            and _is_bf16_tensor(node.args[1], ndim=2)
            and _is_bf16_tensor(node, ndim=2)
        )
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
    return isinstance(node.args[3], float)


def _is_qwen3_quant(node: fx.Node, format_spec: _CodaFormat) -> bool:
    assert format_spec != BF16_FORMAT
    if node.op != "call_function" or node.target not in format_spec.quant_targets:
        return False
    if format_spec == BLOCK_FP8_FORMAT:
        return (
            len(node.args) == 2
            and isinstance(node.args[1], int)
            and node.args[1] == QWEN3_GROUP_SIZE
        )
    return len(node.args) == 1


def _uses_block_fp8_triton_gemm(node: fx.Node) -> bool:
    return node.target in {
        BLOCK_FP8_PREFILL_GEMM_OP,
        BLOCK_FP8_PREFILL_GEMM_OP_PACKET,
    }


def _rmsnorm_variance_size(node: fx.Node) -> Argument:
    return node.args[4] if len(node.args) == 5 else None


def _is_bf16_tensor(value: Argument, *, ndim: int) -> bool:
    if not isinstance(value, fx.Node):
        return False
    tensor = value.meta.get("val")
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.dtype == torch.bfloat16
        and tensor.ndim == ndim
    )
