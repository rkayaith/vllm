# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm MXFP8 GEMM/residual RMSNorm/requant rewrite."""

import operator
from dataclasses import dataclass

import torch
from torch import fx
from torch._ops import OpOverload
from torch.fx.node import Argument

import vllm.ir.ops  # noqa: F401
from vllm.config import VllmConfig
from vllm.model_executor.kernels.linear.mxfp8 import rocm_native  # noqa: F401
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize_fake,
)
from vllm.utils.torch_utils import direct_register_custom_op

from ..fx_utils import find_getitem_maybe, is_func
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

MXFP8_GEMM_OP = torch.ops.vllm.rocm_mxfp8_dot_scaled_mm.default
FUSED_ADD_RMS_NORM_OP = torch.ops.vllm_ir.fused_add_rms_norm.default
MXFP8_QUANT_OP = torch.ops.vllm.mxfp8_quantize.default

REFERENCE_OP_NAME = "rocm_mxfp8_gemm_rmsnorm_quant_reference"
PASS_NAME = "rocm_mxfp8_gemm_rmsnorm_quant_fusion_pass"
QWEN3_EPSILON = 1e-6


def rocm_mxfp8_gemm_rmsnorm_quant_reference(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    epsilon: float,
    variance_size: int | None,
    is_sf_swizzled_layout: bool,
    alignment: int,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the Qwen3-32B MXFP8 candidate with existing operations.

    This temporary correctness implementation exposes the contract expected
    from a future fused AITER kernel. It intentionally keeps the three stages
    explicit and delegates each stage to the same operation as the unfused
    model.
    """
    # Stage 1: run the existing gfx950 tl.dot_scaled GEMM on MXFP8 values and
    # their row-major E8M0 scales.
    gemm = MXFP8_GEMM_OP(
        input,
        input_scale,
        weight,
        weight_scale,
        output_dtype,
    )

    # Stage 2: add the running residual and normalize with the same vllm_ir op
    # as the original graph. The updated residual is discarded here because
    # the original RMSNorm remains live and continues the decoder residual.
    normalized, _ = FUSED_ADD_RMS_NORM_OP(
        gemm,
        residual,
        rms_weight,
        epsilon,
        variance_size,
    )

    # Stage 3: requantize the normalized activation into FP8 E4M3 values with
    # one E8M0 scale byte per 32 values, matching the next MXFP8 GEMM input.
    return MXFP8_QUANT_OP(normalized, is_sf_swizzled_layout, alignment)


def rocm_mxfp8_gemm_rmsnorm_quant_reference_fake(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    epsilon: float,
    variance_size: int | None,
    is_sf_swizzled_layout: bool,
    alignment: int,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    del input, input_scale, weight, weight_scale, rms_weight, epsilon
    del variance_size, output_dtype
    return mxfp8_e4m3_quantize_fake(
        residual,
        is_sf_swizzled_layout,
        alignment,
    )


direct_register_custom_op(
    op_name=REFERENCE_OP_NAME,
    op_func=rocm_mxfp8_gemm_rmsnorm_quant_reference,
    fake_impl=rocm_mxfp8_gemm_rmsnorm_quant_reference_fake,
)
REFERENCE_OP: OpOverload = getattr(torch.ops.vllm, REFERENCE_OP_NAME).default


@dataclass(frozen=True)
class _Qwen3Mxfp8FusionMatch:
    gemm: fx.Node
    rmsnorm: fx.Node
    quant: fx.Node
    quantized: fx.Node
    scales: fx.Node


class RocmMxfp8GemmRMSNormQuantFusionPass(VllmInductorPass):
    """Replace Qwen3-32B's MXFP8 GEMM/RMSNorm/requant activation path.

    Only the quantized activation and its E8M0 scales are redirected to the
    reference node. The original GEMM and fused residual/RMSNorm remain live,
    preserving the decoder residual and any independent normalized consumers.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.pass_name = PASS_NAME
        self.matched_count = 0

    def uuid(self) -> str:
        return self.hash_source(
            self,
            f"{REFERENCE_OP_NAME}:{QWEN3_EPSILON}",
            rocm_mxfp8_gemm_rmsnorm_quant_reference,
            rocm_mxfp8_gemm_rmsnorm_quant_reference_fake,
            _match_qwen3_mxfp8_chain,
            _replace_quant_outputs,
            _is_qwen3_mxfp8_gemm,
            _is_qwen3_rmsnorm,
            _is_qwen3_mxfp8_quant,
            _rmsnorm_variance_size,
            _quant_is_sf_swizzled_layout,
            _quant_alignment,
        )

    @VllmInductorPass.time_and_log  # pyright: ignore[reportArgumentType]
    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, graph: fx.Graph
    ) -> None:
        self.matched_count = 0
        quant_nodes = [node for node in graph.nodes if _is_qwen3_mxfp8_quant(node)]
        for quant in quant_nodes:
            match = _match_qwen3_mxfp8_chain(quant)
            if match is None:
                continue
            _replace_quant_outputs(graph, match)
            self.matched_count += 1

        VllmPatternMatcherPass.match_table[self.pass_name] += self.matched_count
        graph.eliminate_dead_code()
        graph.lint()


def _match_qwen3_mxfp8_chain(quant: fx.Node) -> _Qwen3Mxfp8FusionMatch | None:
    """Walk backward from MXFP8 requant through the cleaned Qwen3 chain."""
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

    gemm = rmsnorm.args[0]
    if not isinstance(gemm, fx.Node) or not _is_qwen3_mxfp8_gemm(gemm):
        return None

    quantized = find_getitem_maybe(quant, 0)
    scales = find_getitem_maybe(quant, 1)
    if quantized is None or scales is None:
        return None

    # Both tuple outputs must be known so replacing them makes the standalone
    # requant node dead. Independent users of the normalized BF16 value remain
    # valid because the original RMSNorm is deliberately not replaced.
    if set(quant.users) != {quantized, scales}:
        return None

    return _Qwen3Mxfp8FusionMatch(
        gemm=gemm,
        rmsnorm=rmsnorm,
        quant=quant,
        quantized=quantized,
        scales=scales,
    )


def _replace_quant_outputs(
    graph: fx.Graph,
    match: _Qwen3Mxfp8FusionMatch,
) -> None:
    """Insert the reference node and redirect only the requantized outputs."""
    gemm = match.gemm
    rmsnorm = match.rmsnorm

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
                _rmsnorm_variance_size(rmsnorm),
                _quant_is_sf_swizzled_layout(match.quant),
                _quant_alignment(match.quant),
                gemm.args[4],
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


def _is_qwen3_mxfp8_gemm(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == MXFP8_GEMM_OP
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


def _is_qwen3_mxfp8_quant(node: fx.Node) -> bool:
    if not (
        node.op == "call_function"
        and node.target == MXFP8_QUANT_OP
        and len(node.args) in {1, 2, 3}
        and _quant_is_sf_swizzled_layout(node) is False
    ):
        return False
    alignment = _quant_alignment(node)
    return isinstance(alignment, int) and alignment == 0


def _rmsnorm_variance_size(node: fx.Node) -> Argument:
    return node.args[4] if len(node.args) == 5 else None


def _quant_is_sf_swizzled_layout(node: fx.Node) -> Argument:
    return node.args[1] if len(node.args) >= 2 else False


def _quant_alignment(node: fx.Node) -> Argument:
    return node.args[2] if len(node.args) == 3 else 0
