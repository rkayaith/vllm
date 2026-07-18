# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation-only fused markers for ROCm FP8 GEMM/RMSNorm/quant chains.

The passes in this file rewrite the Llama-shaped FX chains into custom ops
that call the original operations. They are probes for validating that the
candidate fused-op regions can be replaced before implementing a real kernel.

The two passes are intended to run in this order:
  1. ``RocmAiterGemmRMSQuantGemmValidationPass``
  2. ``RocmAiterGemmRMSQuantValidationPass``

The long pass must run first because every long chain contains the prefix
chain as a subregion. Running the prefix pass first would consume the requant
nodes and prevent the larger replacement from being validated.
"""

import operator
from typing import Any

import torch
from torch import fx

import vllm.ir.ops  # noqa: F401
import vllm.model_executor.kernels.linear.scaled_mm.rocm  # noqa: F401
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from ..fx_utils import find_getitem_maybe, is_func
from ..vllm_inductor_pass import VllmInductorPass


FP8_DTYPE = current_platform.fp8_dtype()
GEMM_OP = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl.default
GEMM_OP_PACKET = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl
QUANT_OP = torch.ops.vllm.rocm_aiter_per_tensor_quant.default
QUANT_OP_PACKET = torch.ops.vllm.rocm_aiter_per_tensor_quant
FUSED_ADD_RMS_NORM_OP = torch.ops.vllm_ir.fused_add_rms_norm.default
FUSED_ADD_RMS_NORM_OP_PACKET = torch.ops.vllm_ir.fused_add_rms_norm
FUSED_ADD_RMS_NORM_MAYBE_INPLACE_OP = (
    torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace
)

PREFIX_FUSED_OP_NAME = "rocm_aiter_gemm_rms_quant_fused_validation"
LONG_FUSED_OP_NAME = "rocm_aiter_gemm_rms_quant_gemm_fused_validation"
PREFIX_PASS_NAME = "rocm_aiter_gemm_rms_quant_validation_pass"
LONG_PASS_NAME = "rocm_aiter_gemm_rms_quant_gemm_validation_pass"


def rocm_aiter_gemm_rms_quant_fused_validation(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    quant_scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gemm = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
        input, weight, torch.bfloat16, input_scale, weight_scale, None
    )
    gemm = torch.narrow(gemm, 0, 0, residual.shape[0]).view(residual.shape)
    norm, residual_out = torch.ops.vllm_ir.fused_add_rms_norm.default(
        gemm, residual, rms_weight, eps, None
    )
    norm = norm.view(-1, residual.shape[-1])
    quant, scale = torch.ops.vllm.rocm_aiter_per_tensor_quant.default(
        norm, FP8_DTYPE, quant_scale
    )
    return quant, scale, residual_out


def rocm_aiter_gemm_rms_quant_fused_validation_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    quant_scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quant = torch.empty(
        (residual.numel() // residual.shape[-1], residual.shape[-1]),
        device=residual.device,
        dtype=FP8_DTYPE,
    )
    return quant, torch.empty_like(quant_scale), torch.empty_like(residual)


def rocm_aiter_gemm_rms_quant_gemm_fused_validation(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    quant_scale: torch.Tensor,
    next_weight: torch.Tensor,
    next_weight_scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    quant, scale, residual_out = rocm_aiter_gemm_rms_quant_fused_validation(
        input,
        weight,
        input_scale,
        weight_scale,
        residual,
        rms_weight,
        quant_scale,
        eps,
    )
    next_gemm = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
        quant, next_weight, torch.bfloat16, scale, next_weight_scale, None
    )
    next_gemm = torch.narrow(next_gemm, 0, 0, quant.shape[0]).view(
        *quant.shape[:-1], next_weight.shape[1]
    )
    return next_gemm, residual_out


def rocm_aiter_gemm_rms_quant_gemm_fused_validation_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    quant_scale: torch.Tensor,
    next_weight: torch.Tensor,
    next_weight_scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    next_output = torch.empty(
        (*residual.shape[:-1], next_weight.shape[1]),
        device=residual.device,
        dtype=torch.bfloat16,
    )
    return next_output, torch.empty_like(residual)


direct_register_custom_op(
    op_name=PREFIX_FUSED_OP_NAME,
    op_func=rocm_aiter_gemm_rms_quant_fused_validation,
    fake_impl=rocm_aiter_gemm_rms_quant_fused_validation_fake,
)

direct_register_custom_op(
    op_name=LONG_FUSED_OP_NAME,
    op_func=rocm_aiter_gemm_rms_quant_gemm_fused_validation,
    fake_impl=rocm_aiter_gemm_rms_quant_gemm_fused_validation_fake,
)

PREFIX_FUSED_OP = getattr(torch.ops.vllm, PREFIX_FUSED_OP_NAME).default
LONG_FUSED_OP = getattr(torch.ops.vllm, LONG_FUSED_OP_NAME).default


class RocmAiterGemmRMSQuantValidationPass(VllmInductorPass):
    """Replace ``GEMM -> residual RMSNorm -> static FP8 requant``."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.pass_name = PREFIX_PASS_NAME

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self._replace_matches(graph)

    def _replace_matches(self, graph: fx.Graph) -> None:
        # Prefix chain:
        #   GEMM -> RESIDUAL_ADD -> RMSNORM -> FP8_REQUANT
        #
        # Match from the consuming requant op and walk through producers:
        #   requant input -> optional view -> RMSNorm result getitem
        #   -> fused_add_rms_norm
        #   fused_add_rms_norm input -> prior GEMM
        #
        # Production runs NoOpEliminationPass before these validation markers.
        # That centralized cleanup removes identity scaled-mm wrapper
        # narrow/view nodes. If a wrapper survives, it is semantic and must not
        # be counted as a fixed-function fused-op candidate.
        #
        # This is the fallback replacement after the long pass has already had
        # the chance to consume any chain with a trailing GEMM. In the current
        # Llama model-forward probe, all visible prefix candidates are part of
        # long chains; the expected standalone prefix candidate belongs to the
        # final norm/lm_head path, which is outside that probe.
        for quant_node in [node for node in graph.nodes if _is_quant_op(node)]:
            match = _match_llama_prefix_from_quant(quant_node)
            if match is None:
                continue
            _replace_prefix_match(graph, match)

        graph.eliminate_dead_code()
        graph.lint()


class RocmAiterGemmRMSQuantGemmValidationPass(RocmAiterGemmRMSQuantValidationPass):
    """Replace ``GEMM -> residual RMSNorm -> static FP8 requant -> GEMM``."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.pass_name = LONG_PASS_NAME

    def _replace_matches(self, graph: fx.Graph) -> None:
        # Long chain:
        #   GEMM -> RESIDUAL_ADD -> RMSNORM -> FP8_REQUANT -> GEMM
        #
        # Anchor at the trailing GEMM, verify that its activation and scale
        # inputs are getitems from the same requant tuple, then reuse the
        # prefix producer walk to prove the requant is fed by the residual
        # RMSNorm chain.
        #
        # Do not restrict the trailing GEMM by projection name or shape here:
        # the Llama analysis intentionally counts both intra-layer chains
        # such as ``o_proj -> post_attention_layernorm -> gate_up_proj`` and
        # cross-layer chains such as
        # ``down_proj -> next input_layernorm -> next qkv_proj``. The model
        # forward graph currently exposes 159 such long chains.
        for trailing_gemm in [node for node in graph.nodes if _is_gemm_op(node)]:
            quant_node = _match_gemm_quant_inputs(trailing_gemm)
            if quant_node is None:
                continue
            match = _match_llama_prefix_from_quant(quant_node)
            if match is None:
                continue
            _replace_long_match(graph, match, trailing_gemm)

        graph.eliminate_dead_code()
        graph.lint()

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self._replace_matches(graph)


class _LlamaPrefixMatch:
    def __init__(
        self,
        *,
        rms_node: fx.Node,
        gemm_node: fx.Node,
        quant_node: fx.Node,
        quant_output: fx.Node,
        scale_output: fx.Node,
        residual_output: fx.Node | None,
    ) -> None:
        self.rms_node = rms_node
        self.gemm_node = gemm_node
        self.quant_node = quant_node
        self.quant_output = quant_output
        self.scale_output = scale_output
        self.residual_output = residual_output


def _replace_prefix_match(graph: fx.Graph, match: _LlamaPrefixMatch) -> None:
    # Replace the three live outputs of the prefix chain:
    #   requant value, requant scale, and updated residual if it is still live.
    #
    # The updated residual getitem is optional for prefix-only replacement.
    # After the long pass runs and dead-code cleanup executes, some candidate
    # RMSNorm tuple outputs may no longer have a live residual getitem. That
    # should not prevent replacing the still-live quant/scale outputs.
    # Dead original GEMM/RMSNorm/requant nodes are erased by dead-code cleanup.
    with graph.inserting_before(match.quant_node):
        fused = graph.call_function(
            PREFIX_FUSED_OP,
            args=_prefix_fused_args(match),
        )
        quant = graph.call_function(operator.getitem, args=(fused, 0))
        scale = graph.call_function(operator.getitem, args=(fused, 1))
        if match.residual_output is not None:
            residual = graph.call_function(operator.getitem, args=(fused, 2))

    # Manually inserted FX nodes do not get fake-value metadata from tracing.
    # Later vLLM IR lowering reads meta["val"] from node inputs, so preserve the
    # metadata of the exact graph values each getitem replaces.
    if "val" in match.quant_output.meta and "val" in match.scale_output.meta:
        fused_values = [match.quant_output.meta["val"], match.scale_output.meta["val"]]
        if match.residual_output is not None and "val" in match.residual_output.meta:
            fused_values.append(match.residual_output.meta["val"])
        elif "val" in match.rms_node.meta:
            fused_values.append(match.rms_node.meta["val"][1])
        if len(fused_values) == 3:
            fused.meta["val"] = tuple(fused_values)
    quant.meta.update(match.quant_output.meta)
    scale.meta.update(match.scale_output.meta)
    if match.residual_output is not None:
        residual.meta.update(match.residual_output.meta)

    match.quant_output.replace_all_uses_with(quant)
    match.scale_output.replace_all_uses_with(scale)
    if match.residual_output is not None:
        match.residual_output.replace_all_uses_with(residual)


def _replace_long_match(
    graph: fx.Graph, match: _LlamaPrefixMatch, trailing_gemm: fx.Node
) -> None:
    # Replace the two live outputs of the long chain:
    #   trailing GEMM output and updated residual.
    # The requant value/scale are internal to the fused marker op.
    #
    # Unlike the prefix replacement, the long marker has to preserve the
    # residual stream that continues through the decoder stack. If that residual
    # output is not live, this is not a valid long-chain replacement.
    if match.residual_output is None:
        return
    prefix_args = _prefix_fused_args(match)
    with graph.inserting_before(trailing_gemm):
        fused = graph.call_function(
            LONG_FUSED_OP,
            args=(
                *prefix_args[:-1],
                trailing_gemm.args[1],
                trailing_gemm.args[4],
                prefix_args[-1],
            ),
        )
        next_output = graph.call_function(operator.getitem, args=(fused, 0))
        residual = graph.call_function(operator.getitem, args=(fused, 1))

    # Preserve fake-value metadata for downstream passes that inspect node args.
    if "val" in trailing_gemm.meta and "val" in match.residual_output.meta:
        fused.meta["val"] = (
            trailing_gemm.meta["val"],
            match.residual_output.meta["val"],
        )
    next_output.meta.update(trailing_gemm.meta)
    residual.meta.update(match.residual_output.meta)

    trailing_gemm.replace_all_uses_with(next_output)
    match.residual_output.replace_all_uses_with(residual)


def _prefix_fused_args(match: _LlamaPrefixMatch) -> tuple[Any, ...]:
    return (
        match.gemm_node.args[0],
        match.gemm_node.args[1],
        match.gemm_node.args[3],
        match.gemm_node.args[4],
        match.rms_node.args[1],
        match.rms_node.args[2],
        match.quant_node.args[2],
        match.rms_node.args[3],
    )


def _match_llama_prefix_from_quant(quant: fx.Node) -> _LlamaPrefixMatch | None:
    # Validate the producer path:
    #   quant(norm_view(getitem(rms, 0)), fp8_dtype, quant_scale)
    #   quant(getitem(rms, 0), fp8_dtype, quant_scale)
    #   rms(gemm(...), residual, weight, eps, None?)
    #
    # The optional trailing GEMM is intentionally not checked here; the long
    # pass checks it at its own consumer anchor before calling this helper.
    # Only the tuple elements needed by replacement must be live:
    #   getitem(quant, 0) and getitem(quant, 1) are required.
    #   getitem(rms, 1) is optional for prefix replacement and required again
    #   by _replace_long_match.
    if not _is_quant_op(quant):
        return None

    norm_getitem = quant.args[0]
    # The pre-split model-forward graph has an explicit view before requant,
    # but vLLM's real post-grad piecewise graphs may have already removed it.
    # Accept only these two Llama shapes:
    #   quant(view(getitem(rms, 0)), fp8_dtype, quant_scale)
    #   quant(getitem(rms, 0), fp8_dtype, quant_scale)
    if isinstance(norm_getitem, fx.Node) and _is_view_op(norm_getitem):
        norm_getitem = norm_getitem.args[0]
    if (
        not isinstance(norm_getitem, fx.Node)
        or not is_func(norm_getitem, operator.getitem)
        or norm_getitem.args[1] != 0
    ):
        return None

    rms = norm_getitem.args[0]
    if not isinstance(rms, fx.Node) or not _is_fused_add_rms_norm_op(rms):
        return None

    gemm = rms.args[0]
    if not isinstance(gemm, fx.Node) or not _is_gemm_op(gemm):
        return None

    quant_output = find_getitem_maybe(quant, 0)
    scale_output = find_getitem_maybe(quant, 1)
    residual = find_getitem_maybe(rms, 1)
    if quant_output is None or scale_output is None:
        return None

    return _LlamaPrefixMatch(
        rms_node=rms,
        gemm_node=gemm,
        quant_node=quant,
        quant_output=quant_output,
        scale_output=scale_output,
        residual_output=residual,
    )


def _match_gemm_quant_inputs(gemm: fx.Node) -> fx.Node | None:
    # Validate the trailing GEMM consumes both activation and scale from the
    # same requant tuple:
    #   gemm(getitem(quant, 0), weight, bf16, getitem(quant, 1), weight_scale, None)
    if not _is_gemm_op(gemm):
        return None
    quant = gemm.args[0]
    scale = gemm.args[3]
    if not (
        isinstance(quant, fx.Node)
        and isinstance(scale, fx.Node)
        and is_func(quant, operator.getitem)
        and is_func(scale, operator.getitem)
        and quant.args[1] == 0
        and scale.args[1] == 1
        and quant.args[0] is scale.args[0]
        and isinstance(quant.args[0], fx.Node)
        and _is_quant_op(quant.args[0])
    ):
        return None
    return quant.args[0]


def _is_gemm_op(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target in {GEMM_OP, GEMM_OP_PACKET}
        and len(node.args) == 6
        and node.args[2] == torch.bfloat16
        and node.args[5] is None
    )


def _is_quant_op(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target in {QUANT_OP, QUANT_OP_PACKET}
        and len(node.args) == 3
        and node.args[1] == FP8_DTYPE
    )


def _is_fused_add_rms_norm_op(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target
        in {
            FUSED_ADD_RMS_NORM_OP,
            FUSED_ADD_RMS_NORM_OP_PACKET,
            FUSED_ADD_RMS_NORM_MAYBE_INPLACE_OP,
        }
        and len(node.args) in {4, 5}
        and (len(node.args) == 4 or node.args[4] is None)
    )


def _is_view_op(node: fx.Node) -> bool:
    return node.target in {
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
        "reshape",
        "view",
    }
