# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm AITER fused GEMM/RMSNorm/requant/GEMM rewrite."""

import operator

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._inductor.pattern_matcher import Match, PatternMatcherPass

import vllm.ir.ops  # noqa: F401
import vllm.model_executor.kernels.linear.scaled_mm.rocm  # noqa: F401
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.platforms import current_platform

from ..fx_utils import is_func
from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass


FP8_DTYPE = current_platform.fp8_dtype()
GEMM_OP = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl.default
GEMM_OP_PACKET = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl
QUANT_OP = torch.ops.vllm.rocm_aiter_per_tensor_quant.default
QUANT_OP_PACKET = torch.ops.vllm.rocm_aiter_per_tensor_quant
FUSED_ADD_RMS_NORM_OP = torch.ops.vllm_ir.fused_add_rms_norm.default
FUSED_ADD_RMS_NORM_PACKET = torch.ops.vllm_ir.fused_add_rms_norm
PASS_NAME = "rocm_aiter_hipb_mm_residual_rmsnorm_requant_mm_fusion_pass"


class RocmAiterHipBMMResidualRMSNormRequantMMFusionPass(VllmPatternMatcherPass):
    """Replace ``GEMM -> residual RMSNorm -> static FP8 requant -> GEMM``.

    This intentionally targets the Llama-shaped static FP8 per-tensor chain
    that AITER's fixed-function ``hipb_mm_residual_rmsnorm_mm`` launcher
    supports today. It does not try to be a general GEMM/RMSNorm fusion pass:
    all matched internal values must be private because the fused op returns
    only the trailing GEMM output and the updated residual.

    ``VllmIRInplaceFunctionalizationPass`` runs as a pre-grad pass and changes
    ``fused_add_rms_norm.maybe_inplace`` to the functional default overload
    before this post-grad pattern runs. That is required for Inductor's
    ``PatternMatcherPass``: its built-in mutation-region guard rejects patterns
    that span both sides of a mutating node. This pass therefore registers only
    the functional/default RMSNorm form used by the production post-grad graph.

    ``NoOpEliminationPass`` then runs before this pass in the post-grad
    pipeline and removes identity scaled-mm ``narrow``/``view`` wrappers using
    fake tensor shape equality. This pass intentionally relies on that
    centralized cleanup instead of proving identity wrappers itself. If a
    wrapper remains inside the fused region, it is treated as semantic and the
    fixed-function AITER op is not a valid replacement.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.pass_name = PASS_NAME
        self.fused_op = rocm_aiter_ops.get_hipb_mm_residual_rmsnorm_requant_mm_op()
        self.patterns = PatternMatcherPass(pass_name=PASS_NAME)

        # Llama post-grad graphs observed for this fixed-function integration
        # use 1e-5; keep 1e-6 because the adjacent RMSNorm+quant fusions already
        # cover both common vLLM epsilons and the AITER launcher takes epsilon
        # as a scalar argument.
        for epsilon in [1e-5, 1e-6]:
            _register_pattern(self.patterns, self.fused_op, epsilon)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        VllmPatternMatcherPass.match_table[self.pass_name] += self.matched_count
        graph.eliminate_dead_code()
        graph.lint()


def _register_pattern(
    pm_pass: PatternMatcherPass,
    fused_op: object,
    epsilon: float,
) -> None:
    # Match the long chain from its final consumer:
    #
    #   leading GEMM
    #     -> fused_add_rms_norm(..., residual, rms_weight, eps)
    #     -> getitem(..., 0)
    #     -> static FP8 requant
    #     -> getitem(quant, 0), getitem(quant, 1)
    #     -> trailing GEMM
    #
    # Registering the larger pattern gives this fusion first chance to consume
    # the RMSNorm/requant region before the existing RMSNorm+quant passes see
    # the overlapping shorter pattern.
    def pattern(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        residual: torch.Tensor,
        rms_weight: torch.Tensor,
        quant_scale: torch.Tensor,
        next_weight: torch.Tensor,
        next_weight_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gemm = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
            input, weight, torch.bfloat16, input_scale, weight_scale, None
        )
        norm, residual_out = torch.ops.vllm_ir.fused_add_rms_norm.default(
            gemm, residual, rms_weight, epsilon
        )
        quant, scale = torch.ops.vllm.rocm_aiter_per_tensor_quant.default(
            norm, FP8_DTYPE, quant_scale
        )
        next_output = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
            quant, next_weight, torch.bfloat16, scale, next_weight_scale, None
        )
        return next_output, residual_out

    def replacement(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        residual: torch.Tensor,
        rms_weight: torch.Tensor,
        quant_scale: torch.Tensor,
        next_weight: torch.Tensor,
        next_weight_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The fused op takes exactly the dataflow values in the original chain:
        #   leading GEMM activation/weight/scales,
        #   residual and RMSNorm parameters,
        #   static FP8 requant scale,
        #   trailing GEMM weight/scale,
        #   RMS epsilon and GEMM output dtype.
        #
        # It returns:
        #   0. trailing GEMM output
        #   1. updated residual from fused_add_rms_norm
        fused = fused_op(
            input,
            weight,
            input_scale,
            weight_scale,
            residual,
            rms_weight,
            quant_scale,
            next_weight,
            next_weight_scale,
            epsilon,
            torch.bfloat16,
        )
        return fused[0], fused[1]

    inputs = [
        torch.empty(5, 16, dtype=FP8_DTYPE, device="cuda"),
        torch.empty(16, 16, dtype=FP8_DTYPE, device="cuda"),
        torch.empty(1, dtype=torch.float32, device="cuda"),
        torch.empty(1, dtype=torch.float32, device="cuda"),
        torch.empty(5, 16, dtype=torch.bfloat16, device="cuda"),
        torch.empty(16, dtype=torch.bfloat16, device="cuda"),
        torch.empty(1, dtype=torch.float32, device="cuda"),
        torch.empty(16, 64, dtype=FP8_DTYPE, device="cuda"),
        torch.empty(1, dtype=torch.float32, device="cuda"),
    ]

    pm.register_replacement(
        pattern,
        replacement,
        inputs,
        pm.fwd_only,
        pm_pass,
        extra_check=_match_outputs_are_private,
    )


def _match_outputs_are_private(match: Match) -> bool:
    output_nodes = match.output_nodes()
    if len(output_nodes) != 2:
        return False
    trailing_gemm, residual_output = output_nodes
    if not (
        isinstance(trailing_gemm, fx.Node)
        and isinstance(residual_output, fx.Node)
        and _is_gemm_op(trailing_gemm)
        and is_func(residual_output, operator.getitem)
        and residual_output.args[1] == 1
    ):
        return False

    quant_output = trailing_gemm.args[0]
    scale_output = trailing_gemm.args[3]
    if not (
        isinstance(quant_output, fx.Node)
        and isinstance(scale_output, fx.Node)
        and is_func(quant_output, operator.getitem)
        and is_func(scale_output, operator.getitem)
        and quant_output.args[1] == 0
        and scale_output.args[1] == 1
        and quant_output.args[0] is scale_output.args[0]
        and isinstance(quant_output.args[0], fx.Node)
        and _is_quant_op(quant_output.args[0])
    ):
        return False
    quant_node = quant_output.args[0]

    norm_getitem = quant_node.args[0]
    if (
        not isinstance(norm_getitem, fx.Node)
        or not is_func(norm_getitem, operator.getitem)
        or norm_getitem.args[1] != 0
    ):
        return False

    rms = norm_getitem.args[0]
    if (
        not isinstance(rms, fx.Node)
        or not _is_fused_add_rms_norm_op(rms)
        or residual_output.args[0] is not rms
    ):
        return False

    # PatternMatcherPass proves the dataflow shape; this extra_check proves the
    # replacement contract. The fixed-function AITER op returns only the
    # trailing GEMM output and updated residual, so every other matched value
    # must be private to the region. Shared quant, norm, or leading-GEMM values
    # are left for the existing smaller fusions.
    if not _has_only_user(norm_getitem, quant_node):
        return False
    if not (
        _has_only_user(quant_output, trailing_gemm)
        and _has_only_user(scale_output, trailing_gemm)
        and set(quant_node.users) == {quant_output, scale_output}
    ):
        return False

    # NoOpEliminationPass owns identity ``narrow``/``view`` cleanup. Do not
    # unwrap or commute shape/indexing ops here: AITER's fixed-function fused op
    # does not expose a slicing/view contract, so a surviving wrapper in the
    # fused region must block this rewrite.
    leading_gemm = rms.args[0]
    if not isinstance(leading_gemm, fx.Node) or not _is_gemm_op(leading_gemm):
        return False
    if not _has_only_user(leading_gemm, rms):
        return False

    # The fused op preserves the residual stream but not the RMS tuple itself.
    # Llama's matched region has exactly getitem(0) for norm and getitem(1)
    # for residual_out; any extra RMS user would need a different fused API.
    return set(rms.users) == {norm_getitem, residual_output}


def _has_only_user(node: fx.Node, user: fx.Node) -> bool:
    return len(node.users) == 1 and user in node.users


def _is_gemm_op(node: fx.Node) -> bool:
    # Keep this tied to the Llama static-FP8 path that produced the validated
    # opportunity: vLLM's ROCm scaled-mm custom op, bf16 output, and no bias.
    # Supporting other GEMM wrappers or bias would require extending the AITER
    # fused launcher contract and the replacement argument list together.
    return (
        node.op == "call_function"
        and node.target in {GEMM_OP, GEMM_OP_PACKET}
        and len(node.args) == 6
        and node.args[2] == torch.bfloat16
        and node.args[5] is None
    )


def _is_quant_op(node: fx.Node) -> bool:
    # The AITER fused op currently models static per-tensor requant: the quant
    # op receives the caller-provided scale and returns that same scale for the
    # next GEMM. Dynamic or group quant schemes need different scale semantics
    # and are intentionally left to the existing RMSNorm+quant passes.
    return (
        node.op == "call_function"
        and node.target in {QUANT_OP, QUANT_OP_PACKET}
        and len(node.args) == 3
        and node.args[1] == FP8_DTYPE
    )


def _is_fused_add_rms_norm_op(node: fx.Node) -> bool:
    # Production post-grad graphs have already passed through
    # VllmIRInplaceFunctionalizationPass, so matching the functional/default
    # overload is both sufficient and important: a raw maybe_inplace graph would
    # cross Inductor's mutation-region barrier because this fusion also consumes
    # the leading GEMM before the RMSNorm.
    return (
        node.op == "call_function"
        and node.target in {FUSED_ADD_RMS_NORM_OP, FUSED_ADD_RMS_NORM_PACKET}
        and len(node.args) == 4
    )
