# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

from tests.compile.passes.rocm_aiter_test_utils import (
    configure_therock_rocm_environment,
)


configure_therock_rocm_environment()

import sys
import types

import pytest
import torch
from torch import fx
from torch.fx.experimental.proxy_tensor import make_fx

import vllm.config
import vllm._aiter_ops as aiter_ops_module
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.compilation.passes.fusion.rocm_aiter_residual_rmsnorm_requant_mm_fusion import (  # noqa: E501
    GEMM_OP,
    GEMM_OP_PACKET,
    QUANT_OP,
    QUANT_OP_PACKET,
    RocmAiterHipBMMResidualRMSNormRequantMMFusionPass,
)
from vllm.compilation.passes.inductor_pass import pass_context
from vllm.compilation.passes.ir.inplace_functionalization import (
    VllmIRInplaceFunctionalizationPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.config import CompilationConfig, CompilationMode, VllmConfig
from vllm.config.utils import Range
from vllm.platforms import current_platform


FP8_DTYPE = current_platform.fp8_dtype()
HIDDEN_SIZE = 16
MLP_SIZE = 64
TOKENS = 5
PADDED_GEMM_TOKENS = TOKENS + 1
EPSILON = 1e-5
NUM_LONG_CHAINS = 2
NUM_PREFIX_CHAINS = 1


def _llama_long_chain(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    quant_scale: torch.Tensor,
    next_weight: torch.Tensor,
    next_weight_scale: torch.Tensor,
    view_before_quant: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    gemm = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
        input, weight, torch.bfloat16, input_scale, weight_scale, None
    )
    gemm = torch.narrow(gemm, 0, 0, residual.shape[0]).view(residual.shape)
    norm, residual_out = torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace(
        gemm, residual, rms_weight, EPSILON, None
    )
    if view_before_quant:
        norm = norm.view(-1, HIDDEN_SIZE)
    quant, scale = torch.ops.vllm.rocm_aiter_per_tensor_quant.default(
        norm, FP8_DTYPE, quant_scale
    )
    next_gemm = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
        quant, next_weight, torch.bfloat16, scale, next_weight_scale, None
    )
    next_gemm = torch.narrow(next_gemm, 0, 0, quant.shape[0]).view(TOKENS, MLP_SIZE)
    return next_gemm, residual_out


def _llama_prefix_chain(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    quant_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gemm = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
        input, weight, torch.bfloat16, input_scale, weight_scale, None
    )
    gemm = torch.narrow(gemm, 0, 0, residual.shape[0]).view(residual.shape)
    norm, residual_out = torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace(
        gemm, residual, rms_weight, EPSILON, None
    )
    quant, scale = torch.ops.vllm.rocm_aiter_per_tensor_quant.default(
        norm, FP8_DTYPE, quant_scale
    )
    return quant, scale, residual_out


def _long_chain_with_extra_quant_use(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    quant_scale: torch.Tensor,
    next_weight: torch.Tensor,
    next_weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gemm = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
        input, weight, torch.bfloat16, input_scale, weight_scale, None
    )
    gemm = torch.narrow(gemm, 0, 0, residual.shape[0]).view(residual.shape)
    norm, residual_out = torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace(
        gemm, residual, rms_weight, EPSILON, None
    )
    quant, scale = torch.ops.vllm.rocm_aiter_per_tensor_quant.default(
        norm, FP8_DTYPE, quant_scale
    )
    next_gemm = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
        quant, next_weight, torch.bfloat16, scale, next_weight_scale, None
    )
    next_gemm = torch.narrow(next_gemm, 0, 0, quant.shape[0]).view(TOKENS, MLP_SIZE)
    return next_gemm, residual_out, quant


def _long_chain_with_non_identity_leading_narrow(
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
    gemm = torch.narrow(gemm, 0, 0, residual.shape[0]).view(residual.shape)
    norm, residual_out = torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace(
        gemm, residual, rms_weight, EPSILON, None
    )
    quant, scale = torch.ops.vllm.rocm_aiter_per_tensor_quant.default(
        norm, FP8_DTYPE, quant_scale
    )
    next_gemm = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl(
        quant, next_weight, torch.bfloat16, scale, next_weight_scale, None
    )
    next_gemm = torch.narrow(next_gemm, 0, 0, quant.shape[0]).view(TOKENS, MLP_SIZE)
    return next_gemm, residual_out


def _long_then_prefix_chains(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    quant_scale: torch.Tensor,
    mlp_weight: torch.Tensor,
    mlp_weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    # Small production-shaped graph:
    #   NUM_LONG_CHAINS copies of
    #     GEMM -> RESIDUAL_ADD -> RMSNORM -> FP8_REQUANT -> GEMM
    #   followed by NUM_PREFIX_CHAINS copies of
    #     GEMM -> RESIDUAL_ADD -> RMSNORM -> FP8_REQUANT
    #
    # The production pass should rewrite only the long chains. The standalone
    # prefix chain remains available for the existing RMSNorm+quant passes.
    outputs: list[torch.Tensor] = []
    for index in range(NUM_LONG_CHAINS):
        next_out, next_residual = _llama_long_chain(
            input,
            weight,
            input_scale,
            weight_scale,
            residual.clone(),
            rms_weight,
            quant_scale,
            mlp_weight,
            mlp_weight_scale,
            view_before_quant=(index == 0),
        )
        outputs.extend([next_out, next_residual])
    outputs.extend(
        _llama_prefix_chain(
            input,
            weight,
            input_scale,
            weight_scale,
            residual.clone(),
            rms_weight,
            quant_scale,
        )
    )
    return tuple(outputs)


def _example_inputs(input_tokens: int = TOKENS) -> list[torch.Tensor]:
    return [
        torch.empty(input_tokens, HIDDEN_SIZE, dtype=FP8_DTYPE, device="cuda"),
        torch.empty(HIDDEN_SIZE, HIDDEN_SIZE, dtype=FP8_DTYPE, device="cuda"),
        torch.empty(1, dtype=torch.float32, device="cuda"),
        torch.empty(1, dtype=torch.float32, device="cuda"),
        torch.empty(TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"),
        torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"),
        torch.empty(1, dtype=torch.float32, device="cuda"),
        torch.empty(HIDDEN_SIZE, MLP_SIZE, dtype=FP8_DTYPE, device="cuda"),
        torch.empty(1, dtype=torch.float32, device="cuda"),
    ]


def _count_targets(graph: fx.Graph, targets: set[object]) -> int:
    return sum(1 for node in graph.nodes if node.target in targets)


def _run_noop_cleanup_and_fusion(graph: fx.Graph, vllm_config: VllmConfig) -> None:
    VllmIRInplaceFunctionalizationPass(vllm_config)(graph)
    NoOpEliminationPass(vllm_config)(graph)
    fusion_pass = RocmAiterHipBMMResidualRMSNormRequantMMFusionPass(vllm_config)
    fusion_pass(graph)


def test_rocm_aiter_residual_rmsnorm_requant_mm_impl_calls_aiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[torch.Tensor, ...]] = []

    def fake_hipb_mm_residual_rmsnorm_mm(
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        mat3: torch.Tensor,
        scaleA0: torch.Tensor,
        scaleB0: torch.Tensor,
        requantScale: torch.Tensor,
        scaleA1: torch.Tensor,
        scaleB1: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(
            (
                mat1,
                mat2,
                residual,
                gamma,
                mat3,
                scaleA0,
                scaleB0,
                requantScale,
                scaleA1,
                scaleB1,
            )
        )
        output = torch.empty((mat1.shape[0], mat3.shape[1]), dtype=residual.dtype)
        return output, torch.empty_like(residual)

    aiter_module = types.ModuleType("aiter")
    aiter_module.__path__ = []
    fake_aiter_ops_module = types.ModuleType("aiter.ops")
    fake_aiter_ops_module.__path__ = []
    gradlib_module = types.ModuleType("aiter.ops.gradlib")
    gradlib_module.hipb_mm_residual_rmsnorm_mm = (
        fake_hipb_mm_residual_rmsnorm_mm
    )
    aiter_module.ops = fake_aiter_ops_module
    fake_aiter_ops_module.gradlib = gradlib_module
    monkeypatch.setitem(sys.modules, "aiter", aiter_module)
    monkeypatch.setitem(sys.modules, "aiter.ops", fake_aiter_ops_module)
    monkeypatch.setitem(sys.modules, "aiter.ops.gradlib", gradlib_module)
    monkeypatch.setattr(
        aiter_ops_module, "_ensure_hipb_mm_extension_initialized", lambda: None
    )

    mat1 = torch.empty(TOKENS, HIDDEN_SIZE, dtype=FP8_DTYPE)
    mat2 = torch.empty(HIDDEN_SIZE, HIDDEN_SIZE, dtype=FP8_DTYPE)
    scaleA0 = torch.empty(1, dtype=torch.float32)
    scaleB0 = torch.empty(1, dtype=torch.float32)
    residual = torch.empty(TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16)
    gamma = torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16)
    requant_scale = torch.empty(1, dtype=torch.float32)
    mat3 = torch.empty(HIDDEN_SIZE, MLP_SIZE, dtype=FP8_DTYPE)
    scaleB1 = torch.empty(1, dtype=torch.float32)

    output, residual_out = (
        aiter_ops_module._rocm_aiter_hipb_mm_residual_rmsnorm_requant_mm_impl(
            mat1,
            mat2,
            scaleA0,
            scaleB0,
            residual,
            gamma,
            requant_scale,
            mat3,
            scaleB1,
            EPSILON,
        )
    )

    assert calls == [
        (
            mat1,
            mat2,
            residual,
            gamma,
            mat3,
            scaleA0,
            scaleB0,
            requant_scale,
            requant_scale,
            scaleB1,
        )
    ]
    assert output.shape == (TOKENS, MLP_SIZE)
    assert output.dtype == torch.bfloat16
    assert residual_out.shape == residual.shape
    assert residual_out.dtype == residual.dtype


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)
def test_rocm_aiter_residual_rmsnorm_requant_mm_fusion_replaces_long_chain() -> None:
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
    )
    fused_op = rocm_aiter_ops.get_hipb_mm_residual_rmsnorm_requant_mm_op()

    def f(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _long_then_prefix_chains(*args)

    with (
        vllm.config.set_current_vllm_config(vllm_config),
        pass_context(Range(start=1, end=8)),
    ):
        graph_module = make_fx(f, tracing_mode="fake")(*_example_inputs())
        _run_noop_cleanup_and_fusion(graph_module.graph, vllm_config)

    assert _count_targets(graph_module.graph, {fused_op}) == NUM_LONG_CHAINS
    fused_nodes = [
        node for node in graph_module.graph.nodes if node.target == fused_op
    ]
    for fused_node in fused_nodes:
        # The leading GEMM input should flow directly into the fixed-function
        # fused op. Identity scaled-mm wrapper slices/views are removed by
        # NoOpEliminationPass; this fusion no longer inserts or commutes a
        # replacement slice around AITER.
        leading_input = fused_node.args[0]
        assert isinstance(leading_input, fx.Node)
        assert leading_input.op == "placeholder"
    assert _count_targets(graph_module.graph, {QUANT_OP, QUANT_OP_PACKET}) == (
        NUM_PREFIX_CHAINS
    )
    assert _count_targets(graph_module.graph, {GEMM_OP, GEMM_OP_PACKET}) == (
        NUM_PREFIX_CHAINS
    )


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)
def test_rocm_aiter_residual_rmsnorm_requant_mm_fusion_skips_shared_quant() -> None:
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
    )
    fused_op = rocm_aiter_ops.get_hipb_mm_residual_rmsnorm_requant_mm_op()

    def f(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _long_chain_with_extra_quant_use(*args)

    with (
        vllm.config.set_current_vllm_config(vllm_config),
        pass_context(Range(start=1, end=8)),
    ):
        graph_module = make_fx(f, tracing_mode="fake")(*_example_inputs())
        _run_noop_cleanup_and_fusion(graph_module.graph, vllm_config)

    assert _count_targets(graph_module.graph, {fused_op}) == 0
    assert _count_targets(graph_module.graph, {QUANT_OP, QUANT_OP_PACKET}) == 1
    assert _count_targets(graph_module.graph, {GEMM_OP, GEMM_OP_PACKET}) == 2


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)
def test_rocm_aiter_residual_rmsnorm_requant_mm_fusion_skips_real_narrow() -> None:
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
    )
    fused_op = rocm_aiter_ops.get_hipb_mm_residual_rmsnorm_requant_mm_op()

    def f(*args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _long_chain_with_non_identity_leading_narrow(*args)

    with (
        vllm.config.set_current_vllm_config(vllm_config),
        pass_context(Range(start=1, end=8)),
    ):
        graph_module = make_fx(f, tracing_mode="fake")(
            *_example_inputs(PADDED_GEMM_TOKENS)
        )
        _run_noop_cleanup_and_fusion(graph_module.graph, vllm_config)

    assert _count_targets(graph_module.graph, {fused_op}) == 0
    assert _count_targets(graph_module.graph, {QUANT_OP, QUANT_OP_PACKET}) == 1
    assert _count_targets(graph_module.graph, {GEMM_OP, GEMM_OP_PACKET}) == 2
