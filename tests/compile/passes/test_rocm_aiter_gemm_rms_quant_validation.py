# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

from tests.compile.passes.rocm_aiter_test_utils import (
    configure_therock_rocm_environment,
)


configure_therock_rocm_environment()

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import vllm.config
from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.compilation.passes.fusion.rocm_aiter_gemm_rms_quant_validation import (
    LONG_FUSED_OP,
    PREFIX_FUSED_OP,
    RocmAiterGemmRMSQuantGemmValidationPass,
    RocmAiterGemmRMSQuantValidationPass,
)
from vllm.compilation.passes.inductor_pass import pass_context
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.config import CompilationConfig, CompilationMode, VllmConfig
from vllm.config.utils import Range
from vllm.platforms import current_platform

FP8_DTYPE = current_platform.fp8_dtype()
HIDDEN_SIZE = 16
MLP_SIZE = 64
TOKENS = 5
EPSILON = 1e-5
NUM_LONG_CHAINS = 2
NUM_PREFIX_CHAINS = 1


def _llama_shaped_chain(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    quant_scale: torch.Tensor,
    next_weight: torch.Tensor,
    next_weight_scale: torch.Tensor,
    next_features: int,
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
    next_gemm = torch.narrow(next_gemm, 0, 0, quant.shape[0]).view(
        TOKENS, next_features
    )
    return next_gemm, residual_out


def _llama_prefix_chain(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    quant_scale: torch.Tensor,
    view_before_quant: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return quant, scale, residual_out


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
    # Small unit-test graph for production-order behavior:
    #   NUM_LONG_CHAINS copies of
    #     GEMM -> RESIDUAL_ADD -> RMSNORM -> FP8_REQUANT -> GEMM
    #   followed by NUM_PREFIX_CHAINS copies of
    #     GEMM -> RESIDUAL_ADD -> RMSNORM -> FP8_REQUANT
    #
    # The long pass should consume only the first group. The prefix pass then
    # runs on the same graph and should consume only the standalone prefix
    # group, proving that we are not counting both passes on independent graph
    # copies.
    outputs: list[torch.Tensor] = []
    for index in range(NUM_LONG_CHAINS):
        next_out, next_residual = _llama_shaped_chain(
            input,
            weight,
            input_scale,
            weight_scale,
            residual.clone(),
            rms_weight,
            quant_scale,
            mlp_weight,
            mlp_weight_scale,
            MLP_SIZE,
            index == 0,
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
            False,
        )
    )
    return tuple(outputs)


def _example_inputs() -> list[torch.Tensor]:
    return [
        torch.empty(TOKENS, HIDDEN_SIZE, dtype=FP8_DTYPE, device="cuda"),
        torch.empty(HIDDEN_SIZE, HIDDEN_SIZE, dtype=FP8_DTYPE, device="cuda"),
        torch.empty(1, dtype=torch.float32, device="cuda"),
        torch.empty(1, dtype=torch.float32, device="cuda"),
        torch.empty(TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"),
        torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"),
        torch.empty(1, dtype=torch.float32, device="cuda"),
        torch.empty(HIDDEN_SIZE, MLP_SIZE, dtype=FP8_DTYPE, device="cuda"),
        torch.empty(1, dtype=torch.float32, device="cuda"),
    ]


def _count_nodes(graph: torch.fx.Graph, target: object) -> int:
    return sum(1 for node in graph.nodes if node.target == target)


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)
def test_rocm_aiter_gemm_rms_quant_validation_replaces_llama_shape() -> None:
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
    )

    def f(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _long_then_prefix_chains(*args)

    with (
        vllm.config.set_current_vllm_config(vllm_config),
        pass_context(Range(start=1, end=8)),
    ):
        graph_module = make_fx(f, tracing_mode="fake")(*_example_inputs())
        NoOpEliminationPass(vllm_config)(graph_module.graph)

        long_pass = RocmAiterGemmRMSQuantGemmValidationPass(vllm_config)
        long_pass(graph_module.graph)
        assert _count_nodes(graph_module.graph, LONG_FUSED_OP) == NUM_LONG_CHAINS

        prefix_pass = RocmAiterGemmRMSQuantValidationPass(vllm_config)
        prefix_pass(graph_module.graph)
        assert _count_nodes(graph_module.graph, LONG_FUSED_OP) == NUM_LONG_CHAINS
        assert _count_nodes(graph_module.graph, PREFIX_FUSED_OP) == NUM_PREFIX_CHAINS
