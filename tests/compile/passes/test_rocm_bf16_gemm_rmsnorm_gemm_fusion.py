# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator

import pytest
import torch
from torch import fx
from torch.fx.experimental.proxy_tensor import make_fx

import vllm.config
import vllm.ir.ops
import vllm.kernels.aiter_ops  # noqa: F401
import vllm.model_executor.layers.utils  # noqa: F401
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops

if not hasattr(torch.ops.vllm, "rocm_aiter_gemm_a8w8_blockscale"):
    rocm_aiter_ops.register_ops_once.__wrapped__()  # pyright: ignore[reportFunctionMemberAccess]

from vllm.compilation.passes.fusion.rocm_coda_gemm_rmsnorm_gemm_fusion import (
    BF16_FUSED_OP,
    BF16_GEMM_OP,
    FUSED_ADD_RMS_NORM_OP,
    QWEN3_EPSILON,
    RocmCodaGemmRMSNormGemmFusionPass,
)
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    KernelConfig,
    VllmConfig,
)

TOKENS = 4
INPUT_SIZE = 256
HIDDEN_SIZE = 256
OUTPUT_SIZE = 384


def _bf16_chain(
    input: torch.Tensor,
    gemm0_weight: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    gemm1_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gemm0 = BF16_GEMM_OP(input, gemm0_weight)
    normalized, residual_out = FUSED_ADD_RMS_NORM_OP(
        gemm0,
        residual,
        rms_weight,
        QWEN3_EPSILON,
    )
    return residual_out, BF16_GEMM_OP(normalized, gemm1_weight)


def _example_inputs(*, tokens: int = TOKENS, device: str = "cpu") -> list[torch.Tensor]:
    return [
        torch.empty(tokens, INPUT_SIZE, dtype=torch.bfloat16, device=device),
        torch.empty(HIDDEN_SIZE, INPUT_SIZE, dtype=torch.bfloat16, device=device),
        torch.empty(tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
        torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
        torch.empty(OUTPUT_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
    ]


def _run_fusion(
    graph: fx.Graph,
    *,
    use_hipbmm_fused_epilogue: bool = True,
) -> RocmCodaGemmRMSNormGemmFusionPass:
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
        device_config=DeviceConfig(device=torch.device("cuda")),
        kernel_config=KernelConfig(use_hipbmm_fused_epilogue=use_hipbmm_fused_epilogue),
    )
    with vllm.config.set_current_vllm_config(vllm_config):
        fusion_pass = RocmCodaGemmRMSNormGemmFusionPass(vllm_config)
        fusion_pass(graph)
    return fusion_pass


def _nodes_with_target(graph: fx.Graph, target: object) -> list[fx.Node]:
    return [node for node in graph.nodes if node.target == target]


def test_fusion_replaces_all_supported_bf16_chains() -> None:
    def two_chains(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        first = _bf16_chain(*args)
        second = _bf16_chain(*args)
        return *first, *second

    graph_module = make_fx(two_chains, tracing_mode="fake")(*_example_inputs())
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 2
    assert fusion_pass.matched_count_by_format == {
        "block_fp8": 0,
        "mxfp8": 0,
        "bf16": 2,
    }
    assert not _nodes_with_target(graph_module.graph, BF16_GEMM_OP)
    assert not _nodes_with_target(graph_module.graph, FUSED_ADD_RMS_NORM_OP)

    fused = _nodes_with_target(graph_module.graph, BF16_FUSED_OP)
    assert len(fused) == 2
    for node in fused:
        outputs = {
            user.args[1]: user for user in node.users if user.target == operator.getitem
        }
        assert set(outputs) == {0, 1}
        assert outputs[0].meta["val"].shape == (TOKENS, HIDDEN_SIZE)
        assert outputs[0].meta["val"].stride() == (HIDDEN_SIZE, 1)
        assert outputs[1].meta["val"].shape == (TOKENS, OUTPUT_SIZE)
        assert outputs[1].meta["val"].stride() == (OUTPUT_SIZE, 1)


def test_fusion_leaves_fp16_chain_unchanged() -> None:
    inputs = [tensor.to(torch.float16) for tensor in _example_inputs()]
    graph_module = make_fx(_bf16_chain, tracing_mode="fake")(*inputs)

    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 0
    assert len(_nodes_with_target(graph_module.graph, BF16_GEMM_OP)) == 2
    assert len(_nodes_with_target(graph_module.graph, FUSED_ADD_RMS_NORM_OP)) == 1


def test_fusion_requires_hipbmm_fused_epilogue() -> None:
    graph_module = make_fx(_bf16_chain, tracing_mode="fake")(*_example_inputs())

    fusion_pass = _run_fusion(
        graph_module.graph,
        use_hipbmm_fused_epilogue=False,
    )

    assert fusion_pass.matched_count == 0
    assert len(_nodes_with_target(graph_module.graph, BF16_GEMM_OP)) == 2
    assert len(_nodes_with_target(graph_module.graph, FUSED_ADD_RMS_NORM_OP)) == 1


@pytest.mark.parametrize("biased_gemm", [0, 1], ids=["gemm0", "gemm1"])
def test_fusion_leaves_biased_bf16_chain_unchanged(biased_gemm: int) -> None:
    def biased_chain(
        input: torch.Tensor,
        gemm0_weight: torch.Tensor,
        residual: torch.Tensor,
        rms_weight: torch.Tensor,
        gemm1_weight: torch.Tensor,
        gemm0_bias: torch.Tensor,
        gemm1_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gemm0 = BF16_GEMM_OP(
            input,
            gemm0_weight,
            gemm0_bias if biased_gemm == 0 else None,
        )
        normalized, residual_out = FUSED_ADD_RMS_NORM_OP(
            gemm0,
            residual,
            rms_weight,
            QWEN3_EPSILON,
        )
        gemm1 = BF16_GEMM_OP(
            normalized,
            gemm1_weight,
            gemm1_bias if biased_gemm == 1 else None,
        )
        return residual_out, gemm1

    inputs = _example_inputs()
    inputs.extend(
        [
            torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16),
            torch.empty(OUTPUT_SIZE, dtype=torch.bfloat16),
        ]
    )
    graph_module = make_fx(biased_chain, tracing_mode="fake")(*inputs)

    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 0
    assert len(_nodes_with_target(graph_module.graph, BF16_GEMM_OP)) == 2
    assert len(_nodes_with_target(graph_module.graph, FUSED_ADD_RMS_NORM_OP)) == 1


def test_fusion_leaves_variance_size_chain_unchanged() -> None:
    def variance_size_chain(
        input: torch.Tensor,
        gemm0_weight: torch.Tensor,
        residual: torch.Tensor,
        rms_weight: torch.Tensor,
        gemm1_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gemm0 = BF16_GEMM_OP(input, gemm0_weight)
        normalized, residual_out = FUSED_ADD_RMS_NORM_OP(
            gemm0,
            residual,
            rms_weight,
            QWEN3_EPSILON,
            HIDDEN_SIZE // 2,
        )
        return residual_out, BF16_GEMM_OP(normalized, gemm1_weight)

    graph_module = make_fx(variance_size_chain, tracing_mode="fake")(*_example_inputs())

    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 0
    assert len(_nodes_with_target(graph_module.graph, BF16_GEMM_OP)) == 2
    assert len(_nodes_with_target(graph_module.graph, FUSED_ADD_RMS_NORM_OP)) == 1


def test_fusion_leaves_normalized_fanout_chain_unchanged() -> None:
    def normalized_fanout_chain(
        input: torch.Tensor,
        gemm0_weight: torch.Tensor,
        residual: torch.Tensor,
        rms_weight: torch.Tensor,
        gemm1_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gemm0 = BF16_GEMM_OP(input, gemm0_weight)
        normalized, residual_out = FUSED_ADD_RMS_NORM_OP(
            gemm0,
            residual,
            rms_weight,
            QWEN3_EPSILON,
        )
        gemm1 = BF16_GEMM_OP(normalized, gemm1_weight)
        return residual_out, gemm1, normalized

    graph_module = make_fx(normalized_fanout_chain, tracing_mode="fake")(
        *_example_inputs()
    )

    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 0
    assert len(_nodes_with_target(graph_module.graph, BF16_GEMM_OP)) == 2
    assert len(_nodes_with_target(graph_module.graph, FUSED_ADD_RMS_NORM_OP)) == 1


def test_fusion_leaves_dead_residual_output_chain_unchanged() -> None:
    def dead_residual_chain(
        input: torch.Tensor,
        gemm0_weight: torch.Tensor,
        residual: torch.Tensor,
        rms_weight: torch.Tensor,
        gemm1_weight: torch.Tensor,
    ) -> torch.Tensor:
        gemm0 = BF16_GEMM_OP(input, gemm0_weight)
        normalized, _ = FUSED_ADD_RMS_NORM_OP(
            gemm0,
            residual,
            rms_weight,
            QWEN3_EPSILON,
        )
        return BF16_GEMM_OP(normalized, gemm1_weight)

    graph_module = make_fx(dead_residual_chain, tracing_mode="fake")(*_example_inputs())

    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 0
    assert len(_nodes_with_target(graph_module.graph, BF16_GEMM_OP)) == 2
    assert len(_nodes_with_target(graph_module.graph, FUSED_ADD_RMS_NORM_OP)) == 1


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)
@pytest.mark.parametrize("tokens", [TOKENS, 1], ids=["prefill", "decode"])
def test_bf16_fused_op_matches_model_sequence(tokens: int) -> None:
    torch.manual_seed(0)
    input, gemm0_weight, residual, rms_weight, gemm1_weight = _example_inputs(
        tokens=tokens, device="cuda"
    )
    input.normal_()
    gemm0_weight.normal_(std=INPUT_SIZE**-0.5)
    residual.normal_()
    residual_before = residual.clone()
    rms_weight.normal_(mean=1.0, std=0.02)
    gemm1_weight.normal_(std=HIDDEN_SIZE**-0.5)

    gemm0 = BF16_GEMM_OP(input, gemm0_weight)
    expected_normalized, expected_residual = FUSED_ADD_RMS_NORM_OP(
        gemm0,
        residual,
        rms_weight,
        QWEN3_EPSILON,
    )
    expected_output = BF16_GEMM_OP(expected_normalized, gemm1_weight)

    actual_residual, actual_output = BF16_FUSED_OP(
        input,
        gemm0_weight,
        residual,
        rms_weight,
        QWEN3_EPSILON,
        gemm1_weight,
    )
    torch.cuda.synchronize()

    # The fused producer retains GEMM0's FP32 accumulator through the residual
    # add, while the source graph materializes GEMM0 as BF16 first. Bound that
    # expected BF16-rounding difference rather than requiring bitwise equality.
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0.02, atol=0.04)
    torch.testing.assert_close(actual_output, expected_output, rtol=0.02, atol=0.04)
    assert actual_output.stride() == (OUTPUT_SIZE, 1)
    assert torch.equal(residual, residual_before)
