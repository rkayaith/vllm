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
from vllm._aiter_ops import rocm_aiter_ops

# The combined pass also registers the block-FP8 backend. Structural tests run
# on builders where the platform guard skips AITER registration, so create the
# same schemas before importing the pass.
if not hasattr(torch.ops.vllm, "rocm_aiter_gemm_a8w8_blockscale"):
    rocm_aiter_ops.register_ops_once.__wrapped__()  # pyright: ignore[reportFunctionMemberAccess]

from vllm.compilation.passes.fusion.rocm_coda_gemm_rmsnorm_gemm_fusion import (
    FUSED_ADD_RMS_NORM_OP,
    MXFP8_FUSED_OP,
    MXFP8_GEMM_OP,
    MXFP8_QUANT_OP,
    QWEN3_EPSILON,
    RocmCodaGemmRMSNormGemmFusionPass,
    rocm_mxfp8_gemm_rmsnorm_gemm,
)
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    KernelConfig,
    VllmConfig,
)
from vllm.platforms import current_platform

TOKENS = 4
INPUT_SIZE = 256
HIDDEN_SIZE = 256
OUTPUT_SIZE = 384
MXFP8_BLOCK_SIZE = 32


def _qwen3_mxfp8_chain(
    input_bf16: torch.Tensor,
    gemm0_weight: torch.Tensor,
    gemm0_weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    gemm1_weight: torch.Tensor,
    gemm1_weight_scale: torch.Tensor,
    *,
    epsilon: float = QWEN3_EPSILON,
    variance_size: int | None = None,
    gemm0_output_dtype: torch.dtype = torch.bfloat16,
    gemm1_output_dtype: torch.dtype = torch.bfloat16,
    extra_output: str | None = None,
) -> tuple[torch.Tensor, ...]:
    input, input_scale = MXFP8_QUANT_OP(input_bf16)
    gemm0 = MXFP8_GEMM_OP(
        input,
        input_scale,
        gemm0_weight,
        gemm0_weight_scale,
        gemm0_output_dtype,
    )
    normalized, residual_out = FUSED_ADD_RMS_NORM_OP(
        gemm0,
        residual,
        rms_weight,
        epsilon,
        variance_size,
    )
    quantized, quantized_scale = MXFP8_QUANT_OP(normalized)
    gemm1 = MXFP8_GEMM_OP(
        quantized,
        quantized_scale,
        gemm1_weight,
        gemm1_weight_scale,
        gemm1_output_dtype,
    )

    outputs = (residual_out, gemm1)
    extra_outputs = {
        "input_quantized": input,
        "input_scale": input_scale,
        "gemm0": gemm0,
        "normalized": normalized,
        "quantized": quantized,
        "scale": quantized_scale,
    }
    if extra_output is not None:
        return outputs + (extra_outputs[extra_output],)
    return outputs


def _example_inputs(
    *,
    tokens: int = TOKENS,
    device: str = "cuda",
) -> list[torch.Tensor]:
    return [
        torch.empty(tokens, INPUT_SIZE, dtype=torch.bfloat16, device=device),
        torch.empty(
            HIDDEN_SIZE,
            INPUT_SIZE,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        torch.empty(
            HIDDEN_SIZE,
            INPUT_SIZE // MXFP8_BLOCK_SIZE,
            dtype=torch.uint8,
            device=device,
        ),
        torch.empty(tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
        torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
        torch.empty(
            OUTPUT_SIZE,
            HIDDEN_SIZE,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        torch.empty(
            OUTPUT_SIZE,
            HIDDEN_SIZE // MXFP8_BLOCK_SIZE,
            dtype=torch.uint8,
            device=device,
        ),
    ]


def _trace_chain(
    *,
    epsilon: float = QWEN3_EPSILON,
    variance_size: int | None = None,
    gemm0_output_dtype: torch.dtype = torch.bfloat16,
    gemm1_output_dtype: torch.dtype = torch.bfloat16,
    extra_output: str | None = None,
) -> fx.GraphModule:
    def f(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _qwen3_mxfp8_chain(
            *args,
            epsilon=epsilon,
            variance_size=variance_size,
            gemm0_output_dtype=gemm0_output_dtype,
            gemm1_output_dtype=gemm1_output_dtype,
            extra_output=extra_output,
        )

    return make_fx(f, tracing_mode="fake")(*_example_inputs())


def _run_fusion(graph: fx.Graph) -> RocmCodaGemmRMSNormGemmFusionPass:
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
        device_config=DeviceConfig(device=torch.device("cuda")),
        kernel_config=KernelConfig(use_hipbmm_fused_epilogue=True),
    )
    with vllm.config.set_current_vllm_config(vllm_config):
        fusion_pass = RocmCodaGemmRMSNormGemmFusionPass(vllm_config)
        fusion_pass(graph)
    return fusion_pass


def _nodes_with_target(graph: fx.Graph, target: object) -> list[fx.Node]:
    return [node for node in graph.nodes if node.target == target]


def test_fusion_replaces_complete_chain() -> None:
    graph_module = _trace_chain()
    original_quants = _nodes_with_target(graph_module.graph, MXFP8_QUANT_OP)
    assert len(original_quants) == 2
    input_quant = original_quants[0]
    input_quantized, input_scale = tuple(input_quant.users)
    if input_quantized.args[1] != 0:
        input_quantized, input_scale = input_scale, input_quantized
    original_gemms = _nodes_with_target(graph_module.graph, MXFP8_GEMM_OP)
    assert len(original_gemms) == 2
    gemm0, gemm1 = original_gemms
    gemm0_args = gemm0.args
    gemm1_weight = gemm1.args[2]
    gemm1_weight_scale = gemm1.args[3]
    rmsnorms = _nodes_with_target(graph_module.graph, FUSED_ADD_RMS_NORM_OP)
    assert len(rmsnorms) == 1
    residual = rmsnorms[0].args[1]
    rms_weight = rmsnorms[0].args[2]
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 1
    assert fusion_pass.matched_count_by_format == {
        "block_fp8": 0,
        "mxfp8": 1,
        "bf16": 0,
    }
    fused = _nodes_with_target(graph_module.graph, MXFP8_FUSED_OP)
    assert len(fused) == 1
    assert not _nodes_with_target(graph_module.graph, MXFP8_GEMM_OP)
    assert not _nodes_with_target(graph_module.graph, FUSED_ADD_RMS_NORM_OP)
    assert _nodes_with_target(graph_module.graph, MXFP8_QUANT_OP) == [input_quant]

    fused_op = fused[0]
    assert fused_op.args == (
        input_quantized,
        input_scale,
        gemm0_args[2],
        gemm0_args[3],
        residual,
        rms_weight,
        QWEN3_EPSILON,
        gemm1_weight,
        gemm1_weight_scale,
        torch.bfloat16,
        torch.bfloat16,
    )
    fused_outputs = {
        user.args[1]: user for user in fused_op.users if user.target == operator.getitem
    }
    assert set(fused_outputs) == {0, 1}
    assert fused_outputs[0].meta["val"].shape == (TOKENS, HIDDEN_SIZE)
    assert fused_outputs[0].meta["val"].stride() == (HIDDEN_SIZE, 1)
    assert fused_outputs[1].meta["val"].shape == (TOKENS, OUTPUT_SIZE)
    assert fused_outputs[1].meta["val"].stride() == (OUTPUT_SIZE, 1)
    assert fused_outputs[0].meta["val"].dtype == torch.bfloat16
    assert fused_outputs[1].meta["val"].dtype == torch.bfloat16


def test_fusion_preserves_mxfp8_rmsnorm_epsilon() -> None:
    epsilon = 1e-5
    graph_module = _trace_chain(epsilon=epsilon)
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 1
    fused = _nodes_with_target(graph_module.graph, MXFP8_FUSED_OP)
    assert len(fused) == 1
    assert fused[0].args[6] == epsilon


def test_fusion_allows_independently_used_gemm0() -> None:
    graph_module = _trace_chain(extra_output="gemm0")
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 1
    gemms = _nodes_with_target(graph_module.graph, MXFP8_GEMM_OP)
    fused = _nodes_with_target(graph_module.graph, MXFP8_FUSED_OP)
    assert len(gemms) == 1
    assert len(fused) == 1
    assert gemms[0] not in fused[0].args
    assert len(gemms[0].users) == 1


@pytest.mark.parametrize(
    (
        "epsilon",
        "variance_size",
        "gemm0_output_dtype",
        "gemm1_output_dtype",
    ),
    [
        pytest.param(
            QWEN3_EPSILON,
            128,
            torch.bfloat16,
            torch.bfloat16,
            id="variance-size",
        ),
        pytest.param(
            QWEN3_EPSILON,
            None,
            torch.float16,
            torch.bfloat16,
            id="gemm0-output",
        ),
        pytest.param(
            QWEN3_EPSILON,
            None,
            torch.bfloat16,
            torch.float16,
            id="gemm1-output",
        ),
    ],
)
def test_fusion_rejects_unsupported_mxfp8_constants(
    epsilon: float,
    variance_size: int | None,
    gemm0_output_dtype: torch.dtype,
    gemm1_output_dtype: torch.dtype,
) -> None:
    graph_module = _trace_chain(
        epsilon=epsilon,
        variance_size=variance_size,
        gemm0_output_dtype=gemm0_output_dtype,
        gemm1_output_dtype=gemm1_output_dtype,
    )
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 0
    assert not _nodes_with_target(graph_module.graph, MXFP8_FUSED_OP)
    assert len(_nodes_with_target(graph_module.graph, MXFP8_GEMM_OP)) == 2


@pytest.mark.parametrize("extra_output", ["normalized", "quantized", "scale"])
def test_fusion_rejects_independently_used_intermediate(extra_output: str) -> None:
    graph_module = _trace_chain(extra_output=extra_output)
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 0
    assert not _nodes_with_target(graph_module.graph, MXFP8_FUSED_OP)
    assert len(_nodes_with_target(graph_module.graph, MXFP8_GEMM_OP)) == 2


@pytest.mark.parametrize("extra_output", ["input_quantized", "input_scale"])
def test_fusion_preserves_independently_used_input_quantization(
    extra_output: str,
) -> None:
    graph_module = _trace_chain(extra_output=extra_output)
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 1
    assert len(_nodes_with_target(graph_module.graph, MXFP8_FUSED_OP)) == 1
    assert len(_nodes_with_target(graph_module.graph, MXFP8_QUANT_OP)) == 1
    assert not _nodes_with_target(graph_module.graph, MXFP8_GEMM_OP)


def test_mxfp8_wrapper_uses_corrected_aiter_contract(monkeypatch) -> None:
    from aiter.ops import gradlib

    from vllm.compilation.passes.fusion import (
        rocm_coda_gemm_rmsnorm_gemm_fusion as fusion_module,
    )

    contract_input_size = 512
    epsilon = 1e-5
    input = torch.empty(TOKENS, contract_input_size, dtype=torch.float8_e4m3fn)
    input_scale = torch.empty(32, 16, dtype=torch.uint8)
    gemm0_weight = torch.empty(
        HIDDEN_SIZE, contract_input_size, dtype=torch.float8_e4m3fn
    )
    gemm0_weight_scale = torch.empty(HIDDEN_SIZE, 16, dtype=torch.uint8)
    residual = torch.empty(TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16)
    rms_weight = torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16)
    gemm1_weight = torch.empty(OUTPUT_SIZE, HIDDEN_SIZE, dtype=torch.float8_e4m3fn)
    gemm1_weight_scale = torch.empty(OUTPUT_SIZE, 8, dtype=torch.uint8)
    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_hipb_mm_epilogue_mm(*args: object, **kwargs: object) -> torch.Tensor:
        captured.append((args, kwargs))
        output = kwargs["out"]
        assert isinstance(output, torch.Tensor)
        return output

    def unexpected_staged_op(*args: object, **kwargs: object) -> None:
        raise AssertionError("the MXFP8 wrapper used the staged implementation")

    monkeypatch.setattr(
        fusion_module, "_ensure_hipb_mm_extension_initialized", lambda: None
    )
    monkeypatch.setattr(fusion_module, "MXFP8_GEMM_OP", unexpected_staged_op)
    monkeypatch.setattr(fusion_module, "MXFP8_QUANT_OP", unexpected_staged_op)
    monkeypatch.setattr(gradlib, "hipb_mm_epilogue_mm", fake_hipb_mm_epilogue_mm)

    residual_out, output = rocm_mxfp8_gemm_rmsnorm_gemm(
        input,
        input_scale,
        gemm0_weight,
        gemm0_weight_scale,
        residual,
        rms_weight,
        epsilon,
        gemm1_weight,
        gemm1_weight_scale,
        torch.bfloat16,
        torch.bfloat16,
    )

    assert len(captured) == 1
    args, kwargs = captured[0]
    assert args[0] is input
    assert isinstance(args[1], torch.Tensor)
    assert isinstance(args[2], torch.Tensor)
    assert args[1].shape == (contract_input_size, HIDDEN_SIZE)
    assert args[1].stride() == (1, contract_input_size)
    assert args[2].shape == (HIDDEN_SIZE, OUTPUT_SIZE)
    assert args[2].stride() == (1, HIDDEN_SIZE)
    stages = kwargs["stages"]
    assert isinstance(stages, tuple)
    assert len(stages) == 3
    residual_stage, rmsnorm_stage, requant_stage = stages
    assert isinstance(residual_stage, gradlib.ResidualAdd)
    assert residual_stage.residual is residual
    assert residual_stage.residual_out is residual_out
    assert isinstance(rmsnorm_stage, gradlib.RMSNorm)
    assert rmsnorm_stage.gamma is rms_weight
    assert rmsnorm_stage.eps == epsilon
    assert isinstance(requant_stage, gradlib.Requant)
    assert requant_stage.scale_mode == (
        gradlib.HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT
    )
    assert requant_stage.scale_out.shape == (32, 8)
    assert requant_stage.scale_out.stride() == (8, 1)
    assert requant_stage.scale_out.dtype == torch.uint8
    assert kwargs["intermediate_dtype"] == torch.float8_e4m3fn
    assert kwargs["dtype"] == torch.bfloat16
    assert kwargs["mat1_scale"] is input_scale
    assert kwargs["mat2_scale"] is gemm0_weight_scale
    assert kwargs["mat3_scale"] is gemm1_weight_scale
    scale_mode = gradlib.HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT
    assert kwargs["mat1_scale_mode"] == scale_mode
    assert kwargs["mat2_scale_mode"] == scale_mode
    assert kwargs["mat3_scale_mode"] == scale_mode
    assert output is kwargs["out"]
    assert output.shape == (TOKENS, OUTPUT_SIZE)
    assert output.stride() == (OUTPUT_SIZE, 1)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_mx()),
    reason="The native MXFP8 GEMM requires a ROCm gfx95x GPU",
)
@pytest.mark.parametrize("tokens", [TOKENS, 1], ids=["prefill", "decode"])
def test_mxfp8_op_matches_model_sequence(tokens: int) -> None:
    torch.manual_seed(0)
    input_bf16 = torch.randn(tokens, INPUT_SIZE, dtype=torch.bfloat16, device="cuda")
    gemm0_weight_bf16 = torch.randn(
        HIDDEN_SIZE, INPUT_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    gemm1_weight_bf16 = torch.randn(
        OUTPUT_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    input, input_scale = MXFP8_QUANT_OP(input_bf16)
    gemm0_weight, gemm0_weight_scale = MXFP8_QUANT_OP(gemm0_weight_bf16)
    gemm1_weight, gemm1_weight_scale = MXFP8_QUANT_OP(gemm1_weight_bf16)
    residual = torch.randn(tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual_before = residual.clone()
    rms_weight = (
        torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.02 + 1.0
    )

    gemm0 = MXFP8_GEMM_OP(
        input,
        input_scale,
        gemm0_weight,
        gemm0_weight_scale,
        torch.bfloat16,
    )
    normalized, expected_residual = FUSED_ADD_RMS_NORM_OP(
        gemm0,
        residual,
        rms_weight,
        QWEN3_EPSILON,
        None,
    )
    quantized, quantized_scale = MXFP8_QUANT_OP(normalized)
    expected_model_output = MXFP8_GEMM_OP(
        quantized,
        quantized_scale,
        gemm1_weight,
        gemm1_weight_scale,
        torch.bfloat16,
    )

    with torch.no_grad():
        actual_residual, actual_model_output = MXFP8_FUSED_OP(
            input,
            input_scale,
            gemm0_weight,
            gemm0_weight_scale,
            residual,
            rms_weight,
            QWEN3_EPSILON,
            gemm1_weight,
            gemm1_weight_scale,
            torch.bfloat16,
            torch.bfloat16,
        )
    torch.cuda.synchronize()

    assert actual_model_output.dtype == torch.bfloat16
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0.02, atol=0.04)
    output_difference = actual_model_output.float() - expected_model_output.float()
    relative_l2_error = torch.linalg.vector_norm(
        output_difference
    ) / torch.linalg.vector_norm(expected_model_output.float())
    cosine_similarity = torch.nn.functional.cosine_similarity(
        actual_model_output.float().flatten(),
        expected_model_output.float().flatten(),
        dim=0,
    )
    # The fused path preserves the FP32 GEMM0 accumulator across the residual
    # and RMSNorm stages. The source graph first rounds GEMM0 to BF16, so the
    # following MXFP8 requant can choose adjacent power-of-two scale buckets.
    assert relative_l2_error < 0.05
    assert cosine_similarity > 0.999
    assert actual_model_output.stride() == (OUTPUT_SIZE, 1)
    assert torch.equal(residual, residual_before)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_mx()),
    reason="The native MXFP8 GEMM requires a ROCm gfx95x GPU",
)
def test_mxfp8_op_compiles_with_inductor() -> None:
    torch.manual_seed(1)
    input_bf16 = torch.randn(1, INPUT_SIZE, dtype=torch.bfloat16, device="cuda")
    gemm0_weight_bf16 = torch.randn(
        HIDDEN_SIZE, INPUT_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    gemm1_weight_bf16 = torch.randn(
        OUTPUT_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    input, input_scale = MXFP8_QUANT_OP(input_bf16)
    gemm0_weight, gemm0_weight_scale = MXFP8_QUANT_OP(gemm0_weight_bf16)
    gemm1_weight, gemm1_weight_scale = MXFP8_QUANT_OP(gemm1_weight_bf16)
    residual = torch.randn(1, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    rms_weight = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    args = (
        input,
        input_scale,
        gemm0_weight,
        gemm0_weight_scale,
        residual,
        rms_weight,
        QWEN3_EPSILON,
        gemm1_weight,
        gemm1_weight_scale,
        torch.bfloat16,
        torch.bfloat16,
    )

    expected = MXFP8_FUSED_OP(*args)
    compiled = torch.compile(MXFP8_FUSED_OP, backend="inductor", fullgraph=True)
    actual = compiled(*args)
    torch.cuda.synchronize()

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
