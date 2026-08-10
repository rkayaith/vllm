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

from vllm.compilation.passes.fusion import (
    rocm_coda_gemm_rmsnorm_gemm_fusion as coda_fusion,
)
from vllm.compilation.passes.fusion.rocm_coda_gemm_rmsnorm_gemm_fusion import (
    FUSED_ADD_RMS_NORM_OP,
    MXFP8_GEMM_OP,
    MXFP8_QUANT_OP,
    MXFP8_REFERENCE_OP,
    QWEN3_EPSILON,
    RocmCodaGemmRMSNormGemmFusionPass,
)
from vllm.config import CompilationConfig, CompilationMode, DeviceConfig, VllmConfig
from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
    _mxfp8_dot_scaled_linear,
    _rocm_mxfp8_dot_scaled_mm_impl,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform

TOKENS = 4
INPUT_SIZE = 256
HIDDEN_SIZE = 256
OUTPUT_SIZE = 384
MXFP8_BLOCK_SIZE = 32


def _qwen3_mxfp8_chain(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    gemm0_weight: torch.Tensor,
    gemm0_weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    gemm1_weight: torch.Tensor,
    gemm1_weight_scale: torch.Tensor,
    *,
    epsilon: float = QWEN3_EPSILON,
    variance_size: int | None = None,
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
    gemm0_output_dtype: torch.dtype = torch.bfloat16,
    gemm1_output_dtype: torch.dtype = torch.bfloat16,
    extra_output: str | None = None,
) -> tuple[torch.Tensor, ...]:
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
    quantized, quantized_scale = MXFP8_QUANT_OP(
        normalized,
        is_sf_swizzled_layout,
        alignment,
    )
    gemm1 = MXFP8_GEMM_OP(
        quantized,
        quantized_scale,
        gemm1_weight,
        gemm1_weight_scale,
        gemm1_output_dtype,
    )

    outputs = (residual_out, gemm1)
    extra_outputs = {
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
        torch.empty(tokens, INPUT_SIZE, dtype=torch.float8_e4m3fn, device=device),
        torch.empty(
            tokens,
            INPUT_SIZE // MXFP8_BLOCK_SIZE,
            dtype=torch.uint8,
            device=device,
        ),
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
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
    gemm0_output_dtype: torch.dtype = torch.bfloat16,
    gemm1_output_dtype: torch.dtype = torch.bfloat16,
    extra_output: str | None = None,
) -> fx.GraphModule:
    def f(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _qwen3_mxfp8_chain(
            *args,
            epsilon=epsilon,
            variance_size=variance_size,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            alignment=alignment,
            gemm0_output_dtype=gemm0_output_dtype,
            gemm1_output_dtype=gemm1_output_dtype,
            extra_output=extra_output,
        )

    return make_fx(f, tracing_mode="fake")(*_example_inputs())


def _run_fusion(graph: fx.Graph) -> RocmCodaGemmRMSNormGemmFusionPass:
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
        device_config=DeviceConfig(device=torch.device("cuda")),
    )
    with vllm.config.set_current_vllm_config(vllm_config):
        fusion_pass = RocmCodaGemmRMSNormGemmFusionPass(vllm_config)
        fusion_pass(graph)
    return fusion_pass


def _nodes_with_target(graph: fx.Graph, target: object) -> list[fx.Node]:
    return [node for node in graph.nodes if node.target == target]


def test_fusion_replaces_complete_chain() -> None:
    graph_module = _trace_chain()
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
    assert fusion_pass.matched_count_by_format == {"block_fp8": 0, "mxfp8": 1}
    references = _nodes_with_target(graph_module.graph, MXFP8_REFERENCE_OP)
    assert len(references) == 1
    assert not _nodes_with_target(graph_module.graph, MXFP8_GEMM_OP)
    assert not _nodes_with_target(graph_module.graph, FUSED_ADD_RMS_NORM_OP)
    assert not _nodes_with_target(graph_module.graph, MXFP8_QUANT_OP)

    reference = references[0]
    assert reference.args == (
        gemm0_args[0],
        gemm0_args[2],
        gemm0_args[1],
        gemm0_args[3],
        residual,
        rms_weight,
        QWEN3_EPSILON,
        gemm1_weight,
        gemm1_weight_scale,
        torch.bfloat16,
        torch.bfloat16,
    )
    reference_outputs = {
        user.args[1]: user
        for user in reference.users
        if user.target == operator.getitem
    }
    assert set(reference_outputs) == {0, 2}
    assert reference_outputs[0].meta["val"].shape == (TOKENS, HIDDEN_SIZE)
    assert reference.meta["val"][1].shape == (TOKENS, OUTPUT_SIZE)
    assert reference_outputs[2].meta["val"].shape == (TOKENS, OUTPUT_SIZE)
    assert reference_outputs[0].meta["val"].dtype == torch.bfloat16
    assert reference.meta["val"][1].dtype == torch.float32
    assert reference_outputs[2].meta["val"].dtype == torch.bfloat16


def test_fusion_allows_independently_used_gemm0() -> None:
    graph_module = _trace_chain(extra_output="gemm0")
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 1
    gemms = _nodes_with_target(graph_module.graph, MXFP8_GEMM_OP)
    references = _nodes_with_target(graph_module.graph, MXFP8_REFERENCE_OP)
    assert len(gemms) == 1
    assert len(references) == 1
    assert gemms[0] not in references[0].args
    assert len(gemms[0].users) == 1


def test_reference_uses_fp32_gemm1(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[torch.dtype] = []

    def fake_gemm(
        input: torch.Tensor,
        input_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        calls.append(output_dtype)
        return torch.zeros(
            input.shape[0],
            weight.shape[0],
            dtype=output_dtype,
            device=input.device,
        )

    def fake_quant(
        input: torch.Tensor,
        is_sf_swizzled_layout: bool,
        alignment: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not is_sf_swizzled_layout
        assert alignment == 0
        return (
            input.to(torch.float8_e4m3fn),
            torch.zeros(
                input.shape[0],
                input.shape[1] // MXFP8_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
        )

    monkeypatch.setattr(coda_fusion, "MXFP8_GEMM_OP", fake_gemm)
    monkeypatch.setattr(coda_fusion, "MXFP8_QUANT_OP", fake_quant)

    inputs = _example_inputs(tokens=1, device="cpu")
    residual, transformed, model_output = (
        coda_fusion.rocm_mxfp8_gemm_rmsnorm_gemm_reference(
            inputs[0],
            inputs[2],
            inputs[1],
            inputs[3],
            inputs[4],
            inputs[5],
            QWEN3_EPSILON,
            inputs[6],
            inputs[7],
            torch.bfloat16,
            torch.bfloat16,
        )
    )

    assert calls == [torch.bfloat16, torch.float32]
    assert residual.dtype == torch.bfloat16
    assert transformed.dtype == torch.float32
    assert model_output.dtype == torch.bfloat16


@pytest.mark.parametrize(
    (
        "epsilon",
        "variance_size",
        "is_sf_swizzled_layout",
        "alignment",
        "gemm0_output_dtype",
        "gemm1_output_dtype",
    ),
    [
        pytest.param(
            1e-5, None, False, 0, torch.bfloat16, torch.bfloat16, id="epsilon"
        ),
        pytest.param(
            QWEN3_EPSILON,
            128,
            False,
            0,
            torch.bfloat16,
            torch.bfloat16,
            id="variance-size",
        ),
        pytest.param(
            QWEN3_EPSILON,
            None,
            True,
            0,
            torch.bfloat16,
            torch.bfloat16,
            id="swizzled-scales",
        ),
        pytest.param(
            QWEN3_EPSILON,
            None,
            False,
            32,
            torch.bfloat16,
            torch.bfloat16,
            id="aligned-scales",
        ),
        pytest.param(
            QWEN3_EPSILON,
            None,
            False,
            0,
            torch.float16,
            torch.bfloat16,
            id="gemm0-output",
        ),
        pytest.param(
            QWEN3_EPSILON,
            None,
            False,
            0,
            torch.bfloat16,
            torch.float16,
            id="gemm1-output",
        ),
    ],
)
def test_fusion_rejects_non_qwen3_mxfp8_constants(
    epsilon: float,
    variance_size: int | None,
    is_sf_swizzled_layout: bool,
    alignment: int,
    gemm0_output_dtype: torch.dtype,
    gemm1_output_dtype: torch.dtype,
) -> None:
    graph_module = _trace_chain(
        epsilon=epsilon,
        variance_size=variance_size,
        is_sf_swizzled_layout=is_sf_swizzled_layout,
        alignment=alignment,
        gemm0_output_dtype=gemm0_output_dtype,
        gemm1_output_dtype=gemm1_output_dtype,
    )
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 0
    assert not _nodes_with_target(graph_module.graph, MXFP8_REFERENCE_OP)
    assert len(_nodes_with_target(graph_module.graph, MXFP8_GEMM_OP)) == 2


@pytest.mark.parametrize("extra_output", ["normalized", "quantized", "scale"])
def test_fusion_rejects_independently_used_intermediate(extra_output: str) -> None:
    graph_module = _trace_chain(extra_output=extra_output)
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 0
    assert not _nodes_with_target(graph_module.graph, MXFP8_REFERENCE_OP)
    assert len(_nodes_with_target(graph_module.graph, MXFP8_GEMM_OP)) == 2


@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_mx()),
    reason="The native MXFP8 GEMM requires a ROCm gfx95x GPU",
)
@pytest.mark.parametrize("tokens", [TOKENS, 1], ids=["prefill", "decode"])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_graph_visible_mxfp8_gemm_matches_direct_triton_path(
    tokens: int,
    output_dtype: torch.dtype,
) -> None:
    torch.manual_seed(0)
    input_bf16 = torch.randn(tokens, INPUT_SIZE, dtype=torch.bfloat16, device="cuda")
    weight_bf16 = torch.randn(
        HIDDEN_SIZE, INPUT_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    input, input_scale = mxfp8_e4m3_quantize(input_bf16)
    weight, weight_scale = mxfp8_e4m3_quantize(weight_bf16)

    expected = _rocm_mxfp8_dot_scaled_mm_impl(
        input, input_scale, weight, weight_scale, output_dtype
    )
    actual = MXFP8_GEMM_OP(input, input_scale, weight, weight_scale, output_dtype)
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    if output_dtype == torch.bfloat16:
        linear_output = _mxfp8_dot_scaled_linear(input_bf16, weight, weight_scale)
        assert torch.equal(linear_output, expected)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_mx()),
    reason="The native MXFP8 GEMM requires a ROCm gfx95x GPU",
)
@pytest.mark.parametrize("tokens", [TOKENS, 1], ids=["prefill", "decode"])
def test_reference_op_is_bitwise_equal_to_fp32_gemm1_ordering(tokens: int) -> None:
    torch.manual_seed(0)
    input_bf16 = torch.randn(tokens, INPUT_SIZE, dtype=torch.bfloat16, device="cuda")
    gemm0_weight_bf16 = torch.randn(
        HIDDEN_SIZE, INPUT_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    gemm1_weight_bf16 = torch.randn(
        OUTPUT_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    input, input_scale = MXFP8_QUANT_OP(input_bf16, False, 0)
    gemm0_weight, gemm0_weight_scale = MXFP8_QUANT_OP(gemm0_weight_bf16, False, 0)
    gemm1_weight, gemm1_weight_scale = MXFP8_QUANT_OP(gemm1_weight_bf16, False, 0)
    residual = torch.randn(tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
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
    expected_residual = gemm0 + residual
    inverse_rms = torch.rsqrt(
        expected_residual.float().pow(2).mean(dim=-1, keepdim=True) + QWEN3_EPSILON
    )
    quantized, quantized_scale = MXFP8_QUANT_OP(
        expected_residual * rms_weight, False, 0
    )
    expected_gemm1 = MXFP8_GEMM_OP(
        quantized,
        quantized_scale,
        gemm1_weight,
        gemm1_weight_scale,
        torch.float32,
    )
    expected = expected_gemm1 * inverse_rms

    with torch.no_grad():
        actual_residual, actual, actual_model_output = MXFP8_REFERENCE_OP(
            input,
            gemm0_weight,
            input_scale,
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

    assert actual.dtype == torch.float32
    assert actual_model_output.dtype == torch.bfloat16
    assert torch.equal(actual_residual, expected_residual)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_model_output, expected.to(torch.bfloat16))
    assert not torch.equal(actual, expected.to(torch.bfloat16).float())
