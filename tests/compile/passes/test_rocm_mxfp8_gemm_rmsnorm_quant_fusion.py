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
from vllm.compilation.passes.fusion.rocm_mxfp8_gemm_rmsnorm_quant_fusion import (
    FUSED_ADD_RMS_NORM_OP,
    MXFP8_GEMM_OP,
    MXFP8_QUANT_OP,
    QWEN3_EPSILON,
    REFERENCE_OP,
    RocmMxfp8GemmRMSNormQuantFusionPass,
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
MXFP8_BLOCK_SIZE = 32


def _qwen3_mxfp8_chain(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    epsilon: float = QWEN3_EPSILON,
    variance_size: int | None = None,
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
    output_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gemm = MXFP8_GEMM_OP(
        input,
        input_scale,
        weight,
        weight_scale,
        output_dtype,
    )
    normalized, residual_out = FUSED_ADD_RMS_NORM_OP(
        gemm,
        residual,
        rms_weight,
        epsilon,
        variance_size,
    )
    quantized, scales = MXFP8_QUANT_OP(
        normalized,
        is_sf_swizzled_layout,
        alignment,
    )

    # Keep the normalized value observable. The rewrite is allowed to replace
    # the requant outputs without taking ownership of this intermediate.
    return normalized, residual_out, quantized, scales


def _example_inputs(
    *,
    tokens: int = TOKENS,
    device: str = "cuda",
) -> list[torch.Tensor]:
    return [
        torch.empty(
            tokens,
            INPUT_SIZE,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
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
        torch.empty(
            tokens,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device=device,
        ),
        torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
    ]


def _trace_chain(
    *,
    epsilon: float = QWEN3_EPSILON,
    variance_size: int | None = None,
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
    output_dtype: torch.dtype = torch.bfloat16,
) -> fx.GraphModule:
    def f(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _qwen3_mxfp8_chain(
            *args,
            epsilon=epsilon,
            variance_size=variance_size,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            alignment=alignment,
            output_dtype=output_dtype,
        )

    return make_fx(f, tracing_mode="fake")(*_example_inputs())


def _run_fusion(graph: fx.Graph) -> RocmMxfp8GemmRMSNormQuantFusionPass:
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
        device_config=DeviceConfig(device=torch.device("cuda")),
    )
    with vllm.config.set_current_vllm_config(vllm_config):
        fusion_pass = RocmMxfp8GemmRMSNormQuantFusionPass(vllm_config)
        fusion_pass(graph)
    return fusion_pass


def _nodes_with_target(graph: fx.Graph, target: object) -> list[fx.Node]:
    return [node for node in graph.nodes if node.target == target]


def test_fusion_replaces_only_mxfp8_quant_outputs() -> None:
    graph_module = _trace_chain()
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 1
    assert len(_nodes_with_target(graph_module.graph, REFERENCE_OP)) == 1
    assert not _nodes_with_target(graph_module.graph, MXFP8_QUANT_OP)

    # The original GEMM and RMSNorm remain live for the normalized and residual
    # graph outputs. Only the two requantized tuple elements are redirected.
    gemm = _nodes_with_target(graph_module.graph, MXFP8_GEMM_OP)[0]
    rmsnorm = _nodes_with_target(graph_module.graph, FUSED_ADD_RMS_NORM_OP)[0]
    assert rmsnorm.args[0] is gemm

    fused = _nodes_with_target(graph_module.graph, REFERENCE_OP)[0]
    assert fused.args[:4] == gemm.args[:4]
    assert fused.args[4:10] == rmsnorm.args[1:4] + (None, False, 0)
    assert fused.args[10] == torch.bfloat16

    fused_getitems: list[int] = []
    for node in fused.users:
        if node.target == operator.getitem and node.args[0] is fused:
            index = node.args[1]
            assert isinstance(index, int)
            fused_getitems.append(index)
    assert set(fused_getitems) == {0, 1}


@pytest.mark.parametrize(
    (
        "epsilon",
        "variance_size",
        "is_sf_swizzled_layout",
        "alignment",
        "output_dtype",
    ),
    [
        pytest.param(
            1e-5,
            None,
            False,
            0,
            torch.bfloat16,
            id="wrong-epsilon",
        ),
        pytest.param(
            QWEN3_EPSILON,
            128,
            False,
            0,
            torch.bfloat16,
            id="variance-size",
        ),
        pytest.param(
            QWEN3_EPSILON,
            None,
            True,
            0,
            torch.bfloat16,
            id="swizzled-scales",
        ),
        pytest.param(
            QWEN3_EPSILON,
            None,
            False,
            32,
            torch.bfloat16,
            id="aligned-scales",
        ),
        pytest.param(
            QWEN3_EPSILON,
            None,
            False,
            0,
            torch.float16,
            id="wrong-output-dtype",
        ),
    ],
)
def test_fusion_rejects_non_qwen3_mxfp8_constants(
    epsilon: float,
    variance_size: int | None,
    is_sf_swizzled_layout: bool,
    alignment: int,
    output_dtype: torch.dtype,
) -> None:
    graph_module = _trace_chain(
        epsilon=epsilon,
        variance_size=variance_size,
        is_sf_swizzled_layout=is_sf_swizzled_layout,
        alignment=alignment,
        output_dtype=output_dtype,
    )
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 0
    assert not _nodes_with_target(graph_module.graph, REFERENCE_OP)
    assert len(_nodes_with_target(graph_module.graph, MXFP8_QUANT_OP)) == 1


def test_fusion_rejects_unexpected_quant_tuple_user() -> None:
    graph_module = _trace_chain()
    graph = graph_module.graph
    quant = _nodes_with_target(graph, MXFP8_QUANT_OP)[0]
    output = next(node for node in graph.nodes if node.op == "output")

    with graph.inserting_before(output):
        extra_quantized = graph.call_function(operator.getitem, args=(quant, 0))
    original_outputs = output.args[0]
    assert isinstance(original_outputs, tuple)
    output.args = (original_outputs + (extra_quantized,),)
    graph.lint()

    fusion_pass = _run_fusion(graph)

    assert fusion_pass.matched_count == 0
    assert not _nodes_with_target(graph, REFERENCE_OP)
    assert len(_nodes_with_target(graph, MXFP8_QUANT_OP)) == 1


@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_mx()),
    reason="The native MXFP8 GEMM requires a ROCm gfx95x GPU",
)
@pytest.mark.parametrize("tokens", [TOKENS, 1], ids=["prefill", "decode"])
def test_graph_visible_mxfp8_gemm_matches_direct_triton_path(tokens: int) -> None:
    torch.manual_seed(0)
    input_bf16 = torch.randn(
        tokens,
        INPUT_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    weight_bf16 = torch.randn(
        HIDDEN_SIZE,
        INPUT_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    input, input_scale = mxfp8_e4m3_quantize(input_bf16)
    weight, weight_scale = mxfp8_e4m3_quantize(weight_bf16)

    # This is the direct call path used before the Triton GEMM was exposed as
    # a graph-visible custom op for matching.
    expected = _rocm_mxfp8_dot_scaled_mm_impl(
        input,
        input_scale,
        weight,
        weight_scale,
        torch.bfloat16,
    )
    actual = MXFP8_GEMM_OP(
        input,
        input_scale,
        weight,
        weight_scale,
        torch.bfloat16,
    )
    linear_output = _mxfp8_dot_scaled_linear(
        input_bf16,
        weight,
        weight_scale,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    assert torch.equal(linear_output, expected)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_mx()),
    reason="The native MXFP8 GEMM requires a ROCm gfx95x GPU",
)
@pytest.mark.parametrize("tokens", [TOKENS, 1], ids=["prefill", "decode"])
def test_reference_op_is_bitwise_equal_to_unfused_chain(tokens: int) -> None:
    torch.manual_seed(0)
    input_bf16 = torch.randn(
        tokens,
        INPUT_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    weight_bf16 = torch.randn(
        HIDDEN_SIZE,
        INPUT_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    input, input_scale = MXFP8_QUANT_OP(input_bf16, False, 0)
    weight, weight_scale = MXFP8_QUANT_OP(weight_bf16, False, 0)
    residual = torch.randn(
        tokens,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    rms_weight = torch.randn(
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    inputs = [input, input_scale, weight, weight_scale, residual, rms_weight]

    with (
        torch.no_grad(),
        vllm.ir.ops.fused_add_rms_norm.set_priority(["aiter", "native"]),
    ):
        expected = _qwen3_mxfp8_chain(*inputs)[2:]
        actual = REFERENCE_OP(
            *inputs,
            QWEN3_EPSILON,
            None,
            False,
            0,
            torch.bfloat16,
        )
    torch.cuda.synchronize()

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
