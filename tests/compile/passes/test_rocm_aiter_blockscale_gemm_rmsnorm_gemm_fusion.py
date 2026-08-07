# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the ROCm block-FP8 two-GEMM RMSNorm rewrite."""

import operator

import pytest
import torch
from torch import fx
from torch._ops import OpOverload
from torch.fx.experimental.proxy_tensor import make_fx

import vllm.config
import vllm.kernels.aiter_ops  # noqa: F401
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops

# The production pass manager imports this fusion only after AITER support has
# registered its wrapper ops. Structural tests also run on non-ROCm builders,
# where that platform guard intentionally skips registration, so register the
# same schemas explicitly before importing the pass under test.
if not hasattr(torch.ops.vllm, "rocm_aiter_gemm_a8w8_blockscale"):
    rocm_aiter_ops.register_ops_once.__wrapped__()  # pyright: ignore[reportFunctionMemberAccess]

from vllm.compilation.passes.fusion import (
    rocm_aiter_blockscale_gemm_rmsnorm_gemm_fusion as blockscale_fusion,
)
from vllm.compilation.passes.fusion.rocm_aiter_blockscale_gemm_rmsnorm_gemm_fusion import (  # noqa: E501
    DECODE_GEMM_OP,
    PREFILL_GEMM_OP,
    QWEN3_EPSILON,
    QWEN3_GROUP_SIZE,
    REFERENCE_OP,
    RocmAiterBlockscaleGemmRMSNormGemmFusionPass,
)
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    ModelConfig,
    PassConfig,
    VllmConfig,
)
from vllm.platforms import current_platform

RMSNORM_QUANT_OP = torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant.default
GROUP_QUANT_OP = torch.ops.vllm.rocm_aiter_group_fp8_quant.default
FP8_DTYPE = current_platform.fp8_dtype()
TOKENS = 4
INPUT_SIZE = 256
HIDDEN_SIZE = 256
OUTPUT_SIZE = 384


def _qwen3_block_fp8_chain(
    input: torch.Tensor,
    gemm0_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gemm0_weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    gemm1_weight: torch.Tensor,
    gemm1_weight_scale: torch.Tensor,
    *,
    gemm0_op: OpOverload,
    gemm1_op: OpOverload,
    epsilon: float = QWEN3_EPSILON,
    group_size: int = QWEN3_GROUP_SIZE,
    gemm0_output_dtype: torch.dtype = torch.bfloat16,
    gemm1_output_dtype: torch.dtype = torch.bfloat16,
    extra_output: str | None = None,
) -> tuple[torch.Tensor, ...]:
    gemm0 = gemm0_op(
        input,
        gemm0_weight,
        input_scale,
        gemm0_weight_scale,
        gemm0_output_dtype,
    )
    quantized, residual_out, quantized_scale = RMSNORM_QUANT_OP(
        gemm0,
        residual,
        rms_weight,
        epsilon,
        group_size,
    )
    gemm1 = gemm1_op(
        quantized,
        gemm1_weight,
        quantized_scale,
        gemm1_weight_scale,
        gemm1_output_dtype,
    )

    outputs = (residual_out, gemm1)
    if extra_output == "gemm0":
        return outputs + (gemm0,)
    if extra_output == "quantized":
        return outputs + (quantized,)
    if extra_output == "scale":
        return outputs + (quantized_scale,)
    return outputs


def _example_inputs(
    *,
    tokens: int = TOKENS,
    device: str = "cuda",
) -> list[torch.Tensor]:
    return [
        torch.empty(tokens, INPUT_SIZE, dtype=FP8_DTYPE, device=device),
        torch.empty(HIDDEN_SIZE, INPUT_SIZE, dtype=FP8_DTYPE, device=device),
        torch.empty(
            tokens,
            INPUT_SIZE // QWEN3_GROUP_SIZE,
            dtype=torch.float32,
            device=device,
        ),
        torch.empty(
            HIDDEN_SIZE // QWEN3_GROUP_SIZE,
            INPUT_SIZE // QWEN3_GROUP_SIZE,
            dtype=torch.float32,
            device=device,
        ),
        torch.empty(tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
        torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
        torch.empty(OUTPUT_SIZE, HIDDEN_SIZE, dtype=FP8_DTYPE, device=device),
        torch.empty(
            OUTPUT_SIZE // QWEN3_GROUP_SIZE,
            HIDDEN_SIZE // QWEN3_GROUP_SIZE,
            dtype=torch.float32,
            device=device,
        ),
    ]


def _trace_chain(
    *,
    gemm0_op: OpOverload,
    gemm1_op: OpOverload,
    epsilon: float = QWEN3_EPSILON,
    group_size: int = QWEN3_GROUP_SIZE,
    gemm0_output_dtype: torch.dtype = torch.bfloat16,
    gemm1_output_dtype: torch.dtype = torch.bfloat16,
    extra_output: str | None = None,
) -> fx.GraphModule:
    def f(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _qwen3_block_fp8_chain(
            *args,
            gemm0_op=gemm0_op,
            gemm1_op=gemm1_op,
            epsilon=epsilon,
            group_size=group_size,
            gemm0_output_dtype=gemm0_output_dtype,
            gemm1_output_dtype=gemm1_output_dtype,
            extra_output=extra_output,
        )

    return make_fx(f, tracing_mode="fake")(*_example_inputs())


def _run_fusion(
    graph: fx.Graph,
) -> RocmAiterBlockscaleGemmRMSNormGemmFusionPass:
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
        device_config=DeviceConfig(device=torch.device("cuda")),
    )
    with vllm.config.set_current_vllm_config(vllm_config):
        fusion_pass = RocmAiterBlockscaleGemmRMSNormGemmFusionPass(vllm_config)
        fusion_pass(graph)
    return fusion_pass


def _nodes_with_target(graph: fx.Graph, target: object) -> list[fx.Node]:
    return [node for node in graph.nodes if node.target == target]


@pytest.mark.parametrize(
    ("gemm_op", "use_triton_gemm"),
    [
        pytest.param(PREFILL_GEMM_OP, True, id="prefill-triton"),
        pytest.param(DECODE_GEMM_OP, False, id="decode-ck"),
    ],
)
def test_fusion_replaces_gemm1_and_residual_and_retains_gemm0(
    gemm_op: OpOverload,
    use_triton_gemm: bool,
) -> None:
    graph_module = _trace_chain(gemm0_op=gemm_op, gemm1_op=gemm_op)
    original_gemms = _nodes_with_target(graph_module.graph, gemm_op)
    assert len(original_gemms) == 2
    original_rmsnorm_quant = _nodes_with_target(graph_module.graph, RMSNORM_QUANT_OP)
    assert len(original_rmsnorm_quant) == 1
    rmsnorm_args = original_rmsnorm_quant[0].args[1:5]
    gemm1_weight = original_gemms[1].args[1]
    gemm1_weight_scale = original_gemms[1].args[3]
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 1
    fused = _nodes_with_target(graph_module.graph, REFERENCE_OP)
    assert len(fused) == 1

    # GEMM1 and the updated residual are replaced. GEMM0 stays live through
    # its independent numerical-check output, while the combined RMSNorm/quant
    # operation and its old tuple projections become dead.
    gemms = _nodes_with_target(graph_module.graph, gemm_op)
    assert len(gemms) == 1
    gemm0 = gemms[0]
    assert not _nodes_with_target(graph_module.graph, RMSNORM_QUANT_OP)

    reference = fused[0]
    assert reference.args[:4] == gemm0.args[:4]
    assert reference.args[4:8] == rmsnorm_args
    assert reference.args[8:10] == (gemm1_weight, gemm1_weight_scale)
    assert reference.args[10] is gemm0
    assert reference.args[11:] == (
        use_triton_gemm,
        torch.bfloat16,
        torch.bfloat16,
    )
    reference_outputs = {
        node.args[1]: node
        for node in reference.users
        if node.target == operator.getitem
    }
    assert set(reference_outputs) == {0, 2}
    assert reference_outputs[0].meta["val"].shape == (TOKENS, HIDDEN_SIZE)
    assert reference_outputs[2].meta["val"].shape == (TOKENS, OUTPUT_SIZE)
    assert reference_outputs[0].meta["val"].dtype == torch.bfloat16
    assert reference.meta["val"][1].shape == (TOKENS, OUTPUT_SIZE)
    assert reference.meta["val"][1].dtype == torch.float32
    assert reference_outputs[2].meta["val"].dtype == torch.bfloat16
    assert reference_outputs[0].users
    assert reference_outputs[2].users


def test_fusion_allows_independently_used_gemm0() -> None:
    graph_module = _trace_chain(
        gemm0_op=PREFILL_GEMM_OP,
        gemm1_op=PREFILL_GEMM_OP,
        extra_output="gemm0",
    )
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 1
    gemms = _nodes_with_target(graph_module.graph, PREFILL_GEMM_OP)
    assert len(gemms) == 1
    reference = _nodes_with_target(graph_module.graph, REFERENCE_OP)
    assert len(reference) == 1
    assert reference[0].args[10] is gemms[0]
    assert len(gemms[0].users) == 2


def test_reference_uses_fp32_triton_gemm1_for_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, torch.dtype]] = []

    def fake_gemm(
        name: str,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        del input_scale, weight_scale
        calls.append((name, output_dtype))
        return torch.zeros(
            input.shape[0],
            weight.shape[0],
            dtype=output_dtype,
            device=input.device,
        )

    def fake_triton_gemm(*args: object) -> torch.Tensor:
        return fake_gemm("triton", *args)  # type: ignore[arg-type]

    def fake_decode_gemm(*args: object) -> torch.Tensor:
        return fake_gemm("decode", *args)  # type: ignore[arg-type]

    monkeypatch.setattr(blockscale_fusion, "PREFILL_GEMM_OP", fake_triton_gemm)
    monkeypatch.setattr(blockscale_fusion, "DECODE_GEMM_OP", fake_decode_gemm)

    inputs = _example_inputs(tokens=1, device="cpu")
    (
        input,
        gemm0_weight,
        input_scale,
        gemm0_weight_scale,
        residual_input,
        rms_weight,
        gemm1_weight,
        gemm1_weight_scale,
    ) = inputs
    retained_gemm0 = torch.empty(1, HIDDEN_SIZE, dtype=torch.bfloat16)
    residual, transformed, model_output = (
        blockscale_fusion.rocm_aiter_blockscale_gemm_rmsnorm_gemm_reference(
            input,
            gemm0_weight,
            input_scale,
            gemm0_weight_scale,
            residual_input,
            rms_weight,
            QWEN3_EPSILON,
            QWEN3_GROUP_SIZE,
            gemm1_weight,
            gemm1_weight_scale,
            retained_gemm0,
            False,
            torch.bfloat16,
            torch.bfloat16,
        )
    )

    assert calls == [
        ("decode", torch.bfloat16),
        ("triton", torch.float32),
    ]
    assert residual.dtype == torch.bfloat16
    assert transformed.dtype == torch.float32
    assert model_output.dtype == torch.bfloat16


@pytest.mark.skipif(
    not rocm_aiter_ops.is_enabled(),
    reason="Pass manager registers ROCm AITER passes only when AITER is enabled",
)
def test_fusion_runs_after_aiter_rmsnorm_quant_fusion() -> None:
    from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
        RocmAiterRMSNormQuantFusionPass,
    )
    from vllm.compilation.passes.pass_manager import PostGradPassManager

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=torch.bfloat16),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config=PassConfig(fuse_norm_quant=True),
        ),
        device_config=DeviceConfig(device=torch.device("cuda")),
    )
    pass_manager = PostGradPassManager()
    pass_manager.configure(vllm_config)

    rmsnorm_quant_index = next(
        index
        for index, pass_ in enumerate(pass_manager.passes)
        if isinstance(pass_, RocmAiterRMSNormQuantFusionPass)
    )
    two_gemm_index = next(
        index
        for index, pass_ in enumerate(pass_manager.passes)
        if isinstance(pass_, RocmAiterBlockscaleGemmRMSNormGemmFusionPass)
    )
    assert rmsnorm_quant_index < two_gemm_index


@pytest.mark.parametrize(
    ("epsilon", "group_size", "gemm0_output_dtype", "gemm1_output_dtype"),
    [
        pytest.param(
            1e-5,
            QWEN3_GROUP_SIZE,
            torch.bfloat16,
            torch.bfloat16,
            id="wrong-epsilon",
        ),
        pytest.param(
            QWEN3_EPSILON,
            64,
            torch.bfloat16,
            torch.bfloat16,
            id="wrong-group-size",
        ),
        pytest.param(
            QWEN3_EPSILON,
            QWEN3_GROUP_SIZE,
            torch.float16,
            torch.bfloat16,
            id="wrong-gemm0-output-dtype",
        ),
        pytest.param(
            QWEN3_EPSILON,
            QWEN3_GROUP_SIZE,
            torch.bfloat16,
            torch.float16,
            id="wrong-gemm1-output-dtype",
        ),
    ],
)
def test_fusion_rejects_non_qwen3_block_fp8_constants(
    epsilon: float,
    group_size: int,
    gemm0_output_dtype: torch.dtype,
    gemm1_output_dtype: torch.dtype,
) -> None:
    graph_module = _trace_chain(
        gemm0_op=PREFILL_GEMM_OP,
        gemm1_op=PREFILL_GEMM_OP,
        epsilon=epsilon,
        group_size=group_size,
        gemm0_output_dtype=gemm0_output_dtype,
        gemm1_output_dtype=gemm1_output_dtype,
    )
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 0
    assert not _nodes_with_target(graph_module.graph, REFERENCE_OP)
    assert len(_nodes_with_target(graph_module.graph, PREFILL_GEMM_OP)) == 2


@pytest.mark.parametrize("extra_output", ["quantized", "scale"])
def test_fusion_rejects_independently_used_quant_output(extra_output: str) -> None:
    graph_module = _trace_chain(
        gemm0_op=PREFILL_GEMM_OP,
        gemm1_op=PREFILL_GEMM_OP,
        extra_output=extra_output,
    )
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 0
    assert not _nodes_with_target(graph_module.graph, REFERENCE_OP)
    assert len(_nodes_with_target(graph_module.graph, PREFILL_GEMM_OP)) == 2


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)
@pytest.mark.parametrize(
    ("gemm_op", "tokens", "use_triton_gemm"),
    [
        pytest.param(PREFILL_GEMM_OP, TOKENS, True, id="prefill-triton"),
        pytest.param(DECODE_GEMM_OP, 1, False, id="decode-ck"),
    ],
)
def test_reference_op_is_bitwise_equal_to_fp32_gemm1_ordering(
    gemm_op: OpOverload,
    tokens: int,
    use_triton_gemm: bool,
) -> None:
    torch.manual_seed(0)
    inputs = _random_example_inputs(tokens=tokens)

    with torch.no_grad():
        retained_gemm0, expected_residual, expected = _fp32_gemm1_ordering(
            *inputs, gemm0_op=gemm_op
        )
        original = _original_ordering(*inputs, gemm0_op=gemm_op, gemm1_op=gemm_op)
        actual_residual, actual, actual_model_output = REFERENCE_OP(
            *inputs[:6],
            QWEN3_EPSILON,
            QWEN3_GROUP_SIZE,
            *inputs[6:],
            retained_gemm0,
            use_triton_gemm,
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
    assert not torch.equal(actual, original.float())
    relative_error = torch.linalg.vector_norm(actual.float() - original.float()) / (
        torch.linalg.vector_norm(original.float())
    )
    assert relative_error < 0.05


def _fp32_gemm1_ordering(
    input: torch.Tensor,
    gemm0_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gemm0_weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    gemm1_weight: torch.Tensor,
    gemm1_weight_scale: torch.Tensor,
    *,
    gemm0_op: OpOverload,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gemm0 = gemm0_op(
        input,
        gemm0_weight,
        input_scale,
        gemm0_weight_scale,
        torch.bfloat16,
    )
    updated_residual = gemm0 + residual
    inverse_rms = torch.rsqrt(
        updated_residual.float().pow(2).mean(dim=-1, keepdim=True) + QWEN3_EPSILON
    )
    weighted = updated_residual * rms_weight
    quantized, quantized_scale = _quantize_fp8_tile_activation(
        weighted, QWEN3_GROUP_SIZE
    )
    # GEMM1 deliberately uses Triton for both prefill and decode because the
    # CK wrapper cannot emit the FP32 result required by this ordering.
    gemm1 = PREFILL_GEMM_OP(
        quantized,
        gemm1_weight,
        quantized_scale,
        gemm1_weight_scale,
        torch.float32,
    )
    return gemm0, updated_residual, gemm1 * inverse_rms


def _original_ordering(
    input: torch.Tensor,
    gemm0_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gemm0_weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    gemm1_weight: torch.Tensor,
    gemm1_weight_scale: torch.Tensor,
    *,
    gemm0_op: OpOverload,
    gemm1_op: OpOverload,
) -> torch.Tensor:
    gemm0 = gemm0_op(
        input,
        gemm0_weight,
        input_scale,
        gemm0_weight_scale,
        torch.bfloat16,
    )
    quantized, _, quantized_scale = RMSNORM_QUANT_OP(
        gemm0,
        residual,
        rms_weight,
        QWEN3_EPSILON,
        QWEN3_GROUP_SIZE,
    )
    return gemm1_op(
        quantized,
        gemm1_weight,
        quantized_scale,
        gemm1_weight_scale,
        torch.bfloat16,
    )


def _quantize_fp8_tile_activation(
    input: torch.Tensor,
    tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_info = torch.finfo(torch.float8_e4m3fn)
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
    quantized = (
        (input.float() * scale)
        .clamp(fp8_info.min, fp8_info.max)
        .to(torch.float8_e4m3fn)
    )
    return quantized, (absolute_maximum / fp8_info.max).to(torch.float32)


def _random_example_inputs(*, tokens: int) -> list[torch.Tensor]:
    inputs = _example_inputs(tokens=tokens)
    inputs[0].copy_(
        (torch.randn_like(inputs[0], dtype=torch.bfloat16) * 0.01).to(FP8_DTYPE)
    )
    inputs[1].copy_(
        (torch.randn_like(inputs[1], dtype=torch.bfloat16) * 0.01).to(FP8_DTYPE)
    )
    inputs[2].fill_(1.0)
    inputs[3].fill_(1.0)
    inputs[4].copy_(torch.randn_like(inputs[4]) * 0.01)
    inputs[5].copy_(torch.randn_like(inputs[5]) * 0.01 + 1.0)
    inputs[6].copy_(
        (torch.randn_like(inputs[6], dtype=torch.bfloat16) * 0.01).to(FP8_DTYPE)
    )
    inputs[7].fill_(1.0)
    return inputs
