# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.config import (
    DeviceConfig,
    KernelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.kernels import linear as linear_kernels
from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
    RocmDotScaledMxfp8LinearKernel,
    RocmHipbmmMxfp8LinearKernel,
    _mxfp8_dot_scaled_linear_gfx950,
    _rocm_aiter_mxfp8_quantize_gfx950_fake,
    _rocm_mxfp8_dot_scaled_mm_gfx950_impl,
    _rocm_mxfp8_dot_scaled_mm_impl,
    shuffle_mxfp8_scale_gfx950,
)
from vllm.platforms import current_platform


def _gfx950_scale_offset(row: int, column: int, padded_columns: int) -> int:
    row_block = row // 32
    row_half = (row // 16) % 2
    row_in_half = row % 16
    column_block = column // 8
    column_half = (column // 4) % 2
    column_in_half = column % 4
    return (
        row_block * padded_columns * 32
        + column_block * 256
        + column_in_half * 64
        + row_in_half * 4
        + column_half * 2
        + row_half
    )


@pytest.mark.parametrize(
    ("rows", "columns"),
    [(1, 8), (31, 9), (32, 16), (33, 17), (65, 24)],
)
def test_shuffle_mxfp8_scale_gfx950_exact_layout(rows: int, columns: int) -> None:
    scale = torch.arange(rows * columns, dtype=torch.int64).reshape(rows, columns)
    scale = (scale % 251).to(torch.uint8)

    shuffled = shuffle_mxfp8_scale_gfx950(scale)

    padded_rows = (rows + 31) // 32 * 32
    padded_columns = (columns + 7) // 8 * 8
    assert shuffled.shape == (padded_rows, padded_columns)
    assert shuffled.is_contiguous()

    shuffled_storage = shuffled.flatten()
    occupied = torch.zeros_like(shuffled_storage, dtype=torch.bool)
    for row in range(rows):
        for column in range(columns):
            offset = _gfx950_scale_offset(row, column, padded_columns)
            assert shuffled_storage[offset] == scale[row, column]
            occupied[offset] = True
    assert torch.count_nonzero(shuffled_storage[~occupied]) == 0


@pytest.mark.parametrize(
    ("rows", "columns", "expected_scale_shape"),
    [
        (1, 256, (32, 8)),
        (33, 256, (64, 8)),
        (33, 512, (64, 16)),
        (33, 29568, (64, 928)),
    ],
)
def test_gfx950_quantize_fake_uses_padded_scale_storage(
    rows: int,
    columns: int,
    expected_scale_shape: tuple[int, int],
) -> None:
    input = torch.empty(rows, columns, dtype=torch.bfloat16)

    output, scale = _rocm_aiter_mxfp8_quantize_gfx950_fake(input)

    assert output.shape == input.shape
    assert output.dtype == torch.float8_e4m3fn
    assert scale.shape == expected_scale_shape
    assert scale.dtype == torch.uint8


def test_gfx950_quantize_requires_k_multiple_of_128() -> None:
    input = torch.empty(1, 64, dtype=torch.bfloat16)

    with pytest.raises(
        AssertionError, match="input columns must be divisible by 128, got 64"
    ):
        _rocm_aiter_mxfp8_quantize_gfx950_fake(input)


def test_gfx950_quantize_rejects_unsupported_input_dtype() -> None:
    input = torch.empty(1, 256, dtype=torch.float32)

    with pytest.raises(
        AssertionError,
        match="input dtype must be float16 or bfloat16, got torch.float32",
    ):
        _rocm_aiter_mxfp8_quantize_gfx950_fake(input)


def test_gfx950_scale_layout_kernel_requires_k_multiple_of_128() -> None:
    input = torch.empty(1, 64, dtype=torch.float8_e4m3fn)
    weight = torch.empty(32, 64, dtype=torch.float8_e4m3fn)
    input_scale = torch.empty(32, 8, dtype=torch.uint8)
    weight_scale = torch.empty(32, 8, dtype=torch.uint8)

    with pytest.raises(
        AssertionError, match="input columns must be divisible by 128, got 64"
    ):
        _rocm_mxfp8_dot_scaled_mm_gfx950_impl(
            input,
            input_scale,
            weight,
            weight_scale,
            torch.bfloat16,
        )


@pytest.mark.parametrize("input_size", [512, 29568])
def test_hipbmm_backend_replaces_weight_scale_with_gfx950_layout(
    input_size: int,
) -> None:
    output_size = 33
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    row_major_scale = torch.arange(
        output_size * (input_size // 32), dtype=torch.int64
    ).reshape(output_size, input_size // 32)
    row_major_scale = (row_major_scale % 251).to(torch.uint8)
    layer.weight_scale = torch.nn.Parameter(row_major_scale, requires_grad=False)
    kernel = object.__new__(RocmHipbmmMxfp8LinearKernel)

    kernel.process_weights_after_loading(layer)

    assert torch.equal(
        layer.weight_scale,
        shuffle_mxfp8_scale_gfx950(row_major_scale),
    )
    expected_scale_columns = (input_size // 32 + 7) // 8 * 8
    assert layer.weight_scale.shape == (64, expected_scale_columns)
    assert layer.weight_scale.is_contiguous()


def test_hipbmm_backend_graph_uses_gfx950_ops() -> None:
    output_size = 32
    input_size = 256
    graph = make_fx(_mxfp8_dot_scaled_linear_gfx950, tracing_mode="fake")(
        torch.empty(1, input_size, dtype=torch.bfloat16),
        torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
        torch.empty(32, 8, dtype=torch.uint8),
    )
    targets = {node.target for node in graph.graph.nodes}

    assert torch.ops.vllm.rocm_aiter_mxfp8_quantize_gfx950.default in targets
    assert torch.ops.vllm.rocm_mxfp8_dot_scaled_mm_gfx950.default in targets
    assert torch.ops.vllm.mxfp8_quantize.default not in targets
    assert torch.ops.vllm.rocm_mxfp8_dot_scaled_mm.default not in targets


@pytest.mark.parametrize(
    ("use_hipbmm_fused_epilogue", "expected_kernel"),
    [
        (False, RocmDotScaledMxfp8LinearKernel),
        (True, RocmHipbmmMxfp8LinearKernel),
    ],
)
def test_hipbmm_feature_selects_mxfp8_backend(
    monkeypatch: pytest.MonkeyPatch,
    use_hipbmm_fused_epilogue: bool,
    expected_kernel: type,
) -> None:
    monkeypatch.setitem(
        linear_kernels._POSSIBLE_MXFP8_KERNELS,
        current_platform._enum,
        [RocmDotScaledMxfp8LinearKernel],
    )
    for kernel_type in (
        RocmDotScaledMxfp8LinearKernel,
        RocmHipbmmMxfp8LinearKernel,
    ):
        monkeypatch.setattr(
            kernel_type,
            "is_supported",
            classmethod(lambda cls, compute_capability=None: (True, None)),
        )
        monkeypatch.setattr(
            kernel_type,
            "can_implement",
            classmethod(lambda cls, config: (True, None)),
        )

    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        kernel_config=KernelConfig(use_hipbmm_fused_epilogue=use_hipbmm_fused_epilogue),
    )
    with set_current_vllm_config(config):
        kernel = linear_kernels.init_mxfp8_linear_kernel()

    assert type(kernel) is expected_kernel


def test_hipbmm_feature_changes_kernel_config_hash() -> None:
    disabled = KernelConfig(use_hipbmm_fused_epilogue=False)
    enabled = KernelConfig(use_hipbmm_fused_epilogue=True)

    assert disabled.compute_hash() != enabled.compute_hash()


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only default")
def test_hipbmm_feature_defaults_to_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_ROCM_USE_AITER_LINEAR_HIPBMM", True)

    inherited = VllmConfig(
        device_config=DeviceConfig(device="cuda"),
        kernel_config=KernelConfig(),
    )
    explicitly_disabled = VllmConfig(
        device_config=DeviceConfig(device="cuda"),
        kernel_config=KernelConfig(use_hipbmm_fused_epilogue=False),
    )

    assert inherited.kernel_config.use_hipbmm_fused_epilogue is True
    assert explicitly_disabled.kernel_config.use_hipbmm_fused_epilogue is False


@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_mx()),
    reason="The native MXFP8 GEMM requires a ROCm gfx95x GPU",
)
@pytest.mark.parametrize(
    ("tokens", "output_size", "input_size"),
    [(1, 33, 256), (1, 256, 256), (33, 384, 768)],
)
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_gfx950_scale_layout_kernel_matches_row_major(
    tokens: int,
    output_size: int,
    input_size: int,
    output_dtype: torch.dtype,
) -> None:
    torch.manual_seed(0)
    input = torch.randn(tokens, input_size, device="cuda").to(torch.float8_e4m3fn)
    weight = torch.randn(output_size, input_size, device="cuda").to(torch.float8_e4m3fn)
    input_scale = torch.randint(
        119,
        128,
        (tokens, input_size // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    weight_scale = torch.randint(
        119,
        128,
        (output_size, input_size // 32),
        dtype=torch.uint8,
        device="cuda",
    )

    expected = _rocm_mxfp8_dot_scaled_mm_impl(
        input,
        input_scale,
        weight,
        weight_scale,
        output_dtype,
    )
    actual = _rocm_mxfp8_dot_scaled_mm_gfx950_impl(
        input,
        shuffle_mxfp8_scale_gfx950(input_scale),
        weight,
        shuffle_mxfp8_scale_gfx950(weight_scale),
        output_dtype,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_mx()),
    reason="The native MXFP8 GEMM requires a ROCm gfx95x GPU",
)
def test_gfx950_scale_layout_kernel_reads_final_128_value_tile() -> None:
    tokens = 2
    output_size = 32
    input_size = 29568
    torch.manual_seed(3)
    input = torch.zeros(tokens, input_size, dtype=torch.float8_e4m3fn, device="cuda")
    weight = torch.zeros(
        output_size, input_size, dtype=torch.float8_e4m3fn, device="cuda"
    )
    input[:, -128:] = torch.randn(tokens, 128, device="cuda").to(torch.float8_e4m3fn)
    weight[:, -128:] = torch.randn(output_size, 128, device="cuda").to(
        torch.float8_e4m3fn
    )
    input_scale = torch.full(
        (tokens, input_size // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    )
    weight_scale = torch.full(
        (output_size, input_size // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    )

    expected = _rocm_mxfp8_dot_scaled_mm_impl(
        input,
        input_scale,
        weight,
        weight_scale,
        torch.bfloat16,
    )
    actual = _rocm_mxfp8_dot_scaled_mm_gfx950_impl(
        input,
        shuffle_mxfp8_scale_gfx950(input_scale),
        weight,
        shuffle_mxfp8_scale_gfx950(weight_scale),
        torch.bfloat16,
    )
    torch.cuda.synchronize()

    assert torch.count_nonzero(expected) > 0
    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_mx()),
    reason="The native MXFP8 GEMM requires a ROCm gfx95x GPU",
)
def test_aiter_producer_feeds_gfx950_scale_layout_kernel() -> None:
    from aiter.ops.quant import dynamic_per_group_scaled_quant

    tokens = 33
    output_size = 384
    input_size = 768
    torch.manual_seed(1)
    input = torch.randn(
        tokens, input_size, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    weight = torch.randn(output_size, input_size, device="cuda").to(torch.float8_e4m3fn)
    weight_scale = torch.randint(
        119,
        128,
        (output_size, input_size // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    row_quantized = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    row_scale = torch.zeros(
        tokens,
        input_size // 32,
        dtype=torch.uint8,
        device="cuda",
    )
    dynamic_per_group_scaled_quant(
        row_quantized,
        input,
        row_scale,
        group_size=32,
        shuffle_scale=False,
    )

    quantized, scale = torch.ops.vllm.rocm_aiter_mxfp8_quantize_gfx950(input)

    assert torch.equal(quantized, row_quantized)
    expected = _rocm_mxfp8_dot_scaled_mm_impl(
        row_quantized,
        row_scale,
        weight,
        weight_scale,
        torch.bfloat16,
    )
    actual = _rocm_mxfp8_dot_scaled_mm_gfx950_impl(
        quantized,
        scale,
        weight,
        shuffle_mxfp8_scale_gfx950(weight_scale),
        torch.bfloat16,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_mx()),
    reason="The native MXFP8 GEMM requires a ROCm gfx95x GPU",
)
def test_gfx950_scale_layout_linear_compiles() -> None:
    input_size = 256
    output_size = 33
    torch.manual_seed(2)
    input = torch.randn(1, input_size, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(output_size, input_size, device="cuda").to(torch.float8_e4m3fn)
    weight_scale = torch.randint(
        119,
        128,
        (output_size, input_size // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    weight_scale = shuffle_mxfp8_scale_gfx950(weight_scale)
    compiled_linear = torch.compile(
        _mxfp8_dot_scaled_linear_gfx950,
        backend="inductor",
        fullgraph=True,
    )

    expected = _mxfp8_dot_scaled_linear_gfx950(input, weight, weight_scale)
    actual = compiled_linear(input, weight, weight_scale)
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
