# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native MXFP8 linear GEMM for AMD CDNA4 (gfx950) via Triton ``tl.dot_scaled``.

Consumes the FP8 E4M3 weights + E8M0 block scales directly (no dequant-to-BF16);
activations are MXFP8-quantized per token. Uses the CDNA4 hardware microscaling
matrix cores. Falls back (via the kernel selector) to the BF16
``EmulationMxfp8LinearKernel`` on archs without native MX or for shapes with
``K % 128 != 0``.
"""

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


def shuffle_mxfp8_scale_gfx950(scale: torch.Tensor) -> torch.Tensor:
    """Convert row-major MXFP8 scales to gfx950 ``32x8`` storage."""
    if scale.ndim != 2:
        raise ValueError(f"scale must be two-dimensional, got shape {scale.shape}")

    rows, columns = scale.shape
    padded_rows = (rows + 31) // 32 * 32
    padded_columns = (columns + 7) // 8 * 8
    padded = torch.zeros(
        (padded_rows, padded_columns), dtype=scale.dtype, device=scale.device
    )
    padded[:rows, :columns] = scale
    return (
        padded.view(padded_rows // 32, 2, 16, padded_columns // 8, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .contiguous()
        .view(padded_rows, padded_columns)
    )


@triton.jit
def _gfx950_scale_offset(row, column, padded_columns):
    return (
        (row // 32) * padded_columns * 32
        + (column // 8) * 256
        + (column % 4) * 64
        + (row % 16) * 4
        + ((column // 4) % 2) * 2
        + ((row // 16) % 2)
    )


@triton.jit
def _mxfp8_linear_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_xsm,
    stride_xsk,
    stride_wn,
    stride_wk,
    stride_wsn,
    stride_wsk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALES_GFX950: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    if SCALES_GFX950:
        tl.static_assert(BLOCK_K % 128 == 0)
    else:
        xs_ptrs = xs_ptr + offs_m[:, None] * stride_xsm + offs_sk[None, :] * stride_xsk
        ws_ptrs = ws_ptr + offs_n[:, None] * stride_wsn + offs_sk[None, :] * stride_wsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        if SCALES_GFX950:
            scale_columns = k_block * (BLOCK_K // 32) + offs_sk
            xs_ptrs = xs_ptr + _gfx950_scale_offset(
                offs_m[:, None], scale_columns[None, :], stride_xsm
            )
            ws_ptrs = ws_ptr + _gfx950_scale_offset(
                offs_n[:, None], scale_columns[None, :], stride_wsn
            )
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
        xs = tl.load(xs_ptrs, mask=m_mask[:, None], other=0)
        ws = tl.load(ws_ptrs, mask=n_mask[:, None], other=0)
        acc += tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3")
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        if not SCALES_GFX950:
            xs_ptrs += (BLOCK_K // 32) * stride_xsk
            ws_ptrs += (BLOCK_K // 32) * stride_wsk

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :]
    )


def _rocm_mxfp8_dot_scaled_mm_impl(
    x: torch.Tensor,  # [M, K] fp8 e4m3
    x_scale: torch.Tensor,  # [M, K//32] uint8 (E8M0)
    w: torch.Tensor,  # [N, K] fp8 e4m3
    w_scale: torch.Tensor,  # [N, K//32] uint8 (E8M0)
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return _rocm_mxfp8_dot_scaled_mm_launch(
        x, x_scale, w, w_scale, output_dtype, scales_gfx950=False
    )


def _rocm_mxfp8_dot_scaled_mm_gfx950_impl(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    assert K % 128 == 0, f"input columns must be divisible by 128, got {K}"
    expected_scale_columns = (K // MXFP8_BLOCK_SIZE + 7) // 8 * 8
    assert x_scale.shape == ((M + 31) // 32 * 32, expected_scale_columns)
    assert w_scale.shape == ((N + 31) // 32 * 32, expected_scale_columns)
    assert x_scale.dtype == MXFP8_SCALE_DTYPE
    assert w_scale.dtype == MXFP8_SCALE_DTYPE
    assert x_scale.is_contiguous()
    assert w_scale.is_contiguous()
    return _rocm_mxfp8_dot_scaled_mm_launch(
        x, x_scale, w, w_scale, output_dtype, scales_gfx950=True
    )


def _rocm_mxfp8_dot_scaled_mm_launch(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    output_dtype: torch.dtype,
    *,
    scales_gfx950: bool,
) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), dtype=output_dtype, device=x.device)
    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = _select_cfg(M, N, K)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _mxfp8_linear_kernel[grid](
        x,
        x_scale,
        w,
        w_scale,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        w.stride(0),
        w.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SCALES_GFX950=scales_gfx950,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _rocm_mxfp8_dot_scaled_mm_fake(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    del x_scale, w_scale
    return torch.empty(
        (x.shape[0], w.shape[0]),
        dtype=output_dtype,
        device=x.device,
    )


def _rocm_aiter_mxfp8_quantize_gfx950_impl(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.quant import dynamic_per_group_scaled_quant

    output, scale = _allocate_mxfp8_quantize_gfx950_outputs(input)
    dynamic_per_group_scaled_quant(
        output,
        input,
        scale,
        group_size=MXFP8_BLOCK_SIZE,
        shuffle_scale=True,
    )
    return output, scale


def _rocm_aiter_mxfp8_quantize_gfx950_fake(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _allocate_mxfp8_quantize_gfx950_outputs(input)


def _allocate_mxfp8_quantize_gfx950_outputs(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = input.shape
    assert input.dtype in {
        torch.float16,
        torch.bfloat16,
    }, f"input dtype must be float16 or bfloat16, got {input.dtype}"
    assert columns % 128 == 0, f"input columns must be divisible by 128, got {columns}"
    padded_rows = (rows + 31) // 32 * 32
    scale_columns = (columns // MXFP8_BLOCK_SIZE + 7) // 8 * 8
    return (
        torch.empty_like(input, dtype=torch.float8_e4m3fn),
        torch.empty(
            (padded_rows, scale_columns),
            dtype=MXFP8_SCALE_DTYPE,
            device=input.device,
        ),
    )


direct_register_custom_op(
    op_name="rocm_mxfp8_dot_scaled_mm",
    op_func=_rocm_mxfp8_dot_scaled_mm_impl,
    fake_impl=_rocm_mxfp8_dot_scaled_mm_fake,
)

direct_register_custom_op(
    op_name="rocm_aiter_mxfp8_quantize_gfx950",
    op_func=_rocm_aiter_mxfp8_quantize_gfx950_impl,
    fake_impl=_rocm_aiter_mxfp8_quantize_gfx950_fake,
)

direct_register_custom_op(
    op_name="rocm_mxfp8_dot_scaled_mm_gfx950",
    op_func=_rocm_mxfp8_dot_scaled_mm_gfx950_impl,
    fake_impl=_rocm_mxfp8_dot_scaled_mm_fake,
)


def _mxfp8_dot_scaled_linear(
    x: torch.Tensor,  # [M, K] bf16/fp16
    w: torch.Tensor,  # [N, K] fp8 e4m3
    w_scale: torch.Tensor,  # [N, K//32] uint8 (E8M0)
) -> torch.Tensor:
    x_q, x_scale = mxfp8_e4m3_quantize(x)
    return torch.ops.vllm.rocm_mxfp8_dot_scaled_mm(
        x_q,
        x_scale,
        w,
        w_scale,
        x.dtype,
    )


def _mxfp8_dot_scaled_linear_gfx950(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    x_q, x_scale = torch.ops.vllm.rocm_aiter_mxfp8_quantize_gfx950(x)
    return torch.ops.vllm.rocm_mxfp8_dot_scaled_mm_gfx950(
        x_q,
        x_scale,
        w,
        w_scale,
        x.dtype,
    )


def _select_cfg(M, N, K):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) — graph-tuned on gfx950.

    The M-bucketed, shape-adaptive tile selection here is the speedup over the
    upstream 2-bucket launcher. Tiles are pipelined (num_stages>=2, larger BLOCK_K)
    and occupancy- and shape-aware: keyed on the LOCAL (M, N, K), so it adapts to the
    TP-sharded shapes (e.g. MiniMax-M3 TP=4 vs TP=8, where local N and K differ) —
    large-K prefill uses BLOCK_K=256; short-K (K=768) widens N. BLOCK_K must divide K
    (the K-loop is unmasked), so every BLOCK_K below is guarded to be K-divisible
    (served K: 384/768/1024/2048/6144).
    """
    if M <= 64:
        # decode (M in {1,32,64}): tiny-M GEMV is weight-BW + GPU-OCCUPANCY bound. The
        # lever is NARROW BLOCK_N=16 (maximize N-tiles so more CUs stream the weight in
        # parallel) + LARGE BLOCK_K (fewer K-iters, bigger coalesced weight loads).
        # Tuned by CUDA-graph replay latency. Optimal at both TP=4 and TP=8.
        if K % 1024 == 0:  # K=2048, 6144 -> graph-best 16x16x1024 (all M)
            return 16, 16, 1024, 2, 2
        if K % 512 == 0:
            return 16, 16, 512, 2, 3
        if K % 256 == 0:  # K=768 (shared_down) -> graph-best 16x32x256
            return 16, 32, 256, 4, 3
        return 16, 32, 128, 4, 3
    # mid-M (65..256) on SMALL local-N: still occupancy-bound (a 64x64 tile makes too
    # few N-tiles), so the narrow-BLOCK_N decode-style tile fills the CUs better.
    # N<=1536 covers the real fused-qkv local N at TP=8: q heads shard but the GQA KV
    # (4) + sparse-indexer (4) heads are < TP=8, so vLLM replicates them to 1/rank ->
    # N = 1024 + 4*128 = 1536 (not 2560/2=1280). For the wider 1280<N<=1536 band the
    # narrow tile only wins up to M=128 (at M=256 the 64x64 tile is better), so cap it
    # there; N<=1280 keeps the narrow tile through M=256. TP=4 qkv N=2560 is unchanged.
    if (M <= 256 and N <= 1280) or (M <= 128 and N <= 1536):
        if K % 1024 == 0:
            return 16, 16, 1024, 2, 2
        if K % 512 == 0:
            return 16, 16, 512, 2, 3
        if K % 256 == 0:
            return 16, 16, 256, 2, 3
        return 16, 16, 128, 2, 3
    # right-sized launch grid (host-side), used to gate the tall 256-BLOCK_M tile.
    occ = triton.cdiv(M, 256) * triton.cdiv(N, 128)
    if K <= 1024:  # short-K (shared_down K=384/768; TP=8 o_proj K=1024)
        if M <= 256:
            return (64, 64, 256, 8, 2) if K % 256 == 0 else (64, 64, 128, 8, 2)
        # large prefill. (The former 256x128x256 tall tile was faster only on triton
        # 3.6; on triton 3.7 its large BLOCK_M register/LDS footprint spills or hits
        # "out of resources", so use 128x128x256 -- within the known-good footprint.)
        if M >= 4096 and K >= 1024 and K % 256 == 0 and occ >= 256:
            return 128, 128, 256, 8, 3
        return 128, 128, 128, 8, 3
    # large-K (K >= 2048). BLOCK_K is K-divisibility-guarded (the K-loop is unmasked):
    # served large-K is 2048/6144 (%256==0), but fall back to 128 (always divides, since
    # the entry requires K%128==0) for any other K to stay correct.
    if M <= 256:  # conc~128 decode + small prefill chunk: occupancy tile
        if K % 512 == 0:
            return 64, 64, 512, 8, 2
        return (64, 64, 256, 8, 2) if K % 256 == 0 else (64, 64, 128, 8, 2)
    if M <= 1024:  # medium prefill chunk: BN=64 keeps small-N occupied
        return (128, 64, 256, 8, 3) if K % 256 == 0 else (128, 64, 128, 8, 3)
    # large prefill (M > 1024). The previously graph-tuned tall 256x128x256 and deep
    # 128x128x512 tiles won only on triton 3.6; on triton 3.7 their larger BLOCK_M /
    # BLOCK_K register+LDS footprint spills (or hits "out of resources" on stricter
    # ROCm/triton builds). The 128x128x256 tile is equal-or-faster on triton 3.7 (the
    # M=4096,N=2048,K=6144 shape: ~104us vs the 256x128x256 tile's ~184us), within ~5%
    # on 3.6, and inside the footprint of the tiles used elsewhere in this selector.
    # Covers the qkv-class local N=1536 (TP=8 qkv / TP=4 shared_gate_up) and the deep-K
    # / very-large-M shapes.
    if K % 256 == 0 and (1280 < N <= 1536 or (occ >= 128 and (K >= 4096 or M >= 4096))):
        return 128, 128, 256, 8, 3
    # small local-N (e.g. TP=8 shared_gate_up N=768): a 64-wide BLOCK_N doubles the
    # N-tile count -> better CU fill than 128x128 at this mid-large M (~1.4x there).
    if N <= 1024 and K % 256 == 0:
        return 128, 64, 256, 8, 3
    return (128, 128, 256, 8, 2) if K % 256 == 0 else (128, 256, 128, 8, 3)


class RocmDotScaledMxfp8LinearKernel(Mxfp8LinearKernel):
    """Native CDNA4 (gfx950) MXFP8 linear via Triton ``tl.dot_scaled``."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "not ROCm"
        # supports_mx() == gfx95x (CDNA4 native microscaling hardware). On other
        # archs dot_scaled would upcast to BF16, so the kernel selector falls
        # through to the BF16 emulation (hipBLASLt) path instead.
        if not current_platform.supports_mx():
            return False, "native MX requires CDNA4 (gfx95x)"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K] fp8
        N, K = weight.shape
        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if layer.weight_scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(
                f"Expected {MXFP8_SCALE_DTYPE} weight_scale, got "
                f"{layer.weight_scale.dtype}."
            )
        out_shape = (*x.shape[:-1], layer.weight.shape[0])
        x2d = x.reshape(-1, x.shape[-1])
        if x2d.shape[-1] % 128 == 0:
            out = _mxfp8_dot_scaled_linear(x2d, layer.weight, layer.weight_scale)
        else:
            # dot_scaled tiling needs K % 128 == 0; dequantize fallback otherwise.
            w_bf16 = dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale)
            out = torch.nn.functional.linear(x2d, w_bf16).to(x.dtype)
        out = out.reshape(out_shape)
        if bias is not None:
            out = out + bias
        return out


class RocmHipbmmMxfp8LinearKernel(RocmDotScaledMxfp8LinearKernel):
    """gfx950 MXFP8 backend for the hipbmm fused-epilogue scale layout."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        N, K = weight.shape
        assert K % 128 == 0, f"weight columns must be divisible by 128, got {K}"
        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(
            shuffle_mxfp8_scale_gfx950(weight_scale), requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if layer.weight_scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(
                f"Expected {MXFP8_SCALE_DTYPE} weight_scale, got "
                f"{layer.weight_scale.dtype}."
            )
        assert x.shape[-1] % 128 == 0, (
            f"input columns must be divisible by 128, got {x.shape[-1]}"
        )
        out_shape = (*x.shape[:-1], layer.weight.shape[0])
        x2d = x.reshape(-1, x.shape[-1])
        out = _mxfp8_dot_scaled_linear_gfx950(
            x2d, layer.weight, layer.weight_scale
        ).reshape(out_shape)
        if bias is not None:
            out = out + bias
        return out
