# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

from tests.compile.passes.rocm_aiter_test_utils import (
    configure_therock_rocm_environment,
)

configure_therock_rocm_environment()

import operator
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import fx
from torch._ops import OpOverload
from torch.fx.experimental.proxy_tensor import make_fx

import vllm.config
import vllm.ir.ops
import vllm.kernels.aiter_ops  # noqa: F401
import vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe as rocm_aiter_moe
from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.compilation.passes.fusion.rocm_aiter_blockscale_gemm_rmsnorm_quant_fusion import (  # noqa: E501
    DECODE_GEMM_OP,
    FUSED_ADD_RMS_NORM_OP,
    GROUP_QUANT_OP,
    PREFILL_GEMM_OP,
    QWEN_EPSILON,
    QWEN_GROUP_SIZE,
    REFERENCE_OP,
    RocmAiterBlockscaleGemmRMSNormQuantFusionPass,
)
from vllm.config import CompilationConfig, CompilationMode, VllmConfig
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.platforms import current_platform

FP8_DTYPE = current_platform.fp8_dtype()
TOKENS = 4
INPUT_SIZE = 256
HIDDEN_SIZE = 256
ROUTER_SIZE = 8


def _qwen_chain(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    router_weight: torch.Tensor,
    *,
    gemm_op: OpOverload,
    epsilon: float = QWEN_EPSILON,
    group_size: int = QWEN_GROUP_SIZE,
    output_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gemm = gemm_op(
        input,
        weight,
        input_scale,
        weight_scale,
        output_dtype,
    )
    normalized, residual_out = FUSED_ADD_RMS_NORM_OP(
        gemm,
        residual,
        rms_weight,
        epsilon,
    )

    # Qwen's normalized BF16 value has an independent router consumer. The
    # fusion must preserve this edge instead of requiring the normalized value
    # to be private to group quantization.
    router = torch.mm(normalized, router_weight)
    quantized, scales = GROUP_QUANT_OP(normalized, group_size)
    return router, residual_out, quantized, scales


def _example_inputs(
    *,
    tokens: int = TOKENS,
    device: str = "cuda",
) -> list[torch.Tensor]:
    return [
        torch.empty(tokens, INPUT_SIZE, dtype=FP8_DTYPE, device=device),
        torch.empty(HIDDEN_SIZE, INPUT_SIZE, dtype=FP8_DTYPE, device=device),
        torch.empty(tokens, INPUT_SIZE // 128, dtype=torch.float32, device=device),
        torch.empty(
            HIDDEN_SIZE // 128,
            INPUT_SIZE // 128,
            dtype=torch.float32,
            device=device,
        ),
        torch.empty(tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
        torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
        torch.empty(
            HIDDEN_SIZE,
            ROUTER_SIZE,
            dtype=torch.bfloat16,
            device=device,
        ),
    ]


def _trace_chain(
    *,
    gemm_op: OpOverload,
    epsilon: float = QWEN_EPSILON,
    group_size: int = QWEN_GROUP_SIZE,
    output_dtype: torch.dtype = torch.bfloat16,
) -> fx.GraphModule:
    def f(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _qwen_chain(
            *args,
            gemm_op=gemm_op,
            epsilon=epsilon,
            group_size=group_size,
            output_dtype=output_dtype,
        )

    return make_fx(f, tracing_mode="fake")(*_example_inputs())


def _run_fusion(graph: fx.Graph) -> RocmAiterBlockscaleGemmRMSNormQuantFusionPass:
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
    )
    with vllm.config.set_current_vllm_config(vllm_config):
        fusion_pass = RocmAiterBlockscaleGemmRMSNormQuantFusionPass(vllm_config)
        fusion_pass(graph)
    return fusion_pass


def _nodes_with_target(graph: fx.Graph, target: object) -> list[fx.Node]:
    return [node for node in graph.nodes if node.target == target]


@pytest.mark.parametrize(
    ("gemm_op", "use_triton_gemm"),
    [
        pytest.param(PREFILL_GEMM_OP, True, id="prefill-cktile"),
        pytest.param(DECODE_GEMM_OP, False, id="decode-ck"),
    ],
)
def test_fusion_replaces_only_qwen_group_quant_outputs(
    gemm_op: OpOverload,
    use_triton_gemm: bool,
) -> None:
    graph_module = _trace_chain(gemm_op=gemm_op)
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 1
    assert len(_nodes_with_target(graph_module.graph, REFERENCE_OP)) == 1
    assert not _nodes_with_target(graph_module.graph, GROUP_QUANT_OP)

    # The original GEMM and RMSNorm remain live. In particular, the normalized
    # RMSNorm getitem still feeds Qwen's router matmul, while the residual
    # getitem remains a graph output.
    gemm = _nodes_with_target(graph_module.graph, gemm_op)[0]
    rmsnorm = _nodes_with_target(graph_module.graph, FUSED_ADD_RMS_NORM_OP)[0]
    router = _nodes_with_target(graph_module.graph, torch.ops.aten.mm.default)[0]
    normalized = router.args[0]
    assert isinstance(normalized, fx.Node)
    assert normalized.target == operator.getitem
    assert normalized.args == (rmsnorm, 0)
    assert rmsnorm.args[0] is gemm

    fused = _nodes_with_target(graph_module.graph, REFERENCE_OP)[0]
    assert fused.args[:4] == gemm.args[:4]
    assert fused.args[4:7] == rmsnorm.args[1:4]
    assert fused.args[7:] == (QWEN_GROUP_SIZE, use_triton_gemm)

    fused_getitems: set[int] = set()
    for node in fused.users:
        if node.target != operator.getitem or node.args[0] is not fused:
            continue
        index = node.args[1]
        assert isinstance(index, int)
        fused_getitems.add(index)
    assert fused_getitems == {0, 1}


@pytest.mark.parametrize(
    ("epsilon", "group_size", "output_dtype"),
    [
        pytest.param(1e-5, QWEN_GROUP_SIZE, torch.bfloat16, id="wrong-epsilon"),
        pytest.param(QWEN_EPSILON, 64, torch.bfloat16, id="wrong-group-size"),
        pytest.param(QWEN_EPSILON, QWEN_GROUP_SIZE, torch.float16, id="wrong-dtype"),
    ],
)
def test_fusion_rejects_non_qwen_constants(
    epsilon: float,
    group_size: int,
    output_dtype: torch.dtype,
) -> None:
    graph_module = _trace_chain(
        gemm_op=PREFILL_GEMM_OP,
        epsilon=epsilon,
        group_size=group_size,
        output_dtype=output_dtype,
    )
    fusion_pass = _run_fusion(graph_module.graph)

    assert fusion_pass.matched_count == 0
    assert not _nodes_with_target(graph_module.graph, REFERENCE_OP)
    assert len(_nodes_with_target(graph_module.graph, GROUP_QUANT_OP)) == 1


def test_fusion_rejects_unexpected_quant_tuple_user() -> None:
    graph_module = _trace_chain(gemm_op=PREFILL_GEMM_OP)
    graph = graph_module.graph
    quant = _nodes_with_target(graph, GROUP_QUANT_OP)[0]
    output = next(node for node in graph.nodes if node.op == "output")

    # Add a second observation of tuple element 0. The rewrite redirects the
    # two known tuple elements, so any third tuple user must conservatively
    # block the match.
    with graph.inserting_before(output):
        extra_quantized = graph.call_function(operator.getitem, args=(quant, 0))
    original_outputs = output.args[0]
    assert isinstance(original_outputs, tuple)
    output.args = (original_outputs + (extra_quantized,),)
    graph.lint()

    fusion_pass = _run_fusion(graph)

    assert fusion_pass.matched_count == 0
    assert not _nodes_with_target(graph, REFERENCE_OP)
    assert len(_nodes_with_target(graph, GROUP_QUANT_OP)) == 1


def test_runner_hoists_quantization_from_directly_owned_modular_kernel() -> None:
    hidden_states = torch.empty(TOKENS, HIDDEN_SIZE)
    expected = (torch.empty_like(hidden_states), torch.empty(TOKENS, 2))

    class Kernel:
        def maybe_prequantize_input(
            self, actual_hidden_states: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert actual_hidden_states is hidden_states
            return expected

    class RoutedExperts:
        def __init__(self) -> None:
            self.quant_method = SimpleNamespace(moe_kernel=Kernel())
            self.quant_config_initialized = False

        def _ensure_moe_quant_config_init(self) -> None:
            self.quant_config_initialized = True

    routed_experts = RoutedExperts()
    runner = SimpleNamespace(
        do_naive_dispatch_combine=False,
        moe_config=SimpleNamespace(pcp_size=1),
        routed_experts=routed_experts,
    )

    actual = MoERunner._maybe_prequantize_input(cast(Any, runner), hidden_states)

    assert routed_experts.quant_config_initialized
    assert actual is expected


def test_routed_experts_pass_prequantized_input_to_directly_owned_kernel() -> None:
    hidden_states = torch.empty(TOKENS, HIDDEN_SIZE)
    quantized = torch.empty_like(hidden_states, dtype=FP8_DTYPE)
    scales = torch.empty(TOKENS, HIDDEN_SIZE // QWEN_GROUP_SIZE)
    expected = torch.empty_like(hidden_states)

    class Kernel:
        def __init__(self) -> None:
            self.arguments: dict[str, object] = {}

        def apply(self, **kwargs: object) -> torch.Tensor:
            self.arguments = kwargs
            return expected

    kernel = Kernel()
    routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(is_monolithic=False, moe_kernel=kernel),
        w13_weight=torch.empty(1),
        w2_weight=torch.empty(1),
        activation=object(),
        global_num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    topk_weights = torch.empty(TOKENS, 2)
    topk_ids = torch.empty(TOKENS, 2, dtype=torch.int32)

    actual = RoutedExperts.forward_modular(
        cast(Any, routed_experts),
        hidden_states,
        topk_weights,
        topk_ids,
        prequantized_hidden_states=quantized,
        prequantized_hidden_states_scale=scales,
    )

    assert actual is expected
    assert kernel.arguments["hidden_states"] is hidden_states
    assert kernel.arguments["prequantized_hidden_states"] is quantized
    assert kernel.arguments["prequantized_hidden_states_scale"] is scales


def test_aiter_external_blockscale_input_supplies_valid_row_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quantized = torch.empty(TOKENS, HIDDEN_SIZE, dtype=FP8_DTYPE)
    scales = torch.empty(TOKENS, HIDDEN_SIZE // QWEN_GROUP_SIZE)
    output = torch.empty(TOKENS, HIDDEN_SIZE)
    captured: dict[str, object] = {}

    def fake_fused_experts(**kwargs: object) -> torch.Tensor:
        captured.update(kwargs)
        return torch.empty_like(output)

    monkeypatch.setattr(
        rocm_aiter_moe,
        "rocm_aiter_fused_experts",
        fake_fused_experts,
    )
    monkeypatch.setattr(
        rocm_aiter_moe.rocm_aiter_ops,
        "get_moe_dispatch_policy",
        lambda: 0,
    )
    experts = SimpleNamespace(
        quant_config=SimpleNamespace(
            block_shape=[128, 128],
            quant_dtype=FP8_DTYPE,
        ),
        moe_config=object(),
    )

    rocm_aiter_moe.AiterExperts.apply(
        cast(Any, experts),
        output=output,
        hidden_states=quantized,
        w1=torch.empty(1),
        w2=torch.empty(1),
        topk_weights=torch.empty(TOKENS, 2),
        topk_ids=torch.empty(TOKENS, 2, dtype=torch.int32),
        activation=cast(Any, object()),
        global_num_experts=8,
        expert_map=None,
        a1q_scale=scales,
        a2_scale=None,
        workspace13=torch.empty(0),
        workspace2=torch.empty(0),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    num_local_tokens = captured["num_local_tokens"]
    assert isinstance(num_local_tokens, torch.Tensor)
    assert num_local_tokens.dtype == torch.int32
    assert num_local_tokens.tolist() == [TOKENS]


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)
@pytest.mark.parametrize(
    ("gemm_op", "tokens", "use_triton_gemm"),
    [
        pytest.param(PREFILL_GEMM_OP, TOKENS, True, id="prefill-cktile"),
        pytest.param(DECODE_GEMM_OP, 1, False, id="decode-ck"),
    ],
)
def test_reference_op_is_bitwise_equal_to_unfused_chain(
    gemm_op: OpOverload,
    tokens: int,
    use_triton_gemm: bool,
) -> None:
    torch.manual_seed(0)
    inputs = _random_example_inputs(tokens=tokens)

    # Match the worker-level priority used by the Qwen model validation. The
    # reference op calls vllm_ir RMSNorm, so its selected implementation is part
    # of the execution contract being checked here.
    with (
        torch.no_grad(),
        vllm.ir.ops.fused_add_rms_norm.set_priority(["aiter", "native"]),
    ):
        expected = _qwen_quant_outputs(inputs, gemm_op)
        actual = REFERENCE_OP(
            *inputs[:6],
            QWEN_EPSILON,
            QWEN_GROUP_SIZE,
            use_triton_gemm,
        )
    torch.cuda.synchronize()

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


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
    inputs[5].copy_(torch.randn_like(inputs[5]) * 0.01)
    inputs[6].copy_(torch.randn_like(inputs[6]) * 0.01)
    return inputs


def _qwen_quant_outputs(
    inputs: list[torch.Tensor],
    gemm_op: Callable[..., torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    gemm = gemm_op(*inputs[:4], torch.bfloat16)
    # The model's IR RMSNorm is lowered to this AITER implementation before
    # execution. Compare the replacement against that actual execution path,
    # rather than calling the native vllm_ir correctness reference here.
    normalized, _ = torch.ops.vllm_aiter.fused_add_rms_norm.default(
        gemm,
        inputs[4],
        inputs[5],
        QWEN_EPSILON,
    )
    return GROUP_QUANT_OP(normalized, QWEN_GROUP_SIZE)
