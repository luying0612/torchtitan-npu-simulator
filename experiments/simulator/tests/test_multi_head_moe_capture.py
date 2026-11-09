# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-head-MoE capture contracts: virtual SLA2 branch fusions and memory plans."""

from collections import Counter

import pytest
import torch
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.capture.saved_tensors import AutogradSavedTensorCapture
from torchtitan_simulator.compat.multi_head_moe import (
    sla2_linear_attention,
    sla2_sparse_attention,
)
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.cost.op_cost_model import annotate_workload_costs
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.validation import validate_workload

# Debug-model geometry (torchtitan_npu.models.multi_head_moe.config_registry):
# dim=256, vocab=102400, 8 layers (front-2/back-2 dense FFN, MoE in between),
# n_heads=4, head_dim=64, block=64, router proj 64, inter_dim=512,
# MoE heads S=4, head hidden hs=64, Es=16, Ks=4, expert mult 2 (inter=128),
# shared inter 128, FP32 parameters.
_D, _V, _HID = 256, 102400, 512
_H, _HD, _PROJ = 4, 64, 64
_HS, _ES, _MULT = 64, 16, 2
_SHARED = 128
_DENSE_LAYER = (
    4 * (_H * _HD * _D)  # to_q/k/v/out
    + 2 * _HD  # norm_q + norm_k
    + 2 * (_PROJ * _HD)  # block router proj_q/proj_k
    + _H  # alpha [H, 1]
    + 2 * _D  # attention_norm + ffn_norm
    + 3 * (_HID * _D)  # dense w1/w2/w3
)
_MOE_LAYER = (
    _DENSE_LAYER
    - 3 * (_HID * _D)
    + _D * (_H * _HS)  # proj_in
    + _H * _HS * _ES  # gate router [S, hs, Es]
    + 3 * (_ES * _H) * (_HS * _MULT) * _HS  # grouped experts w1/w2/w3
    + (_H * _HS) * _D  # proj_out
    + 3 * (_SHARED * _D)  # shared w1/w2/w3
)
_TOTAL_PARAMS = 2 * _V * _D + _D + 4 * _DENSE_LAYER + 4 * _MOE_LAYER
# Dense layer: qkv/out + 2 head norms + 2 router projs + alpha + 2 block norms
# + w1/w2/w3 = 14; MoE layer: 14 attention/norm entries + proj_in + gate
# router + 3 experts + proj_out + 3 shared = 20; plus embeddings, head and
# final norm.
_TOTAL_PARAM_TENSORS = 3 + 4 * 14 + 4 * 20

_SPARSE_FWD = "multi_head_moe_sim.sla2_sparse_attention.default"
_SPARSE_BWD = "multi_head_moe_sim.sla2_sparse_attention_grad.default"
_LINEAR_FWD = "multi_head_moe_sim.sla2_linear_attention.default"
_LINEAR_BWD = "multi_head_moe_sim.sla2_linear_attention_grad.default"


def test_multi_head_moe_param_formula_matches_debug_geometry():
    assert _TOTAL_PARAMS == 63_395_104
    assert _TOTAL_PARAM_TENSORS == 139


@pytest.mark.parametrize(
    "world,replicate,rank,ac",
    [
        (1, 1, 0, "none"),
        (2, 1, 0, "none"),
        (2, 1, 1, "none"),
        (4, 2, 0, "none"),
        (1, 1, 0, "full"),
        (1, 1, 0, "selective"),
    ],
    ids=["single", "fsdp-first", "fsdp-last", "hsdp", "full-ac", "selective-ac"],
)
def test_multi_head_moe_capture_reaches_iteration_with_sla2_fusion_boundaries(world, replicate, rank, ac):
    workload = simulate(
        SimulationConfig(
            config="multi_head_moe_debugmodel",
            seq_len=16,
            world_size=world,
            logical_rank=rank,
            data_parallel_replicate_degree=replicate,
            activation_checkpoint_mode=ac,
        )
    )
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # 8 SLA2 layers. Full AC recomputes the forward inside backward; the
    # default selective policy saves only upstream compute-intensive ops, so
    # the virtual fusions are recomputed there as well.
    recompute = 2 if ac in ("full", "selective") else 1
    assert counts[_SPARSE_FWD] == 8 * recompute
    assert counts[_LINEAR_FWD] == 8 * recompute
    assert counts[_SPARSE_BWD] == 8
    assert counts[_LINEAR_BWD] == 8
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _TOTAL_PARAM_TENSORS
    degree = world // replicate
    memory = workload.iteration.schedule.annotations["memory_plan"]
    # The formulas below assume world is a power of two that divides the
    # parameter shard evenly (no padding imbalance); debug geometry guarantees
    # numel is even at every tested degree.
    local_params = _TOTAL_PARAMS if degree == 1 else _TOTAL_PARAMS // degree
    assert memory.persistent_param_bytes == local_params * 4
    # Native fused AdamW keeps two FP32 moments and one device FP32 step
    # scalar per participating parameter.
    assert memory.optimizer_state_bytes == local_params * 8 + _TOTAL_PARAM_TENSORS * 4
    custom = [n for n in nodes if "tensor_semantics" in n.attrs]
    assert {n.annotations["raw_op_type"] for n in custom} == {_SPARSE_FWD, _SPARSE_BWD, _LINEAR_FWD, _LINEAR_BWD}
    assert all(
        "tensor_semantics" not in n.attrs
        for n in nodes
        if n.annotations["raw_op_type"].startswith(("aten.", "npu.", "cann_ops_transformer."))
    )
    source_nodes = {
        node.op_id: node
        for graph in workload.captured_workload.step_templates.values()
        for node in graph.nodes.values()
    }
    for node in custom:
        assert node.attrs["tensor_semantics"] == source_nodes[node.op_id].attrs["tensor_semantics"]
        for key, tensors in (("inputs", node.inputs), ("outputs", node.outputs)):
            assert [item["index"] for item in node.attrs["tensor_semantics"][key]] == list(range(len(tensors)))
            for item in node.attrs["tensor_semantics"][key]:
                assert len(item["axes"]) == len(tensors[item["index"]].shape)


def test_multi_head_moe_ep2_dispatches_all_to_all_and_reaches_iteration():
    workload = simulate(
        SimulationConfig(
            config="multi_head_moe_debugmodel",
            seq_len=16,
            world_size=2,
            logical_rank=0,
            expert_parallel_degree=2,
        )
    )
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # 4 MoE layers x (token dispatch + gradient combine) per dispatch pair.
    assert counts["_c10d_functional.all_to_all_single.default"] == 20
    # Collectives carry their logical cross-rank volume: in+out bytes, once,
    # independently recomputed here from the captured tensor facts.
    from torchtitan_simulator.capture.tensor_utils import tensor_volume_bytes

    annotate_workload_costs(workload)
    comm = [n for n in nodes if n.op_type in ("all_to_all", "reduce_scatter")]
    assert len(comm) == 20 + 11
    for node in comm:
        assert not node.annotations.get("cost_unknown")
        expected = sum(
            tensor_volume_bytes(t.shape, t.dtype) for t in (*node.inputs, *node.outputs)
        )
        assert node.annotations["comm_bytes"] == expected
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _TOTAL_PARAM_TENSORS


def test_multi_head_moe_rejects_tensor_parallel_at_config_level():
    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        simulate(
            SimulationConfig(
                config="multi_head_moe_debugmodel",
                seq_len=16,
                world_size=2,
                tensor_parallel_degree=2,
            )
        )


def test_multi_head_moe_rejects_context_parallel():
    with pytest.raises(NotImplementedError, match="context parallel"):
        simulate(
            SimulationConfig(
                config="multi_head_moe_debugmodel",
                seq_len=16,
                world_size=2,
                context_parallel_degree=2,
            )
        )


def test_multi_head_moe_preset_rejects_mtp_modules():
    from torchtitan_simulator.presets import multi_head_moe_debugmodel

    with pytest.raises(ValueError, match="no MTP modules"):
        multi_head_moe_debugmodel(num_mtp_layers=1)


def test_multi_head_moe_fusion_semantics_declare_roles_and_axes():
    query = torch.empty(1, _H, 16, _HD, device="meta", requires_grad=True)
    key = torch.empty(1, _H, 16, _HD, device="meta", requires_grad=True)
    value = torch.empty(1, _H, 16, _HD, device="meta", requires_grad=True)
    block_mask = torch.empty(1, _H, 1, 1, device="meta", requires_grad=True)
    capture = OpDispatchCapture()
    with capture:
        sparse_out = sla2_sparse_attention(query.sin(), key.sin(), value.sin(), block_mask.sin(), 1)
        sparse_out.backward(torch.empty_like(sparse_out))
        linear_out = sla2_linear_attention(query.sin(), key.sin(), value.sin(), 1.0 - block_mask.sin(), 0)
        linear_out.backward(torch.empty_like(linear_out))

    nodes = {node.annotations["raw_op_type"]: node for node in capture.build_nodes().values()}
    sparse = nodes[_SPARSE_FWD]
    sparse_backward = nodes[_SPARSE_BWD]
    assert [(item["role"], item["axes"]) for item in sparse.attrs["tensor_semantics"]["inputs"]] == [
        ("query", ("batch", "head", "sequence", "head_dim")),
        ("key", ("batch", "head", "sequence", "head_dim")),
        ("value", ("batch", "head", "sequence", "head_dim")),
        ("block_mask", ("batch", "head", "q_blocks", "kv_blocks")),
    ]
    assert [item["role"] for item in sparse_backward.attrs["tensor_semantics"]["inputs"]] == [
        "query",
        "key",
        "value",
        "block_mask",
        "output_gradient",
    ]
    assert [item["role"] for item in sparse_backward.attrs["tensor_semantics"]["outputs"]] == [
        "query_gradient",
        "key_gradient",
        "value_gradient",
        "block_mask_gradient",
    ]
    linear = nodes[_LINEAR_FWD]
    assert [item["role"] for item in linear.attrs["tensor_semantics"]["inputs"]] == [
        "query",
        "key",
        "value",
        "complement_block_mask",
    ]


def test_multi_head_moe_virtual_fusion_saves_actual_inputs_and_preserves_backward_identity():
    shapes = [(1, _H, 16, _HD), (1, _H, 16, _HD), (1, _H, 16, _HD), (1, _H, 1, 1)]
    leaves = [torch.empty(shape, device="meta", requires_grad=True) for shape in shapes]
    capture = OpDispatchCapture()
    with capture, AutogradSavedTensorCapture():
        inputs = [leaf.sin() for leaf in leaves]
        output = sla2_sparse_attention(*inputs, 1)
        dy = torch.empty_like(output).sin()
        output.backward(dy)
    saved = [value for value in inputs if isinstance(value, torch.Tensor)]
    for leaf, value in zip(leaves, inputs, strict=True):
        assert leaf.grad is not None and leaf.grad.shape == value.shape
        assert leaf.grad.dtype == value.dtype
    forward = next(
        event for event in capture.memory_events() if event.raw_op_type == str(sla2_sparse_attention._opoverload)
    )
    backward = next(
        event
        for event in capture.memory_events()
        if event.raw_op_type == "multi_head_moe_sim.sla2_sparse_attention_grad.default"
    )
    expected_ids = [capture.tensor_id(value) for value in saved]
    assert [t.tensor_id for t in forward.inputs if t.tensor_id in expected_ids] == expected_ids
    backward_inputs = [t.alias_of if t.alias_of is not None else t.tensor_id for t in backward.inputs]
    for value_id in expected_ids:
        assert value_id in backward_inputs
    assert backward.inputs[-1].tensor_id == capture.tensor_id(dy)
    assert [tuple(t.shape) for t in backward.outputs] == [tuple(t.shape) for t in saved]
    assert [t.dtype for t in backward.outputs] == [str(t.dtype).removeprefix("torch.") for t in saved]


def _sla2_hardware():
    """Real DV120 profile: 32-bit cube/vec entries match the declared FP32 vec config."""
    from torchtitan_simulator.cost.hardware import dv120_hardware

    return dv120_hardware()


def test_multi_head_moe_cost_model_routes_sla2_fusions_through_the_library():
    plain = simulate(SimulationConfig(config="multi_head_moe_debugmodel", seq_len=16))
    annotate_workload_costs(plain)
    nodes = [n for template in plain.step_templates.values() for n in template.nodes.values()]
    sla2 = [n for n in nodes if n.annotations["raw_op_type"].startswith("multi_head_moe_sim.sla2_")]
    # The library roofline requires chip capabilities: without a hardware
    # profile the fusions stay unknown rather than partially annotated.
    assert len(sla2) == 32
    assert all(node.annotations.get("cost_unknown") for node in sla2)

    workload = simulate(SimulationConfig(config="multi_head_moe_debugmodel", seq_len=16))
    hardware = _sla2_hardware()
    annotate_workload_costs(workload, hardware=hardware)
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    flops = {}
    for node in nodes:
        raw = node.annotations["raw_op_type"]
        if raw not in (_SPARSE_FWD, _SPARSE_BWD, _LINEAR_FWD, _LINEAR_BWD):
            continue
        flops.setdefault(raw, node.annotations.get("flops"))
        # Library roofline: declared coarse duration, cube+vec work, traffic.
        assert node.annotations.get("duration_us", 0) > 0
        assert node.annotations.get("memory_traffic_bytes", 0) > 0
        # Envelope invariant: the declared composition
        # max(cube + vec, memory) + overhead can never dip below the pure
        # compute floor at cube peak or the pure HBM transfer at full bandwidth.
        work = node.annotations["flops"]
        floor_us = max(
            work / hardware.chip.cube.peak_flops_per_second(32) * 1e6,
            node.annotations["memory_traffic_bytes"]
            / hardware.chip.hbm.peak_bytes_per_second
            * 1e6,
        )
        assert node.annotations["duration_us"] >= floor_us - 1e-9
    # Capture pads tokens to the 64-token block multiple: seq_len 16 -> T=64,
    # one query/key block per head, selected=1, complement=0. Library families:
    # sparse forward cube 4*B*H*T*selected_keys*Dh plus 6 softmax units per
    # selected score; gradient 10 GEMM units plus 9 softmax units per score and
    # one per mask element; linear state GEMMs 2*B*H*T*Dh^2*(complement+1) plus
    # 2 normalize units per element; gradient doubles the GEMM population and
    # uses 5 normalize units plus one per mask element.
    assert flops == {
        _SPARSE_FWD: 4_292_608,
        _SPARSE_BWD: 10_633_220,
        _LINEAR_FWD: 2_129_920,
        _LINEAR_BWD: 4_276_228,
    }


def test_multi_head_moe_seq_len_128_exercises_real_block_selection():
    workload = simulate(SimulationConfig(config="multi_head_moe_debugmodel", seq_len=128))
    annotate_workload_costs(workload, hardware=_sla2_hardware())
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    flops = {}
    selected = {}
    for node in nodes:
        raw = node.annotations["raw_op_type"]
        if raw in (_SPARSE_FWD, _LINEAR_FWD):
            selected.setdefault(raw, node.attrs.get("selected_blocks"))
            flops.setdefault(raw, node.annotations.get("flops"))
    # Two 64-token key blocks per head; the router picks 1 of 2, so the sparse
    # branch keeps one block and the linear complement covers the other.
    assert selected == {_SPARSE_FWD: 1, _LINEAR_FWD: 1}
    # Sparse: 4*B*H*T*(1*64)*head_dim + 6 softmax units per selected score;
    # linear: 2*B*H*T*head_dim^2*(1+1) + 2 normalize units per element.
    assert flops == {_SPARSE_FWD: 8_585_216, _LINEAR_FWD: 8_454_144}
