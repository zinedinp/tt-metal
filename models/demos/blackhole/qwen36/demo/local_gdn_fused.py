# local-ai (Phase H3): GDN decode recurrence on ONE P150 -- today's single-device composite
# (recurrent_gated_delta_rule_decode_inplace_ttnn, BF16 state) against the fused C++ device op
# ttnn.transformer.fused_recurrent_gated_delta_rule (FP32 state, one Tensix core per head), which so far only
# the 4-card TP path uses. Shapes of Qwen3.8-27B decode: 48 value heads, K = V = 128.
#
# 1) device time per step (trace capture + replay, no host in the loop)
# 2) accuracy over GDN_STEPS steps against an FP32 torch reference of the same recurrence
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    fused_recurrent_gated_delta_rule_ttnn,
    recurrent_gated_delta_rule_decode_inplace_ttnn,
)

H, K, V = 48, 128, 128
STEPS = int(os.environ.get("GDN_STEPS", "64"))
REPS = int(os.environ.get("GDN_REPS", "50"))
SCALE = K**-0.5


def _inputs(gen):
    q = torch.randn(1, 1, H, K, generator=gen)
    k = torch.randn(1, 1, H, K, generator=gen)
    v = torch.randn(1, 1, H, V, generator=gen)
    beta = torch.sigmoid(torch.randn(1, 1, H, generator=gen))
    g = -torch.nn.functional.softplus(torch.randn(1, 1, H, generator=gen)) * 0.5
    return q, k, v, beta, g


def _ref_step(q, k, v, beta, g, S):
    q, k, v, beta, g = (t.reshape(H, -1).double() if t.dim() == 4 else t.reshape(H).double() for t in (q, k, v, beta, g))
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6) * SCALE
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    S = S * torch.exp(g)[:, None, None]
    u = beta[:, None] * (v - torch.einsum("hk,hkv->hv", k, S))
    S = S + k[:, :, None] * u[:, None, :]
    return torch.einsum("hk,hkv->hv", q, S), S


def _dev(t, dev, dtype=ttnn.bfloat16, mc=ttnn.L1_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


def _old(dev, state):
    def step(q, k, v, beta, g):
        o, _ = recurrent_gated_delta_rule_decode_inplace_ttnn(q, k, v, beta, g, state_buffer=state, device=dev)
        return o

    return step


def _new(dev, state):
    def step(q, k, v, beta, g):
        o, s = fused_recurrent_gated_delta_rule_ttnn(q, k, v, beta, g, initial_state=state, device=dev, high_precision=True)
        ttnn.copy(s, state)
        ttnn.deallocate(s)
        return o

    return step


def _time(dev, step, ins):
    step(*ins)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    o = step(*ins)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(REPS):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t0) * 1e3 / REPS
    ttnn.release_trace(dev, tid)
    ttnn.deallocate(o)
    return ms


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 50_000_000}], indirect=True)
def test_gdn_fused(mesh_device):
    dev = mesh_device
    dev.enable_program_cache()
    gen = torch.Generator().manual_seed(0)
    zeros = torch.zeros(1, H, K, V)
    variants = {
        "composite bf16 state": (_old, ttnn.bfloat16),
        "fused fp32 state": (_new, ttnn.float32),
    }
    seqs = [_inputs(gen) for _ in range(STEPS)]
    S_ref = torch.zeros(H, K, V, dtype=torch.float64)
    o_ref = []
    for ins in seqs:
        o, S_ref = _ref_step(*ins, S_ref)
        o_ref.append(o)

    for name, (make, sdt) in variants.items():
        state = _dev(zeros, dev, sdt, ttnn.DRAM_MEMORY_CONFIG)
        step = make(dev, state)
        # accuracy: STEPS eager steps from a zero state
        worst = 0.0
        for i, ins in enumerate(seqs):
            q, k, v, beta, g = (_dev(t, dev) for t in ins)
            o = step(q, k, v, beta, g)
            got = ttnn.to_torch(o).double().reshape(H, V)
            ref = o_ref[i]
            rel = ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item()
            worst = max(worst, rel)
            for t in (q, k, v, beta, g, o):
                ttnn.deallocate(t)
        S_got = ttnn.to_torch(state).double().reshape(H, K, V)
        s_rel = ((S_got - S_ref).norm() / S_ref.norm()).item()
        # device time per step (state keeps evolving; values do not matter for timing)
        ins = tuple(_dev(t, dev) for t in seqs[0])
        ms = _time(dev, step, ins)
        logger.info(
            f"[GDN] {name:<22s} {ms * 1e3:8.1f} us/step   o rel.err max over {STEPS} steps {worst:.2e}   "
            f"state rel.err {s_rel:.2e}"
        )
        for t in ins:
            ttnn.deallocate(t)
        ttnn.deallocate(state)
