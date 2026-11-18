# local-ai (Phase H3): the fused GDN decode kernel (tt/gdn/fused_decode.py) against an FP64 torch reference
# over GDN_STEPS steps in the real decode layouts, and its device time (trace replay).
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE
from models.demos.blackhole.qwen36.tt.gdn.fused_decode import gdn_decode_fused

HK, HV, K, V = 16, 48, 128, 128
D = 2 * HK * K + HV * V
STEPS = int(os.environ.get("GDN_STEPS", "32"))
REPS = int(os.environ.get("GDN_REPS", "50"))


def _ref(qkv, decay, beta, S):
    q = qkv[: HK * K].view(HK, K).repeat_interleave(HV // HK, 0)
    k = qkv[HK * K : 2 * HK * K].view(HK, K).repeat_interleave(HV // HK, 0)
    v = qkv[2 * HK * K :].view(HV, V)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * K**-0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    S = S * decay[:, None, None]
    u = beta[:, None] * (v - torch.einsum("hk,hkv->hv", k, S))
    S = S + k[:, :, None] * u[:, None, :]
    return torch.einsum("hk,hkv->hv", q, S), S


def _dev(t, dev, mc=ttnn.L1_MEMORY_CONFIG):
    return ttnn.from_torch(t.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 50_000_000}], indirect=True)
def test_gdn_kernel(mesh_device):
    dev = mesh_device
    dev.enable_program_cache()
    gen = torch.Generator().manual_seed(0)
    kw = dict(num_k_heads=HK, num_v_heads=HV, head_k_dim=K, head_v_dim=V)
    state = _dev(torch.zeros(1, HV, K, V), dev, ttnn.DRAM_MEMORY_CONFIG)
    o_heads = _dev(torch.zeros(1, 1, HV, V), dev)
    S_ref = torch.zeros(HV, K, V, dtype=torch.float64)
    worst = 0.0
    for i in range(STEPS):
        qkv = torch.randn(D, generator=gen, dtype=torch.float64)
        decay = torch.exp(-torch.nn.functional.softplus(torch.randn(HV, generator=gen, dtype=torch.float64)) * 0.5)
        beta = torch.sigmoid(torch.randn(HV, generator=gen, dtype=torch.float64))
        o_ref, S_ref = _ref(qkv, decay, beta, S_ref)
        x, d, b = _dev(qkv.view(1, 1, D), dev), _dev(decay.view(1, 1, HV), dev), _dev(beta.view(1, 1, HV), dev)
        gdn_decode_fused(x, d, b, state, o_heads, **kw)
        got = ttnn.to_torch(o_heads).double().view(HV, V)
        rel = ((got - o_ref).norm() / o_ref.norm()).item()
        if i < 3 or rel > 1e-3:
            logger.info(f"[GDNK] step {i}: o rel.err {rel:.2e}")
        worst = max(worst, rel)
        for t in (x, d, b):
            ttnn.deallocate(t)
    S_got = ttnn.to_torch(state).double().view(HV, K, V)
    s_rel = ((S_got - S_ref).norm() / S_ref.norm()).item()

    x, d, b = _dev(qkv.view(1, 1, D), dev), _dev(decay.view(1, 1, HV), dev), _dev(beta.view(1, 1, HV), dev)
    gdn_decode_fused(x, d, b, state, o_heads, **kw)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    gdn_decode_fused(x, d, b, state, o_heads, **kw)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(REPS):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    us = (time.perf_counter() - t0) * 1e6 / REPS
    ttnn.release_trace(dev, tid)
    logger.info(f"[GDNK] fused kernel {us:7.1f} us/step   o rel.err max {worst:.2e} over {STEPS} steps   state rel.err {s_rel:.2e}")
    # FPU matmuls run fp32 operands as TF32: ~5e-3 relative, no drift (composite BF16 path: 1.1e-2).
    assert worst < 2e-2 and s_rel < 2e-2
