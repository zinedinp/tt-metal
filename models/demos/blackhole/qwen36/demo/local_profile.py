# local-ai (Phase H0): decode device time per layer on ONE P150, without the Tracy profiler.
#
# The release wheel has no TRACY_ENABLE, so TT_METAL_DEVICE_PROFILER is unavailable. Instead every
# piece of the decode step (embedding, each of the 64 layers, final norm + LM head, and the whole
# step) is captured as its OWN trace and replayed REPS times back to back: replay time / REPS is the
# pure device time of that piece (no Python, no dispatch). Weight bytes per layer come from the device
# tensors the layer holds, so each line also shows the achieved DRAM bandwidth (P150: 512 GB/s peak).
#
#   PROFILE_POS="8192,131072,262144"   decode positions (attention layers are re-timed per position)
#   PROFILE_REPS=20
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, BLOCK_SIZE
from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

POS = [int(x) for x in os.environ.get("PROFILE_POS", "8192").split(",") if x.strip()]
REPS = int(os.environ.get("PROFILE_REPS", "20"))
PEAK_GBS = 512.0

_TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
    ttnn.float32: 4096,
    ttnn.uint32: 4096,
    ttnn.int32: 4096,
}


def _tensor_bytes(t):
    try:
        if t.storage_type() != ttnn.StorageType.DEVICE:
            return 0
        shp = list(t.padded_shape)
        vol = 1
        for d in shp:
            vol *= int(d)
        per = _TILE_BYTES.get(t.dtype)
        if per is None:
            return 0
        if t.layout == ttnn.TILE_LAYOUT:
            return vol // 1024 * per
        return vol * per // 1024
    except Exception:
        return 0


def _weight_bytes(obj, seen=None, depth=0):
    """Sum of device tensor bytes reachable from obj's attributes (weights, norms, conv taps)."""
    if seen is None:
        seen = set()
    if id(obj) in seen or depth > 6:
        return 0
    seen.add(id(obj))
    if isinstance(obj, ttnn.Tensor):
        return _tensor_bytes(obj)
    total = 0
    if isinstance(obj, dict):
        items = obj.values()
    elif isinstance(obj, (list, tuple)):
        items = obj
    elif hasattr(obj, "__dict__"):
        items = [v for k, v in vars(obj).items() if not k.startswith("_cap")]
    else:
        return 0
    for v in items:
        if isinstance(v, (int, float, str, bytes, torch.Tensor)) or v is None:
            continue
        if isinstance(v, ttnn.Tensor) or isinstance(v, (dict, list, tuple)) or (
            hasattr(v, "__dict__") and type(v).__module__.startswith(("models.", "__main__"))
        ):
            total += _weight_bytes(v, seen, depth + 1)
    return total


def _time_trace(dev, fn, reps=REPS):
    """Capture fn() as a trace, replay reps times, return ms per replay."""
    fn()  # compile / warm
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t0) * 1e3 / reps
    ttnn.release_trace(dev, tid)
    for o in out if isinstance(out, (list, tuple)) else [out]:
        if isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)
    return ms


def _line(name, ms, nbytes):
    gbs = nbytes / (ms * 1e-3) / 1e9 if ms > 0 else 0.0
    floor = nbytes / (PEAK_GBS * 1e9) * 1e3
    logger.info(
        f"[PROFILE] {name:<22s} {ms:8.3f} ms  weights {nbytes / 1e6:8.1f} MB  {gbs:6.1f} GB/s "
        f"({100 * gbs / PEAK_GBS:5.1f} % of peak, floor {floor:6.3f} ms)"
    )
    return gbs


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "num_command_queues": 2, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.timeout(7200)
def test_decode_profile(mesh_device):
    dev = mesh_device
    dev.enable_program_cache()
    max_pos = max(POS)
    num_blocks = (max_pos + 2 * BLOCK_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    model = Qwen36Model.from_pretrained(dev, max_batch_size=1, max_seq_len=num_blocks * BLOCK_SIZE, enable_mtp=False)
    kv_shape = [num_blocks, model.args.n_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

    def inputs(pos):
        d = model.prepare_inputs_decode(torch.tensor([[1234]], dtype=torch.long), pos, page_table=page_table)
        cos, sin = unpack_rope(d[2])
        return d[0], d[1], cos, sin, d[3]

    tok, cur, cos, sin, pt = inputs(POS[0])
    x0 = model.embd(tok)
    x0 = ttnn.to_memory_config(x0, ttnn.DRAM_MEMORY_CONFIG)
    # one full eager step compiles every program
    ttnn.deallocate(model._forward_decode(tok, cos, sin, cur, pt))
    ttnn.synchronize_device(dev)

    rows = []
    lm_bytes = _tensor_bytes(model.lm_head_weight) + _weight_bytes(model.norm)
    total_w = lm_bytes

    ms = _time_trace(dev, lambda: model.embd(tok))
    rows.append(("embedding", ms, 0))

    per_kind = {"gdn": [], "attn": {p: [] for p in POS}}
    for li, layer in enumerate(model.layers):
        wb = _weight_bytes(layer)
        total_w += wb
        if layer.is_full_attention:
            for p in POS:
                tok_p, cur_p, cos_p, sin_p, pt_p = inputs(p)
                ms = _time_trace(
                    dev,
                    lambda: layer.forward(
                        ttnn.clone(x0, memory_config=ttnn.DRAM_MEMORY_CONFIG), cos_p, sin_p,
                        position_tensor=cur_p, page_table=pt_p, mode="decode",
                    ),
                )
                per_kind["attn"][p].append((ms, wb))
                rows.append((f"L{li:02d} attn @{p // 1024}K", ms, wb))
        else:
            ms = _time_trace(
                dev, lambda: layer.forward(ttnn.clone(x0, memory_config=ttnn.DRAM_MEMORY_CONFIG), mode="decode")
            )
            per_kind["gdn"].append((ms, wb))
            rows.append((f"L{li:02d} gdn", ms, wb))

    # ---- split one GDN layer and one attention layer into norm / mixer / mlp ----
    from models.tt_transformers.tt.common import Mode

    l1 = {"output_mem_config": ttnn.L1_MEMORY_CONFIG}
    for li in (0, 3):
        layer = model.layers[li]
        tok_p, cur_p, cos_p, sin_p, pt_p = inputs(POS[0])
        xn = layer.attention_norm(ttnn.clone(x0, memory_config=ttnn.DRAM_MEMORY_CONFIG), mode=Mode.DECODE, norm_config=l1)
        xn = ttnn.to_memory_config(xn, ttnn.DRAM_MEMORY_CONFIG)
        ms_n = _time_trace(dev, lambda: layer.attention_norm(
            ttnn.clone(x0, memory_config=ttnn.DRAM_MEMORY_CONFIG), mode=Mode.DECODE, norm_config=l1))
        if layer.is_full_attention:
            mix = lambda: layer.attention.forward(
                ttnn.to_memory_config(xn, ttnn.L1_MEMORY_CONFIG), cos_p, sin_p, position_tensor=cur_p, page_table=pt_p)
        else:
            mix = lambda: layer.attention.forward(ttnn.to_memory_config(xn, ttnn.L1_MEMORY_CONFIG), mode="recurrent")
        ms_mix = _time_trace(dev, mix)
        ms_mlp = _time_trace(dev, lambda: layer.feed_forward.forward(
            ttnn.to_memory_config(xn, ttnn.L1_MEMORY_CONFIG), mode="decode"))
        wb_mix = _weight_bytes(layer.attention) if not layer.is_full_attention else 0
        rows.append((f"L{li:02d} norm", ms_n, 0))
        rows.append((f"L{li:02d} mixer", ms_mix, wb_mix))
        rows.append((f"L{li:02d} mlp", ms_mlp, _weight_bytes(layer.feed_forward)))
        ttnn.deallocate(xn)

    def head():
        y = model._final_norm_decode(ttnn.clone(x0, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        return model._lm_head(y)

    ms_head = _time_trace(dev, head)
    rows.append(("final norm + lm_head", ms_head, lm_bytes))

    full = {}
    for p in POS:
        tok_p, cur_p, cos_p, sin_p, pt_p = inputs(p)
        full[p] = _time_trace(dev, lambda: model._forward_decode(tok_p, cos_p, sin_p, cur_p, pt_p))

    logger.info("=" * 100)
    for name, ms, wb in rows:
        _line(name, ms, wb)
    logger.info("-" * 100)
    g_ms = sum(m for m, _ in per_kind["gdn"])
    g_b = sum(b for _, b in per_kind["gdn"])
    _line(f"sum gdn x{len(per_kind['gdn'])}", g_ms, g_b)
    for p in POS:
        a_ms = sum(m for m, _ in per_kind["attn"][p])
        a_b = sum(b for _, b in per_kind["attn"][p])
        _line(f"sum attn x{len(per_kind['attn'][p])} @{p // 1024}K", a_ms, a_b)
        parts = g_ms + a_ms + ms_head + rows[0][1]
        logger.info(
            f"[PROFILE] @{p // 1024}K: whole-step trace {full[p]:.2f} ms ({1e3 / full[p]:.2f} tok/s device-only), "
            f"sum of parts {parts:.2f} ms, weights {total_w / 1e9:.2f} GB -> floor {total_w / PEAK_GBS / 1e6:.2f} ms "
            f"({PEAK_GBS * 1e9 / total_w:.1f} tok/s), achieved {total_w / (full[p] * 1e-3) / 1e9:.0f} GB/s"
        )
    logger.info("=" * 100)
