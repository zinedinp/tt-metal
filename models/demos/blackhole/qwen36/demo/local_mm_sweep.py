# local-ai (Phase H1): decode matmul sweep on ONE P150 for the real Qwen3.8-27B shapes (M = 1 token).
#
# Variants per shape: ttnn auto (what the single-device path uses today), 1D mcast_in0 on N cores
# (interleaved weight, as the TP path), DRAM width-sharded (weight re-laid out). Each variant is captured
# as a trace and replayed REPS times -> pure device time; GB/s against the 512 GB/s GDDR6 peak.
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tt import tp_common as tpc

REPS = int(os.environ.get("SWEEP_REPS", "50"))
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.bfloat16: 2048}
SHAPES = [  # name, K, N, weight dtype
    ("mlp gate/up", 5120, 17408, ttnn.bfloat4_b),
    ("mlp down bf4", 17408, 5120, ttnn.bfloat4_b),
    ("mlp down bf8", 17408, 5120, ttnn.bfloat8_b),
    ("gdn qkvzab", 5120, 16384 + 96, ttnn.bfloat8_b),
    ("gdn out", 6144, 5120, ttnn.bfloat8_b),
    ("attn qg+k+v", 5120, 14336, ttnn.bfloat8_b),
    ("attn wo", 6144, 5120, ttnn.bfloat8_b),
]
CORES = [int(c) for c in os.environ.get("SWEEP_CORES", "22,33,44,55,66,77,88,110").split(",")]


def _time(dev, fn):
    out = fn()
    ttnn.deallocate(out)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(REPS):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    us = (time.perf_counter() - t0) * 1e6 / REPS
    ttnn.release_trace(dev, tid)
    ttnn.deallocate(out)
    return us


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 50_000_000}], indirect=True
)
def test_mm_sweep(device):
    dev = device
    dev.enable_program_cache()
    grid = dev.compute_with_storage_grid_size()
    logger.info(f"[SWEEP] worker grid {grid.x}x{grid.y}, dram banks {dev.dram_grid_size()}")
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    x_t = torch.randn(1, 1, 32, 5120 * 4, dtype=torch.bfloat16)
    for name, K, N, wdt in SHAPES:
        w = ttnn.from_torch(torch.randn(1, 1, K, N, dtype=torch.bfloat16) * 0.02, dtype=wdt,
                            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.from_torch(x_t[..., :K], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.L1_MEMORY_CONFIG)
        nbytes = (K // 32) * ((N + 31) // 32) * TILE_BYTES[wdt]
        res = []

        def rec(tag, us):
            gbs = nbytes / (us * 1e-6) / 1e9
            res.append((us, tag))
            logger.info(f"[SWEEP] {name:<14s} {tag:<16s} {us:8.1f} us  {gbs:6.1f} GB/s ({100 * gbs / 512:5.1f} %)")

        rec("auto", _time(dev, lambda: ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)))
        for c in CORES:
            if c > grid.x * grid.y:
                continue
            try:
                pc = tpc.create_matmul_1d_decode_progcfg(1, K, N, num_cores=c, grid_w=grid.x)
                rec(f"1d {c}c", _time(dev, lambda: ttnn.linear(
                    x, w, compute_kernel_config=ckc, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG)))
            except Exception as e:  # invalid block/grid combo
                logger.info(f"[SWEEP] {name:<14s} 1d {c}c failed: {str(e).splitlines()[0][:120]}")
        try:
            wmc = tpc.create_dram_sharded_mem_config(K, N)
            pad_n = wmc.shard_spec.shape[1] * tpc.DRAM_CORES
            wt = torch.zeros(1, 1, K, pad_n, dtype=torch.bfloat16)
            ws = ttnn.from_torch(wt, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=wmc)
            pcd = tpc.create_dram_sharded_matmul_program_config(1, K, pad_n)
            xs_cfg = tpc.create_activation_shard_config(K)
            xs = ttnn.to_memory_config(x, xs_cfg)
            rec("dram-sharded", _time(dev, lambda: ttnn.linear(
                xs, ws, compute_kernel_config=ckc, program_config=pcd,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)))
            ttnn.deallocate(ws)
            ttnn.deallocate(xs)
        except Exception as e:
            logger.info(f"[SWEEP] {name:<14s} dram-sharded failed: {str(e).splitlines()[0][:160]}")
        best = min(res)
        logger.info(f"[SWEEP] {name:<14s} BEST {best[1]} {best[0]:.1f} us (auto {res[0][0]:.1f} us, "
                    f"floor {nbytes / 512e9 * 1e6:.1f} us)")
        ttnn.deallocate(w)
        ttnn.deallocate(x)


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 50_000_000}], indirect=True
)
def test_norm_sweep(device):
    """Decode RMSNorm [1,1,32,5120]: today's single-core interleaved norm vs width-sharded multi-core."""
    dev = device
    dev.enable_program_cache()
    H = 5120
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    xt = torch.randn(1, 1, 32, H, dtype=torch.bfloat16)
    wt = torch.randn(1, 1, H // 32, 32, dtype=torch.bfloat16)
    w = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    for src in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=src)
        tag = "dram" if src == ttnn.DRAM_MEMORY_CONFIG else "l1"
        us = _time(dev, lambda: ttnn.rms_norm(x, epsilon=1e-6, weight=w, compute_kernel_config=ckc,
                                              memory_config=ttnn.L1_MEMORY_CONFIG))
        logger.info(f"[NORM] interleaved in {tag:<4s}            {us:7.1f} us")
        ref = ttnn.to_torch(ttnn.rms_norm(x, epsilon=1e-6, weight=w, compute_kernel_config=ckc)).float()
        for gx, gy in ((4, 4), (8, 4), (8, 5), (10, 8), (8, 8)):
            n = gx * gy
            if (H // 32) % n:
                continue
            bw = H // 32 // n
            shard = ttnn.create_sharded_memory_config(
                shape=(32, H // n), core_grid=ttnn.CoreGrid(x=gx, y=gy), strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(gx, gy), subblock_w=bw if bw <= 4 else max(d for d in (1, 2, 3, 4) if bw % d == 0),
                block_h=1, block_w=bw, inplace=False)

            def run():
                xs = ttnn.to_memory_config(x, shard)
                ys = ttnn.rms_norm(xs, epsilon=1e-6, weight=w, program_config=pc, memory_config=shard,
                                   compute_kernel_config=ckc)
                ttnn.deallocate(xs)
                y = ttnn.sharded_to_interleaved(ys, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(ys)
                return y

            try:
                us = _time(dev, run)
                got = ttnn.to_torch(run()).float()
                pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
                logger.info(f"[NORM] sharded {gx}x{gy}={n:3d} cores (+reshard) {us:7.1f} us  pcc {pcc:.6f}")
            except Exception as e:
                logger.info(f"[NORM] sharded {gx}x{gy} failed: {str(e).splitlines()[0][:140]}")
        ttnn.deallocate(x)


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 50_000_000}], indirect=True
)
def test_logits_readback(device):
    """Per-token logits readback: what the demo does today vs on-device shrink first."""
    dev = device
    V = 248320
    lt = torch.randn(1, 1, 32, V, dtype=torch.bfloat16)
    lg = ttnn.from_torch(lt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ref = int(lt[0, 0, 0].float().argmax())

    def host(fn, n=10):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            r = fn()
        return (time.perf_counter() - t0) * 1e3 / n, r

    ms, r = host(lambda: int(ttnn.to_torch(lg).float().reshape(-1, V)[0].argmax()))
    logger.info(f"[READBACK] today: to_torch full tile + float + argmax   {ms:7.2f} ms  ok={r == ref}")

    def rm_slice():
        x = ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y = ttnn.slice(x, [0, 0, 0, 0], [1, 1, 1, V])
        ttnn.deallocate(x)
        return y

    us = _time(dev, rm_slice)
    ms, r = host(lambda: int(ttnn.to_torch(rm_slice()).float().reshape(-1)[:V].argmax()))
    logger.info(f"[READBACK] device to_layout RM + slice row 0: dev {us:7.1f} us, host total {ms:7.2f} ms ok={r == ref}")

    def sl_tile():
        return ttnn.slice(lg, [0, 0, 0, 0], [1, 1, 1, V])

    us = _time(dev, sl_tile)
    ms, r = host(lambda: int(ttnn.to_torch(sl_tile()).float().reshape(-1)[:V].argmax()))
    logger.info(f"[READBACK] device slice row 0 (tile):        dev {us:7.1f} us, host total {ms:7.2f} ms ok={r == ref}")

    def amax():
        y = rm_slice()
        a = ttnn.argmax(y, dim=-1)
        ttnn.deallocate(y)
        return a

    try:
        us = _time(dev, amax)
        ms, r = host(lambda: int(ttnn.to_torch(amax()).reshape(-1)[0]))
        logger.info(f"[READBACK] device argmax (RM row):           dev {us:7.1f} us, host total {ms:7.2f} ms ok={r == ref}")
    except Exception as e:
        logger.info(f"[READBACK] device argmax failed: {str(e).splitlines()[0][:160]}")
