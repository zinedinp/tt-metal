# local-ai (Phase H1): single-P150 decode tuning, measured with demo/local_mm_sweep.py (PROTOKOLL #104ff).
#
# * linear(): M = 1 tile decode matmuls on an interleaved weight get the measured-best 1D mcast_in0
#   program config instead of ttnn's auto pick (e.g. mlp down BFP4 350 -> 180 us, o_proj 133 -> 99 us).
#   Everything else (prefill, sharded inputs, unknown shapes) goes to ttnn.linear unchanged.
# * rms_norm_decode(): the decode RMSNorm on a [.., 32, H] tile row runs on ONE core when interleaved
#   (85 us). Width-sharding it over 16 cores, norm, back to interleaved costs 17 us (pcc 0.99998).
#
# Switches: QWEN36_MM_TUNE=0 / QWEN36_NORM_SHARD=0 restore the stock calls.
import math
import os

import ttnn

_MM = os.environ.get("QWEN36_MM_TUNE", "1") == "1"
_NORM = os.environ.get("QWEN36_NORM_SHARD", "1") == "1"

# (K, N, weight dtype or None for any) -> cores of the 1D decode grid (11 wide on P150)
_TABLE = {
    (17408, 5120, ttnn.bfloat4_b): 66,
    (17408, 5120, ttnn.bfloat8_b): 33,
    (6144, 5120, None): 33,
}
_pc_cache = {}
_norm_cache = {}


def _is_decode_row(x):
    try:
        shp = list(x.padded_shape)
    except Exception:
        return False
    lead = 1
    for d in shp[:-2]:
        lead *= int(d)
    return lead == 1 and int(shp[-2]) == ttnn.TILE_SIZE and not x.is_sharded()


def _decode_pc(x, w):
    K, N = int(w.padded_shape[-2]), int(w.padded_shape[-1])
    cores = _TABLE.get((K, N, w.dtype)) or _TABLE.get((K, N, None))
    if cores is None:
        return None
    key = (K, N, cores)
    pc = _pc_cache.get(key)
    if pc is None:
        from models.demos.blackhole.qwen36.tt.tp_common import create_matmul_1d_decode_progcfg

        grid_w = x.device().compute_with_storage_grid_size().x
        pc = create_matmul_1d_decode_progcfg(1, K, N, num_cores=cores, grid_w=grid_w)
        _pc_cache[key] = pc
    return pc


def linear(x, w, **kw):
    if _MM and kw.get("program_config") is None and _is_decode_row(x) and not w.is_sharded():
        pc = _decode_pc(x, w)
        if pc is not None:
            kw["program_config"] = pc
            if kw.get("memory_config") is None:
                kw["memory_config"] = ttnn.L1_MEMORY_CONFIG
    return ttnn.linear(x, w, **kw)


def _norm_cfg(H):
    cfg = _norm_cache.get(H)
    if cfg is None:
        tiles = H // ttnn.TILE_SIZE
        gx, gy = 4, 4
        while tiles % (gx * gy):
            gy -= 1
        n = gx * gy
        bw = tiles // n
        shard = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, H // n),
            core_grid=ttnn.CoreGrid(x=gx, y=gy),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            subblock_w=max(d for d in (1, 2, 3, 4) if bw % d == 0),
            block_h=1,
            block_w=bw,
            inplace=False,
        )
        cfg = _norm_cache[H] = (shard, pc)
    return cfg


def norm_shard_ok(x):
    return _NORM and _is_decode_row(x) and (int(x.padded_shape[-1]) // ttnn.TILE_SIZE) % 4 == 0


def rms_norm_decode(norm, x, out_mc=ttnn.L1_MEMORY_CONFIG):
    """models.common.rmsnorm.RMSNorm `norm` on a decode tile row, multi-core. Same math as norm(x)."""
    shard, pc = _norm_cfg(int(x.padded_shape[-1]))
    xs = ttnn.to_memory_config(x, shard)
    ys = ttnn.rms_norm(
        xs,
        epsilon=norm.eps,
        weight=norm.weight,
        program_config=pc,
        memory_config=shard,
        compute_kernel_config=norm.compute_kernel_config_hifi2,
    )
    ttnn.deallocate(xs)
    y = ttnn.sharded_to_interleaved(ys, out_mc)
    ttnn.deallocate(ys)
    return y
