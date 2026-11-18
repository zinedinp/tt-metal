# local-ai (Phase H3): fused GDN decode step on one P150 as ONE device program (ttnn.generic_op).
#
# Replaces, per GDN layer and token, the q/k/v slices + to_layout + reshape + repeat_interleave, both
# L2 norms, the query scale and the whole recurrent composite (decay, k.S, delta, rank-1 update, q.S,
# state copy-back) with one launch on 48 Tensix cores (one per value head). Inputs are read straight
# from the decode activations; the state lives in FP32 and is updated in place; o is written in the
# [1,1,HV,V] layout the gated RMSNorm reads. Kernels: gdn/kernels/gdn_decode_{reader,compute,writer}.cpp.
import os
import struct

import ttnn

_KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_SRC = {}
_TILE_F32 = 4096


def _src(name):
    if name not in _SRC:
        with open(os.path.join(_KDIR, name)) as f:
            _SRC[name] = f.read()
    return _SRC[name]


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


_CBS = (  # (index, tiles): see gdn_decode_compute.cpp; Kt = Vt = 4 for Qwen3.8-27B
    ("q", 0, "Kt"), ("k", 1, "Kt"), ("v", 2, "Vt"), ("decay", 3, 1), ("beta", 4, 1), ("S", 5, "kv"),
    ("out", 6, "Vt"), ("state", 7, "kv"), ("sd", 10, "kv"), ("vread", 11, "Vt"),
    ("u", 12, "Vt"), ("kcol", 13, "Kt"), ("ss", 14, 1), ("r", 15, 1), ("qn", 16, "Kt"), ("kn", 17, "Kt"),
    ("scr", 19, 1),
)


def gdn_decode_fused(qkv, decay, beta, state, o_heads, *, num_k_heads, num_v_heads, head_k_dim, head_v_dim, eps=1e-6):
    """One decode step of the gated delta rule for every value head.

    qkv     [1,1,Dq+Dk+Dv] fp32 TILE  post-conv/SiLU projection (q | k | v)
    decay   [1,1,HV]       fp32 TILE  exp(g)
    beta    [1,1,HV]       fp32 TILE  sigmoid(b)
    state   [1,HV,K,V]     fp32 TILE  updated in place
    o_heads [1,1,HV,V]     fp32 TILE  output (written in place, returned)
    """
    dev = qkv.device()
    grid = dev.compute_with_storage_grid_size()
    Kt, Vt = head_k_dim // 32, head_v_dim // 32
    G = num_v_heads // num_k_heads
    q0, k0, v0 = 0, num_k_heads * Kt, 2 * num_k_heads * Kt
    n = {"Kt": Kt, "Vt": Vt, "kv": Kt * Vt}
    coords = [ttnn.CoreCoord(h // grid.y, h % grid.y) for h in range(num_v_heads)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])

    cbs = [
        ttnn.CBDescriptor(
            total_size=(n[t] if isinstance(t, str) else t) * _TILE_F32,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=_TILE_F32)],
        )
        for _, i, t in _CBS
    ]

    ta = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rd_rt, wr_rt, cp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for h, c in enumerate(coords):
        rd_rt[c.x][c.y] = [h, h // G, qkv.buffer_address(), decay.buffer_address(), beta.buffer_address(),
                           state.buffer_address()]
        wr_rt[c.x][c.y] = [h, o_heads.buffer_address(), state.buffer_address()]
        cp_rt[c.x][c.y] = []

    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=_src("gdn_decode_reader.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=[Kt, Vt, q0, k0, v0] + ta(qkv) + ta(decay) + ta(beta) + ta(state),
                runtime_args=rd_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_src("gdn_decode_writer.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=[Kt, Vt] + ta(o_heads) + ta(state),
                runtime_args=wr_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_src("gdn_decode_compute.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=[Kt, Vt, _f32_bits(eps), _f32_bits(head_k_dim**-0.5)],
                runtime_args=cp_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                ),
            ),
        ],
        semaphores=[],
        cbs=cbs,
    )
    ttnn.generic_op([qkv, decay, beta, state, o_heads], program)
    return o_heads
