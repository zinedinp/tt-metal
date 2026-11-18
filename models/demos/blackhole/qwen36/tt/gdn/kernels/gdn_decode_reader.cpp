// local-ai H3: fused GDN decode, reader (one core per value head hv).
//
// Reads straight from the decode activations, no relayout:
//   q, k, v: the post-conv/SiLU projection row [1,1,D] (fp32 TILE). Its single tile row holds every
//            channel in row 0, so head hk's q is tiles q0 + hk*Kt .. +Kt, k is k0 + hk*Kt, v is v0 + hv*Vt.
//            GVA: hk = hv / G (passed as a runtime arg).
//   decay, beta: rows [1,1,HV] (fp32 TILE); element hv is copied into [0,0] of a zeroed scalar tile.
//   S: the persistent state [1,HV,K,V] fp32 TILE, head hv's Kt*Vt tiles (updated in place by the writer).
// Rows 1..31 of the q/k/v tiles are padding; they only need to be finite (see the compute kernel).
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_decay = 3, cb_beta = 4, cb_S = 5, cb_scr = 19;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t q0 = get_compile_time_arg_val(2);
    constexpr uint32_t k0 = get_compile_time_arg_val(3);
    constexpr uint32_t v0 = get_compile_time_arg_val(4);
    constexpr auto x_a = TensorAccessorArgs<5>();
    constexpr auto d_a = TensorAccessorArgs<x_a.next_compile_time_args_offset()>();
    constexpr auto b_a = TensorAccessorArgs<d_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();

    const uint32_t hv = get_arg_val<uint32_t>(0);
    const uint32_t hk = get_arg_val<uint32_t>(1);
    const uint32_t x_addr = get_arg_val<uint32_t>(2);
    const uint32_t d_addr = get_arg_val<uint32_t>(3);
    const uint32_t b_addr = get_arg_val<uint32_t>(4);
    const uint32_t s_addr = get_arg_val<uint32_t>(5);

    const uint32_t tb = get_tile_size(cb_q);  // fp32 tile, 4096 B
    const auto x_acc = TensorAccessor(x_a, x_addr, tb);
    const auto d_acc = TensorAccessor(d_a, d_addr, tb);
    const auto b_acc = TensorAccessor(b_a, b_addr, tb);
    const auto s_acc = TensorAccessor(s_a, s_addr, tb);
    constexpr uint32_t kv = Kt * Vt;

    Noc noc;
    auto read_into = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t n) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = base + t}, {.offset_bytes = t * tb});
        }
        noc.async_read_barrier();
        cb.push_back(n);
    };

    // Element hv of a [1,1,HV] row -> [0,0] of a zeroed scalar tile (fp32 tile: faces of 16x16,
    // row 0 cols 0..15 in face 0, cols 16..31 in face 1 at element 256).
    auto scalar_into = [&](const auto& acc, uint32_t cb_id) {
        CircularBuffer scr(cb_scr);
        scr.reserve_back(1);
        noc.async_read(acc, scr, tb, {.page_id = hv / 32}, {.offset_bytes = 0});
        noc.async_read_barrier();
        const CoreLocalMem<volatile uint32_t> src(scr.get_write_ptr());
        const uint32_t c = hv % 32;
        const uint32_t val = src[c < 16 ? c : 256 + (c - 16)];
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        const CoreLocalMem<volatile uint32_t> dst(cb.get_write_ptr());
        for (uint32_t i = 1; i < tb / 4; i++) {
            dst[i] = 0;
        }
        dst[0] = val;
        cb.push_back(1);
    };

    read_into(s_acc, cb_S, hv * kv, kv);
    read_into(x_acc, cb_q, q0 + hk * Kt, Kt);
    read_into(x_acc, cb_k, k0 + hk * Kt, Kt);
    read_into(x_acc, cb_v, v0 + hv * Vt, Vt);
    scalar_into(d_acc, cb_decay);
    scalar_into(b_acc, cb_beta);
}
