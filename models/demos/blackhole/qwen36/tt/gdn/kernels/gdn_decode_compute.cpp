// local-ai H3: fused GDN decode, compute (one value head, one token).
//
// Derived from ttnn's fused_recurrent_gated_delta_rule compute kernel (Apache-2.0, Tenstorrent), with
// the L2 normalisation and query scale moved on-core so the inputs can be the raw projection rows:
//   rq = rsqrt(q.q + eps) * scale,  qn = q * rq          (q.q as a 1xKt @ Kt x1 matmul, B transposed)
//   rk = rsqrt(k.k + eps),          kn = k * rk
//   sd    = S * decay
//   vread = kn . sd
//   u     = beta * (v - vread)
//   S'    = sd + kn^T (x) u
//   o     = qn . S'
// Tiles carry the token in row 0. Padding rows only have to be finite: beta's tile is zero outside
// [0,0], so u's padding rows are exactly 0 and the rank-1 update ignores kn's padding columns.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_decay = 3, cb_beta = 4, cb_S = 5;
constexpr uint32_t cb_out = 6, cb_state = 7, cb_sd = 10;
constexpr uint32_t cb_vread = 11, cb_u = 12, cb_kcol = 13, cb_ss = 14, cb_r = 15, cb_qn = 16, cb_kn = 17;
constexpr uint32_t DST_TILES = 4;

void mm(uint32_t a, uint32_t b, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t Nt, bool tr) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(Mt * Nt);
    matmul_init(a, b, tr ? 1 : 0);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            for (uint32_t ki = 0; ki < Kt; ki++) {
                uint32_t bi = tr ? (ni * Kt + ki) : (ki * Nt + ni);
                matmul_tiles(a, b, mi * Kt + ki, bi, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_o.push_back(Mt * Nt);
}

// r[0,0] = rsqrt(x . x + eps) * mul
void inv_norm(uint32_t x, uint32_t Kt, uint32_t eps_bits, uint32_t mul_bits) {
    mm(x, x, cb_ss, 1, Kt, 1, true);
    CircularBuffer ss(cb_ss), r(cb_r);
    ss.wait_front(1);
    r.reserve_back(1);
    tile_regs_acquire();
    copy_init(cb_ss);
    copy_tile(cb_ss, 0, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, mul_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_r, 0);
    tile_regs_release();
    r.push_back(1);
    ss.pop_front(1);
}

void bcast_scalar_mul(uint32_t a, uint32_t scal, uint32_t o, uint32_t n) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    mul_bcast_scalar_init(a, scal);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(a, scal, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_o.push_back(n);
}

template <uint32_t Kt, uint32_t Vt>
void rank1_update(uint32_t sd, uint32_t kcol, uint32_t u, uint32_t o) {
    constexpr uint32_t kv = Kt * Vt;
    CircularBuffer cb_o(o), cb_st(cb_state);
    cb_o.reserve_back(kv);
    cb_st.reserve_back(kv);
    for (uint32_t mi = 0; mi < Kt; mi++) {
        for (uint32_t n0 = 0; n0 < Vt; n0 += DST_TILES) {
            const uint32_t nn = (Vt - n0 < DST_TILES) ? (Vt - n0) : DST_TILES;
            tile_regs_acquire();
            copy_init(sd);
            for (uint32_t j = 0; j < nn; j++) {
                copy_tile(sd, mi * Vt + n0 + j, j);
            }
            matmul_init(kcol, u, 0);
            for (uint32_t j = 0; j < nn; j++) {
                matmul_tiles(kcol, u, mi, n0 + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nn; j++) {
                pack_tile<true>(j, o, mi * Vt + n0 + j);
                pack_tile<true>(j, cb_state, mi * Vt + n0 + j);
            }
            tile_regs_release();
        }
    }
    cb_o.push_back(kv);
    cb_st.push_back(kv);
}

void delta_rule_residual(uint32_t a, uint32_t b, uint32_t beta, uint32_t o, uint32_t n) {
    constexpr uint32_t per_batch = DST_TILES - 1;
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    for (uint32_t i0 = 0; i0 < n; i0 += per_batch) {
        const uint32_t nn = (n - i0 < per_batch) ? (n - i0) : per_batch;
        tile_regs_acquire();
        sub_init(a, b);
        for (uint32_t j = 0; j < nn; j++) {
            sub_tiles(a, b, i0 + j, i0 + j, j);
        }
        copy_init(beta);
        copy_tile(beta, 0, nn);
        sfpu_mul_bcast_col_init();
        for (uint32_t j = 0; j < nn; j++) {
            sfpu_mul_bcast_col(j, nn);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nn; j++) {
            pack_tile<true>(j, o, i0 + j);
        }
        tile_regs_release();
    }
    cb_o.push_back(n);
}

void transpose_block(uint32_t in, uint32_t o, uint32_t n) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    transpose_init(in);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        transpose_tile(in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_o.push_back(n);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(2);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(3);
    constexpr uint32_t one_bits = 0x3F800000u;
    constexpr uint32_t kv = Kt * Vt;

    compute_kernel_hw_startup(cb_q, cb_v, cb_out);

    CircularBuffer q(cb_q), k(cb_k), v(cb_v), decay(cb_decay), beta(cb_beta), S(cb_S);
    CircularBuffer sd(cb_sd), vread(cb_vread), u(cb_u), kcol(cb_kcol), r(cb_r), qn(cb_qn), kn(cb_kn);

    // qn = q * rsqrt(q.q + eps) * scale
    q.wait_front(Kt);
    inv_norm(cb_q, Kt, eps_bits, scale_bits);
    r.wait_front(1);
    bcast_scalar_mul(cb_q, cb_r, cb_qn, Kt);
    r.pop_front(1);
    q.pop_front(Kt);

    // kn = k * rsqrt(k.k + eps)
    k.wait_front(Kt);
    inv_norm(cb_k, Kt, eps_bits, one_bits);
    r.wait_front(1);
    bcast_scalar_mul(cb_k, cb_r, cb_kn, Kt);
    r.pop_front(1);
    k.pop_front(Kt);
    kn.wait_front(Kt);

    // sd = S * decay
    decay.wait_front(1);
    S.wait_front(kv);
    bcast_scalar_mul(cb_S, cb_decay, cb_sd, kv);
    decay.pop_front(1);
    S.pop_front(kv);
    sd.wait_front(kv);

    // vread = kn . sd
    mm(cb_kn, cb_sd, cb_vread, 1, Kt, Vt, false);
    vread.wait_front(Vt);

    // u = beta * (v - vread)
    v.wait_front(Vt);
    beta.wait_front(1);
    delta_rule_residual(cb_v, cb_vread, cb_beta, cb_u, Vt);
    v.pop_front(Vt);
    vread.pop_front(Vt);
    beta.pop_front(1);
    u.wait_front(Vt);

    // kcol = kn^T, S' = sd + kcol (x) u  (also emitted as the state output). S' goes back into cb_S,
    // which the reader filled once and the decay step has popped: one 64 KiB buffer less, so the
    // program fits next to larger L1 activations.
    transpose_block(cb_kn, cb_kcol, Kt);
    kn.pop_front(Kt);
    kcol.wait_front(Kt);
    rank1_update<Kt, Vt>(cb_sd, cb_kcol, cb_u, cb_S);
    kcol.pop_front(Kt);
    u.pop_front(Vt);
    sd.pop_front(kv);
    S.wait_front(kv);

    // o = qn . S'
    qn.wait_front(Kt);
    mm(cb_qn, cb_S, cb_out, 1, Kt, Vt, false);
    qn.pop_front(Kt);
    S.pop_front(kv);
}
