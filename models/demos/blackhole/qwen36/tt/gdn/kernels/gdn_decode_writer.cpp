// local-ai H3: fused GDN decode, writer. o (row 0 of Vt fp32 tiles) goes straight into row hv of the
// [1,1,HV,V] output (the layout the gated RMSNorm reads), as two 64-byte face-row segments per tile;
// the new state overwrites head hv's tiles of the persistent state in place.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 6, cb_state = 7;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr auto o_a = TensorAccessorArgs<2>();
    constexpr auto s_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();

    const uint32_t hv = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_arg_val<uint32_t>(2);

    const uint32_t tb = get_tile_size(cb_out);
    const auto o_acc = TensorAccessor(o_a, o_addr, tb);
    const auto s_acc = TensorAccessor(s_a, s_addr, tb);
    constexpr uint32_t kv = Kt * Vt;

    Noc noc;
    CircularBuffer cbout(cb_out), cbst(cb_state);

    // Row r of an fp32 tile: face0/face1 for r < 16, face2/face3 below; 16 floats (64 B) per face row.
    const uint32_t r = hv % 32;
    const uint32_t face_a = (r < 16 ? 0u : 2048u) + (r % 16) * 64u;
    const uint32_t face_b = face_a + 1024u;
    const uint32_t tile_row = hv / 32;

    cbout.wait_front(Vt);
    auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cbout);
    for (uint32_t j = 0; j < Vt; j++) {
        const uint32_t page = tile_row * Vt + j;
        noc.async_write(src, o_acc, 64, {.offset_bytes = j * tb}, {.page_id = page, .offset_bytes = face_a});
        noc.async_write(src, o_acc, 64, {.offset_bytes = j * tb + 1024}, {.page_id = page, .offset_bytes = face_b});
    }
    cbst.wait_front(kv);
    auto ssrc = use<CircularBuffer::AddrSelector::READ_PTR>(cbst);
    for (uint32_t t = 0; t < kv; t++) {
        noc.async_write(ssrc, s_acc, tb, {.offset_bytes = t * tb}, {.page_id = hv * kv + t});
    }
    noc.async_write_barrier();
    cbout.pop_front(Vt);
    cbst.pop_front(kv);
}
