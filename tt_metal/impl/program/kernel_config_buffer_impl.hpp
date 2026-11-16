// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/kernel_config_buffer.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-umd/types/cluster_descriptor_types.hpp>  // ChipId

namespace tt::tt_metal {

class Buffer;
class IDevice;
class Program;

namespace experimental {

class KernelConfigBufferImpl {
public:
    // One per-core allocation: a `size`-byte shard on each of `cores`, each at its own address. Backed by a
    // MeshBuffer, or by per-device Buffers for the fast-dispatch kernels, which are launched per device rather than
    // as a MeshWorkload.
    struct Part {
        CoreRangeSet cores;
        uint32_t size = 0;
        std::shared_ptr<distributed::MeshBuffer> mesh_buffer;
        std::vector<std::shared_ptr<Buffer>> device_buffers;
    };

    // Parts on disjoint cores, all MeshBuffer-backed on one MeshDevice or all per-device. `lockstep_address`: every
    // shard is at one address, checked when allocated.
    KernelConfigBufferImpl(std::vector<Part> parts, bool lockstep_address);

    const CoreRangeSet& cores() const { return cores_; }
    // The largest shard, and the shard on `core`.
    uint32_t size() const { return size_; }
    uint32_t size(const CoreCoord& core) const;
    // The MeshDevice a MeshBuffer-backed buffer is on; null for per-device buffers.
    const distributed::MeshDevice* mesh_device() const;

    DeviceAddr address(const distributed::MeshCoordinate& device_coord, const CoreCoord& core) const;
    // Same, keyed by the physical device dispatch writes to.
    DeviceAddr address(ChipId device_id, const CoreCoord& core) const;
    bool is_allocated() const;
    void deallocate();
    // Whether the buffer has a shard on `device_id` that this host can address.
    bool has_shard_on(ChipId device_id) const;
    // Whether every shard is at one address. Only such a buffer runs a program with Tensix program semaphores: a
    // kernel signals a peer's semaphore at its own semaphore's address.
    bool lockstep_address() const { return lockstep_address_; }

private:
    struct LocalPart {
        Part part;
        // Local devices only: a non-local device has no shard address on this host.
        std::unordered_map<ChipId, const Buffer*> device_buffers;
    };
    const LocalPart& part_on(const CoreCoord& core) const;

    std::vector<LocalPart> parts_;
    CoreRangeSet cores_;
    uint32_t size_ = 0;
    bool lockstep_address_ = false;
};

// Allocates `size` bytes (rounded up to L1 alignment) on each of `cores`, at a per-core address: a one-part buffer.
// With `lockstep_address`, every core's shard is at one address (see CreateKernelConfigBuffer).
KernelConfigBuffer AllocateKernelConfigBuffer(
    distributed::MeshDevice& device, const CoreRangeSet& cores, uint32_t size, bool lockstep_address = false);

// CreateKernelConfigBuffer for programs launched per device rather than as a MeshWorkload, such as the fast-dispatch
// command queue programs: one program per device, already laid out. No slow-dispatch check, since fast dispatch may be
// active. Each device gets its own per-core buffers, for the same cores at the same sizes.
std::shared_ptr<KernelConfigBufferImpl> CreateKernelConfigBuffer(
    const std::vector<std::pair<IDevice*, Program*>>& programs, bool lockstep_address);

// The largest Tensix kernel-config block among `program`'s kernel groups. The program must be laid out.
uint32_t LargestTensixKernelConfigBlock(Program& program);

// Checks that `program`, laid out, can run from `buffer` on `device_id`: the buffer has a shard there on every Tensix
// core of the program, each kernel group's block fits in the shard on each of its cores, and the program has no Tensix
// program semaphores unless the buffer has a lockstep address.
void ValidateKernelConfigBuffer(Program& program, const KernelConfigBufferImpl& buffer, ChipId device_id);

}  // namespace experimental
}  // namespace tt::tt_metal
