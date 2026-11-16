// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kernel_config_buffer_impl.hpp"

#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <optional>
#include <utility>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/program_preparation.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "distributed/sd_mesh_command_queue.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/buffers/buffer_impl.hpp"
#include "impl/buffers/semaphore.hpp"
#include "impl/context/metal_context.hpp"
#include "program_impl.hpp"
#include "slow_dispatch.hpp"

namespace tt::tt_metal::experimental {

namespace {

uint32_t tensix_index(const Program& program) {
    return MetalContext::instance(program.impl().get_context_id())
        .hal()
        .get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
}

// Tensix cores the program runs kernels on.
CoreRangeSet program_tensix_cores(Program& program) {
    CoreRangeSet cores;
    for (const auto& kernel_group : program.impl().get_kernel_groups(tensix_index(program))) {
        cores = cores.merge(kernel_group->core_ranges);
    }
    return cores;
}

void check_no_tensix_semaphores(Program& program) {
    for (const Semaphore& semaphore : program.impl().semaphores()) {
        TT_FATAL(
            semaphore.core_type() != CoreType::WORKER,
            "A program run from a kernel config buffer cannot have Tensix program semaphores (semaphore {} on {}): "
            "each core's block, and so each semaphore, sits at a different address, but a kernel signals a peer's "
            "semaphore at its own address. Use a global semaphore instead.",
            semaphore.id(),
            semaphore.core_range_set());
    }
}

void check_slow_dispatch(distributed::MeshDevice& device) {
    TT_FATAL(
        !MetalContext::instance(device.impl().get_context_id()).rtoptions().get_fast_dispatch(),
        "A kernel config buffer needs slow dispatch: fast dispatch places kernel-config blocks in its own ring "
        "buffer");
}

}  // namespace

KernelConfigBuffer::KernelConfigBuffer(std::shared_ptr<KernelConfigBufferImpl> impl) : impl_(std::move(impl)) {}

const CoreRangeSet& KernelConfigBuffer::cores() const { return impl_->cores(); }

uint32_t KernelConfigBuffer::size() const { return impl_->size(); }

uint32_t KernelConfigBuffer::size(const CoreCoord& core) const { return impl_->size(core); }

DeviceAddr KernelConfigBuffer::address(const distributed::MeshCoordinate& device_coord, const CoreCoord& core) const {
    return impl_->address(device_coord, core);
}

void KernelConfigBuffer::deallocate() { impl_->deallocate(); }

KernelConfigBufferImpl::KernelConfigBufferImpl(std::vector<Part> parts, bool lockstep_address) :
    lockstep_address_(lockstep_address) {
    TT_FATAL(!parts.empty(), "A kernel config buffer needs at least one part");
    for (Part& part : parts) {
        TT_FATAL(!cores_.intersects(part.cores), "Kernel config buffer parts overlap on {}", part.cores);
        cores_ = cores_.merge(part.cores);
        size_ = std::max(size_, part.size);
        LocalPart local{.part = std::move(part)};
        if (local.part.mesh_buffer != nullptr) {
            distributed::MeshDevice& mesh_device = *local.part.mesh_buffer->device();
            for (const distributed::MeshCoordinate& coord : distributed::MeshCoordinateRange(mesh_device.shape())) {
                if (mesh_device.is_local(coord)) {
                    const Buffer* device_buffer = local.part.mesh_buffer->get_device_buffer(coord);
                    local.device_buffers.emplace(device_buffer->device()->id(), device_buffer);
                }
            }
        }
        for (const auto& buffer : local.part.device_buffers) {
            const auto [_, inserted] = local.device_buffers.emplace(buffer->device()->id(), buffer.get());
            TT_FATAL(inserted, "Kernel config buffer part has two buffers on device {}", buffer->device()->id());
        }
        parts_.push_back(std::move(local));
    }
    const bool on_mesh = parts_.front().part.mesh_buffer != nullptr;
    for (const LocalPart& local : parts_) {
        TT_FATAL(
            (local.part.mesh_buffer != nullptr) == on_mesh &&
                (!on_mesh || local.part.mesh_buffer->device() == parts_.front().part.mesh_buffer->device()),
            "Kernel config buffer parts must all be on one MeshDevice, or all per-device");
    }
}

const KernelConfigBufferImpl::LocalPart& KernelConfigBufferImpl::part_on(const CoreCoord& core) const {
    for (const LocalPart& local : parts_) {
        if (local.part.cores.contains(core)) {
            return local;
        }
    }
    TT_THROW("Kernel config buffer has no shard on core {}", core);
}

uint32_t KernelConfigBufferImpl::size(const CoreCoord& core) const { return part_on(core).part.size; }

const distributed::MeshDevice* KernelConfigBufferImpl::mesh_device() const {
    const auto& mesh_buffer = parts_.front().part.mesh_buffer;
    return mesh_buffer != nullptr ? mesh_buffer->device() : nullptr;
}

bool KernelConfigBufferImpl::is_allocated() const {
    return std::all_of(parts_.begin(), parts_.end(), [](const LocalPart& local) {
        if (local.part.mesh_buffer != nullptr) {
            return local.part.mesh_buffer->is_allocated();
        }
        return std::all_of(local.part.device_buffers.begin(), local.part.device_buffers.end(), [](const auto& buffer) {
            return buffer->is_allocated();
        });
    });
}

void KernelConfigBufferImpl::deallocate() {
    for (LocalPart& local : parts_) {
        if (local.part.mesh_buffer != nullptr) {
            local.part.mesh_buffer->deallocate();
        }
        for (const auto& buffer : local.part.device_buffers) {
            buffer->impl().deallocate(*buffer);
        }
    }
}

bool KernelConfigBufferImpl::has_shard_on(ChipId device_id) const {
    return std::all_of(
        parts_.begin(), parts_.end(), [&](const LocalPart& local) { return local.device_buffers.contains(device_id); });
}

DeviceAddr KernelConfigBufferImpl::address(
    const distributed::MeshCoordinate& device_coord, const CoreCoord& core) const {
    const Part& part = part_on(core).part;
    TT_FATAL(part.mesh_buffer != nullptr, "This kernel config buffer is per device: address it by device id");
    return per_core_allocation::get_per_core_address(*part.mesh_buffer->get_device_buffer(device_coord), core);
}

DeviceAddr KernelConfigBufferImpl::address(ChipId device_id, const CoreCoord& core) const {
    const LocalPart& local = part_on(core);
    const auto it = local.device_buffers.find(device_id);
    TT_FATAL(it != local.device_buffers.end(), "Kernel config buffer has no shard on device {}", device_id);
    return per_core_allocation::get_per_core_address(*it->second, core);
}

namespace {

// One page per core, each core allocated on its own, so the buffer takes L1 only on its cores.
BufferShardingArgs per_core_sharding(const CoreRangeSet& cores) {
    const uint32_t num_cores = cores.num_cores();
    ShardSpecBuffer shard_parameters(cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});
    BufferShardingArgs sharding_args(std::move(shard_parameters), TensorMemoryLayout::HEIGHT_SHARDED);
    per_core_allocation::set_per_core_allocation(sharding_args, true);
    return sharding_args;
}

// For a lockstep-address buffer: every core of `buffer` on its device has its shard at one address.
void check_lockstep_address(const Buffer& buffer, const CoreRangeSet& cores) {
    std::optional<DeviceAddr> first;
    for (const CoreCoord& core : corerange_to_cores(cores)) {
        const DeviceAddr address = per_core_allocation::get_per_core_address(buffer, core);
        TT_FATAL(
            !first.has_value() || *first == address,
            "Device {}: a lockstep-address kernel config buffer landed at 0x{:x} on one core and 0x{:x} on core {}: "
            "its "
            "cores' free L1 differs",
            buffer.device()->id(),
            *first,
            address,
            core);
        first = address;
    }
}

// One part: a `size`-byte shard on each of `cores`. With `lockstep_address`, allocated bottom up and required to land
// at one address on every core, which holds when the cores' free L1 starts at the same address.
KernelConfigBufferImpl::Part allocate_mesh_part(
    distributed::MeshDevice& device, const CoreRangeSet& cores, uint32_t size, bool lockstep_address) {
    auto buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = static_cast<DeviceAddr>(size) * cores.num_cores()},
        distributed::DeviceLocalBufferConfig{
            .page_size = size,
            .buffer_type = BufferType::L1,
            .sharding_args = per_core_sharding(cores),
            .bottom_up = lockstep_address ? std::optional<bool>(true) : std::nullopt,
        },
        &device);
    if (lockstep_address) {
        for (const distributed::MeshCoordinate& coord : distributed::MeshCoordinateRange(device.shape())) {
            if (device.is_local(coord)) {
                check_lockstep_address(*buffer->get_device_buffer(coord), cores);
            }
        }
    }
    return {.cores = cores, .size = size, .mesh_buffer = std::move(buffer)};
}

// Each core's block: its kernel group's, the largest if it is in different groups in different `programs` (the
// devices of a workload), aligned to `alignment`.
std::unordered_map<CoreCoord, uint32_t> blocks_on_cores(const std::vector<Program*>& programs, uint32_t alignment) {
    std::unordered_map<CoreCoord, uint32_t> block_on_core;
    for (Program* program : programs) {
        const uint32_t index = tensix_index(*program);
        const ProgramConfig& program_config = program->impl().get_program_config(index);
        for (const auto& kernel_group : program->impl().get_kernel_groups(index)) {
            const uint32_t block = tt::align(kernel_config_footprint(*kernel_group, program_config), alignment);
            for (const CoreCoord& core : corerange_to_cores(kernel_group->core_ranges)) {
                uint32_t& size = block_on_core[core];
                size = std::max(size, block);
            }
        }
    }
    TT_FATAL(!block_on_core.empty(), "Programs have no Tensix kernels, so no Tensix kernel config block");
    return block_on_core;
}

// The parts' (size, cores): one per block size, or, with a lockstep address, one at the largest block.
std::map<uint32_t, CoreRangeSet> shards_by_size(
    const std::unordered_map<CoreCoord, uint32_t>& block_on_core, bool lockstep_address) {
    std::map<uint32_t, std::vector<CoreRange>> cores_by_size;
    uint32_t largest = 0;
    for (const auto& [core, size] : block_on_core) {
        largest = std::max(largest, size);
    }
    for (const auto& [core, size] : block_on_core) {
        cores_by_size[lockstep_address ? largest : size].emplace_back(core);
    }
    std::map<uint32_t, CoreRangeSet> shards;
    for (auto& [size, cores] : cores_by_size) {
        shards.emplace(size, CoreRangeSet(cores).merge_ranges());
    }
    return shards;
}

[[maybe_unused]] std::string describe(const std::map<uint32_t, CoreRangeSet>& shards) {
    std::string out;
    for (const auto& [size, cores] : shards) {
        out += fmt::format(" {} B x {} cores;", size, cores.num_cores());
    }
    return out;
}

void check_hybrid(distributed::MeshDevice& device) {
    TT_FATAL(
        device.allocator_impl()->get_config().allocator_mode == AllocatorMode::HYBRID,
        "A kernel config buffer is allocated per core, which needs the device opened in the HYBRID allocator mode "
        "(set TT_METAL_ALLOCATOR_MODE_HYBRID=1 before opening the device)");
}

}  // namespace

KernelConfigBuffer AllocateKernelConfigBuffer(
    distributed::MeshDevice& device, const CoreRangeSet& cores, uint32_t size, bool lockstep_address) {
    check_hybrid(device);
    TT_FATAL(!cores.empty(), "A kernel config buffer needs at least one core");
    TT_FATAL(size > 0, "A kernel config buffer needs a non-zero size");
    const uint32_t aligned_size = tt::align(size, device.allocator_impl()->get_alignment(BufferType::L1));
    std::vector<KernelConfigBufferImpl::Part> parts;
    parts.push_back(allocate_mesh_part(device, cores, aligned_size, lockstep_address));
    return KernelConfigBuffer(std::make_shared<KernelConfigBufferImpl>(std::move(parts), lockstep_address));
}

void ValidateKernelConfigBuffer(Program& program, const KernelConfigBufferImpl& buffer, ChipId device_id) {
    TT_FATAL(buffer.is_allocated(), "Kernel config buffer was deallocated");

    const CoreRangeSet program_cores = program_tensix_cores(program);
    TT_FATAL(
        buffer.cores().contains(program_cores),
        "Kernel config buffer cores {} do not cover the program's Tensix cores {}",
        buffer.cores(),
        program_cores);
    TT_FATAL(buffer.has_shard_on(device_id), "Kernel config buffer has no shard on device {}", device_id);
    if (!buffer.lockstep_address()) {
        check_no_tensix_semaphores(program);
    }

    // Each core holds only its kernel group's part of the block, in a shard sized for it.
    const uint32_t index = tensix_index(program);
    const ProgramConfig& program_config = program.impl().get_program_config(index);
    for (const auto& kernel_group : program.impl().get_kernel_groups(index)) {
        const uint32_t footprint = kernel_config_footprint(*kernel_group, program_config);
        for (const CoreCoord& core : corerange_to_cores(kernel_group->core_ranges)) {
            TT_FATAL(
                footprint <= buffer.size(core),
                "Kernel group on {} needs a {} B kernel config block, but its kernel config buffer has {} B on core {}",
                kernel_group->core_ranges,
                footprint,
                buffer.size(core),
                core);
        }
    }
}

KernelConfigBuffer CreateKernelConfigBuffer(
    distributed::MeshDevice& device, distributed::MeshWorkload& workload, bool lockstep_address) {
    check_slow_dispatch(device);
    if (!lockstep_address) {
        for (auto& [_, program] : workload.get_programs()) {
            check_no_tensix_semaphores(program);
        }
    }
    // Laid out as launch lays it out, so launch reuses this. Nothing to do if the caller already compiled it.
    program_preparation::compile(device, workload);

    check_hybrid(device);
    std::vector<Program*> programs;
    for (auto& [_, program] : workload.get_programs()) {
        programs.push_back(&program);
    }
    const auto shards = shards_by_size(
        blocks_on_cores(programs, device.allocator_impl()->get_alignment(BufferType::L1)), lockstep_address);
    std::vector<KernelConfigBufferImpl::Part> parts;
    for (const auto& [size, cores] : shards) {
        parts.push_back(allocate_mesh_part(device, cores, size, lockstep_address));
    }
    log_debug(tt::LogMetal, "Kernel config buffer on MeshDevice {}:{}", device.id(), describe(shards));
    return KernelConfigBuffer(std::make_shared<KernelConfigBufferImpl>(std::move(parts), lockstep_address));
}

void EnqueueMeshWorkload(
    distributed::MeshCommandQueue& mesh_cq,
    distributed::MeshWorkload& workload,
    const KernelConfigBuffer& buffer,
    bool blocking) {
    distributed::MeshDevice& device = *mesh_cq.device();
    check_slow_dispatch(device);
    TT_FATAL(buffer.impl().is_allocated(), "Kernel config buffer was deallocated");
    const distributed::MeshDevice* buffer_device = buffer.impl().mesh_device();
    TT_FATAL(
        buffer_device == &device,
        "Kernel config buffer is allocated on MeshDevice {}, but the workload is enqueued on MeshDevice {}",
        buffer_device != nullptr ? std::to_string(buffer_device->id()) : std::string("(none)"),
        device.id());
    // Short-circuit for inactive MeshDevices (no-op), as distributed::EnqueueMeshWorkload does.
    if (device.get_device_ids().empty()) {
        return;
    }
    auto* sd_mesh_cq = dynamic_cast<distributed::SDMeshCommandQueue*>(&mesh_cq);
    TT_FATAL(sd_mesh_cq != nullptr, "A kernel config buffer needs a slow-dispatch mesh command queue");
    sd_mesh_cq->enqueue_mesh_workload(workload, buffer.impl(), blocking);
}

}  // namespace tt::tt_metal::experimental
