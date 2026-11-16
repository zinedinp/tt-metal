// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>
#include <utility>

#include <tt-metalium/experimental/kernel_config_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program_cache.hpp>

#include "ttnn/distributed/types.hpp"

// Experimental and subject to change.

namespace ttnn::device_operation {

// A program factory's mesh workload, with the factory's index, type-erased the way the program cache stores it, and how
// to find the workload in it.
struct ErasedCachedMeshWorkload {
    tt::tt_metal::program_cache::detail::CachedProgramFactory cached_program_factory;
    tt::tt_metal::distributed::MeshWorkload& (*workload_of)(
        tt::tt_metal::program_cache::detail::CachedProgramFactory&) = nullptr;

    tt::tt_metal::distributed::MeshWorkload& workload() { return workload_of(cached_program_factory); }
};

// A device operation's mesh workload, compiled and laid out for one set of operation attributes and tensors
// (device_operation::compile), outside the program cache. device_operation::run launches it with any operation
// attributes and tensors that have the same program cache key, applying their runtime arguments as a cache hit does,
// optionally from a kernel config buffer (create_kernel_config_buffer).
//
// Tied to the MeshDevice it was compiled for. Move-only.
class CompiledProgram {
public:
    CompiledProgram(
        ErasedCachedMeshWorkload workload,
        tt::tt_metal::program_cache::detail::ProgramCacheKey key,
        std::string operation_name,
        ttnn::MeshDevice* mesh_device) :
        workload_(std::move(workload)),
        key_(std::move(key)),
        operation_name_(std::move(operation_name)),
        mesh_device_(mesh_device) {}

    CompiledProgram(CompiledProgram&&) noexcept = default;
    CompiledProgram& operator=(CompiledProgram&&) noexcept = default;
    CompiledProgram(const CompiledProgram&) = delete;
    CompiledProgram& operator=(const CompiledProgram&) = delete;

    tt::tt_metal::distributed::MeshWorkload& workload() { return workload_.workload(); }
    tt::tt_metal::program_cache::detail::CachedProgramFactory& cached_program_factory() {
        return workload_.cached_program_factory;
    }
    std::size_t program_factory_index() const { return workload_.cached_program_factory.program_factory_index; }
    const tt::tt_metal::program_cache::detail::ProgramCacheKey& key() const { return key_; }
    // The device operation it was compiled from.
    const std::string& operation_name() const { return operation_name_; }
    ttnn::MeshDevice& mesh_device() const { return *mesh_device_; }

private:
    ErasedCachedMeshWorkload workload_;
    tt::tt_metal::program_cache::detail::ProgramCacheKey key_;
    std::string operation_name_;
    ttnn::MeshDevice* mesh_device_;
};

// A kernel config buffer for `compiled` (experimental::CreateKernelConfigBuffer), for device_operation::run to launch
// it from instead of the KERNEL_CONFIG window. Needs slow dispatch and the device opened with
// TT_METAL_ALLOCATOR_MODE_HYBRID=1; rejects Tensix program semaphores unless `lockstep_address`.
inline tt::tt_metal::experimental::KernelConfigBuffer create_kernel_config_buffer(
    CompiledProgram& compiled, bool lockstep_address = false) {
    return tt::tt_metal::experimental::CreateKernelConfigBuffer(
        compiled.mesh_device(), compiled.workload(), lockstep_address);
}

}  // namespace ttnn::device_operation
