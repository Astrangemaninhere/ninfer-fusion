#pragma once

#include "artifact/binder.h"
#include "core/arena.h"
#include "core/device.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::artifact {

struct LoadProgress {
    std::function<void(std::string_view, std::uint64_t, std::uint64_t)> callback;
};

struct MaterializationStats {
    std::uint64_t file_bytes              = 0;
    std::uint64_t h2d_bytes               = 0;
    std::uint64_t device_capacity_bytes   = 0;
    std::uint64_t retained_resource_bytes = 0;
    std::uint64_t peak_staging_bytes      = 0;
    std::size_t tensor_count              = 0;
    std::size_t resource_count            = 0;
    double upload_seconds                 = 0.0;
    // W13: the offload's three numbers, so the startup path can be honest about
    // where the bytes went instead of implying nothing moved.
    std::uint64_t weight_host_bytes         = 0; // pinned host mirror
    std::uint64_t weight_device_arena_bytes = 0; // device working set kept back
    std::uint64_t weight_device_bytes_freed = 0; // the point of the exercise
    // Mapped-tensor placements: bytes left in the artifact's file mapping and how many objects
    // that was.  Both stay zero for every target that does not call retain_mapped_tensor.
    // `tensor_count` above intentionally keeps counting ONLY device tensors.
    std::uint64_t mapped_tensor_bytes = 0;
    std::size_t mapped_tensor_count   = 0;
};

class MaterializedArtifact {
public:
    MaterializedArtifact()                                           = default;
    ~MaterializedArtifact()                                          = default;
    MaterializedArtifact(MaterializedArtifact&&) noexcept            = default;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept = default;
    MaterializedArtifact(const MaterializedArtifact&)                = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&)     = delete;

    void* device_data(ObjectHandle handle) const;
    std::span<const std::byte> resource_bytes(ObjectHandle handle) const;
    // Host-mapped payload of an object bound with Binder::retain_mapped_tensor.  Throws when the
    // handle was not planned as a mapped tensor.
    //
    // LIFETIME CONTRACT, ENFORCED: the returned span aliases the Reader's file mapping, and this
    // artifact HOLDS A LEASE on that mapping (mapping_lease_ below), so the span stays valid for as
    // long as the artifact does -- including after the Reader that produced it has been destroyed.
    // The donor's design was the same (igorls/ninfer @ 5e4a66d src/artifact/materializer.h:58,
    // `mapping_lease_`).
    //
    // This used to be an open use-after-munmap, not a documented hazard: this tree had no
    // Reader::mapping_lease(), so no lease could be taken and the contract could only ASK callers
    // to outlive the Reader. MEASURED 2026-09-18 (FN-GATE) with a probe that destroys the Reader
    // and then reads the span: SIGSEGV without the lease, intact bytes with it. See
    // scratch/PATCHSET/FNGATE/REPORT.md and probe/mapped_lease_probe.cpp.
    std::span<const std::byte> mapped_tensor_bytes(ObjectHandle handle) const;
    std::vector<std::byte> take_resource_bytes(ObjectHandle handle);

    const MaterializationStats& stats() const noexcept { return stats_; }

    DeviceArena& device_arena();

    // W13: the weight residency runtime, or nullptr when nothing was offloaded.
    // The engine reads it to drive the layer-ordered fetch (see
    // text_context_impl.h run_layers) and to print the counters. It is owned here
    // so the CUDA backend stays inside materializer.cpp.
    // The runtime mutates its counters and its arena contents, so this handle is
    // mutable even off a const artifact -- the same shape core/weight.h's as_dense
    // uses for a const Weight's buffer.
    [[nodiscard]] product::WeightResidencyRuntime* weight_residency() const noexcept {
        return weight_residency_.get();
    }

private:
    friend MaterializedArtifact materialize(const Reader&, const MaterializationPlan&,
                                            DeviceContext&, LoadProgress*);

    struct ObjectStorage {
        void* device = nullptr;
        std::vector<std::byte> resource;
        std::span<const std::byte> mapped;
    };

    std::unique_ptr<DeviceArena> device_arena_;
    std::unique_ptr<product::WeightResidencyRuntime> weight_residency_;
    // A share of the Reader's file mapping, so `objects_[i].mapped` stays readable after the
    // Reader dies. Populated by materialize() ONLY when the plan actually has mapped tensors, so
    // every target that does not call Binder::retain_mapped_tensor (all of them but FlashNext)
    // keeps exactly the file descriptor and address-space behaviour it had before -- no lease, no
    // extra mapping held for the artifact's lifetime.
    //
    // Declared BEFORE objects_ on purpose: members destruct in reverse declaration order, so
    // objects_ (whose mapped spans point into the mapping) is destroyed first and the lease last.
    std::shared_ptr<const void> mapping_lease_;
    // Held so the injected backend outlives the runtime that makes calls into it.
    std::shared_ptr<product::WeightResidencyDevice> weight_backend_;
    std::vector<ObjectStorage> objects_;
    MaterializationStats stats_;
};

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device, LoadProgress* progress = nullptr);

} // namespace ninfer::artifact
