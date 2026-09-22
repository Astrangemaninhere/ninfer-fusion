#pragma once

#include "artifact/reader.h"
#include "product/weight_offload_budget.h"
#include "product/weight_residency.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::artifact {

enum class TensorPlacement : std::uint8_t {
    Device,
    ValidateOnly,
};

struct ObjectHandle {
    std::size_t index = 0;
};

struct DeviceMaterialization {
    ObjectHandle object;
    // W13: when `offloaded` is set this object's bytes do NOT come from the
    // resident device arena. `bytes`/`alignment` keep their meaning (the span
    // inside the artifact mmap); `offset` is recomputed for the resident set only.
    bool offloaded          = false;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 0;
};

struct HostMaterialization {
    ObjectHandle object;
};

// A tensor whose bytes stay in the artifact's own file mapping on the host: nothing is copied to
// the device and nothing is retained as an owning resource.  Added for the copied Flash-Next
// target, which keeps BF16 payloads in the mapping and quantizes them from there.  Donor:
// igorls/ninfer @ 5e4a66d src/artifact/binder.h:31.
struct MappedTensorMaterialization {
    ObjectHandle object;
};

struct MaterializationPlan {
    std::size_t object_count            = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::vector<DeviceMaterialization> device_objects;
    std::vector<HostMaterialization> host_objects;
    // W13 P0 carrier (rides the plan like the KV tables layer_residual /
    // layer_sliding_windows): per-(layer, expert) weight residency decided at
    // load time. EMPTY = every device object stays resident (today's behaviour;
    // the materializer is unchanged in P0 and ignores the field until the P1
    // GEMM residency hook lands).
    std::vector<product::WeightSpan> weight_residency;
    // W13 P1: the resolved offload plan (which layer groups leave the device
    // arena, the cyclic device arena, the bytes actually freed). EMPTY = every
    // device object stays resident, i.e. today's behaviour.
    product::WeightOffloadPlan weight_offload;
    // Mapped-tensor placements (see MappedTensorMaterialization).  EMPTY for every target that
    // does not call Binder::retain_mapped_tensor, i.e. for all of them but Flash-Next.
    std::vector<MappedTensorMaterialization> mapped_tensor_objects;
};

class Binder {
public:
    explicit Binder(const Reader& reader);

    // The reader this binder plans against.  Added for Flash-Next, which asks the artifact
    // directly whether a tensor exists.  Donor: igorls/ninfer @ 5e4a66d src/artifact/binder.h:49.
    [[nodiscard]] const Reader& reader() const noexcept { return reader_; }

    ObjectHandle require_tensor(std::string_view name, NumericFormat format, StorageLayout layout,
                                std::span<const std::uint64_t> shape);
    ObjectHandle require_resource(std::string_view name, ResourceEncoding encoding);

    // Read-only lookup of what the artifact declares under `name`: nullptr when the
    // object is absent, and nullptr when it is a resource.  It consumes nothing and
    // asserts nothing, so a target can read the artifact's own contract (format,
    // layout, shape) and then bind that object through require_tensor.
    [[nodiscard]] const TensorDescriptor* find_tensor(std::string_view name) const noexcept;

    const ObjectDescriptor& descriptor(ObjectHandle handle) const;
    PayloadSpan payload(ObjectHandle handle) const;
    void materialize_on_device(ObjectHandle handle);
    void retain_on_host(ObjectHandle handle);
    // Plan `handle` to stay in the artifact's file mapping on the host: no device copy, no owning
    // resource.  The materialized span is valid only while the Reader that planned it is alive;
    // see the note on MaterializedArtifact::mapped_tensor_bytes.
    void retain_mapped_tensor(ObjectHandle handle);
    void validate_only(ObjectHandle handle);
    // finish() returns the plan, and the W13 offload decision (when limits were set)
    // is applied inside it: the decision has to move bytes OFF device_capacity_bytes
    // before the materializer sizes the resident arena. Callers therefore only set
    // the limits, so no per-target load path has to change.
    MaterializationPlan finish();

    // W13: the offload budget. Zero host bytes keeps the field empty and changes
    // nothing (offload stays opt-in). Call before finish().
    void set_weight_offload_limits(product::WeightOffloadLimits limits);

    // W13: the device weight objects as the artifact describes them (object name
    // + mmap span), in binding order. Read from the reader and the plan, never
    // from a compiled-in per-model table.
    [[nodiscard]] std::vector<product::WeightSpanSource> weight_span_sources() const;

    // W13: classify the device objects against `limits`, mark the chosen ones as
    // offloaded, and recompute device_capacity_bytes so the freed bytes are real
    // rather than merely declared. Must run after every binding and before
    // finish(); running it twice is an error. Returns the resolved plan.
    product::WeightOffloadPlan plan_weight_offload(const product::WeightOffloadLimits& limits);

private:
    ObjectHandle find_unconsumed(std::string_view name);

    const Reader& reader_;
    std::vector<bool> consumed_;
    std::vector<bool> planned_;
    bool planned_offload_        = false;
    bool has_offload_limits_     = false;
    product::WeightOffloadLimits offload_limits_{};
    MaterializationPlan materialization_;
};

} // namespace ninfer::artifact
