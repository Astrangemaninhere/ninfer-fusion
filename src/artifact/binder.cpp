#include "artifact/binder.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace ninfer::artifact {
namespace {

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    const std::uint64_t mask = alignment - 1;
    if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
        throw ArtifactError("materialization plan size overflows u64");
    }
    return (value + mask) & ~mask;
}

} // namespace

Binder::Binder(const Reader& reader)
    : reader_(reader), consumed_(reader.objects().size(), false),
      planned_(reader.objects().size(), false) {
    materialization_.object_count = reader.objects().size();
}

ObjectHandle Binder::find_unconsumed(std::string_view name) {
    const auto& objects            = reader_.objects();
    const ObjectDescriptor* object = reader_.find(name);
    if (object == nullptr) {
        throw ArtifactError("required artifact object is missing: " + std::string(name));
    }
    const auto index = static_cast<std::size_t>(object - objects.data());
    if (consumed_[index]) {
        throw ArtifactError("artifact object was bound more than once: " + std::string(name));
    }
    consumed_[index] = true;
    return ObjectHandle{index};
}

ObjectHandle Binder::require_tensor(std::string_view name, NumericFormat format,
                                    StorageLayout layout, std::span<const std::uint64_t> shape) {
    const ObjectHandle handle = find_unconsumed(name);
    const auto* tensor        = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) {
        throw ArtifactError("required tensor is a resource: " + std::string(name));
    }
    if (tensor->format != format || tensor->layout != layout ||
        !std::equal(tensor->shape.begin(), tensor->shape.end(), shape.begin(), shape.end())) {
        throw ArtifactError("tensor descriptor does not match target contract: " +
                            std::string(name));
    }
    return handle;
}

ObjectHandle Binder::require_resource(std::string_view name, ResourceEncoding encoding) {
    const ObjectHandle handle = find_unconsumed(name);
    const auto* resource      = std::get_if<ResourceDescriptor>(&descriptor(handle));
    if (resource == nullptr) {
        throw ArtifactError("required resource is a tensor: " + std::string(name));
    }
    if (resource->encoding != encoding) {
        throw ArtifactError("resource encoding does not match target contract: " +
                            std::string(name));
    }
    return handle;
}

const TensorDescriptor* Binder::find_tensor(std::string_view name) const noexcept {
    const ObjectDescriptor* object = reader_.find(name);
    return object == nullptr ? nullptr : std::get_if<TensorDescriptor>(object);
}

const ObjectDescriptor& Binder::descriptor(ObjectHandle handle) const {
    if (handle.index >= reader_.objects().size()) {
        throw ArtifactError("artifact object handle is out of range");
    }
    return reader_.objects()[handle.index];
}

PayloadSpan Binder::payload(ObjectHandle handle) const {
    return reader_.payload(descriptor(handle));
}

void Binder::materialize_on_device(ObjectHandle handle) {
    const auto* tensor = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) {
        throw ArtifactError("resource cannot be materialized as a device tensor");
    }
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(tensor->name));
    }
    const std::uint64_t alignment = tensor_alignment(tensor->layout);
    const std::uint64_t offset    = align_up(materialization_.device_capacity_bytes, alignment);
    if (tensor->bytes > std::numeric_limits<std::uint64_t>::max() - offset) {
        throw ArtifactError("materialization plan size overflows u64");
    }
    materialization_.device_objects.push_back(DeviceMaterialization{
        .object = handle, .offset = offset, .bytes = tensor->bytes, .alignment = alignment});
    materialization_.device_capacity_bytes = offset + tensor->bytes;
    planned_[handle.index]                 = true;
}

void Binder::retain_on_host(ObjectHandle handle) {
    const auto* resource = std::get_if<ResourceDescriptor>(&descriptor(handle));
    if (resource == nullptr) {
        throw ArtifactError("tensor cannot be retained as a host resource");
    }
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(resource->name));
    }
    materialization_.host_objects.push_back(HostMaterialization{handle});
    planned_[handle.index] = true;
}

void Binder::retain_mapped_tensor(ObjectHandle handle) {
    const auto* tensor = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) {
        throw ArtifactError("resource cannot be retained as a mapped tensor");
    }
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(tensor->name));
    }
    materialization_.mapped_tensor_objects.push_back(MappedTensorMaterialization{handle});
    planned_[handle.index] = true;
}

void Binder::validate_only(ObjectHandle handle) {
    const ObjectDescriptor& object = descriptor(handle);
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(object_name(object)));
    }
    planned_[handle.index] = true;
}

MaterializationPlan Binder::finish() {
    const auto it = std::find(consumed_.begin(), consumed_.end(), false);
    if (it != consumed_.end()) {
        const auto index = static_cast<std::size_t>(it - consumed_.begin());
        throw ArtifactError("artifact object was not consumed by the selected target: " +
                            std::string(object_name(reader_.objects()[index])));
    }
    const auto unplanned = std::find(planned_.begin(), planned_.end(), false);
    if (unplanned != planned_.end()) {
        const auto index = static_cast<std::size_t>(unplanned - planned_.begin());
        throw ArtifactError("artifact object has no materialization placement: " +
                            std::string(object_name(reader_.objects()[index])));
    }
    if (has_offload_limits_) {
        if (planned_offload_) {
            throw ArtifactError("weight offload was planned before finish()");
        }
        (void)plan_weight_offload(offload_limits_);
    }
    return std::move(materialization_);
}

std::vector<product::WeightSpanSource> Binder::weight_span_sources() const {
    std::vector<product::WeightSpanSource> sources;
    sources.reserve(materialization_.device_objects.size());
    for (const DeviceMaterialization& placement : materialization_.device_objects) {
        const ObjectDescriptor& object = reader_.objects().at(placement.object.index);
        const PayloadSpan payload      = reader_.payload(object);
        if (payload.data.size() != placement.bytes) {
            throw ArtifactError("artifact payload does not match the planned span");
        }
        sources.push_back(product::WeightSpanSource{std::string(object_name(object)),
                                                    payload.absolute_offset, placement.bytes});
    }
    return sources;
}

void Binder::set_weight_offload_limits(product::WeightOffloadLimits limits) {
    if (has_offload_limits_) { throw ArtifactError("weight offload limits were set twice"); }
    has_offload_limits_ = true;
    offload_limits_     = limits;
}

product::WeightOffloadPlan Binder::plan_weight_offload(const product::WeightOffloadLimits& limits) {
    if (planned_offload_) { throw ArtifactError("weight offload was planned twice"); }
    planned_offload_ = true;
    if (limits.host_pinned_bytes == 0) { return {}; }

    // The span floor is DERIVED, not chosen: a span must be at least 1/4096 of the
    // device weight arena and never below 1 MiB, or a slot reserved for it is waste.
    product::WeightOffloadLimits effective = limits;
    if (effective.min_span_bytes == 0) {
        effective.min_span_bytes = std::max<std::uint64_t>(
            1ULL << 20, materialization_.device_capacity_bytes / 4096);
    }

    product::WeightOffloadPlan plan = product::build_weight_offload_plan(weight_span_sources(),
                                                                        effective);

    // Which device objects leave the arena. The plan is the authority, and its
    // membership is per OBJECT, not per layer: build_weight_offload_plan() drops
    // every object below `min_span_bytes` (weight_residency.h: "Objects smaller
    // than this are not worth a slot (norms, biases, a_log...)"), so a selected
    // layer keeps its norms resident while its large planes move. That is the
    // contract tests/test_weight_residency.cpp pins ("the sub-floor norms stay
    // resident", "offloaded bytes count only spans above the floor").
    //
    // Returning membership by LAYER instead -- which is what this did -- counted
    // the sub-floor bytes no span covers, so the guard below fired on every
    // artifact that has a norm inside a layer, i.e. on all of them.
    //
    // source_indices index weight_span_sources(), which is built from
    // materialization_.device_objects in order, so they are device-object indices.
    std::vector<bool> offloaded(materialization_.device_objects.size(), false);
    std::uint64_t offloaded_bytes = 0;
    for (const std::size_t object_index : plan.source_indices) {
        if (object_index >= materialization_.device_objects.size()) {
            throw ArtifactError("weight offload plan names an object that is not a device "
                                "materialization");
        }
        if (offloaded[object_index]) {
            throw ArtifactError("weight offload plan names a device object twice");
        }
        const DeviceMaterialization& placement = materialization_.device_objects[object_index];
        const ObjectDescriptor& object         = reader_.objects().at(placement.object.index);
        const std::optional<product::WeightSpanLocation> at =
            product::weight_span_location(object_name(object));
        if (!at.has_value()) {
            throw ArtifactError("weight offload plan offloads an object that has no layer: " +
                                std::string(object_name(object)));
        }
        // The span floor is the plan's own selection rule, so a plan that offloads
        // a sub-floor object is malformed. Asserting it here is what makes this
        // loop a check rather than a restatement of the plan.
        if (placement.bytes < effective.min_span_bytes) {
            throw ArtifactError(
                "weight offload plan offloads an object below the span floor: " +
                std::string(object_name(object)) + " (" + std::to_string(placement.bytes) +
                " B < " + std::to_string(effective.min_span_bytes) + " B)");
        }
        offloaded[object_index]  = true;
        offloaded_bytes += placement.bytes;
    }
    if (offloaded_bytes != plan.offloaded_bytes) {
        throw ArtifactError(
            "weight offload plan does not match the artifact's device objects: the plan "
            "offloads " + std::to_string(plan.offloaded_bytes) +
            " B but the device objects it names weigh " + std::to_string(offloaded_bytes) +
            " B of the " + std::to_string(materialization_.device_objects.size()) +
            " device objects in this artifact");
    }

    // Re-lay the resident arena. Iterating device_objects in binding order and
    // skipping the offloaded ones reproduces exactly what materialize_on_device()
    // computed for the survivors (the same align_up chain), so no resident offset
    // moves and no payload comparison in materialize() can change meaning.
    std::uint64_t capacity = 0;
    for (std::size_t i = 0; i < materialization_.device_objects.size(); ++i) {
        DeviceMaterialization& placement = materialization_.device_objects[i];
        if (offloaded[i]) {
            placement.offloaded = true;
            continue;
        }
        const std::uint64_t offset = align_up(capacity, placement.alignment);
        if (placement.bytes > std::numeric_limits<std::uint64_t>::max() - offset) {
            throw ArtifactError("materialization plan size overflows u64");
        }
        placement.offset = offset;
        capacity         = offset + placement.bytes;
    }
    if (capacity == 0) {
        throw ArtifactError("weight offload moved every device tensor to the host; the engine "
                            "needs at least one resident device tensor");
    }
    materialization_.device_capacity_bytes = capacity;
    materialization_.weight_offload        = plan;
    return plan;
}

} // namespace ninfer::artifact
