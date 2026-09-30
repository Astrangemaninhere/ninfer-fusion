#include "artifact/materializer.h"

// F1059: `ops/stream_capture.h` is deliberately NOT included any more. The fetch no longer needs
// to ASK whether a capture is running -- the fork in fetch_enroll() is issued unconditionally, so
// it is correct in both regimes and the branch that its predicate would have driven is gone.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::artifact {
namespace {

constexpr std::size_t kSlotBytes        = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumSlotCount = 4;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) { throw ArtifactError(label); }
    return a + b;
}

std::uint64_t align_down(std::uint64_t value, std::uint64_t alignment) {
    return value / alignment * alignment;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, const char* label) {
    return checked_add(value, alignment - 1, label) / alignment * alignment;
}

class Slot {
public:
    explicit Slot(std::size_t bytes) : buffer(bytes) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }

    ~Slot() {
        if (pending) { (void)cudaEventSynchronize(event); }
        if (event != nullptr) { (void)cudaEventDestroy(event); }
    }

    void wait() {
        if (pending) {
            CUDA_CHECK(cudaEventSynchronize(event));
            pending = false;
        }
    }

    PinnedHostBuffer buffer;
    cudaEvent_t event = nullptr;
    bool pending      = false;
};

struct CopyRange {
    std::uint64_t source_begin = 0;
    std::uint64_t source_end   = 0;
    std::byte* destination     = nullptr;
};

struct ReadSpan {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};


// W13: the CUDA backend behind product::WeightResidencyDevice. The pinned mirror
// uses cudaHostAllocDefault -- the same flag the KV cold tier's disk staging uses
// (program_impl.h:920), so the two host tiers behave identically under WSL.
class CudaWeightResidencyDevice final : public product::WeightResidencyDevice {
public:
    // THREE streams now, and the third one is the fix (F1059/streamfix).
    //
    // notehook split `stream_` into `consumer_` + `transfer_` and made the H2D ride
    // `transfer_` = `device.transfer_stream`. That is the ENGINE'S OWN stream: it carries the
    // context/state pipeline (`context_completion_.record(device.transfer_stream)` at
    // program_impl.h:6237/6294/6417/7278/9195/9632, `context_source_ready_.wait(
    // device.transfer_stream)` at :9151/:9561, `state_store->begin_host_to_device(...)` at
    // :6105/:6759), and the engine reads that timeline back through its own transfer timers
    // (:1006-1008). Injecting 8.56 GB per pass into it is what turned the offloaded run into the
    // engine's degenerate-head token 0 -- measured, not inferred: F1057's `CUDA_LAUNCH_BLOCKING=1`
    // and `NINFER_HEADDBG=1` arms are byte-identical to the resident arm.
    //
    // So the fetch gets a stream of its OWN, `fetch_`. Nothing is added to the engine's stream and
    // nothing on it is synchronised: the ordering the layer walk needs is one event per layer,
    // recorded on `fetch_` and waited on by `consumer_` (fetch_event_record / consumer_wait), and
    // a copy that has already landed makes that wait free.
    CudaWeightResidencyDevice(cudaStream_t consumer, cudaStream_t transfer)
        : consumer_(consumer), transfer_(transfer) {
        if (cudaStreamCreateWithFlags(&fetch_, cudaStreamNonBlocking) != cudaSuccess) {
            (void)cudaGetLastError();
            throw std::runtime_error("weight offload fetch stream creation failed");
        }
        // The fork event: published on the CONSUMER while the consumer is being captured, which is
        // what enrols `fetch_` in that capture (cudaStreamWaitEvent PROPAGATES the capture to the
        // stream it waits from -- src/core/decode_graph_peer.h:296-300). DisableTiming because it
        // is an edge, never a measurement.
        if (cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming) != cudaSuccess) {
            (void)cudaGetLastError();
            (void)cudaStreamDestroy(fetch_);
            fetch_ = nullptr;
            throw std::runtime_error("weight offload capture fork event creation failed");
        }
    }
    ~CudaWeightResidencyDevice() override {
        if (fork_ != nullptr) { (void)cudaEventDestroy(fork_); }
        if (fetch_ != nullptr) { (void)cudaStreamDestroy(fetch_); }
    }

    void* device_alloc(std::uint64_t bytes) override {
        void* slot = nullptr;
        if (cudaMalloc(&slot, static_cast<std::size_t>(bytes)) != cudaSuccess) {
            (void)cudaGetLastError();
            return nullptr;
        }
        return slot;
    }
    void device_free(void* slot) noexcept override { (void)cudaFree(slot); }
    void* pinned_alloc(std::uint64_t bytes) override {
        void* block = nullptr;
        if (cudaHostAlloc(&block, static_cast<std::size_t>(bytes), cudaHostAllocDefault) !=
            cudaSuccess) {
            (void)cudaGetLastError();
            return nullptr;
        }
        return block;
    }
    void pinned_free(void* pinned) noexcept override { (void)cudaFreeHost(pinned); }
    void enqueue_h2d(void* device_slot, const void* pinned, std::uint64_t bytes) override {
        // F1059: rides `fetch_`, NOT the engine's `transfer_`. A HOST->DEVICE cudaMemcpyAsync
        // from pinned memory is CAPTURABLE (it becomes a memcpy node); it is the HOST-destination
        // form that is illegal under capture, which is why `transfer_` cannot be used here and
        // why the old code had to skip the whole hook while capturing.
        CUDA_CHECK(cudaMemcpyAsync(device_slot, pinned, static_cast<std::size_t>(bytes),
                                   cudaMemcpyHostToDevice, fetch_));
    }
    // F1059: PUBLISH THE CONSUMER'S POSITION AND MAKE THE FETCH STREAM WAIT ON IT, before this
    // layer's copies are enqueued. TWO jobs, one pair of calls, and the first one is the fix.
    //
    //  1. THE ORDERING -- and this is NOT about graphs, which is why it is UNCONDITIONAL.
    //     slot(i) = i % arena_layers, and fetch_layer(index) prefetches layer
    //     `index + arena_layers - 1`, whose slot is
    //         (index + arena_layers - 1) % arena_layers = (index - 1) % arena_layers.
    //     That is EXACTLY slot(index - 1): the strip belonging to the layer the consumer enqueued
    //     ONE STEP EARLIER. At this host call that layer has only been ENQUEUED -- its GEMMs are
    //     queued or executing, and the CPU runs layers ahead -- so without this wait the copy lands
    //     in a strip that is still being read, and `fetch_layer`'s own eviction loop clears
    //     `resident_[index - 1]` while nothing joins the write to the read. It is a SYSTEMATIC
    //     ordering defect rather than a rare race (with k strips the overwritten strip is ALWAYS
    //     the previous layer's), which is why F1057's 1-strip and 6-strip runs and its 11-token and
    //     3,400-token runs all produced the same wrong bytes. Waiting on an event recorded on the
    //     CONSUMER here makes the copy's first byte land after everything the consumer has enqueued
    //     so far, the previous layer's GEMMs included.
    //  2. THE CAPTURE. The same pair is what enrols `fetch_` in a capture: cudaStreamWaitEvent
    //     PROPAGATES the capture to the stream it waits from (src/core/decode_graph_peer.h:296-300),
    //     so issued while the consumer is capturing, the copies become graph NODES and each layer's
    //     record/wait pair becomes a graph EDGE. One mechanism, both defects.
    //
    // Per LAYER rather than per capture on purpose: this tree has more than one capture site
    // (decode_impl.h:85, graph_impl.h:26, mtp_impl.h:420, dflash_impl.h:615), so a flag that
    // latched "already enrolled" would enrol the first capture and silently miss the second. A
    // per-layer fork carries no state across calls, and the matching rejoin is the per-layer
    // consumer_wait() that already exists.
    //
    // The cost is two stream ops per fetch (116 per 8-token round in the measured arm) and the
    // prefetch overlap the arena exists for is PRESERVED: `arena_layers - 1` layers of compute
    // still separate a copy from its consumer.
    void fetch_enroll() override {
        CUDA_CHECK(cudaEventRecord(fork_, consumer_));
        CUDA_CHECK(cudaStreamWaitEvent(fetch_, fork_, 0));
    }
    // NOTHING in this backend calls this any more, and that is the point: `synchronize()` used to
    // drain the stream the fetch ran on, and F1057 measured that the fetch path never called it
    // (it has no caller in weight_residency.h). It is kept as a HOST wait on `fetch_` so that a
    // future caller that wants a barrier gets one on the offload's OWN stream, not the engine's.
    void synchronize() override { CUDA_CHECK(cudaStreamSynchronize(fetch_)); }
    std::uint64_t now_ns() const noexcept override {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
    }
    // THE JOIN. `cudaEventDisableTiming` because nothing reads this event's duration: it is an
    // edge, not a measurement. The destroy tolerates nullptr so a partially constructed runtime
    // can still be torn down.
    void* fetch_event_create() override {
        cudaEvent_t event = nullptr;
        if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
            (void)cudaGetLastError();
            return nullptr;
        }
        return static_cast<void*>(event);
    }
    void fetch_event_record(void* event) override {
        CUDA_CHECK(cudaEventRecord(static_cast<cudaEvent_t>(event), fetch_));
    }
    void consumer_wait(void* event) override {
        CUDA_CHECK(cudaStreamWaitEvent(consumer_, static_cast<cudaEvent_t>(event), 0));
    }
    void fetch_event_destroy(void* event) noexcept override {
        if (event != nullptr) {
            (void)cudaEventDestroy(static_cast<cudaEvent_t>(event));
        }
    }

private:
    cudaStream_t consumer_ = nullptr;
    cudaStream_t transfer_ = nullptr;
    // F1059: the offload's OWN stream for the H2D, and the event that enrols it in a capture.
    cudaStream_t fetch_ = nullptr;
    cudaEvent_t fork_   = nullptr;
};

} // namespace

void* MaterializedArtifact::device_data(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].device == nullptr) {
        throw ArtifactError("object handle does not name a materialized tensor");
    }
    return objects_[handle.index].device;
}

std::span<const std::byte> MaterializedArtifact::mapped_tensor_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].mapped.empty()) {
        throw ArtifactError("object handle does not name a mapped tensor");
    }
    return objects_[handle.index].mapped;
}

std::span<const std::byte> MaterializedArtifact::resource_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    return objects_[handle.index].resource;
}

std::vector<std::byte> MaterializedArtifact::take_resource_bytes(ObjectHandle handle) {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    auto& resource = objects_[handle.index].resource;
    stats_.retained_resource_bytes -= resource.size();
    return std::move(resource);
}

DeviceArena& MaterializedArtifact::device_arena() {
    if (!device_arena_) { throw ArtifactError("artifact has no device tensor backing"); }
    return *device_arena_;
}

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device, LoadProgress* progress) {
    MaterializedArtifact out;
    // Take a share of the Reader's file mapping BEFORE anything can alias it: the spans assigned
    // below point into that mapping, and without this share the artifact would leave them dangling
    // the moment the caller's Reader goes out of scope. Gated on the plan having mapped tensors so
    // that a target with none keeps no mapping alive (see the member's note in materializer.h).
    if (!plan.mapped_tensor_objects.empty()) { out.mapping_lease_ = reader.mapping_lease(); }
    out.objects_.resize(plan.object_count);
    const std::uint64_t capacity = plan.device_capacity_bytes;
    if (capacity == 0 || capacity > static_cast<std::uint64_t>(SIZE_MAX)) {
        throw ArtifactError("artifact tensor backing size is invalid");
    }
    out.device_arena_ = std::make_unique<DeviceArena>(static_cast<std::size_t>(capacity));
    out.stats_.device_capacity_bytes = capacity;
    out.stats_.tensor_count          = plan.device_objects.size();
    out.stats_.resource_count        = plan.host_objects.size();
    out.stats_.mapped_tensor_count   = plan.mapped_tensor_objects.size();
    // Mapped placements: no device copy, no owning resource -- the span aliases the artifact's
    // own file mapping.  See the lifetime contract on mapped_tensor_bytes().
    for (const MappedTensorMaterialization& placement : plan.mapped_tensor_objects) {
        if (placement.object.index >= plan.object_count) {
            throw ArtifactError("mapped tensor placement does not name an artifact object");
        }
        const PayloadSpan payload = reader.payload(reader.objects()[placement.object.index]);
        out.objects_.at(placement.object.index).mapped = payload.data;
        out.stats_.mapped_tensor_bytes = checked_add(out.stats_.mapped_tensor_bytes,
                                                     payload.data.size(),
                                                     "mapped artifact tensor bytes overflow u64");
    }

    // W13: the offload arena and its runtime are built here (and only here) so the
    // CUDA backend above stays inside this translation unit.
    const product::WeightOffloadPlan& offload = plan.weight_offload;
    std::vector<std::size_t> offload_of_object(plan.device_objects.size(), SIZE_MAX);
    if (!offload.empty()) {
        if (offload.source_indices.size() != offload.spans.size()) {
            throw ArtifactError("weight offload plan is not internally consistent");
        }
        for (std::size_t s = 0; s < offload.spans.size(); ++s) {
            if (offload.source_indices[s] >= plan.device_objects.size()) {
                throw ArtifactError("weight offload span does not name a device object");
            }
            offload_of_object[offload.source_indices[s]] = s;
        }
        std::uint64_t declared = 0;
        for (std::size_t i = 0; i < plan.device_objects.size(); ++i) {
            if (plan.device_objects[i].offloaded != (offload_of_object[i] != SIZE_MAX)) {
                throw ArtifactError("weight offload flag does not match the offload plan");
            }
            if (plan.device_objects[i].offloaded) { declared += plan.device_objects[i].bytes; }
        }
        if (declared != offload.offloaded_bytes) {
            throw ArtifactError("weight offload byte accounting does not match the plan");
        }
        // notehook: the CONSUMING stream is passed as well, so the runtime can make it wait on
        // each layer's fetch event. `device.stream` is the stream the layer walk computes on
        // (TextContext holds this same DeviceContext and uses ctx_.stream throughout).
        out.weight_backend_ =
            std::make_shared<CudaWeightResidencyDevice>(device.stream, device.transfer_stream);

        out.weight_residency_ =
            std::make_unique<product::WeightResidencyRuntime>(offload, out.weight_backend_.get());
        out.stats_.weight_host_bytes         = offload.offloaded_bytes;
        out.stats_.weight_device_arena_bytes = offload.device_arena_bytes;
        out.stats_.weight_device_bytes_freed = offload.device_bytes_freed;
        std::fprintf(stderr,
                     "[weight-offload] host=%llu B arena=%llu B freed=%llu B layers=%zu "
                     "arena_layers=%u stride=%u B\n",
                     static_cast<unsigned long long>(offload.offloaded_bytes),
                     static_cast<unsigned long long>(offload.device_arena_bytes),
                     static_cast<unsigned long long>(offload.device_bytes_freed),
                     offload.layers.size(), offload.arena_layers, offload.layer_stride);
    }

    for (const HostMaterialization& placement : plan.host_objects) {
        auto& resource            = out.objects_.at(placement.object.index).resource;
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        resource.assign(payload.data.begin(), payload.data.end());
        out.stats_.retained_resource_bytes += resource.size();
        out.stats_.file_bytes =
            checked_add(out.stats_.file_bytes, resource.size(), "artifact read bytes overflow u64");
    }

    std::vector<CopyRange> ranges;
    ranges.reserve(plan.device_objects.size());
    std::uint64_t copied         = 0;
    std::uint64_t last_published = 0;
    std::uint64_t total          = 0;
    for (std::size_t object_index = 0; object_index < plan.device_objects.size(); ++object_index) {
        const DeviceMaterialization& placement = plan.device_objects[object_index];
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        if (placement.offloaded) {
            // Byte-exact mirror of the mmap into pinned host memory. The device
            // address handed back is STABLE from here on, so nothing downstream has
            // to know this object ever left the device.
            //
            // These bytes are deliberately NOT added to `total`: nothing here reads
            // them from the artifact (adopt_span fills their mirror straight from the
            // mmap), and `copied` only ever accumulates the resident ranges, so
            // counting them made `copied != total` true by construction and the
            // completeness check below threw on every run that offloaded anything.
            out.objects_.at(placement.object.index).device =
                out.weight_residency_->adopt_span(offload_of_object.at(object_index),
                                                  payload.data.data());
            continue;
        }
        DeviceSpan storage =
            out.device_arena_->alloc_bytes(static_cast<std::size_t>(placement.bytes),
                                           static_cast<std::size_t>(placement.alignment));
        const auto actual_offset =
            static_cast<std::uint64_t>(static_cast<std::byte*>(storage.data) -
                                       static_cast<std::byte*>(out.device_arena_->base()));
        if (actual_offset != placement.offset || payload.data.size() != placement.bytes) {
            throw ArtifactError("materialization plan does not match artifact payload");
        }
        out.objects_.at(placement.object.index).device = storage.data;
        ranges.push_back(CopyRange{
            .source_begin = payload.absolute_offset,
            .source_end   = checked_add(payload.absolute_offset, placement.bytes,
                                        "artifact tensor source range overflows u64"),
            .destination  = static_cast<std::byte*>(storage.data),
        });
        total = checked_add(total, placement.bytes, "artifact tensor byte count overflows u64");
    }
    if (ranges.empty()) { throw ArtifactError("materialization plan has no device tensors"); }
    std::sort(ranges.begin(), ranges.end(), [](const CopyRange& a, const CopyRange& b) {
        return a.source_begin < b.source_begin;
    });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].source_begin < ranges[i - 1].source_end) {
            throw ArtifactError("materialization source ranges overlap");
        }
    }

    constexpr std::uint64_t alignment = Reader::direct_io_alignment;
    std::vector<ReadSpan> read_spans;
    read_spans.reserve(ranges.size());
    std::uint64_t aligned_read_bytes = 0;
    for (const CopyRange& range : ranges) {
        const std::uint64_t begin = align_down(range.source_begin, alignment);
        if (read_spans.empty() || begin > align_up(read_spans.back().end, alignment,
                                                   "artifact direct I/O span overflows u64")) {
            read_spans.push_back(ReadSpan{begin, range.source_end});
        } else {
            read_spans.back().end = std::max(read_spans.back().end, range.source_end);
        }
    }
    for (const ReadSpan& span : read_spans) {
        aligned_read_bytes = checked_add(
            aligned_read_bytes,
            align_up(span.end - span.begin, alignment, "artifact direct I/O span overflows u64"),
            "artifact direct I/O byte count overflows u64");
    }
    const std::size_t slot_bytes =
        static_cast<std::size_t>(std::min<std::uint64_t>(kSlotBytes, aligned_read_bytes));
    const std::size_t slot_count = static_cast<std::size_t>(
        std::min<std::uint64_t>(kMaximumSlotCount, 1 + (aligned_read_bytes - 1) / slot_bytes));
    std::vector<std::unique_ptr<Slot>> slots;
    slots.reserve(slot_count);
    for (std::size_t i = 0; i < slot_count; ++i) {
        slots.push_back(std::make_unique<Slot>(slot_bytes));
    }
    out.stats_.peak_staging_bytes = static_cast<std::uint64_t>(slot_bytes) * slot_count;

    std::size_t next_slot  = 0;
    std::size_t next_range = 0;
    const auto start       = std::chrono::steady_clock::now();
    if (progress != nullptr && progress->callback) { progress->callback("weights", 0, total); }
    for (const ReadSpan& span : read_spans) {
        for (std::uint64_t source = span.begin; source < span.end; source += slot_bytes) {
            Slot& slot = *slots[next_slot++ % slot_count];
            slot.wait();

            const std::uint64_t remaining = span.end - source;
            const std::size_t request     = static_cast<std::size_t>(std::min<std::uint64_t>(
                slot_bytes,
                align_up(remaining, alignment, "artifact direct I/O request overflows u64")));
            auto destination =
                std::span<std::byte>(static_cast<std::byte*>(slot.buffer.data()), request);
            const std::size_t bytes_read = reader.read_direct(source, destination);
            const std::uint64_t required = std::min<std::uint64_t>(request, remaining);
            if (bytes_read < required) {
                throw ArtifactError("direct artifact read ended before the planned tensor range");
            }
            out.stats_.file_bytes =
                checked_add(out.stats_.file_bytes, bytes_read, "artifact read bytes overflow u64");
            const std::uint64_t chunk_end =
                checked_add(source, bytes_read, "artifact direct I/O result overflows u64");

            while (next_range < ranges.size() && ranges[next_range].source_end <= source) {
                ++next_range;
            }
            std::size_t range_index = next_range;
            while (range_index < ranges.size() && ranges[range_index].source_begin < chunk_end) {
                const CopyRange& range         = ranges[range_index];
                const std::uint64_t copy_begin = std::max(source, range.source_begin);
                const std::uint64_t copy_end   = std::min(chunk_end, range.source_end);
                if (copy_begin < copy_end) {
                    const auto amount = static_cast<std::size_t>(copy_end - copy_begin);
                    CUDA_CHECK(cudaMemcpyAsync(
                        range.destination +
                            static_cast<std::size_t>(copy_begin - range.source_begin),
                        static_cast<std::byte*>(slot.buffer.data()) +
                            static_cast<std::size_t>(copy_begin - source),
                        amount, cudaMemcpyHostToDevice, device.transfer_stream));
                    copied =
                        checked_add(copied, amount, "artifact copied byte count overflows u64");
                }
                if (range.source_end <= chunk_end) {
                    ++range_index;
                } else {
                    break;
                }
            }
            next_range = range_index;
            CUDA_CHECK(cudaEventRecord(slot.event, device.transfer_stream));
            slot.pending = true;

            if (progress != nullptr && progress->callback && copied != last_published &&
                copied < total) {
                last_published = copied;
                progress->callback("weights", copied, total);
            }
        }
    }
    for (const auto& slot : slots) { slot->wait(); }
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    if (copied != total || next_range != ranges.size()) {
        throw ArtifactError("direct materialization did not cover every tensor byte");
    }
    out.stats_.h2d_bytes = copied;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (progress != nullptr && progress->callback) { progress->callback("weights", copied, total); }
    return out;
}

} // namespace ninfer::artifact
