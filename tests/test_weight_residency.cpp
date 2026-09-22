// Unit test for W13 weight host-offload: product/weight_residency.h (the plan and
// the fetch schedule) and product/weight_offload_budget.h (the transfer math).
//
// Host-only by construction: the schedule talks to an injected
// product::WeightResidencyDevice, so a counting/faking backend exercises the whole
// path -- classification, arena layout, prefetch ordering, eviction, byte-exact
// mirroring, address stability -- with plain g++ and no CUDA, GPU or artifact.
#include "product/weight_offload_budget.h"
#include "product/weight_residency.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

namespace p = ninfer::product;

int failures = 0;
// TIEROFFLOAD: the number of assertions actually EXECUTED, so "all checks passed"
// can be quoted as a count instead of an adjective.
std::uint64_t checks_run = 0;

void check(bool condition, const std::string& message) {
    ++checks_run;
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

template <typename Fn>
void rejects(const std::string& needle, Fn&& operation) {
    ++checks_run;
    try {
        operation();
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(needle) == std::string::npos) {
            std::cerr << "FAIL: error '" << error.what() << "' does not mention '" << needle
                      << "'\n";
            ++failures;
        }
        return;
    }
    std::cerr << "FAIL: expected a throw mentioning '" << needle << "'\n";
    ++failures;
}

// A backend that allocates real host memory, counts every primitive, and performs
// the "H2D" as a memcpy so the test can inspect what was actually transferred.
class FakeDevice final : public p::WeightResidencyDevice {
public:
    void* device_alloc(std::uint64_t bytes) override {
        ++device_allocs;
        device_bytes += bytes;
        return std::malloc(static_cast<std::size_t>(bytes));
    }
    void device_free(void* slot) noexcept override {
        if (slot != nullptr) { std::free(slot); }
        ++device_frees;
    }
    void* pinned_alloc(std::uint64_t bytes) override {
        ++pinned_allocs;
        pinned_bytes += bytes;
        return std::malloc(static_cast<std::size_t>(bytes));
    }
    void pinned_free(void* pinned) noexcept override {
        if (pinned != nullptr) { std::free(pinned); }
    }
    void enqueue_h2d(void* device_slot, const void* pinned, std::uint64_t bytes) override {
        ++copies;
        bytes_copied += bytes;
        std::memcpy(device_slot, pinned, static_cast<std::size_t>(bytes));
    }
    void synchronize() override { ++syncs; }
    std::uint64_t now_ns() const noexcept override { return clock; }

    std::uint64_t clock        = 0;
    std::uint64_t device_allocs = 0;
    std::uint64_t device_frees  = 0;
    std::uint64_t device_bytes  = 0;
    std::uint64_t pinned_allocs = 0;
    std::uint64_t pinned_bytes  = 0;
    std::uint64_t copies        = 0;
    std::uint64_t bytes_copied  = 0;
    std::uint64_t syncs         = 0;
};

// A synthetic artifact: `layers` layers, two 1 KiB spans each, plus one tiny
// layer-scoped norm and one object that carries no layer at all.
std::vector<p::WeightSpanSource> make_sources(std::uint32_t layers,
                                              std::uint64_t span_bytes) {
    std::vector<p::WeightSpanSource> sources;
    std::uint64_t offset = 0;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        for (const char* name : {"attention/query_key", "mlp/down"}) {
            sources.push_back(p::WeightSpanSource{prefix + name, offset, span_bytes});
            offset += span_bytes;
        }
        sources.push_back(p::WeightSpanSource{prefix + "input_norm", offset, 64});
        offset += 64;
    }
    sources.push_back(p::WeightSpanSource{"text/final_norm", offset, 64});
    return sources;
}

p::WeightOffloadLimits limits_for(std::uint64_t host_bytes, std::uint32_t prefetch) {
    p::WeightOffloadLimits limits;
    limits.host_pinned_bytes = host_bytes;
    limits.min_span_bytes    = 100;
    limits.prefetch_layers   = prefetch;
    // The FakeDevice-backed schedule DOES enter every offloaded layer through
    // note_layer() on every pass, so these fixtures assert the residency contract
    // the plan builder now requires. The silent caller is the case under test in
    // test_offload_requires_a_per_layer_fetch(), which sets this back to false.
    limits.fetch_per_layer_entry = true;
    return limits;
}

std::vector<std::byte> pattern(std::uint64_t bytes, std::uint8_t seed) {
    std::vector<std::byte> out(static_cast<std::size_t>(bytes));
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<std::byte>((seed + i) & 0xFF);
    }
    return out;
}

void test_location_parsing() {
    const auto dense = p::weight_span_location("text/layers/7/attention/query_key");
    check(dense.has_value() && dense->layer == 7 && dense->expert == -1,
          "the layer index is read out of the artifact name");
    const auto first = p::weight_span_location("text/layers/0/gdn/a_projection");
    check(first.has_value() && first->layer == 0, "layer 0 parses");
    const auto moe = p::weight_span_location("text/layers/3/experts/11/mlp/down");
    check(moe.has_value() && moe->layer == 3 && moe->expert == 11,
          "the expert index is read when the name carries one");
    check(!p::weight_span_location("text/token_embedding").has_value(),
          "the token embedding has no layer");
    check(!p::weight_span_location("text/final_norm").has_value(),
          "the final norm has no layer");
    check(!p::weight_span_location("vision/merger/fc2").has_value(),
          "the vision tower has no layer");
    check(!p::weight_span_location("text/layers/x/input_norm").has_value(),
          "a non-numeric layer segment is rejected");
    check(!p::weight_span_location("text/layers/").has_value(),
          "an empty layer segment is rejected");
}

void test_classification_is_a_deep_tail() {
    const auto sources = make_sources(8, 1000);

    const p::WeightOffloadPlan all = p::build_weight_offload_plan(sources, limits_for(16000, 2));
    check(all.layers.size() == 8 && all.layers.front() == 0 && all.layers.back() == 7,
          "a budget covering every layer offloads every layer");
    check(all.spans.size() == 16, "the sub-floor norms stay resident");
    check(all.offloaded_bytes == 16000, "offloaded bytes count only spans above the floor");
    check(all.layer_stride == 2048, "one arena strip holds the widest layer, 256-aligned");
    check(all.arena_layers == 2, "the arena is the prefetch window, not the offloaded set");
    check(all.device_arena_bytes == 4096, "arena bytes are strips x stride");
    check(all.device_bytes_freed == 16000 - 4096, "freed bytes are offloaded minus arena");
    check(all.arena_slot_of(0) == 0 && all.arena_slot_of(1) == 1 && all.arena_slot_of(2) == 0,
          "arena slots cycle with period arena_layers, which is what keeps addresses stable");
    check(all.device_offset_of(0) == all.device_offset_of(4),
          "layer 0 span 0 and layer 2 span 0 land on the same stable address");

    const p::WeightOffloadPlan tail = p::build_weight_offload_plan(sources, limits_for(13000, 2));
    check(tail.layers.size() == 6 && tail.layers.front() == 2 && tail.layers.back() == 7,
          "a tight budget takes a contiguous DEEP tail, never a scatter");
    check(tail.offloaded_bytes == 12000, "the tail's byte count is exact");
    check(tail.device_bytes_freed == 12000 - 4096, "and so are the bytes it frees");
}

void test_rejections_are_loud(const std::vector<p::WeightSpanSource>& sources) {
    p::WeightOffloadLimits off;
    off.host_pinned_bytes = 0;
    check(p::build_weight_offload_plan(sources, off).empty(),
          "a zero budget is off, not an error");

    rejects("does not admit a single layer group",
            [&] { (void)p::build_weight_offload_plan(sources, limits_for(100, 2)); });
    // Four layers of 2000 B with a two-strip arena: the offload would move bytes but
    // free none, which is a promise the design refuses to make silently.
    rejects("would free no device memory",
            [&] { (void)p::build_weight_offload_plan(sources, limits_for(4000, 2)); });
    rejects("would free no device memory",
            [&] { (void)p::build_weight_offload_plan(sources, limits_for(4000, 8)); });

    p::WeightOffloadLimits deep = limits_for(100000, 1);
    rejects("prefetch depth below 2",
            [&] { (void)p::build_weight_offload_plan(sources, deep); });

    p::WeightOffloadLimits floored = limits_for(1000000, 2);
    floored.min_span_bytes         = 1u << 30;
    rejects("no layer-scoped weight object reached the span floor",
            [&] { (void)p::build_weight_offload_plan(sources, floored); });

    p::WeightOffloadLimits narrow = limits_for(100000, 2);
    narrow.device_arena_bytes     = 1;
    rejects("cannot hold one layer strip",
            [&] { (void)p::build_weight_offload_plan(sources, narrow); });

    // An operator-set arena big enough for every offloaded layer is the same
    // contradiction reached from the other side.
    p::WeightOffloadLimits wide = limits_for(8000, 4);
    wide.device_arena_bytes     = 8192;
    rejects("would free no device memory",
            [&] { (void)p::build_weight_offload_plan(sources, wide); });
}

void test_runtime_schedule_mirror_and_eviction() {
    const auto sources = make_sources(8, 1000);
    p::WeightOffloadPlan plan = p::build_weight_offload_plan(sources, limits_for(16000, 2));

    FakeDevice device;
    p::WeightResidencyRuntime runtime(plan, &device);
    check(runtime.enabled(), "the runtime is enabled once something is offloaded");
    check(device.device_allocs == 1 && device.device_bytes == plan.device_arena_bytes,
          "exactly one device arena is allocated, sized to the working set");
    check(runtime.pinned_bytes() == plan.offloaded_bytes,
          "the pinned mirror is the offloaded bytes: that is the host RAM bill");

    std::vector<std::vector<std::byte>> mirrors;
    mirrors.reserve(plan.spans.size());
    for (std::size_t s = 0; s < plan.spans.size(); ++s) {
        mirrors.push_back(pattern(plan.spans[s].bytes, static_cast<std::uint8_t>(s * 7 + 1)));
        void* address = runtime.adopt_span(s, mirrors[s].data());
        check(address == runtime.device_address(s),
              "adopt_span returns the same STABLE address device_address reports");
    }
    check(device.pinned_bytes == plan.offloaded_bytes, "the backend saw the same pinned bytes");

    const void* const address_of_first = runtime.device_address(0);
    check(address_of_first == runtime.device_address(4),
          "layers arena_layers apart share a slot: that IS the sliding window");

    // One pass over the layer order. Each layer is byte-checked the moment it is
    // entered: the arena only holds the last `arena_layers` strips, so a span is
    // only required to be correct while its own layer is being computed.
    std::uint64_t byte_checked = 0;
    for (std::uint32_t layer = 0; layer < 8; ++layer) {
        runtime.note_layer(layer);
        for (std::size_t s = 0; s < plan.spans.size(); ++s) {
            if (plan.spans[s].layer != layer) { continue; }
            const auto* on_device = static_cast<const std::byte*>(runtime.device_address(s));
            check(std::memcmp(on_device, mirrors[s].data(),
                              static_cast<std::size_t>(plan.spans[s].bytes)) == 0,
                  "the layer being computed is a byte-exact mirror of its source");
            ++byte_checked;
        }
    }
    check(byte_checked == plan.spans.size(),
          "every offloaded span was byte-checked while its layer was live");
    const p::WeightResidencyCounters round1 = runtime.counters();
    check(round1.layers_entered == 8, "every layer entry is counted");
    check(round1.faults == 1, "only the first layer faults; the rest were prefetched");
    check(round1.prefetch_hits == 7, "seven layer entries were served by a prefetch");
    check(round1.prefetches_issued == 8, "every offloaded layer is fetched exactly once");
    check(round1.h2d_bytes == plan.offloaded_bytes, "one round moves the offloaded bytes once");
    check(device.bytes_copied == plan.offloaded_bytes, "the backend saw exactly that traffic");

    const std::uint64_t copies_before = device.copies;
    runtime.note_layer(99);
    check(device.copies == copies_before, "entering a non-offloaded layer issues no transfer");

    // A second pass must look exactly like the first. The cyclic arena evicted the
    // layers the first pass had overwritten, so nothing is treated as stale-resident
    // and nothing is skipped -- and the same layers come back byte-correct.
    std::uint64_t rechecked = 0;
    for (std::uint32_t layer = 0; layer < 8; ++layer) {
        runtime.note_layer(layer);
        for (std::size_t s = 0; s < plan.spans.size(); ++s) {
            if (plan.spans[s].layer != layer) { continue; }
            const auto* on_device = static_cast<const std::byte*>(runtime.device_address(s));
            check(std::memcmp(on_device, mirrors[s].data(),
                              static_cast<std::size_t>(plan.spans[s].bytes)) == 0,
                  "an evicted-and-refetched layer reads back byte-exactly");
            ++rechecked;
        }
    }
    check(rechecked == plan.spans.size(), "every span was rechecked in the second pass");
    check(runtime.counters().faults == 2 && runtime.counters().prefetch_hits == 14,
          "the second pass has exactly the first pass's shape");
    check(device.copies == 2 * plan.spans.size(), "two passes, one copy per span per pass");
    check(runtime.device_address(0) == address_of_first,
          "the address of layer 0 is unchanged after two full passes");
}

// W13 x NVFP4 -- the `algn=0/0` failure. Every offloaded span must START on an
// aligned address inside its arena strip, not just the strip base.
//
// Mechanism, all read off the real code: build_weight_offload_plan() packs the
// spans of one layer back to back into ONE strip; device_offset_of() hands the
// GEMM `arena + slot*layer_stride + layer_offsets[i]`; the resident path gets the
// same guarantee from Binder::materialize_on_device -> align_up(capacity,
// tensor_alignment(layout)), and tensor_alignment() is 256 for every layout. An
// NVFP4 payload is align_up(n*k/2,256) + n*k/16 + sizeof(float) == 4 (mod 16) --
// always -- so a back-to-back pack puts the SECOND NVFP4 span of a strip at
// 4 (mod 16) and validate_nvfp4_weight (src/ops/linear/nvfp4/nvfp4_format.cpp)
// refuses it: "[nvfp4] invalid weight ... algn=0/0". Only the offload path can
// produce that refusal, which is exactly what the engine printed.
//
// The fixture uses the real failing geometry (n=5120 k=17408 -> 50,135,044 B) and
// the test FIRST proves the fixture is a misalignment case by re-deriving the
// naive offsets -- otherwise it would pass with the bug still present.
void test_span_starts_are_strip_aligned() {
    constexpr std::uint64_t kNvfp4Down = 50'135'044; // n=5120 k=17408, == 4 (mod 16)
    constexpr std::uint64_t kFp8Sized  = 8'388'608;  // 16 | size: never misaligned
    std::vector<p::WeightSpanSource> sources;
    std::uint64_t offset = 0;
    for (std::uint32_t layer = 0; layer < 6; ++layer) {
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        for (const char* name : {"mlp/gate_up", "mlp/down"}) {
            sources.push_back(p::WeightSpanSource{prefix + name, offset, kNvfp4Down});
            offset += kNvfp4Down;
        }
        sources.push_back(p::WeightSpanSource{prefix + "mlp/o_proj", offset, kFp8Sized});
        offset += kFp8Sized;
    }
    const p::WeightOffloadPlan plan =
        p::build_weight_offload_plan(sources, limits_for(1ULL << 32, 2));
    check(plan.spans.size() == 18, "every above-floor span of every layer is offloaded");
    check(plan.offloaded_bytes == 6 * (2 * kNvfp4Down + kFp8Sized),
          "offloaded_bytes is the pure sum of span bytes: alignment padding is not billed to "
          "the pinned mirror");

    // (a) CONTROL ON THE FIXTURE -- the naive packing this test exists to catch.
    std::unordered_map<std::uint32_t, std::uint64_t> naive;
    std::size_t naive_misaligned               = 0;
    std::uint64_t first_naive_misaligned_bytes = 0;
    for (std::size_t s = 0; s < plan.spans.size(); ++s) {
        std::uint64_t& cursor = naive[plan.spans[s].layer];
        if (cursor % 16 != 0) {
            if (naive_misaligned == 0) { first_naive_misaligned_bytes = plan.spans[s].bytes; }
            ++naive_misaligned;
        }
        cursor += plan.spans[s].bytes;
    }
    check(naive_misaligned == 12,
          "the fixture IS a misalignment case: 12 of its 18 naive offsets are not 16-aligned");
    check(first_naive_misaligned_bytes == kNvfp4Down,
          "and the first misaligned one is an NVFP4 `mlp/down`, as in the engine's report");

    // (b) THE REQUIREMENT -- drop the align_up() in the plan builder and this fails.
    for (std::size_t s = 0; s < plan.spans.size(); ++s) {
        check(plan.layer_offsets[s] % 256 == 0,
              "every offloaded span starts on a 256-byte boundary (== tensor_alignment)");
        check(plan.layer_offsets[s] + plan.spans[s].bytes <= plan.layer_stride,
              "every offloaded span fits inside its own arena strip");
    }
    check(plan.layer_offsets[1] == 50'135'296,
          "the second span of layer 0 starts at the next 256-boundary, not at 50,135,044");
    check(plan.layer_stride == 108'659'200,
          "the strip is sized from the ALIGNED cursor, so each span still fits");
}

// W13 x --vision: THE LAYER KEY IS ROOT-QUALIFIED -- the guard for B13.
//
// weight_span_location() reads a layer index only under the `text/` root. Every other
// user of the "/layers/" spelling -- the vision tower's vision/layers/N/... and the
// draft heads' dflash{2,}/layers/N/... -- belongs to no layer, exactly as the contract
// above weight_span_location() says, and stays resident instead of being grouped into a
// text layer's offload strip. This test is the pin INVERTED: it is RED on a tree without
// the guard and GREEN with it.
//
// Reachability, before the guard: the selected set is always a DEEPEST-FIRST contiguous
// suffix and vision only touches layers 0..26, so the collision was NOT reachable at the
// budget that produced the `algn=0/0` failure (suffix 52..63, 4,094,521,376 B). It WAS
// reachable at any budget that reaches layer <= 26 -- the cheapest such budget offloads
// 10,934,961,392 B (span floor in [2,679,744, 3,290,112]) of pinned host memory -- and
// the span floor is not a guard: its default is derived (binder.cpp: max(1 MiB,
// device_capacity_bytes / 4096)), so any arena below 4 GiB drops it to 1 MiB and the
// 2,679,744 B vision spans pass it, as does any --weight-span-floor-bytes <= 2,679,744.
void test_vision_layer_keys_are_root_qualified() {
    const auto vision = p::weight_span_location("vision/layers/3/attention/qkv");
    check(!vision.has_value(),
          "a vision object has no layer key: vision/layers/3 is not text layer 3");

    constexpr std::uint64_t kMiB = 1u << 20;
    std::vector<p::WeightSpanSource> sources;
    for (std::uint32_t layer = 0; layer < 4; ++layer) {
        sources.push_back(p::WeightSpanSource{
            "text/layers/" + std::to_string(layer) + "/mlp/down",
            static_cast<std::uint64_t>(layer) * kMiB, kMiB});
    }
    // The one vision span. It is not a text layer's span, so it stays resident.
    sources.push_back(p::WeightSpanSource{"vision/layers/0/attention/qkv", 4 * kMiB, kMiB});

    const p::WeightOffloadPlan plan =
        p::build_weight_offload_plan(sources, limits_for(64 * kMiB, 2));
    check(plan.layers.size() == 4 && plan.layers.front() == 0 && plan.layers.back() == 3,
          "four layer groups are selected, none of them a 'vision' one");
    check(plan.spans.size() == 4,
          "one span per selected text layer: the vision span is not a layer-0 span");
    check(plan.spans[0].layer == 0 && plan.spans[1].layer == 1,
          "each text layer keeps exactly its own span -- the vision span is in neither");
    check(plan.offloaded_bytes == 4 * kMiB,
          "the vision bytes are not billed to a text layer's offload budget");
    check(plan.layer_stride == kMiB,
          "no text layer's strip is widened by the vision span");

    // The control that makes the above mean something: the guard is the ROOT, not "has
    // a slash" -- a text object still parses, and a layer-less vision object still does
    // not. This is the control the pin called for and the guard must not overshoot.
    const auto text = p::weight_span_location("text/layers/0/mlp/down");
    check(text.has_value() && text->layer == 0,
          "control: a text layer object still parses to its own layer");
    check(!p::weight_span_location("vision/merger/fc2").has_value(),
          "control: a non-layer-scoped vision object has no layer key");
    check(!p::weight_span_location("vision/patch_embedding").has_value(),
          "control: and neither does the patch embedding");
}

// W13 -- THE RESIDENCY CONTRACT, AND THE REFUSAL THAT MAKES ITS ABSENCE LOUD.
//
// The arena is cyclic and strictly smaller than the offloaded set, so at any instant
// some offloaded layers' strips belong to other layers. The mechanism is transparent
// ONLY because note_layer() is entered at every layer of every pass. This test is the
// pin INVERTED: it DEMONSTRATES the stale read first (so a tree that stopped refusing
// could not pass it), then requires the plan builder to refuse a caller that does not
// promise the per-layer fetch.
void test_offload_requires_a_per_layer_fetch() {
    const auto sources        = make_sources(8, 1000);
    p::WeightOffloadPlan plan = p::build_weight_offload_plan(sources, limits_for(16000, 2));

    FakeDevice device;
    p::WeightResidencyRuntime runtime(plan, &device);
    std::vector<std::vector<std::byte>> mirrors;
    for (std::size_t s = 0; s < plan.spans.size(); ++s) {
        mirrors.push_back(pattern(plan.spans[s].bytes, static_cast<std::uint8_t>(s * 7 + 1)));
        (void)runtime.adopt_span(s, mirrors[s].data());
    }

    // The pass the engine actually runs today: every layer enters through the hook.
    for (std::uint32_t layer = 0; layer < 8; ++layer) { runtime.note_layer(layer); }

    // The pass the engine runs NEXT when the hook is prefill-only: nothing is entered,
    // and every layer whose strip was overwritten is now holding another layer's bytes.
    std::size_t stale_layers = 0;
    for (std::uint32_t layer = 0; layer < 8; ++layer) {
        for (std::size_t s = 0; s < plan.spans.size(); ++s) {
            if (plan.spans[s].layer != layer) { continue; }
            const auto* on_device = static_cast<const std::byte*>(runtime.device_address(s));
            if (std::memcmp(on_device, mirrors[s].data(),
                            static_cast<std::size_t>(plan.spans[s].bytes)) != 0) {
                ++stale_layers;
                break;
            }
        }
    }
    check(stale_layers == 6,
          "CONTROL: a 2-strip arena over 8 offloaded layers leaves 6 layers reading a strip "
          "that belongs to another layer when the hook is not entered -- the measured engine "
          "failure (64 ids of 0 at rc=0) in miniature");

    // ...which is why the plan refuses a caller that does not promise the fetch.
    p::WeightOffloadLimits silent = limits_for(16000, 2);
    silent.fetch_per_layer_entry  = false;
    rejects("cannot be numerically transparent",
            [&] { (void)p::build_weight_offload_plan(sources, silent); });
    // The refusal must NAME what has no home: the layer range and the shortfall.
    rejects("L0..L7", [&] { (void)p::build_weight_offload_plan(sources, silent); });
    rejects("6 of them have no strip of their own",
            [&] { (void)p::build_weight_offload_plan(sources, silent); });

    // Controls: the contract is the only thing being asked for, and "off" is still off.
    check(!p::build_weight_offload_plan(sources, limits_for(16000, 2)).empty(),
          "control: the identical limits are admitted once the caller asserts the per-layer "
          "fetch, so the refusal is about the contract and not about the bytes");
    p::WeightOffloadLimits off     = limits_for(0, 2);
    off.fetch_per_layer_entry      = false;
    check(p::build_weight_offload_plan(sources, off).empty(),
          "control: a zero budget is OFF (empty plan, no refusal) whatever the contract says");
}

void test_kv_capacity_fit() {
    // The engine's own two reported 1M points: rk4v4 (4.25 bits/element) and nvfp4
    // (4.50). Two observations, one exact affine solve -- nothing typed in beyond
    // the observations themselves.
    constexpr std::uint64_t kOneMillion = 1'048'576;
    const p::KvCapacityPoint rk4v4{kOneMillion, 18'912'736'512ULL, 4.25};
    const p::KvCapacityPoint nvfp4{kOneMillion, 20'000'740'608ULL, 4.50};
    const p::KvCapacityFit fit = p::kv_capacity_fit_from(rk4v4, nvfp4);

    const double back_rk4v4 = fit.bytes_for(kOneMillion, 4.25);
    check(back_rk4v4 > 18912736512.0 - 2.0 && back_rk4v4 < 18912736512.0 + 2.0,
          "the fit reproduces the engine's rk4v4 1M requirement");
    const double back_nvfp4 = fit.bytes_for(kOneMillion, 4.50);
    check(back_nvfp4 > 20000740608.0 - 2.0 && back_nvfp4 < 20000740608.0 + 2.0,
          "the fit reproduces the engine's nvfp4 1M requirement");
    const double bits_now = fit.bits_within(kOneMillion, 11'600'323'584ULL);
    check(bits_now > 2.55 && bits_now < 2.58,
          "the reported 11.6 GB gives about 2.57 bits/element at 1M");

    rejects("two different bit budgets", [&] { (void)p::kv_capacity_fit_from(rk4v4, rk4v4); });
    rejects("non-zero token count",
            [&] { (void)p::kv_capacity_fit_from(p::KvCapacityPoint{0, 1, 1.0}, nvfp4); });

    // What W13 has to buy for 1M: the bytes that put nvfp4's 4.50 inside the budget.
    const std::uint64_t needed = 20'000'740'608ULL - 11'600'323'584ULL;
    const p::WeightOffloadKvHeadroom head =
        p::weight_offload_kv_headroom(fit, kOneMillion, 11'600'323'584ULL, needed);
    check(head.bits_after > 4.49 && head.bits_after < 4.51,
          "freeing 8.4 GB puts nvfp4 (4.50 bits/element) inside the budget at 1M");
    check(head.bits_gained > 1.9, "and it is worth about two whole bits per element");
    check(head.describe().find("bits/el") != std::string::npos, "the headroom line is printable");

    // rk4v4 is cheaper on the ladder and is NOT usable: the four-arm exact-prefix
    // measurement put pure-rk4v4 KV at 0.020, i.e. random. The fit is a capacity model
    // only and says nothing about that -- so the tier to pay for is nvfp4.
    const std::uint64_t rk4v4_needed = 18'912'736'512ULL - 11'600'323'584ULL;
    const p::WeightOffloadKvHeadroom rk4v4_head =
        p::weight_offload_kv_headroom(fit, kOneMillion, 11'600'323'584ULL, rk4v4_needed);
    check(rk4v4_head.bits_after > 4.24 && rk4v4_head.bits_after < 4.26,
          "freeing 6.8 GB would fit rk4v4's 4.25 bits/element, which is not worth doing");
    check(rk4v4_head.bits_gained < head.bits_gained, "and it buys strictly less than nvfp4 does");
}

void test_offload_verdict() {
    // Dense decode: 6 GiB of weights per round across PCIe against a 20 ms round.
    const p::WeightOffloadVerdict dense =
        p::weight_offload_verdict(6.0 * 1024 * 1024 * 1024, 25e9, 1792e9, 0.020);
    check(!dense.net_positive, "GBs per decode token is not net-positive");
    check(dense.slowdown_ratio > 10.0, "and the verdict says how bad, not just that it is bad");
    check(dense.pcie_seconds_per_round > 0.2 && dense.pcie_seconds_per_round < 0.3,
          "6 GiB at 25 GB/s is about a quarter of a second");

    // Prefill: the same bytes amortized over a 4096-token chunk are essentially free.
    const p::WeightOffloadVerdict prefill =
        p::weight_offload_verdict(6.0 * 1024 * 1024 * 1024 / 4096, 25e9, 1792e9, 0.020);
    check(prefill.net_positive, "the same offload IS net-positive once it is amortized");
    check(prefill.slowdown_ratio < 1.01, "and the slowdown is under a percent");

    rejects("positive bandwidths", [&] { (void)p::weight_offload_verdict(1.0, 0.0, 1.0, 1.0); });
}

} // namespace

int main() {
    test_location_parsing();
    test_classification_is_a_deep_tail();
    test_rejections_are_loud(make_sources(4, 1000));
    test_runtime_schedule_mirror_and_eviction();
    test_span_starts_are_strip_aligned();
    test_vision_layer_keys_are_root_qualified();
    test_offload_requires_a_per_layer_fetch();
    test_kv_capacity_fit();
    test_offload_verdict();
    if (failures != 0) {
        std::cerr << failures << " of " << checks_run << " check(s) failed\n";
        return 1;
    }
    std::cout << "test_weight_residency: all checks passed (n=" << checks_run << ")\n";
    return 0;
}
