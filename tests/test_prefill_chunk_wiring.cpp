// W6 wiring test: the prefill-chunk handle the engine's bandwidth governor needs from a target
// Program. Two halves, because the failure this exists to catch has two shapes:
//
//   * the members must EXIST with the exact shape the engine's guard and its clamp contract rely
//     on -- a compile-time assertion, so losing them is a build failure;
//   * the members must be DEFINED -- a member-function pointer is an odr-use, so a declaration
//     kept without a definition is a link error rather than a silently discarded call.
//
// Why a dedicated test. engine_core.h installs the governor's shrunken prefill unit behind
// `if constexpr (bandwidth_detail::exposes_prefill_chunk_wire<Program>)`. Before
// ninfer/targets/qwen3_6/runtime.h and impl/runtime/api_impl.h grew the two forwarders that
// predicate was false for EVERY target, so the whole block -- the capacity readback, the
// set_prefill_chunk() call and the trace -- compiled to nothing, while the engine still built,
// linked and passed the entire suite. tests/test_bandwidth_governor.cpp cannot see that: it tests
// BandwidthGovernor itself, and its `prefill_chunk_for(3072) < 3072` line is satisfied by the class
// even when nothing on earth calls it. That is the "the self-test shares the wrong assumption with
// the object under test" failure mode; the only check that closes it is one that (a) names the wire
// and (b) odr-uses it.
//
// The predicate asserted here is the SAME named definition engine_core.h uses, so this test cannot
// drift into asserting a private copy of the guard. See REPORT.md for the measured negative
// controls: with the two forwarders commented out this TU does not compile and (with the loud else
// branch) src/runtime/engine/engine.cpp does not compile either, while the pristine engine_core.h
// compiled fine in exactly that state -- which is the silent hole this closes.
#include "runtime/engine/bandwidth_governor.h"

#include <ninfer/targets/muse_glimmer_30b/package.h>
#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6_35b_a3b/package.h>

#include <concepts>
#include <cstdint>
#include <iostream>

namespace {

using ninfer::runtime::BandwidthGovernor;
using ninfer::runtime::bandwidth_detail::exposes_prefill_chunk_wire;

// The three packages the engine instantiates EngineCore over (src/runtime/engine/engine.cpp
// :243-245) -- the same three whose Program specializations carry the forwarders.
using Dense27 = ninfer::targets::qwen3_6_27b::Package;
using Moe35   = ninfer::targets::qwen3_6_35b_a3b::Package;
using Muse30  = ninfer::targets::muse_glimmer_30b::Package;

// The engine's own predicate, for each target.
template <class Package>
constexpr bool wired() {
    return exposes_prefill_chunk_wire<typename Package::Program>;
}

// Split out of the predicate above so a failure names the missing half instead of the whole wire.
template <class Package>
constexpr bool exposes_setter() {
    return requires(typename Package::Program& program, std::uint32_t chunk) {
        program.set_prefill_chunk(chunk);
    };
}

template <class Package>
constexpr bool exposes_capacity() {
    return requires(typename Package::Program& program) {
        { program.prefill_chunk_capacity() } noexcept -> std::same_as<std::uint32_t>;
    };
}

static_assert(wired<Dense27>(),
              "qwen3_6_27b::Program lost the prefill-chunk wire: engine_core.h's governor block "
              "(capacity readback + set_prefill_chunk + trace) compiles to nothing again");
static_assert(wired<Moe35>(),
              "qwen3_6_35b_a3b::Program lost the prefill-chunk wire: engine_core.h's governor "
              "block (capacity readback + set_prefill_chunk + trace) compiles to nothing again");
static_assert(wired<Muse30>(),
              "muse_glimmer_30b::Program lost the prefill-chunk wire: engine_core.h's governor "
              "block (capacity readback + set_prefill_chunk + trace) compiles to nothing again");

static_assert(exposes_setter<Dense27>() && exposes_setter<Moe35>() && exposes_setter<Muse30>(),
              "a Program lost set_prefill_chunk(std::uint32_t)");
static_assert(exposes_capacity<Dense27>() && exposes_capacity<Moe35>() && exposes_capacity<Muse30>(),
              "a Program lost prefill_chunk_capacity(), or its signature/exception specification "
              "drifted from `std::uint32_t () const noexcept`");

// The exact call the engine makes, spelled the way engine_core.h spells it.
static_assert(requires(Dense27::Program& program) { program.set_prefill_chunk(0u); },
              "the engine's `requires { program->set_prefill_chunk(0u); }` guard is false for "
              "qwen3_6_27b");

// odr-use: with the runtime.h declaration kept and the api_impl.h definition deleted, these are
// undefined references, so "declared but no longer forwards" cannot pass by static_assert alone.
// One pointer per (package, member): the three Program types are distinct specializations, so
// there is no single pointer type across them.
//
// `volatile` is load-bearing, not decoration. A plain `const <mfp> k = &F;` that is only compared
// against nullptr is folded away at -O1: measured, with the api_impl.h definition renamed the
// object file exported NO reference to set_prefill_chunk, so the "link-time half" was vacuous. A
// volatile object must have its value in memory, which keeps the relocation that names the
// function -- and that relocation is the check.
template <class Package>
using ProgramOf = typename Package::Program;

template <class Package>
using SetPrefillChunk = void (ProgramOf<Package>::*)(std::uint32_t) noexcept;

template <class Package>
using PrefillCapacity = std::uint32_t (ProgramOf<Package>::*)() const noexcept;

volatile SetPrefillChunk<Dense27> kSetChunkDense27 = &Dense27::Program::set_prefill_chunk;
volatile SetPrefillChunk<Moe35> kSetChunkMoe35     = &Moe35::Program::set_prefill_chunk;
volatile SetPrefillChunk<Muse30> kSetChunkMuse30   = &Muse30::Program::set_prefill_chunk;
volatile PrefillCapacity<Dense27> kCapacityDense27 = &Dense27::Program::prefill_chunk_capacity;
volatile PrefillCapacity<Moe35> kCapacityMoe35     = &Moe35::Program::prefill_chunk_capacity;
volatile PrefillCapacity<Muse30> kCapacityMuse30   = &Muse30::Program::prefill_chunk_capacity;

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

// Synthetic observation window, same shape as tests/test_bandwidth_governor.cpp: the governor
// reads engine counters, so a window is "so many decode tokens/rounds plus some prefill work".
struct Window {
    double us_per_token          = 1.0;
    std::uint64_t tokens         = 100;
    std::uint64_t rounds         = 100;
    std::uint64_t prefill_units  = 50;
};

class Feed {
public:
    Feed(BandwidthGovernor::Tuning tuning, bool enabled)
        : governor_(tuning, enabled), window_ns_(tuning.window_ns) {}

    void step(const Window& window) {
        counters_.decode_device_ns += static_cast<std::uint64_t>(
            window.us_per_token * 1000.0 * static_cast<double>(window.tokens));
        counters_.decode_tokens += window.tokens;
        counters_.decode_rounds += window.rounds;
        counters_.prefill_device_ns += window.prefill_units * 2'000'000ULL;
        counters_.prefill_units += window.prefill_units;
        now_ns_ += window_ns_;
        governor_.observe(now_ns_, counters_);
    }

    [[nodiscard]] BandwidthGovernor& governor() noexcept { return governor_; }

private:
    BandwidthGovernor governor_;
    BandwidthGovernor::Counters counters_{};
    std::uint64_t window_ns_ = 50'000'000ULL;
    std::uint64_t now_ns_    = 1'000'000'000ULL;
};

// One second-scale window, so the runtime half finishes in milliseconds.
BandwidthGovernor::Tuning millisecond_tuning() {
    BandwidthGovernor::Tuning tuning;
    tuning.window_ns = 1'000'000ULL;
    return tuning;
}

void drive_contention(Feed& feed) {
    Window calm;
    calm.us_per_token = 1.0;
    for (int i = 0; i < 6; ++i) { feed.step(calm); }
    Window contended;
    contended.us_per_token = 2.0;
    for (int i = 0; i < 12; ++i) { feed.step(contended); }
}

// The two pointers must be non-null: this is also what keeps the odr-use alive at -O3.
void test_symbols_resolve() {
    expect(kSetChunkDense27 != nullptr && kSetChunkMoe35 != nullptr && kSetChunkMuse30 != nullptr,
           "a Program::set_prefill_chunk member-function pointer did not resolve");
    expect(kCapacityDense27 != nullptr && kCapacityMoe35 != nullptr && kCapacityMuse30 != nullptr,
           "a Program::prefill_chunk_capacity member-function pointer did not resolve");
}
// The contract engine_core.h leans on: while unthrottled the installed unit IS the capacity, in
// both modes. That identity is what makes the trace's `installed=1 chunk=<capacity>` line the
// same value the run was already using.
void test_unthrottled_identity() {
    const BandwidthGovernor::Tuning tuning = millisecond_tuning();
    const std::uint32_t capacities[]       = {128u, 384u, 3072u, 4096u};

    BandwidthGovernor fresh(tuning, true);
    BandwidthGovernor off(tuning, false);
    for (const std::uint32_t capacity : capacities) {
        expect(fresh.prefill_chunk_for(capacity) == capacity,
               "a fresh enabled governor changed the prefill unit before any evidence existed");
        expect(off.prefill_chunk_for(capacity) == capacity,
               "a disabled governor changed the prefill unit");
    }
}

// Throttling shrinks the unit into [128, capacity] on the 128-token prefill alignment -- the
// range program.h's set_prefill_chunk re-clamps into.
void test_throttled_shrink() {
    const BandwidthGovernor::Tuning tuning = millisecond_tuning();
    Feed feed(tuning, true);
    drive_contention(feed);
    const std::uint32_t unit = feed.governor().prefill_chunk_for(3072);
    expect(feed.governor().snapshot().share < 1.0,
           "the contention sequence did not throttle, so the shrink below was not exercised");
    expect(unit < 3072, "a throttled governor did not shrink the prefill unit");
    expect(unit % 128 == 0, "the shrunk prefill unit is not 128-aligned");
    expect(unit >= 128, "the shrunk prefill unit fell below the prefill alignment");
    expect(feed.governor().prefill_chunk_for(128) == 128,
           "a unit already at the alignment floor was scaled below it");
    expect(feed.governor().prefill_chunk_for(4096) <= 4096,
           "the shrunk prefill unit exceeded the capacity it was given");
}

// Negative control: remove the rule (the disturbance band) and the SAME window sequence leaves the
// unit at the capacity. Without this the shrink assertions above would pass for a governor that
// shrank unconditionally.
void test_control_rule_removed() {
    BandwidthGovernor::Tuning no_rule = millisecond_tuning();
    no_rule.tol_hi                    = 1.0e9;
    Feed control(no_rule, true);
    drive_contention(control);
    expect(control.governor().prefill_chunk_for(3072) == 3072,
           "control: with the throttle rule removed the prefill unit shrank anyway, so the shrink "
           "checks are not evidence about the rule");
}

} // namespace

int main() {
    test_symbols_resolve();
    test_unthrottled_identity();
    test_throttled_shrink();
    test_control_rule_removed();
    if (failures != 0) {
        std::cerr << failures << " prefill-chunk wiring checks failed\n";
        return 1;
    }
    std::cout << "prefill-chunk wiring checks passed\n";
    return 0;
}
