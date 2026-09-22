// The prefill unit's TWO modes (PrefillChunkMode, include/ninfer/types.h): `dynamic` is the
// bandwidth governor owning the unit, `manual` is the unit pinned to the caller's value with the
// governor disabled. The user's ruling was that both must exist -- "prefill chunk 要能动态调整或者
// 手动调整两种模式" -- so this test exists for the three ways that can be false while everything
// still builds and runs:
//
//   * the mode vocabulary exists but the resolution ignores the override, so --prefill-chunk-mode
//     manual is accepted and then thrown away (the "parses but changes nothing" failure mode this
//     project has paid for repeatedly);
//   * "manual" is implemented as "enabled but never throttles", which still lets a *later* share
//     change move the unit -- manual has to be the identity for EVERY base, not just the unthrottled
//     one;
//   * the manual value is masked by the default 3072 (engine.cpp:57 records that exact trap on the
//     scoring path: "Overwriting every caller's value with 3072 made --prefill-chunk ... silently
//     inert here").
//
// Every pair of assertions below is two-sided: the mode that must be the identity and the mode that
// must NOT be, on the same feed, in the same run. A governor that ignored the mode would satisfy one
// half and fail the other, so this file cannot pass by asserting a code path.
//
// What this file does NOT cover (see REPORT.md): the engine-level application of the resolved mode
// (normalize_engine_options -> EngineCore's governor construction -> the trace line) is not
// reachable from here -- it needs a device and an artifact. That end is pinned by an end-to-end run
// whose stderr trace prints `mode=manual installed=0 chunk=<requested>` and, in the same
// configuration with the forced throttle band, `mode=dynamic installed=1 chunk=<shrunk>`.
#include "runtime/engine/bandwidth_governor.h"

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using ninfer::PrefillChunkMode;
using ninfer::runtime::BandwidthGovernor;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

// The generator every environment-dependent assertion needs. std::getenv is the engine's own
// reader (bandwidth_detail::env_raw), so setting the variable through the process environment is
// the same switch the operator sets -- there is no test-only override to drift from it.
void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    ::setenv(name, value, 1);
#endif
}

void unset_env(const char* name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    ::unsetenv(name);
#endif
}

// The governor is the mechanism, so "which mode is this" is asked through the same mapping the
// engine uses (adapts()), and a mode is judged by three things at once: the switch, the identity
// property, and whether a share change can move the unit.
struct Mechanism {
    bool enabled = false;
    bool identity_for_every_base = false;
    bool a_throttle_moved_it = false;
};

// Same synthetic window shape as tests/test_bandwidth_governor.cpp and
// tests/test_prefill_chunk_wiring.cpp: the governor reads engine counters, so a window is "so many
// decode tokens/rounds plus some prefill work", and the disturbance band is widened/narrowed to
// make throttling deterministic instead of timing-dependent.
struct Window {
    double us_per_token         = 1.0;
    std::uint64_t tokens        = 100;
    std::uint64_t rounds        = 100;
    std::uint64_t prefill_units = 50;
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

// The three judgements above, measured for one mode by running it against one contention feed.
// `cap` is the capacity the engine would read back from the Program
// (ProgramImplCore::prefill_chunk_capacity).
Mechanism measure(PrefillChunkMode mode, std::uint32_t cap) {
    Mechanism out;
    Feed feed(millisecond_tuning(), BandwidthGovernor::adapts(mode));
    out.enabled = feed.governor().enabled();

    // Identity for EVERY base, not just the unthrottled one: this is the half that separates
    // "manual" from "enabled but calm", and it is measured AFTER the contention so a share change
    // has had every chance to move the unit.
    drive_contention(feed);
    bool identity = true;
    for (const std::uint32_t base : {128u, 256u, 1024u, 3072u, 4096u}) {
        if (feed.governor().prefill_chunk_for(base) != base) { identity = false; }
    }
    out.identity_for_every_base = identity;
    out.a_throttle_moved_it      = feed.governor().prefill_chunk_for(cap) != cap;
    return out;
}

// ---- the vocabulary ------------------------------------------------------------------------
// One spelling table, read by all three front ends. If this drifts, a flag that the CLI accepts
// stops being a mode the mechanism knows.
void test_spellings() {
    expect(BandwidthGovernor::parse_mode("dynamic") == PrefillChunkMode::Dynamic,
           "parse_mode(\"dynamic\") is not Dynamic");
    expect(BandwidthGovernor::parse_mode("manual") == PrefillChunkMode::Manual,
           "parse_mode(\"manual\") is not Manual");
    expect(std::string(BandwidthGovernor::mode_name(PrefillChunkMode::Dynamic)) == "dynamic",
           "mode_name(Dynamic) is not \"dynamic\"");
    expect(std::string(BandwidthGovernor::mode_name(PrefillChunkMode::Manual)) == "manual",
           "mode_name(Manual) is not \"manual\"");
    // Round trip: the trace line's `mode=` field is read back by the operator as a flag value.
    expect(BandwidthGovernor::parse_mode(BandwidthGovernor::mode_name(PrefillChunkMode::Manual)) ==
               PrefillChunkMode::Manual,
           "mode_name/parse_mode do not round-trip for Manual");

    // The rejection half, without which the two positive cases above would also pass for a parser
    // that returned Manual for everything.
    bool threw = false;
    try {
        (void)BandwidthGovernor::parse_mode("Manual");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "parse_mode accepted \"Manual\" (the vocabulary is case-sensitive)");
    threw = false;
    try {
        (void)BandwidthGovernor::parse_mode("");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "parse_mode accepted the empty string");
    threw = false;
    try {
        (void)BandwidthGovernor::parse_mode("3072");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "parse_mode accepted a bare number: --prefill-chunk-mode is not --prefill-chunk");
}

// ---- the mode -> mechanism mapping ---------------------------------------------------------
void test_adapts_mapping() {
    expect(BandwidthGovernor::adapts(PrefillChunkMode::Dynamic),
           "adapts(Dynamic) is false, so dynamic mode would never install a unit");
    expect(!BandwidthGovernor::adapts(PrefillChunkMode::Manual),
           "adapts(Manual) is true, so manual mode would still let the governor write");
}

// ---- "CLI > environment > default", both directions ----------------------------------------
// The NINFER_FT_BW_GOV values are the documented ones (bandwidth_governor.h: the kill switch is
// =0, =1 forces it on, unset takes the default). Each override assertion is paired with the
// environment read it has to beat; a resolver that let the environment win in either direction
// fails exactly one of the two pairs.
void test_resolution_precedence() {
    const struct {
        const char* env;  // nullptr == unset
        const char* label;
        PrefillChunkMode expected;
    } cases[] = {
        {nullptr, "unset -> default Dynamic", PrefillChunkMode::Dynamic},
        {"1", "NINFER_FT_BW_GOV=1 -> Dynamic", PrefillChunkMode::Dynamic},
        {"0", "NINFER_FT_BW_GOV=0 -> Manual", PrefillChunkMode::Manual},
        {"false", "NINFER_FT_BW_GOV=false -> Manual", PrefillChunkMode::Manual},
        {"off", "NINFER_FT_BW_GOV=off -> Manual", PrefillChunkMode::Manual},
        {"yes", "NINFER_FT_BW_GOV=yes -> Dynamic", PrefillChunkMode::Dynamic},
    };
    for (const auto& one : cases) {
        if (one.env == nullptr) {
            unset_env("NINFER_FT_BW_GOV");
        } else {
            set_env("NINFER_FT_BW_GOV", one.env);
        }
        // Env-only resolution, and the same answer through an unset override.
        expect(BandwidthGovernor::resolve_mode() == one.expected, one.label);
        expect(BandwidthGovernor::resolve_mode(std::nullopt) == one.expected, one.label);

        // ... and now the override, which has to win in BOTH directions for every env value.
        const PrefillChunkMode opposite = one.expected == PrefillChunkMode::Dynamic
                                              ? PrefillChunkMode::Manual
                                              : PrefillChunkMode::Dynamic;
        expect(BandwidthGovernor::resolve_mode(PrefillChunkMode::Dynamic) ==
                   PrefillChunkMode::Dynamic,
               std::string("an explicit dynamic override lost to the environment (") + one.label +
                   ")");
        expect(BandwidthGovernor::resolve_mode(PrefillChunkMode::Manual) ==
                   PrefillChunkMode::Manual,
               std::string("an explicit manual override lost to the environment (") + one.label +
                   ")");
        // The override must not be a no-op that accidentally agrees: for the case where the
        // environment already says Dynamic, the Manual override must still produce Manual.
        if (opposite == PrefillChunkMode::Manual) {
            expect(!BandwidthGovernor::adapts(BandwidthGovernor::resolve_mode(PrefillChunkMode::Manual)),
                   std::string("manual override left the mechanism enabled (") + one.label + ")");
        }
    }
    unset_env("NINFER_FT_BW_GOV");
}

// ---- the two modes are different modes, on one feed -----------------------------------------
void test_two_modes_differ() {
    const std::uint32_t cap = 3072;
    const Mechanism dynamic = measure(PrefillChunkMode::Dynamic, cap);
    const Mechanism manual  = measure(PrefillChunkMode::Manual, cap);

    expect(dynamic.enabled, "dynamic mode left the governor disabled");
    expect(!manual.enabled, "manual mode left the governor enabled");
    expect(dynamic.a_throttle_moved_it,
           "dynamic mode did not move the prefill unit under contention, so its half of the "
           "comparison is vacuous");
    expect(manual.identity_for_every_base,
           "manual mode changed the prefill unit for some base: manual must be the identity for "
           "every value, not only while the share is 1.0");
    expect(dynamic.identity_for_every_base != manual.identity_for_every_base,
           "both modes behaved identically: the mode made no difference to the mechanism");
}

// ---- the trap that motivated the mode: a manual value must not be masked --------------------
// engine.cpp:57 records that a fixed 3072 made --prefill-chunk silently inert on the scoring path.
// The same shape is possible in manual mode if anything re-derives the unit from a default, so the
// test pins the identity for values that are NOT the default and are not the capacity: a caller
// who asked for 256 or 1024 must get 256 or 1024, and the default must not appear anywhere.
void test_manual_is_verbatim() {
    Feed feed(millisecond_tuning(), BandwidthGovernor::adapts(PrefillChunkMode::Manual));
    drive_contention(feed);
    for (const std::uint32_t requested : {128u, 256u, 384u, 1024u, 1664u, 3072u, 4096u}) {
        const std::uint32_t installed = feed.governor().prefill_chunk_for(requested);
        expect(installed == requested,
               "manual mode returned " + std::to_string(installed) + " for a requested " +
                   std::to_string(requested) + " (a default masked the caller's value)");
    }
    expect(feed.governor().prefill_chunk_for(1024) != 3072u,
           "manual mode replaced a requested 1024 with the 3072 default: this is the exact trap "
           "engine.cpp records for the scoring path");
    expect(feed.governor().share() == 1.0,
           "a disabled governor moved its share: manual mode is not merely 'a governor that has "
           "not throttled yet'");
    expect(feed.governor().snapshot().windows == 0,
           "a disabled governor counted observation windows: manual mode still reads the engine's "
           "bandwidth counters");
}

// ---- negative control ----------------------------------------------------------------------
// Same declaration as every case above, with ONE thing removed: the mode that is handed to the
// mechanism. If the two modes were really interchangeable -- i.e. if the assertions above were
// asserting a code path rather than a behaviour -- this control would also hold, and it must not:
// `adapts(Manual)` has to be the only thing standing between the throttled feed and a shrunk unit.
void test_control_mode_removed() {
    Feed feed(millisecond_tuning(), BandwidthGovernor::adapts(PrefillChunkMode::Dynamic));
    drive_contention(feed);
    // The contention feed above is the same one test_two_modes_differ() uses, so the control has to
    // show what that feed does when the mode does NOT stop it: the share falls to min_share (0.125,
    // bandwidth_governor.h Tuning) and 0.125 * 3072 = 384, still on the 128-token alignment.
    expect(feed.governor().prefill_chunk_for(3072) != 3072u,
           "control: with the mode forced to Dynamic the feed did not throttle, so the "
           "manual-mode identity assertions above are not evidence about the mode");
    expect(feed.governor().prefill_chunk_for(3072) == 384u,
           "control: the throttled feed shrank 3072 to " +
               std::to_string(feed.governor().prefill_chunk_for(3072)) +
               " instead of min_share * 3072 = 384, so the control does not describe the shape it "
               "claims to (if Tuning::min_share changed, this number is what has to be re-read)");
    expect(feed.governor().share() < 1.0,
           "control: the share never moved, so the two assertions above are not about throttling");
}

// ---- what a prefill unit CONSUMED, not what the engine INSTALLED --------------------------
// The mode surface above is all expressed through prefill_chunk_for(), i.e. through the value
// the engine would write. Nothing in this file, and nothing in the tree, said what the prefill
// loop then sliced with it -- so 'manual pins the unit' could hold on paper while the loop ran
// something else. note_installed_unit()/note_prefill_unit() are that pairing, and the three
// things they have to get right are asserted here: the manual unit reports the requested value
// on every full unit, the dynamic unit reports the shrunk one, and a unit can never report MORE
// than the chunk it ran under (the shape a loop takes when it stops reading the field).
void test_unit_tokens_pair_with_the_installed_chunk() {
    // Manual: the engine writes nothing, the Program holds `capacity`, and every unit is that.
    BandwidthGovernor manual(BandwidthGovernor::Tuning{}, /*enabled=*/false);
    manual.note_installed_unit(1024, 1024, /*installed=*/false);
    for (int i = 0; i < 3; ++i) { manual.note_prefill_unit(1024); }
    expect(manual.unit_tokens().units == 3,
           "manual: " + std::to_string(manual.unit_tokens().units) +
               " units were accounted, not 3");
    expect(manual.unit_tokens().tokens == 3072,
           "manual: the units summed to " + std::to_string(manual.unit_tokens().tokens) +
               " tokens, not 3 * 1024");
    expect(manual.unit_tokens().min == 1024 && manual.unit_tokens().max == 1024,
           "manual: the per-unit range was [" + std::to_string(manual.unit_tokens().min) +
               ", " + std::to_string(manual.unit_tokens().max) +
               "], so a unit did not consume the requested 1024");
    expect(manual.unit_tokens().over == 0,
           "manual: a unit was counted as consuming more than the installed chunk");

    // Dynamic after a throttle: the pairing has to follow the installed value down.
    BandwidthGovernor dynamic(BandwidthGovernor::Tuning{}, /*enabled=*/true);
    dynamic.note_installed_unit(1024, 1024, /*installed=*/true);
    dynamic.note_prefill_unit(1024);
    dynamic.note_installed_unit(128, 1024, /*installed=*/true);
    dynamic.note_prefill_unit(128);
    expect(dynamic.unit_tokens().last == 128,
           "dynamic: the unit after the throttle reported " +
               std::to_string(dynamic.unit_tokens().last) + " tokens, not the installed 128");
    expect(dynamic.unit_tokens().min == 128 && dynamic.unit_tokens().max == 1024,
           "dynamic: the per-unit range was [" + std::to_string(dynamic.unit_tokens().min) +
               ", " + std::to_string(dynamic.unit_tokens().max) +
               "], so the shrink is not visible in the readback");
    expect(dynamic.unit_tokens().over == 0,
           "dynamic: a unit consumed more than the chunk it ran under");

    // The one assertion that is about the loop rather than about the log: a unit run under a
    // 128-token chunk may not report 1024 tokens. This is the readback's own negative control --
    // it is what a loop that kept using the capacity / the caller's value / any other constant
    // would produce, and it has to be noticed rather than printed as a curious number.
    dynamic.note_installed_unit(128, 1024, /*installed=*/true);
    dynamic.note_prefill_unit(1024);
    expect(dynamic.unit_tokens().over == 1,
           "a unit consumed 1024 tokens under an installed chunk of 128 and over= stayed " +
               std::to_string(dynamic.unit_tokens().over) +
               ": the readback cannot tell an ignored install from an honoured one");
}

// ---- the two guards on the disabled path, and why only one of them is load-bearing --------
// prefill_chunk_for() and observe() EACH check `enabled_`, so deleting either check alone is
// behaviour-preserving and no assertion can tell the difference. (The line that wrote this file
// recorded exactly that: the injection removing only the guard inside prefill_chunk_for() made
// nothing red, and was filed as 'not discriminable'.)
//
// That is not an unpinned path. `share_` is the only state that can move the unit and observe()
// is its only writer, so a disabled governor's share_ stays 1.0 whatever the counters do -- and
// with share_ == 1.0 the remaining term in prefill_chunk_for() is the identity on its own. That
// is an invariant about the FEED, so it is asserted directly here: run the contention feed that
// takes an ENABLED governor to min_share against a DISABLED one and require it to move nothing.
// Removing observe()'s guard turns this red, which is the direction that matters, so the
// arrangement is pinned and prefill_chunk_for()'s copy is defence in depth.
void test_disabled_governor_ignores_a_hostile_feed() {
    Feed off(millisecond_tuning(), BandwidthGovernor::adapts(PrefillChunkMode::Manual));
    drive_contention(off);
    // Far more hostile than the shared feed, in case 'the feed was not strong enough' is the
    // explanation for a calm governor.
    Window hostile;
    hostile.us_per_token = 8.0;
    for (int i = 0; i < 64; ++i) { off.step(hostile); }

    const BandwidthGovernor::Snapshot snap = off.governor().snapshot();
    const std::string prefix = "disabled governor: ";
    expect(!off.governor().enabled(), prefix + "the manual mode left it enabled");
    expect(snap.windows == 0,
           prefix + "counted " + std::to_string(snap.windows) +
               " observation windows, so observe() no longer returns before touching share_");
    expect(snap.share == 1.0,
           prefix + "share moved to " + std::to_string(snap.share) +
               ", so the disabled path is no longer the identity for free");
    expect(snap.decode_us_per_token == 0.0,
           prefix + "smoothed the decode latency, so it is reading the engine's counters");
    expect(snap.credit == 0.0, prefix + "banked prefill credit");
    for (const std::uint32_t base : {128u, 256u, 1024u, 3072u}) {
        expect(off.governor().prefill_chunk_for(base) == base,
               prefix + "moved the unit for base " + std::to_string(base));
    }

    // Two-sided: the same feed with the flag flipped must move, or the test above would pass
    // against a feed that simply never contended.
    Feed on(millisecond_tuning(), BandwidthGovernor::adapts(PrefillChunkMode::Dynamic));
    drive_contention(on);
    for (int i = 0; i < 64; ++i) { on.step(hostile); }
    const std::string control = "control (same feed, mode dynamic): ";
    expect(on.governor().snapshot().windows > 0,
           control + "it counted no windows, so this feed is not evidence about the disabled "
               "governor either");
    expect(on.governor().share() < 1.0,
           control + "share stayed at 1.0, so the feed never throttled");
    expect(on.governor().prefill_chunk_for(1024) < 1024u,
           control + "the unit did not move, so the second guard is untested by this feed");
}

} // namespace

int main() {
    test_spellings();
    test_adapts_mapping();
    test_resolution_precedence();
    test_two_modes_differ();
    test_manual_is_verbatim();
    test_control_mode_removed();
    test_unit_tokens_pair_with_the_installed_chunk();
    test_disabled_governor_ignores_a_hostile_feed();
    if (failures != 0) {
        std::cerr << failures << " prefill-chunk mode checks failed\n";
        return 1;
    }
    std::cout << "prefill-chunk mode checks passed\n";
    return 0;
}
