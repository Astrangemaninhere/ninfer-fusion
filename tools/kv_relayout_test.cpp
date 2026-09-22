// tools/kv_relayout_test.cpp — host-only unit test for the FreeToken step-2
// decision logic (src/serve/kv_auto_relayout.cpp). No CUDA device required.
//
// Cross-checks the C++ policy against tools/archkit/ft_tiers.py (the Python
// reference) on the same synthetic energy table, plus the hysteresis seam.
//
// Gap 1 (free-VRAM axis) additions, all host-only:
//   * build_ft_spec(..., rk4v4_shift = 0) is the pre-axis spec, term for term;
//   * a disarmed axis (no probe / probe returns 0 / reference 0 / bridge 0 /
//     kill switch) produces exactly the shift-0 spec;
//   * the shift rule itself (deadband, sign, rk4v4 exposure cap);
//   * the axis end to end through decide_once() with a fake probe;
//   * the CLI > env > default precedence of from_env(CliOverrides).
//
// Build (WSL or Windows) -- the -isystem CUDA path is needed because
// ops/common/ft_stats.h includes <cuda_runtime.h> for its inline observe();
// no libcudart is linked (the decision TU names no CUDA function, which is the
// whole point of keeping the free-VRAM probe in serve/kv_vram_probe.cpp):
//   g++ -std=c++20 -I <repo>/src -I <repo>/include \
//       -isystem /usr/local/cuda/include tools/kv_relayout_test.cpp \
//       <repo>/src/serve/kv_auto_relayout.cpp -o kv_relayout_test
// Run:
//   ./kv_relayout_test --expect "<spec printed by ft_tiers.py>"
#include "serve/kv_auto_relayout.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

using ninfer::serve::build_ft_spec;
using ninfer::serve::KvAutoRelayout;
using Energy = std::vector<std::pair<int, double>>;

std::vector<std::pair<int, double>> synthetic_energy() {
    // 16 full-attention layers, strictly ascending energy so the tertile split
    // is unambiguous: low third (0-4) -> rk4v4, middle (5-9) -> iso4e, high -> nvfp4;
    // layers 13-15 are the deep band and must stay nvfp4 regardless.
    std::vector<std::pair<int, double>> energy;
    for (int layer = 0; layer < 16; ++layer) {
        energy.emplace_back(layer, 1.0 + 0.5 * layer);
    }
    return energy;
}

// Every shape the policy can be handed: dense, sparse (unmeasured layers),
// degenerate single-entry, out-of-range layer, and a layer outside the
// full-attention range.
std::vector<Energy> equivalence_tables() {
    std::vector<Energy> tables;
    tables.push_back(synthetic_energy());
    {
        Energy sparse;
        for (int layer = 0; layer < 16; layer += 3) { sparse.emplace_back(layer, 2.0 + layer); }
        tables.push_back(sparse);
    }
    tables.push_back({{0, 1.0}});
    tables.push_back({{0, 1.0}, {99, 3.0}});
    tables.push_back({{15, 1.0}, {14, 5.0}, {13, 2.0}, {0, 9.0}});
    {
        Energy descending;  // energy order is the ranking key, not the layer index
        for (int layer = 0; layer < 16; ++layer) {
            descending.emplace_back(layer, 20.0 - static_cast<double>(layer));
        }
        tables.push_back(descending);
    }
    return tables;
}

int count_tier(const std::string& spec, const std::string& tier) {
    int count = 0;
    std::size_t pos = 0;
    const std::string needle = ":" + tier;
    while ((pos = spec.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

bool require(bool condition, const char* what, bool& ok) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ok = false;
    }
    return condition;
}

} // namespace

int main(int argc, char** argv) {
    std::string expect;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--expect") == 0 && i + 1 < argc) { expect = argv[++i]; }
    }

    const auto energy = synthetic_energy();
    const std::string spec = ninfer::serve::build_ft_spec(energy, 16, 0.2);
    std::printf("spec=%s\n", spec.c_str());

    bool ok = true;
    if (!expect.empty()) {
        const bool match = expect == spec;
        std::printf("python-reference: %s\n", match ? "MATCH" : "MISMATCH");
        if (!match) {
            std::printf("  expected=%s\n  actual  =%s\n", expect.c_str(), spec.c_str());
            ok = false;
        }
    }

    // Hysteresis: the first cycle only records the candidate; the second cycle
    // applies it; an identical third cycle is a no-op.
    int applies = 0;
    ninfer::serve::KvAutoRelayout::Config config;
    config.interval_secs = 0; // decide_once is called directly
    config.full_attn_layers = 16;
    ninfer::serve::KvAutoRelayout relayout(config, [&](std::string_view applied) {
        ++applies;
        std::printf("apply[%d]=%s\n", applies, std::string(applied).c_str());
        return true;
    });
    const std::string first = relayout.decide_once(energy);
    const std::string second = relayout.decide_once(energy);
    const std::string third = relayout.decide_once(energy);
    std::printf("hysteresis: first=%s second=%s third=%s\n",
                first.empty() ? "(none)" : first.c_str(),
                second.empty() ? "(none)" : second.c_str(),
                third.empty() ? "(none)" : third.c_str());
    if (!first.empty() || second != spec || !third.empty() || applies != 1) {
        std::printf("hysteresis FAIL\n");
        ok = false;
    }

    // format_kv_table round trip: the applied table renders back to the spec.
    std::array<ninfer::KvCacheStorage, ninfer::kKvLayerStorageSlots> table{};
    for (int layer = 0; layer < 16; ++layer) { table[static_cast<std::size_t>(layer)] = ninfer::KvCacheStorage::BFloat16; }
    table[0] = ninfer::KvCacheStorage::E8Group64;
    table[1] = ninfer::KvCacheStorage::E8Group64;
    table[2] = ninfer::KvCacheStorage::Nvfp4Group16;
    const std::string formatted = ninfer::serve::format_kv_table(table, 3);
    std::printf("format_kv_table=%s\n", formatted.c_str());
    if (formatted != "0-1:rk4v4,2:nvfp4") {
        std::printf("format_kv_table FAIL\n");
        ok = false;
    }

    // =====================================================================
    // Gap 1: the free-VRAM axis
    // =====================================================================

    // (a) Shift 0 is the pre-axis policy, term for term, on every table shape.
    {
        int mismatches = 0;
        for (const Energy& e : equivalence_tables()) {
            const std::string base = build_ft_spec(e, 16, 0.2);
            const std::string zero = build_ft_spec(e, 16, 0.2, 0);
            if (base != zero) { ++mismatches; }
        }
        std::printf("shift0_equivalence: tables=%zu mismatches=%d\n",
                    equivalence_tables().size(), mismatches);
        require(mismatches == 0, "build_ft_spec(shift=0) differs from the pre-axis spec", ok);
    }

    // (b) A disarmed axis yields the shift-0 spec through decide_once(): the
    //     probe fails (0), the probe is absent, the bridge is absent, the
    //     kill switch is off, and the loop is disabled.
    {
        struct Case {
            const char* name;
            bool probe_present;
            bool kill_switch_off;
            std::uint64_t layer_bytes;
            int interval_secs;
        };
        const Case cases[] = {
            {"probe-fails", true, false, 32ULL << 20, 0},
            {"no-probe", false, false, 32ULL << 20, 0},
            {"no-bridge", true, false, 0, 0},
            {"kill-switch", true, true, 32ULL << 20, 0},
            {"loop-disabled", true, false, 32ULL << 20, 0},
        };
        int bad = 0;
        for (const Case& item : cases) {
            for (const Energy& e : equivalence_tables()) {
                KvAutoRelayout::Config c;
                c.interval_secs                  = item.interval_secs;
                c.full_attn_layers               = 16;
                c.vram_alpha                     = 1.0;
                c.vram_axis_enabled              = !item.kill_switch_off;
                c.vram_reference_bytes           = 8ULL << 30;
                c.vram_layer_bytes               = item.layer_bytes;
                if (item.probe_present) {
                    // A probe that cannot read the device reports 0.
                    c.free_vram_bytes = [] { return std::uint64_t{0}; };
                }
                KvAutoRelayout r(c, [](std::string_view) { return true; });
                (void)r.decide_once(e);                 // records the candidate
                const std::string landed = r.decide_once(e);  // applies it
                if (r.last_vram_shift() != 0 || landed != build_ft_spec(e, 16, 0.2)) { ++bad; }
            }
        }
        std::printf("disarmed_axis_equivalence: cases=%zu tables=%zu mismatches=%d\n",
                    sizeof(cases) / sizeof(cases[0]), equivalence_tables().size(), bad);
        require(bad == 0, "a disarmed VRAM axis changed the applied spec", ok);
    }

    // (c) The rule itself: whole-layer granularity, both directions, cap, and
    //     the disabled-bridge cases. Sign convention: the argument is
    //     free - reference, so a NEGATIVE argument (a deficit) is what grows the
    //     rk4v4 band and a positive one shrinks it.
    {
        const std::uint64_t layer = 32ULL << 20;             // one layer's KV bytes
        const std::uint64_t quantum = layer / 18;            // its nvfp4->rk4v4 saving
        const std::int64_t q = static_cast<std::int64_t>(quantum);
        const auto shift = [&](std::int64_t deviation) {
            return ninfer::serve::ft_vram_shift_for(deviation, layer);
        };
        const bool a = require(shift(0) == 0, "zero deviation moved the band", ok);
        const bool b = require(shift(-(q - 1)) == 0, "sub-quantum deficit moved the band", ok);
        const bool c = require(shift(-q) == 1, "one-quantum deficit did not move one layer", ok);
        const bool d = require(shift(-4 * q) == 4, "multi-quantum deficit", ok);
        const bool e = require(shift(3 * q) == -3, "surplus did not relax the band", ok);
        const bool f = require(shift(-1000 * q) == 64, "shift is not bounded by the table", ok);
        const bool g = require(ninfer::serve::ft_vram_shift_for(-q, 0) == 0,
                               "a missing byte bridge still produced a shift", ok);
        const bool h = require(ninfer::serve::ft_vram_shift_for(INT64_MIN, layer) == 64,
                               "INT64_MIN deviation is not handled", ok);
        std::printf("shift_rule: %s\n", (a && b && c && d && e && f && g && h) ? "ok" : "FAIL");
    }

    // (d) build_ft_spec with a shift: the rk4v4 band grows only by whole layers of
    //     the measured set, unobserved layers stay on the conservative iso4e, the
    //     deep band never moves, and the rk4v4 exposure cap is respected.
    {
        const std::string s0 = build_ft_spec(energy, 16, 0.2, 0);
        const std::string s_tight = build_ft_spec(energy, 16, 0.2, 1);
        const std::string s_loose = build_ft_spec(energy, 16, 0.2, -1);
        const std::string s_max = build_ft_spec(energy, 16, 0.2, 1000);
        const std::string s_min = build_ft_spec(energy, 16, 0.2, -1000);
        // 12 measured layers (0..11): base bands are 4/4/4 -> +1 moves layer 5
        // from iso4e to rk4v4, -1 moves layer 4 from rk4v4 to iso4e.
        require(count_tier(s0, "rk4v4") == 4 && count_tier(s0, "iso4e") == 4, "base tertiles", ok);
        require(count_tier(s_tight, "rk4v4") == 5 && count_tier(s_tight, "iso4e") == 3,
                "tight shift did not grow the rk4v4 band", ok);
        require(count_tier(s_loose, "rk4v4") == 3 && count_tier(s_loose, "iso4e") == 5,
                "loose shift did not shrink the rk4v4 band", ok);
        require(count_tier(s_loose, "nvfp4") == count_tier(s0, "nvfp4"),
                "the byte-neutral iso4e/nvfp4 boundary moved", ok);
        // A14 (redtest). The cap is product/kv_bit_budget.h's kKvBitBudgetE8LayerLimit,
        // which is 8: it is the size of the measured rk4v4 window (layers 0..7), and that
        // header's own banner note ("the banner is stale on that one", :18) records that an
        // older banner still claiming 10 is out of date. build_ft_spec clamps rk4v4_lo to
        // min(n, that constant), so on this table's 12 measured layers the maximal-deficit
        // band is min(12, 8) = 8: rk4v4 takes measured layers 0..7, iso4e is squeezed out
        // entirely, and nvfp4 keeps the 4 measured layers 8..11 plus the 4 protected deep
        // layers 12..15. The expectation below is the cap's value; the assertion still
        // DISCRIMINATES the cap, because with no cap rk4v4 would be n = 12.
        // The pre-image expected 10 / 6, i.e. the pre-tightening limit, and the test was in
        // no CMakeLists so nothing ever ran it.
        require(count_tier(s_max, "rk4v4") == 8 && count_tier(s_max, "nvfp4") == 8,
                "the rk4v4 exposure cap (kKvBitBudgetE8LayerLimit) was not enforced", ok);
        require(count_tier(s_min, "rk4v4") == 0, "the loose end did not reach an empty rk4v4 band", ok);
        require(count_tier(s_max, "nvfp4") >= 4, "deep protection was lost", ok);
        const std::string lo = s_max.substr(s_max.find("12:"));
        require(lo.rfind("12:nvfp4", 0) == 0, "a deep layer left nvfp4 under a shift", ok);
        std::printf("spec_shift: rk4v4 %d->%d (+1) %d (-1) %d (cap)\n", count_tier(s0, "rk4v4"),
                    count_tier(s_tight, "rk4v4"), count_tier(s_loose, "rk4v4"),
                    count_tier(s_max, "rk4v4"));

        // A sparse table: only the observed layers can enter the rk4v4 band, the
        // rest must keep the conservative iso4e even under a maximal deficit.
        const Energy sparse = equivalence_tables()[1];
        const std::string sparse_tight = build_ft_spec(sparse, 16, 0.2, 1000);
        int sparse_iso4e = 0;
        int sparse_rk4v4 = 0;
        for (int layer = 0; layer < 12; ++layer) {
            const std::string tag = "," + std::to_string(layer) + ":";
            const std::size_t at = sparse_tight.find(tag);
            if (at == std::string::npos && layer != 0) { continue; }
            const std::size_t begin = at == std::string::npos ? 0 : at + tag.size();
            const std::string tier = sparse_tight.substr(begin, sparse_tight.find(',', begin) - begin);
            if (tier == "iso4e") { ++sparse_iso4e; }
            if (tier == "rk4v4") { ++sparse_rk4v4; }
        }
        std::printf("spec_shift_sparse: rk4v4=%d iso4e=%d\n", sparse_rk4v4, sparse_iso4e);
        require(sparse_iso4e >= 6, "unobserved layers lost the conservative iso4e default", ok);
    }

    // (e) The axis end to end: a fake probe, alpha = 1 so the value is exact.
    {
        const std::uint64_t reference = 8ULL << 30;
        const std::uint64_t layer_bytes = 32ULL << 20;
        const std::uint64_t quantum = layer_bytes / 18;
        std::uint64_t free_now = reference;
        KvAutoRelayout::Config c;
        c.interval_secs        = 0;
        c.full_attn_layers     = 16;
        c.vram_alpha           = 1.0;   // no damping: deterministic arithmetic
        c.vram_reference_bytes = reference;
        c.vram_layer_bytes     = layer_bytes;
        c.free_vram_bytes      = [&free_now] { return free_now; };

        int applied_count = 0;
        std::string landed;
        KvAutoRelayout r(c, [&](std::string_view s) {
            ++applied_count;
            landed = std::string(s);
            return true;
        });

        // Tight: 4 quanta below the baseline -> shift +4.
        free_now = reference - 4 * quantum;
        const std::string c1 = r.decide_once(energy);
        const std::string c2 = r.decide_once(energy);
        require(c1.empty() && r.last_vram_shift() == 4, "tight reading did not shift by 4", ok);
        require(c2 == build_ft_spec(energy, 16, 0.2, 4) && landed == c2,
                "the tight shift was not applied", ok);
        std::printf("axis_tight: shift=%d rk4v4=%d\n", r.last_vram_shift(), count_tier(landed, "rk4v4"));

        // Loose: 2 quanta above the baseline -> shift -2 (hysteresis again).
        free_now = reference + 2 * quantum;
        const std::string c3 = r.decide_once(energy);
        const std::string c4 = r.decide_once(energy);
        require(c3.empty(), "hysteresis did not defer the loose candidate", ok);
        require(c4 == build_ft_spec(energy, 16, 0.2, -2) && r.last_vram_shift() == -2,
                "the loose shift was not applied", ok);
        std::printf("axis_loose: shift=%d rk4v4=%d\n", r.last_vram_shift(), count_tier(landed, "rk4v4"));

        // The EWMA damps a one-cycle spike: with alpha 0.5 a single tight read
        // moves the damped value only halfway, so a quarter-quantum spike is
        // absorbed instead of reloading the table.
        KvAutoRelayout::Config damped = c;
        damped.vram_alpha = 0.5;
        KvAutoRelayout dr(damped, [](std::string_view) { return true; });
        free_now = reference;
        (void)dr.decide_once(energy);          // seeds the EWMA at the baseline
        free_now = reference - quantum / 4;    // small spike
        (void)dr.decide_once(energy);
        require(dr.last_vram_shift() == 0, "the EWMA did not absorb a sub-quantum spike", ok);
        free_now = reference - 8 * quantum;    // sustained, large: must still act
        (void)dr.decide_once(energy);
        (void)dr.decide_once(energy);
        require(dr.last_vram_shift() > 0, "a sustained 4-quantum (damped) deficit did not act", ok);
        std::printf("axis_damped: shift=%d\n", dr.last_vram_shift());
        require(applied_count == 2, "an unexpected number of reloads was applied", ok);
    }

    // (f) CLI > env > default for the gap-2 layer of from_env().
    {
#if !defined(_WIN32)
        using Overrides = KvAutoRelayout::CliOverrides;
        setenv("NINFER_FT_RELOAD_SECS", "30", 1);
        setenv("NINFER_FT_VRAM_AXIS", "0", 1);
        const bool a = require(KvAutoRelayout::from_env().interval_secs == 30,
                               "the env interval was not read", ok);
        const bool b = require(KvAutoRelayout::from_env(Overrides{.interval_secs = 7}).interval_secs == 7,
                               "the CLI interval did not beat the env", ok);
        const bool c = require(KvAutoRelayout::from_env(Overrides{.interval_secs = 0}).interval_secs == 0,
                               "--kv-auto-relayout 0 did not beat the env", ok);
        const bool d = require(KvAutoRelayout::from_env(Overrides{}).interval_secs == 30,
                               "an unset CLI override stopped deferring to the env", ok);
        const bool e = require(!KvAutoRelayout::from_env().vram_axis_enabled,
                               "the env kill switch was ignored", ok);
        const bool f = require(KvAutoRelayout::from_env(Overrides{.vram_axis_enabled = true})
                                   .vram_axis_enabled,
                               "--ft-vram-axis on did not beat the env kill switch", ok);
        const bool g = require(!KvAutoRelayout::from_env(Overrides{.vram_axis_enabled = false})
                                   .vram_axis_enabled,
                               "--ft-vram-axis off was ignored", ok);
        unsetenv("NINFER_FT_RELOAD_SECS");
        unsetenv("NINFER_FT_VRAM_AXIS");
        const bool h = require(KvAutoRelayout::from_env().interval_secs == 0,
                               "the relayout is not off by default", ok);
        const bool i = require(KvAutoRelayout::from_env().vram_axis_enabled,
                               "the axis is not armed by default", ok);
        std::printf("cli_env_precedence: %s\n", (a && b && c && d && e && f && g && h && i) ? "ok" : "FAIL");
#else
        std::printf("cli_env_precedence: skipped (Windows)\n");
#endif
    }

    std::printf("%s\n", ok ? "KV_RELAYOUT_TEST PASS" : "KV_RELAYOUT_TEST FAIL");
    return ok ? 0 : 1;
}
