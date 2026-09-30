#include "serve/kv_auto_relayout.h"

#include "ops/common/ft_stats.h"
#include "product/kv_bit_budget.h"  // kKvBitBudgetE8LayerLimit (gap 1 rk4v4 cap)
#include "serve/kv_cold_policy.h"


#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>

namespace ninfer::serve {
namespace {

const char* kv_storage_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16: return "bf16";
    case KvCacheStorage::Int8Group64: return "int8";
    // BOTH fp8 spellings name the SAME target tier (layouts_impl.h
    // target_kv_cache_profile / product::kv_dtype_for_storage): Fp8E4M3Row256 is
    // what --kv-dtype emits, Fp8Group16 is what product::parse_kv_storage emits
    // for a per-layer spec. Naming only one of them renamed the other to "bf16".
    case KvCacheStorage::Fp8E4M3Row256: return "fp8";
    case KvCacheStorage::Fp8Group16: return "fp8-g16";
    case KvCacheStorage::Nvfp4Group16: return "nvfp4";
    case KvCacheStorage::Iso3Group16: return "iso4e";
    case KvCacheStorage::E8Group64: return "rk4v4";
    // The rk4v4 family's narrower K planes. Unreachable through the planner (both refuse
    // by name in product::kv_dtype_for_storage / target_kv_cache_profile), and named
    // here so that this switch keeps the no-silent-default property its trailing throw
    // documents: the next enumerator must land in one of the two named camps, not fall
    // past the switch with a message about a code no enumerator names.
    case KvCacheStorage::E8K3Group64: return "rk3v4";
    case KvCacheStorage::E8K2Group64: return "rk2v4";
    case KvCacheStorage::Dropped:
        // types.h:38-43: a discarded layer has NO storage at all. "bf16" is not a
        // tier it falls back to -- BFloat16 is also the "unset / inherit the global
        // --kv-dtype" sentinel (product/kv_component_switch.h kv_resolve_slot_dtype),
        // so rendering it as bf16 silently said "ignore the operator's dtype here",
        // exactly the laundering kv_storage_dtype.h:10-19 was written to remove.
        return "dropped";
    }
    // No silent default arm (kv_storage_dtype.h:27-29 is the in-tree statement of
    // this idiom): an out-of-range byte -- KvCacheStorage is a uint8_t, so a cast or
    // untrusted input can produce one -- must be LOUD, not "bf16".
    throw std::invalid_argument(
        "kv_auto_relayout: KV storage code " +
        std::to_string(static_cast<unsigned>(storage)) +
        " has no name; the per-layer KV table holds a value no enumerator names");
}

KvCacheStorage kv_storage_from_name(std::string_view name) {
    if (name == "int8") { return KvCacheStorage::Int8Group64; }
    if (name == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (name == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (name == "iso4e") { return KvCacheStorage::Iso3Group16; }
    if (name == "rk4v4") { return KvCacheStorage::E8Group64; }
    // The inverse of the two arms above, so format_kv_table() -> parse_kv_table() stays
    // total. Unlike the unknown-text fallback below, these are NOT "CLI tolerance": the
    // name is recognized and resolves to a real tier, which then REFUSES at the mapper
    // with the real reason. Silently landing on bf16 -- what the fallback would do --
    // would be the laundering kv_storage_dtype.h:10-19 exists to remove.
    if (name == "rk3v4") { return KvCacheStorage::E8K3Group64; }
    if (name == "rk2v4") { return KvCacheStorage::E8K2Group64; }
    // Inverse of the two arms added above, so format_kv_table() -> parse_kv_table()
    // stays total for every value the table can now render. (The BFloat16 fallback
    // below is NOT changed: unknown text staying at bf16 is this parser's
    // documented CLI tolerance, see the comment above parse_kv_table.)
    if (name == "fp8-g16") { return KvCacheStorage::Fp8Group16; }
    if (name == "dropped") { return KvCacheStorage::Dropped; }
    return KvCacheStorage::BFloat16;
}

// Parses "0-11:rk4v4,12-15:nvfp4" into a per-layer table (unknown text is left at
// bf16, mirroring the CLI parser's tolerance for the pieces it understands).
std::array<KvCacheStorage, kKvLayerStorageSlots> parse_kv_table(std::string_view spec) {
    std::array<KvCacheStorage, kKvLayerStorageSlots> table{};
    std::size_t pos = 0;
    while (pos <= spec.size()) {
        const std::size_t comma = spec.find(',', pos);
        const std::string_view item = spec.substr(
            pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos);
        const std::size_t colon = item.find(':');
        if (colon != std::string_view::npos) {
            const std::string_view range = item.substr(0, colon);
            const KvCacheStorage storage = kv_storage_from_name(item.substr(colon + 1));
            const std::size_t dash = range.find('-');
            int first = 0;
            int last = 0;
            if (range == "all") {
                // "all" 前缀此前会被 atoi 解析成 0 ⇒ `all:bf16` 只设第 0 层，
                // 与"未设置"无法区分（_TODO.md 98/117 U3）。显式展开全层。
                first = 0;
                last = static_cast<int>(kKvLayerStorageSlots) - 1;
            } else if (dash == std::string_view::npos) {
                first = last = std::atoi(std::string(range).c_str());
            } else {
                first = std::atoi(std::string(range.substr(0, dash)).c_str());
                last = std::atoi(std::string(range.substr(dash + 1)).c_str());
            }
            if (first < 0) { first = 0; }
            if (last >= static_cast<int>(kKvLayerStorageSlots)) {
                last = static_cast<int>(kKvLayerStorageSlots) - 1;
            }
            for (int layer = first; layer <= last; ++layer) {
                table[static_cast<std::size_t>(layer)] = storage;
            }
        }
        if (comma == std::string_view::npos) { break; }
        pos = comma + 1;
    }
    return table;
}

// Gap 1: the free-VRAM axis is armed only when it has a probe, a baseline and
// the byte -> layer bridge. Shared by observe_vram_shift() and the cycle log so
// neither can act on a half-configured axis.
bool vram_axis_usable(const KvAutoRelayout::Config& config) {
    return config.vram_axis_enabled && static_cast<bool>(config.free_vram_bytes) &&
           config.vram_reference_bytes != 0 && config.vram_layer_bytes != 0;
}

} // namespace

std::string format_kv_table(const std::array<KvCacheStorage, kKvLayerStorageSlots>& table,
                            int layers) {
    if (layers <= 0 || layers > static_cast<int>(kKvLayerStorageSlots)) {
        layers = static_cast<int>(kKvLayerStorageSlots);
    }
    std::string out;
    int start = 0;
    while (start < layers) {
        const KvCacheStorage storage = table[static_cast<std::size_t>(start)];
        int end = start;
        while (end + 1 < layers && table[static_cast<std::size_t>(end + 1)] == storage) {
            ++end;
        }
        if (!out.empty()) { out.push_back(','); }
        if (start == end) {
            out += std::to_string(start);
        } else {
            out += std::to_string(start) + "-" + std::to_string(end);
        }
        out.push_back(':');
        out += kv_storage_name(storage);
        start = end + 1;
    }
    return out;
}

int ft_vram_shift_for(std::int64_t free_minus_reference_bytes, std::uint64_t layer_bytes) {
    // Whole-layer granularity: one layer that moves from the 4.50-bit tiers to
    // rk4v4 gives back layer_bytes/18 (kFtNvfp4ToRk4v4SavingDenominator).
    const std::uint64_t saving_per_layer = layer_bytes / kFtNvfp4ToRk4v4SavingDenominator;
    if (saving_per_layer == 0) { return 0; }  // no bridge: stay inert
    const bool tight = free_minus_reference_bytes < 0;
    // INT64_MIN-safe negation (the deviation is a byte count, but a caller can
    // hand over any int64).
    const std::uint64_t magnitude =
        tight ? static_cast<std::uint64_t>(-(free_minus_reference_bytes + 1)) + 1ULL
              : static_cast<std::uint64_t>(free_minus_reference_bytes);
    // Floor division == the deadband: a deviation below one whole layer's
    // saving demands nothing.
    const std::uint64_t layers = magnitude / saving_per_layer;
    if (layers == 0) { return 0; }
    // The landed value is clamped again in build_ft_spec (band bounds + the rk4v4
    // exposure cap); the table width is the only bound that matters here.
    const std::uint64_t capped =
        std::min<std::uint64_t>(layers, static_cast<std::uint64_t>(kKvLayerStorageSlots));
    return tight ? static_cast<int>(capped) : -static_cast<int>(capped);
}

std::string build_ft_spec(const std::vector<std::pair<int, double>>& energy,
                          int full_attn_layers, double deep_frac, int rk4v4_shift) {
    if (full_attn_layers <= 0) { return {}; }
    const int deep_start = static_cast<int>(static_cast<double>(full_attn_layers) *
                                            (1.0 - deep_frac));
    std::map<int, double> by_layer;
    for (const auto& item : energy) {
        if (item.first >= 0 && item.first < full_attn_layers) { by_layer[item.first] = item.second; }
    }

    std::vector<int> unmeasured;
    std::vector<std::pair<int, double>> measured; // (layer, energy)
    for (int layer = 0; layer < deep_start; ++layer) {
        const auto it = by_layer.find(layer);
        if (it == by_layer.end()) {
            unmeasured.push_back(layer);
        } else {
            measured.emplace_back(layer, it->second);
        }
    }
    // Energy ascending: the lowest-energy third is the most compressible.
    std::sort(measured.begin(), measured.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<std::string> parts;
    parts.reserve(static_cast<std::size_t>(full_attn_layers));
    for (int layer = deep_start; layer < full_attn_layers; ++layer) {
        parts.push_back(std::to_string(layer) + ":nvfp4"); // deep protection
    }
    for (const int layer : unmeasured) {
        parts.push_back(std::to_string(layer) + ":iso4e"); // conservative default
    }
    const int n = static_cast<int>(measured.size());
    // Tertile boundaries. rk4v4_shift == 0 reproduces the pre-axis expression
    // exactly (rk4v4_lo == n/3, rk4v4_hi == 2*(n/3)).
    int rk4v4_lo = n / 3;
    int rk4v4_hi = 2 * (n / 3);
    if (rk4v4_shift != 0) {
        // Gap 1: only the rk4v4/iso4e boundary moves (the only byte-moving step of
        // the ladder), and the rk4v4 band may not exceed the measurement-backed
        // exposure cap of product/kv_bit_budget.h:69-82.
        const int rk4v4_cap = std::min(n, static_cast<int>(product::kKvBitBudgetE8LayerLimit));
        rk4v4_lo = std::clamp(rk4v4_lo + rk4v4_shift, 0, rk4v4_cap);
        // The iso4e band keeps its own upper edge; it cannot invert, though, when
        // a small measured set makes the two boundaries meet.
        rk4v4_hi = std::max(rk4v4_hi, rk4v4_lo);
    }
    // kvfix (F893): THE BYTE-NEUTRAL BOUNDARY IS REFUSED, BY NAME. The four readings this
    // rests on are on the constant's declaration in serve/kv_auto_relayout.h. The condition
    // is exactly "the proxy would place at least one MEASURED layer on iso4e": with
    // rk4v4_hi == rk4v4_lo there is no such boundary, so a table too small to separate the
    // pair is returned unchanged and this guard is not a blanket disable.
    if (rk4v4_hi > rk4v4_lo) {
        throw std::invalid_argument(kFtByteNeutralBoundaryWithheld);
    }
    for (int i = 0; i < n; ++i) {
        const char* tier = i < rk4v4_lo ? "rk4v4" : (i < rk4v4_hi ? "iso4e" : "nvfp4");
        parts.push_back(std::to_string(measured[static_cast<std::size_t>(i)].first) + ":" + tier);
    }
    std::string out;
    for (const auto& part : parts) {
        if (!out.empty()) { out.push_back(','); }
        out += part;
    }
    return out;
}

KvAutoRelayout::Config KvAutoRelayout::from_env() { return from_env(CliOverrides{}); }

KvAutoRelayout::Config KvAutoRelayout::from_env(const CliOverrides& cli) {
    Config config;
    if (const char* secs = std::getenv("NINFER_FT_RELOAD_SECS")) {
        const int value = std::atoi(secs);
        if (value > 0) { config.interval_secs = value; }
    }
    if (const char* layers = std::getenv("NINFER_FT_FULL_ATTN_LAYERS")) {
        const int value = std::atoi(layers);
        if (value > 0) { config.full_attn_layers = value; }
    }
    if (const char* frac = std::getenv("NINFER_FT_DEEP_FRAC")) {
        const double value = std::atof(frac);
        if (value > 0.0 && value < 1.0) { config.deep_frac = value; }
    }
    if (const char* axis = std::getenv("NINFER_FT_VRAM_AXIS")) {
        // Kill switch only: the default is armed, so this env can only take the
        // axis away. "0"/"off"/"false" disarm; anything else (including "1")
        // keeps the default, which is why an unrecognised spelling is not an
        // error here.
        const std::string_view text(axis);
        config.vram_axis_enabled = !(text == "0" || text == "off" || text == "false");
    }
    // Gap 2: the CLI layer, applied ON TOP of the environment (CLI > env >
    // default). An unset optional defers, so from_env()'s behaviour for every
    // pre-existing caller is bit-for-bit what it was.
    if (cli.interval_secs.has_value()) {
        config.interval_secs = std::max(*cli.interval_secs, 0);
    }
    if (cli.vram_axis_enabled.has_value()) {
        config.vram_axis_enabled = *cli.vram_axis_enabled;
    }
    return config;
}

KvAutoRelayout::KvAutoRelayout(Config config, ApplyFn apply)
    : config_(std::move(config)), apply_(std::move(apply)) {
    applied_table_ = config_.current_table;
}

KvAutoRelayout::~KvAutoRelayout() { stop(); }

void KvAutoRelayout::start() {
    if (config_.interval_secs <= 0 || !apply_ || running_.exchange(true)) { return; }
    thread_ = std::thread([this] { loop(); });
}

void KvAutoRelayout::stop() {
    if (!running_.exchange(false)) { return; }
    if (thread_.joinable()) { thread_.join(); }
}

bool KvAutoRelayout::vram_axis_armed() const noexcept { return vram_axis_usable(config_); }

int KvAutoRelayout::observe_vram_shift() {
    last_vram_shift_       = 0;
    last_free_vram_bytes_  = 0;
    if (!vram_axis_armed()) { return 0; }
    const std::uint64_t free_now = config_.free_vram_bytes();
    if (free_now == 0) { return 0; }  // query failed: stay inert, do not guess
    last_free_vram_bytes_ = free_now;

    if (!(config_.vram_alpha > 0.0 && config_.vram_alpha <= 1.0)) {
        // A weight outside (0,1] is a configuration error: use the raw reading
        // rather than folding a non-finite value into the decision.
        vram_ewma_ = static_cast<double>(free_now);
    } else if (vram_ewma_ <= 0.0) {
        vram_ewma_ = static_cast<double>(free_now);  // seed with the first reading
    } else {
        vram_ewma_ = config_.vram_alpha * static_cast<double>(free_now) +
                     (1.0 - config_.vram_alpha) * vram_ewma_;
    }
    // Byte counts stay far below 2^53, so the double round-trip is exact enough
    // that the integer rule below still sees whole bytes.
    const std::int64_t deviation =
        static_cast<std::int64_t>(std::llround(vram_ewma_)) -
        static_cast<std::int64_t>(config_.vram_reference_bytes);
    last_vram_shift_ = ft_vram_shift_for(deviation, config_.vram_layer_bytes);
    return last_vram_shift_;
}

std::string KvAutoRelayout::decide_once(const std::vector<std::pair<int, double>>& energy) {
    if (energy.empty()) { return {}; } // no observation yet
    // Gap 1: 0 unless the axis is armed AND the free VRAM has moved a whole
    // layer's worth of savings away from (or towards) its baseline.
    const int vram_shift = observe_vram_shift();
    // kvfix (F893): a WITHHELD per-layer table is a REFUSAL, not a silent no-op. It is
    // printed by name, once per distinct reason, and NOTHING is landed -- the loop keeps
    // running and the applied table keeps whatever it had. The catch has to live HERE and
    // not at the apply lambda in apps/serve/main.cpp: that lambda is only reached once a
    // candidate EXISTS, so the throw would escape decide_once(), unwind loop()'s thread and
    // call std::terminate -- a refusal turned into a crash.
    std::string candidate;
    try {
        candidate = build_ft_spec(energy, config_.full_attn_layers,
                                  config_.deep_frac, vram_shift);
    } catch (const std::exception& withheld) {
        if (withheld_reported_ != withheld.what()) {
            withheld_reported_ = withheld.what();
            std::fprintf(stderr, "[ft] %s\n", withheld_reported_.c_str());
        }
        return {};
    }
    withheld_reported_.clear();
    if (candidate.empty()) { return {}; }

    // Semantic comparison: only a real table change counts.
    const auto candidate_table = parse_kv_table(candidate);
    bool same_as_applied = true;
    for (std::size_t layer = 0; layer < candidate_table.size(); ++layer) {
        if (candidate_table[layer] != applied_table_[layer]) {
            same_as_applied = false;
            break;
        }
    }
    if (same_as_applied) {
        pending_spec_.clear();
        return {};
    }

    // Hysteresis: the same candidate must be seen in two consecutive cycles.
    // The VRAM axis rides on the same gate: a shift that comes and goes within
    // one cycle changes the candidate string, so the two-cycle rule takes two
    // identical cycles (shift included) before anything is reloaded.
    if (candidate != pending_spec_) {
        pending_spec_ = candidate;
        return {};
    }
    if (!apply_(candidate)) { return {}; } // reload busy: retry next cycle
    applied_table_ = candidate_table;
    applied_spec_ = candidate;
    pending_spec_.clear();
    return candidate;
}

void KvAutoRelayout::loop() {
    using namespace std::chrono_literals;
    int elapsed_ms = 0;
    while (running_.load()) {
        std::this_thread::sleep_for(250ms);
        elapsed_ms += 250;
        if (elapsed_ms < config_.interval_secs * 1000) { continue; }
        elapsed_ms = 0;

        std::vector<ops::ft::EnergySample> samples(static_cast<std::size_t>(ops::ft::kMaxLayers));
        const int count = ops::ft::snapshot(samples.data(), static_cast<int>(samples.size()));
        if (count <= 0) { continue; } // ft stats disabled or nothing observed yet
        std::vector<std::pair<int, double>> energy;
        energy.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) { energy.emplace_back(samples[i].layer, samples[i].mean_l); }

        // S34 closed-loop cold residency tap: windowed energies come from
        // snapshot deltas (ft::snapshot is cumulative since process start,
        // so (sum1-sum0)/(count1-count0) is the exact window mean). DRY by
        // default -- NINFER_FT_COLD_MODE=live is reserved for the S30 pool
        // path and intentionally NOT wired in this first patch.
        static const int cold_pages = [] {
            const char* e = std::getenv("NINFER_FT_COLD_PAGES");
            return e ? std::atoi(e) : 0;
        }();
        if (cold_pages > 0) {
            static serve::ColdPolicyConfig cold_cfg;
            static serve::ColdPolicyState cold_state;
            static std::map<int, std::pair<double, std::uint64_t>> prev_cum;
            cold_cfg.full_attn_layers = config_.full_attn_layers;
            cold_cfg.deep_frac = config_.deep_frac;
            cold_cfg.cold_cap = cold_pages;
            std::vector<serve::ColdObservation> cold_obs;
            for (int i = 0; i < count; ++i) {
                const auto& s = samples[static_cast<std::size_t>(i)];
                const double sum = s.mean_l * static_cast<double>(s.rounds);
                const auto it = prev_cum.find(s.layer);
                if (it != prev_cum.end()) {
                    double wm = 0.0;
                    std::uint64_t wr = 0;
                    serve::window_from_cumulative(it->second.first,
                                                  it->second.second, sum,
                                                  s.rounds, wm, wr);
                    if (wr > 0) {
                        cold_obs.push_back({s.layer, wm, wr});
                    }
                }
                prev_cum[s.layer] = {sum, s.rounds};
            }
            const serve::ColdDecision cold = serve::decide_cold_residency(
                cold_obs, cold_cfg, cold_state, energy);
            const auto bad = serve::invariant_violations(cold, cold_cfg);
            for (const std::string& v : bad) {
                std::fprintf(stderr, "[ft][cold] INVARIANT VIOLATION: %s\n",
                             v.c_str());
            }
            if (cold.changed && bad.empty()) {
                std::string cold_list;
                for (const int layer : cold.cold_layers) {
                    cold_list += std::to_string(layer);
                    cold_list.push_back(' ');
                }
                std::fprintf(stderr,
                             "[ft][cold:dry] would cold=[%s] cut_reasons=%zu\n",
                             cold_list.c_str(), cold.reasons.size());
            }
        }

        const std::string applied = decide_once(energy);
        // Gap 1 forensics, printed only while the axis is armed so a build with
        // no probe keeps exactly the stderr it had before.
        if (vram_axis_armed()) {
            std::fprintf(stderr, "[ft][vram] free=%llu ref=%llu layer=%llu shift=%+d\n",
                         static_cast<unsigned long long>(last_free_vram_bytes_),
                         static_cast<unsigned long long>(config_.vram_reference_bytes),
                         static_cast<unsigned long long>(config_.vram_layer_bytes),
                         last_vram_shift_);
        }
        if (!applied.empty()) {
            std::fprintf(stderr, "[ft] auto-relayout -> %s\n", applied.c_str());
        }
    }
}

} // namespace ninfer::serve
