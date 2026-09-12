#include "serve/kv_auto_relayout.h"

#include "ops/common/ft_stats.h"
#include "serve/kv_cold_policy.h"


#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>

namespace ninfer::serve {
namespace {

const char* kv_storage_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16: return "bf16";
    case KvCacheStorage::Int8Group64: return "int8";
    case KvCacheStorage::Fp8E4M3Row256: return "fp8";
    case KvCacheStorage::Nvfp4Group16: return "nvfp4";
    case KvCacheStorage::Iso3Group16: return "iso3";
    case KvCacheStorage::E8Group64: return "e8";
    }
    return "bf16";
}

KvCacheStorage kv_storage_from_name(std::string_view name) {
    if (name == "int8") { return KvCacheStorage::Int8Group64; }
    if (name == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (name == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (name == "iso3") { return KvCacheStorage::Iso3Group16; }
    if (name == "e8") { return KvCacheStorage::E8Group64; }
    return KvCacheStorage::BFloat16;
}

// Parses "0-11:e8,12-15:nvfp4" into a per-layer table (unknown text is left at
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

std::string build_ft_spec(const std::vector<std::pair<int, double>>& energy,
                          int full_attn_layers, double deep_frac) {
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
        parts.push_back(std::to_string(layer) + ":iso3"); // conservative default
    }
    const int n = static_cast<int>(measured.size());
    for (int i = 0; i < n; ++i) {
        const char* tier = i < n / 3 ? "e8" : (i < 2 * (n / 3) ? "iso3" : "nvfp4");
        parts.push_back(std::to_string(measured[static_cast<std::size_t>(i)].first) + ":" + tier);
    }
    std::string out;
    for (const auto& part : parts) {
        if (!out.empty()) { out.push_back(','); }
        out += part;
    }
    return out;
}

KvAutoRelayout::Config KvAutoRelayout::from_env() {
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

std::string KvAutoRelayout::decide_once(const std::vector<std::pair<int, double>>& energy) {
    if (energy.empty()) { return {}; } // no observation yet
    const std::string candidate = build_ft_spec(energy, config_.full_attn_layers,
                                                config_.deep_frac);
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
        if (!applied.empty()) {
            std::fprintf(stderr, "[ft] auto-relayout -> %s\n", applied.c_str());
        }
    }
}

} // namespace ninfer::serve
