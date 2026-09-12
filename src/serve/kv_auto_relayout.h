#pragma once

// FreeToken step 2 — periodic KV relayout (design: _TODO.md 55, policy: the
// ft_tiers.py decision logic). While the server runs, the ft energy table
// (ops::ft) is sampled every N seconds; a per-layer KV table is derived
// (deep-layer protection + energy tertiles) and applied through
// GenerationService::reload_kv_storage once the decision has been stable for
// two consecutive cycles (hysteresis). Disabled unless NINFER_FT_RELOAD_SECS>0.
//
// The spec strings are compared *semantically* (parsed per-layer tables), so a
// formatting difference never triggers a spurious reload.

#include "ninfer/types.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::serve {

class KvAutoRelayout {
public:
    // Applies a new spec; returns false when the reload was rejected (e.g. one
    // is already in flight) so the caller retries on the next cycle.
    using ApplyFn = std::function<bool(std::string_view spec)>;

    struct Config {
        int interval_secs = 0;      // 0 disables the thread (NINFER_FT_RELOAD_SECS)
        int full_attn_layers = 16;  // deep-protection range (NINFER_FT_FULL_ATTN_LAYERS)
        double deep_frac = 0.2;     // last 20% of full-attn layers keep nvfp4
        std::array<KvCacheStorage, kKvLayerStorageSlots> current_table{};
        bool current_table_explicit = false;
    };

    static Config from_env();
    KvAutoRelayout(Config config, ApplyFn apply);
    ~KvAutoRelayout();

    KvAutoRelayout(const KvAutoRelayout&) = delete;
    KvAutoRelayout& operator=(const KvAutoRelayout&) = delete;

    void start();
    void stop();

    // Test seam: run one decision cycle with a supplied energy table; returns
    // the spec that would be applied (empty when nothing changed).
    std::string decide_once(const std::vector<std::pair<int, double>>& energy);

private:
    void loop();

    Config config_;
    ApplyFn apply_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::string pending_spec_;  // candidate seen in the previous cycle
    std::string applied_spec_;  // spec handed to apply_ successfully
    std::array<KvCacheStorage, kKvLayerStorageSlots> applied_table_{};
};

// ft_tiers.py policy in C++: layers [deep_start, total) keep nvfp4; the rest are
// split by energy tertiles (lowest third -> e8, middle -> iso3, highest ->
// nvfp4); layers with no observation get the conservative iso3.
std::string build_ft_spec(const std::vector<std::pair<int, double>>& energy,
                          int full_attn_layers, double deep_frac = 0.2);

// "0-11:e8,12-15:nvfp4" formatting for a per-layer table (first `layers` slots).
std::string format_kv_table(const std::array<KvCacheStorage, kKvLayerStorageSlots>& table,
                            int layers);

} // namespace ninfer::serve
