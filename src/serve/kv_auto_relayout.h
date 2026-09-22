#pragma once

// FreeToken step 2 — periodic KV relayout (design: _TODO.md 55, policy: the
// ft_tiers.py decision logic). While the server runs, the ft energy table
// (ops::ft) is sampled every N seconds; a per-layer KV table is derived
// (deep-layer protection + energy tertiles) and applied through
// GenerationService::reload_kv_storage once the decision has been stable for
// two consecutive cycles (hysteresis). Disabled by default: it runs only when
// the interval is > 0, set by NINFER_FT_RELOAD_SECS (env) or --kv-auto-relayout
// SECS (CLI, which wins).
//
// The spec strings are compared *semantically* (parsed per-layer tables), so a
// formatting difference never triggers a spurious reload.
//
// GAP 1 (free-VRAM axis): the tertile split used to be a function of the energy
// table alone, so the live memory situation could not influence the tier of a
// single layer. `Config::free_vram_bytes` (a caller-injected probe),
// `Config::vram_reference_bytes` (the free-VRAM baseline the axis is relative
// to) and `Config::vram_layer_bytes` (one full-attention layer's device KV
// payload, the byte -> layer bridge) now move the rk4v4 band boundary by whole
// layers; see ft_vram_shift_for() for the rule and its evidence. With no probe,
// no baseline or no bridge the shift is 0 and this file's output is
// byte-for-byte the pre-axis policy.
//
// GAP 2 (CLI): `--kv-auto-relayout SECS` / `--ft-vram-axis on|off` / `--ft-stats
// on|off` are the operator surface (apps/serve + apps/cli). The precedence is
// CLI > environment > default everywhere: from_env(CliOverrides{}) is the
// environment layer, and an unset optional means "defer", so from_env() keeps
// its old meaning for every caller that passes nothing. The line stays OPT-IN:
// the default interval is 0 = disabled.

#include "ninfer/types.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::serve {

// EWMA weight for the live free-VRAM reading. 0.5 is the value the cold
// residency policy already uses for its per-layer energy EWMA
// (kv_cold_policy.h ColdPolicyConfig::ewma_alpha), so the two loops that hang
// off the same observation cadence damp their inputs identically.
inline constexpr double kFtVramDefaultAlpha = 0.5;

// The ONLY byte-moving step of the KV tier ladder is nvfp4 -> rk4v4: 4.50 -> 4.25
// bits per element (product/kv_bit_budget.h kKvBitBudgetTiers, bits_x100 450 ->
// 425), i.e. 25/450 == 1/18 of that layer's device bytes. iso4e shares nvfp4's
// plane geometry (450 == 450), so re-cutting the iso4e/nvfp4 boundary would
// trade quality for exactly zero bytes.
inline constexpr std::uint64_t kFtNvfp4ToRk4v4SavingDenominator = 18;

class KvAutoRelayout {
public:
    // Applies a new spec; returns false when the reload was rejected (e.g. one
    // is already in flight) so the caller retries on the next cycle.
    using ApplyFn = std::function<bool(std::string_view spec)>;

    // Gap 1: "bytes free on the device right now", injected by the caller.
    // Returning 0 (or leaving this empty) means "unknown" and disarms the axis.
    // Injected rather than called here so neither this header nor the decision
    // TU depends on CUDA -- see serve/kv_vram_probe.h for the device side.
    using FreeVramFn = std::function<std::uint64_t()>;

    struct Config {
        int interval_secs = 0;      // 0 disables the thread (NINFER_FT_RELOAD_SECS)
        int full_attn_layers = 16;  // deep-protection range (NINFER_FT_FULL_ATTN_LAYERS)
        double deep_frac = 0.2;     // last 20% of full-attn layers keep nvfp4
        std::array<KvCacheStorage, kKvLayerStorageSlots> current_table{};
        bool current_table_explicit = false;

        // ---- gap 1: the free-VRAM axis (all four default to "inert") -------
        // Live free-VRAM probe; empty or a 0 return disarms the axis.
        FreeVramFn free_vram_bytes;
        // The free-VRAM baseline the axis is relative to, in bytes: a reading
        // taken while the process is already in its steady state (apps/serve
        // samples it after warmup). The axis is *relative* on purpose -- no
        // device total, reserve or headroom is baked into this layer, so the
        // rule survives a different GPU, a WSL GPU-PV allocation or a
        // concurrent process. 0 disarms the axis.
        std::uint64_t vram_reference_bytes = 0;
        // Device KV payload bytes of ONE full-attention layer at the resolved
        // capacity -- the byte -> layer bridge. The caller knows the geometry
        // (apps/serve derives it from MemorySummary::kv_payload_bytes), the
        // decision layer does not. 0 disarms the axis.
        std::uint64_t vram_layer_bytes = 0;
        // EWMA weight applied to the live reading before it is compared with
        // the baseline; 1.0 disables smoothing. Values outside (0,1] fall back
        // to the raw reading (never into a NaN decision).
        double vram_alpha = kFtVramDefaultAlpha;
        // Kill switch (NINFER_FT_VRAM_AXIS=0 / --ft-vram-axis off).
        bool vram_axis_enabled = true;
    };

    // Gap 2: the CLI layer of "CLI > env > default". An unset optional means
    // "defer to the environment (then to the default)", so a caller that passes
    // nothing gets today's behaviour unchanged.
    struct CliOverrides {
        std::optional<int> interval_secs;         // --kv-auto-relayout SECS (0 = off)
        std::optional<bool> vram_axis_enabled;    // --ft-vram-axis on|off
    };

    static Config from_env();
    // Environment layer + CLI layer on top (CLI wins). Same env names as
    // from_env(): NINFER_FT_RELOAD_SECS / NINFER_FT_FULL_ATTN_LAYERS /
    // NINFER_FT_DEEP_FRAC / NINFER_FT_VRAM_AXIS.
    static Config from_env(const CliOverrides& cli);

    KvAutoRelayout(Config config, ApplyFn apply);
    ~KvAutoRelayout();

    KvAutoRelayout(const KvAutoRelayout&) = delete;
    KvAutoRelayout& operator=(const KvAutoRelayout&) = delete;

    void start();
    void stop();

    // Test seam: run one decision cycle with a supplied energy table; returns
    // the spec that would be applied (empty when nothing changed). Single
    // threaded by contract (loop() is its one production caller): the VRAM axis
    // keeps its EWMA state here, so two concurrent callers are not supported.
    std::string decide_once(const std::vector<std::pair<int, double>>& energy);

    // Forensics for the cycle logging (gap 1): the shift the last decide_once()
    // used and the raw reading it came from. Valid after the first non-empty
    // decide_once() call; both are 0 while the axis is disarmed.
    int last_vram_shift() const noexcept { return last_vram_shift_; }
    std::uint64_t last_free_vram_bytes() const noexcept { return last_free_vram_bytes_; }

private:
    void loop();
    // Reads the probe, damps it, and converts the deviation from the baseline
    // into an rk4v4-band shift. Pure bookkeeping, no device call of its own.
    int observe_vram_shift();
    // True when the axis has everything it needs to act.
    bool vram_axis_armed() const noexcept;

    Config config_;
    ApplyFn apply_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::string pending_spec_;  // candidate seen in the previous cycle
    std::string applied_spec_;  // spec handed to apply_ successfully
    std::array<KvCacheStorage, kKvLayerStorageSlots> applied_table_{};
    double vram_ewma_ = 0.0;              // damped free-VRAM reading (bytes)
    int last_vram_shift_ = 0;             // forensics (gap 1)
    std::uint64_t last_free_vram_bytes_ = 0;
};

// Gap 1 rule, pure arithmetic (no I/O, no device) so it is unit-testable:
// how many whole layers the rk4v4 band boundary moves for a signed deviation of
// the (damped) free VRAM from its baseline.
//
//   positive shift = tight (free < baseline): a BIGGER rk4v4 band, because the
//     layers that enter it give up 1/18 of their device bytes each
//     (kFtNvfp4ToRk4v4SavingDenominator);
//   negative shift = loose (free > baseline): a SMALLER rk4v4 band, because a
//     surplus buys back the tier with the weakest long-context evidence.
//
// The deviation is floored to whole layers' worth of that saving, which IS the
// deadband: a deviation that cannot pay for one whole layer never moves the
// table, so allocator churn and the probe's own noise cannot flap the bands.
// The caller clamps the result against the band bounds and the rk4v4 exposure cap
// (build_ft_spec); this function reports the demand, not the landed value.
int ft_vram_shift_for(std::int64_t free_minus_reference_bytes, std::uint64_t layer_bytes);

// ft_tiers.py policy in C++: layers [deep_start, total) keep nvfp4; the rest are
// split by energy tertiles (lowest third -> rk4v4, middle -> iso4e, highest ->
// nvfp4); layers with no observation get the conservative iso4e.
//
// `rk4v4_shift` (gap 1, default 0) moves the rk4v4/iso4e boundary by whole layers: it
// is the only boundary the free-VRAM axis is allowed to touch (iso4e costs the
// same bytes as nvfp4, so the iso4e/nvfp4 boundary is byte-neutral and moving it
// would trade quality away for nothing). The band is clamped to
// [0, min(measured, kKvBitBudgetE8LayerLimit)] -- the cap is the measurement-
// backed rk4v4 exposure limit of product/kv_bit_budget.h:69-82 (the shipped policy
// places rk4v4 in 10 layers and is verified at 131072 tokens; all 16 on rk4v4 measures
// 0/27 needle). With rk4v4_shift == 0 the tertile expression is the pre-axis one,
// term for term.
std::string build_ft_spec(const std::vector<std::pair<int, double>>& energy,
                          int full_attn_layers, double deep_frac = 0.2,
                          int rk4v4_shift = 0);

// "0-11:rk4v4,12-15:nvfp4" formatting for a per-layer table (first `layers` slots).
std::string format_kv_table(const std::array<KvCacheStorage, kKvLayerStorageSlots>& table,
                            int layers);

} // namespace ninfer::serve
