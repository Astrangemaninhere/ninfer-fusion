#pragma once

// W13 P1 fetch schedule: WHEN a weight span is brought back, and what it costs.
//
// The transfer math is the whole justification of W13, so it is a pure function
// here rather than a comment somewhere.
//
// Per decode round the engine reads every resident weight once (measured: 19.0 GB
// of weights for the 27B artifact, VRAM.md). Per decode round it also reads the
// whole KV of the sequence. Their ratio is what decides whether pushing weights
// off the device is a win:
//
//   * KV bytes/round  grows LINEARLY with context.
//   * weight bytes/round is CONSTANT.
//
// So the honest statement is a threshold, not a slogan:
//
//   offloading is worthwhile  <=>  time_saved(KV stays device-resident)
//                                   >  offloaded_bytes / pcie_bandwidth
//
// The saved side only exists when the alternative is NOT "resident" but "the
// request does not fit at all". At short context the resident alternative is
// strictly better (the weights are already on HBM at ~1.8 TB/s); at 1M the
// request does not fit without moving something, and THAT is W13's regime.
//
// Everything below is dependency-free (no CUDA, no ninfer/types.h) so it is
// unit-testable with plain g++.

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::product {

// One capacity observation of the engine: "at `tokens` context, the engine needs
// exactly `bytes` of device memory for KV when the plan averages `bits_per_element`
// per KV element". Two of these determine the affine model.
struct KvCapacityPoint {
    std::uint64_t tokens = 0;
    std::uint64_t bytes  = 0;
    double bits_per_element = 0.0;
};

// KV device bytes = bytes_per_token_bit * tokens * bits + constant_bytes.
// Solved exactly from two points; no fitted-looking constants are typed here.
struct KvCapacityFit {
    double bytes_per_token_bit = 0.0;
    double constant_bytes      = 0.0;

    [[nodiscard]] double bytes_for(std::uint64_t tokens, double bits) const noexcept {
        return bytes_per_token_bit * static_cast<double>(tokens) * bits + constant_bytes;
    }
    // The average bits/element the SAME engine can afford in `available_bytes`.
    [[nodiscard]] double bits_within(std::uint64_t tokens,
                                     std::uint64_t available_bytes) const noexcept {
        const double span = bytes_per_token_bit * static_cast<double>(tokens);
        if (span <= 0.0) { return 0.0; }
        return (static_cast<double>(available_bytes) - constant_bytes) / span;
    }
};

// Exact two-point solve. The two points must differ in bits; identical bits are a
// contradiction (one bits value cannot need two byte counts) and throw.
[[nodiscard]] inline KvCapacityFit kv_capacity_fit_from(const KvCapacityPoint& a,
                                                        const KvCapacityPoint& b) {
    if (a.tokens == 0 || b.tokens == 0) {
        throw std::invalid_argument("KV capacity fit needs a non-zero token count");
    }
    const double bits_span = a.bits_per_element - b.bits_per_element;
    if (bits_span == 0.0) {
        throw std::invalid_argument("KV capacity fit needs two different bit budgets");
    }
    KvCapacityFit fit;
    fit.bytes_per_token_bit =
        (static_cast<double>(a.bytes) - static_cast<double>(b.bytes)) / bits_span /
        static_cast<double>(a.tokens);
    fit.constant_bytes = static_cast<double>(a.bytes) -
                         fit.bytes_per_token_bit * static_cast<double>(a.tokens) *
                             a.bits_per_element;
    return fit;
}

// The balance sheet of ONE offload decision, so "it will not be slow" is a
// computable claim instead of a hope. Pure, and UNPRINTED: the `--weight-report`
// flag this comment used to name does not exist anywhere in the tree (checked with
// git grep --untracked), and the only exerciser of these three functions is
// tests/test_weight_residency.cpp. What is open here is a PRINTER -- a verdict line
// at load time, or an actual --weight-report -- not more math.
struct WeightOffloadVerdict {
    double offloaded_bytes_per_round = 0.0;
    double pcie_seconds_per_round    = 0.0;  // offloaded_bytes_per_round / bandwidth
    double resident_seconds_per_round = 0.0; // the same bytes from HBM
    double compute_seconds_per_round  = 0.0; // KV attention + everything else
    double net_seconds_per_round      = 0.0; // pcie - resident, added to compute
    double slowdown_ratio             = 1.0; // (compute + net) / compute
    bool   net_positive               = false;
    std::string reason;

    [[nodiscard]] std::string describe() const {
        std::string out = "w13 verdict: offloaded/round=";
        out += std::to_string(offloaded_bytes_per_round);
        out += " B pcie=" + std::to_string(pcie_seconds_per_round) + " s";
        out += " hbm=" + std::to_string(resident_seconds_per_round) + " s";
        out += " compute=" + std::to_string(compute_seconds_per_round) + " s";
        out += " slowdown=" + std::to_string(slowdown_ratio) + "x";
        out += net_positive ? " NET+" : " NET-";
        out += " (" + reason + ")";
        return out;
    }
};

// `hbm_bytes_per_second` is the device bandwidth the offloaded bytes would have
// enjoyed had they stayed resident; both bandwidths are operator inputs (measured
// or assumed) so this function never invents a hardware number.
[[nodiscard]] inline WeightOffloadVerdict
weight_offload_verdict(double offloaded_bytes_per_round, double pcie_bytes_per_second,
                       double hbm_bytes_per_second, double compute_seconds_per_round) {
    if (!(pcie_bytes_per_second > 0.0) || !(hbm_bytes_per_second > 0.0)) {
        throw std::invalid_argument("weight offload verdict needs positive bandwidths");
    }
    WeightOffloadVerdict v;
    v.offloaded_bytes_per_round  = offloaded_bytes_per_round;
    v.pcie_seconds_per_round     = offloaded_bytes_per_round / pcie_bytes_per_second;
    v.resident_seconds_per_round = offloaded_bytes_per_round / hbm_bytes_per_second;
    v.compute_seconds_per_round  = compute_seconds_per_round;
    v.net_seconds_per_round      = v.pcie_seconds_per_round - v.resident_seconds_per_round;
    const double baseline        = compute_seconds_per_round + v.resident_seconds_per_round;
    v.slowdown_ratio = baseline > 0.0 ? (baseline + v.net_seconds_per_round) / baseline : 1.0;
    // "Net positive" is stated only when the offload does not dominate the round.
    // A 10x round is still a legitimate trade when the alternative is OOM, so the
    // caller decides; this flag is the operator-facing warning, not a veto.
    v.net_positive = v.net_seconds_per_round <= compute_seconds_per_round;
    v.reason       = v.net_positive ? "offload transfer fits inside the round's compute"
                                    : "offload transfer dominates the round (only worth it "
                                      "to make an otherwise-impossible context run)";
    return v;
}

// The final question the deliverable has to answer: how much device memory does the
// offload free, and what does that buy in KV bits/element at a target context.
struct WeightOffloadKvHeadroom {
    std::uint64_t device_bytes_freed = 0;
    double bits_before               = 0.0;
    double bits_after                = 0.0;
    double bits_gained               = 0.0;

    [[nodiscard]] std::string describe() const {
        return "w13 kv headroom: freed=" + std::to_string(device_bytes_freed) +
               " B bits/el " + std::to_string(bits_before) + " -> " +
               std::to_string(bits_after) + " (+" + std::to_string(bits_gained) + ")";
    }
};

[[nodiscard]] inline WeightOffloadKvHeadroom
weight_offload_kv_headroom(const KvCapacityFit& fit, std::uint64_t tokens,
                           std::uint64_t available_bytes, std::uint64_t device_bytes_freed) {
    WeightOffloadKvHeadroom h;
    h.device_bytes_freed = device_bytes_freed;
    h.bits_before        = fit.bits_within(tokens, available_bytes);
    h.bits_after = fit.bits_within(tokens, available_bytes + device_bytes_freed);
    h.bits_gained = h.bits_after - h.bits_before;
    return h;
}

} // namespace ninfer::product
