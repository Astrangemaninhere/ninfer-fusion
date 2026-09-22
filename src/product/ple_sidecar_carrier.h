#pragma once

// NGRAM-BUILD / N3: the n-gram (PLE) table's CARRIER layer — which medium holds the
// 95 GiB table, and how that choice is budgeted.
//
// WHY THIS FILE EXISTS
//   The table's carrier has been implicit: src/ops/ple/ple_table.{h,cu} always assumes
//   one carrier (SSD sidecar + bounded pinned page cache), and nothing in the tree can
//   express "keep this one in memory instead". The user's rule for FlashNext is a
//   two-part decision, verbatim:
//       "检测到 flashnext 就默认 ngram 丢 ssd，但到时候 gui 允许不丢 ssd 保留在内存里"
//   i.e. (a) detection of the FlashNext target is the TRIGGER, (b) the DEFAULT it selects
//   is SSD, (c) an explicit override (the GUI, later) may force memory. That is a policy
//   over a small carrier set, not a compile-time choice, so it gets its own carrier
//   header instead of a per-model branch in the engine.
//
// SHAPE: the same 判据 / 载体 / 预算 triple the cold-KV tier already uses, because this is
// the same problem (a big host/SSD-resident payload paged into a bounded pinned pool):
//   * src/product/kv_tier_formats.h:53 kv_layer_class_cold_capable + :73
//     kv_cold_pool_reachable — the CLASS predicate, kept separate from the policy so the
//     "can this tree serve it" question never hides inside a "should it".
//   * src/product/kv_cold_tier_budget.h — the pure, host-only, unit-testable budget
//     ladder, and its doctrine: an unmet budget is a CAPACITY EVENT, counted and
//     reported, never thrown (throwing turns memory pressure into a dead server); the
//     only hard errors are STARTUP CONTRADICTIONS.
//   * src/product/weight_residency.h:115-123 — the conclusion that budgets must stay
//     SEPARATE per consumer (KV cold pages age per request; weight spans are static
//     after load). The n-gram table's pinned page cache is a THIRD lifetime (per-token
//     random access, fixed after load), so it gets its own budget rather than sharing.
//
// SCOPE OF THIS HEADER (N3)
//   * The memory side is IMPLEMENTED (the decision ladder, the budget arithmetic, the
//     startup contradiction).
//   * The SSD side is NAMED, not implemented: ple_ssd_backend_linked() is the hook and
//     ple_ssd_backend_requirement() says exactly what an implementer must add. Because
//     the backend is absent today, the SSD default resolves to memory WITH A RECORDED
//     REASON — never silently. That gap is deliberate and is why the default the user
//     asked for (SSD) is not yet the default the engine gets.
//
// Deliberately pure and host-only: no CUDA, no engine headers, unit-testable with plain
// g++ (tests/test_ple_carrier.cpp), matching tests/test_kv_cold_tier_budget.cpp.

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninfer::product {

// The carrier classes. Named after the cold-KV tier vocabulary (a hot/host medium and a
// disk medium) rather than inventing a second one.
enum class PleCarrierClass : std::uint8_t {
    MemoryResident = 0, // whole table (or its rows) stays in host memory
    SsdPaged       = 1, // table stays on SSD; touched rows fault into the pinned cache
};

[[nodiscard]] inline const char* ple_carrier_name(PleCarrierClass cls) noexcept {
    switch (cls) {
    case PleCarrierClass::MemoryResident: return "memory";
    case PleCarrierClass::SsdPaged: return "ssd";
    }
    return "?";
}

// ---------------------------------------------------------------------------------
// 判据 (predicate)
// ---------------------------------------------------------------------------------

// THE SSD HOOK. Returns false in this tree: no SSD carrier backend is linked. This
// function is the named seam an implementer fills in, and the reason the user's SSD
// default cannot be honoured yet. An SSD carrier needs all of:
//   1. a resident-set budget for the SSD path (PleCarrierBudget::ssd_bytes) that the
//      eviction policy can enforce, mirroring the cold-KV disk tier's cap-plus-working-set
//      pair (kv_cold_tier_budget.h:32-34) instead of the single pinned cap;
//   2. a page-granular fault path in ops/ple/ple_table.cu — the existing fault_in()
//      (ple_table.cu:126) is already 4096-aligned and LRU-bounded, so the work is the
//      admission ladder (memory-first, like the KV ladder), not a new reader;
//   3. a prefetch worker (PleTableOptions::prefetch_workers defaults to 16 and is
//      currently unused — declared at ple_table.h:31, read by nothing);
//   4. the option surface: PleTableOptions/PleRuntime::Options gain the carrier class,
//      and the serve CLI gains the carrier flag next to the sidecar root
//      (--ple-sidecar; the flag itself does not exist yet either).
[[nodiscard]] inline bool ple_ssd_backend_linked() noexcept { return false; }

// What is missing, as one line a report or a CLI error can print verbatim.
[[nodiscard]] inline std::string ple_ssd_backend_requirement() {
    return "no SSD carrier backend is linked (ops/ple/ple_table.cu fault path is the only "
           "reader today; needs an ssd_bytes budget, the memory-first admission ladder, "
           "the unused prefetch_workers pool, and the --ple-carrier option surface)";
}

// The CLASS-level admission: can this tree actually serve the carrier class at all? Kept
// separate from the policy request so "should" never answers "can".
[[nodiscard]] inline bool ple_carrier_class_servable(PleCarrierClass cls) noexcept {
    switch (cls) {
    case PleCarrierClass::MemoryResident: return true;
    case PleCarrierClass::SsdPaged: return ple_ssd_backend_linked();
    }
    return false;
}

// ---------------------------------------------------------------------------------
// 预算 (budget) — the THIRD budget, deliberately not shared (see file header)
// ---------------------------------------------------------------------------------
struct PleCarrierBudget {
    // Pinned host bytes for the n-gram page cache. Default matches PleTableOptions
    // (ple_table.h:30, 512 MiB) so the header cannot drift from the op.
    std::uint64_t pinned_bytes = 512ULL << 20;
    // SSD-side resident-set cap. 0 until the backend exists; the memory-first ladder
    // then has nowhere to demote to, which is exactly why the downgrade is recorded.
    std::uint64_t ssd_bytes = 0;
};

// ---------------------------------------------------------------------------------
// 载体 (carrier): the request -> resolution ladder
// ---------------------------------------------------------------------------------
struct PleCarrierRequest {
    // (a) THE TRIGGER. Set when the artifact identity is the FlashNext target
    // (qwen4_exp), which is the only model with a PLE n-gram sidecar today.
    bool flashnext_detected = false;
    // (c) THE OVERRIDE. The GUI/CLI wins over the default, per the user's rule.
    std::optional<PleCarrierClass> explicit_override;
};

struct PleCarrierResolution {
    PleCarrierClass intended = PleCarrierClass::MemoryResident;
    PleCarrierClass resolved = PleCarrierClass::MemoryResident;
    bool downgraded          = false;
    std::string reason; // non-empty iff downgraded

    [[nodiscard]] bool served() const noexcept { return !downgraded; }

    // One greppable line (same convention as PleForensics::to_line).
    [[nodiscard]] std::string to_line() const {
        std::string out = "ple-carrier intended=";
        out += ple_carrier_name(intended);
        out += " resolved=";
        out += ple_carrier_name(resolved);
        out += downgraded ? " DOWNGRADED: " : " ok";
        out += reason;
        return out;
    }
};

// The rule, in one place: override wins; otherwise the FlashNext trigger selects SSD and
// everything else stays memory.
//
// NEVER THROWS. Following the cold-tier precedent, an unavailable carrier is a capacity
// event that is RECORDED and reported; the throwing path is startup validation, below.
[[nodiscard]] inline PleCarrierResolution ple_carrier_resolve(const PleCarrierRequest& req) {
    PleCarrierResolution out;
    out.intended = req.explicit_override.value_or(
        req.flashnext_detected ? PleCarrierClass::SsdPaged : PleCarrierClass::MemoryResident);
    out.resolved = out.intended;
    if (!ple_carrier_class_servable(out.intended)) {
        // The only demotion target that exists is memory. Record why, loudly: a silent
        // fallback here would be the "flag that silently keeps everything resident"
        // failure mode weight_residency.h:111-113 calls out by name.
        out.resolved   = PleCarrierClass::MemoryResident;
        out.downgraded = true;
        out.reason     = std::string("intended ") + ple_carrier_name(out.intended) +
                     " but " + ple_ssd_backend_requirement();
    }
    return out;
}

// Can the resolved carrier's budget actually hold the payload? The memory side is
// implemented here (pure arithmetic); the SSD side is unbounded from this header's point
// of view (its working set is the backend's business).
[[nodiscard]] inline std::optional<std::string> ple_carrier_budget_verdict(
    const PleCarrierResolution& resolution, const PleCarrierBudget& budget,
    std::uint64_t table_bytes) {
    if (resolution.resolved == PleCarrierClass::MemoryResident) {
        if (budget.pinned_bytes < table_bytes) {
            return std::string("memory-resident table needs ") + std::to_string(table_bytes) +
                   " bytes but the pinned budget is " + std::to_string(budget.pinned_bytes) +
                   " (ple_table.h:30 default is a page cache, not a whole-table pin)";
        }
        return std::nullopt;
    }
    return std::nullopt;
}

// STARTUP CONTRADICTION CHECK — the one place this header may throw.
//
// Doctrine (kv_cold_tier_budget.h:35-38): budgets are met or reported; only startup
// contradictions are hard errors. A carrier the tree cannot serve, selected by an
// EXPLICIT override the operator typed, is a contradiction: the operator asked for
// something this binary cannot do, and honouring it silently would mean the flag lied.
// An UNSERVABLE DEFAULT is not a contradiction — it is the recorded downgrade, because
// nobody asked for it by name.
[[nodiscard]] inline PleCarrierResolution ple_carrier_validate_startup(
    const PleCarrierRequest& req, const PleCarrierBudget& budget, std::uint64_t table_bytes) {
    PleCarrierResolution out = ple_carrier_resolve(req);
    if (out.downgraded) {
        if (req.explicit_override.has_value()) {
            throw std::invalid_argument("PLE carrier explicitly requested as '" +
                                        std::string(ple_carrier_name(*req.explicit_override)) +
                                        "' cannot be served: " + out.reason);
        }
    }
    if (const auto verdict = ple_carrier_budget_verdict(out, budget, table_bytes)) {
        throw std::invalid_argument("PLE carrier contradiction: " + *verdict);
    }
    return out;
}

} // namespace ninfer::product
