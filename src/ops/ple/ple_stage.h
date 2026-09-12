#pragma once

// PLE stage contract: phase semantics, the runtime declaration, and the
// counters that make an n-gram gather observable.
//
// src/ops/ple/ple_table.{h,cu} can only gather "every column of the batch it is
// handed". Nothing in the tree decides WHICH columns a layer-sidecar stage
// actually owes, and nothing in the tree says whether a model owes anything at
// all. This header pins down the three answers so a future runtime cannot
// silently drift from them. See _collab/build/T4_ngram_gather.md.
//
// 1. PHASE SEMANTICS (ple_phase_window)
//    Prefill and Draft owe every column of their batch. Decode runs one column.
//    A speculative Verify pass runs K+1 columns but keeps only the LAST one:
//    every earlier proposal column is discarded before the residual reaches the
//    next layer, so gathering them is pure SSD traffic. Gathering the full batch
//    in Verify would cost K+1 times the faults for one column of useful data.
//
// 2. THE DECLARATION (PleStageDecl)
//    "This model has a PLE layer-sidecar at layer L, ngram_size=3,
//    heads_per_ngram=8, ple_embed_dim=2560" is DATA. It is already written down
//    in the arch-spec schema (tools/archkit/arch_spec.py, optional group "ple";
//    tools/archkit/gen_target.py:104-116 emits it as a PLEConfig stub that no
//    engine code consumes). PleStageDecl reads that same block at runtime, so
//    the engine never grows a per-model branch: an undeclared model answers
//    `present == false` and pays one bool test per declared-layer check.
//
// 3. THE OBSERVABLE (PleForensics)
//    A wired PLE stage that never gathers looks exactly like an unwired one from
//    the outside. These counters are the difference, and they are what an
//    acceptance run greps for ("forensics 标记非零"). Gated by NINFER_PLE_STATS=1,
//    matching the NINFER_HEADDBG / NINFER_KV_ROWSCALE convention.
//
// Host-only: no CUDA, no sidecar ownership. The device half stays in PleTable;
// this header is the vocabulary both halves share.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::ops::ple {

// The four ways the engine ever enters a layer. Draft is the proposal model's
// own pass over its block; Verify is the target model's pass over a proposed
// block (K+1 columns, last one kept).
enum class PlePhase : std::uint8_t { Prefill = 0, Decode = 1, Verify = 2, Draft = 3 };

[[nodiscard]] constexpr const char* ple_phase_name(PlePhase phase) noexcept {
    switch (phase) {
    case PlePhase::Prefill: return "prefill";
    case PlePhase::Decode: return "decode";
    case PlePhase::Verify: return "verify";
    case PlePhase::Draft: return "draft";
    }
    return "unknown";
}

// Half-open [first_token, first_token + tokens) window into a batch column set.
struct PlePhaseWindow {
    std::size_t first_token = 0;
    std::size_t tokens      = 0;
};

// The single place the phase -> column-set mapping lives. Callers hand a
// PlePhaseWindow to PleTable::gather_phase; they must not re-derive it.
[[nodiscard]] constexpr PlePhaseWindow ple_phase_window(PlePhase phase,
                                                        std::size_t n_tokens) noexcept {
    if (n_tokens == 0) { return PlePhaseWindow{0, 0}; }
    switch (phase) {
    case PlePhase::Prefill:
    case PlePhase::Draft: return PlePhaseWindow{0, n_tokens};
    case PlePhase::Decode:
    case PlePhase::Verify: return PlePhaseWindow{n_tokens - 1, 1};
    }
    return PlePhaseWindow{0, n_tokens};
}

// Host-side counters. Relaxed atomics because a fault can be satisfied from a
// prefetch thread while the gather thread is still deriving rows.
struct PleForensics {
    std::atomic<std::uint64_t> gathers{0};
    std::atomic<std::uint64_t> tokens_offered{0};  // columns the caller handed in
    std::atomic<std::uint64_t> tokens_gathered{0}; // columns the phase actually owes
    std::atomic<std::uint64_t> rows_derived{0};    // n_heads * tokens_gathered
    std::atomic<std::uint64_t> rows_resolved{0};   // row -> (file, offset) lookups
    std::atomic<std::uint64_t> rows_unique{0};     // distinct rows after dedup
    std::atomic<std::uint64_t> rows_reused{0};     // repeats served from dedup/LRU
    std::atomic<std::uint64_t> cache_hits{0};      // rows found resident
    std::atomic<std::uint64_t> cache_faults{0};    // rows read from the sidecar
    std::atomic<std::uint64_t> bytes_read{0};      // sidecar bytes actually pread
    std::atomic<std::uint64_t> gathers_by_phase[4]{};

    void reset() noexcept {
        gathers.store(0, std::memory_order_relaxed);
        tokens_offered.store(0, std::memory_order_relaxed);
        tokens_gathered.store(0, std::memory_order_relaxed);
        rows_derived.store(0, std::memory_order_relaxed);
        rows_resolved.store(0, std::memory_order_relaxed);
        rows_unique.store(0, std::memory_order_relaxed);
        rows_reused.store(0, std::memory_order_relaxed);
        cache_hits.store(0, std::memory_order_relaxed);
        cache_faults.store(0, std::memory_order_relaxed);
        bytes_read.store(0, std::memory_order_relaxed);
        for (auto& slot : gathers_by_phase) { slot.store(0, std::memory_order_relaxed); }
    }

    // True once a stage has demonstrably done work. This is the "非零" predicate.
    [[nodiscard]] bool observed() const noexcept {
        return gathers.load(std::memory_order_relaxed) != 0 &&
               rows_derived.load(std::memory_order_relaxed) != 0;
    }

    // One greppable line; the CLI's print_generation_summary style, but returned
    // as text so host-only callers (tools/, tests/) can print it too.
    [[nodiscard]] std::string to_line() const {
        const auto load = [](const std::atomic<std::uint64_t>& value) {
            return value.load(std::memory_order_relaxed);
        };
        std::string out = "ple";
        out += " gathers=" + std::to_string(load(gathers));
        out += " p=(" + std::to_string(load(gathers_by_phase[0])) + "," +
               std::to_string(load(gathers_by_phase[1])) + "," +
               std::to_string(load(gathers_by_phase[2])) + "," +
               std::to_string(load(gathers_by_phase[3])) + ")";
        out += " tokens=" + std::to_string(load(tokens_gathered)) + "/" +
               std::to_string(load(tokens_offered));
        out += " rows=" + std::to_string(load(rows_derived));
        out += " unique=" + std::to_string(load(rows_unique));
        out += " reused=" + std::to_string(load(rows_reused));
        out += " hits=" + std::to_string(load(cache_hits));
        out += " faults=" + std::to_string(load(cache_faults));
        out += " bytes=" + std::to_string(load(bytes_read));
        return out;
    }
};

// NINFER_PLE_STATS=1 turns the counters on for callers that would otherwise skip
// them entirely (the counters themselves are cheap, but a decode loop pays for
// the string only when asked).
[[nodiscard]] inline bool ple_stats_enabled() {
    static const bool enabled = [] {
        const char* raw = std::getenv("NINFER_PLE_STATS");
        if (raw == nullptr || raw[0] == '\0') { return false; }
        const std::string_view value(raw);
        return value != "0" && value != "off" && value != "false";
    }();
    return enabled;
}

// The data-driven sidecar declaration. This is the ONLY thing that decides
// whether a layer has a PLE stage, and it comes from data, never from a
// compiled-in per-model list.
struct PleStageDecl {
    bool present                 = false; // false => every accessor is a no-op
    std::uint32_t ngram_size     = 3;
    std::uint32_t heads_per_ngram = 8;
    std::uint32_t embed_dim      = 0;
    std::uint32_t eos_token_id   = 0;
    std::vector<std::uint32_t> layer_ids; // layer indices carrying a sidecar stage

    // n-gram level 2..ngram_size, heads_per_ngram heads per level. With
    // ngram_size=3 / heads_per_ngram=8 this is 16, which is exactly what
    // PleLayout::n_heads carries in a real ple-manifest.json.
    [[nodiscard]] std::uint32_t n_heads() const noexcept {
        return ngram_size < 2 ? 0 : (ngram_size - 1) * heads_per_ngram;
    }

    [[nodiscard]] bool declares_layer(std::uint32_t layer) const noexcept {
        for (const std::uint32_t id : layer_ids) {
            if (id == layer) { return true; }
        }
        return false;
    }

    // Structural checks that need no sidecar. Throws std::invalid_argument.
    void validate() const;

    // Geometry cross-check against a loaded sidecar (PleLayout::ngram_size /
    // n_heads / embedding_row_dimension). Throws std::runtime_error naming both
    // sides of the mismatch. Row dimension is passed as a plain integer so this
    // header never has to include ple_layout.h.
    void validate_against(std::uint32_t sidecar_ngram_size, std::uint32_t sidecar_heads_per_ngram,
                          std::uint32_t sidecar_n_heads, std::uint32_t sidecar_row_dim) const;

    // Reads the arch-spec "ple" block (tools/archkit/arch_spec.py schema; see
    // tools/archkit/specs/qwen4_exp_spec.json). A spec without a "ple" block is
    // a valid declaration of absence -> returns a default-constructed (absent)
    // decl. Malformed geometry throws; a missing file throws.
    //
    // Defined in ple_layout.cpp, which already pays for <nlohmann/json.hpp>, so
    // including this header never adds a JSON parse to a heavy TU.
    [[nodiscard]] static PleStageDecl from_spec_file(const std::string& spec_path);

    // Same, from already-read text. Useful for host-only tools and tests.
    [[nodiscard]] static PleStageDecl from_spec_text(const std::string& spec_text,
                                                      const std::string& origin);
};

} // namespace ninfer::ops::ple
