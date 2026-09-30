#pragma once

// qwen4_exp / THE PER-SEQUENCE HALF OF THE PLE PATH, and the ONE call site of
// ple_seam_gather_step.
//
// THE THREE ZERO-COUNT ITEMS THIS HEADER CLOSES (each one implemented, each one called by
// nothing in the engine before this):
//
//  1. detail::PleNgramWindow (impl/ple_ngram.h:48-91) -- the predecessor producer. Its own
//     lifecycle note reads "one instance per sequence (the n-gram window is per-sequence state,
//     not global): the engine's single-sequence text path has one, and future concurrent
//     sequences get one each" (ple_ngram.h:44-46). Nothing in the tree held one. Here it is:
//     windows_ has exactly one entry per sequence slot the engine can run, and the constructor
//     is the only construction site.
//
//  2. ple_seam_gather_step (impl/ple_seam_window.h:113-124) -- the production composition of
//     the window with the real-table gather. Nothing called it. gather_with_phase() below is
//     its single call site, reached from the four phase entry points.
//
//  3. the phase -> column mapping. THE CALL SITE DOES NOT COMPUTE IT. It passes a PlePhase and
//     the owed column set is taken inside the seam (ple_seam_plan -> ops::ple::ple_phase_window,
//     ple_stage.h:70) and again inside PleTable::gather_phase. There is no `n_tokens - 1` and no
//     comparison against a PlePhase anywhere in this file, on purpose: a second copy of that
//     mapping is the defect ple_seam_window.h:20-23 forbids in its own words ("the single place
//     that mapping lives; ... callers must not re-derive it"). dst_elems() below is the one
//     place a caller may ask how big the destination is, and it also goes through
//     ple_seam_dst_elems -> ple_seam_plan, so even the sizing question has one answer.
//
// LIFETIME AND OWNERSHIP, stated so a future caller cannot get it wrong:
//   * PleRuntime (the sidecar: 4 fds + a bounded pinned LRU, ple_table.h:35-38) is per ENGINE.
//     This class does not build one; it takes an already-attached one, because "PLE is off" is
//     spelled by the ABSENCE of this object (PleRuntime::attach returns nullptr), and a
//     constructor able to build its own sidecar would erase that distinction.
//   * PleNgramWindow is per SEQUENCE, and committing is a SEPARATE step from gathering: a
//     speculative Verify pass feeds K+1 columns and keeps only the last, so the rejected
//     proposal columns must never enter the history (ple_ngram.h:62-67). commit() is that
//     step, and nothing in this file calls it -- the caller does, after the sampler answers.
//   * The destination buffer is caller-owned (ple_seam_window.h:111-112) so a decode loop
//     allocates nothing per step; prevs_scratch_ is reused across steps for the same reason.

#include "ops/ple/ple_layout.h"
#include "ops/ple/ple_stage.h"
#include "targets/qwen4_exp/impl/ple_ngram.h"
#include "targets/qwen4_exp/impl/ple_runtime.h"
#include "targets/qwen4_exp/impl/ple_seam_window.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen4_exp::detail {

class PleSessionSet {
public:
    // == kMaximumConcurrency (include/ninfer/types.h:24) and == the bound registry.cpp's
    // validate_options() enforces. A set sized to anything else could not cover the engine.
    static constexpr std::size_t kMaximumSequences = 16;

    PleSessionSet(std::unique_ptr<PleRuntime> runtime, std::size_t sequences, std::int32_t eos)
        : runtime_(std::move(runtime)) {
        if (runtime_ == nullptr) {
            throw std::invalid_argument(
                "PLE session set needs an attached PleRuntime; PLE off is spelled by not "
                "constructing one at all (PleRuntime::attach returns nullptr)");
        }
        if (sequences == 0 || sequences > kMaximumSequences) {
            throw std::invalid_argument(
                "PLE session set: sequences must be in [1," +
                std::to_string(kMaximumSequences) + "] (" + std::to_string(sequences) +
                " given)");
        }
        const std::uint32_t ngram = runtime_->layout().ngram_size;
        windows_.reserve(sequences);
        for (std::size_t slot = 0; slot < sequences; ++slot) {
            windows_.emplace_back(ngram, eos); // ONE WINDOW PER SEQUENCE, built here only
        }
    }

    PleSessionSet(const PleSessionSet&) = delete;
    PleSessionSet& operator=(const PleSessionSet&) = delete;

    [[nodiscard]] std::size_t sequences() const noexcept { return windows_.size(); }
    [[nodiscard]] std::uint32_t ngram_size() const noexcept {
        return runtime_->layout().ngram_size;
    }

    // ---- the four phases. Each one names a PlePhase and nothing else; the column set it owes
    // is taken from ple_phase_window inside the seam. Prefill and Draft owe every column,
    // Decode and Verify owe the last one -- and none of that is restated here.
    void prefill_step(std::size_t slot, std::span<const std::int32_t> tokens, void* dst,
                      cudaStream_t stream) {
        gather_with_phase(ops::ple::PlePhase::Prefill, slot, tokens, dst, stream);
    }
    void decode_step(std::size_t slot, std::span<const std::int32_t> tokens, void* dst,
                     cudaStream_t stream) {
        gather_with_phase(ops::ple::PlePhase::Decode, slot, tokens, dst, stream);
    }
    void verify_step(std::size_t slot, std::span<const std::int32_t> tokens, void* dst,
                     cudaStream_t stream) {
        gather_with_phase(ops::ple::PlePhase::Verify, slot, tokens, dst, stream);
    }
    void draft_step(std::size_t slot, std::span<const std::int32_t> tokens, void* dst,
                    cudaStream_t stream) {
        gather_with_phase(ops::ple::PlePhase::Draft, slot, tokens, dst, stream);
    }

    // The number of BF16 elements the last step for `phase` wrote into dst: plan.columns *
    // n_heads * row_dim (a Verify caller that passed K+1 columns must read dst as
    // [n_heads * row_dim, 1]). One lookup, through the same plan the gather used.
    [[nodiscard]] std::size_t dst_elems(ops::ple::PlePhase phase, std::size_t n_tokens) const noexcept {
        return ple_seam_dst_elems(phase, runtime_->layout().ngram_size,
                                  runtime_->layout().n_heads,
                                  runtime_->layout().embedding_row_dimension, n_tokens);
    }

    // The acceptance side. Deliberately NOT folded into the step calls: only the columns the
    // sampler actually kept may become history.
    void commit(std::size_t slot, std::span<const std::int32_t> tokens) {
        window(slot).commit(tokens);
    }
    void reset(std::size_t slot) { window(slot).reset(); }
    [[nodiscard]] std::size_t committed(std::size_t slot) const { return window(slot).committed(); }

    // The acceptance observable (ple_stage.h:110-115): a wired stage that never gathers is
    // indistinguishable from an unwired one from the outside.
    [[nodiscard]] const ops::ple::PleForensics& forensics() const noexcept {
        return runtime_->forensics();
    }
    [[nodiscard]] std::size_t gathers() const noexcept {
        return runtime_->forensics().gathers.load(std::memory_order_relaxed);
    }

private:
    [[nodiscard]] PleNgramWindow& window(std::size_t slot) {
        if (slot >= windows_.size()) {
            throw std::out_of_range("PLE session set: sequence slot " + std::to_string(slot) +
                                    " of " + std::to_string(windows_.size()) +
                                    " (one window per sequence)");
        }
        return windows_[slot];
    }
    [[nodiscard]] const PleNgramWindow& window(std::size_t slot) const {
        return const_cast<PleSessionSet*>(this)->window(slot);
    }

    // ---- THE call site of ple_seam_gather_step. One, shared by all four phases, because a
    // second copy of it is how a phase would come to have its own window rule. The eos comes
    // from the window that owns the history it is about.
    void gather_with_phase(ops::ple::PlePhase phase, std::size_t slot,
                           std::span<const std::int32_t> tokens, void* dst,
                           cudaStream_t stream) {
        if (tokens.empty()) { return; }
        ple_seam_gather_step(*runtime_, window(slot), tokens, phase, window(slot).eos(), dst,
                             stream, prevs_scratch_);
    }

    std::unique_ptr<PleRuntime> runtime_;     // the sidecar: one per engine, injected
    std::vector<PleNgramWindow> windows_;     // ONE PER SEQUENCE
    std::vector<std::int32_t> prevs_scratch_; // reused: a decode loop allocates nothing per step
};

} // namespace ninfer::targets::qwen4_exp::detail
