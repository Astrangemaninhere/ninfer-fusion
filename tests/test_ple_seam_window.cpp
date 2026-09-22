// tests/test_ple_seam_window.cpp — host-only acceptance for the PLE seam composition
// (src/targets/qwen4_exp/impl/ple_seam_window.h).
//
// NO GPU, NO CUDA, NO SIDECAR. That is possible because ple_seam_window.h takes the
// CUDA half as a template parameter, so the WIRING (not the kernel) is provable on a
// plain host compiler. The GPU half of the same composition is instantiated for real in
// tests/ops/ple_seam_gather_e2e_test.cu.
//
// What this file is FOR: the N6 gap is that the n-gram window and the PLE seam both
// existed with nothing joining them, and the two in-tree call sites hand-filled
// `prevs = {eos, eos}`, so the bigram/trigram mixers never saw a real predecessor. A
// test that only checked "a gather happened" would stay green if the window were ripped
// out again. So every check here is paired with the control that must go red:
//
//   | # | check                                              | control that makes it RED |
//   |---|----------------------------------------------------|---------------------------|
//   | 1 | window prevs change the derived rows               | replace the window call with an eos fill -> 0 rows differ |
//   | 2 | phase -> owed columns (Decode 1, Prefill/Draft T)  | ple_phase_window always returning the whole batch -> Decode claims T |
//   | 3 | prevs layout is nearest-first and order-bearing    | swap the two predecessors -> rows change |
//   | 4 | a real EOS predecessor equals the eos fill exactly | re-implement the cut in the seam -> rows diverge |
//   | 5 | deriving does NOT commit; step() commit does       | make the seam commit -> Verify would commit rejects |
//   | 6 | the composed step hands the RUNTIME the window prevs| the runtime sees {eos, eos} instead of {t_a, eos} |
//
// Check 6 is the load-bearing one for the wiring itself: it inspects the arguments that
// actually reach gather_phase.
//
// ⚠ DOC/CODE MISMATCH found while writing this test (reported, not papered over):
// ple_layout.h:75-79, ple_ngram.h and ple_runtime.h:32 all describe `prevs` as
// "oldest first". The code is NEAREST-first: PleNgramWindow::fill_prevs maps s=0 to the
// most recently committed token, and PleLayout::derive_rows_one maps prevs[s-1] to
// ctx[s] (so prevs[0] -> ctx[1] = 1-back). tools/ple_reference.py derives the same way,
// so producer and consumer AGREE and the prose is the outlier. This test asserts the
// code's convention because that is what the sidecar/checkpoint were built against;
// flipping the prose is a doc fix, flipping the code would change model numerics.

#include "targets/qwen4_exp/impl/ple_seam_window.h"

#include <cstdio>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace {

using ninfer::ops::ple::PleLayout;
using ninfer::ops::ple::PlePhase;
using ninfer::targets::qwen4_exp::detail::PleNgramWindow;
using ninfer::targets::qwen4_exp::detail::PleSeamPlan;

constexpr std::int32_t kEos = 151645; // Qwen3.8 vocabulary EOS (matches the .cu tests)
constexpr std::int32_t kTa  = 162042;
constexpr std::int32_t kTb  = 9707;
constexpr int kHeads        = 16;
constexpr int kRowDim       = 160;

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    } else {
        std::printf("ok: %s\n", what);
    }
}

// The fixture geometry of tools/ple_reference.py gen (per-head vocab 2000, offsets
// h*2000), built by hand so this test needs no sidecar, no JSON and no python.
PleLayout make_layout() {
    PleLayout layout;
    layout.ngram_size              = 3;
    layout.heads_per_ngram         = 8;
    layout.n_heads                 = kHeads;
    layout.embedding_row_dimension = kRowDim;
    layout.row_stride_bytes        = 2 * kRowDim;
    layout.per_head_vocab_sizes.assign(kHeads, 2000);
    layout.per_head_offsets.resize(kHeads);
    for (int h = 0; h < kHeads; ++h) {
        layout.per_head_offsets[static_cast<std::size_t>(h)] =
            static_cast<std::uint32_t>(2000 * h);
    }
    return layout;
}

// Derives with an explicit prevs fill (the old hand-filled way, for contrast).
std::vector<std::int32_t> derive_with_fill(const PleLayout& layout,
                                           const std::vector<std::int32_t>& tokens,
                                           const std::vector<std::int32_t>& prevs) {
    std::vector<std::int32_t> rows(tokens.size() * kHeads);
    layout.derive_rows(std::span<const std::int32_t>(tokens),
                       std::span<const std::int32_t>(prevs), kEos, rows.data());
    return rows;
}

// Derives through the seam (window-driven), for the columns `phase` owes.
std::vector<std::int32_t> derive_with_seam(const PleLayout& layout,
                                           const PleNgramWindow& window,
                                           const std::vector<std::int32_t>& tokens,
                                           PlePhase phase,
                                           std::vector<std::int32_t>& scratch) {
    const PleSeamPlan plan =
        ninfer::targets::qwen4_exp::detail::ple_seam_plan(phase, layout.ngram_size,
                                                          tokens.size());
    std::vector<std::int32_t> rows(plan.columns * kHeads);
    ninfer::targets::qwen4_exp::detail::ple_seam_derive_rows(
        layout, window, std::span<const std::int32_t>(tokens), phase, kEos, scratch,
        rows.data());
    return rows;
}

std::size_t count_differing(const std::vector<std::int32_t>& a,
                            const std::vector<std::int32_t>& b) {
    const std::size_t n = a.size() < b.size() ? a.size() : b.size();
    std::size_t differ = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) { ++differ; }
    }
    return differ;
}

// The duck-typed runtime the composition drives. Records the exact arguments that
// gather_phase receives, which is where a missing window shows up.
struct RecordingRuntime {
    struct Call {
        std::vector<std::int32_t> tokens;
        std::vector<std::int32_t> prevs;
        std::int32_t eos          = 0;
        PlePhase phase            = PlePhase::Prefill;
        void* dst                 = nullptr;
        int stream                = 0;
    };
    mutable std::vector<Call> calls;

    void gather_phase(std::span<const std::int32_t> tokens,
                      std::span<const std::int32_t> prevs, std::int32_t eos, PlePhase phase,
                      void* dst, int stream) const {
        calls.push_back(Call{{tokens.begin(), tokens.end()}, {prevs.begin(), prevs.end()},
                             eos, phase, dst, stream});
    }
};

bool throws_invalid_argument(const std::function<void()>& body) {
    try {
        body();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

} // namespace

int main() {
    using namespace ninfer::targets::qwen4_exp::detail;
#ifdef NGRAM_BUILD_MUTANT
    // The mutation-testing controls compile a MUTANT copy of ple_seam_window.h ahead of
    // the tree's on the include path (-I order). Printing the marker makes a mutant run
    // self-identifying in the log, so "the mutation was never applied" cannot be mistaken
    // for "the test passed".
    std::printf("MUTANT BUILD: %s\n", NGRAM_BUILD_MUTANT);
#endif
    const PleLayout layout = make_layout();

    // ---- 1. The window changes the rows (the whole point of not hand-filling prevs).
    {
        PleNgramWindow window(3, kEos);
        window.commit(std::span<const std::int32_t>(&kTa, 1)); // one committed token
        std::vector<std::int32_t> scratch;
        const std::vector<std::int32_t> seam_rows =
            derive_with_seam(layout, window, {kTb}, PlePhase::Decode, scratch);
        const std::vector<std::int32_t> eos_rows =
            derive_with_fill(layout, {kTb}, {kEos, kEos});
        const std::size_t differ = count_differing(seam_rows, eos_rows);
        std::printf("seam vs eos-fill rows differ in %zu/%zu head ids\n", differ,
                    seam_rows.size());
        expect(differ > 0,
               "window prevs change the derived rows (control: eos fill -> 0 differ)");
        expect(scratch.size() == 2 && scratch[0] == kTa && scratch[1] == kEos,
               "prevs for a 1-deep history are {nearest committed, eos}");
    }

    // ---- 2. Phase -> owed columns, and the column mapping is consistent.
    {
        const PleSeamPlan decode  = ple_seam_plan(PlePhase::Decode, 3, 3);
        const PleSeamPlan prefill = ple_seam_plan(PlePhase::Prefill, 3, 3);
        const PleSeamPlan verify  = ple_seam_plan(PlePhase::Verify, 3, 3);
        const PleSeamPlan draft   = ple_seam_plan(PlePhase::Draft, 3, 3);
        const PleSeamPlan single  = ple_seam_plan(PlePhase::Decode, 3, 1);
        const PleSeamPlan empty   = ple_seam_plan(PlePhase::Decode, 3, 0);
        expect(decode.columns == 1 && decode.first_column == 2,
               "Decode owes exactly the last column");
        expect(prefill.columns == 3 && prefill.first_column == 0 && draft.columns == 3,
               "Prefill/Draft owe every column");
        expect(verify.columns == 1 && verify.first_column == 2,
               "Verify owes only the last column (K+1 gathered = K+1x faults for 1 column)");
        expect(single.columns == 1 && single.first_column == 0,
               "a 1-column Decode batch owes that column");
        expect(empty.columns == 0, "an empty batch owes nothing");
        expect(decode.prevs_per_token == 2, "each owed column carries ngram_size-1 prevs");

        // Prefill's last column == Decode's only column, for the same window.
        PleNgramWindow window(3, kEos);
        std::vector<std::int32_t> scratch;
        const std::vector<std::int32_t> prefill_rows =
            derive_with_seam(layout, window, {11, 22, 33}, PlePhase::Prefill, scratch);
        const std::vector<std::int32_t> decode_rows =
            derive_with_seam(layout, window, {11, 22, 33}, PlePhase::Decode, scratch);
        const std::vector<std::int32_t> prefill_last(prefill_rows.end() - kHeads,
                                                     prefill_rows.end());
        expect(decode_rows == prefill_last,
               "Decode's column == Prefill's last column (same window)");
    }

    // ---- 3. prevs layout is nearest-first, and the order is load-bearing.
    {
        PleNgramWindow window(3, kEos);
        const std::int32_t older = 101;
        const std::int32_t newer = 202;
        window.commit(std::span<const std::int32_t>(&older, 1));
        window.commit(std::span<const std::int32_t>(&newer, 1));
        std::vector<std::int32_t> prevs(2);
        window.fill_prevs(std::span<const std::int32_t>(&kTb, 1),
                          std::span<std::int32_t>(prevs));
        std::printf("prevs after committing {%d,%d}: [%d,%d]\n", older, newer, prevs[0],
                    prevs[1]);
        expect(prevs[0] == newer && prevs[1] == older,
               "prevs[0] is the NEAREST predecessor (code convention, see file header)");

        std::vector<std::int32_t> scratch;
        const std::vector<std::int32_t> ordered =
            derive_with_seam(layout, window, {kTb}, PlePhase::Decode, scratch);
        const std::vector<std::int32_t> swapped =
            derive_with_fill(layout, {kTb}, {prevs[1], prevs[0]});
        expect(count_differing(ordered, swapped) > 0,
               "swapping the two predecessors changes the rows (order is load-bearing)");
    }

    // ---- 4. A REAL eos predecessor passes through and equals the eos fill exactly.
    {
        PleNgramWindow window(3, kEos);
        window.commit(std::span<const std::int32_t>(&kEos, 1));
        std::vector<std::int32_t> scratch;
        const std::vector<std::int32_t> seam_rows =
            derive_with_seam(layout, window, {kTb}, PlePhase::Decode, scratch);
        const std::vector<std::int32_t> eos_rows =
            derive_with_fill(layout, {kTb}, {kEos, kEos});
        expect(seam_rows == eos_rows,
               "history ending in a real eos == the eos fill (seam does not re-cut)");
    }

    // ---- 5. Deriving does not commit; step() does (Verify must not commit rejects).
    {
        PleNgramWindow window(3, kEos);
        window.commit(std::span<const std::int32_t>(&kTa, 1));
        const std::size_t before = window.committed();
        std::vector<std::int32_t> scratch;
        (void)derive_with_seam(layout, window, {kTb, kTa}, PlePhase::Verify, scratch);
        expect(window.committed() == before, "Verify-style derive leaves the history alone");
        std::vector<std::int32_t> prevs(2);
        window.step(std::span<const std::int32_t>(&kTb, 1), std::span<std::int32_t>(prevs));
        expect(window.committed() == 2, "step() commits (history grows to ngram_size-1)");
        window.step(std::span<const std::int32_t>(&kTa, 1), std::span<std::int32_t>(prevs));
        expect(window.committed() == 2, "history is capped at ngram_size-1 = 2");
    }

    // ---- 6. LOAD-BEARING: the composed step hands the RUNTIME the window prevs.
    {
        PleNgramWindow window(3, kEos);
        window.commit(std::span<const std::int32_t>(&kTa, 1));
        RecordingRuntime runtime;
        std::vector<std::int32_t> scratch;
        int dst_token = 0;
        ple_seam_gather_step(runtime, window, std::span<const std::int32_t>(&kTb, 1),
                             PlePhase::Decode, kEos, &dst_token, /*stream=*/0, scratch);
        expect(runtime.calls.size() == 1, "the composed step drove the runtime exactly once");
        if (runtime.calls.size() == 1) {
            const RecordingRuntime::Call& call = runtime.calls.front();
            expect(call.phase == PlePhase::Decode, "the composed step passes the phase through");
            expect(call.tokens == std::vector<std::int32_t>{kTb},
                   "the composed step passes the tokens through");
            expect(call.prevs.size() == 2 && call.prevs[0] == kTa,
                   "the runtime receives the WINDOW prevs, not an eos fill");
            std::printf("runtime saw prevs=[%d,%d] (an eos fill would be [%d,%d])\n",
                        call.prevs[0], call.prevs[1], kEos, kEos);
        }
        expect(ple_seam_dst_elems(PlePhase::Decode, 3, kHeads, kRowDim, 3) ==
                   static_cast<std::size_t>(kHeads * kRowDim),
               "dst sizing helper matches a 1-column Decode gather");
        expect(ple_seam_dst_elems(PlePhase::Prefill, 3, kHeads, kRowDim, 3) ==
                   static_cast<std::size_t>(3 * kHeads * kRowDim),
               "dst sizing helper matches a 3-column Prefill gather");
    }

    // ---- 7. Contract violations are loud, not silent.
    {
        PleNgramWindow narrow(2, kEos); // ngram_size mismatching the sidecar
        std::vector<std::int32_t> scratch;
        std::vector<std::int32_t> rows(kHeads);
        expect(throws_invalid_argument([&] {
                   ple_seam_derive_rows(layout, narrow, std::span<const std::int32_t>(&kTb, 1),
                                        PlePhase::Decode, kEos, scratch, rows.data());
               }),
               "a window/sidecar ngram_size mismatch throws instead of deriving garbage");
        PleNgramWindow window(3, kEos);
        std::vector<std::int32_t> short_prevs(1); // must be 2
        expect(throws_invalid_argument([&] {
                   window.fill_prevs(std::span<const std::int32_t>(&kTb, 1),
                                     std::span<std::int32_t>(short_prevs));
               }),
               "an undersized prevs buffer throws");
    }

    std::printf(failures == 0 ? "PLE_SEAM_WINDOW_PASS\n" : "PLE_SEAM_WINDOW_FAIL\n");
    return failures == 0 ? 0 : 1;
}
