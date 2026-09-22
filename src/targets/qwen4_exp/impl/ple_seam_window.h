#pragma once

// N6 / NGRAM-BUILD: the production composition of the n-gram window and the PLE seam.
//
// WHY THIS FILE EXISTS (the exact gap it closes)
// ---------------------------------------------------------------
// Two production halves already exist in this tree and NOTHING joined them, which is
// why the N6 ledger row calls the module "死代码（src/ops/ple/ 之外零引用）":
//
//   * detail::PleNgramWindow (impl/ple_ngram.h) produces the (tokens, prevs, eos)
//     triple, and its own header records that "树里没有任何东西生产 (tokens, prevs,
//     eos) 这个三元组：现存的两个调用点都是手填 prevs = eos（tests/test_ple_layout.cpp:96
//     与 :176），而 ... bigram/trigram mixers never see a real predecessor". So the
//     authoritative row formula was only ever exercised with every predecessor == eos.
//   * ops::ple::PleTable / PleRuntime (impl/ple_runtime.h) can gather the REAL SSD
//     sidecar through the device kernel, but nothing constructed them either.
//
// Joining them is this header's whole job. It owns the one thing a caller must not
// re-derive:
//   * WHICH columns a phase owes -> ops/ple/ple_stage.h ple_phase_window (the single
//     place that mapping lives; a Verify pass carries K+1 columns but keeps only the
//     last, so gathering all of them is K+1 times the SSD faults for one column of
//     useful data).
//   * WHERE the predecessors come from -> the window, i.e. committed history. Never an
//     eos fill, and never a second copy of the EOS-cut rule (that stays in
//     PleLayout::derive_rows_one, ple_layout.cpp:122-145).
//
// HOST-ONLY BY CONSTRUCTION: the CUDA half enters through a template parameter, so this
// header includes no CUDA header and can be verified by a plain host compiler
// (tests/test_ple_seam_window.cpp does exactly that, with no GPU and no sidecar).
// Instantiating the template is what keeps it out of the dead-code state the module is
// in today: tests/ops/ple_seam_gather_e2e_test.cu instantiates it against a real
// PleTable/GPU gather.

#include "ops/ple/ple_layout.h"
#include "ops/ple/ple_stage.h"
#include "targets/qwen4_exp/impl/ple_ngram.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen4_exp::detail {

// Which columns of a batch a phase owes, and how many predecessors each column carries.
struct PleSeamPlan {
    std::size_t first_column    = 0;
    std::size_t columns         = 0;
    std::size_t prevs_per_token = 0;
};

// The phase -> column-set mapping, with the predecessor width attached so a caller
// cannot pair a plan with a wrongly sized prevs buffer.
[[nodiscard]] inline PleSeamPlan ple_seam_plan(ops::ple::PlePhase phase,
                                               std::uint32_t ngram_size,
                                               std::size_t n_tokens) noexcept {
    const ops::ple::PlePhaseWindow window = ops::ple::ple_phase_window(phase, n_tokens);
    return PleSeamPlan{window.first_token, window.tokens,
                       PleNgramWindow::prevs_per_token(ngram_size)};
}

// Builds the prevs triple for a whole batch out of committed history. Exposed because
// the row derivation and the gather both need the same buffer, and because the "window
// actually fed real predecessors" check is a host-side fact worth asserting on its own
// (tests/test_ple_seam_window.cpp).
inline void ple_seam_fill_prevs(const PleNgramWindow& window,
                               std::span<const std::int32_t> tokens,
                               std::span<std::int32_t> prevs_out) {
    window.fill_prevs(tokens, prevs_out);
}

// Derives the row ids for the columns `phase` owes, taking predecessors from the window.
// rows_out must hold plan.columns * layout.n_heads ids, packed exactly like
// PleTable::gather's input (token-major, head-minor) so the two are interchangeable.
//
// Note what this deliberately does NOT do: it does not commit the batch to the history
// (a speculative Verify pass must not commit rejected proposal columns), and it does not
// re-implement the EOS cut. Both are the window's and PleLayout's business.
inline void ple_seam_derive_rows(const ops::ple::PleLayout& layout,
                                 const PleNgramWindow& window,
                                 std::span<const std::int32_t> tokens,
                                 ops::ple::PlePhase phase, std::int32_t eos,
                                 std::vector<std::int32_t>& prevs_scratch,
                                 std::int32_t* rows_out) {
    if (window.ngram_size() != layout.ngram_size) {
        throw std::invalid_argument(
            "PLE seam: window ngram_size != sidecar ngram_size (" +
            std::to_string(window.ngram_size()) + " vs " +
            std::to_string(layout.ngram_size) + ")");
    }
    const PleSeamPlan plan = ple_seam_plan(phase, layout.ngram_size, tokens.size());
    prevs_scratch.resize(tokens.size() * plan.prevs_per_token);
    ple_seam_fill_prevs(window, tokens, std::span<std::int32_t>(prevs_scratch));
    for (std::size_t i = 0; i < plan.columns; ++i) {
        const std::size_t column = plan.first_column + i;
        layout.derive_rows_one(tokens[column],
                               prevs_scratch.data() + column * plan.prevs_per_token, eos,
                               rows_out + i * layout.n_heads);
    }
}

// The one call a production step makes: window -> owed columns -> real-table gather.
//
// `Runtime` is duck-typed on purpose so this header stays CUDA-free. It must provide
//   gather_phase(tokens, prevs, eos, phase, dst, stream)
// which both ops::ple::PleTable (ple_table.h:42-44) and PleRuntime (ple_runtime.h:132)
// do. `Stream` is whatever stream type the runtime takes (cudaStream_t for both).
//
// The buffer is caller-owned so a decode loop allocates nothing per step.
template <class Runtime, class Stream>
inline void ple_seam_gather_step(Runtime& runtime, const PleNgramWindow& window,
                                 std::span<const std::int32_t> tokens,
                                 ops::ple::PlePhase phase, std::int32_t eos, void* dst,
                                 Stream stream,
                                 std::vector<std::int32_t>& prevs_scratch) {
    const std::size_t n_prev = PleNgramWindow::prevs_per_token(window.ngram_size());
    prevs_scratch.resize(tokens.size() * n_prev);
    ple_seam_fill_prevs(window, tokens, std::span<std::int32_t>(prevs_scratch));
    runtime.gather_phase(tokens, std::span<const std::int32_t>(prevs_scratch), eos, phase,
                         dst, stream);
}

// Number of rows the gathered dst holds for this phase, i.e. what a caller must size
// `dst` to: plan.columns * n_heads * row_dim BF16 elements. Kept next to the plan so the
// Verify case ("you passed T columns, read dst as [n_heads*row_dim, 1]") is one lookup
// instead of a re-derivation at every call site.
[[nodiscard]] inline std::size_t ple_seam_dst_elems(ops::ple::PlePhase phase,
                                                    std::uint32_t ngram_size,
                                                    std::uint32_t n_heads,
                                                    std::uint32_t row_dim,
                                                    std::size_t n_tokens) noexcept {
    return ple_seam_plan(phase, ngram_size, n_tokens).columns * n_heads * row_dim;
}

} // namespace ninfer::targets::qwen4_exp::detail
