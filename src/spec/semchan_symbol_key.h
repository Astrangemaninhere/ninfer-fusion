#pragma once

// src/spec/semchan_symbol_key.h -- THE SEMANTIC CANDIDATE CHANNEL, ITS SYMBOL-KEY PRODUCER,
// AND THE ONE THING KVMem'S AUTHORS SAY THEY THROW AWAY: THE BLOCK'S ORIGIN.
//
// ===========================================================================================
// WHAT THIS IS, IN THE TREE'S OWN WORDS
// ===========================================================================================
// This header implements the second candidate SOURCE the tree named and never had. It is the
// `fusiondesign` landing `src/spec/sum_dir_sign_key.h` + "候选集第二来源" whose three missing
// pieces F480 sec.6 listed by name: "树里真缺的只有符号键生产者 + 测试 + 裸 add_executable"
// (dl/_orch/DEFECTS.md, F480 sec.6, live line 30058).
//
// The discipline is not mine and is not a suggestion. Three sites state it, and each is quoted
// here with its file, because F544 sec.3 warns by name that the same sentence exists as a COPY in
// program_impl.h.pre_land / .frontierinv.pre / .frontierinv.post and that a line number without a
// filename is therefore ambiguous:
//
//   * program_impl.h [text: candidate-set UNION, never in place of this arbiter; = :15950 on 2026-09-25] (the LIVE file):
//       "A semantic channel (KVMem's Mean-K) is complementary and belongs in the candidate-set
//        UNION, never in place of this arbiter (KVMINE REPORT sec.2)."
//     (the same sentence is at program_impl.h.pre_land:14396-14397, .frontierinv.pre:15297-15298
//      and .frontierinv.post:15310-15311; those are COPIES, not the live lines)
//   * src/spec/sum_dir_query_key.h:37-39:
//       "an EMBEDDING vector -- a similarity score may narrow (the candidate-set UNION) and may
//        not decide"
//     and its own refusal text at :121: "a similarity score may enter the candidate-set UNION and
//     nothing else".
//   * src/spec/sum_dir_determined.h:24-27: "It may only contribute to the candidate-set UNION".
//
// ===========================================================================================
// ANCHOR SNAPSHOT -- THE LINE NUMBERS IN THIS FILE ARE A READING, AND ONE OF THEM ALREADY ROTTED
// ===========================================================================================
// Every `program_impl.h:<n>` below was taken with `sed -n '<n>p'` on the LIVE file at
// **sha256-16 `049eae0dcfd711f3` / 1,086,785 B** (`dl/semchan/logs/02_live_lines.txt`). While this
// line was still working, ANOTHER LINE REPLACED THAT FILE: at 2026-09-23 23:59:02 it became
// **`42f47ad02022b481` / 1,099,037 B**, and **every one of these numbers moved** --
// `dl/semchan/logs/91_reanchor.txt` has both readings side by side:
//
//     the doctrine sentence     :15723-15724  ->  :15917-15918      (+194)
//     `reach_request_for_term`  :15920-15928  ->  :16114   <- STALE (kvfix F893): LIVE is `reach_request_for_term` at :16146
//     `install_index_recall_provider` :15985  ->  :16179   <- STALE (kvfix F893): LIVE is `install_index_recall_provider` at :16219
//     `index_path_stamp().round_consulted = false`  ->  :15306
//
// THIS IS NOT A DEFECT I FIXED BY EDITING NUMBERS: it is the reason the ORDER says "锚点会腐烂，取行号
// 要当场取活行". THE DURABLE REFERENCE IS THE IDENTIFIER, NOT THE NUMBER. So every citation in this
// file NAMES the function, member or field it points at as well as the number, and a reader who
// wants the current line re-takes it with `grep -n '<identifier>' <file>`. The numbers are kept
// because a number plus its sha is a *reading* and readings are what this project trades in.
//
// AND THE ONE CITATION THAT MATTERS MOST IS UNAFFECTED IN SUBSTANCE, which is why this landing was
// not invalidated by the concurrent edit: the sentence at `:15917-15918` (was `:15723-15724`) still
// reads "A semantic channel (KVMem's Mean-K) is complementary and belongs in the candidate-set
// UNION, never in place of this arbiter (KVMINE REPORT sec.2)", and `grep -cF 'candidate-set UNION'`
// over the live file still returns 1. Re-measured, not assumed: see `logs/91_reanchor.txt`.
//
// ===========================================================================================
// THE PRECISION DISCIPLINE, WHICH IS THE LOAD-BEARING PART
// ===========================================================================================
// A mean over a block is a STATISTIC, and a statistic cannot carry the block's interior. KVMem
// states the price of its own default in its own units -- Mean-K = 32,768 B per block, about 128x
// the cargo it summarizes (dl/indexdesign, the Mean-K mathematics section) -- and its authors say
// it in words too: Mean-K necessarily loses the variation INSIDE a block, and it "dilutes and
// fragments the relevant evidence in long sessions" (KVMem's own
// docs/kvmem_known_issues.md, KVMI-007, status RESEARCH, priority P2: it is "the main accuracy
// bottleneck of the current default mean-k"). docs/kvmem_tiered_io_design.md sec.1 gives the same
// ratio from the other side: ~64 KiB of KV per token at block-tokens 256, so one block is ~16 MiB
// of K/V against a 32 KiB mean.
//
// => THEREFORE: this channel may NARROW the candidate set and it may NOT DECIDE. The exactness
//    has to come from our arbiter (`sum_dir_recall_span_reachable`, spec/sum_dir_reach.h), which
//    is exact-or-silent by construction (program_impl.h [text: exact-or-silent, never approximately right; = :15948 on 2026-09-25]). Every function below is
//    written so that "the semantic side decides" is not merely forbidden but INEXPRESSIBLE: the
//    union carries the arbiter's anchor through untouched, and there is no function in this
//    header that returns a determination.
//
// WHY THE VALUE IS CLAIMED ON THE NARROWING AXIS AND NOWHERE ELSE: `recallplan:387` says it
// without flattery -- "recall is a re-prefill, i.e. 1x on this axis, while KVMem's reported gain
// over compaction is 11.4-53.8x." So this header claims NOTHING about compression. Its one claim
// is: a query the lexical family cannot express can still put a page INTO the candidate set, and
// the arbiter still chooses.
//
// ===========================================================================================
// THE HOLE THE AUTHORS THEMSELVES POINT AT, AND WHAT THIS HEADER DOES ABOUT IT
// ===========================================================================================
// Three sentences, and they are the whole of the design decision:
//
//   1. WHAT THEY THROW AWAY. KVMem selects blocks and then re-RoPEs each block's K into a
//      compact position frame -- every byte of the block goes through the rotation -- so the model
//      is told "this block sits at position p'" while the block actually came from position p, and
//      NOTHING IN THE KV CARRIES THE ORIGIN. Attention cannot notice the substitution, because
//      after a re-based rotation there is no substitution to notice. Their own P0 issue records
//      the arithmetic consequence: "fp16/fp8 K 反复原地 re-RoPE 会累积漂移" -- repeated IN-PLACE
//      re-RoPE of fp16/fp8 K accumulates drift (docs/kvmem_known_issues.md, KVMI-011, P0,
//      IMPLEMENTED / PARTIAL REAL-SAMPLE VALIDATION), whose fix on their side is an immutable
//      raw-K with a bounded reset.
//   2. WE CARRY THE ORIGIN. Every candidate this header emits carries a `SemChanProvenance`: the
//      block's SOURCE span in the sequence it was summarized from, its generation tag, and its
//      position-free 128-bit content identity (sum_dir.h:212-227 `sum_dir_block_digest`, and
//      `SumDirRow`'s own comment at sum_dir.h:103-110 says the span is "BOOKKEEPING ... NOT part
//      of the identity"). The provenance is a COLUMN BESIDE the position frame, never a
//      recomputation of it, and the two are never collapsed into one value.
//   3. THE VERDICT IS THE ARBITER'S. A similarity may not decide -- so the origin is not used to
//      score, and it is not used to dedupe silently either: two candidates that share a position
//      frame but differ in origin are BOTH admitted (see `SemChanOriginPolicy` and the union's
//      `origins_per_page`), and the arbiter's exact-or-silent evidence picks between them.
//
// THE TREE ALREADY REFUSES TO GUESS ABOUT ORIGINS, AND THIS HEADER INHERITS THAT RATHER THAN
// INVENTING IT. `sum_dir.h` builds a `page_conflict_` column (declared :2316, built :2204-2219) and
// its guarded reader REFUSES a page that two GENERATIONS cover (`sum_dir.h:1821` verbatim:
// "if (page < page_conflict_.size() && page_conflict_[page] != 0U) { return false; }"), and
// `sum_dir_reach.h:917-996` states the consequence for the arbiter in its own words -- it "is
// generation-blind -- MEASURED: `generation` and `page_conflict_` have ZERO occurrences in this
// whole header -- so it must not answer a question the authority declines." A channel that
// RESOLVED two origins by score would be answering exactly that declined question. MEASURED on this
// landing's own P1 arm (dl/semchan/logs/21_poc_green.txt): the arbiter returns
// `status=refused-length` on a two-generation page while the channel hands it both origins -- the
// two layers behaving as the tree's own notes say they must, and neither of them guessing.
//
// WE DO NOT TAKE THE re-RoPE ROAD, AND THE COST IF ANYONE EVER DOES IS NAMED: our tree stores K
// ALREADY ROTATED -- `TextContext::mtp_forward_core` rotates in place and then appends, rope then
// `ops::gqa_kv_append` (src/targets/qwen3_6/impl/runtime/text_context_impl.h:791-797, cited by
// dl/semchan/logs/02_live_lines.txt; F480 sec.4 correction (a) is why the two e8 flags are NOT
// evidence for this and the rope->append ORDER is) -- so a design that re-RoPEs a restored block
// into a new frame would repeat exactly the operation KVMI-011 measures as drift-accumulating, on
// a stored-already-rotated K, which is strictly worse than KVMem's raw-K case. THIS CHANNEL DOES
// NOT ROTATE ANYTHING: it reads token IDS and never touches a K byte, so the re-RoPE failure mode
// is not avoided by care, it is out of reach by construction. What remains un-audited is named in
// dl/semchan/REPORT.md sec.6: the engine's own restore path is not this header's to certify.
//
// ===========================================================================================
// WHICH KVMem SCORER THIS IS, AND WHERE IT DIFFERS FROM KVMem's OWN SEMANTICS
// ===========================================================================================
// KVMem's --kvmem-retrieval-method accepts exactly five names (src/qw3_cli.cpp:143-145 verbatim:
// "mean-k|per-token|sub-block-mean-k|key-direction-fixed4|key-direction-adaptive"), dispatched
// through `KvMemRetrievalMethod` (include/qw3/kvmem_store.hpp:159:
// "enum class KvMemRetrievalMethod : uint8_t { MeanK = 0, PerToken = 1, SubBlockMeanK = 2,
//  DeltaNet = 3 };") whose DEFAULT is MeanK (:249). THIS HEADER IMPLEMENTS `mean-k` AND REFUSES
// THE OTHER FOUR BY NAME (see SemChanScorer below).
//
// THE DIFFERENCE, STATED PLAINLY, BECAUSE IT IS REAL AND IT IS NOT SMALL:
//   KVMem's mean-k averages the layer's KEY VECTORS: "block_attn_score_softmax_pages_kernel: one
//   CUDA block per (layer,token) ... logit[w] = scale*(q[l,t,h] . kbar[l,w,h/group]) for every
//   page w, softmaxes those logits OVER PAGES" (src/kernels_cuda.cu:5330-5338), where `kbar` is
//   the per-block MEAN of the fp16 K rows (src/kernels_cuda.cu:8040-8041). Its kbar is a LEARNED
//   object living in the model's key space, so its similarity is semantic.
//   THIS header's "symbol key" averages the block's own TOKEN IDS as one-hot symbols --
//   kbar_s = (1/n) * #{i in block : id_i == s} -- and scores with the same dot product shape,
//   q . kbar_s. It is therefore ORDER-INSENSITIVE (which is exactly the Mean-K failure mode, and
//   is why it is admissible as a candidate source) but it is NOT PARAPHRASE-INSENSITIVE.
//   THE REASON IT IS NOT THE KVMem OBJECT: this header is host-only and std-only. The tree has no
//   embedder and `sum_dir_query_key.h:40-41` already says so in the same breath it refuses the
//   embedding-vector source: "`sum_dir_vector.h`'s own default offer needs an embedder that does
//   not exist in the tree, so it cannot produce a KEY at all". This header does not invent one.
//   => The honest name for what is implemented here is "mean-k over the symbol alphabet": the
//      same REDUCTION (mean over the block), the same SCORE (dot with the query), the same
//      BLINDNESS (order), on the only per-token vector this layer can see.
//
// WHAT IT CANNOT DO, SAID SO NOBODY ASSUMES OTHERWISE: a paraphrase, a re-tokenisation, or any
// wording that is not the same ids is invisible to the ORDERING too -- this channel sees the same
// bag. It can be FOOLED by a block that holds the same multiset in a different order, and the test
// beside this header builds exactly that block on purpose, because that is the case where the
// arbiter must overrule this channel. It narrows; it never decides.
//
// ===========================================================================================
// WHAT THIS HEADER DOES NOT TOUCH (the "only add" rule)
// ===========================================================================================
//   * It does not modify `sum_dir.h`, `sum_dir_reach.h`, `sum_dir_query_key.h`,
//     `sum_dir_determined.h` or `turn_recall_journal.h`. It only #includes them and reads them.
//   * It does not replace, weaken or reorder anything in the engine. It has no engine-side caller:
//     `install_index_recall_provider` (program_impl.h:15985) does not include it, so landing this
//     file CANNOT move the engine binary by one byte -- the same property
//     `sum_dir_query_key.h:53-55` claims for itself.
//   * It does not touch the storage layer. A provenance that must survive a spill to disk and be
//     read back is an INTERFACE REQUIREMENT against `cold_host_tier` / `kv_cold_tier_budget.h`,
//     not something this header may reach into; it is named as such in REPORT.md sec.7.
//   * WHERE IT WOULD ATTACH, named so the seam is not a mystery: the arbiter's candidate input is
//     `appended_pages` (`reach_request_for_term`, program_impl.h:15920-15928), and F480 sec.6
//     already records that `appended_pages` "本来就是一个候选集入参". A second source cannot ride
//     the EXCLUSION list in; the additive seam is the arbiter's own `SumDirReachResult::alternatives`
//     column plus the `RecallRequest` the provider returns. Wiring it is a decision about the
//     engine's behaviour and is NOT taken here -- see dl/semchan/REPORT.md sec.5 for why, by name.

#include "spec/sum_dir.h"
#include "spec/sum_dir_query_key.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::spec::semchan {

// ---------------------------------------------------------------------------
// 1. THE NAMED SCORERS -- one implemented, four refused by name
// ---------------------------------------------------------------------------
// The five names are KVMem's, verbatim from its CLI validation (src/qw3_cli.cpp:582-593) and its
// enum (include/qw3/kvmem_store.hpp:159). Declaring all five is the point: a reader looking for
// "which of KVMem's scorers does this tree have" gets an answer per name rather than a silence.
enum class SemChanScorer : std::uint8_t {
    // IMPLEMENTED. KVMem's default (KvMemRetrievalMethod::MeanK == 0).
    MeanK = 0,
    // REFUSED (needs the per-token K the candidate space does not hold).
    PerToken = 1,
    // REFUSED (needs the K vectors, one mean per sub-block).
    SubBlockMeanK = 2,
    // REFUSED (needs the K vectors to cluster into direction prototypes).
    KeyDirectionFixed4 = 3,
    // REFUSED (needs the K vectors, plus the residual-gain thresholds).
    KeyDirectionAdaptive = 4,
};

[[nodiscard]] constexpr const char* semchan_scorer_name(SemChanScorer scorer) noexcept {
    switch (scorer) {
    case SemChanScorer::MeanK: return "mean-k";
    case SemChanScorer::PerToken: return "per-token";
    case SemChanScorer::SubBlockMeanK: return "sub-block-mean-k";
    case SemChanScorer::KeyDirectionFixed4: return "key-direction-fixed4";
    case SemChanScorer::KeyDirectionAdaptive: return "key-direction-adaptive";
    }
    return "unknown";
}

// The predicate that makes the refusals executable rather than editorial, and it is `constexpr` so
// the assertions at the bottom of this file fail the BUILD rather than an argument.
[[nodiscard]] constexpr bool semchan_scorer_is_implemented(SemChanScorer scorer) noexcept {
    return scorer == SemChanScorer::MeanK;
}

// THE SENTENCE THAT IS THE WHOLE POINT. There is no value of `SemChanScorer` for which this is
// true, and there never will be: the channel's output is a SIMILARITY, and a similarity may not
// determine (sum_dir_query_key.h:37-39, :121; program_impl.h [text: candidate-set UNION, never in place of this arbiter; = :15950 on 2026-09-25]).
[[nodiscard]] constexpr bool semchan_scorer_may_determine(SemChanScorer) noexcept { return false; }

// THE SAME PREDICATE FOR THE CHANNEL AS A WHOLE, so the compile-time half has one constant to pin
// and a reader has one place to look. There is no configuration under which a candidate source
// determines an answer in this tree: `sum_dir_reach.h`'s selector is "exact-or-silent, never
// approximately right" (program_impl.h [text: exact-or-silent, never approximately right; = :15948 on 2026-09-25]) and that property is what makes the family
// admissible at all.
inline constexpr bool kSemChanMayDetermine = false;

// Why, in one sentence per name, so a caller printing the reason never prints a null.
[[nodiscard]] constexpr const char* semchan_scorer_reason(SemChanScorer scorer) noexcept {
    switch (scorer) {
    case SemChanScorer::MeanK:
        return "implemented: the mean of the block's per-token SYMBOL vectors, scored by dot with "
               "the query's own symbol vector -- KVMem's default reduction and the same score, on "
               "the alphabet this host-only layer can see (NOT on the learned K)";
    case SemChanScorer::PerToken:
        return "refused: per-token needs the per-token K rows, which this host-only layer does not "
               "have; KVMem's own exactmass store is also capped (their known issue KVMI-005, "
               "OPEN/P1: a 48 KiB shared-memory budget, failing past ~12K blocks)";
    case SemChanScorer::SubBlockMeanK:
        return "refused: sub-block-mean-k needs the K vectors to average per sub-block; with no K "
               "there is nothing to sub-divide, and inventing a sub-block split over symbols would "
               "be a different scorer wearing KVMem's name";
    case SemChanScorer::KeyDirectionFixed4:
        return "refused: key-direction-fixed4 clusters every 32-token slice into four DIRECTION "
               "prototypes of the K vectors (KVMem's own CLI help); directions of one-hot symbol "
               "vectors carry no information";
    case SemChanScorer::KeyDirectionAdaptive:
        return "refused: key-direction-adaptive retains 1/2/4 packed prototypes by normalized "
               "residual gain over the K vectors; the residual of a symbol histogram is a "
               "histogram, not a direction";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// 2. THE PROVENANCE -- where the block CAME FROM, carried beside the frame and never recomputed
// ---------------------------------------------------------------------------
// WHY THIS TYPE EXISTS AT ALL, in the authors' own terms: after a re-based rotation, "the model is
// told where the block now sits and nothing tells it where the block came from". A candidate that
// arrives without an origin is therefore not a weaker candidate -- it is the same hole one layer
// over, and this header refuses it (see SemChanOriginPolicy).
//
// THE FOUR FIELDS ARE READS, NOT INVENTIONS, and each names its source column:
//   * `source_begin` / `source_end` -- the row's `token_begin` / `token_end`
//     (sum_dir.h:107-108), which the tree DEFINES as "BOOKKEEPING: this is the caller's token
//     space and is NOT part of the identity ... its job is to let a retrieval answer be reported
//     as a token range". That is exactly a provenance, and the fact that it is NOT identity is
//     exactly why it must be carried beside one.
//   * `generation` -- the row's `generation` (sum_dir.h:110), the sequence tag.
//   * `content_identity` -- the row's `block_identity` (sum_dir.h:102), the position-free 128-bit
//     content identity computed by `sum_dir_block_digest` from the block's tokens and nothing else
//     (sum_dir.h:212-227). This is the field that survives a re-prefill, which is why
//     `sum_dir_query_key.h:139` refuses position-bearing sources as identities.
//   * `row` -- the row index, so a reader can go back to the exact row that produced this.
//
// WHAT IT IS NOT: it is not a position frame. Nothing here is derived by rotating, re-basing or
// re-numbering anything. `origin_span_tokens()` is the SOURCE span; a caller that wants a window
// frame computes it itself, elsewhere, and this type does not offer it.
struct SemChanProvenance {
    std::uint32_t               source_begin = 0;
    std::uint32_t               source_end   = 0; // exclusive; in the sequence that was summarized
    std::uint32_t               generation   = 0;
    sum_dir::SumDirDigest       content_identity{};
    std::uint32_t               row          = 0;

    // `SumDirDigest::is_unset()` is the tree's own predicate (sum_dir.h:190-193: "The pristine
    // value, i.e. 'no digest'. Unreachable for a real block ... which is exactly why every caller
    // may use it as the invalid value"), and `sum_dir_row_well_formed` reads it the same way
    // (sum_dir.h:591). This header does not invent a second spelling of "no identity".
    [[nodiscard]] bool is_set() const noexcept {
        return !content_identity.is_unset() && content_identity.lo != 0U &&
               content_identity.hi != 0U && source_end > source_begin;
    }
    [[nodiscard]] std::uint32_t origin_span_tokens() const noexcept {
        return source_end > source_begin ? source_end - source_begin : 0U;
    }
    // The one thing a comparison needs, spelled out so two callers cannot disagree about what
    // "same origin" means: the identity AND the source span AND the generation. Two blocks with
    // the same content identity but different spans are the same TEXT at two places, which is a
    // different fact from the same text at one place, and the project's whole refusal discipline
    // rests on not confusing those two.
    [[nodiscard]] friend bool operator==(const SemChanProvenance& a,
                                         const SemChanProvenance& b) noexcept {
        return a.content_identity == b.content_identity && a.source_begin == b.source_begin &&
               a.source_end == b.source_end && a.generation == b.generation;
    }
};

// THE POLICY, DECLARED AND NOT DEFAULTED-IN-SILENCE. The house idiom for "a policy that must be
// named or the tree does not build" is `kInexactAdmissionPolicy` / `kRecallBudgetEdgePolicy`
// (turn_recall_journal.h:1519-1527, and the static_assert at program_impl.h:15090-15092 that
// refuses `Unnamed`). This is that idiom, for the origin axis.
enum class SemChanOriginPolicy : std::uint8_t {
    Unnamed           = 0, // the tree must NOT build with this: see the static_assert below
    RefuseWholePass   = 1, // a block with no origin makes the PASS refuse -- default
    DropAndCountLoudly = 2, // the origin-less block is dropped and COUNTED in the printed line
};

inline constexpr SemChanOriginPolicy kSemChanOriginPolicy = SemChanOriginPolicy::RefuseWholePass;

[[nodiscard]] constexpr const char* semchan_origin_policy_name(
    SemChanOriginPolicy policy) noexcept {
    switch (policy) {
    case SemChanOriginPolicy::Unnamed: return "unnamed";
    case SemChanOriginPolicy::RefuseWholePass: return "refuse-whole-pass";
    case SemChanOriginPolicy::DropAndCountLoudly: return "drop-and-count-loudly";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// 3. THE SYMBOL KEY -- the producer the tree did not have
// ---------------------------------------------------------------------------
// The mean is taken over the block's own token ids: a one-hot per token, averaged. Written out,
// because the shape matters more than the code:
//
//     kbar_s = (1 / n) * SUM_{i in block} [ id_i == s ]         for every symbol s in the block
//     q_s    = (1 / m) * SUM_{j in query} [ id_j == s ]
//     score  = SUM_s q_s * kbar_s                                <-- KVMem's `q . kbar`
//
// `kbar` and `q` are kept as SPARSE (symbol, weight) pairs in ASCENDING symbol order, because the
// alphabet is sparse in any real block and because a sorted vector makes the dot product exact and
// the tie-break deterministic without an unordered_map's iteration order leaking into a ranking.
//
// `weights` are L1-normalized, so a LONG block and a SHORT block are comparable, which is what a
// mean means.
struct SemChanSymbolKey {
    std::uint32_t tokens = 0;            // how many ids the key was derived from; 0 == no key
    std::vector<std::uint32_t> symbols;  // ASCENDING, distinct
    std::vector<double> weights;         // parallel to `symbols`; sums to 1 when tokens != 0

    [[nodiscard]] bool is_unset() const noexcept { return tokens == 0 || symbols.empty(); }

    // The weight this key gives a symbol, or 0. Binary search over the sorted symbols, so the
    // readout is O(log |alphabet|) and cannot allocate.
    [[nodiscard]] double weight_of(std::uint32_t symbol) const noexcept {
        const auto it = std::lower_bound(symbols.begin(), symbols.end(), symbol);
        if (it == symbols.end() || *it != symbol) { return 0.0; }
        return weights[static_cast<std::size_t>(it - symbols.begin())];
    }
};

// Every refusal the producer can make has a named field, so a zero key is never a silent answer.
// The two bounds are the block's own granularity: `kSumDirBlockTokens` is 64
// (sum_dir.h:138-139 `kSumDirBlockTokens = turn_recall::kRecallPageTokens` with a static_assert of
// 64), and a SHORT tail block is legitimate, so 1..64 is admitted and 0 / >64 are refused by name.
struct SemChanKeyReport {
    std::uint32_t requested_tokens = 0;
    bool on_axis                   = false; // 1 <= count <= kSemChanMaxBlockTokens
    bool refused_empty_span        = false; // ids == nullptr or count == 0
    bool refused_over_block        = false; // count > kSemChanMaxBlockTokens
};

inline constexpr std::uint32_t kSemChanMaxBlockTokens = sum_dir::kSumDirBlockTokens;

[[nodiscard]] inline SemChanSymbolKey
semchan_symbol_key_from_ids(const std::uint32_t* ids, std::uint32_t count,
                            SemChanKeyReport* report = nullptr) noexcept {
    SemChanSymbolKey key;
    SemChanKeyReport local;
    local.requested_tokens = count;

    if (ids == nullptr || count == 0) {
        local.refused_empty_span = true;
        if (report != nullptr) { *report = local; }
        return key; // tokens stays 0: "not a key"
    }
    if (count > kSemChanMaxBlockTokens) {
        local.refused_over_block = true;
        if (report != nullptr) { *report = local; }
        return key;
    }

    key.symbols.assign(ids, ids + count);
    std::sort(key.symbols.begin(), key.symbols.end());
    // Collapse the sorted run into (distinct symbol, multiplicity) and normalize in the same pass:
    // one traversal, and the multiplicity is an INTEGER before it is a weight, so the mean is
    // computed by counting rather than by accumulating -- no floating-point drift from the sum.
    std::vector<std::uint32_t> counts;
    std::vector<std::uint32_t> distinct;
    counts.reserve(key.symbols.size());
    distinct.reserve(key.symbols.size());
    for (std::size_t i = 0; i < key.symbols.size();) {
        const std::uint32_t symbol = key.symbols[i];
        std::size_t j = i;
        while (j < key.symbols.size() && key.symbols[j] == symbol) { ++j; }
        distinct.push_back(symbol);
        counts.push_back(static_cast<std::uint32_t>(j - i));
        i = j;
    }
    key.symbols = std::move(distinct);
    key.weights.resize(counts.size());
    for (std::size_t i = 0; i < counts.size(); ++i) {
        key.weights[i] = static_cast<double>(counts[i]) / static_cast<double>(count);
    }
    key.tokens = count;
    local.on_axis = true;
    if (report != nullptr) { *report = local; }
    return key;
}

[[nodiscard]] inline SemChanSymbolKey
semchan_symbol_key_from_ids(const std::vector<std::uint32_t>& ids,
                            SemChanKeyReport* report = nullptr) noexcept {
    return semchan_symbol_key_from_ids(ids.data(), static_cast<std::uint32_t>(ids.size()), report);
}

// ---------------------------------------------------------------------------
// 4. THE SCORER, NAMED -- mean-k's dot product
// ---------------------------------------------------------------------------

// KVMem's mean-k scores a query against a block with a single dot product against the block mean
// (src/kernels_cuda.cu:5330-5338, the per-(layer,token) softmax over pages). This is that dot,
// over the sparse symbol vectors, two-pointer so it is exact and allocation-free.
//
// NOTE ON THE ORIGIN AND THE SCORE: the provenance is DELIBERATELY NOT AN INPUT to this function.
// Adding it would make the origin a ranking signal, i.e. a way for the channel to decide, which is
// the one thing it may not do. The origin travels WITH the candidate and is read by the ARBITER.
//
// NEGATIVE LOGITS ARE NOT CLAMPED HERE, AND THAT IS A DIFFERENCE, NAMED: KVMem's step kernel
// accumulates `block_score += dot > 0.0f ? dot : 0.0f;` (src/kernels_cuda.cu:5362-5364). Both
// weight functions are non-negative here, so the dot is non-negative by construction and the clamp
// would be dead code -- but a reader who knows KVMem's kernel would otherwise wonder, so the
// difference is written down rather than left to be discovered.
[[nodiscard]] inline double semchan_mean_k_score(const SemChanSymbolKey& query,
                                                 const SemChanSymbolKey& block) noexcept {
    std::size_t i = 0;
    std::size_t j = 0;
    double score = 0.0;
    while (i < query.symbols.size() && j < block.symbols.size()) {
        const std::uint32_t a = query.symbols[i];
        const std::uint32_t b = block.symbols[j];
        if (a == b) {
            score += query.weights[i] * block.weights[j];
            ++i;
            ++j;
        } else if (a < b) {
            ++i;
        } else {
            ++j;
        }
    }
    return score;
}

// ---------------------------------------------------------------------------
// 5. THE INDEX -- the block side, read off the engine's OWN columns
// ---------------------------------------------------------------------------
// Nothing new is stored and no engine state is invented: `SumDirRow` already keeps its block's
// tokens -- `content_first` / `content_count` index `SumDir::content()`, and the row's own comment
// says why they are there: "the block's own contents ... kept so a query can be run against the
// block itself and not only against its catalogue line" (sum_dir.h:111-114). And it already keeps
// the origin -- `token_begin` / `token_end` / `generation` / `block_identity` (sum_dir.h:102-110).
// So the index is a READ of the directory, exactly like `SumDirReachPostings::build`
// (sum_dir_reach.h:435-520), and it cannot hold anything the directory does not.
struct SemChanBlockEntry {
    std::uint32_t          page = 0; // the row's own logical page (sum_dir.h:109)
    SemChanSymbolKey       key;
    SemChanProvenance      provenance;
};

struct SemChanIndexReport {
    std::uint32_t rows_total      = 0;
    std::uint32_t rows_indexed    = 0;
    std::uint32_t rows_not_live   = 0; // state != Live: the engine's byte axis, read not invented
    std::uint32_t rows_empty      = 0; // content_count == 0
    std::uint32_t rows_over_block = 0; // content_count > kSumDirBlockTokens: refused, not truncated
    std::uint32_t rows_no_origin  = 0; // an unset content identity or an empty source span
    std::uint32_t blocks          = 0;
    // The count the provenance criterion is about: blocks that share a position frame (the same
    // CONTENT IDENTITY) and differ in origin. On a directory built by re-basing a window, this is
    // the number of places the frame alone would have lied.
    std::uint32_t same_frame_distinct_origin = 0;
};

[[nodiscard]] inline std::vector<SemChanBlockEntry>
semchan_index_blocks(const sum_dir::SumDir& directory, SemChanIndexReport* report = nullptr) {
    SemChanIndexReport local;
    std::vector<SemChanBlockEntry> index;
    const std::vector<sum_dir::SumDirRow>& rows = directory.rows();
    const std::vector<std::uint32_t>& content = directory.content();
    local.rows_total = static_cast<std::uint32_t>(rows.size());
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const sum_dir::SumDirRow& row = rows[r];
        if (row.state != sum_dir::SumDirState::Live) {
            ++local.rows_not_live;
            continue;
        }
        if (row.content_count == 0U) {
            ++local.rows_empty;
            continue;
        }
        const std::size_t first = row.content_first;
        const std::size_t last  = first + row.content_count;
        if (last > content.size()) {
            // A row that points outside its own content column is malformed, and
            // `sum_dir_row_well_formed` (sum_dir.h) is the tree's own predicate for that. Refusing
            // here rather than reading past the end is the same "refuse, do not clamp" rule the
            // rest of this header follows.
            ++local.rows_empty;
            continue;
        }
        SemChanKeyReport key_report;
        SemChanSymbolKey key = semchan_symbol_key_from_ids(content.data() + first,
                                                           row.content_count, &key_report);
        if (key_report.refused_over_block) {
            ++local.rows_over_block;
            continue;
        }
        if (key.is_unset()) {
            ++local.rows_empty;
            continue;
        }
        SemChanBlockEntry entry;
        entry.page = row.page;
        entry.provenance.source_begin     = row.token_begin;
        entry.provenance.source_end       = row.token_end;
        entry.provenance.generation       = row.generation;
        entry.provenance.content_identity = row.block_identity;
        entry.provenance.row              = static_cast<std::uint32_t>(r);
        entry.key = std::move(key);
        if (!entry.provenance.is_set()) { ++local.rows_no_origin; }
        // COUNTED, NOT RESOLVED: walk back over what is already in the index and note every entry
        // that carries this block's identity from a DIFFERENT origin. The index does not merge
        // them and does not keep only the newest -- choosing between two origins is the arbiter's
        // act, and a channel that pre-chose would have decided.
        for (const SemChanBlockEntry& prior : index) {
            if (prior.provenance.content_identity == entry.provenance.content_identity &&
                !(prior.provenance == entry.provenance)) {
                ++local.same_frame_distinct_origin;
            }
        }
        index.push_back(std::move(entry));
        ++local.rows_indexed;
    }
    local.blocks = static_cast<std::uint32_t>(index.size());
    if (report != nullptr) { *report = local; }
    return index;
}

// ---------------------------------------------------------------------------
// 6. THE CANDIDATE SOURCE -- narrows, carries the origin, and has no way to decide
// ---------------------------------------------------------------------------
enum class SemChanCandidateStatus : std::uint8_t {
    Candidates                  = 0,
    RefusedEmptyQuery           = 1, // the query's key is unset: not a query, not a miss
    RefusedNoBlocks             = 2, // the index is empty
    RefusedNoPositiveScore      = 3, // every block scored exactly 0 -- refused, NOT "none found"
    RefusedBoundBinds           = 4, // more positive blocks than the fanout can hold: REFUSE
    RefusedScorerNotImplemented = 5,
    RefusedOriginMissing        = 6, // under RefuseWholePass: a block with no origin, no pass
};

[[nodiscard]] constexpr const char* semchan_candidate_status_name(
    SemChanCandidateStatus status) noexcept {
    switch (status) {
    case SemChanCandidateStatus::Candidates: return "candidates";
    case SemChanCandidateStatus::RefusedEmptyQuery: return "refused-empty-query";
    case SemChanCandidateStatus::RefusedNoBlocks: return "refused-no-blocks";
    case SemChanCandidateStatus::RefusedNoPositiveScore: return "refused-no-positive-score";
    case SemChanCandidateStatus::RefusedBoundBinds: return "refused-bound-binds";
    case SemChanCandidateStatus::RefusedScorerNotImplemented:
        return "refused-scorer-not-implemented";
    case SemChanCandidateStatus::RefusedOriginMissing: return "refused-origin-missing";
    }
    return "unknown";
}

// A CANDIDATE IS A PAGE, A SCORE AND AN ORIGIN. The origin is not optional and not decorative:
// it is the field the arbiter reads to tell "this is the block the question is about" from "this
// is a block that happens to look like it", which is the confusion the authors' own P0 issue
// describes from the other end.
struct SemChanCandidate {
    std::uint32_t     page  = 0;
    double            score = 0.0;
    SemChanProvenance provenance;
};

// THE EDGE RULE, TAKEN FROM THE FILE THAT OWNS IT, NOT RESTATED: `kRecallBudgetEdgePolicy`
// (turn_recall_journal.h:1527) is `BudgetEdgePolicy::RefuseNotTruncate`. The semantic side obeys
// the same rule, and F480 sec.6 is the reason it must not invent its own: the tree has ALREADY
// paid for a layer that "TRUNCATES from the low end and drops the anchor the selector rank-1'd"
// while claiming coverage, and the instruction there is to take the edge rule "from
// `sum_dir_reach.h:773`, NEVER from `lo += drop`". A truncating candidate source would be that
// same defect in a new file.
struct SemChanCandidates {
    SemChanScorer          scorer = SemChanScorer::MeanK;
    SemChanOriginPolicy    origin_policy = kSemChanOriginPolicy;
    SemChanCandidateStatus status = SemChanCandidateStatus::RefusedEmptyQuery;
    std::uint32_t fanout    = 0; // the cap the caller set
    std::uint32_t scored    = 0; // blocks the scorer ran over
    std::uint32_t positive  = 0; // blocks with a strictly positive score
    std::uint32_t page_tokens = 0;
    std::uint32_t dropped_no_origin = 0; // only reachable under DropAndCountLoudly
    std::uint32_t distinct_pages = 0;
    // candidates with the SAME page but a DIFFERENT origin. Non-zero means the page number alone
    // is not enough to describe what this channel found, which is the whole provenance criterion.
    std::uint32_t same_page_distinct_origin = 0;
    std::vector<SemChanCandidate> candidates; // score DESC, then page ASC, then origin ASC
    std::string refusal;                      // non-empty iff status != Candidates

    [[nodiscard]] bool found() const noexcept { return status == SemChanCandidateStatus::Candidates; }

    // The pages, one entry per candidate, so a page named twice for two origins appears twice.
    // A caller that wants the SET asks for `distinct_pages`; a caller that wants to know what the
    // channel saw asks THIS, and the difference between the two vectors is the provenance.
    [[nodiscard]] std::vector<std::uint32_t> pages() const {
        std::vector<std::uint32_t> out;
        out.reserve(candidates.size());
        for (const SemChanCandidate& c : candidates) { out.push_back(c.page); }
        return out;
    }
    // Every origin this channel names for one page. Empty when the channel did not name the page.
    [[nodiscard]] std::vector<SemChanProvenance> origins_of(std::uint32_t page) const {
        std::vector<SemChanProvenance> out;
        for (const SemChanCandidate& c : candidates) {
            if (c.page == page) { out.push_back(c.provenance); }
        }
        return out;
    }
};

// The diagnostic line. Two counts that must never be added are printed on the same line with
// their own labels: `candidates` counts (page, origin) records, `distinct_pages` counts pages, and
// their denominators are different sets -- the project rule is "比集合不比个数".
[[nodiscard]] inline std::string semchan_candidates_line(const SemChanCandidates& c) {
    std::string out = "[semchan] scorer=";
    out += semchan_scorer_name(c.scorer);
    out += " status=";
    out += semchan_candidate_status_name(c.status);
    out += " origin_policy=";
    out += semchan_origin_policy_name(c.origin_policy);
    out += " fanout=" + std::to_string(c.fanout);
    out += " scored=" + std::to_string(c.scored);
    out += " positive=" + std::to_string(c.positive);
    out += " candidates=" + std::to_string(c.candidates.size());
    out += " distinct_pages=" + std::to_string(c.distinct_pages);
    out += " same_page_distinct_origin=" + std::to_string(c.same_page_distinct_origin);
    out += " dropped_no_origin=" + std::to_string(c.dropped_no_origin);
    out += " page_tokens=" + std::to_string(c.page_tokens);
    out += " may_determine=0";
    if (!c.refusal.empty()) {
        out += " refusal=";
        out += c.refusal;
    }
    return out;
}

// A candidate that cannot be held must be HELD or the pass must REFUSE -- sum_dir_reach.h [text: A candidate that cannot be held must be HELD or the run must REFUSE; = :102 on 2026-09-25]
// states the rule for the arbiter's own budget, and this is the same rule.
[[nodiscard]] inline SemChanCandidates
semchan_candidate_pages(const std::vector<SemChanBlockEntry>& index,
                        const SemChanSymbolKey& query, std::uint32_t fanout,
                        SemChanOriginPolicy origin_policy = kSemChanOriginPolicy) {
    SemChanCandidates out;
    out.origin_policy = origin_policy;
    out.fanout        = fanout;
    out.page_tokens   = sum_dir::kSumDirBlockTokens;
    if (!semchan_scorer_is_implemented(out.scorer)) {
        out.status  = SemChanCandidateStatus::RefusedScorerNotImplemented;
        out.refusal = "the configured scorer is declared and not implemented: see "
                      "semchan_scorer_reason() for the per-name reason";
        return out;
    }
    if (query.is_unset()) {
        out.status  = SemChanCandidateStatus::RefusedEmptyQuery;
        out.refusal = "the query's symbol key is unset (0 tokens): this is 'not asked', and it is "
                      "not the same answer as 'nothing found'";
        return out;
    }
    if (index.empty()) {
        out.status  = SemChanCandidateStatus::RefusedNoBlocks;
        out.refusal = "the block index is empty: there is nothing to narrow";
        return out;
    }

    std::vector<SemChanCandidate> found;
    found.reserve(index.size());
    for (const SemChanBlockEntry& entry : index) {
        ++out.scored;
        const double score = semchan_mean_k_score(query, entry.key);
        if (!(score > 0.0)) { continue; } // exactly-zero and NaN both land here; see the note below
        ++out.positive;
        // THE ORIGIN GATE, BEFORE THE CANDIDATE EXISTS. A score is the only thing the channel is
        // allowed to compute; an origin is the only thing it is allowed to pass on. A block whose
        // origin is unset cannot become a candidate under either policy -- what the policy decides
        // is whether the PASS continues.
        if (!entry.provenance.is_set()) {
            if (origin_policy == SemChanOriginPolicy::RefuseWholePass) {
                out.status  = SemChanCandidateStatus::RefusedOriginMissing;
                out.refusal = "a positively-scored block carries no origin (an unset content "
                              "identity or an empty source span). Emitting it would be a candidate "
                              "whose provenance is exactly the thing KVMem's re-based frame loses "
                              "(KVMI-011), so the pass refuses rather than answering with an "
                              "anonymised block";
                return out;
            }
            ++out.dropped_no_origin; // DropAndCountLoudly: dropped, and printed in the line
            continue;
        }
        SemChanCandidate candidate;
        candidate.page       = entry.page;
        candidate.score      = score;
        candidate.provenance = entry.provenance;
        found.push_back(std::move(candidate));
    }
    if (found.empty()) {
        out.status  = SemChanCandidateStatus::RefusedNoPositiveScore;
        out.refusal = "no block shares a single symbol with the query: a zero-overlap answer is a "
                      "SILENT MISS if it is reported as 'nothing found' (the arbiter's own "
                      "NoCandidate refusal exists for the same reason, sum_dir_reach.h:257)";
        return out;
    }
    if (fanout != 0U && found.size() > fanout) {
        out.status  = SemChanCandidateStatus::RefusedBoundBinds;
        out.refusal = "the bound binds: " + std::to_string(found.size()) +
                      " positive record(s) > fanout " + std::to_string(fanout) +
                      ". Refusing rather than truncating (kRecallBudgetEdgePolicy = "
                      "RefuseNotTruncate, turn_recall_journal.h:1527)";
        return out;
    }

    // A TOTAL ORDER, NOT A RANK. Ties break on the page, then on the origin, so the same directory
    // and the same query always yield the same list -- a candidate order that depended on the
    // index's own iteration order would make two readers of one run disagree, which is the defect
    // this project calls a "plausible-looking wrong anchor" one layer over. NOTE that the tie
    // break is on the ORIGIN and never on the frame: two records that differ only in origin are
    // kept in BOTH orders' sight, side by side and distinct.
    std::vector<std::size_t> order(found.size());
    for (std::size_t i = 0; i < order.size(); ++i) { order[i] = i; }
    std::sort(order.begin(), order.end(), [&found](std::size_t a, std::size_t b) {
        if (found[a].score != found[b].score) { return found[a].score > found[b].score; }
        if (found[a].page != found[b].page) { return found[a].page < found[b].page; }
        if (found[a].provenance.source_begin != found[b].provenance.source_begin) {
            return found[a].provenance.source_begin < found[b].provenance.source_begin;
        }
        return found[a].provenance.generation < found[b].provenance.generation;
    });
    out.candidates.reserve(order.size());
    for (const std::size_t i : order) { out.candidates.push_back(found[i]); }

    {
        std::vector<std::uint32_t> distinct;
        distinct.reserve(out.candidates.size());
        for (const SemChanCandidate& c : out.candidates) { distinct.push_back(c.page); }
        std::sort(distinct.begin(), distinct.end());
        distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
        out.distinct_pages = static_cast<std::uint32_t>(distinct.size());
    }
    for (std::size_t a = 0; a < out.candidates.size(); ++a) {
        for (std::size_t b = a + 1; b < out.candidates.size(); ++b) {
            if (out.candidates[a].page == out.candidates[b].page &&
                !(out.candidates[a].provenance == out.candidates[b].provenance)) {
                ++out.same_page_distinct_origin;
            }
        }
    }
    out.status = SemChanCandidateStatus::Candidates;
    return out;
}

// ---------------------------------------------------------------------------
// 7. THE UNION -- where the second source meets the arbiter, and loses every argument it must
// ---------------------------------------------------------------------------
// THE SHAPE, STATED SO IT CANNOT BE MISREAD: the union is the ARBITER'S page list with the
// channel's pages APPENDED (deduplicated by PAGE, never by origin). The arbiter's order is
// preserved, the arbiter's anchor is carried through untouched, and `anchor_preserved` is a
// CALCULATED field rather than a claim. There is deliberately no function here that takes a union
// and returns an anchor of its own: the only anchor in this struct is the one the arbiter handed
// in.
//
// THE ORIGINS SURVIVE THE DEDUPLICATION, which is the part the provenance criterion is about: if
// the channel named one page from two origins, `origins_per_page` holds BOTH, and
// `pages_with_multiple_origins` counts the page once. Nothing is merged away silently, and the
// arbiter is handed every origin the channel saw.
struct SemChanUnion {
    std::uint32_t anchor_page = 0;      // the ARBITER's anchor, copied, never recomputed
    bool     anchor_preserved = false;  // == (anchor_page == the arbiter's anchor)
    bool     arbiter_found    = false;
    std::uint32_t from_arbiter      = 0; // pages the arbiter named
    std::uint32_t from_semchan_only = 0; // pages ONLY the channel named -- the channel's whole gift
    std::uint32_t in_both           = 0; // pages both named -- agreement, counted
    std::uint32_t candidates        = 0; // the union's cardinality, in PAGES
    std::uint32_t cap               = 0;
    std::uint32_t pages_with_multiple_origins = 0;
    bool     refused_over_cap       = false;
    std::string refusal;
    std::vector<std::uint32_t> pages;
    // Parallel to `pages`: every origin the channel named for that page, in the channel's own
    // order. EMPTY for a page only the arbiter named -- and empty is a different fact from
    // "the arbiter's page has an origin", which the arbiter states for itself.
    std::vector<std::vector<SemChanProvenance>> origins_per_page;

    [[nodiscard]] bool holds(std::uint32_t page) const noexcept {
        return std::find(pages.begin(), pages.end(), page) != pages.end();
    }
    [[nodiscard]] bool determined_by_semchan() const noexcept { return false; } // by construction

    // Every origin the channel named for one page, in the channel's own order; empty when the
    // channel named nothing for that page (or the page is the arbiter's own). A caller that gets
    // MORE THAN ONE here has been handed a decision it must make itself -- this header will not
    // make it (see `semchan_origin_may_be_chosen_by_channel`).
    [[nodiscard]] std::vector<SemChanProvenance> origins_of(std::uint32_t page) const {
        const auto it = std::find(pages.begin(), pages.end(), page);
        if (it == pages.end()) { return {}; }
        return origins_per_page[static_cast<std::size_t>(it - pages.begin())];
    }
};

// THE CHANNEL MAY NOT CHOOSE BETWEEN ORIGINS EITHER. Choosing the "newer" or the "closer" origin
// by anything but exact evidence is a decision, and a candidate source does not decide. It PUSHES
// both and says so; the arbiter -- whose family is exact-or-silent (program_impl.h [text: exact-or-silent, never approximately right; = :15948 on 2026-09-25]) -- is
// the only layer here allowed to prefer one. This predicate is `constexpr` and pinned below so the
// rule cannot be relaxed by editing a branch.
[[nodiscard]] constexpr bool semchan_origin_may_be_chosen_by_channel() noexcept { return false; }

[[nodiscard]] inline SemChanUnion
semchan_union_candidates(const std::vector<std::uint32_t>& arbiter_pages,
                         std::uint32_t arbiter_anchor, bool arbiter_found,
                         const SemChanCandidates& channel, std::uint32_t cap) {
    SemChanUnion out;
    out.anchor_page      = arbiter_anchor;
    out.anchor_preserved = (out.anchor_page == arbiter_anchor); // trivially true, and PRINTED
    out.arbiter_found    = arbiter_found;
    out.cap              = cap;

    out.pages = arbiter_pages; // the arbiter's order, verbatim
    out.origins_per_page.assign(arbiter_pages.size(), {});
    out.from_arbiter = static_cast<std::uint32_t>(arbiter_pages.size());

    // A REFUSED CHANNEL CONTRIBUTES NOTHING. It is not an empty list: it is a named absence, and
    // the union says so by leaving `from_semchan_only` at 0 while the channel's own status carries
    // the reason. This is the one place where the two sides could be confused, so it is the one
    // place the distinction is written out.
    if (channel.found()) {
        for (const SemChanCandidate& candidate : channel.candidates) {
            const auto it = std::find(out.pages.begin(), out.pages.end(), candidate.page);
            if (it != out.pages.end()) {
                const std::size_t index = static_cast<std::size_t>(it - out.pages.begin());
                // THE PAGE IS ALREADY IN, THE ORIGIN IS NOT: the origin is appended, not merged
                // away, so the arbiter receives both. `in_both` counts the PAGE once; the second
                // origin is not a second page and is not counted as one.
                if (out.origins_per_page[index].empty()) { ++out.in_both; }
                out.origins_per_page[index].push_back(candidate.provenance);
                continue;
            }
            if (cap != 0U && out.pages.size() >= cap) {
                out.refused_over_cap = true;
                out.refusal = "the candidate-set UNION would exceed the cap: refusing rather than "
                              "truncating (kRecallBudgetEdgePolicy = RefuseNotTruncate). Nothing "
                              "is dropped silently and no page is reordered to make room";
                return out;
            }
            out.pages.push_back(candidate.page);
            out.origins_per_page.push_back({candidate.provenance});
            ++out.from_semchan_only;
        }
    }
    out.candidates = static_cast<std::uint32_t>(out.pages.size());
    for (const std::vector<SemChanProvenance>& origins : out.origins_per_page) {
        if (origins.size() > 1U) { ++out.pages_with_multiple_origins; }
    }
    return out;
}

// The diagnostic line. It prints the two sides SEPARATELY and never adds them: `in_both` counts
// pages the arbiter also named, `added` counts pages only the channel named, and their
// denominators are different sets. The project rule is "比集合不比个数" -- compare SETS, not counts
// -- so the line names the sets it compared, and it prints the origin multiplicity because that is
// the field the provenance criterion is decided on.
[[nodiscard]] inline std::string semchan_union_line(const SemChanUnion& u) {
    std::string out = "[semchan-union] arbiter_found=";
    out += (u.arbiter_found ? "1" : "0");
    out += " anchor=" + std::to_string(u.anchor_page);
    out += " anchor_preserved=" + std::to_string(u.anchor_preserved ? 1 : 0);
    out += " determined_by_semchan=0";
    out += " from_arbiter=" + std::to_string(u.from_arbiter);
    out += " in_both=" + std::to_string(u.in_both);
    out += " from_semchan_only=" + std::to_string(u.from_semchan_only);
    out += " union_pages=" + std::to_string(u.candidates);
    out += " pages_with_multiple_origins=" + std::to_string(u.pages_with_multiple_origins);
    out += " cap=" + std::to_string(u.cap);
    if (u.refused_over_cap) { out += " REFUSED_OVER_CAP"; }
    if (!u.refusal.empty()) {
        out += " refusal=";
        out += u.refusal;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 8. THE COMPILE-TIME HALF -- the refusals as build errors
// ---------------------------------------------------------------------------

// A refused KVMem scorer is a build error at this line if a future edit implements one without
// saying so, and the "may determine" predicate is pinned so that admitting the channel to a
// determination is not a diff anyone can make quietly.
static_assert(semchan_scorer_is_implemented(SemChanScorer::MeanK),
              "mean-k is the one scorer this header implements");
static_assert(!semchan_scorer_is_implemented(SemChanScorer::PerToken),
              "per-token is declared and refused: it needs the per-token K this layer has not got");
static_assert(!semchan_scorer_is_implemented(SemChanScorer::SubBlockMeanK),
              "sub-block-mean-k is declared and refused: no K, no sub-block mean");
static_assert(!semchan_scorer_is_implemented(SemChanScorer::KeyDirectionFixed4),
              "key-direction-fixed4 is declared and refused: directions need the K vectors");
static_assert(!semchan_scorer_is_implemented(SemChanScorer::KeyDirectionAdaptive),
              "key-direction-adaptive is declared and refused: prototypes need the K vectors");
static_assert(!semchan_scorer_may_determine(SemChanScorer::MeanK),
              "a similarity score may narrow the candidate set; it may never determine "
              "(sum_dir_query_key.h:37-39/:121, program_impl.h [text: candidate-set UNION, never in place of this arbiter; = :15950 on 2026-09-25])");
static_assert(!kSemChanMayDetermine,
              "the semantic channel is a candidate SOURCE. If this is ever flipped, the arbiter's "
              "exact-or-silent property (program_impl.h [text: exact-or-silent, never approximately right; = :15948 on 2026-09-25]) has been traded away and the "
              "trade must be argued for in the open, not made by editing a constant");
static_assert(!semchan_origin_may_be_chosen_by_channel(),
              "the channel pushes every origin it found and chooses none of them; choosing is the "
              "arbiter's act");

// THE ORIGIN POLICY MUST BE NAMED, the same way `kRecallBudgetEdgePolicy != Unnamed` is required
// at program_impl.h:15090-15092: an origin policy that is Unnamed is a disagreement left silent.
static_assert(kSemChanOriginPolicy != SemChanOriginPolicy::Unnamed,
              "the origin policy must be NAMED: 'a candidate with no origin' is the exact shape "
              "KVMem's re-based frame produces (its own P0 issue KVMI-011), and leaving the "
              "response to it unnamed is how it stays silent");

// THE TREE'S OWN REFUSAL, RE-ASSERTED HERE AS AN INDEPENDENT READER. `sum_dir_query_key.h` already
// refuses `EmbeddingVector` as a KEY producer (its own static_assert at :142). This file declares
// the same refusal a second time, from a different file, so that a single edit inside that header
// cannot silence the refusal for the whole tree: the two readers would have to be edited together,
// in two files, which is a different act from editing one line.
static_assert(!sum_dir::query_key::sum_dir_query_key_source_admitted(
                  sum_dir::query_key::SumDirQueryKeySource::EmbeddingVector),
              "a similarity score may enter the candidate-set UNION and nothing else "
              "(sum_dir_query_key.h:121); this channel is exactly that source");
static_assert(
    std::string_view(sum_dir::query_key::sum_dir_query_key_source_reason(
                         sum_dir::query_key::SumDirQueryKeySource::EmbeddingVector)) !=
        std::string_view(),
    "the tree's refusal of the embedding-vector source must carry a reason, or the refusal is a "
    "silence rather than an answer");

// THE GRANULARITY IS THE PAGE, THE SAME ONE `sum_dir.h` PINS. Not a new constant: this is a
// re-read of the directory's own block, so a channel that returned pages at a different
// granularity than the arbiter would be un-unionable by construction, and that is a build error.
static_assert(kSemChanMaxBlockTokens == sum_dir::kSumDirBlockTokens,
              "the channel's block must be the directory's block, or a union of the two is "
              "meaningless");
static_assert(sum_dir::kSumDirBlockTokens == 64U,
              "the directory's block is the engine's Paged-KV page (sum_dir.h:138-139)");

} // namespace ninfer::spec::semchan
