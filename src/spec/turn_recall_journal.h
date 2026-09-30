#pragma once

// ⑥ per-round external recall ("逐轮外挂召回"): the three pieces the tree is missing.
//
// WHAT IS ALREADY IN THE TREE (the four legs -- reused here, never re-implemented):
//
//   P1 spill      program_impl.h:10610-10887  enqueue_cold_compressions() packs one
//                                             logical page (64 tokens, ALL text layers)
//                                             into each layer's cold-slot bytes and
//                                             mirrors them into that layer's spill file
//                                             at offset `file_slot * slot_stride`;
//                                             file_slot allocated at :10588-10596,
//                                             entry recorded in
//                                             SequenceState::cold_pages
//                                             (program.h:487-492).
//   P2 read-back  program_impl.h:11023-11158  restore_cold_page(): file -> pinned
//                                             staging (cold_disk_staging) -> device cold
//                                             slot -> rANS/int8 decode into the K/V
//                                             planes -> publish the physical page index.
//   P3 prefetch   program_impl.h:11163-11190  prefetch_cold_pages(): pre-read a batch of
//                                             file regions into pinned staging so the read
//                                             latency hides behind the previous decode.
//   P4 key type   prefix_identity.h:42-57     PrefixShortlistDigests::at(frontier) -- one
//                                             rolling 128-bit digest per token frontier;
//                                             used as CheckpointSummary::shortlist_key
//                                             (program_impl.h:7182-7200).
//
// WHAT IS MISSING (the reason this header exists):
//
//   P5 per-round trigger     warm_cold_prefix() is called from exactly ONE place
//                            (program_impl.h:9426, inside start_sequence's retained-
//                            materialization branch) and describes itself as
//                            "rewrite/resume paths only"; it is in no decode_*_batch.
//                            => the steady-state round never re-reads the spill.
//                            The hook lives in program_impl.h
//                            (ensure_sequence_kv_mapped_for_round); this header only
//                            carries the decision, never the hook.
//   P6 page selection        warm_cold_prefix(sequence, end_page) restores [0, end_page)
//                            unconditionally, in page order, with no notion of "which
//                            pages do I actually need" and no byte budget.
//                            => RecallPagePlan / plan_recall_pages() below.
//   P7 persistent log        restore_cold_page() ends by RELEASING the file slot and
//                            ERASING the cold_pages entry (:11149-11156), and
//                            cold_frontier is reset to 0. The (page -> file_slot)
//                            mapping is therefore memory-only AND deleted: after a
//                            restore, "which spill region holds span [a,b)" is
//                            unanswerable -- also across processes.
//                            => TurnRecallJournal below.
//
// EXCHANGE RATES (measured; TODO.md:5830-5831, :5884, :6182 -- quoted, not re-derived):
//
//   SSD read of packed KV        2.6 us/token (7 GB/s) .. 6.1 us/token (3 GB/s)
//   re-prefill the same token    365 us/token (2740 tok/s, in-tree load) .. 690 us
//   one decode step              38.5 ms/token
//   per-token streaming of 1M    18.00 GiB/step = 0.36 tok/s  <-- FORBIDDEN
//
//   => batched read : re-prefill = 60-260 : 1 (buy compute with bandwidth, cheaply).
//   => per-token streaming is ~10^3.6 WORSE than re-prefill: it is not "expensive",
//      it is slower than doing nothing. Everything below therefore exists to keep the
//      read BATCHED and ONCE PER ROUND, and to REFUSE rather than approximate.
//
// CODEC POLICY (in code, not in a comment): nvfp4 (18,432 B/token) or int8
// (33,792 B/token). rk4v4 is REFUSED: it would save 5.6% of the payload while its
// measured exact-prefix accuracy is 9/448 = 0.020 (chance) against nvfp4 0.949 and
// int8 0.977 (TODO.md §44.1) -- the rk4v4 arm is a structural failure, so no external
// recall tier may depend on it.
//
// HONESTY RULES THIS HEADER ENCODES:
//
//   1. The journal is a MIRROR, never the only replica of an attended page. (The tree
//      says the same about the spill files: "spill 文件是 mirror 不是 tier".) Losing
//      the last record of a crash costs a re-read, never correctness; that is why the
//      default flush policy is "no fsync".
//   2. The digest in a record is a SHORTLIST, not an identity. prefix_identity.h:37-39
//      says exactly that: "This is only a content shortlist: exact token and
//      ResidentPrefixIdentity comparison remains authoritative for reuse." The caller
//      must still go through the engine's own exact comparison
//      (prefix_matches / reuse_base) before publishing restored bytes. Nothing here
//      authorizes reuse on its own.
//   3. A hole is a REFUSAL, not a clamp. Pages without a live record are counted and
//      dropped; the run is cut at the first hole rather than filled with a neighbour's
//      bytes (KV is a prefix function: page p's bytes are only valid under the prefix
//      that produced them, `KV(p) = f(T[0..p))`).
//   4. No token-space search happens here. "Which span do I want back?" is asked of the
//      retrieval layer -- suffix_lookup (tail-anchored; measured R@1 0.73-1.00 on the
//      verbatim-at-end shape only) or an any-position/BM25 index for the medial and
//      reworded shapes (TODO.md:5890-5893). This header only answers "where do those
//      pages' bytes live, are they still there, and what does reading them cost".
//
// Host-only, std-only: no CUDA, no engine headers, unit-testable with plain g++ --
// the same shape as serve/kv_cold_policy.h and product/kv_cold_tier_budget.h.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

// NAMESPACE: `ninfer::spec::turn_recall`, deliberately nested rather than `ninfer::spec`.
// A second in-flight recall header (scratch/rc2's src/spec/turn_recall.h) also lives in
// `ninfer::spec` and declares `RecallCodec` (with a different underlying type) and
// `kRecallPageTokens`; the two files are complementary -- that one owns the token-space
// index, this one owns the file-backed journal and the page plan -- and they will be
// included from the same translation unit, so the collision is real and not theoretical
// (measured: including both in one TU fails to compile on exactly those two names). A
// nested namespace keeps the two patches independently landable, which matters more than
// the shorter spelling. If they are ever unified into one header, this namespace is the
// thing to delete first.
namespace ninfer::spec::turn_recall {

// ---------------------------------------------------------------------------
// codec
// ---------------------------------------------------------------------------

inline constexpr std::uint32_t kRecallPageTokens          = 64;    // kPagedKVPageSize
inline constexpr std::uint32_t kRecallNvfp4BytesPerToken  = 18432; // 4.50 b/el
inline constexpr std::uint32_t kRecallInt8BytesPerToken   = 33792; // 8.25 b/el
inline constexpr std::uint32_t kRecallRk4v4BytesPerToken     = 17408; // refused, see above

enum class RecallCodec : std::uint8_t {
    // [CODECBLIND] CORRECTED 2026-09-22. The two claims this enumerator carried had both
    // been contradicted by the code beside them:
    //   * "or a table mixing codecs" -- `Mixed` is ADMITTED (`recall_codec_admitted`
    //     below, since 2026-09-18, MIXEDPROOF/REPORT.md): both cold IO legs are PER LAYER,
    //     so one region descriptor covers a two-stride page.
    //   * "a table containing rk4v4" was NEVER said here, and that omission was the defect:
    //     a table with rk4v4 beside an admitted codec is classified as the ADMITTED one and
    //     its rk4v4 layers are invisible (see `CodecBlindPolicy` below).
    Rejected = 0, // rk4v4, bf16, fp8, iso4e: a codec the journal cannot describe, and the
                  // only value the classifier returns when it can name NOTHING.
    Nvfp4    = 2,
    Int8     = 3,
    Mixed    = 4, // several codecs across the layers of one page, each layer addressed at
                  // its OWN extent: ADMITTED, not refused (program_impl.h's two cold IO
                  // legs are per-layer and the record's codec byte is a findability filter)
};

[[nodiscard]] inline const char* recall_codec_name(RecallCodec codec) noexcept {
    switch (codec) {
    case RecallCodec::Nvfp4: return "nvfp4";
    case RecallCodec::Int8: return "int8";
    case RecallCodec::Mixed: return "mixed";
    case RecallCodec::Rejected: break;
    }
    return "rejected";
}

// The ONE admission decision. rk4v4 is refused with its measurement, not by omission.
// constexpr so the refusal can be asserted at COMPILE time (a regression that admitted
// rk4v4 would fail the build, not a test run).
//
// `Mixed` IS ADMITTED. The 2026-09-18 revision of this rule was a REFUSAL whose stated
// reason -- "a page whose layers use different codecs has no single stride" -- is FALSE
// AS A CLAIM ABOUT WHAT THE ENGINE NEEDS, and it was measured rather than argued
// (PATCHSET/MIXEDPROOF/REPORT.md):
//   * BOTH IO legs are PER LAYER. The spill writes `cold_disk_files[layer]` at
//     `file_slot * cold_slots.nb[3]` (program_impl.h:11814-11833) and the restore reads
//     the same pair (program_impl.h:12559-12579); each layer is addressed at its OWN
//     extent in its OWN file.
//   * The restore's codec comes from `view.dtype` (program_impl.h:12592 NVFP4,
//     program_impl.h:12629 I8), NOT from the record.
//   * Therefore a page whose layers carry two strides needs ONE region descriptor --
//     one `file_slot` and one `layer_bytes` -- not two. The record's single `codec` byte
//     has exactly ONE reader in the whole tree, `index()` below, where it is a
//     FINDABILITY filter and never an offset.
// So the refusal was over-strict, and because the classifier is STACK-LEVEL
// (program_impl.h:1071, one call) it cost recall for EVERY spilled page of a two-codec
// stack, not only for the two-codec ones. It is admitted here; what replaces it is the
// per-layer view, which is where the read path already got its codec.
[[nodiscard]] constexpr bool recall_codec_admitted(RecallCodec codec) noexcept {
    return codec == RecallCodec::Nvfp4 || codec == RecallCodec::Int8 ||
           codec == RecallCodec::Mixed;
}

[[nodiscard]] constexpr std::uint32_t
recall_codec_bytes_per_token(RecallCodec codec) noexcept {
    switch (codec) {
    case RecallCodec::Nvfp4: return kRecallNvfp4BytesPerToken;
    case RecallCodec::Int8: return kRecallInt8BytesPerToken;
    // NOT a byte rate. A Mixed page has one rate PER LAYER and no single one, so there is
    // no number this function could return that a cost model may multiply by a token
    // count. 0 is the honest "not a single codec's rate" and it is safe because this
    // function has no runtime caller: its consumers are the static_asserts below and in
    // sum_dir.h, all of which ask about Nvfp4/Int8 only. The page's real cost is
    // `turn_recall_page_bytes()` (program_impl.h:12845-12855), the SUM over the
    // slot-bearing layers -- which is what the writer already stores (program_impl.h:12926)
    // and what the plan arithmetic already charges (program_impl.h:12845 -> :12989 ->
    // turn_recall_journal.h:738/:813/:825). If a budget input ever needs Mixed's number,
    // it must call that function and not this one.
    case RecallCodec::Mixed: return 0;
    case RecallCodec::Rejected: break;
    }
    return 0;
}

// rk4v4 can never be admitted; the tier's own arithmetic is checked at compile time too.
static_assert(!recall_codec_admitted(RecallCodec::Rejected),
              "rk4v4/bf16/fp8/iso4e are not externally recallable");
// The Mixed row is the OTHER direction, and the reason it is stated here is the reason
// the old sentence had to go: the old one named a requirement ("one stride") that no
// consumer has. `layer_bytes` is documented as a per-LAYER stride but is WRITTEN as the
// per-page SUM (program_impl.h:12926 -> turn_recall_page_bytes(), :12845-12855), so it
// could not have been the single-stride check the old sentence promised even if it were
// read -- and it has no reader at all (MIXEDPROOF REPORT.md sec.3 claim 3).
static_assert(recall_codec_admitted(RecallCodec::Mixed),
              "a two-stride page IS recallable: both IO legs address the layer's own file "
              "at the layer's own nb[3] (program_impl.h:11814-11833 / :12559-12579) and "
              "the restore's codec comes from view.dtype, not from this record; one region "
              "descriptor therefore covers every layer");
static_assert(recall_codec_admitted(RecallCodec::Nvfp4) &&
                  recall_codec_admitted(RecallCodec::Int8),
              "nvfp4 and int8 are the two admitted tiers");
static_assert(recall_codec_bytes_per_token(RecallCodec::Nvfp4) <
                  recall_codec_bytes_per_token(RecallCodec::Int8),
              "nvfp4 is the smaller tier");

[[nodiscard]] inline std::string recall_codec_refusal(RecallCodec codec) {
    // `Mixed` no longer reaches here (recall_codec_admitted admits it), and the branch is
    // kept because the honest text for it is the record of WHY it stopped being a
    // refusal. The old string -- "the page's layers do not share one cold codec, so a
    // single record cannot describe the stride" -- asserted a requirement the engine does
    // not have; it is replaced by a statement of what is actually true if it ever fires
    // again (i.e. if someone re-narrows recall_codec_admitted).
    if (codec == RecallCodec::Mixed) {
        return "turn recall refused: Mixed was re-narrowed out of recall_codec_admitted. "
               "That rule refuses a page the engine CAN serve -- both cold IO legs are "
               "per-layer (program_impl.h:11814-11833 / :12559-12579) and the record's "
               "codec byte is only a findability filter (index()), so one region "
               "descriptor covers a two-stride page. See MIXEDPROOF/REPORT.md.";
    }
    return "turn recall refused: the KV tier is not nvfp4/int8/mixed. rk4v4 would save 5.6% "
           "of the payload while its measured exact-prefix accuracy is 9/448 = 0.020 (nvfp4 "
           "0.949, int8 0.977), i.e. a structural failure -- not a candidate for an "
           "external recall tier";
}

// ---------------------------------------------------------------------------
// [CODECBLIND] THE CLASSIFIER'S BLIND SPOT, NAMED INSTEAD OF SILENT (2026-09-22)
// ---------------------------------------------------------------------------
//
// WHAT IS WRONG, MEASURED RATHER THAN ARGUED. The tree has exactly ONE producer of the
// admission decision: the target's classifier, `ProgramImplCore::turn_recall_codec_of_layers()`
// (program_impl.h). It skips every layer with no cold slot and then reads
//
//     if (view.dtype == DType::NVFP4) { nvfp4 = true; }
//     else if (view.dtype == DType::I8) { i8 = true; }
//
// with **no further arm**. A cold-slot-bearing layer of any OTHER dtype therefore sets
// NEITHER flag, and the consequence is not symmetric:
//
//   * PURE -- rk4v4 only: nothing is named, the classifier returns `Rejected`,
//     `recall_codec_admitted` refuses it, and program_impl.h prints
//     `[recall] RECALL DISABLED for this run: ...`. LOUD already; not the defect.
//   * BESIDE AN ADMITTED CODEC -- e.g. the factory tier table's rk4v4 layers next to its
//     nvfp4 ones: the classifier returns `Nvfp4`, it is ADMITTED, the chain fires, and
//     NOTHING said that the rk4v4 layers were invisible to the decision. SILENT. This is
//     the defect, and it is not cosmetic: `ProgramImplCore::turn_recall_page_bytes()`
//     sums `cold_slots.nb[3]` over EVERY slot-bearing layer, so the page whose stride the
//     record describes CAN contain bytes written by a codec the journal refuses to
//     describe. The classifier's answer and the record's stride are therefore about two
//     different sets of layers, and only one of them was ever printed.
// (The same narrowness is described independently, in another line's own words, at
// `src/product/kv_plane_census.h` -- which also records that this classifier's comment
// claimed "Mixed ... admits nothing" long after `recall_codec_admitted` had admitted it.)
//
// WHAT THIS LANDS, AND WHAT IT DELIBERATELY DOES NOT. The count of unnamed
// slot-bearing layers gains a NAME, a LINE, a COUNTER and a POLICY, so the silent case
// stops being silent AND the decision to keep firing (or to refuse) is on the record:
//
//   * `ReportOnly` (the shipped value below) executes today's run UNCHANGED and prints the
//     blindness by name. No existing capture moves; the run is bit-identical.
//   * `RefuseRun` refuses the whole run by name -- `refused-codec-blind`, before any page is
//     spilled or restored. It is one token away and it is deliberately NOT taken here:
//     refusing an rk4v4-bearing stack turns the FACTORY DEFAULT's recall off, which is an
//     ADMISSION change and belongs to the line that owns admission, not to the line that
//     measured the blindness. Whoever owns admission changes
//     `kCodecBlindPolicy = CodecBlindPolicy::RefuseRun;` and nothing else.
//   * `Unnamed` is refused by an engine-side `static_assert`, so the choice cannot be left
//     unmade silently -- the `kInexactAdmissionPolicy` / `kRecallBudgetEdgePolicy` idiom
//     this header already uses twice.
//
// RE-ADMITTING rk4v4 INTO THE TIER is a DIFFERENT question and stays refused where it
// already is, with its measurement: `recall_codec_refusal` above states 9/448 = 0.020
// exact-prefix against nvfp4 0.949 / int8 0.977. This block does not reopen it.
enum class CodecBlindPolicy : std::uint8_t {
    Unnamed = 0,  // not a policy: the ABSENCE of one. Engine-side static_assert.
    ReportOnly,   // today's behaviour, NAMED: the run executes and the blindness is printed
    RefuseRun,    // the named refusal `refused-codec-blind`; nothing is spilled or restored
};
inline constexpr CodecBlindPolicy kCodecBlindPolicy = CodecBlindPolicy::ReportOnly;

[[nodiscard]] inline constexpr const char* codec_blind_policy_name(
    CodecBlindPolicy policy) noexcept {
    switch (policy) {
        case CodecBlindPolicy::Unnamed: return "unnamed";
        case CodecBlindPolicy::ReportOnly: return "report-only";
        case CodecBlindPolicy::RefuseRun: return "refuse-run";
    }
    return "unknown";
}

// The refusal's own name, so a run, a test and a report cannot spell it three ways (the
// `refused-prefill-budget` rule this header states for the token budget). A plain constant,
// not a `[[nodiscard]]` one: the attribute is meaningless on a variable and gcc says so.
inline constexpr const char* kCodecBlindRefusalName = "refused-codec-blind";

// The decision as a PURE FUNCTION of the count and the policy, so the engine's branch is one
// call and a host test can exercise every policy against every count with no GPU and no
// decoder. The count itself is produced by the TARGET layer, which is why this header stays
// free of `DType` (src/spec/ depends on no other tree layer, and this change keeps it so).
//
// The `blind_layers == 0` arm is the one a careless edit loses: a predicate written
// `return policy == RefuseRun;` would refuse a stack with NOTHING blind in it, i.e. it would
// refuse the homogeneous nvfp4 tier that recall works on. The sibling test has a RED arm
// for exactly that.
[[nodiscard]] inline bool codec_blind_refuses_run(std::uint32_t blind_layers,
                                                 CodecBlindPolicy policy) noexcept {
    if (policy != CodecBlindPolicy::RefuseRun) { return false; }
    return blind_layers != 0;
}

// ---------------------------------------------------------------------------
// P7: the persistent L0 journal
// ---------------------------------------------------------------------------
//
// One fixed 64-byte record per (re)spill or release of ONE logical page -- 64 tokens
// across ALL text layers, because that is the block table's addressing unit and the
// unit the sentinel encodes (cold_host_tier.h:34-41), not a per-layer or per-head one.
//
// Records are appended in the order the pages are spilled, and are replayed on load to
// rebuild the (digest, page) -> file_slot map, so the map survives the process that
// wrote it. `sequence` disambiguates a reused page number across sessions; tombstones
// (`kind = Release`) are what stop a replay from pointing at a file region that has
// since been handed to another page (the file-slot allocator is process-global and the
// space IS reused -- program.h:783-789).
//
// KEY = (digest, page). Why not a similarity score: KV is a prefix function, so the
// only legal recall is a recall whose generating prefix is byte-identical, and the
// engine's own rolling digest at frontier=(page+1)*page_tokens is precisely the
// certificate of "T[0, frontier) is this". A span of text that merely *looks* like the
// remembered one is not recallable, which is why the n-gram-as-KV-substitute idea
// measures 0.000% coverage.
//
// VALUE = file_slot (+ the stride the writer used, so a reader with a different
// geometry refuses instead of mis-reading).

inline constexpr std::uint32_t kRecallJournalMagic   = 0x314C4352U; // 'R','C','L','1'
inline constexpr std::uint8_t  kRecallJournalVersion = 1U;
inline constexpr std::uint32_t kRecallRecordBytes    = 64U;
inline constexpr std::uint32_t kRecallCrcBytes       = 48U; // crc32 covers [0, 48)

enum class RecallKind : std::uint8_t {
    Spill   = 1, // the page's bytes now live at file_slot
    Release = 2, // they do not any more (restored hot, sequence gone, slot recycled)
};

enum class RecallFlags : std::uint8_t {
    None          = 0,
    GdnStateInLog = 1U << 0, // the GDN recurrent state for this frontier is in the log
                            // too: it is NOT part of the KV pages, so a recall that
                            // needs it must load it or recompute it (never silently
                            // proceed with a stale recurrent state)
};

struct RecallRecord {
    std::uint32_t magic       = kRecallJournalMagic; //  0
    std::uint8_t  version     = kRecallJournalVersion; //  4
    std::uint8_t  kind        = static_cast<std::uint8_t>(RecallKind::Spill); // 5
    std::uint8_t  codec       = static_cast<std::uint8_t>(RecallCodec::Rejected); // 6
    std::uint8_t  flags       = 0;                   //  7
    std::uint32_t page        = 0;                   //  8 logical page (64 tokens x all layers)
    std::uint32_t layer_bytes = 0;                   // 12 per-layer slot stride as written
    std::int32_t  file_slot   = -1;                  // 16 -1 = no on-disk region
    std::uint32_t sequence    = 0;                   // 20 session/sequence counter
    std::uint32_t frontier    = 0;                   // 24 (page + 1) * kRecallPageTokens
    std::uint32_t reserved    = 0;                   // 28
    std::uint64_t digest_lo   = 0;                   // 32 PrefixShortlistDigests::at(frontier)
    std::uint64_t digest_hi   = 0;                   // 40
    std::uint32_t crc32       = 0;                   // 48
    std::uint32_t reserved2   = 0;                   // 52
    // 56/60: THE SHARD AXIS. These were `pad0`/`pad1`. The byte layout is unchanged
    // (still two 32-bit words, sizeof still 64, the static_asserts below still hold), but
    // the fields are now NAMED, because a multi-device world needs exactly this and a
    // pad field cannot carry a rule.
    //
    // (0, 1) means "a single-device world" and is the DEFAULT, so every record written
    // before this change reads back as the world it was actually written in. A reader MUST
    // refuse a record whose (shard_rank, shard_world) is not its own -- the same shape of
    // guard `layer_bytes` already carries ("the stride the writer used, so a reader with a
    // different geometry refuses instead of mis-reading"). Leaving this as padding is what
    // lets two ranks write the same journal region and read each other's pages.
    std::uint32_t shard_rank  = 0;                   // 56
    std::uint32_t shard_world = 1;                   // 60
    // 64 bytes, all four of them explicit: the record is one fixed unit, and a torn
    // append loses at most this record. The padding is spelled out so that the byte
    // layout is the same whether it is read by this struct, by a packed reader, or by
    // an offline tool (sizeof is 64 by alignment, and the explicit pad1 keeps a packed
    // reader at 64 as well instead of relying on that).
};

static_assert(sizeof(RecallRecord) == kRecallRecordBytes,
              "the L0 record must stay one fixed 64-byte unit");
static_assert(kRecallRecordBytes == 64U, "record stride is the log's only format guarantee");

// ===========================================================================
// THE RECORD'S WORD LEDGER -- ONE AUTHORITY, COMPLETE AND DISJOINT
// ===========================================================================
//
// The record is sixteen four-byte words. Every one of them is listed here EXACTLY ONCE with
// the thing that owns it, and the ledger's completeness is a static_assert rather than a
// comment -- so a word silently claimed by a new field, or a field renamed onto a word this
// ledger still calls free, is a COMPILE ERROR at this one place instead of a number that
// goes stale in a comment three hundred lines away. That IS the defect this block replaces:
// the previous ledger's free-word list still named `pad0`/`pad1` at 56/60 long after they
// had become the shard axis, and still counted `reserved`@28 as free although the release
// reason had already taken it.
//
//   offset  word                       owner
//   ------  -------------------------  -----------------------------------------
//      0    magic                      payload
//      4    version/kind/codec/flags   payload
//      8    page                       payload
//     12    layer_bytes                payload  <- PRIMARY region descriptor (the stride)
//     16    file_slot                  payload  <- PRIMARY region descriptor (which region)
//     20    sequence                   payload
//     24    frontier                   payload
//     28    reserved                   RELEASE REASON (low byte live)
//     32    digest_lo                  payload  (2 words)
//     40    digest_hi                  payload  (2 words)
//     48    crc32                      payload
//     52    reserved2                  UNASSIGNED
//     56    shard_rank                 SHARD AXIS
//     60    shard_world                SHARD AXIS
//
// WHY `reserved`@28 IS NOT FREE -- the correction this ledger exists to make permanent.
// recall_release_reason() reads `record.reserved & 0xFF`; recall_set_release_reason() WRITES
// THE WHOLE WORD (`record.reserved = (record.reserved & 0xFFFFFF00U) | reason`); and that
// writer is LIVE at program_impl.h:11351 (RecallReleaseReason::SequenceEnd). A second region
// descriptor would put its `layer_bytes` at @28 and have that stride's low byte overwritten
// by the next release -- so the word is OCCUPIED even though three of its four bytes read as
// zero in every record on disk. Counting it as free is exactly what let the old ledger
// reach R=2.
//
// AND R>=2 IS NOT WHAT A TWO-STRIDE PAGE NEEDS. The ledger's arithmetic above was read as
// "two strides therefore two regions", and that step is the unexamined premise the old
// sentence below used to reach the Mixed refusal. It is wrong: both cold IO legs are
// PER LAYER -- the spill writes `cold_disk_files[layer]` at `file_slot * cold_slots.nb[3]`
// (program_impl.h:11814-11833) and the restore reads that same pair (program_impl.h:12559-12579),
// and the restore's codec comes from `view.dtype` (program_impl.h:12592/:12629), never from
// this record. ONE `file_slot` therefore addresses every layer of a page, so a page with two
// strides needs one region descriptor and Mixed is admitted (see recall_codec_admitted). What
// the byte budget below still bounds is a WORLD that wants two genuinely different REGIONS,
// AND R=2 IS NOT NEEDED, SO LOSING IT COSTS NOTHING. The old ledger's `>= 1U` guarantee was
// read as "a split world needs a second region or it cannot record where its bytes went".
// That premise is false on this tree, in the paragraph immediately above: R>=2 on the
// HEAD-axis split keeps ONE stride per layer by construction, so it fits the record
// unchanged. What does NOT fit, and never did, is R>=2 WITH a per-region codec -- the
// `codec` field is one byte for the whole page, two regions need two of them, and the
// honest pool below is four bytes with no whole byte to spare. The layer-axis split is a
// separate question and is NOT settled by this paragraph.

inline constexpr std::uint32_t kRecallRegionDescriptorBytes = 8U;  // layer_bytes + file_slot

// The ledger's three numbers. These are the ONLY places these counts are written down.
inline constexpr std::uint32_t kRecallRecordFreeBytes    = 16U; // words no payload field owns
inline constexpr std::uint32_t kRecallReleaseReasonBytes = 4U;  // `reserved` @28
inline constexpr std::uint32_t kRecallShardAxisBytes     = 8U;  // rank @56 + world @60

static_assert(kRecallRecordFreeBytes + 12U * sizeof(std::uint32_t) == kRecallRecordBytes,
              "the ledger must be COMPLETE AND DISJOINT: 16 free words + the 12 payload words "
              "listed in the table above must add up to the record exactly, 64 bytes. If a "
              "field is added, removed or moved, fix the table above and this fires");

// ---------------------------------------------------------------------------
// THE ONE AUTHORITY. Every other site -- this file's own static_asserts, the region
// predicate in core/shard_rank_axis.h, and all three test TUs -- is asserted AGAINST these
// functions and never against a repeated literal. A second literal that happens to agree
// today is how the previous ledger went stale.
// ---------------------------------------------------------------------------

// The bytes a NEW field could actually occupy: what no payload field owns, minus the word
// the release reason has already taken, minus the shard axis. The remainder is `reserved2`
// @52, which no reader anywhere in the tree touches.
[[nodiscard]] constexpr std::uint32_t recall_record_free_bytes_honest() noexcept {
    return kRecallRecordFreeBytes - kRecallReleaseReasonBytes - kRecallShardAxisBytes; // 4
}

// The number the RAW subtraction reaches -- what the previous ledger claimed, and what
// core/shard_rank_axis.h's predicate read until this change. Kept as a NAMED, asserted
// quantity rather than deleted, because the difference between it and the honest one IS the
// correction, and a difference nobody can name is a difference nobody can test.
[[nodiscard]] constexpr std::uint32_t recall_record_free_bytes_raw_after_shard_axis() noexcept {
    return kRecallRecordFreeBytes - kRecallShardAxisBytes; // 8
}

// Extra region descriptors the record can describe ONCE the shard axis is present. A
// descriptor is kRecallRegionDescriptorBytes of CONTIGUOUS bytes (the primary region's pair
// sits at @12/@16); the honest pool is one word, so this is ZERO -- there is no room for a
// second region, which is the honest replacement for the previous `1`.
[[nodiscard]] constexpr std::uint32_t
recall_extra_regions_after_shard_axis() noexcept {
    return recall_record_free_bytes_honest() / kRecallRegionDescriptorBytes; // 0
}

// The assertions that turn the ledger above into a build-time fact. The FIRST is the one the
// old ledger had backwards: it read `>= 1U` because it counted the release reason's word as
// free. `== 0U` is the honest guarantee.
static_assert(recall_extra_regions_after_shard_axis() == 0U,
              "the record has room for the PRIMARY region (@12/@16, both NAMED fields) and "
              "for NO second one: a second descriptor needs 8 CONTIGUOUS bytes and the "
              "honest pool is 4. If this ever passes, a word this ledger calls occupied "
              "became free (or vice versa) and the whole accounting must be redone");
static_assert(recall_record_free_bytes_honest() < kRecallRegionDescriptorBytes,
              "the honest pool must be SMALLER than one descriptor, and that inequality is "
              "the REASON the count above is 0. Stated separately so the reason survives if "
              "the count is ever re-derived by a different route");
static_assert(recall_record_free_bytes_raw_after_shard_axis() >
                  recall_record_free_bytes_honest(),
              "the raw subtraction MUST exceed the honest one, because the one word it "
              "miscounts (`reserved`@28) is the release reason and a release reason is a "
              "live writer's field, not slack. THIS IS THE CORRECTION, asserted");
static_assert(recall_codec_admitted(RecallCodec::Mixed),
              "a two-stride page is served by ONE region descriptor, because the cold IO "
              "legs are per-layer; the byte budget above bounds extra REGIONS, and it does "
              "not bound Mixed. Re-narrowing this line re-breaks every spilled page of a "
              "mixed stack, not just the mixed ones (the classifier is stack-level)");

// ===========================================================================
// P7 -- THE DURABLE PAGE -> REGION MAPPING: WHAT IT NEEDS (and what it must NOT cost)
// ===========================================================================
//
// THE QUESTION THIS SECTION ANSWERS, from the header's own "WHAT IS MISSING" list:
// restore_cold_page() ends by RELEASING the file slot and ERASING the cold_pages entry
// (program_impl.h:11149-11156), and cold_frontier is reset to 0, so "after a restore,
// which spill region holds span [a,b) is unanswerable -- also across processes". Does the
// on-disk record already carry enough to rebuild the mapping, or does the format need a
// field?
//
// ---------------------------------------------------------------------------
// 1. THE MAPPING IS ALREADY DURABLE. The header above says so, and index() is the proof.
// ---------------------------------------------------------------------------
//
// One fixed 64-byte RecallRecord per (re)spill or release of ONE logical page, carrying
//
//     page (8)          -- the logical page, i.e. the block-table coordinate
//     file_slot (16)    -- WHERE the bytes went
//     layer_bytes (12)  -- the stride the writer used, so a foreign geometry REFUSES
//     sequence (20)     -- disambiguates a reused page number across sessions
//     frontier (24), digest_lo (32), digest_hi (40) -- the (digest, page) KEY
//     shard_rank (56), shard_world (60) -- so two ranks cannot read each other's pages
//
// and TurnRecallJournal replays the whole log AT OPEN (load() -> index(), :634/:731) into
// live_, which find_live()/live_page()/live_pages() consult. So the (digest, page) ->
// file_slot map DOES survive the process that wrote it. P7 is NOT "there is no format".
//
// ---------------------------------------------------------------------------
// 2. WHAT THE FORMAT GENUINELY CANNOT ANSWER, AND IT IS ONE THING
// ---------------------------------------------------------------------------
//
// A `file_slot` is a PROCESS-LOCAL coordinate, not a durable one. The spill files are
// opened "w+b" (program_impl.h:1032-1033), i.e. TRUNCATED at open, the file-slot
// allocator is process-global, and `truncate` is hard-coded false
// (program_impl.h:1087) -- so this run's slot 7 and the previous run's slot 7 are
// DIFFERENT REGIONS with no recorded relationship. A replayed record is therefore a
// geographically valid description of a file that no longer exists in that shape, and
// nothing in the record can tell: `sequence` disambiguates a reused PAGE number, never a
// reused SLOT, and `layer_bytes` pins the geometry but not the run.
//
// THE FIELD THAT WOULD CLOSE IT, AND WHY IT DOES NOT FIT -- this is byte arithmetic, not
// an opinion, and the ledger above is now the ONE AUTHORITY that states it:
//
//   the honest pool        = recall_record_free_bytes_honest()      = 4 bytes (`reserved2` @52)
//   a region descriptor    = kRecallRegionDescriptorBytes           = 8 CONTIGUOUS bytes
//   => recall_extra_regions_after_shard_axis() == 0
//   => a durable run epoch costs ALL 4 bytes, so the trade is the whole pool.
//
// THIS IS NOW A LANDED FACT AND NOT A WARNING. The previous revision of this section
// reported the miscount and deliberately left the constant alone, on the stated grounds that
// correcting it would break the build and a second line's wiring test. The second half of
// that was right, and larger than it was measured to be: moving the value moves EVERY site
// that pinned the old number, and the sites are not three. They are NINE, in three test
// files plus one engine predicate:
//
//   src/spec/turn_recall_journal.h          (:366, :369)     2 static_assert  <- compile time
//   src/spec/turn_recall_journal_test.cpp   (:71)            1 runtime check
//   tests/test_multidev_wiring.cpp          (:222, :225-226, :247)  3 runtime checks
//   tests/test_shard_rank_axis.cpp          (:391, :402, :406)      3 runtime checks
//   src/core/shard_rank_axis.h              (:410)           recall_region_budget_holds()
//
// THE CORRECT ORDER IS FORCED, and it is not a preference. The header's two static_asserts
// are the ONLY compile-time sites, so the header moves FIRST and every runtime site moves in
// the SAME change:
//
//   header alone      -> five runtime checks still encode the old number: they COMPILE and
//                        then FAIL. A green build over a red suite.
//   a site alone      -> it asserts a value the header no longer derives, from a token the
//                        header no longer exports.
//   header + sites    -> coherent. There is no partial move that is.
//
// So this one is all-or-nothing, and it is landed as one change with the pre-images kept.
// The header still adds NO field and makes NO version bump: the correction is to the
// ACCOUNTING, not to the record. Every site now asserts against the authority above.
//
// The one word a durable epoch would have to occupy is DERIVED from that authority rather
// than restated: it is not "4 bytes, and coincidentally the honest pool is also 4" -- it is
// "the honest pool, which happens to be exactly one word". Two literals that agree today is
// the shape that went stale here before.
inline constexpr std::uint32_t kRecallWordBytes      = 4U;
inline constexpr std::uint32_t kRecallEpochWordBytes = kRecallWordBytes;

static_assert(recall_record_free_bytes_honest() == kRecallEpochWordBytes,
              "the honest free-byte count is exactly one word, and an epoch axis needs ALL of "
              "it: the trade for a durable run epoch is the whole pool, which is why it is a "
              "trade. If this ever fails the ledger above changed and the trade must be "
              "redone before anyone writes a field at @52");

// ---------------------------------------------------------------------------
// 3. THE DESIGN THAT NEEDS NO FIELD AT ALL: MAKE THE REGION COORDINATE THE PAGE
// ---------------------------------------------------------------------------
//
// There is a second way to make the mapping durable, and it costs zero bytes because it
// removes the thing that had to be remembered. Today the on-disk offset of page p is
// `file_slot * stride`, with file_slot handed out by an allocator -- so the map has to be
// remembered. If instead the offset IS `p * stride`, then file_slot == p, the mapping is
// DERIVED rather than stored, and it survives processes, crashes and slot reuse by
// construction.
//
// THIS IS NOT AN INVENTION: it is the rule the text cargo already obeys, stated in this
// very file for the very same reason --
//
//   "THE INDEX IS THE PREFIX'S OWN BLOCK NUMBER, i.e. `token_begin / kRecallPageTokens`.
//    It is NOT the file slot. ... the cargo is addressed by the CONTENT's own coordinate,
//    so the directory never has to expose -- and a reader never has to know -- where a
//    file region happens to sit."  (the cargo format note above)
//
// THE COST, STATED RATHER THAN HIDDEN: disk becomes proportional to the FRONTIER, not to
// the live cold set. At the 1M configuration the reachable frontier is 15,782 pages
// (INDEX1M2 sec.5.5a) and one page's cold slots are 1.13-1.21 MiB across all text layers
// (`turn_recall_page_bytes()`), so a densely page-addressed spill family costs
// 17.4-18.2 GiB of DISK against a `--cold-disk-bytes` default that this line's arms never
// pushed past 2 GiB. That is affordable only with sparse files (a page never spilled leaves
// its region unwritten), and it is the reason this section ships the arithmetic and the
// offset rule but does NOT rewire the writer: the writer's change is a layout change that
// `--cold-disk-bytes`' page arithmetic and the per-layer file sizing both read.
[[nodiscard]] constexpr std::uint64_t
recall_page_region_offset(std::uint32_t page, std::uint64_t page_stride) noexcept {
    return static_cast<std::uint64_t>(page) * page_stride;
}

// The inverse, so the two directions cannot drift: this is the only place the relation is
// spelled, and the round trip is asserted below at the 1M page index.
[[nodiscard]] constexpr std::uint32_t
recall_region_page_of_offset(std::uint64_t offset, std::uint64_t page_stride) noexcept {
    return page_stride == 0 ? 0U : static_cast<std::uint32_t>(offset / page_stride);
}

static_assert(recall_region_page_of_offset(recall_page_region_offset(15781U, 1181696ULL),
                                           1181696ULL) == 15781U,
              "a page-addressed region must round-trip at the 1M page index");

// ---------------------------------------------------------------------------
// 4. WHAT THIS FILE CAN DO WITHOUT ANY OF THAT: A LOCATOR THAT REFUSES
// ---------------------------------------------------------------------------
//
// The capability P5/P6 need is one DECISION -- "for this page, is there a region to fetch
// from, and may I believe it?" -- answerable from the replayed records ALONE, with the
// guards the format already carries. It makes no claim the record cannot support: a record
// whose geometry or world is not the caller's is REFUSED by name (the rule `layer_bytes`
// and the shard axis were added for), a page with a standing Release is GONE and is NOT a
// fetch source, and a page with no record at all is UNKNOWN -- which is a refusal, never a
// guess.
//
// WHY `layer_bytes` IS CHECKED AS THE PER-PAGE SUM: the field is DOCUMENTED as "per-layer
// slot stride as written" (@12) and WRITTEN as the per-page sum (program_impl.h:12926 ->
// turn_recall_page_bytes(), :12845-12855). INDEX1M2 sec.5.2 named exactly this as the reason
// the check could not be wired ("wiring it as-is would install a wrong-stride check"). It is
// wired here against the SUM, because the sum is what EVERY writer has ever written, so no
// existing record is newly refused -- and the ambiguity is closed by a caller that declares
// the SUM it reads with rather than by a guess about which of the two it holds.
enum class RecallPagePlace : std::uint8_t {
    Unknown = 0, // no record has ever stood for this page in this log
    Spill   = 1, // the bytes are at `file_slot`, and the geometry matches
    Gone    = 2, // a Release stands: the region was returned and is NOT a fetch source
    Refused = 3, // a record exists and names another geometry / world / codec
};

[[nodiscard]] inline const char* recall_page_place_name(RecallPagePlace place) noexcept {
    switch (place) {
    case RecallPagePlace::Unknown: return "unknown";
    case RecallPagePlace::Spill: return "spill";
    case RecallPagePlace::Gone: return "gone";
    case RecallPagePlace::Refused: return "refused";
    }
    return "invalid";
}

struct RecallPageLocation {
    RecallPagePlace place       = RecallPagePlace::Unknown;
    std::int32_t    file_slot   = -1;
    std::uint32_t   layer_bytes = 0;
    std::uint32_t   frontier    = 0;
    std::uint32_t   sequence    = 0;
    std::string     refusal; // non-empty iff place == Refused
    [[nodiscard]] bool fetchable() const noexcept { return place == RecallPagePlace::Spill; }
};

// The caller's own geometry: what IT will read with, and which world it is in.
struct RecallLocatorGeometry {
    std::uint32_t page_tokens = kRecallPageTokens;
    std::uint32_t layer_count = 0; // text layers carrying a cold slot (0 => stride unchecked)
    // The PER-PAGE SUM this caller reads with -- `turn_recall_page_bytes()`, the same number
    // the writer stored in `layer_bytes`. 0 => unset, and the stride check is then SKIPPED
    // rather than passed by accident.
    std::uint64_t page_bytes  = 0;
    std::uint32_t shard_rank  = 0;
    std::uint32_t shard_world = 1;
};

class RecallPageLocator {
public:
    explicit RecallPageLocator(RecallLocatorGeometry geometry) : geometry_(geometry) {}

    // REPLAY ORDER, and the SAME rule index() applies (:731): a Spill installs, a Release
    // erases. Sharing the rule is what makes a locator and the journal unable to disagree
    // about a page -- a second spelling here would be the drift this whole header removes.
    void observe(const RecallRecord& record) {
        if (record.magic != kRecallJournalMagic || record.version != kRecallJournalVersion) {
            return;
        }
        if (record.kind == static_cast<std::uint8_t>(RecallKind::Spill)) {
            live_[record.page] = record;
            return;
        }
        live_.erase(record.page);
        // A Release is REMEMBERED as a tombstone, not merely erased: "Gone" is a different
        // answer from "never seen" -- the first says a region WAS this page's and is now
        // reusable, the second says this log never covered the page. Only the second is a
        // hole in the log; collapsing them is how a released region gets read as a live one.
        released_.insert(record.page);
    }

    [[nodiscard]] std::size_t live_count() const noexcept { return live_.size(); }
    [[nodiscard]] std::size_t released_count() const noexcept { return released_.size(); }

    [[nodiscard]] RecallPageLocation locate(std::uint32_t page) const {
        RecallPageLocation location;
        const auto entry = live_.find(page);
        if (entry == live_.end()) {
            location.place = released_.count(page) != 0U ? RecallPagePlace::Gone
                                                         : RecallPagePlace::Unknown;
            return location;
        }
        const RecallRecord& record = entry->second;
        // THE SHARD GUARD. "A reader MUST refuse a record whose (shard_rank, shard_world) is
        // not its own -- the same shape of guard `layer_bytes` already carries" (the shard
        // axis's own comment above). A v1 record reads back as (0, 1), which is the identity
        // world and IS the world those records were written in.
        if (record.shard_rank != geometry_.shard_rank ||
            record.shard_world != geometry_.shard_world) {
            location.place   = RecallPagePlace::Refused;
            location.refusal = "a record for page " + std::to_string(page) + " names shard (" +
                               std::to_string(record.shard_rank) + "," +
                               std::to_string(record.shard_world) + "), this caller is (" +
                               std::to_string(geometry_.shard_rank) + "," +
                               std::to_string(geometry_.shard_world) + ")";
            return location;
        }
        if (!recall_codec_admitted(static_cast<RecallCodec>(record.codec))) {
            location.place   = RecallPagePlace::Refused;
            location.refusal = std::string("a record for page ") + std::to_string(page) +
                               " carries a codec no external tier may read: " +
                               recall_codec_refusal(static_cast<RecallCodec>(record.codec));
            return location;
        }
        if (record.file_slot < 0) {
            // A Spill with no region is malformed, and a malformed record is a REFUSAL -- the
            // writer never emits one, so this can only be corruption or a foreign format.
            location.place   = RecallPagePlace::Refused;
            location.refusal = "a Spill record for page " + std::to_string(page) +
                               " carries file_slot < 0, which the writer never emits";
            return location;
        }
        // THE GEOMETRY GUARD. Skipped only when the caller declared no stride.
        if (geometry_.page_bytes != 0 &&
            static_cast<std::uint64_t>(record.layer_bytes) != geometry_.page_bytes) {
            location.place   = RecallPagePlace::Refused;
            location.refusal = "a record for page " + std::to_string(page) + " names " +
                               std::to_string(record.layer_bytes) +
                               " B per page, this caller reads with " +
                               std::to_string(geometry_.page_bytes) + " B";
            return location;
        }
        location.place       = RecallPagePlace::Spill;
        location.file_slot   = record.file_slot;
        location.layer_bytes = record.layer_bytes;
        location.frontier    = record.frontier;
        location.sequence    = record.sequence;
        return location;
    }

    // The plan-facing form: which of these pages can this caller actually fetch, and which
    // are REFUSALS that must be counted rather than dropped. A refusal is never folded into
    // a miss -- sum_dir.h:364-368's "two states on purpose" rule, applied here.
    struct Census {
        std::uint32_t fetchable = 0;
        std::uint32_t gone      = 0;
        std::uint32_t unknown   = 0;
        std::uint32_t refused   = 0;
        std::vector<std::string> refusals;
    };

    [[nodiscard]] Census census(std::span<const std::uint32_t> pages) const {
        Census out;
        for (const std::uint32_t page : pages) {
            const RecallPageLocation located = locate(page);
            switch (located.place) {
            case RecallPagePlace::Spill: ++out.fetchable; break;
            case RecallPagePlace::Gone: ++out.gone; break;
            case RecallPagePlace::Unknown: ++out.unknown; break;
            case RecallPagePlace::Refused:
                ++out.refused;
                if (out.refusals.size() < 8U) { out.refusals.push_back(located.refusal); }
                break;
            }
        }
        return out;
    }

private:
    RecallLocatorGeometry geometry_{};
    std::unordered_map<std::uint32_t, RecallRecord> live_;
    std::unordered_set<std::uint32_t> released_;
};

// WHY A REASON CODE EXISTS (w-find4, the G9 fix): the two record kinds cannot say WHICH of
// the three different things happened to a page's bytes, and the third one is the one that
// makes a standing record dangerous:
//   * Spill                      the bytes are in the file and that region is this page's;
//   * Release "restored hot"     the device replica is back; the region was returned and the
//                                page can be spilled again LATER, into a DIFFERENT slot,
//                                with a fresh Spill record;
//   * Release "sequence ended"   the page is not coming back and its region is already in
//                                the process-global pool, so the next page to spill anywhere
//                                in this process can own it.
// `file_slot == -1` cannot separate those last two, and a Release ALWAYS carries -1
// (program_impl.h:11605), so a reader that must tell "moved, will come back" from "gone,
// the region is not trustworthy" has nothing to read today. The state machine has exactly
// two states on purpose (sum_dir.h:364-368 says so in as many words), so the answer is a
// FIELD and not a third kind: index() drops every kind that is neither Spill nor Release
// (:567), which would make a third kind a silent no-op -- and a state no lookup can see is
// worse than no state at all.
//
// THE FIELD IS `reserved` (offset 28). It sits INSIDE the CRC's [0, 48) coverage, so the
// reason is sealed and validated exactly like the rest of the header; it costs no byte, so
// the record stays one 64-byte unit and kRecallJournalVersion does not move; and a reader
// written before this change reads the same 0 it always did.
enum class RecallReleaseReason : std::uint8_t {
    Unspecified  = 0, // no reason recorded: what every record written before this change says
    RestoredHot  = 1, // the page went hot again; it may be spilled later into a new slot
    SequenceEnd  = 2, // the sequence owning the page went away (release_sequence_kv)
    SlotRecycled = 3, // the file slot was handed to another page
};

[[nodiscard]] inline RecallReleaseReason
recall_release_reason(const RecallRecord& record) noexcept {
    return static_cast<RecallReleaseReason>(record.reserved & 0xFFU);
}

// Only meaningful on a Release: a Spill's `reserved` word is not a reason.
inline void recall_set_release_reason(RecallRecord& record, RecallReleaseReason reason) noexcept {
    record.reserved = (record.reserved & 0xFFFFFF00U) | static_cast<std::uint32_t>(reason);
}

// Table-less CRC-32 (IEEE 802.3, reflected): 4 bytes per 64-byte record is free next to
// the read it protects, and it keeps this header dependency-free.
//
// THIS IS THE SPEC LAYER'S ONLY reflected-CRC-32 LOOP. It is exposed in its raw seeded form
// because spec/sum_dir.h seals its own header with the same polynomial but keeps the running
// register (it covers a prefix and then a trailing payload), and restating the loop there is
// exactly how "standard crc32 (0xEDB88320, reflected)" had already become two independently
// spelt copies. sum_dir_crc32() is now a call to recall_crc32_raw(); the two conventions are
// related by ONE provable identity, asserted in tests/test_sum_dir.cpp:
//     recall_crc32(d, n) == ~recall_crc32_raw(0xFFFFFFFFU, d, n)
[[nodiscard]] constexpr std::uint32_t recall_crc32_raw(std::uint32_t crc,
                                                       const std::uint8_t* data,
                                                       std::size_t bytes) noexcept {
    for (std::size_t i = 0; i < bytes; ++i) {
        crc ^= static_cast<std::uint32_t>(data[i]);
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = (crc & 1U) ? 0xEDB88320U : 0U;
            crc = (crc >> 1U) ^ mask;
        }
    }
    return crc;
}

// The sealed form used by recall_record_seal(): init all ones, answer inverted.
[[nodiscard]] constexpr std::uint32_t recall_crc32(const std::uint8_t* data,
                                                   std::size_t bytes) noexcept {
    return ~recall_crc32_raw(0xFFFFFFFFU, data, bytes);
}

// The published check value of CRC-32/ISO-HDLC over "123456789" is 0xCBF43926, and this
// primitive has to reproduce it or it is not the algorithm the format claims to read. Pinned
// at COMPILE time because the polynomial is inherited: this journal's record seals and
// spec/sum_dir.h's header seal are the same loop, so a change here moves both at once and the
// build is the only place that can say so.
inline constexpr std::uint8_t kRecallCrcCheckBytes[9] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
static_assert(recall_crc32(kRecallCrcCheckBytes, 9U) == 0xCBF43926U,
              "recall_crc32 is CRC-32/ISO-HDLC (check value over \"123456789\")");

inline void recall_record_seal(RecallRecord& record) noexcept {
    record.magic   = kRecallJournalMagic;
    record.version = kRecallJournalVersion;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&record);
    record.crc32 = recall_crc32(bytes, kRecallCrcBytes);
}

[[nodiscard]] inline bool recall_record_valid(const RecallRecord& record) noexcept {
    if (record.magic != kRecallJournalMagic) { return false; }
    if (record.version != kRecallJournalVersion) { return false; }
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&record);
    return recall_crc32(bytes, kRecallCrcBytes) == record.crc32;
}

// The map key: (digest, page). Deliberately not a hash of the whole record -- two
// records for the same page under different prefixes must be distinguishable, and the
// digest is the only field that carries the prefix.
[[nodiscard]] inline std::uint64_t recall_key(std::uint64_t digest_lo, std::uint64_t digest_hi,
                                              std::uint32_t page) noexcept {
    std::uint64_t h = 0x9E3779B97F4A7C15ULL;
    h ^= digest_lo + 0x9E3779B97F4A7C15ULL + (h << 6U) + (h >> 2U);
    h ^= digest_hi + 0x9E3779B97F4A7C15ULL + (h << 6U) + (h >> 2U);
    h ^= static_cast<std::uint64_t>(page) + 0x9E3779B97F4A7C15ULL + (h << 6U) + (h >> 2U);
    return h;
}

// fsync policy. The journal is a mirror, so the DEFAULT IS NO FSYNC: a lost trailing
// record costs a re-read (the page is still device-readable, or re-prefillable), never
// correctness. PerRound is the strongest policy the hot path may use -- one batched
// sync at a round boundary, never one per record. PerRecord exists for tests and for
// the cross-process case where the log must be durable before another process reads it.
enum class JournalFlushPolicy : std::uint8_t {
    None      = 0, // fwrite only (stdio buffering; the OS flushes)
    PerRound  = 1, // fflush + fsync once per round, after the round's records are in
    PerRecord = 2, // fflush + fsync after every append (tests / cross-process handoff)
};

[[nodiscard]] inline const char* journal_flush_policy_name(JournalFlushPolicy p) noexcept {
    switch (p) {
    case JournalFlushPolicy::None: return "none";
    case JournalFlushPolicy::PerRound: return "round";
    case JournalFlushPolicy::PerRecord: return "record";
    }
    return "?";
}

// The ENV spelling of the policy above, parsed in ONE place because the parsing rule is part
// of the handle's contract and not of any single call site:
//
//     absent / "" / "0" / anything non-numeric or <= 0  ->  None      (OFF)
//     "1"                                                 ->  PerRound
//     "2" or more                                        ->  PerRecord
//
// A PRESENCE test (`getenv(...) != nullptr`) is the failure this exists to make impossible:
// it reads the same for "0" as for "1", so the one value every sibling flag in this tree
// uses to mean OFF -- NINFER_TURN_RECALL=0, NINFER_FT_BW_GOV=0, NINFER_MTP_ADAPTIVE=0 --
// turned fsync ON here instead. The ladder is the enum's own numbering so the cross-process
// rung is reachable at all; before this, PerRecord had no spelling outside the tests.
[[nodiscard]] inline JournalFlushPolicy journal_flush_policy_from_env(const char* value) noexcept {
    if (value == nullptr) { return JournalFlushPolicy::None; }
    const long asked = std::strtol(value, nullptr, 10);
    if (asked <= 0) { return JournalFlushPolicy::None; }
    if (asked == 1) { return JournalFlushPolicy::PerRound; }
    return JournalFlushPolicy::PerRecord;
}

struct JournalOpenOptions {
    std::string path;
    bool truncate = false;      // a fresh session log; a resume keeps the old one
    JournalFlushPolicy flush = JournalFlushPolicy::None;
};

struct JournalLoadReport {
    std::uint64_t records    = 0; // well-formed records read
    std::uint64_t live       = 0; // Spill entries still standing after replay
    std::uint64_t tombstones = 0; // Release entries
    std::uint64_t torn_tail  = 0; // a short/crc-bad record at the end: crash mid-append
    std::uint64_t dropped_tail_records = 0; // records after the torn one (not trusted)
};

class TurnRecallJournal {
public:
    explicit TurnRecallJournal(JournalOpenOptions options) : options_(std::move(options)) {
        if (options_.path.empty()) {
            throw std::invalid_argument("turn recall journal needs a path");
        }
        const char* mode = options_.truncate ? "wb" : "ab";
        file_            = std::fopen(options_.path.c_str(), mode);
        // "ab" also creates; opening for read-back is a separate fopen in load(), so a
        // write-only failure is reported by good() rather than thrown: an unwritable
        // log must degrade to "no recall", not take the engine down.
        //
        // A journal opened WITHOUT truncate is a CONTINUATION -- that is what the flag
        // means one line up ("a resume keeps the old one") -- so the records already in
        // the file are still the answer to "is page p recallable" and their map is
        // replayed here, AT OPEN. Without this, the engine's own options (truncate =
        // false, program_impl.h:960) made every record an earlier process wrote
        // invisible to find_live() while its 64 bytes sat in the file: the same
        // "written, therefore findable" rule append() keeps on the hot path.
        //
        // A replay that had to stop (a torn tail, i.e. a crash mid-append) SAYS SO on
        // stderr instead of returning a quietly empty map -- a silent empty map is the
        // exact failure this header exists to make impossible.
        if (file_ != nullptr && !options_.truncate) {
            const JournalLoadReport replay = load();
            if (replay.torn_tail != 0) {
                std::fprintf(stderr,
                             "[recall] journal %s: replay stopped at a torn tail "
                             "(records=%llu live=%llu torn=%llu dropped=%llu); the bytes "
                             "after it are NOT trusted and their pages are not recallable\n",
                             options_.path.c_str(),
                             static_cast<unsigned long long>(replay.records),
                             static_cast<unsigned long long>(replay.live),
                             static_cast<unsigned long long>(replay.torn_tail),
                             static_cast<unsigned long long>(replay.dropped_tail_records));
            }
        }
    }

    ~TurnRecallJournal() {
        if (file_ != nullptr) {
            std::fflush(file_);
            std::fclose(file_);
            file_ = nullptr;
        }
    }

    TurnRecallJournal(const TurnRecallJournal&) = delete;
    TurnRecallJournal& operator=(const TurnRecallJournal&) = delete;

    [[nodiscard]] bool good() const noexcept { return file_ != nullptr; }
    [[nodiscard]] const std::string& path() const noexcept { return options_.path; }
    [[nodiscard]] JournalFlushPolicy flush_policy() const noexcept { return options_.flush; }
    void set_flush_policy(JournalFlushPolicy policy) noexcept { options_.flush = policy; }

    [[nodiscard]] std::uint64_t bytes_written() const noexcept { return bytes_written_; }
    [[nodiscard]] std::uint64_t records_written() const noexcept { return records_written_; }

    // Appends one record. Returns false (never throws) when the log is unusable, so the
    // engine's per-round hook stays noexcept-friendly.
    bool append(RecallRecord record) {
        if (file_ == nullptr) { return false; }
        recall_record_seal(record);
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(&record);
        if (std::fwrite(bytes, 1, kRecallRecordBytes, file_) != kRecallRecordBytes) {
            return false;
        }
        bytes_written_ += kRecallRecordBytes;
        ++records_written_;
        ++records_since_sync_;
        // A record that was WRITTEN must be FINDABLE by the process that wrote it.
        // live_ is the ONLY thing find_live() / live_count() / live_pages() / live_page()
        // consult, and before this line only load() ever filled it -- so an append() left
        // the map empty and the whole P6 hit criterion (program_impl.h:11700) plus the
        // built-in provider (program_impl.h:11838-11841) answered "nothing" for records
        // that were already on disk. index() is the rule load() applies, applied at the
        // moment the bytes are in stdio.
        index(record);
        if (options_.flush == JournalFlushPolicy::PerRecord) { sync(); }
        return true;
    }

    void flush() {
        if (file_ != nullptr) { std::fflush(file_); }
    }

    // The only fsync in this header. Called at a round boundary (PerRound) or by a test.
    void sync() {
        if (file_ == nullptr) { return; }
        std::fflush(file_);
#if defined(__unix__) || defined(__APPLE__)
        const int fd = ::fileno(file_);
        if (fd >= 0) { (void)::fsync(fd); }
#endif
        records_since_sync_ = 0;
    }

    [[nodiscard]] std::uint64_t records_since_sync() const noexcept { return records_since_sync_; }

    // Replays the whole log and rebuilds the live map. A torn tail ends the replay: the
    // bytes of a record that a crash cut in half are not trusted, and the page it
    // described is simply "not recallable" (the mirror rule).
    JournalLoadReport load() {
        JournalLoadReport report;
        live_.clear();
        std::FILE* in = std::fopen(options_.path.c_str(), "rb");
        if (in == nullptr) { return report; }

        RecallRecord record{};
        bool torn = false;
        for (;;) {
            const std::size_t got =
                std::fread(&record, 1, kRecallRecordBytes, in);
            if (got == 0) { break; } // clean EOF
            if (got != kRecallRecordBytes) {
                torn = true;
                ++report.torn_tail;
                break;
            }
            if (!recall_record_valid(record)) {
                torn = true;
                ++report.torn_tail;
                break;
            }
            ++report.records;
            if (torn) { ++report.dropped_tail_records; }
            if (static_cast<RecallKind>(record.kind) == RecallKind::Release) {
                ++report.tombstones;
            }
            // ONE rule, ONE spelling: index() is shared with the append() path, so a
            // replay can never disagree with the process that wrote the record about
            // what is live. It counts nothing and skips nothing itself -- a Release
            // erases the key it names (admitted codec or not), a Spill indexes only
            // under an admitted codec, and anything else is ignored. `report.live` is
            // recounted from the map below, and the map is the only thing lookups use.
            index(record);
        }
        std::fclose(in);
        // Tombs that removed entries must not be counted as live: recount from the map,
        // which is the only thing lookups consult.
        report.live = live_.size();
        return report;
    }

    [[nodiscard]] const RecallRecord* find_live(std::uint64_t digest_lo, std::uint64_t digest_hi,
                                                std::uint32_t page) const noexcept {
        const auto it = live_.find(recall_key(digest_lo, digest_hi, page));
        return it == live_.end() ? nullptr : &it->second;
    }

    [[nodiscard]] std::size_t live_count() const noexcept { return live_.size(); }

    // The page numbers with a live record, ascending: the input the P6 planner needs.
    [[nodiscard]] std::vector<std::uint32_t> live_pages() const {
        std::vector<std::uint32_t> pages;
        pages.reserve(live_.size());
        for (const auto& [key, record] : live_) {
            (void)key;
            pages.push_back(record.page);
        }
        std::sort(pages.begin(), pages.end());
        pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
        return pages;
    }

    [[nodiscard]] bool live_page(std::uint32_t page) const noexcept {
        for (const auto& [key, record] : live_) {
            (void)key;
            if (record.page == page) { return true; }
        }
        return false;
    }

    // The ONE spelling of the rule that maps a record onto live_, shared by BOTH writers
    // of that map: the hot path (append) and the replay (load). They must agree exactly:
    // tombstone FIRST (a Release erases whatever key it names, admitted codec or not),
    // then Spill, and only under an admitted codec -- an rk4v4/bf16/fp8/iso4e record is
    // counted and skipped there precisely so that no later lookup can treat it as a hit.
    // Having two copies of this rule is what let the tree write records that its own log
    // could not find.
    //
    // `Mixed` PASSES THIS FILTER since 2026-09-18, and it needed no edit here: the filter
    // delegates to `recall_codec_admitted`, so admitting Mixed in that ONE rule is what
    // stopped this line hiding it. That is the right shape and it is worth naming, because
    // the alternative (a second predicate here) is what would let the hot path and the
    // replay disagree again.
    //
    // WHAT THE FILTER IS NOT: it asks "may a later lookup treat this record as a hit",
    // i.e. FINDABILITY. It is not an offset, not a stride, and not a geometry check -- it
    // compares the record's codec to nothing, because the journal has no reader geometry
    // to compare against. The one field that exists for that comparison, `layer_bytes`, is
    // documented as a per-layer stride and WRITTEN as the per-page sum
    // (program_impl.h:12926 -> turn_recall_page_bytes()), so wiring it here as-is would
    // install a WRONG-stride check; this line therefore reads it zero times. The
    // cross-run hazard that guard was meant to close is stated in MIXEDPROOF/REPORT.md
    // sec.5(ii) and is NOT closed by admitting Mixed.
    void index(const RecallRecord& record) {
        const std::uint64_t key =
            recall_key(record.digest_lo, record.digest_hi, record.page);
        if (static_cast<RecallKind>(record.kind) == RecallKind::Release) {
            live_.erase(key);
            return;
        }
        if (static_cast<RecallKind>(record.kind) != RecallKind::Spill) { return; }
        if (!recall_codec_admitted(static_cast<RecallCodec>(record.codec))) { return; }
        live_.insert_or_assign(key, record);
    }

private:
    JournalOpenOptions options_;
    std::FILE* file_                 = nullptr;
    std::uint64_t bytes_written_     = 0;
    std::uint64_t records_written_   = 0;
    std::uint64_t records_since_sync_ = 0;
    std::unordered_map<std::uint64_t, RecallRecord> live_;
};

// ---------------------------------------------------------------------------
// P6: page selection
// ---------------------------------------------------------------------------
//
// THE TOKEN-SPACE HALF IS NOT HERE. "Which span of the conversation do I want back?"
// is answered by the retrieval layer and injected, because that is a different key
// space with its own measurements: the in-tree suffix_lookup is tail-anchored and
// measures R@1 0.73-1.00 when the quote ends at the newest token but 0.00-0.03 when it
// is followed by a follow-up (medial) or is reworded -- the medial shape needs an
// any-position n-gram index and the reworded one needs BM25 (TODO.md:5890-5893). This
// header must not grow a second token index, and it must not re-derive what
// ResidentPrefixIdentity already decides.
struct RecallRequest {
    std::uint32_t token_begin = 0; // absolute, in the sequence's own token space
    std::uint32_t token_end = 0;   // exclusive

    // =======================================================================================
    // [F-1019] THE NOMINATION CHANNEL -- HOW N CONCURRENT AGENTS SHARE ONE LOOKUP.
    // =======================================================================================
    // WHAT WAS WRONG WITH THIS TYPE, IN THE RECORD'S OWN WORDS. `spec/semchan_stage2.h:25-31`
    // states the structural fact and the whole shape of the second stage follows from it:
    // "`RecallRequest` IS NOT A SET ... Two integers. ONE CONTIGUOUS SPAN. No container, no page
    // list, no cap -- so there is no field a nominated page could enter and no place a union
    // could be written." The consequence is at `:46-51`: "a nominated page OUTSIDE the arbiter's
    // run cannot be added to the recall. It can only be ENCLOSED -- by growing the run upward
    // until it contains the nomination, which drags in EVERY page in between. **The union route
    // through `RecallRequest` is a route that COSTS KV, not one that saves it.**"
    //
    // WHY A PAGE LIST, AND NOT A SPAN LIST AND NOT A PER-AGENT BINDING. Three candidates were
    // on the table; this field pair is two of them fused, and the reason is the granularity this
    // file already fixes at `:1246`: "one logical page = kRecallPageTokens (64) tokens ACROSS ALL
    // TEXT LAYERS ... the block table's addressing unit is the page, and a cold-slot sentinel
    // encodes ONE slot base that the attention kernels index in every layer's cold_slots at once,
    // so a page can only be moved whole".
    //   * A SPAN LIST cannot be expressed in the delivery unit. A union of N agents' spans is a
    //     set of spans whose members are PAGES; re-deriving pages from spans re-introduces the
    //     rounding at `program_impl.h:14321-14325` (which rounds the end OUTWARD to a whole page)
    //     and therefore re-introduces enclosure for every span whose end is mid-page.
    //   * A PER-AGENT BINDING ALONE DOES NOT SHARE ANYTHING. It changes who owns a page, not
    //     which pages are looked up: N agents' spans still become N contiguous runs and N calls
    //     into `plan_recall_pages`, i.e. N lookups. The owner's order is 「lookup共用」 -- the
    //     LOOKUP must be shared -- and bookkeeping is not what is shared.
    //   * A PAGE LIST is the minimal shape that can carry the union WITHOUT enclosure, and N
    //     agents' nominations CONCATENATE into ONE list, so ONE `RecallRequest` reaches ONE
    //     `plan_recall_pages` call and ONE pass over the journal index. That is the sharing.
    //
    // AND THE PARALLEL OWNER VECTOR IS THE OTHER HALF, NOT A CONVENIENCE. One shared lookup must
    // still answer N agents SEPARATELY, and after the plan is built the only surviving fact about
    // a nominated page is its POSITION in this list -- a page number is not an agent. So
    // `nominated_owner` is positionally parallel to `nominated_pages` and each entry is the
    // caller's own agent index for the page at the same position. Without it the sharing is
    // lossy: N agents would receive the union and each would have to re-derive its own share from
    // page geometry, which is the enclosure problem again in the consumer.
    //
    // THE INVARIANTS, ALL CHECKED BY THE PLANNER, NONE OF THEM ASSUMED HERE:
    //   * `nominated_pages` is ASCENDING and DUPLICATE-FREE, in PAGE units, exactly as
    //     `RecallPagePlan::pages` is (`:1297`). `nominated_owner.size() == nominated_pages.size()`.
    //   * EMPTY IS THE DEFAULT AND THE PRE-IMAGE BEHAVIOUR. With both vectors empty this struct
    //     is byte-for-byte the two-integer struct above and every consumer takes the same path it
    //     took before this field existed. That is what makes this change revertible and testable.
    //   * A NOMINATION NEVER MOVES `token_begin` OR `token_end`. The arbiter's run is the run;
    //     nominations are an ADDITION to it, which is the property `semchan_stage2.h:160-163`
    //     states for its own extension ("`token_begin` NEVER MOVES UP") and the reason a refusal
    //     here cannot corrupt the anchor at the low end of the run.
    std::vector<std::uint32_t> nominated_pages; // ascending, page units, dedup'd; empty = none
    std::vector<std::uint32_t> nominated_owner; // PARALLEL: the agent index for each page above

    // True iff this request carries nominations. A named predicate so no caller writes
    // `!nominated_pages.empty()` in its own spelling and drifts from the invariant.
    [[nodiscard]] bool has_nominations() const noexcept { return !nominated_pages.empty(); }
};

// `frontier` is the sequence's committed-token count, so a provider can bound its
// answer to what exists. Returning an empty span (token_end <= token_begin) means
// "nothing to recall this round".
using RecallRequestProvider = std::function<RecallRequest(std::uint32_t frontier)>;

//
// GRANULARITY: one logical page = kRecallPageTokens (64) tokens ACROSS ALL TEXT LAYERS.
// Never per layer, never per head, never per token:
//   * the block table's addressing unit is the page, and a cold-slot sentinel encodes
//     ONE slot base that the attention kernels index in every layer's cold_slots at
//     once (cold_host_tier.h:34-41), so a page can only be moved whole;
//   * per-layer selection would fragment one logical page across `layers` files with
//     independent lifetimes and cannot be made atomic.
//
// ORDER: ascending page number. The on-disk offset of page p is monotone in p
// (`file_slot * stride`, with file slots handed out in allocation order), so ascending
// order is the order that lets consecutive pages merge into one batched read per layer
// file. Descending (recency first) would defeat that, and the newest pages are exactly
// the ones that are still device-resident anyway.
//
// BUDGET: truncation drops from the LOW end (the OLDEST pages). That keeps the selected
// set a *contiguous, gap-free* run, which is the only shape `warm_cold_prefix` can
// restore today (restore_cold_page is per page, but a hole would leave a log whose
// middle is missing while its tail is present, and every page that IS loaded must have
// a live record).
//
// HIT/MISS: `has_live_page` is the whole hit criterion. A page is a HIT iff a
// well-formed Spill record with an admitted codec and the CURRENT digest still stands
// for it (a later Release tombstone erases it). A page that misses CUTS the run: the
// pages below it are counted as dropped_hole and never filled with a neighbour's bytes.
// The caller supplies the predicate (the engine binds it to the journal); it is a
// std::function so this header stays host-only and testable.

struct RecallPagePlanRequest {
    std::uint32_t frontier = 0;          // committed tokens now
    std::uint32_t page_tokens = kRecallPageTokens;
    std::uint32_t wanted_begin_page = 0; // from the retrieval layer (token space)
    std::uint32_t wanted_end_page = 0;   // exclusive; 0 => pages_for(frontier)
    std::uint64_t page_bytes = 0;        // sum of layer strides for one page
    std::uint64_t byte_budget = 0;       // 0 => unbounded (still bounded by the run)

    // ---- [PREFILLBUDGET] THE SAME ROUND, IN THE DIMENSION THE COST IS ACTUALLY IN -----------
    // `byte_budget` above is 256 MiB because a page is 1,181,696 B of DEVICE MEMORY -- and the
    // tree's own comment on that default (program_impl.h, `NINFER_TURN_RECALL_BYTES`) says it "is
    // a MEMORY-SAFETY limit rather than a speed limit". Under the inject shape the marginal cost
    // of a recall is not bytes, it is a RE-PREFILL, priced PER TOKEN, so the round has had no
    // speed bound at all. This field is that bound, and it is a SEPARATE field rather than a
    // reinterpretation of `byte_budget`: the two are different dimensions, one of them is a
    // memory figure inherited from the KV mirror's page width, and collapsing them would make the
    // 227-page ceiling (227 x 64 = 14,528 tokens) look like a policy when it is arithmetic.
    //
    // 0 == UNBOUNDED == the flag and the environment variable were both absent. That is the
    // default, and it makes every existing run bit-identical: a zero here cannot refuse anything.
    std::uint64_t token_budget = 0;      // 0 => unbounded (neither the flag nor the env said so)

    // ---- [F-1019] THE NOMINATION CHANNEL, CARRIED INTO THE DECISION --------------------------
    // Plumbed straight through from `RecallRequest` by the ONE consumer (`program_impl.h`, the
    // `plan_request` builder) -- the planner is handed the union and never re-derives it. Both
    // vectors default empty, and the empty case is the pre-image plan byte-for-byte.
    std::vector<std::uint32_t> nominated_pages; // ascending, page units; empty = the pre-image
    std::vector<std::uint32_t> nominated_owner; // PARALLEL to the above
};

struct RecallPagePlan {
    std::vector<std::uint32_t> pages;   // ascending
    std::uint32_t dropped_budget  = 0;  // oldest pages dropped to fit byte_budget
    // A TOTAL, and it totals two different losses. Exactly ONE page of a run has no live
    // record -- the page the scan stopped at; every page ABOVE it has a live record and is
    // dropped only by the contiguity rule below ("a partial run is still a valid (shorter)
    // contiguous run"). `hole=N` therefore reads as "N pages are gone" when N-1 of them are
    // recallable. The measured instance (RECALLFIX, D1 at 1M, pinned eedfb8aad55f8d7e):
    // wanted run [0,8) with page 1 the only page lacking a live record printed `hole=7`,
    // i.e. 1 absent + 6 cut -- and the same event was reported by the sequence-end line as
    // `pages=5/5 refused_cost=0 digest_unavailable=0`, whose reader took it for complete.
    std::uint32_t dropped_hole    = 0;  // pages cut off above a hole (total)
    std::uint32_t dropped_absent  = 0;  // OF dropped_hole: how many have no live record (0/1)
    std::uint32_t dropped_clamped = 0;  // asked beyond the committed frontier
    std::uint64_t bytes           = 0;  // pages.size() * page_bytes

    // ---- [INEXACTGATE D] THE WANTED RUN, SO `pages=R/P` CAN BE READ FOR WHAT IT IS ----
    // `count()` is the CUT count and the sequence-end line's denominator is Σ count(), so the
    // more severely a run was cut the MORE COMPLETE that line reads (measured: D1 printed
    // `pages=5/5` on 3 of 5 rounds carrying `hole=7 INEXACT`; C5 printed `pages=6/20` with zero
    // INEXACT -- dl/holegate/REPORT.md sec.3.1). The three fields below are the denominator a
    // reader believes `pages=R/P` already has. They are RECORDED, never inferred: a later
    // reader cannot recover them from `count()` plus the drop counters once a plan is refused
    // with no plan object at all (which is finding 2's case).
    std::uint32_t wanted_begin_page = 0;  // as the REQUEST asked it, before any clamp
    std::uint32_t wanted_end_page   = 0;  // exclusive; the request's end, or `committed` if 0
    // [begin, end) of the OLDEST pages the byte budget dropped, so "the ANCHOR is the first
    // thing dropped" (`lo += drop`, :1263) is visible on the line without the caller's anchor.
    std::uint32_t dropped_budget_begin_page = 0;
    std::uint32_t dropped_budget_end_page   = 0;

    // ---- [F-1019] THE NOMINATIONS, AND EVERY WAY ONE OF THEM CAN FAIL TO ARRIVE -------------
    // Same discipline as the drop counters above: RECORDED, never inferred, and a total that
    // names which of two different losses it totals. `nominated_kept` is the pages actually
    // appended to `pages`; `nominated_missing` is the ones the journal could not serve (the ONLY
    // reason a nomination is refused, because the criterion is `has_live_page` and nothing else);
    // `nominated_gap` is a nomination the CALLER mis-built -- below the byte-budget edge, out of
    // ascending order, or with no owner entry -- which is a programming error and is reported as
    // such rather than silently repaired.
    std::uint32_t nominated_kept    = 0;
    std::uint32_t nominated_missing = 0;
    std::uint32_t nominated_gap     = 0;
    // The per-agent view of the SHARED lookup: for every agent index that nominated at least one
    // page, the OWNER label and the pages of its nomination the plan actually holds, ascending.
    // This is the whole delivery of "N agents, one lookup": the lookup was done once, and this is
    // how the ONE result is handed back per agent without a second pass over the index.
    std::vector<std::uint32_t> nominated_page_agents;      // distinct agent indices, ascending
    std::vector<std::uint32_t> nominated_pages_kept;       // PARALLEL to nominated_pages above

    // ---- [PREFILLBUDGET] THE TOKEN BUDGET'S OWN REFUSAL, RECORDED, NEVER INFERRED ------------
    // dl/vectorkey/REPORT.md:284-288 states the hazard in its own words: "a plan may legitimately
    // name more blocks than the token budget allows -- and then something must choose" -- and the
    // only admissible choice is the named refusal, because "if it truncates the run, the injected
    // text is a PREFIX of the answer's context, which is a PARTIAL answer ... at best and a
    // confident wrong answer at worst". So this is a BOOL plus the two numbers, not a clamp: the
    // plan that would have exceeded the budget is returned EMPTY and these three fields are its
    // whole refusal. `dropped_budget`/`dropped_hole`/`dropped_clamped` are deliberately NOT
    // touched -- they name the BYTE budget's loss and the frontier's shortfall, and reusing one of
    // them here would put two dimensions in one field pair, which is the defect class this file's
    // [INEXACTGATE] sections were written to stop.
    bool          refused_prefill_budget = false;
    // The run that would have been injected, in TOKENS, and the budget that refused it. Both are
    // recorded so the refusal is ARITHMETIC on the line (`wanted_tokens=` / `budget_tokens=`)
    // rather than a bare word a reader has to trust.
    std::uint64_t prefill_tokens_wanted   = 0;
    std::uint64_t prefill_token_budget    = 0;

    [[nodiscard]] bool empty() const noexcept { return pages.empty(); }
    // The half of `dropped_hole` that is NOT the absent page: pages whose bytes ARE in the
    // journal and are thrown away by the contiguity rule. Derived, never stored, so it cannot
    // drift from the two fields it is the difference of.
    [[nodiscard]] std::uint32_t cut_above_hole() const noexcept {
        return dropped_hole >= dropped_absent ? dropped_hole - dropped_absent : 0U;
    }
    [[nodiscard]] bool exact() const noexcept {
        // [PREFILLBUDGET] `refused_prefill_budget` is part of the definition, and it has to be:
        // a plan refused by the TOKEN budget holds NO page while all three drop counters can
        // legitimately be zero (nothing was dropped -- the whole run was refused before anything
        // was cut), so without this term `exact()` would be TRUE on a plan that holds nothing.
        // That inversion is the exact class of bug the [INEXACTGATE D] block above exists to
        // remove, so it is refused here rather than left for a reader to notice.
        return dropped_budget == 0 && dropped_hole == 0 && dropped_clamped == 0 &&
               !refused_prefill_budget;
    }
    [[nodiscard]] std::uint32_t count() const noexcept {
        return static_cast<std::uint32_t>(pages.size());
    }

    // ---- [INEXACTGATE A] WHY THE PLAN WAS CUT, NAMED --------------------------------
    // `exact()` is ONE bool over THREE counters, and the three counters are three different
    // defects with three different consequences for the text that executes:
    //
    //   BudgetTruncated  the OLDEST pages were dropped to fit `byte_budget`
    //                    (`lo += drop`, :1263). The run's LOW END -- and the selector's anchor
    //                    sits at the low end (`sum_dir_reach.h`'s first_page/anchor_page) -- is
    //                    therefore NOT in the plan, so sum_dir_reach.h:762-765's claim that
    //                    "the passage's text contains query[covered_begin, covered_end)
    //                    verbatim" is FALSE for the run that actually executes (measured:
    //                    dl/indexfan, `anchor in plan = NO`; dl/holegate sec.5.4).
    //   HoleCut          the run is cut at the first page with no live record (:1243-1253).
    //                    `dropped_absent` of the dropped pages are GONE; `cut_above_hole()` of
    //                    them are LIVE and recallable and were discarded by the contiguity rule.
    //                    The executed text is a PREFIX shorter than the wanted run.
    //   FrontierClamped  the request asked past the committed frontier. NOTHING THAT EXISTS WAS
    //                    DROPPED: the plan holds the whole run the frontier can serve. This is a
    //                    different axis from the other two and must not be read as a loss --
    //                    naming it here is what stops that conflation.
    //
    // The order is fixed ONLY so that `defect()` is a function of the plan; a plan may carry
    // more than one, and the reader who needs all three reads the counters, which are untouched.
    // `dropped_absent` is deliberately NOT a case: it is never non-zero unless `dropped_hole` is
    // (there is always an absent page when a hole cuts the run), and a gate on
    // `cut_above_hole() > 0` would MISS the pure-tail case where the top page is the only one
    // missing and nothing recallable was thrown away -- the most severe hole of all for the
    // reader, and the one a naive "was anything recoverable lost?" test would call clean.
    enum class Defect : std::uint8_t {
        None = 0,
        // [PREFILLBUDGET] The TOKEN budget refused the whole round. It is checked FIRST because it
        // fires FIRST: the refusal happens before the byte budget is consulted and before any page
        // is kept, so a plan cannot carry it together with the byte budget's own defect except in
        // the case where both would refuse -- and then this one is the one that named the cause.
        PrefillBudget,
        BudgetTruncated,
        HoleCut,
        FrontierClamped,
        // [F-1019] A NOMINATION THE JOURNAL COULD NOT SERVE. Its own member, and NOT folded into
        // HoleCut: a hole is a fact about the ARBITER'S RUN (its bytes are not this prefix's bytes
        // any more, so the run must end there), while a missing nomination is a fact about a page
        // the CALLER named out of a DIFFERENT lookup -- a shared one. Folding them would put two
        // dimensions in one member, which is the defect class the [INEXACTGATE] block above exists
        // to stop. It is checked LAST: the run's own defects are about the run, and the run is what
        // the anchor and the selector's `covered_*` claim depend on.
        NominationMissing,
    };
    [[nodiscard]] Defect defect() const noexcept {
        if (refused_prefill_budget) { return Defect::PrefillBudget; }
        if (dropped_budget != 0) { return Defect::BudgetTruncated; }
        if (dropped_hole != 0) { return Defect::HoleCut; }
        if (dropped_clamped != 0) { return Defect::FrontierClamped; }
        if (nominated_missing != 0 || nominated_gap != 0) { return Defect::NominationMissing; }
        return Defect::None;
    }
    [[nodiscard]] bool inexact() const noexcept { return !exact(); }
    // The run the retrieval layer asked for, in pages, as it asked for it.
    [[nodiscard]] std::uint32_t wanted_pages() const noexcept {
        return wanted_end_page > wanted_begin_page ? wanted_end_page - wanted_begin_page : 0U;
    }
    // ... of which the committed frontier could serve this many (== wanted - dropped_clamped).
    [[nodiscard]] std::uint32_t admitted_pages() const noexcept {
        const std::uint32_t want = wanted_pages();
        return want >= dropped_clamped ? want - dropped_clamped : 0U;
    }
    // ⭐ THE IDENTITY that makes the shortfall auditable: the pages the plan did NOT hold out of
    // the ones it could have held. `count() == admitted_pages() - cut_pages()` is an INVARIANT of
    // `plan_recall_pages` (both cuts are subtractions from the same [lo, hi)), and the sibling
    // test asserts it over a sweep, so a future edit that loses a dropped page fails a check
    // instead of only lowering a ratio.
    [[nodiscard]] std::uint32_t cut_pages() const noexcept {
        const std::uint32_t admitted = admitted_pages();
        return admitted > count() ? admitted - count() : 0U;
    }
};

// The NAME of a defect, so a refusal, a test and a report cannot spell it three ways.
[[nodiscard]] inline const char* recall_plan_defect_name(RecallPagePlan::Defect defect) noexcept {
    switch (defect) {
        case RecallPagePlan::Defect::None: return "none";
        case RecallPagePlan::Defect::PrefillBudget: return "prefill-budget";
        case RecallPagePlan::Defect::BudgetTruncated: return "budget-truncated";
        case RecallPagePlan::Defect::HoleCut: return "hole-cut";
        case RecallPagePlan::Defect::FrontierClamped: return "frontier-clamped";
        case RecallPagePlan::Defect::NominationMissing: return "nomination-missing";
    }
    return "unknown";
}

// ---- [PREFILLBUDGET] THE NAME A RUN IS JUDGED BY, IN THE SPELLING THE SPECIFICATION USES ----
// dl/vectorkey/REPORT.md:44 and :282 name the edge `refused-prefill-budget` in lowercase hyphens
// because that is the fleet's checker vocabulary (`scripts/check_gold.py`'s verdict table, cited
// at dl/vectorkey/REPORT.md:313-325). The tree's own printed header keeps its UPPERCASE shape
// (`[recall] REFUSED-<defect>`) and this is the SECOND, lowercase token appended beside it, so a
// harness can grep ONE string and the two refusal vocabularies of this path (sum_dir_reach.h's
// `refused-budget` status, and this plan's defects) can never be confused for each other.
//
// The byte budget's defect gets its OWN lowercase name on purpose: it must NOT be spelled
// `refused-budget`, because that is `SumDirReachStatus::RefusedBudget`'s name at
// sum_dir_reach.h:269 and one string for two different refusals is precisely the ambiguity
// dl/vectorkey/REPORT.md:281-282 refuses.
[[nodiscard]] inline const char* recall_plan_refusal_name(RecallPagePlan::Defect defect) noexcept {
    switch (defect) {
        case RecallPagePlan::Defect::None: return "none";
        case RecallPagePlan::Defect::PrefillBudget: return "refused-prefill-budget";
        case RecallPagePlan::Defect::BudgetTruncated: return "refused-byte-budget";
        case RecallPagePlan::Defect::HoleCut: return "refused-hole-cut";
        case RecallPagePlan::Defect::FrontierClamped: return "refused-frontier-clamp";
        case RecallPagePlan::Defect::NominationMissing: return "refused-nomination";
    }
    return "unknown";
}

[[nodiscard]] inline std::uint32_t recall_pages_for_frontier(std::uint32_t frontier,
                                                             std::uint32_t page_tokens) noexcept {
    if (page_tokens == 0) { return 0; }
    return frontier / page_tokens;
}

// ---------------------------------------------------------------------------
// [INEXACTGATE E] THE BUDGET EDGE'S RULE -- the ONE place the tree's two opposite
// policies meet, and the reason a NEW budget (dl/vectorkey ROW 2) must not copy this site.
// ---------------------------------------------------------------------------
//
// The tree holds BOTH edge policies for the same named concept ("the budget cannot hold this
// run"), wired IN SERIES:
//
//   sum_dir_recall_span_reachable   REFUSES   `sum_dir_reach.h:773-793`: "---- THE REFUSAL,
//                                   NAMED. A candidate that cannot be held is NOT clamped. ---"
//                                   setting `result.pages = 0`, enumerated as
//                                   `RefusedBudget = 3, // the strongest class cannot be held;
//                                   refusing, not truncating` (:257).
//   plan_recall_pages (this file)   TRUNCATES `lo += static_cast<std::uint32_t>(drop);` (:1263),
//                                   whose stated reason is contiguity (`:1165`: "BUDGET:
//                                   truncation drops from the LOW end (the OLDEST pages)").
//
// The layer the tree's own rule governs conforms; the layer that decides what text reaches the
// model violates it -- and the anchor the selector rank-1'd sits at the LOW end, so the first
// thing the truncation drops is the anchor. dl/vectorkey ROW 2 proposes a NEW token-denominated
// prefill budget and states its edge as "`refused-prefill-budget`, refusing not truncating --
// never a partial injection" (dl/vectorkey/REPORT.md:44): that is the sum_dir_reach rule, and
// ROW 2 must take it from THERE, never from `lo += drop` here (dl/holegate sec.5.5).
//
//
// ⭐ THE POLICY IS NAMED, AND BY THE OWNER. `Unnamed` was the correct INTERMEDIATE, never a
// destination: it existed so that this line could not install a policy in either direction, and the
// choice has now been made on the record, in the owner's own words, three ways over:
//
//   1. the owner's standing ruling on this budget family -- 「prefill 必须有上限」 and at the edge
//      「点名拒绝而不是截断」 (a NAMED REFUSAL, not a truncation);
//   2. `dl/vectorkey` ROW 2, which is the landing of that ruling: `refused-prefill-budget`,
//      "never a partial injection" (dl/vectorkey/REPORT.md:44);
//   3. ⭐ `dl/holegate`'s ruling on ROW 2, which this line is a successor to: its edge rule must be
//      taken from `sum_dir_reach.h:773`, NEVER from `lo += drop` (this file, :1263).
//
// and the fleet's absolute rule that all three express: **a silent wrong answer is worse than a
// refusal.**
//
// The arithmetic inside `plan_recall_pages_with_policy` is NOT touched by this: it still computes
// `drop`, still records `dropped_budget`, and now returns the plan EMPTY with the dropped range
// named, exactly as `sum_dir_reach.h:775` sets `result.pages = 0`. The other two rules stay
// expressible (and are exercised by the sibling test) because a future owner may need them --
// `TruncateOldest` is what this site did, `TruncateNewestKeepAnchor` is the only variant that
// keeps `sum_dir_reach.h:762-765`'s claim true, and BOTH are truncations that the tree's rule
// refuses.
enum class BudgetEdgePolicy : std::uint8_t {
    Unnamed = 0,               // not a policy: the ABSENCE of one. Engine-side static_assert.
    RefuseNotTruncate,         // ⭐ NAMED. sum_dir_reach.h:257/:773's rule -- "refusing, not
                               // truncating" -- which dl/vectorkey ROW 2 must inherit
    TruncateOldest,            // what this site did: drops the anchor. Refused by the tree's rule
    TruncateNewestKeepAnchor,  // keeps the anchor (and sum_dir_reach.h:762-765's claim);
                               // still a truncation, so sum_dir_reach.h itself would refuse it
};
inline constexpr BudgetEdgePolicy kRecallBudgetEdgePolicy = BudgetEdgePolicy::RefuseNotTruncate;
[[nodiscard]] inline constexpr const char* recall_budget_edge_policy_name(
    BudgetEdgePolicy policy) noexcept {
    switch (policy) {
        case BudgetEdgePolicy::Unnamed: return "unnamed";
        case BudgetEdgePolicy::RefuseNotTruncate: return "refuse-not-truncate";
        case BudgetEdgePolicy::TruncateOldest: return "truncate-oldest";
        case BudgetEdgePolicy::TruncateNewestKeepAnchor: return "truncate-newest-keep-anchor";
    }
    return "unknown";
}

[[nodiscard]] inline RecallPagePlan
plan_recall_pages_with_policy(const RecallPagePlanRequest& request,
                              const std::function<bool(std::uint32_t)>& has_live_page,
                              BudgetEdgePolicy edge_policy) {
    RecallPagePlan plan;
    const std::uint32_t page_tokens =
        request.page_tokens == 0 ? kRecallPageTokens : request.page_tokens;
    const std::uint32_t committed = recall_pages_for_frontier(request.frontier, page_tokens);
    std::uint32_t hi = request.wanted_end_page == 0 ? committed : request.wanted_end_page;
    std::uint32_t lo = request.wanted_begin_page;
    // [INEXACTGATE D] The run AS ASKED, recorded before the clamp moves `hi`. This is the
    // denominator the sequence-end line's `pages=R/P` has never had.
    plan.wanted_begin_page = lo;
    plan.wanted_end_page   = hi;
    if (hi > committed) {
        // [INEXACTGATE D / line `redkv`] `dropped_clamped` is "the part of the ASK the frontier
        // cannot serve" (`program.h:1078`), i.e. the OVERLAP of [lo, hi) with [committed, +inf) --
        // NOT `hi - committed`, which counts the pages BELOW the ask's own start as well. The
        // difference is measurable and it is not cosmetic: `program.h:1081` states
        // `pages_admitted + pages_clamped == SUM wanted_pages()`, and the sibling test's sweep
        // (22400 cases: 4 frontiers x 5 begins x 5 ends x 4 budgets x 14 holes x 4 policies) shows
        // 6720 of them violating exactly that, ALL of them with `lo > committed` -- e.g. frontier=0
        // asked for [1,3) reported `wanted=2, admitted=0, clamped=3`.
        const std::uint32_t serve_from = lo > committed ? lo : committed;
        plan.dropped_clamped = hi > serve_from ? hi - serve_from : 0U;
        hi                   = committed;
    }
    if (lo > hi) { lo = hi; }
    if (lo == hi) { return plan; }

    // Cut at the first hole. Nothing below the hole is recalled: a hole means those
    // bytes are not this prefix's bytes any more, and a partial run is still a valid
    // (shorter) contiguous run, whereas filling the hole would be an approximation.
    for (std::uint32_t page = lo; page < hi; ++page) {
        if (!has_live_page(page)) {
            plan.dropped_hole += (hi - page);
            // One of the pages just dropped is the page that is GONE; the rest are live and
            // cut only because the run must stay gap-free. Recording which is which is what
            // makes `hole=N` auditable from the line alone.
            plan.dropped_absent += 1U;
            hi = page;
            break;
        }
    }
    if (lo >= hi) { return plan; }

    // ---- [PREFILLBUDGET] THE TOKEN BUDGET. THE EDGE IS A REFUSAL. ---------------------------
    // WHERE THIS SITS, AND WHY EXACTLY HERE. It is AFTER the frontier clamp and AFTER the hole
    // cut, so the run it measures is the run the plan COULD HOLD -- not the raw ask. That
    // placement is the whole distinction the [INEXACTGATE D] block above fights for:
    //   * a request past the committed frontier drops `dropped_clamped` pages that never existed,
    //     and `TurnRecallCounters`' comment states outright that this "must not be read as a loss";
    //   * a hole cuts a run because those bytes are not this prefix's bytes any more.
    // Neither of those is the budget's business, so neither may trip this refusal: firing on the
    // RAW ask would turn a legitimate clamp into a refusal and REGRESS the tree.
    //
    // AND IT IS BEFORE THE BYTE BUDGET, deliberately: `lo += drop` below is the truncation the
    // tree's own rule refuses (sum_dir_reach.h:257/:773, "refusing, not truncating"), and the
    // anchor the selector rank-1'd sits at the LOW end of the run, so it is the FIRST thing a
    // truncation drops. When the token budget refuses, NO page is kept and NO token is
    // re-prefilled -- which is the requirement stated at dl/vectorkey/REPORT.md:287-288: "The
    // refusal must fire at PLAN time, on the plan's own `needed` count, before a single token is
    // re-prefilled." This function is a DECISION and does no I/O, so "before a single token" is
    // structural here, not a promise.
    //
    // NOT A CLAMP. `min(want, budget/page_tokens)` does not appear below and must not: the tree's
    // rule for its own budget is "A REFUSAL threshold here, never a clamp" (sum_dir_reach.h:249-251)
    // and dl/vectorkey/REPORT.md:290-291 says the prefill budget "inherits it verbatim".
    {
        // The run is MEASURED in tokens whether or not a budget exists: a reader can then always
        // see how big the round it is reading actually was, and `prefill_tokens_wanted` is a
        // function of the plan rather than of the knob, so it cannot drift from `count()`.
        // [F-1019] THE DENOMINATOR IS THE WHOLE PLAN, NOT ONLY THE RUN. This figure is the token
        // budget's own measure of what the round would re-prefill, and a nomination OUTSIDE the run
        // is restored by the very same executor (`recall_cold_pages_for_round` walks `plan.pages`
        // page by page), so leaving it out would under-count the round and pass a budget the round
        // exceeds -- the exact inversion [PREFILLBUDGET] above exists to remove. Nominations INSIDE
        // the run are already in `hi - lo` and are not counted twice.
        const std::uint64_t run_pages_now = static_cast<std::uint64_t>(hi - lo);
        std::uint64_t       want_pages_now = run_pages_now;
        for (const std::uint32_t page : request.nominated_pages) {
            if (page < lo || page >= hi) { ++want_pages_now; }
        }
        plan.prefill_tokens_wanted = want_pages_now * static_cast<std::uint64_t>(page_tokens);
        plan.prefill_token_budget  = request.token_budget;
        if (request.token_budget != 0 && plan.prefill_tokens_wanted > request.token_budget) {
            plan.refused_prefill_budget = true;
            plan.pages.clear();
            plan.bytes = 0;
            return plan;
        }
    }

    // Budget: keep the NEWEST pages (drop the oldest), so the result stays gap-free.
    if (request.byte_budget != 0 && request.page_bytes != 0) {
        const std::uint64_t max_pages = request.byte_budget / request.page_bytes;
        const std::uint64_t want      = static_cast<std::uint64_t>(hi - lo);
        if (want > max_pages) {
            const std::uint64_t drop = want - max_pages;
            plan.dropped_budget      = static_cast<std::uint32_t>(drop);
            // [INEXACTGATE E] The dropped range, named. `lo += drop` is the whole hazard: with
            // the anchor at the LOW end of the run (sum_dir_reach.h's first_page/anchor_page),
            // the anchor is the first thing dropped, so the selector's `covered_*` claim is
            // false for the run that executes. Recording the range makes that auditable from
            // the line alone, against the `first_page=` the selector printed one stage earlier.
            plan.dropped_budget_begin_page = lo;
            plan.dropped_budget_end_page   = lo + static_cast<std::uint32_t>(drop);
            if (edge_policy == BudgetEdgePolicy::RefuseNotTruncate) {
                // sum_dir_reach.h:257/:773's rule. The plan is returned EMPTY and the defect
                // (BudgetTruncated) plus the dropped range are the refusal's whole content.
                //
                // [line `redkv`] ALL `want` OF THEM WERE DROPPED, not just `drop`. The plan holds
                // NO page, so the `max_pages` it would have kept were removed by this same edge;
                // recording only the excess left one page unaccounted for per refusal, which is
                // the page the refusal's own next field (`executed=0` beside `wanted=343`) already
                // counts as lost. The plan's OWN derived shortfall agrees: `cut_pages()` is
                // `admitted_pages() - count()` = 343 - 0, and its doc comment calls it "the pages
                // the plan did NOT hold out of the ones it could have held". The identity the
                // sibling test asserts ("no page is lost unaccounted for") is what forces this.
                plan.dropped_budget = static_cast<std::uint32_t>(want);
                // ... and the range it names is the range it dropped: [lo, hi) of the clamped run,
                // which for a whole-run refusal is the whole run. Leaving the narrow
                // [lo, lo+drop) range beside a `dropped_budget` of `want` would put two different
                // ranges in one field pair.
                plan.dropped_budget_begin_page = lo;
                plan.dropped_budget_end_page   = hi;
                plan.pages.clear();
                plan.bytes = 0;
                return plan;
            } else if (edge_policy == BudgetEdgePolicy::TruncateNewestKeepAnchor) {
                // Keeps the anchor's end of the run: still a truncation, but the claim at
                // sum_dir_reach.h:762-765 stays true for the text that executes.
                hi -= static_cast<std::uint32_t>(drop);
            } else {
                lo += static_cast<std::uint32_t>(drop);
            }
        }
    }
    if (lo >= hi) { return plan; }

    plan.pages.reserve(hi - lo);
    for (std::uint32_t page = lo; page < hi; ++page) { plan.pages.push_back(page); }

    // =======================================================================================
    // [F-1019] THE NOMINATIONS -- ADDITIVE, AND EACH ONE ACCOUNTED FOR.
    // =======================================================================================
    // WHERE THIS SITS, AND WHY IT IS THE LAST THING THAT TOUCHES `pages`. Everything above it is
    // the ARBITER'S RUN and is left byte-identical: the frontier clamp, the hole cut, the token
    // budget's refusal and the byte budget's edge all still see exactly the run they saw before,
    // so the anchor-preservation the edge policy exists for is untouched. A nomination is an
    // ADDITION to a decided run, never an input to deciding it -- which is what stops a shared
    // lookup from moving the anchor (`semchan_stage2.h:160-163`).
    //
    // THE CRITERION IS THE SAME ONE CRITERION. A nominated page is kept iff `has_live_page(page)`
    // says so -- literally the predicate the hole cut at `:1573` calls -- so a shared lookup cannot
    // hold a page the standalone lookup would refuse. There is no second admissibility rule here,
    // deliberately: two predicates on one page is what let the hot path and the replay disagree.
    if (!request.nominated_pages.empty()) {
        const std::size_t n_nom = request.nominated_pages.size();
        if (request.nominated_owner.size() != n_nom) {
            // A CALLER DEFECT, NAMED. The owner vector is how the ONE shared result is handed back
            // per agent; without a length match there is no agent to attribute a page to, and
            // guessing one would be the silent substitution this project refuses. The plan is
            // returned with the run intact and the defect recorded -- NOT thrown, because a plan
            // is a DECISION (this function does no I/O) and the caller owns the repair.
            plan.nominated_gap = static_cast<std::uint32_t>(n_nom);
        } else {
            for (std::size_t i = 0; i < n_nom; ++i) {
                const std::uint32_t page  = request.nominated_pages[i];
                const std::uint32_t agent = request.nominated_owner[i];
                if (i != 0 && page <= request.nominated_pages[i - 1]) {
                    // NOT ASCENDING or duplicated: the invariant the struct documents. Counted,
                    // not repaired -- a sort here would hide the caller's bug behind an order the
                    // caller did not ask for.
                    ++plan.nominated_gap;
                    continue;
                }
                if (page < lo || page >= hi) {
                    // OUTSIDE THE DECIDED RUN, i.e. the case the whole field exists for. It is
                    // still bounded by the committed frontier (the run's own `hi` already was) and
                    // it still has to be live.
                    if (!has_live_page(page)) {
                        ++plan.nominated_missing;
                        continue;
                    }
                    plan.pages.push_back(page);
                    ++plan.nominated_kept;
                    plan.nominated_page_agents.push_back(agent);
                    plan.nominated_pages_kept.push_back(page);
                } else {
                    // ALREADY IN THE RUN: cost 0, gain 0. Recorded as KEPT (it IS in the plan) and
                    // attributed to the agent that named it, so the per-agent view is complete.
                    ++plan.nominated_kept;
                    plan.nominated_page_agents.push_back(agent);
                    plan.nominated_pages_kept.push_back(page);
                }
            }
        }
        // `pages` stays ASCENDING, the order `:1254-1258` makes load-bearing ("ascending order is
        // the order that lets consecutive pages merge into one batched read per layer file").
        // `std::sort` is used rather than an insertion pass because the two sources (run, then
        // nominations) interleave arbitrarily and a sort states the invariant outright.
        std::sort(plan.pages.begin(), plan.pages.end());
        plan.pages.erase(std::unique(plan.pages.begin(), plan.pages.end()), plan.pages.end());
    }

    plan.bytes = static_cast<std::uint64_t>(plan.pages.size()) * request.page_bytes;
    return plan;
}

// The engine's call sites keep this spelling; it forwards the tree's edge policy. The sibling
// test calls `plan_recall_pages_with_policy` directly, so every policy is EXERCISED without
// editing the constant -- and the constant is what the engine's static_assert refuses while the
// owner has not named one.
[[nodiscard]] inline RecallPagePlan
plan_recall_pages(const RecallPagePlanRequest& request,
                  const std::function<bool(std::uint32_t)>& has_live_page) {
    return plan_recall_pages_with_policy(request, has_live_page, kRecallBudgetEdgePolicy);
}

// ---------------------------------------------------------------------------
// the cost model, with the measured anchors as defaults
// ---------------------------------------------------------------------------
//
// This exists so the refusal is arithmetic rather than taste: a plan whose per-token
// cost is not below re-prefill must not be executed, and a plan that would stream
// per token must be refused outright (it is ~10^3.6 worse than doing nothing).
//
// ⛔ [INEXACTGATE C] NAMED GAP -- the SECOND stated refusal above has no predicate at all.
// "a plan that would stream per token must be refused outright" is not implemented anywhere in
// this file or in its caller: the caller consults `net_positive()` and nothing else
// (`program_impl.h:14159`). It is expressible with declared fields -- `read_ms()` vs
// `decode_us_per_token`, the comparison the sibling test already asserts at
// `turn_recall_journal_test.cpp:345-349` -- and wiring it NARROWS admission, which is the
// owning line's decision, so this line names it and does not take it. Do not read the prose
// above as a check: as of this writing there is no check behind it.

// The struct's own defaults, named ONCE, so the detector below cannot drift from the
// initialisers it is a statement about.
inline constexpr double kRecallCostDefaultReadUsPerToken      = 6.1;     // 2.6 .. 6.1, measured
inline constexpr double kRecallCostDefaultReprefillUsPerToken  = 365.0;   // 2740 tok/s in-tree
inline constexpr double kRecallCostDefaultDecodeUsPerToken     = 38500.0; // 25.97 tok/s in-tree

struct RecallCost {
    double read_us_per_token      = kRecallCostDefaultReadUsPerToken;
    double reprefill_us_per_token = kRecallCostDefaultReprefillUsPerToken;
    double decode_us_per_token    = kRecallCostDefaultDecodeUsPerToken;
    std::uint64_t tokens          = 0;

    [[nodiscard]] double read_ms() const noexcept {
        return static_cast<double>(tokens) * read_us_per_token / 1000.0;
    }
    [[nodiscard]] double reprefill_ms() const noexcept {
        return static_cast<double>(tokens) * reprefill_us_per_token / 1000.0;
    }
    [[nodiscard]] double speedup_vs_reprefill() const noexcept {
        return read_us_per_token > 0.0 ? reprefill_us_per_token / read_us_per_token : 0.0;
    }
    [[nodiscard]] bool net_positive() const noexcept {
        return read_us_per_token < reprefill_us_per_token;
    }

    // ---- [INEXACTGATE C] WHAT THE GATE ACTUALLY DECIDED ------------------------------
    // `net_positive()` compares two RATES, so `tokens` cancels: that is CORRECT and is not the
    // defect -- both sides are per token, so the comparison is plan-independent by construction.
    // The defect is what a reader does with the result. A run that printed `refused_cost=0` was
    // read as "a check passed", when with the shipped defaults the predicate is the compile-time
    // constant `6.1 < 365.0` and the `return;` in the caller CANNOT be reached: on that path
    // `refused_cost=0` is evidence that NO CHECK RAN (dl/holegate sec.2.5, sec.6.1 -- and no
    // capture in the fleet's census contains `refused_cost=[1-9]`).
    //
    // This does not change `net_positive()`'s meaning by one character. It names WHICH of the
    // three states the caller is in, so the line can say it and a reader no longer has to
    // infer a check from a zero.
    enum class Verdict : std::uint8_t {
        NotPositive = 0,          // the read does not beat re-prefill: the caller refuses
        PositiveMeasured,         // the predicate ran on rates that came from a measurement
        PositiveDefaultConstant,  // ⭐ the predicate IS the initialiser: no check ran
    };
    [[nodiscard]] Verdict verdict() const noexcept {
        if (!net_positive()) { return Verdict::NotPositive; }
        const bool defaults =
            read_us_per_token == kRecallCostDefaultReadUsPerToken &&
            reprefill_us_per_token == kRecallCostDefaultReprefillUsPerToken;
        return defaults ? Verdict::PositiveDefaultConstant : Verdict::PositiveMeasured;
    }
    // The same fact as a bool, for a caller that wants to count or assert it.
    [[nodiscard]] bool rate_gate_is_default_constant() const noexcept {
        return verdict() == Verdict::PositiveDefaultConstant;
    }
};

// ===========================================================================
// [RECALLCHEAP] RESTORE-FIRST -- WHICH OF THE TWO RESTORE LEGS RUNS, PER PLAN.
// ===========================================================================
//
// THE TWO LEGS. Both are live, and both live in ONE function,
// `program_impl.h`'s `recall_cold_pages_for_round`:
//
//   BYTE LEG  -- `restore_cold_page` (`program_impl.h:13281`). Read the page's packed
//                record (nvfp4 rANS / i8 / bf16 / rk4v4), decode it back into the K/V
//                plane the attention kernels read, publish the physical page index.
//                NO new tokens, no extent growth, no workspace re-plan: the page becomes
//                hot IN PLACE.
//   TEXT LEG  -- the arm gated on `NINFER_RECALL_TEXT` (`program_impl.h:13875`). Read the
//                block's 256 B cargo record and RE-PREFILL the text (append, or
//                original-position rebuild under `NINFER_RECALL_TEXT_REBUILD`).
//
// MEASURED ON THIS TREE, not modelled: one planned page = 64 tokens, `--cold-policy disk`,
// the 1M geometry of `dl/fusionrefresh/runs/A2_armed_fixed`:
//
//   [context-append] lane=0 tokens=64 source=text-cargo blocks=1 cargo_bytes=256 prefill_ms=2218.87
//   [recall] pages=1 bytes=1232896 restored=1 read_ms=0.390400 reprefill_ms=23.360000 ratio=59.836066:1 hook_ms=2225.490
//
// !! THE TWO NUMBERS ON THE `[recall]` LINE ARE THE COST MODEL, NOT A MEASUREMENT. `read_ms()`
// and `reprefill_ms()` are `tokens * <rate constant>` (the RecallCost defaults above, 6.1 and
// 365.0 us/token). The engine's own measured cost of the SAME 64 tokens is `prefill_ms=2218.87`
// = 34,670 us/token, i.e. 95x the model's 365.0. The model does not know which leg ran, so a
// line that prints it does not report what was spent. `prefill_ms` on the `[context-append]` /
// `[context-rebuild]` lines is the measurement; that is the number this file's predicate is
// priced against.
//
// WHY A SWITCH IS NEEDED AT ALL. Today the legs are chosen by a GLOBAL env PRESENCE test, so
// `NINFER_RECALL_TEXT` REPLACES the byte leg wholesale -- for every plan, including the plans
// whose bytes are still materialisable. The choice belongs to the PLAN, not to the process, and
// this predicate is that choice.
//
// ALL-OR-NOTHING, PER PLAN, deliberately: a plan that injected half its pages as restored bytes
// and half as re-prefilled text would put two different POSITION semantics into one round's
// attention set (the byte leg restores rows IN PLACE, the append arm puts them AFTER the
// frontier). That is the "two mechanisms inside one decision" shape `sum_dir_reach.h [text: substitute the positional first-fit silently; = :110 on 2026-09-25]`
// already refuses for the selector. One plan, one mechanism, and the line says which.
//
// HOST-ONLY, PURE, std-only: no CUDA, no engine header, no I/O -- the same shape as
// `RecallCost` above. `tests/test_recall_restore_first.cpp` pins it with a
// GREEN / MUTANT / SPECIFIC triple.
// ===========================================================================

// The env name. The VALUE is the mode word; absent or unrecognised is the DEFAULT, never a
// silent third mode, and never "0 means off" (the tree's sibling recall flags spell OFF as "0"
// as a PRESENCE test -- this one does not, and says so, because it has three states, not two).
inline constexpr const char* kRecallRestoreFirstEnv = "NINFER_RECALL_RESTORE_FIRST";

enum class RecallRestoreFirst : std::uint8_t {
    // TODAY'S BEHAVIOUR, and the default: an armed text route replaces the byte leg.
    Text  = 0,
    // The byte leg is preferred per plan; the text route is the FALLBACK for byte-absent plans.
    Bytes = 1,
};

inline constexpr RecallRestoreFirst kRecallRestoreFirstDefault = RecallRestoreFirst::Text;

[[nodiscard]] inline const char* recall_restore_first_name(RecallRestoreFirst mode) noexcept {
    switch (mode) {
        case RecallRestoreFirst::Text:  return "text";
        case RecallRestoreFirst::Bytes: return "bytes";
    }
    return "text";
}

[[nodiscard]] inline RecallRestoreFirst recall_restore_first_parse(const char* value) noexcept {
    if (value == nullptr) { return kRecallRestoreFirstDefault; }
    const std::string text(value);
    if (text == "bytes") { return RecallRestoreFirst::Bytes; }
    if (text == "text")  { return RecallRestoreFirst::Text; }
    return kRecallRestoreFirstDefault;
}

// THE DECISION. TRUE iff the TEXT route runs.
//
//   text_route_armed                -- `NINFER_RECALL_TEXT` present AND the cargo exists. FALSE
//                                      makes this predicate FALSE, which is why the predicate is
//                                      a no-op on every path that has no text route at all.
//   planned_pages                   -- `plan.pages.size()`.
//   pages_with_materialisable_bytes -- how many of those pages satisfy the BYTE LEG'S OWN two
//                                      conditions (`program_impl.h:14299-14302`): still
//                                      `cold_compressed`, and still carrying a `cold_pages` entry.
//                                      No new admission is invented here.
[[nodiscard]] inline bool recall_text_route_wins(RecallRestoreFirst mode,
                                                 bool text_route_armed,
                                                 std::size_t planned_pages,
                                                 std::size_t pages_with_materialisable_bytes) noexcept {
    if (!text_route_armed) { return false; }
    if (mode != RecallRestoreFirst::Bytes) { return true; }
    if (planned_pages == 0U) { return true; }
    return pages_with_materialisable_bytes != planned_pages;
}

[[nodiscard]] inline const char* recall_cost_verdict_name(RecallCost::Verdict verdict) noexcept {
    switch (verdict) {
        case RecallCost::Verdict::NotPositive: return "not-positive";
        case RecallCost::Verdict::PositiveMeasured: return "positive-measured";
        case RecallCost::Verdict::PositiveDefaultConstant: return "positive-default-const";
    }
    return "unknown";
}

// The one line the engine prints for a round that actually recalled something (and
// nothing at all for a round that did not: silence must mean "no-op", not "unclear").
// ⚠ [INEXACTGATE B] THAT CONTRACT IS NOT TRUE OF THE SHIPPED ENGINE, and this line does not
// change it: the engine prints nothing at all for a round it REFUSED, and it prints the loud
// `INEXACT` word for a round it EXECUTED. `program_impl.h:14139` returns on an empty plan with
// no output, and an empty plan is what a hole at the first requested page or a sub-page budget
// produces (`plan_recall_pages` returns at :1254/:1266 with the counters SET) -- so today
// silence means "no-op OR a total refusal", two different facts. `recall_refusal_line()` below
// is the named form for the second; the engine prints it at that branch (patch [B]).
// ---- [RESTOREUNITS] THE RESTORE UNIT, DECIDED IN ONE PLACE SO IT CAN BE EXECUTED -----------
// `pages_restored` had ONE name and TWO units: the text-cargo append arm and the context-rebuild
// arm incremented it once per ROUND, and the KV read-back arm added the pages it really
// restored. Every recall run on the shipped engine takes the append arm, so the counter read
// `6` where 8 pages had been appended -- and the same number was printed as the NUMERATOR of
// `pages=R/P` beside a denominator (`pages_planned`) that IS a page count. A field whose unit is
// not fixed cannot be compared with anything, so the decision is moved here, into one constexpr
// function that every arm and the round's own census call.
//
// ⛔ WHAT THIS FUNCTION DELIBERATELY DOES NOT DO: it does NOT return the planned count as the
// restored count. `restored = planned` would make every run read as complete -- including a run
// whose append arm `break`s on a page whose directory row is dead, or whose cargo record is
// missing, and a run whose read-back arm `continue`s on a page that is not compressed. Both are
// reachable (see the two `break`s and the two `continue`s in the restore arms), so a plan of N
// pages CAN bring back fewer, and that fact belongs on the log rather than in a smoothing
// function. `short_rounds` / `short_pages` below are its NAMED form.
struct RecallRestoreDelta {
    std::uint64_t pages        = 0;  // the pages this round ACTUALLY brought back
    std::uint64_t short_rounds = 0;  // 1 when the round did not bring back what its plan asked
    std::uint64_t short_pages  = 0;  // the deficit in pages (0 when it brought back enough)
};

// A round whose restored count differs from its plan's count is COUNTED, in both directions:
// fewer means pages were dropped silently by an arm, more means the number handed in is not
// this round's restore at all (the unit drifted again). Neither is smoothed away.
[[nodiscard]] constexpr RecallRestoreDelta recall_restore_delta(std::uint64_t planned_pages,
                                                               std::uint64_t restored_pages) noexcept {
    RecallRestoreDelta d{};
    d.pages = restored_pages;
    if (restored_pages != planned_pages) {
        d.short_rounds = 1;
        d.short_pages  = planned_pages > restored_pages ? planned_pages - restored_pages : 0;
    }
    return d;
}

// The run-level form of the same fact, carried by two counters so the sequence-end line can
// print it. `pages` is not here: it is already the counter the arms advance, and duplicating it
// would create a second place for the two to disagree.
struct RecallRestoreTally {
    std::uint64_t short_rounds = 0;
    std::uint64_t short_pages  = 0;
};

[[nodiscard]] constexpr RecallRestoreTally recall_restore_tally_add(
    RecallRestoreTally tally, const RecallRestoreDelta& round) noexcept {
    tally.short_rounds += round.short_rounds;
    tally.short_pages  += round.short_pages;
    return tally;
}

[[nodiscard]] inline std::string recall_line(const RecallPagePlan& plan,
                                             const RecallCost& cost,
                                             std::uint64_t pages_restored,
                                             std::uint64_t refused_runs) {
    std::string out = "[recall] pages=" + std::to_string(plan.count()) +
                      " bytes=" + std::to_string(plan.bytes) +
                      " restored=" + std::to_string(pages_restored) +
                      " read_ms=" + std::to_string(cost.read_ms()) +
                      " reprefill_ms=" + std::to_string(cost.reprefill_ms()) +
                      " ratio=" + std::to_string(cost.speedup_vs_reprefill()) + ":1";
    if (plan.dropped_hole != 0) {
        out += " hole=" + std::to_string(plan.dropped_hole);
        out += " absent=" + std::to_string(plan.dropped_absent);
        out += " cut_above_hole=" + std::to_string(plan.cut_above_hole());
    }
    if (plan.dropped_budget != 0) { out += " budget=" + std::to_string(plan.dropped_budget); }
    if (plan.dropped_clamped != 0) { out += " clamped=" + std::to_string(plan.dropped_clamped); }
    if (refused_runs != 0) { out += " refused_runs=" + std::to_string(refused_runs); }
    // [INEXACTGATE A/D] The defect's NAME and the wanted run, ADDED. Everything above keeps its
    // exact bytes and its exact meaning -- in particular the last field stays byte-identical, so
    // every grep the fleet already uses (`INEXACT`, `never approximated`, `hole=`) still hits.
    if (!plan.exact()) {
        out += " defect=" + std::string(recall_plan_defect_name(plan.defect()));
        out += " wanted=" + std::to_string(plan.wanted_pages());
        out += " cut=" + std::to_string(plan.cut_pages());
    }
    if (plan.dropped_budget != 0) {
        out += " budget_dropped=[" + std::to_string(plan.dropped_budget_begin_page) + "," +
               std::to_string(plan.dropped_budget_end_page) + ")";
    }
    if (!plan.exact()) { out += " INEXACT(never approximated)"; }
    return out;
}

// ⭐ [INEXACTGATE A/B] ONE producer for the refusal sentence. The empty-plan refusal (finding 2)
// and the inexact-plan refusal (finding 1) therefore carry identical content, and a reader who
// greps a run for `REFUSED-` sees every refusal the recall path made -- including the ones that
// today print nothing at all. Call it only when `defect() != None`; it names `none` if misused
// rather than pretending.
[[nodiscard]] inline std::string recall_refusal_line(const RecallPagePlan& plan,
                                                    const RecallCost& cost) {
    std::string out = "[recall] REFUSED-";
    out += recall_plan_defect_name(plan.defect());
    // [PREFILLBUDGET] the lowercase, greppable name (`refused-prefill-budget` for the token
    // budget), plus the two numbers the refusal is arithmetic on. Appended, never substituted:
    // the `REFUSED-<defect>` header above is what every existing harness greps and its meaning is
    // not redefined here.
    out += " refusal=" + std::string(recall_plan_refusal_name(plan.defect()));
    if (plan.defect() == RecallPagePlan::Defect::PrefillBudget) {
        out += " wanted_tokens=" + std::to_string(plan.prefill_tokens_wanted);
        out += " budget_tokens=" + std::to_string(plan.prefill_token_budget);
    }
    out += " wanted=" + std::to_string(plan.wanted_pages());
    out += " admitted=" + std::to_string(plan.admitted_pages());
    out += " executed=" + std::to_string(plan.count());
    if (plan.cut_above_hole() != 0) {
        out += " recallable_but_cut=" + std::to_string(plan.cut_above_hole());
    }
    out += " " + recall_line(plan, cost, 0, 0);
    return out;
}

// ---------------------------------------------------------------------------
// [INEXACTGATE A] THE OWNER'S POLICY SITE -- and there is deliberately NO DEFAULT HERE.
// ---------------------------------------------------------------------------
//
// dl/holegate/REPORT.md sec.4.2 named this site and refused to design it, because the two
// policies are two different PRODUCTS and not a bug fix:
//
//   (A) RefuseRound     the round does not execute; the answer then comes from the resident
//                       prompt. The record already contains the natural experiment: C1's recall
//                       was refused and it answered correctly (`330`), so (A) is demonstrated,
//                       on this binary family, at this scale, to be survivable.
//   (B) ReportOnly      today's behaviour: the round executes and the record says it was inexact.
//                       C4 (truncated answer) and D1 (wrong answer) are what that produces --
//                       and sec.6.3 shows the wrongness is NOT attributable to the recall, so
//                       (B) is a defensible product too. Nothing in the captures decides it.
//
// `Unnamed` is not one of the two: it is the ABSENCE of a choice. The engine's call site
// static_asserts it away, so this patch cannot land a policy silently in either direction.
//
// ⭐ AND THE CHOICE HAS BEEN MADE: `RefuseRound`. The owner's ruling on this budget family is
// 「点名拒绝而不是截断」 -- a NAMED REFUSAL, not a truncation -- and `dl/vectorkey` ROW 2's
// "never a partial injection" is that ruling landed (dl/vectorkey/REPORT.md:44); the fleet's
// absolute rule is that a silent wrong answer is worse than a refusal. Policy (B), `ReportOnly`,
// is what D1 did: it executed a run cut from 8 pages to 1 and produced a wrong answer whose
// cause the record cannot attribute to the recall (dl/holegate sec.6.3) -- i.e. it is the shape
// the fleet's rule forbids, not a conservative default. `ReportOnly` remains expressible so the
// experiment in holegate's LEAD can still be run against BOTH arms by changing one token.
enum class InexactAdmissionPolicy : std::uint8_t {
    Unnamed = 0,  // not a policy: the engine's static_assert refuses it (regression guard)
    ReportOnly,   // executes as today, named on the line (what D1 did)
    RefuseRound,  // ⭐ NAMED: does not execute; one named line, no restore
};
inline constexpr InexactAdmissionPolicy kInexactAdmissionPolicy =
    InexactAdmissionPolicy::RefuseRound;
[[nodiscard]] inline constexpr const char* inexact_admission_policy_name(
    InexactAdmissionPolicy policy) noexcept {
    switch (policy) {
        case InexactAdmissionPolicy::Unnamed: return "unnamed";
        case InexactAdmissionPolicy::ReportOnly: return "report-only";
        case InexactAdmissionPolicy::RefuseRound: return "refuse-round";
    }
    return "unknown";
}

// WHICH OF THE THREE DROP COUNTERS MAKE A CUT REFUSE-WORTHY, by the TREE'S OWN stated rules --
// this is a mapping, not the owner's policy (the owner's policy is the enum above):
//   BudgetTruncated  YES. The tree's rule for this exact named concept is `RefusedBudget`,
//                    "refusing, not truncating" (sum_dir_reach.h:257, :773-793), and dl/vectorkey
//                    ROW 2 requires that rule for a new budget. This site does the opposite.
//   HoleCut          YES. `trace:turn_recall_journal.h:80` "A hole is a REFUSAL, not a clamp"
//                    -- but that refusal is the NARROW one (never fill the hole with a
//                    neighbour's bytes). What it does NOT say is whether the resulting SHORT
//                    run may feed a generation, and the hole's own arithmetic is why it must:
//                    N-1 of the dropped pages are LIVE and recallable and were discarded by the
//                    contiguity rule (dropped_absent names the one that is truly gone), so the
//                    executed text is a prefix of the wanted run with no record of that fact
//                    anywhere the answer channel can see.
//   FrontierClamped  NO -- and this is the one reading a reviewer should check. Nothing that
//                    exists was dropped: the plan holds the whole run the committed frontier can
//                    serve, and `dropped_clamped` counts the part of the ASK that has no bytes
//                    yet. A refusal on this reason alone would refuse rounds whose recall is
//                    complete. ⭐ THE OWNER CONFIRMED THIS READING when naming `RefuseRound`
//                    ("not on dropped_clamped alone (nothing that exists was dropped)"), and the
//                    sibling test now has TWO RED arms for it: a clamped-only fixture must
//                    produce NO refusal, and a pure-tail hole (dropped_hole=1, dropped_absent=1,
//                    cut_above_hole=0) MUST -- because a gate on `cut_above_hole() > 0` would
//                    miss exactly that case. A gate that fires on the wrong counter is the same
//                    defect class this patch exists to fix.
[[nodiscard]] inline constexpr bool recall_plan_refuses_by_tree_rule(
    RecallPagePlan::Defect defect) noexcept {
    // [PREFILLBUDGET] PrefillBudget is YES, and for the STRONGEST form of the same reason as
    // BudgetTruncated: the plan was refused instead of truncated, so it holds nothing at all and
    // there is no short run for a round to execute. It is listed here so the mapping stays TOTAL
    // over `Defect` -- a new defect that is silently "not refuse-worthy" is how a refusal stops
    // being a refusal, and this chain is the one place that rule is written down.
    return defect == RecallPagePlan::Defect::PrefillBudget ||
           defect == RecallPagePlan::Defect::HoleCut ||
           defect == RecallPagePlan::Defect::BudgetTruncated;
}

// The refusal decision, as a pure function of the plan and the owner's NAMED policy, so the
// engine's branch is one call and the test can exercise every policy without editing a constant.
[[nodiscard]] inline bool recall_plan_refuses_round(const RecallPagePlan& plan,
                                                   InexactAdmissionPolicy policy) noexcept {
    if (policy != InexactAdmissionPolicy::RefuseRound) { return false; }
    return recall_plan_refuses_by_tree_rule(plan.defect());
}

// ---------------------------------------------------------------------------
// THE TEXT CARGO -- the cold payload, held as TEXT (2026-09-15)
// ---------------------------------------------------------------------------
//
// WHY THIS IS IN THIS FILE AND NOT IN A SECOND FORMAT HEADER
//
//   This header already owns the one thing a retired block's record is: the 64-byte L0
//   record above, keyed (digest, page), with `file_slot` naming "where the bytes went".
//   The cargo changes WHAT the bytes are, not what the record says -- and a second file
//   owning "what a retired page looks like" is exactly the drift (three spellings of one
//   concept) that recall_identity.h's header comment was written to remove.
//
// THE ORDER IT COMES FROM (user, 2026-09-15, verbatim)
//
//   "冷kv按语义卸载，多并发共用一个卸载总结亚并发，然后用重新 prefill 来实现超长文本"
//   "我不是要改变 64 的结构……就拿 64 的块拼卸载文本"
//   "召回 = 把这段文字重新 prefill"
//
//   So the durable payload of ONE retired block is its 64 TOKEN IDS, and a recall
//   re-prefills those ids instead of loading KV back.
//
// THE FORMAT, AND WHY IT NEEDS NO NEW KEY AND NO NEW FIELD
//
//   one block record = 64 token ids, little-endian uint32, FIXED stride 256 B.
//
//   * THE INDEX IS THE PREFIX'S OWN BLOCK NUMBER, i.e. `token_begin / kRecallPageTokens`.
//     It is NOT the file slot. That matters for the user's requirement 4 (目录抽象): the
//     cargo is addressed by the CONTENT's own coordinate, so the directory never has to
//     expose -- and a reader never has to know -- where a file region happens to sit.
//     (An earlier draft indexed the cargo by `file_slot`, which would have made the
//     catalog's answer depend on the allocator's mood; it was abandoned for this reason,
//     not for cost.)
//   * The block's IDENTITY is recomputed from the record's own 64 ids by
//     `spec/recall_identity.h`'s `recall_identity_tokens()`. Nothing positional is stored,
//     because a recalled block is re-prefilled at NEW positions (recall_identity.h:30-40):
//     a stored positional digest would fail to match its own record. That is the whole
//     binding -- and it is a RE-DERIVATION, not a second copy of the digest, so the two
//     cannot disagree.
//   * A block may sit in several cargo file *records* (N:1 aggregation is done by the
//     UNLOAD TEXT layer in sum_dir.h, which spans whole blocks); one cargo record per
//     block, always. Overlap is a property of the aggregation, never of the record.
//
// THE MAGIC IS A TRAP FOR THE ONE MISTAKE THAT MATTERS
//
//   A cargo file read as if it were a KV mirror (or the reverse) produces 256-byte records
//   of noise. The header carries the record stride AND the block size, and the reader
//   REFUSES a file whose stride is not 256 -- the same "the stride the writer used, so a
//   reader with a different geometry refuses instead of mis-reading" rule the 64-byte L0
//   record already carries in its `layer_bytes` field.
inline constexpr std::uint32_t kRecallCargoMagic      = 0x31475243U; // 'C','R','G','1'
inline constexpr std::uint8_t  kRecallCargoVersion    = 1U;
inline constexpr std::uint32_t kRecallCargoHeaderBytes = 32U;
inline constexpr std::uint32_t kRecallCargoBlockTokens = kRecallPageTokens;              // 64
inline constexpr std::uint32_t kRecallCargoBytesPerToken = 4U;                           // uint32 LE
inline constexpr std::uint32_t kRecallCargoBlockBytes =
    kRecallCargoBlockTokens * kRecallCargoBytesPerToken;                                 // 256

static_assert(kRecallCargoBlockBytes == 256U,
              "the cargo record stride is the format's only guarantee; the header states it");
static_assert(kRecallCargoBlockTokens == kRecallPageTokens,
              "one cargo record is ONE block, and the block is the engine's 64-token page");

// WHERE THE CARGO FILE LIVES: `<--cold-disk-path>/ninfer_text.cargo`, i.e. beside the
// per-layer KV spill files it replaces. No new CLI flag, no new option, no new API surface:
// a run that spills at all already names the directory the payload goes to.
//
// =========================================================================================
// THE KEY IS (WRITER, BLOCK INDEX), AND THE WRITER IS THE FILE.
// =========================================================================================
//
// WHAT THE KEY WAS, AND WHY IT ALIASED. `write_block` (:1738) and `read_block` (:1789) both
// compute `offset = kRecallCargoHeaderBytes + block_index * kRecallCargoBlockBytes`, and
// `block_index` is the page number INSIDE ONE SEQUENCE -- every lane counts it from 0
// (`SequenceState::ledger`, program.h:457; `cold_frontier`, program.h:526). So the key was the
// block index ALONE, and two writers that retire the same page number computed the same
// ABSOLUTE file offset: the second `fwrite` won, and the first writer's reader was handed the
// second writer's tokens. That is a SILENT wrong value -- `read_block` refuses only past the
// high-water mark and compares nothing at all.
//
// WHY THE RECORD STILL CANNOT CARRY THE WRITER. The record is 256 B = 64 uint32 lanes
// (:1377-1382) and those 64 lanes are FULLY CONSUMED by token ids on a full block. There is no
// byte in it for an identity, and adding one would move the stride, which is the format's own
// stated guarantee.
//
// SO THE WRITER LIVES IN THE FILE -- the one dimension that was free. Two writers must not
// share a path: given distinct writers the mapping below is INJECTIVE, so the two key spaces
// are disjoint and NO offset in one writer's file can be another writer's record. Note what
// this buys and what it does not: WITHIN one writer's own file the block index alone is still
// a COMPLETE key -- which is exactly why the record needs no byte for it -- and the high-water
// mark (`blocks_written_`) keeps precisely its old meaning, per writer, so no page a writer
// never wrote can be read back as a hole.
//
// WRITER 0 KEEPS THE LEGACY NAME, character for character. `--max-concurrency 1` is the regime
// every existing reading was taken in, its artefact path must not move, and a cargo file
// written before this change stays readable by writer 0 (same name, same header, same stride).
//
// WHAT THIS DOES *NOT* FIX, NAMED SO IT IS NOT MISTAKEN FOR FIXED:
//   * the DIRECTORY's join is still page-keyed. `SumDir::row_for_page(page, row)`
//     (sum_dir.h:1772) consults a dense page->row index whose tie rule is FIRST-WRITER-WINS,
//     so with two writers covering one page the LENGTH a reader is handed
//     (`sum_dir_row_tokens`, reached through `live_row_for_page`) can be the OTHER writer's.
//     The bytes are then this writer's but the length is not, so the read can splice this
//     writer's own zero padding. Fixing that means keying the lookup on the row's `generation`
//     column (sum_dir.h:381-387), i.e. binding `SequenceState::recall_sequence_tag` -- the
//     prerequisite the directory's own note already names, and NOT a change to this header.
//   * a WRITER SLOT IS REUSED by the next sequence on that lane, so a later sequence's page
//     number can still collide with a retained continuation's records -- the single-writer
//     aliasing the refuse site at program_impl.h:1320-1326 already marks NOT FIXED HERE. That
//     one needs the DURABLE (unbounded) identity, which is the tag above; and it cannot be the
//     file dimension, because an unbounded identity would mint unboundedly many files.
[[nodiscard]] inline std::string recall_cargo_path(const std::string& cold_disk_path,
                                                  std::uint32_t writer = 0U) {
    const std::string dir = cold_disk_path.empty() ? std::string("/tmp") : cold_disk_path;
    // Injective in `writer`: writer 0 is the legacy name; every other writer is disjoint from
    // it and from every sibling. This is the ONE place the writer->file mapping is spelled.
    if (writer == 0U) { return dir + "/ninfer_text.cargo"; }
    return dir + "/ninfer_text.w" + std::to_string(writer) + ".cargo";
}

// Little-endian, and written byte by byte so the format does not depend on the host's
// endianness -- the same discipline the L0 record's explicit padding carries.
inline void recall_cargo_encode_block(const std::uint32_t* tokens, std::uint8_t* out) noexcept {
    for (std::uint32_t i = 0; i < kRecallCargoBlockTokens; ++i) {
        const std::uint32_t value = tokens[i];
        out[i * 4U + 0U] = static_cast<std::uint8_t>(value & 0xFFU);
        out[i * 4U + 1U] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
        out[i * 4U + 2U] = static_cast<std::uint8_t>((value >> 16U) & 0xFFU);
        out[i * 4U + 3U] = static_cast<std::uint8_t>((value >> 24U) & 0xFFU);
    }
}

inline void recall_cargo_decode_block(const std::uint8_t* in, std::uint32_t* out) noexcept {
    for (std::uint32_t i = 0; i < kRecallCargoBlockTokens; ++i) {
        out[i] = static_cast<std::uint32_t>(in[i * 4U + 0U]) |
                 (static_cast<std::uint32_t>(in[i * 4U + 1U]) << 8U) |
                 (static_cast<std::uint32_t>(in[i * 4U + 2U]) << 16U) |
                 (static_cast<std::uint32_t>(in[i * 4U + 3U]) << 24U);
    }
}

// The file header, packed the same way (32 B): magic, version, stride, block tokens,
// block count, then the four bytes of a little-endian probe so a big-endian reader
// refuses instead of decoding every id byte-swapped.
struct RecallCargoHeader {
    std::uint32_t magic         = kRecallCargoMagic;
    std::uint8_t  version       = kRecallCargoVersion;
    std::uint8_t  reserved[3]   = {0, 0, 0};
    std::uint32_t record_bytes  = kRecallCargoBlockBytes;
    std::uint32_t block_tokens  = kRecallCargoBlockTokens;
    std::uint64_t block_count   = 0; // blocks WRITTEN so far (a high-water mark)
    std::uint32_t endian_probe  = 0x04030201U;
    std::uint32_t reserved2     = 0;
};

static_assert(sizeof(RecallCargoHeader) == kRecallCargoHeaderBytes,
              "the cargo header is one fixed 32-byte unit");

inline void recall_cargo_pack_header(const RecallCargoHeader& header,
                                     std::uint8_t* out) noexcept {
    const auto put32 = [out](std::size_t at, std::uint32_t value) {
        out[at + 0] = static_cast<std::uint8_t>(value & 0xFFU);
        out[at + 1] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
        out[at + 2] = static_cast<std::uint8_t>((value >> 16U) & 0xFFU);
        out[at + 3] = static_cast<std::uint8_t>((value >> 24U) & 0xFFU);
    };
    const auto get32 = [&put32, out](std::size_t at, std::uint64_t value) {
        put32(at + 0U, static_cast<std::uint32_t>(value & 0xFFFFFFFFULL));
        put32(at + 4U, static_cast<std::uint32_t>((value >> 32U) & 0xFFFFFFFFULL));
    };
    put32(0U, header.magic);
    out[4] = header.version;
    out[5] = 0;
    out[6] = 0;
    out[7] = 0;
    put32(8U, header.record_bytes);
    put32(12U, header.block_tokens);
    get32(16U, header.block_count);
    put32(24U, header.endian_probe);
    put32(28U, header.reserved2);
}

[[nodiscard]] inline RecallCargoHeader recall_cargo_unpack_header(const std::uint8_t* in) noexcept {
    const auto get32 = [in](std::size_t at) {
        return static_cast<std::uint32_t>(in[at + 0]) |
               (static_cast<std::uint32_t>(in[at + 1]) << 8U) |
               (static_cast<std::uint32_t>(in[at + 2]) << 16U) |
               (static_cast<std::uint32_t>(in[at + 3]) << 24U);
    };
    RecallCargoHeader header;
    header.magic        = get32(0U);
    header.version      = in[4];
    header.record_bytes = get32(8U);
    header.block_tokens = get32(12U);
    header.block_count  = static_cast<std::uint64_t>(get32(16U)) |
                          (static_cast<std::uint64_t>(get32(20U)) << 32U);
    header.endian_probe = get32(24U);
    header.reserved2    = get32(28U);
    return header;
}

[[nodiscard]] inline bool recall_cargo_header_ok(const RecallCargoHeader& header) noexcept {
    return header.magic == kRecallCargoMagic && header.version == kRecallCargoVersion &&
           header.record_bytes == kRecallCargoBlockBytes &&
           header.block_tokens == kRecallCargoBlockTokens &&
           header.endian_probe == 0x04030201U;
}

// ---------------------------------------------------------------------------
// THE PROVENANCE OF A BLOCK -- what "put it back where it was" reads
// ---------------------------------------------------------------------------
//
// NOTHING NEW IS WRITTEN, AND THE STRIDE DOES NOT MOVE. The position a block must be
// rebuilt at is ALREADY in the file, as the record's own index -- the format note above
// states it verbatim ("THE INDEX IS THE PREFIX'S OWN BLOCK NUMBER, i.e. `token_begin /
// kRecallPageTokens`"), the writer obeys it, and the reader is HANDED it (read_block's
// first argument) and today discards it. So the minimal sufficient persistence is ZERO new
// bytes: there is no version to bump and no old file to reject.
//
// WHY NOT A FIELD: a per-record position would be four more bytes on a 256-byte record,
// which breaks `static_assert(kRecallCargoBlockBytes == 256U)` and makes every cargo file
// written before the change unreadable. The index already carries the position at full
// fidelity, so a field would be a SECOND copy of one fact -- the drift recall_identity.h's
// header comment was written to remove.
//
// WHAT SPACE THE POSITION IS IN: the same LEDGER space `sequence.ledger` and a page plan's
// `pages` live in -- token 0 is index 0 -- so `token_begin` needs no conversion before it
// reaches a prefill's position base.

// One block AND the provenance its coordinate already implies.
struct RecallCargoLocatedBlock {
    std::uint64_t block_index = 0; // the coordinate the record was written at
    std::uint32_t token_begin = 0; // block_index * kRecallCargoBlockTokens, ledger space
    // THE BLOCK'S REAL LENGTH, from the directory row (sum_dir_row_tokens, sum_dir.h:502-504)
    // -- NOT the record stride. On success it is `row_tokens`; on refusal it is 0.
    std::uint32_t token_count = 0;
    std::uint32_t ids[kRecallCargoBlockTokens] = {};
};

// Pure: the ledger position of block `block_index`'s first token. The record's coordinate
// IS the position (see above), so this is the whole conversion.
[[nodiscard]] inline std::uint32_t
recall_cargo_position_of_block(std::uint64_t block_index) noexcept {
    return static_cast<std::uint32_t>(block_index * kRecallCargoBlockTokens);
}

// A RUN of located blocks: the ids in ONE flat vector -- byte-for-byte the `segment` the
// append leg builds today -- plus the provenance of each block, in the same ascending
// order. `positions[i]` is the ledger position of `ids[i * kRecallCargoBlockTokens]`.
struct RecallCargoLocatedRun {
    std::vector<std::uint32_t> ids;       // ascending, cut at the first refusal
    std::vector<std::uint32_t> positions; // one entry per block ACTUALLY READ
    // ⭐ PER-BLOCK LENGTH, parallel to `positions` (`block_tokens[i]` is how many of
    // `ids[i * <stride>]` onward are REAL). A run whose blocks are all full is unchanged in
    // every respect; a run with a short tail can only be rebuilt correctly by reading this.
    // A consumer that walks `ids` in fixed kRecallCargoBlockTokens strides instead of using
    // this is the bug the "padding would create tokens the model never saw" rule forbids.
    std::vector<std::uint32_t> block_tokens;
    std::uint64_t first_block = 0;

    [[nodiscard]] bool empty() const noexcept { return ids.empty(); }
    // The OLD invariant (positions.size() * kRecallCargoBlockTokens == ids.size()) holds
    // only while every block is FULL. It is replaced by the exact statement:
    // sum(block_tokens) == ids.size(), with one entry per block actually read.
    [[nodiscard]] bool lengths_consistent() const noexcept {
        if (block_tokens.size() != positions.size()) { return false; }
        std::uint64_t total = 0;
        for (const std::uint32_t count : block_tokens) {
            if (count == 0 || count > kRecallCargoBlockTokens) { return false; }
            total += count;
        }
        return total == ids.size();
    }
    [[nodiscard]] std::uint64_t blocks_read() const noexcept { return positions.size(); }
    [[nodiscard]] std::uint64_t token_count() const noexcept { return ids.size(); }
    [[nodiscard]] std::uint32_t first_token() const noexcept {
        return positions.empty() ? recall_cargo_position_of_block(first_block) : positions[0];
    }
    // A plan may hold NON-ADJACENT pages (a hole or a budget drop cut it), so a run is
    // always contiguous in the FILE but NOT necessarily in POSITION. When this is false the
    // caller must position EACH block by its own `positions[i]`: assuming one base would
    // rebuild every block after the gap at the wrong position.
    //
    // The stride is the previous block's REAL length, not the record stride: with a short
    // tail, `positions[i] == positions[i-1] + block_tokens[i-1]` is the contiguous case.
    [[nodiscard]] bool position_contiguous() const noexcept {
        if (block_tokens.size() != positions.size()) { return false; }
        for (std::size_t i = 1; i < positions.size(); ++i) {
            if (positions[i] != positions[i - 1U] + block_tokens[i - 1U]) { return false; }
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// HOW THE HANDLE IS OPENED: the ONE thing the write leg and the read leg want to be
// OPPOSITE about, so it is a PARAMETER and not a guess
// ---------------------------------------------------------------------------
//
// Both legs are right, about the same path, and they want opposite things:
//
//   * THE WRITE LEG (the tree's only call site, program_impl.h:1026:
//     `std::make_unique<TurnRecallTextCargo>(recall_cargo_path(dir))`) must CREATE the file
//     and TRUNCATE whatever a previous run left at that path. A fresh run that inherited the
//     previous turn's high-water mark would publish `records` it never wrote, and would read
//     back another turn's ids as if they were its own.
//   * THE READ LEG must open the file a previous turn ALREADY WROTE and must not lose one
//     byte of it -- which is exactly what `fopen(..., "w+b")` did on its first instruction.
//
// "Open the existing file if it exists, else create" LOOKS like it satisfies both and does
// not: it silently hands the WRITE leg the READ leg's semantics (no truncation), i.e. the
// failure above. So the mode is SPELLED OUT, and its default is the behaviour every existing
// call site already had -- the write leg's call site is unchanged, character for character.
enum class RecallCargoOpen : std::uint8_t {
    // The WRITE leg: fopen("w+b"). Creates, or TRUNCATES an existing file to the 32 B header.
    CreateOrTruncate = 0,
    // The READ leg: fopen("rb"). The file MUST already exist, and it is opened READ-ONLY, so
    // that the OPERATING SYSTEM -- not a comment, and not this type -- is what makes it
    // impossible for this handle to write over the file it is reading.
    ReadExisting = 1,
};

// The handle. Host-only and std-only like everything else in this namespace: it is a FILE*
// and two counters, and the reason it exists as a type rather than as three free functions
// is that the ONE rule it enforces -- "a record for block b is ALWAYS at byte offset
// header + b * 256" -- must be spelled in one place, or the writer and the reader drift.
class TurnRecallTextCargo {
public:
    // The default argument is load-bearing: it is what keeps the write leg's call site
    // byte-identical to what it was before a read mode existed.
    // THE WRITER IDENTITY. `writer` is the dimension the key is partitioned by, and it is a
    // PARAMETER rather than a re-derivation of `path` because this type must not be handed a
    // path it cannot check: `recall_cargo_open_writer` below is the ONE place that turns a
    // writer into a path AND constructs the handle from it, so the two cannot disagree.
    // Defaulted to 0, which is the legacy single-writer identity, so every existing call site
    // is byte-identical to what it was before this parameter existed.
    explicit TurnRecallTextCargo(const std::string& path,
                                 RecallCargoOpen open_mode = RecallCargoOpen::CreateOrTruncate,
                                 std::uint32_t writer = 0U)
        : path_(path), open_mode_(open_mode), writer_(writer) {
        if (open_mode_ == RecallCargoOpen::CreateOrTruncate) {
            file_ = std::fopen(path_.c_str(), "w+b");
            if (file_ == nullptr) { return; }
            RecallCargoHeader header;
            std::uint8_t bytes[kRecallCargoHeaderBytes] = {};
            recall_cargo_pack_header(header, bytes);
            if (std::fwrite(bytes, 1, kRecallCargoHeaderBytes, file_) != kRecallCargoHeaderBytes) {
                std::fclose(file_);
                file_ = nullptr;
            }
            return;
        }
        // READ AN EXISTING FILE. Two things happen here that never happened before, and BOTH
        // are needed before the handle can read the file at all -- a fix for either one alone
        // reads back exactly ZERO blocks:
        //
        //   (1) THE MODE. "rb", not "w+b": opened READ-ONLY, so opening it cannot truncate it.
        //       This is the mode that makes "point the reader at the cargo file the previous
        //       turn wrote" possible at all; before it, the constructor's first instruction
        //       set that file's length to 0 and its content to nothing.
        //   (2) THE COUNT. `blocks_written_` is loaded FROM THE HEADER. `read_block` refuses
        //       exactly when `block_index + 1 > blocks_written_`, and `blocks_written_` used
        //       to be advanced ONLY by `write_block`, i.e. only in the process that wrote the
        //       file. Without this line a freshly opened read handle has `blocks_written_ == 0`
        //       and REFUSES EVERY BLOCK, so even a non-truncating open would read nothing.
        //
        // The header is validated with the SAME `recall_cargo_header_ok` the format note above
        // promises ("the reader REFUSES a file whose stride is not 256"), so a foreign file --
        // or a KV mirror read as a cargo -- is refused here (`good() == false`) instead of
        // being decoded as 256-byte records of noise.
        //
        // `header.block_count` is deliberately NOT range-checked against the file size here: a
        // count that is too large is ALREADY safe, because `read_block`'s `fread` comes back
        // SHORT past EOF and refuses that block. A bound would therefore add a new refusal path
        // without closing a measured hazard.
        file_ = std::fopen(path_.c_str(), "rb");
        if (file_ == nullptr) { return; }
        std::uint8_t bytes[kRecallCargoHeaderBytes] = {};
        if (std::fread(bytes, 1, kRecallCargoHeaderBytes, file_) != kRecallCargoHeaderBytes) {
            std::fclose(file_);
            file_ = nullptr;
            return;
        }
        const RecallCargoHeader header = recall_cargo_unpack_header(bytes);
        if (!recall_cargo_header_ok(header)) {
            std::fclose(file_);
            file_ = nullptr;
            return;
        }
        blocks_written_ = header.block_count;
    }

    ~TurnRecallTextCargo() {
        if (file_ != nullptr) {
            // The header's block_count is a high-water mark, refreshed on every write, so a
            // crash costs the trailing records at most -- the same "the log is a MIRROR"
            // rule the L0 journal above states.
            //
            // Only a WRITABLE handle has anything to flush: C11 7.21.5.2 leaves `fflush` on a
            // purely input stream undefined, and a read handle may not write at all, so the
            // call is made exactly where it was made before for the write leg.
            if (writable()) { std::fflush(file_); }
            std::fclose(file_);
            file_ = nullptr;
        }
    }

    TurnRecallTextCargo(const TurnRecallTextCargo&) = delete;
    TurnRecallTextCargo& operator=(const TurnRecallTextCargo&) = delete;

    [[nodiscard]] bool good() const noexcept { return file_ != nullptr; }
    // Which leg this handle is. `false` means the file was opened READ-ONLY: `write_block`
    // refuses BY CONSTRUCTION below, and closing this handle cannot change one byte of it.
    [[nodiscard]] bool writable() const noexcept {
        return open_mode_ == RecallCargoOpen::CreateOrTruncate;
    }
    // The identity the key is partitioned by. Exposed so a caller (and a probe) can assert
    // WHICH writer's key space this handle addresses, instead of inferring it from the path.
    [[nodiscard]] std::uint32_t writer() const noexcept { return writer_; }
    [[nodiscard]] const std::string& path() const noexcept { return path_; }
    [[nodiscard]] std::uint64_t blocks_written() const noexcept { return blocks_written_; }
    [[nodiscard]] std::uint64_t bytes_written() const noexcept { return bytes_written_; }
    [[nodiscard]] std::uint64_t blocks_refused() const noexcept { return blocks_refused_; }

    // Writes ONE block at the coordinate the CONTENT owns (never at the file slot; see the
    // format note above). `count` is the block's REAL token length and may be SHORT
    // (`count < kRecallCargoBlockTokens`). The users' requirement, verbatim:
    //   「最后一块没存你就补齐了存呗」
    // -- a book or a corpus never ends on a 64 boundary, so refusing the tail block means
    // the last turn of every document is silently unstoreable.
    //
    // ⭐ THE OLD COMMENT IS NOT CONTRADICTED, IT IS ANSWERED. It said a short block is a
    // REFUSAL because "padding would create tokens the model never saw" -- a CORRECT worry.
    // The fix is on the READ side, not here: this writer still fills the fixed stride, but
    // it writes ZERO in every lane at and beyond `count`, and it records NO length of its
    // own; the block's real length travels in the DIRECTORY ROW that owns the block
    // (`sum_dir_row_tokens(row) = row.token_end - row.token_begin`, sum_dir.h:502-504), and
    // `read_located_block` below now takes exactly that count and hands it to the caller
    // instead of hard-setting the stride. So the padded lanes are never rebuilt from, which
    // is the property the old comment was protecting.
    //
    // ⚠️ NO WIRE CHANGE: the record stays kRecallCargoBlockBytes, the stride does not move,
    // the header does not move, and no byte was repurposed for a length. That is affordable
    // only because a SumDirRow has ALWAYS carried an arbitrary
    // `token_begin`/`token_end`/`content_count` triple and `append()` never checked the
    // count against 64 (sum_dir.h:809, verbatim), and because `recall_block_digest` already
    // folds `tokens.size()` into the digest (kv_recall_block.h:216-219), so a short block
    // and its own 64-lane extension stay distinguishable.
    //
    // ⚠️ NAMED UNSAFE CONSUMERS: relaxing THIS guard is INERT on its own -- the one in-tree
    // caller (program_impl.h:11929) still passes kPagedKVPageSize. It becomes live only when
    // that caller's own `base + kPagedKVPageSize <= sequence.ledger.size()` gate
    // (program_impl.h:11926) is relaxed, and THREE consumers then read a short block as if
    // it were 64. They are listed, with the two that cannot be fixed from their own
    // translation unit, in scratch/PATCHSET/REPORT.md section 3.
    //
    // A READ handle refuses here BY CONSTRUCTION, not merely because a FILE* opened "rb"
    // happens to fail its `fwrite`: the whole point of the read leg is that the file it just
    // read is byte-for-byte the file it leaves behind, and "the C library will refuse" is a
    // property of the mode STRING, not of this type.
    bool write_block(std::uint64_t block_index, const std::uint32_t* tokens,
                     std::uint32_t count) {
        if (file_ == nullptr || open_mode_ != RecallCargoOpen::CreateOrTruncate ||
            tokens == nullptr || count == 0 || count > kRecallCargoBlockTokens) {
            ++blocks_refused_;
            return false;
        }
        std::uint8_t record[kRecallCargoBlockBytes] = {};
        // Encode ONLY the `count` real ids and leave every remaining lane ZERO. The explicit
        // zeroed lane buffer is what makes that literal instead of incidental.
        std::uint32_t lanes[kRecallCargoBlockTokens] = {};
        for (std::uint32_t lane = 0; lane < count; ++lane) { lanes[lane] = tokens[lane]; }
        recall_cargo_encode_block(lanes, record);
        const std::uint64_t offset = static_cast<std::uint64_t>(kRecallCargoHeaderBytes) +
                                     block_index * kRecallCargoBlockBytes;
        if (offset > static_cast<std::uint64_t>(0x7FFFFFFFFFFFULL)) {
            ++blocks_refused_;
            return false;
        }
        if (std::fseek(file_, static_cast<long>(offset), SEEK_SET) != 0 ||
            std::fwrite(record, 1, kRecallCargoBlockBytes, file_) != kRecallCargoBlockBytes) {
            ++blocks_refused_;
            return false;
        }
        if (block_index + 1U > blocks_written_) { blocks_written_ = block_index + 1U; }
        bytes_written_ += kRecallCargoBlockBytes;
        std::uint8_t header_bytes[kRecallCargoHeaderBytes] = {};
        RecallCargoHeader header;
        header.block_count = blocks_written_;
        recall_cargo_pack_header(header, header_bytes);
        std::fflush(file_);
        const long here = std::ftell(file_);
        if (std::fseek(file_, 0, SEEK_SET) == 0) {
            (void)std::fwrite(header_bytes, 1, kRecallCargoHeaderBytes, file_);
            (void)std::fseek(file_, here, SEEK_SET);
        }
        return true;
    }

    // Reads one block. Returns false when the coordinate is outside the file, so a caller
    // can never read padding as if it were text (the same rule write_block enforces).
    //
    // ⭐ THE READ SIDE NOW MATCHES THE RELAXED WRITER -- this was the second half of
    // 「最后一块没存你就补齐了存呗」. The old body refused unless `count == 64`, which made the
    // writer's acceptance of `count < 64` unreachable from any read: a stored tail block could
    // not be read back by the length it was stored with. The rule is now the WRITER's own rule,
    // stated once and identically at both ends of the file: `count == 0 || count > 64` refuses.
    //
    // ⚠️ WHY THE DECODE IS *NOT* HANDED `out` DIRECTLY ANY MORE. `recall_cargo_decode_block`
    // unconditionally writes all 64 lanes of its destination, because the record it decodes is
    // ALWAYS 64 lanes (the stride does not move -- :1222-1245). Passing a caller's
    // `count`-sized buffer to it would write past that buffer for any `count < 64`: the exact
    // silent out-of-bounds write a relaxed guard invites. So the decode lands in an explicit
    // 64-lane local and only the first `count` lanes are copied out. For `count == 64` this is
    // the same bytes, in the same order, as before -- the loop is the identity -- so every
    // existing caller's reading is unmoved, and the padding the writer left in lanes
    // [count, 64) is never handed to a caller that did not ask for it.
    bool read_block(std::uint64_t block_index, std::uint32_t* out,
                    std::uint32_t count) const {
        if (file_ == nullptr || out == nullptr || count == 0 ||
            count > kRecallCargoBlockTokens) {
            return false;
        }
        if (block_index + 1U > blocks_written_) { return false; }
        const std::uint64_t offset = static_cast<std::uint64_t>(kRecallCargoHeaderBytes) +
                                     block_index * kRecallCargoBlockBytes;
        std::uint8_t record[kRecallCargoBlockBytes] = {};
        if (std::fseek(file_, static_cast<long>(offset), SEEK_SET) != 0 ||
            std::fread(record, 1, kRecallCargoBlockBytes, file_) != kRecallCargoBlockBytes) {
            return false;
        }
        std::uint32_t lanes[kRecallCargoBlockTokens] = {};
        recall_cargo_decode_block(record, lanes);
        for (std::uint32_t lane = 0; lane < count; ++lane) { out[lane] = lanes[lane]; }
        return true;
    }

    // READ THE BLOCK *AND* ITS POSITION. Same record, same stride, same file, same bytes as
    // read_block() above -- the ONLY difference is that the caller is handed the coordinate
    // it already had to supply. `const` for the same reason read_block is: it reads, and a
    // rebuild must not be able to mutate the payload it is rebuilding from.
    //
    // ERROR SHAPE: on refusal `out.token_count` is 0 and `out.ids` is left UNTOUCHED, so a
    // caller that ignores the return value still cannot rebuild from tokens the model never
    // saw. Never padded, never substituted, never truncated -- the same rule write_block and
    // read_block each state for themselves.
    // `row_tokens` IS the directory row's own length for this block
    // (`sum_dir_row_tokens(row) = row.token_end - row.token_begin`, sum_dir.h:502-504). It
    // is an ARGUMENT and not a constant because THIS header is the FILE FORMAT, not the
    // directory: the record carries a fixed 64-lane stride and no length field
    // (turn_recall_journal.h:912-915 leaves no byte for one), so the length can only come
    // from the row -- which is exactly why no wire change is needed (sum_dir.h:809).
    bool read_located_block(std::uint64_t block_index, std::uint32_t row_tokens,
                            RecallCargoLocatedBlock& out) const {
        out.block_index = block_index;
        out.token_begin = recall_cargo_position_of_block(block_index);
        out.token_count = 0;
        // Same refusal shape write_block now uses, so a caller cannot be handed a length the
        // record cannot hold and cannot be handed "zero tokens" as if it were a success.
        if (row_tokens == 0 || row_tokens > kRecallCargoBlockTokens) { return false; }
        // Read EXACTLY the row's lanes: the writer's zero padding in [row_tokens, 64) is not
        // read, so it cannot be consumed even by a caller that inspects the whole array.
        if (!read_block(block_index, out.ids, row_tokens)) { return false; }
        // Lanes [0, row_tokens) are the real ids; lanes [row_tokens, 64) are the writer's
        // zeros and MUST NOT be consumed. The count travels with the block from here on.
        out.token_count = row_tokens;
        return true;
    }

private:
    std::string     path_;
    RecallCargoOpen open_mode_    = RecallCargoOpen::CreateOrTruncate;
    std::uint32_t   writer_       = 0;  // the key's writer dimension; see recall_cargo_path
    std::FILE*      file_         = nullptr;
    std::uint64_t   blocks_written_ = 0;
    std::uint64_t   bytes_written_  = 0;
    std::uint64_t   blocks_refused_ = 0;
};

// ---------------------------------------------------------------------------
// THE ONE DOOR THAT TURNS A WRITER INTO A HANDLE.
// ---------------------------------------------------------------------------
// Every caller that wants a cargo handle for a WRITER goes through here, so the writer cannot
// be stated twice and cannot disagree with the path: the path is DERIVED from the writer by
// `recall_cargo_path` above, and the same value is carried into the handle. That is the whole
// of "a writer identity in the key" -- the key is (writer, block index), the writer is the
// file, and the file is a pure function of the writer in ONE place.
//
// `writer` is a BOUNDED slot (the engine uses the lane: `SequenceState::lane`, assigned once at
// program_impl.h:7953), for the reason the path note above states: an unbounded identity would
// mint unboundedly many files.
[[nodiscard]] inline std::unique_ptr<TurnRecallTextCargo>
recall_cargo_open_writer(const std::string& cold_disk_path, std::uint32_t writer,
                         RecallCargoOpen open_mode = RecallCargoOpen::CreateOrTruncate) {
    return std::make_unique<TurnRecallTextCargo>(recall_cargo_path(cold_disk_path, writer),
                                                open_mode, writer);
}

// THE RUN READER. This is the traversal the append leg performs inline today (ascending
// page order, the run CUT at the first coordinate read_block REFUSES) factored out, so that
// the append leg and an original-position REBUILD leg cannot silently diverge on WHICH
// tokens are recalled. "Two paths, one of them dead" is the shape the coordinator forbade;
// "two paths that disagree about a hole" is worse, because both of them run.
//
// WHAT "CUT" MEANS, MEASURED RATHER THAN ASSUMED (locate_test, out of tree): read_block
// refuses exactly when `block_index + 1 > blocks_written_`, i.e. past the file's
// HIGH-WATER MARK. An index BELOW the mark that was never written is NOT refused -- it
// reads back as the 64 ZEROS the writer left at that offset. So a run cuts at the mark, and
// an in-range gap is read as zeros, identically for both legs. This is inherited from
// read_block and is deliberately NOT changed here: making it stricter would change what the
// append leg recalls, and a rebuild that recalls MORE than the append leg is a new bug.
//
// An empty run for a non-empty `pages` is the append leg's existing
// `REASON=no-cargo-record-for-planned-pages` case, not a new error.
[[nodiscard]] inline RecallCargoLocatedRun
recall_cargo_read_run(const TurnRecallTextCargo& cargo, std::span<const std::uint32_t> pages,
                      std::span<const std::uint32_t> page_tokens = {}) {
    RecallCargoLocatedRun run;
    if (!page_tokens.empty() && page_tokens.size() != pages.size()) { return run; }
    run.ids.reserve(pages.size() * static_cast<std::size_t>(kRecallCargoBlockTokens));
    run.positions.reserve(pages.size());
    run.block_tokens.reserve(pages.size());
    for (std::size_t index = 0; index < pages.size(); ++index) {
        const std::uint32_t page = pages[index];
        // The block's REAL length: the caller's directory row when it has one, else the
        // record stride (which reproduces the pre-patch behaviour byte-for-byte).
        const std::uint32_t row_tokens =
            page_tokens.empty() ? kRecallCargoBlockTokens : page_tokens[index];
        RecallCargoLocatedBlock block;
        // The cut: past the high-water mark there is no record AT ALL, so the run ends --
        // the tokens below it are not this prefix's text, and filling them with a
        // neighbour's ids would be an approximation. See the note above for what this does
        // and does not catch.
        if (!cargo.read_located_block(page, row_tokens, block)) { break; }
        if (run.positions.empty()) { run.first_block = page; }
        run.positions.push_back(block.token_begin);
        run.block_tokens.push_back(block.token_count);
        // ⭐ ONLY the real ids. Inserting all kRecallCargoBlockTokens lanes here is the
        // exact shape that would rebuild this run from the writer's zero padding.
        run.ids.insert(run.ids.end(), block.ids, block.ids + block.token_count);
    }
    return run;
}

} // namespace ninfer::spec::turn_recall
