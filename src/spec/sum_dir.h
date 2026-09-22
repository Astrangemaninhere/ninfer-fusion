#pragma once

// src/spec/sum_dir.h -- the model's own DIRECTORY over recall blocks.
//
//   Independent feature. EXPERIMENTAL. OFF BY DEFAULT. Host-only, std-only: no CUDA, no engine
//   headers, so it compiles and unit-tests under plain `g++` (tests/test_sum_dir.cpp).
//
// WHAT THIS IS
//   A per-sequence table whose ROWS ARE THE RECALL BLOCKS. While the engine is idle the model is
//   asked to write one catalogue line per block ("what is in here"), and that line -- not a
//   similarity score over the block's bytes -- becomes the key a later recall is targeted with.
//   The point of a directory is that you look a thing up BY ITS DESCRIPTION, whereas the
//   block-level machinery the tree already has (below) can only answer "where did I put these
//   bytes", never "which bytes are about X".
//
// WHAT ALREADY EXISTS IN THIS TREE (reused, never re-implemented -- read these before editing)
//
//   B1 block granularity   cold_host_tier.h:34-41  one logical Paged-KV page = 64 tokens ACROSS
//                          ALL TEXT LAYERS; a cold-slot sentinel encodes ONE slot base that every
//                          layer's cold_slots is indexed by at once, so a page moves whole.
//                          kColdHostPageTokens (there, :67) == kPagedKVPageSize
//                          (core/paged_kv_cache.h:17) == 64, asserted in both places.
//   B2 block identity      prefix_identity.h:40-57  PrefixShortlistDigests -- one rolling 128-bit
//                          digest per token frontier. It is a CONTENT SHORTLIST, not an identity:
//                          prefix_identity.h:37-39 says so and names the authoritative check (the
//                          engine's own exact comparison, prefix_matches /
//                          ResidentPrefixIdentity). The digest's construction is a 2-lane FNV-1a
//                          whose constants are OWNED BY spec/fnv_convention.h (see the note
//                          below): kDigestOffset (:60), kDigestPrime (:61), mix_digest (:66, the
//                          lane-1 rotation is :69), and a zero lane pinned to 1 on :114 so that
//                          "0" stays "no digest". Cited by SYMBOL: the line numbers here were
//                          re-checked on 2026-09-13, and the ones this comment used to carry
//                          (:61-62, :99, :103-111) pointed at the wrong lines.
//   B3 where blocks live   program.h:506  SequenceState::ColdPageEntry{page, slot,
//                          file_slot}, with the trap written into the comment above it: `slot` is
//                          a DEVICE WORKING-SET INDEX that gets recycled (allocate_cold_slot
//                          hands back the LOWEST free index), so it is a LOCATION, not an
//                          identity; `file_slot` is the stable one and even it only means "where
//                          the bytes were written".
//   B4 the idle hook       serve/kv_auto_relayout.cpp:320-325  the tree's existing "wake every N
//                          seconds and decide" loop: a 250 ms tick, hysteresis over two
//                          consecutive cycles, a semantic comparison so a formatting difference
//                          is not a change. It has exactly one apply call,
//                          GenerationService::reload_kv_storage (generation_service.h:134-139),
//                          whose drain body (generation_service.cpp:518-550) is the tree's
//                          EXISTING quiescence primitive: raise reload_in_progress_, then wait
//                          for request_capacity_->active == 0 (the counter every admitted request
//                          holds until its response is released, generation_service.cpp:21-43 and
//                          :299-312), with new admissions refused under the same lock. The
//                          summarizer is therefore a SECOND CONSUMER OF THAT LOOP, not a new
//                          thread and not a new definition of idle: see sum_dir_idle_eligible()
//                          and SumDirKnobs::from_env().
//
// THE BLOCK LEDGER (formerly cited here as B4) -- NOW INCLUDED, NOT RESTATED
//   src/spec/turn_recall_journal.h (journal + page plan + cost model) is the ledger: one fixed
//   64-byte L0 record per (re)spill or release of ONE logical page, keyed
//   (digest_lo, digest_hi, page), RecallKind::Release tombstones, recall_record_seal() crc32.
//   The three earlier states of this comment (a citation to a file that did not exist, then a
//   denial of a file that did, then "field for field" prose) are all gone, because the types are
//   now SHARED rather than described: this header includes that one and DERIVES from it.
//
//   What is taken from the ledger, by symbol, so a renumbering cannot go unnoticed:
//     * turn_recall::kRecallPageTokens       -> kSumDirBlockTokens, SumDir's own granularity
//     * turn_recall::RecallCodec             -> the values behind SumDirCodec's admitted tiers
//     * turn_recall::recall_codec_admitted() -> sum_dir_codec_admitted()   (the ONE admission rule)
//     * turn_recall::recall_codec_bytes_per_token() -> sum_dir_codec_bytes_per_token()
//     * turn_recall::RecallKind              -> the values behind SumDirState::Live / ::Dead
//     * turn_recall::recall_crc32_raw()      -> the body of sum_dir_crc32()
//   What is NOT taken: the digests. SumDirDigest is this header's own two-lane FNV with its own
//   domain separation (kSumDirBlockDomain / kSumDirSummaryDomain), because a BLOCK digest and the
//   engine's rolling PREFIX digest are different objects with different inputs -- unifying them
//   would be the third spelling, not the removal of one.
//
//   What that means for this header, stated plainly:
//     * `sum_dir_row_bound()` below is still ONLY a row check: it checks what its caller hands
//       it. If a caller passes `ledger_live = true` without a ledger behind it, this header
//       cannot tell -- it verifies the row, not the world. Sharing the codec set does not change
//       that, and the parameters stay named `ledger_*` for the contract they name.
//     * The rules that USED to be cited from the ledger and are still load-bearing remain stated
//       here in full (the identity is content-only; a page number is part of the binding; a
//       recycled slot is Dead and not a hit), so a reader of this file still needs no other file
//       to follow it.
//     * The two numbers that motivated the refusal -- the rk4v4 exact-prefix accuracy
//       (9/448 = 0.020) against nvfp4 0.949 and int8 0.977 -- are RETAINED but their provenance
//       is NOT in this tree. They are not an inference here: they are the reason the refusal is
//       hard-coded in the ledger, and this header now inherits that decision instead of
//       re-deciding it (see SumDirCodec).
//
// THE ONE RULE THIS HEADER EXISTS TO PROTECT
//   IDENTITY IS CONTENT, NOT POSITION. A recalled block is rebuilt at a NEW position, so nothing
//   positional may enter `block_identity`: not the block index, not the absolute token offset,
//   not the sequence, not the device cold slot, not the spill file slot, not the wall time at
//   which the catalogue line was written. All of those DO appear on the row -- they are how the
//   row is found again and checked -- and each is marked as bookkeeping. The self-test PROVES the
//   position independence (the same content at two offsets has equal identity; one token anywhere
//   in the block changed has a different identity) instead of asserting it in a comment.
//
// ETIQUETTE TOWARD 95-3 (injected recall must look to the model like ordinary context)
//   Nothing in this file is ever rendered into a prompt. A row is host-side bookkeeping, so the
//   rule does not bind this header; it binds the future patch that turns a row back into tokens,
//   which is listed as a blocker in the report rather than assumed away here.

// The FNV conventions this file's digests are built on. It is std-only (like this
// file), so including it does not cost the plain-g++ property; the alternative --
// re-spelling the four constants here -- is what let the engine's offset and the
// published one drift apart as two unrelated literals in the first place.
#include "spec/fnv_convention.h"

// The block ledger that OWNS the codec set, the 64-byte record shape, the crc32 and the
// spill/release lifecycle. Same discipline as fnv_convention.h above, one file over: this
// header USES those types now instead of restating them, because the restatement had already
// drifted (see the codec section below). It is host-only and std-only too, so including it
// does not cost the plain-g++ property either.
#include "spec/turn_recall_journal.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ninfer::spec::sum_dir {

// ---------------------------------------------------------------------------
// granularity
// ---------------------------------------------------------------------------

// Must equal kPagedKVPageSize (core/paged_kv_cache.h:17) and kColdHostPageTokens
// (cold_host_tier.h:67). The value now comes FROM the ledger that owns the block-keyed
// record, so a change has one place to be made instead of two; turn_recall::kRecallPageTokens
// states the rule ("one logical page = 64 tokens x ALL text layers").
// tests/test_sum_dir.cpp still pins the resolved value to 64 and names both engine lines.
inline constexpr std::uint32_t kSumDirBlockTokens = turn_recall::kRecallPageTokens;
static_assert(kSumDirBlockTokens == 64U,
              "the directory's block IS the engine's 64-token Paged-KV page");

// "this row has no catalogue line yet" -- a normal state, since rows exist as soon as a block
// spills while the catalogue is written asynchronously afterwards.
inline constexpr std::uint32_t kSumDirNoSummary = 0xFFFFFFFFU;
// The "no row covers this block" reading of the block->row index below. It is the same value as
// `kSumDirNoSummary` because both mean "the column has no entry here", and they are kept as two
// names so a future change to either cannot silently move the other.
inline constexpr std::uint32_t kSumDirNoRow = 0xFFFFFFFFU;

// ---------------------------------------------------------------------------
// identity -- two-lane FNV-1a over the block's CONTENT ONLY
// ---------------------------------------------------------------------------

// These numbers are THE ENGINE'S, and they now come from the one header that owns both FNV
// conventions in this tree: spec/fnv_convention.h. That header states how the engine's offset
// differs from the published FNV-1a 64 basis (one decimal digit), names every artefact that
// depends on each convention, and carries the static_asserts that stop a third convention -- or a
// silent "tidy-up" of this pair into the published one -- from being introduced. The lane-1
// rotation is the engine's (prefix_identity.cpp:103-111: `rotl(value ^ 0x9e3779b97f4a7c15, 29)`),
// so feeding this function the same block contents the engine's own mix sees yields the same 128
// bits, which is what makes the binding to B2 checkable rather than hopeful.
inline constexpr std::uint64_t kSumDirDigestOffset0  = fnv::kEngineDigestLane0Offset;
inline constexpr std::uint64_t kSumDirDigestOffset1  = fnv::kEngineDigestLane1Offset;
inline constexpr std::uint64_t kSumDirDigestPrime0   = fnv::kEngineDigestPrime0;
inline constexpr std::uint64_t kSumDirDigestPrime1   = fnv::kEngineDigestPrime1;
inline constexpr std::uint64_t kSumDirLaneSkew       = fnv::kEngineDigestLaneSkew;
inline constexpr std::uint32_t kSumDirLaneSkewRotate = fnv::kEngineDigestLaneRotate;

// Domain separation, so a block digest can never be confused with a whole-prefix digest: the
// engine's token domain is 0x6e696e6665722d74 ("ninfer-t", prefix_identity.cpp:99) and only its
// low byte changes here.
inline constexpr std::uint64_t kSumDirBlockDomain   = 0x6e696e6665722d62ULL; // "ninfer-b"
inline constexpr std::uint64_t kSumDirSummaryDomain = 0x6e696e6665722d73ULL; // "ninfer-s"

struct SumDirDigest {
    std::uint64_t lo = kSumDirDigestOffset0;
    std::uint64_t hi = kSumDirDigestOffset1;

    [[nodiscard]] friend bool operator==(const SumDirDigest& a, const SumDirDigest& b) noexcept {
        return a.lo == b.lo && a.hi == b.hi;
    }
    [[nodiscard]] friend bool operator!=(const SumDirDigest& a, const SumDirDigest& b) noexcept {
        return !(a == b);
    }
    // Total order so rows can be sorted and binary-searched. Only equality is semantic; this is a
    // stable arrangement and nothing more.
    [[nodiscard]] friend bool operator<(const SumDirDigest& a, const SumDirDigest& b) noexcept {
        return a.lo != b.lo ? a.lo < b.lo : a.hi < b.hi;
    }
    // The pristine value, i.e. "no digest". Unreachable for a real block (the zero-lane pin below
    // makes it so), which is exactly why every caller may use it as the invalid value.
    [[nodiscard]] constexpr bool is_unset() const noexcept {
        return lo == kSumDirDigestOffset0 && hi == kSumDirDigestOffset1;
    }
};

inline void sum_dir_mix(SumDirDigest& digest, std::uint64_t value) noexcept {
    const std::uint64_t skewed = value ^ kSumDirLaneSkew;
    const std::uint64_t lane1 =
        skewed << kSumDirLaneSkewRotate | skewed >> (64U - kSumDirLaneSkewRotate);
    digest.lo ^= value;
    digest.lo *= kSumDirDigestPrime0;
    digest.hi ^= lane1;
    digest.hi *= kSumDirDigestPrime1;
}

// THE identity of one block: fed the block's token ids, in order, and their count, and nothing
// else. Deliberately absent -- block index, absolute token offset, sequence tag, cold slot, file
// slot, catalogue text, wall clock. Each of those changes when the same content is recalled into a
// new place, so each would make two copies of one block compare unequal, which is the exact
// failure this function exists to make impossible.
[[nodiscard]] inline SumDirDigest sum_dir_block_digest(const std::uint32_t* tokens,
                                                       std::size_t count) noexcept {
    SumDirDigest digest;
    sum_dir_mix(digest, kSumDirBlockDomain);
    sum_dir_mix(digest, static_cast<std::uint64_t>(count));
    for (std::size_t i = 0; i < count; ++i) {
        sum_dir_mix(digest, static_cast<std::uint64_t>(tokens[i]));
    }
    // prefix_identity.cpp:113-115 pins a zero lane to 1 for the same reason: 0 would otherwise be
    // reachable and every caller would have to special-case it.
    if (digest.lo == 0) { digest.lo = 1; }
    if (digest.hi == 0) { digest.hi = 1; }
    return digest;
}

[[nodiscard]] inline SumDirDigest
sum_dir_block_digest(const std::vector<std::uint32_t>& tokens) noexcept {
    return sum_dir_block_digest(tokens.data(), tokens.size());
}

// The catalogue line's own digest: what a later query is compared against. Same mix, different
// domain, so a query can never be mistaken for a block. NOT an identity of the block -- two
// summaries of one block are two values, which is correct: a description may be rewritten at any
// time, and rewriting it must not disturb a single identity.
[[nodiscard]] inline SumDirDigest sum_dir_text_digest(std::string_view text) noexcept {
    SumDirDigest digest;
    sum_dir_mix(digest, kSumDirSummaryDomain);
    sum_dir_mix(digest, static_cast<std::uint64_t>(text.size()));
    for (const char c : text) {
        sum_dir_mix(digest, static_cast<std::uint64_t>(static_cast<std::uint8_t>(c)));
    }
    if (digest.lo == 0) { digest.lo = 1; }
    if (digest.hi == 0) { digest.hi = 1; }
    return digest;
}

// ---------------------------------------------------------------------------
// codec -- DERIVED from the record that owns the codec set, never restated
// ---------------------------------------------------------------------------
//
// turn_recall::RecallCodec (spec/turn_recall_journal.h) is THE spelling of "which KV tiers may
// back an external recall record". This header used to keep a second copy of it, and the copy
// had already drifted where it is least visible: the ledger numbers Rejected 0 and has no
// Unset at all, this header numbered Unset 0 and Rejected 5, while the comment above it
// claimed "values kept". The admitted set and the two byte rates are CALLS now, so there is
// nothing left to drift; sum_dir_codec_to_recall() is the total, asserted mapping they all go
// through.
//
// The refusal itself is not taste: rk4v4 has no external recall tier behind it, and a page whose
// layers disagree has no single stride for one row to describe. The accuracy figures that
// motivated the first half (rk4v4 9/448 = 0.020 exact-prefix against nvfp4 0.949 and int8 0.977)
// were measured in an earlier slice whose record is NOT in this repository; they are kept as
// the stated reason for a decision that is hard-coded in the ledger, not as a citation a
// reader here can open.
//
// Two enumerators survive on purpose, and neither is a second authoritative spelling:
//   * `Unset` (0)    -- sum_dir-only. A row that has not yet been told its block's codec. The
//                       ledger never needs it: a record is only ever written for a page whose
//                       bytes exist, so "no codec" is refused there under the name Rejected.
//   * `Rejected` (5) -- this header's WIRE value for a refused codec, kept so the row format
//                       does not move. The ledger spells the same state 0. The mapping below
//                       is what ties them together, and the static_asserts fail the BUILD (not
//                       a test run) if either side is renumbered.
enum class SumDirCodec : std::uint8_t {
    Unset    = 0, // sum_dir-only: a row may not describe a codec it does not know
    Nvfp4    = static_cast<std::uint8_t>(turn_recall::RecallCodec::Nvfp4),
    Int8     = static_cast<std::uint8_t>(turn_recall::RecallCodec::Int8),
    Mixed    = static_cast<std::uint8_t>(turn_recall::RecallCodec::Mixed),
    Rejected = 5, // sum_dir wire value; turn_recall::RecallCodec::Rejected is 0
};

// The bridge. `Unset` maps to the ledger's Rejected because "no codec is known" is refused for
// exactly the reason "this codec is not admitted" is: there is no external tier either way.
[[nodiscard]] constexpr turn_recall::RecallCodec
sum_dir_codec_to_recall(SumDirCodec codec) noexcept {
    switch (codec) {
    case SumDirCodec::Nvfp4: return turn_recall::RecallCodec::Nvfp4;
    case SumDirCodec::Int8: return turn_recall::RecallCodec::Int8;
    case SumDirCodec::Mixed: return turn_recall::RecallCodec::Mixed;
    case SumDirCodec::Rejected:
    case SumDirCodec::Unset: break;
    }
    return turn_recall::RecallCodec::Rejected;
}

static_assert(static_cast<std::uint8_t>(SumDirCodec::Nvfp4) ==
                  static_cast<std::uint8_t>(turn_recall::RecallCodec::Nvfp4),
              "sum_dir's nvfp4 tier must BE the ledger's");
static_assert(static_cast<std::uint8_t>(SumDirCodec::Int8) ==
                  static_cast<std::uint8_t>(turn_recall::RecallCodec::Int8),
              "sum_dir's int8 tier must BE the ledger's");
static_assert(static_cast<std::uint8_t>(SumDirCodec::Mixed) ==
                  static_cast<std::uint8_t>(turn_recall::RecallCodec::Mixed),
              "sum_dir's mixed marker must BE the ledger's");
static_assert(sum_dir_codec_to_recall(SumDirCodec::Rejected) ==
                  turn_recall::RecallCodec::Rejected,
              "a refused codec is the ledger's refused codec");
static_assert(sum_dir_codec_to_recall(SumDirCodec::Unset) == turn_recall::RecallCodec::Rejected,
              "an unset codec is refused, never admitted");
static_assert(kSumDirBlockTokens == turn_recall::kRecallPageTokens,
              "the directory's block is the ledger's page");

[[nodiscard]] constexpr bool sum_dir_codec_admitted(SumDirCodec codec) noexcept {
    return turn_recall::recall_codec_admitted(sum_dir_codec_to_recall(codec));
}

[[nodiscard]] inline const char* sum_dir_codec_name(SumDirCodec codec) noexcept {
    // "unset" stays local: the ledger has no such state and must not grow one.
    if (codec == SumDirCodec::Unset) { return "unset"; }
    return turn_recall::recall_codec_name(sum_dir_codec_to_recall(codec));
}

// The byte rates are the ledger's constants, so the two 4.50/8.25 b/el numbers exist once.
[[nodiscard]] constexpr std::uint32_t sum_dir_codec_bytes_per_token(SumDirCodec codec) noexcept {
    return turn_recall::recall_codec_bytes_per_token(sum_dir_codec_to_recall(codec));
}

static_assert(!sum_dir_codec_admitted(SumDirCodec::Rejected),
              "rk4v4/bf16/fp8/iso4e have no external recall tier");
// The row's codec byte is one byte per ROW for a row that may span two strides, and that is
// fine for exactly the reason the ledger gives: the cold IO legs are per-layer, so one row
// and one file_slot address every layer at that layer's own extent. This used to read
// `!sum_dir_codec_admitted(SumDirCodec::Mixed)` with the reason "a page whose layers
// disagree has no single stride" -- a requirement no consumer of the row has. It is
// ADMITTED now (see turn_recall::recall_codec_admitted, the ONE rule this delegates to).
static_assert(sum_dir_codec_admitted(SumDirCodec::Mixed),
              "a mixed-codec row is admitted: sum_dir's refusal was a delegation to the "
              "ledger (sum_dir.h:313-315), so the ledger's 2026-09-18 admission moved this "
              "line with it; a row with two strides needs one file_slot, not two");
static_assert(!sum_dir_codec_admitted(SumDirCodec::Unset),
              "a row may not describe a codec it does not know");
static_assert(sum_dir_codec_admitted(SumDirCodec::Nvfp4) &&
                  sum_dir_codec_admitted(SumDirCodec::Int8),
              "nvfp4 and int8 are the two admitted tiers");
// The admitted tiers' VALUES are wire-visible: SumDirRow serialises the codec as ONE BYTE
// (sum_dir_pack_row -> static_cast<std::uint8_t>(row.codec), the loader reads it back at
// offset 64), so 2/3/4 are not an implementation detail. Pinned here so that renumbering the
// ledger -- whose record has no codec byte of its own -- fails the build instead of silently
// reinterpreting every row an older build wrote.
static_assert(static_cast<std::uint8_t>(SumDirCodec::Nvfp4) == 2U &&
                  static_cast<std::uint8_t>(SumDirCodec::Int8) == 3U &&
                  static_cast<std::uint8_t>(SumDirCodec::Mixed) == 4U &&
                  static_cast<std::uint8_t>(SumDirCodec::Rejected) == 5U,
              "the row's codec byte keeps the values 2/3/4/5 it was written with");
// The byte rates are pinned BOTH ways on purpose: to their literals (the numbers this row's
// cost model was written against) and to the ledger's constants (the single place they are
// now stated). Pinning only the second would be tautological -- it would still hold if the
// ledger itself moved -- and pinning only the first is the duplication being removed.
static_assert(sum_dir_codec_bytes_per_token(SumDirCodec::Nvfp4) == 18432U,
              "4.50 b/el is the nvfp4 rate the row cost model was written against");
static_assert(sum_dir_codec_bytes_per_token(SumDirCodec::Int8) == 33792U,
              "8.25 b/el is the int8 rate the row cost model was written against");
static_assert(sum_dir_codec_bytes_per_token(SumDirCodec::Nvfp4) ==
                  turn_recall::kRecallNvfp4BytesPerToken,
              "one byte rate, one spelling");
static_assert(sum_dir_codec_bytes_per_token(SumDirCodec::Int8) ==
                  turn_recall::kRecallInt8BytesPerToken,
              "one byte rate, one spelling");
static_assert(sum_dir_codec_bytes_per_token(SumDirCodec::Nvfp4) <
                  sum_dir_codec_bytes_per_token(SumDirCodec::Int8),
              "nvfp4 is the smaller tier");

// ---------------------------------------------------------------------------
// lifecycle -- aligned with the block ledger, never with a slot index
// ---------------------------------------------------------------------------
//
// Exactly the two things that happen to a block's bytes (B3): they were written to a spill
// region (Live), and they are not there any more -- restored hot, sequence gone, or the file slot
// recycled into another page (Dead). There is deliberately no third state and no place to park a
// slot number: a recycled slot is simply Dead, and Dead is what stops a lookup from treating the
// row as a hit. `generation` carries the sequence tag -- and THIS HEADER IS THE ONLY THING THAT
// DEFINES THAT TAG: the engine's own tag IS `recall_sequence_tag` (program.h:535), and it is
// really read: `record.sequence = sequence.recall_sequence_tag;` in
// impl/runtime/program_impl.h. But this header's `sequence_tag` is a caller-supplied session id
// and is NOT bound to that engine value -- binding the two is a prerequisite for landing this
// feature, not a citation. Until then a page number a later session reuses CAN, in principle, be
// read as this one's row, and nothing in this header can detect that.
//
// WHY THE OPPOSITE SENTENCE STOOD HERE UNTIL 2026-09-14 (kept as a date, so the next reader can
// tell drift from error): it used to read "there is no recall_sequence_tag anywhere in this tree",
// naming `program.h:536-538` as a citation that pointed at a blank line. That was TRUE OF THE TREE
// IT WAS WRITTEN AGAINST. At commit 3944a53 -- the base of this working tree -- the engine had no
// such symbol: `git show HEAD:src/targets/qwen3_6/impl/runtime/program.h | grep
// recall_sequence_tag` is EMPTY, and HEAD's program.h:536-538 is three unrelated declarations. The
// sentence became false when the field landed in the working tree (program.h mtime 08:55:24).
//
// AND THE HALF THAT MATTERS FOR THIS HEADER: that field is DECLARED BUT NEVER WIRED. The two lines
// named above are its ONLY occurrences in the whole tree, and nothing anywhere assigns it a value
// other than its own initialiser, so every record a run writes carries sequence = 0. A tag that is
// always 0 cannot distinguish one session from another, so binding this header's `sequence_tag` to
// it would buy nothing today: the identity a persisted row can carry is still the caller's. That
// is why the binding is a prerequisite and not a one-line change.
// The two states ARE the ledger's two record kinds, and they are numbered FROM it here
// instead of being numbered independently. `Live` is a standing RecallKind::Spill record,
// `Dead` is a RecallKind::Release tombstone -- the static_asserts below make that a build
// failure to change, not a comment to forget.
enum class SumDirState : std::uint8_t {
    Live = static_cast<std::uint8_t>(turn_recall::RecallKind::Spill),   // 1
    Dead = static_cast<std::uint8_t>(turn_recall::RecallKind::Release), // 2
};

[[nodiscard]] constexpr turn_recall::RecallKind
sum_dir_state_to_recall_kind(SumDirState state) noexcept {
    return state == SumDirState::Live ? turn_recall::RecallKind::Spill
                                      : turn_recall::RecallKind::Release;
}

static_assert(static_cast<std::uint8_t>(SumDirState::Live) == 1U &&
                  static_cast<std::uint8_t>(SumDirState::Dead) == 2U,
              "the persisted row state must keep the values 1 and 2");
static_assert(sum_dir_state_to_recall_kind(SumDirState::Live) == turn_recall::RecallKind::Spill &&
                  sum_dir_state_to_recall_kind(SumDirState::Dead) ==
                      turn_recall::RecallKind::Release,
              "live is a standing Spill record; dead is a Release tombstone");

[[nodiscard]] inline const char* sum_dir_state_name(SumDirState state) noexcept {
    return state == SumDirState::Live ? "live" : "dead";
}

// ---------------------------------------------------------------------------
// the row
// ---------------------------------------------------------------------------
//
// The row IS the design: it binds a BLOCK IDENTITY (B2, position-free) to a CATALOGUE LINE (the
// model's summary), carrying just enough bookkeeping to find the block again and to check that it
// is still there.
//
// THE CATALOGUE INVARIANT, stated once because the whole class depends on it:
//
//     summaries_.size() == rows_.size() always, and row k's line is summaries_[k].
//
// Keeping the column in ROW LOCKSTEP -- one slot per row, empty until written, never reordered --
// is what makes the format rejection-safe: a row that a loader cannot accept is simply dropped,
// and every other row's line stays attached to its own row. (The first version indexed the column
// by iteration order instead, which silently re-pointed every row after a dropped one; section 5
// of the self-test is what caught it, and the invariant above is what makes it unrepeatable.)
struct SumDirRow {
    // -- identity ------------------------------------------------------- position-free
    SumDirDigest block_identity{};
    // -- where the block sat in the sequence that was summarized ---------- BOOKKEEPING: this is
    //    the caller's token space and is NOT part of the identity. Its job is to let a retrieval
    //    answer be reported as a token range, and to let a caller sanity-check a candidate against
    //    the sequence it came from.
    std::uint32_t token_begin = 0;
    std::uint32_t token_end   = 0; // exclusive
    std::uint32_t page        = 0; // logical page == token_begin / kSumDirBlockTokens
    std::uint32_t generation  = 0; // sequence tag: this header's own concept, not the engine's
    // -- the block's own contents -------------------------------------- kept so a query can be run
    //    against the block itself and not only against its catalogue line
    std::uint32_t content_first = 0;
    std::uint32_t content_count = 0;
    // -- the catalogue slot -------------------------------------------- always the row's own index;
    //    NO_SUMMARY marks a row whose line has not been written yet
    std::uint32_t summary_index = kSumDirNoSummary;
    std::uint32_t summary_count = 0; // bytes of that line; 0 = nothing to read
    SumDirDigest  summary_digest{};  // sum_dir_text_digest of those bytes
    // -- the block's bytes --------------------------------------------- HINTS, not identity
    SumDirCodec  codec     = SumDirCodec::Unset;
    SumDirState  state     = SumDirState::Live;
    std::int32_t file_slot = -1; // -1 = no on-disk region. A LOCATION: two rows differing only
                                 // here describe the SAME block, which is why this is outside
                                 // block_identity.
    // -- the block's FRAME AXIS --------------------------------------------- the r side that was
    //    THROWN AWAY. Bit j == 1  <=>  token offset j of this block is a TEMPLATE position or an
    //    injected ADDED/CONTROL token, rather than bytes the CALLER supplied.
    //
    //    The producer has always existed: `chat_template.cpp:68 append_template` inserts template
    //    bytes and records NOTHING, while `:70-74 append_literal` records a byte span -- so the
    //    template's extent is the complement of `RenderedChat::literal_spans` (chat_template.h:120).
    //    And `Tokenizer::encode_with_boundaries` (tokenizer.cpp:796-897) knows the class at the
    //    instant it pushes each id: the `:873 append_ordinary(...)` call pushes ordinary-text ids and
    //    `:883 encoded.input_ids.push_back(match_token->id)` pushes an added/special token's id. It
    //    was lost because not one of the three carrier structs had a field for it
    //    (tokenizer.h:128-131, processor.h:119-134, processor.h:101-117), and the ONE per-token byte
    //    column that does exist -- `ProcessedInput::token_types` (processor.h:103) -- is the VISION
    //    MODALITY column and cannot be borrowed: non-zero there means "a Vision chunk is owed" and
    //    the prefill path THROWS when it is missing (product/kv_recall_block.h:352-360, verified).
    //    See `spec/frame_axis.h` for the column, the bitmap algebra, and how it is recovered.
    //
    //    ZERO IS A MEANINGFUL VALUE, not a sentinel: it says "no template/control position in this
    //    block", which is exactly what a writer predating this field produced. An old file
    //    deserializes with `frame_bits == 0` and means what it always meant, and the payload digest
    //    still matches because `serialize` zero-initializes the whole stride
    //    (`std::uint8_t wire[kSumDirRowWireBytes] = {}`) and digests all 80 bytes of it.
    std::uint64_t frame_bits = 0;
};

// 16 + 16 + 8 + 12 + 12 + 8 = 76 payload bytes, padded to one fixed stride so a packed reader and
// this struct see the same layout (the same explicitness the absent block ledger's specification
// used for its own 64-byte record: fixed stride, every byte accounted for).
//
// `frame_bits` (below) is the 8 bytes at offset 70, and it is why the padding comment above is no
// longer the whole story. THE STRIDE ITSELF DOES NOT MOVE: `pack_row` used to write its last byte at
// offset 70 and `unpack_row` used to read its last byte at offset 70, so offsets 70..80 of this
// 80-byte stride were neither written nor read. Writing one `std::uint64_t` there lands the struct
// at exactly 80 -- `sizeof(SumDirRow) == 80` -- so the assertion below (80 <= 80) still holds and
// `kSumDirRowWireBytes` is UNCHANGED. That is the whole reason this column is affordable:
// NO WIRE CHANGE, no new version, and a file written before this field reads back with
// `frame_bits == 0`, which is exactly the semantics it always had.
inline constexpr std::uint32_t kSumDirRowWireBytes = 80U;
static_assert(sizeof(SumDirRow) <= kSumDirRowWireBytes,
              "the row must fit the one stride the format advertises");

[[nodiscard]] inline std::uint32_t sum_dir_row_tokens(const SumDirRow& row) noexcept {
    return row.token_end > row.token_begin ? row.token_end - row.token_begin : 0U;
}

[[nodiscard]] inline bool sum_dir_row_has_summary(const SumDirRow& row) noexcept {
    return row.summary_index != kSumDirNoSummary && row.summary_count != 0;
}

// ---------------------------------------------------------------------------
// THE ONE RENDERER OF A BLOCK'S TOKEN STREAM INTO THE CATALOGUE LINE'S BYTE SPACE
// ---------------------------------------------------------------------------
//
// `QSummarySubstring` asks "which blocks' catalogue lines contain byte string w?" and names
// `search_summaries()` as its answerer, with the parenthesis "byte substring; no tokenizer by
// design" (spec/sum_dir_query_registry.h:103). That parenthesis is the whole specification of this
// function: the engine's inference runtime has NO tokenizer in it (nothing under
// impl/runtime/ includes frontend/tokenizer.h), so the only text a producer can honestly write
// from a block is a RENDERING of the block's own ids -- and the rendering that is lossless,
// tokenizer-free and unambiguous under substring search is the space-delimited decimal id list,
// DELIMITED ON BOTH SIDES.
//
// The leading and trailing spaces are load-bearing, not cosmetic. Without them a query for the
// two-token phrase `12 34` would match inside the rendering of the ONE token `1234`, so a
// substring hit would not be an n-gram hit. With them, a query rendered by this same function can
// only match at token boundaries, and `search_summaries()` becomes an exact n-gram search over the
// token stream -- which is the "lexical (n-gram / BM25)" retriever this tree's registry names,
// with the substring scan standing in for the BM25 scores it does not yet compute.
//
// WHY BOTH SIDES MUST CALL *THIS* FUNCTION. The producer is `ProgramImplCore`'s retire path (where
// a block is stored) and the consumer is the same engine's recall provider (where a query is
// formed from the sequence's own newest tokens). They are two call sites in one file, and if each
// spelled the line for itself they would be free to drift -- a producer writing "12 34 " and a
// consumer searching "12  34" silently returns nothing, and the failure looks like "the index has
// nothing", which is the exact misreading this whole patch exists to end. One function, two
// callers, so a drift is not expressible.
//
// COST, STATED: a 64-token block renders to ~700 B, not to `SumDirKnobs::max_summary_bytes`'s
// 160 B. That 160 B knob is dead code in this tree (its only reader, `sum_dir_idle_eligible`, has
// zero callers -- see the inventory), and this patch deliberately does NOT adopt it: capping a
// 64-token block's catalogue line at 160 B would make only the first ~14 tokens of each block
// searchable, i.e. it would silently drop detail, which is the one thing the requirement forbids
// (「不能丢任何细节」). The searchable unit is therefore the WHOLE block, and the bound that is
// enforced is the number of blocks one pass may return (see `RecallIndexPolicy` in
// impl/runtime/program_impl.h), not a byte cap on the text.
[[nodiscard]] inline std::string sum_dir_render_token_line(const std::uint32_t* tokens,
                                                          std::size_t count) {
    std::string line;
    if (tokens == nullptr || count == 0) { return line; }
    // 11 bytes per token = the widest decimal uint32 (10 digits) plus its separator, plus the
    // trailing space, so the reserve is exact rather than a guess.
    line.reserve(count * 11U + 1U);
    line.push_back(' ');
    for (std::size_t index = 0; index < count; ++index) {
        const std::uint32_t value = tokens[index];
        char digits[10] = {};
        std::size_t used = 0;
        std::uint32_t rest = value;
        do {
            digits[used++] = static_cast<char>('0' + static_cast<int>(rest % 10U));
            rest /= 10U;
        } while (rest != 0U);
        while (used != 0U) { line.push_back(digits[--used]); }
        line.push_back(' ');
    }
    return line;
}

// The inverse direction, for a caller that holds a query as TEXT rather than as ids: render an
// already-decoded id list. It exists so a shell/python harness can build the SAME term the
// producer wrote, without linking this header.
[[nodiscard]] inline std::string sum_dir_render_token_line(std::span<const std::uint32_t> tokens) {
    return sum_dir_render_token_line(tokens.data(), tokens.size());
}

// One function, so an appender and a loader cannot disagree about what "well-formed" is.
[[nodiscard]] inline bool sum_dir_row_well_formed(const SumDirRow& row) noexcept {
    if (row.block_identity.is_unset()) { return false; }
    if (row.block_identity.lo == 0 || row.block_identity.hi == 0) { return false; }
    if (!sum_dir_codec_admitted(row.codec)) { return false; }
    if (row.state != SumDirState::Live && row.state != SumDirState::Dead) { return false; }
    if (row.token_end < row.token_begin) { return false; }
    if (row.file_slot < -1) { return false; }
    // A Live row must point at bytes. A Dead row may still carry a stale slot number -- that is
    // exactly the stale hint the ledger binding has to catch -- so it is not rejected here; what a
    // Dead row may not do is claim to be Live.
    if (row.state == SumDirState::Live && row.file_slot < 0) { return false; }
    // A catalogue line is all-or-nothing: an index without a length (or the reverse) is a
    // half-written row, and a half-written row must never look like a hit.
    const bool has_index = row.summary_index != kSumDirNoSummary;
    const bool has_count = row.summary_count != 0;
    if (has_index != has_count) { return false; }
    if (has_count && row.summary_digest.is_unset()) { return false; }
    return true;
}

// The binding to the block ledger, expressed as a predicate rather than as a copy of the journal.
// THE LEDGER ITSELF IS NOT IN THIS TREE (see the note at the top of this file): the caller hands
// in what a ledger WOULD say about this block and this page, and this decides whether the row may
// be offered to recall. That is deliberately all this can do -- it checks the row, and it cannot
// check the caller. A caller that passes `ledger_live = true` with no ledger behind it is making
// an unchecked claim, which is exactly why the rules are restated here instead of cited:
//
//   * the row must be well-formed and Live;
//   * a live spill record must stand for the SAME (identity, page) -- a recycled or tombstoned
//     region is not a hit.
//
// The caller's own recomputation of the block digest from the tokens the ledger points at must
// equal the row's identity; that argument is passed IN as `ledger_identity` rather than re-derived
// here precisely so this header need not carry the block bytes. It is NOT the authoritative
// comparison the tree requires before reuse -- that remains prefix_matches /
// ResidentPrefixIdentity (prefix_identity.h:37-39). True here means "this row may be offered",
// never "reuse these bytes".
[[nodiscard]] inline bool sum_dir_row_bound(const SumDirRow& row,
                                            const SumDirDigest& ledger_identity,
                                            std::uint32_t ledger_page,
                                            bool ledger_live) noexcept {
    if (!sum_dir_row_well_formed(row)) { return false; }
    if (row.state != SumDirState::Live) { return false; }
    if (!ledger_live) { return false; }
    if (!(ledger_identity == row.block_identity)) { return false; }
    return ledger_page == row.page;
}

// ---------------------------------------------------------------------------
// the idle hook -- default OFF
// ---------------------------------------------------------------------------
//
// Cadence knobs only. The summarizer does not invent a sleep of its own: it is a second consumer
// of the loop that already exists (B4, kv_auto_relayout.cpp:320-325) and inherits that loop's
// tick. from_env() reads the same style of variable this feature family already uses (NINFER_FT_*,
// ops/common/ft_stats.h:5-6 -- the one kv_auto_relayout.cpp:3 includes, and NOT the same-named
// impl/runtime/ft_stats.h -- and kv_auto_relayout.cpp:151-166) under one new prefix, so it stays
// separable -- and the default is OFF, which is the rule here for everything shipped
// experimentally.
// ---------------------------------------------------------------------------
// THE STRATEGY SURFACE -- "can this page's plane be dropped?", as ONE selectable policy
// ---------------------------------------------------------------------------
//
// The question "can a plane be dropped" has no PER-LAYER answer in this engine and cannot have one:
// a page is ONE residency bit across ALL text layers, one `file_slot` mirrors every layer's bytes,
// and a page whose layers disagree has no single stride. All of that is argued at length in the
// unload-text section below, with the four tree anchors.
//
// What IS real, and what this enum makes selectable, is the PAGE's byte axis: whether a Spill record
// still stands for the page. `row.state` is that fact -- `Live` is `RecallKind::Spill`, `Dead` is
// `RecallKind::Release` -- and `append()` has derived it from `file_slot` since the row existed.
// So the replacement for the deleted `plane_dropped` is a PAGE property the row already carries, and
// the verdict needs NO new engine hook.
//
// EVERY VALUE BELOW IS A COMPLETE ANSWER, not one knob of several: each one says what an absent byte
// axis means, end to end. That is deliberate -- a strategy that had to be assembled from three
// orthogonal flags would have no single place to be reviewed, logged or defaulted.
enum class SumDirByteAxisPolicy : std::uint8_t {
    // DEFAULT. A page with no byte axis is NOT OFFERED.
    //
    // This is the tree's answer TODAY and not a new opinion: `sum_dir_row_bound()` refuses a row
    // whose `state != Live`, and `search_summaries()` skips it. The default is therefore the
    // compatibility setting -- landing this patch changes no observable behaviour until a caller
    // asks for something else -- and the self-test pins the agreement with `sum_dir_row_bound()`
    // itself rather than with a restatement of it.
    RefuseWhenAbsent = 0,
    // OPT-IN. A page with no byte axis is OFFERED as `AdmissibleTextOnly`: still unloadable, but the
    // unload text holding it loses `byte_recallable` and the price is counted in `blocks_text_only`
    // and `unload_texts_text_only`. This is the behaviour the design argues for, and it is opt-in
    // because turning it on WITHOUT also splitting the two lookup gates above would make the OFFER
    // and the BINDING disagree about the same row. That split is a separate change; this patch
    // supplies the policy it will select.
    PriceWhenAbsent = 1,
    // OPT-IN. The byte axis is never consulted: every block with content is `Admissible`, so the
    // price reads 0. This is the pre-question behaviour -- the one the deleted `plane_dropped` field
    // silently encoded, since no caller ever set it -- kept as a strategy so that the difference it
    // makes is MEASURED by the self-test instead of argued about in a comment.
    IgnoreByteAxis = 2,
};

inline constexpr SumDirByteAxisPolicy kSumDirByteAxisPolicyDefault =
    SumDirByteAxisPolicy::RefuseWhenAbsent;

[[nodiscard]] inline const char* sum_dir_byte_axis_policy_name(SumDirByteAxisPolicy policy) noexcept {
    switch (policy) {
    case SumDirByteAxisPolicy::RefuseWhenAbsent: return "refuse";
    case SumDirByteAxisPolicy::PriceWhenAbsent: return "price";
    case SumDirByteAxisPolicy::IgnoreByteAxis: return "ignore";
    }
    return "unknown";
}

// The flag's parser. An unknown value is NEVER silently accepted: it keeps the default and reports
// `recognised = false`, so a typo is a visible message rather than an invisibly different strategy.
[[nodiscard]] inline SumDirByteAxisPolicy
sum_dir_byte_axis_policy_parse(const char* text, bool& recognised) noexcept {
    recognised = true;
    if (text == nullptr) { recognised = false; return kSumDirByteAxisPolicyDefault; }
    const std::string_view value(text);
    if (value == "refuse") { return SumDirByteAxisPolicy::RefuseWhenAbsent; }
    if (value == "price") { return SumDirByteAxisPolicy::PriceWhenAbsent; }
    if (value == "ignore") { return SumDirByteAxisPolicy::IgnoreByteAxis; }
    recognised = false;
    return kSumDirByteAxisPolicyDefault;
}

// HOW MANY PAGES ONE PASS MAY RETURN -- THE FACTORY DEFAULT, AND WHY THIS NUMBER.
//
// This is NOT a new policy. The tree already states the per-round recall cost bound exactly
// once, and in BYTES: `turn_recall_byte_budget`, default 256 MiB (`program_impl.h`, env
// NINFER_TURN_RECALL_BYTES). The per-pass cap is the SAME bound expressed in PAGES, and the
// two denominations have to agree -- when they disagree the cap becomes a second, tighter
// answer to a question the byte budget has already answered. That is the state the 1M
// configuration met:
//
//     256 MiB / (64 tokens x 18,464 B/token) = 268,435,456 / 1,181,696 = 227 whole pages.
//
// MEASURED at the 1M configuration's own scale (dl/recallplan/REPORT.md sec.3.3, reproduced
// at the same scale by dl/idx1m/probe/probe_cap_admit.cpp): over 15,782 rows x 64 =
// 1,010,048 tokens the strongest admissible passage is a contiguous run of 43 pages. At the
// previous default of 8 the selector answered `RefusedBudget` with `pages=0`; at 227 the same
// run, carrying the same `need=43`, is `Found` with `pages=43`.
//
// THE REFUSAL IS NOT WEAKENED AND TRUNCATION IS NOT INTRODUCED. The tree's named rule at the
// edge is `kRecallBudgetEdgePolicy = RefuseNotTruncate` (`spec/turn_recall_journal.h`), and a
// run the cap cannot hold is still refused, naming the run, the cap and the pages it would
// have dropped. What changes is only WHICH NUMBER has to be moved to admit the run -- and it
// is no longer a number an operator has to move to reach the tree's own declared default.
//
// The override is unchanged: NINFER_SUM_DIR_BLOCKS=<pages> wins, as `from_env()` below reads.
inline constexpr std::uint32_t kSumDirDefaultBlocksPerPass = 227U;
static_assert(kSumDirDefaultBlocksPerPass ==
                  ((256ULL << 20) / (kSumDirBlockTokens * 18464ULL)),
              "the default pass bound IS the round's own byte budget restated in whole "
              "64-token pages: 227 == floor(256 MiB / (64 x 18,464 B))");

struct SumDirKnobs {
    bool enabled                      = false; // NINFER_SUM_DIR=1 (or any non-"0" value)
    int  min_idle_secs                = 30;    // quiescence must have lasted this long
    // The per-pass page bound. kSumDirDefaultBlocksPerPass above carries the derivation;
    // NINFER_SUM_DIR_BLOCKS overrides it.
    std::uint32_t max_blocks_per_pass = kSumDirDefaultBlocksPerPass;
    std::uint32_t max_summary_bytes   = 160;   // a catalogue line, not a paragraph
    // The byte-axis strategy. DEFAULT = today's answer (see SumDirByteAxisPolicy above):
    // `sum_dir_row_bound()` already refuses a row whose `state != Live`, so landing this
    // patch with the default in place changes nothing until a caller asks for something
    // else. `byte_axis_policy_error` records an unparseable NINFER_SUM_DIR_BYTE_AXIS.
    SumDirByteAxisPolicy byte_axis_policy       = kSumDirByteAxisPolicyDefault;
    bool                 byte_axis_policy_error = false;

    [[nodiscard]] static SumDirKnobs from_env() {
        SumDirKnobs knobs;
        if (const char* enable = std::getenv("NINFER_SUM_DIR")) {
            if (enable[0] != '\0' && enable[0] != '0') { knobs.enabled = true; }
        }
        if (const char* secs = std::getenv("NINFER_SUM_DIR_IDLE_SECS")) {
            const int value = std::atoi(secs);
            if (value >= 0) { knobs.min_idle_secs = value; }
        }
        if (const char* blocks = std::getenv("NINFER_SUM_DIR_BLOCKS")) {
            const int value = std::atoi(blocks);
            if (value > 0) { knobs.max_blocks_per_pass = static_cast<std::uint32_t>(value); }
        }
        if (const char* axis = std::getenv("NINFER_SUM_DIR_BYTE_AXIS")) {
            knobs.byte_axis_policy =
                sum_dir_byte_axis_policy_parse(axis, knobs.byte_axis_policy_error);
        }
        return knobs;
    }
};

// The single rule the caller must satisfy before summarizing. Stated once, here, so that no future
// call site can quietly summarize with traffic in flight: "idle" is ALREADY DEFINED in this tree
// as request_capacity_->active == 0 with admissions refused (B4, generation_service.cpp:518-550),
// and `drain_closed` is that fact. Quiescence is the precondition because a catalogue pass re-reads
// spilled blocks, i.e. it spends the same bandwidth a decode round needs -- which is a cost claim,
// not a citation: the calibration that priced it lived in the ledger specification that is not in
// this tree, so treat the number as unmeasured here.
[[nodiscard]] inline bool sum_dir_idle_eligible(const SumDirKnobs& knobs, bool drain_closed,
                                               int idle_secs) noexcept {
    if (!knobs.enabled) { return false; }
    if (!drain_closed) { return false; }
    return idle_secs >= knobs.min_idle_secs;
}

// ---------------------------------------------------------------------------
// wire format
// ---------------------------------------------------------------------------
//
// Little-endian, host-independent: a fixed 48-byte header, then the rows at one fixed stride, then
// the catalogue bytes in row order (one line per row, skipped when empty), then the block contents.
// Dense and sorted, with no index column: a directory is small and is read whole, so sorting on
// load is cheap and removes an entire class of "the index disagrees with the data" bugs.
//
// The seal is an ACCIDENT DETECTOR, not authenticity: it says "the bytes on disk are the bytes this
// writer wrote", and nothing more is claimed. Standard crc32 (0xEDB88320, reflected), the same
// construction the (absent) ledger specification used, so one tool can check an artefact of either
// shape once both exist.

inline constexpr std::uint32_t kSumDirMagic         = 0x31445253U; // 'S','R','D','1'
inline constexpr std::uint8_t  kSumDirVersion       = 1U;
inline constexpr std::uint32_t kSumDirHeaderBytes   = 48U;
inline constexpr std::uint32_t kSumDirHeaderCovered = 44U; // the header seal covers [0, 44)
inline constexpr std::uint64_t kSumDirMaxRows       = 1ULL << 26; // the bound the spill bitmap uses
                                                                  // (program_impl.h:930 kMaxFileSlots)

// ONE implementation, two call conventions: this is the ledger's reflected-CRC-32 primitive
// (spec/turn_recall_journal.h recall_crc32_raw). The header seal keeps the running register --
// it covers a prefix [0, 44) and then the trailing payload, so it needs a seed -- while a
// record seal finalises one. Same loop, same polynomial, no second copy to drift.
[[nodiscard]] constexpr std::uint32_t sum_dir_crc32(std::uint32_t seed, const std::uint8_t* data,
                                                   std::size_t bytes) noexcept {
    return turn_recall::recall_crc32_raw(seed, data, bytes);
}

// The seal this header writes is CRC-32/ISO-HDLC left UN-finalised (the writer chains a
// header prefix and then the payload through one running register, so the xor-out belongs to
// whoever knows the last byte). Pinned to the published check value of "123456789" exactly as
// the ledger pins its finalised form, which is also the proof that the two are one algorithm:
//     recall_crc32(d, n) == ~sum_dir_crc32(0xFFFFFFFFU, d, n)
inline constexpr std::uint8_t kSumDirCrcCheckBytes[9] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
static_assert(sum_dir_crc32(0xFFFFFFFFU, kSumDirCrcCheckBytes, 9U) == 0x340BC6D9U,
              "sum_dir's seal is the ledger's reflected CRC-32 (0xCBF43926 finalised)");
static_assert(~sum_dir_crc32(0xFFFFFFFFU, kSumDirCrcCheckBytes, 9U) ==
                  turn_recall::recall_crc32(kSumDirCrcCheckBytes, 9U),
              "the two call conventions are one algorithm");

struct SumDirHeader {
    std::uint32_t magic          = kSumDirMagic;   //  0
    std::uint8_t  version        = kSumDirVersion; //  4
    std::uint8_t  block_tokens   = 0;              //  5  u8 on the wire, validated on load
    std::uint16_t reserved0      = 0;              //  6  must be 0
    std::uint64_t sequence_tag   = 0;              //  8  which session wrote this (caller-supplied)
    std::uint64_t row_count      = 0;              // 16  < kSumDirMaxRows
    std::uint64_t summary_bytes  = 0;              // 24
    std::uint64_t content_tokens = 0;              // 32
    std::uint32_t payload_digest = 0;              // 40  crc32 over rows ++ summaries ++ content
    std::uint32_t header_digest  = 0;              // 44  crc32 over [0, 44)
    // 48 bytes, both digests explicit so a torn header is detectable.
};
static_assert(sizeof(SumDirHeader) == kSumDirHeaderBytes,
              "the header is one fixed unit; a packed reader must see the same 48");

namespace detail {

inline void sum_dir_pack_u32_at(std::uint8_t* out, std::size_t offset, std::uint32_t value) noexcept {
    for (std::uint32_t byte = 0; byte < 4; ++byte) {
        out[offset + byte] = static_cast<std::uint8_t>(value >> (8U * byte));
    }
}
inline void sum_dir_pack_u64_at(std::uint8_t* out, std::size_t offset, std::uint64_t value) noexcept {
    for (std::uint32_t byte = 0; byte < 8; ++byte) {
        out[offset + byte] = static_cast<std::uint8_t>(value >> (8U * byte));
    }
}
inline std::uint32_t sum_dir_unpack_u32(const std::uint8_t* in, std::size_t offset) noexcept {
    std::uint32_t value = 0;
    for (std::uint32_t byte = 0; byte < 4; ++byte) {
        value |= static_cast<std::uint32_t>(in[offset + byte]) << (8U * byte);
    }
    return value;
}
inline std::uint64_t sum_dir_unpack_u64(const std::uint8_t* in, std::size_t offset) noexcept {
    std::uint64_t value = 0;
    for (std::uint32_t byte = 0; byte < 8; ++byte) {
        value |= static_cast<std::uint64_t>(in[offset + byte]) << (8U * byte);
    }
    return value;
}
inline void sum_dir_pack_bytes(std::vector<std::uint8_t>& out, const void* data, std::size_t bytes) {
    const auto* bytes_in = static_cast<const std::uint8_t*>(data);
    out.insert(out.end(), bytes_in, bytes_in + bytes);
}

} // namespace detail

// ---------------------------------------------------------------------------
// SumDir -- the directory
// ---------------------------------------------------------------------------

struct SumDirLoadReport {
    std::uint64_t rows_read      = 0;
    std::uint64_t rows_rejected  = 0; // not well-formed, or out of range of its own column
    std::uint64_t summaries_read = 0;
    std::uint64_t content_tokens = 0;
    bool header_ok  = false;
    bool payload_ok = false;
    bool truncated  = false;
    std::string error;
};

// ---------------------------------------------------------------------------
// THE UNLOAD-TEXT LAYER -- 64-blocks concatenated, judged block by block
// ---------------------------------------------------------------------------
//
// WHAT THIS SECTION ADDS, AND WHAT IT REPLACES
//   It answers the user's two questions as code rather than as prose:
//     "take the 64-blocks to concatenate unload text; overlap is fine but a gap is not; and there
//      has to be a judgement of which blocks can be unloaded and which cannot."
//     "I am not asking to change the 64 structure, I mean you concatenate unload text out of 64."
//
//   * the block stays the engine's 64-token Paged-KV page. `static_assert(kSumDirBlockTokens ==
//     64U)` above STAYS. The block-to-page map does NOT move.
//   * block -> unload text is N:1. Want a longer unit? Concatenate more blocks; never stretch one.
//   * OVERLAP IS A FEATURE (one block may sit in several unload texts) and a GAP IS A BUG UNLESS A
//     VERDICT OR A NAMED REFUSAL EXPLAINS IT.
//   * every block gets a verdict, with a reason. No block may be silently skipped.
//
// WHY NO WIRE CHANGE IS NEEDED FOR ANY OF THIS
//   An unload text over blocks [i, j] is the contiguous token range [64i, 64(j+1)). A `SumDirRow`
//   has ALWAYS carried an arbitrary `token_begin`/`token_end`/`content_count` triple and `append()`
//   never checked the count against 64 -- so a row whose span is a multiple of 64 is expressible
//   TODAY, with the same 80-byte stride, the same loader and the same header. Nothing about 64
//   moves, and this section is pure addition: not one line above it is changed.
//
// ---------------------------------------------------------------------------
// THE FIELD THAT IS DELETED, NOT RENAMED
// ---------------------------------------------------------------------------
// An earlier draft carried `SumDirBlockFacts::plane_dropped` = "a plane was dropped for a layer
// this block's text uses". No such engine state exists, and four independent tree anchors say so:
//
//   (a) `cold_host_tier.h:34-41` -- "GRANULARITY: one logical page ACROSS ALL TEXT LAYERS, never a
//       slice of one. ... a cold-slot sentinel encodes ONE slot base and the attention kernels
//       index that slot number in every layer's cold_slots at once, so a device page can only be
//       demoted whole-slot x all layers. The tier therefore has NO per-layer and no per-head
//       eviction notion at all: one page is ONE residency bit."
//   (b) `program_impl.h` -- "Mirror every layer's slot bytes into its spill file. The slot is a
//       fixed-stride unit and this page owns `file_slot`": the loop is `for layer < layers` and the
//       offset is `file_slot * bytes` in EVERY layer's file. ONE slot per page, not per plane.
//   (c) `turn_recall_journal.h:170-171` -- `static_assert(!recall_codec_admitted(RecallCodec::
//       Mixed), "a page whose layers use different codecs has no single stride")`. A per-layer
//       split is not merely absent, it is INEXPRESSIBLE in the recall format.
//   (d) `turn_recall_journal.h:609-617` -- "per-layer selection would fragment one logical page
//       across `layers` files with independent lifetimes and cannot be made atomic."
//
// `kv_bit_budget.h`'s "(page, kv_head, K|V plane)" is a PRICING unit -- how many bytes one slot
// costs on the ladder -- and it is NOT an eviction unit. The two were conflated.
//
// => the byte axis is a PAGE property and the row already carries it: `row.state == Live` means "a
//    Spill record stands for this page", which is exactly the two-valued `SumDirState` above
//    (`Live = RecallKind::Spill`, `Dead = RecallKind::Release`). The replacement fact is
//    `byte_axis_present`, it needs NO new engine hook, and it is derived in exactly one place:
//    `sum_dir_byte_axis_present()` below.
//
// A NOTE ON TRUST: `state == Live` is what the row SAYS, and a released slot is recycled (the
// engine zeroes its file-used table), so a stale row can point at another page's bytes.
// `sum_dir_row_bound()` above already says true-from-`bound` means "this row may be OFFERED",
// never "reuse these bytes". So `Admissible` may be REPORTED from `state == Live`, but the byte
// axis is only USED after `sum_dir_row_bound()` agrees. The verdict is a plan; `bound` is the
// authority. Never let the verdict replace `bound`.

// ---------------------------------------------------------------------------
// THE PER-BLOCK VERDICT
// ---------------------------------------------------------------------------
//
// ORDER IS A CONTRACT, but NOT uniformly -- two of the six adjacencies are load-bearing for
// CORRECTNESS, two for LIVENESS / lever soundness, one for AUDITABILITY, and one is informational.
//
//   1. Empty            -- nothing to unload. NOT a policy outcome: for a block a real sequence
//                          owns this is provably unreachable (see `sum_dir_resident_block`), so a
//                          non-zero count is a CALLER BUG, not a verdict.
//   2. SystemPrefix     -- inside the fixed template/system prefix. ECONOMIC (a per-request
//                          re-prefill), and it has NO tree anchor at all.
//   3. Resident         -- the partial tail page, i.e. the tokens past the last committed page
//                          boundary. STRUCTURAL, and computable from the sequence length alone.
//   4. PinnedByAnchor   -- an anchor frontier sits inside it. ECONOMIC + SCHEDULING (the engine
//                          can release the anchor), engine-side anchor exists.
//   5. ControlToken     -- carries an injected control/template token. The only SEMANTIC keep:
//                          splicing it into an unload text changes the chat structure.
//   6. AdmissibleTextOnly -- a page property: the byte axis is gone, the text axis is not. Whether
//                          this is OFFERED is the strategy (`SumDirByteAxisPolicy`).
//   7. Admissible       -- unloadable on both axes.
enum class SumDirAdmissibility : std::uint8_t {
    Admissible         = 0,
    AdmissibleTextOnly = 1,
    ControlToken       = 2,
    PinnedByAnchor     = 3,
    Resident           = 4,
    SystemPrefix       = 5,
    Empty              = 6,
};

// The rank IS the contract. Higher rank wins. `sum_dir_avl_rank` is the one place the order is
// written down, so a reordering is a one-line diff and the 64-mask truth table in the self-test
// catches it. (The enum VALUES are the wire order and are deliberately different; do not read the
// enum's numbering as the precedence.)
[[nodiscard]] constexpr std::uint8_t sum_dir_avl_rank(SumDirAdmissibility verdict) noexcept {
    switch (verdict) {
    case SumDirAdmissibility::Admissible: return 0;
    case SumDirAdmissibility::AdmissibleTextOnly: return 1;
    case SumDirAdmissibility::ControlToken: return 2;
    case SumDirAdmissibility::PinnedByAnchor: return 3;
    case SumDirAdmissibility::Resident: return 4;
    case SumDirAdmissibility::SystemPrefix: return 5;
    case SumDirAdmissibility::Empty: return 6;
    }
    return 0;
}

[[nodiscard]] inline const char* sum_dir_admissibility_name(SumDirAdmissibility verdict) noexcept {
    switch (verdict) {
    case SumDirAdmissibility::Admissible: return "admissible";
    case SumDirAdmissibility::AdmissibleTextOnly: return "admissible-text-only";
    case SumDirAdmissibility::ControlToken: return "control-token";
    case SumDirAdmissibility::PinnedByAnchor: return "pinned-by-anchor";
    case SumDirAdmissibility::Resident: return "resident";
    case SumDirAdmissibility::SystemPrefix: return "system-prefix";
    case SumDirAdmissibility::Empty: return "empty";
    }
    return "unknown";
}

// WHAT THE JUDGE IS ALLOWED TO KNOW. Six fields, every one a fact about the SEQUENCE, about a
// CONTENT token, or about the row's OWN stated byte axis. There is no `page`, no `slot`, no
// `file_slot`, no device column -- and that absence is visible in the definition, so "the judge
// must not know where bytes are" is enforced by the TYPE and not by a comment. It is also why the
// deleted `plane_dropped` cannot come back by accident: a per-LAYER fact has no field to live in.
struct SumDirBlockFacts {
    bool content_present      = false; // the block's tokens exist and are final
    bool inside_system_prefix = false; // the fixed template/system prefix covers this block
    bool overlap_frontier     = false; // the partial tail page (see `sum_dir_resident_block`)
    bool pinned_by_anchor     = false; // an anchor / checkpoint frontier sits inside it
    bool has_control_token    = false; // an injected control or template token is in it
    bool byte_axis_present    = false; // row.state == Live: a Spill record stands for this page
};

// THE ONE PLACE THE FACT IS DERIVED FROM THE ROW -- the strategy's first seam.
//
// `policy == IgnoreByteAxis` does not consult the row at all, which is why this overload takes the
// POINTER: "the directory has no row for this block" and "the row says Dead" are the same answer
// under `IgnoreByteAxis` and different answers under the other two, and a `RowState`-only
// signature could not express that.
[[nodiscard]] inline bool sum_dir_byte_axis_present(const SumDirRow* row,
                                                    SumDirByteAxisPolicy policy) noexcept {
    if (policy == SumDirByteAxisPolicy::IgnoreByteAxis) { return true; }
    return row != nullptr && row->state == SumDirState::Live;
}

// The ergonomic form. With the default policy this is EXACTLY the predicate `sum_dir_row_bound()`
// applies to liveness above (`if (row.state != SumDirState::Live) { return false; }`), which is
// what makes the default a compatibility setting rather than a new opinion. The self-test pins
// that equality against the real `sum_dir_row_bound()` and not against a restatement of it.
[[nodiscard]] inline bool
sum_dir_byte_axis_present(const SumDirRow& row,
                          SumDirByteAxisPolicy policy = kSumDirByteAxisPolicyDefault) noexcept {
    return sum_dir_byte_axis_present(&row, policy);
}

// Rank-argmax over the reasons that hold, driven BY `sum_dir_avl_rank`. The rank table is therefore
// the single source of the order: renumbering a rank changes the judge, it does not merely
// contradict it. (An earlier draft was an if-cascade that never consulted `sum_dir_avl_rank`; an
// injection caught it, because a rank change then left the judge's output untouched and only the
// numeric pins reddened. That is a real defect class -- "two encodings of the contract, one of them
// dead weight" -- and it is now impossible here.)
//
// `content_present == false` short-circuits to `Empty`: "there is nothing here" is not a reason
// that holds ALONGSIDE the others, it is the absence of a subject. That is the one place the order
// is not an argmax, and it is why `Empty`'s rank is only pinned numerically.
[[nodiscard]] inline SumDirAdmissibility
sum_dir_block_admissibility(const SumDirBlockFacts& facts) noexcept {
    if (!facts.content_present) { return SumDirAdmissibility::Empty; }
    // The byte axis is a MODE and not a reason: it contributes exactly one of two candidates, and
    // both of them are weaker than every keep reason.
    SumDirAdmissibility best = facts.byte_axis_present ? SumDirAdmissibility::Admissible
                                                       : SumDirAdmissibility::AdmissibleTextOnly;
    const SumDirAdmissibility reasons[4] = {SumDirAdmissibility::ControlToken,
                                            SumDirAdmissibility::PinnedByAnchor,
                                            SumDirAdmissibility::Resident,
                                            SumDirAdmissibility::SystemPrefix};
    const bool holds[4] = {facts.has_control_token, facts.pinned_by_anchor,
                           facts.overlap_frontier, facts.inside_system_prefix};
    for (int k = 0; k < 4; ++k) {
        if (holds[k] && sum_dir_avl_rank(reasons[k]) >= sum_dir_avl_rank(best)) {
            best = reasons[k];
        }
    }
    return best;
}

// NOTE ON LINE NUMBERS: every `file:line` below was read off THIS file at HEAD 3944a53 BEFORE
// the landing, and the landing inserts lines above several of them, so those numbers are stale.
// Each reference is also identified by SYMBOL. Current values, re-verified after the landing:
//   SumDirRow:435  frame_bits field:483  kSumDirRowWireBytes:498  static_assert:499
//   SumDirBlockFacts:916  holds[] (the judge's ONLY read of the three):968
//   sum_dir_block_admissibility:958  sum_dir_frame_facts_apply:1028
//   SumDir::append (7-arg):1490  (8-arg, frame_bits):1528  set_frame_bits:1549
//   payload_digest:1689  serialize:1714  pack_row:1892  pack_row frame_bits:1913
//   unpack_row:1917  unpack_row frame_bits:1936  sum_dir_unload_plan_core:2005
//   sum_dir_unload_plan (4-arg, unchanged):2060  (6-arg, with the axis):2073
//
// ---------------------------------------------------------------------------
// THE FRAME AXIS -> THE THREE FACTS THAT HAD NO ASSIGNER. THE WHOLE POINT OF THIS LANDING.
// ---------------------------------------------------------------------------
//
// MEASURED, BEFORE THIS FUNCTION EXISTED (the evidence, not a claim):
//     grep -n 'has_control_token\|pinned_by_anchor\|inside_system_prefix' src/spec/sum_dir.h
//   returned exactly six lines -- :885 / :887 / :888 (their own declarations, all `= false`),
//   :935 / :936 (the four-element `holds[]` array the judge reads), and nothing else. NOT ONE
//   LEFT-HAND SIDE. That is the exact mechanism behind "7 条判据 0/7 按意思": three of the seven
//   verdicts could never be produced by ANY input, and the only two live unload verdicts were
//   separated by page membership alone (`byte_axis_present`, which is `row.state`).
//
// WHAT THIS FUNCTION IS. The missing assigner -- and nothing more. It is a pure function of the
// row's own frame axis and of frontiers the sequence ALREADY carries, so it cannot invent engine
// state and it cannot change what any existing caller sees (the 4-argument `sum_dir_unload_plan`
// below passes empty spans and therefore still produces the all-false facts it always did).
//
//   frame_bits    -- `SumDirRow::frame_bits` for THIS block. Bit j = offset j is template/control.
//   boundary_bits -- bit j = an anchor / checkpoint / message / cache frontier sits at offset j of
//                    this block. The frontiers come from what the request already publishes
//                    (`EncodedChat::message_boundaries`, `.cache_boundaries`,
//                    `.rewrite_execution_frontiers`, processor.h:130-133). A frontier that sits on a
//                    block EDGE is inside NO block and sets no bit -- which is what keeps "the
//                    anchor sits INSIDE it" (:887) from claiming the block that merely begins there.
//   tokens        -- the block's own token count (`sum_dir_row_tokens`).
//
// HONEST BOUNDARY OF EACH DERIVATION (this is the part that must not be overstated):
//   has_control_token    EXACT. `frame_bits != 0` means a template or added-token position is in
//                        this block, which is verbatim what :888 says the field means.
//   pinned_by_anchor     EXACT given the frontiers: `boundary_bits != 0` means a frontier is strictly
//                        inside the block, which is verbatim what :887 says.
//   inside_system_prefix SUFFICIENT, NOT NECESSARY. `frame_bits == mask` says "every position of
//                        this block is template", which every block of the system preamble
//                        satisfies -- so `SystemPrefix` becomes reachable for the first time. But a
//                        block of pure template GLUE between two user turns satisfies it too, so it
//                        can over-report. Narrowing it to the true preamble extent needs
//                        `RenderedChat::message_boundaries` (chat_template.h:125-128) crossed with
//                        the template's own structure. That is named as a follow-up; it is NOT
//                        claimed here.
[[nodiscard]] inline SumDirBlockFacts sum_dir_frame_facts_apply(SumDirBlockFacts facts,
                                                               std::uint64_t frame_bits,
                                                               std::uint64_t boundary_bits,
                                                               std::uint32_t tokens) noexcept {
    const std::uint64_t mask = tokens == 0U              ? 0ULL
                               : tokens >= kSumDirBlockTokens ? ~0ULL
                                                             : ((1ULL << tokens) - 1ULL);
    facts.has_control_token    = frame_bits != 0ULL;
    facts.inside_system_prefix = tokens != 0U && frame_bits == mask;
    facts.pinned_by_anchor     = boundary_bits != 0ULL;
    return facts;
}

// The RECALL side of the same column: a consumer holding only the row can now ask "does this block
// begin at / contain a role boundary?" -- the question whose absence at `program_impl.h:13041`
// (the append arm restarting from `sequence.execution_frontier` instead of the record's own index)
// is what let a re-sent block be read as a fresh user turn.
[[nodiscard]] inline std::uint64_t sum_dir_row_frame_bits(const SumDirRow& row) noexcept {
    return row.frame_bits;
}

[[nodiscard]] inline bool sum_dir_row_has_frame_position(const SumDirRow& row) noexcept {
    return row.frame_bits != 0ULL;
}

// Does the frame axis say this block may NOT be re-sent as ordinary input? `frame_bits != 0` is not
// sufficient on its own (template glue is harmless), so the caller supplies the offset it cares
// about; bit 0 is the one the loss at program_impl.h:13041 was about.
[[nodiscard]] inline bool sum_dir_row_frame_at(const SumDirRow& row, std::uint32_t offset) noexcept {
    return offset < kSumDirBlockTokens && ((row.frame_bits >> offset) & 1ULL) != 0ULL;
}

[[nodiscard]] inline bool sum_dir_is_unloadable(SumDirAdmissibility verdict) noexcept {
    return verdict == SumDirAdmissibility::Admissible ||
           verdict == SumDirAdmissibility::AdmissibleTextOnly;
}

// `byte_axis_present` is the ONLY thing that separates the two unloadable verdicts, so it is the
// ONLY thing that can make an unload text lose its byte recall.
[[nodiscard]] inline bool sum_dir_verdict_recalls_bytes(SumDirAdmissibility verdict) noexcept {
    return verdict == SumDirAdmissibility::Admissible;
}

[[nodiscard]] inline bool sum_dir_keep_verdict(SumDirAdmissibility verdict) noexcept {
    return !sum_dir_is_unloadable(verdict);
}

// The judge applied to every block at once. This is the RIGHT SIDE of the audit's
// `verdict_facts_mismatch` check -- and in production it is genuinely a different pass from the
// report under audit, because the directory is persisted (see `serialize`/`deserialize` above)
// while the facts are recomputed now. That is what stops the check from being a tautology.
[[nodiscard]] inline std::vector<SumDirAdmissibility>
sum_dir_admissibility_report_from(const std::vector<SumDirBlockFacts>& facts) {
    std::vector<SumDirAdmissibility> verdicts;
    verdicts.reserve(facts.size());
    for (const SumDirBlockFacts& block : facts) {
        verdicts.push_back(sum_dir_block_admissibility(block));
    }
    return verdicts;
}

// ---------------------------------------------------------------------------
// EXHAUSTIVENESS, AND THE TWO FACTS THE SEQUENCE ITSELF DECIDES
// ---------------------------------------------------------------------------
//
// The number of judgements is fixed by the SEQUENCE and not by the caller. A block that gets no
// entry is a bug a `size()` comparison catches, as long as the audit is handed `token_count` and
// compares, instead of trusting `verdicts.size()` as the denominator.
[[nodiscard]] inline std::uint32_t sum_dir_block_count(std::uint32_t token_count) noexcept {
    if (token_count == 0) { return 0; }
    return (token_count + kSumDirBlockTokens - 1U) / kSumDirBlockTokens;
}

inline constexpr std::uint32_t kSumDirNoBlock = 0xFFFFFFFFU;

// `Resident`, computed from the sequence length ALONE, through the tree's own arithmetic.
//
//   `recall_pages_for_frontier(frontier, page_tokens)` is `frontier / page_tokens` -- a FLOOR. So
//   the page `frontier / 64` is NOT committed while `frontier % 64 != 0`, and it is exactly the one
//   page that further generation will continue writing. The engine says the same thing from the
//   other side when it advances `cold_frontier` by a whole page at a time.
//
// => at most ONE block per sequence is `Resident`, it is `token_count / 64`, and it exists iff
//    `token_count % 64 != 0`. That is a TRIPWIRE: `resident_blocks <= 1`, and
//    `resident_blocks == (token_count % 64 != 0)`. No engine hook, no frontier argument, no
//    `overlap_frontier` guessed by a caller.
[[nodiscard]] inline std::uint32_t sum_dir_resident_block(std::uint32_t token_count) noexcept {
    if (token_count == 0 || token_count % kSumDirBlockTokens == 0) { return kSumDirNoBlock; }
    return turn_recall::recall_pages_for_frontier(token_count, kSumDirBlockTokens);
}

// The blocks a recall can serve at all: `[0, committed_pages)`. The unload-text window is a SUBSET
// of this, and the check "every text lives inside the committed window" is what makes `Resident`
// non-negotiable instead of a matter of taste.
[[nodiscard]] inline std::uint32_t sum_dir_committed_blocks(std::uint32_t token_count) noexcept {
    return turn_recall::recall_pages_for_frontier(token_count, kSumDirBlockTokens);
}

struct SumDirBlockFactsFromSequence {
    SumDirBlockFacts facts;
    bool resident_here = false;
};

// The sequence-derived part of the facts, for ALL blocks at once. `content_present` is true by
// construction for every block a real sequence owns: block i covers `[64i, min(64(i+1), len))`,
// which is non-empty for every i < ceil(len/64). So `Empty` is unreachable here BY CONSTRUCTION and
// a non-zero `Empty` count means the caller invented a block.
//
// The byte axis is NOT set here, and that is deliberate: the sequence cannot know it. It is the
// row's column, and `sum_dir_plan_unload()` below is where the two halves are joined.
[[nodiscard]] inline std::vector<SumDirBlockFactsFromSequence>
sum_dir_facts_from_sequence(std::uint32_t token_count) noexcept {
    std::vector<SumDirBlockFactsFromSequence> out(sum_dir_block_count(token_count));
    const std::uint32_t resident = sum_dir_resident_block(token_count);
    for (std::uint32_t i = 0; i < out.size(); ++i) {
        out[i].facts.content_present = true;
        out[i].resident_here         = (i == resident);
        out[i].facts.overlap_frontier = (i == resident);
    }
    return out;
}

// ---------------------------------------------------------------------------
// CONCATENATING BLOCKS INTO AN UNLOAD TEXT -- rule C1
// ---------------------------------------------------------------------------
//
// An unload text is a MAXIMAL RUN OF CONSECUTIVE offered blocks. Its token range is the plain
// interval [64*i, 64*(j+1)) -- contiguous text, one row, no new field, and therefore no wire change.
//
// WHY NOT SPLICE ACROSS A HOLE: a hole means the run's tokens are not contiguous, and one row
// carries exactly ONE `[token_begin, token_end)`. Splicing would need a new field -- a wire change,
// which was explicitly not asked for. OVERLAP is the answer: build two overlapping runs around the
// hole and let the coverage rule account for both.
//
// LIVENESS: the outer loop index MUST advance on every path. An earlier draft's first sabotage did
// not redden its probe, it HUNG, because the builder's progress depended on the coalescing loop.
// Terminate by construction: `index` advances in both branches.
struct SumDirUnloadText {
    std::uint32_t first_block = 0; // inclusive
    std::uint32_t last_block  = 0; // inclusive
    std::uint32_t token_begin = 0;
    std::uint32_t token_end   = 0; // exclusive; token_end - token_begin is a MULTIPLE of the block
    bool byte_recallable      = false; // false iff some member block is AdmissibleTextOnly
    SumDirDigest identity{}; // the EXISTING digest over the CONCATENATED tokens -- no new identity
};

// The offering gate: the strategy's second seam. Under `RefuseWhenAbsent` a block whose byte axis
// is absent is not offered; under the other two policies it is. The JUDGE is untouched by the
// policy -- only the offering is, which is why no eighth verdict is needed and the 7-verdict
// contract (and its 64-mask truth table) is policy-independent.
[[nodiscard]] inline bool sum_dir_verdict_is_offered(SumDirAdmissibility verdict,
                                                     SumDirByteAxisPolicy policy) noexcept {
    if (!sum_dir_is_unloadable(verdict)) { return false; }
    if (policy == SumDirByteAxisPolicy::RefuseWhenAbsent) {
        return sum_dir_verdict_recalls_bytes(verdict);
    }
    return true;
}

[[nodiscard]] inline std::vector<SumDirUnloadText>
sum_dir_unload_texts(const std::vector<SumDirAdmissibility>& verdicts,
                     const std::vector<std::uint32_t>& tokens) noexcept {
    std::vector<SumDirUnloadText> texts;
    const std::uint32_t blocks = static_cast<std::uint32_t>(verdicts.size());
    std::uint32_t index        = 0;
    while (index < blocks) {
        if (!sum_dir_is_unloadable(verdicts[index])) { ++index; continue; } // progress guaranteed
        const std::uint32_t first = index;
        bool byte_recallable      = true;
        while (index < blocks && sum_dir_is_unloadable(verdicts[index])) {
            if (!sum_dir_verdict_recalls_bytes(verdicts[index])) { byte_recallable = false; }
            ++index;
        }
        const std::uint32_t last = index - 1U;
        SumDirUnloadText text;
        text.first_block = first;
        text.last_block  = last;
        text.token_begin = first * kSumDirBlockTokens;
        const std::uint64_t raw_end = static_cast<std::uint64_t>(last + 1U) * kSumDirBlockTokens;
        text.token_end = raw_end < tokens.size() ? static_cast<std::uint32_t>(raw_end)
                                                 : static_cast<std::uint32_t>(tokens.size());
        text.byte_recallable = byte_recallable;
        if (text.token_end > text.token_begin) {
            text.identity = sum_dir_block_digest(tokens.data() + text.token_begin,
                                                 text.token_end - text.token_begin);
        }
        texts.push_back(text);
    }
    return texts;
}

// Rule C1 WITH the offering gate. It deviates from `sum_dir_unload_texts` by ONE conjunct, and it
// delegates for every policy that declines nothing, so there is exactly one implementation of the
// run rule on those paths. Where the two must agree -- a policy that refuses no block -- the
// self-test asserts they return identical texts over a whole corpus, so the pair cannot drift
// silently into "two encodings, one of them dead weight".
[[nodiscard]] inline std::vector<SumDirUnloadText>
sum_dir_unload_texts_offered(const std::vector<SumDirAdmissibility>& verdicts,
                             const std::vector<std::uint32_t>& tokens,
                             SumDirByteAxisPolicy policy) noexcept {
    if (policy != SumDirByteAxisPolicy::RefuseWhenAbsent) {
        return sum_dir_unload_texts(verdicts, tokens);
    }
    std::vector<SumDirUnloadText> texts;
    const std::uint32_t blocks = static_cast<std::uint32_t>(verdicts.size());
    std::uint32_t index        = 0;
    while (index < blocks) {
        if (!sum_dir_verdict_is_offered(verdicts[index], policy)) { ++index; continue; }
        const std::uint32_t first = index;
        bool byte_recallable      = true;
        while (index < blocks && sum_dir_verdict_is_offered(verdicts[index], policy)) {
            if (!sum_dir_verdict_recalls_bytes(verdicts[index])) { byte_recallable = false; }
            ++index;
        }
        const std::uint32_t last = index - 1U;
        SumDirUnloadText text;
        text.first_block = first;
        text.last_block  = last;
        text.token_begin = first * kSumDirBlockTokens;
        const std::uint64_t raw_end = static_cast<std::uint64_t>(last + 1U) * kSumDirBlockTokens;
        text.token_end = raw_end < tokens.size() ? static_cast<std::uint32_t>(raw_end)
                                                 : static_cast<std::uint32_t>(tokens.size());
        text.byte_recallable = byte_recallable;
        if (text.token_end > text.token_begin) {
            text.identity = sum_dir_block_digest(tokens.data() + text.token_begin,
                                                 text.token_end - text.token_begin);
        }
        texts.push_back(text);
    }
    return texts;
}

// ---------------------------------------------------------------------------
// WHY `uncovered_admissible == 0` IS NOT ENOUGH
// ---------------------------------------------------------------------------
//
// The equation `{blocks covered by some text} == {blocks the verdicts call unloadable}` is worth
// having, and the number that must be zero is:
//
//     uncovered_admissible = #{ i : is_unloadable(verdicts[i]) and i is in NO text }
//
// But look at what it CANNOT see, by construction:
//
//     a wrongly-inadmissible block has verdict k in the KEEP set, and
//     `is_unloadable(k)` is false for every keep verdict -- so it can never be counted.  QED.
//
// i.e. the equation is BLIND to exactly the error "an admissible block judged inadmissible". It
// catches a BUILDER bug (verdicts say loadable, the builder forgot the block) and it is
// structurally incapable of catching a VERDICT bug. Two more numbers are needed, and they need two
// more inputs:
//
//   * `verdict_facts_mismatch` -- needs the FACTS: recompute `sum_dir_block_admissibility(facts[i])`
//     and require it to equal `verdicts[i]`. This is the non-blind check, and it is not a tautology
//     in production because the verdicts and the facts come from DIFFERENT PASSES: the directory is
//     persisted while the facts are recomputed now.
//   * `uncovered_refuted_claim` -- needs the SEQUENCE: an uncovered block whose keep verdict is
//     refuted by `token_count` alone. Today exactly two keep reasons are refutable this way:
//     `Resident` (against `sum_dir_resident_block`) and `Empty` (against `content_present`). The
//     other three keep reasons have NO tree witness for their trigger -- that is a measured gap,
//     not an oversight, and it is named per-reason below.
//
// And a block that is uncovered while its verdict IS a keep is not a gap at all: the keep verdict
// IS the reason. "A gap must never be silent" is satisfied by requiring every uncovered block to
// carry a claim, and by refuting every claim the sequence can refute. What cannot be refuted is
// COUNTED and REPORTED as unrefuted, never dropped.
struct SumDirCoverageCost {
    // -- the sequence, not the caller, fixes the denominator
    std::uint32_t blocks_total  = 0;
    std::uint32_t report_len_ok = 0; // 1 iff verdicts.size() == blocks_total
    // -- the Resident tripwire (sequence-computable, no engine input)
    std::uint32_t resident_blocks   = 0; // must be <= 1
    std::uint32_t resident_misfiled = 0; // verdict Resident but not the partial tail block
    std::uint32_t resident_missing  = 0; // a partial tail exists but no block claims Resident
    // -- the Empty tripwire (sequence-computable: Empty is unreachable for a real block)
    std::uint32_t empty_on_real_block = 0; // must be 0
    // -- the judge-vs-report check (needs the FACTS)
    std::uint32_t verdict_facts_mismatch = 0; // must be 0
    // -- coverage
    std::uint32_t blocks_unloadable      = 0;
    std::uint32_t blocks_text_only       = 0; // the byte-axis price, counted in blocks
    std::uint32_t blocks_covered         = 0;
    std::uint32_t unload_texts           = 0;
    std::uint32_t unload_texts_text_only = 0; // the byte-axis price, counted in unload texts
    std::uint64_t covered_tokens         = 0;
    // -- the ONE number that must be zero (a builder / persisted-directory gap)
    std::uint32_t uncovered_admissible = 0;
    // -- the accounting for the rest, so no gap is silent
    std::uint32_t uncovered_total             = 0;
    std::uint32_t uncovered_with_keep_verdict = 0; // has a reason -- not a gap
    std::uint32_t uncovered_refuted_claim     = 0; // must be 0: the SEQUENCE refutes the reason
    std::uint32_t uncovered_unrefuted_claim   = 0; // INFO: no tree witness exists yet
    // -- the shape rules (all must hold; each is a separate red)
    std::uint32_t span_not_multiple_of_block = 0; // a text's span must be a multiple of 64
    std::uint32_t text_beyond_sequence       = 0; // a text must not run past token_count
    std::uint32_t text_beyond_committed      = 0; // a text must not enter the partial tail block
    std::uint32_t text_edge_mismatch         = 0; // text.token_begin/end must match its block range
};

// The sequence-only refutation of a keep claim. `Empty` and `Resident` are the only two verdicts
// whose trigger the sequence decides by itself; everything else returns "no witness".
[[nodiscard]] inline bool sum_dir_sequence_refutes(SumDirAdmissibility verdict,
                                                   std::uint32_t block,
                                                   std::uint32_t token_count) noexcept {
    const std::uint32_t resident = sum_dir_resident_block(token_count);
    if (verdict == SumDirAdmissibility::Resident) { return block != resident; }
    if (verdict == SumDirAdmissibility::Empty) { return block < sum_dir_block_count(token_count); }
    return false;
}

// THE AUDIT. Inputs, and who supplies each one:
//   `token_count` -- the sequence's COMMITTED token count (the engine's `frontier`). This is the
//                    independent variable; everything else is derived or asserted against it.
//   `verdicts`    -- the report under audit. May come from a PREVIOUS pass (persisted rows).
//   `facts`       -- the recomputed facts. Empty means "not supplied" and the two fact-dependent
//                    checks are then SKIPPED, not silently passed.
//   `texts`       -- the unload texts, as block ranges.
[[nodiscard]] inline SumDirCoverageCost
sum_dir_avl_audit(std::uint32_t token_count, const std::vector<SumDirAdmissibility>& verdicts,
                  const std::vector<SumDirBlockFacts>& facts,
                  const std::vector<SumDirUnloadText>& texts) noexcept {
    SumDirCoverageCost cost;
    cost.blocks_total  = sum_dir_block_count(token_count);
    cost.report_len_ok = (verdicts.size() == cost.blocks_total) ? 1U : 0U;

    std::vector<std::uint8_t> covered(cost.blocks_total, 0);
    const std::uint32_t resident = sum_dir_resident_block(token_count);

    for (std::uint32_t i = 0; i < verdicts.size(); ++i) {
        const SumDirAdmissibility verdict = verdicts[i];
        if (verdict == SumDirAdmissibility::Resident) {
            ++cost.resident_blocks;
            if (i != resident) { ++cost.resident_misfiled; }
        }
        if (verdict == SumDirAdmissibility::Empty && i < cost.blocks_total) {
            ++cost.empty_on_real_block;
        }
        if (i < facts.size()) {
            const bool block_is_real = (i < cost.blocks_total);
            if (facts[i].content_present != block_is_real ||
                sum_dir_block_admissibility(facts[i]) != verdict) {
                ++cost.verdict_facts_mismatch;
            }
        }
        if (!sum_dir_is_unloadable(verdict)) { continue; }
        ++cost.blocks_unloadable;
        if (!sum_dir_verdict_recalls_bytes(verdict)) { ++cost.blocks_text_only; }
    }
    if (resident != kSumDirNoBlock && cost.resident_blocks == 0) { ++cost.resident_missing; }

    for (const SumDirUnloadText& text : texts) {
        ++cost.unload_texts;
        if (!text.byte_recallable) { ++cost.unload_texts_text_only; }
        cost.covered_tokens += text.token_end - text.token_begin;

        const std::uint32_t span = text.token_end - text.token_begin;
        if (span % kSumDirBlockTokens != 0) { ++cost.span_not_multiple_of_block; }
        if (text.token_end > token_count) { ++cost.text_beyond_sequence; }
        if (text.last_block >= sum_dir_committed_blocks(token_count)) {
            ++cost.text_beyond_committed;
        }
        const std::uint32_t expect_begin = text.first_block * kSumDirBlockTokens;
        const std::uint64_t expect_end =
            static_cast<std::uint64_t>(text.last_block + 1U) * kSumDirBlockTokens;
        const std::uint32_t clipped_end = expect_end < token_count
                                              ? static_cast<std::uint32_t>(expect_end)
                                              : token_count;
        if (text.token_begin != expect_begin || text.token_end != clipped_end) {
            ++cost.text_edge_mismatch;
        }
        for (std::uint32_t block = text.first_block;
             block <= text.last_block && block < cost.blocks_total; ++block) {
            if (block < covered.size() && covered[block] == 0) {
                covered[block] = 1;
                ++cost.blocks_covered;
            }
        }
    }

    for (std::uint32_t i = 0; i < cost.blocks_total; ++i) {
        const SumDirAdmissibility verdict =
            i < verdicts.size() ? verdicts[i] : SumDirAdmissibility::Empty;
        if (covered[i] != 0) { continue; }
        ++cost.uncovered_total;
        if (sum_dir_is_unloadable(verdict)) {
            ++cost.uncovered_admissible; // MUST be 0: a gap with no verdict behind it
            continue;
        }
        ++cost.uncovered_with_keep_verdict; // the verdict IS the reason -- not a silent gap
        if (sum_dir_sequence_refutes(verdict, i, token_count)) {
            ++cost.uncovered_refuted_claim; // MUST be 0: the SEQUENCE refutes the reason
        } else {
            ++cost.uncovered_unrefuted_claim; // INFO: no tree witness exists yet for this reason
        }
    }
    return cost;
}

// The two wholesale checks, kept as named predicates so a caller cannot forget one.
[[nodiscard]] inline bool sum_dir_avl_no_gap(const SumDirCoverageCost& cost) noexcept {
    return cost.uncovered_admissible == 0;
}
[[nodiscard]] inline bool sum_dir_avl_verdicts_sound(const SumDirCoverageCost& cost) noexcept {
    return cost.verdict_facts_mismatch == 0 && cost.resident_misfiled == 0 &&
           cost.resident_missing == 0 && cost.empty_on_real_block == 0 &&
           cost.uncovered_refuted_claim == 0 && cost.report_len_ok == 1;
}
[[nodiscard]] inline bool sum_dir_avl_shapes_ok(const SumDirCoverageCost& cost) noexcept {
    return cost.span_not_multiple_of_block == 0 && cost.text_beyond_sequence == 0 &&
           cost.text_beyond_committed == 0 && cost.text_edge_mismatch == 0;
}

// ---------------------------------------------------------------------------
// THE PLAN -- the strategy applied, in one value a caller can inspect and log
// ---------------------------------------------------------------------------
//
// `verdicts` is the judge's HONEST output under every policy: the policy does not change what is
// true, it changes what is OFFERED. That separation is what keeps `verdict_facts_mismatch` a real
// check (a policy that rewrote the verdicts would make the audit disagree with itself) and it is
// why `blocks_refused_no_bytes` exists as its own counter: a block the strategy declined to offer
// is named, so declining cannot become a silent gap.
struct SumDirUnloadPlan {
    std::vector<SumDirAdmissibility> verdicts;
    std::vector<SumDirUnloadText>    texts;
    SumDirCoverageCost               cost;
    SumDirByteAxisPolicy             policy       = kSumDirByteAxisPolicyDefault;
    std::uint32_t                    blocks_total = 0;
    std::uint32_t                    blocks_offered           = 0;
    std::uint32_t                    blocks_refused_no_bytes  = 0;
};

// The gap equation, with the strategy's own refusals accounted for BY NAME. Under the default
// policy `blocks_refused_no_bytes` is what carries that accounting; under a policy that declines
// nothing it is 0 and this reduces to `uncovered_admissible == 0` -- the design's original
// equation, unchanged.
[[nodiscard]] inline bool sum_dir_unload_plan_no_silent_gap(const SumDirUnloadPlan& plan) noexcept {
    return plan.cost.uncovered_admissible == plan.blocks_refused_no_bytes;
}

class SumDir {
public:
    SumDir() = default;

    explicit SumDir(std::uint64_t sequence_tag, std::uint32_t block_tokens = kSumDirBlockTokens)
        : sequence_tag_(sequence_tag), block_tokens_(block_tokens) {
        if (block_tokens_ != kSumDirBlockTokens) {
            throw std::invalid_argument(
                "sum directory: the granularity is the Paged-KV page, and only that");
        }
    }

    [[nodiscard]] std::uint64_t sequence_tag() const noexcept { return sequence_tag_; }
    [[nodiscard]] std::uint32_t block_tokens() const noexcept { return block_tokens_; }
    [[nodiscard]] const std::vector<SumDirRow>& rows() const noexcept { return rows_; }
    [[nodiscard]] const std::vector<std::uint32_t>& content() const noexcept { return content_; }
    [[nodiscard]] const std::vector<std::string>& summaries() const noexcept { return summaries_; }
    [[nodiscard]] std::size_t size() const noexcept { return rows_.size(); }
    [[nodiscard]] bool empty() const noexcept { return rows_.empty(); }
    [[nodiscard]] bool sorted() const noexcept { return sorted_; }

    // Appends a row plus the block whose content it catalogues. The identity is COMPUTED here,
    // from `tokens` and from nothing else: an API taking a caller-supplied identity would be an API
    // through which position could sneak back in.
    [[nodiscard]] std::size_t append(const std::uint32_t* tokens, std::size_t count,
                                     std::uint32_t token_begin, std::uint32_t generation,
                                     SumDirCodec codec, std::int32_t file_slot,
                                     std::uint32_t page) {
        if (rows_.size() >= kSumDirMaxRows) {
            throw std::length_error("sum directory: row count exceeds the format's bound");
        }
        SumDirRow row;
        row.block_identity = sum_dir_block_digest(tokens, count);
        row.content_first  = static_cast<std::uint32_t>(content_.size());
        row.content_count  = static_cast<std::uint32_t>(count);
        content_.insert(content_.end(), tokens, tokens + count);
        row.token_begin = token_begin;
        row.token_end   = token_begin + static_cast<std::uint32_t>(count);
        row.page        = page;
        row.generation  = generation;
        row.codec       = codec;
        row.state       = file_slot >= 0 ? SumDirState::Live : SumDirState::Dead;
        row.file_slot   = file_slot;
        rows_.push_back(row);
        // The catalogue column is created in lockstep with the rows and never reordered.
        summaries_.emplace_back();
        sorted_ = false;
        // A new row can be the first row to cover its page, so the block->row index is stale.
        page_index_built_ = false;
        page_generation_index_built_ = false;
        return rows_.size() - 1U;
    }

    [[nodiscard]] std::size_t append(const std::vector<std::uint32_t>& tokens,
                                     std::uint32_t token_begin, std::uint32_t generation,
                                     SumDirCodec codec, std::int32_t file_slot,
                                     std::uint32_t page) {
        return append(tokens.data(), tokens.size(), token_begin, generation, codec, file_slot, page);
    }

    // THE FRAME AXIS CARRIER. The same append, plus the block's frame bitmap -- the column the two
    // overloads above pass 0 for, which means exactly what a legacy row means. It is the ONLY new
    // input, and it is a fact about the block's PROVENANCE, not about its content, so it is
    // deliberately NOT folded into `block_identity`: `sum_dir_block_digest` stays a function of the
    // tokens and of nothing else (`:1372-1374`).
    [[nodiscard]] std::size_t append(const std::uint32_t* tokens, std::size_t count,
                                     std::uint32_t token_begin, std::uint32_t generation,
                                     SumDirCodec codec, std::int32_t file_slot, std::uint32_t page,
                                     std::uint64_t frame_bits) {
        const std::size_t row =
            append(tokens, count, token_begin, generation, codec, file_slot, page);
        rows_[row].frame_bits = frame_bits;
        return row;
    }

    [[nodiscard]] std::size_t append(const std::vector<std::uint32_t>& tokens,
                                     std::uint32_t token_begin, std::uint32_t generation,
                                     SumDirCodec codec, std::int32_t file_slot, std::uint32_t page,
                                     std::uint64_t frame_bits) {
        return append(tokens.data(), tokens.size(), token_begin, generation, codec, file_slot, page,
                      frame_bits);
    }

    // Corrects a row's frame axis without touching anything the identity depends on. The axis is
    // provenance, so this is as safe as `set_summary` below -- and unlike a catalogue line it is 8
    // bytes, so it can be written for every row on the idle path without a budget argument.
    [[nodiscard]] bool set_frame_bits(std::size_t row, std::uint64_t frame_bits) {
        if (row >= rows_.size()) { return false; }
        rows_[row].frame_bits = frame_bits;
        return true;
    }

    // Attaches a catalogue line to a row. THE one mutation the idle summarizer performs, and it
    // touches nothing the identity depends on -- which is the property that lets the catalogue be
    // written, or rewritten, at any time without invalidating a single identity.
    void set_summary(std::size_t row, std::string_view text) {
        if (row >= rows_.size()) { throw std::out_of_range("sum directory: row out of range"); }
        summaries_[row] = std::string(text);
        if (text.empty()) {
            // An empty line is the same thing as no line, and the format says so.
            rows_[row].summary_index  = kSumDirNoSummary;
            rows_[row].summary_count  = 0;
            rows_[row].summary_digest = SumDirDigest{};
            return;
        }
        rows_[row].summary_index  = static_cast<std::uint32_t>(row);
        rows_[row].summary_count  = static_cast<std::uint32_t>(text.size());
        rows_[row].summary_digest = sum_dir_text_digest(text);
    }

    [[nodiscard]] std::string_view summary_of(std::size_t row) const {
        if (row >= rows_.size()) { throw std::out_of_range("sum directory: row out of range"); }
        if (!sum_dir_row_has_summary(rows_[row])) { return {}; }
        return summaries_[row];
    }

    // Marks a block's bytes gone (restored hot, sequence ended, file slot recycled). The row stays:
    // a Dead row is the tombstone that stops a lookup from hitting -- the "Release" state of the
    // block-ledger specification, which is NOT in this tree; the state machine here is complete on
    // its own terms (Live bytes / Dead bytes) and needs nothing from it. Its catalogue line is
    // kept, because "which block was this" is still worth answering after the bytes are gone.
    bool release(std::size_t row) {
        if (row >= rows_.size()) { return false; }
        rows_[row].state     = SumDirState::Dead;
        rows_[row].file_slot = -1;
        return true;
    }

    // Required for lookup and recorded in the file. A separate step so a caller can append many
    // rows on the idle path and pay for the order once. The catalogue column is permuted WITH the
    // rows, which is what keeps the class invariant (row k <-> summaries_[k]) true across a sort.
    void sort_rows() {
        if (rows_.size() != summaries_.size()) {
            throw std::logic_error("sum directory: the catalogue column is out of lockstep");
        }
        const std::size_t n = rows_.size();
        std::vector<std::size_t> order(n);
        for (std::size_t i = 0; i < n; ++i) { order[i] = i; }
        std::sort(order.begin(), order.end(), [this](std::size_t a, std::size_t b) {
            const SumDirRow& left  = rows_[a];
            const SumDirRow& right = rows_[b];
            if (left.block_identity != right.block_identity) {
                return left.block_identity < right.block_identity;
            }
            return left.token_begin < right.token_begin;
        });
        std::vector<SumDirRow> sorted_rows;
        std::vector<std::string> sorted_summaries;
        sorted_rows.reserve(n);
        sorted_summaries.reserve(n);
        for (const std::size_t index : order) {
            sorted_rows.push_back(rows_[index]);
            sorted_summaries.push_back(std::move(summaries_[index]));
        }
        rows_.swap(sorted_rows);
        summaries_.swap(sorted_summaries);
        for (std::size_t i = 0; i < n; ++i) {
            rows_[i].summary_index = sum_dir_row_has_summary(rows_[i])
                                         ? static_cast<std::uint32_t>(i)
                                         : kSumDirNoSummary;
        }
        sorted_ = true;
        // The permutation moved every row's ORDINAL, which is what the block->row index stores.
        page_index_built_ = false;
        page_generation_index_built_ = false;
    }

    // Binary search by identity: the index of the first match plus the [begin, end) run of
    // duplicates, since the same content may legitimately sit at several positions in one session.
    // A caller must still run sum_dir_row_bound() before trusting any of them.
    [[nodiscard]] bool find(const SumDirDigest& identity, std::size_t& begin, std::size_t& end) const {
        if (!sorted_) { throw std::logic_error("sum directory: sort_rows() before lookup"); }
        const auto lower = std::lower_bound(
            rows_.begin(), rows_.end(), identity,
            [](const SumDirRow& row, const SumDirDigest& value) {
                return row.block_identity < value;
            });
        if (lower == rows_.end() || !(lower->block_identity == identity)) { return false; }
        const auto upper = std::upper_bound(
            lower, rows_.end(), identity,
            [](const SumDirDigest& value, const SumDirRow& row) {
                return value < row.block_identity;
            });
        begin = static_cast<std::size_t>(lower - rows_.begin());
        end   = static_cast<std::size_t>(upper - rows_.begin());
        return true;
    }

    // =======================================================================
    // THE BLOCK-TO-ROW INDEX -- the join `sum_dir.h` left open ON PURPOSE
    // =======================================================================
    //
    // The note this pair closes is at the "THE BLOCK-TO-ROW MAP" paragraph below (search for
    // "stated because it is the one genuinely underspecified part"): "Closing that gap (an explicit
    // block->row index) is a design decision this patch leaves open on purpose rather than
    // inventing a convention that the directory's own doctrine -- 'a CONTENT SHORTLIST, not an
    // identity' -- has no basis for." This is that decision, and it invents no convention: it
    // adopts the one the same paragraph already states, verbatim --
    //
    //     "a row names the block it covers with its `page` column
    //      (`page == token_begin / kSumDirBlockTokens`)"
    //
    // -- and it adopts the FIRST-WRITER-WINS tie rule the dead unload plan already uses
    // (`sum_dir_unload_plan_core`, the `row_of_block` loop: "First writer wins, so a directory
    // holding several rows for one page is read deterministically instead of arbitrarily"). So the
    // two callers that need "which row covers block p" now agree by construction instead of by
    // coincidence, and the doctrine objection is answered rather than waived: this is not an
    // identity, it is an ADDRESS, and it is derived from a column the row already carried.
    //
    // WHY THIS IS THE SEAM THE WHOLE SLICE TURNED ON. Three things were blocked on it:
    //   * the length-aware read: `read_located_block` takes a `row_tokens` it must get from
    //     `sum_dir_row_tokens(row)` (turn_recall_journal.h, the `row_tokens IS the directory row's
    //     own length` note), and nothing could produce that row. `recall_cargo_read_run`'s
    //     `page_tokens` argument therefore had NO possible supplier and its one caller omitted it.
    //   * the byte axis: `sum_dir_byte_axis_present(row, policy)` needs the row for a block, and
    //     the only place that ever built the map was the transient `row_of_block` inside the DEAD
    //     unload plan -- which is why the engine's live judge had no byte axis at all.
    //   * the partial tail: the one row that MUST exist for a short block to be read back at its
    //     real length is the row for the LAST block, and it is exactly the row nothing wrote.
    //
    // COMPLEXITY, STATED: the index is a dense `page -> row ordinal` vector, built once on first
    // query and invalidated by any append or sort. Lookup is then O(1) and the whole directory is
    // scanned once, not once per block -- which is the difference that matters for a book of
    // 10 M tokens (156,250 blocks): a per-call linear scan would make the read leg quadratic.
    // Memory is 4 B per block covered.
    //
    // ⚠️ WHAT IT IS *NOT*: it is not an identity, and a hit here is not an authorisation to reuse
    // bytes. It says "this row describes that block". `sum_dir_row_bound()` still has to admit the
    // row, and the engine's own identity comparison still has to authorise the reuse -- the same
    // two gates the doctrine above requires.
    [[nodiscard]] bool row_for_page(std::uint32_t page, std::size_t& row) const {
        ensure_page_index();
        if (page >= page_index_.size()) { return false; }
        // THE TRIPWIRE. A page that several GENERATIONS cover has no single answer, and the one
        // this function used to give was the first writer's -- which is the right ADDRESS and the
        // wrong LENGTH for every other writer (see `row_for_page_generation` below). Refusing is
        // the only answer that cannot splice a wrong length into a read: the caller gets "no row",
        // which the engine's read legs already handle as a NAMED refusal
        // (`REASON=no-cargo-record-for-planned-pages`, program_impl.h:13737-13745), instead of a
        // length that is not its own.
        //
        // This is INERT while the tag is unbound: every row a run writes carries generation 0, so
        // no page can be covered by two generations, and every lookup here -- and therefore every
        // existing caller -- behaves exactly as it did before this landing.
        if (page < page_conflict_.size() && page_conflict_[page] != 0U) { return false; }
        const std::uint32_t ordinal = page_index_[page];
        if (ordinal == kSumDirNoRow || ordinal >= rows_.size()) { return false; }
        row = static_cast<std::size_t>(ordinal);
        return true;
    }

    // =======================================================================================
    // THE WRITER-AWARE JOIN: `row_for_page` ABOVE IS AN ADDRESS, NOT A LENGTH.
    // =======================================================================================
    //
    // WHAT WAS WRONG, AND WHY IT TOOK TWO WRITERS TO SEE. `row_for_page` answers "which row covers
    // block p" with the FIRST-WRITER-WINS tie rule `ensure_page_index` states, and that is the right
    // rule for an ADDRESS: a directory holding several rows for one page is read deterministically
    // instead of arbitrarily. It is the WRONG rule for a LENGTH, and a length is what the engine
    // takes from it -- `live_row_for_page` -> `sum_dir_row_tokens` is the value `read_located_block`
    // and `recall_cargo_read_run`'s `page_tokens` are handed
    // (impl/runtime/program_impl.h:13722-13724 and :13918-13919). So when two writers retire the
    // SAME page index with DIFFERENT lengths -- two lanes of one round covering one page, which is
    // the routine case, not a corner -- a reader can be handed the OTHER writer's length while the
    // bytes it then reads are its own, and the read consumes its own ZERO PADDING past the real end
    // of its record. (turn_recall_journal.h names that padding as its own hazard.)
    //
    // THE TAG IS THE ROW'S OWN `generation` COLUMN, AND IT IS ALREADY ON THE WIRE. `SumDirRow::
    // generation` (:457) is written by `append` from the caller's tag (:1583), and `pack_row` /
    // `unpack_row` already carry it at offset 28 of the row's fixed 80-byte stride (:2093, :2118,
    // `kSumDirRowWireBytes = 80` at :510). THE STRIDE DOES NOT MOVE, NO COLUMN IS ADDED, NO VERSION
    // IS BUMPED, and a file written before this join existed reads back unchanged: its rows carry
    // generation 0, and a caller that asks with 0 gets exactly the rows it always got. That is the
    // same property the frame axis used (:498-510).
    //
    // THE JOIN IS A REFUSAL WHEN IT CANNOT ANSWER HONESTLY. A tag no row carries returns FALSE -- it
    // does NOT fall back to the first writer's row, because that fallback IS the defect. The one
    // case that looks like an exception is not one: a caller asking with generation 0 finds the rows
    // at generation 0, which is what a directory whose tag has never been bound contains, so the
    // joiner's behaviour BEFORE the tag is bound is what it always was, byte for byte.
    //
    // A LENGTH OF 0 IS A REFUSAL, NOT A LENGTH. `tokens_for_page_generation` returns 0 both for "no
    // row of mine covers this block" and for a row of zero length; the format does not produce the
    // latter from `append` (a row is appended with a token count, and `sum_dir_row_well_formed` only
    // requires `token_end >= token_begin`), and a caller that cannot tell them apart is a caller
    // that has to treat 0 as "do not read", which is the safe reading in both cases.
    //
    // WARNING: WHAT THIS DOES *NOT* FIX, NAMED SO IT IS NOT MISTAKEN FOR FIXED.
    //   (i) This is the DIRECTORY's dimension. The CARGO's dimension is the writer's FILE, and that
    //       is a separate change (`dl/cargoid/`: `recall_cargo_path(dir, writer)` +
    //       `recall_cargo_open_writer`). A length that is now the right one does not make the BYTES
    //       the right ones while one engine still owns one cargo file for all lanes.
    //   (ii) The tag is only as good as its binder. The engine writes `sequence.recall_sequence_tag`
    //       into this column (program_impl.h:12584) and that field is DECLARED AND NEVER ASSIGNED
    //       (program.h:546), so every row a run writes today carries 0 and THIS JOIN IS INERT until
    //       the binding lands at the one site where a sequence takes its lane
    //       (program_impl.h:7953, `sequence.lane = lane;`). An inert fix is not a fix; it is a fix
    //       with its remaining work named and located.
    //   (iii) A PER-INCARNATION IDENTITY CANNOT BE THE FILE DIMENSION. It must be unique per
    //       sequence incarnation, a server admits unboundedly many, and a file name minted per
    //       incarnation is unboundedly many files with no reclamation. So the FILE stays keyed by
    //       the BOUNDED slot (the lane) and the unbounded identity stays the ROW's problem -- which
    //       is exactly where this join consults it.
    [[nodiscard]] bool row_for_page_generation(std::uint32_t page, std::uint32_t generation,
                                              std::size_t& row) const {
        ensure_page_generation_index();
        const std::uint64_t key = (static_cast<std::uint64_t>(generation) << 32) |
                                  static_cast<std::uint64_t>(page);
        const auto hit = page_generation_index_.find(key);
        if (hit == page_generation_index_.end()) { return false; }
        row = static_cast<std::size_t>(hit->second);
        return true;
    }

    // The block's REAL token count, as the directory states it -- 0 when no row covers the block.
    // This is the ONE producer of the `row_tokens` that `read_located_block` and
    // `recall_cargo_read_run`'s `page_tokens` take, and it is a reading of the row rather than a
    // re-derivation from the page index precisely so that a SHORT block (the tail of a document)
    // reports its own short length instead of the 64-token stride.
    [[nodiscard]] std::uint32_t tokens_for_page(std::uint32_t page) const {
        std::size_t row = 0;
        if (!row_for_page(page, row)) { return 0U; }
        return sum_dir_row_tokens(rows_[row]);
    }

    // The writer-aware form: the length of THIS writer's row for the block, and 0 when this writer
    // has no row for it -- see the "A LENGTH OF 0 IS A REFUSAL" note above.
    [[nodiscard]] std::uint32_t tokens_for_page_generation(std::uint32_t page,
                                                          std::uint32_t generation) const {
        std::size_t row = 0;
        if (!row_for_page_generation(page, generation, row)) { return 0U; }
        return sum_dir_row_tokens(rows_[row]);
    }

    // The same reading, but only for a row that may be offered to recall: `sum_dir_row_bound`'s
    // "Live and pointing at bytes" rule, without the caller-supplied ledger half (which this
    // header cannot check -- see its own note). A Dead row keeps its catalogue line and its
    // length, so this is the function that keeps a tombstoned block from being read back.
    [[nodiscard]] bool live_row_for_page(std::uint32_t page, std::size_t& row) const {
        if (!row_for_page(page, row)) { return false; }
        return rows_[row].state == SumDirState::Live;
    }

    // THE WRITER-AWARE FORM OF THE FUNCTION THE READ LEGS CALL. `program_impl.h:13722` (the
    // original-position rebuild) and `:13918` (the append leg) are the ONLY two places a `row_tokens`
    // reaches a reader, and both call `live_row_for_page`. Routing them here is the whole of the
    // join-side fix; it is NOT done by this landing (program_impl.h is not this line's to write
    // today) and the handoff names the two anchors.
    [[nodiscard]] bool live_row_for_page_generation(std::uint32_t page, std::uint32_t generation,
                                                    std::size_t& row) const {
        if (!row_for_page_generation(page, generation, row)) { return false; }
        return rows_[row].state == SumDirState::Live;
    }

    // How many blocks this directory covers. The index's own extent, exposed so a caller can
    // state a bound instead of guessing one.
    [[nodiscard]] std::uint32_t indexed_blocks() const {
        ensure_page_index();
        return static_cast<std::uint32_t>(page_index_.size());
    }

    // THE MINIMUM USEFUL RETRIEVAL, and the whole of slice 1's: every Live row whose catalogue line
    // contains `term` as a case-sensitive substring. Case sensitivity is deliberate -- a tokenizer
    // is a policy this header must not invent. This is what makes the feature TARGETED: a candidate
    // is found by what its description says, not by where it sits. A real retriever (BM25 over the
    // summaries; or an any-position index, which the absent ledger specification argued the medial
    // shape needs) replaces this function's body, not its contract.
    //
    // ⭐ THE PRODUCER THIS FUNCTION NEVER HAD. Until this landing `summaries_` was empty in every
    // real run: `set_summary`'s only callers were two test files (the inventory's Q3, confirmed at
    // bedrock), so the one catalogue retriever could only ever return an empty vector and
    // `QSummarySubstring` was a registered question whose column had no writer. It is now written
    // in the ENGINE, in the same retire path that stores a block's text
    // (impl/runtime/program_impl.h, the `[textcargo]` write site), with the line produced by
    // `sum_dir_render_token_line` above -- the same function the recall provider uses to render its
    // query, so a query is expressed in exactly the alphabet this function searches.
    [[nodiscard]] std::vector<std::size_t> search_summaries(std::string_view term) const {
        std::vector<std::size_t> hits;
        if (term.empty()) { return hits; }
        for (std::size_t index = 0; index < rows_.size(); ++index) {
            if (rows_[index].state != SumDirState::Live) { continue; }
            const std::string_view text = summary_of(index);
            if (text.empty()) { continue; }
            if (text.find(term) != std::string_view::npos) { hits.push_back(index); }
        }
        return hits;
    }

    // The block's own token ids, so a query can be run against the block as well as against its
    // description.
    [[nodiscard]] std::vector<std::uint32_t> block_content(std::size_t row) const {
        if (row >= rows_.size()) { throw std::out_of_range("sum directory: row out of range"); }
        const SumDirRow& target = rows_[row];
        if (target.content_first + target.content_count > content_.size()) { return {}; }
        return std::vector<std::uint32_t>(content_.begin() + target.content_first,
                                          content_.begin() + target.content_first +
                                              target.content_count);
    }

    // -----------------------------------------------------------------------
    // serialization
    // -----------------------------------------------------------------------

    [[nodiscard]] std::uint64_t summary_bytes() const noexcept {
        std::uint64_t total = 0;
        for (const std::string& text : summaries_) { total += text.size(); }
        return total;
    }

    // Rows ++ summaries ++ content, in that order: one digest over the payload the header
    // describes, so a loader can separate "the rows and the text agree" from "they do not".
    [[nodiscard]] std::uint32_t payload_digest() const noexcept {
        std::uint32_t crc = 0xFFFFFFFFU;
        for (const SumDirRow& row : rows_) {
            std::uint8_t wire[kSumDirRowWireBytes] = {};
            pack_row(row, wire);
            crc = sum_dir_crc32(crc, wire, kSumDirRowWireBytes);
        }
        for (const std::string& text : summaries_) {
            if (text.empty()) { continue; }
            crc = sum_dir_crc32(crc, reinterpret_cast<const std::uint8_t*>(text.data()),
                                text.size());
        }
        if (!content_.empty()) {
            std::vector<std::uint8_t> packed;
            packed.reserve(content_.size() * 4U);
            for (const std::uint32_t token : content_) {
                std::uint8_t word[4];
                detail::sum_dir_pack_u32_at(word, 0, token);
                detail::sum_dir_pack_bytes(packed, word, 4);
            }
            crc = sum_dir_crc32(crc, packed.data(), packed.size());
        }
        return ~crc;
    }

    [[nodiscard]] std::vector<std::uint8_t> serialize() const {
        std::vector<std::uint8_t> out;
        out.reserve(kSumDirHeaderBytes + rows_.size() * kSumDirRowWireBytes + summary_bytes() +
                    content_.size() * 4U);

        std::uint8_t header[kSumDirHeaderBytes] = {};
        detail::sum_dir_pack_u32_at(header, 0, kSumDirMagic);
        header[4] = kSumDirVersion;
        header[5] = static_cast<std::uint8_t>(block_tokens_);
        header[6] = 0;
        header[7] = 0;
        detail::sum_dir_pack_u64_at(header, 8, sequence_tag_);
        detail::sum_dir_pack_u64_at(header, 16, rows_.size());
        detail::sum_dir_pack_u64_at(header, 24, summary_bytes());
        detail::sum_dir_pack_u64_at(header, 32, content_.size());
        detail::sum_dir_pack_u32_at(header, 40, payload_digest());
        std::uint32_t seal = sum_dir_crc32(0xFFFFFFFFU, header, kSumDirHeaderCovered);
        seal = ~seal;
        detail::sum_dir_pack_u32_at(header, 44, seal);

        detail::sum_dir_pack_bytes(out, header, kSumDirHeaderBytes);
        for (const SumDirRow& row : rows_) {
            std::uint8_t wire[kSumDirRowWireBytes] = {};
            pack_row(row, wire);
            detail::sum_dir_pack_bytes(out, wire, kSumDirRowWireBytes);
        }
        // The catalogue bytes, in row order, an unwritten line contributing nothing.
        for (const std::string& text : summaries_) {
            if (text.empty()) { continue; }
            detail::sum_dir_pack_bytes(out, text.data(), text.size());
        }
        for (const std::uint32_t token : content_) {
            std::uint8_t word[4];
            detail::sum_dir_pack_u32_at(word, 0, token);
            detail::sum_dir_pack_bytes(out, word, 4);
        }
        return out;
    }

    // STRICT by design: an unknown version, a mismatched granularity, a digest that does not match,
    // or a row that is not well-formed is a REFUSAL -- an empty directory plus a report, never a
    // partial load. A half-loaded directory would be worse than none, because the absent rows would
    // read as "not recallable" rather than "not read".
    [[nodiscard]] static SumDir deserialize(const std::uint8_t* data, std::size_t bytes,
                                            SumDirLoadReport& report) {
        SumDir dir;
        if (data == nullptr || bytes < kSumDirHeaderBytes) {
            report.error     = "shorter than one header";
            report.truncated = true;
            return dir;
        }
        if (detail::sum_dir_unpack_u32(data, 0) != kSumDirMagic) {
            report.error = "bad magic";
            return dir;
        }
        if (data[4] != kSumDirVersion) {
            report.error = "unsupported version";
            return dir;
        }
        if (data[5] != kSumDirBlockTokens) {
            report.error = "block granularity is not the Paged-KV page";
            return dir;
        }
        if (data[6] != 0 || data[7] != 0) {
            report.error = "reserved header bits are set";
            return dir;
        }
        std::uint32_t header_seal = sum_dir_crc32(0xFFFFFFFFU, data, kSumDirHeaderCovered);
        header_seal = ~header_seal;
        if (header_seal != detail::sum_dir_unpack_u32(data, 44)) {
            report.error = "header digest mismatch";
            return dir;
        }
        report.header_ok = true;

        const std::uint64_t row_count      = detail::sum_dir_unpack_u64(data, 16);
        const std::uint64_t summary_bytes  = detail::sum_dir_unpack_u64(data, 24);
        const std::uint64_t content_tokens = detail::sum_dir_unpack_u64(data, 32);
        if (row_count > kSumDirMaxRows) {
            report.error = "row count exceeds the format's bound";
            return dir;
        }
        const std::uint64_t payload_bytes =
            row_count * kSumDirRowWireBytes + summary_bytes + content_tokens * 4ULL;
        if (payload_bytes > bytes - kSumDirHeaderBytes) {
            report.error     = "payload shorter than the header describes";
            report.truncated = true;
            return dir;
        }
        const std::uint8_t* payload = data + kSumDirHeaderBytes;
        std::uint32_t payload_seal =
            sum_dir_crc32(0xFFFFFFFFU, payload, static_cast<std::size_t>(payload_bytes));
        payload_seal = ~payload_seal;
        if (payload_seal != detail::sum_dir_unpack_u32(data, 40)) {
            report.error = "payload digest mismatch";
            return dir;
        }
        report.payload_ok = true;

        const std::uint8_t* summary_base = payload + row_count * kSumDirRowWireBytes;
        const std::uint8_t* content_base = summary_base + summary_bytes;

        dir.sequence_tag_ = detail::sum_dir_unpack_u64(data, 8);
        dir.block_tokens_ = kSumDirBlockTokens;
        dir.rows_.reserve(static_cast<std::size_t>(row_count));

        // Pass 1: shape acceptance. Only the row's structure is judged here, so a row can be
        // dropped without disturbing any other row's catalogue line.
        std::vector<SumDirRow> accepted;
        accepted.reserve(static_cast<std::size_t>(row_count));
        for (std::uint64_t index = 0; index < row_count; ++index) {
            const SumDirRow row = unpack_row(payload + index * kSumDirRowWireBytes);
            const bool shape_ok = sum_dir_row_well_formed(row) &&
                                  row.content_first + row.content_count <= content_tokens;
            if (!shape_ok) {
                ++report.rows_rejected;
                continue;
            }
            accepted.push_back(row);
        }

        // Pass 2: the catalogue bytes, in row order (the writer's layout), with each line verified
        // against its own digest. A line that is absent or does not verify leaves that row without a
        // catalogue; it never receives a neighbour's bytes or a truncated one. The column is filled
        // in ORIGINAL row order, because that is the order the writer laid it out in.
        std::vector<char> has_line(accepted.size(), 0);
        std::vector<std::string> lines(accepted.size());
        std::uint64_t line_offset = 0;
        for (std::size_t k = 0; k < accepted.size(); ++k) {
            const SumDirRow& row = accepted[k];
            if (!sum_dir_row_has_summary(row)) { continue; }
            if (line_offset + row.summary_count > summary_bytes) { continue; }
            const std::string_view text(
                reinterpret_cast<const char*>(summary_base + line_offset), row.summary_count);
            if (!(sum_dir_text_digest(text) == row.summary_digest)) {
                // The line is present but does not belong to this row. The row keeps its identity
                // and loses its catalogue: a description is never worth trusting on faith.
                continue;
            }
            has_line[k] = 1;
            lines[k]    = std::string(text);
            line_offset += row.summary_count;
        }
        // A row whose line was rejected loses the row's own summary columns too, so that the
        // re-serialized bytes describe exactly what is in memory (and the round trip stays exact).
        for (std::size_t k = 0; k < accepted.size(); ++k) {
            if (!sum_dir_row_has_summary(accepted[k])) { continue; }
            if (has_line[k] != 0) { continue; }
            accepted[k].summary_index  = kSumDirNoSummary;
            accepted[k].summary_count  = 0;
            accepted[k].summary_digest = SumDirDigest{};
        }

        for (SumDirRow& row : accepted) { dir.rows_.push_back(row); }
        // The catalogue column must exist in lockstep before the sort, which permutes the two
        // together.
        dir.summaries_ = std::move(lines);
        report.rows_read      = accepted.size();
        report.summaries_read = dir.summaries_.size();

        dir.content_.resize(static_cast<std::size_t>(content_tokens));
        for (std::size_t index = 0; index < dir.content_.size(); ++index) {
            dir.content_[index] = detail::sum_dir_unpack_u32(content_base, index * 4U);
        }
        report.content_tokens = dir.content_.size();
        // The format's own invariant. A directory that is not sorted is a bug in a writer, and
        // sorting here means no reader has to care which one wrote it. sort_rows() also re-derives
        // every row's summary_index from its final position.
        dir.sort_rows();
        return dir;
    }

    [[nodiscard]] static SumDir deserialize(const std::vector<std::uint8_t>& bytes,
                                            SumDirLoadReport& report) {
        return deserialize(bytes.data(), bytes.size(), report);
    }

private:
    // The block->row index, built lazily and invalidated by every mutation that can move a row's
    // ordinal or add one: `append` (and therefore `set_frame_bits`, which appends) and
    // `sort_rows` (which permutes). `mutable` so a `const` lookup can build it -- the alternative
    // is a caller-visible "build the index" step, which is exactly the un-called prerequisite this
    // slice is trying to abolish.
    void ensure_page_index() const {
        if (page_index_built_) { return; }
        std::uint32_t extent = 0;
        for (const SumDirRow& row : rows_) {
            if (row.page == kSumDirNoRow) { continue; }
            if (row.page + 1U > extent) { extent = row.page + 1U; }
        }
        page_index_.assign(extent, kSumDirNoRow);
        page_conflict_.assign(extent, 0U);
        for (std::size_t row = 0; row < rows_.size(); ++row) {
            const std::uint32_t page = rows_[row].page;
            if (page >= page_index_.size()) { continue; }
            // First writer wins -- the same tie rule `sum_dir_unload_plan_core`'s transient
            // `row_of_block` uses, so the two readings of one directory cannot disagree.
            if (page_index_[page] == kSumDirNoRow) {
                page_index_[page] = static_cast<std::uint32_t>(row);
            } else if (rows_[page_index_[page]].generation != rows_[row].generation) {
                // TWO GENERATIONS COVER ONE PAGE. The ADDRESS is still first-writer-wins, but the
                // LENGTH is no longer a single answer, so the page is swept into the tripwire
                // `row_for_page` consults. Note what this costs and what it does not: the generation
                // of the first-writer row is read back through `page_index_` itself, so the
                // authority stays in ONE column and no second dense `page -> generation` vector is
                // introduced.
                page_conflict_[page] = 1U;
            }
        }
        page_index_built_ = true;
    }

    // THE WRITER-AWARE INDEX: `(generation, page) -> row ordinal`. Built LAZILY, on the first tagged
    // lookup, and rebuilt after any mutation that moves an ordinal or adds a row -- so a caller that
    // never asks a tagged question pays NOTHING for it, which is why the engine as it stands today
    // (whose two read legs still call the untagged form) pays nothing.
    //
    // It is a HASH MAP and not a dense vector because the key's high half is the tag, which is
    // unbounded. That is a real cost and it is stated rather than hidden: it is the FIRST structure
    // in this header whose per-block cost is not a small constant, so if this ever needs to be a
    // dense column, this is the line to change. `page_conflict_` above IS dense, and it is 1 B per
    // block covered against the 4 B the address index already pays.
    void ensure_page_generation_index() const {
        if (page_generation_index_built_) { return; }
        ensure_page_index();
        page_generation_index_.clear();
        page_generation_index_.reserve(rows_.size());
        for (std::size_t row = 0; row < rows_.size(); ++row) {
            const SumDirRow& r = rows_[row];
            if (r.page == kSumDirNoRow) { continue; }
            const std::uint64_t key = (static_cast<std::uint64_t>(r.generation) << 32) |
                                      static_cast<std::uint64_t>(r.page);
            // First writer wins here too, for the same reason: one (generation, page) pair names at
            // most one row in any directory this header writes, and if a caller has hand-built a
            // directory where that is false, the FIRST row is the one the address index would have
            // given, so the tagged join agrees with the untagged one wherever the untagged one is
            // still willing to answer.
            page_generation_index_.emplace(key, static_cast<std::uint32_t>(row));
        }
        page_generation_index_built_ = true;
    }

    static void pack_row(const SumDirRow& row, std::uint8_t* wire) noexcept {
        std::size_t offset = 0;
        detail::sum_dir_pack_u64_at(wire, offset, row.block_identity.lo); offset += 8;
        detail::sum_dir_pack_u64_at(wire, offset, row.block_identity.hi); offset += 8;
        detail::sum_dir_pack_u32_at(wire, offset, row.token_begin); offset += 4;
        detail::sum_dir_pack_u32_at(wire, offset, row.token_end); offset += 4;
        detail::sum_dir_pack_u32_at(wire, offset, row.page); offset += 4;
        detail::sum_dir_pack_u32_at(wire, offset, row.generation); offset += 4;
        detail::sum_dir_pack_u32_at(wire, offset, row.content_first); offset += 4;
        detail::sum_dir_pack_u32_at(wire, offset, row.content_count); offset += 4;
        detail::sum_dir_pack_u32_at(wire, offset, row.summary_index); offset += 4;
        detail::sum_dir_pack_u32_at(wire, offset, row.summary_count); offset += 4;
        detail::sum_dir_pack_u64_at(wire, offset, row.summary_digest.lo); offset += 8;
        detail::sum_dir_pack_u64_at(wire, offset, row.summary_digest.hi); offset += 8;
        wire[offset++] = static_cast<std::uint8_t>(row.codec);
        wire[offset++] = static_cast<std::uint8_t>(row.state);
        detail::sum_dir_pack_u32_at(wire, offset, static_cast<std::uint32_t>(row.file_slot));
        offset += 4;
        // offset == 70 here, and 70..80 was dead space before this line: the writer stopped at 70
        // and the reader stopped at 70. Adding these 8 bytes brings the struct to the stride's full
        // 80 bytes and leaves `kSumDirRowWireBytes` and `kSumDirVersion` untouched.
        detail::sum_dir_pack_u64_at(wire, offset, row.frame_bits);
        offset += 8;
    }

    [[nodiscard]] static SumDirRow unpack_row(const std::uint8_t* wire) noexcept {
        SumDirRow row;
        row.block_identity.lo = detail::sum_dir_unpack_u64(wire, 0);
        row.block_identity.hi = detail::sum_dir_unpack_u64(wire, 8);
        row.token_begin       = detail::sum_dir_unpack_u32(wire, 16);
        row.token_end         = detail::sum_dir_unpack_u32(wire, 20);
        row.page              = detail::sum_dir_unpack_u32(wire, 24);
        row.generation        = detail::sum_dir_unpack_u32(wire, 28);
        row.content_first     = detail::sum_dir_unpack_u32(wire, 32);
        row.content_count     = detail::sum_dir_unpack_u32(wire, 36);
        row.summary_index     = detail::sum_dir_unpack_u32(wire, 40);
        row.summary_count     = detail::sum_dir_unpack_u32(wire, 44);
        row.summary_digest.lo = detail::sum_dir_unpack_u64(wire, 48);
        row.summary_digest.hi = detail::sum_dir_unpack_u64(wire, 56);
        row.codec             = static_cast<SumDirCodec>(wire[64]);
        row.state             = static_cast<SumDirState>(wire[65]);
        row.file_slot         = static_cast<std::int32_t>(detail::sum_dir_unpack_u32(wire, 66));
        // Offsets 70..80: written by `pack_row` above, and read here for the first time. A file
        // written before this field has zeros there, which is the "no frame position" reading.
        row.frame_bits        = detail::sum_dir_unpack_u64(wire, 70);
        return row;
    }

    std::uint64_t sequence_tag_ = 0;
    std::uint32_t block_tokens_ = kSumDirBlockTokens;
    bool sorted_ = true;
    std::vector<SumDirRow> rows_;
    std::vector<std::string> summaries_;
    std::vector<std::uint32_t> content_;
    // The block->row index (see `ensure_page_index`). Kept last so its declaration order does not
    // suggest it participates in the wire format -- it does not: nothing here is serialized.
    mutable std::vector<std::uint32_t> page_index_;
    mutable bool page_index_built_ = false;
    // The tripwire set: 1 for every page that two or more GENERATIONS cover. Dense over the same
    // extent `page_index_` is, and rebuilt by the same `ensure_page_index` pass, so it cannot fall
    // out of step with the address index it qualifies.
    mutable std::vector<std::uint8_t> page_conflict_;
    // The writer-aware index (see `ensure_page_generation_index`). LAZY: a caller that never asks a
    // tagged question never builds it.
    mutable std::unordered_map<std::uint64_t, std::uint32_t> page_generation_index_;
    mutable bool page_generation_index_built_ = false;
};

// ---------------------------------------------------------------------------
// THE FAN-OUT: "a term was found -> WHICH PAGES DO I ASK FOR?"
// ---------------------------------------------------------------------------
//
// This is the whole of step 4 of the retrieval wiring, and it lives HERE -- in the directory, as a
// pure function over the directory -- rather than inside the engine's provider, for one reason that
// is not style: a DECISION that can only be exercised by running the engine is a decision nothing can
// test. The engine's provider is a three-line adapter over this function (see
// `ProgramImplCore::index_request_for_term`), so the acceptance demo and the engine are not two
// implementations of the fan-out that might disagree -- they are one implementation with two callers.
// `sum_dir_unload_plan` was left in exactly the opposite state (implemented, zero callers), and that
// is the failure this factoring is designed not to repeat.
//
// WHAT IT ANSWERS, in the order the answers are tried:
//
//   1. EVERY WINDOW OF THE QUERY, NOT JUST THE SHRINKING TAIL. The catalogue line is PER BLOCK, so a
//      phrase that straddles a block boundary appears in NEITHER block's line -- measured, not
//      assumed: the first run of the acceptance demo printed "matches 0 block line(s)" for exactly
//      this shape. The windows that DO lie inside one block are the query's own PREFIXES and
//      SUFFIXES -- a phrase spanning pages A..B has some prefix inside A and the remainder inside B
//      -- so the windows tried are: the whole query, then `head-k` and `tail-k` for every k. The
//      FIRST implementation shrank only the tail, and the demo caught the consequence in numbers: for
//      a 40-token phrase whose split falls 12 tokens into page A it returned page B alone, and the
//      reassembled run did NOT contain the phrase. Both ends are needed, and the measurement is what
//      said so.
//
//   2. ADMISSIBILITY. A hit block is admissible only when its whole span lies at or below
//      `frontier` -- the sequence's committed-token count -- AND strictly below the QUERY's own first
//      token. The second half is what stops the search from FINDING THE QUESTION: the query is formed
//      from the newest committed tokens, so its own block is in the directory and would otherwise be
//      the nearest hit. Without it the retriever answers "the page you are standing on", a
//      plausible-looking wrong answer -- the worst kind. Excluding `end > frontier - query_tokens`
//      once, for every window, is sound because every window is a substring of the query, so the only
//      blocks a window can match by accident are the query's own.
//
//   3. THE CONTIGUOUS RUN over the UNION of every window's admissible pages, from the lowest page,
//      capped at `fanout_cap` blocks. A run rather than a single page because material that spans more
//      than one page must come back whole (「不能丢任何细节」): one page of a five-page passage is not
//      four fifths of an answer, it is a different one. A run rather than "every hit" because a plan
//      is a list of pages to move BEFORE the round runs and its cost is linear in that list;
//      unbounded would be the whole-prefix recall this design forbids by name.
//
//   4. THE END IS THE LAST BLOCK'S OWN END, from its row (`sum_dir_row_tokens`), never
//      `last_page * block_tokens + block_tokens`. For a document whose final block is partial --
//      which is every document -- the stride expression claims `block_tokens - count` tokens past the
//      real text, i.e. exactly the writer's zero padding. This is the same rule the cargo's
//      `read_located_block` applies, applied at the point where the span is chosen rather than after
//      it has been handed out.
//
// COST, MEASURED AND NAMED: the window sweep is bounded at `kSumDirMaxQueryWindows` k-values, and it
// STOPS EARLY the moment the accumulated pages form a run of two or more contiguous blocks -- which
// is exactly the cross-boundary answer arriving. Each window costs one `search_summaries` scan over
// every row, so the cost is O(windows x rows x term); the demo measures it (625 rows: tens of
// microseconds to about a millisecond per query). THE NEXT STEP, named so this is not mistaken for
// it: a real inverted index over block n-grams (postings n-gram -> block) removes the scan and makes
// each window O(1). `search_summaries`'s own contract permits exactly that replacement -- "a real
// retriever replaces this function's body, not its contract".
inline constexpr std::uint32_t kSumDirMaxQueryWindows = 24U;

struct SumDirRecallSpan {
    bool          found       = false;
    std::uint32_t window      = 0; // tokens in the longest window that hit
    std::size_t   hits        = 0; // raw search_summaries() hits, summed over the windows
    std::size_t   admissible  = 0; // hits that survived the frontier/self-match test
    std::uint32_t windows     = 0; // how many windows were actually searched
    std::uint32_t pages       = 0; // pages in the returned run
    std::uint32_t first_page  = 0;
    std::uint32_t last_page   = 0;
    std::uint32_t token_begin = 0;
    std::uint32_t token_end   = 0; // clamped to `frontier`
};

// The number of tokens in a rendered line: every token is delimited on both sides, so the count is
// exactly `separators - 1`. Kept as its own function because the producer's alphabet and the
// consumer's alphabet must be the same object, and this is the only place the line is parsed.
[[nodiscard]] inline std::uint32_t sum_dir_token_line_count(std::string_view line) noexcept {
    std::uint32_t separators = 0;
    for (const char byte : line) {
        if (byte == ' ') { ++separators; }
    }
    return separators == 0U ? 0U : separators - 1U;
}

// Drop the first `skip` tokens from a rendered line, landing exactly on a separator boundary (which
// is why a re-slice is safe and a character offset would not be). A `skip` at or past the end yields
// an empty line, which `search_summaries` treats as "no question" and never as "everything".
[[nodiscard]] inline std::string_view sum_dir_token_line_from(std::string_view line,
                                                              std::uint32_t skip) noexcept {
    if (skip == 0U) { return line; }
    std::uint32_t seen = 0;
    for (std::size_t index = 1; index < line.size(); ++index) {
        if (line[index] != ' ') { continue; }
        if (++seen == skip) { return line.substr(index); }
    }
    return std::string_view();
}

// Keep only the first `count` tokens of a rendered line, ending exactly on a separator boundary. The
// mirror of `sum_dir_token_line_from`, and the reason both exist is item 1 above: a phrase has a
// prefix inside one block AND a suffix inside another, so a search that can look at only one end of
// the phrase can only ever find one of the two blocks.
[[nodiscard]] inline std::string_view sum_dir_token_line_head(std::string_view line,
                                                             std::uint32_t count) noexcept {
    if (count == 0U) { return std::string_view(); }
    std::uint32_t seen = 0;
    for (std::size_t index = 1; index < line.size(); ++index) {
        if (line[index] != ' ') { continue; }
        if (++seen == count) { return line.substr(0, index + 1); }
    }
    return line; // the line holds `count` or fewer tokens, so the whole line is the head
}

[[nodiscard]] inline SumDirRecallSpan
sum_dir_recall_span_for_term(const SumDir& directory, std::string_view term, std::uint32_t frontier,
                             std::uint32_t fanout_cap,
                             std::uint32_t block_tokens = kSumDirBlockTokens) {
    SumDirRecallSpan span;
    const std::uint32_t query_tokens = sum_dir_token_line_count(term);
    if (query_tokens == 0U || block_tokens == 0U || fanout_cap == 0U) { return span; }
    // The query occupies [frontier - query_tokens, frontier). A block whose END reaches into that
    // range is a block the query covers, so it is not admissible -- for ANY window, because every
    // window is a substring of the query.
    const std::uint32_t query_begin = frontier > query_tokens ? frontier - query_tokens : 0U;

    std::vector<std::uint32_t> pages;
    std::size_t raw_hits = 0;
    std::size_t admissible = 0;
    std::uint32_t windows = 0;
    std::uint32_t longest_window = 0;
    bool head_found = false;
    bool tail_found = false;

    const auto harvest = [&](std::string_view window, bool* side) {
        if (window.empty()) { return; }
        const std::vector<std::size_t> hits = directory.search_summaries(window);
        ++windows;
        raw_hits += hits.size();
        const std::uint32_t window_tokens = sum_dir_token_line_count(window);
        for (const std::size_t row : hits) {
            const SumDirRow& hit = directory.rows()[row];
            const std::uint32_t tokens = sum_dir_row_tokens(hit);
            if (tokens == 0U) { continue; }
            const std::uint64_t end = static_cast<std::uint64_t>(hit.page) * block_tokens + tokens;
            if (end > frontier || end > query_begin) { continue; }
            ++admissible;
            pages.push_back(hit.page);
            if (window_tokens > longest_window) { longest_window = window_tokens; }
            if (side != nullptr) { *side = true; }
        }
    };

    // THE FULL WINDOW FIRST, AND IF IT HITS, THE ANSWER IS ALREADY COMPLETE. A full-window hit means
    // the WHOLE phrase was found inside one block's line, i.e. inside one page -- so there is nothing
    // for the sweep to add, and running it anyway would multiply the cost by 2*kSumDirMaxQueryWindows
    // for no gain. (The demo measured that cost before this early stop existed: 47 windows and 8.2 ms
    // for a query whose FIRST window had already answered.) The sweep exists only for the phrase that
    // crosses a boundary and therefore appears in no single line.
    harvest(term, nullptr);

    // THE SWEEP, LONGEST WINDOW FIRST. `k` DESCENDS, so the first hit found on either side is that
    // side's LONGEST usable window -- which is the most specific evidence available about where the
    // phrase's prefix (head side) and its suffix (tail side) each live. Ascending k would find the
    // shortest, least specific windows first: a single-token window matches every block that holds
    // that token, and a common token would then bracket the answer onto whatever the earliest of those
    // blocks is. Descending k costs the same number of searches and is strictly better evidence.
    //
    // It stops as soon as BOTH sides have hit -- at that point the phrase is bracketed from both ends
    // and further windows can only add weaker evidence -- and is bounded at kSumDirMaxQueryWindows
    // k-values regardless.
    if (pages.empty() && query_tokens >= 2U) {
        const std::uint32_t k_max =
            std::min<std::uint32_t>(query_tokens - 1U, kSumDirMaxQueryWindows);
        for (std::uint32_t k = k_max; k >= 1U; --k) {
            harvest(sum_dir_token_line_head(term, k), &head_found);
            harvest(sum_dir_token_line_from(term, query_tokens - k), &tail_found);
            if (head_found && tail_found) { break; }
            if (k == 1U) { break; } // an unsigned loop must not rely on wraparound
        }
    }
    if (pages.empty()) { return span; }

    // THE RUN: FROM THE FIRST HARVESTED PAGE TO THE LAST ONE, CAPPED. Stated as "first to last"
    // rather than "the maximal contiguous run of harvested pages" because the harvested set is not
    // guaranteed to be contiguous even when the passage is: a phrase spanning pages A..C shows up in
    // A's line (as a head window) and in C's line (as a tail window), and page B's line -- which holds
    // only the phrase's middle -- need not match at any window length. Requiring contiguity would then
    // return page A alone for a passage that covers A, B and C. The gap is FILLED instead, and the
    // fill is bounded by `fanout_cap`, which is precisely what that cap is for: the overshoot between
    // two far-apart matches can never exceed the per-pass budget. The user's requirement is that a
    // passage comes back WHOLE; a bounded, stated overshoot is the cheaper error than a truncated
    // passage, and the cap is what keeps it bounded.
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    const std::uint32_t first_page = pages.front();
    const std::uint32_t last_page  = std::min<std::uint32_t>(first_page + fanout_cap - 1U,
                                                            pages.back());
    const std::uint32_t run_pages  = last_page - first_page + 1U;
    const std::uint32_t begin_token = first_page * block_tokens;
    if (begin_token >= frontier) { return span; }
    const std::uint32_t last_tokens = directory.tokens_for_page(last_page);
    const std::uint32_t end_token =
        last_page * block_tokens + (last_tokens != 0U ? last_tokens : block_tokens);

    span.found       = true;
    span.window      = longest_window;
    span.hits        = raw_hits;
    span.admissible  = admissible;
    span.windows     = windows;
    span.pages       = run_pages;
    span.first_page  = first_page;
    span.last_page   = last_page;
    span.token_begin = begin_token;
    span.token_end   = std::min(end_token, frontier);
    return span;
}


// ---------------------------------------------------------------------------
// THE ENTRY POINT -- who writes the fact, who reads it, and where the verdict takes effect
// ---------------------------------------------------------------------------
//
// This is the whole chain in one call, so a caller has exactly ONE thing to call and each link of
// the chain is checkable:
//
//   WRITER 1  (pre-existing, untouched)  `SumDir::append()` derives `row.state` from `file_slot`
//                                        (`state = file_slot >= 0 ? Live : Dead`). Nothing in this
//                                        patch changes it: the byte axis is already on the row.
//   WRITER 2  (new)                      `sum_dir_byte_axis_present(row, policy)` -- the ONE place
//                                        the row's state becomes the fact the judge reads, and the
//                                        ONE place the strategy can change what is read.
//   READER    (new)                      `sum_dir_block_admissibility(facts)` -- the verdict.
//   EFFECTS   (new + pre-existing)       the verdict decides (a) whether a block is unloadable at
//                                        all -- the judge; (b) whether rule C1 puts it in an unload
//                                        text -- `sum_dir_unload_texts_offered`; (c) whether that
//                                        text keeps `byte_recallable`; (d) the price the audit
//                                        counts in `blocks_text_only` / `unload_texts_text_only`.
//
// WHERE IT IS *NOT* WIRED, NAMED SO IT IS NOT MISTAKEN FOR DONE: the engine-side call site. Nothing
// in the engine calls this yet -- the idle loop that would (`serve/kv_auto_relayout.cpp`, the
// "wake every N seconds and decide" tick this header's B4 already names) is an engine file this
// patch does not touch. And the two lines that make `state == Live` a *lookup gate* rather than a
// verdict -- `sum_dir_row_bound()`'s `if (row.state != SumDirState::Live) { return false; }` and
// `search_summaries()`'s `if (rows_[index].state != SumDirState::Live) { continue; }` -- are
// deliberately NOT changed here: they are a separate concern, and changing them in the same patch
// would make it impossible to tell which change a red check belongs to. With the DEFAULT policy
// this function agrees with both of them; that agreement is asserted in the self-test rather than
// assumed.
//

// THE BLOCK-TO-ROW MAP, stated because it is the one genuinely underspecified part: a row names the
// block it covers with its `page` column (`page == token_begin / kSumDirBlockTokens`), and a
// concatenated unload-text row names its FIRST block. So a directory that stores one row per 64-block
// gives a complete byte-axis reading, and a directory that stores only concatenated rows gives a
// reading for the first block of each run and "no row" for the rest. Both are reported honestly:
// "no row for this block" and "the row says Dead" are the same answer under two of the three
// policies and different answers under `IgnoreByteAxis`, which is why the fact writer takes a
// POINTER. Closing that gap (an explicit block->row index) is a design decision this patch leaves
// open on purpose rather than inventing a convention that the directory's own doctrine -- "a
// CONTENT SHORTLIST, not an identity" -- has no basis for.
// THE JUDGE, WITH AND WITHOUT THE FRAME AXIS -- ONE IMPLEMENTATION, TWO ENTRY POINTS.
//
// The axis arrives as two per-block bitmaps rather than as a new parameter on the facets:
//   * `frame_bits[block]`   -- SumDirRow::frame_bits, i.e. the block's own template/control column;
//   * `boundary_bits[block]`-- the block's anchor/checkpoint frontier column.
// The 4-argument entry point passes NEITHER, so for it `sum_dir_frame_facts_apply` is called with
// (0, 0, tokens) and returns exactly the facts it received -- `has_control_token = false`,
// `pinned_by_anchor = false`, `inside_system_prefix = false` for every non-empty block. That is
// byte-for-byte the behaviour this function had before the axis existed, which is what makes this
// refactor safe to land next to a live build: NO EXISTING CALLER'S OUTPUT MOVES.
//
// A bitmap vector SHORTER than `blocks_total` means "no axis for the blocks past the end" and those
// blocks keep the all-false facts. An absent axis therefore reads as "no frame position", which is a
// legitimate value and is made distinguishable from a present-but-zero one by the vector's SIZE --
// the same distinction the fact writer already draws with a POINTER for the byte axis (`:894-897`).
[[nodiscard]] inline SumDirUnloadPlan
sum_dir_unload_plan_core(const SumDir& directory, std::uint32_t token_count,
                         const std::vector<std::uint32_t>& sequence_tokens,
                         SumDirByteAxisPolicy policy, std::span<const std::uint64_t> frame_bits,
                         std::span<const std::uint64_t> boundary_bits) noexcept {
    SumDirUnloadPlan plan;
    plan.policy       = policy;
    plan.blocks_total = sum_dir_block_count(token_count);

    // The directory's own answer, per block. First writer wins, so a directory holding several rows
    // for one page is read deterministically instead of arbitrarily.
    std::vector<const SumDirRow*> row_of_block(plan.blocks_total, nullptr);
    for (const SumDirRow& row : directory.rows()) {
        if (row.page < plan.blocks_total && row_of_block[row.page] == nullptr) {
            row_of_block[row.page] = &row;
        }
    }

    // The two halves joined: the sequence decides content and the tail, the row decides the byte axis.
    // The frame axis joins them on the CONTENT side, which is the one the judge was blind to.
    std::vector<SumDirBlockFacts> facts(plan.blocks_total);
    for (std::uint32_t index = 0; index < plan.blocks_total; ++index) {
        SumDirBlockFacts block;
        block.content_present  = true; // by construction -- see `sum_dir_facts_from_sequence`
        block.overlap_frontier = (index == sum_dir_resident_block(token_count));
        block.byte_axis_present = sum_dir_byte_axis_present(row_of_block[index], policy);
        const std::uint64_t frame = index < frame_bits.size() ? frame_bits[index] : 0ULL;
        const std::uint64_t edge  = index < boundary_bits.size() ? boundary_bits[index] : 0ULL;
        const std::uint64_t begin = static_cast<std::uint64_t>(index) * kSumDirBlockTokens;
        const std::uint64_t stop  = std::min<std::uint64_t>(begin + kSumDirBlockTokens, token_count);
        const std::uint32_t tokens =
            begin >= token_count ? 0U : static_cast<std::uint32_t>(stop - begin);
        facts[index] = sum_dir_frame_facts_apply(block, frame, edge, tokens);
    }

    plan.verdicts = sum_dir_admissibility_report_from(facts);
    plan.texts    = sum_dir_unload_texts_offered(plan.verdicts, sequence_tokens, policy);

    if (policy == SumDirByteAxisPolicy::RefuseWhenAbsent) {
        for (std::uint32_t index = 0; index < plan.verdicts.size() && index < plan.blocks_total;
             ++index) {
            if (plan.verdicts[index] == SumDirAdmissibility::AdmissibleTextOnly) {
                ++plan.blocks_refused_no_bytes;
            }
        }
    }
    plan.blocks_offered = plan.blocks_total >= plan.blocks_refused_no_bytes
                              ? plan.blocks_total - plan.blocks_refused_no_bytes
                              : 0U;
    plan.cost = sum_dir_avl_audit(token_count, plan.verdicts, facts, plan.texts);
    return plan;
}

// THE UNCHANGED ENTRY POINT. Empty spans => the all-false frame facts => the plan this function
// produced before this landing, bit for bit.
[[nodiscard]] inline SumDirUnloadPlan
sum_dir_unload_plan(const SumDir& directory, std::uint32_t token_count,
                    const std::vector<std::uint32_t>& sequence_tokens,
                    SumDirByteAxisPolicy policy = kSumDirByteAxisPolicyDefault) noexcept {
    return sum_dir_unload_plan_core(directory, token_count, sequence_tokens, policy,
                                   std::span<const std::uint64_t>(), std::span<const std::uint64_t>());
}

// THE ENTRY POINT THAT CAN SEE THE BLOCK. `frame_bits` and `boundary_bits` are indexed by block
// number; supplying them is what makes `has_control_token` / `inside_system_prefix` /
// `pinned_by_anchor` non-constant, and therefore what makes three of the seven verdicts reachable
// by MEANING instead of by page arithmetic. See `sum_dir_frame_facts_apply` for the exact
// derivation and its honest boundary.
[[nodiscard]] inline SumDirUnloadPlan
sum_dir_unload_plan(const SumDir& directory, std::uint32_t token_count,
                    const std::vector<std::uint32_t>& sequence_tokens, SumDirByteAxisPolicy policy,
                    const std::vector<std::uint64_t>& frame_bits,
                    const std::vector<std::uint64_t>& boundary_bits) noexcept {
    return sum_dir_unload_plan_core(directory, token_count, sequence_tokens, policy, frame_bits,
                                   boundary_bits);
}

} // namespace ninfer::spec::sum_dir
