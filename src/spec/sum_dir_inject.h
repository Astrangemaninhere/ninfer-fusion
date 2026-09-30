#pragma once

// src/spec/sum_dir_inject.h -- THE BINDING: a `sum_dir` ROW becomes CONTEXT.
//
// WHY THIS FILE EXISTS, in the record's own words. `src/spec/sum_dir.h:98` names the blocker it is
// not allowed to assume away:
//
//   "ETIQUETTE TOWARD 95-3 (injected recall must look to the model like ordinary context)"
//   "Nothing in this file is ever rendered into a prompt. A row is host-side bookkeeping, so the
//    rule does not bind this header; it binds the future patch that turns a row back into tokens,
//    which is listed as a blocker in the report rather than assumed away here."
//
// `dl/injectchan` (F-738) turned that blocker into an INGRESS: `src/spec/inject_channel.h` declares
// a chosen tensor and the position range it will occupy, `inject_ingress.h` writes it into the
// input-embedding matrix `x` immediately after `ops::embedding`, and under `dtype=bf16, scale=1`
// the write is arithmetic-free -- measured as the identity over all 65,536 bf16 patterns, 0
// differing -- so the etiquette requirement is met as BYTE EQUALITY and not as a tolerance. And it
// said, by name, what it did not do:
//
//   "No `sum_dir` row wired to the channel; the wall is open, the binding is the next landing."
//
// This header IS that binding, and it is deliberately shaped as a CALLER of the ingress rather than
// as a second ingress. It builds one `inject::Declaration` -- the same struct, the same fields --
// and hands it to `inject::admit_against_engine()` and `inject::bake()`, the ingress's OWN two
// functions. That choice is the whole reason the reuse table below is short and true: every
// declaration-side refusal a row-borne binding can provoke is provoked BY THE INGRESS'S CODE, not
// by a re-implementation of it. `src/targets/qwen3_6/impl/runtime/inject_ingress.h` is not touched.
//
// ---------------------------------------------------------------------------------------------
// THE POSITION RULE, IN `sum_dir.h:88-96`'s OWN TERMS, AND WHY IT IS NOT MINE TO CHANGE
// ---------------------------------------------------------------------------------------------
//
//   "IDENTITY IS CONTENT, NOT POSITION. A recalled block is rebuilt at a NEW position, so nothing
//    positional may enter `block_identity`: not the block index, not the absolute token offset, not
//    the sequence, not the device cold slot, not the spill file slot, not the wall time at which
//    the catalogue line was written."
//   "The self-test PROVES the position independence (the same content at two offsets has equal
//    identity; one token anywhere in the block changed has a different identity) instead of
//    asserting it in a comment."
//
// A binding that turned a row into a declaration would be exactly the place that rule could be
// broken, because a declaration has a `position0` and a row has a `token_begin`. The rule says the
// first is the caller's free choice ("rebuilt at a NEW position") and the second is BOOKKEEPING
// (`sum_dir.h:453-456`: "this is the caller's token space and is NOT part of the identity"). So this
// header:
//
//   * takes `position0` from the CALLER and never derives it from the row -- not from `token_begin`,
//     not from `page`, not from `file_slot`, not from `generation`;
//   * CARRIES `token_begin`/`token_end` into the binding record and reports them, marked
//     BOOKKEEPING, and puts them in NO digest, NO payload and NO declaration field;
//   * recomputes `sum_dir_block_digest(row's content tokens)` and REFUSES the binding when it does
//     not equal `row.block_identity` (`refused-row-content-digest-mismatch`) -- the rule is not
//     asserted here, it is CHECKED, from the content, on every binding;
//   * exposes `row_binding_position_self_test()`, which is `sum_dir.h`'s own self-test in the
//     binding's terms: the same row bound at two positions has equal identity, a byte-equal
//     payload and declarations equal except for `position0`, while a one-token change moves the
//     identity.
//
// `sum_dir_vector.h:361` states the consequence this header carries one level up: two copies of one
// block at different offsets "receive the SAME key and are one target". So `same_target()` is
// identity equality with the position IGNORED, and two bindings that are one target MUST carry one
// payload -- `refused-row-same-identity-different-payload` is what makes that a check rather than a
// remark. (Under `dtype=bf16, scale=1` the payload is `embed(tokens)` and nothing else, so
// position-free content implies byte-identical bytes; the refusal is the guard for the day it is
// not.)
//
// ---------------------------------------------------------------------------------------------
// THE PAYLOAD: THE ENGINE'S OWN BYTES, PINNED
// ---------------------------------------------------------------------------------------------
//
// A row holds token ids. The alphabet of the ingress is the INPUT-EMBEDDING SPACE, so the payload
// is `embed(row's tokens)` -- and the only honest source of those bytes in this tree is the
// engine's own `direction=egress`, which is `dl/injectchan`'s own definition of "equivalent to
// tokens the engine could have computed" ("that is not a host re-implementation of the embedding
// table"). A row-borne binding therefore REQUIRES the payload to be PINNED (`digest=`, in the
// ingress's own spelling): a binding to "whatever file is at that path now" is the silent-corruption
// shape this record keeps finding, and it is refused by name before anything is read.
//
// The consequence is a checkable JOIN, and it is the observability requirement met:
//
//   binder record:  identity=<row's block_identity> ingested_digest=<bake()'s baked_digest>
//   engine stderr:  [inject] ADMITTED ... source_digest=... ingested_digest=<the same field>
//
// `baked_digest` is over the bytes that will enter `x`, and it is the field the engine PRINTS
// (inject_ingress.h, the ADMITTED/COMPLETE lines), so the two records are joined by a measurement
// and not by a filename. If the file moved between the bind and the run, the engine itself refuses
// it (`refused-source-digest-mismatch`, the ingress's own name).
//
// ---------------------------------------------------------------------------------------------
// THE TWO NAME SPACES, AND WHY THERE ARE TWO
// ---------------------------------------------------------------------------------------------
//
// The declaration-side names -- the 24 in `inject::refusal_name()` -- are REUSED, by calling the
// ingress's code. The row-side names in this file are NEW, because they are about a ROW and not
// about a declaration: which row, its state, its content, its length, its identity. They are
// spelled ONCE, here, in `row_refusal_name()`, and every one is rendered with the `row-` prefix and
// with the FIELD and the NUMBERS it is about.
//
// ⚠ WHY NOT IN `inject_channel.h`. `dl/injectchan`'s caliber enumerates its `Refusal` enum and
// REQUIRES every name to be provoked by a case, so a name no case can reach FAILS the test as a
// dead guard (that is the discipline that made `refused-position-negative` and
// `refused-columns-zero` live rather than decorative). Appending row names to that enum would
// therefore BREAK their 24-of-24 population check by construction. Two enumerations, one per name
// space, each with its own population check, is the shape that keeps both calibers honest.
//
// This file is std-only and host-only -- it includes `spec/sum_dir.h` and `spec/inject_channel.h`,
// both of which state the same discipline for themselves -- so the WHOLE binding is testable with
// plain g++ and no device, no cmake and no CUDA.

#include "spec/inject_channel.h"
#include "spec/sum_dir.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::spec::sum_dir_inject {

namespace inject = ninfer::spec::inject;

// `sum_dir_inject` is a SIBLING of `sum_dir`, not a child, so the directory's own vocabulary is
// named once here rather than qualified at every use. Nothing is re-declared: these are the
// directory header's own types and functions.
namespace sum_dir = ninfer::spec::sum_dir;
using sum_dir::SumDir;
using sum_dir::SumDirCodec;
using sum_dir::SumDirDigest;
using sum_dir::SumDirRow;
using sum_dir::SumDirState;
using sum_dir::kSumDirBlockTokens;
using sum_dir::kSumDirNoSummary;
using sum_dir::sum_dir_block_digest;
using sum_dir::sum_dir_row_tokens;
using sum_dir::sum_dir_row_well_formed;
using sum_dir::sum_dir_state_name;

// ---------------------------------------------------------------------------
// THE ROW-SIDE REFUSAL VOCABULARY (NEW -- 10 names, none of them one of the ingress's 24)
// ---------------------------------------------------------------------------
//
// Each name answers a question a DECLARATION cannot be asked, and each is provokable: the caliber
// drives one case per name and compares the SET it produced against this enumeration, so a name no
// case can reach fails the test as a dead guard.
enum class RowRefusal : std::uint8_t {
    None = 0,
    // The caller named no row at all. A "binding" with no row behind it is an unchecked claim, and
    // it is the one failure that cannot be expressed as a field of anything.
    RefusedRowNoRowSource,
    // The row index is outside the directory. (A directory of 3 rows asked for row 3.)
    RefusedRowOutOfRange,
    // `sum_dir_row_well_formed()` says false: a half-written row -- an index without a length, an
    // identity unset, a codec the directory will not admit.
    RefusedRowNotWellFormed,
    // `state != Live`. A recycled slot is Dead and is not a hit (`sum_dir.h:406-411`, and the rule
    // `sum_dir_row_bound()` applies at `:632`).
    RefusedRowNotLive,
    // The row's own content window is not inside the directory's content column, or the row's token
    // count is 0, so the row has no bytes to stand for. `SumDir::block_content()` returns an EMPTY
    // vector in the first case (`sum_dir.h:1968-1974`) and a caller that reads an empty block as
    // "nothing to inject" would inject nothing and report an admission.
    RefusedRowContentAbsent,
    // The length the caller declared for the block differs from the row's OWN token count
    // (`sum_dir_row_tokens`). The payload's column count IS the block's length: a payload sized for
    // one block bound to a row of another length is a permutation of the right bytes.
    RefusedRowLengthMismatch,
    // `sum_dir_block_digest(row's content tokens) != row.block_identity`. THE CONTENT-NOT-POSITION
    // CHECK, recomputed from the bytes the row points at, on every binding.
    RefusedRowContentDigestMismatch,
    // The payload was not pinned (`inject::Declaration::has_digest` false). A row-borne declaration
    // whose payload is "whatever is at that path now" cannot be re-run and cannot be audited.
    RefusedRowPayloadUnpinned,
    // Two bindings whose position ranges INTERSECT and whose identities DIFFER: which bytes win
    // would depend on write order, which is not a precedence rule anybody stated -- the same refusal
    // shape the ingress applies to a vision scatter destination, one level up. (Two bindings that
    // are ONE TARGET -- equal identity, therefore byte-equal payload -- over intersecting ranges are
    // the SAME BYTES twice and are ADMITTED; `duplicate_writes` counts them.)
    RefusedRowRangeOverlapsPlan,
    // Two bindings with the SAME identity (`same_target()`) and DIFFERENT ingested digests. One
    // target is one block's worth of content (`sum_dir_vector.h:361`), so two payloads claiming it
    // are two different contents wearing one key, and the identity is then not content.
    RefusedRowSameIdentityDifferentPayload,
};

inline constexpr std::size_t kRowRefusalCount = 10U;

[[nodiscard]] inline const char* row_refusal_name(RowRefusal refusal) noexcept {
    switch (refusal) {
    case RowRefusal::None: return "admitted";
    case RowRefusal::RefusedRowNoRowSource: return "refused-row-no-row-source";
    case RowRefusal::RefusedRowOutOfRange: return "refused-row-out-of-range";
    case RowRefusal::RefusedRowNotWellFormed: return "refused-row-not-well-formed";
    case RowRefusal::RefusedRowNotLive: return "refused-row-not-live";
    case RowRefusal::RefusedRowContentAbsent: return "refused-row-content-absent";
    case RowRefusal::RefusedRowLengthMismatch: return "refused-row-length-mismatch";
    case RowRefusal::RefusedRowContentDigestMismatch: return "refused-row-content-digest-mismatch";
    case RowRefusal::RefusedRowPayloadUnpinned: return "refused-row-payload-unpinned";
    case RowRefusal::RefusedRowRangeOverlapsPlan: return "refused-row-range-overlaps-plan";
    case RowRefusal::RefusedRowSameIdentityDifferentPayload:
        return "refused-row-same-identity-different-payload";
    }
    return "refused-row-unknown";
}

// The enumeration, in one place, so a population check can be written against the NAMES rather
// than against a list typed a second time (the same instrument `dl/injectchan` used for its 24).
[[nodiscard]] inline std::vector<RowRefusal> all_row_refusals() {
    std::vector<RowRefusal> names;
    names.reserve(kRowRefusalCount);
    for (std::uint8_t index = 1U; index <= kRowRefusalCount; ++index) {
        names.push_back(static_cast<RowRefusal>(index));
    }
    return names;
}

// A refusal a caller cannot read is a silent failure with a stderr line, so every one carries its
// NAME, the FIELD it is about, and the NUMBERS.
[[nodiscard]] inline std::string render_row_refusal(RowRefusal refusal, std::string_view field,
                                                    std::string_view detail) {
    std::string line = "REFUSED ";
    line += row_refusal_name(refusal);
    if (!field.empty()) {
        line += " field=";
        line += field;
    }
    if (!detail.empty()) {
        line += " -- ";
        line += detail;
    }
    return line;
}

// ---------------------------------------------------------------------------
// THE ROW AS A VALUE: two spellings, ONE validator
// ---------------------------------------------------------------------------
//
// A caller may hold the row because it read a DIRECTORY (the bytes the engine persists,
// `SumDir::serialize()` / `SumDir::deserialize()`) or because it has the row's fields and the
// block's token ids in hand. Both end in the same `RowView`, and `bind_row()` below cannot tell
// which spelling it was given -- so a rule cannot be enforced on one path and not the other.
struct RowView {
    SumDirRow row{};
    // The block's token ids, in order, exactly as the row's content window holds them.
    std::vector<std::uint32_t> content;
    // Which spelling this is. A caller that forgot to fill either one says so, rather than
    // silently binding against a default-constructed row (whose identity is unset and which
    // `sum_dir_row_well_formed` would refuse -- but as "not well formed", which is a different
    // statement from "you named no row").
    bool from_directory = false;
    bool from_fields    = false;
    // A refusal the SOURCE produced while building the view, carried into `bind_row` so that a
    // caller cannot read the refusal and then forget to act on it: the binding reports it as its
    // own. Out-of-range is the one that lives here, because the DIRECTORY is what knows its extent.
    RowRefusal  source_refusal = RowRefusal::None;
    std::string source_field;
    std::string source_detail;

    [[nodiscard]] bool valid_source() const noexcept {
        return from_directory || from_fields || source_refusal != RowRefusal::None;
    }
};

// (A) A row read out of the DIRECTORY, with the bytes the directory itself holds for it.
[[nodiscard]] inline RowView row_view_from_directory(const SumDir& directory, std::size_t index) {
    RowView view;
    view.from_directory = true;
    if (index >= directory.size()) {
        view.from_directory  = false;
        view.source_refusal  = RowRefusal::RefusedRowOutOfRange;
        view.source_field    = "row";
        view.source_detail   = "row " + std::to_string(index) + "; the directory holds " +
                               std::to_string(directory.size()) + " row(s)";
        return view;
    }
    view.row     = directory.rows()[index];
    view.content = directory.block_content(index);
    return view;
}

// (B) A row supplied as fields, with the block's token ids supplied alongside. Nothing here is
// trusted: `bind_row()` recomputes the identity from `content` and refuses when it disagrees with
// `row.block_identity`.
[[nodiscard]] inline RowView row_view_from_fields(const SumDirRow& row,
                                                  std::span<const std::uint32_t> content) {
    RowView view;
    view.row         = row;
    view.content.assign(content.begin(), content.end());
    view.from_fields = true;
    return view;
}

[[nodiscard]] inline RowView row_view_from_fields(const SumDirRow& row,
                                                  const std::vector<std::uint32_t>& content) {
    return row_view_from_fields(row, std::span<const std::uint32_t>(content));
}

// ---------------------------------------------------------------------------
// WHAT THE CALLER PINS ABOUT THE PAYLOAD
// ---------------------------------------------------------------------------
//
// Everything here is a DECLARATION field, so it travels to the engine unchanged. `digest` has no
// default and `has_digest` is required to be true: see the header comment above.
struct PayloadPin {
    inject::ElementType dtype = inject::ElementType::Bf16;
    double scale              = 1.0;
    std::string path;
    bool has_digest           = false;
    std::uint64_t digest      = 0;
    // 0 = "take the length from the row". A non-zero value is the caller's own claim about the
    // block's length, and it is checked against the row's (`refused-row-length-mismatch`) -- which
    // is what makes the refusal provokable rather than unreachable.
    std::uint32_t declared_block_tokens = 0;
};

// ---------------------------------------------------------------------------
// THE BINDING RESULT -- and it is also the observability record
// ---------------------------------------------------------------------------
struct RowBinding {
    bool bound = false;

    // (1) The declaration, exactly the struct the ingress consumes.
    inject::Declaration declaration;
    // (2) The ingress's OWN two verdicts, carried rather than summarised: `settlement` from
    //     `admit_against_engine()`, `payload` from `bake()` (which carries the two digests).
    inject::Settlement settlement;
    inject::PayloadSettlement payload;
    // (3) The row-side verdict, when the binding failed before a declaration could exist.
    RowRefusal refusal = RowRefusal::None;
    std::string field;
    std::string detail;

    // ---- the provenance record: what was bound, and how the identity survived it -------------
    std::size_t row_index = 0;   // meaningful only when the source was a directory
    bool from_directory   = false;
    SumDirDigest block_identity{};   // the row's CONTENT identity (position-free)
    std::uint32_t block_tokens   = 0;
    // BOOKKEEPING, reported and never used: `sum_dir.h:453-456`. They appear on the record so a
    // reader can see that the new position is not derived from them.
    std::uint32_t source_token_begin = 0;
    std::uint32_t source_token_end   = 0;
    // The position the caller CHOSE (as requested, sign included), and whether it differs from the
    // row's own span.
    std::int32_t requested_position0 = 0;
    bool position_is_new             = false;
    // Set when this binding is the SAME BYTES as an earlier one in the plan (one target, one
    // payload, intersecting ranges): permitted, counted, and named.
    bool duplicate_writes = false;
    // The identity of the row as RECOMPUTED from the content bytes the row points at.
    SumDirDigest recomputed_identity{};

    [[nodiscard]] bool admitted() const noexcept {
        return bound && refusal == RowRefusal::None && settlement.admitted() && payload.settled();
    }

    // THE ONE RENDERER OF THE BINDING, the counterpart of the ingress's ADMITTED line. It carries
    // every field of the declaration, both digests, and the row provenance the declaration cannot
    // hold.
    [[nodiscard]] std::string line() const {
        if (!admitted()) {
            if (refusal != RowRefusal::None) {
                return "[rowbind] " + render_row_refusal(refusal, field, detail);
            }
            if (!settlement.admitted()) {
                return "[rowbind] REFUSED " + std::string(inject::refusal_name(settlement.refusal)) +
                       " field=" + settlement.field + " -- " + settlement.detail;
            }
            return "[rowbind] REFUSED " + std::string(inject::refusal_name(payload.refusal)) +
                   " field=" + payload.field + " -- " + payload.detail;
        }
        char buffer[640];
        std::snprintf(buffer, sizeof(buffer),
                      "[rowbind] BOUND row=%zu identity=0x%016llx:0x%016llx block_tokens=%u "
                      "source_span=[%u..%u)(BOOKKEEPING) new_positions=[%d..%d) "
                      "position_is_new=%d dtype=%s layout=%s rows=%d cols=%d "
                      "scale=%.6f bytes=%zu source_digest=fnv1a64:0x%016llx "
                      "ingested_digest=fnv1a64:0x%016llx %s",
                      row_index, static_cast<unsigned long long>(block_identity.lo),
                      static_cast<unsigned long long>(block_identity.hi), block_tokens,
                      source_token_begin, source_token_end, declaration.position0,
                      declaration.last_position(), position_is_new ? 1 : 0,
                      inject::dtype_spelling(declaration.dtype), declaration.layout.c_str(),
                      declaration.rows, declaration.cols, declaration.scale,
                      payload.payload.size() * sizeof(std::uint16_t),
                      static_cast<unsigned long long>(payload.source_digest),
                      static_cast<unsigned long long>(payload.baked_digest),
                      duplicate_writes ? "duplicate_writes=1" : "duplicate_writes=0");
        return buffer;
    }
};

// THE "ONE TARGET" PREDICATE. `sum_dir_vector.h:361`: two copies of one block at different offsets
// "receive the SAME key and are one target". The key IS `block_identity`, so the target is the
// identity with the position IGNORED -- which is why this function does not read `position0`.
[[nodiscard]] inline bool same_target(const RowBinding& a, const RowBinding& b) noexcept {
    return a.block_identity == b.block_identity;
}

// ---------------------------------------------------------------------------
// THE ONE BINDER
// ---------------------------------------------------------------------------
//
// Order is part of the contract: the row's own fields are checked before a declaration is built,
// the declaration is checked by the INGRESS before a byte is read, and the plan is consulted last
// so that a binding which is wrong on its own terms never reports a plan conflict as its reason.
//
//   source          the row and its content (one of the two spellings above)
//   rows            the engine's hidden width (supplied by the engine, like ModelLimits)
//   limits          the ingress's own ModelLimits
//   new_position0   THE CALLER'S CHOICE. Never derived from the row.
//   pin             the payload and its mandatory digest
//   plan            the bindings already made for this run (may be empty)
[[nodiscard]] inline RowBinding bind_row(const RowView& source, std::int32_t rows,
                                         const inject::ModelLimits& limits,
                                         std::int32_t new_position0, const PayloadPin& pin,
                                         std::span<const RowBinding> plan = {}) {
    RowBinding out;
    out.requested_position0 = new_position0;
    out.from_directory      = source.from_directory;

    // A refusal the SOURCE already made (an index outside the directory) is this binding's
    // refusal: the view carries it so it cannot be dropped between the two calls.
    if (source.source_refusal != RowRefusal::None) {
        out.refusal = source.source_refusal;
        out.field   = source.source_field;
        out.detail  = source.source_detail;
        return out;
    }

    if (!source.valid_source()) {
        out.refusal = RowRefusal::RefusedRowNoRowSource;
        out.field   = "row";
        out.detail  = "neither a directory row nor a row's own fields were supplied";
        return out;
    }

    if (!sum_dir_row_well_formed(source.row)) {
        out.refusal = RowRefusal::RefusedRowNotWellFormed;
        out.field   = "block_identity";
        out.detail  = "the row is not well formed: identity_unset=" +
                      std::string(source.row.block_identity.is_unset() ? "1" : "0") +
                      " codec=" + std::to_string(static_cast<unsigned>(source.row.codec)) +
                      " summary_index=" + std::to_string(source.row.summary_index) +
                      " summary_count=" + std::to_string(source.row.summary_count) +
                      " file_slot=" + std::to_string(source.row.file_slot);
        return out;
    }

    if (source.row.state != SumDirState::Live) {
        out.refusal = RowRefusal::RefusedRowNotLive;
        out.field   = "state";
        out.detail  = "the row is " + std::string(sum_dir_state_name(source.row.state)) +
                      "; a recycled slot is not a hit";
        return out;
    }

    out.source_token_begin = source.row.token_begin;
    out.source_token_end   = source.row.token_end;
    out.block_identity     = source.row.block_identity;
    out.block_tokens       = sum_dir_row_tokens(source.row);
    out.position_is_new    = new_position0 >= 0 &&
                             static_cast<std::uint32_t>(new_position0) != source.row.token_begin;

    // The row's own bytes. `block_content()` returns an EMPTY vector when the window does not fit,
    // which is why the count -- and not `empty()` -- is the predicate.
    if (out.block_tokens == 0 || source.content.size() != out.block_tokens) {
        out.refusal = RowRefusal::RefusedRowContentAbsent;
        out.field   = "content";
        out.detail  = "the row names " + std::to_string(out.block_tokens) + " token(s) but " +
                      std::to_string(source.content.size()) +
                      " are available; content_count=" + std::to_string(source.row.content_count) +
                      " content_first=" + std::to_string(source.row.content_first);
        return out;
    }

    if (pin.declared_block_tokens != 0 && pin.declared_block_tokens != out.block_tokens) {
        out.refusal = RowRefusal::RefusedRowLengthMismatch;
        out.field   = "block_tokens";
        out.detail  = "the caller declared " + std::to_string(pin.declared_block_tokens) +
                      " token(s) for the block; the row holds " +
                      std::to_string(out.block_tokens) + " (token_begin=" +
                      std::to_string(source.row.token_begin) + " token_end=" +
                      std::to_string(source.row.token_end) + ")";
        return out;
    }

    // THE CONTENT-NOT-POSITION CHECK. Not asserted, recomputed from the tokens the row points at.
    out.recomputed_identity = sum_dir_block_digest(source.content);
    if (!(out.recomputed_identity == source.row.block_identity)) {
        out.refusal = RowRefusal::RefusedRowContentDigestMismatch;
        out.field   = "block_identity";
        char recomputed[64];
        std::snprintf(recomputed, sizeof(recomputed), "0x%016llx:0x%016llx",
                      static_cast<unsigned long long>(out.recomputed_identity.lo),
                      static_cast<unsigned long long>(out.recomputed_identity.hi));
        out.detail = std::string("the row's content digests to ") + recomputed + ", not the row's " +
                     "block_identity -- the row's content window no longer stands for its identity";
        return out;
    }

    if (!pin.has_digest) {
        out.refusal = RowRefusal::RefusedRowPayloadUnpinned;
        out.field   = "digest";
        out.detail  = "a row-borne declaration must pin the payload's bytes; '" + pin.path +
                      "' was named with no digest";
        return out;
    }

    // THE DECLARATION. `cols` IS the row's token count -- the block's length is the payload's
    // column count, and it is read from the row, not chosen.
    out.declaration.direction = inject::Direction::Ingest;
    out.declaration.dtype     = pin.dtype;
    out.declaration.layout    = std::string(inject::kIngestLayout);
    out.declaration.rows      = rows;
    out.declaration.cols      = static_cast<std::int32_t>(out.block_tokens);
    out.declaration.position0 = new_position0;
    out.declaration.scale     = pin.scale;
    out.declaration.path      = pin.path;
    out.declaration.has_digest = true;
    out.declaration.digest    = pin.digest;

    // THE INGRESS'S OWN TWO FUNCTIONS. Not a second spelling of their rules.
    out.settlement = inject::admit_against_engine(out.declaration, limits);
    if (!out.settlement.admitted()) { return out; }
    out.payload = inject::bake(out.declaration);
    if (!out.payload.settled()) { return out; }

    // THE PLAN. Two refusals, from the two sides of `same_target()`.
    for (const RowBinding& earlier : plan) {
        if (!earlier.bound) { continue; }
        if (same_target(earlier, out)) {
            if (earlier.payload.baked_digest != out.payload.baked_digest) {
                out.refusal = RowRefusal::RefusedRowSameIdentityDifferentPayload;
                out.field   = "digest";
                char other[64];
                std::snprintf(other, sizeof(other), "fnv1a64:0x%016llx",
                              static_cast<unsigned long long>(earlier.payload.baked_digest));
                out.detail = std::string("two payloads claim one block_identity (they are one ")
                             + "target, sum_dir_vector.h:361): this one ingests " + other +
                             ", the earlier binding ingested 0x" + std::to_string(out.payload.baked_digest);
                return out;
            }
            // ONE TARGET, ONE PAYLOAD, intersecting ranges: the SAME BYTES twice. Admitted, and
            // counted -- it is idempotent, so there is no write order to depend on.
            const std::int32_t a0 = earlier.declaration.position0;
            const std::int32_t a1 = earlier.declaration.last_position();
            if (a0 <= out.declaration.last_position() && out.declaration.position0 <= a1) {
                out.duplicate_writes = true;
            }
            continue;
        }
        const std::int32_t b0 = earlier.declaration.position0;
        const std::int32_t b1 = earlier.declaration.last_position();
        if (b0 <= out.declaration.last_position() && out.declaration.position0 <= b1) {
            out.refusal = RowRefusal::RefusedRowRangeOverlapsPlan;
            out.field   = "position0";
            out.detail  = "range [" + std::to_string(out.declaration.position0) + ", " +
                          std::to_string(out.declaration.last_position()) +
                          "] intersects an earlier binding's [" + std::to_string(b0) + ", " +
                          std::to_string(b1) +
                          "] for a DIFFERENT block; which bytes win would depend on write order";
            return out;
        }
    }

    out.bound = true;
    return out;
}

// ---------------------------------------------------------------------------
// THE POSITION-INDEPENDENCE SELF-TEST, in the binding's own terms
// ---------------------------------------------------------------------------
//
// `sum_dir.h:94-96` requires the proof to exist rather than the assertion. The header's own
// self-test proves it for `block_identity`; this one proves it for the OBJECT THE BINDING ACTUALLY
// HANDS THE ENGINE -- the declaration -- because that is where a position could sneak in:
//
//   * the same row bound at two DIFFERENT positions has EQUAL identity (the row's content is what
//     it is, wherever it is rebuilt);
//   * the two payloads are BYTE-EQUAL (the bytes are `embed(tokens)`; position is not an input);
//   * the two declarations are EQUAL IN EVERY FIELD EXCEPT `position0`;
//   * one token changed ANYWHERE in the block moves the identity (the other half of the sentence).
struct PositionIndependenceResult {
    bool holds = false;
    std::uint32_t bound_positions = 0;
    bool identity_equal          = false;
    bool payload_byte_equal      = false;
    bool ingested_digest_equal   = false;
    bool declarations_equal_except_position0 = false;
    bool one_token_change_moves_identity     = false;
    std::string detail;
};

[[nodiscard]] inline bool declarations_equal_except_position0(const inject::Declaration& a,
                                                              const inject::Declaration& b) {
    return a.direction == b.direction && a.dtype == b.dtype && a.layout == b.layout &&
           a.rows == b.rows && a.cols == b.cols && a.scale == b.scale && a.path == b.path &&
           a.has_digest == b.has_digest && a.digest == b.digest;
}

// `variant` is the row's content with ONE token changed, supplied by the caller so this function
// does not decide what "changed" means. Its identity must differ.
[[nodiscard]] inline PositionIndependenceResult row_binding_position_self_test(
    const RowView& source, std::int32_t rows, const inject::ModelLimits& limits,
    const PayloadPin& pin, std::int32_t position_a, std::int32_t position_b, const RowView& variant) {
    PositionIndependenceResult result;

    const RowBinding a = bind_row(source, rows, limits, position_a, pin);
    const RowBinding b = bind_row(source, rows, limits, position_b, pin);
    if (!a.admitted() || !b.admitted()) {
        result.detail = "the self-test could not bind: a=" + a.line() + " b=" + b.line();
        return result;
    }
    result.bound_positions = 2;
    result.identity_equal  = a.block_identity == b.block_identity;
    result.payload_byte_equal = a.payload.payload == b.payload.payload;
    result.ingested_digest_equal = a.payload.baked_digest == b.payload.baked_digest;
    result.declarations_equal_except_position0 =
        declarations_equal_except_position0(a.declaration, b.declaration) &&
        a.declaration.position0 != b.declaration.position0;

    const RowBinding c = bind_row(variant, rows, limits, position_a, pin);
    // The variant must be a DIFFERENT block: binding it must fail the content-digest check (its
    // content no longer hashes to the identity the row carries) -- which is exactly the statement
    // "one token anywhere in the block changed has a different identity", made by the same code
    // that makes it in production.
    result.one_token_change_moves_identity =
        !c.bound && c.refusal == RowRefusal::RefusedRowContentDigestMismatch;

    result.holds = result.identity_equal && result.payload_byte_equal &&
                   result.ingested_digest_equal && result.declarations_equal_except_position0 &&
                   result.one_token_change_moves_identity;
    if (!result.holds) {
        result.detail = "identity_equal=" + std::string(result.identity_equal ? "1" : "0") +
                        " payload_byte_equal=" + std::string(result.payload_byte_equal ? "1" : "0") +
                        " ingested_digest_equal=" +
                        std::string(result.ingested_digest_equal ? "1" : "0") +
                        " declarations_equal_except_position0=" +
                        std::string(result.declarations_equal_except_position0 ? "1" : "0") +
                        " one_token_change_moves_identity=" +
                        std::string(result.one_token_change_moves_identity ? "1" : "0");
    }
    return result;
}

// ---------------------------------------------------------------------------
// EMITTING THE DECLARATION THE INGRESS CONSUMES
// ---------------------------------------------------------------------------
//
// The binding's output IS a spec file: the operator then runs the engine with
// `--inject-spec <file>`, which parses it with `inject::parse_spec_file()` -- the same grammar the
// binder writes here. So the binder cannot write a declaration the engine would refuse to parse
// without the caliber noticing (the caliber round-trips every emitted spec through that parser).
//
// The row provenance rides as `#` comment lines: the parser SKIPS them (inject_channel.h, "`#`
// comment lines, blank lines skipped"), so they cost the engine nothing, and they make the one
// artifact the engine consumes also the durable record of WHICH ROW it came from.
[[nodiscard]] inline std::string format_spec_text(const RowBinding& binding) {
    if (!binding.admitted()) { return std::string(); }
    std::string text;
    text += "# rowbind (src/spec/sum_dir_inject.h): a sum_dir ROW bound to the inject channel\n";
    text += "# row=" + std::to_string(binding.row_index) + " from_directory=" +
            std::string(binding.from_directory ? "1" : "0") + "\n";
    {
        char identity[96];
        std::snprintf(identity, sizeof(identity), "# block_identity=0x%016llx:0x%016llx\n",
                      static_cast<unsigned long long>(binding.block_identity.lo),
                      static_cast<unsigned long long>(binding.block_identity.hi));
        text += identity;
    }
    text += "# block_tokens=" + std::to_string(binding.block_tokens) +
            " source_span=[" + std::to_string(binding.source_token_begin) + ".." +
            std::to_string(binding.source_token_end) + ") BOOKKEEPING\n";
    text += "# new_positions=[" + std::to_string(binding.declaration.position0) + ".." +
            std::to_string(binding.declaration.last_position() + 1) +
            ") position_is_new=" + std::string(binding.position_is_new ? "1" : "0") + "\n";
    {
        char digests[192];
        std::snprintf(digests, sizeof(digests),
                      "# source_digest=fnv1a64:0x%016llx ingested_digest=fnv1a64:0x%016llx\n",
                      static_cast<unsigned long long>(binding.payload.source_digest),
                      static_cast<unsigned long long>(binding.payload.baked_digest));
        text += digests;
    }

    const inject::Declaration& decl = binding.declaration;
    text += "direction=" + std::string(decl.direction == inject::Direction::Ingest ? "ingest"
                                                                                   : "egress") + "\n";
    text += "dtype=" + std::string(decl.dtype == inject::ElementType::Bf16   ? "bf16"
                                   : decl.dtype == inject::ElementType::F16 ? "f16"
                                                                            : "f32") +
            "\n";
    text += "layout=" + decl.layout + "\n";
    text += "rows=" + std::to_string(decl.rows) + "\n";
    text += "cols=" + std::to_string(decl.cols) + "\n";
    text += "position0=" + std::to_string(decl.position0) + "\n";
    {
        char scale[64];
        std::snprintf(scale, sizeof(scale), "scale=%.9g\n", decl.scale);
        text += scale;
    }
    text += "path=" + decl.path + "\n";
    {
        char digest[64];
        std::snprintf(digest, sizeof(digest), "digest=fnv1a64:0x%016llx\n",
                      static_cast<unsigned long long>(decl.digest));
        text += digest;
    }
    return text;
}

// The provenance sidecar, next to the payload: the durable record of the binding, so a run's
// evidence does not depend on a log line having been captured.
[[nodiscard]] inline std::string format_record_tsv(const RowBinding& binding) {
    if (!binding.admitted()) { return std::string(); }
    std::string text = "field\tvalue\n";
    text += "row\t" + std::to_string(binding.row_index) + "\n";
    text += "from_directory\t" + std::string(binding.from_directory ? "1" : "0") + "\n";
    {
        char identity[96];
        std::snprintf(identity, sizeof(identity), "0x%016llx:0x%016llx",
                      static_cast<unsigned long long>(binding.block_identity.lo),
                      static_cast<unsigned long long>(binding.block_identity.hi));
        text += "block_identity\t" + std::string(identity) + "\n";
    }
    text += "block_tokens\t" + std::to_string(binding.block_tokens) + "\n";
    text += "source_token_begin\t" + std::to_string(binding.source_token_begin) + "\n";
    text += "source_token_end\t" + std::to_string(binding.source_token_end) + "\n";
    text += "new_position0\t" + std::to_string(binding.declaration.position0) + "\n";
    text += "new_last_position\t" + std::to_string(binding.declaration.last_position()) + "\n";
    text += "position_is_new\t" + std::string(binding.position_is_new ? "1" : "0") + "\n";
    text += "rows\t" + std::to_string(binding.declaration.rows) + "\n";
    text += "cols\t" + std::to_string(binding.declaration.cols) + "\n";
    text += "payload_bytes\t" +
            std::to_string(binding.payload.payload.size() * sizeof(std::uint16_t)) + "\n";
    {
        char digests[192];
        std::snprintf(digests, sizeof(digests), "fnv1a64:0x%016llx\tfnv1a64:0x%016llx",
                      static_cast<unsigned long long>(binding.payload.source_digest),
                      static_cast<unsigned long long>(binding.payload.baked_digest));
        const std::string pair = digests;
        const std::size_t tab = pair.find('\t');
        text += "source_digest\t" + pair.substr(0, tab) + "\n";
        text += "ingested_digest\t" + pair.substr(tab + 1) + "\n";
    }
    text += "duplicate_writes\t" + std::string(binding.duplicate_writes ? "1" : "0") + "\n";
    return text;
}

[[nodiscard]] inline bool write_text_file(const std::string& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { return false; }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    return out.good();
}

} // namespace ninfer::spec::sum_dir_inject
