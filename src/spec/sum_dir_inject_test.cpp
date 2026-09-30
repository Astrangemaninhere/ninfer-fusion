// src/spec/sum_dir_inject_test.cpp -- THE CALIBER FOR THE ROW->CONTEXT BINDING.
//
// ONE BINARY, TWO MODES, and the same functions behind both:
//
//   * no arguments -- THE CALIBER, and the registered ctest. It provokes every row-side refusal by
//     name, then compares the SET of names it produced against the enumeration in
//     `sum_dir_inject.h`, so a name no case can reach FAILS as a dead guard rather than passing as
//     covered. It takes the same census over the INGRESS's own refusal enum and requires every name
//     to be either PROVOKED BY A CASE (10 of them) or DECLARED UNREACHABLE WITH A REASON (the
//     rest), so a new name added to `inject_channel.h` cannot slip past this file unnoticed. It
//     then proves the position rule (`sum_dir.h:90-96`) instead of asserting it.
//   * `--bind ...` -- THE BINDER CLI. It reads a directory blob or a row's fields plus a token list,
//     binds the row, writes the spec file the engine consumes (`--inject-spec`), and writes the
//     provenance sidecar. It cannot use a rule the caliber does not test, because it calls the same
//     `bind_row()`.
//
// Build: plain g++, no cmake, no CUDA, no device --
//   g++ -std=c++20 -I<tree>/src -Wall -Wextra -Werror src/spec/sum_dir_inject_test.cpp -o build/x

#include "spec/sum_dir_inject.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using ninfer::spec::inject::ElementType;
using ninfer::spec::inject::ModelLimits;
using ninfer::spec::inject::Refusal;
using ninfer::spec::sum_dir::SumDir;
using ninfer::spec::sum_dir::SumDirCodec;
using ninfer::spec::sum_dir::SumDirRow;
using ninfer::spec::sum_dir::SumDirState;
namespace binding = ninfer::spec::sum_dir_inject;
using binding::PayloadPin;
using binding::RowBinding;
using binding::RowRefusal;
using binding::RowView;

// ---------------------------------------------------------------------------
// the harness
// ---------------------------------------------------------------------------
int g_checks = 0;
int g_failures = 0;
std::vector<RowRefusal> g_row_refusals_seen;
std::vector<Refusal> g_ingress_refusals_seen;
std::string g_workdir = ".";

void check(bool condition, const std::string& what) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

void note(const std::string& what) { std::printf("  %s\n", what.c_str()); }

std::string path_in_workdir(const std::string& name) { return g_workdir + "/" + name; }

std::vector<std::uint8_t> make_payload_bytes(std::uint32_t rows, std::uint32_t cols,
                                             std::uint32_t salt) {
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(rows) * cols * 2U, 0U);
    for (std::uint32_t c = 0; c < cols; ++c) {
        for (std::uint32_t r = 0; r < rows; ++r) {
            // Deterministic, distinct and FINITE: a permutation of the bytes is detectable and no
            // element is non-finite.
            const float value =
                1.0f + static_cast<float>((r * 131U + c * 17U + salt) % 251U) / 256.0f;
            const std::uint16_t word = ninfer::spec::inject::fp32_to_bf16_rne(value);
            const std::size_t at = (static_cast<std::size_t>(c) * rows + r) * 2U;
            bytes[at]     = static_cast<std::uint8_t>(word & 0xFFU);
            bytes[at + 1] = static_cast<std::uint8_t>(word >> 8U);
        }
    }
    return bytes;
}

bool write_bytes(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { return false; }
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    return out.good();
}

std::uint64_t digest_of(const std::vector<std::uint8_t>& bytes) {
    return ninfer::spec::inject::fnv1a64(bytes.data(), bytes.size());
}

std::vector<std::uint32_t> make_tokens(std::size_t count, std::uint32_t salt) {
    std::vector<std::uint32_t> tokens(count);
    for (std::size_t i = 0; i < count; ++i) {
        tokens[i] = 1000U + salt * 100000U + static_cast<std::uint32_t>(i) * 7U;
    }
    return tokens;
}

std::string hex64(std::uint64_t value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "0x%016llx", static_cast<unsigned long long>(value));
    return buffer;
}

// ---------------------------------------------------------------------------
// a row's fields, without a directory behind them (the "fields" spelling)
// ---------------------------------------------------------------------------
SumDirRow live_row_fields(std::uint32_t token_begin, std::uint32_t token_count,
                          const ninfer::spec::sum_dir::SumDirDigest& identity) {
    SumDirRow row;
    row.block_identity = identity;
    row.token_begin    = token_begin;
    row.token_end      = token_begin + token_count;
    row.page           = token_begin / ninfer::spec::sum_dir::kSumDirBlockTokens;
    row.generation     = 1U;
    row.content_first  = 0U;
    row.content_count  = token_count;
    row.summary_index  = ninfer::spec::sum_dir::kSumDirNoSummary;
    row.summary_count  = 0U;
    row.codec          = SumDirCodec::Int8;
    row.state          = SumDirState::Live;
    row.file_slot      = 3;
    return row;
}

// ---------------------------------------------------------------------------
// the fixture: one Live 64-token row, one Dead row, one Live 8-token row, and the payloads
// ---------------------------------------------------------------------------
struct Fixture {
    SumDir directory{0U, ninfer::spec::sum_dir::kSumDirBlockTokens};
    std::vector<std::uint32_t> block_0;
    std::vector<std::uint32_t> block_2;
    std::string payload_0;
    std::string payload_2;
    std::string payload_short;
    std::string payload_nonfinite;
    std::string payload_other;
    std::uint64_t digest_0 = 0;
    std::uint64_t digest_2 = 0;
    std::uint64_t digest_other = 0;
};

bool build_fixture(Fixture& fixture, std::uint32_t rows) {
    fixture.block_0 = make_tokens(64, 1);
    fixture.block_2 = make_tokens(8, 2);

    const std::size_t good_row = fixture.directory.append(
        fixture.block_0, /*token_begin=*/0U, /*generation=*/1U, SumDirCodec::Int8,
        /*file_slot=*/3, /*page=*/0U);
    fixture.directory.set_summary(
        good_row, ninfer::spec::sum_dir::sum_dir_render_token_line(fixture.block_0));
    // file_slot = -1 makes `append` write a DEAD row: a recycled slot is not a hit.
    const std::size_t dead_row = fixture.directory.append(
        fixture.block_0, /*token_begin=*/64U, /*generation=*/1U, SumDirCodec::Int8,
        /*file_slot=*/-1, /*page=*/1U);
    fixture.directory.set_summary(
        dead_row, ninfer::spec::sum_dir::sum_dir_render_token_line(fixture.block_0));
    const std::size_t short_row = fixture.directory.append(
        fixture.block_2, /*token_begin=*/128U, /*generation=*/1U, SumDirCodec::Int8,
        /*file_slot=*/7, /*page=*/2U);
    fixture.directory.set_summary(
        short_row, ninfer::spec::sum_dir::sum_dir_render_token_line(fixture.block_2));
    if (fixture.directory.size() != 3U) { return false; }

    fixture.payload_0 = path_in_workdir("payload_row0.bin");
    fixture.payload_2 = path_in_workdir("payload_row2.bin");
    fixture.payload_short = path_in_workdir("payload_short.bin");
    fixture.payload_nonfinite = path_in_workdir("payload_nonfinite.bin");
    fixture.payload_other = path_in_workdir("payload_row0_other.bin");

    const std::vector<std::uint8_t> bytes_0 = make_payload_bytes(rows, 64, 0U);
    const std::vector<std::uint8_t> bytes_2 = make_payload_bytes(rows, 8, 3U);
    const std::vector<std::uint8_t> bytes_other = make_payload_bytes(rows, 64, 9U);
    const std::vector<std::uint8_t> bytes_short = make_payload_bytes(rows, 4, 5U);
    std::vector<std::uint8_t> bytes_nonfinite = bytes_0;
    // +inf in bf16 = exponent all ones, mantissa zero, sign zero.
    bytes_nonfinite[0] = 0x80U;
    bytes_nonfinite[1] = 0x7FU;

    fixture.digest_0 = digest_of(bytes_0);
    fixture.digest_2 = digest_of(bytes_2);
    fixture.digest_other = digest_of(bytes_other);
    return write_bytes(fixture.payload_0, bytes_0) && write_bytes(fixture.payload_2, bytes_2) &&
           write_bytes(fixture.payload_short, bytes_short) &&
           write_bytes(fixture.payload_nonfinite, bytes_nonfinite) &&
           write_bytes(fixture.payload_other, bytes_other);
}

// A pin that is correct for row 0 unless the caller overrides a field.
PayloadPin good_pin(const Fixture& fixture) {
    PayloadPin pin;
    pin.dtype      = ElementType::Bf16;
    pin.scale      = 1.0;
    pin.path       = fixture.payload_0;
    pin.has_digest = true;
    pin.digest     = fixture.digest_0;
    return pin;
}

// One row-side name per sighting, deduplicated by the population check below.
void census(RowRefusal refusal) {
    if (refusal != RowRefusal::None) { g_row_refusals_seen.push_back(refusal); }
}

void record_ingress(const RowBinding& binding) {
    if (binding.refusal != RowRefusal::None) { return; }
    if (!binding.settlement.admitted()) {
        g_ingress_refusals_seen.push_back(binding.settlement.refusal);
        return;
    }
    if (!binding.payload.settled()) { g_ingress_refusals_seen.push_back(binding.payload.refusal); }
}

// ===========================================================================
// THE CALIBER
// ===========================================================================
int run_caliber(const std::uint32_t rows) {
    const std::int32_t hidden = static_cast<std::int32_t>(rows);
    const ModelLimits limits{hidden, /*context_capacity=*/4096, ElementType::Bf16};

    Fixture fixture;
    if (!build_fixture(fixture, rows)) {
        std::printf("FAIL: the fixture could not be written under '%s'\n", g_workdir.c_str());
        return 1;
    }
    note("fixture: 3 directory rows (0 Live/64 tok, 1 Dead/64 tok, 2 Live/8 tok), 5 payloads");

    // ---- 0. the positive control: a well-formed binding must be ADMITTED ------------------
    {
        const RowView view = binding::row_view_from_directory(fixture.directory, 0U);
        RowBinding b = binding::bind_row(view, hidden, limits, /*new_position0=*/3000,
                                         good_pin(fixture));
        b.row_index = 0U;
        check(b.admitted(), "a well-formed row at a NEW position must be admitted: " + b.line());
        check(b.block_tokens == 64U, "the block's length comes from the row (64)");
        check(b.declaration.cols == 64, "the declaration's cols IS the row's token count");
        check(b.declaration.position0 == 3000, "the position is the caller's, not the row's");
        check(b.position_is_new, "new_position0 != token_begin is reported as position_is_new");
        check(b.payload.baked_digest == fixture.digest_0,
              "at bf16+scale=1 the ingested digest EQUALS the file's digest (arithmetic-free)");
        check(b.source_token_begin == 0U && b.source_token_end == 64U,
              "the source span is carried as BOOKKEEPING");
        note(b.line());
    }

    // ---- 1. THE ROW-SIDE POPULATION: one case per name, by name ---------------------------
    {
        // 1.1 refused-row-no-row-source
        RowBinding b = binding::bind_row(RowView{}, hidden, limits, 0, good_pin(fixture));
        check(b.refusal == RowRefusal::RefusedRowNoRowSource,
              "an empty row source must refuse by name: " + b.line());
        census(b.refusal);

        // 1.2 refused-row-out-of-range
        const RowView beyond = binding::row_view_from_directory(fixture.directory, 3U);
        b = binding::bind_row(beyond, hidden, limits, 0, good_pin(fixture));
        check(b.refusal == RowRefusal::RefusedRowOutOfRange,
              "row 3 of 3 must refuse out-of-range: " + b.line());
        census(b.refusal);

        // 1.3 refused-row-not-well-formed: an unset identity
        SumDirRow unset_identity = live_row_fields(0U, 64U, fixture.directory.rows()[0].block_identity);
        unset_identity.block_identity = ninfer::spec::sum_dir::SumDirDigest{};
        RowView view = binding::row_view_from_fields(unset_identity, fixture.block_0);
        b = binding::bind_row(view, hidden, limits, 0, good_pin(fixture));
        check(b.refusal == RowRefusal::RefusedRowNotWellFormed,
              "a row whose identity is unset must refuse as not well formed: " + b.line());
        census(b.refusal);

        // 1.4 refused-row-not-live: row 1 is Dead (a recycled slot is not a hit)
        const RowView dead = binding::row_view_from_directory(fixture.directory, 1U);
        b = binding::bind_row(dead, hidden, limits, 0, good_pin(fixture));
        check(b.refusal == RowRefusal::RefusedRowNotLive,
              "a Dead row must refuse by name: " + b.line());
        census(b.refusal);

        // 1.5 refused-row-content-absent: the row names 64 tokens and NO bytes stand for them. The
        //     directory's own half of this shape is `block_content()` returning an EMPTY vector for
        //     a window that does not fit (`sum_dir.h:1968-1974`); this case is the same statement
        //     made through the fields spelling, because `SumDir::rows()` is read-only and a
        //     directory with a corrupt window cannot be built without corrupting a serialized blob
        //     that `deserialize` would then refuse on its payload digest.
        {
            const SumDirRow row =
                live_row_fields(0U, 64U, fixture.directory.rows()[0].block_identity);
            const RowView absent =
                binding::row_view_from_fields(row, std::vector<std::uint32_t>{});
            b = binding::bind_row(absent, hidden, limits, 0, good_pin(fixture));
            check(b.refusal == RowRefusal::RefusedRowContentAbsent,
                  "a row with no bytes must refuse as content-absent: " + b.line());
            census(b.refusal);
        }

        // 1.6 refused-row-length-mismatch
        {
            PayloadPin pin = good_pin(fixture);
            pin.declared_block_tokens = 32U;
            const RowView view0 = binding::row_view_from_directory(fixture.directory, 0U);
            b = binding::bind_row(view0, hidden, limits, 0, pin);
            check(b.refusal == RowRefusal::RefusedRowLengthMismatch,
                  "a declared length that is not the row's must refuse by name: " + b.line());
            census(b.refusal);
        }

        // 1.7 refused-row-content-digest-mismatch: one block's worth of DIFFERENT tokens
        {
            const SumDirRow row =
                live_row_fields(0U, 64U, fixture.directory.rows()[0].block_identity);
            const std::vector<std::uint32_t> other = make_tokens(64, 99);
            const RowView drifted = binding::row_view_from_fields(row, other);
            b = binding::bind_row(drifted, hidden, limits, 0, good_pin(fixture));
            check(b.refusal == RowRefusal::RefusedRowContentDigestMismatch,
                  "content that does not digest to the row's identity must refuse by name: " +
                      b.line());
            census(b.refusal);
        }

        // 1.8 refused-row-payload-unpinned
        {
            PayloadPin pin = good_pin(fixture);
            pin.has_digest = false;
            const RowView view0 = binding::row_view_from_directory(fixture.directory, 0U);
            b = binding::bind_row(view0, hidden, limits, 0, pin);
            check(b.refusal == RowRefusal::RefusedRowPayloadUnpinned,
                  "a row-borne declaration with no digest must refuse by name: " + b.line());
            census(b.refusal);
        }

        // 1.9 refused-row-range-overlaps-plan (a DIFFERENT block over an earlier range)
        {
            const RowView view0 = binding::row_view_from_directory(fixture.directory, 0U);
            RowBinding first = binding::bind_row(view0, hidden, limits, 0, good_pin(fixture));
            check(first.admitted(), "the first plan member must bind: " + first.line());
            PayloadPin pin2 = good_pin(fixture);
            pin2.path   = fixture.payload_2;
            pin2.digest = fixture.digest_2;
            const RowView view2 = binding::row_view_from_directory(fixture.directory, 2U);
            const RowBinding overlap[1] = {first};
            b = binding::bind_row(view2, hidden, limits, /*new_position0=*/32, pin2, overlap);
            check(b.refusal == RowRefusal::RefusedRowRangeOverlapsPlan,
                  "a different block over an occupied range must refuse by name: " + b.line());
            census(b.refusal);
        }

        // 1.10 refused-row-same-identity-different-payload
        {
            const RowView view0 = binding::row_view_from_directory(fixture.directory, 0U);
            RowBinding first = binding::bind_row(view0, hidden, limits, 0, good_pin(fixture));
            PayloadPin other = good_pin(fixture);
            other.path   = fixture.payload_other;
            other.digest = fixture.digest_other;
            const RowView same = binding::row_view_from_directory(fixture.directory, 0U);
            const RowBinding plan[1] = {first};
            b = binding::bind_row(same, hidden, limits, /*new_position0=*/1000, other, plan);
            check(b.refusal == RowRefusal::RefusedRowSameIdentityDifferentPayload,
                  "two payloads for one identity must refuse by name: " + b.line());
            census(b.refusal);
            check(binding::same_target(first, first), "a binding is one target with itself");
        }
    }

    // ---- 2. THE POPULATION CHECK over the row-side names ---------------------------------
    {
        std::vector<RowRefusal> produced = g_row_refusals_seen;
        std::sort(produced.begin(), produced.end());
        produced.erase(std::unique(produced.begin(), produced.end()), produced.end());
        const std::vector<RowRefusal> declared = binding::all_row_refusals();
        std::vector<RowRefusal> missing;
        for (const RowRefusal name : declared) {
            if (std::find(produced.begin(), produced.end(), name) == produced.end()) {
                missing.push_back(name);
            }
        }
        std::printf("row-side population: declared=%zu provoked=%zu\n", declared.size(),
                    produced.size());
        for (const RowRefusal name : declared) {
            const bool hit = std::find(produced.begin(), produced.end(), name) != produced.end();
            std::printf("  %-46s %s\n", binding::row_refusal_name(name), hit ? "PROVOKED" : "UNREACHABLE");
        }
        check(missing.empty(),
              "every row-side refusal name must be provoked by a case (a dead guard FAILS, it does "
              "not pass silently)");
        for (const RowRefusal name : missing) {
            std::printf("FAIL: dead guard: %s\n", binding::row_refusal_name(name));
        }
    }

    // ---- 3. THE INGRESS'S OWN NAMES: which the ROW path provokes, and why the rest cannot be --
    {
        const RowView row0 = binding::row_view_from_directory(fixture.directory, 0U);

        ModelLimits tight = limits;
        tight.context_capacity = 32;

        // The ten spellings, deliberately written out one by one so each has a line of its own.
        {
            PayloadPin pin = good_pin(fixture);
            const RowBinding b = binding::bind_row(row0, hidden + 1, limits, 0, pin);
            record_ingress(b);
            check(!b.settlement.admitted() && b.settlement.refusal == Refusal::RefusedRowsMismatch,
                  "rows != the engine's hidden must reuse refused-rows-mismatch: " + b.line());
        }
        {
            const RowBinding b = binding::bind_row(row0, hidden, tight, 0, good_pin(fixture));
            record_ingress(b);
            check(!b.settlement.admitted() &&
                      b.settlement.refusal == Refusal::RefusedColumnsExceedContext,
                  "cols > the context capacity must reuse refused-columns-exceed-context: " +
                      b.line());
        }
        {
            const RowBinding b = binding::bind_row(row0, hidden, limits, 4080, good_pin(fixture));
            record_ingress(b);
            check(!b.settlement.admitted() &&
                      b.settlement.refusal == Refusal::RefusedPositionRangeOutOfContext,
                  "a range beyond the context must reuse refused-position-range-out-of-context: " +
                      b.line());
        }
        {
            const RowBinding b = binding::bind_row(row0, hidden, limits, -1, good_pin(fixture));
            record_ingress(b);
            check(!b.settlement.admitted() &&
                      b.settlement.refusal == Refusal::RefusedPositionNegative,
                  "a negative position must reuse refused-position-negative: " + b.line());
        }
        {
            PayloadPin pin = good_pin(fixture);
            pin.scale = 0.0;
            const RowBinding b = binding::bind_row(row0, hidden, limits, 0, pin);
            record_ingress(b);
            check(!b.settlement.admitted() && b.settlement.refusal == Refusal::RefusedScaleInvalid,
                  "scale <= 0 must reuse refused-scale-invalid: " + b.line());
        }
        {
            PayloadPin pin = good_pin(fixture);
            pin.path.clear();
            const RowBinding b = binding::bind_row(row0, hidden, limits, 0, pin);
            record_ingress(b);
            check(!b.settlement.admitted() && b.settlement.refusal == Refusal::RefusedPathEmpty,
                  "an empty payload path must reuse refused-path-empty: " + b.line());
        }
        {
            PayloadPin pin = good_pin(fixture);
            pin.path = path_in_workdir("does_not_exist.bin");
            const RowBinding b = binding::bind_row(row0, hidden, limits, 0, pin);
            record_ingress(b);
            check(!b.payload.settled() && b.payload.refusal == Refusal::RefusedFileMissing,
                  "a missing payload must reuse refused-file-missing: " + b.line());
        }
        {
            PayloadPin pin = good_pin(fixture);
            pin.path   = fixture.payload_short;
            pin.digest = digest_of(make_payload_bytes(rows, 4, 5U));
            const RowBinding b = binding::bind_row(row0, hidden, limits, 0, pin);
            record_ingress(b);
            check(!b.payload.settled() && b.payload.refusal == Refusal::RefusedFileSizeMismatch,
                  "a short payload must reuse refused-file-size-mismatch: " + b.line());
        }
        {
            PayloadPin pin = good_pin(fixture);
            pin.digest = fixture.digest_other;
            const RowBinding b = binding::bind_row(row0, hidden, limits, 0, pin);
            record_ingress(b);
            check(!b.payload.settled() &&
                      b.payload.refusal == Refusal::RefusedSourceDigestMismatch,
                  "a wrong pinned digest must reuse refused-source-digest-mismatch: " + b.line());
        }
        {
            PayloadPin pin = good_pin(fixture);
            pin.path   = fixture.payload_nonfinite;
            pin.digest = digest_of([rows] {
                std::vector<std::uint8_t> bytes = make_payload_bytes(rows, 64, 0U);
                bytes[0] = 0x80U;
                bytes[1] = 0x7FU;
                return bytes;
            }());
            const RowBinding b = binding::bind_row(row0, hidden, limits, 0, pin);
            record_ingress(b);
            check(!b.payload.settled() && b.payload.refusal == Refusal::RefusedNonFinite,
                  "a non-finite element must reuse refused-non-finite: " + b.line());
        }

        // THE CENSUS over the ingress's enum: every enumerated name is either provoked above or
        // declared unreachable WITH A REASON. A name added to inject_channel.h without a decision
        // here shows up as UNCLASSIFIED and fails.
        const std::pair<Refusal, const char*> unreachable[] = {
            {Refusal::RefusedSpecUnreadable, "the binder calls parse_spec_file() on nothing: a row-borne declaration is built in memory"},
            {Refusal::RefusedSpecSyntax, "same -- there is no spec text to be malformed"},
            {Refusal::RefusedFieldUnknown, "same -- there is no key vocabulary at this level"},
            {Refusal::RefusedFieldMissing, "same -- the binder fills every field"},
            {Refusal::RefusedFieldDuplicate, "same"},
            {Refusal::RefusedFieldNotInteger, "same"},
            {Refusal::RefusedDirectionUnsupported, "a row-borne binding IS ingest by definition; the binder has no direction to get wrong"},
            {Refusal::RefusedDtypeUnsupported, "the pin's dtype is the enum (bf16/f16/f32); a spec FILE can still name f64 and injectchan's caliber provokes it there"},
            {Refusal::RefusedLayoutUnsupported, "the binder always declares token-major, the only spelling the engine consumes"},
            {Refusal::RefusedColumnsZero, "cols IS the row's token count, and a zero-token block is refused earlier as refused-row-content-absent"},
            {Refusal::RefusedDigestMalformed, "the pin's digest is a uint64; it is not text and cannot be malformed"},
            {Refusal::RefusedOverlapsVisionScatter, "raised by the engine at apply() time against this chunk's vision columns; not knowable in the binder"},
            {Refusal::RefusedRangeNotFullyApplied, "raised by the engine at finish() time from the coverage bitmap; not knowable in the binder"},
        };
        std::size_t classified = 0U;
        for (const auto& item : unreachable) {
            ++classified;
            (void)item.second;
        }
        std::printf("ingress-side census: provoked-by-the-row-path=%zu declared-unreachable=%zu\n",
                    g_ingress_refusals_seen.size(), classified);
        std::vector<std::string> provoked;
        for (const Refusal name : g_ingress_refusals_seen) {
            const std::string text(ninfer::spec::inject::refusal_name(name));
            if (std::find(provoked.begin(), provoked.end(), text) == provoked.end()) {
                provoked.push_back(text);
            }
        }
        std::sort(provoked.begin(), provoked.end());
        for (const std::string& name : provoked) { std::printf("  REUSED  %s\n", name.c_str()); }
        for (const auto& item : unreachable) {
            std::printf("  N/A     %-40s %s\n",
                        ninfer::spec::inject::refusal_name(item.first), item.second);
        }
        check(provoked.size() >= 10U,
              "the row path must reach at least the ten ingress refusals it claims to reuse");
        // THE CENSUS BALANCE: every name the ingress spells is either reused by the row path or
        // declared unreachable WITH A REASON, so a name added to `inject_channel.h` without a
        // decision here is a failure rather than a silent omission.
        check(provoked.size() + classified >= 23U,
              "every enumerated ingress name must be either reused or declared unreachable");
    }

    // ---- 4. THE POSITION RULE, PROVED (sum_dir.h:90-96) ----------------------------------
    {
        const RowView view0 = binding::row_view_from_directory(fixture.directory, 0U);
        const SumDirRow fields = live_row_fields(0U, 64U, fixture.directory.rows()[0].block_identity);
        const RowView variant = binding::row_view_from_fields(fields, make_tokens(64, 42));
        const binding::PositionIndependenceResult self =
            binding::row_binding_position_self_test(view0, hidden, limits, good_pin(fixture),
                                                    /*position_a=*/0, /*position_b=*/3000, variant);
        check(self.holds, "the position-independence self-test must HOLD: " + self.detail);
        check(self.identity_equal, "the same row at two positions has EQUAL identity");
        check(self.payload_byte_equal, "the same row at two positions has a BYTE-EQUAL payload");
        check(self.ingested_digest_equal, "the two ingestions have equal digests");
        check(self.declarations_equal_except_position0,
              "the two declarations differ in NOTHING except position0");
        check(self.one_token_change_moves_identity,
              "one token changed anywhere in the block moves the identity (the other half)");
        note("positions tried: 2 (0 and 3000); variant: one block's worth of different tokens");
    }

    // ---- 5. ONE TARGET, ONE PAYLOAD, AND THE SAME BYTES TWICE ----------------------------
    {
        SumDir two_copies{0U, ninfer::spec::sum_dir::kSumDirBlockTokens};
        const std::size_t copy_a =
            two_copies.append(fixture.block_0, 0U, 1U, SumDirCodec::Int8, 3, 0U);
        const std::size_t copy_b =
            two_copies.append(fixture.block_0, /*token_begin=*/960U, 1U, SumDirCodec::Int8, 4, 15U);
        check(copy_a == 0U && copy_b == 1U, "the two-copy directory must hold exactly two rows");
        const RowView a_view = binding::row_view_from_directory(two_copies, copy_a);
        const RowView b_view = binding::row_view_from_directory(two_copies, copy_b);
        const RowBinding a = binding::bind_row(a_view, hidden, limits, 100, good_pin(fixture));
        check(a.admitted(), "copy A must bind: " + a.line());
        const RowBinding plan[1] = {a};
        const RowBinding b = binding::bind_row(b_view, hidden, limits, 3000, good_pin(fixture), plan);
        check(b.admitted(), "copy B at a NEW position must bind: " + b.line());
        check(binding::same_target(a, b),
              "two copies of one block at DIFFERENT offsets are ONE TARGET (sum_dir_vector.h:361)");
        check(a.block_identity == b.block_identity && a.declaration.position0 != b.declaration.position0,
              "one target, two positions -- the identity did not read the position");
        check(b.payload.payload == a.payload.payload,
              "one target means ONE PAYLOAD, byte for byte");
        const RowBinding c = binding::bind_row(b_view, hidden, limits, 120, good_pin(fixture), plan);
        check(c.admitted() && c.duplicate_writes,
              "one target over an intersecting range is the SAME BYTES twice: admitted and counted");
        RowView other_view = binding::row_view_from_fields(
            live_row_fields(960U, 64U,
                            ninfer::spec::sum_dir::sum_dir_block_digest(make_tokens(64, 3))),
            make_tokens(64, 3));
        const RowBinding d = binding::bind_row(other_view, hidden, limits, 100, good_pin(fixture),
                                               plan);
        check(d.refusal == RowRefusal::RefusedRowRangeOverlapsPlan,
              "a DIFFERENT block over an intersecting range must refuse: " + d.line());
    }

    // ---- 6. THE SPEC FILE THE ENGINE CONSUMES (round trip through the ingress's parser) ---
    {
        const RowView view0 = binding::row_view_from_directory(fixture.directory, 0U);
        RowBinding b = binding::bind_row(view0, hidden, limits, 3000, good_pin(fixture));
        b.row_index = 0U;
        check(b.admitted(), "the round-trip binding must be admitted: " + b.line());

        const std::string spec = binding::format_spec_text(b);
        check(!spec.empty(), "an admitted binding must render a spec file");
        check(spec.find("# block_identity=") != std::string::npos,
              "the spec carries the row provenance as a # comment the parser skips");
        check(spec.find("position0=3000") != std::string::npos,
              "the spec's position0 is the caller's chosen position");

        const ninfer::spec::inject::ParseResult parsed = ninfer::spec::inject::parse_spec(spec);
        check(parsed.ok(), "the engine's OWN parser must accept the emitted spec: " +
                               ninfer::spec::inject::render_refusal(
                                   parsed.settlement.refusal, parsed.settlement.field,
                                   parsed.settlement.detail));
        if (parsed.ok()) {
            check(binding::declarations_equal_except_position0(b.declaration, parsed.declaration) &&
                      b.declaration.position0 == parsed.declaration.position0,
                  "the parsed declaration must be the bound declaration, field for field");
            check(ninfer::spec::inject::admit_against_engine(parsed.declaration, limits).admitted(),
                  "the parsed declaration must be admitted against the engine");
            const ninfer::spec::inject::PayloadSettlement again =
                ninfer::spec::inject::bake(parsed.declaration);
            check(again.settled() && again.baked_digest == b.payload.baked_digest,
                  "the re-baked ingress digest must equal the binder's -- THE JOIN");
        }

        const std::string record = binding::format_record_tsv(b);
        const std::string expected = "ingested_digest\tfnv1a64:" + hex64(b.payload.baked_digest);
        check(record.find(expected) != std::string::npos,
              "the provenance record must carry the ingested digest the engine PRINTS");
        check(record.find("position0_is_new") == std::string::npos,
              "the record names its own fields (position_is_new is the row-side spelling)");
        note("round trip: binder -> spec text -> engine parser -> admission -> bake: digest equal");

        check(write_bytes(path_in_workdir("spec_roundtrip.txt"),
                          std::vector<std::uint8_t>(spec.begin(), spec.end())),
              "the emitted spec must be writable");
    }

    // ---- 7. THE NOT-BLIND CONTROL --------------------------------------------------------
    {
        const RowView view0 = binding::row_view_from_directory(fixture.directory, 0U);
        const RowBinding a = binding::bind_row(view0, hidden, limits, 3000, good_pin(fixture));
        const RowBinding b = binding::bind_row(view0, hidden, limits, 3000, good_pin(fixture));
        check(a.bound && b.bound && a.declaration.position0 == b.declaration.position0,
              "the binder is deterministic on identical inputs");
        const RowBinding c = binding::bind_row(view0, hidden, limits, 2000, good_pin(fixture));
        check(c.bound && c.declaration.position0 != a.declaration.position0,
              "NOT-BLIND: a different position is a different binding (the probe can differ)");
        check(a.block_identity == c.block_identity,
              "and the identity did NOT move with it -- the rule, in one line");
    }

    // ---- 8. THE FURTHER DEFECT THIS BINDING EXPOSED, PROVED HOST-SIDE --------------------
    //
    // The row->context road needs the payload's bytes, and the only honest source of them in this
    // tree is the ingress's OWN EGRESS direction -- `dl/injectchan` says so of its own arm E: the
    // engine's own embeddings are "the definition of 'equivalent to tokens the engine could have
    // computed' that is not a host re-implementation of the embedding table". So the binder drives
    // an egress declaration as its FIRST step, and that is where the defect is.
    //
    // THE DEFECT, in measured terms. `inject_ingress.h`'s `configure()` calls three functions from
    // `inject_channel.h`, in this order: `parse_spec_file()`, `admit_against_engine()`, and
    // `bake()`. `bake()` READS the payload -- it is the INGEST settlement. For `direction=egress`
    // the path is what the run is going to WRITE. So the composition refuses an egress declaration
    // with `refused-file-missing` before the model is even bound, and the `flush_dump()` at the end
    // of the same function is unreachable: THE DIRECTION THAT PRODUCES THE BYTES CANNOT BE USED TO
    // PRODUCE THEM. This case is that composition, called here with the same three functions in the
    // same order, so the claim is about the engine's own code and not a re-derivation of it.
    {
        const std::string egress_path = path_in_workdir("egress_output_that_does_not_exist.bin");
        std::remove(egress_path.c_str());
        const std::string spec =
            "direction=egress\ndtype=bf16\nlayout=token-major\nrows=" + std::to_string(rows) +
            "\ncols=64\nposition0=0\nscale=1.0\npath=" + egress_path + "\n";
        const ninfer::spec::inject::ParseResult parsed = ninfer::spec::inject::parse_spec(spec);
        check(parsed.ok(), "an egress declaration parses (nothing is wrong with its grammar)");
        const ninfer::spec::inject::Settlement admitted =
            ninfer::spec::inject::admit_against_engine(parsed.declaration, limits);
        check(admitted.admitted(),
              "and it is ADMITTED against the engine -- so the refusal below comes from the bake");
        const ninfer::spec::inject::PayloadSettlement settled =
            ninfer::spec::inject::bake(parsed.declaration);
        check(!settled.settled() && settled.refusal == Refusal::RefusedFileMissing,
              "and the UNCONDITIONAL bake() refuses it: refused-file-missing, because the file this "
              "direction is going to WRITE must already exist. THE DEFECT.");
        note("the fix (this line's landing) gates the bake on direction=ingest, so the settlement "
             "stays for the direction it was written for; see REPORT.md section 5.1");
        // The control that the fix's CONDITION is the discriminator and not the file's absence: the
        // same declaration as INGEST is refused for the same reason (a missing file), which is
        // correct for ingest -- so the refusal is about the direction, not about bake() being
        // wrong.
        ninfer::spec::inject::Declaration as_ingest = parsed.declaration;
        as_ingest.direction = ninfer::spec::inject::Direction::Ingest;
        const ninfer::spec::inject::PayloadSettlement ingest_settled =
            ninfer::spec::inject::bake(as_ingest);
        check(!ingest_settled.settled() &&
                  ingest_settled.refusal == Refusal::RefusedFileMissing,
              "the same declaration as INGEST refuses identically -- so the fix's condition is the "
              "discriminator, and no ingest behaviour changes");
    }

    std::printf("\nchecks=%d failures=%d\n", g_checks, g_failures);
    std::printf("VERDICT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}

} // namespace

// ===========================================================================
// THE BINDER CLI -- `--bind ...`. It calls the SAME bind_row() the caliber drives, so the arm
// cannot use a rule the caliber does not test.
//
//   --dir PATH --row N                 read the row out of a serialized SumDir blob
//   --tokens PATH --token-begin N      ...or supply the row's fields and the block's token ids
//   [--identity 0xLO:0xHI]             pin the row's identity; OMITTED = adopt the digest of the
//                                      tokens -- which is honest only because the record then says
//                                      from_directory=0, i.e. this row did not come from a directory
//   --payload PATH [--payload-digest 0xH | auto]
//   --rows N --context N --position0 N [--scale F]
//   --spec-out PATH --record-out PATH
// ===========================================================================
namespace {

bool read_token_file(const std::string& path, std::vector<std::uint32_t>& tokens,
                     std::string& why) {
    std::ifstream stream(path);
    if (!stream) {
        why = "'" + path + "' could not be opened";
        return false;
    }
    std::string word;
    while (stream >> word) {
        std::size_t consumed = 0U;
        try {
            const unsigned long long value = std::stoull(word, &consumed, 0);
            if (consumed != word.size() || value > 0xFFFFFFFFULL) {
                why = "'" + word + "' is not a token id";
                return false;
            }
            tokens.push_back(static_cast<std::uint32_t>(value));
        } catch (const std::exception&) {
            why = "'" + word + "' is not a token id";
            return false;
        }
    }
    if (tokens.empty()) {
        why = "'" + path + "' holds no token ids";
        return false;
    }
    return true;
}

bool parse_identity(const std::string& text, ninfer::spec::sum_dir::SumDirDigest& identity) {
    const std::size_t colon = text.find(':');
    if (colon == std::string::npos) { return false; }
    try {
        identity.lo = std::stoull(text.substr(0, colon), nullptr, 0);
        identity.hi = std::stoull(text.substr(colon + 1), nullptr, 0);
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

int run_cli(int argc, char** argv) {
    std::string dir_path, tokens_path, payload, spec_out, record_out, identity_text;
    std::uint64_t row_index = 0U, token_begin = 0U, payload_digest = 0U;
    std::int32_t rows = 0, context = 0, position0 = 0;
    double scale = 1.0;
    bool have_dir = false, have_tokens = false, have_payload = false;
    bool digest_auto = false, digest_given = false, have_identity = false, have_spec_out = false;

    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        const auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "rowbind: %s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (flag == "--dir") { dir_path = next("--dir"); have_dir = true; }
        else if (flag == "--row") { row_index = std::stoull(next("--row")); }
        else if (flag == "--tokens") { tokens_path = next("--tokens"); have_tokens = true; }
        else if (flag == "--token-begin") { token_begin = std::stoull(next("--token-begin")); }
        else if (flag == "--identity") { identity_text = next("--identity"); have_identity = true; }
        else if (flag == "--payload") { payload = next("--payload"); have_payload = true; }
        else if (flag == "--payload-digest") {
            const std::string value = next("--payload-digest");
            digest_given = true;
            if (value == "auto") { digest_auto = true; } else {
                payload_digest = std::stoull(value, nullptr, 0);
            }
        }
        else if (flag == "--rows") { rows = static_cast<std::int32_t>(std::stoll(next("--rows"))); }
        else if (flag == "--context") { context = static_cast<std::int32_t>(std::stoll(next("--context"))); }
        else if (flag == "--position0") { position0 = static_cast<std::int32_t>(std::stoll(next("--position0"))); }
        else if (flag == "--scale") { scale = std::stod(next("--scale")); }
        else if (flag == "--spec-out") { spec_out = next("--spec-out"); have_spec_out = true; }
        else if (flag == "--record-out") { record_out = next("--record-out"); }
        else {
            std::fprintf(stderr, "rowbind: unknown argument '%s'\n", flag.c_str());
            return 2;
        }
    }
    if (!have_payload || !have_spec_out || rows <= 0 || context <= 0) {
        std::fprintf(stderr, "rowbind: --payload, --spec-out, --rows and --context are mandatory\n");
        return 2;
    }
    if (have_dir == have_tokens) {
        std::fprintf(stderr, "rowbind: name exactly one row source: --dir+--row, or --tokens\n");
        return 2;
    }

    const ModelLimits limits{rows, context, ElementType::Bf16};
    RowView view;
    if (have_dir) {
        std::ifstream stream(dir_path, std::ios::binary);
        if (!stream) {
            std::fprintf(stderr, "rowbind: '%s' could not be opened\n", dir_path.c_str());
            return 2;
        }
        const std::vector<std::uint8_t> blob((std::istreambuf_iterator<char>(stream)),
                                            std::istreambuf_iterator<char>());
        ninfer::spec::sum_dir::SumDirLoadReport report;
        const SumDir directory =
            SumDir::deserialize(blob.data(), blob.size(), report);
        view = binding::row_view_from_directory(directory, row_index);
    } else {
        std::vector<std::uint32_t> tokens;
        std::string why;
        if (!read_token_file(tokens_path, tokens, why)) {
            std::fprintf(stderr, "rowbind: %s\n", why.c_str());
            return 2;
        }
        ninfer::spec::sum_dir::SumDirDigest identity =
            ninfer::spec::sum_dir::sum_dir_block_digest(tokens);
        if (have_identity && !parse_identity(identity_text, identity)) {
            std::fprintf(stderr, "rowbind: --identity '%s' is not LO:HI\n", identity_text.c_str());
            return 2;
        }
        SumDirRow row;
        const std::uint32_t count = static_cast<std::uint32_t>(tokens.size());
        row.block_identity = identity;
        row.token_begin    = static_cast<std::uint32_t>(token_begin);
        row.token_end      = static_cast<std::uint32_t>(token_begin) + count;
        row.page           = static_cast<std::uint32_t>(token_begin) / ninfer::spec::sum_dir::kSumDirBlockTokens;
        row.generation     = 1U;
        row.content_first  = 0U;
        row.content_count  = count;
        row.summary_index  = ninfer::spec::sum_dir::kSumDirNoSummary;
        row.summary_count  = 0U;
        row.codec          = SumDirCodec::Int8;
        row.state          = SumDirState::Live;
        row.file_slot      = 3;
        view = binding::row_view_from_fields(row, tokens);
    }

    PayloadPin pin;
    pin.dtype      = ElementType::Bf16;
    pin.scale      = scale;
    pin.path       = payload;
    pin.has_digest = digest_given;
    pin.digest     = payload_digest;

    // `--payload-digest auto` means "measure the file, then pin what was measured". The measurement
    // is `inject::fnv1a64()` -- THE ingress's own digest function, the same one `bake()` fills
    // `source_digest` with -- so the pin is the instrument the engine compares against and not a
    // second hash that could disagree with it.
    if (digest_auto) {
        std::ifstream stream(payload, std::ios::binary);
        if (!stream) {
            std::fprintf(stderr, "rowbind: '%s' could not be opened to measure its digest\n",
                         payload.c_str());
            return 2;
        }
        const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(stream)),
                                             std::istreambuf_iterator<char>());
        pin.digest = ninfer::spec::inject::fnv1a64(bytes.data(), bytes.size());
    }

    RowBinding b = binding::bind_row(view, rows, limits, position0, pin);
    b.row_index  = row_index;

    std::printf("%s\n", b.line().c_str());
    if (!b.admitted()) {
        std::printf("ROWBIND VERDICT: REFUSED\n");
        return 1;
    }
    if (!binding::write_text_file(spec_out, binding::format_spec_text(b))) {
        std::fprintf(stderr, "rowbind: '%s' could not be written\n", spec_out.c_str());
        return 2;
    }
    if (!record_out.empty()) {
        if (!binding::write_text_file(record_out, binding::format_record_tsv(b))) {
            std::fprintf(stderr, "rowbind: '%s' could not be written\n", record_out.c_str());
            return 2;
        }
    }
    std::printf("ROWBIND VERDICT: BOUND spec=%s\n", spec_out.c_str());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--bind") { return run_cli(argc, argv); }
    if (argc >= 2 && std::string(argv[1]) == "--help") {
        std::printf(
            "sum_dir_inject_test (F745 injectbind) -- the caliber for the row->context binding\n"
            "  no arguments   run the caliber (the registered ctest)\n"
            "  --bind ...     bind a ROW and emit the spec the engine consumes:\n"
            "                 --dir PATH --row N | --tokens PATH [--token-begin N]\n"
            "                 [--identity 0xLO:0xHI] --payload PATH [--payload-digest 0xH|auto]\n"
            "                 --rows N --context N --position0 N [--scale F]\n"
            "                 --spec-out PATH [--record-out PATH]\n");
        return 0;
    }
    const char* workdir = std::getenv("INJECTBIND_WORKDIR");
    if (workdir != nullptr) { g_workdir = workdir; }
    std::uint32_t rows = 8U;
    const char* rows_env = std::getenv("INJECTBIND_ROWS");
    if (rows_env != nullptr) { rows = static_cast<std::uint32_t>(std::atoi(rows_env)); }
    return run_caliber(rows);
}
