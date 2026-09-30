#pragma once

// src/spec/inject_channel.h -- THE INGRESS SURFACE: HOW A CHOSEN TENSOR BECOMES CONTEXT.
//
// WHY THIS HEADER EXISTS. Nine of the ten input channels this engine already has are TOKEN
// channels, and the tenth -- `media` -- carries *bytes that decode to an image*
// (src/product/prompt_input/prompt_input.cpp acquires them, media_acquire fetches them), i.e. its
// alphabet is the RANGE of `merger o layers o patch_embed` and nothing else. An arbitrary tensor
// therefore cannot be submitted: there is no field that accepts one. That is the wall, and this
// header is the field.
//
// WHAT `sum_dir.h:98` REQUIRES, IN ITS OWN WORDS. It is the rule this file has to satisfy, not a
// header it may delegate to:
//
//   "ETIQUETTE TOWARD 95-3 (injected recall must look to the model like ordinary context)
//    Nothing in this file is ever rendered into a prompt. A row is host-side bookkeeping, so the
//    rule does not bind this header; it binds the future patch that turns a row back into tokens,
//    which is listed as a blocker in the report rather than assumed away here."
//
// So the requirement is: WHAT A CALLER INJECTS MUST BE INDISTINGUISHABLE, TO THE MODEL, FROM
// ORDINARY CONTEXT. This header cannot satisfy that by declaration and does not pretend to. It
// satisfies it the only way the engine allows, which is stated as three named facts:
//
//   (F1) THE CONSUMPTION POINT IS THE INPUT-EMBEDDING MATRIX, NOT A LATER ONE.
//        The bytes this channel delivers land in the same tensor `x` (BF16 [hidden, tokens],
//        contiguous, ne[0]=hidden fastest) that `ops::embedding` fills and that `run_layers`
//        consumes -- at the same column, before the same first-layer rmsnorm. There is no
//        separate "injected" tensor and no separate arithmetic: after the write, nothing in the
//        forward pass can tell an injected column from a gathered one, because there is nothing
//        left to tell them apart with.
//
//   (F2) THE ALPHABET IS THE EMBEDDING SPACE, SO A ROUND TRIP IS EXACT, NOT APPROXIMATE.
//        A payload whose rows are `embed(t)` for tokens t the engine could have gathered itself
//        must produce, at the declared columns, the SAME 16-bit words the gather would have
//        written. Under `dtype=bf16, scale=1` this header performs no arithmetic at all (the
//        identity conversion is bit-exact by construction -- see element_to_bf16 below), so that
//        equality is BYTE equality, and the etiquette test is an equality test with no tolerance
//        to choose. `sum_dir.h:98` is thereby a MEASUREMENT, not an aspiration: see
//        inject_channel_test.cpp (the host half) and dl/injectchan/REPORT.md (the engine half).
//
//   (F3) AND THE DECLARED RANGE IS CONTIGUOUS, WHICH IS A STATED LIMIT, NOT AN OVERSIGHT.
//        A gap-free run is the only shape the tree's own recall path can rebuild today
//        (src/spec/turn_recall_journal.h:1261: "keeps the selected set a *contiguous, gap-free*
//        run, which is the only shape `warm_cold_prefix` can restore today"). A scattered set
//        would need each block positioned by its own `positions[i]`
//        (turn_recall_journal.h:2349-2356, `position_contiguous()`), i.e. the vision path's
//        index-driven `ops::scatter`, and this channel deliberately does not open that spelling:
//        a contiguous run is ONE cudaMemcpy of a known length, and a length is checkable.
//
// WHAT THIS HEADER IS NOT. It is host-only and std-only on purpose -- the same discipline
// spec/sum_dir.h states for itself ("It is std-only (like this file), so including it does not
// cost the plain-g++ property"). It contains no CUDA, no artifact and no engine header, so the
// WHOLE admission surface is testable with plain g++ and the engine's only obligation is to hand
// it a ModelLimits and copy the bytes it returns. It touches no device, allocates no arena and has
// no persistent state.
//
// IT ALSO DOES NOT DECIDE WHETHER A PAYLOAD IS A GOOD IDEA. Admission is about whether the engine
// CAN consume the declaration as written; there is no opinion here about whether the vectors are
// useful. Every field is either checked or reported un-checked; nothing is accepted and ignored.

#include "spec/fnv_convention.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::spec::inject {

// ---------------------------------------------------------------------------
// THE VOCABULARY
// ---------------------------------------------------------------------------

// Which way the bytes travel. INGEST is the wall being opened: a chosen tensor becomes context.
// EGRESS is the same declaration read backwards -- it writes the columns the engine WOULD have
// computed at the declared positions to `path` -- and it exists for one reason that is worth
// stating: without it, "a payload equivalent to tokens the engine could have computed" has no
// definition that is not a re-implementation of the embedding table on the host, i.e. a second
// gather that the test would be comparing against. With it, the payload used by the etiquette
// test IS the engine's own gather, byte for byte, taken from the very matrix the token path
// consumes. EGRESS therefore exists to make the ingest test honest, not as a feature.
enum class Direction { Ingest, Egress };

// The dtype the PAYLOAD FILE is written in. The engine's alphabet at the consumption point is
// BF16 only (ops/embedding.h: "out is contiguous BF16 [D,T]"), so f16 and f32 are declared
// conversions and their arithmetic is stated in element_to_bf16 below.
enum class ElementType { Bf16, F16, F32 };

// EVERY REFUSAL HAS A NAME, because a refusal a caller cannot read is a silent failure with a
// stderr line. The order of the enumerators is NOT the order of the checks; admission_check()
// states the order, and the first failing field is the one reported.
enum class Refusal {
    None,
    // spec file
    RefusedSpecUnreadable,       // the spec file could not be opened at all
    RefusedSpecSyntax,           // a non-blank, non-comment line that is not key=value
    RefusedFieldUnknown,         // a key that is not in this vocabulary
    RefusedFieldMissing,         // a mandatory key is absent
    RefusedFieldDuplicate,       // the same key appears twice
    RefusedFieldNotInteger,      // a key whose field admits only an integer did not get one
    // declaration
    RefusedDirectionUnsupported, // direction is neither ingest nor egress
    RefusedDtypeUnsupported,     // dtype is not bf16|f16|f32
    RefusedLayoutUnsupported,    // layout is not the one spelling the engine consumes
    RefusedPositionNegative,     // position0 < 0
    RefusedColumnsZero,          // cols < 1: an empty injection is not a no-op, it is unmeasurable
    RefusedScaleInvalid,         // scale is not finite, or is <= 0
    RefusedDigestMalformed,      // digest= is neither "fnv1a64:0x..." nor a bare hex word
    // against the engine
    RefusedRowsMismatch,              // rows != the model's hidden
    RefusedColumnsExceedContext,      // cols > the engine's context capacity
    RefusedPositionRangeOutOfContext, // position0 + cols - 1 exceeds the context capacity
    // against the payload
    RefusedPathEmpty,
    RefusedFileMissing,
    RefusedFileSizeMismatch,     // bytes != rows * cols * size_of(dtype); stated in the detail
    RefusedSourceDigestMismatch, // the file's FNV-1a 64 is not the one the caller pinned
    RefusedNonFinite,            // an element is inf/nan after the scale (names its index)
    // raised by the engine's own application of an admitted declaration
    RefusedOverlapsVisionScatter, // the declared columns collide with the vision path's columns
    RefusedRangeNotFullyApplied,  // the run's prefill never covered the whole declared range
};

// The name of a refusal, spelled once. Callers print this; nothing else may spell it.
[[nodiscard]] inline const char* refusal_name(Refusal refusal) noexcept {
    switch (refusal) {
    case Refusal::None:
        return "none";
    case Refusal::RefusedSpecUnreadable:
        return "refused-spec-unreadable";
    case Refusal::RefusedSpecSyntax:
        return "refused-spec-syntax";
    case Refusal::RefusedFieldUnknown:
        return "refused-field-unknown";
    case Refusal::RefusedFieldMissing:
        return "refused-field-missing";
    case Refusal::RefusedFieldDuplicate:
        return "refused-field-duplicate";
    case Refusal::RefusedFieldNotInteger:
        return "refused-field-not-integer";
    case Refusal::RefusedDirectionUnsupported:
        return "refused-direction-unsupported";
    case Refusal::RefusedDtypeUnsupported:
        return "refused-dtype-unsupported";
    case Refusal::RefusedLayoutUnsupported:
        return "refused-layout-unsupported";
    case Refusal::RefusedPositionNegative:
        return "refused-position-negative";
    case Refusal::RefusedColumnsZero:
        return "refused-columns-zero";
    case Refusal::RefusedScaleInvalid:
        return "refused-scale-invalid";
    case Refusal::RefusedDigestMalformed:
        return "refused-digest-malformed";
    case Refusal::RefusedRowsMismatch:
        return "refused-rows-mismatch";
    case Refusal::RefusedColumnsExceedContext:
        return "refused-columns-exceed-context";
    case Refusal::RefusedPositionRangeOutOfContext:
        return "refused-position-range-out-of-context";
    case Refusal::RefusedPathEmpty:
        return "refused-path-empty";
    case Refusal::RefusedFileMissing:
        return "refused-file-missing";
    case Refusal::RefusedFileSizeMismatch:
        return "refused-file-size-mismatch";
    case Refusal::RefusedSourceDigestMismatch:
        return "refused-source-digest-mismatch";
    case Refusal::RefusedNonFinite:
        return "refused-non-finite";
    case Refusal::RefusedOverlapsVisionScatter:
        return "refused-overlaps-vision-scatter";
    case Refusal::RefusedRangeNotFullyApplied:
        return "refused-range-not-fully-applied";
    }
    return "refused-unspelled";
}

// The layout spelling the engine consumes. It is not a preference: `ne[0]` is the fastest-varying
// dimension (core/tensor.cpp:51-56, set_contiguous_strides), so a BF16 [hidden, tokens] matrix is
// stored token-major -- token t's hidden vector contiguous, token t+1 immediately after it. A
// payload in any other order would be scattered into the model as a permutation of itself, which
// is exactly the silent-corruption class this record keeps finding. One spelling, named, checked.
inline constexpr std::string_view kIngestLayout = "token-major";

// ---------------------------------------------------------------------------
// WHAT THE ENGINE CAN CONSUME (supplied by the engine -- NOT by the caller)
// ---------------------------------------------------------------------------

struct ModelLimits {
    // The consumption point's row count. ops::embedding's `out` must be [d,T] with d == the table's
    // hidden width, and the same tensor is what run_layers reads: a payload whose rows are anything
    // else cannot be consumed, and the refusal is by name rather than by a truncated copy.
    std::int32_t hidden = 0;
    // The engine's own context capacity in positions (EngineOptions / the KV plan's token
    // capacity). A declared range beyond it is refused here rather than clipped later.
    std::int32_t context_capacity = 0;
    // The alphabet the consumption point accepts. BF16 today, and this field exists so that the
    // day it is not, the declaration that names the old one is refused by name instead of
    // converted silently.
    ElementType consumed_dtype = ElementType::Bf16;
};

// ---------------------------------------------------------------------------
// WHAT THE CALLER DECLARES
// ---------------------------------------------------------------------------

struct Declaration {
    Direction direction   = Direction::Ingest;
    ElementType dtype     = ElementType::Bf16;
    std::string layout;                    // must equal kIngestLayout
    std::int32_t rows     = 0;             // == hidden
    std::int32_t cols     = 0;             // number of positions; >= 1
    std::int32_t position0 = 0;            // first absolute position; the range is [p0, p0+cols)
    double scale          = 1.0;           // applied in fp32 BEFORE the conversion; must be finite, > 0
    std::string path;                      // the payload file (ingest: read; egress: written)
    bool has_digest       = false;         // if set, the FILE's fnv1a64 must equal `digest`
    std::uint64_t digest  = 0;

    [[nodiscard]] std::int32_t last_position() const noexcept {
        return position0 + cols - 1;
    }
};

// The outcome of every check that can be made without touching the payload's bytes.
struct Settlement {
    Refusal refusal = Refusal::None;
    std::string field;   // WHICH field failed -- empty only for whole-file refusals
    std::string detail;  // the numbers the refusal is about; never a restatement of the name

    [[nodiscard]] bool admitted() const noexcept { return refusal == Refusal::None; }
};

// The outcome of every check that needs the bytes.
struct PayloadSettlement {
    Refusal refusal = Refusal::None;
    std::string field;
    std::string detail;
    // The exact bytes that will be copied into the input-embedding matrix, token-major, BF16.
    // Empty iff refused. `settled()` is the predicate, not `!payload.empty()`: a zero-column
    // declaration is refused, so an empty payload can never be a legitimate admission, but saying
    // which is the predicate removes the ambiguity rather than relying on it.
    std::vector<std::uint16_t> payload;
    std::uint64_t source_digest = 0; // FNV-1a 64 over the FILE's bytes (what the caller can pin)
    std::uint64_t baked_digest  = 0; // FNV-1a 64 over `payload`'s bytes (what the model gets)

    [[nodiscard]] bool settled() const noexcept { return refusal == Refusal::None; }
};

// ---------------------------------------------------------------------------
// PARSING: the flat `key=value` spelling, and every way it can be wrong
// ---------------------------------------------------------------------------
//
// A flat text file, not JSON: this header is std-only by (the same) design, so it may not pull a
// JSON parser into every translation unit that includes it, and the grammar is small enough to
// state in full -- one `key=value` per line, `#` comment lines, blank lines skipped, no quoting,
// no escapes, no nesting. Every key is checked against the vocabulary above; nothing is ignored.
//
// THE KEYS, AND WHICH ARE MANDATORY:
//     direction   ingest | egress                       mandatory
//     dtype       bf16 | f16 | f32                      mandatory
//     layout      token-major                           mandatory
//     rows        integer > 0                           mandatory
//     cols        integer > 0                           mandatory
//     position0   integer >= 0                          mandatory
//     path        the payload file                      mandatory
//     scale       finite, > 0                           optional, default 1.0
//     digest      fnv1a64:0x... | 0x... | decimal       optional; pins the FILE's bytes
//
// `scale` and `digest` are the only optional keys. Every other key's absence is
// `refused-field-missing` NAMING THE KEY, because a defaulted geometry is a geometry the caller
// did not choose.

[[nodiscard]] inline std::string_view trim(std::string_view text) noexcept {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t' || text.front() == '\r' ||
                             text.front() == '\n')) {
        text.remove_prefix(1);
    }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r' ||
                             text.back() == '\n')) {
        text.remove_suffix(1);
    }
    return text;
}

[[nodiscard]] inline bool parse_i64(std::string_view text, std::int64_t& out) noexcept {
    if (text.empty()) { return false; }
    std::size_t at = 0;
    bool negative  = false;
    if (text[at] == '+' || text[at] == '-') {
        negative = text[at] == '-';
        ++at;
        if (at == text.size()) { return false; }
    }
    std::int64_t value = 0;
    for (; at < text.size(); ++at) {
        const char c = text[at];
        if (c < '0' || c > '9') { return false; }
        value = value * 10 + (c - '0');
        if (value > std::numeric_limits<std::int64_t>::max() / 16) { return false; }
    }
    out = negative ? -value : value;
    return true;
}

[[nodiscard]] inline bool parse_f64(std::string_view text, double& out) noexcept {
    if (text.empty()) { return false; }
    std::string buffer(text);
    char* end = nullptr;
    const double value = std::strtod(buffer.c_str(), &end);
    if (end == nullptr || *end != '\0') { return false; }
    out = value;
    return true;
}

// `digest=` accepts the three spellings a person actually writes: the prefixed form the engine
// prints (`fnv1a64:0x...`), a bare hex word, and a decimal. Anything else is
// refused-digest-malformed NAMING the spelling, rather than being accepted as 0 -- a digest that
// silently parses to zero matches nothing, and a check that can only report a mismatch is not the
// check the caller asked for.
[[nodiscard]] inline bool parse_digest(std::string_view text, std::uint64_t& out) noexcept {
    std::string_view body = trim(text);
    const std::string_view prefix = "fnv1a64:";
    if (body.starts_with(prefix)) { body.remove_prefix(prefix.size()); }
    body = trim(body);
    if (body.empty()) { return false; }
    int base = 10;
    if (body.size() > 2 && body[0] == '0' && (body[1] == 'x' || body[1] == 'X')) {
        body.remove_prefix(2);
        base = 16;
        if (body.empty()) { return false; }
    }
    std::uint64_t value = 0;
    for (char c : body) {
        int digit = -1;
        if (c >= '0' && c <= '9') {
            digit = c - '0';
        } else if (base == 16 && c >= 'a' && c <= 'f') {
            digit = 10 + (c - 'a');
        } else if (base == 16 && c >= 'A' && c <= 'F') {
            digit = 10 + (c - 'A');
        } else {
            return false;
        }
        const auto radix = static_cast<std::uint64_t>(base);
        if (value > (std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(digit)) /
                        radix) {
            return false;
        }
        value = value * radix + static_cast<std::uint64_t>(digit);
    }
    out = value;
    return true;
}

struct ParseResult {
    Settlement settlement;
    Declaration declaration;
    [[nodiscard]] bool ok() const noexcept { return settlement.admitted(); }
};

// The whole spec text -> a Declaration, or the FIRST thing wrong with it, by name. The order is
// fixed and is part of the contract: a line's own shape is checked before its key is looked up, a
// key's value is checked before the next key is read, and the whole-file verdicts (unknown,
// missing, duplicate) are reported last so that a malformed VALUE is never masked by a MISSING
// KEY elsewhere.
[[nodiscard]] inline ParseResult parse_spec(std::string_view text) {
    ParseResult result;
    Declaration& decl = result.declaration;

    bool seen_direction = false, seen_dtype = false, seen_layout = false;
    bool seen_rows = false, seen_cols = false, seen_position0 = false;
    bool seen_path = false, seen_scale = false, seen_digest = false;

    std::size_t line_number = 0;
    while (!text.empty()) {
        ++line_number;
        const std::size_t newline = text.find('\n');
        std::string_view line = newline == std::string_view::npos ? text : text.substr(0, newline);
        text = newline == std::string_view::npos ? std::string_view{} : text.substr(newline + 1);
        line = trim(line);
        if (line.empty() || line.front() == '#') { continue; }

        const std::size_t equals = line.find('=');
        if (equals == std::string_view::npos) {
            result.settlement = {Refusal::RefusedSpecSyntax, {},
                                 "line " + std::to_string(line_number) +
                                     " is neither blank, a # comment, nor key=value"};
            return result;
        }
        const std::string_view key = trim(line.substr(0, equals));
        const std::string_view value = trim(line.substr(equals + 1));
        if (key.empty()) {
            result.settlement = {Refusal::RefusedSpecSyntax, {},
                                 "line " + std::to_string(line_number) + " has an empty key"};
            return result;
        }

        const auto duplicate = [&](bool& seen, std::string_view name) -> bool {
            if (seen) {
                result.settlement = {Refusal::RefusedFieldDuplicate, std::string(name),
                                     "line " + std::to_string(line_number) +
                                         " repeats a key already given"};
                return true;
            }
            seen = true;
            return false;
        };

        if (key == "direction") {
            if (duplicate(seen_direction, key)) { return result; }
            if (value == "ingest") {
                decl.direction = Direction::Ingest;
            } else if (value == "egress") {
                decl.direction = Direction::Egress;
            } else {
                result.settlement = {Refusal::RefusedDirectionUnsupported, std::string(key),
                                     "declared '" + std::string(value) +
                                         "'; the vocabulary is ingest|egress"};
                return result;
            }
        } else if (key == "dtype") {
            if (duplicate(seen_dtype, key)) { return result; }
            if (value == "bf16") {
                decl.dtype = ElementType::Bf16;
            } else if (value == "f16") {
                decl.dtype = ElementType::F16;
            } else if (value == "f32") {
                decl.dtype = ElementType::F32;
            } else {
                result.settlement = {Refusal::RefusedDtypeUnsupported, std::string(key),
                                     "declared '" + std::string(value) +
                                         "'; the payload may be written bf16|f16|f32"};
                return result;
            }
        } else if (key == "layout") {
            if (duplicate(seen_layout, key)) { return result; }
            decl.layout.assign(value);
        } else if (key == "rows" || key == "cols" || key == "position0") {
            std::int64_t parsed = 0;
            // ROWS AND COLS MUST BE NON-NEGATIVE HERE, BUT position0 MAY BE NEGATIVE, AND THE
            // ASYMMETRY IS DELIBERATE: refusing a negative position at PARSE time would make
            // RefusedPositionNegative unreachable and leave the guard dead -- a needle that can
            // only report 0 is not a predicate. Letting the sign through to the admission check
            // keeps that refusal a live branch with a named field, and keeps `cols=0` (which the
            // parse cannot refuse, and which the admission refuses as RefusedColumnsZero) live in
            // the same way.
            const bool signed_field = key == "position0";
            if (!parse_i64(value, parsed) || parsed > std::numeric_limits<std::int32_t>::max() ||
                (!signed_field && parsed < 0) || parsed < std::numeric_limits<std::int32_t>::min()) {
                result.settlement = {Refusal::RefusedFieldNotInteger, std::string(key),
                                     "'" + std::string(value) + "' is not a " +
                                         (signed_field ? "signed" : "non-negative") +
                                         " 32-bit integer"};
                return result;
            }
            if (key == "rows") {
                if (duplicate(seen_rows, key)) { return result; }
                decl.rows = static_cast<std::int32_t>(parsed);
            } else if (key == "cols") {
                if (duplicate(seen_cols, key)) { return result; }
                decl.cols = static_cast<std::int32_t>(parsed);
            } else {
                if (duplicate(seen_position0, key)) { return result; }
                decl.position0 = static_cast<std::int32_t>(parsed);
            }
        } else if (key == "scale") {
            if (duplicate(seen_scale, key)) { return result; }
            double parsed = 0.0;
            if (!parse_f64(value, parsed)) {
                result.settlement = {Refusal::RefusedFieldNotInteger, std::string(key),
                                     "'" + std::string(value) + "' is not a number"};
                return result;
            }
            decl.scale = parsed;
        } else if (key == "path") {
            if (duplicate(seen_path, key)) { return result; }
            decl.path.assign(value);
        } else if (key == "digest") {
            if (duplicate(seen_digest, key)) { return result; }
            std::uint64_t parsed = 0;
            if (!parse_digest(value, parsed)) {
                result.settlement = {Refusal::RefusedDigestMalformed, std::string(key),
                                     "'" + std::string(value) +
                                         "' is none of fnv1a64:0x<hex>, 0x<hex>, <hex>, <decimal>"};
                return result;
            }
            decl.has_digest = true;
            decl.digest     = parsed;
        } else {
            result.settlement = {
                Refusal::RefusedFieldUnknown, std::string(key),
                "line " + std::to_string(line_number) +
                    " names a key this channel does not have; the vocabulary is "
                    "direction|dtype|layout|rows|cols|position0|scale|path|digest"};
            return result;
        }
    }

    // Whole-file verdicts, reported only after every line was individually well formed.
    const std::pair<const char*, bool> mandatory[] = {
        {"direction", seen_direction}, {"dtype", seen_dtype},   {"layout", seen_layout},
        {"rows", seen_rows},           {"cols", seen_cols},     {"position0", seen_position0},
        {"path", seen_path}};
    for (const auto& [name, seen] : mandatory) {
        if (!seen) {
            result.settlement = {Refusal::RefusedFieldMissing, name,
                                 "the spec does not declare it and it has no default"};
            return result;
        }
    }
    return result;
}

[[nodiscard]] inline ParseResult parse_spec_file(const std::string& path) {
    ParseResult result;
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        result.settlement = {Refusal::RefusedSpecUnreadable, {},
                             "'" + path + "' could not be opened"};
        return result;
    }
    std::string text((std::istreambuf_iterator<char>(stream)),
                     std::istreambuf_iterator<char>());
    return parse_spec(text);
}

// ---------------------------------------------------------------------------
// ADMISSION AGAINST THE ENGINE, in this order, and each step names its own field
// ---------------------------------------------------------------------------

[[nodiscard]] inline Settlement admit_against_engine(const Declaration& decl,
                                                     const ModelLimits& limits) {
    if (decl.layout != kIngestLayout) {
        return {Refusal::RefusedLayoutUnsupported, "layout",
                "declared '" + decl.layout + "'; the engine consumes '" +
                    std::string(kIngestLayout) +
                    "' only (ne[0] is the fastest-varying dimension, core/tensor.cpp:51-56)"};
    }
    if (decl.rows != limits.hidden) {
        return {Refusal::RefusedRowsMismatch, "rows",
                "declared " + std::to_string(decl.rows) + "; the input-embedding matrix is [" +
                    std::to_string(limits.hidden) + ", tokens]"};
    }
    if (decl.cols < 1) {
        return {Refusal::RefusedColumnsZero, "cols",
                "declared " + std::to_string(decl.cols) +
                    " columns; a zero-column injection consumes nothing and would be reported as "
                    "an admission"};
    }
    if (decl.position0 < 0) {
        return {Refusal::RefusedPositionNegative, "position0",
                "declared " + std::to_string(decl.position0) + "; positions are absolute and start at 0"};
    }
    if (!std::isfinite(decl.scale) || decl.scale <= 0.0) {
        return {Refusal::RefusedScaleInvalid, "scale",
                "declared " + std::to_string(decl.scale) +
                    "; the scale is applied in fp32 before the conversion and must be finite and > 0"};
    }
    if (decl.cols > limits.context_capacity) {
        return {Refusal::RefusedColumnsExceedContext, "cols",
                std::to_string(decl.cols) + " columns exceed the context capacity " +
                    std::to_string(limits.context_capacity)};
    }
    if (decl.last_position() >= limits.context_capacity) {
        return {Refusal::RefusedPositionRangeOutOfContext, "position0",
                "range [" + std::to_string(decl.position0) + ", " +
                    std::to_string(decl.last_position()) + "] exceeds the context capacity " +
                    std::to_string(limits.context_capacity)};
    }
    if (decl.path.empty()) {
        return {Refusal::RefusedPathEmpty, "path", "the payload path is empty"};
    }
    if (decl.dtype != limits.consumed_dtype && decl.dtype != ElementType::Bf16) {
        // f16/f32 ARE admitted against a BF16 consumption point -- that is the declared conversion
        // below -- but a declaration naming a dtype the engine does not have a conversion INTO is
        // refused here rather than converted by a rule nobody stated.
        return {Refusal::RefusedDtypeUnsupported, "dtype",
                "the consumption point's alphabet is BF16 and this channel converts only from "
                "bf16|f16|f32"};
    }
    return {};
}

// ---------------------------------------------------------------------------
// THE ARITHMETIC, STATED
// ---------------------------------------------------------------------------
//
[[nodiscard]] inline std::uint16_t fp32_to_bf16_rne(float value) noexcept {
    // fp32 -> bf16, round-to-nearest-even. The 16 dropped mantissa bits are rounded by adding the
    // half-way constant plus the dropped field's own low bit, which is the classic RNE form: the
    // +lsb makes the tie break toward the even neighbour instead of always upward.
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
}

// f16 -> fp32, WIDE and therefore EXACT for every input including subnormals: every finite f16
// is exactly representable in fp32 (5 exponent bits and 10 mantissa bits are strictly fewer
// than 8 and 23), so this step rounds nothing and cannot be the source of a disagreement with
// the gather path.
[[nodiscard]] inline float fp16_to_fp32(std::uint16_t bits) noexcept {
    const std::uint32_t sign     = static_cast<std::uint32_t>(bits >> 15) << 31;
    const std::uint32_t exponent = (bits >> 10) & 0x1FU;
    const std::uint32_t mantissa = bits & 0x3FFU;
    std::uint32_t fp32           = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            fp32 = sign; // +/- 0
        } else {
            // Subnormal f16: renormalise into fp32's exponent range, exactly. The 1 of the f16
            // subnormal's implicit leading position is shifted up until it reaches bit 10.
            std::uint32_t normalised = mantissa;
            std::uint32_t exponent_out = 127U - 15U + 1U;
            while ((normalised & 0x400U) == 0) {
                normalised <<= 1U;
                --exponent_out;
            }
            fp32 = sign | (exponent_out << 23) | ((normalised & 0x3FFU) << 13);
        }
    } else if (exponent == 0x1FU) {
        fp32 = sign | 0x7F800000U | (mantissa << 13); // inf / nan, carried through to be refused
    } else {
        fp32 = sign | ((exponent - 15U + 127U) << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &fp32, sizeof(value));
    return value;
}

// element_to_bf16 is the ONLY place this channel performs arithmetic, and it performs it in fp32
// between two exact widening/narrowing steps:
//
//   bf16 -> fp32: wide (exact);  fp32 -> bf16: ROUND-TO-NEAREST-EVEN on the 16-bit payload.
//   f16  -> fp32: wide (exact);  fp32 -> bf16: round-to-nearest-even.
//   f32  -> bf16: round-to-nearest-even.
//
// RNE, and not truncation and not round-half-away, because that is what the engine's own BF16 casts
// do and a second rounding rule would make the ingest path disagree with the gather path on exactly
// the values where the disagreement is invisible. THE CONSEQUENCE THAT MATTERS: for
// `dtype=bf16, scale=1.0` the whole function is the identity -- bf16 -> fp32 is exact, the
// multiply by 1.0f is exact, and the RNE narrowing of an exactly-representable bf16 value is the
// same 16 bits -- so the bytes copied into the model are the file's bytes, and the etiquette test
// is BYTE equality rather than equality within a tolerance.
//
// NaN and infinity are REFUSED (refused-non-finite, naming the element index) rather than
// propagated: the tree's own vector admission refuses non-finite embeddings for the same reason
// (spec/sum_dir_vector.h, SumDirVectorSetResult::RefusedNotFinite), and an injected inf is a
// silent corruption of every column after it once the first rmsnorm runs.
[[nodiscard]] inline std::uint16_t element_to_bf16(const void* raw, ElementType dtype,
                                                  std::size_t element, double scale,
                                                  float& out_value) noexcept {
    float widened = 0.0f;
    if (dtype == ElementType::Bf16) {
        const auto* words          = static_cast<const std::uint16_t*>(raw);
        const std::uint32_t fp32   = static_cast<std::uint32_t>(words[element]) << 16;
        std::memcpy(&widened, &fp32, sizeof(widened));
    } else if (dtype == ElementType::F16) {
        const auto* words = static_cast<const std::uint16_t*>(raw);
        widened           = fp16_to_fp32(words[element]);
    } else {
        const auto* words = static_cast<const float*>(raw);
        widened           = words[element];
    }
    out_value = static_cast<float>(widened * static_cast<float>(scale));
    return fp32_to_bf16_rne(out_value);
}

// ---------------------------------------------------------------------------
// THE GEOMETRY: where a declared range meets one prefill chunk
// ---------------------------------------------------------------------------
//
// Factored out of the engine's application loop ON PURPOSE, and it is the same reason the rest of
// this header is host-only: the two ways a declaration can be wrong about a chunk -- it does not
// reach it, or it collides with the vision path's columns in it -- are the refusals a negative
// control has to provoke, and a refusal that can only be provoked by running a model is a refusal
// nobody checks. The engine calls these; the self-test exercises them.
//
// THE CONVENTION, once: both the declared range and a chunk are half-open intervals of ABSOLUTE
// positions. `position_base` is the absolute position of `x`'s column 0, so column c of the chunk
// holds absolute position (position_base + c), and (lo - position_base) is a column of x while
// (lo - position0) is a column of the PAYLOAD. Those two offsets are the same number only when the
// declaration starts exactly where the chunk starts, which is why the overlap record carries both.
struct Overlap {
    std::int32_t first_column         = 0; // first column of x covered, in the chunk's column space
    std::int32_t count                = 0; // number of x columns covered
    std::size_t payload_element_offset = 0; // element index into the payload's token-major payload
};

[[nodiscard]] inline std::optional<Overlap> overlap_with_chunk(const Declaration& decl,
                                                               std::int32_t position_base,
                                                               std::int32_t chunk_columns) noexcept {
    const std::int32_t p0 = decl.position0;
    const std::int32_t p1 = decl.last_position();
    const std::int32_t lo = p0 > position_base ? p0 : position_base;
    const std::int32_t hi_exclusive = (p1 + 1) < (position_base + chunk_columns)
                                          ? (p1 + 1)
                                          : (position_base + chunk_columns);
    if (hi_exclusive <= lo) { return std::nullopt; }
    Overlap overlap;
    overlap.first_column = lo - position_base;
    overlap.count        = hi_exclusive - lo;
    overlap.payload_element_offset =
        static_cast<std::size_t>(lo - p0) * static_cast<std::size_t>(decl.rows);
    return overlap;
}

// Does the covered run touch any of `columns` (which are in the CHUNK's column space, as the
// vision path's destination columns are)? Returns the first colliding column or -1.
[[nodiscard]] inline std::int32_t first_collision(const Overlap& overlap, const std::int32_t* columns,
                                                 std::size_t column_count) noexcept {
    for (std::size_t i = 0; i < column_count; ++i) {
        if (columns[i] >= overlap.first_column &&
            columns[i] < overlap.first_column + overlap.count) {
            return columns[i];
        }
    }
    return -1;
}

// The completeness bookkeeping, as a BITMAP rather than a counter, so that a schedule which
// records its plan and then executes it -- which TextContext does, being a stack object per
// recording/execution -- covers each declared column once for the purposes of the verdict
// instead of reporting twice as many columns as were declared.
struct Coverage {
    std::vector<std::uint8_t> covered;
    std::int32_t count = 0;

    void reset(std::int32_t columns) {
        covered.assign(static_cast<std::size_t>(columns < 0 ? 0 : columns), 0U);
        count = 0;
    }
    // Returns true when this column had not been covered before.
    bool cover(std::int32_t column) {
        if (column < 0 || static_cast<std::size_t>(column) >= covered.size()) { return false; }
        if (covered[static_cast<std::size_t>(column)] != 0U) { return false; }
        covered[static_cast<std::size_t>(column)] = 1U;
        ++count;
        return true;
    }
    [[nodiscard]] bool complete() const noexcept {
        return !covered.empty() && static_cast<std::size_t>(count) == covered.size();
    }
};

[[nodiscard]] inline std::uint64_t fnv1a64(const std::uint8_t* bytes, std::size_t count) noexcept {
    std::uint64_t digest = fnv::kFnv1a64OffsetBasis;
    for (std::size_t i = 0; i < count; ++i) {
        digest ^= static_cast<std::uint64_t>(bytes[i]);
        digest *= fnv::kFnv1a64Prime;
    }
    return digest;
}

[[nodiscard]] inline std::size_t element_bytes(ElementType dtype) noexcept {
    return dtype == ElementType::F32 ? 4U : 2U;
}

// Read the payload, convert it, and settle it. This is the only place a byte is read or written.
//
// THE DIGESTS, AND WHICH ONE A CALLER PINS. `source_digest` is over the FILE's bytes, and it is
// the one `digest=` in the spec is compared against, because it is the one the caller can compute
// without knowing this header's conversion. `baked_digest` is over `payload` -- the bytes that
// actually enter the input-embedding matrix -- and it is the one the engine PRINTS, because a
// digest of what the model did not receive is not a reading of what the model received.
[[nodiscard]] inline PayloadSettlement bake(const Declaration& decl) {
    PayloadSettlement result;
    std::ifstream stream(decl.path, std::ios::binary);
    if (!stream) {
        result.refusal = Refusal::RefusedFileMissing;
        result.field   = "path";
        result.detail  = "'" + decl.path + "' could not be opened for reading";
        return result;
    }
    std::vector<std::uint8_t> source((std::istreambuf_iterator<char>(stream)),
                                     std::istreambuf_iterator<char>());
    const std::uint64_t elements =
        static_cast<std::uint64_t>(decl.rows) * static_cast<std::uint64_t>(decl.cols);
    const std::uint64_t expected = elements * static_cast<std::uint64_t>(element_bytes(decl.dtype));
    if (static_cast<std::uint64_t>(source.size()) != expected) {
        result.refusal = Refusal::RefusedFileSizeMismatch;
        result.field   = "path";
        result.detail  = "'" + decl.path + "' is " + std::to_string(source.size()) +
                        " bytes; rows*cols*sizeof(" +
                        (decl.dtype == ElementType::Bf16 ? "bf16"
                         : decl.dtype == ElementType::F16 ? "f16" : "f32") +
                        ") = " + std::to_string(expected);
        return result;
    }
    result.source_digest = fnv1a64(source.data(), source.size());
    if (decl.has_digest && result.source_digest != decl.digest) {
        result.refusal = Refusal::RefusedSourceDigestMismatch;
        result.field   = "digest";
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "fnv1a64:0x%016llx",
                      static_cast<unsigned long long>(result.source_digest));
        result.detail = std::string("the file hashes to ") + buffer + ", not the pinned " +
                        std::to_string(decl.digest);
        return result;
    }

    result.payload.resize(static_cast<std::size_t>(elements));
    for (std::uint64_t i = 0; i < elements; ++i) {
        float value = 0.0f;
        const std::uint16_t baked = element_to_bf16(source.data(), decl.dtype,
                                                    static_cast<std::size_t>(i), decl.scale, value);
        if (!std::isfinite(value)) {
            result.refusal = Refusal::RefusedNonFinite;
            result.field   = "path";
            result.detail  = "element " + std::to_string(i) + " (row " +
                            std::to_string(i % static_cast<std::uint64_t>(decl.rows)) + ", column " +
                            std::to_string(i / static_cast<std::uint64_t>(decl.rows)) +
                            ") is not finite after the scale";
            result.payload.clear();
            return result;
        }
        result.payload[static_cast<std::size_t>(i)] = baked;
    }
    result.baked_digest =
        fnv1a64(reinterpret_cast<const std::uint8_t*>(result.payload.data()),
                result.payload.size() * sizeof(std::uint16_t));
    return result;
}

// One call for the whole host-side admission: parse, admit against the engine, bake.
struct Admission {
    Settlement settlement;
    Declaration declaration;
    PayloadSettlement payload;
    [[nodiscard]] bool admitted() const noexcept {
        return settlement.admitted() && payload.settled();
    }
};

[[nodiscard]] inline Admission admit(std::string_view spec_text, const ModelLimits& limits) {
    Admission result;
    ParseResult parsed  = parse_spec(spec_text);
    result.settlement   = parsed.settlement;
    result.declaration  = parsed.declaration;
    if (!parsed.ok()) { return result; }
    result.settlement = admit_against_engine(result.declaration, limits);
    if (!result.settlement.admitted()) { return result; }
    result.payload = bake(result.declaration);
    return result;
}

// ---------------------------------------------------------------------------
// OBSERVABILITY: what the engine prints, spelled once, so a reading and a claim cannot drift
// ---------------------------------------------------------------------------
//
// An ingress nobody can see is an ingress nobody can trust, so the admitted line carries EVERY
// field the caller declared plus the two measurements that are not in the declaration: the byte
// count actually consumed and the digest of the bytes actually consumed. The refusal line carries
// the refusal's NAME, the FIELD it is about, and the numbers -- so a reader can tell a wrong
// declaration from a wrong guard without reading this file.

[[nodiscard]] inline const char* dtype_spelling(ElementType dtype) noexcept {
    switch (dtype) {
    case ElementType::Bf16:
        return "bf16";
    case ElementType::F16:
        return "f16";
    case ElementType::F32:
        return "f32";
    }
    return "unspelled";
}

[[nodiscard]] inline std::string render_refusal(const Refusal refusal, const std::string& field,
                                                const std::string& detail) {
    std::string line = "[inject] REFUSED ";
    line += refusal_name(refusal);
    if (!field.empty()) {
        line += " field=";
        line += field;
    }
    if (!detail.empty()) {
        line += " : ";
        line += detail;
    }
    return line;
}

[[nodiscard]] inline std::string render_admission(const Declaration& decl,
                                                  const PayloadSettlement& payload) {
    char digest[80];
    std::snprintf(digest, sizeof(digest), "fnv1a64:0x%016llx",
                  static_cast<unsigned long long>(payload.baked_digest));
    char source[80];
    std::snprintf(source, sizeof(source), "fnv1a64:0x%016llx",
                  static_cast<unsigned long long>(payload.source_digest));
    const bool identity = decl.dtype == ElementType::Bf16 && decl.scale == 1.0;
    std::string line = "[inject] ADMITTED direction=";
    line += decl.direction == Direction::Ingest ? "ingest" : "egress";
    line += " dtype=";
    line += dtype_spelling(decl.dtype);
    line += identity ? "(identity)" : "(converted)";
    line += " layout=";
    line += decl.layout;
    line += " shape=[rows=" + std::to_string(decl.rows) + ", cols=" + std::to_string(decl.cols) +
            "]";
    line += " positions=[" + std::to_string(decl.position0) + ".." +
            std::to_string(decl.last_position()) + "]";
    line += " scale=" + std::to_string(decl.scale);
    line += " bytes=" + std::to_string(payload.payload.size() * sizeof(std::uint16_t));
    line += " source_digest=";
    line += source;
    line += " ingested_digest=";
    line += digest;
    line += " path=";
    line += decl.path;
    return line;
}

} // namespace ninfer::spec::inject
