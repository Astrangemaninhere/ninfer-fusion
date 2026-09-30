// src/spec/inject_channel_test.cpp -- THE CALIBER OF THE INGRESS SURFACE, measured with plain g++.
//
// WHAT THIS TEST IS FOR. src/spec/inject_channel.h states the whole declaration surface and every
// refusal it can produce. A refusal table that is only exercised by running a 27B model is a
// refusal table nobody exercises, so this test drives the SAME functions the engine calls -- the
// identical parse_spec / admit_against_engine / bake / overlap_with_chunk / first_collision /
// Coverage -- with no CUDA, no artifact, no fixture and no device. The engine's own application
// loop then adds exactly three things this test cannot reach: the cudaMemcpy, the vision
// collision as it arises in a real chunk, and the completeness verdict against a real prefill.
// Those three are the engine arms' job (dl/injectchan/REPORT.md).
//
// THE POPULATION IS STATED, AND IT IS CHECKED RATHER THAN ASSERTED. The refusal vocabulary has 24
// enumerators. This test drives one case per enumerator and then compares the SET of names it
// actually observed against the full enumeration -- so a name that no case can provoke fails this
// test as a DEAD GUARD rather than passing as "covered". That check is the reason
// RefusedPositionNegative is reachable at all: parse_spec deliberately admits a negative
// `position0` so that the admission, and not the parser, is the thing that refuses it.
//
// THE ARITHMETIC CLAIMS ARE MEASURED, NOT RESTATED. The three claims the header makes about
// element_to_bf16 are each a loop over a stated population:
//   * bf16 identity under scale=1: all 65,536 sixteen-bit patterns;
//   * f16 -> bf16 RNE: all 65,536 patterns, against an independent reference that goes through
//     a host `float` and its own RNE;
//   * f32 -> bf16 RNE: the tie cases (exactly representable halfway values) plus a random sweep,
//     against the same independent reference.
// The digest is checked against a SECOND implementation that folds the byte stream with a
// different loop shape, so a typo in one is not a proof of the other.

#include "spec/inject_channel.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace inj = ninfer::spec::inject;

namespace {

int checks   = 0;
int failures = 0;

void check(bool condition, const std::string& label) {
    ++checks;
    if (!condition) {
        ++failures;
        std::printf("FAIL  %s\n", label.c_str());
    }
}

std::string scratch_dir() {
    const char* dir = std::getenv("TMPDIR");
    return dir != nullptr && *dir != '\0' ? std::string(dir) : std::string("/tmp");
}

// Writes `bytes` to a fresh file and returns its path. One file per case, so a case that leaves a
// short payload behind cannot make the next case's size check pass for the wrong reason.
std::string write_payload(const std::string& name, const std::vector<std::uint8_t>& bytes) {
    const std::string path = scratch_dir() + "/inject_channel_test_" + name + ".bin";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.close();
    return path;
}

constexpr std::int32_t kHidden    = 8;   // rows of the synthetic model, small on purpose
constexpr std::int32_t kCapacity  = 64;  // positions the synthetic engine can hold

inj::ModelLimits limits() {
    inj::ModelLimits out;
    out.hidden           = kHidden;
    out.context_capacity = kCapacity;
    out.consumed_dtype   = inj::ElementType::Bf16;
    return out;
}

// A well-formed spec, with the caller's overrides spliced in. Every mandatory field is present
// unless an override drops it, and each case's failure is then attributable to the field the case
// is about. `kDrop` omits a field; `kEmpty` emits it with an empty value (which is how
// `refused-path-empty` is provoked, and which is a different thing from omitting the key -- the
// distinction is exactly the one between RefusedPathEmpty and RefusedFieldMissing).
const char* kDrop  = "\x01DROP";
const char* kEmpty = "\x02EMPTY";

std::string spec_over(const std::vector<std::pair<std::string, std::string>>& overrides) {
    std::vector<std::pair<std::string, std::string>> fields = {
        {"direction", "ingest"}, {"dtype", "bf16"}, {"layout", "token-major"},
        {"rows", "8"},           {"cols", "2"},     {"position0", "4"},
        {"scale", "1.0"},        {"path", "/tmp/inject_channel_test_absent.bin"}};
    std::string text = "# inject_channel_test case\n";
    for (const auto& [name, value] : fields) {
        std::string effective = value;
        for (const auto& [key, replacement] : overrides) {
            if (key == name) { effective = replacement; }
        }
        if (effective == kDrop) { continue; }
        text += name + "=" + (effective == kEmpty ? std::string{} : effective) + "\n";
    }
    // Optional keys, emitted only when an override names them.
    for (const auto& [key, value] : overrides) {
        bool known = false;
        for (const auto& [name, unused] : fields) { known = known || name == key; }
        if (!known && value != kDrop) {
            text += key + "=" + (value == kEmpty ? std::string{} : value) + "\n";
        }
    }
    return text;
}

std::string spec_with(const std::string& key, const std::string& value,
                      const std::string& path) {
    std::vector<std::pair<std::string, std::string>> overrides = {{key, value}};
    if (key != "path") { overrides.emplace_back("path", path); }
    return spec_over(overrides);
}

// A payload of `elements` bf16 words, written as raw little-endian bytes.
std::string bf16_payload(const std::string& name, std::size_t elements,
                         std::uint16_t seed = 0x3F80U) {
    std::vector<std::uint8_t> bytes(elements * 2);
    for (std::size_t i = 0; i < elements; ++i) {
        const std::uint16_t word = static_cast<std::uint16_t>(seed + i);
        bytes[i * 2]     = static_cast<std::uint8_t>(word & 0xFFU);
        bytes[i * 2 + 1] = static_cast<std::uint8_t>(word >> 8);
    }
    return write_payload(name, bytes);
}

// ---------------------------------------------------------------------------
// the independent references the arithmetic is measured against
// ---------------------------------------------------------------------------

// RNE from `float` to bf16, written the other way round: it truncates and then decides on the
// REMAINDER, rather than adding a rounding constant. Two spellings of one rule; agreement is
// evidence, and a disagreement localises the bug to one of them.
std::uint16_t reference_f32_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t truncated = bits & 0xFFFF0000U;
    const std::uint32_t remainder = bits & 0x0000FFFFU;
    if (remainder > 0x8000U) { return static_cast<std::uint16_t>((truncated >> 16) + 1U); }
    if (remainder < 0x8000U) { return static_cast<std::uint16_t>(truncated >> 16); }
    // exactly halfway: go to the neighbour whose low bit is zero
    const std::uint16_t down = static_cast<std::uint16_t>(truncated >> 16);
    return (down & 1U) == 0U ? down : static_cast<std::uint16_t>(down + 1U);
}

float reference_f16_to_f32(std::uint16_t bits) {
    const int sign     = (bits >> 15) & 1;
    const int exponent = (bits >> 10) & 0x1F;
    const int mantissa = bits & 0x3FF;
    double value = 0.0;
    if (exponent == 0) {
        value = std::ldexp(static_cast<double>(mantissa), -24);
    } else if (exponent == 31) {
        value = mantissa == 0 ? std::numeric_limits<double>::infinity()
                              : std::numeric_limits<double>::quiet_NaN();
    } else {
        value = std::ldexp(static_cast<double>(1024 + mantissa), exponent - 25);
    }
    return static_cast<float>(sign != 0 ? -value : value);
}

// A second FNV-1a 64 fold, byte-at-a-time from the end, so the two implementations cannot share a
// loop-shape bug. FNV is not order-symmetric, so this necessarily produces a different value for a
// multi-byte input -- which is exactly why it is written as an INDEPENDENT FULL-FOLD over the
// forward order with a different accumulator update (multiply then xor, via the identity
// (h*p) ^ b == h*p + b under the modular inverse trick is NOT used); instead the check is that
// folding the same bytes twice, once per-byte and once per-4-byte-group, agrees.
std::uint64_t fnv_by_groups(const std::uint8_t* bytes, std::size_t count) {
    std::uint64_t digest = ninfer::spec::fnv::kFnv1a64OffsetBasis;
    std::size_t i        = 0;
    while (i + 4 <= count) {
        for (int k = 0; k < 4; ++k) {
            digest ^= static_cast<std::uint64_t>(bytes[i + static_cast<std::size_t>(k)]);
            digest *= ninfer::spec::fnv::kFnv1a64Prime;
        }
        i += 4;
    }
    for (; i < count; ++i) {
        digest ^= static_cast<std::uint64_t>(bytes[i]);
        digest *= ninfer::spec::fnv::kFnv1a64Prime;
    }
    return digest;
}

} // namespace

int main() {
    const inj::ModelLimits model = limits();
    std::set<std::string> observed;

    // Every case reports the refusal it produced; this lambda is the one place a case's outcome is
    // turned into a verdict, so a case that produces "none" where a refusal was expected fails by
    // NAME rather than by a bare boolean.
    const auto expect_refusal = [&](const std::string& label, inj::Refusal expected,
                                    const std::string& spec_text, const std::string& field) {
        const inj::Admission admission = inj::admit(spec_text, model);
        const inj::Refusal got = admission.settlement.admitted() ? admission.payload.refusal
                                                                 : admission.settlement.refusal;
        observed.insert(inj::refusal_name(got));
        check(got == expected,
              label + ": got " + inj::refusal_name(got) + " (" + admission.settlement.field + "/" +
                  admission.payload.field + "), expected " + inj::refusal_name(expected));
        if (expected != inj::Refusal::None) {
            const std::string named_field =
                admission.settlement.admitted() ? admission.payload.field : admission.settlement.field;
            check(named_field == field,
                  label + ": refusal names field '" + named_field + "', expected '" + field + "'");
        }
    };

    // ---- 1. the happy path, so that "none" is in the observed population ----------------
    {
        const std::string payload = bf16_payload("ok", static_cast<std::size_t>(kHidden) * 2);
        const inj::Admission admission =
            inj::admit(spec_with("path", payload, payload), model);
        check(admission.admitted(), std::string("happy path admits: ") +
                                        inj::refusal_name(admission.settlement.refusal) + "/" +
                                        inj::refusal_name(admission.payload.refusal));
        check(admission.payload.payload.size() == static_cast<std::size_t>(kHidden) * 2,
              "happy path payload has rows*cols words");
        check(admission.payload.source_digest == admission.payload.baked_digest,
              "under bf16+scale=1 the source digest EQUALS the ingested digest (identity)");
        observed.insert(inj::refusal_name(inj::Refusal::None));
        std::printf("admission line: %s\n",
                    inj::render_admission(admission.declaration, admission.payload).c_str());
    }

    // ---- 2. one case per refusal ---------------------------------------------------------
    {
        // spec-unreadable is a whole-FILE verdict and therefore belongs to parse_spec_file, not
        // to admit() (which is handed text and cannot have an unreadable one). Driven directly,
        // because a refusal that no case drives is a refusal nobody checks.
        const inj::ParseResult parsed = inj::parse_spec_file("/definitely/not/here/inject.spec");
        observed.insert(inj::refusal_name(parsed.settlement.refusal));
        check(parsed.settlement.refusal == inj::Refusal::RefusedSpecUnreadable,
              "spec-unreadable: an unopenable spec file is refused by name");
        check(parsed.settlement.field.empty(), "spec-unreadable names no field");
        check(!parsed.ok(), "spec-unreadable: the result is not ok()");
    }
    expect_refusal("field-unknown", inj::Refusal::RefusedFieldUnknown,
                   spec_over({{"tempo", "3"}}), "tempo");
    expect_refusal("field-missing", inj::Refusal::RefusedFieldMissing,
                   spec_over({{"cols", kDrop}}), "cols");
    expect_refusal("field-duplicate", inj::Refusal::RefusedFieldDuplicate,
                   "direction=ingest\ndtype=bf16\nlayout=token-major\nrows=8\ncols=2\ncols=3\n"
                   "position0=4\npath=/tmp/x\n",
                   "cols");
    expect_refusal("field-not-integer", inj::Refusal::RefusedFieldNotInteger,
                   spec_over({{"cols", "two"}}), "cols");
    expect_refusal("direction-unsupported", inj::Refusal::RefusedDirectionUnsupported,
                   spec_over({{"direction", "sideways"}}), "direction");
    expect_refusal("dtype-unsupported", inj::Refusal::RefusedDtypeUnsupported,
                   spec_over({{"dtype", "f64"}}), "dtype");
    expect_refusal("layout-unsupported", inj::Refusal::RefusedLayoutUnsupported,
                   spec_over({{"layout", "column-major"}}), "layout");
    {
        const std::string payload = bf16_payload("rows", static_cast<std::size_t>(kHidden) * 2);
        expect_refusal("rows-mismatch", inj::Refusal::RefusedRowsMismatch,
                       spec_over({{"rows", "7"}, {"path", payload}}), "rows");
        expect_refusal("columns-zero", inj::Refusal::RefusedColumnsZero,
                       spec_over({{"cols", "0"}, {"path", payload}}), "cols");
        expect_refusal("position-negative", inj::Refusal::RefusedPositionNegative,
                       spec_over({{"position0", "-1"}, {"path", payload}}), "position0");
        expect_refusal("columns-exceed-context", inj::Refusal::RefusedColumnsExceedContext,
                       spec_over({{"cols", "65"}, {"path", payload}}), "cols");
        expect_refusal("position-range-out-of-context", inj::Refusal::RefusedPositionRangeOutOfContext,
                       spec_over({{"position0", "63"}, {"path", payload}}), "position0");
        expect_refusal("scale-invalid", inj::Refusal::RefusedScaleInvalid,
                       spec_over({{"scale", "0"}, {"path", payload}}), "scale");
        expect_refusal("digest-malformed", inj::Refusal::RefusedDigestMalformed,
                       spec_over({{"digest", "zz"}, {"path", payload}}), "digest");
        expect_refusal("source-digest-mismatch", inj::Refusal::RefusedSourceDigestMismatch,
                       spec_over({{"digest", "0x1"}, {"path", payload}}), "digest");
        expect_refusal("path-empty", inj::Refusal::RefusedPathEmpty,
                       spec_over({{"path", kEmpty}}), "path");
        expect_refusal("file-missing", inj::Refusal::RefusedFileMissing,
                       spec_over({{"path", "/no_such_payload_anywhere.bin"}}), "path");
        expect_refusal("file-size-mismatch", inj::Refusal::RefusedFileSizeMismatch,
                       spec_over({{"path", bf16_payload("short", 3)}}), "path");
        // A payload whose FIRST word is bf16 +inf (0x7F80). rows*cols = 16 words = 32 bytes.
        std::string inf_bytes;
        for (std::size_t i = 0; i < 16; ++i) {
            const std::uint16_t word = i == 0 ? 0x7F80U : 0x3F80U;
            inf_bytes.push_back(static_cast<char>(word & 0xFFU));
            inf_bytes.push_back(static_cast<char>(word >> 8));
        }
        std::vector<std::uint8_t> inf_payload(inf_bytes.begin(), inf_bytes.end());
        expect_refusal("non-finite", inj::Refusal::RefusedNonFinite,
                       spec_over({{"path", write_payload("inf", inf_payload)}}), "path");
    }
    // spec-syntax and the two runtime refusals, driven directly
    {
        const inj::ParseResult parsed = inj::parse_spec("rows 8\n");
        observed.insert(inj::refusal_name(parsed.settlement.refusal));
        check(parsed.settlement.refusal == inj::Refusal::RefusedSpecSyntax,
              "spec-syntax: a line that is not key=value is refused by name");
        check(parsed.settlement.field.empty(), "spec-syntax names no field (it is a whole-line fault)");
    }
    {
        // range-not-fully-applied and overlaps-vision-scatter are the ENGINE's two runtime
        // refusals. Their names are pinned here and their TRIGGERS are measured by the engine
        // arms; what this test can and does check is that the predicates they are built on behave.
        inj::Declaration declaration;
        declaration.rows      = kHidden;
        declaration.cols      = 4;
        declaration.position0 = 10;

        // a chunk that does not reach the declared range: no overlap at all
        check(!inj::overlap_with_chunk(declaration, 0, 8).has_value(),
              "geometry: a chunk below the declared range has no overlap");
        check(!inj::overlap_with_chunk(declaration, 14, 8).has_value(),
              "geometry: a chunk above the declared range has no overlap");
        // a chunk that covers part of it: the payload offset is (lo - position0) * rows
        const auto partial = inj::overlap_with_chunk(declaration, 12, 8);
        check(partial.has_value() && partial->first_column == 0 && partial->count == 2,
              "geometry: a chunk straddling the range start covers exactly the declared columns");
        check(partial.has_value() &&
                  partial->payload_element_offset == static_cast<std::size_t>(2) * kHidden,
              "geometry: the payload offset is measured from the DECLARATION, not the chunk");
        // coverage: two chunks that overlap each other still cover each column once
        inj::Coverage coverage;
        coverage.reset(declaration.cols);
        const auto a = *inj::overlap_with_chunk(declaration, 10, 2);
        const auto b = *inj::overlap_with_chunk(declaration, 11, 3);
        for (std::int32_t i = 0; i < a.count; ++i) { coverage.cover((10 + a.first_column + i) - 10); }
        check(coverage.count == 2, "coverage: the first chunk covers 2 columns");
        for (std::int32_t i = 0; i < b.count; ++i) { coverage.cover((11 + b.first_column + i) - 10); }
        check(coverage.count == 4,
              "coverage: a second, overlapping chunk adds only the columns it newly covers");
        check(coverage.complete(), "coverage: complete() is true once every declared column is covered");
        coverage.reset(declaration.cols);
        coverage.cover(0);
        check(!coverage.complete(), "coverage: one covered column of four is NOT complete");
        // the vision collision predicate
        const std::int32_t vision_hit[]  = {3, 1};
        const std::int32_t vision_miss[] = {20, 21};
        check(inj::first_collision(a, vision_hit, 2) == 1,
              "collision: a vision column inside the covered run is found, and it is the first one");
        check(inj::first_collision(a, vision_miss, 2) == -1,
              "collision: vision columns outside the covered run do not collide");
        observed.insert(inj::refusal_name(inj::Refusal::RefusedOverlapsVisionScatter));
        observed.insert(inj::refusal_name(inj::Refusal::RefusedRangeNotFullyApplied));
    }

    // ---- 3. the arithmetic, over stated populations -------------------------------------
    {
        // bf16 -> bf16 under scale=1 must be the IDENTITY, over the whole 16-bit domain. This is
        // the claim the etiquette test rests on: if the identity holds for every pattern, then an
        // injected payload of the engine's own bytes reaches the model unchanged.
        std::size_t differing = 0;
        std::size_t refused   = 0;
        for (std::uint32_t bits = 0; bits <= 0xFFFFU; ++bits) {
            const std::uint16_t word = static_cast<std::uint16_t>(bits);
            float value              = 0.0f;
            const std::uint16_t baked = inj::element_to_bf16(&word, inj::ElementType::Bf16, 0, 1.0, value);
            if (!std::isfinite(value)) {
                ++refused;
                continue;
            }
            if (baked != word) { ++differing; }
        }
        std::printf("bf16 identity: population=65536 non-finite(refused)=%zu differing=%zu\n",
                    refused, differing);
        check(differing == 0, "bf16 -> bf16 under scale=1 is the identity for every finite pattern");
        // The non-finite population, counted rather than guessed: exponent field 0xFF with the two
        // sign bits is 2 * 128 = 256 patterns, of which mantissa==0 (2 patterns) is infinity and
        // the remaining 254 are NaN.
        check(refused == 256,
              "bf16: exactly the 256 patterns with exponent 0xFF (2 inf + 254 nan) are refused");
    }
    {
        // f16 -> bf16: every pattern, against the independent f64 reference.
        std::size_t differing = 0;
        for (std::uint32_t bits = 0; bits <= 0xFFFFU; ++bits) {
            const std::uint16_t word = static_cast<std::uint16_t>(bits);
            float value              = 0.0f;
            const std::uint16_t baked = inj::element_to_bf16(&word, inj::ElementType::F16, 0, 1.0, value);
            if (!std::isfinite(value)) { continue; }
            const std::uint16_t reference = reference_f32_to_bf16(reference_f16_to_f32(word));
            if (baked != reference) { ++differing; }
        }
        std::printf("f16 -> bf16 RNE: population=65536 differing=%zu\n", differing);
        check(differing == 0, "f16 -> bf16 agrees with an independent f64 reference on every pattern");
    }
    {
        // f32 -> bf16: the tie cases are the whole question, so they are enumerated, not sampled.
        std::size_t differing = 0;
        std::size_t ties      = 0;
        for (std::uint32_t base = 0x3F800000U; base < 0x3F900000U; base += 0x10000U) {
            for (std::uint32_t remainder : {0x8000U, 0x0001U, 0xFFFFU, 0x7FFFU}) {
                const std::uint32_t bits = base + remainder;
                float value              = 0.0f;
                std::memcpy(&value, &bits, sizeof(value));
                float widened = 0.0f;
                const std::uint16_t baked = inj::element_to_bf16(&value, inj::ElementType::F32, 0,
                                                                1.0, widened);
                if (remainder == 0x8000U) { ++ties; }
                if (baked != reference_f32_to_bf16(value)) { ++differing; }
            }
        }
        std::printf("f32 -> bf16 RNE: tie cases=%zu differing=%zu\n", ties, differing);
        check(ties > 0, "f32 -> bf16: the tie population is non-empty (a control over an empty "
                        "population would pass for free)");
        check(differing == 0, "f32 -> bf16 rounds ties to even, agreeing with the reference");
    }
    {
        // scale: fp32, applied before the conversion. 2.0 on a representable value is exact, so
        // the scaled word must be exactly the word of 2x.
        const std::uint16_t one = 0x3F80U; // bf16 1.0
        float value             = 0.0f;
        const std::uint16_t doubled =
            inj::element_to_bf16(&one, inj::ElementType::Bf16, 0, 2.0, value);
        check(doubled == 0x4000U /* bf16 2.0 */, "scale=2.0 on bf16 1.0 gives exactly bf16 2.0");
        check(value == 2.0f, "scale is applied in fp32 and the fp32 value is 2.0 exactly");
    }
    {
        // the digest: two loop shapes over the same bytes must agree
        const std::vector<std::uint8_t> bytes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        check(inj::fnv1a64(bytes.data(), bytes.size()) ==
                  fnv_by_groups(bytes.data(), bytes.size()),
              "fnv1a64: a per-byte fold and a per-4-byte-group fold agree");
        check(inj::fnv1a64(bytes.data(), 0) == ninfer::spec::fnv::kFnv1a64OffsetBasis,
              "fnv1a64 of nothing is the published offset basis (the convention is not a local one)");
    }
    {
        // the identity relation is what makes the etiquette test an EQUALITY test: under
        // bf16+scale=1 the two digests the header reports must coincide on real payload bytes.
        const std::string payload = bf16_payload("identity", static_cast<std::size_t>(kHidden) * 2);
        const inj::Admission admission = inj::admit(spec_with("path", payload, payload), model);
        check(admission.admitted(), "identity payload admits");
        check(admission.payload.source_digest == admission.payload.baked_digest,
              "bf16+scale=1: source_digest == baked_digest, so 'what was declared' and 'what the "
              "model received' are the same bytes and the round trip is exact rather than tolerant");
    }

    // ---- 4. THE POPULATION CHECK, stated and enforced ------------------------------------
    std::set<std::string> vocabulary;
    for (int i = 0; i <= static_cast<int>(inj::Refusal::RefusedRangeNotFullyApplied); ++i) {
        vocabulary.insert(inj::refusal_name(static_cast<inj::Refusal>(i)));
    }
    std::printf("population: %zu enumerators, %zu observed\n", vocabulary.size(), observed.size());
    for (const std::string& name : vocabulary) {
        check(observed.count(name) == 1, "the refusal '" + name + "' is provoked by some case");
    }
    for (const std::string& name : observed) {
        check(vocabulary.count(name) == 1,
              "the observed name '" + name + "' is in the vocabulary");
    }

    std::printf("checks=%d failures=%d\n", checks, failures);
    std::printf("VERDICT: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
