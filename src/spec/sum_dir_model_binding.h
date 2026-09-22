#pragma once

// src/spec/sum_dir_model_binding.h -- THE FOURTH ARTEFACT: A PER-DIRECTORY MODEL FINGERPRINT.
// (line `indexgaps`, gap (3))
//
// THE FAILURE IT EXISTS TO CATCH, NAMED
//   The retrieval key is a function of the block's IDS and of nothing else (the ONE RULE,
//   sum_dir.h:89-96; re-stated at sum_dir_vector.h:30-32). A WEIGHT CHANGE does not move it. The
//   re-prefill's OUTPUT does. So after a weight set moves, the key still verifies, the block is
//   still found, `DETERMINED` is still TRUE -- and the reproduction has silently changed. No
//   existing artefact can see it, and that is the whole reason this file exists:
//
//     block_identity            -- a function of the ids; MUST NOT carry model or position
//                                  (sum_dir.h:89-96, and sum_dir.h:1640 says so again)
//     payload_digest            -- covers *rows ++ summaries ++ content* (sum_dir.h:1988-2010):
//                                  the DIRECTORY's own bytes, and it moves whenever a row is
//                                  rewritten for any reason at all
//     directory_payload_digest  -- binds a RETRIEVAL COLUMN to a DIRECTORY
//                                  (sum_dir_vector.h:273-275, :295, refused at :689); it does not
//                                  carry a model at all
//
//   THE TREE HAS THE PATTERN THREE TIMES AND THIS IS THE FOURTH: per-row content digest, per-file
//   payload CRC, per-column directory binding, and now PER-DIRECTORY MODEL BINDING.
//
// POSITION AND DISCIPLINE: THE SAME AS `directory_payload_digest`
//   * stored ONCE per directory, not per row;
//   * a value that belongs to the ARTEFACT and travels with it, so a directory a different model
//     wrote cannot be read back as if it were this one's;
//   * compared BEFORE a read-back is admitted, and a disagreement is a NAMED REFUSAL rather than a
//     re-interpretation -- the same shape as `sum_dir_vector.h:689`'s refusal;
//   * covered by its own header seal, over [0, 36), for the same reason the vector column's seal
//     excludes its own field (`kSumDirVectorHeaderCovered`, sum_dir_vector.h:279-283): a seal that
//     covered itself could never agree between writer and reader.
//
// WHY IT IS ITS OWN FILE AND NOT A FIELD IN sum_dir.h's HEADER -- stated rather than glossed
//   The design asks for this value in the DIRECTORY header. `SumDir`'s header is a FROZEN 48-byte
//   unit (`kSumDirHeaderBytes`, magic `SRD1`, a version byte, and `payload_digest` defined over
//   *rows ++ summaries ++ content*), and `sum_dir.h`'s own doctrine for the neighbouring landing is
//   that a new column gets its own file and its own magic so that every byte an older build wrote
//   still loads (sum_dir_vector.h:26-29, stated for exactly this reason). Adding a field to
//   `SumDir`'s 48 bytes would change what `payload_digest` covers for every existing artefact, i.e.
//   it would break the compatibility property rather than extend the header.
//   SO THIS IS A **PER-DIRECTORY SIDEcar**, carrying the SAME discipline at the same position in the
//   read-back path. It is one file per directory, keyed to that directory by its `payload_digest`.
//   ⛔ The residual difference is real and is NOT hidden: it is not literally a field inside
//   `SumDir`'s 48-byte header. Folding it there needs a `SumDir` format version bump, which is the
//   owner's call and is not made here.
//
// WHAT IT COVERS -- the three components the design names, each a value and not a name
//   1. TOKENIZER identity   -- a digest of the tokenizer's own definition. Two tokenizers can
//                              produce colliding ids for different text; a string name cannot
//                              detect that, a digest of the definition can.
//   2. WEIGHT-SET identity  -- a digest of the weights the blocks were GENERATED under. This is the
//                              component that moves when the re-prefill's output moves.
//   3. GENERATION-ID digest -- a digest of the ids the blocks were generated under, i.e. of the
//                              content the model itself produced into this directory.
//
// WHAT THIS HEADER DOES NOT DO, NAMED SO IT IS NOT MISTAKEN FOR DONE
//   * It does not COMPUTE the three components. Hashing a tokenizer's definition or a weight set is
//     an engine-side job with a real cost; this header is the artefact, the wire format and the
//     comparison, i.e. the half that is provably correct once a producer exists. Every component is
//     taken from the caller exactly as `sum_dir_vector.h` takes a float vector from the caller and
//     refuses to embed anything itself.
//   * It is not wired into the engine, and it deliberately has no engine-side caller: the same
//     property `sum_dir_vector.h:49-52` claims, for the same reason.
//   * It does NOT replace `sum_dir_row_bound()`. That check answers "may this row be offered"; this
//     one answers "may this DIRECTORY's bytes be read back at all". Both must hold.

#include "spec/sum_dir.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::spec::sum_dir::model_binding {

// ---------------------------------------------------------------------------
// wire format -- its own magic, its own version, its own seal
// ---------------------------------------------------------------------------

inline constexpr std::uint32_t kSumDirModelBindingMagic = 0x31424D53U; // 'S','M','B','1'
inline constexpr std::uint8_t  kSumDirModelBindingVersion = 1U;
inline constexpr std::uint32_t kSumDirModelBindingHeaderBytes = 48U;
// The seal covers [0, 36) and NOT [0, 40): its own field lives at [36, 40), and a seal that covered
// itself could never agree between the writer (which hashes while the field is still zero) and the
// reader (which hashes with the field already written).
inline constexpr std::uint32_t kSumDirModelBindingHeaderCovered = 36U;

// A component that is 0 means UNSET, and an unset component is a refusal rather than a wildcard --
// "we do not know" must never compare equal to anything. This is the same discipline as
// `SumDirDigest::is_unset()` and as `sum_dir_vector.h`'s `dim_missing`, which refuses to arm an
// embedder whose width nobody stated.
inline constexpr std::uint64_t kSumDirModelComponentUnset = 0ULL;

// The fingerprint itself. Three 64-bit components, so the artefact is 24 bytes of payload and one
// 48-byte header with room to spare -- deliberately NOT three digests, because the comparison
// semantics are per-component ("the tokenizer moved" and "the weights moved" are different
// refusals, and collapsing them would lose which one it was).
struct SumDirModelFingerprint {
    std::uint64_t tokenizer_identity   = kSumDirModelComponentUnset; // 0 == unset
    std::uint64_t weights_identity     = kSumDirModelComponentUnset;
    std::uint64_t generation_ids_digest = kSumDirModelComponentUnset;

    [[nodiscard]] constexpr bool is_unset() const noexcept {
        return tokenizer_identity == kSumDirModelComponentUnset ||
               weights_identity == kSumDirModelComponentUnset ||
               generation_ids_digest == kSumDirModelComponentUnset;
    }
    [[nodiscard]] friend bool operator==(const SumDirModelFingerprint& a,
                                         const SumDirModelFingerprint& b) noexcept {
        return a.tokenizer_identity == b.tokenizer_identity &&
               a.weights_identity == b.weights_identity &&
               a.generation_ids_digest == b.generation_ids_digest;
    }
    [[nodiscard]] friend bool operator!=(const SumDirModelFingerprint& a,
                                         const SumDirModelFingerprint& b) noexcept {
        return !(a == b);
    }
};

struct SumDirModelBindingHeader {
    std::uint32_t magic    = kSumDirModelBindingMagic; //  0
    std::uint8_t  version  = kSumDirModelBindingVersion; // 4
    std::uint8_t  reserved8[3] = {0, 0, 0};           //  5..7
    std::uint64_t tokenizer_identity   = kSumDirModelComponentUnset; //  8
    std::uint64_t weights_identity     = kSumDirModelComponentUnset; // 16
    std::uint64_t generation_ids_digest = kSumDirModelComponentUnset; // 24
    // The DIRECTORY this binding was taken against. Same position and same job as
    // `SumDirVectorHeader::directory_payload_digest`: a binding attached to a DIFFERENT directory
    // is a refusal, not a re-interpretation.
    std::uint32_t directory_payload_digest = 0;        // 32
    std::uint32_t header_digest            = 0;        // 36  crc over [0, 36)
    std::uint32_t reserved0 = 0;                       // 40  must be 0
    std::uint32_t reserved1 = 0;                       // 44  must be 0
};
static_assert(sizeof(SumDirModelBindingHeader) == kSumDirModelBindingHeaderBytes,
              "the model binding header is one fixed unit; a packed reader must see the same 48");

// The three components ARE the fingerprint; this accessor exists so a caller names the concept
// rather than the offsets, and so a stored header can be compared without a second type.
[[nodiscard]] inline SumDirModelFingerprint
sum_dir_model_fingerprint(const SumDirModelBindingHeader& header) noexcept {
    SumDirModelFingerprint fingerprint;
    fingerprint.tokenizer_identity    = header.tokenizer_identity;
    fingerprint.weights_identity      = header.weights_identity;
    fingerprint.generation_ids_digest = header.generation_ids_digest;
    return fingerprint;
}

// ---------------------------------------------------------------------------
// the comparison -- one named verdict per thing that moved
// ---------------------------------------------------------------------------

enum class SumDirModelBindingVerdict : std::uint8_t {
    Bound                     = 0, // every component agrees
    RefusedFingerprintUnset   = 1, // this binding carries no model -- "unknown" is not a wildcard
    RefusedDirectoryMismatch  = 2, // the binding belongs to a different directory
    RefusedTokenizerChanged   = 3,
    RefusedWeightSetChanged   = 4, // ⭐ the one that makes "DETERMINED stays true, the reproduction
                                  //    changes" visible
    RefusedGenerationChanged  = 5,
    RefusedFormatChanged      = 6, // magic / version / seal -- a format disagreement
};

[[nodiscard]] constexpr const char* sum_dir_model_binding_verdict_name(
    SumDirModelBindingVerdict verdict) noexcept {
    switch (verdict) {
    case SumDirModelBindingVerdict::Bound: return "bound";
    case SumDirModelBindingVerdict::RefusedFingerprintUnset: return "refused-fingerprint-unset";
    case SumDirModelBindingVerdict::RefusedDirectoryMismatch: return "refused-directory-mismatch";
    case SumDirModelBindingVerdict::RefusedTokenizerChanged: return "refused-tokenizer-changed";
    case SumDirModelBindingVerdict::RefusedWeightSetChanged: return "refused-weight-set-changed";
    case SumDirModelBindingVerdict::RefusedGenerationChanged: return "refused-generation-changed";
    case SumDirModelBindingVerdict::RefusedFormatChanged: return "refused-format-changed";
    }
    return "unknown";
}

// THE COMPARISON. Three independent, ORDERED tests, so a run can tell which component moved --
// checked weights FIRST among the three, because a moved weight set is the case that leaves the key
// valid and the output changed, i.e. the case nothing else in the tree can see.
[[nodiscard]] inline SumDirModelBindingVerdict sum_dir_model_binding_check(
    const SumDirModelBindingHeader& stored, const SumDirModelFingerprint& current,
    std::uint32_t directory_payload_digest) noexcept {
    if (stored.magic != kSumDirModelBindingMagic) {
        return SumDirModelBindingVerdict::RefusedFormatChanged;
    }
    if (stored.version != kSumDirModelBindingVersion) {
        return SumDirModelBindingVerdict::RefusedFormatChanged;
    }
    if (stored.directory_payload_digest != directory_payload_digest) {
        return SumDirModelBindingVerdict::RefusedDirectoryMismatch;
    }
    if (stored.weights_identity == kSumDirModelComponentUnset ||
        current.weights_identity == kSumDirModelComponentUnset) {
        return SumDirModelBindingVerdict::RefusedFingerprintUnset;
    }
    if (stored.weights_identity != current.weights_identity) {
        return SumDirModelBindingVerdict::RefusedWeightSetChanged;
    }
    if (stored.tokenizer_identity == kSumDirModelComponentUnset ||
        current.tokenizer_identity == kSumDirModelComponentUnset) {
        return SumDirModelBindingVerdict::RefusedFingerprintUnset;
    }
    if (stored.tokenizer_identity != current.tokenizer_identity) {
        return SumDirModelBindingVerdict::RefusedTokenizerChanged;
    }
    if (stored.generation_ids_digest == kSumDirModelComponentUnset ||
        current.generation_ids_digest == kSumDirModelComponentUnset) {
        return SumDirModelBindingVerdict::RefusedFingerprintUnset;
    }
    if (stored.generation_ids_digest != current.generation_ids_digest) {
        return SumDirModelBindingVerdict::RefusedGenerationChanged;
    }
    return SumDirModelBindingVerdict::Bound;
}

// The ergonomic form: build the header from a fingerprint and a directory digest. Serialisation and
// the seal live below; this is what a writer calls.
[[nodiscard]] inline SumDirModelBindingHeader sum_dir_model_binding_make(
    const SumDirModelFingerprint& fingerprint, std::uint32_t directory_payload_digest) noexcept {
    SumDirModelBindingHeader header;
    header.tokenizer_identity    = fingerprint.tokenizer_identity;
    header.weights_identity      = fingerprint.weights_identity;
    header.generation_ids_digest = fingerprint.generation_ids_digest;
    header.directory_payload_digest = directory_payload_digest;
    return header;
}

// ---------------------------------------------------------------------------
// serialization -- 48 bytes, one fixed unit, its own seal
// ---------------------------------------------------------------------------

[[nodiscard]] inline std::vector<std::uint8_t>
sum_dir_model_binding_serialize(const SumDirModelBindingHeader& header) noexcept {
    std::vector<std::uint8_t> out(kSumDirModelBindingHeaderBytes, 0U);
    std::uint8_t* wire = out.data();
    detail::sum_dir_pack_u32_at(wire, 0, kSumDirModelBindingMagic);
    wire[4] = kSumDirModelBindingVersion;
    wire[5] = 0;
    wire[6] = 0;
    wire[7] = 0;
    detail::sum_dir_pack_u64_at(wire, 8, header.tokenizer_identity);
    detail::sum_dir_pack_u64_at(wire, 16, header.weights_identity);
    detail::sum_dir_pack_u64_at(wire, 24, header.generation_ids_digest);
    detail::sum_dir_pack_u32_at(wire, 32, header.directory_payload_digest);
    detail::sum_dir_pack_u32_at(wire, 36, 0U); // the seal hashes while its own field is zero
    detail::sum_dir_pack_u32_at(wire, 40, 0U);
    detail::sum_dir_pack_u32_at(wire, 44, 0U);
    std::uint32_t seal = sum_dir_crc32(0xFFFFFFFFU, wire, kSumDirModelBindingHeaderCovered);
    seal = ~seal;
    detail::sum_dir_pack_u32_at(wire, 36, seal);
    return out;
}

struct SumDirModelBindingLoadReport {
    bool ok         = false;
    bool sealed     = false; // the header seal agreed
    bool truncated  = false;
    bool wrong_magic = false;
    bool wrong_version = false;
    std::string error;
};

[[nodiscard]] inline SumDirModelBindingHeader
sum_dir_model_binding_load(const std::uint8_t* wire, std::size_t bytes,
                           SumDirModelBindingLoadReport* report = nullptr) noexcept {
    SumDirModelBindingHeader header;
    // EVERY failure path returns this value, and its magic is ZEROED: a caller must never be handed
    // a half-believed header whose magic still claims to be one this reader understood. A default
    // constructed `SumDirModelBindingHeader` carries the real magic, which is exactly the trap this
    // line closes.
    SumDirModelBindingHeader refused;
    refused.magic = 0;
    SumDirModelBindingLoadReport local;
    if (wire == nullptr || bytes < kSumDirModelBindingHeaderBytes) {
        local.truncated = true;
        local.error = "model binding: fewer than 48 bytes";
        if (report != nullptr) { *report = local; }
        return refused;
    }
    local.sealed = (detail::sum_dir_unpack_u32(wire, 36) ==
                    static_cast<std::uint32_t>(
                        ~sum_dir_crc32(0xFFFFFFFFU, wire, kSumDirModelBindingHeaderCovered)));
    const std::uint32_t magic = detail::sum_dir_unpack_u32(wire, 0);
    local.wrong_magic = magic != kSumDirModelBindingMagic;
    local.wrong_version = wire[4] != kSumDirModelBindingVersion;
    if (local.wrong_magic) {
        local.error = "model binding: magic is not 'SMB1'";
    } else if (local.wrong_version) {
        local.error = "model binding: version is not 1";
    } else if (!local.sealed) {
        local.error = "model binding: header seal disagrees -- the artefact is not the one written";
    } else if (detail::sum_dir_unpack_u32(wire, 40) != 0U ||
               detail::sum_dir_unpack_u32(wire, 44) != 0U) {
        // The two reserved words must be zero; a non-zero one is a format this reader does not know,
        // which is a refusal and not an ignored field (the vector header's own rule).
        local.error = "model binding: a reserved word is non-zero";
    } else {
        local.ok = true;
        header.magic = magic;
        header.version = wire[4];
        header.tokenizer_identity    = detail::sum_dir_unpack_u64(wire, 8);
        header.weights_identity      = detail::sum_dir_unpack_u64(wire, 16);
        header.generation_ids_digest = detail::sum_dir_unpack_u64(wire, 24);
        header.directory_payload_digest = detail::sum_dir_unpack_u32(wire, 32);
        header.header_digest = detail::sum_dir_unpack_u32(wire, 36);
    }
    if (report != nullptr) { *report = local; }
    return local.ok ? header : refused;
}

} // namespace ninfer::spec::sum_dir::model_binding
