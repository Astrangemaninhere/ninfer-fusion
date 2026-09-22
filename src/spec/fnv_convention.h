#pragma once

// src/spec/fnv_convention.h -- THE FNV-1a 64 CONVENTIONS THIS TREE USES, in one owning place.
//
// WHY THIS HEADER EXISTS. Two offset-basis values are alive in this tree and they
// differ by exactly a factor of ten, because ONE DECIMAL DIGIT of the published
// literal is missing from the other:
//
//     published FNV-1a 64   0xcbf29ce484222325 = 14695981039346656037   (20 digits)
//     the engine's lane 0                        1469598103934665603    (19 digits)
//
// Both are legal FNV (any non-zero basis is), BOTH ARE ALREADY SPELT INTO LANDED
// ARTEFACTS AND TESTS, and neither may be "tidied" into the other: changing either
// value changes every digest computed with it, which invalidates records already
// on disk. So this header does not fix anything -- it PINS what is there, names
// which artefact depends on which convention, and makes a third convention
// impossible to add by accident.
//
// HOW THE PIN WORKS. The static_asserts at the bottom do not merely record the
// two values: they assert the exact arithmetic relation between them, which only
// the two values actually in use can satisfy.
//
//     kFnv1a64OffsetBasis / 10 == kEngineDigestLane0Offset   (decimal truncation)
//     kFnv1a64OffsetBasis %  10 == 7                         (the dropped digit)
//     kFnv1a64OffsetBasis      != kEngineDigestLane0Offset   (never unify them)
//
// A reader who "corrects" either value makes this header fail to compile, and the
// only way back to green is to delete or rewrite a tripwire -- a deliberate,
// visible act rather than a tidy-up. tests/test_fnv_convention.cpp adds the third
// half of the pin: it walks src/, tests/ and tools/, collects every 64-bit value
// that is one of the known spellings, and asserts the DISTINCT VALUE SET is
// exactly the five registered here. A sixth value -- a new convention -- fails
// that test and prints where it came from.
//
// WHO USES WHICH (this is the dependency record; keep it in step with the test):
//
//   PUBLISHED convention (this pair)
//     product/kv_recall_block.h:188-189      recall_block_digest() -- its own
//                                            comment states the reason: a pure
//                                            fold shared by the write side (host,
//                                            at eviction) and the read side.
//     tools/ple_table_test.cu:31-32          PLE table byte digest (tool).
//     tools/ple_gather_test.cu:45-46         PLE gather check (tool).
//     tools/archkit/ple_gather_check.py:231  the Python twin of the above.
//
//   ENGINE convention (lane-0 offset + lane-1 offset + two primes + lane skew)
//     targets/qwen3_6/impl/runtime/prefix_identity.cpp:60-61  THE OWNER: the
//                                            rolling 128-bit prefix/block
//                                            shortlist digest. Not included from
//                                            here (a target's impl/ pulls engine
//                                            headers), so the test pins its values
//                                            BY TEXT against this header.
//     targets/qwen3_6/impl/runtime/program_impl.h:1484,1817,7734   single-lane
//     targets/qwen3_6/impl/runtime/pressure_planner.h:213,1029     single-lane
//     runtime/engine/resource_manager.h:1225-1229 (two-lane), :2646  single-lane
//     product/kv_rowscale_persist.h:144      single-lane
//     spec/sum_dir.h                         row/catalogue digests -- now aliases
//                                            of the constants below instead of a
//                                            re-spelling (this header is why).
//     tests/test_resource_manager.cpp:1387   the test's own recomputation.
//
// NOT UNIFIED HERE, AND WHY: those other single-lane sites spell the engine's
// lane-0 offset inline. Several of them live in another author's zone, so this
// header records them rather than editing them; unifying the SPELLING is safe and
// welcome (include this header and use the names), but changing any VALUE is not.

#include <cstdint>

namespace ninfer::spec::fnv {

// ---------------------------------------------------------------------------
// convention 1: the published FNV-1a 64 pair
// offset_basis = 14695981039346656037 = 0xcbf29ce484222325, prime = 1099511628211 = 0x100000001b3
// ---------------------------------------------------------------------------
inline constexpr std::uint64_t kFnv1a64OffsetBasis = 0xcbf29ce484222325ULL;
inline constexpr std::uint64_t kFnv1a64Prime       = 0x100000001b3ULL;

// ---------------------------------------------------------------------------
// convention 2: THE ENGINE'S convention, exactly as the owner spells it
// (targets/qwen3_6/impl/runtime/prefix_identity.cpp:60-61). Two lanes, two
// primes; lane 1 feeds rotl(value ^ kEngineDigestLaneSkew, kEngineDigestLaneRotate)
// (that file's :29-31 and :103-111).
//
// kEngineDigestLane0Offset is NOT the published basis: it is the published basis
// with its last decimal digit dropped. It is retained, not corrected, because
// digests computed with it have already landed.
// ---------------------------------------------------------------------------
inline constexpr std::uint64_t kEngineDigestLane0Offset = 1469598103934665603ULL;
inline constexpr std::uint64_t kEngineDigestLane1Offset = 7809847782465536322ULL;
inline constexpr std::uint64_t kEngineDigestPrime0      = 1099511628211ULL;
inline constexpr std::uint64_t kEngineDigestPrime1      = 14029467366897019727ULL;
inline constexpr std::uint64_t kEngineDigestLaneSkew    = 0x9e3779b97f4a7c15ULL;
inline constexpr std::uint32_t kEngineDigestLaneRotate  = 29U;

// How many distinct conventions this tree is allowed to contain. A third one must
// be registered here, in the static_asserts below, and in
// tests/test_fnv_convention.cpp's expected-value set -- there is no way to add one
// quietly, and that is the whole point of this header.
inline constexpr int kFnvConventionCount = 2;

// ---------------------------------------------------------------------------
// the tripwires (compile-time)
// ---------------------------------------------------------------------------

static_assert(kFnvConventionCount == 2,
              "a third digest convention needs to be registered in src/spec/fnv_convention.h "
              "and in tests/test_fnv_convention.cpp -- that is deliberate friction, not a bug");

static_assert(kFnv1a64OffsetBasis / 10U == kEngineDigestLane0Offset,
              "the engine's lane-0 offset is the published FNV-1a 64 basis truncated by one "
              "decimal digit (14695981039346656037 -> 1469598103934665603). If you changed "
              "either constant you changed a digest that landed records depend on -- see the "
              "file comment for the dependency list");
static_assert(kFnv1a64OffsetBasis % 10U == 7U,
              "the digit the engine's convention drops is the trailing 7 of 14695981039346656037");
static_assert(kFnv1a64OffsetBasis != kEngineDigestLane0Offset,
              "the two conventions must NOT be unified: each is referenced by landed artefacts "
              "(the list is in this header's comment)");

// The engine's lane 0 reuses the published prime; the two lanes must differ, or
// there would be only one lane; and the rotate count is part of the convention
// (a different count is a different digest over the same bytes).
static_assert(kEngineDigestPrime0 == kFnv1a64Prime,
              "the engine's lane-0 prime is the published FNV prime; if this ever stops being "
              "true, that is a new convention, not a tidy-up");
static_assert(kEngineDigestPrime1 != kEngineDigestPrime0,
              "the two lanes must not share a prime");
static_assert(kEngineDigestLaneRotate > 0U && kEngineDigestLaneRotate < 64U,
              "a rotation must be a real rotation");

} // namespace ninfer::spec::fnv
