#pragma once

// THE D256 KV-CACHE DTYPE TABLE, and the ONE place the ops layer learns the leading
// extent of a code plane.
//
// WHY `code_leading_extent` IS HERE AND NOT AT THE CALL SITES
// ----------------------------------------------------------
// The causal-attention validation used to derive it inline as
//     cache.dtype == DType::E8Kv ? kHeadDim / 2 : kHeadDim
// with the comment "everything else stores one code per element". That ternary is an
// admission that the PACKED tiers do not store one byte per element, and it is wrong for
// every packed tier except the one it names: a 3-bit plate is 96 B/row and a 2-bit plate
// is 64 B/row (product/kv_e8_width.h e8_kv_row_code_bytes), not 256. Deriving the extent
// from the dtype here, once, is what keeps a second and third packed width from silently
// inheriting the 4-bit tier's answer.
//
// The extent IS derived from the geometry of record rather than written down: the
// static_asserts below recompute the head-page K plane out of the per-tier row extent and
// the g64 scale block and require it to equal product::e8_kv_k_plane_bytes, so an edit to
// either header that moves the geometry reds this file's compile.
//
// ⚠ THE SCALE BLOCK IS NOT PART OF THIS EXTENT. `scale_leading_extent` is a SEPARATE
// tensor (k_scale_pages / v_scale_pages). Adding the 512 B/head-page scale block into
// code_leading_extent is the double-count the identity below forbids: the correct W3
// extent is 96, never 96 + 8.
//
// ⚠⚠ AND THERE ARE TWO CODE EXTENTS, NOT ONE. `code_leading_extent` is the K PLANE's row
// extent. The V plane's row extent is a SEPARATE field (`v_code_leading_extent`) because
// this family narrows K only: a W3 layer is a 96-byte K plate beside a 128-byte V plate.
// See the field's own comment below for what one shared extent would break.

#include "core/dtype.h"
#include "product/kv_e8_width.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

inline constexpr std::int32_t kD256KVCacheHeadDim = 256;

struct D256KVCacheProfile {
    DType code_dtype;
    std::int32_t quant_group;
    std::int32_t scale_leading_extent;
    // Bytes of CODE per row (one row = one head over one token). One byte per element for
    // the unpacked tiers; product::e8_kv_row_code_bytes() for the packed e8 family.
    std::int32_t code_leading_extent;
    // ⚠ THE V PLANE HAS ITS OWN EXTENT, AND FOR THIS FAMILY IT IS *NOT* code_leading_extent.
    // `product/kv_e8_width.h:133-139` is the rule: "the V plane is NOT a function of the
    // width: this family narrows K only ... the shipped V is i4 at 4 bits + the same g64
    // FP16 scale plane, i.e. the W4 geometry" -- so a W3 row has a 96-byte K plate and a
    // 128-byte V plate, and a W2 row 64 and 128. A consumer that derives ONE extent and
    // checks both planes with it rejects every CORRECT narrow pool (V at 128 != 96/64) or
    // accepts a wrong one; that is why the two are separate fields and not one. For the
    // unpacked tiers V is the same 256 as K, so recognising them here moves nothing.
    std::int32_t v_code_leading_extent;
};

// THE e8 FAMILY, NAMED ONCE. Three tiers share one K+V plane pair and one g64 FP16 scale
// plane; only the K code width differs (4 / 3 / 2 bits). A consumer that can serve the
// shipped 4-bit tier but not the narrow ones -- or that has no e8 arm at all -- must refuse
// the family BY NAME rather than let it fall through a dtype-blind shape check or an
// anonymous allow-list. A refusal that blames the wrong thing (a valid profile reported as
// an "invalid profile", a correctly laid out pool reported as an "invalid shape") is a
// defect in its own right: it sends the next reader to fix the geometry instead of the arm.
[[nodiscard]] constexpr bool d256_kv_cache_is_e8_family(DType dtype) noexcept {
    return dtype == DType::E8Kv || dtype == DType::E8K3Kv || dtype == DType::E8K2Kv;
}

// The scale plane is one FP16 per quant_group channels over the whole head, so its
// leading extent is head_dim / group. Named here so the e8 rows and the identity below
// cannot drift from the geometry header.
inline constexpr std::int32_t kD256ScaleExtentPerGroup =
    product::kE8KvHeadDim / product::kE8KvScaleGroup;   // 4

// THE FAMILY'S V ROW EXTENT, NAMED ONCE. `e8_kv_v_plane_bytes()` is 8704 at every width
// BY CONSTRUCTION (kv_e8_width.h sizes it as the W4 K plane and static_asserts it), so the
// V row extent is the W4 row's code bytes -- 128 -- for E8Kv, E8K3Kv and E8K2Kv alike.
// Writing `e8_kv_row_code_bytes(W4)` here rather than the literal 128 is what makes the
// V extent follow the geometry header if the shipped plane ever moves, and it is the same
// spelling the reader's `kE8AppendVCodeBytesPerRow` is asserted against.
inline constexpr std::int32_t kD256E8VCodeExtent =
    product::e8_kv_row_code_bytes(product::E8KvWidth::W4);   // 128

inline constexpr D256KVCacheProfile d256_kv_cache_profile(DType dtype) {
    switch (dtype) {
    case DType::BF16:
        // Unpacked: V is one code per element too, so the two extents coincide.
        return {DType::BF16, 0, 0, kD256KVCacheHeadDim, kD256KVCacheHeadDim};
    case DType::I8:
        return {DType::I8, 64, 4, kD256KVCacheHeadDim, kD256KVCacheHeadDim};
    case DType::FP8_E4M3FN:
        return {DType::FP8_E4M3FN, 256, 1, kD256KVCacheHeadDim, kD256KVCacheHeadDim};
    case DType::E8Kv:
        // Packed 4-bit E8-lattice K / i4 V with g64 scales (int8-kernel path). At W4 the
        // two extents are equal because K IS the W4 plane; the V field still carries it.
        return {DType::E8Kv, product::kE8KvScaleGroup, kD256ScaleExtentPerGroup,
                product::e8_kv_row_code_bytes(product::E8KvWidth::W4), kD256E8VCodeExtent};
    // THE TWO NARROW TIERS. Same K+V plane pair and the same per-64 FP16 scale plane as
    // E8Kv; only the K code plane is narrower -- 3 bits per element (8 codes per 3 bytes)
    // and 2 bits per element (4 codes per byte). The extent is the row's code bytes:
    // 96 and 64. Recognising them here is what lets a narrow plate be read at ITS OWN
    // extent instead of the 4-bit tier's 128 or the unpacked tiers' 256.
    //
    // ⚠ K NARROWS, V DOES NOT: both narrow rows carry `kD256E8VCodeExtent` (128) for the V
    // plane. A row that put 96/64 here would describe a V plane this family does not have.
    case DType::E8K3Kv:
        return {DType::E8K3Kv, product::kE8KvScaleGroup, kD256ScaleExtentPerGroup,
                product::e8_kv_row_code_bytes(product::E8KvWidth::W3), kD256E8VCodeExtent};
    case DType::E8K2Kv:
        return {DType::E8K2Kv, product::kE8KvScaleGroup, kD256ScaleExtentPerGroup,
                product::e8_kv_row_code_bytes(product::E8KvWidth::W2), kD256E8VCodeExtent};
    default:
        // UNCHANGED AND DELIBERATE. A dtype this table does not know is still refused
        // loudly here; the narrow tiers were added as rows, not by widening this arm.
        throw std::invalid_argument("unsupported D256 KV-cache dtype");
    }
}

// The two head-dim constants the ops layer and the geometry header both fix must agree.
static_assert(kD256KVCacheHeadDim == product::kE8KvHeadDim,
              "ops D256 head dim must equal product::kE8KvHeadDim");

// THE ARITHMETIC GATE. For each e8 tier: the extent this table publishes, times the page's
// rows, plus the g64 FP16 scale block, must equal the geometry of record's K plane.
//   3 bits: 96 * 64 + 4 * 64 * 2 = 6144 + 512 = 6656 B/head-page
//   2 bits: 64 * 64 + 4 * 64 * 2 = 4096 + 512 = 4608 B/head-page
//   4 bits: 128 * 64 + 512       = 8192 + 512 = 8704 B/head-page
// A wrong extent (256, 128-for-W3, 64-for-W4) or a double-counted scale block (+512)
// satisfies none of these, so it cannot compile.
static_assert(d256_kv_cache_profile(DType::E8Kv).code_leading_extent *
                          product::kE8KvPageTokens +
                      kD256ScaleExtentPerGroup * product::kE8KvPageTokens *
                          product::kE8KvScaleBytesPerGroup ==
                  product::e8_kv_k_plane_bytes(product::E8KvWidth::W4),
              "e8 W4: code extent * page tokens + g64 fp16 scales must be 8704 B/head-page");
static_assert(d256_kv_cache_profile(DType::E8K3Kv).code_leading_extent *
                          product::kE8KvPageTokens +
                      kD256ScaleExtentPerGroup * product::kE8KvPageTokens *
                          product::kE8KvScaleBytesPerGroup ==
                  product::e8_kv_k_plane_bytes(product::E8KvWidth::W3),
              "e8 W3: code extent * page tokens + g64 fp16 scales must be 6656 B/head-page");
static_assert(d256_kv_cache_profile(DType::E8K2Kv).code_leading_extent *
                          product::kE8KvPageTokens +
                      kD256ScaleExtentPerGroup * product::kE8KvPageTokens *
                          product::kE8KvScaleBytesPerGroup ==
                  product::e8_kv_k_plane_bytes(product::E8KvWidth::W2),
              "e8 W2: code extent * page tokens + g64 fp16 scales must be 4608 B/head-page");

// The literals, pinned so a *plausible* wrong width (W3 read at W2's 64, or at W4's 128)
// reds here rather than at a call site.
static_assert(d256_kv_cache_profile(DType::E8Kv).code_leading_extent == 128,
              "e8 W4 code extent is 128 B/row");
static_assert(d256_kv_cache_profile(DType::E8K3Kv).code_leading_extent == 96,
              "e8 W3 code extent is 96 B/row");
static_assert(d256_kv_cache_profile(DType::E8K2Kv).code_leading_extent == 64,
              "e8 W2 code extent is 64 B/row");

// ---------------------------------------------------------------------------
// THE V-PLANE IDENTITY. The V extent is the W4 row for all three e8 tiers, and this is
// what reds the file if a later edit makes the V extent follow the K width -- the single
// most plausible wrong reading of the field above, and the one that would make every
// correctly laid out W3/W2 pool fail an admission check.
// ---------------------------------------------------------------------------
static_assert(kD256E8VCodeExtent == 128,
              "the e8 family's V row extent is the shipped W4 plane's 128 B/row");
static_assert(kD256E8VCodeExtent * product::kE8KvPageTokens +
                      kD256ScaleExtentPerGroup * product::kE8KvPageTokens *
                          product::kE8KvScaleBytesPerGroup ==
                  product::e8_kv_v_plane_bytes(),
              "e8 V: 128 * 64 + 512 must be the geometry of record's 8704 B/head-page V");
static_assert(d256_kv_cache_profile(DType::E8Kv).v_code_leading_extent == kD256E8VCodeExtent &&
                  d256_kv_cache_profile(DType::E8K3Kv).v_code_leading_extent == kD256E8VCodeExtent &&
                  d256_kv_cache_profile(DType::E8K2Kv).v_code_leading_extent == kD256E8VCodeExtent,
              "all three e8 tiers must publish the SAME (W4) V extent: this family narrows "
              "K only, so a V extent that tracked the K width describes a plane that does "
              "not exist");
// The unpacked tiers keep V at the K extent, so this change moves nothing for them.
static_assert(d256_kv_cache_profile(DType::BF16).v_code_leading_extent == kD256KVCacheHeadDim &&
                  d256_kv_cache_profile(DType::I8).v_code_leading_extent == kD256KVCacheHeadDim &&
                  d256_kv_cache_profile(DType::FP8_E4M3FN).v_code_leading_extent ==
                      kD256KVCacheHeadDim,
              "the unpacked tiers store one code per element on BOTH planes: 256");
static_assert(!d256_kv_cache_is_e8_family(DType::BF16) &&
                  !d256_kv_cache_is_e8_family(DType::I8) &&
                  !d256_kv_cache_is_e8_family(DType::FP8_E4M3FN) &&
                  d256_kv_cache_is_e8_family(DType::E8Kv) &&
                  d256_kv_cache_is_e8_family(DType::E8K3Kv) &&
                  d256_kv_cache_is_e8_family(DType::E8K2Kv),
              "the family predicate must name exactly the three e8 tiers and no unpacked one");

} // namespace ninfer::ops
