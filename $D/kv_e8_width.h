// ===========================================================================================
// kvfix (F893) -- THIS FILE IS A STALE MIRROR, AND IT IS TRACKED.  READ THE LIVE ONE.
// ===========================================================================================
// `$D/` was created by an unexpanded shell variable (the literal name `$D`) and it is IN THE
// GIT INDEX -- `git ls-files -- '$D'` lists these three files.  They are copies of the live KV
// surfaces taken 2026-09-20 and NEVER UPDATED, so they still carry the PRE-repair spellings:
//   * `$D/kv_bit_budget.h` keeps the `--kv-bit-budget 0-7:8,8-63:4.5` example, which the engine
//     refuses by name ("ranges must tile layers 0..15 in order"); the LIVE file at
//     src/product/kv_bit_budget.h:1163 carries the corrected, layer-count-agnostic
//     `0-7:8,8-15:4.5`.
//   * `$D/kv_formats.h` keeps the `--kv-tier-formats hot=bf16,tail=fp16,cold=iso4e` example,
//     which the engine refuses by name ("this engine has no tail tier"); the LIVE file at
//     src/kvcfg/kv_formats.h:5 carries `hot=bf16,cold=iso4e` and names the gap in place.
// MEASURED CONSEQUENCE: a whole-tree grep for either spelling still finds it HERE, so a reader
// concludes the tree ships an example the engine refuses.  Two lines of this record did exactly
// that (dl/namedmech's F-855 triage B1/E3, which counted "NINE places"; dl/kvadv F-892 carried
// it forward).  The live count is ZERO.
// THE FIX, AND WHY IT IS ONLY THIS BANNER: the tree's own convention is set out at
// src/kvcfg/kv_formats.h:38 -- "就地删名会把这个缺口藏起来, 故此处只标注事实" (deleting the name in
// place would HIDE the gap, so only the fact is annotated here).  Nothing is deleted, renamed or
// rewritten; the mirror is named at the top of its own files so the next grep's reader sees it.
// The live file for this mirror is: src/product/kv_e8_width.h
// ===========================================================================================

#pragma once
// kv_e8_width.h -- the ONE decision point for the e8 KV family's K-plane CODE WIDTH.
//
// WHY THIS FILE EXISTS
// --------------------
// src/kvcfg/kv_formats.h's `E8` comment promised "3/2bit 变体后续" (3/2-bit variants,
// later) and the GUI twin tools/gui/kv_tiers.py went further and *priced* e8 at 2 bits
// ({"e8": 2}) while bits_of(KvFormat::E8) returned 4. The width of this family was
// therefore written down in three places and agreed in none of them. This header makes
// the width the single input and derives every other number from it, so the plane
// geometry, the ladder cost and the vocabulary's bit column cannot drift apart again.
//
// THE AXIS (read this before adding a row)
// ----------------------------------------
// Two different things in this ecosystem are called "e8 <N> bit":
//
//   AXIS 1 (THIS FILE) -- the e8 K-plane code width. One signed integer code per
//     element, packed at N bits per element, over the same per-64-channel FP16 scale
//     plane the shipped 4-bit e8 already uses.
//
//   AXIS 2 -- the ecosystem's `rk<N>v<M>-e8` naming, which is the ROTATED-K bit width
//     against the V bit width, with e8 in the mix. `rk4v4-e8` = rotated K at 4 bits,
//     V at 4 bits; `rk2v4-e8` = K at 2 bits, V at 4 bits.
//
// The two axes are THE SAME AXIS ON THE K PLANE and they differ only in whether the WIDTH
// is spelled for a plane or for a K/V pair. That is why THIS FILE NARROWS THE K PLANE AND
// LEAVES V AT i4: it is what the family does (V carries no lattice projection, so there is
// no e8 reason to narrow it) and it is what the ecosystem did. So:
//
//   token   K plane                          V plane        layer K+V
//   e8      8704 B  (D/2 codes, 4 b/el)      8704 B (i4)    17408 B = 4.25 b/el  [shipped]
//   e8k3    6656 B  (D*3/8 codes, 3 b/el)    8704 B (i4)    15360 B = 3.75 b/el  [new]
//   e8k2    4608 B  (D/4 codes, 2 b/el)      8704 B (i4)    13312 B = 3.25 b/el  [new]
//
// THE TWO AXES MEET AT EXACTLY TWO POINTS, and both are arithmetic, not claims:
//   * ecosystem rk4v4-e8 K plane = D/2 code bytes + 16 bits per 64-channel group
//                               = 128*64 + 512 = 8704 B/head-page = our e8 K plane.
//   * ecosystem rk2v4-e8 K plane = D/4 code bytes + 16 bits per 64-channel group
//                               =  64*64 + 512 = 4608 B/head-page = our e8k2 K plane,
//     and with its V left at 4 bits its LAYER cost is 4608+8704 = 13312 B = 3.25 b/el,
//     which is exactly our e8k2 ladder row.
// They are the same GEOMETRY. They are NOT the same CODEC: rk2v4-e8's K is a
// 2-byte E8 ROOT-CYLINDER code per 8 dims (a 240-root codebook index + a radius byte),
// i.e. a genuine E8 codebook, while this file's e8k2 is per-element codes over a shared
// scale (the same plane contract our shipped 4-bit e8 uses). See
// docs/plans/e8-kv-port-handoff-*.md in the BenWu/ninfer e8-kv fork for the codebook.
//
// THE LADDER ROW IS THE LAYER COST, K+V AVERAGED -- kv_bit_budget.h states it verbatim
// ("per KV element, K+V averaged") -- so a K-plane-only narrowing moves the row by HALF
// of one bit per element. bits_x100 = 50*w + 225, which reproduces the shipped 425 at
// w=4: that is the check that this derivation is the shipped one and not a lookalike.
//
// THE LATTICE FLOOR (why "smaller" is not free on this axis) -- AND THAT IT IS CLEARED
// ---------------------------------------------------------------------------------
// src/ops/kernel/e8_lattice.cuh:125-137 measured the constraint that decides whether a
// 3-bit or 2-bit e8 lattice code can exist in THIS plane shape: the Conway-Sloane
// projection returns D8 points (integer coords) OR D8+1/2 points (all-eight-coordinates
// half-integer), and 47.6% of real K/V blocks take the half-integer coset; the plane
// this file sizes holds "one signed integer per coordinate", so the `rintf` that must
// follow moves those coordinates 0.5 step, measured at +3.65 dB of error (2.32x MSE)
// versus plain rounding AT THE SAME BIT RATE. The note's own remedy is to hold
// 2*coordinate -- i.e. ONE MORE BIT PER ELEMENT (5 bits to represent +-15).
//
// That floor binds the INTEGER-CODE consumer, and for it the conclusion below stands: a
// 3-bit or 2-bit e8 with one integer per coordinate cannot carry the projection, and at
// w=3 the floor is 4 and at w=2 the floor is 3, so BOTH new widths sit strictly BELOW it.
//
// IT IS NOT THE ONLY CONSUMER, AND THE SECOND ONE IS NOW BUILT. e8_lattice.cuh:136-137
// names a second escape -- "where the consumer reconstructs the lattice point rather
// than an integer code" -- and that consumer needs no extra bit at all: its codeword is
// an 8-dimensional POINT of E8 +- 1/4, 16 bits at 2 b/el and 24 at 3 b/el, which is
// exactly 2 and 3 bytes per 8 elements, i.e. exactly this file's byte counts for W2/W3.
// It lives in src/ops/kv/e8_lattice_plane_codec.cuh and is the codec of record for both
// narrow widths (e8_kv_plane_codec_of_record). So the W3 and W2 rows below are the REAL
// E8 lattice and not "a plain narrower code in the same plane shape" -- the arithmetic
// in this header (the byte counts, the ladder, the static_asserts) is unchanged and did
// not have to move for that to be true, because the codeword sizes were already right.
//
// The floor arithmetic is kept above because it is still the reason the SCALAR codec in
// product/kv_e8_width_codec.h is the codec of record for W4 and the priced control arm
// for the other two: it is what a one-integer-per-coordinate plane can carry, and it is
// what every lattice number is measured against at the same bits and side information.

#include <cstdint>

namespace ninfer::product {

// The e8 family's K-plane geometry constants. kE8KvHeadDim/kE8KvPageTokens are the
// engine's own head-page shape (src/product/kv_tier_formats.h COLD CODEC RULE block
// prices every plane at head_dim 256 x 64 tokens) and kE8KvScaleGroup is the group the
// shipped e8 row's "nibble + FP16/g64" is built from.
inline constexpr std::int32_t kE8KvHeadDim    = 256;
inline constexpr std::int32_t kE8KvPageTokens = 64;
inline constexpr std::int32_t kE8KvScaleGroup = 64;
// One FP16 scale per 64-channel group -> 16 bits per 64 elements = 0.25 bits/element,
// the term that turns every width N into the ladder's N.25.
inline constexpr std::int32_t kE8KvScaleBytesPerGroup = 2;

// The three widths this family ships. The numeric value IS the K-plane bits per element.
enum class E8KvWidth : std::uint8_t {
    W4 = 4,   // shipped tier ("e8"); equals ecosystem rk4v4-e8's K plane
    W3 = 3,   // new; below the lattice floor (needs 4) -- no ecosystem counterpart
    W2 = 2,   // new; below the lattice floor (needs 3); equals rk2v4-e8's K GEOMETRY only
};

// Widths whose codes divide a byte evenly. W3 does NOT: it packs 8 elements into 24
// bits (3 bytes), which is the tree's own documented-but-unimplemented ISO3 reference
// layout ("符号+2bit 幅值、8 元素/3 字节", kv_formats.h:27-29) -- byte-aligned at the
// 8-element group, never at the element.
[[nodiscard]] constexpr std::int32_t e8_kv_codes_per_byte(E8KvWidth w) noexcept {
    return w == E8KvWidth::W4 ? 2 : (w == E8KvWidth::W2 ? 4 : 0);
}

// Bytes of code per 8-element group: W4 4, W3 3, W2 2. W2's 2 is the ecosystem's
// "2-byte E8 root cylinder code per 8 rotated dims" byte count as well.
[[nodiscard]] constexpr std::int32_t e8_kv_code_bytes_per_8(E8KvWidth w) noexcept {
    return (8 * static_cast<std::int32_t>(w) + 7) / 8;
}

// Code bytes in one row of head_dim elements.
[[nodiscard]] constexpr std::int32_t e8_kv_row_code_bytes(E8KvWidth w) noexcept {
    return kE8KvHeadDim * static_cast<std::int32_t>(w) / 8;   // 128 / 96 / 64
}

// The K plane, per (page, kv_head, K plane): code bytes + the g64 scale plane.
[[nodiscard]] constexpr std::int32_t e8_kv_k_plane_bytes(E8KvWidth w) noexcept {
    return e8_kv_row_code_bytes(w) * kE8KvPageTokens +
           (kE8KvHeadDim / kE8KvScaleGroup) * kE8KvPageTokens * kE8KvScaleBytesPerGroup;
}

// The V plane is NOT a function of the width: this family narrows K only (see the header
// block), and the shipped V is i4 at 4 bits + the same g64 FP16 scale plane, i.e. the
// W4 geometry. Named rather than repeated so a reader can see the V plane is untouched.
[[nodiscard]] constexpr std::int32_t e8_kv_v_plane_bytes() noexcept {
    return e8_kv_k_plane_bytes(E8KvWidth::W4);   // 8704
}
static_assert(e8_kv_v_plane_bytes() == 8704, "V stays i4 at 4.25 b/el (not a width input)");

// The LAYER cost: K plane + V plane, the quantity kv_bit_budget.h's ladder prices
// ("per KV element, K+V averaged"). Two planes of kE8KvElementsPerHeadPage elements.
[[nodiscard]] constexpr std::int32_t e8_kv_layer_bytes(E8KvWidth w) noexcept {
    return e8_kv_k_plane_bytes(w) + e8_kv_v_plane_bytes();
}

// The ladder cost in 0.01-bit units (kKvBitBudgetScale), derived from the layer bytes
// rather than written down: layer_bytes * 8 * 100 / (2 * 16384) == 50*w + 225.
[[nodiscard]] constexpr std::int32_t e8_kv_bits_x100(E8KvWidth w) noexcept {
    return (e8_kv_layer_bytes(w) * 8 * 100) / (2 * kE8KvHeadDim * kE8KvPageTokens);
}

// Elements one head-page of one plane covers: 256 wide x 64 tokens.
inline constexpr std::int32_t kE8KvElementsPerHeadPage =
    kE8KvHeadDim * kE8KvPageTokens;   // 16384, the kKvBitBudgetElementsPerHeadPage shape

// ---------------------------------------------------------------------------
// The geometry, pinned. W4 is the SHIPPED tier and must reproduce the four
// static_asserts src/product/kv_tier_formats.h already carries (8704 B K plane,
// 4.25 b/el layer): that is the check that this header's derivation is the shipped one
// and not a lookalike. W3/W2 are new and are pinned here for the first time.
// ---------------------------------------------------------------------------
static_assert(e8_kv_k_plane_bytes(E8KvWidth::W4) == 8704,
              "e8 W4 K plane: 128*64 + 512 = 8704 B/head-page (the shipped plane)");
static_assert(e8_kv_k_plane_bytes(E8KvWidth::W3) == 6656,
              "e8 W3 K plane: 96*64 + 512 = 6656 B/head-page");
static_assert(e8_kv_k_plane_bytes(E8KvWidth::W2) == 4608,
              "e8 W2 K plane: 64*64 + 512 = 4608 B/head-page (== rk2v4-e8's K geometry)");
static_assert(e8_kv_layer_bytes(E8KvWidth::W4) == 17408, "e8 W4 layer: 8704+8704");
static_assert(e8_kv_layer_bytes(E8KvWidth::W3) == 15360, "e8 W3 layer: 6656+8704");
static_assert(e8_kv_layer_bytes(E8KvWidth::W2) == 13312,
              "e8 W2 layer: 4608+8704 = 13312 B = 3.25 b/el (== rk2v4-e8's layer cost)");
static_assert(e8_kv_bits_x100(E8KvWidth::W4) == 425, "e8 W4: 4.25 b/el (the shipped row)");
static_assert(e8_kv_bits_x100(E8KvWidth::W3) == 375, "e8 W3: 3.75 b/el");
static_assert(e8_kv_bits_x100(E8KvWidth::W2) == 325, "e8 W2: 3.25 b/el");
static_assert(e8_kv_row_code_bytes(E8KvWidth::W4) == 128 && e8_kv_row_code_bytes(E8KvWidth::W3) == 96 &&
                  e8_kv_row_code_bytes(E8KvWidth::W2) == 64,
              "e8 K code bytes per 256-wide row: 128 / 96 / 64");
static_assert(e8_kv_code_bytes_per_8(E8KvWidth::W4) == 4 && e8_kv_code_bytes_per_8(E8KvWidth::W3) == 3 &&
                  e8_kv_code_bytes_per_8(E8KvWidth::W2) == 2,
              "e8 K code bytes per 8-element group: 4 / 3 / 2");
// The three widths must stay distinct and ordered, because the vocabulary's
// bits_of() sort (hot >= tail) and the ladder's cost column both read them.
static_assert(static_cast<std::int32_t>(E8KvWidth::W4) > static_cast<std::int32_t>(E8KvWidth::W3) &&
                  static_cast<std::int32_t>(E8KvWidth::W3) > static_cast<std::int32_t>(E8KvWidth::W2),
              "e8 widths must be strictly ordered W4 > W3 > W2");

// The K-plane saving against the shipped plane, in bytes/head-page. The LAYER saving is
// the same number (V is untouched), and the ladder row moves by half a bit per element.
[[nodiscard]] constexpr std::int32_t e8_kv_k_bytes_saved_vs_w4(E8KvWidth w) noexcept {
    return e8_kv_k_plane_bytes(E8KvWidth::W4) - e8_kv_k_plane_bytes(w);
}
static_assert(e8_kv_k_bytes_saved_vs_w4(E8KvWidth::W3) == 2048, "W3 gives back 2048 B/head-page");
static_assert(e8_kv_k_bytes_saved_vs_w4(E8KvWidth::W2) == 4096, "W2 gives back 4096 B/head-page");

} // namespace ninfer::product
