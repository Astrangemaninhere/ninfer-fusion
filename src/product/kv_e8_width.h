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
#include <string>
#include <string_view>

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

// THE SHIPPED V PLANE, NAMED FOR WHAT IT IS (i4) RATHER THAN FOR WHERE IT SITS.
//
// THE PRE-IMAGE OF THIS LINE WAS `e8_kv_v_plane_bytes()` -- "THE V plane" -- and that definite
// article is the defect, not the arithmetic (line dl/e8vaxis, marker F1166). A V plane is not a
// constant of the family; it is a FORMAT somebody chose, and this function names exactly one of
// them: the shipped i4 (packed two codes per byte + one FP16 per 64-channel group, 8704
// B/head-page). Because `e8_kv_layer_bytes(w)` below folds it in as a CONSTANT, the family had
// one degree of freedom and it was not on V -- which is what "a KV-bound design" means here.
//
// THE VALUE IS UNCHANGED SO THAT NO CALLER MOVES. What changed is that it is now one format
// among the three the plane axis (E8KvPlaneFormat, at the foot of this file) can put on EITHER
// plane, and the V it describes is format B4 on that axis.
[[nodiscard]] constexpr std::int32_t e8_kv_i4_v_plane_bytes() noexcept {
    return e8_kv_k_plane_bytes(E8KvWidth::W4);   // 8704
}
static_assert(e8_kv_i4_v_plane_bytes() == 8704, "the shipped V is i4 at 4.25 b/el");

// THE PRE-IMAGE SPELLING, KEPT AS AN ALIAS and not as a second definition, so the two cannot
// drift apart. Every existing caller and every existing pin keeps its exact meaning; what a
// reader must NOT do is read this as "the V plane of the family". It is the SHIPPED i4 V, and
// the plane axis at the foot of this file is where a V plane becomes a CHOICE.
[[nodiscard]] constexpr std::int32_t e8_kv_v_plane_bytes() noexcept {
    return e8_kv_i4_v_plane_bytes();
}
static_assert(e8_kv_v_plane_bytes() == 8704,
              "e8_kv_v_plane_bytes() is the SHIPPED i4 V and nothing more; the V of the plane "
              "axis is e8_kv_v_plane_bytes(E8KvPlaneFormat)");

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

// ===========================================================================
// THE PLANE AXIS -- E8 AS A FORMAT, SELECTABLE INDEPENDENTLY ON K AND ON V
// ===========================================================================
// WHY THIS BLOCK EXISTS (line dl/e8vaxis, marker F1166, on the owner's order: "e8 应该是类似
// int8 那种独立的东西而不是个 kv 绑死的设计 ... 最低 kv 都可以是 e8 2bit").
//
// EVERYTHING ABOVE THIS LINE IS KV-BOUND, and the binding is one fact: the family's only input
// is `E8KvWidth w`, that input names the **K** plane, and V is folded in as the constant
// `e8_kv_v_plane_bytes()`. So "e8k2" never meant "e8 at 2 bits"; it meant "K at 2 bits AND V at
// i4" -- two planes in one token. The tree says the same thing in three other places and each of
// them is honest about it: src/core/dtype.h calls the two narrow tiers "the SAME K+V plane
// pair"; the writer (ops/kv_cache/append/e8_lattice_narrow_kernel.cuh) heads a block "THE V
// PLANE IS NOT NARROWED, AND THAT IS THE FAMILY'S OWN RULE"; and ops/kv_cache/d256_profile.h
// static_asserts that all three e8 tiers publish the SAME V extent.
//
// WHAT REPLACES IT. A plane is chosen by a FORMAT, and the format does not know which plane it
// is being chosen for -- that is the whole edit. E8 quantises eight coordinates along the
// CHANNEL axis; it does not know and does not care which plane it is quantising, so the SAME
// lattice, the SAME 256-entry stage-2 table and the SAME `e8_lattice_hadamard64()` serve both
// planes. That is why this costs no mathematics: the K codec IS the V codec, not a lookalike of
// it, and section 10.2 of src/ops/kv/e8_width_codec_test.cpp pins that by BIT IDENTITY.
enum class E8KvPlane : std::uint8_t {
    K = 0,
    V = 1,
};

// THE FORMATS. `bits` IS the plane's bits per element, so the enum's own value is the axis.
enum class E8KvPlaneFormat : std::uint8_t {
    // 4.00 b/el on the plane: 8 elements per 4 code bytes, plus one FP16 per 64-channel group.
    // ONE byte geometry, TWO codecs -- on K it is the SCALAR codec (the shipped W4 row), on V it
    // is the shipped packed i4. That is the tree's own "SAME SIZE but a different MEANING"
    // (product/kv_e8_width_codec.h), and it is why the CODEC is a function of (plane, format)
    // while the BYTE COUNT is a function of the format alone.
    B4 = 4,
    // 3.00 b/el on the plane: the REAL E8 lattice, a 24-bit codeword per 8 dimensions.
    B3 = 3,
    // 2.00 b/el on the plane: the REAL E8 lattice, a 16-bit codeword per 8 dimensions.
    // THE FLOOR THE OWNER NAMED: B2 on BOTH planes is 2.25 b/el per plane.
    B2 = 2,
};

[[nodiscard]] constexpr std::int32_t e8_kv_plane_format_bits(E8KvPlaneFormat f) noexcept {
    return static_cast<std::int32_t>(f);
}

// Code bytes per 8-element group: 4 at B4 (which is also W4's scalar layout), 3 at B3, 2 at B2.
// Derived from the bits, so a fourth format cannot be added with a hand-written byte count.
[[nodiscard]] constexpr std::int32_t e8_kv_plane_format_code_bytes_per_8(
    E8KvPlaneFormat f) noexcept {
    return (8 * e8_kv_plane_format_bits(f) + 7) / 8;
}

// Code bytes in one row of head_dim elements: 128 / 96 / 64.
[[nodiscard]] constexpr std::int32_t e8_kv_plane_format_row_code_bytes(
    E8KvPlaneFormat f) noexcept {
    return kE8KvHeadDim * e8_kv_plane_format_bits(f) / 8;
}

// ONE PLANE at that format: its code bytes over the page's rows, plus the per-64-channel FP16
// scale plane every format in this family carries (16 bits per 64 elements = 0.25 b/el).
[[nodiscard]] constexpr std::int32_t e8_kv_plane_format_bytes(E8KvPlaneFormat f) noexcept {
    return e8_kv_plane_format_row_code_bytes(f) * kE8KvPageTokens +
           (kE8KvHeadDim / kE8KvScaleGroup) * kE8KvPageTokens * kE8KvScaleBytesPerGroup;
}

// THE PAIR. This is the object the KV-bound design did not have: TWO independent inputs. Its
// exact pre-image is `e8_kv_layer_bytes(E8KvWidth w)`, which is this same sum with the second
// operand replaced by the constant 8704.
[[nodiscard]] constexpr std::int32_t e8_kv_pair_bytes(E8KvPlaneFormat k,
                                                      E8KvPlaneFormat v) noexcept {
    return e8_kv_plane_format_bytes(k) + e8_kv_plane_format_bytes(v);
}

// The pair's ladder cost in 0.01-bit units (kKvBitBudgetScale): K+V averaged over the two planes,
// which is the quantity kv_bit_budget.h's ladder prices ("per KV element, K+V averaged").
// Derived from the bytes and never written down -- the same derivation e8_kv_bits_x100() uses,
// which is what makes (B4,B4) reproduce the shipped 425 rather than merely equal it.
[[nodiscard]] constexpr std::int32_t e8_kv_pair_bits_x100(E8KvPlaneFormat k,
                                                          E8KvPlaneFormat v) noexcept {
    return (e8_kv_pair_bytes(k, v) * 8 * 100) / (2 * kE8KvHeadDim * kE8KvPageTokens);
}

// THE V PLANE FOR A FORMAT -- the question the pre-image's e8_kv_v_plane_bytes() could not be
// asked, because it answered for one plane and took no argument.
[[nodiscard]] constexpr std::int32_t e8_kv_v_plane_bytes(E8KvPlaneFormat v) noexcept {
    return e8_kv_plane_format_bytes(v);
}

// ===========================================================================
// THE PINS. The shipped row must be REPRODUCED through the new axis -- that is the check that
// this axis is the shipped geometry re-spelled and not a lookalike -- and the owner's floor must
// be a NUMBER rather than a promise.
// ===========================================================================
// (1) the three planes, as literals: 128*64+512 / 96*64+512 / 64*64+512.
static_assert(e8_kv_plane_format_bytes(E8KvPlaneFormat::B4) == 8704,
              "e8 B4 plane: 128*64 + 512 = 8704 B/head-page (the shipped V's plane)");
static_assert(e8_kv_plane_format_bytes(E8KvPlaneFormat::B3) == 6656,
              "e8 B3 plane: 96*64 + 512 = 6656 B/head-page");
static_assert(e8_kv_plane_format_bytes(E8KvPlaneFormat::B2) == 4608,
              "e8 B2 plane: 64*64 + 512 = 4608 B/head-page");
// (2) THE AXIS REPRODUCES THE SHIPPED ROW. (B4,B4) is the number the shipped e8 tier has, and
// e8_kv_layer_bytes(W4) is that same number -- so the axis has not invented a geometry, it has
// re-spelled the old one with V promoted from a constant to an input.
static_assert(e8_kv_pair_bytes(E8KvPlaneFormat::B4, E8KvPlaneFormat::B4) == 17408,
              "the shipped pair: 8704 + 8704 = 17408 B/head-page");
static_assert(e8_kv_pair_bytes(E8KvPlaneFormat::B4, E8KvPlaneFormat::B4) ==
                  e8_kv_layer_bytes(E8KvWidth::W4),
              "the plane axis must REPRODUCE the shipped W4 layer, not replace it");
static_assert(e8_kv_pair_bits_x100(E8KvPlaneFormat::B4, E8KvPlaneFormat::B4) == 425,
              "the shipped pair is 4.25 b/el -- the same 425 the ladder prices");
// (3) THE SHIPPED V IS THE B4 PLANE. This is the ONE place the pre-image constant and the new
// axis are required to agree, so no later edit can leave them describing two planes.
static_assert(e8_kv_v_plane_bytes() == e8_kv_plane_format_bytes(E8KvPlaneFormat::B4),
              "the shipped i4 V and the B4 plane must be the same plane, byte for byte");
static_assert(e8_kv_v_plane_bytes(E8KvPlaneFormat::B2) == 4608,
              "e8 on V at 2 bits is a 4608 B/head-page plane");
// (4) THE OWNER'S FLOOR, AS ARITHMETIC. "最低 kv 都可以是 e8 2bit": B2 on BOTH planes.
static_assert(e8_kv_pair_bytes(E8KvPlaneFormat::B2, E8KvPlaneFormat::B2) == 9216,
              "e8 B2 on BOTH planes: 4608 + 4608 = 9216 B/head-page");
static_assert(e8_kv_pair_bits_x100(E8KvPlaneFormat::B2, E8KvPlaneFormat::B2) == 225,
              "e8 B2/B2 is 2.25 b/el per plane -- THE FLOOR THE OWNER NAMED");
static_assert(e8_kv_pair_bits_x100(E8KvPlaneFormat::B3, E8KvPlaneFormat::B3) == 325,
              "e8 B3/B3 is 3.25 b/el per plane");
// (5) THE AXIS IS NOT A RENAME OF THE WIDTH. If B2-on-both ever stops being STRICTLY cheaper than
// the cheapest K-only row the ladder has today, the axis has stopped being an axis.
static_assert(e8_kv_pair_bits_x100(E8KvPlaneFormat::B2, E8KvPlaneFormat::B2) <
                  e8_kv_bits_x100(E8KvWidth::W2),
              "B2 on BOTH planes (9216 B) must cost LESS than the K-only W2 row (4608+8704=13312) "
              "-- that is what makes the pair an ADDED degree of freedom");


// ===========================================================================
// [dl/e8names, marker F1194] THE PLANE'S NAME, AND A PAIR SPELLED FROM BOTH PLANES
// ===========================================================================
// THE DEFECT THIS CLOSES. Everything above prices a pair; nothing above could NAME one without
// the binding era leaking into the name. The family's user-visible tokens are `rk4v4`, `rk3v4`
// and `rk2v4`: a PAIR token whose `v4` is a LITERAL, so the token has no position for V, and the
// owner's three-rung ladder ("4bit / 3bit / 2bit -- all three rungs must work") is spelled three
// different ways. MEASURED CONSEQUENCE, in one file: the SAME dtype is named `rk2v4` by the
// dtype-name switch and `a 16-bit plane (no cold codec)` by the codec switch beside it
// (src/targets/qwen3_6/impl/runtime/program_impl.h). A name that cannot say which V it means is
// what this block exists to stop being the ONLY spelling.
//
// THE PLANE'S NAME IS DERIVED FROM THE FORMAT, AND IT NAMES ONE PLANE. `e8-b<N>` is ONE PLANE at
// N bits per element -- it does not name K or V, because the format does not know which plane it
// is being chosen for. That is the axis F1166 landed, and it is the same reason the CODEC is a
// function of (plane, format) while the BYTE COUNT is a function of the format alone. So there is
// exactly one name per E8KvPlaneFormat, and a fourth format adds a fourth name and nothing else.
//
// A PAIR IS SPELLED BY SPELLING BOTH OF ITS PLANES, K FIRST, V SECOND. `e8-b3/e8-b4` is the
// sentence `rk3v4` could not write; `e8-b2/e8-b2` -- THE OWNER'S FLOOR -- was unspellable in the
// old vocabulary at all. The pair's name is COMPOSED from the two plane names rather than written
// down again, so a pair and its spelling cannot drift.
//
// THE BINDING-ERA TOKENS ARE KEPT, BYTE FOR BYTE, AS ALIASES -- AND THEY ARE NOT THE PRIMARY
// SPELLING ANY MORE. They are the DEPLOYED vocabulary: `--kv-layer-storage` specs,
// `kvcfg::KvFormat`'s names, `kKvBitBudgetTiers`' `spec_name`, `KvCacheStorage`'s `rk3v4-g64`
// tokens, and the tests that assert every one of them. Nothing in this header removes or
// re-spells one. `e8_kv_pair_name_legacy()` reproduces them FROM the pair -- never from a second
// hand-written table, so the pair and its legacy name cannot drift.

// THE PLANE'S NAME. One name per format, and the name carries the format's own bits, which is the
// same number `e8_kv_plane_format_bits()` returns and the same value the enumerator holds.
[[nodiscard]] constexpr const char* e8_kv_plane_format_name(E8KvPlaneFormat f) noexcept {
    switch (f) {
    case E8KvPlaneFormat::B4: return "e8-b4";
    case E8KvPlaneFormat::B3: return "e8-b3";
    case E8KvPlaneFormat::B2: return "e8-b2";
    }
    return "e8-b?";
}

// THE PAIR, COMPOSED FROM THE TWO PLANE NAMES AND FROM NOTHING ELSE. K first, V second, always:
// `e8_kv_pair_name(B3, B4)` reads "the K plane is B3 and the V plane is B4".
[[nodiscard]] inline std::string e8_kv_pair_name(E8KvPlaneFormat k, E8KvPlaneFormat v) {
    return std::string(e8_kv_plane_format_name(k)) + "/" + std::string(e8_kv_plane_format_name(v));
}

// LEGACY. THE BINDING-ERA PAIR TOKENS, AND ONLY THE THREE THE TREE SHIPS. Reproduced FROM the
// pair -- not from a second table -- so a pair and its legacy name cannot drift apart. A pair the
// old vocabulary cannot spell returns `none`, which is a REFUSAL and not an empty answer.
[[nodiscard]] constexpr const char* e8_kv_pair_name_legacy(E8KvPlaneFormat k,
                                                           E8KvPlaneFormat v) noexcept {
    if (k == E8KvPlaneFormat::B4 && v == E8KvPlaneFormat::B4) { return "rk4v4"; }
    if (k == E8KvPlaneFormat::B3 && v == E8KvPlaneFormat::B4) { return "rk3v4"; }
    if (k == E8KvPlaneFormat::B2 && v == E8KvPlaneFormat::B4) { return "rk2v4"; }
    return "none";
}

// THE PINS. (a) one name per format, and the three are pairwise distinct; (b) the legacy tokens
// are the DEPLOYED ones byte for byte -- a re-spelling fails HERE instead of quietly splitting the
// shipped vocabulary from its alias; (c) a pair the old vocabulary cannot spell REFUSES by name;
// (d) THE TWO VOCABULARIES STAY DISJOINT IN BOTH DIRECTIONS, so no later edit can make a plane
// name claim to be a pair token (or the reverse) and have every reader of one silently start
// reading the other.
static_assert(e8_kv_plane_format_name(E8KvPlaneFormat::B4) == std::string_view{"e8-b4"} &&
                  e8_kv_plane_format_name(E8KvPlaneFormat::B3) == std::string_view{"e8-b3"} &&
                  e8_kv_plane_format_name(E8KvPlaneFormat::B2) == std::string_view{"e8-b2"},
              "one plane name per format: e8-b4 / e8-b3 / e8-b2");
static_assert(std::string_view{e8_kv_plane_format_name(E8KvPlaneFormat::B4)} !=
                      std::string_view{e8_kv_plane_format_name(E8KvPlaneFormat::B3)} &&
                  std::string_view{e8_kv_plane_format_name(E8KvPlaneFormat::B3)} !=
                      std::string_view{e8_kv_plane_format_name(E8KvPlaneFormat::B2)} &&
                  std::string_view{e8_kv_plane_format_name(E8KvPlaneFormat::B2)} !=
                      std::string_view{e8_kv_plane_format_name(E8KvPlaneFormat::B4)},
              "the plane names are PAIRWISE DISTINCT: a name two formats shared would make every "
              "report ambiguous, which is the defect this block closes");
static_assert(e8_kv_pair_name_legacy(E8KvPlaneFormat::B4, E8KvPlaneFormat::B4) ==
                      std::string_view{"rk4v4"} &&
                  e8_kv_pair_name_legacy(E8KvPlaneFormat::B3, E8KvPlaneFormat::B4) ==
                      std::string_view{"rk3v4"} &&
                  e8_kv_pair_name_legacy(E8KvPlaneFormat::B2, E8KvPlaneFormat::B4) ==
                      std::string_view{"rk2v4"},
              "the three shipped K-only pairs keep the tree's DEPLOYED tokens, byte for byte");
static_assert(e8_kv_pair_name_legacy(E8KvPlaneFormat::B2, E8KvPlaneFormat::B2) ==
                      std::string_view{"none"} &&
                  e8_kv_pair_name_legacy(E8KvPlaneFormat::B3, E8KvPlaneFormat::B3) ==
                      std::string_view{"none"} &&
                  e8_kv_pair_name_legacy(E8KvPlaneFormat::B3, E8KvPlaneFormat::B2) ==
                      std::string_view{"none"},
              "THE OWNER'S FLOOR HAS NO BINDING-ERA TOKEN: e8-b2 on BOTH planes is a pair the old "
              "vocabulary cannot spell, so the legacy name REFUSES (`none`) instead of inventing "
              "`rk2v2`. That refusal is the measurement that the old token really was K-only");
static_assert(e8_kv_plane_format_name(E8KvPlaneFormat::B4) !=
                      e8_kv_pair_name_legacy(E8KvPlaneFormat::B4, E8KvPlaneFormat::B4) &&
                  e8_kv_plane_format_name(E8KvPlaneFormat::B3) !=
                      e8_kv_pair_name_legacy(E8KvPlaneFormat::B3, E8KvPlaneFormat::B4) &&
                  e8_kv_plane_format_name(E8KvPlaneFormat::B2) !=
                      e8_kv_pair_name_legacy(E8KvPlaneFormat::B2, E8KvPlaneFormat::B4),
              "the plane vocabulary and the binding-era vocabulary must stay DISJOINT in both "
              "directions");
// =============================================================================================
// [F1231 2026-09-29] F-6: THE OWNER'S FLOOR IS NAMEABLE AND NOT ASKABLE, AND HERE IS THE DECISION.
// =============================================================================================
//
// THE ASYMMETRY, IN THIS HEADER'S OWN NUMBERS (pinned by the static_asserts at the foot of this
// block, so a later round cannot move one side without the other failing to compile):
//
//   THE FLOOR THE DESCENT PRICES: `e8-b2` on BOTH planes. `CellMode::E8_2bit` is priced at
//   `plane_bytes == 4,608` (kv_cell_modes.h, static_asserted), and the descent charges
//   `2 * kv_heads * sum_l plane_bytes`, so the floor's charge is B2/B2 -- pair bytes 9,216.
//   THE NARROWEST TOKEN THE CLI CAN ASK FOR: `rk2v4` IS `e8_kv_pair_name_legacy(B2, B4)`, i.e. K at
//   2 bits and V at 4 -- pair bytes 4,608 + 8,704 = 13,312. **A DIFFERENT TIER.** Asking for the
//   narrowest token does not get the floor, and NOTHING IN THE TREE SAYS SO AT THE POINT OF ASKING.
//
// AND WHY THE FRONT END CANNOT SIMPLY GAIN A PARSER: `DType` deliberately has no `E8KvB2B2`
// (src/core/dtype.h, whose own comment records the omission as deliberate), so all seven input
// surfaces refuse `e8-bN` -- 19 `e8-bN` occurrences in the tree are comments, name-producers or
// static_asserts and not one of them is a parse. A parser added HERE, ahead of the enumerator,
// would make the front end ACCEPT a spelling the engine cannot honour: it would move the refusal
// from "unknown token" to "accepted, then silently ignored", which is strictly worse than the
// asymmetry it fixes. (The design's own rule: refusals are named, never fabricated.)
//
// SO THE DECISION THIS WINDOW TAKES, STATED ONCE AND WITH ITS REASON: **THE FLOOR STAYS BEYOND THE
// FRONT END, BY NAME, UNTIL THE ENGINE CAN HONOUR IT.** The engine-side change is the prerequisite
// and it is NOT landed here because it is not this line's edit set: it is a fourth `DType`
// enumerator plus its geometry, and `DType` is read by the artifact reader, the codec switches,
// the kernel routes and every one of the seven input surfaces. Landing half of it (an enumerator
// with no geometry, or a parser with no enumerator) is the failure mode F-6 itself warns about.
//
// THE OWED WORK, NAMED SO THE NEXT LINE DOES NOT HAVE TO RE-DERIVE IT:
//   (1) `DType::E8KvB2B2` in `src/core/dtype.h`, with the geometry this header already carries
//       (`e8_kv_pair_bytes(B2, B2) == 9216`, `e8_kv_pair_bits_x100(B2, B2) == 225`);
//   (2) THEN `e8_kv_pair_from_name()` HERE, whose contract is already fixed by the vocabulary
//       above: two plane names, K first, V second (`"e8-b2/e8-b2"`); the three binding-era aliases
//       (`rk4v4`/`rk3v4`/`rk2v4`) accepted as INPUT aliases because they are the deployed
//       vocabulary; an unknown token REFUSED by name and not defaulted; and a round-trip pin --
//       `from_name(pair_name(k, v)) == (k, v)` for every pair -- so a parser and its composer
//       cannot drift;
//   (3) and the ONE reading that would have caught this asymmetry earlier: any surface that
//       accepts a KV-format token must be able to name the TIER it selected, not just the token.
//
// ⚠ THIS BLOCK CHANGES NO CODE. It is a statement of the decision, placed in the header that owns
// the vocabulary, and the static_asserts below pin the two tiers apart so the statement cannot
// silently become false.
static_assert(e8_kv_pair_bytes(E8KvPlaneFormat::B2, E8KvPlaneFormat::B2) == 9216,
              "F1231/F-6: the descent's floor is B2 on BOTH planes -- pair bytes 9,216 "
              "(2 x 4,608), and CellMode::E8_2bit is priced at 4,608 per plane");
static_assert(e8_kv_pair_bytes(E8KvPlaneFormat::B2, E8KvPlaneFormat::B4) == 13312,
              "F1231/F-6: the CLI's narrowest askable token is K=B2, V=B4 -- pair bytes 13,312 "
              "(4,608 + 8,704), a DIFFERENT TIER from the floor above");
static_assert(e8_kv_pair_bytes(E8KvPlaneFormat::B2, E8KvPlaneFormat::B2) !=
                  e8_kv_pair_bytes(E8KvPlaneFormat::B2, E8KvPlaneFormat::B4),
              "F1231/F-6: THE ASYMMETRY ITSELF, AS AN INEQUALITY: the floor and the narrowest "
              "askable tier are not the same object, and a later round that made them equal would "
              "have to delete this assertion rather than silently change what `rk2v4` means");

// ===========================================================================
// [F1260 kvplanar] THE PLANE'S ORDER, AND THE PLANE'S READ SIDE -- TWO FACTS ABOUT A
// PLANE FORMAT THAT A CELL-LEVEL DECISION NEEDS AND THAT THIS FILE OWNS.
// ===========================================================================
//
// WHY HERE. A cell's two planes are ROWS of `kCellRungs` and every e8 row's plane is ONE enumerator
// of `E8KvPlaneFormat` (`Rk4v4` = B4, `Rk3v4` = B3, `E8_2bit` = B2 -- pinned in kv_cell_modes.h at
// the pair's own block). So "which plane format is next below this one on ITS OWN ladder" and "is
// this plane format realizable on the device today" are questions about the PLANE AXIS and belong
// to the header that defines it, not to the walk.

// THE ORDER OVER THE PLANE FORMATS, DERIVED FROM THE BYTES AND NOT DECLARED AS A LIST. The rule is
// the one `cell_next_cheaper` uses one level up, so the two orders cannot disagree: **the next
// format below you is the priciest format strictly cheaper than you**. It returns `here` when
// nothing is strictly cheaper, and the caller reads `next == here` as "this plane is at its floor"
// -- the same statement the walk's `AlreadyAtFloor` carries.
[[nodiscard]] constexpr E8KvPlaneFormat e8_kv_plane_next_cheaper(E8KvPlaneFormat here) noexcept {
    const E8KvPlaneFormat all[3] = {E8KvPlaneFormat::B4, E8KvPlaneFormat::B3, E8KvPlaneFormat::B2};
    E8KvPlaneFormat best = here;
    bool found = false;
    for (const E8KvPlaneFormat candidate : all) {
        if (e8_kv_plane_format_bytes(candidate) >= e8_kv_plane_format_bytes(here)) { continue; }
        if (!found || e8_kv_plane_format_bytes(candidate) > e8_kv_plane_format_bytes(best)) {
            best = candidate;
            found = true;
        }
    }
    return best;   // `here` when nothing is strictly cheaper: THE PLANE'S FLOOR
}
static_assert(e8_kv_plane_next_cheaper(E8KvPlaneFormat::B4) == E8KvPlaneFormat::B3 &&
                  e8_kv_plane_next_cheaper(E8KvPlaneFormat::B3) == E8KvPlaneFormat::B2 &&
                  e8_kv_plane_next_cheaper(E8KvPlaneFormat::B2) == E8KvPlaneFormat::B2,
              "the plane order is B4 > B3 > B2 and B2 is the floor -- derived from the bytes above, "
              "so a fourth format gets its rung here without editing this assertion");

// WHICH PLANE FORMATS HAVE A READ-SIDE CONSUMER TODAY. THIS IS A READING OF THE TREE AND NOT A
// PROMISE, and it is the fact a pair's PRICE must be read beside:
//
//   * B4 IS SHIPPED ON BOTH PLANES. On V it is the published i4 (this file's own
//     `e8_kv_v_plane_bytes() == e8_kv_plane_format_bytes(B4)` static_assert above); on K it is the
//     shipped scalar plane, decoded as packed NIBBLES (`gqa_attention_decode_i8.cuh:269-271` via
//     `gqa_kv_i4_code_index`).
//   * B3 AND B2 ARE WRITABLE AND NOT READABLE. The tree states it twice, verbatim:
//     `kv_storage_dtype.h:113-117` -- "decoder_state.cpp:422-442 lays its K plate at its own 96/64 B
//     row beside the family's 128 B V plate, and e8_lattice_narrow_kernel.cuh:187 appends it through
//     the ENCODER ..., so the tier is WRITABLE and NOT READABLE ... (e8_lattice_kv_plane.cuh,
//     `e8_kv_lattice_decode_group<3>` / `<2>`) has NO CALLER"; and again at
//     `causal_softmax_attention.cpp:81-89` and `:181-189`.
//
// ⚠ SO A NARROW PLANE IS *PRICED* AND NOT *DEPLOYABLE*, AND THE TWO ARE DIFFERENT ANSWERS. A caller
// that reported a B3/B2 pair as available would be quoting a lattice no reader can read: the missing
// DECODER is plane-agnostic (one `e8_kv_lattice_decode_group<WBits>` serves K and V alike), which is
// why the tree's refusal names the DECODER and not the plane.
enum class E8KvPlaneReadSide : std::uint8_t {
    Shipped  = 0,  // a consumer exists on a runtime path today
    Reserved = 1,  // the ENCODER exists and the DECODER has no caller: priced, not deployable
};

[[nodiscard]] constexpr E8KvPlaneReadSide e8_kv_plane_read_side(E8KvPlane plane,
                                                                E8KvPlaneFormat f) noexcept {
    // The plane is an INPUT and not a qualifier: the missing decoder is plane-agnostic, so B3/B2 are
    // Reserved on K and on V alike, and a landing that adds the caller flips ONE row here rather than
    // a sentence in a report.
    (void)plane;
    return f == E8KvPlaneFormat::B4 ? E8KvPlaneReadSide::Shipped : E8KvPlaneReadSide::Reserved;
}
static_assert(e8_kv_plane_read_side(E8KvPlane::K, E8KvPlaneFormat::B4) ==
                      E8KvPlaneReadSide::Shipped &&
                  e8_kv_plane_read_side(E8KvPlane::V, E8KvPlaneFormat::B4) ==
                      E8KvPlaneReadSide::Shipped &&
                  e8_kv_plane_read_side(E8KvPlane::K, E8KvPlaneFormat::B3) ==
                      E8KvPlaneReadSide::Reserved &&
                  e8_kv_plane_read_side(E8KvPlane::V, E8KvPlaneFormat::B2) ==
                      E8KvPlaneReadSide::Reserved,
              "the read side is a fact of the FORMAT and not of the plane: B4 is shipped on both "
              "(the published i4 on V, the nibble-decoded scalar plane on K) and the two narrow "
              "lattice plates have an encoder and no caller");

} // namespace ninfer::product
