#pragma once

// F1172 -- THE CELL: `(block, layer)` IS THE DECISION POSITION, AND A CELL'S ALLOWED MODES ARE
// A **SET**, NOT A POSITION ON ONE LADDER.
//
// WHY THIS FILE EXISTS. The order states the unit of allocation verbatim: "it is NOT the whole
// cell at int8 -- SOME layers inside it are int8; inside a cell there should be ALL KINDS of
// quantisation modes, not one uniform mode for the whole cell." That sentence names a set of
// modes per cell. The tree's only carrier for a mode is `RecallBlockCodec` (`kv_recall_block.h`)
// and it has THREE values -- `None / Int8Raw / Nvfp4Rans` -- none of which is `bf16`, while the
// order's own default is "int8 OR bf16". So the enum cannot express the order's START, let
// alone its set. GAP-MODESET, and this header is where it is answered.
//
// THE SECOND THING THIS FILE ANSWERS. The order states the walk's shape verbatim: "IT IS
// DEMOTED ONE RUNG AT A TIME". A rung needs an order, and the tree's only ordering of these
// modes is `kv_e8_width.h`'s `bits_x100 = 50*w + 225`, which is ONE dimension over ONE family
// (`w` in 2..4 over the e8 K plane). The order's set spans FAMILIES (bf16 / int8 / rk4v4 /
// e8-2bit / nvfp4), and no single scalar orders them. So the order here is NOT declared: it is
// DERIVED, per cell, from each rung's COLD RECORD BYTES -- the same quantity the engine already
// charges `kv_block_stage_bytes_per_block` with. See `cell_next_cheaper()`.
//
// WHAT THIS FILE IS NOT. It is not a kernel, it does not pack anything, and it does not decide
// anything. It is the type of the decision plus the arithmetic that orders it. Every byte in the
// table below is taken FROM THE TREE (`kv_tier_formats.h`) and the two that are NOT are marked
// `PriceState::Unpriced` with the gap that owes them named, rather than guessed.

#include "core/dtype.h"
// [F1260 kvplanar] THE PLANE AXIS, AND WHY THIS EDGE IS A DAG. `e8_kv_pair_bytes(K, V)` /
// `E8KvPlaneFormat` / `e8_kv_plane_read_side` are the e8 family's own facts, and a CELL's two planes
// are chosen from that axis, so the cell's header must see it. The direction is one-way: kv_e8_width.h
// includes only <cstdint>/<string>/<string_view> and nothing of this file's, so no cycle is created
// and the include sits here rather than being re-typed as a second table.
#include "product/kv_e8_width.h"
#include "product/kv_tier_formats.h"

#include <array>
#include <cstdint>
#include <string>

namespace ninfer::product {

// ---------------------------------------------------------------------------------------------
// 1. THE COLD CODEC AXIS, AS VALUES. (This is the axis the string table below is about.)
// ---------------------------------------------------------------------------------------------
//
// WHY A NEW ENUM WHEN `ColdCodec` EXISTS (kv_tier_formats.h: "None / Int8Raw / Nvfp4Rans").
// Because the two files disagree about the axis and both are in the same build. MEASURED, both
// READ IN SOURCE this session:
//
//   * The stage's `cold_codec_of` (impl/runtime/program_impl.h, the switch at byte ~691,771)
//     has SEVEN arms and names FOUR codecs:
//         case DType::I8:    return "the int8 raw slot";
//         case DType::NVFP4: return "the nvfp4 rANS slot";
//         case DType::ISO3:  return "an iso4e plane (no cold codec)";
//         case DType::E8Kv:  return "the rk4v4 raw slot (verbatim 4-bit codes + requantized g64)";
//         case DType::BF16:  return "the bf16 raw slot (E2M1 g64)";
//         case DType::FP8_E4M3FN: return "an fp8 plane (no cold codec)";
//         default:           return "a 16-bit plane (no cold codec)";
//   * `kv_tier_formats.h`'s class table (bytes ~26,641 / ~29,900 / ~30,100) gives TWO codec
//     values and folds the other two names into one of them:
//         case KvLayerClass::Nvfp4Fusion: return {ColdCodec::Nvfp4Rans, "nvfp4 rANS slot", ...};
//         case KvLayerClass::Rk4v4Fusion: return {ColdCodec::Int8Raw,  "rk4v4 -> int8 raw slot", ...};
//         case KvLayerClass::Classic16:   return {ColdCodec::Int8Raw,  "bf16 -> int8 raw slot", ...};
//
// SO: FOUR CODEC NAMES, TWO CODEC VALUES, AND THE DISTINCTION THE STAGE KEEPS IS THE ONE THE
// PRODUCT TABLE DROPS. That is not a style difference -- `bf16` and `rk4v4` are the order's
// default and its floor-adjacent rung respectively, and a 2-valued axis cannot name either.
// This enum is the 4-valued axis, with the 3 "no codec" arms collapsing to `None` on purpose
// (they have no pack arm, so for a per-cell MODE they are the same answer).
enum class LayerColdCodec : std::uint8_t {
    None      = 0,  // no pack arm at all (ISO3 / FP8_E4M3FN / 16-bit, per the switch above)
    Int8Raw   = 1,  // "the int8 raw slot"
    Nvfp4Rans = 2,  // "the nvfp4 rANS slot"
    Bf16Raw   = 3,  // "the bf16 raw slot (E2M1 g64)"
    Rk4v4Raw  = 4,  // "the rk4v4 raw slot (verbatim 4-bit codes + requantized g64)"
};

[[nodiscard]] constexpr const char* layer_cold_codec_name(LayerColdCodec codec) noexcept {
    switch (codec) {
    case LayerColdCodec::Int8Raw: return "int8-raw-slot";
    case LayerColdCodec::Nvfp4Rans: return "nvfp4-rans-slot";
    case LayerColdCodec::Bf16Raw: return "bf16-raw-slot";
    case LayerColdCodec::Rk4v4Raw: return "rk4v4-raw-slot";
    case LayerColdCodec::None: break;
    }
    return "none";
}

// WHETHER AN ARM EXISTS. This is the CODEC question and it is deliberately separate from the
// RECORD question below -- the stage's own comment says so verbatim (at byte ~691,600):
//     "the two facts this message exists to keep apart are CODEC and RECORD."
[[nodiscard]] constexpr bool layer_cold_codec_has_pack_arm(LayerColdCodec codec) noexcept {
    return codec != LayerColdCodec::None;
}

// THE DTYPE -> CODEC AXIS AS A VALUE, WHICH IS THE ONE THING THE STAGE'S STRING TABLE CANNOT BE.
//
// TWO STRING TABLES FOR THIS ONE AXIS LIVE IN THE SAME FILE AND THEY DISAGREE. Both READ IN
// SOURCE this session, both in `impl/runtime/program_impl.h`:
//
//   (1) `cold_codec_of`, the SEVEN-arm switch at byte ~691,771. Its arms are I8, NVFP4, ISO3,
//       E8Kv, BF16, FP8_E4M3FN, default. `DType::E8K3Kv` and `DType::E8K2Kv` HAVE NO ARM, so they
//       fall to `default: return "a 16-bit plane (no cold codec)";`
//   (2) the dtype-name switch at byte ~820,377, TEN arms, which is the one that gets it right:
//           case DType::NVFP4: continue;
//           case DType::I8: continue;
//           case DType::E8Kv: return "rk4v4";
//           case DType::E8K3Kv: return "rk3v4";
//           case DType::E8K2Kv: return "rk2v4";
//
// SO THE ORDER'S FLOOR IS NAMED **"a 16-bit plane"** BY ONE TABLE AND **"rk2v4"** BY THE OTHER.
// `DType::E8K2Kv` is a real 2-bit K-plane dtype (`core/dtype.h:34`, "the K code plane packed at
// 3 or 2 bits per element"), the census's codec column is printed by table (1), and the order's
// "in the END they go to e8 2bit" therefore has a WRONG NAME in the one place a reader would
// look for it. This function is that axis, once, as a value; the strings stay where they are,
// for printing.
[[nodiscard]] constexpr LayerColdCodec layer_cold_codec_of(DType dtype) noexcept {
    switch (dtype) {
    case DType::I8: return LayerColdCodec::Int8Raw;
    case DType::NVFP4: return LayerColdCodec::Nvfp4Rans;
    case DType::BF16: return LayerColdCodec::Bf16Raw;
    case DType::E8Kv: return LayerColdCodec::Rk4v4Raw;
    // THE FLOOR, AND THE HOLE. The dtype exists; the COLD CODEC does not. Named, not folded into
    // `default` with the fp8/16-bit planes, because those are different answers and the whole
    // point of the stage's own "no codec is not no record" message is that the two must not be
    // conflated.
    case DType::E8K2Kv:
    case DType::E8K3Kv:
    case DType::ISO3:
    case DType::FP8_E4M3FN:
    case DType::FP32:
    case DType::I32:
    case DType::U8:
    case DType::I64:
    case DType::FP16: return LayerColdCodec::None;
    }
    return LayerColdCodec::None;
}

// ---------------------------------------------------------------------------------------------
// 2. THE COLD RECORD AXIS. Two records; every record question is answered HERE, not by a codec.
// ---------------------------------------------------------------------------------------------
//
// `recall_block_codec_name`'s own header comment states the field's purpose as a RECORD question
// ("to let a reader decide whether the payload is a raw slot (kKvColdInt8PayloadBytes, 9232 B) or
// an rANS slot (kKvColdPoolStrideBytes, 9632 B)") while its name and its type are a CODEC
// question. That is the conflation, quoted from the header itself. The mapping is split here.
//
// THE MAPPING IS 4 -> 2, AND THAT IS THE POINT. Three codecs (`Int8Raw`, `Bf16Raw`, `Rk4v4Raw`)
// write ONE record. `decoder_state.cpp` proves it with three static_asserts (cited in
// `blob_F1168.md` section 4.3 E3, READ IN SOURCE there): `cold_slot_stride_for(DType::I8)`,
// `(DType::BF16)` and `(DType::E8Kv)` all `== ops::kColdI8SlotBytes`. `kv_tier_formats.h` proves
// the same thing from the product side with `ColdCodec::Int8Raw` above.
enum class CellRecord : std::uint8_t {
    None       = 0,  // nothing is written
    RawSlot    = 1,  // kKvColdInt8PayloadBytes  == 9232
    RansSlot   = 2,  // kKvColdPoolStrideBytes   == 9632
};

[[nodiscard]] constexpr CellRecord cell_record_of(LayerColdCodec codec) noexcept {
    switch (codec) {
    case LayerColdCodec::Nvfp4Rans: return CellRecord::RansSlot;
    case LayerColdCodec::Int8Raw:
    case LayerColdCodec::Bf16Raw:
    case LayerColdCodec::Rk4v4Raw: return CellRecord::RawSlot;
    case LayerColdCodec::None: break;
    }
    return CellRecord::None;
}

// The record's stride, FROM THE TREE'S OWN CONSTANTS and never re-typed. `-1` is not a value this
// returns for a record that exists; a record with no stride is `CellRecord::None`.
[[nodiscard]] constexpr std::int32_t cell_record_bytes(CellRecord record) noexcept {
    switch (record) {
    case CellRecord::RawSlot: return kKvColdInt8PayloadBytes;   // 9232, pinned == 9232 below it
    case CellRecord::RansSlot: return kKvColdPoolStrideBytes;   // 9632, pinned == 9632 below it
    case CellRecord::None: break;
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------
// 3. THE MODE SET. The order's own five names, PLUS the 3-bit tier the owner ruled in on
// 2026-09-29 ("4bit 3bit 2bit 的三层降级都得是通的").
// ---------------------------------------------------------------------------------------------
//
// The five names are the order's own set: "bf16 / int8 / rk4v4 / e8-2bit / nvfp4".
//
// THE NUMERIC VALUES CARRY EXACTLY ONE MEANING AND NO LADDER: **ZERO IS THE ORDER'S DEFAULT.**
// A value-initialised `BlockVector` is therefore the order's STARTING POINT mechanically, which
// is what makes the "OFF is byte-for-byte the pre-image" property a theorem about a
// default-constructed object rather than a switch a caller has to remember to leave alone. The
// values are NOT a price order and NOT a rung order -- `cell_next_cheaper()` derives the rung
// order from `plane_bytes`, and reading these numbers as a ladder is the one-dimensional habit
// the order's second sentence forbids.
//
// ⚠⚠ [F1231, from F1230 1.3] THE TWO LINES ABOVE USED TO SAY `record_bytes` / `record_bytes` AND
// THE FIRST OF THEM WAS FALSE -- F1209's site 2 moved the derived order onto the PLANE ruler
// (`candidate.plane_bytes >= here.plane_bytes` below), and a comment that describes a closed
// defect in the PRESENT TENSE is how a later round re-opens it. The one-dimensional habit the
// sentence warns about is now named exactly: A RUNG IS NOT ITS COLD RECORD.
enum class CellMode : std::uint8_t {
    // THE TWO COLUMNS, IN THE ORDER THE FILE USES THEM: `plane_bytes` is the memory a demotion
    // GIVES BACK and the ruler the order, the charge and the step's delta are all denominated in;
    // `record_bytes` is the COLD SLOT the pool STAGES (two values only: 9,232 raw / 9,632 rANS) and
    // it is carried as a READING. [F1231/F-5 site 3] The rows used to name only the record, which
    // is the column that CONFLATES six rungs into two -- so a reader could not see the ladder in
    // the very table that defines it. The record is not deleted: `decoder_state.cpp`'s three
    // static_asserts are about rungs SHARING a record, and that fact must stay legible.
    Int8    = 0,  // plane 16,896 (record 9,232)  -- one of the two defaults AND the zero value
    Bf16    = 1,  // plane 32,768 (record 9,232)  -- the order's other default
    Rk4v4   = 2,  // plane  8,704 (record 9,232)  -- e8 lattice at 4 bits (the 4-bit tier)
    Nvfp4   = 3,  // plane  9,216 (record 9,632)  -- the priciest rung strictly below int8
    E8_2bit = 4,  // plane  4,608 (record 9,216)  -- THE FLOOR. See G6 below.
    // [planes 2026-09-29] THE 3-BIT TIER, ADDED ON THE OWNER'S RULING: "4bit 3bit 2bit 的三层降级
    // 都得是通的" -- without it the 3-bit layer has NO EDGE and the ladder is 4 -> 2 with a hole in
    // it. The header predicted this one-line change verbatim (see the MIDDLE RUNGS note below: "ADDING
    // IT IS A ONE-LINE CHANGE TO `CellMode` AND `kCellRungs`; the walk, the order and the histogram
    // are all written to take it without any other edit") -- and that prediction was VERIFIED before
    // the write: the histogram printer iterates `for i < kCellModeCount` over `kCellRungs[i].mode`
    // (kv_block_descent.h:254-259), and `cell_rung` / `cell_default_tier` / `cell_vector_charge` /
    // `CellAllowedSet` all iterate or bit-mask over the count, so no other file changes.
    // APPENDED, NOT INSERTED: `Int8 == 0` is the "zero is the order's default" theorem and must not
    // move, and the table's tie-break reads "first encountered wins", which a strictly-decreasing
    // `plane_bytes` column (asserted below) makes moot.
    Rk3v4   = 5,  // raw record, 9232 B  -- e8 lattice at 3 bits (the 3-bit tier)
};

inline constexpr std::uint32_t kCellModeCount = 6;

[[nodiscard]] constexpr const char* cell_mode_name(CellMode mode) noexcept {
    switch (mode) {
    case CellMode::Nvfp4: return "nvfp4";
    case CellMode::Int8: return "int8";
    case CellMode::Bf16: return "bf16";
    case CellMode::Rk4v4: return "rk4v4";
    case CellMode::Rk3v4: return "rk3v4";
    case CellMode::E8_2bit: return "e8-2bit";
    }
    return "?";
}

// THE PRICE, AND ITS PROVENANCE. A number that is in the tree carries the place it was read; a
// number that is NOT is `Unpriced` and carries the gap that owes it. There is no third state and
// no silent default: a rung whose price state is not `Priced` CANNOT be entered by the walk
// (`cell_next_cheaper` refuses it BY NAME), which is how the order's "in the END they go to
// e8 2bit" is recorded as unreachable today rather than as a TODO.
enum class PriceState : std::uint8_t {
    Priced   = 0,  // `record_bytes` is a fact read from the tree
    Unpriced = 1,  // no byte cost exists upstream; `record_bytes` is NOT meaningful
};

struct CellRung {
    CellMode mode = CellMode::Int8;
    LayerColdCodec codec = LayerColdCodec::Int8Raw;
    // THE DTYPE THIS RUNG IS SPELLED BY UPSTREAM. A value, not prose, so the mode axis and the
    // dtype axis cannot drift -- and so that the order's floor (`DType::E8K2Kv`) is reachable
    // from a mode by construction.
    DType dtype = DType::I8;
    PriceState price_state = PriceState::Priced;
    std::int32_t record_bytes = 0;
    // [planes 2026-09-29] THE RESIDENT-PLANE PRICE, AND THE RULER THE DESCENT READS.
    //
    // THE CALIBER, NAMED AND NOT ASSUMED. `plane_bytes` is ONE resident plane per head-page -- the
    // caliber `kv_tier_formats.h:538-541` names verbatim: "Resident plane bytes per head-page of one
    // class, from the same plane geometry the bit ladder uses (product/kv_bit_budget.h: bits_x100 *
    // 16384 / 800), so the two can never disagree: 32768 / 16896 / 17408 / 9216 / 9216 / 8704 for
    // bf16 / int8 / fp8 / nvfp4 / iso4e / rk4v4." Every value in this column is pinned by a
    // `static_assert` in that file (`:575-586`), so none of them is re-typed or re-derived here.
    //
    // ONE PLANE, NOT A PAIR, AND THE SAME UNIT `record_bytes` IS ALREADY IN. A cell's engine charge
    // is `2 * kv_heads * record_bytes` (`cell_vector_charge` below), where the 2 IS the K and V
    // planes -- and the ledger pins `bytes_per_block = 1,181,696 = 16 layers * 4 KV heads * 2 planes
    // * 9232`. So both columns are per-plane and the factor 2 lives in exactly one place. **A pair
    // sum must NOT be put in this column**: the decoupled e8 pair is `2 * plane_bytes`, and the
    // first draft of this landing did put one there (`pair(B2,B2) = 9216`) against four one-plane
    // values, which priced the 2-bit FLOOR ABOVE the 4-bit rung `rk4v4 = 8704` and made the order's
    // own sentence ("in the END they go to e8 2bit") unreachable through the ruler.
    std::int32_t plane_bytes = 0;
    const char* price_source = "";
};

// THE TABLE. Ordered by `mode`'s value, NOT by price -- the price order is derived per cell in
// `cell_next_cheaper()`, and keeping the table in any particular price order would invite a
// reader to treat the row index as the rung number (the exact one-dimensional habit the order's
// second sentence forbids).
inline constexpr std::array<CellRung, kCellModeCount> kCellRungs{{
    // Int8: THE ORDER'S DEFAULT. "the int8 raw slot" / "rk4v4 -> int8 raw slot".
    // [planes] plane_bytes = 16896, ONE int8 plane, READ IN SOURCE: kv_tier_formats.h:352
    //     static_assert(kKvColdResidentInt8Bytes == 16896, "int8 resident plane per head-page")
    //     and, in the caliber family itself, :577 "int8: 8.25 b/el" (:310-312 is the expression).
    {CellMode::Int8, LayerColdCodec::Int8Raw, DType::I8, PriceState::Priced,
     kKvColdInt8PayloadBytes,
     16896,
     "kv_tier_formats.h kKvColdInt8PayloadBytes (pinned == 9232)"},
    // Bf16: THE ORDER'S OTHER DEFAULT. kv_tier_formats.h KvLayerClass::Classic16 ->
    // {ColdCodec::Int8Raw, kKvColdInt8PayloadBytes, ...} and its comment says so verbatim:
    // "Reachable: enqueue_cold_compressions packs a bf16 layer into the raw slot through
    // entropy_cold_requant's Bf16G64 arm and cold_i8_slot_pack_raw".
    // [planes] plane_bytes = 32768, ONE bf16 plane, READ IN SOURCE: kv_tier_formats.h:460
    //     verbatim "the resident plane this codec gives back is the BF16 one (32768), not the int8
    //     one (16896)" -- and :575 static_assert(kv_cold_resident_bytes_of(Classic16) == 32768,
    //     "bf16: 16.00 b/el"); :546 is the expression (head_dim * tokens * 2, no scale plane).
    {CellMode::Bf16, LayerColdCodec::Bf16Raw, DType::BF16, PriceState::Priced,
     kKvColdInt8PayloadBytes,
     32768,
     "kv_tier_formats.h KvLayerClass::Classic16 (bf16 -> int8 raw slot, 9232)"},
    // Rk4v4: kv_tier_formats.h KvLayerClass::Rk4v4Fusion -> ColdCodec::Int8Raw with
    // kKvColdInt8PayloadBytes. NOTE, and it is load-bearing: this is the SAME 9232 as Int8.
    // [planes] plane_bytes = 8704, AND THIS ONE IS COMPUTED, NOT ASSERTED -- the only row in this
    // column that is. The expression is kv_tier_formats.h:451-452 (inside the Rk4v4Fusion arm):
    //     (kKvColdHeadDim / 2) * kKvColdPageTokens
    //   + (kKvColdHeadDim / kKvColdInt8Group) * kKvColdPageTokens * 2
    //   = (256 / 2) * 64 + (256 / 64) * 64 * 2
    //   =       8,192    +          512        = 8,704   B/head-page
    // The geometry is :259-261 (head_dim 256, page_tokens 64, int8_group 64); the arm's own comment
    // is :558-560 verbatim "rk4v4: a packed 4-bit lattice code per two elements + one FP16 scale per
    // 64." It is ONE plane (8192 B of 4-bit codes + 512 B of g64 fp16 scales), which is why it is
    // directly comparable to the per-plane record 9232 -- :434-436 verbatim "an rk4v4 layer's
    // resident plane is 8704 B/head-page ... while the shared raw record is 9232 B, so this codec
    // COSTS 528 B/head-page". Same number again at kv_e8_width.h:310 "e8 B4 plane: 128*64 + 512 =
    // 8704 B/head-page"; the DECOUPLED pair is twice it (:331 pair(B4,B4) = 17408).
    {CellMode::Rk4v4, LayerColdCodec::Rk4v4Raw, DType::E8Kv, PriceState::Priced,
     kKvColdInt8PayloadBytes,
     8704,
     "kv_tier_formats.h KvLayerClass::Rk4v4Fusion (rk4v4 -> int8 raw slot, 9232)"},
    // Rk3v4: THE 3-BIT TIER, the row that makes "4bit 3bit 2bit" a connected ladder.
    // [planes] plane_bytes = 6656, ONE 3-bit plane, READ IN SOURCE twice, both verbatim:
    //   kv_tier_formats.h:583-584  static_assert(kv_cold_resident_bytes_of(KvLayerClass::Rk3v4Fusion)
    //                              == 6656, "rk3v4 K plane: 96*64 + 512 = 6656 B/head-page");
    //   kv_tier_formats.h:561-566  the body it pins: (256*3/8)*64 + (256/64)*64*2
    //                              = 96*64 + 512 = 6,144 + 512 = 6,656
    //   kv_e8_width.h:185-186      static_assert(e8_kv_k_plane_bytes(E8KvWidth::W3) == 6656,
    //                              "e8 W3 K plane: 96*64 + 512 = 6656 B/head-page");
    // NAMED TRAP, because the e8 family carries two rulers one letter apart: the WIDTH ladder's
    // `e8_kv_layer_bytes(W3) == 15360` (kv_e8_width.h:190, verbatim "e8 W3 layer: 6656+8704") is
    // K@B3 + the SHIPPED i4 V -- NOT the decoupled pair. The decoupled 3-bit pair is
    // pair(B3,B3) = 6656 + 6656 = 13312 (pinned by e8_kv_pair_bits_x100(B3,B3) == 325 at
    // kv_e8_width.h:336-337; check 13312*8*100/(2*256*64) = 325). At ONE PLANE the two calibers
    // agree exactly, so 6656 is the value on this column's caliber with no ambiguity left.
    // The codec and dtype follow the FLOOR row's precedent (the master's ruling moved the floor to
    // the decoupled e8 spelling): Rk4v4Raw over the raw slot, so its RECORD is the shared 9232 and
    // its step is a codec change on the same record -- which is exactly the G7 shape this ladder
    // has always been made of, now named on a third tier instead of hidden between two.
    {CellMode::Rk3v4, LayerColdCodec::Rk4v4Raw, DType::E8K3Kv, PriceState::Priced,
     kKvColdInt8PayloadBytes,
     6656,
     "kv_tier_formats.h:583-584 kv_cold_resident_bytes_of(Rk3v4Fusion) == 6656 (96*64 + 512); "
     "kv_e8_width.h:185-186 e8_kv_k_plane_bytes(W3) == 6656 -- ONE plane, same caliber as int8"},
    // Nvfp4: the widest record. "nvfp4 rANS slot", kv_tier_formats.h::
    // static_assert(kKvColdPoolStrideBytes == 9632, ...)
    // [planes] plane_bytes = 9216, ONE nvfp4 plane, READ IN SOURCE: kv_tier_formats.h:351
    //     static_assert(kKvColdResidentNvfp4Bytes == 9216, "nvfp4 resident plane per head-page")
    //     and :578 "nvfp4: 4.50 b/el" (:307-309 is the expression: nibble codes + E4M3 g16 scales).
    //     The prose at :489 and :413 calls this geometry a "plane pair"; the EXPRESSION is one
    //     plane, the same family :538-541 lines up with rk4v4's 8704, and the per-plane reading is
    //     the one the caliber family and the 9232/9632 records are both in.
    {CellMode::Nvfp4, LayerColdCodec::Nvfp4Rans, DType::NVFP4, PriceState::Priced,
     kKvColdPoolStrideBytes,
     9216,
     "kv_tier_formats.h kKvColdPoolStrideBytes (pinned == 9632)"},
    // E8_2bit: UNPRICED, AND THIS ROW IS A GAP, NOT A PLACEHOLDER.
    //   THE DTYPE IS REAL. `DType::E8K2Kv` (`core/dtype.h:34`) is "the e8 family, narrower K-plane
    //   code widths (e8k3 / e8k2) ... with the K code plane packed at 3 or 2 bits per element",
    //   and the engine names it `"rk2v4"` in its ten-arm dtype switch (program_impl.h byte
    //   ~820,377). So the order's floor HAS an identity upstream.
    //   THE BYTE COST DOES NOT. G6, READ IN SOURCE: `kv_e8_width.h`'s floor rung is `W2` =
    //   13312 B = 3.25 b/el and `e8_kv_v_plane_bytes()` is not a function of the width at all --
    //   its own static_assert reads `"V stays i4 at 4.25 b/el (not a width input)"`. And that
    //   13312 is a RESIDENT plane price, not a COLD RECORD: no cold slot exists for this rung
    //   (the seven-arm codec switch has no arm for it -- see `layer_cold_codec_of`), which is the
    //   other half of why this row cannot be filled from the tree.
    //   CLAIMED ONLY, NOT USED: a sibling line (e8vaxis / F1166) is quoted as giving
    //   `pair(B2,B2)` = 4608+4608 = 9216 B. That number is NOT READ IN SOURCE by this line, so
    //   it is NOT written here. When it is read, it goes in this row's `record_bytes` and nowhere
    //   else -- and NOTE it would be 16 B BELOW the raw record, so the descent would gain exactly
    //   0.17 % of a raw slot per cell, which is a number worth having before any test is planned.
    // ⚠⚠ SUPERSEDED BY THE MASTER'S RULING (2026-09-29): "rk3v4 换成解耦后的 e8". The
    // block above is KEPT AS HISTORY and is now WRONG as a description of this row -- do not read
    // it as the current state. The ruling replaced the old K-side rk3v4 axis with the DECOUPLED
    // e8 PAIR (F1166's plane-agnostic format) and priced this row from the pair family:
    //     kv_e8_width.h:333  static_assert(e8_kv_pair_bytes(E8KvPlaneFormat::B2, E8KvPlaneFormat::B2) == 9216, ...)
    //     kv_e8_width.h:314-315  static_assert(e8_kv_plane_format_bytes(E8KvPlaneFormat::B2) == 4608, ...)  (two planes, :333 == 225 = 2.25 b/el)
    // NAMED READING, NOT SMOOTHED OVER: the price 9216 is a PLANE price and this row's RECORD is
    // the raw slot at 9232 -- 16 B apart. That 16 B IS the currency of the step INTO the floor
    // (9232 - 9216 = 16 B/head-page = 0.17 %), and it is also what makes the step EXIST under
    // `cell_next_cheaper`'s strict "smaller bytes" rule (9216 < 9232) -- which is why no
    // GAP-CODECENUM cold slot has to be invented. Flattening the 16 B would hide "is the floor
    // worth descending to", so it is written here by name.
    // [planes 2026-09-29] ⚠ THIS ROW'S `plane_bytes` IS **4608**, NOT THE 9216 THE FIRST DRAFT OF
    // THIS COLUMN PUT HERE. The 9216 above is `pair(B2,B2)` = `plane(B2) + plane(B2)` = the sum of
    // TWO planes (kv_e8_width.h:285-286 is the definition of `pair_bytes`; :312-313 is the one-plane
    // assert), while every other row of `plane_bytes` is ONE plane -- so 9216 in this column put a
    // full ruler next to four half ones and priced the 2-bit FLOOR above the 4-bit rung (8704),
    // which makes the order's own "in the END they go to e8 2bit" unreachable through the ruler.
    // 4608 is the same number twice in the tree, both READ IN SOURCE:
    //   kv_e8_width.h:314-315   static_assert(e8_kv_plane_format_bytes(E8KvPlaneFormat::B2) == 4608,
    //                           "e8 B2 plane: 64*64 + 512 = 4608 B/head-page");
    //   kv_tier_formats.h:585-586 static_assert(kv_cold_resident_bytes_of(Rk2v4Fusion) == 4608,
    //                           "rk2v4 K plane: 64*64 + 512 = 4608 B/head-page (== rk2v4-e8's K geometry)");
    // The DECOUPLED pair is exactly twice it, 9216, so nothing about the master's ruling changed --
    // only which half of it this column carries. `record_bytes` is deliberately LEFT AT 9216: it was
    // set by that ruling and it is the field the 16 B floor-step currency is written about.
    {CellMode::E8_2bit, LayerColdCodec::Rk4v4Raw, DType::E8Kv, PriceState::Priced, 9216,
     4608,
     "kv_e8_width.h:333 static_assert(e8_kv_pair_bytes(E8KvPlaneFormat::B2, E8KvPlaneFormat::B2) "
     "== 9216) -- the DECOUPLED e8 pair price, a PLANE price; the record it lands in is the raw "
     "slot at 9232, delta 16 B (the currency of the floor step, named not smoothed)"},
}};

// THE MIDDLE RUNGS OF THE E8 FAMILY, NAME-BY-COORDINATE, AND WHY THEY ARE NOT IN THE SET ABOVE.
// READ IN SOURCE: `core/dtype.h` also carries `E8K3Kv = 11` ("rk3v4") and the engine's ten-arm
// switch names it; `kv_e8_width.h` prices it at `W3` = 15360 B/head-page = 3.75 b/el with a
// 6656 B K plane. It sits BETWEEN `int8` (16896 B resident, 9232 B cold) and `e8-2bit`, so it is
// the obvious candidate for a rung the descent could actually spend -- and it is NOT in the
// order's mode set, so it is not in `CellMode`. Named here as a candidate for the floor's slot
// geometry (see `blob_F1172.md` section 6, dispatch sentence 3) rather than added to the set on
// this line's own authority. ADDING IT IS A ONE-LINE CHANGE TO `CellMode` AND `kCellRungs`; the
// walk, the order and the histogram are all written to take it without any other edit, which is
// the test that the order really is derived rather than declared.
// [G6 2026-09-29] THE FLOOR'S DTYPE, FOLLOWING THE ROW. The master's ruling moved the floor
// row to `DType::E8Kv` (the decoupled e8 family spelling), so this constant -- whose whole job
// is to NAME the floor -- follows it. `DType::E8K2Kv` itself is NOT deleted and NOT deprecated:
// it is a real dtype on other paths (the append side's 64-byte one, `program_impl.h`'s "rk2v4"
// name, `decoder_state.cpp`'s arms), and this line never had the authority to remove it.
inline constexpr DType kCellFloorDtype = DType::E8Kv;

[[nodiscard]] constexpr const CellRung& cell_rung(CellMode mode) noexcept {
    for (const CellRung& rung : kCellRungs) {
        if (rung.mode == mode) { return rung; }
    }
    return kCellRungs[1];  // unreachable for a valid CellMode; Int8 is the safe identity
}

[[nodiscard]] constexpr bool cell_mode_is_priced(CellMode mode) noexcept {
    return cell_rung(mode).price_state == PriceState::Priced;
}

// ---------------------------------------------------------------------------------------------
// 3b. THE PLANE COLUMN IS ONE RULER, AND IT IS STRICTLY ORDERED. [planes 2026-09-29]
// ---------------------------------------------------------------------------------------------
//
// WHY THESE ASSERTS EXIST AT ALL. The first draft of this column mixed two calibers: four rows that
// are ONE plane and one row (`E8_2bit`) that was the SUM of two. The damage was not cosmetic and it
// was not the sort of thing a reader would catch: `nvfp4 9216` and `e8-2bit 9216` then TIED, so
// `argmax{ plane_bytes' < plane_bytes }` gave no edge between them, and `rk4v4 8704` came out BELOW
// `e8-2bit 9216` -- i.e. the 2-bit FLOOR was priced ABOVE the 4-bit rung, so the order's own
// sentence ("in the END they go to e8 2bit") could not be reached at all. The mixed column needed a
// second "equal means no edge" note to explain its tie; the ONE-caliber column has no tie to
// explain, and that is the difference between a derived order and a declared one.
//
// "EQUAL MEANS NO EDGE" HAS EXACTLY ONE INSTANCE IN THIS TREE, AND IT IS NOT IN THIS COLUMN. The
// rule is real -- the order reads STRICTLY smaller, so equally-priced rungs have no edge either way
// -- and its one instance is the COLD RECORD: `bf16` and `int8` both write the 9232 B raw slot
// (`decoder_state.cpp:870-872` verbatim, "bf16 cold slot must be the int8 raw record"), which is
// why `record_bytes` alone can never separate the order's two defaults. The SECOND instance the
// first draft wanted to write here (nvfp4 vs the e8 pair, both 9216) was an artifact of the mixed
// caliber and DISSOLVES under one ruler: nvfp4 is 9216 and e8-2bit is 4608, and neither is the
// other's next rung by equality -- they are simply ordered.
//
// AND THE x2 LIVES IN EXACTLY ONE PLACE. `plane_bytes` is one plane; `cell_vector_charge` applies
// the K+V pair itself (`2 * kv_heads * ...`), which is the `2` the ledger's
// `bytes_per_block = 1,181,696 = 16 * 4 * 2 * 9232` also carries. Put a pair sum in this column and
// that 2 is applied twice.
static_assert(cell_rung(CellMode::Bf16).plane_bytes   == 32768, "bf16  plane: 16.00 b/el");
static_assert(cell_rung(CellMode::Int8).plane_bytes   == 16896, "int8  plane:  8.25 b/el");
static_assert(cell_rung(CellMode::Nvfp4).plane_bytes   == 9216, "nvfp4 plane:  4.50 b/el");
static_assert(cell_rung(CellMode::Rk4v4).plane_bytes   == 8704, "rk4v4 plane:  4.25 b/el");
static_assert(cell_rung(CellMode::Rk3v4).plane_bytes   == 6656, "rk3v4 plane:  3.25 b/el");
static_assert(cell_rung(CellMode::E8_2bit).plane_bytes == 4608, "e8-2bit plane: 2.25 b/el "
              "(ONE B2 plane -- the decoupled pair is 2*4608 = 9216 and is NOT this column's unit)");
static_assert(cell_rung(CellMode::Bf16).plane_bytes > cell_rung(CellMode::Int8).plane_bytes &&
                  cell_rung(CellMode::Int8).plane_bytes > cell_rung(CellMode::Nvfp4).plane_bytes &&
                  cell_rung(CellMode::Nvfp4).plane_bytes > cell_rung(CellMode::Rk4v4).plane_bytes &&
                  cell_rung(CellMode::Rk4v4).plane_bytes > cell_rung(CellMode::Rk3v4).plane_bytes &&
                  cell_rung(CellMode::Rk3v4).plane_bytes > cell_rung(CellMode::E8_2bit).plane_bytes,
              "THE PLANE COLUMN MUST BE STRICTLY DECREASING along bf16 > int8 > nvfp4 > rk4v4 > "
              "rk3v4 > e8-2bit. A tie here is an EDGE that the order does not have (it reads "
              "strictly smaller); an inversion here is worse -- it makes the order's floor "
              "unreachable and the ladder's 4/3/2-bit tiers disconnected, which is the whole "
              "ruling this row set exists to carry. If a future rung has no byte cost today, put "
              "it in the column as `PriceState::Unpriced` and let `cell_next_cheaper` refuse it BY "
              "NAME; do NOT widen this column's caliber to make room for it.");

// =============================================================================================
// [F1260 kvplanar] ⭐ 3c. THE PAIR: A CELL'S STATE IS (K FORMAT, V FORMAT), NOT ONE RUNG.
// =============================================================================================
//
// THE DEFECT THIS ANSWERS, IN ONE SENTENCE. `plane_bytes` is ONE plane and `cell_vector_charge`
// applies the `2` itself (`2 * kv_heads * sum_l plane_bytes`), so **the `2` is the implicit
// statement "this cell's K plane and V plane are the same format"** -- and the ladder that follows
// is a DIAGONAL ladder (`bf16/bf16 -> int8/int8 -> nvfp4/nvfp4 -> B4/B4 -> B3/B3 -> B2/B2`). A cell
// whose two planes differ **cannot be represented in the type**, so it can be neither printed, nor
// charged, nor reached by a step. MEASURED, in the tree's own words, that the deployed tiers are NOT
// on that diagonal:
//
//   * `kv_kv_bits.h:279` -- "The two narrow widths keep the SAME V plane (i4): **only the K code
//     width moves**".
//   * `kv_bit_budget.h:301-312` -- "TWO ROWS ARE K-NARROWED PAIRS: `rk3v4 375` and `rk2v4 325`,
//     whose V STAYS i4 ... **and it is NOT the geometry the descent's `CellMode` can express**".
//   * `kv_bit_budget.h:483-490` -- "`rk3v4` is THE ONE ROW where the ladder's pair ruler and the
//     descent's symmetric-cell ruler disagree (**375 vs 325**)".
//   * `kv_e8_width.h` (`e8_kv_pair_name_legacy`) -- `rk4v4=(B4,B4)`, `rk3v4=(B3,B4)`,
//     `rk2v4=(B2,B4)`, and `e8_kv_pair_bytes` prices every one of them.
//
// SO THE PAIR IS A VALUE, AND ITS PRICE IS THE SUM OF THE TWO PLANES' OWN PRICES -- the SAME
// one-plane column, added twice, which is what makes the two currencies ONE number:
//
//      cell_pair_bytes(p) := cell_rung(p.k).plane_bytes + cell_rung(p.v).plane_bytes
//      cell_pair_charge(p, kv_heads) := kv_heads * cell_pair_bytes(p)
//
// ⚠ AND THE PRE-IMAGE IS A THEOREM ABOUT A DIAGONAL PAIR, NOT A SWITCH: for every row of the table,
// `cell_pair_bytes(m, m) == 2 * cell_rung(m).plane_bytes` (asserted below for all six rows), so
// `cell_pair_charge(m, m, h)` IS today's `cell_vector_charge` number for the same vector. The
// `plane_bytes` COLUMN IS NOT TOUCHED -- its own note forbids putting a pair sum in it, and this
// block is exactly what that note asks for instead.
struct CellPair {
    CellMode k = CellMode::Int8;   // the K plane's format, one row of `kCellRungs`
    CellMode v = CellMode::Int8;   // the V plane's format, the SAME table chosen independently

    // VALUE EQUALITY, BECAUSE THIS TYPE IS COMPARED AND C++20 DOES NOT SYNTHESISE IT FOR AN
    // AGGREGATE. `!=` comes from `==`. Used by the walk's deepest-pair census and by the plan's
    // "uniform layer" test, both of which must compare the PAIR and not one coordinate of it.
    [[nodiscard]] constexpr bool operator==(const CellPair& other) const noexcept = default;
};

[[nodiscard]] constexpr std::int32_t cell_pair_bytes(CellPair p) noexcept {
    return cell_rung(p.k).plane_bytes + cell_rung(p.v).plane_bytes;
}

[[nodiscard]] constexpr bool cell_pair_is_diagonal(CellPair p) noexcept { return p.k == p.v; }

// THE CHARGE OF ONE CELL. `kv_heads *` and NOT `2 * kv_heads *`: the `2` has become the second
// addend of `cell_pair_bytes`, so it lives in exactly one place still -- the pair.
[[nodiscard]] constexpr std::int64_t cell_pair_charge(CellPair p, std::int32_t kv_heads) noexcept {
    if (kv_heads <= 0) { return 0; }
    return static_cast<std::int64_t>(kv_heads) * static_cast<std::int64_t>(cell_pair_bytes(p));
}

// THE PAIR'S NAME, COMPOSED FROM THE ROWS' OWN NAMES, K FIRST, V SECOND -- the same grammar
// `e8_kv_pair_name(k, v)` uses one level down, so a pair and its spelling cannot drift.
[[nodiscard]] inline std::string cell_pair_name(CellPair p) {
    return std::string("k:") + cell_mode_name(p.k) + "/v:" + cell_mode_name(p.v);
}

// THE ROW'S PLANE, ON THE E8 SIDE. The three e8 rows ARE three `E8KvPlaneFormat`s, and the mapping
// is by GEOMETRY and not by name: `Rk4v4`'s 8,704 == B4's plane, `Rk3v4`'s 6,656 == B3's, and
// `E8_2bit`'s 4,608 == B2's. `true` is the answer for a row that is not in the e8 family, and the
// caller must read the out-parameter only when it is true.
[[nodiscard]] constexpr bool cell_row_e8_plane(CellMode row, E8KvPlaneFormat& out) noexcept {
    if (row == CellMode::Rk4v4) { out = E8KvPlaneFormat::B4; return true; }
    if (row == CellMode::Rk3v4) { out = E8KvPlaneFormat::B3; return true; }
    if (row == CellMode::E8_2bit) { out = E8KvPlaneFormat::B2; return true; }
    return false;
}

// ⭐ (1) THE PRE-IMAGE THEOREM, FOR EVERY ROW OF THE TABLE. This is the assertion that makes
// "unset = byte-for-byte the pre-image" a property of the TYPE rather than a promise about the
// walk: a diagonal pair charges exactly what the pre-image charged, for every rung, by arithmetic.
[[nodiscard]] constexpr bool cell_pair_diagonal_is_the_preimage_charge() noexcept {
    for (const CellRung& rung : kCellRungs) {
        const CellPair p{rung.mode, rung.mode};
        if (cell_pair_bytes(p) != 2 * rung.plane_bytes) { return false; }
    }
    return true;
}
static_assert(cell_pair_diagonal_is_the_preimage_charge(),
              "F1260: a DIAGONAL pair must charge exactly 2 x the row's one-plane price -- that is "
              "what makes `cell_pair_charge(m, m, h)` the pre-image `cell_vector_charge(m, h)` for "
              "every rung, and therefore what makes the OFF/default path byte-for-byte unchanged");

// ⭐ (2) THE PAIR'S PRICE IS THE PLANE AXIS' PRICE, TO THE BYTE. The e8 rows' planes are
// `E8KvPlaneFormat`s, and `e8_kv_pair_bytes(K, V)` is the tree's arbitrary-pair price
// (`kv_e8_width.h`); the two must be ONE number or the plan's currency and the CLI storage
// grammar's would be two. All FOUR deployed/mixed pairs the tree names are asserted, and the
// citation for each is the string the header that owns the vocabulary prints.
static_assert(cell_pair_bytes(CellPair{CellMode::Rk4v4, CellMode::Rk4v4}) ==
                  e8_kv_pair_bytes(E8KvPlaneFormat::B4, E8KvPlaneFormat::B4),
              "F1260: (rk4v4, rk4v4) IS (B4, B4) -- the shipped pair, 17,408 B");
static_assert(cell_pair_bytes(CellPair{CellMode::Rk3v4, CellMode::Rk4v4}) ==
                  e8_kv_pair_bytes(E8KvPlaneFormat::B3, E8KvPlaneFormat::B4),
              "F1260: (rk3v4, rk4v4) IS THE DEPLOYED rk3v4 -- K at 3 bits, V still i4 = (B3,B4) = "
              "15,360 B. This pair is what `kv_bit_budget.h:301-312` says the descent's CellMode "
              "CANNOT express; from this landing it can");
static_assert(cell_pair_bytes(CellPair{CellMode::E8_2bit, CellMode::Rk4v4}) ==
                  e8_kv_pair_bytes(E8KvPlaneFormat::B2, E8KvPlaneFormat::B4),
              "F1260: (e8-2bit, rk4v4) IS THE DEPLOYED rk2v4 -- K at 2 bits, V still i4 = (B2,B4) = "
              "13,312 B, the pair `kv_bit_budget.h:301-312` names as unexpressible");
static_assert(cell_pair_bytes(CellPair{CellMode::E8_2bit, CellMode::E8_2bit}) ==
                  e8_kv_pair_bytes(E8KvPlaneFormat::B2, E8KvPlaneFormat::B2),
              "F1260: (e8-2bit, e8-2bit) IS THE OWNER'S FLOOR -- 9,216 B, the diagonal pair");
// AND THE TWO PRICES THE COORDINATOR'S CORRECTION SEPARATES: 13,312 and 9,216 are TWO DIFFERENT
// PAIRS, and the assertion below is the whole difference between them: a 2-bit ladder that has to
// leave V on the family's 128 B plate costs 13,312 per cell; one where V narrows too costs 9,216.
// ⚠ AND THE CHEAPER ONE IS NOT DEPLOYABLE TODAY (see `cell_pair_read_side` below): the pair that is
// PRICED and the pair that can be READ are different answers, and both are printed.
static_assert(cell_pair_bytes(CellPair{CellMode::E8_2bit, CellMode::Rk4v4}) -
                      cell_pair_bytes(CellPair{CellMode::E8_2bit, CellMode::E8_2bit}) ==
                  e8_kv_plane_format_bytes(E8KvPlaneFormat::B4) -
                      e8_kv_plane_format_bytes(E8KvPlaneFormat::B2),
              "F1260: the 13,312-vs-9,216 gap IS the V plane's own narrowing (8,704 -> 4,608). So "
              "the cheap 2-bit cell is bought by narrowing V, and there is no other way to buy it");

// ⭐ (3) REALIZABILITY, IN TYPED TERMS, MIRRORING THE TREE'S OWN ORACLE. `kv_kv_bits.h`'s
// `kv_e8_plane_pair_realizable(k_format, v_format)` is the family's own answer in STRINGS:
//   "(e8-lattice, i4)=rk4v4, (e8-lattice-3, i4)=rk3v4, (e8-lattice-2, i4)=rk2v4,
//    (e8-lattice-3, e8-lattice-3), (e8-lattice-2, e8-lattice-2)=THE FLOOR,
//    (e8-lattice-3, e8-lattice-2), (e8-lattice-2, e8-lattice-3)"
// and it refuses a 4-bit K plane carrying a lattice V ("there is no 4-bit lattice"). This file
// cannot include that one (kv_kv_bits.h reaches this header through kv_bit_budget.h, so an include
// here would close a cycle), so the rule is re-expressed over ROWS and the MIRROR IS PINNED IN A TU
// THAT CAN SEE BOTH (`dl/kvplanar/out/probe.cpp`) -- the tree's own discipline for a pin whose two
// headers cannot meet in one file.
[[nodiscard]] constexpr bool cell_pair_realizable(CellPair p) noexcept {
    E8KvPlaneFormat kf{};
    E8KvPlaneFormat vf{};
    // A ROW PAIRED WITH ITSELF IS ALWAYS REALIZABLE, and that is the whole of the non-e8 answer:
    // `int8`/`bf16`/`nvfp4` are the two-plane-SYMMETRIC rows (the ladder's own note: "K and V are
    // the same tier"), so `(m, m)` is exactly the plane pair the engine builds for them.
    if (p.k == p.v) { return true; }
    // ⚠ AND EVERY CROSS-FAMILY PAIR IS REFUSED, IN BOTH DIRECTIONS. `(int8, nvfp4)` is a pair NO
    // TIER BUILDS: the engine's V for an int8 K is int8 (`kv_bits_tier_v_format` returns the layer
    // dtype for every non-NVFP4 dtype), and an `nvfp4` K's V is ISO4E or E2M1 -- never int8. A
    // lenient default here would have let the walk spend a pair the engine would refuse one layer
    // down, and this function's first draft DID exactly that (MEASURED: the mixed selection
    // `k:lattice,v:priciest` spent 32 cells on `(int8, nvfp4)`, which no kernel can build). The
    // refusal is by construction now, not by luck.
    const bool k_e8 = cell_row_e8_plane(p.k, kf);
    const bool v_e8 = cell_row_e8_plane(p.v, vf);
    if (!k_e8 || !v_e8) { return false; }
    // BOTH in the family: the family's own table above, as arithmetic. A 4-bit K plane (B4) may not
    // take a lattice V; a lattice K may take i4 (B4) or a lattice V that is not wider than itself.
    const std::int32_t kb = e8_kv_plane_format_bits(kf);
    const std::int32_t vb = e8_kv_plane_format_bits(vf);
    if (kb == 4) { return vb == 4; }
    return vb == 4 || vb <= kb;
}
static_assert(cell_pair_realizable(CellPair{CellMode::Rk4v4, CellMode::Rk4v4}) &&
                  cell_pair_realizable(CellPair{CellMode::Rk3v4, CellMode::Rk4v4}) &&
                  cell_pair_realizable(CellPair{CellMode::E8_2bit, CellMode::Rk4v4}) &&
                  cell_pair_realizable(CellPair{CellMode::Rk3v4, CellMode::E8_2bit}) &&
                  cell_pair_realizable(CellPair{CellMode::E8_2bit, CellMode::E8_2bit}) &&
                  !cell_pair_realizable(CellPair{CellMode::Rk4v4, CellMode::Rk3v4}) &&
                  !cell_pair_realizable(CellPair{CellMode::Rk4v4, CellMode::E8_2bit}),
              "F1260: the family's seven pairs, and the two refusals the oracle names -- a 4-bit K "
              "plane may not carry a lattice V, and a lattice K plane may not carry a 4-bit V "
              "(there is no 4-bit lattice)");
// AND THE CROSS-FAMILY REFUSALS, PINNED SO THE LENIENT DEFAULT CANNOT COME BACK: the rows an
// existing arm could actually reach are exactly the ones whose pairs are symmetric, and a MIXED row
// pair is not a tier any tier table names.
static_assert(cell_pair_realizable(CellPair{CellMode::Int8, CellMode::Int8}) &&
                  cell_pair_realizable(CellPair{CellMode::Nvfp4, CellMode::Nvfp4}) &&
                  cell_pair_realizable(CellPair{CellMode::Bf16, CellMode::Bf16}) &&
                  !cell_pair_realizable(CellPair{CellMode::Int8, CellMode::Nvfp4}) &&
                  !cell_pair_realizable(CellPair{CellMode::Nvfp4, CellMode::Int8}) &&
                  !cell_pair_realizable(CellPair{CellMode::Int8, CellMode::Rk4v4}) &&
                  !cell_pair_realizable(CellPair{CellMode::Bf16, CellMode::Int8}),
              "F1260: the symmetric rows are realizable and EVERY cross-family pair is refused -- "
              "the engine's V for a non-NVFP4 dtype IS the layer dtype, and its V for NVFP4 is "
              "ISO4E/E2M1, so `(int8, nvfp4)` is a pair no tier builds");

// ⭐ (4) PRICED IS NOT THE SAME AS READABLE. A pair whose every plane is read today is `Shipped`;
// a pair with a narrow lattice plane is PRICED and RESERVED, because the family's decoder for that
// plate (`e8_kv_lattice_decode_group<3>/<2>`) has no caller (see `e8_kv_plane_read_side`,
// `kv_e8_width.h`, and the refusal text at `kv_storage_dtype.h:113-117`). A caller that reported a
// Reserved pair as available would be quoting a lattice no reader can read.
[[nodiscard]] constexpr E8KvPlaneReadSide cell_pair_read_side(CellPair p) noexcept {
    E8KvPlaneFormat kf{};
    E8KvPlaneFormat vf{};
    const E8KvPlaneReadSide ks = cell_row_e8_plane(p.k, kf)
                                     ? e8_kv_plane_read_side(E8KvPlane::K, kf)
                                     : E8KvPlaneReadSide::Shipped;
    const E8KvPlaneReadSide vs = cell_row_e8_plane(p.v, vf)
                                     ? e8_kv_plane_read_side(E8KvPlane::V, vf)
                                     : E8KvPlaneReadSide::Shipped;
    return (ks == E8KvPlaneReadSide::Reserved || vs == E8KvPlaneReadSide::Reserved)
               ? E8KvPlaneReadSide::Reserved
               : E8KvPlaneReadSide::Shipped;
}
static_assert(cell_pair_read_side(CellPair{CellMode::Rk4v4, CellMode::Rk4v4}) ==
                      E8KvPlaneReadSide::Shipped &&
                  cell_pair_read_side(CellPair{CellMode::E8_2bit, CellMode::Rk4v4}) ==
                      E8KvPlaneReadSide::Reserved &&
                  cell_pair_read_side(CellPair{CellMode::E8_2bit, CellMode::E8_2bit}) ==
                      E8KvPlaneReadSide::Reserved,
              "F1260: rk4v4 is deployable, and BOTH 2-bit pairs (13,312 and 9,216) are RESERVED -- "
              "the difference between the two prices is a plane, and neither has a reader yet");

// =============================================================================================
// [F1239 kvrate] ⭐ THE RATE RULER, DERIVED FROM THE TABLE ABOVE -- ONE DATUM, SIX PINS.
//
// WHY A RATE AND NOT A TOTAL. An absolute byte budget is a statement about ONE pass, and this
// engine's candidate population GROWS INSIDE ONE PREFILL: MEASURED on the new pair, the population
// walked 736 -> 1248 -> 1760 -> ... -> 10,208 cells over 24 passes, so against ONE fixed budget of
// 36,700,160 B the engine read `min_possible_bytes` 27,131,904 -> 46,006,272 -> 64,880,640 and
// answered `sat_yes=1 / sat_NO=23` -- satisfiable on pass 1 ONLY. A RATE ("bits per element") is the
// same statement on EVERY pass, because the population it is a rate OVER is the pass's own: the
// ceiling scales with W and the floor scales with W, so W cancels out of the satisfiability
// question and the answer stops depending on which pass asked.
//
// THE ONE DATUM, AND WHY IT IS NOT THE THING THE OWNER FORBADE. `plane_bytes` is LINEAR in b/el and
// the slope is the plane GEOMETRY: one head-page of 16,384 elements priced on a 0.01-bit grid over
// 8 bits per byte. His objection is to numbers that STEER a decision
// (「不应该写死任何数字来干预引擎决策」); a UNIT CONVERSION steers nothing -- it is the ruler the
// operator's own rate is read against, and without one a rate is not a quantity. What must hold is
// 少写死多适配, and it does, as a TESTABLE PROPERTY: ⚠ NO PER-MODEL NUMBER APPEARS IN THIS BLOCK. The
// target is the operator's rate; the population's block count and the per-block charge are both
// computed from the running stack at runtime, so ADDING A MODEL TOUCHES NO LINE HERE AND NO KV
// CONSTANT.
//
// AND THE DATUM IS PINNED AGAINST THE TABLE, so it cannot go stale: six rows of `kCellRungs` are
// already asserted in b/el (16.00 / 8.25 / 4.50 / 4.25 / 3.25 / 2.25), and the six
// `static_assert`s below check this conversion against each of those plane prices TO THE BYTE.
// Re-pricing a row without re-deriving the conversion is a BUILD FAILURE -- which is the point: a
// stale conversion would silently re-denominate every rate arm in the tree.
inline constexpr std::int32_t kKvCellBitsPerElementScale = 100;   // the 0.01-bit grid
inline constexpr std::int32_t kKvCellElementsPerHeadPage   = 16384;  // head_dim 256 x page_tokens 64
inline constexpr std::int32_t kKvCellBitsPerByte           = 8;

// A b/el target -> the ONE-PLANE byte price the descent's whole currency is denominated in.
[[nodiscard]] constexpr std::int32_t
kv_cell_plane_bytes_at_bits_x100(std::int32_t bits_x100) noexcept {
    return static_cast<std::int32_t>(
        (static_cast<std::int64_t>(bits_x100) * kKvCellElementsPerHeadPage) /
        (kKvCellBitsPerByte * kKvCellBitsPerElementScale));
}

// The same conversion the other way, so a rung's own b/el is a READING and never re-typed.
[[nodiscard]] constexpr std::int32_t
kv_cell_bits_x100_of_plane_bytes(std::int32_t plane_bytes) noexcept {
    return static_cast<std::int32_t>(
        (static_cast<std::int64_t>(plane_bytes) * kKvCellBitsPerByte *
         kKvCellBitsPerElementScale) /
        kKvCellElementsPerHeadPage);
}

// ⭐ THE FLOOR'S OWN RATE, AS A READING -- the cheapest priced rung of this very table, in b/el.
// This is the number a budget must be at or above, in the SAME currency an operator writes, so a
// refusal can NAME the gap ("the budget is at 1.27 b/el and this population's floor is at 2.25")
// instead of only saying "unsatisfiable". It is derived from the table, so a new rung cheap enough
// to become the floor MOVES THIS READING BY ITSELF -- no constant to update.
[[nodiscard]] constexpr std::int32_t kv_cell_cheapest_bits_x100() noexcept {
    std::int32_t cheapest = 0;
    bool any = false;
    for (const CellRung& rung : kCellRungs) {
        if (rung.price_state != PriceState::Priced) { continue; }
        if (!any || rung.plane_bytes < cheapest) { cheapest = rung.plane_bytes; any = true; }
    }
    return any ? kv_cell_bits_x100_of_plane_bytes(cheapest) : 0;
}

static_assert(kv_cell_plane_bytes_at_bits_x100(1600) == 32768, "bf16 16.00 b/el -> 32768 B plane");
static_assert(kv_cell_plane_bytes_at_bits_x100(825)  == 16896, "int8  8.25 b/el -> 16896 B plane");
static_assert(kv_cell_plane_bytes_at_bits_x100(450)  ==  9216, "nvfp4 4.50 b/el ->  9216 B plane");
static_assert(kv_cell_plane_bytes_at_bits_x100(425)  ==  8704, "rk4v4 4.25 b/el ->  8704 B plane");
static_assert(kv_cell_plane_bytes_at_bits_x100(325)  ==  6656, "rk3v4 3.25 b/el ->  6656 B plane");
static_assert(kv_cell_plane_bytes_at_bits_x100(225)  ==  4608, "e8-2bit 2.25 b/el -> 4608 B plane");
static_assert(kv_cell_plane_bytes_at_bits_x100(825) == cell_rung(CellMode::Int8).plane_bytes &&
                  kv_cell_plane_bytes_at_bits_x100(225) ==
                      cell_rung(CellMode::E8_2bit).plane_bytes &&
                  kv_cell_plane_bytes_at_bits_x100(1600) == cell_rung(CellMode::Bf16).plane_bytes,
              "THE CONVERSION AND THE TABLE'S OWN assertED PRICES MUST BE ONE NUMBER. The two "
              "columns above and the `static_assert`s on `kCellRungs` are the same claim spelled "
              "twice; if they ever disagree, one of them is a lie and the rate arms would be "
              "denominated in neither.");
static_assert(kv_cell_cheapest_bits_x100() == 225,
              "THE FLOOR IS e8-2bit at 2.25 b/el -- read off the table, not typed in the budget. "
              "If a future rung becomes cheaper this assert fires, and the correct fix is to "
              "re-read which rung is the floor, NOT to change this constant.");

// ---------------------------------------------------------------------------------------------
// 4. `F_(l,b)`: THE CELL'S ALLOWED SET. A SET, and the order says so.
// ---------------------------------------------------------------------------------------------
//
// The tree already has a per-layer allowed set and its own words for it, `kv_adapt_solver.h`:
//     "* Decision variable is per-layer: T(l) in F_l, F_l an ARBITRARY subset."
// `F_l` is per-LAYER. The order's unit is per-CELL. The relation between them is DESIGN, not a
// reading, and it is stated rather than implied:
//
//     F_(l,b)  :=  F_l INTERSECT { modes whose record the pool can hold for this layer }
//
// Today the second factor is "everything" for every layer that has a pack arm, because the pool
// reserves ONE record per (page, head, plane) at the layer's own stride -- G3 / GAP-SPAN: a cell
// whose F contains two modes with DIFFERENT records needs the pool to answer a question it has
// no place for ("which stride does this layer's slot take"), and THAT ANSWER IS NOT THIS FILE'S.
// So `cell_allowed_set` below is total by default and the narrowing is a caller's act.
struct CellAllowedSet {
    std::uint32_t mask = 0;

    [[nodiscard]] constexpr bool has(CellMode mode) const noexcept {
        return (mask & (1U << static_cast<std::uint32_t>(mode))) != 0U;
    }
    constexpr void add(CellMode mode) noexcept {
        mask |= (1U << static_cast<std::uint32_t>(mode));
    }
    constexpr void remove(CellMode mode) noexcept {
        mask &= ~(1U << static_cast<std::uint32_t>(mode));
    }
    [[nodiscard]] constexpr std::uint32_t count() const noexcept {
        std::uint32_t n = 0;
        for (std::uint32_t b = 0; b < kCellModeCount; ++b) {
            n += ((mask >> b) & 1U);
        }
        return n;
    }
};

[[nodiscard]] constexpr CellAllowedSet cell_all_modes() noexcept {
    CellAllowedSet set{};
    set.mask = (1U << kCellModeCount) - 1U;
    return set;
}

// `F_l` as a set: the intersection the relation above names. Both halves are the caller's, and
// this is the one place the two become one set.
[[nodiscard]] constexpr CellAllowedSet cell_allowed_set(CellAllowedSet layer_set,
                                                        CellAllowedSet pool_supported) noexcept {
    return CellAllowedSet{layer_set.mask & pool_supported.mask};
}

// ---------------------------------------------------------------------------------------------
// 5. THE DEFAULT TIER: THE ONE THING IN THIS FILE THAT IS **NOT** DERIVED FROM BYTES.
// ---------------------------------------------------------------------------------------------
//
// The order: "at the START they all default to int8 or bf16". Bytes cannot choose between them:
// `bytes(int8) == bytes(bf16) == 9232` (row 2 and row 3 of the table above, both READ IN
// SOURCE). So the default is a PREFERENCE ORDER over the set, and this is where the order's own
// words become the tie-break:
//
//     default(S)  :=  Int8   if Int8 in S
//                     Bf16   else if Bf16 in S
//                     the priciest priced mode in S   else
//                     NoDefault                       else   (a NAMED refusal, not a guess)
//
// The middle transfer is the whole point: a cell that cannot be int8 but can be bf16 starts at
// bf16, which is what "int8 or bf16" means. Falling through to "priciest priced" is the only
// derivation the byte table can support and is stated as such.
enum class DefaultTier : std::uint8_t {
    Int8 = 0,      // the order's first default
    Bf16 = 1,      // the order's second default
    Derived = 2,   // not int8 and not bf16 in this set: the priciest priced rung in F-(l,b)
    NoDefault = 3, // no priced rung in F-(l,b) at all: named, and the cell cannot start
};

struct CellDefault {
    CellMode mode = CellMode::Int8;   // meaningful only when state != NoDefault
    DefaultTier state = DefaultTier::Int8;
};

[[nodiscard]] constexpr CellDefault cell_default_tier(CellAllowedSet set) noexcept {
    if (set.has(CellMode::Int8)) { return CellDefault{CellMode::Int8, DefaultTier::Int8}; }
    if (set.has(CellMode::Bf16)) { return CellDefault{CellMode::Bf16, DefaultTier::Bf16}; }
    CellMode best = CellMode::Nvfp4;
    bool any = false;
    for (const CellRung& rung : kCellRungs) {
        if (!set.has(rung.mode) || rung.price_state != PriceState::Priced) { continue; }
        // [F1231, F-5 site 1] THE RULER. This branch picks "the priciest priced rung in S" and it
        // used to ask the question in `record_bytes` -- the COLD SLOT -- while every other order in
        // this file (sites 1 and 2 of the derived order, the charge, the step's delta) is on
        // `plane_bytes`. The two rulers AGREE on today's table and DISAGREE in general, and the
        // disagreement is an INVERSION, not a tie-break: for `int8` vs `nvfp4` the record ruler
        // says int8 is cheaper (9,232 < 9,632) while the plane ruler says int8 is pricier
        // (16,896 > 9,216). So a future row could have moved the START rung silently -- and the
        // start is R1 ("the start state is the user's declared high bit"), which is not a place a
        // unit mix-up may live. Corrected to the plane ruler: the column the walk prices and the
        // stop condition sums.
        if (!any || rung.plane_bytes > cell_rung(best).plane_bytes) {
            best = rung.mode;
            any = true;
        }
    }
    if (!any) { return CellDefault{CellMode::Int8, DefaultTier::NoDefault}; }
    return CellDefault{best, DefaultTier::Derived};
}

// ---------------------------------------------------------------------------------------------
// 6. THE ORDER: DERIVED FROM BYTES, PER CELL. THIS IS THE "ONE RUNG AT A TIME" STEP.
// ---------------------------------------------------------------------------------------------
//
// There is no rung NUMBER here and no ladder index. The step is a function from a cell's current
// mode to the cell's next mode, and its definition is the ORDER's definition of a rung:
//
//     next(S, f)  :=  argmax_{ f' in S : bytes(f') < bytes(f) } bytes(f')
//
// i.e. "the rung below you is the priciest rung in your own set that is strictly cheaper than
// you". This is why the order needs no global scalar: it uses only DIFFERENCES of `bytes`, and
// `bytes` is a record width the engine already has. `bits_x100 = 50*w + 225` is NOT used and
// could not be: it is one-dimensional over the e8 family only.
//
// THE FOUR WAYS A STEP REFUSES, ALL NAMED. A refusal is a RETURN VALUE, never a fallback --
// `kv_block_budget_stage.h`'s own rule ("never silently collapse") applied to a step.
enum class StepRefusal : std::uint8_t {
    None                = 0,  // the step is legal and strictly cheaper
    AlreadyAtFloor      = 1,  // nothing in F-(l,b) is cheaper than the cell's current mode
    RungBytesUnknown    = 2,  // a cheaper-by-reputation rung exists but is Unpriced ==> G6
    NoByteCurrency      = 3,  // the next rung is a different CODEC at the SAME record ==> G7
    CurrentModeNotInSet = 4,  // the cell's mode is outside its own allowed set (a caller bug)
    CurrentModeUnpriced = 5,  // the cell is already on an Unpriced rung: no order is defined
};

[[nodiscard]] constexpr const char* step_refusal_name(StepRefusal refusal) noexcept {
    switch (refusal) {
    case StepRefusal::AlreadyAtFloor: return "already-at-floor";
    case StepRefusal::RungBytesUnknown: return "rung-bytes-unknown(G6)";
    case StepRefusal::NoByteCurrency: return "no-byte-currency(G7)";
    case StepRefusal::CurrentModeNotInSet: return "current-mode-not-in-set";
    case StepRefusal::CurrentModeUnpriced: return "current-mode-unpriced";
    case StepRefusal::None: break;
    }
    return "none";
}

struct CellStep {
    CellMode to = CellMode::Int8;       // meaningful only when refusal == None
    StepRefusal refusal = StepRefusal::None;
    std::int32_t bytes_saved = 0;       // > 0 exactly when refusal == None
    // THE REASONS, ALL OF THEM. `refusal` above is the PRECEDENCE answer (one value, so a caller
    // can switch on it), and these two are the FACTS behind it. They are kept separate on
    // purpose: a cell whose set contains an unpriced rung AND whose priced neighbour writes the
    // same record is blocked by G6 and G7 AT ONCE, and a single refusal value would hide one of
    // the two behind the other. The walk counts both, so the accounting line cannot under-report
    // either gap. MEASURED on the shipped stack's full mode set: 32 of 32 cells carry BOTH.
    bool blocked_by_unpriced_rung = false;   // G6: a cheaper-by-reputation rung has no byte cost
    bool blocked_by_same_record = false;     // G7 reading: some priced rung writes the same RECORD
    // [F1231, F-5 site 2] G7's REFUSAL FACT, ON THE RULER THE STEP IS PRICED IN. Appended with a
    // default so every existing aggregate construction of `CellStep` still compiles unchanged, and
    // so both facts travel together: the record one is the CODEC question, the plane one is the
    // CURRENCY question, and only the second may refuse a step.
    bool blocked_by_same_plane = false;      // G7 refusal: some priced rung has the same PLANE bytes
    [[nodiscard]] constexpr bool legal() const noexcept { return refusal == StepRefusal::None; }
};

[[nodiscard]] constexpr CellStep cell_next_cheaper(CellAllowedSet set, CellMode from) noexcept {
    const CellRung& here = cell_rung(from);
    if (here.price_state != PriceState::Priced) {
        return CellStep{from, StepRefusal::CurrentModeUnpriced, 0, false, false};
    }
    if (!set.has(from)) {
        return CellStep{from, StepRefusal::CurrentModeNotInSet, 0, false, false};
    }

    CellMode best = from;
    bool found = false;
    bool blocked_by_unpriced = false;
    bool blocked_by_same_record = false;
    // [F1231, F-5 site 2] THE PRE-PLANE PLURAL FACT, AND IT IS THE ONE THE REFUSAL USES. G7's
    // question is "does the only other priced rung in this set save a byte?", and "a byte" is a
    // PLANE byte: that is what `cell_next_cheaper`'s delta is denominated in (site 3) and what the
    // driver's `bytes_saved <= 0` guard refuses on. Asking it in `record_bytes` made the refusal a
    // statement about the cold SLOT while the step it refuses is priced in resident PLANES, so the
    // two facts can disagree in both directions once `F_(l,b)` is narrowed. Both are computed and
    // both travel; the REFUSAL is the plane one.
    bool blocked_by_same_plane = false;
    for (const CellRung& candidate : kCellRungs) {
        if (candidate.mode == from || !set.has(candidate.mode)) { continue; }
        // G6 LANDS HERE, AND IT IS A REFUSAL, NOT A SKIP. A rung the tree has no byte cost for
        // cannot be entered: entering it would make `bytes_saved` unknown, and the walk's whole
        // stop condition is a byte total. The refusal is recorded by NAME so the reason the
        // walk halted above the floor is readable, and `cell_mode_is_priced` is the only test
        // that decides it. The order's "in the END they go to e8 2bit" therefore stops at this
        // line today, and the line says so.
        if (candidate.price_state != PriceState::Priced) {
            blocked_by_unpriced = true;
            continue;
        }
        // ① AND ② READ `plane_bytes`, NOT `record_bytes`. [planes 2026-09-29] THE ORDER IS DERIVED
        // FROM THE RESIDENT PLANE, because that is the memory a demotion actually gives back;
        // `record_bytes` is the COLD SLOT the pool stages, and it separates only nvfp4 (9632) from
        // everything else (9232) -- which is why the record column alone could only ever express
        // "every layer is the raw slot" (see `cell_vector_charge`'s note).
        if (candidate.plane_bytes >= here.plane_bytes) { continue; }                    // ← ①
        if (!found || candidate.plane_bytes > cell_rung(best).plane_bytes) {            // ← ②
            best = candidate.mode;
            found = true;
        }
    }
    // G7 LANDS HERE. If the only other priced rungs in the set write the SAME record, the step
    // exists as a CODEC change and does NOT save a byte. That is a real state of this tree, and
    // it is not an error in the caller -- it is the conflation `decoder_state.cpp`'s three
    // static_asserts prove. It must NOT be taken, because a step with zero currency cannot reach
    // a byte budget and would make the walk's termination argument false.
    for (const CellRung& candidate : kCellRungs) {
        if (candidate.mode == from || !set.has(candidate.mode)) { continue; }
        if (candidate.price_state != PriceState::Priced) { continue; }
        // BOTH READINGS, ONE PER RULER, AND THE NAMES SAY WHICH IS WHICH so a later round cannot
        // re-conflate them. The record reading is kept (it is the historical one and it is still
        // printed); the plane fact is what the refusal below is made on.
        if (candidate.record_bytes == here.record_bytes) { blocked_by_same_record = true; }
        if (candidate.plane_bytes == here.plane_bytes) { blocked_by_same_plane = true; }
    }
    // PRECEDENCE: G6 first, because an unpriced rung makes the byte ORDER itself unknown --
    // whether `rk4v4` saves anything is a question one cannot ask while "the next rung" might
    // have been `e8-2bit` all along. The two facts are BOTH returned either way.
    //
    // [planes 2026-09-29] ⭐ AND G6 IS A STOP, NOT A SKIP -- THIS TEST SITS ABOVE THE `found`
    // RETURN ON PURPOSE. Before this landing the `found` return came first, so a strictly cheaper
    // PRICED rung was handed back while an unpriced rung was merely counted: the loop SKIPPED the
    // unknown rung and spent the one below it, which is exactly the forbidden step (it treats an
    // unknown price as if it were an infinite one). The master's ruling is "if the next rung has no
    // byte cost today, STOP AT THIS RUNG, do not skip it" -- so the refusal is raised here, before
    // any step can be returned, and `from` is the answer. The G7 loop above is deliberately run
    // before this point (and no longer only when `found` is false) so that BOTH facts still travel
    // together on the G6 refusal, as this comment block has always promised.
    if (blocked_by_unpriced) {
        return CellStep{from, StepRefusal::RungBytesUnknown, 0, true, blocked_by_same_record};
    }
    if (found) {
        // ③ THE CURRENCY IS `plane_bytes` -- THE RULER ① AND ② ORDER ON, AND THE RULER THE STOP
        // CONDITION SUMS. [F1209 2026-09-29] THIS SITE WAS THE LAST THIRD OF THE SPLIT RULER.
        //
        // ⚠ WHAT THIS SITE USED TO BE, KEPT AS HISTORY AND NOW SUPERSEDED. It returned
        // `here.record_bytes - cell_rung(best).record_bytes`, and the comment here argued, in these
        // words, that the record ruler was deliberate:
        //   "The stop condition is `plan.total_bytes <= budget`, whose total is the ENGINE'S
        //    CHARGE. That charge is `cell_vector_charge` = `2 * kv_heads * cell_record_bytes(...)`
        //    (9232 / 9632), i.e. a RECORD total, and `kv_block_descent.h:434` subtracts this delta
        //    straight out of it."
        // THAT PREMISE IS DEAD. The cut at [planes 2026-09-29] had already changed the charge: by
        // the time this comment was written, `cell_vector_charge` summed `rung.plane_bytes` (the
        // `per_layer_pair += rung.plane_bytes;` line in section 8 below), NOT `cell_record_bytes`.
        // So the "RECORD total" the record ruler was justified by no longer existed -- the comment
        // justified a behaviour using a state of the code it was standing on, and it did not notice
        // the charge had moved.
        // THE SAME COMMENT'S SECOND CLAIM IS ALSO SUPERSEDED: it said that making the delta buy
        // plane bytes "is a change to the CHARGE (`cell_vector_charge`) and through it to
        // `bytes_per_block = 1,181,696` ... a separate, separately-authorised change, not this
        // landing". The charge was ALREADY that change -- it is what the cut did -- so this delta
        // needs no separate authorisation; it is the missing half of a change already made. Its
        // line citation has moved too: the subtraction is not at `:434` any more, it is at
        // `kv_block_descent.h:485` (`plan.total_bytes -= ladder_step_total_delta(step, kv_heads);`).
        // NO REASONING WAS DELETED: the superseded argument is quoted above, marked, and left here.
        //
        // AND THE MEASURED CONSEQUENCE OF THE OLD FORM, WHICH IS WHY THIS IS A CLOSURE AND NOT A
        // TIDY-UP: from `int8` the plane-cheaper rung the order reaches is `nvfp4`, whose RECORD is
        // 9,632, so the step read `9,232 - 9,632 = -400`; the driver's `bytes_saved <= 0` guard
        // (kv_block_descent.h) refused it, and EVERY cell on EVERY budget was refused --
        // `refused = cells`, `steps = 0`, `saved_bytes = 0`. MEASURED twice and independently:
        // `blob_F1192.md` 33.4 and `blob_F1201.md` (measured on the card, `refused=8064=cells`).
        // The pre-image step is PINNED in `kv_tier_ladder.h`'s static_asserts so a later round can
        // read what changed rather than trust this paragraph.
        //
        // ⚠⚠ AND THE DELTA IS A ONE-PLANE, ONE-LAYER NUMBER: THE CALLER MUST SCALE IT.
        // The total it is subtracted from is `cell_vector_charge = 2 * kv_heads * sum_l
        // plane_bytes(V[l])` (section 8's `charge.bytes` line), so ONE cell's demotion changes that
        // total by `2 * kv_heads * (this delta)`. A driver that subtracts this number directly is
        // SHORT BY THE FACTOR `2 * kv_heads` -- 8 on the shipped 4-head stack. That is
        // DEVELOPMENT-3, it is derived with its arithmetic at `block_descent_plan`'s subtraction
        // (`kv_block_descent.h`), and `ladder_step_total_delta(step, kv_heads)` is the ONE spelling
        // of the factor. This function returns the per-plane number because a per-plane number is
        // what the order compares; the factor belongs to whoever owns the total.
        // THE STEP IS STILL STRICTLY POSITIVE FOR EVERY RUNG THE ORDER CAN REACH: the `plane_bytes`
        // column of `kCellRungs` is strictly decreasing (section 3b's static_asserts), so the
        // driver's termination argument -- every accepted step strictly decreases the quantity the
        // stop reads -- now holds ON THE STOP'S OWN RULER, which is exactly what it could not do
        // while the order and the delta were in different units.
        return CellStep{best, StepRefusal::None, here.plane_bytes - cell_rung(best).plane_bytes,
                        false, false};
    }
    if (blocked_by_same_plane) {
        // THE REFUSAL'S CURRENCY IS THE PLANE RULER; the record reading is carried beside it for a
        // reader that wants the codec answer too. `StepRefusal::NoByteCurrency` therefore now means
        // exactly "the reachable rungs save zero of the bytes the step is priced in".
        return CellStep{from, StepRefusal::NoByteCurrency, 0, false, blocked_by_same_record, true};
    }
    return CellStep{from, StepRefusal::AlreadyAtFloor, 0, false, false};
}

// ---------------------------------------------------------------------------------------------
// 7b. THE STAGE'S CHARGE INPUT, AS A TABLE. Three fields, and the third is what keeps a
// partially planned pass LEGAL: a block outside the covered range charges the scalar.
// ---------------------------------------------------------------------------------------------
//
// THE ORIGIN IS A PAGE NUMBER because the stage's `decide(block)` is handed a PAGE number -- the
// engine's own identity, quoted in `kv_block_budget_stage.h`: "A page's block index IS the page
// number under the engine's own identity". The table therefore travels WITH an origin rather than
// assuming index 0, so the plan can cover any window.
struct BlockChargeTable {
    const std::int64_t* bytes = nullptr;
    std::uint32_t origin = 0;
    std::uint32_t count = 0;

    [[nodiscard]] bool covers(std::uint32_t block) const noexcept {
        return bytes != nullptr && block >= origin && (block - origin) < count;
    }
    // THE FALLBACK IS THE SCALAR'S JOB, NOT THIS ONE'S. Returning 0 here would price an
    // unplanned block as free, which is the one way this table could make the ceiling admit
    // something it must refuse. So there is no fallback value here: the caller tests `covers()`.
};

// ---------------------------------------------------------------------------------------------
// 8. `V(b)`: THE PER-LAYER VECTOR, AND IT IS THE SIDECAR'S PAYLOAD, NOT THE COLD RECORD'S.
// ---------------------------------------------------------------------------------------------
//
// THE CARRIER IS THE SIDECAR AND ONLY THE SIDECAR. `kv_recall_block.h` has the precedent verbatim
// (its `shard_rank`/`shard_world` comment): "The fields sit at the END of the struct and default
// to the identity world ... Appended rather than inserted so no existing aggregate initialisation
// or member offset moves." The cold slot record must NOT be touched, and the same header says
// why, verbatim: "The cold slot record has NO spare space. Its stride is a function of the page
// ... every byte of both codecs is spoken for". So the vector lives on the sidecar, it is
// APPENDED, and nothing about the payload moves.
//
// THE AXIS SIZE. `full_attention_layers(total_layers)` = `total_layers / kHybridAttentionInterval`
// with `kHybridAttentionInterval = 4` (targets/qwen3_6/export/ninfer/targets/qwen3_6/
// hybrid_topology.h:7,13-15), so the shipped 64-layer stack has 16. That is the 16 bytes this
// append costs. The pin that this constant IS `full_attention_layers(64)` belongs in a test TU
// where both headers can be included -- the SAME caveat `kv_recall_block.h` already writes about
// `kRecallBlockTokens` ("this header cannot include that ... so the pin lives in
// tests/test_kv_recall_block.cpp where both can be included"). This header is in `product/` and
// must not reach into `targets/`, so it carries the constant and not the include.
inline constexpr std::uint32_t kBlockVectorLayerAxis = 16;

// THE VECTOR. Default-constructed it is the ORDER'S STARTING POINT: every layer at `int8`.
// AND THAT IS NOT A COINCIDENCE WORTH PASSING OVER: today's engine charge is the literal
// `rans_layers = 0` at program_impl.h byte ~765,805, i.e. `2 * kv_heads * layers * 9232` -- the
// statement "every layer is the raw slot". So THE DEFAULT VECTOR **IS** THE PRE-IMAGE CHARGE,
// and the "OFF is byte-for-byte" property stops being a switch and becomes a theorem about this
// default. See `cell_vector_charge()`'s note, and `blob_F1172.md` section 2 for the pin.
struct BlockVector {
    std::array<CellMode, kBlockVectorLayerAxis> mode{};
    std::uint32_t layers = kBlockVectorLayerAxis;
    // =========================================================================================
    // [F1260 kvplanar] ⭐ THE SECOND COORDINATE, AND WHY IT IS APPENDED HERE.
    // `mode` IS THE K PLANE, keeps its name, its type and its position, and every reader outside
    // this file that asks `vector.mode[layer]` is reading the K plane -- which is what it was always
    // reading, because until this landing the two planes could only be one value. The V plane is a
    // SEPARATE array so that:
    //   * a VALUE-INITIALISED vector is the DIAGONAL (pre-image) state BY CONSTRUCTION: both arrays
    //     default to `CellMode::Int8` (the order's own "at the START they all default to int8"), so
    //     the "OFF is byte-for-byte the pre-image" theorem stays a theorem about a default object
    //     rather than a switch somebody has to remember to leave alone;
    //   * `set(layer, m)` KEEPS ITS PRE-IMAGE MEANING -- a diagonal write, both planes to `m` -- so
    //     every existing caller (`kv_tier_ladder.h:581/635`, `kv_descent_control.h:437/464`,
    //     `kv_cell_alloc_solve.h`) keeps producing exactly the vectors it produced before and the
    //     charge it reads back is exactly the same number;
    //   * a TWO-SIDED write is a new, explicitly named act (`set_pair`).
    // ⚠ THE INVARIANT THE TWO ARRAYS SHARE: `v_plane[l] == mode[l]` unless a pair step put them
    // apart. It is not enforced by a flag -- the only writer that can break it is `set_pair`, which
    // is this file's own new call site -- and `cell_vector_charge` above reads whichever the array
    // holds, so a vector that never met a pair step charges the pre-image number by arithmetic.
    std::array<CellMode, kBlockVectorLayerAxis> v_plane{};

    [[nodiscard]] constexpr CellMode at(std::uint32_t layer) const noexcept {
        return layer < layers ? mode[layer] : CellMode::Int8;
    }
    [[nodiscard]] constexpr CellMode v_at(std::uint32_t layer) const noexcept {
        return layer < layers ? v_plane[layer] : CellMode::Int8;
    }
    [[nodiscard]] constexpr CellPair pair_at(std::uint32_t layer) const noexcept {
        return CellPair{at(layer), v_at(layer)};
    }
    // THE PRE-IMAGE WRITE: ONE MODE, BOTH PLANES. Every existing caller keeps this meaning.
    constexpr void set(std::uint32_t layer, CellMode value) noexcept {
        if (layer < layers) {
            mode[layer] = value;
            v_plane[layer] = value;
        }
    }
    // THE TWO-SIDED WRITE: the walk's own, and the only way the two planes can part.
    constexpr void set_pair(std::uint32_t layer, CellPair value) noexcept {
        if (layer < layers) {
            mode[layer] = value.k;
            v_plane[layer] = value.v;
        }
    }
};

[[nodiscard]] constexpr BlockVector block_vector_default(std::uint32_t layers) noexcept {
    BlockVector vector{};
    vector.layers = layers < kBlockVectorLayerAxis ? layers : kBlockVectorLayerAxis;
    return vector;
}

// THE CHARGE, AS A VALUE WITH A STATE. `priced == false` is NOT an error and NOT zero: it is
// the statement "at least one cell in this vector sits on a rung the tree has no byte cost for",
// which is exactly what happens the moment a cell reaches `e8-2bit`. A caller that treated it as
// zero would be reading a free lunch.
struct CellCharge {
    bool priced = true;
    std::int64_t bytes = 0;
    std::uint32_t unpriced_layers = 0;
};

//     bytes(V) := 2 * kv_heads * sum_l plane_bytes(V[l])          [planes 2026-09-29]
//
// ⭐ [planes 2026-09-29] THE ORDER'S RULER AND THE CHARGE'S RULER DISAGREED; THIS MAKES THEM ONE.
// WHICH SIDE MOVED, WHICH DID NOT, AND WHY. Until this cut the charge summed `record_bytes` (the
// COLD SLOT the pool stages: 9,232 / 9,632) while `cell_next_cheaper` ordered its rungs by
// `plane_bytes` (the RESIDENT PLANE the demotion gives back). Two rulers, and the MEASURED
// consequence, on the first binding arm this line ever ran (`PLN_cap45`, budget 53,176,320):
// every one of the 768 cells was refused with `steps=0`, because from `int8` the cheaper-PLANE rung
// is `nvfp4` and its RECORD is 9,632 -- so the step's delta was `9,232 - 9,632 = -400`, negative,
// and the driver's strictly-decreasing guard (`kv_block_descent.h:429`) refused it as it must.
// NO BUDGET COULD PRODUCE `saved_bytes > 0`: the five rungs below `int8` all share the 9,232 raw
// record (zero delta) and the one that does not is wider. That is a deadlock, not a shortfall.
// SO THE CHARGE MOVED, ONTO THE `<plane_bytes>` COLUMN, for two reasons and only these:
//   * the ORDER is already on that ruler (`argmax{ plane_bytes' < plane_bytes }`), and a stop
//     condition must be denominated in the same quantity the steps move;
//   * the FLOOR only exists on that ruler -- `e8-2bit`'s 4,608 B is a resident plane, and its COLD
//     RECORD is the raw slot at 9,232, i.e. on the record ruler the floor's step is 16 B of shape
//     and nothing of currency. A charge that cannot express the floor cannot price a walk to it.
// THE OTHER SIDE DID NOT MOVE, AND MUST NOT: `record_bytes` is still the truth about the COLD SLOT
// (`kv_tier_formats.h`'s 9,232 / 9,632, and the rANS tier's own unsatisfiability pins), and it is
// still what `cell_record_bytes` / `cell_mode_record_bytes` report. Those two keep their names and
// their ruler; they are simply no longer what the CHARGE sums. `kv_cold_codec_reduces` and the
// FIT/PAY table are untouched -- they answer the codec question, never the budget question.
// ⚠ THE LEGACY CHARGE IS KEPT AS A NAMED, QUOTABLE READING, NOT DELETED -- because the ledger
// (`blob_F1160.md` section 8.1), the stage's own `ceiling.txt = 302,514,176` and every arm already
// run tonight (including `sparse` / `cap*` and the host-side section 8.1 table) are priced in it,
// and without it the two batches of readings cannot be compared. It is a READING of a retired
// ruler, and nothing charges from it any more:
inline constexpr std::int64_t kLegacyChargePerBlockRawInt8_16x4 =
    2LL * 4 * 16 * kKvColdInt8PayloadBytes;   // 2 planes * KV heads 4 * layers 16 * raw 9232
static_assert(kLegacyChargePerBlockRawInt8_16x4 == 1181696,
              "the LEGACY charge per block at the shipped 16x4 all-raw stack == 1,181,696, the "
              "number the ledger and ceiling.txt pin. If this fires, this is not a reading of the "
              "retired ruler any more and the before/after comparison of tonight's arms is void");
// And the SAME stack under the ruler this header now charges from -- the number the stage will
// print as `bytes_per_block` on the next run, derived here so the two can be compared at a glance:
inline constexpr std::int64_t kChargePerBlockRawInt8_16x4 =
    2LL * 4 * 16 * 16896;                      // 2 planes * KV heads 4 * layers 16 * int8 plane
static_assert(kChargePerBlockRawInt8_16x4 == 2162688,
              "the resident-plane charge per block at the shipped 16x4 all-int8 stack == 2,162,688 "
              "(= 1,181,696 * 16896/9232); the budget a run supplies is now spent in THIS unit");

//
// THE SAME SUM THE ENGINE ALREADY HAS. `kv_block_stage_bytes_per_block(layers, kv_heads,
// rans_layers)` computes `2 * kv_heads * (rans_layers * 9632 + raw_layers * 9232)`. THAT IS THIS
// FUNCTION AT A VECTOR WHOSE MODES TAKE AT MOST TWO VALUES, with `rans_layers` a COUNT standing
// in for the multiset. The count is a SUFFICIENT STATISTIC FOR THIS SUM IF AND ONLY IF THE RUNG
// TABLE HAS AT MOST TWO DISTINCT PRICED RECORDS -- true today (9232, 9632) and false the moment
// the order's floor gets a stride of its own. So this function is not a replacement; it is the
// same sum with a strictly larger domain, and the count is demoted to a reading. The equality
// itself is a pin for a test TU (this header cannot include the stage header, which includes it).
[[nodiscard]] constexpr CellCharge cell_vector_charge(const BlockVector& vector,
                                                      std::int32_t kv_heads) noexcept {
    CellCharge charge{};
    if (kv_heads <= 0) { charge.priced = false; return charge; }
    std::int64_t per_layer_pair = 0;
    for (std::uint32_t layer = 0; layer < vector.layers; ++layer) {
        // [F1260 kvplanar] ⭐ THE SUM IS OVER THE PAIR, AND THE FACTOR `2` HAS MOVED INTO IT. The
        // `2` in the pre-image spelling (`2 * kv_heads * sum_l plane_bytes`) was the K plane and the
        // V plane; `cell_pair_bytes` now adds those two planes EXPLICITLY, so
        //     diagonal pair : cell_pair_bytes(m,m) == 2 * plane_bytes(m)   [asserted, all six rows]
        //     charge        : kv_heads * sum_l cell_pair_bytes(pair_l)     [this line]
        // and for every vector whose cells are diagonal this expression IS the pre-image one, term
        // for term. For a vector whose cells are NOT, it charges the two planes the cell actually
        // has -- which is the whole of what "the walkway can express a pair" means on the budget
        // side.
        const CellPair pair = vector.pair_at(layer);
        const CellRung& k_rung = cell_rung(pair.k);
        const CellRung& v_rung = cell_rung(pair.v);
        if (k_rung.price_state != PriceState::Priced ||
            v_rung.price_state != PriceState::Priced) {
            charge.unpriced_layers += 1;
            continue;
        }
        // [planes 2026-09-29] THE CHARGE SUMS THE RESIDENT PLANES, NOT THE COLD RECORD -- the one
        // line that puts the order and the stop condition on one ruler. See the long note above.
        per_layer_pair += cell_pair_bytes(pair);
    }
    if (charge.unpriced_layers != 0) { charge.priced = false; }
    charge.bytes = static_cast<std::int64_t>(kv_heads) * per_layer_pair;
    return charge;
}

// The charge of ONE cell. ⚠ [planes 2026-09-29] THIS IS NOW THE **RECORD** PRICE AND NOT THE
// CHARGE'S RULER: `cell_vector_charge` sums `cell_rung(mode).plane_bytes`, so a caller that wants a
// cell's charge reads `plane_bytes` (`:16896 / 32768 / 9216 / 8704 / 6656 / 4608`), and this
// function is kept because the COLD SLOT question is still a real question (`cell_record_bytes`,
// 9232 / 9632). The `2 * kv_heads` factor is the caller's in both cases.
[[nodiscard]] constexpr std::int32_t cell_mode_record_bytes(CellMode mode) noexcept {
    const CellRung& rung = cell_rung(mode);
    if (rung.price_state != PriceState::Priced) { return -1; }
    return cell_record_bytes(cell_record_of(rung.codec));
}

// IS EVERY CELL OF THIS VECTOR ON ONE MODE? -- the question the SIDECAR's single `codec` byte
// has been answering by accident. `RecallBlockCodec` has one value per block while a block's
// payload is a vector; the refusal name says so itself (plural): "NoCodec, // the block's layers
// have no cold codec, so nothing was packed". This predicate is the ONE thing that has to be
// true for the scalar spelling to be a faithful description, and it is what the sidecar
// invariant is written against.
[[nodiscard]] constexpr bool block_vector_is_uniform(const BlockVector& vector) noexcept {
    for (std::uint32_t layer = 1; layer < vector.layers; ++layer) {
        if (vector.mode[layer] != vector.mode[0]) { return false; }
    }
    return true;
}

// The one codec a uniform vector can be described by, or `None` when it is NOT uniform. `None`
// therefore means TWO different things and the sidecar must not conflate them: "this vector has
// one mode and it has no pack arm" and "this vector has more than one mode". The sidecar patch
// (`kv_recall_block.h`) splits the refusal into `NoCodec` and `MixedLayers` for exactly this
// reason, because a mixed block is the state the order DEMANDS, not a failure.
[[nodiscard]] constexpr LayerColdCodec block_vector_uniform_codec(const BlockVector& vector) noexcept {
    if (!block_vector_is_uniform(vector)) { return LayerColdCodec::None; }
    return cell_rung(vector.mode[0]).codec;
}

// The per-layer echo the placer owes the directory. NOTE THE FLOOR'S HOLE: a cell at `e8-2bit`
// has `LayerColdCodec::None`, so an echo built from this function reports "nothing was packed"
// for a cell that is SUPPOSED to have been packed. That is a consequence of G6 and it is named
// here rather than papered over: **a cell that reached the order's floor cannot be echoed
// today.** It is the third independent place G6 bites (the other two: `cell_next_cheaper`'s
// `RungBytesUnknown` and the unpriced row of `kCellRungs`).
[[nodiscard]] constexpr LayerColdCodec cell_mode_codec(CellMode mode) noexcept {
    return cell_rung(mode).codec;
}

// The ONE-LINE reading of a vector, for the accounting line and for a test's failure message.
[[nodiscard]] inline std::string block_vector_describe(const BlockVector& vector) {
    std::string out = "layers=" + std::to_string(vector.layers) + " [";
    for (std::uint32_t layer = 0; layer < vector.layers; ++layer) {
        if (layer != 0) { out += " "; }
        out += cell_mode_name(vector.mode[layer]);
    }
    out += "]";
    return out;
}

} // namespace ninfer::product
