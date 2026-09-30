#pragma once

#include <cstddef>
#include <cstdint>

namespace ninfer {

enum class DType : std::uint8_t {
    BF16       = 0,
    FP32       = 1,
    I32        = 2,
    U8         = 3,
    I64        = 4,
    I8         = 5,
    FP16       = 6,
    FP8_E4M3FN = 7,
    // Packed E2M1 nibble plane (two codes per byte) with per-16-channel
    // E4M3FN scales; see the per-layer KV storage table.
    NVFP4      = 8,
    ISO3       = 9,
    // Packed 4-bit E8-lattice K codes + i4 V codes (two codes per byte) with
    // per-64-channel FP16 scales; consumed by the int8 attention kernels
    // (stage unpacks nibbles to i8). See the per-layer KV storage table.
    // LEGACY SPELLINGS (line dl/e8names, marker F1194). The three names in this enum are
    // KEPT at their values, byte for byte, because each is the DEPLOYED token -- see the
    // plane-level primary block at the foot of this enum, which spells the same three
    // pairs per plane and is equal to these BY VALUE.
    E8Kv       = 10,
    // e8 family, narrower K-plane code widths (e8k3 / e8k2): the SAME K+V plane pair
    // and the same per-64 FP16 scale plane as E8Kv, with the K code plane packed at 3
    // or 2 bits per element. Geometry and cost are derived in product/kv_e8_width.h
    // (K plane 8704 -> 6656 -> 4608 B/head-page; layer 4.25 -> 3.75 -> 3.25 b/element,
    // K+V averaged, which is what the bit-budget ladder prices).
    // NOT a different codebook: product/kv_e8_width.h records that the ecosystem's
    // rk2v4-e8 reproduces this exact GEOMETRY while its 240-root cylinder CODEC does
    // not, and that the measured lattice floor puts both widths BELOW the projection
    // they are named for.
    E8K3Kv     = 11,   // LEGACY (F1194): pair(B3,B4) -- primary spelling E8KvB3B4
    E8K2Kv     = 12,   // LEGACY (F1194): pair(B2,B4) -- primary spelling E8KvB2B4

    // =====================================================================================
    // [dl/e8names, marker F1194] THE PLANE-LEVEL PRIMARY SPELLINGS
    // =====================================================================================
    // WHY THESE EXIST, IN THE OWNER'S OWN TERMS. The order is that the 4-bit, 3-bit and
    // 2-bit rungs must ALL work. The three names above spell that ladder three different
    // ways, and each of them writes V as a LITERAL: `E8Kv` has no V position at all, and
    // `E8K3Kv` / `E8K2Kv` put K's width in the name and give V nowhere to sit. So a table
    // keyed by these names cannot notice that it has no arm for the 3-bit rung -- and one
    // did not: the same dtype was `rk2v4` in one table and `a 16-bit plane (no cold
    // codec)` in the table beside it (impl/runtime/program_impl.h, both ends fixed by
    // this landing).
    //
    // THE CONVENTION THIS SPELLING NEEDS, STATED ONCE. `E8Kv` is the family; the FIRST
    // `B<n>` is the K plane's format and the SECOND is the V plane's, where `B<n>` is one
    // of `product::E8KvPlaneFormat`'s three values (product/kv_e8_width.h, whose `e8-b<n>`
    // names are those same three formats spelled for ONE plane). SO THE OWNER'S FLOOR GETS A
    // NAME WITH THIS LANDING AND DELIBERATELY NOT AN ENUMERATOR: the floor is `e8-b2/e8-b2`
    // (product/kv_e8_width.h's `e8_kv_pair_name(B2,B2)`), which is spellable today and was
    // unspellable in the old vocabulary, while `DType` keeps THREE e8 values because three
    // is what the engine has a geometry and a writer for. Adding a fourth enumerator here
    // would be a tier the build layer refuses -- the same defect one plane over.
    // The floor's bytes are already priced and pinned, cited by its own text in
    // product/kv_e8_width.h and NOT by a line number: this landing's own two include lines
    // moved that number from 331 to 333, which is the whole reason this tree's citations
    // are read as anchors and not as offsets.
    //
    // SAME VALUE, SAME BYTES, SAME EVERY PATH. These are ALIASES: an enumerator is its
    // value, so every `switch (DType)` in the tree keeps its exact meaning and not one .o
    // moves. The switches keep naming the legacy spellings on purpose -- naming BOTH
    // spellings of one value in one switch is a compile error (duplicate case value), and
    // the deployed literal is what the tests assert -- and GCC's -Wswitch / -Wswitch-enum
    // are satisfied by EITHER spelling alone, because both are the same value. MEASURED,
    // both directions, dl/e8names/t_dup2.cpp.
    E8KvB4B4 = 10,   // K plane format B4, V plane format B4
    E8KvB3B4 = 11,   // K plane format B3, V plane format B4
    E8KvB2B4 = 12,   // K plane format B2, V plane format B4
};


std::size_t dtype_size(DType dtype);

} // namespace ninfer
