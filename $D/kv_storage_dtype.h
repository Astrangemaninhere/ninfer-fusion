#pragma once

// The ONE KvCacheStorage -> DType mapping, and it REFUSES an unmapped value.
//
// WHY THIS HEADER EXISTS. The per-layer KV table is a
// std::array<KvCacheStorage, kKvLayerStorageSlots> built by the front ends
// (--kv-layer-storage, /reload_kv) and by embedders through the public
// ninfer::Engine per-layer overload. The mapping from those storages to the
// DType the page pool is actually built from used to live as a nested
// conditional whose LAST ARM was `: DType::BF16`.
//
// DType::BF16 is not an ordinary tier: it is also the "inherit the global
// --kv-dtype" sentinel (decoder_state.cpp PagedKVCache::plan_cache's
// layer_dtype(), product/kv_component_switch.h kv_resolved_layer_dtype). So an
// unmapped storage value did not merely lose its codec, it silently expressed
// "ignore the operator's dtype on this layer". Both halves are silent and
// neither is reported, and plan_cache()'s own validator ("Paged KV per-layer
// dtype is invalid", decoder_state.cpp) could never fire, because the laundering
// happened one function earlier. A silent wrong answer is worse than a crash.
//
// The tree already contains the correct idiom ~1300 lines above the defect:
// target_kv_cache_profile() (layouts_impl.h) switches over the SAME enum with no
// default arm and throws "unknown KV-cache storage profile" after the switch.
// This header is that idiom, placed where both the target plan and a host test
// can reach it.
//
// Deliberately NOT a switch with a default arm: every enumerator is named and the
// throw sits AFTER the switch, so adding an enumerator makes -Wswitch point here
// and a stale table cannot slip through in either direction.

#include "ninfer/types.h"
#include "core/dtype.h"

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

// `where` is the caller's own provenance for the slot, e.g.
// "--kv-layer-storage[7]", so the refusal names the thing the operator wrote.
// Throws std::invalid_argument for a storage the engine has no KV codec for.
[[nodiscard]] inline DType kv_dtype_for_storage(KvCacheStorage storage, std::string_view where) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return DType::BF16;
    case KvCacheStorage::Int8Group64:
        return DType::I8;
    // BOTH fp8 spellings name the SAME target tier: KvCacheStorage carries the
    // standalone ops/kv_cache row-scaled codec's old name (Fp8E4M3Row256) for
    // --kv-dtype / request-log naming, while product::parse_kv_storage emits
    // Fp8Group16 for the per-layer spec, and target_kv_cache_profile() maps both
    // to (DType::FP8_E4M3FN, kKvFp8QuantGroup).
    case KvCacheStorage::Fp8E4M3Row256:
    case KvCacheStorage::Fp8Group16:
        return DType::FP8_E4M3FN;
    case KvCacheStorage::Nvfp4Group16:
        return DType::NVFP4;
    case KvCacheStorage::Iso3Group16:
        return DType::ISO3;
    case KvCacheStorage::E8Group64:
        return DType::E8Kv;
    case KvCacheStorage::E8K3Group64:
    case KvCacheStorage::E8K2Group64:
        // NAMED AND REFUSED, exactly as Dropped is -- for the opposite reason. Dropped
        // names a real layer that owns no planes; these two name a real TIER that owns
        // no READER. The vocabulary (kvcfg/kv_formats.h "rk3v4"/"rk2v4", bits_of 3 and 2),
        // the plane geometry (product/kv_e8_width.h: K 6656 / 4608 B per head-page, the
        // second being byte-identical to the ecosystem's rk2v4-e8 K plane), the ladder cost
        // (375 / 325, K+V averaged), the host codec of record
        // (src/ops/kv/e8_lattice_plane_codec.cuh, post-image b550e0c50991120e) and the device
        // arm (src/ops/kernel/e8_lattice_kv_plane.cuh, post-image c733d788c9260997, host
        // suite 75/75, 0 FAIL) all exist and are pinned by static_assert.
        //
        // ⚠ RE-DERIVED (dl/e8decode), because BOTH reasons this refusal used to give had
        // become false on the tree's own bytes, and a refusal that blames an already-fixed
        // defect sends the next reader to fix the wrong thing:
        //   * "no page-pool planner has an arm for it" -- FALSE. decoder_state.cpp:422-442 is
        //     a dedicated rk3v4/rk2v4 arm and lays the K plate at
        //     head_dim / 8 * product::e8_kv_code_bytes_per_8(width) -- 96 B/row at W3, 64 at
        //     W2 -- beside the family's unchanged 128-byte V plate. These tiers no longer
        //     reach the NVFP4/ISO4E arm's head_dim/2.
        //   * "no CMake target compiles the device arm" -- FALSE. src/CMakeLists.txt:173
        //     compiles ops/kernel/e8_lattice_kv_plane_inst.cu, the forced-instantiation TU
        //     that is the reader's compile floor (its comment block starts at :141).
        //
        // WHAT IS ACTUALLY MISSING IS A CALLER, and that alone is why this refusal stands.
        // The reader for a 3-bit/2-bit K plate exists and is COMPILED, and NOTHING ON ANY
        // RUNTIME PATH CALLS IT: `e8_kv_lattice_decode_group<3>` / `<2>` and
        // `e8_kv_lattice_decode_group_codes<3>` / `<2>` occur only at their own definitions
        // (src/ops/kernel/e8_lattice_kv_plane.cuh:361,400) and in the forced-instantiation TU
        // (src/ops/kernel/e8_lattice_kv_plane_inst.cu:88,89,92,93). The append arm writes the
        // plate through the ENCODER (src/ops/kv_cache/append/e8_lattice_narrow_kernel.cuh:187,
        // `e8_kv_lattice_encode_group<WBits>`), so the tier is WRITABLE and NOT READABLE.
        // Every attention arm in this engine is 4-bit: the gqa e8 decode kernel unpacks packed
        // NIBBLES (src/ops/kernel/gqa_attention_decode_i8.cuh:269-271 via
        // gqa_kv_i4_code_index), and the causal family has no e8 kernel at all
        // (src/ops/softmax_attention/dense/causal_cache/prompt.cu:34 is {i8, else bf16}).
        //
        // Both wrong resolutions are the ones this function exists to prevent, so neither
        // is allowed here: DType::E8Kv would run the 4-bit nibble reader over a narrower
        // plate and misread every row (silent corruption), and DType::BF16 would silently
        // inherit the global --kv-dtype and ignore the request. Refuse by name and say
        // which of the two the operator asked for.
        throw std::invalid_argument(
            std::string(where) + ": '" +
            (storage == KvCacheStorage::E8K3Group64 ? "rk3v4" : "rk2v4") +
            "' is a DEFINED rk4v4 tier whose codec, K plate layout and WRITER all exist but "
            "which NO RUNTIME PATH CAN READ. Its contract is complete and "
            "checked (product/kv_e8_width.h: K plane 6656 B at 3 bits / 4608 B at 2 bits "
            "per head-page, per-64 FP16 scales, ladder cost 3.75 / 3.25 b/element), "
            "decoder_state.cpp:422-442 lays its K plate at its own 96/64 B row beside the "
            "family's 128 B V plate, and e8_lattice_narrow_kernel.cuh:187 appends it through "
            "the encoder -- but the reader that decodes a 3-bit/2-bit plate "
            "(e8_lattice_kv_plane.cuh, e8_kv_lattice_decode_group<3>/<2>) has NO CALLER "
            "outside its own definition and the forced-instantiation TU "
            "(e8_lattice_kv_plane_inst.cu:88,89), and every attention arm in the engine reads "
            "the 4-BIT nibble plane, so a 96/64-byte K plate would be decoded at the "
            "128-byte extent. It is refused rather than resolved to rk4v4 (which would "
            "misread every row through the 4-bit nibble reader) or to bf16 (which would "
            "silently inherit the global --kv-dtype). Use rk4v4, int8, nvfp4 or iso4e.");
    // INTEGRATE4's five. NAMED, each refusing for its OWN reason, because the trailing throw
    // would tell an operator that no enumerator names their storage code -- the same
    // misdirection the Dropped arm below was added to remove, and measurably still present
    // before this landing (dl/e8mixwire/probe/out_absent.txt, codes 10..14: "KV storage code
    // 10 names no enumerator, so it has no short name").
    case KvCacheStorage::Fp8KeyNvfp4Value:
        // A K/V CODEC PAIR, not one tier: its resolved planes
        // (src/core/paged_kv_storage.h:92-99) are {FP8_E4M3FN, 256, FP16, 1} for K --
        // byte-for-byte Fp8E4M3Row256 -- and {U8, 128, U8, 16} for V -- byte-for-byte
        // Nvfp4Group16. DType has one enumerator per PLANE CODEC, so there is no DType that
        // means "fp8 K with an nvfp4 V": resolving either half would build a layer whose other
        // plane is read through the wrong reader.
        throw std::invalid_argument(
            std::string(where) + ": 'fp8k-nvfp4v-g64' is a K/V codec PAIR (fp8 K, nvfp4 V), "
            "not a single tier. Its K plane is Fp8E4M3Row256's and its V plane is "
            "Nvfp4Group16's (core/paged_kv_storage.h:92-99), and this engine's per-layer table "
            "carries ONE DType per layer, so neither half can stand for the whole. Refused "
            "rather than resolved to either half, which would build one plane correctly and "
            "read the other through the wrong reader.");
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
    case KvCacheStorage::RK4V4E8:
    case KvCacheStorage::RK2V4E8:
        // The fork's int8-family storages. Their geometry exists
        // (core/paged_kv_storage.h:103-122) and that file says about itself what decides this
        // arm: "ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not
        // GLOB), so this file is inert until someone adds it deliberately". This tree's page
        // pool has no geometry for them, so a resolved DType here would name a plane the
        // decoder cannot lay out.
        //
        // RK4V4E8/RK2V4E8 are TWO CHARACTERS from this tree's own rk4v4/rk3v4/rk2v4 TIERS and
        // describe a different codec on the same geometry (the 240-root root-cylinder
        // specimen); dl/e8dev measured that aliasing them onto the lattice is "a WRONG CODEC
        // UNDER A RIGHT NAME". They must never resolve to DType::E8Kv / E8K3Kv / E8K2Kv.
        throw std::invalid_argument(
            std::string(where) + ": '" +
            (storage == KvCacheStorage::RK4V4E8     ? "rk4v4e8"
             : storage == KvCacheStorage::RK2V4E8   ? "rk2v4e8"
             : storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 ? "rot-i8k-i4v-g64"
                                                    : "rot-i4k-i4v-g64") +
            "' is a FORK-SURVEY storage (sergiuszm/ninfer-4090 @ rtx4090-port). Its geometry is "
            "declared in core/paged_kv_storage.h but that file is ADDITIVE and not wired into "
            "any build target, so this engine's page pool has no layout for it. Refused by name "
            "rather than resolved: rk4v4e8/rk2v4e8 name a root-cylinder specimen that is NOT "
            "this tree's rk4v4/rk3v4/rk2v4 lattice tiers, and resolving either onto them would "
            "put a different codec behind a right-looking name.");
    case KvCacheStorage::Dropped:
        // NAMED, not left to fall past the switch. It has to be named for two
        // separate reasons, and neither of them is "make -Wswitch quiet":
        //
        // 1. Falling past the switch already THROWS, so the semantics were right --
        //    but the message was not: the trailing throw says "KV storage code 7 has
        //    no DType", which claims no enumerator names it. types.h:29 does name it.
        //    An operator who asked for a dropped layer was told their storage code
        //    was unknown, when the truth is that the layer was discarded
        //    (NINFER_KV_DROP_LAYERS) and owns no KV planes to build a codec for.
        // 2. It makes THIS switch answer the question the header comment above
        //    promises it answers ("adding an enumerator makes -Wswitch point here").
        //    A switch that is one arm short is a switch that reported nothing about
        //    that arm; -Wswitch naming Dropped here is the same signal the next
        //    enumerator will produce, and now it is explicit which of the two cases
        //    any given enumerator is in.
        //
        // The sibling mapping over the SAME enum already decided this the same way:
        // layouts_impl.h target_kv_cache_profile():105 names Dropped and throws.
        // And the refusal is ALREADY pinned executably by
        // tests/test_kv_component_switch.cpp:171-184, whose 256-code sweep requires
        // exactly 249 refusals -- code 7 among them -- and zero codes resolving to
        // BF16. Returning anything else here turns that test red.
        throw std::invalid_argument(
            std::string(where) + ": this layer's KV storage was DISCARDED "
            "(NINFER_KV_DROP_LAYERS), so it owns no KV planes and has no codec to "
            "resolve. It is not a bf16 layer -- DType::BF16 would also mean "
            "\"inherit the global --kv-dtype\", i.e. act on the request instead of "
            "refusing it.");
    }
    // Reachable only for a value no enumerator names: an out-of-range byte in the
    // table (KvCacheStorage is a std::uint8_t, so a cast or an untrusted input can
    // produce one) or an enumerator added without a codec here. Refuse LOUDLY.
    // Falling back to DType::BF16 here is precisely the silent downgrade this
    // function exists to remove: it would drop the request AND inherit the global
    // dtype, i.e. do nothing while reporting success.
    throw std::invalid_argument(
        std::string(where) + ": KV storage code " +
        std::to_string(static_cast<unsigned>(storage)) +
        " has no DType, so this layer's KV codec is unknown. Refusing rather than "
        "resolving it to DType::BF16, which would silently inherit the global "
        "--kv-dtype and ignore the request. A per-layer slot must be one of "
        "bf16, int8, fp8, nvfp4, iso4e, rk4v4, rk3v4, rk2v4 (the last two refuse by name "
        "until their K code plate gains a reader -- see product/kv_e8_width.h). This "
        "message deliberately does NOT enumerate the fork-survey storages: it is the "
        "GENERIC arm, and a fallback that lists every tier satisfies any test asking "
        "whether a refusal names its tier -- measured, by deleting the Fp8KeyNvfp4Value "
        "arm and watching the sweep stay green (dl/e8mixwire/land/run_suite.py, MUT-B).");
}

// ---------------------------------------------------------------------------
// THE CANONICAL SHORT NAME OF EVERY KvCacheStorage, AND WHY IT IS AN ARRAY.
//
// This tree carries NINE hand-written KvCacheStorage -> text tables and every one
// of them was at least one enumerator short (REPORT.md section T5). The reason is
// structural: all nine are `switch` statements, and in THIS build a short switch
// is not even a warning -- CMakeLists.txt opens no warning flag at all, and even
// with -Wall -Wswitch the message is a warning, never an error. A comment that
// claims "a new enumerator breaks this build" is therefore false until something
// is actually an error.
//
// An array whose LENGTH is pinned by a static_assert IS an error. The asserts
// below close BOTH directions, which is the property the switches only claimed:
//   * add an enumerator and forget the name    -> the count assert fires;
//   * add a name and forget to grow the pin    -> the length assert fires.
//
// The table is built with std::to_array deliberately: its size is DEDUCED from the
// initialiser list, so `static_assert(std::size(...) == N)` can fail. With
// `std::array<T, N> t{...}` it cannot -- the unwritten tail is value-initialised
// and size() is N no matter how many entries were written. (Pinned by the "naive"
// control in gen/t4/, which compiles with a table one entry short.)
//
// kKvCacheStorageCount is pinned to the LAST enumerator rather than being a free
// literal, so appending an enumerator after Dropped fires COUNTER_ASSERT and
// forces the author to look at this block.
inline constexpr std::size_t kKvCacheStorageCount = 15;
// RE-PINNED to the TRUE last enumerator. The pin used to be
// `E8K2Group64 + 1 == kKvCacheStorageCount`, which was correct only while E8K2Group64 WAS
// last. INTEGRATE4 appended five enumerators after it (Fp8KeyNvfp4Value and four fork
// spellings), so the old pin still evaluated 9 + 1 == 10 -- TRUE -- while the enumeration had
// grown to 15. A pin whose whole promise is "appending an enumerator fires COUNTER_ASSERT"
// must be anchored to the end of the enumeration, not to whatever was last when it was
// written; that is exactly the failure this comment used to claim could not happen.
//
// Measured (dl/e8mixwire/probe/mixwire_probe.cpp, out_absent.txt): with the old pin, codes
// 10..14 -- Fp8KeyNvfp4Value, RotatedInt8KeyInt4ValueGroup64, RotatedInt4KeyInt4ValueGroup64,
// RK4V4E8, RK2V4E8 -- each threw "names no enumerator", which is false about all five.
// Pinned to the LAST enumerator, and it now MEANS it. The pin named E8K2Group64 while that
// enumerator was last; INTEGRATE4 then appended five more, and an assert pinned to
// E8K2Group64 evaluated 9 + 1 == 10 -- TRUE -- with the enumeration at 15. Pinning to a
// NAMED enumerator is only a pin while that enumerator is last, so the anchor here is
// RK2V4E8, the true end of the enumeration as include/ninfer/types.h declares it.
//
// This assert was MEASURED firing during this landing: the first splice raised the count to
// 15 and left this expression on E8K2Group64, and the header refused to compile with
// "(10 == 15)" (dl/e8mixwire/land/, pass 1). A pin that can only ever be true is a label;
// this one produced a diagnostic on a real edit, which is what the claim above asks for.
static_assert(static_cast<std::size_t>(KvCacheStorage::RK2V4E8) + 1 == kKvCacheStorageCount,
              "KvCacheStorage grew an enumerator: bump kKvCacheStorageCount AND name the new "
              "value in kKvStorageNames below (do not silence this assert)");

// One entry per enumerator, in ENUMERATOR ORDER. The four shipped spellings are
// untouched; the four appended tiers use the "source" vocabulary of
// include/ninfer/types.h. The register an OPERATOR reads
// (serve/kv_auto_relayout.cpp: "int8", "fp8", "nvfp4") is deliberately a
// different one and stays where it is -- see REPORT.md section T4 for why this
// table does not absorb it.
inline constexpr auto kKvStorageNames = std::to_array<std::string_view>({
    "bf16",        // KvCacheStorage::BFloat16
    "int8-g64",    // KvCacheStorage::Int8Group64
    "fp8-e4m3-r256",  // KvCacheStorage::Fp8E4M3Row256
    "nvfp4-g16",   // KvCacheStorage::Nvfp4Group16
    "fp8-g16",     // KvCacheStorage::Fp8Group16
    "iso4e-g16",    // KvCacheStorage::Iso3Group16
    "rk4v4-g64",      // KvCacheStorage::E8Group64
    "dropped",     // KvCacheStorage::Dropped
    "rk3v4-g64",    // KvCacheStorage::E8K3Group64
    "rk2v4-g64",    // KvCacheStorage::E8K2Group64
    // INTEGRATE4's five, appended at the enum's own indices. The token is the enumerator in
    // kebab case, and where kebab-casing would collide with a TIER name the collision is
    // broken by the source's own `E8` suffix: `RK4V4E8` -> "rk4v4e8", one character from the
    // tier "rk4v4" and from this table's own "rk4v4-g64" (KvCacheStorage::E8Group64). That
    // distinction is load-bearing, not cosmetic: the four `-Wswitch` sites in
    // dl/warnsurface/E8_SWITCH_SITES.md have no `default:`, and dl/e8dev measured that
    // mapping the RK* pair onto the E8 lattice tiers would put a root-cylinder specimen
    // behind the lattice's name -- a wrong codec under a right name.
    "fp8k-nvfp4v-g64",  // KvCacheStorage::Fp8KeyNvfp4Value
    "rot-i8k-i4v-g64",  // KvCacheStorage::RotatedInt8KeyInt4ValueGroup64
    "rot-i4k-i4v-g64",  // KvCacheStorage::RotatedInt4KeyInt4ValueGroup64
    "rk4v4e8",          // KvCacheStorage::RK4V4E8
    "rk2v4e8",          // KvCacheStorage::RK2V4E8
});
static_assert(std::size(kKvStorageNames) == kKvCacheStorageCount,
              "kKvStorageNames must carry exactly one entry per KvCacheStorage enumerator");

// Positional truth: every entry must sit at ITS OWN enumerator's index, so a table
// that is complete but REORDERED is refused too. Reordering is the other silent
// failure mode -- it names every value and mislabels all of them. It is not
// hypothetical: serve/kv_auto_relayout.cpp's switch lists Fp8Group16 BEFORE
// Nvfp4Group16 while the enum has it after, so a mechanical switch->array
// conversion there would swap those two tiers.
static_assert(kKvStorageNames[static_cast<std::size_t>(KvCacheStorage::BFloat16)] == "bf16");
static_assert(kKvStorageNames[static_cast<std::size_t>(KvCacheStorage::Int8Group64)] == "int8-g64");
static_assert(kKvStorageNames[static_cast<std::size_t>(KvCacheStorage::Fp8E4M3Row256)] == "fp8-e4m3-r256");
static_assert(kKvStorageNames[static_cast<std::size_t>(KvCacheStorage::Nvfp4Group16)] == "nvfp4-g16");
static_assert(kKvStorageNames[static_cast<std::size_t>(KvCacheStorage::Fp8Group16)] == "fp8-g16");
static_assert(kKvStorageNames[static_cast<std::size_t>(KvCacheStorage::Iso3Group16)] == "iso4e-g16");
static_assert(kKvStorageNames[static_cast<std::size_t>(KvCacheStorage::E8Group64)] == "rk4v4-g64");
static_assert(kKvStorageNames[static_cast<std::size_t>(KvCacheStorage::Dropped)] == "dropped");
static_assert(kKvStorageNames[static_cast<std::size_t>(KvCacheStorage::E8K3Group64)] == "rk3v4-g64");
static_assert(kKvStorageNames[static_cast<std::size_t>(KvCacheStorage::E8K2Group64)] == "rk2v4-g64");
static_assert(kKvStorageNames[static_cast<std::size_t>(KvCacheStorage::Fp8KeyNvfp4Value)] == "fp8k-nvfp4v-g64");
static_assert(kKvStorageNames[static_cast<std::size_t>(KvCacheStorage::RotatedInt8KeyInt4ValueGroup64)] == "rot-i8k-i4v-g64");
static_assert(kKvStorageNames[static_cast<std::size_t>(KvCacheStorage::RotatedInt4KeyInt4ValueGroup64)] == "rot-i4k-i4v-g64");
static_assert(kKvStorageNames[static_cast<std::size_t>(KvCacheStorage::RK4V4E8)] == "rk4v4e8");
static_assert(kKvStorageNames[static_cast<std::size_t>(KvCacheStorage::RK2V4E8)] == "rk2v4e8");

// The lookup keeps the LOUD contract of kv_dtype_for_storage above: the table is
// total over the enum, but a value no enumerator names (KvCacheStorage is a
// uint8_t, so a cast or untrusted text produces one) would index out of range.
// Refuse it instead of reading past the end.
[[nodiscard]] inline std::string_view kv_storage_token(KvCacheStorage storage) {
    if (static_cast<std::size_t>(storage) >= kKvStorageNames.size()) {
        throw std::invalid_argument(
            std::string("KV storage code ") + std::to_string(static_cast<unsigned>(storage)) +
            " names no enumerator, so it has no short name");
    }
    return kKvStorageNames[static_cast<std::size_t>(storage)];
}

// The name the OPERATOR reads: what a front end prints on its "kv cache dtype" line.
//
// This is NOT a second table. It is the canonical token above, returned as an owning
// string so a print site can concatenate or store it. Deriving it HERE, beside the
// table it reads, is the whole point: apps/cli/main.cpp used to carry its own switch
// (main.cpp:101-122) that had drifted to six stale spellings -- iso3-group16 for the
// row this table calls iso4e-g16, e8-group64 for rk4v4-g64, plus int8-group64,
// fp8-e4m3-row256, nvfp4-group16 and fp8-group16 -- and that knew only EIGHT of the
// ten enumerators, so E8K3Group64/E8K2Group64 printed "unknown". The line it feeds is
// the line an operator reads to confirm WHICH TIER ACTUALLY RAN, so a stale name there
// makes every measurement checked against it ambiguous. Nothing forced the two to
// agree: the drift was silent, need not have been atomic, and no test failed.
//
// A name written twice drifts twice. A name derived once cannot.
//
// The refusal is inherited rather than re-invented: a storage code no enumerator names
// throws, exactly as kv_storage_token does, instead of returning a name that looks like
// a tier. tests/test_kv_operator_name.cpp is the bar that keeps this and the table
// equal -- by literal pin, by derivation, by totality, and by a text census over the
// CLI sources that catches a re-introduced copy.
[[nodiscard]] inline std::string kv_operator_token(KvCacheStorage storage) {
    return std::string(kv_storage_token(storage));
}

} // namespace ninfer::product
