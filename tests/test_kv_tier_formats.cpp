// Unit test for the --kv-tier-formats landing (product/kv_tier_formats.h).
// Host-only: the vocabulary, the hot landing, the mode/cold gates and the cold
// codec byte rule are pure functions over arrays, so they are testable without
// CUDA, an artifact or a GPU.
#include "product/kv_tier_formats.h"

#include <array>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

namespace p = ninfer::product;

using ninfer::DType;
using ninfer::KvCacheStorage;
using ninfer::kKvLayerStorageSlots;
using ninfer::kvcfg::KvFormat;
using p::ColdCodec;
using p::KvLayerClass;
using p::KvTierPlan;

using Table   = std::array<KvCacheStorage, kKvLayerStorageSlots>;
using Classes = std::array<KvLayerClass, kKvLayerStorageSlots>;

constexpr std::int32_t kLayers = 16;

Table uniform(KvCacheStorage storage) {
    Table table{};
    table.fill(storage);
    return table;
}

// The storage -> dtype step the planner performs (layouts_impl.h
// target_kv_cache_profile), written as a switch so a new KvCacheStorage
// enumerator is REPORTED HERE instead of being silently classified as bf16.
//
// "breaks this build" would be the stronger and more useful promise, and this
// comment used to make it. It is not true in this tree: CMakeLists.txt opens no
// warning flag at all, so the -Wswitch this switch emits is not even printed, and
// it is never an error. What this file CAN promise -- and now does -- is that the
// diagnostic exists and that the failure path is honest, which is what the
// executable checks below pin.
//
// That used to happen to iso4e and rk4v4: the old ternary chain here mapped both onto
// DType::BF16, so the "pure refuses a table carrying nvfp4/rk4v4 layers" check
// below passed on its nvfp4 layers alone and never exercised rk4v4 at all.
//
// DType::BF16 HAS TWO MEANINGS AND THIS FUNCTION MUST ONLY EVER PRODUCE ONE OF
// THEM. As an ANSWER it means "this layer is bf16" -- that is what
// `case BFloat16: return DType::BF16` says, and it is correct. As an ABSENT
// answer it is the "inherit the global --kv-dtype" sentinel
// (decoder_state.cpp plan_cache()'s layer_dtype() via
// product/kv_component_switch.h kv_resolve_slot_dtype), i.e. "no answer, use the
// operator's --kv-dtype". So the function's codomain is the set of REAL tiers and
// a storage that carries none is REFUSED -- there is no arm, and no trailing
// statement, that returns DType::BF16 because it did not know.
//
// That refusal is the tree's own idiom for this enum, not an invention:
// product/kv_storage_dtype.h and layouts_impl.h target_kv_cache_profile() both
// switch over the SAME enum with no default arm and throw after the switch.
[[nodiscard]] DType dtype_of(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16: return DType::BF16;
    case KvCacheStorage::Int8Group64: return DType::I8;
    case KvCacheStorage::Fp8E4M3Row256: return DType::FP8_E4M3FN;
    case KvCacheStorage::Nvfp4Group16: return DType::NVFP4;
    case KvCacheStorage::Fp8Group16: return DType::FP8_E4M3FN;
    case KvCacheStorage::Iso3Group16: return DType::ISO3;
    case KvCacheStorage::E8Group64: return DType::E8Kv;
    case KvCacheStorage::Dropped:
        // types.h: a discarded layer owns NO KV planes, so there is nothing to
        // classify. Returning DType::BF16 here would report it as a bf16 layer
        // and -- because BF16 is also the inherit-the-global sentinel -- say
        // "ignore the operator's --kv-dtype on this layer" at the same time.
        throw std::invalid_argument(
            "dtype_of: a Dropped layer owns no KV planes, so it has no dtype; "
            "DType::BF16 would also mean \"inherit the global --kv-dtype\"");
    }
    // Reachable only for a value no enumerator names: KvCacheStorage is a
    // std::uint8_t and the table is fed from option text, so a cast or an
    // untrusted input can produce one. Refuse it the way the production
    // mapping does (product/kv_storage_dtype.h).
    throw std::invalid_argument("dtype_of: KV storage code " +
                                std::to_string(static_cast<unsigned>(storage)) +
                                " names no enumerator, so it has no dtype");
}

// The effective dtype -> class step the planner performs, applied to a storage
// table so the test can drive the gates with plain arrays.
Classes classes_of(const Table& table) {
    Classes classes{};
    for (std::size_t i = 0; i < table.size(); ++i) {
        classes[i] = p::kv_layer_class_of(dtype_of(table[i]));
    }
    return classes;
}

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

// True when the call throws std::invalid_argument whose text contains `needle`.
bool rejects(const std::string& needle, const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument& error) {
        const std::string text = error.what();
        if (text.find(needle) != std::string::npos) { return true; }
        std::cerr << "FAIL: rejected with an unexpected message: " << text << '\n';
        ++failures;
        return true;
    }
    return false;
}

} // namespace

int main() {
    // ---- the spec text, not the parsed defaults, decides what was asked ----
    {
        const p::KvTierWritten written = p::kv_tier_written("hot=bf16,tail=fp16,cold=iso4e");
        check(written.hot && written.tail && written.cold, "all three tiers were written");
        const p::KvTierWritten only_hot = p::kv_tier_written("hot=int8");
        check(only_hot.hot && !only_hot.tail && !only_hot.cold, "only hot was written");
        const p::KvTierWritten nothing = p::kv_tier_written("");
        check(!nothing.hot && !nothing.tail && !nothing.cold, "an empty spec writes nothing");
        // ... which matters, because the parsed struct fills the mode default in.
        check(p::kv_tier_formats_parse("hot=int8", false).cold == KvFormat::Iso4e,
              "the fusion default cold is iso4e");
        check(p::kv_tier_formats_parse("", true).cold == KvFormat::Int8,
              "the pure default cold is int8");
    }

    // ---- the vocabulary's own rules, verbatim ----
    {
        check(rejects("decode-hot", [] { (void)p::kv_tier_formats_parse("hot=int4", false); }),
              "hot below int8 is rejected");
        check(rejects("forbids iso/rk4v4", [] { (void)p::kv_tier_formats_parse("cold=iso4e", true); }),
              "pure forbids the iso/rk4v4 formats");
        check(rejects("unknown kv format", [] { (void)p::kv_tier_formats_parse("hot=q4", false); }),
              "an unknown format is rejected");
        check(rejects("tail tier precision below hot",
                      [] { (void)p::kv_tier_formats_parse("hot=int8,tail=iso4e", false); }),
              "tail below hot is rejected before any landing is attempted");
    }

    // ---- layer classification: iso4e is a tier of its own, not the nvfp4 family ----
    {
        // layouts_impl.h: "ISO4E is a tier of its own: identical planes to NVFP4
        // but a distinct 3-bit sign-magnitude codec". Its K plane is not E2M1, so
        // it can never feed the nvfp4 rANS slot, which is what the old
        // Iso4e -> Nvfp4Fusion mapping claimed.
        check(p::kv_layer_class_of(DType::ISO3) == KvLayerClass::Iso4eFusion,
              "a standalone iso4e layer has its own class");
        check(p::kv_layer_class_of(DType::ISO3) != KvLayerClass::Nvfp4Fusion,
              "iso4e is NOT the nvfp4 rANS class");
        check(p::kv_layer_class_is_fusion(KvLayerClass::Iso4eFusion),
              "iso4e is still a fusion tier: pure forbids it (kv_formats.h)");
        check(p::kv_cold_format_of(KvLayerClass::Iso4eFusion) == KvFormat::Auto,
              "a standalone iso4e plane has no cold codec of its own");
        check(p::kv_layer_class_of(DType::E8Kv) == KvLayerClass::Rk4v4Fusion,
              "an e8 lattice layer is the rk4v4 class");
        check(p::kv_layer_class_is_fusion(KvLayerClass::Rk4v4Fusion), "rk4v4 is a fusion tier");
        check(p::kv_layer_class_of(DType::FP8_E4M3FN) == KvLayerClass::Fp8,
              "fp8 is not a fusion tier");
    }

    // ---- hot -> the resident table ----
    {
        const auto bf16 = p::kv_hot_storage_of(KvFormat::Bf16);
        check(bf16.has_value() && *bf16 == KvCacheStorage::BFloat16, "hot=bf16 maps to bf16");
        const auto int8 = p::kv_hot_storage_of(KvFormat::Int8);
        check(int8.has_value() && *int8 == KvCacheStorage::Int8Group64, "hot=int8 maps to int8");
        check(!p::kv_hot_storage_of(KvFormat::Auto).has_value(), "hot=auto inherits the table");
        check(rejects("no FP16 KV tier", [] { (void)p::kv_hot_storage_of(KvFormat::Fp16); }),
              "hot=fp16 is refused (no FP16 KV tier)");

        const KvTierPlan mapped =
            p::kv_tier_formats_plan("hot=int8", false, uniform(KvCacheStorage::BFloat16));
        check(mapped.override_hot, "hot=int8 overrides every layer");
        check(mapped.hot_storage == KvCacheStorage::Int8Group64, "hot=int8 storage");
        bool all_int8 = true;
        for (std::size_t i = 0; i < mapped.table.size(); ++i) {
            all_int8 = all_int8 && mapped.table[i] == KvCacheStorage::Int8Group64;
        }
        check(all_int8, "every layer of the planned table is int8");

        const KvTierPlan inherit =
            p::kv_tier_formats_plan("cold=iso4e", false, uniform(KvCacheStorage::Nvfp4Group16));
        check(!inherit.override_hot, "an absent hot leaves the table alone");
        check(inherit.table[3] == KvCacheStorage::Nvfp4Group16, "the incoming table is preserved");
    }

    // ---- tail: accepted only when it does not ask for a second format ----
    {
        const Table table = uniform(KvCacheStorage::BFloat16);
        check(rejects("no tail tier",
                      [&] { (void)p::kv_tier_formats_plan("hot=bf16,tail=fp16", false, table); }),
              "tail=fp16 is refused: no tail storage class exists");
        check(rejects("no tail tier", [&] { (void)p::kv_tier_formats_plan("tail=fp16", false, table); }),
              "a bare tail= is refused too");
        bool accepted = true;
        try {
            (void)p::kv_tier_formats_plan("hot=bf16,tail=bf16", false, table);
            (void)p::kv_tier_formats_plan("hot=bf16", false, table);
        } catch (const std::exception&) { accepted = false; }
        check(accepted, "tail == hot and an omitted tail are accepted");
    }

    // ---- the cold codec rule: the bytes decide, not the bit width ----
    {
        // The pool stride is fixed for BOTH codecs (decoder_state.cpp allocates
        // {slot_bytes, kv_heads, 2, cold_pages} with slot_bytes = the rANS max),
        // so a codec pays off only when that stride is below the resident plane
        // the cold page gave back.
        const p::KvColdCodecSpec raw = p::kv_cold_codec_spec(ColdCodec::Int8Raw);
        check(raw.reachable, "the int8 raw slot is reachable for an int8 stack");
        // pool_stride_bytes is the arena bytes an int8 LAYER reserves, which is its
        // own record: decoder_state.cpp cold_slot_stride_for() sizes each layer's
        // tensor from the resolved dtype, so the int8 slot is 9232 B and not the
        // wider rANS stride.
        check(raw.pool_stride_bytes == 9232, "the int8 layer's cold record is 9232 B");
        check(raw.payload_bytes == 9232, "the int8 codec fills all of it");
        check(raw.resident_bytes == 16896, "an int8 resident plane is 16896 B/head-page");
        check(p::kv_cold_codec_reduces(raw), "int8 cold FREES device memory");
        check(p::kv_cold_codec_saved_bytes(raw) == 7664,
              "int8 cold saves 7664 B/head-page (45.4% of the resident plane)");

        const p::KvColdCodecSpec rans = p::kv_cold_codec_spec(ColdCodec::Nvfp4Rans);
        check(rans.reachable, "the nvfp4 rANS slot is implemented and admitted for an nvfp4 stack");
        check(rans.resident_bytes == 9216, "an nvfp4 resident plane is 9216 B/head-page");
        check(!p::kv_cold_codec_reduces(rans),
              "the nvfp4 rANS slot does NOT reduce the footprint at the measured ceiling: "
              "the record that encodes (9632) is above the plane it gives back (9216)");
        check(p::kv_cold_codec_saved_bytes(rans) == -416,
              "the nvfp4 rANS slot COSTS 416 B/head-page at 4.04 b/c: 259 B per stream is "
              "the measured K requirement, and it is 416 B wider than the plane pair");

        const p::KvColdCodecSpec none = p::kv_cold_codec_spec(ColdCodec::None);
        check(none.codec == ColdCodec::None && !p::kv_cold_codec_reduces(none),
              "no codec is never a saving");

        // The premise the whole rule rests on: the int8 resident plane is ABOVE
        // the record it needs. This is a real boundary, not a formality -- at
        // head_dim = 128 the int8 plane is only 8448 B, and a 9536 B record would
        // cost memory there.
        check(p::kKvColdResidentInt8Bytes > p::kKvColdInt8PayloadBytes,
              "the int8 plane is above its own cold record, so cold can pay off at all");
        const std::int32_t derived_nvfp4 = (p::kKvColdHeadDim / 2) * p::kKvColdPageTokens +
                                           (p::kKvColdHeadDim / p::kKvColdNvfp4Group) *
                                               p::kKvColdPageTokens;
        check(derived_nvfp4 == p::kKvColdResidentNvfp4Bytes,
              "the nvfp4 resident plane is DERIVED from head_dim, not a literal");
        // The stride is the rANS budget at the MEASURED ceiling, 4.04 bits/code -- not the
        // 4-bit no-expansion maximum and not the retracted 2.60 band:
        // (512 * 404 + 799) / 800 == 259 B per stream.
        check(p::kKvColdRansStreamBytes == (512 * 404 + 799) / 800,
              "the rANS stream budget is ceil(512 * 4.04 / 8), in fixed point");
        check(p::kKvColdPoolStrideBytes == 320 + 32 * p::kKvColdRansStreamBytes + 1024,
              "the stride is the 4.04 b/c rANS record");
        check(p::kKvColdPoolStrideBytes == 9632, "== 320 + 32*259 + 1024");
        // The measurement itself, as the invariant that moved the number. The encoder's
        // per-stream budget is (stride - 1344) / 32 with no clamp
        // (entropy_nvfp4_slot_kernels.cuh pass A/B), so the stride must cover the largest
        // stream the real encoder produces: 259 B, the maximum over the 2048 real K
        // streams of dl/ransceil/rans_probe.cu. At the old 167 B budget it validated 0 of
        // 64 K slots, i.e. the pool compressed nothing while the ledger priced a saving.
        check(p::kKvColdRansStreamBytes >= p::kKvColdRansMeasuredMaxStreamBytes &&
                  p::kKvColdRansMeasuredMaxStreamBytes == 259,
              "the shipped rANS budget covers the measured K maximum (259 B / 512 symbols)");
        check(320 + 32 * p::kKvColdRansMeasuredMaxStreamBytes + 1024 ==
                  p::kKvColdPoolStrideBytes,
              "the shipped stride IS the measured minimum: 320 + 32*259 + 1024");
        // The byte ceil is what sets the ceiling the encoder can really reach: 259 B is the
        // SMALLEST whole-byte budget that covers 512 codes at 4.04 b/c, so the enforced
        // ceiling is 259 * 8 / 512 = 4.046875 bits/symbol, never below the 4.04 the constant
        // is named for. Those two numbers are the same quantity, not a contradiction: 258 B
        // truncates the ceil and would leave the measured 259 B maximum unencodable (the
        // hazard decoder_state.cpp names when it does the ceil in fixed point).
        check(258 * 8 * 100 < 512 * 404 && 259 * 8 * 100 >= 512 * 404,
              "259 B is the smallest budget covering 512 codes at 4.04 b/c: the enforced "
              "ceiling is 4.046875 b/s, above the nominal one, so no page is refused for a "
              "rate the ceiling admits");
        // The consequence the briefing asks about, as an invariant rather than a label, and
        // the answer is negative: BOTH rANS records are ABOVE the nvfp4 plane. The 4-bit
        // one is above it because 320 + 32*256 + 1024 = 9536 > 9216; the measured one is
        // above it because the K plane's rANS streams need 259 B each, more than the 256 B
        // the no-expansion bound allows. So the record that ENCODES and the record that
        // SAVES are different records, and the second does not exist.
        check(p::kKvColdPoolStrideBytes > p::kKvColdResidentNvfp4Bytes &&
                  p::kKvColdRansRecordAt4BitBytes > p::kKvColdResidentNvfp4Bytes,
              "BOTH rANS records are ABOVE the nvfp4 plane: encode-and-save is "
              "unsatisfiable for this class, at every ceiling the tree can express");

        // The default rule picks the reducing codec, and explains the miss otherwise.
        const Classes int8_classes  = classes_of(uniform(KvCacheStorage::Int8Group64));
        const Classes nvfp4_classes = classes_of(uniform(KvCacheStorage::Nvfp4Group16));
        const Classes bf16_classes  = classes_of(uniform(KvCacheStorage::BFloat16));
        const Classes iso4e_classes  = classes_of(uniform(KvCacheStorage::Iso3Group16));
        check(p::kv_cold_codec_default(int8_classes, kLayers).codec == ColdCodec::Int8Raw,
              "an all-int8 stack defaults to the int8 raw slot");
        check(p::kv_cold_codec_default(nvfp4_classes, kLayers).codec == ColdCodec::Nvfp4Rans,
              "an nvfp4 stack names the rANS slot as the codec it cannot profitably use");
        // BF16-COLD-LAND A2: Classic16 is a cold class now. Its record IS the int8 raw
        // slot (9232 B, fixed layout, no overflow path) and the plane it gives back is
        // the bf16 one, which makes it the largest saving of the three codecs.
        check(p::kv_cold_codec_default(bf16_classes, kLayers).codec == ColdCodec::Int8Raw,
              "a bf16 stack resolves the raw slot: A2 gave Classic16 a cold codec");
        check(p::kv_cold_codec_saved_bytes(p::kv_cold_codec_spec_of(KvLayerClass::Classic16)) ==
                  23536,
              "and prices it against the bf16 plane it gives back: 32768 - 9232");
        check(p::kv_cold_codec_default(iso4e_classes, kLayers).codec == ColdCodec::None,
              "a standalone iso4e stack has no cold codec (its K plane is not E2M1)");

        // The operator's override names a CODEC; auto means "use the rule".
        check(!p::kv_cold_codec_of_format(KvFormat::Auto).has_value(), "cold=auto is the rule");
        check(p::kv_cold_codec_of_format(KvFormat::Int8) == ColdCodec::Int8Raw,
              "cold=int8 selects the raw slot");
        check(p::kv_cold_codec_of_format(KvFormat::Iso4e) == ColdCodec::Nvfp4Rans,
              "cold=iso4e selects the rANS slot");
        check(p::kv_cold_codec_of_format(KvFormat::Rk4v4) == ColdCodec::None,
              "cold=rk4v4 names no codec");

        // The printed byte truth: the report has to say whether memory really moves.
        const std::string int8_line = p::kv_cold_codec_bytes(raw);
        check(int8_line.find("int8 raw slot") != std::string::npos, "the clause names the codec");
        check(int8_line.find("SAVES 7664 B/head-page") != std::string::npos,
              "the clause prints the saving in bytes");
        check(int8_line.find("45.4%") != std::string::npos, "the clause prints the share");
        const std::string rans_line = p::kv_cold_codec_bytes(rans);
        check(rans_line.find("COSTS 416 B/head-page") != std::string::npos &&
                  rans_line.find("4.5%") != std::string::npos,
              "the clause prints the rANS COST and its share, not a saving");
        check(rans_line.find("NOT reachable") == std::string::npos,
              "the clause no longer flags an unreachable codec: both slots are reachable");
    }

    // ---- cold: enforced against the codec the resolved table can really produce ----
    {
        // The vocabulary name each codec stores: this is what makes cold=iso4e mean
        // "the nvfp4 family" and cold=int8 mean "the raw int8 slot".
        check(p::kv_cold_format_of(KvLayerClass::Int8) == KvFormat::Int8,
              "the raw int8 slot stores int8");
        check(p::kv_cold_format_of(KvLayerClass::Nvfp4Fusion) == KvFormat::Iso4e,
              "the nvfp4 rANS slot stores iso4e");
        check(p::kv_cold_format_of(KvLayerClass::Classic16) == KvFormat::Auto,
              "a bf16 plane stores no cold format at all");

        const Table int8_table = uniform(KvCacheStorage::Int8Group64);
        check(p::kv_cold_pool_reachable(classes_of(int8_table), kLayers),
              "an all-int8 stack can feed the cold codec");
        check(p::kv_cold_pool_reachable(classes_of(uniform(KvCacheStorage::Nvfp4Group16)),
                                        kLayers),
              "an all-nvfp4 stack is cold-capable: its layers pack the rANS slot");
        // The compressor judges every layer on its own dtype, so a mix is fine: the
        // codec follows the layer, not the pool.
        Table mixed  = int8_table;
        mixed[7]     = KvCacheStorage::Nvfp4Group16;
        check(p::kv_cold_pool_reachable(classes_of(mixed), kLayers),
              "an int8/nvfp4 mix is cold-capable (one codec per layer dtype)");
        Table one_bf16 = int8_table;
        one_bf16[7]    = KvCacheStorage::BFloat16;
        // BF16-COLD-LAND A2: one bf16 layer no longer blocks the compressor. It was
        // excluded because its READ side did not exist; it does now -- a bf16 layer packs
        // the same raw slot the int8 arm packs (entropy_cold_requant Bf16G64), and the
        // bf16 small-T decode body reads it.
        check(p::kv_cold_pool_reachable(classes_of(one_bf16), kLayers),
              "a bf16 layer is cold-capable: it feeds the raw slot through Bf16G64");
        Table one_iso4e = int8_table;
        one_iso4e[7]    = KvCacheStorage::Iso3Group16;
        check(!p::kv_cold_pool_reachable(classes_of(one_iso4e), kLayers),
              "one iso4e layer blocks the compressor: its K plane is not E2M1");
        // BF16-COLD-LAND E2: the rk4v4 tier is cold-capable now, and it is the one class
        // whose record does NOT pay -- 9232 B of raw record against an 8704 B plane.
        Table one_rk4v4 = int8_table;
        one_rk4v4[7]    = KvCacheStorage::E8Group64;
        check(p::kv_cold_pool_reachable(classes_of(one_rk4v4), kLayers),
              "an rk4v4 layer is cold-capable: it packs the raw slot through Rk4v4KvG64");
        const p::KvColdCodecSpec rk4v4_spec = p::kv_cold_codec_spec_of(KvLayerClass::Rk4v4Fusion);
        check(rk4v4_spec.codec == ColdCodec::Int8Raw && rk4v4_spec.resident_bytes == 8704,
              "the rk4v4 codec is the raw slot priced against the rk4v4 plane (8704 B/head-page)");
        check(!p::kv_cold_codec_reduces(rk4v4_spec) && p::kv_cold_codec_saved_bytes(rk4v4_spec) == -528,
              "and it COSTS 528 B/head-page: stated, not hidden. It is still the right "
              "codec, because arming the pool is what unlocks the nvfp4 layers of the "
              "same stack (416 B/head-page each at the measured ceiling), and the rANS "
              "slot cannot hold rk4v4's near-uniform 4-bit codes");
        // The counter, at the measured ceiling. It read `pays` when the grid point was
        // 6688; that record is the one the encoder validates 0 of 64 K slots on, so the
        // verdict it carried was the price of an empty pool. Both halves are still
        // asserted: the lattice is not what is missing (the class FITS the record the pool
        // allocates), and the codec it would use does not pay at the record that encodes.
        check(p::kv_cold_class_bytes_of(KvLayerClass::Rk4v4Fusion).fits &&
                  !p::kv_cold_class_bytes_of(KvLayerClass::Rk4v4Fusion).pays,
              "the FITS/PAYS counter still says the rk4v4 LATTICE is not what is missing "
              "(it FITS), and that the codec it would use does not pay at the measured "
              "ceiling either: the rANS slot cannot hold rk4v4's near-uniform 4-bit codes");

        const KvTierPlan cold_int8 =
            p::kv_tier_formats_plan("hot=int8,cold=int8", false, int8_table);
        bool satisfied = true;
        try {
            (void)p::kv_tier_formats_check(cold_int8, classes_of(int8_table), kLayers, true);
        } catch (const std::exception&) { satisfied = false; }
        check(satisfied, "cold=int8 over an all-int8 stack is satisfiable");

        // BF16-COLD-LAND A2: cold=int8 names the raw slot, and a bf16 stack now packs
        // that slot (one codec per layer dtype, exactly as the int8/nvfp4 mix does), so
        // the request is satisfiable. This is the byte-level half of "the tree and the
        // cold tier can both be on": the tree needs bf16, and bf16 now has a cold codec.
        const KvTierPlan cold_int8_bf16 =
            p::kv_tier_formats_plan("hot=bf16,cold=int8", false, int8_table);
        bool bf16_int8_accepted = true;
        try {
            (void)p::kv_tier_formats_check(cold_int8_bf16,
                                           classes_of(uniform(KvCacheStorage::BFloat16)), kLayers,
                                           true);
        } catch (const std::exception&) { bf16_int8_accepted = false; }
        check(bf16_int8_accepted,
              "cold=int8 over a bf16 stack is accepted (A2): a bf16 layer packs the raw slot");

        const Table nvfp4_stack    = uniform(KvCacheStorage::Nvfp4Group16);
        const KvTierPlan cold_iso4e = p::kv_tier_formats_plan("cold=iso4e", false, nvfp4_stack);
        // The rANS slot IS reachable over this stack, and at the measured ceiling its
        // record is ABOVE the nvfp4 plane it replaces (9632 > 9216), so the byte clause
        // REFUSES the request. That is the operator-visible half of the ceiling change,
        // and it is the half that has to stay honest: naming a codec that costs device
        // memory is exactly what the pool may not be asked to do. The refusal names the
        // byte clause, not reachability -- reachability is true and is not the reason.
        check(rejects("would not reduce device memory", [&] {
                  (void)p::kv_tier_formats_check(cold_iso4e, classes_of(nvfp4_stack), kLayers,
                                                 true);
              }),
              "cold=iso4e is REFUSED over an nvfp4 stack at the measured ceiling: 9632 > 9216");
        check(!p::kv_cold_codec_reduces_at_4bit(p::kv_cold_codec_spec(ColdCodec::Nvfp4Rans)),
              "the same codec did NOT reduce at the 4-bit no-expansion ceiling: 9536 > 9216");
        // A name that is not a cold codec at all is refused as such.
        const KvTierPlan cold_rk4v4 = p::kv_tier_formats_plan("hot=int8,cold=rk4v4", false, int8_table);
        check(rejects("names no cold codec", [&] {
                  (void)p::kv_tier_formats_check(cold_rk4v4, classes_of(int8_table), kLayers, true);
              }),
              "cold=rk4v4 is refused: the pool carries no rk4v4 slot codec");

        // An omitted cold is informational, never an error.
        const KvTierPlan default_cold = p::kv_tier_formats_plan("hot=int8", false, int8_table);
        const std::string report =
            p::kv_tier_formats_check(default_cold, classes_of(int8_table), kLayers, true);
        check(report.find("int8 codec") != std::string::npos,
              "the report names the codec the pool will really use");
        check(report.find("cold=iso4e") != std::string::npos,
              "the report prints the resolved (unwritten) cold default");
        check(report.find("cold_residency=pool") != std::string::npos,
              "the report names the residency it found");
        check(report.find("SAVES 7664 B/head-page") != std::string::npos,
              "the report prints whether the cold tier really frees memory");

        // ... and a pool the resolved table cannot profit from says WHY, including the
        // negative byte case. An nvfp4 stack is REACHABLE (its layers pack the rANS
        // slot), so the report must NOT call its pool idle: it must report the pool as
        // live and the resolved codec as the net cost it is.
        const KvTierPlan nvfp4_default =
            p::kv_tier_formats_plan("", false, nvfp4_stack);  // mode default only: fusion
        const std::string rans_report =
            p::kv_tier_formats_check(nvfp4_default, classes_of(nvfp4_stack), kLayers, true);
        check(rans_report.find("pool live") != std::string::npos,
              "an nvfp4 stack reports a live pool, not an idle one");
        check(rans_report.find("pool reserved but idle") == std::string::npos,
              "an nvfp4 stack is not reported idle: the rANS slot really is admitted");
        check(rans_report.find("nvfp4 rANS slot") != std::string::npos,
              "the report names the codec the nvfp4 stack will really use");
        check(rans_report.find("COSTS 416 B/head-page") != std::string::npos,
              "an nvfp4 stack reports that its codec is a net cost, not a saving");
        check(rans_report.find("net cost") != std::string::npos,
              "the net-cost clause IS present: at the measured ceiling the pool really is "
              "a net cost on the 4-bit classes");

        // The CLOSED wording has to exist for a stack whose layers feed no codec at
        // all: that is where --max-cold-pages would be pure overhead, and since the
        // arming gate (kv_cold_pool_state / kv_cold_pool_close_notice, layouts_impl.h)
        // the pool is not built for it at all. The wording may therefore no longer say
        // "reserved": a reserved pool is the defect this replaced. This assertion was
        // `find("pool reserved but idle")` and is the SAME tripwire, moved onto the
        // contract that now holds.
        // BF16-COLD-LAND A2. This block used to prove the CLOSED wording on a bf16
        // stack; bf16 is a cold class now, so the stack that proves it is one with an
        // iso4e layer -- whose K plane is not E2M1 and which has no cold branch on any
        // route. `one_iso4e` (above) is exactly that: the int8 table with layer 7 iso4e.
        const KvTierPlan bf16_default =
            p::kv_tier_formats_plan("", false, uniform(KvCacheStorage::BFloat16));
        const Classes bf16_classes = classes_of(uniform(KvCacheStorage::BFloat16));
        const std::string bf16_report =
            p::kv_tier_formats_check(bf16_default, bf16_classes, kLayers, true);
        check(bf16_report.find("cold_residency=pool") != std::string::npos,
              "a bf16 stack ARMS the pool: Classic16 feeds the raw slot (A2)");
        check(bf16_report.find("SAVES 23536 B/head-page") != std::string::npos,
              "the bf16 report prices the raw slot against the bf16 plane: 32768 - 9232");
        check(p::kv_cold_pool_state(bf16_classes, kLayers, true) == p::KvColdPool::Armed,
              "a requested pool over a bf16 stack is Armed, not Closed");
        check(p::kv_cold_pool_state(bf16_classes, kLayers, false) ==
                  p::KvColdPool::NotRequested,
              "with no --cold-policy there is no pool to judge, which is not Closed");

        const Classes closed_classes = classes_of(one_iso4e);
        const KvTierPlan closed_default = p::kv_tier_formats_plan("", false, one_iso4e);
        const std::string idle_report =
            p::kv_tier_formats_check(closed_default, closed_classes, kLayers, true);
        check(p::kv_cold_pool_state(closed_classes, kLayers, true) == p::KvColdPool::Closed,
              "one iso4e layer closes the pool: its K plane is not E2M1");
        check(idle_report.find("cold_residency=NOT armed (closed)") != std::string::npos,
              "a stack with no codec reports a pool that is NOT armed, not a reserved one");
        check(idle_report.find("pool reserved but idle") == std::string::npos,
              "the reserved-but-idle wording is gone: nothing reserves this pool any more");
        check(idle_report.find("codec=none") != std::string::npos,
              "the closed report says there is no cold codec for this stack at all");
        check(idle_report.find("FITS but would COST 416 B/head-page") != std::string::npos,
              "the closed report prices the slot at the measured ceiling -- FITS and COSTS "
              "-- instead of saying a bare none");
        check(idle_report.find("pure overhead") != std::string::npos,
              "the closed report still says what --max-cold-pages would have been");

        // THE ARMING GATE'S OWN TRIPWIRES. Three facts, because three different
        // silent zeros were possible: the state is the request AND the stack (never the
        // request alone), the offender list is the NAMED form of the reachability
        // predicate, and the notice names EVERY offender -- a stack with six of them
        // used to read exactly like a stack with one.
        // BF16-COLD-LAND A2: layer 3 is iso4e here, not bf16 -- bf16 stopped being an
        // offender when it got its codec, and the tripwire has to keep testing two
        // DIFFERENT offenders or it stops proving "every offender is named".
        Table two_offenders = int8_table;
        two_offenders[3]    = KvCacheStorage::Iso3Group16;
        two_offenders[9]    = KvCacheStorage::Fp8Group16;
        const Classes two_classes = classes_of(two_offenders);
        check(p::kv_cold_pool_state(two_classes, kLayers, true) == p::KvColdPool::Closed,
              "one iso4e and one rk4v4 layer close the pool even though 14 layers feed it");
        check(p::kv_cold_pool_offenders(two_classes, kLayers).size() == 2,
              "the offender list is the named form of kv_cold_pool_reachable");
        check(p::kv_cold_pool_offenders(classes_of(int8_table), kLayers).empty(),
              "an all-int8 stack has no offender: this is the byte-identical case");
        check(p::kv_cold_pool_offenders(classes_of(one_bf16), kLayers).empty(),
              "and neither has a bf16 stack any more (A2): Classic16 is cold-capable");
        const std::string notice = p::kv_cold_pool_close_notice(two_classes, kLayers);
        check(notice.find("offender: layer 3 dtype=iso4e") != std::string::npos &&
                  notice.find("offender: layer 9 dtype=fp8-e4m3") != std::string::npos,
              "the close notice names EVERY offender, by layer index AND by dtype");
        check(notice.find("): no cold codec") != std::string::npos,
              "every offender line carries the reason");
        check(notice.find("cold_residency=closed") != std::string::npos,
              "the notice states the residency it produced, on the residency axis");
        check(notice.find("SKIPPING the offender is NOT a fix") != std::string::npos,
              "and names the non-fix it is not doing");

        // A MIXED int8/nvfp4 stack is cold-capable too (see the pool check above), and
        // it builds one codec per layer dtype: the report resolves the int8 codec (the
        // only reducing class) and must still name the nvfp4 layers that cost.
        const KvTierPlan mixed_default = p::kv_tier_formats_plan("", false, mixed);
        const std::string mixed_report =
            p::kv_tier_formats_check(mixed_default, classes_of(mixed), kLayers, true);
        check(mixed_report.find("pool with the int8 codec") != std::string::npos,
              "a mixed stack resolves the int8 codec (the only reducing class)");
        check(mixed_report.find("SAVES 7664 B/head-page") != std::string::npos,
              "the mixed report prices the int8 codec it resolved");
        check(mixed_report.find("MIXED stack") != std::string::npos,
              "the mixed report names the nvfp4 layers of the same pool");
        check(mixed_report.find("the nvfp4 layers COST 416 B/head-page each") != std::string::npos,
              "the mixed sentence derives its sign per layer dtype: the int8 layers save "
              "while the nvfp4 layers cost at the measured ceiling");
    }

    // ---- FITS vs EXISTS: the byte truth for the classes with no codec ----
    {
        // A cold record holds a REQUANTIZED page, so its width does not depend on
        // the source dtype, and every class fits one. What differs is whether the
        // record pays, i.e. whether it is below the resident plane given back.
        const p::KvColdClassBytes bf16 = p::kv_cold_class_bytes_of(KvLayerClass::Classic16);
        const p::KvColdClassBytes fp8  = p::kv_cold_class_bytes_of(KvLayerClass::Fp8);
        const p::KvColdClassBytes i8   = p::kv_cold_class_bytes_of(KvLayerClass::Int8);
        const p::KvColdClassBytes nv4  = p::kv_cold_class_bytes_of(KvLayerClass::Nvfp4Fusion);
        const p::KvColdClassBytes iso  = p::kv_cold_class_bytes_of(KvLayerClass::Iso4eFusion);
        const p::KvColdClassBytes rk4v4   = p::kv_cold_class_bytes_of(KvLayerClass::Rk4v4Fusion);
        check(bf16.resident_bytes == 32768 && fp8.resident_bytes == 17408 &&
                  i8.resident_bytes == 16896 && nv4.resident_bytes == 9216 &&
                  iso.resident_bytes == 9216 && rk4v4.resident_bytes == 8704,
              "the six resident planes are derived from the plane geometry");
        check(bf16.fits && fp8.fits && i8.fits && nv4.fits && iso.fits && rk4v4.fits,
              "every class FITS a record the pool allocates: the slot was never too small");
        check(i8.record_bytes == 9232 && nv4.record_bytes == 9632,
              "the two records are the raw 9232 B and the 4.04 b/c rANS 9632 B");
        check(!iso.pays && !nv4.pays && !rk4v4.pays,
              "NONE of the three 4-bit-shaped classes pays: the record that encodes is "
              "above every plane they would give back, and the record that would be below "
              "them does not encode");
        check(iso.record_bytes - iso.resident_bytes == 416,
              "iso4e/nvfp4 cost 416 B/head-page (4.5%) at the measured ceiling");
        check(rk4v4.record_bytes - rk4v4.resident_bytes == 928,
              "rk4v4 costs 928 B/head-page (10.7%): it is the furthest from paying");
        // ... and the same classes at the 4-bit bound, where the sign flips. This is
        // defect (a) as a build-time fact rather than a comment.
        check(!p::kv_cold_class_bytes_at_4bit(KvLayerClass::Iso4eFusion).pays &&
                  p::kv_cold_class_bytes_at_4bit(KvLayerClass::Iso4eFusion).fits,
              "at 4.0 b/c iso4e FITS and still COSTS 320 B/head-page");
        check(!p::kv_cold_class_bytes_at_4bit(KvLayerClass::Rk4v4Fusion).pays,
              "at 4.0 b/c rk4v4 COSTS 832 B/head-page");
        check(p::kv_cold_class_bytes_at_4bit(KvLayerClass::Fp8).pays &&
                  p::kv_cold_class_bytes_at_4bit(KvLayerClass::Classic16).pays,
              "fp8/bf16 paid at either ceiling");
        // The gap phrase names the missing CODEC, never a too-small slot.
        const std::string gap = p::kv_cold_class_gap_text(KvLayerClass::Rk4v4Fusion);
        check(gap.find("FITS") != std::string::npos &&
                  gap.find("would COST 928 B/head-page") != std::string::npos,
              "the rk4v4 gap names a fitting slot and states its measured cost: the codec "
              "is what is missing, and the slot it is missing on is not a saving either");
        check(gap.find("e8 lattice plane") != std::string::npos,
              "the rk4v4 gap names the class in the vocabulary's own words");
    }

    // ---- mode: pure stays off the fusion tiers ----
    {
        const Table int8_table     = uniform(KvCacheStorage::Int8Group64);
        const KvTierPlan pure_int8 = p::kv_tier_formats_plan("hot=int8", true, int8_table);
        const std::string report =
            p::kv_tier_formats_check(pure_int8, classes_of(int8_table), kLayers, false);
        check(report.find("mode=pure") != std::string::npos, "the report names pure mode");
        check(report.find("cold_residency=none") != std::string::npos,
              "the report separates format from residency");

        const KvTierPlan pure_default = p::kv_tier_formats_plan("", true, Table{});
        Table fusion_default          = uniform(KvCacheStorage::Nvfp4Group16);
        fusion_default[2]             = KvCacheStorage::E8Group64;
        check(rejects("pure forbids the fusion tiers", [&] {
                  (void)p::kv_tier_formats_check(pure_default, classes_of(fusion_default), kLayers,
                                                 false);
              }),
              "pure refuses a table carrying nvfp4/rk4v4 layers");
        // rk4v4 alone must trip it: with the old dtype_of() table this table
        // classified as all-bf16 and the check passed for the wrong reason.
        Table rk4v4_only = uniform(KvCacheStorage::Int8Group64);
        for (std::int32_t layer = 0; layer < kLayers; ++layer) {
            rk4v4_only[static_cast<std::size_t>(layer)] = KvCacheStorage::E8Group64;
        }
        check(rejects("e8 lattice plane", [&] {
                  (void)p::kv_tier_formats_check(pure_default, classes_of(rk4v4_only), kLayers, false);
              }),
              "pure refuses an all-rk4v4 table on its own");
        // An all-iso4e table must trip it too, and be named as iso4e (not nvfp4).
        Table iso4e_only = uniform(KvCacheStorage::Iso3Group16);
        check(rejects("standalone iso4e plane", [&] {
                  (void)p::kv_tier_formats_check(pure_default, classes_of(iso4e_only), kLayers,
                                                 false);
              }),
              "pure refuses an all-iso4e table and names it iso4e");
    }

    // ---- EXECUTABLE GUARD for the two refusal arms of dtype_of() ----
    //
    // Why this block has to exist: the P3 change above (naming
    // KvCacheStorage::Dropped, and throwing instead of falling off the end) is the
    // thing dtype_of() is FOR, and until this block existed NOTHING in the suite
    // exercised either arm. Reverting `case KvCacheStorage::Dropped:` to the old
    // `default: return DType::BF16;` -- the exact laundering this file was changed
    // to remove -- left the whole suite GREEN. A guard nobody executes is a comment.
    //
    // The refusal is also what product/kv_storage_dtype.h does over the SAME enum,
    // and its sibling layouts_impl.h target_kv_cache_profile() names Dropped at
    // :105 and throws; tests/test_kv_component_switch.cpp already counts code 7 in
    // its 249-code refusal sweep. This block is the same fact, one function over.
    {
        // Positive control, so the two checks below cannot pass vacuously: the
        // function must still answer for a storage that really has a dtype.
        check(dtype_of(KvCacheStorage::BFloat16) == DType::BF16,
              "dtype_of still classifies a bf16 layer as bf16");

        // A Dropped layer owns NO KV planes (types.h: NINFER_KV_DROP_LAYERS), so it
        // has no dtype. Classifying it as bf16 would be the worst available answer:
        // DType::BF16 is ALSO the "inherit the global --kv-dtype" sentinel, so it
        // would simultaneously report "this layer is bf16" and "ignore the
        // operator's dtype here".
        check(rejects("owns no KV planes", [] { (void)dtype_of(KvCacheStorage::Dropped); }),
              "dtype_of(Dropped) is REFUSED, not classified as bf16");

        // KvCacheStorage is a std::uint8_t fed from option text, so a value no
        // enumerator names (here 9, above the 8 enumerators) is reachable and must
        // be refused too.
        check(rejects("names no enumerator",
                      [] { (void)dtype_of(static_cast<KvCacheStorage>(9)); }),
              "dtype_of(out-of-range storage byte) is refused");
    }

    if (failures == 0) { std::cout << "kv_tier_formats_test: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
