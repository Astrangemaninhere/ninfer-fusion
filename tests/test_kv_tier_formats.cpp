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
// enumerator breaks this build instead of silently classifying as bf16. That
// used to happen to iso3 and e8: the old ternary chain here mapped both onto
// DType::BF16, so the "pure refuses a table carrying nvfp4/e8 layers" check
// below passed on its nvfp4 layers alone and never exercised e8 at all.
DType dtype_of(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16: return DType::BF16;
    case KvCacheStorage::Int8Group64: return DType::I8;
    case KvCacheStorage::Fp8E4M3Row256: return DType::FP8_E4M3FN;
    case KvCacheStorage::Nvfp4Group16: return DType::NVFP4;
    case KvCacheStorage::Fp8Group16: return DType::FP8_E4M3FN;
    case KvCacheStorage::Iso3Group16: return DType::ISO3;
    case KvCacheStorage::E8Group64: return DType::E8Kv;
    }
    return DType::BF16;
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
        const p::KvTierWritten written = p::kv_tier_written("hot=bf16,tail=fp16,cold=iso3");
        check(written.hot && written.tail && written.cold, "all three tiers were written");
        const p::KvTierWritten only_hot = p::kv_tier_written("hot=int8");
        check(only_hot.hot && !only_hot.tail && !only_hot.cold, "only hot was written");
        const p::KvTierWritten nothing = p::kv_tier_written("");
        check(!nothing.hot && !nothing.tail && !nothing.cold, "an empty spec writes nothing");
        // ... which matters, because the parsed struct fills the mode default in.
        check(p::kv_tier_formats_parse("hot=int8", false).cold == KvFormat::Iso3,
              "the fusion default cold is iso3");
        check(p::kv_tier_formats_parse("", true).cold == KvFormat::Int8,
              "the pure default cold is int8");
    }

    // ---- the vocabulary's own rules, verbatim ----
    {
        check(rejects("decode-hot", [] { (void)p::kv_tier_formats_parse("hot=int4", false); }),
              "hot below int8 is rejected");
        check(rejects("forbids iso/e8", [] { (void)p::kv_tier_formats_parse("cold=iso3", true); }),
              "pure forbids the iso/e8 formats");
        check(rejects("unknown kv format", [] { (void)p::kv_tier_formats_parse("hot=q4", false); }),
              "an unknown format is rejected");
        check(rejects("tail tier precision below hot",
                      [] { (void)p::kv_tier_formats_parse("hot=int8,tail=iso3", false); }),
              "tail below hot is rejected before any landing is attempted");
    }

    // ---- layer classification: iso3 is a tier of its own, not the nvfp4 family ----
    {
        // layouts_impl.h: "ISO3 is a tier of its own: identical planes to NVFP4
        // but a distinct 3-bit sign-magnitude codec". Its K plane is not E2M1, so
        // it can never feed the nvfp4 rANS slot, which is what the old
        // Iso3 -> Nvfp4Fusion mapping claimed.
        check(p::kv_layer_class_of(DType::ISO3) == KvLayerClass::Iso3Fusion,
              "a standalone iso3 layer has its own class");
        check(p::kv_layer_class_of(DType::ISO3) != KvLayerClass::Nvfp4Fusion,
              "iso3 is NOT the nvfp4 rANS class");
        check(p::kv_layer_class_is_fusion(KvLayerClass::Iso3Fusion),
              "iso3 is still a fusion tier: pure forbids it (kv_formats.h)");
        check(p::kv_cold_format_of(KvLayerClass::Iso3Fusion) == KvFormat::Auto,
              "a standalone iso3 plane has no cold codec of its own");
        check(p::kv_layer_class_of(DType::E8Kv) == KvLayerClass::E8Fusion,
              "an e8 lattice layer is the e8 class");
        check(p::kv_layer_class_is_fusion(KvLayerClass::E8Fusion), "e8 is a fusion tier");
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
            p::kv_tier_formats_plan("cold=iso3", false, uniform(KvCacheStorage::Nvfp4Group16));
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
        check(raw.pool_stride_bytes == 9536, "the pool stride is 9536 B");
        check(raw.payload_bytes == 9232, "the int8 codec fills 9232 B of that stride");
        check(raw.resident_bytes == 16896, "an int8 resident plane is 16896 B/head-page");
        check(p::kv_cold_codec_reduces(raw), "int8 cold FREES device memory");
        check(p::kv_cold_codec_saved_bytes(raw) == 7360,
              "int8 cold saves 7360 B/head-page against the stride (7664 against the payload)");

        const p::KvColdCodecSpec rans = p::kv_cold_codec_spec(ColdCodec::Nvfp4Rans);
        check(rans.reachable, "the nvfp4 rANS slot is implemented and admitted for an nvfp4 stack");
        check(rans.resident_bytes == 9216, "an nvfp4 resident plane is 9216 B/head-page");
        check(!p::kv_cold_codec_reduces(rans),
              "the nvfp4 rANS slot does NOT reduce the footprint: 9536 > 9216");
        check(p::kv_cold_codec_saved_bytes(rans) == -320,
              "the nvfp4 rANS slot costs 320 B/head-page");

        const p::KvColdCodecSpec none = p::kv_cold_codec_spec(ColdCodec::None);
        check(none.codec == ColdCodec::None && !p::kv_cold_codec_reduces(none),
              "no codec is never a saving");

        // The premise the whole rule rests on: the int8 resident plane is ABOVE
        // the pool stride. This is a real boundary, not a formality -- at
        // head_dim = 128 the int8 plane is 8448 B, below the 9536 B stride, and
        // even int8 cold would then lose memory.
        check(p::kKvColdResidentInt8Bytes > p::kKvColdPoolStrideBytes,
              "the int8 plane is above the cold stride, so cold can pay off at all");
        const std::int32_t derived_nvfp4 = (p::kKvColdHeadDim / 2) * p::kKvColdPageTokens +
                                           (p::kKvColdHeadDim / p::kKvColdNvfp4Group) *
                                               p::kKvColdPageTokens;
        check(derived_nvfp4 == p::kKvColdResidentNvfp4Bytes,
              "the nvfp4 resident plane is DERIVED from head_dim, not a literal");
        check(p::kKvColdPoolStrideBytes == 320 + 32 * 256 + 1024,
              "the stride is the rANS maximum, so compression never reaches the arena");

        // The default rule picks the reducing codec, and explains the miss otherwise.
        const Classes int8_classes  = classes_of(uniform(KvCacheStorage::Int8Group64));
        const Classes nvfp4_classes = classes_of(uniform(KvCacheStorage::Nvfp4Group16));
        const Classes bf16_classes  = classes_of(uniform(KvCacheStorage::BFloat16));
        const Classes iso3_classes  = classes_of(uniform(KvCacheStorage::Iso3Group16));
        check(p::kv_cold_codec_default(int8_classes, kLayers).codec == ColdCodec::Int8Raw,
              "an all-int8 stack defaults to the int8 raw slot");
        check(p::kv_cold_codec_default(nvfp4_classes, kLayers).codec == ColdCodec::Nvfp4Rans,
              "an nvfp4 stack names the rANS slot as the codec it cannot profitably use");
        check(p::kv_cold_codec_default(bf16_classes, kLayers).codec == ColdCodec::None,
              "a bf16 stack has no cold codec at all");
        check(p::kv_cold_codec_default(iso3_classes, kLayers).codec == ColdCodec::None,
              "a standalone iso3 stack has no cold codec (its K plane is not E2M1)");

        // The operator's override names a CODEC; auto means "use the rule".
        check(!p::kv_cold_codec_of_format(KvFormat::Auto).has_value(), "cold=auto is the rule");
        check(p::kv_cold_codec_of_format(KvFormat::Int8) == ColdCodec::Int8Raw,
              "cold=int8 selects the raw slot");
        check(p::kv_cold_codec_of_format(KvFormat::Iso3) == ColdCodec::Nvfp4Rans,
              "cold=iso3 selects the rANS slot");
        check(p::kv_cold_codec_of_format(KvFormat::E8) == ColdCodec::None,
              "cold=e8 names no codec");

        // The printed byte truth: the report has to say whether memory really moves.
        const std::string int8_line = p::kv_cold_codec_bytes(raw);
        check(int8_line.find("int8 raw slot") != std::string::npos, "the clause names the codec");
        check(int8_line.find("SAVES 7360 B/head-page") != std::string::npos,
              "the clause prints the saving in bytes");
        check(int8_line.find("43.6%") != std::string::npos, "the clause prints the share");
        const std::string rans_line = p::kv_cold_codec_bytes(rans);
        check(rans_line.find("COSTS 320 B/head-page") != std::string::npos,
              "the clause prints the rANS cost, not a saving");
        check(rans_line.find("NOT reachable") == std::string::npos,
              "the clause no longer flags an unreachable codec: both slots are reachable");
    }

    // ---- cold: enforced against the codec the resolved table can really produce ----
    {
        // The vocabulary name each codec stores: this is what makes cold=iso3 mean
        // "the nvfp4 family" and cold=int8 mean "the raw int8 slot".
        check(p::kv_cold_format_of(KvLayerClass::Int8) == KvFormat::Int8,
              "the raw int8 slot stores int8");
        check(p::kv_cold_format_of(KvLayerClass::Nvfp4Fusion) == KvFormat::Iso3,
              "the nvfp4 rANS slot stores iso3");
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
        check(!p::kv_cold_pool_reachable(classes_of(one_bf16), kLayers),
              "one bf16 layer blocks the compressor for the whole sequence");
        Table one_iso3 = int8_table;
        one_iso3[7]    = KvCacheStorage::Iso3Group16;
        check(!p::kv_cold_pool_reachable(classes_of(one_iso3), kLayers),
              "one iso3 layer blocks the compressor: its K plane is not E2M1");

        const KvTierPlan cold_int8 =
            p::kv_tier_formats_plan("hot=int8,cold=int8", false, int8_table);
        bool satisfied = true;
        try {
            (void)p::kv_tier_formats_check(cold_int8, classes_of(int8_table), kLayers, true);
        } catch (const std::exception&) { satisfied = false; }
        check(satisfied, "cold=int8 over an all-int8 stack is satisfiable");

        const KvTierPlan cold_int8_bf16 =
            p::kv_tier_formats_plan("hot=bf16,cold=int8", false, int8_table);
        check(rejects("requires a stack the compressor can pack", [&] {
                  (void)p::kv_tier_formats_check(cold_int8_bf16,
                                                 classes_of(uniform(KvCacheStorage::BFloat16)),
                                                 kLayers, true);
              }),
              "cold=int8 over a bf16 stack is refused");

        const Table nvfp4_stack    = uniform(KvCacheStorage::Nvfp4Group16);
        const KvTierPlan cold_iso3 = p::kv_tier_formats_plan("cold=iso3", false, nvfp4_stack);
        // The rANS slot IS reachable over this stack now, so the refusal has to come
        // from the byte clause: 9536 > 9216, the fixed pool stride is above the
        // resident plane, and the nvfp4 stack has no codec that reduces.
        check(rejects("would not reduce device memory", [&] {
                  (void)p::kv_tier_formats_check(cold_iso3, classes_of(nvfp4_stack), kLayers, true);
              }),
              "cold=iso3 is refused over an nvfp4 stack: the reachable rANS slot costs memory");
        check(rejects("would not reduce device memory", [&] {
                  (void)p::kv_tier_formats_check(cold_iso3, classes_of(nvfp4_stack), kLayers, false);
              }),
              "cold=iso3 is refused even without a pool: the byte verdict is residency-independent");
        check(rejects("COSTS 320 B/head-page", [&] {
                  (void)p::kv_tier_formats_check(cold_iso3, classes_of(nvfp4_stack), kLayers, true);
              }),
              "the refusal states that the rANS slot would cost memory anyway");
        // A name that is not a cold codec at all is refused as such.
        const KvTierPlan cold_e8 = p::kv_tier_formats_plan("hot=int8,cold=e8", false, int8_table);
        check(rejects("names no cold codec", [&] {
                  (void)p::kv_tier_formats_check(cold_e8, classes_of(int8_table), kLayers, true);
              }),
              "cold=e8 is refused: the pool carries no e8 slot codec");

        // An omitted cold is informational, never an error.
        const KvTierPlan default_cold = p::kv_tier_formats_plan("hot=int8", false, int8_table);
        const std::string report =
            p::kv_tier_formats_check(default_cold, classes_of(int8_table), kLayers, true);
        check(report.find("int8 codec") != std::string::npos,
              "the report names the codec the pool will really use");
        check(report.find("cold=iso3") != std::string::npos,
              "the report prints the resolved (unwritten) cold default");
        check(report.find("cold_residency=pool") != std::string::npos,
              "the report names the residency it found");
        check(report.find("SAVES 7360 B/head-page") != std::string::npos,
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
        check(rans_report.find("COSTS 320 B/head-page") != std::string::npos,
              "an nvfp4 stack reports that its codec costs memory");
        check(rans_report.find("net cost") != std::string::npos,
              "the report says --max-cold-pages is a net cost over an nvfp4 stack");

        // The idle wording still has to exist for a stack whose layers feed no codec
        // at all: that is the case where --max-cold-pages really is pure overhead.
        const KvTierPlan bf16_default =
            p::kv_tier_formats_plan("", false, uniform(KvCacheStorage::BFloat16));
        const std::string idle_report = p::kv_tier_formats_check(
            bf16_default, classes_of(uniform(KvCacheStorage::BFloat16)), kLayers, true);
        check(idle_report.find("pool reserved but idle") != std::string::npos,
              "a bf16 stack reports the reserved-but-idle pool");
        check(idle_report.find("codec=none") != std::string::npos,
              "the idle report says there is no cold codec for this stack at all");

        // A MIXED int8/nvfp4 stack is cold-capable too (see the pool check above), and
        // it builds one codec per layer dtype: the report resolves the int8 codec (the
        // only reducing class) and must still name the nvfp4 layers that cost.
        const KvTierPlan mixed_default = p::kv_tier_formats_plan("", false, mixed);
        const std::string mixed_report =
            p::kv_tier_formats_check(mixed_default, classes_of(mixed), kLayers, true);
        check(mixed_report.find("pool with the int8 codec") != std::string::npos,
              "a mixed stack resolves the int8 codec (the only reducing class)");
        check(mixed_report.find("SAVES 7360 B/head-page") != std::string::npos,
              "the mixed report prices the int8 codec it resolved");
        check(mixed_report.find("MIXED stack") != std::string::npos,
              "the mixed report names the nvfp4 layers that cost the same pool");
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
              "pure refuses a table carrying nvfp4/e8 layers");
        // e8 alone must trip it: with the old dtype_of() table this table
        // classified as all-bf16 and the check passed for the wrong reason.
        Table e8_only = uniform(KvCacheStorage::Int8Group64);
        for (std::int32_t layer = 0; layer < kLayers; ++layer) {
            e8_only[static_cast<std::size_t>(layer)] = KvCacheStorage::E8Group64;
        }
        check(rejects("e8 lattice plane", [&] {
                  (void)p::kv_tier_formats_check(pure_default, classes_of(e8_only), kLayers, false);
              }),
              "pure refuses an all-e8 table on its own");
        // An all-iso3 table must trip it too, and be named as iso3 (not nvfp4).
        Table iso3_only = uniform(KvCacheStorage::Iso3Group16);
        check(rejects("standalone iso3 plane", [&] {
                  (void)p::kv_tier_formats_check(pure_default, classes_of(iso3_only), kLayers,
                                                 false);
              }),
              "pure refuses an all-iso3 table and names it iso3");
    }

    if (failures == 0) { std::cout << "kv_tier_formats_test: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
