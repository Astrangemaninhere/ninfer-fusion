#pragma once
// --kv-tier-formats / --nvfp4-mode: engine-side landing of the KV tier
// vocabulary defined in src/kvcfg/kv_formats.h (hot / tail / cold + nvfp4 mode).
//
// Host-only, std-only, no CUDA: the CLI, the server and the planner share one
// resolution path and a plain host test can exercise it (same convention as
// product/kv_bit_budget.h and product/kv_options.h).
//
// ---------------------------------------------------------------------------
// WHAT THE ENGINE CAN CARRY (evidence in REPORT.md)
// ---------------------------------------------------------------------------
//  * ONE DType per full-attention layer, driving BOTH the K and the V plane.
//    EngineOptions::kv_layer_storage -> layouts_impl.h layer_overrides ->
//    SequencePlanningInputs::layer_kv_dtypes -> PagedKVCacheLayout::layer_dtypes
//    -> PagedKVLayerView::dtype. The NVFP4 tier is the single K/V asymmetry: its
//    V side stores ISO3 sign-magnitude nibbles in the same plane geometry
//    ("semantic split via v_dtype", decoder_state.cpp). So `mode = fusion` is
//    already baked into the nvfp4 tier, and `pure` is the request to stay off it.
//    A standalone ISO3 layer is a TIER OF ITS OWN (same planes, distinct 3-bit
//    sign-magnitude codec and its own decode kernel, layouts_impl.h), not a
//    spelling of nvfp4: its K plane is not E2M1, so it cannot feed the nvfp4
//    cold slot. KvLayerClass::Iso3Fusion keeps that apart.
//  * TWO residency classes per page: the resident page pool and the cold slot
//    pool (PagedKVLayerView::cold_slots). There is no third class, so `tail` --
//    a recent high-precision window that is neither the active frontier nor aged
//    out -- has nowhere to land: every page of a layer carries one format. The
//    missing mechanism is the KV precision tail (_TODO.md 96 / issue#164: hot
//    ring page pool + requant on window exit).
//  * The cold slot CODEC is DERIVED from the layer dtype, never selected.
//    enqueue_cold_compressions (program_impl.h) admits a layer iff its dtype is
//    DType::I8 (raw i8 slot) or DType::NVFP4 (rANS slot) AND the layer carries no
//    residual plane, and it requires EVERY layer of the sequence to pass: one
//    sentinel per (sequence, page) retires the physical page for all layers at
//    once, so a single layer with no codec leaves the reserved pool idle. Both
//    cold formats are obtainable today: int8 over an int8 stack, and the nvfp4
//    rANS slot over an nvfp4 stack (a mixed int8/nvfp4 stack packs per layer
//    dtype, since the codec follows the layer dtype and not the pool).
//  * The cold pool lives IN DEVICE MEMORY, not on host/disk: decoder_state.cpp
//    add_tensor()s {slot_bytes, kv_heads, 2, cold_pages} into the device arena,
//    one fixed 9536 B stride per (page, kv_head, plane), "one buffer size serves
//    both codecs". Compression is therefore a FORMAT change, not an eviction: a
//    codec pays off only when that stride is BELOW the resident plane the cold
//    page gave back. See COLD CODEC RULE.
//
// ---------------------------------------------------------------------------
// COLD CODEC RULE (single decision point; REPORT.md carries the full arithmetic)
// ---------------------------------------------------------------------------
// Per head-page (kPagedKVPageSize = 64 tokens, one kv_head, one plane), the
// engine's own plane geometry costs, in bytes:
//
//   nvfp4 resident   head_dim/2*64 + head_dim/16*64    =  8192 + 1024 =  9216
//   int8  resident   head_dim  *64 + head_dim/64*2*64  = 16384 +  512 = 16896
//   cold pool stride (either codec, fixed)             =               9536
//   int8 codec payload (16 hdr + 8192 codes + 1024 sc) =               9232
//
// so the cold tier frees memory over exactly one of them:
//   int8  : 16896 -> 9536 = 7360 B/head-page SAVED (43.6% of the resident plane;
//           45.4% against the 9232 B the codec actually fills)
//   nvfp4 :  9216 -> 9536 =  320 B/head-page LOST (+3.5%): the rANS header and
//           uncompressed scale tail outweigh the 4-bit plane's own advantage.
//           rANS does compress the CODES (~2.0-2.6 bits/code vs ~4.0 measured),
//           but the slot is fixed-stride at the rANS MAXIMUM, so none of that
//           compression reaches the arena.
// Rule: prefer the codec that is BOTH reachable and footprint-reducing. int8 is
// the only codec that does both: an nvfp4 stack IS cold-capable but its rANS slot
// costs, and an iso3/e8/bf16 stack has no codec at all, so the reserved slots
// free nothing there and the report says which case it is instead of quietly
// costing memory.
// kv_cold_codec_default() is the one decision point, kv_cold_codec_spec() is the
// one class->bytes table, and --kv-tier-formats cold= is the operator override.
//
// ---------------------------------------------------------------------------
// RESOLUTION RULES
// ---------------------------------------------------------------------------
//  hot  Auto keeps the table the other knobs build; bf16 / int8 replace every
//       full-attention layer (the only two formats with a resident KV plane
//       codec -- fp16 has no KV tier at all, sub-int8 is rejected by the
//       vocabulary's own decode-hot rule).
//  tail Accepted only when it does not ask for a second format (tail omitted /
//       Auto, or tail == hot). Any other value is refused with the missing
//       mechanism named -- never silently collapsed onto hot.
//  cold A REQUEST, not a default: when written explicitly it must name a codec
//       the resolved layer table can produce AND that reduces the footprint,
//       otherwise it is refused with the reason named. The codec itself is
//       derived from the layer dtype by kv_cold_codec_default(); `cold=` only
//       overrides it, and the byte consequence is printed either way. Residency
//       is a different axis and stays with --cold-policy / --max-cold-pages;
//       this file reports the pairing, it does not allocate.
//  mode pure additionally forbids the fusion tiers (nvfp4 / iso3 / e8) in the
//       resolved table, which is what makes pure a clean attribution baseline.
//
// The split into two stages exists because the layer count lives in the target
// (TextConfig::full_attention_layers()), not in the option parser:
// kv_tier_formats_plan() runs before the per-layer dtype table is mapped and
// applies `hot`; kv_tier_formats_check() runs after it and applies mode/cold
// against the EFFECTIVE dtype of every layer (a BF16 override slot inherits the
// global --kv-dtype, exactly like PagedKVCache::plan_cache resolves it).
// ---------------------------------------------------------------------------

#include "core/dtype.h"
#include "kvcfg/kv_formats.h"
#include "ninfer/types.h"

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

// Everything the tier gates need to know about one layer's dtype, in one
// classification, because two different questions are asked of it:
//   * is the layer on a FUSION tier?  --nvfp4-mode pure says no.
//   * which cold-slot codec can it feed?  The int8 class (raw nibble slot) and the
//     nvfp4 class (rANS slot): those are the two arms of the per-layer codec
//     dispatch in enqueue_cold_compressions. A bf16/fp8/iso3/e8 layer has no cold
//     codec at all, so one such layer anywhere in the stack leaves the pool idle.
// Iso3Fusion is its own entry rather than a member of the nvfp4 family: a
// standalone ISO3 layer shares the nvfp4 PLANE GEOMETRY but not its CODEC
// (layouts_impl.h "ISO3 is a tier of its own"), so it is a fusion tier (pure
// forbids it, kv_formats.h) that cannot feed the nvfp4 cold slot.
enum class KvLayerClass : std::uint8_t {
    Classic16,    // bf16: no cold codec, not a fusion tier
    Fp8,          // fp8: the compressor rejects it, not a fusion tier
    Int8,         // raw int8 slot: reachable, and the only footprint-reducing codec
    Nvfp4Fusion,  // E2M1 K / ISO3 V; rANS slot is reachable but its fixed stride costs memory
    Iso3Fusion,   // standalone 3-bit sign-magnitude plane: no cold codec, fusion tier
    E8Fusion,     // E8 lattice; no cold codec at all
};

// Effective per-layer dtype -> classification. The caller must resolve BF16
// override slots against the global dtype first (PagedKVCache::plan_cache does
// exactly that before it builds a plane).
[[nodiscard]] inline KvLayerClass kv_layer_class_of(DType dtype) noexcept {
    switch (dtype) {
    case DType::I8: return KvLayerClass::Int8;
    case DType::NVFP4: return KvLayerClass::Nvfp4Fusion;
    case DType::ISO3: return KvLayerClass::Iso3Fusion;
    case DType::E8Kv: return KvLayerClass::E8Fusion;
    case DType::FP8_E4M3FN: return KvLayerClass::Fp8;
    default: return KvLayerClass::Classic16;
    }
}

[[nodiscard]] inline bool kv_layer_class_is_fusion(KvLayerClass cls) noexcept {
    return cls == KvLayerClass::Nvfp4Fusion || cls == KvLayerClass::Iso3Fusion ||
           cls == KvLayerClass::E8Fusion;
}

[[nodiscard]] inline const char* kv_layer_class_name(KvLayerClass cls) noexcept {
    switch (cls) {
    case KvLayerClass::Classic16: return "a 16-bit plane (no cold codec)";
    case KvLayerClass::Fp8: return "an fp8 plane (no cold codec)";
    case KvLayerClass::Int8: return "the raw int8 slot";
    case KvLayerClass::Nvfp4Fusion: return "the nvfp4 rANS slot (K E2M1 + V iso3)";
    case KvLayerClass::Iso3Fusion: return "a standalone iso3 plane (fusion tier, no cold codec)";
    case KvLayerClass::E8Fusion: return "an e8 lattice plane (no cold codec)";
    }
    return "an unknown class";
}

// The cold-tier format a layer class stores, in the vocabulary's own names.
// Auto means "this class has no cold codec of its own", so the pairing is
// readable straight off the class.
[[nodiscard]] inline kvcfg::KvFormat kv_cold_format_of(KvLayerClass cls) noexcept {
    switch (cls) {
    case KvLayerClass::Int8: return kvcfg::KvFormat::Int8;
    case KvLayerClass::Nvfp4Fusion: return kvcfg::KvFormat::Iso3;
    default: return kvcfg::KvFormat::Auto;
    }
}

// ---------------------------------------------------------------------------
// The cold codec table: ONE place that pairs a codec with the bytes it costs
// ---------------------------------------------------------------------------
// The page geometry is DERIVED here, not restated, so the rule cannot be "written
// down wrong": a codec's cost is the pool stride it must fit into and its benefit
// is the resident plane the cold page gave back, and both are pure functions of
// the plane geometry (decoder_state.cpp plan_cache):
//
//   resident_nvfp4 = head_dim/2  * page_tokens        (U8 nibble codes)
//                  + head_dim/16 * page_tokens        (E4M3 g16 scales)
//   resident_int8  = head_dim    * page_tokens        (I8 codes)
//                  + head_dim/64 * page_tokens * 2    (FP16 g64 scales)
//
// The premise is the shipped TEXT KV geometry (qwen3_6_27b TextConfig
// head_dim = 256, kPagedKVPageSize = 64 tokens, kKvInt8QuantGroup = 64); the
// VALUES are pinned against the ops owners below, mirroring
// product/kv_bit_budget.h, because those headers (ops/cold_i8.h,
// ops/entropy_nvfp4_slot.h) include cuda_runtime.h and this one must stay
// host-only.
//
// NOTE the asymmetry the whole decision turns on: the RESIDENT plane scales with
// head_dim, the COLD STRIDE does not. At head_dim = 128 the int8 plane is only
// 8448 B, i.e. BELOW the 9536 B stride, so even int8 cold would be a net loss.
// Re-derive per target; never inherit this verdict (d).
inline constexpr std::int32_t kKvColdHeadDim    = 256;  // TextConfig::head_dim
inline constexpr std::int32_t kKvColdPageTokens = 64;   // kPagedKVPageSize
inline constexpr std::int32_t kKvColdInt8Group  = 64;   // kKvInt8QuantGroup
inline constexpr std::int32_t kKvColdNvfp4Group = 16;   // kNvfp4KvQuantGroup

// The pool stride: ONE buffer size serves both codecs, sized at the rANS MAXIMUM
// (320 B header + 32 streams x 256 B + 1024 B uncompressed scale tail), so the
// pool never shrinks to what a page actually compresses to.
inline constexpr std::int32_t kKvColdPoolStrideBytes = 320 + 32 * 256 + 1024;
// The int8 codec's own payload: 16 B header + E2M1 nibble codes + the 1024 B
// scale tail it inherits from the same page-major geometry.
inline constexpr std::int32_t kKvColdInt8PayloadBytes =
    16 + (kKvColdHeadDim / 2) * kKvColdPageTokens +
    (kKvColdHeadDim / kKvColdNvfp4Group) * kKvColdPageTokens;
inline constexpr std::int32_t kKvColdNvfp4PayloadBytes  = kKvColdPoolStrideBytes;
inline constexpr std::int32_t kKvColdResidentNvfp4Bytes =
    (kKvColdHeadDim / 2) * kKvColdPageTokens +
    (kKvColdHeadDim / kKvColdNvfp4Group) * kKvColdPageTokens;
inline constexpr std::int32_t kKvColdResidentInt8Bytes =
    kKvColdHeadDim * kKvColdPageTokens +
    (kKvColdHeadDim / kKvColdInt8Group) * kKvColdPageTokens * 2;

// Pinned against the ops owners and the geometry the measurements were taken on.
static_assert(kKvColdPoolStrideBytes == 9536, "cold stride == ops::kEntropyNvfp4SlotBytes");
static_assert(kKvColdInt8PayloadBytes == 9232, "int8 payload == ops::kColdI8SlotBytes");
static_assert(kKvColdResidentNvfp4Bytes == 9216, "nvfp4 resident plane per head-page");
static_assert(kKvColdResidentInt8Bytes == 16896, "int8 resident plane per head-page");

enum class ColdCodec : std::uint8_t {
    None,       // no codec: the pool cannot compress this stack at all
    Int8Raw,    // raw E2M1-nibble slot over the int8 requant (kColdI8SlotBytes)
    Nvfp4Rans,  // rANS streams plus an uncompressed scale tail (the nvfp4 tier)
};

// Everything the decision needs about one codec: the byte truth AND whether the
// engine can reach it today. Both shipped codecs are reachable (the per-layer codec
// dispatch in enqueue_cold_compressions arms one of them per layer dtype); adding a
// codec (d: an e8 lattice slot) is an edit to the table below and nothing else --
// no gate carries a literal codec any more.
struct KvColdCodecSpec {
    ColdCodec codec;
    const char* name;
    std::int32_t pool_stride_bytes;  // what the pool reserves per (page, head, plane)
    std::int32_t payload_bytes;      // what the codec writes of that stride
    std::int32_t resident_bytes;     // the resident plane the cold page gave back
    bool reachable;                  // can the engine produce it in this tree?
};

// Does the cold tier actually FREE device memory for this codec? This is the
// question S12/A_kvbit_cold.md answered with "cold costs 0.00 hot bits/element",
// which is only true when the pool stride is below the resident plane it
// replaces. For nvfp4 it is not, so the answer must not be assumed.
[[nodiscard]] constexpr bool kv_cold_codec_reduces(const KvColdCodecSpec& spec) noexcept {
    return spec.codec != ColdCodec::None && spec.pool_stride_bytes < spec.resident_bytes;
}

// Negative means the cold tier COSTS that many bytes per head-page.
[[nodiscard]] constexpr std::int32_t
kv_cold_codec_saved_bytes(const KvColdCodecSpec& spec) noexcept {
    return spec.resident_bytes - spec.pool_stride_bytes;
}

// THE class -> codec/bytes mapping.
[[nodiscard]] constexpr KvColdCodecSpec kv_cold_codec_spec_of(KvLayerClass cls) noexcept {
    switch (cls) {
    case KvLayerClass::Int8:
        // Reachable: enqueue_cold_compressions packs an int8 layer into the raw
        // slot, and it is the only codec that reduces the footprint.
        return {ColdCodec::Int8Raw, "int8 raw slot", kKvColdPoolStrideBytes,
                kKvColdInt8PayloadBytes, kKvColdResidentInt8Bytes, true};
    case KvLayerClass::Nvfp4Fusion:
        // Reachable: enqueue_cold_compressions packs an nvfp4 layer into the rANS
        // slot (requant + encode + decode all exist and are admitted), so an nvfp4
        // stack IS compressed -- but at the current 4.0 bits/code the stride (9536)
        // stays above the resident plane (9216), so kv_cold_codec_reduces() is false
        // and `cold=` is refused on the BYTE clause, not on reachability. Nothing
        // here is waiting for a gate to be lifted any more.
        return {ColdCodec::Nvfp4Rans, "nvfp4 rANS slot", kKvColdPoolStrideBytes,
                kKvColdNvfp4PayloadBytes, kKvColdResidentNvfp4Bytes, true};
    default:
        return {ColdCodec::None, "none", 0, 0, 0, false};
    }
}

// Lookup by codec, for the override path: `cold=` names a CODEC, not a class.
[[nodiscard]] constexpr KvColdCodecSpec kv_cold_codec_spec(ColdCodec codec) noexcept {
    switch (codec) {
    case ColdCodec::Int8Raw: return kv_cold_codec_spec_of(KvLayerClass::Int8);
    case ColdCodec::Nvfp4Rans: return kv_cold_codec_spec_of(KvLayerClass::Nvfp4Fusion);
    case ColdCodec::None: break;
    }
    return {ColdCodec::None, "none", 0, 0, 0, false};
}

// The operator's `cold=` name -> the codec it asks for. nullopt means "no
// override, use the default rule" (KvFormat::Auto); ColdCodec::None means "this
// name is not a cold codec at all".
[[nodiscard]] inline std::optional<ColdCodec>
kv_cold_codec_of_format(kvcfg::KvFormat format) noexcept {
    switch (format) {
    case kvcfg::KvFormat::Auto: return std::nullopt;
    case kvcfg::KvFormat::Int8: return ColdCodec::Int8Raw;
    case kvcfg::KvFormat::Iso3: return ColdCodec::Nvfp4Rans;
    default: return ColdCodec::None;
    }
}

// THE default rule. Picks the first layer class whose codec is reachable AND
// footprint-reducing. A stack may MIX int8 and nvfp4 layers (the compressor judges
// each layer on its own dtype), so no class can be assumed to own the pool: the
// loop decides, and it keeps a future per-layer codec from silently picking the
// wrong one. When nothing qualifies, the returned spec
// names the codec that WOULD have been used and why it does not pay, so the
// report can explain the idle pool instead of just reporting it.
[[nodiscard]] inline KvColdCodecSpec
kv_cold_codec_default(const std::array<KvLayerClass, kKvLayerStorageSlots>& classes,
                      std::int32_t layers) noexcept {
    KvColdCodecSpec explanation{ColdCodec::None, "none", 0, 0, 0, false};
    if (layers <= 0) { return explanation; }
    for (std::int32_t layer = 0; layer < layers; ++layer) {
        const KvColdCodecSpec spec = kv_cold_codec_spec_of(classes[static_cast<std::size_t>(layer)]);
        if (spec.codec == ColdCodec::None) { continue; }
        if (spec.reachable && kv_cold_codec_reduces(spec)) { return spec; }
        if (explanation.codec == ColdCodec::None) { explanation = spec; }
    }
    return explanation;
}

// The byte clause of the report: what the pool reserves, what the codec fills,
// what the resident plane gave back, and whether that is a saving or a cost.
// This is the printed answer to "does this really free memory?".
[[nodiscard]] inline std::string kv_cold_codec_bytes(const KvColdCodecSpec& spec) {
    if (spec.codec == ColdCodec::None) { return "codec=none (no cold codec for this stack)"; }
    std::string out = "codec=";
    out += spec.name;
    out += " [pool stride " + std::to_string(spec.pool_stride_bytes) + " B, fills " +
           std::to_string(spec.payload_bytes) + " B, resident plane " +
           std::to_string(spec.resident_bytes) + " B/head-page] -> ";
    const std::int32_t saved = kv_cold_codec_saved_bytes(spec);
    if (saved == 0) {
        out += "NO change";
    } else {
        // x10 integer percent of the resident plane, so the header stays <cmath>-free.
        const std::int64_t scaled = static_cast<std::int64_t>(saved) * 1000;
        const std::int64_t half =
            static_cast<std::int64_t>(spec.resident_bytes) / 2;
        const std::int32_t tenths = static_cast<std::int32_t>(
            (scaled + (scaled < 0 ? -half : half)) / spec.resident_bytes);
        const std::int32_t magnitude = tenths < 0 ? -tenths : tenths;
        out += (saved > 0 ? "SAVES " : "COSTS ");
        out += std::to_string(saved > 0 ? saved : -saved);
        out += " B/head-page (" + std::to_string(magnitude / 10) + "." +
               std::to_string(magnitude % 10) + "% of the resident plane)";
    }
    if (!spec.reachable) { out += " but NOT reachable in this tree"; }
    return out;
}

// The class-level cold admission: a layer feeds a cold slot codec exactly when it
// is int8 (raw nibble slot) or nvfp4 (rANS slot). This is the CLASS twin of the
// per-layer test in enqueue_cold_compressions (program_impl.h), which packs a layer
// iff its dtype is DType::I8 or DType::NVFP4 -- the two arms of the table above.
[[nodiscard]] inline bool kv_layer_class_cold_capable(KvLayerClass cls) noexcept {
    return cls == KvLayerClass::Int8 || cls == KvLayerClass::Nvfp4Fusion;
}

// True when the cold pool can compress pages for this stack: EVERY full-attention
// layer has to feed a slot codec, because one sentinel per (sequence, page) retires
// the physical page for all layers at once and the execution table has no layer
// axis -- a single bf16/fp8/iso3/e8 layer makes the compressor skip the sequence,
// so the reserved pool would compress nothing. Kept as its own predicate because it
// is the COMPRESSOR's precondition, which is narrower than "a codec exists".
//
// BOUNDARY -- class granularity only: KvLayerClass does not carry whether the layer
// owns a residual plane, and enqueue_cold_compressions additionally requires
// k_residual_pages.data == nullptr (the cold restore rebuilds codes and scales
// only, never the residual planes the nvfp4 decode reads). A residual-bearing
// nvfp4 layer is therefore reported here as cold-capable while the per-layer check
// in program_impl.h still refuses to pack it: the pool is reserved, those layers
// stay hot. That per-layer check is the authority; folding a device-side plane
// layout into this host-only classification to restate it is not worth the coupling
// (and the residual axis is a per-layer device fact, not a class fact).
[[nodiscard]] inline bool kv_cold_pool_reachable(
    const std::array<KvLayerClass, kKvLayerStorageSlots>& classes, std::int32_t layers) noexcept {
    if (layers <= 0) { return false; }
    for (std::int32_t layer = 0; layer < layers; ++layer) {
        if (!kv_layer_class_cold_capable(classes[static_cast<std::size_t>(layer)])) {
            return false;
        }
    }
    return true;
}

// Which tier names the spec text actually wrote. The parsed KvTierFormats
// cannot answer this: parse_tier_formats fills the mode default (fusion =>
// cold = iso3), so a written "cold=iso3" and an omitted cold are identical
// there.
struct KvTierWritten {
    bool hot  = false;
    bool tail = false;
    bool cold = false;
};

[[nodiscard]] inline KvTierWritten kv_tier_written(std::string_view spec) {
    KvTierWritten out;
    std::size_t pos = 0;
    while (pos <= spec.size()) {
        const std::size_t comma = spec.find(',', pos);
        const std::string_view item =
            spec.substr(pos, comma == std::string_view::npos ? spec.size() - pos : comma - pos);
        pos = comma == std::string_view::npos ? spec.size() + 1 : comma + 1;
        const std::size_t eq = item.find('=');
        if (eq == std::string_view::npos) { continue; }
        const std::string_view tier = item.substr(0, eq);
        if (tier == "hot") {
            out.hot = true;
        } else if (tier == "tail") {
            out.tail = true;
        } else if (tier == "cold") {
            out.cold = true;
        }
    }
    return out;
}

// parse_tier_formats with the flag name in the message. The vocabulary's own
// wording is reused verbatim, so this twin cannot drift from the kv_tiers.py /
// GUI contract (src/kvcfg/kv_formats.h header note).
[[nodiscard]] inline kvcfg::KvTierFormats kv_tier_formats_parse(std::string_view spec, bool pure) {
    std::string err;
    const kvcfg::Nvfp4Mode mode = pure ? kvcfg::Nvfp4Mode::Pure : kvcfg::Nvfp4Mode::Fusion;
    const auto parsed           = kvcfg::parse_tier_formats(spec, mode, &err);
    if (!parsed.has_value()) { throw std::invalid_argument("--kv-tier-formats: " + err); }
    return *parsed;
}

// The resident storage a vocabulary format maps to; nullopt means "inherit"
// (KvFormat::Auto). Throws for a format the engine has no resident KV codec for.
[[nodiscard]] inline std::optional<KvCacheStorage> kv_hot_storage_of(kvcfg::KvFormat format) {
    switch (format) {
    case kvcfg::KvFormat::Auto: return std::nullopt;
    case kvcfg::KvFormat::Bf16: return KvCacheStorage::BFloat16;
    case kvcfg::KvFormat::Int8: return KvCacheStorage::Int8Group64;
    case kvcfg::KvFormat::Fp16:
        throw std::invalid_argument(
            "--kv-tier-formats hot=fp16: this engine has no FP16 KV tier (KvCacheStorage and "
            "DType carry bf16, int8, fp8, nvfp4, iso3 and e8 KV codecs only, and the per-layer "
            "KV dtype validator rejects FP16). Use hot=bf16, which is also 16-bit.");
    case kvcfg::KvFormat::Int4:
    case kvcfg::KvFormat::Iso4:
    case kvcfg::KvFormat::Iso3:
    case kvcfg::KvFormat::E8:
        throw std::invalid_argument(
            std::string("--kv-tier-formats hot=") + kvcfg::name_of(format) +
            ": the hot tier must stay at int8 or wider (decode-hot rule), and this engine has "
            "no resident codec for that format anyway. Put a sub-int8 format on cold= or in "
            "--kv-layer-storage.");
    }
    return std::nullopt;
}

// Stage 1 result: the vocabulary plus the resident table it asks for.
struct KvTierPlan {
    kvcfg::KvTierFormats formats;
    KvTierWritten written;
    bool override_hot          = false;  // hot was written: replace every full-attention layer
    KvCacheStorage hot_storage = KvCacheStorage::BFloat16;
    std::array<KvCacheStorage, kKvLayerStorageSlots> table{};  // `table` with hot applied
};

// Stage 1: vocabulary syntax + hot + tail, against the per-layer table the other
// knobs resolved (in KvCacheStorage space, so no model knowledge is needed).
// Throws std::invalid_argument for everything the engine cannot express.
[[nodiscard]] inline KvTierPlan kv_tier_formats_plan(
    std::string_view spec, bool pure,
    const std::array<KvCacheStorage, kKvLayerStorageSlots>& table) {
    KvTierPlan plan;
    plan.formats = kv_tier_formats_parse(spec, pure);
    plan.written = kv_tier_written(spec);
    plan.table   = table;

    if (plan.written.hot) {
        // hot=auto is a legal spelling of "keep the table the other knobs built".
        const auto storage = kv_hot_storage_of(plan.formats.hot);
        if (storage.has_value()) {
            plan.override_hot = true;
            plan.hot_storage  = *storage;
            plan.table.fill(*storage);
        }
    }

    if (plan.written.tail && plan.formats.tail != kvcfg::KvFormat::Auto &&
        plan.formats.tail != plan.formats.hot) {
        throw std::invalid_argument(
            std::string("--kv-tier-formats tail=") + kvcfg::name_of(plan.formats.tail) +
            " cannot be honored: this engine has no tail tier. A full-attention layer is built "
            "with exactly one DType for all of its pages (PagedKVLayerView::dtype, planned by "
            "decoder_state.cpp plan_cache), and the only two residency classes are resident "
            "pages and the cold slot pool, so a recent high-precision window that is neither the "
            "active frontier nor aged out has nowhere to live. The missing mechanism is the KV "
            "precision tail (_TODO.md 96 / issue#164: hot ring page pool + requant on window "
            "exit). Until it lands, omit tail (or repeat the hot format), and use "
            "--kv-layer-storage for a static per-layer mix.");
    }
    return plan;
}

// Stage 2: the mode and cold gates, evaluated on the EFFECTIVE per-layer class of
// the table the plan will really build. Returns the one-line stderr report
// (mirrors the existing "[kv-bit-budget] ..." report). Throws
// std::invalid_argument for a request the resolved table cannot satisfy.
[[nodiscard]] inline std::string kv_tier_formats_check(
    const KvTierPlan& plan, const std::array<KvLayerClass, kKvLayerStorageSlots>& classes,
    std::int32_t layers, bool cold_pool_present) {
    if (layers <= 0) { throw std::invalid_argument("--kv-tier-formats: no full-attention layers"); }

    if (plan.formats.mode == kvcfg::Nvfp4Mode::Pure) {
        for (std::int32_t layer = 0; layer < layers; ++layer) {
            const KvLayerClass cls = classes[static_cast<std::size_t>(layer)];
            if (!kv_layer_class_is_fusion(cls)) { continue; }
            throw std::invalid_argument(
                "--nvfp4-mode pure forbids the fusion tiers (the nvfp4 family, iso3 and e8): "
                "layer " +
                std::to_string(layer) + " resolves to " + kv_layer_class_name(cls) +
                ". Pure runs the classic formats only (its cold default is int8), so pass "
                "--kv-tier-formats hot=bf16 or hot=int8, or drop --nvfp4-mode pure.");
        }
    }

    // The codec the resolved table would use, and the codec `cold=` asked for.
    const KvColdCodecSpec resolved = kv_cold_codec_default(classes, layers);
    if (plan.written.cold) {
        const auto asked = kv_cold_codec_of_format(plan.formats.cold);
        if (asked.has_value()) {
            const KvColdCodecSpec spec = kv_cold_codec_spec(*asked);
            if (spec.codec == ColdCodec::None) {
                throw std::invalid_argument(
                    std::string("--kv-tier-formats cold=") + kvcfg::name_of(plan.formats.cold) +
                    " names no cold codec: the cold pool carries the packed E2M1/iso3 slot "
                    "codecs only (int8 raw slot, nvfp4 rANS slot). A 16-bit or i4/iso4/e8 name "
                    "here has no slot codec and no decoder branch, so the pool could only "
                    "reserve and stay idle. Use cold=int8 (reachable) or cold=iso3.");
            }
            if (!spec.reachable) {
                throw std::invalid_argument(
                    std::string("--kv-tier-formats cold=") + kvcfg::name_of(plan.formats.cold) +
                    " has no REACHABLE cold codec in this tree. The cold slot codec is derived "
                    "from the layer dtype, not selected: enqueue_cold_compressions "
                    "(program_impl.h) packs DType::I8 into the raw slot and DType::NVFP4 into "
                    "the rANS slot, and nothing else. Byte check: " +
                    kv_cold_codec_bytes(spec) +
                    ". Reachable today: cold=int8 over an int8 stack, cold=iso3 over an nvfp4 "
                    "stack.");
            }
            if (!kv_cold_codec_reduces(spec)) {
                throw std::invalid_argument(
                    std::string("--kv-tier-formats cold=") + kvcfg::name_of(plan.formats.cold) +
                    " would not reduce device memory: " + kv_cold_codec_bytes(spec) +
                    ". The cold pool is a fixed-stride device arena (decoder_state.cpp "
                    "add_tensor), so a codec whose slot stride is not below the resident plane "
                    "makes the cold tier a net LOSS, and no override can fix that. Keep the "
                    "pages hot (omit cold=), or put the stack on int8 (hot=int8 / the matching "
                    "--kv-dtype / --kv-layer-storage).");
            }
        }
        if (plan.formats.cold == kvcfg::KvFormat::Int8 && !kv_cold_pool_reachable(classes, layers)) {
            // The offender is the first layer that feeds NO cold codec: naming an int8
            // or nvfp4 layer here would blame a layer the compressor packs fine.
            std::int32_t offender = 0;
            for (std::int32_t layer = 0; layer < layers; ++layer) {
                if (!kv_layer_class_cold_capable(classes[static_cast<std::size_t>(layer)])) {
                    offender = layer;
                    break;
                }
            }
            throw std::invalid_argument(
                "cold=int8 requires a stack the compressor can pack: "
                "enqueue_cold_compressions (program_impl.h) skips the sequence unless EVERY "
                "layer is DType::I8 (raw slot) or DType::NVFP4 (rANS slot) and carries no "
                "residual plane, and layer " + std::to_string(offender) + " resolves to " +
                kv_layer_class_name(classes[static_cast<std::size_t>(offender)]) +
                ", so the pool would reserve its slots and compress nothing. Pass "
                "--kv-tier-formats hot=int8 (or the matching --kv-dtype / --kv-layer-storage) to "
                "make the request satisfiable.");
        }
    }

    const bool reachable = kv_cold_pool_reachable(classes, layers);
    std::string out      = "[kv-tier-formats] hot=";
    out += kvcfg::name_of(plan.formats.hot);
    if (!plan.written.hot) { out += "(table from the other knobs)"; }
    out += " tail=";
    out += kvcfg::name_of(plan.formats.tail);
    out += plan.written.tail ? "" : "(==hot)";
    out += " cold=";
    out += kvcfg::name_of(plan.formats.cold);
    if (!plan.written.cold) {
        out += reachable ? "(default; the resolved table feeds a cold codec)"
                         : "(default; the resolved table has no reachable cold codec)";
    }
    out += " mode=";
    out += plan.formats.mode == kvcfg::Nvfp4Mode::Pure ? "pure" : "fusion";
    out += " layers=" + std::to_string(layers);
    out += " cold_residency=";
    if (!cold_pool_present) {
        out += "none (format and residency are orthogonal: the cold tier only exists with "
               "--cold-policy window|disk|host + --max-cold-pages N)";
    } else if (reachable) {
        // EVERY layer feeds a slot codec, so the pool really compresses. The int8
        // phrase is kept for the int8 stack (it names the codec the pool will really
        // use); over an nvfp4 stack the resolved codec is the rANS slot, which does
        // compress but does NOT pay off, so that case must not read as a saving.
        if (resolved.codec == ColdCodec::Int8Raw) {
            out += "pool with the int8 codec; ";
        } else {
            out += "pool live, but no codec pays off; ";
        }
        out += kv_cold_codec_bytes(resolved);
        if (!kv_cold_codec_reduces(resolved)) {
            out += " -- the pool compresses, but this codec's fixed stride is above the resident "
                   "plane it gives back, so --max-cold-pages is a net cost here";
        }
        // A stack may carry BOTH cold-capable classes, and such a pool holds one codec
        // PER LAYER DTYPE: the bytes above price the codec the rule resolved (always the
        // int8 one, since it is the only reducing class), so the nvfp4 layers of the
        // same stack have to be named or the line reads as a whole-stack saving.
        bool has_int8  = false;
        bool has_nvfp4 = false;
        for (std::int32_t layer = 0; layer < layers; ++layer) {
            const KvLayerClass cls = classes[static_cast<std::size_t>(layer)];
            has_int8  = has_int8 || cls == KvLayerClass::Int8;
            has_nvfp4 = has_nvfp4 || cls == KvLayerClass::Nvfp4Fusion;
        }
        if (has_int8 && has_nvfp4) {
            const std::int32_t nvfp4_cost =
                -kv_cold_codec_saved_bytes(kv_cold_codec_spec(ColdCodec::Nvfp4Rans));
            out += " (MIXED stack: the codec is derived per layer dtype, so the int8 layers save "
                   "as priced while the nvfp4 layers COST " + std::to_string(nvfp4_cost) +
                   " B/head-page each)";
        }
    } else {
        // Not every layer feeds a codec, so the compressor skips the whole sequence:
        // the pool reserves its slots and fills none of them.
        out += "pool reserved but idle (not every layer feeds a cold codec); ";
        out += kv_cold_codec_bytes(resolved);
        out += " -- the reserved slots compress nothing, so --max-cold-pages is pure overhead here";
    }
    return out;
}

} // namespace ninfer::product
