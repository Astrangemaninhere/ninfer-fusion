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
//    V side stores ISO4E sign-magnitude nibbles in the same plane geometry
//    ("semantic split via v_dtype", decoder_state.cpp). So `mode = fusion` is
//    already baked into the nvfp4 tier, and `pure` is the request to stay off it.
//    A standalone ISO4E layer is a TIER OF ITS OWN (same planes, a distinct
//    sign-magnitude 4-bit nibble codec -- bit3 sign, magnitude 0..7, scale
//    amax/7, i.e. the ISO4 quantizer with the code map swapped; the 3-bit
//    codec of tools/convert/kv_iso_ref.py is NOT what runs, see
//    src/ops/kernel/gqa_iso3_codec.cuh (the path that exists; an earlier revision of this
//    comment said gqa_iso4e_codec.cuh, which is NOT in this tree -- `gqa_iso4e_*` is the
//    deferred target spelling) -- and its own decode kernel,
//    layouts_impl.h), not a spelling of nvfp4: its K plane is not E2M1, so it
//    cannot feed the nvfp4 cold slot. KvLayerClass::Iso4eFusion keeps that apart.
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
//    add_tensor()s {slot_bytes, kv_heads, 2, cold_pages} into the device arena, one
//    stride per (page, kv_head, plane) PER LAYER DTYPE (decoder_state.cpp
//    cold_slot_stride_for(): 9232 B raw for the int8 codec, 9632 B rANS for the
//    nvfp4 one). "One fixed 9536 B stride ... one buffer size serves both codecs"
//    stood here and is stale twice over: 9536 was the rANS record at the OLD 4-bit
//    ceiling (kKvColdRansRecordAt4BitBytes, kept only as build-time fact), and the
//    two codecs do not share a stride. Compression is a FORMAT change, not an
//    eviction: a codec pays off only when its stride is BELOW the resident plane the
//    cold page gave back. See COLD CODEC RULE.
//
// ---------------------------------------------------------------------------
// COLD CODEC RULE (single decision point; REPORT.md carries the full arithmetic)
// ---------------------------------------------------------------------------
// Per head-page (kPagedKVPageSize = 64 tokens, one kv_head, one plane), the
// engine's own plane geometry costs, in bytes:
//
//   resident plane (one plane, head_dim 256 x 64 tokens):
//     bf16   head_dim*64*2                         = 32768 +    0 = 32768
//     int8   head_dim*64 + head_dim/64*64*2        = 16384 +  512 = 16896
//     fp8    head_dim*64 + head_dim/16*64          = 16384 + 1024 = 17408
//     nvfp4  head_dim/2*64 + head_dim/16*64        =  8192 + 1024 =  9216
//     iso4e   (the same plane pair as nvfp4)        =  8192 + 1024 =  9216
//     rk4v4     head_dim/2*64 + head_dim/64*64*2      =  8192 +  512 =  8704
//   cold record (a requantized plane pair + the codec's header):
//     raw slot   16 + 8192 + 1024                                 =  9232
//     rANS slot 320 + 32*259 + 1024  at 4.04 bits/code            =  9632
//               (320 + 32*256 + 1024 at the 4.00 no-expansion bound = 9536
//                -> only 61 of 64 real K slots; 320 + 32*167 + 1024
//                at 2.60 bits/code = 6688 -> 0 of 64: NOTHING encodes)
//
// The record holds a REQUANTIZED page, so its size does not depend on the source
// dtype: all six classes above fit the same 9232 B raw / 9632 B rANS record. What
// differs is only the resident plane the cold page gives back, i.e. whether the
// record PAYS. At the 4.04 bits/code ceiling in use -- the SMALLEST ceiling whose
// per-stream budget covers the measured K requirement (see kKvColdRansStreamBytes):
//     bf16   32768 -> 9632 = 23136 B/head-page SAVED (70.6%)  [the raw slot, 9232]
//     fp8    17408 -> 9632 =  7776 B/head-page SAVED (44.7%)  [the raw slot, 9232]
//     int8   16896 -> 9232 =  7664 B/head-page SAVED (45.4%)  [the raw slot]
//     nvfp4   9216 -> 9632 =   416 B/head-page COST
//     iso4e    9216 -> 9632 =   416 B/head-page COST
//     rk4v4      8704 -> 9632 =   928 B/head-page COST
// ⭐ THE RESULT, and it is a negative one that has to be stated rather than priced
// around: for the three 4-bit-shaped classes, "the record encodes" and "the record
// saves" are MUTUALLY EXCLUSIVE, at every ceiling the tree can express.
//   * To encode at all the record must be >= 9632 B: the encoder's budget is
//     (stride - 1344)/32 B per 512-symbol stream, unclamped (entropy_nvfp4_slot
//     _kernels.cuh pass A/B), and the K plane's rANS streams were MEASURED at up to
//     259 B (dl/ransceil/rans_probe.cu). 6688 B gives 167 B -> 0 of 64 K slots.
//   * To save, the record must be BELOW the plane it replaces: < 9216 B for the
//     nvfp4/iso4e pair, < 8704 B for rk4v4. That is <= 3.84 / 3.59 bits/code.
//   The two intervals do not overlap. So on a 4-bit stack the cold rANS tier is
// either inert (at 6688) or a net device-memory cost (at 9632); it is never a
// saving. This is why the OLD rule text -- "cold is now CHEAPER per element than
// every tier it can cache for" -- was false at the shipped default in the strongest
// sense: it was true only at the price of a slot that never validates.
// The codecs that DO pay are the RAW 9232 B slot on int8/fp8/bf16, whose resident
// planes (16896 / 17408 / 32768) are wider than any record. That path has no
// entropy budget and therefore no overflow case: it always produces a slot.
// kv_cold_class_bytes_of() below derives BOTH ceilings and static_asserts them.
// Break-even ceilings, closed form: the rANS record is below a resident plane of R
// bytes while ceil(512*b/8) <= (R - 1344)/32, i.e. b <= 3.84 bits/code for the
// 9216 B pair and b <= 3.59 bits/code for rk4v4's 8704 B plane. The measured
// requirement is 4.0469 bits/symbol, ABOVE both: there is no ceiling that both
// encodes and saves.
//
// Rule: prefer a codec that is BOTH reachable and footprint-reducing. For the 4-bit
// classes the two conditions are now in direct conflict (see the table above), so
// kv_cold_class_bytes_of() reports them separately and never folds them: a class can
// FIT the record and still not PAY on it. What rk4v4/iso4e/fp8/bf16 lack is the
// codec itself (the requant mode, the per-dtype dispatch arm in
// enqueue_cold_compressions and the attention kernels' cold branch), NOT slot space.
// int8, bf16 and the nvfp4 rANS slot are the three the engine can produce today, and
// only the first two free device memory.
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
//  mode pure additionally forbids the fusion tiers (nvfp4 / iso4e / rk4v4) in the
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
#include <vector>

namespace ninfer::product {

// Everything the tier gates need to know about one layer's dtype, in one
// classification, because two different questions are asked of it:
//   * is the layer on a FUSION tier?  --nvfp4-mode pure says no.
//   * which cold-slot codec can it feed?  The int8 class (raw nibble slot) and the
//     nvfp4 class (rANS slot): those are the two arms of the per-layer codec
//     dispatch in enqueue_cold_compressions. A bf16/fp8/iso4e/rk4v4 layer has no cold
//     codec at all, so one such layer anywhere in the stack leaves the pool idle.
// Iso4eFusion is its own entry rather than a member of the nvfp4 family: a
// standalone ISO4E layer shares the nvfp4 PLANE GEOMETRY but not its CODEC
// (layouts_impl.h "ISO4E is a tier of its own"), so it is a fusion tier (pure
// forbids it, kv_formats.h) that cannot feed the nvfp4 cold slot.
enum class KvLayerClass : std::uint8_t {
    Classic16,    // bf16: no cold codec, not a fusion tier
    Fp8,          // fp8: the compressor rejects it, not a fusion tier
    Int8,         // raw int8 slot: reachable, and the only footprint-reducing codec
    Nvfp4Fusion,  // E2M1 K / ISO4E V; rANS slot is reachable but its fixed stride costs memory
    Iso4eFusion,   // standalone sign-magnitude nibble plane (4-bit code, 4.50 b/el): no cold codec, fusion tier
    Rk4v4Fusion,     // E8 lattice; no cold codec at all
    // rk4v4 family, narrower K planes. Same family, same V plane, same cold story as
    // Rk4v4Fusion (no cold codec of its own) -- but see kv_layer_class_cold_capable: these
    // two are NOT cold-capable, because unlike Rk4v4Fusion there is no encoder for them to
    // pack with. Geometry: product/kv_e8_width.h.
    Rk3v4Fusion,   // e8 lattice, K plane at 3 bits (rk3v4)
    Rk2v4Fusion,   // e8 lattice, K plane at 2 bits (rk2v4)
};

// Effective per-layer dtype -> classification. The caller must resolve BF16
// override slots against the global dtype first (PagedKVCache::plan_cache does
// exactly that before it builds a plane).
[[nodiscard]] inline KvLayerClass kv_layer_class_of(DType dtype) noexcept {
    switch (dtype) {
    case DType::I8: return KvLayerClass::Int8;
    case DType::NVFP4: return KvLayerClass::Nvfp4Fusion;
    case DType::ISO3: return KvLayerClass::Iso4eFusion;
    case DType::E8Kv: return KvLayerClass::Rk4v4Fusion;
    case DType::E8K3Kv: return KvLayerClass::Rk3v4Fusion;
    case DType::E8K2Kv: return KvLayerClass::Rk2v4Fusion;
    case DType::FP8_E4M3FN: return KvLayerClass::Fp8;
    default: return KvLayerClass::Classic16;
    }
}

[[nodiscard]] inline bool kv_layer_class_is_fusion(KvLayerClass cls) noexcept {
    return cls == KvLayerClass::Nvfp4Fusion || cls == KvLayerClass::Iso4eFusion ||
           cls == KvLayerClass::Rk4v4Fusion || cls == KvLayerClass::Rk3v4Fusion ||
           cls == KvLayerClass::Rk2v4Fusion;
}

[[nodiscard]] inline const char* kv_layer_class_name(KvLayerClass cls) noexcept {
    switch (cls) {
    case KvLayerClass::Classic16: return "a 16-bit plane (no cold codec)";
    case KvLayerClass::Fp8: return "an fp8 plane (no cold codec)";
    case KvLayerClass::Int8: return "the raw int8 slot";
    case KvLayerClass::Nvfp4Fusion: return "the nvfp4 rANS slot (K E2M1 + V iso4e)";
    case KvLayerClass::Iso4eFusion: return "a standalone iso4e plane (fusion tier, no cold codec)";
    case KvLayerClass::Rk4v4Fusion: return "an e8 lattice plane (no cold codec)";
    case KvLayerClass::Rk3v4Fusion: return "an rk3v4 plane (K at 3 bits, no cold codec)";
    case KvLayerClass::Rk2v4Fusion: return "an rk2v4 plane (K at 2 bits, no cold codec)";
    }
    return "an unknown class";
}

// The cold-tier format a layer class stores, in the vocabulary's own names.
// Auto means "this class has no cold codec of its own", so the pairing is
// readable straight off the class.
[[nodiscard]] inline kvcfg::KvFormat kv_cold_format_of(KvLayerClass cls) noexcept {
    switch (cls) {
    case KvLayerClass::Int8: return kvcfg::KvFormat::Int8;
    case KvLayerClass::Nvfp4Fusion: return kvcfg::KvFormat::Iso4e;
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

// The rANS pool stride at the ceiling decoder_state.cpp fixes
// (kColdSlotRansBitsPerCodeX100 = 404, the MEASURED minimum): 320 B header + 32
// streams of ceil(512 codes x 4.04 / 8) = 259 B + the 1024 B uncompressed scale tail.
// Written as the same integer ceil the derivation uses, so the two cannot round apart.
//
// WHY 404 AND NOT 260, in one paragraph, because the two are one keystroke apart and
// the old one is still what the ledger mirrors. The encoder does not take a ceiling:
// it takes the STRIDE and derives its per-stream budget from it at runtime,
// (slot_bytes - 320 - 1024) / 32, with no clamp and no reference to any constant
// (entropy_nvfp4_slot_kernels.cuh pass A and pass B). So the stride IS the budget, and
// a page whose streams exceed it has its valid flag cleared and stays hot. Measured on
// the real encoder over the 2048 K streams of 16 engine-dumped Qwen3.8 27B pages
// (dl/ransceil/rans_probe.cu -- which includes this tree's own kernel header -- and
// dl/ransceil/REPORT.md): min 243 / p50 250 / p90 252 / p99 254 / max 259 B per
// 512-symbol stream, i.e. 4.0469 bits/symbol on the binding (K) plane. A 6688 B
// record gives 167 B and a hit rate of 0 of 64 K slots; 9536 B gives 256 B and 61 of
// 64; 9632 B gives 259 B and 64 of 64. The old 2.60 bits/code ceiling was justified by
// a "2.0-2.6 bits/code measured" band that the requantizer's own header now RETRACTS
// (the same instrument measures 3.31-3.86 bits/code for the marginal entropy -- and
// the marginal entropy is not even the binding quantity here: the per-stream worst
// case is, and it is 4.0469).
//
// The consequence is that this stride is ABOVE the nvfp4 plane pair, so the cold rANS
// tier COSTS device memory on every class that can reach it. That is the price, it is
// pinned below, and it is not the defect this file used to carry: the defect was a
// stride that produced no slot while its own table advertised 2528 B/head-page saved.
inline constexpr std::int32_t kKvColdRansStreamBytes = (512 * 404 + 8 * 100 - 1) / (8 * 100);
inline constexpr std::int32_t kKvColdPoolStrideBytes = 320 + 32 * kKvColdRansStreamBytes + 1024;
// The measured datum, repeated here because this header is deliberately cuda-free and
// cannot include the instrument. Same number, same instrument, one place each: this is
// the second spelling of decoder_state.cpp kColdSlotRansMeasuredMaxStreamBytes and the
// static_asserts below are what keep the two together.
inline constexpr std::int32_t kKvColdRansMeasuredMaxStreamBytes = 259;  // K, per 512 symbols
inline constexpr std::int32_t kKvColdRansMeasuredMaxStreamBytesV = 239; // V, same instrument
inline constexpr std::int32_t kKvColdRansMeasuredKHitSlotsAt2Bit = 0;   // of 64, at 167 B
inline constexpr std::int32_t kKvColdRansMeasuredKHitSlotsAt4Bit = 61;  // of 64, at 256 B
inline constexpr std::int32_t kKvColdRansMeasuredKHitSlots = 64;        // of 64, at 259 B
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
// This header is host-only, so it cannot include the ops header that owns the authority;
// it states the same integer ceil and lets decoder_state.cpp -- the one TU that sees both
// -- assert the two together. That is the whole coupling, and it is deliberately two-sided.
static_assert(kKvColdRansStreamBytes >= kKvColdRansMeasuredMaxStreamBytes,
              "the cold rANS stride's per-stream budget is BELOW the largest K stream the "
              "real encoder produces (259 B / 512 symbols): the encoder clears slot_valid "
              "on real pages and the cold pool compresses NOTHING. Measured hit rates: "
              "167 B -> 0 of 64, 256 B -> 61 of 64, 259 B -> 64 of 64. Re-derive "
              "decoder_state.cpp kColdSlotRansBitsPerCodeX100 from the instrument "
              "(dl/ransceil/rans_probe.cu) before changing this");
// The stride is the number the whole cold cost model is priced at: product/kv_bit_budget.h
// carries the same 9632 as kKvBitBudgetColdSlotBytes, and decoder_state.cpp static_asserts
// cold_slot_stride_for(DType::NVFP4) against this SAME ops literal, so the allocator, the
// cost model and the ops slot geometry cannot drift without failing a build. 9536 is the
// 4-bit no-expansion record and 6688 the 2.60 record; BOTH COST every 4-bit-shaped class
// the plane they are meant to give back -- one by not encoding, the other by being wider.
static_assert(kKvColdPoolStrideBytes == 9632, "cold stride == ops::kEntropyNvfp4SlotBytes");
static_assert(kKvColdPoolStrideBytes > kKvColdResidentNvfp4Bytes,
              "the cold rANS record no longer EXCEEDS the nvfp4 plane pair it replaces. "
              "If it has become narrower, the measurement in dl/ransceil/REPORT.md (the K "
              "plane needs 259 B per stream, more than the 4-bit no-expansion bound) is "
              "stale: re-run dl/ransceil/rans_probe.cu and re-derive EVERY number in the "
              "cold rule together -- the record, the ceiling, the price and the bytes the "
              "ladder charges");
// ⭐ The unsatisfiability pin. The two conditions the old rule text ran together are
// `encodes` (>= 9632 B) and `saves` (< the resident plane). This asserts they cannot both
// hold for any class that can reach the rANS codec -- so no future edit may reintroduce a
// saving by choosing the ceiling that makes the comparison come out positive.
static_assert(320 + 32 * kKvColdRansMeasuredMaxStreamBytes + 1024 >= kKvColdResidentNvfp4Bytes,
              "the smallest cold rANS record that encodes (320 + 32*259 + 1024 = 9632 B) "
              "is at or above the 9216 B nvfp4 plane pair -- and rk4v4's 8704 B plane is "
              "narrower still: encode-and-save is UNSATISFIABLE for nvfp4/iso4e/rk4v4, so "
              "the cold rANS tier is either inert or a net device-memory cost and NEVER a "
              "saving. If this fires, the plane geometry changed and the whole cold rule "
              "has to be re-derived");
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
// ⚠ This predicate is the BYTE question only, and it is deliberately kept that way.
// It returns false for the rANS slot at the shipped stride (9632 > 9216). The
// complementary question -- does the codec produce a slot at all -- is NOT folded in
// here, because folding it would hide which of the two failures is in play. It is
// pinned by the measured-coverage static_assert next to the stride above.
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
        // slot. pool_stride_bytes is the arena bytes an int8 LAYER reserves, which
        // is its own codec width -- decoder_state.cpp cold_slot_stride_for() sizes
        // each layer's tensor from the resolved dtype, so pricing this class
        // against the rANS stride understated the saving by 304 B/head-page at the
        // 4-bit bound and by 2944 B/head-page at the measured 4.04 ceiling in use now
        // (the rANS record is 9632 B, so 9632 - 9232 = 400 the other way: the
        // single-grid-point axis this comment is about, see kKvBitBudgetColdSlotBytesInt8Raw).
        return {ColdCodec::Int8Raw, "int8 raw slot", kKvColdInt8PayloadBytes,
                kKvColdInt8PayloadBytes, kKvColdResidentInt8Bytes, true};
    case KvLayerClass::Nvfp4Fusion:
        // Reachable: enqueue_cold_compressions packs an nvfp4 layer into the rANS
        // slot (requant + encode + decode all exist and are admitted), so an nvfp4
        // stack IS compressed -- `reachable` is true and is not in question.
        // What IS in question is whether compressing it FREES anything, and the answer
        // at the measured ceiling is no: kKvColdPoolStrideBytes is 9632 B (pinned by
        // static_assert below) against a 9216 B plane pair, so kv_cold_codec_reduces()
        // is FALSE here and `cold=` is refused on the BYTE clause. Note this is the
        // same verdict the block used to reach from 9536, for the opposite reason: 9536
        // (kKvColdRansRecordAt4BitBytes) was too WIDE, and 6688 -- the record that was
        // narrow enough -- was too SMALL to encode (0 of 64 K slots). The binding
        // tension is not reachability and never was: it is that the record which fits
        // inside the plane and the record which the encoder fills do not overlap. The
        // same number is the knob: the ceiling sets the codec's per-stream budget
        // budget is DERIVED from (32 streams x ceil(512 * 2.60 / 8) = 167 B), so a
        // full page whose entropy exceeds its per-stream budget overflows it and clears
        // its valid flag -- see the INVALID line in program_impl.h. Pay and overflow turn
        // out to be the same knob and they turn in the same direction: the 2.60 ceiling
        // that made this record PAY was measured to produce 0 of 64 valid K slots, and the
        // 4.04 ceiling that produces 64 of 64 makes it a 416 B/head-page COST. There is no
        // third setting. `reachable` below is unchanged and still true -- the codec exists
        // and is admitted -- but reachable and footprint-reducing are now different
        // questions, and the asserts further down keep them apart instead of folding them.
        return {ColdCodec::Nvfp4Rans, "nvfp4 rANS slot", kKvColdPoolStrideBytes,
                kKvColdNvfp4PayloadBytes, kKvColdResidentNvfp4Bytes, true};
    case KvLayerClass::Rk4v4Fusion:
        // BF16-COLD-LAND E2. Reachable, and the ONLY class here whose record does NOT
        // pay: an rk4v4 layer's resident plane is 8704 B/head-page (8192 B of packed 4-bit
        // lattice codes + 512 B of fp16 g64 scales) while the shared raw record is
        // 9232 B, so this codec COSTS 528 B/head-page on every rk4v4 layer it packs.
        //
        // It is still the right codec, because the pool is all-or-nothing per page:
        // arming it is what unlocks the NVFP4 layers of the SAME stack, whose rANS slot
        // saves 2528 B/head-page each. The 27B factory table is 6 x rk4v4 + 10 x nvfp4, so
        // the pool nets 10*2528 - 6*528 = +22112 B per head-page across the stack -- and
        // the rANS slot cannot be used for rk4v4 instead, because rk4v4's codes are
        // near-uniform 4-bit values, which is the same measured reason the int8 tier
        // uses the raw record rather than the entropy slot.
        // kv_cold_class_bytes_of(Rk4v4Fusion) still reads "fits && pays" and it is not
        // wrong: that table prices the rANS record a class WOULD use, and it is the
        // statement that the e8 lattice is not the missing piece. Reachability is this
        // function; that table is a counter.
        return {ColdCodec::Int8Raw, "rk4v4 -> int8 raw slot", kKvColdInt8PayloadBytes,
                kKvColdInt8PayloadBytes,
                (kKvColdHeadDim / 2) * kKvColdPageTokens +
                    (kKvColdHeadDim / kKvColdInt8Group) * kKvColdPageTokens * 2,
                true};
    case KvLayerClass::Classic16:
        // BF16-COLD-LAND A2. Reachable: enqueue_cold_compressions packs a bf16 layer
        // into the raw slot through entropy_cold_requant's Bf16G64 arm and
        // cold_i8_slot_pack_raw -- the same pair the int8 arm uses (a bf16 layer has no
        // scale plane, so its V side shares the K arm). The name is spelled out rather
        // than reusing the int8 spec because the resident plane this codec gives back is
        // the BF16 one (32768), not the int8 one (16896): 9232 < 32768 makes this the
        // largest saving of the three at 23536 B/head-page (71.8%).
        // The resident plane is spelled as the plane geometry, NOT via
        // kv_cold_resident_bytes_of(): that function is declared further down this file
        // (it belongs with the FITS/PAYS table), and calling it here does not compile.
        // The two cannot drift, because the static_assert on
        // kv_cold_resident_bytes_of(KvLayerClass::Classic16) == 32768 further down pins
        // this same expression to 32768 as a build-time fact.
        return {ColdCodec::Int8Raw, "bf16 -> int8 raw slot", kKvColdInt8PayloadBytes,
                kKvColdInt8PayloadBytes, kKvColdHeadDim * kKvColdPageTokens * 2, true};
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

// ---------------------------------------------------------------------------
// FITS is not EXISTS: the byte truth for the classes that have no codec yet
// ---------------------------------------------------------------------------
// A cold record holds a REQUANTIZED page, so the bytes it needs are the shared
// plane pair (head_dim/2 x tokens of packed 4-bit codes + head_dim/16 x tokens of
// E4M3 group scales) plus the codec's header -- the same for every source dtype.
// The class-dependent number is the resident plane the cold page gives back. Two
// different failures therefore have to be told apart, because their fixes are not
// the same:
//   * !fits         -- no record the pool allocates can hold this class's page.
//                      That is a GEOMETRY change (a wider record), and it is the
//                      case for none of the six classes.
//   * fits && !pays -- the record would be bigger than the plane it replaces:
//                      a CEILING or implementation problem, and a wider slot can
//                      only make it worse.
// Reachability (kv_layer_class_cold_capable) is a third, separate question: it is
// about whether a device-side codec exists. The table below deliberately prices
// the classes that have NO codec, which is exactly what the report needs in order
// to say "the slot fits, the codec is missing" instead of "none".
inline constexpr std::int32_t kKvColdRecordRawBytes =
    16 + (kKvColdHeadDim / 2) * kKvColdPageTokens +
    (kKvColdHeadDim / kKvColdNvfp4Group) * kKvColdPageTokens;
static_assert(kKvColdRecordRawBytes == 9232, "the raw record == ops::kColdI8SlotBytes");
static_assert(kKvColdRecordRawBytes == kKvColdInt8PayloadBytes,
              "one definition of the raw record, spelled two ways");

// The record the rANS codec produced at the 4-bit no-expansion ceiling. Named so
// the effect of the ceiling change is a build-time fact, and it now carries THREE
// measured facts rather than one:
//   * 9536 B is ABOVE the 9216 B nvfp4 plane it replaced -- so at that ceiling the
//     cold tier could only cost device memory (this was the original defect (a));
//   * and it only produces a slot for 61 of 64 real K pages (256 B budget against a
//     measured 259 B maximum), so even the cost it prices is not always paid;
//   * and 6688 B, which IS below that plane, produces 0 of 64 -- so the only rANS
//     record that both validates and is narrower than the plane does not exist.
// 9536 is kept, not deleted, because it is the counterexample that makes the last
// clause true rather than rhetorical.
inline constexpr std::int32_t kKvColdRansRecordAt4BitBytes = 320 + 32 * 256 + 1024;
static_assert(kKvColdRansRecordAt4BitBytes == 9536, "the 4-bit rANS record");

// The requantized plane pair on its own: what BOTH records carry in common.
inline constexpr std::int32_t kKvColdPlanePairBytes =
    (kKvColdHeadDim / 2) * kKvColdPageTokens +
    (kKvColdHeadDim / kKvColdNvfp4Group) * kKvColdPageTokens;

// Every cold record the pool can allocate: the raw one and the rANS one. `fits`
// is measured against the wider of the two, because that is the record the pool
// already reserves for an int8 layer.
[[nodiscard]] constexpr std::int32_t kv_cold_widest_record_bytes() noexcept {
    return kKvColdRecordRawBytes > kKvColdPoolStrideBytes ? kKvColdRecordRawBytes
                                                          : kKvColdPoolStrideBytes;
}

// Resident plane bytes per head-page of one class, from the same plane geometry
// the bit ladder uses (product/kv_bit_budget.h: bits_x100 * 16384 / 800), so the
// two can never disagree: 32768 / 16896 / 17408 / 9216 / 9216 / 8704 for
// bf16 / int8 / fp8 / nvfp4 / iso4e / rk4v4.
[[nodiscard]] constexpr std::int32_t kv_cold_resident_bytes_of(KvLayerClass cls) noexcept {
    switch (cls) {
    case KvLayerClass::Classic16:
        // bf16: head_dim x tokens elements of 2 bytes, and no scale plane at all.
        return kKvColdHeadDim * kKvColdPageTokens * 2;
    case KvLayerClass::Fp8:
        // fp8: one E4M3 code byte per element + one E4M3 scale per 16-channel group.
        return kKvColdHeadDim * kKvColdPageTokens +
               (kKvColdHeadDim / kKvColdNvfp4Group) * kKvColdPageTokens;
    case KvLayerClass::Int8: return kKvColdResidentInt8Bytes;
    case KvLayerClass::Nvfp4Fusion:
    case KvLayerClass::Iso4eFusion:
        // iso4e shares the nvfp4 plane pair -- two codes per byte over per-16
        // E4M3FN scales -- so its resident footprint is identical.
        return kKvColdResidentNvfp4Bytes;
    case KvLayerClass::Rk4v4Fusion:
        // rk4v4: a packed 4-bit lattice code per two elements + one FP16 scale per 64.
        return (kKvColdHeadDim / 2) * kKvColdPageTokens +
               (kKvColdHeadDim / kKvColdInt8Group) * kKvColdPageTokens * 2;
    case KvLayerClass::Rk3v4Fusion:
        // rk3v4: 8 elements into 24 bits (3 bytes), + one FP16 scale per 64.
        // 256*3/8 = 96 bytes/row, 96*64 = 6144, + (256/64)*64*2 = 512 -> 6656.
        // Derived and pinned in product/kv_e8_width.h (e8_kv_k_plane_bytes(W3)).
        return (kKvColdHeadDim * 3 / 8) * kKvColdPageTokens +
               (kKvColdHeadDim / kKvColdInt8Group) * kKvColdPageTokens * 2;
    case KvLayerClass::Rk2v4Fusion:
        // rk2v4: 4 elements per byte, + one FP16 scale per 64.
        // 256/4 = 64 bytes/row, 64*64 = 4096, + 512 -> 4608.
        return (kKvColdHeadDim / 4) * kKvColdPageTokens +
               (kKvColdHeadDim / kKvColdInt8Group) * kKvColdPageTokens * 2;
    }
    return 0;
}
static_assert(kv_cold_resident_bytes_of(KvLayerClass::Classic16) == 32768, "bf16: 16.00 b/el");
static_assert(kv_cold_resident_bytes_of(KvLayerClass::Fp8) == 17408, "fp8: 8.50 b/el");
static_assert(kv_cold_resident_bytes_of(KvLayerClass::Int8) == 16896, "int8: 8.25 b/el");
static_assert(kv_cold_resident_bytes_of(KvLayerClass::Nvfp4Fusion) == 9216, "nvfp4: 4.50 b/el");
static_assert(kv_cold_resident_bytes_of(KvLayerClass::Iso4eFusion) == 9216, "iso4e == nvfp4 planes");
static_assert(kv_cold_resident_bytes_of(KvLayerClass::Rk4v4Fusion) == 8704, "rk4v4: 4.25 b/el");
// The two narrow widths. Same content as product/kv_e8_width.h's K-plane asserts, stated
// here because this file prices the plane and the two must not be able to drift.
static_assert(kv_cold_resident_bytes_of(KvLayerClass::Rk3v4Fusion) == 6656,
              "rk3v4 K plane: 96*64 + 512 = 6656 B/head-page");
static_assert(kv_cold_resident_bytes_of(KvLayerClass::Rk2v4Fusion) == 4608,
              "rk2v4 K plane: 64*64 + 512 = 4608 B/head-page (== rk2v4-e8's K geometry)");

struct KvColdClassBytes {
    std::int32_t record_bytes   = 0; // the record a cold codec for this class would need
    std::int32_t resident_bytes = 0; // the resident plane the cold page gives back
    bool fits                   = false;
    bool pays                   = false; // resident > record, BYTES only: see the note at
                                         // kv_cold_codec_reduces() -- a record can fit and
                                         // still be one no encoder produces
};

// The codec a class would use if one existed. The 4-bit classes (nvfp4, iso4e and
// rk4v4's lattice codes) feed the rANS slot directly -- their code planes have the
// same two-codes-per-byte geometry the encoder already walks -- while int8's
// near-uniform requant codes are the raw slot's own case (entropy_cold_requant's
// own note: rANS gains nothing there).
[[nodiscard]] constexpr ColdCodec kv_cold_codec_class_would_use(KvLayerClass cls) noexcept {
    return cls == KvLayerClass::Int8 ? ColdCodec::Int8Raw : ColdCodec::Nvfp4Rans;
}

[[nodiscard]] constexpr KvColdClassBytes kv_cold_class_bytes_of(KvLayerClass cls) noexcept {
    const std::int32_t record =
        kv_cold_codec_class_would_use(cls) == ColdCodec::Int8Raw ? kKvColdRecordRawBytes
                                                                : kKvColdPoolStrideBytes;
    const std::int32_t resident = kv_cold_resident_bytes_of(cls);
    return {record, resident, record <= kv_cold_widest_record_bytes(), resident > record};
}

// THE answer to "does this class fit / would it pay". Nothing here is about
// whether a codec exists.
static_assert(kv_cold_class_bytes_of(KvLayerClass::Classic16).fits, "bf16 fits the raw record");
static_assert(kv_cold_class_bytes_of(KvLayerClass::Classic16).pays, "bf16 pays even at 4.0 b/c");
static_assert(kv_cold_class_bytes_of(KvLayerClass::Fp8).fits, "fp8 fits the raw record");
static_assert(kv_cold_class_bytes_of(KvLayerClass::Fp8).pays, "fp8 pays even at 4.0 b/c");
static_assert(kv_cold_class_bytes_of(KvLayerClass::Int8).fits, "int8 fits its own raw record");
static_assert(kv_cold_class_bytes_of(KvLayerClass::Int8).pays, "int8 pays");
// ⭐ THE HONEST PRICE, and the assertion that keeps it honest. These three classes FIT
// the rANS record -- and PAY on none of them. Both facts are build-time, and they are the
// SAME fact the rule table above states: the record that encodes (>= 9632 B) is above
// every 4-bit resident plane, and the record that would be below them (< 9216 B) cannot
// encode. An earlier version of this block asserted `pays` here, with the byte
// magnitudes, on the strength of a 2.60 b/c ceiling whose hit rate was never measured;
// the measurement (dl/ransceil/REPORT.md) showed that ceiling produces 0 of 64 valid K
// slots, i.e. the savings were priced on a slot that never validates. A slot that never
// validates has no price, so the verdict is COST and the magnitude is the measured one.
// Pinned with bytes, not only the sign: a ceiling that drifted to 3.50 b/c would keep
// the sign and lose the arithmetic.
static_assert(kv_cold_class_bytes_of(KvLayerClass::Iso4eFusion).fits, "iso4e fits the rANS record");
static_assert(!kv_cold_class_bytes_of(KvLayerClass::Iso4eFusion).pays &&
                  kv_cold_class_bytes_of(KvLayerClass::Iso4eFusion).record_bytes -
                          kv_cold_class_bytes_of(KvLayerClass::Iso4eFusion).resident_bytes ==
                      416,
              "iso4e COSTS 416 B/head-page at 4.04 b/c (9216 -> 9632): this is the SMALLEST "
              "record whose per-stream budget covers the measured K requirement, so a "
              "narrower one is inert and a wider one costs more. Do not re-derive this "
              "into a saving by lowering the ceiling -- that is the 0-of-64 defect");
static_assert(kv_cold_class_bytes_of(KvLayerClass::Nvfp4Fusion).fits, "nvfp4 fits the rANS record");
static_assert(!kv_cold_class_bytes_of(KvLayerClass::Nvfp4Fusion).pays &&
                  kv_cold_class_bytes_of(KvLayerClass::Nvfp4Fusion).record_bytes -
                          kv_cold_class_bytes_of(KvLayerClass::Nvfp4Fusion).resident_bytes ==
                      416,
              "nvfp4 COSTS 416 B/head-page at 4.04 b/c (9216 -> 9632): the rANS slot is the "
              "only codec this tier can feed, so its sign IS the cold tier for it, and the "
              "sign is negative at every ceiling the codec can encode at");
static_assert(kv_cold_class_bytes_of(KvLayerClass::Rk4v4Fusion).fits, "rk4v4 fits the rANS record");
static_assert(!kv_cold_class_bytes_of(KvLayerClass::Rk4v4Fusion).pays &&
                  kv_cold_class_bytes_of(KvLayerClass::Rk4v4Fusion).record_bytes -
                          kv_cold_class_bytes_of(KvLayerClass::Rk4v4Fusion).resident_bytes ==
                      928,
              "rk4v4 COSTS 928 B/head-page at 4.04 b/c (8704 -> 9632); its break-even is the "
              "narrowest in the table at 3.59 b/c, so it is the furthest from paying");

// The same two verdicts at the 4-bit no-expansion ceiling, kept as build-time
// fact rather than prose: this is defect (a) pinned so it cannot come back.
[[nodiscard]] constexpr KvColdClassBytes kv_cold_class_bytes_at_4bit(KvLayerClass cls) noexcept {
    const std::int32_t record =
        kv_cold_codec_class_would_use(cls) == ColdCodec::Int8Raw ? kKvColdRecordRawBytes
                                                                : kKvColdRansRecordAt4BitBytes;
    const std::int32_t resident = kv_cold_resident_bytes_of(cls);
    return {record, resident, record <= kKvColdRansRecordAt4BitBytes, resident > record};
}
static_assert(kv_cold_class_bytes_at_4bit(KvLayerClass::Iso4eFusion).fits &&
                  !kv_cold_class_bytes_at_4bit(KvLayerClass::Iso4eFusion).pays,
              "at the 4-bit bound the rANS slot FITS an iso4e page (9536 >= 9232) and still"
              " COSTS 320 B/head-page: the slot was never too small, the ceiling was too high");
static_assert(kv_cold_class_bytes_at_4bit(KvLayerClass::Rk4v4Fusion).fits &&
                  !kv_cold_class_bytes_at_4bit(KvLayerClass::Rk4v4Fusion).pays,
              "at the 4-bit bound the rANS slot fits an rk4v4 page and costs 832 B/head-page");
static_assert(kv_cold_class_bytes_at_4bit(KvLayerClass::Fp8).pays &&
                  kv_cold_class_bytes_at_4bit(KvLayerClass::Classic16).pays,
              "fp8/bf16 paid at either ceiling: their resident planes are wider than any record");

// The same verdict at the CODEC level, for callers that hold a spec rather than a
// class: did this codec pay before the ceiling was tightened? The shipped path
// does not need it (its verdict is kv_cold_codec_reduces), so it exists to let the
// tests pin the flip instead of restating the numbers.
[[nodiscard]] constexpr bool kv_cold_codec_reduces_at_4bit(const KvColdCodecSpec& spec) noexcept {
    return spec.codec == ColdCodec::Int8Raw ? kKvColdRecordRawBytes < spec.resident_bytes
                                            : kKvColdRansRecordAt4BitBytes < spec.resident_bytes;
}

// Why a class with no codec has an idle slot, in the report's own words. Never
// "the slot is too small" -- for no class is it -- and never a bare "none".
[[nodiscard]] inline std::string kv_cold_class_gap_text(KvLayerClass cls) {
    const KvColdClassBytes bytes = kv_cold_class_bytes_of(cls);
    std::string out              = std::string(kv_layer_class_name(cls)) + ": a cold record needs " +
                      std::to_string(bytes.record_bytes) + " B and the plane is " +
                      std::to_string(bytes.resident_bytes) + " B/head-page, so it ";
    out += bytes.fits ? "FITS" : "does NOT fit";
    out += bytes.pays ? " and would SAVE " : " but would COST ";
    out += std::to_string(bytes.pays ? bytes.resident_bytes - bytes.record_bytes
                                     : bytes.record_bytes - bytes.resident_bytes) +
           " B/head-page if a codec existed";
    return out;
}

// The operator's `cold=` name -> the codec it asks for. nullopt means "no
// override, use the default rule" (KvFormat::Auto); ColdCodec::None means "this
// name is not a cold codec at all".
[[nodiscard]] inline std::optional<ColdCodec>
kv_cold_codec_of_format(kvcfg::KvFormat format) noexcept {
    switch (format) {
    case kvcfg::KvFormat::Auto: return std::nullopt;
    case kvcfg::KvFormat::Int8: return ColdCodec::Int8Raw;
    case kvcfg::KvFormat::Iso4e: return ColdCodec::Nvfp4Rans;
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
// is int8 (raw nibble slot), nvfp4 (rANS slot) or bf16 (the SAME raw nibble slot: a
// bf16 plane is requantized to E2M1/g64 by entropy_cold_requant's Bf16G64 arm and
// restored by cold_i8_slot_restore_bf16_raw -- see decoder_state.cpp
// cold_slot_codec_of). This is the CLASS twin of the per-layer test in
// enqueue_cold_compressions (program_impl.h), which packs a layer iff its dtype is
// DType::I8, DType::NVFP4 or DType::BF16 -- the three arms of the table above.
//
// BF16-COLD-LAND A2: Classic16 was excluded here, and that one exclusion is what
// kept the MTP tree (whose per-column masks exist only on the bf16 routes --
// gqa_attention.cpp refuses a quantized tier by name) from ever arming the pool.
// The arithmetic never excluded it and never needed to: the bf16 resident plane is
// 32768 B/head-page against a 9232 B raw record, so kv_cold_codec_reduces() is true
// and kv_cold_class_bytes_of(Classic16).pays has been a build-time fact all along.
// What was missing was the READ side, which now exists.
[[nodiscard]] inline bool kv_layer_class_cold_capable(KvLayerClass cls) noexcept {
    return cls == KvLayerClass::Int8 || cls == KvLayerClass::Nvfp4Fusion ||
           cls == KvLayerClass::Classic16 || cls == KvLayerClass::Rk4v4Fusion;
    // Rk3v4Fusion / Rk2v4Fusion are deliberately ABSENT: Rk4v4Fusion is cold-capable because
    // its K codes are the packed nibbles the raw record already carries, so an encoder
    // exists (program_impl.h's per-layer check is the real authority). The two narrow
    // widths have no encoder in this tree at all, so arming the pool on them could only
    // reserve device memory that never gets written. A stack containing one of them
    // therefore reports the pool as closed, which is the honest report.
}

// The class's own dtype token. This file is host-only and carries no DType printer
// (core/dtype.h is the enum), so a message that holds a class and no DType reads the
// dtype off the class -- the same vocabulary kv_layer_class_name() is built from.
// "rk4v4"/"iso4e"/"nvfp4"/"i8" are the tokens name_of() and dtype_name() use for the same
// tiers, so a line here cannot be read as a different format than a line there.
[[nodiscard]] inline const char* kv_layer_class_dtype_token(KvLayerClass cls) noexcept {
    switch (cls) {
    case KvLayerClass::Classic16: return "bf16";
    case KvLayerClass::Fp8: return "fp8-e4m3";
    case KvLayerClass::Int8: return "i8";
    case KvLayerClass::Nvfp4Fusion: return "nvfp4";
    case KvLayerClass::Iso4eFusion: return "iso4e";
    case KvLayerClass::Rk4v4Fusion: return "rk4v4";
    case KvLayerClass::Rk3v4Fusion: return "rk3v4";
    case KvLayerClass::Rk2v4Fusion: return "rk2v4";
    }
    return "?";
}

// The layer indices that feed NO cold-slot codec, ascending. This is the NAMED form of
// the question kv_cold_pool_reachable() answers with a bool (both are the same
// kv_layer_class_cold_capable() call, so they cannot drift), and it is what the default
// path needs: a pool that is not built has to say WHO closed it, and naming only the
// first offender makes a stack with six of them read exactly like a stack with one --
// the defect the per-layer warning in enqueue_cold_compressions was already fixed for.
[[nodiscard]] inline std::vector<std::int32_t> kv_cold_pool_offenders(
    const std::array<KvLayerClass, kKvLayerStorageSlots>& classes, std::int32_t layers) {
    std::vector<std::int32_t> out;
    for (std::int32_t layer = 0; layer < layers; ++layer) {
        if (!kv_layer_class_cold_capable(classes[static_cast<std::size_t>(layer)])) {
            out.push_back(layer);
        }
    }
    return out;
}

// True when the cold pool can compress pages for this stack: EVERY full-attention
// layer has to feed a slot codec, because one sentinel per (sequence, page) retires
// the physical page for all layers at once and the execution table has no layer
// axis -- a single bf16/fp8/iso4e/rk4v4 layer makes the compressor skip the sequence,
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

// ---------------------------------------------------------------------------
// THE ARMING GATE: who decides whether the cold pool is built
// ---------------------------------------------------------------------------
// `--kv-tier-formats` is a FORMAT flag, and its report -- the only place the
// `cold_residency=` clause lived -- is printed only when that flag was given.
// Residency is a DIFFERENT axis (--cold-policy / --max-cold-pages, never
// --kv-tier-formats), and it had no report of its own at all: a run that set the
// residency flags and nothing else printed `cold_residency=` exactly 0 times, so the
// instrument was silent about the one axis it was built to describe. That is the same
// disease as the defect it was built to catch, and the fix belongs in the ARMING path,
// not in the report: the state below is what the arming site acts on.
enum class KvColdPool : std::uint8_t {
    NotRequested,  // no --cold-policy: there is no pool, and nothing to report
    Armed,         // a pool was requested and every layer feeds a slot codec
    Closed,        // a pool was requested and this stack cannot fill it
};

[[nodiscard]] inline KvColdPool kv_cold_pool_state(
    const std::array<KvLayerClass, kKvLayerStorageSlots>& classes, std::int32_t layers,
    bool requested) noexcept {
    if (!requested) { return KvColdPool::NotRequested; }
    return kv_cold_pool_reachable(classes, layers) ? KvColdPool::Armed : KvColdPool::Closed;
}

// THE LOUD FORM of "one such layer anywhere in the stack leaves the pool idle", and the
// replacement for the two things that used to stand in for it:
//   * the compressor's per-PAGE "REFUSED at layer ..." (program_impl.h), which is loud
//     about the symptom and silent about the DECISION -- the pool stays armed, reserves
//     --max-cold-pages of device memory and holds nothing, and the operator has to infer
//     "this whole stack is unpackable" from a per-page line; and
//   * a report clause that only fired when an unrelated flag was present.
// Every offender is named with its layer index, its dtype and the reason. Nobody may
// instead SKIP the offender: transfer_to_cold returns the shared physical page and
// restore_cold_page rebuilds only the I8/NVFP4 planes, so a skipped layer would read
// another page's bytes -- a silent wrong answer, which is worse than a loud zero.
[[nodiscard]] inline std::string kv_cold_pool_close_notice(
    const std::array<KvLayerClass, kKvLayerStorageSlots>& classes, std::int32_t layers) {
    const std::vector<std::int32_t> offenders = kv_cold_pool_offenders(classes, layers);
    std::string out =
        "[kv-cold-tier] cold_residency=closed: the cold slot pool is NOT armed -- ";
    out += std::to_string(offenders.size()) + " of " + std::to_string(layers) +
           " full-attention layer(s) feed no cold-slot codec, so no page could ever be "
           "packed into it: one sentinel per (sequence, page) retires the physical page for "
           "EVERY layer at once, and the execution table has no layer axis.\n";
    for (const std::int32_t layer : offenders) {
        const KvLayerClass cls = classes[static_cast<std::size_t>(layer)];
        out += "[kv-cold-tier]   offender: layer " + std::to_string(layer) + " dtype=" +
               kv_layer_class_dtype_token(cls) + " (" + kv_layer_class_name(cls) +
               "): no cold codec\n";
    }
    out +=
        "[kv-cold-tier]   consequence: no cold slot is reserved and --max-cold-pages costs "
        "nothing; every page stays hot, so the spill count is 0 BY CONSTRUCTION, not by "
        "accident.\n";
    out +=
        "[kv-cold-tier]   fix: put every layer on i8 or nvfp4 (--kv-layer-storage, or "
        "--kv-tier-formats hot=), or drop the residency request (--cold-policy). SKIPPING "
        "the offender is NOT a fix: see this function's note on why the pool is "
        "all-or-nothing.\n";
    return out;
}

// The `cold_residency=` clause, in ONE spelling, for the three states. Extracted from
// kv_tier_formats_check()'s return value so the unconditional arming report (which prints
// without --kv-tier-formats) and the format report cannot drift apart: the Armed and
// NotRequested texts below are byte-identical to the ones that lived inside that function.
[[nodiscard]] inline std::string kv_cold_residency_text(
    const std::array<KvLayerClass, kKvLayerStorageSlots>& classes, std::int32_t layers,
    KvColdPool state) {
    const KvColdCodecSpec resolved = kv_cold_codec_default(classes, layers);
    std::string out                = "cold_residency=";
    if (state == KvColdPool::NotRequested) {
        out += "none (format and residency are orthogonal: the cold tier only exists with "
               "--cold-policy window|disk|host + --max-cold-pages N)";
        return out;
    }
    if (state == KvColdPool::Closed) {
        // Closed: the arming site does not build the pool at all (see
        // kv_cold_pool_close_notice(), which names EVERY offender -- this clause keeps the
        // format report to one line and names the first one).
        out += "NOT armed (closed): not every layer feeds a cold codec, so the pool would "
               "reserve --max-cold-pages of device memory and compress nothing; ";
        // NOT `resolved`: a closed pool is never built, so it has no codec to price. This
        // clause used to print `codec=int8 raw slot ... SAVES 7664 B/head-page` for a stack
        // whose pool can never compress a page -- the sentence contradicted itself
        // ("compress nothing" immediately followed by a saving) and it priced a codec this
        // stack will never run. The bytes that do belong on this line are the OFFENDER's own
        // slot geometry, and the gap text below carries them per layer.
        out += kv_cold_codec_bytes(KvColdCodecSpec{ColdCodec::None, "none", 0, 0, 0, false});
        // Name the FIRST layer that feeds nothing, and price its slot: the fix for
        // "no codec" is not the same as the fix for "no room", and the operator
        // cannot tell them apart from a bare "none".
        std::int32_t offender = 0;
        for (std::int32_t layer = 0; layer < layers; ++layer) {
            if (!kv_layer_class_cold_capable(classes[static_cast<std::size_t>(layer)])) {
                offender = layer;
                break;
            }
        }
        out += " -- layer " + std::to_string(offender) + " is " +
               kv_cold_class_gap_text(classes[static_cast<std::size_t>(offender)]);
        out += "; with the tier NOT armed, --max-cold-pages is pure overhead avoided rather "
               "than pure overhead spent";
        return out;
    }
    // EVERY layer feeds a slot codec, so the pool really compresses. The int8 phrase is
    // kept for the int8 stack (it names the codec the pool will really use); over an
    // nvfp4 stack the resolved codec is the rANS slot, which does compress but does NOT
    // pay off, so that case must not read as a saving.
    if (resolved.codec == ColdCodec::Int8Raw) {
        out += "pool with the int8 codec; ";
    } else {
        out += "pool live with the nvfp4 rANS codec; ";
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
        const std::int32_t delta =
            kv_cold_codec_saved_bytes(kv_cold_codec_spec(ColdCodec::Nvfp4Rans));
        // The sign IS the content of this sentence and it flips with the rANS
        // ceiling: the nvfp4 layers COST 320 B/head-page each at 4.0 b/c and
        // SAVE 2528 at 2.60. Deriving the word from the same
        // kv_cold_codec_saved_bytes() the byte clause uses is what keeps the two
        // from contradicting each other; the old hardcoded "COST" would now
        // print "COST -2528 B/head-page".
        out += " (MIXED stack: the codec is derived per layer dtype, so the int8 layers save "
               "as priced while the nvfp4 layers ";
        out += delta > 0 ? "SAVE " : "COST ";
        out += std::to_string(delta > 0 ? delta : -delta) + " B/head-page each)";
    }
    return out;
}

// Which tier names the spec text actually wrote. The parsed KvTierFormats
// cannot answer this: parse_tier_formats fills the mode default (fusion =>
// cold = iso4e), so a written "cold=iso4e" and an omitted cold are identical
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
            "DType carry bf16, int8, fp8, nvfp4, iso4e and rk4v4 KV codecs only, and the per-layer "
            "KV dtype validator rejects FP16). Use hot=bf16, which is also 16-bit.");
    case kvcfg::KvFormat::Int4:
    case kvcfg::KvFormat::Iso4:
    case kvcfg::KvFormat::Iso4e:
    case kvcfg::KvFormat::Rk4v4:
    case kvcfg::KvFormat::Rk3v4:
    case kvcfg::KvFormat::Rk2v4:
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
                "--nvfp4-mode pure forbids the fusion tiers (the nvfp4 family, iso4e and rk4v4): "
                "layer " +
                std::to_string(layer) + " resolves to " + kv_layer_class_name(cls) +
                ". Pure runs the classic formats only (its cold default is int8), so pass "
                "--kv-tier-formats hot=bf16 or hot=int8, or drop --nvfp4-mode pure.");
        }
    }

    // The codec `cold=` asked for. (The codec the resolved TABLE would use is read by
    // kv_cold_residency_text(), which is the one place that still needs it.)
    if (plan.written.cold) {
        const auto asked = kv_cold_codec_of_format(plan.formats.cold);
        if (asked.has_value()) {
            const KvColdCodecSpec spec = kv_cold_codec_spec(*asked);
            if (spec.codec == ColdCodec::None) {
                throw std::invalid_argument(
                    std::string("--kv-tier-formats cold=") + kvcfg::name_of(plan.formats.cold) +
                    " names no cold codec: the cold pool carries the packed E2M1/iso4e slot "
                    "codecs only (int8 raw slot, nvfp4 rANS slot). A 16-bit or i4/iso4/rk4v4 name "
                    "here has no slot codec and no decoder branch, so the pool could only "
                    "reserve and stay idle. Use cold=int8 (reachable) or cold=iso4e.");
            }
            if (!spec.reachable) {
                throw std::invalid_argument(
                    std::string("--kv-tier-formats cold=") + kvcfg::name_of(plan.formats.cold) +
                    " has no REACHABLE cold codec in this tree. The cold slot codec is derived "
                    "from the layer dtype, not selected: enqueue_cold_compressions "
                    "(program_impl.h) packs DType::I8 into the raw slot and DType::NVFP4 into "
                    "the rANS slot, and nothing else. Byte check: " +
                    kv_cold_codec_bytes(spec) +
                    ". Reachable today: cold=int8 over an int8 stack, cold=iso4e over an nvfp4 "
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
    out += " ";
    // One spelling with the unconditional arming report (kv_cold_residency_text): the
    // state the engine ACTS on and the state this line DESCRIBES are one object, so a
    // closed pool cannot be reported as a reserved one.
    out += kv_cold_residency_text(classes, layers,
                                  kv_cold_pool_state(classes, layers, cold_pool_present));
    return out;
}

} // namespace ninfer::product
