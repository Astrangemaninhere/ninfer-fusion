#include <ninfer/targets/qwen3_6/decoder_state.h>
#include "ninfer/ops/cold_i8.h"
#include "ninfer/ops/entropy_nvfp4_slot.h"
#include "ops/kernel/gqa_isoquant_rot_gate.h"
#include "ops/kernel/gqa_isoquant_row_scale_loader.h"

#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets::qwen3_6 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

// Single source of truth for the per-layer KV quant group. It has to agree
// with both the plane geometry plan_cache() allocates for that dtype and the
// ops-level guard in ops/wrapper/gqa_attention.cpp: every packed-16 tier
// (NVFP4, FP8_E4M3FN, ISO3) must carry quant_group 16 with E4M3 group scales,
// while I8/E8Kv carry 64-wide FP16 scales.
std::int32_t kv_layer_quant_group(DType dtype) {
    switch (dtype) {
    case DType::I8:
    case DType::E8Kv:
        return kKvInt8QuantGroup;
    case DType::FP8_E4M3FN:
        return kKvFp8QuantGroup;
    case DType::NVFP4:
    // ISO3 shares the NVFP4 plane geometry -- two codes per byte and per-16
    // E4M3FN scales -- and only swaps the nibble decoder, so it takes the same
    // group. Leaving ISO3 out of this table used to resolve to that group only
    // by accident, through the final `else` arm of the old ternary chain.
    case DType::ISO3:
        return kNvfp4KvQuantGroup;
    default:
        return 0;
    }
}

// V storage for a resolved layer dtype. The NVFP4 tier stores V in the ISO3
// codec (a semantic split inside the same planes); an ISO3 layer keeps ISO3 V,
// which is exactly what gqa_attention_decode_iso3.cuh reads and writes back.
//
// SEPARATION: the NVFP4-tier V codec is now selectable. `codec` only changes
// what the producers encode and the consumers decode -- both codecs use the
// same two-codes-per-byte plane with per-16 E4M3FN scales, so no geometry,
// allocation or launcher change follows from it. E2M1 resolves an NVFP4 layer
// to NVFP4 V, i.e. the `v_dtype == ISO3` tests in the launchers fall through to
// the <NVFP4, NVFP4> kernel that already exists. Plain ISO3 layers keep ISO3 V
// either way: their K is ISO3 too, and the knob is documented as an NVFP4-tier
// switch only.
DType kv_layer_v_dtype(DType dtype, KvVCodec codec) {
    if (dtype != DType::NVFP4) { return dtype; }
    return codec == KvVCodec::E2M1 ? DType::NVFP4 : DType::ISO3;
}

// SEPARATION guard. Two mechanisms hardcode ISO3 V and do NOT read v_dtype, so
// an E2M1 V plane would be silently mis-decoded by them:
//   * V residual planes (gqa_attention_prefill_fill_nvfp4k_iso3v_kernel writes
//     ISO3; the prefill and decode producers only decode the residual on the
//     VVDType == ISO3 / Iso3V branch);
//   * the entropy cold pool for NVFP4 layers (program_impl.h requantizes V with
//     Iso3VG16 unconditionally).
// Refuse loudly instead of approximating -- the repository's own idiom.
void kv_v_codec_check(KvVCodec codec, std::span<const DType> layer_dtypes,
                      std::span<const bool> layer_residual, std::uint32_t layers,
                      std::uint32_t max_cold_pages) {
    if (codec == KvVCodec::Iso3) { return; }
    for (std::uint32_t i = 0; i < layers; ++i) {
        if (layer_dtypes[i] != DType::NVFP4) { continue; }
        if (i < layer_residual.size() && layer_residual[i]) {
            throw std::invalid_argument(
                "kv-v-codec e2m1: NVFP4 layer " + std::to_string(i) +
                " keeps a V residual plane, and the residual codec is ISO3-only; "
                "disable --kv-residual-layers or keep --kv-v-codec iso3");
        }
    }
    if (max_cold_pages != 0) {
        for (std::uint32_t i = 0; i < layers; ++i) {
            if (layer_dtypes[i] == DType::NVFP4) {
                throw std::invalid_argument(
                    "kv-v-codec e2m1: the cold pool requires ISO3 V for NVFP4 layers "
                    "(the eviction requant is Iso3VG16); use --cold-policy none or keep "
                    "--kv-v-codec iso3");
            }
        }
    }
}

// Diagnostic metadata only (PagedKVLayerView::v_quant_group): published by the
// tiers whose V plane is an E4M3 group-scale plane.
std::int32_t kv_layer_v_quant_group(DType dtype) {
    return dtype == DType::NVFP4 || dtype == DType::ISO3 ? kNvfp4KvQuantGroup : 0;
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                              std::int32_t kv_heads, std::int32_t head_dim, DType dtype,
                              std::int32_t quant_group, std::span<const DType> layer_dtypes,
                              std::span<const bool> layer_residual,
                              std::span<const std::uint32_t> layer_windows,
                              std::int32_t table_rows, std::uint32_t physical_page_groups) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    // The per-layer tables below live in std::array<..., 64> slots, and the
    // attention kernels index their own 64-wide tables by KV layer number, so a
    // model with more than 64 KV layers would silently index out of bounds.
    if (layers > kPagedKVCacheMaxLayers) {
        throw std::invalid_argument("Paged KV layer count exceeds the 64-layer table budget");
    }
    if (!layer_dtypes.empty() && layer_dtypes.size() < layers) {
        throw std::invalid_argument("Paged KV per-layer dtype table is shorter than the layer count");
    }
    if (!layer_residual.empty() && layer_residual.size() < layers) {
        throw std::invalid_argument("Paged KV per-layer residual table is shorter than the layer count");
    }
    if (!layer_windows.empty() && layer_windows.size() < layers) {
        throw std::invalid_argument(
            "Paged KV per-layer sliding-window table is shorter than the layer count");
    }
    const auto layer_has_residual = [&](std::uint32_t layer) {
        return !layer_residual.empty() && layer_residual[layer];
    };
    // Per-layer resolution: BF16 entries inherit the global dtype. Accepted
    // per-layer storages are the quantized codecs with their native group.
    const auto layer_dtype = [&](std::uint32_t layer) {
        const DType override_dtype = layer_dtypes.empty() ? DType::BF16 : layer_dtypes[layer];
        const DType selected = override_dtype == DType::BF16 ? dtype : override_dtype;
        if (selected != DType::BF16 && selected != DType::I8 && selected != DType::NVFP4 &&
            selected != DType::FP8_E4M3FN && selected != DType::E8Kv &&
            selected != DType::ISO3) {
            throw std::invalid_argument("Paged KV per-layer dtype is invalid");
        }
        return selected;
    };
    (void)quant_group;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const DType selected = layer_dtype(layer);
        if (selected != DType::BF16) {
            const std::int32_t group = kv_layer_quant_group(selected);
            if (head_dim % group != 0) {
                throw std::invalid_argument("Paged KV per-layer quantization is invalid");
            }
        }
    }

    const std::uint32_t logical_pages = page_count(capacity);
    // An explicit --kv-capacity may floor the device page pool below max_context:
    // the rope domain (4x under YaRN) can exceed the pool, and the cold pool
    // recycles committed pages under pressure so the context keeps growing.
    if (physical_page_groups == 0) {
        throw std::invalid_argument("Paged KV physical page capacity is zero");
    }

    KVPageGeometry geometry;
    geometry.planes.reserve(static_cast<std::size_t>(layers) * 8ULL);
    // Family-capacity slots: must match PagedKVCacheLayout's 64-wide tables.
    std::array<DType, 64> stored{};
    std::array<bool, 64> residual_flags{};
    std::array<std::uint32_t, 64> window_flags{};
    std::array<std::uint32_t, 64> plane_base{};
    std::uint32_t plane_cursor = 0;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const DType selected = layer_dtype(layer);
        stored[layer]        = selected;
        window_flags[layer]  = layer_windows.empty() ? 0U : layer_windows[layer];
        plane_base[layer]    = plane_cursor;
        const std::int32_t group = kv_layer_quant_group(selected);
        if (selected == DType::BF16) {
            geometry.planes.push_back({DType::BF16, head_dim, kv_heads, 256});
            geometry.planes.push_back({DType::BF16, head_dim, kv_heads, 256});
        } else if (selected == DType::I8) {
            geometry.planes.push_back({DType::I8, head_dim, kv_heads, 256});
            geometry.planes.push_back({DType::I8, head_dim, kv_heads, 256});
            geometry.planes.push_back({DType::FP16, head_dim / group, kv_heads, 256});
            geometry.planes.push_back({DType::FP16, head_dim / group, kv_heads, 256});
        } else if (selected == DType::E8Kv) {
            // E8 tier: packed 4-bit E8-lattice K codes + i4 V codes (two per
            // byte) with per-64 FP16 scales (int8-kernel path).
            geometry.planes.push_back({DType::U8, head_dim / 2, kv_heads, 256});
            geometry.planes.push_back({DType::U8, head_dim / 2, kv_heads, 256});
            geometry.planes.push_back({DType::FP16, head_dim / group, kv_heads, 256});
            geometry.planes.push_back({DType::FP16, head_dim / group, kv_heads, 256});
        } else if (selected == DType::FP8_E4M3FN) {
            geometry.planes.push_back({DType::FP8_E4M3FN, head_dim, kv_heads, 256});
            geometry.planes.push_back({DType::FP8_E4M3FN, head_dim, kv_heads, 256});
            // E4M3 group scales, not FP16: the fp8 tier is a `packed16` dtype and the
            // attention op requires an E4M3 scale plane for every one of those
            // (gqa_attention.cpp:208-213), and the decode kernel reads E4M3
            // (gqa_attention_decode_fp8.cuh). Allocating FP16 here made every fp8 run
            // fail at the guard with "invalid NVFP4 KV cache scale dtype".
            geometry.planes.push_back({DType::FP8_E4M3FN, head_dim / group, kv_heads, 256});
            geometry.planes.push_back({DType::FP8_E4M3FN, head_dim / group, kv_heads, 256});
        } else {
            // NVFP4/ISO3 tier: identical plane geometry. K carries E2M1 (NVFP4)
            // or sign-magnitude ISO3 nibbles, both over per-16 E4M3FN scales,
            // and V is ISO3 either way (semantic split via v_dtype, no extra
            // payload). Only NVFP4 layers add the second-stage residual plane
            // set (K/V codes + scales): the residual is read by
            // gqa_attention_decode_nvfp4.cuh and the DType::NVFP4 prefill arm,
            // while gqa_attention_decode_iso3.cuh and the DType::ISO3 prefill
            // arm (which passes nullptrs, gqa_attention_prefill.cu:186-191)
            // never touch a residual plane. Gating it here -- not only in the
            // layer view -- keeps the plane count pushed below in step with
            // plane_cursor, so an ISO3 layer listed in --kv-residual-layers
            // cannot straddle the next layer's plane_base.
            const bool residual = selected == DType::NVFP4 && layer_has_residual(layer);
            residual_flags[layer] = residual;
            geometry.planes.push_back({DType::U8, head_dim / 2, kv_heads, 256});
            geometry.planes.push_back({DType::U8, head_dim / 2, kv_heads, 256});
            geometry.planes.push_back({DType::FP8_E4M3FN, head_dim / group, kv_heads, 256});
            geometry.planes.push_back({DType::FP8_E4M3FN, head_dim / group, kv_heads, 256});
            if (residual) {
                geometry.planes.push_back({DType::U8, head_dim / 2, kv_heads, 256});
                geometry.planes.push_back({DType::U8, head_dim / 2, kv_heads, 256});
                geometry.planes.push_back({DType::FP8_E4M3FN, head_dim / group, kv_heads, 256});
                geometry.planes.push_back({DType::FP8_E4M3FN, head_dim / group, kv_heads, 256});
            }
        }
        plane_cursor += selected == DType::BF16 ? 2U : (residual_flags[layer] ? 8U : 4U);
    }
    return PagedKVCacheLayout{
        .pages = plan_device_kv_page_pool(
            builder, DeviceKVPagePoolSpec{.page_group_count = physical_page_groups,
                                          .geometry         = std::move(geometry)}),
        .execution_tables = plan_kv_execution_tables(
            builder,
            KVExecutionTableSpec{.logical_page_capacity = logical_pages, .table_rows = table_rows}),
        .layers      = layers,
        .max_context = capacity,
        .kv_heads    = kv_heads,
        .head_dim    = head_dim,
        .dtype       = dtype,
        .quant_group = quant_group,
        .layer_dtypes = stored,
        .layer_residual = residual_flags,
        .layer_sliding_windows = window_flags,
        .layer_plane_base = plane_base,
    };
}

// --- Cold-slot record geometry ----------------------------------------------
// One cold slot holds one (physical page, kv_head, K|V) record, and the pool
// keeps one such record per layer per slot. The record stride is DERIVED from
// the attention geometry -- never a literal -- so a layer whose dtype feeds a
// different codec gets a different record instead of every layer paying the
// widest codec's width:
//
//   code plane  = page_tokens * (head_dim / 2)     packed E2M1 nibbles
//   scale plane = page_tokens * (head_dim / 16)    E4M3FN per-16 group scales
//
// Both codecs are built on that plane pair and differ only in header + payload:
//   int8 raw   : the code plane verbatim, 16 B header, no overflow path
//   nvfp4 rANS : `2 * StreamsPerHalf` rANS streams, 320 B header
// The rANS stream geometry follows the decode block shape (one 16-thread group
// per half page, one stream per thread), so a stream covers one 16th of the
// code plane and its budget is the entropy bound at `bits_per_code` per E2M1
// symbol:
//   stream_budget = ceil(stream_symbols * bits_per_code / 8)
// With bits_per_code == 4 (the storage nibble width) the budget equals the
// stream's uncompressed byte count, i.e. the no-expansion bound, and the
// derivation reproduces the two published record sizes exactly (static_asserts
// below). The requantizer measures 2.0-2.6 bits/code on real frames
// (include/ninfer/ops/entropy_cold_requant.h), which is the lever that lets the
// rANS record fall BELOW the nvfp4 resident plane (64 * head_dim/2 +
// 64 * head_dim/16 = 9216 B at head_dim 256): at a 2.6 bits/code ceiling the
// record is 6688 B, 27% smaller than the plane it replaces.
// kColdSlotRansBitsPerCode is the single place that ceiling is chosen; it stays
// at the no-expansion bound by default so the shipped geometry is unchanged. An
// over-tight ceiling is SAFE -- an rANS stream that overflows its budget clears
// the slot's valid flag and the page stays hot -- but it makes the pool inert,
// so it must be a measured decision, not a default.
inline constexpr std::int32_t kColdSlotNibbleBits = 4; // E2M1 symbol width
inline constexpr std::int32_t kColdSlotRansBitsPerCode = kColdSlotNibbleBits;
// Every record is 16-byte aligned: the rANS scale tail and the scale scatter
// move uint4, and every (page, head, plane) offset inside a record is a
// multiple of the stride, so a 16-byte stride keeps them all aligned.
inline constexpr std::int32_t kColdSlotAlignBytes         = 16;
inline constexpr std::int32_t kColdSlotRefHeadDim         = 256; // the codecs' own geometry
inline constexpr std::int32_t kColdSlotRansStreamsPerHalf = 16;  // decode block shape
inline constexpr std::int32_t kColdSlotScaleGroup = kNvfp4KvQuantGroup; // the codec's per-16 tail
inline constexpr std::int32_t kColdSlotRansStreams = 2 * kColdSlotRansStreamsPerHalf;

// The reference plane pair: used to RECOVER the codec constants (their headers)
// instead of repeating byte counts here.
inline constexpr std::int32_t kColdSlotRefCodePlane =
    kPagedKVPageSize * (kColdSlotRefHeadDim / 2);
inline constexpr std::int32_t kColdSlotRefScalePlane =
    kPagedKVPageSize * (kColdSlotRefHeadDim / kColdSlotScaleGroup);
inline constexpr std::int32_t kColdSlotRawHeaderBytes =
    ops::kColdI8SlotBytes - kColdSlotRefCodePlane - kColdSlotRefScalePlane;
inline constexpr std::int32_t kColdSlotRansHeaderBytes =
    ops::kEntropyNvfp4SlotBytes - kColdSlotRefCodePlane - kColdSlotRefScalePlane;

enum class ColdSlotCodec : std::uint8_t { None, Int8Raw, Nvfp4Rans };

// Which cold-slot codec a resolved layer dtype feeds; mirrors the device-side
// dispatch in enqueue_cold_compressions (program_impl.h): int8 -> raw slot,
// nvfp4 -> rANS slot, everything else has no cold codec at all.
[[nodiscard]] constexpr ColdSlotCodec cold_slot_codec_of(DType dtype) noexcept {
    switch (dtype) {
    case DType::I8: return ColdSlotCodec::Int8Raw;
    case DType::NVFP4: return ColdSlotCodec::Nvfp4Rans;
    default: return ColdSlotCodec::None;
    }
}

[[nodiscard]] constexpr std::int32_t cold_slot_round_up(std::int32_t value,
                                                        std::int32_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

// Record stride of one codec at `bits_per_code` entropy per E2M1 symbol.
[[nodiscard]] constexpr std::int32_t cold_slot_stride_bytes(ColdSlotCodec codec,
                                                            std::int32_t head_dim,
                                                            std::int32_t page_tokens,
                                                            std::int32_t bits_per_code) {
    const std::int32_t code_plane  = page_tokens * (head_dim / 2);
    const std::int32_t scale_plane = page_tokens * (head_dim / kColdSlotScaleGroup);
    // rANS stream geometry: one stream per thread of a half-page decode group,
    // each covering one 16th of the code plane; the budget is the entropy bound
    // at `bits_per_code` per E2M1 symbol.
    const std::int32_t stream_symbols = (code_plane * 2) / kColdSlotRansStreams;
    const std::int32_t stream_budget  = (stream_symbols * bits_per_code + 7) / 8;
    const std::int32_t header =
        codec == ColdSlotCodec::Int8Raw ? kColdSlotRawHeaderBytes : kColdSlotRansHeaderBytes;
    const std::int32_t payload =
        codec == ColdSlotCodec::Int8Raw ? code_plane : kColdSlotRansStreams * stream_budget;
    // Floor: a record must hold its header, at least 4 bytes per stream (the
    // rANS stream terminator) and its scale tail.
    const std::int32_t floor_bytes = header + 4 * kColdSlotRansStreams + scale_plane;
    // The RAW slot cannot follow the geometry down: cold_i8_slot_pack_kernel
    // writes a FIXED header + reference code plane + reference scale tail
    // layout whatever stride it is handed, so a narrower record would overrun
    // into the next record. At the codec's own geometry the derived size IS that
    // fixed layout (static_assert below); at any other geometry the codec is
    // outside its contract and the record stays at the safe published width.
    const std::int32_t fixed_raw_record =
        kColdSlotRawHeaderBytes + kColdSlotRefCodePlane + kColdSlotRefScalePlane;
    std::int32_t record = header + payload + scale_plane;
    if (codec == ColdSlotCodec::Int8Raw && record < fixed_raw_record) {
        record = fixed_raw_record;
    }
    if (record < floor_bytes) { record = floor_bytes; }
    return cold_slot_round_up(record, kColdSlotAlignBytes);
}

// The record stride a layer's resolved dtype gets. Dtypes with no cold codec
// keep the widest (default) record: the pool still reserves one per layer, so
// their footprint is exactly what it is today and nothing shrinks behind a
// consumer that does not exist yet.
[[nodiscard]] constexpr std::int32_t cold_slot_stride_for(DType dtype, std::int32_t head_dim,
                                                          std::int32_t page_tokens) {
    if (cold_slot_codec_of(dtype) == ColdSlotCodec::Int8Raw) {
        return cold_slot_stride_bytes(ColdSlotCodec::Int8Raw, head_dim, page_tokens, 0);
    }
    return cold_slot_stride_bytes(ColdSlotCodec::Nvfp4Rans, head_dim, page_tokens,
                                  kColdSlotRansBitsPerCode);
}

// The two published record sizes must FALL OUT of the derivation at the geometry
// the codecs are written for. If either codec's header, stream count or scale
// tail changes, this fails the build instead of silently mis-sizing every pool.
static_assert(cold_slot_stride_for(DType::I8, kColdSlotRefHeadDim, kPagedKVPageSize) ==
                  ops::kColdI8SlotBytes,
              "int8 cold-slot derivation drifted from ops::kColdI8SlotBytes");
static_assert(cold_slot_stride_for(DType::NVFP4, kColdSlotRefHeadDim, kPagedKVPageSize) ==
                  ops::kEntropyNvfp4SlotBytes,
              "nvfp4 cold-slot derivation drifted from ops::kEntropyNvfp4SlotBytes");
// Alignment: the rANS scale tail / scatter move uint4 and every in-record
// offset is a multiple of the stride.
static_assert(cold_slot_stride_for(DType::I8, kColdSlotRefHeadDim, kPagedKVPageSize) %
                      kColdSlotAlignBytes ==
                  0,
              "int8 cold-slot record is not 16-byte aligned");
static_assert(cold_slot_stride_for(DType::NVFP4, kColdSlotRefHeadDim, kPagedKVPageSize) %
                      kColdSlotAlignBytes ==
                  0,
              "nvfp4 cold-slot record is not 16-byte aligned");

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;

    // S28: apply the row-scale sidecar before any KV write path can observe the
    // table. Env-gated (NINFER_KV_ROWSCALE); unset => false, nothing happens.
    // SEPARATION: the switch is now three-state -- auto (baked table, the
    // default), off (kernel-side identity, no file needed) and <path>. The
    // explicit spec from --kv-row-scale wins over the environment.
    if (spec.kv_row_scale_spec.empty()) {
        (void)ninfer::ops::kv_rowscale_sidecar_apply_from_env(
            static_cast<std::uint32_t>(spec.full_attention_layers),
            static_cast<std::uint32_t>(spec.kv_heads),
            static_cast<std::uint32_t>(spec.attention_head_dim),
            0);  // model artifact hash: round 2 (needs options plumbing)
    } else {
        (void)ninfer::ops::kv_rowscale_sidecar_apply_spec(
            spec.kv_row_scale_spec, static_cast<std::uint32_t>(spec.full_attention_layers),
            static_cast<std::uint32_t>(spec.kv_heads),
            static_cast<std::uint32_t>(spec.attention_head_dim), 0);
    }
    // SEPARATION: the SO(4) rotation gate, applied at the same commit point as
    // the row scale so no KV write path can observe a half-separated module.
    // --kv-rotation wins over NINFER_KV_ROTATION; both are state-based (the
    // enabled state is written back explicitly, see kv_rotation_apply_spec).
    if (spec.kv_rotation_off) {
        (void)ninfer::ops::kv_rotation_apply_spec("off");
    } else {
        (void)ninfer::ops::kv_rotation_apply_from_env();
    }
    const std::span<const DType> layer_dtypes(spec.layer_kv_dtypes.data(),
                                              spec.full_attention_layers);
    const std::span<const bool> layer_residual(spec.layer_residual.data(),
                                               spec.full_attention_layers);
    const std::span<const std::uint32_t> layer_windows(spec.layer_sliding_windows.data(),
                                                       spec.full_attention_layers);
    // SEPARATION: refuse an E2M1 V plane on a layer whose V is decoded by a
    // mechanism that only knows ISO3 (residual plane / cold pool).
    kv_v_codec_check(spec.kv_v_codec, layer_dtypes, layer_residual,
                     spec.full_attention_layers, spec.max_cold_pages);
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                layer_dtypes, layer_residual, layer_windows,
                                spec.kv_table_rows, spec.text_physical_page_groups);
    layout.text_kv.kv_v_codec = spec.kv_v_codec;
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                   {}, {}, {}, spec.kv_table_rows,
                                   spec.mtp_physical_page_groups);
        // MTP layers inherit the resolved global dtype (no per-layer table), so
        // they follow the same V codec rule through layer_dtypes_.empty().
        layout.mtp_kv->kv_v_codec = spec.kv_v_codec;
    }
    // Cold pool slots: one record per (page, kv_head, K|V plane) and one tensor
    // per layer. The record stride is the LAYER's codec width, derived from the
    // attention geometry above: the INT8 tier packs requantized E2M1 nibbles
    // into 9232 B raw records, the NVFP4 tier rANS-encodes the same plane pair
    // into 9536 B records (320 B header + the stream budget + the scale tail).
    // Sizing every layer for the rANS maximum used to charge an all-int8 pool
    // 304 B per (page, head, plane) it could never use.
    if (spec.max_cold_pages != 0) {
        const std::uint32_t cold_pages = spec.max_cold_pages;
        std::int32_t widest_slot_bytes = 0;
        layout.text_kv.max_cold_pages  = spec.max_cold_pages;
        for (std::uint32_t layer = 0; layer < spec.full_attention_layers; ++layer) {
            // plan_cache already resolved BF16 override slots against the global
            // dtype, so layer_dtypes[layer] is the dtype every consumer of this
            // layer's cold records (PagedKVCache::layer_view, the codec dispatch
            // in enqueue_cold_compressions) will see.
            const std::int32_t stride = cold_slot_stride_for(
                layout.text_kv.layer_dtypes[layer], spec.attention_head_dim, kPagedKVPageSize);
            layout.text_kv.layer_slot_bytes[layer] = stride;
            widest_slot_bytes = widest_slot_bytes > stride ? widest_slot_bytes : stride;
            layout.text_kv.cold_slots[layer] = builder.add_tensor(
                DType::U8, {stride, static_cast<std::uint32_t>(spec.kv_heads), 2, cold_pages},
                256, "cold slots L" + std::to_string(layer));
            layout.text_kv.cold_slot_valid[layer] = builder.add_tensor(
                DType::I32, {static_cast<std::uint32_t>(spec.kv_heads), 2, cold_pages}, 256,
                "cold slot valid L" + std::to_string(layer));
        }
        // Widest record in the pool: the disk staging bound and the diagnostic
        // reported by PagedKVCache::slot_bytes(). Records are addressed with the
        // per-layer stride, never with this.
        layout.text_kv.slot_bytes = widest_slot_bytes;
    }
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pages_(backing, layout.pages), execution_tables_(backing, layout.execution_tables, pages_),
      layers_(layout.layers), max_context_(layout.max_context), kv_heads_(layout.kv_heads),
      head_dim_(layout.head_dim), dtype_(layout.dtype), quant_group_(layout.quant_group),
      slot_bytes_(layout.slot_bytes), max_cold_pages_(layout.max_cold_pages),
      layer_slot_bytes_(layout.layer_slot_bytes),
      layer_dtypes_(layout.layer_dtypes), layer_residual_(layout.layer_residual),
      layer_sliding_windows_(layout.layer_sliding_windows),
      layer_plane_base_(layout.layer_plane_base), kv_v_codec_(layout.kv_v_codec) {
    cold_slot_used_.assign(max_cold_pages_, 0);
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        if (layout.cold_slots[layer].region.bytes != 0) {
            cold_slots_[layer] = layout.cold_slots[layer].bind(backing);
            cold_slot_valid_[layer] = layout.cold_slot_valid[layer].bind(backing);
        }
    }
}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept
    : cache_(&cache), block_table_(block_table) {}

std::int32_t PagedKVCache::allocate_cold_slot() noexcept {
    if (max_cold_pages_ == 0) { return -1; }
    for (std::uint32_t slot = 0; slot < max_cold_pages_; ++slot) {
        if (!cold_slot_used_[slot]) {
            cold_slot_used_[slot] = true;
            return static_cast<std::int32_t>(slot);
        }
    }
    return -1;
}

void PagedKVCache::release_cold_slot(std::int32_t slot) noexcept {
    if (slot >= 0 && static_cast<std::uint32_t>(slot) < max_cold_pages_) {
        cold_slot_used_[slot] = false;
    }
}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, block_table_);
}

PagedKVCacheView PagedKVCache::execution_view(const KVExecutionRowLease& row) const {
    if (!row.belongs_to(execution_tables_)) {
        throw std::invalid_argument("Paged KV execution row belongs to another cache");
    }
    return PagedKVCacheView(*this, execution_tables_.row(row.handle()));
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    // The scaled/stride decision follows the layer's resolved dtype so a
    // per-layer table (PR1) can mix quantized and BF16 layers in one pool.
    const DType layer_dtype =
        layer_dtypes_.empty() ? dtype_ : layer_dtypes_[layer];
    const bool scaled = layer_dtype == DType::I8 || layer_dtype == DType::FP8_E4M3FN ||
                        layer_dtype == DType::NVFP4 || layer_dtype == DType::E8Kv ||
                        layer_dtype == DType::ISO3;
    const std::size_t base   = layer_plane_base_.empty()
                                    ? static_cast<std::size_t>(layer) * (scaled ? 4ULL : 2ULL)
                                    : layer_plane_base_[layer];
    const bool residual = layer < layers_ && !layer_residual_.empty() &&
                          layer_residual_[layer] && layer_dtype == DType::NVFP4;
    // E3: per-layer SWA window (0 = full attention; every kernel guards on
    // sliding_window > 0, so absent/zero entries are inert).
    const std::uint32_t window = layer < layers_ && !layer_sliding_windows_.empty()
                                     ? layer_sliding_windows_[layer]
                                     : 0U;
    return PagedKVLayerView{
        .k_pages       = pages_.plane(base),
        .v_pages       = pages_.plane(base + 1),
        .k_scale_pages = scaled ? pages_.plane(base + 2) : Tensor(),
        .v_scale_pages = scaled ? pages_.plane(base + 3) : Tensor(),
        .k_residual_pages = residual ? pages_.plane(base + 4) : Tensor(),
        .k_residual_scale_pages = residual ? pages_.plane(base + 5) : Tensor(),
        .v_residual_pages = residual ? pages_.plane(base + 6) : Tensor(),
        .v_residual_scale_pages = residual ? pages_.plane(base + 7) : Tensor(),
        .block_table   = block_table,
        .cold_slots    = cold_slots_[layer],
        .cold_slot_valid = cold_slot_valid_[layer],
        // Per-layer record stride: this layer's codec width, not the pool-wide
        // widest record. Every cold read/write in the kernels and in
        // program_impl.h uses this value.
        .cold_slot_bytes = layer_slot_bytes(layer),
        .slot_bytes = layer_slot_bytes(layer),
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .layer_index   = static_cast<std::int32_t>(layer),
        .dtype         = layer_dtypes_.empty() ? dtype_ : layer_dtypes_[layer],
        .quant_group   = layer_dtypes_.empty()
                             ? quant_group_
                             : kv_layer_quant_group(layer_dtypes_[layer]),
        .v_dtype       = layer_dtypes_.empty() ? dtype_ : kv_layer_v_dtype(layer_dtypes_[layer], kv_v_codec_),
        .v_quant_group = layer_dtypes_.empty()
                             ? quant_group_
                             : kv_layer_v_quant_group(layer_dtypes_[layer]),
        .sliding_window_tokens = window,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    // The scaled/stride decision follows the layer's resolved dtype so a
    // per-layer table (PR1) can mix quantized and BF16 layers in one pool.
    const DType layer_dtype =
        layer_dtypes_.empty() ? dtype_ : layer_dtypes_[layer];
    const bool scaled = layer_dtype == DType::I8 || layer_dtype == DType::FP8_E4M3FN ||
                        layer_dtype == DType::NVFP4 || layer_dtype == DType::E8Kv ||
                        layer_dtype == DType::ISO3;
    const std::size_t base   = layer_plane_base_.empty()
                                    ? static_cast<std::size_t>(layer) * (scaled ? 4ULL : 2ULL)
                                    : layer_plane_base_[layer];
    const bool residual = layer < layers_ && !layer_residual_.empty() &&
                          layer_residual_[layer] && layer_dtype == DType::NVFP4;
    // E3: per-layer SWA window (0 = full attention; every kernel guards on
    // sliding_window > 0, so absent/zero entries are inert).
    const std::uint32_t window = layer < layers_ && !layer_sliding_windows_.empty()
                                     ? layer_sliding_windows_[layer]
                                     : 0U;
    return PagedKVBatchLayerView{
        .k_pages       = pages_.plane(base),
        .v_pages       = pages_.plane(base + 1),
        .k_scale_pages = scaled ? pages_.plane(base + 2) : Tensor(),
        .v_scale_pages = scaled ? pages_.plane(base + 3) : Tensor(),
        .k_residual_pages = residual ? pages_.plane(base + 4) : Tensor(),
        .k_residual_scale_pages = residual ? pages_.plane(base + 5) : Tensor(),
        .v_residual_pages = residual ? pages_.plane(base + 6) : Tensor(),
        .v_residual_scale_pages = residual ? pages_.plane(base + 7) : Tensor(),
        .block_tables  = execution_tables_.matrix(),
        .cold_slots    = cold_slots_[layer],
        .cold_slot_valid = cold_slot_valid_[layer],
        // Per-layer record stride: this layer's codec width, not the pool-wide
        // widest record. Every cold read/write in the kernels and in
        // program_impl.h uses this value.
        .cold_slot_bytes = layer_slot_bytes(layer),
        .slot_bytes = layer_slot_bytes(layer),
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .layer_index   = static_cast<std::int32_t>(layer),
        .dtype         = layer_dtypes_.empty() ? dtype_ : layer_dtypes_[layer],
        .quant_group   = layer_dtypes_.empty()
                             ? quant_group_
                             : kv_layer_quant_group(layer_dtypes_[layer]),
        .v_dtype       = layer_dtypes_.empty() ? dtype_ : kv_layer_v_dtype(layer_dtypes_[layer], kv_v_codec_),
        .v_quant_group = layer_dtypes_.empty()
                             ? quant_group_
                             : kv_layer_v_quant_group(layer_dtypes_[layer]),
        .sliding_window_tokens = window,
    };
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::targets::qwen3_6
