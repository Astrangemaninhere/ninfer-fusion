#include <ninfer/targets/qwen3_6/decoder_state.h>
#include "ninfer/ops/cold_i8.h"
#include "ninfer/ops/entropy_nvfp4_slot.h"
#include "ops/kernel/gqa_isoquant_rot_gate.h"
#include "ops/kernel/gqa_isoquant_row_scale_loader.h"
#include "product/kv_component_switch.h"
// For product::e8_kv_code_bytes_per_8: the code plane's own row arithmetic, so the two
// narrower e8 arms below cannot be a lookalike of the derivation the ladder row and the
// cold table are priced from (product/kv_e8_width.h).
#include "product/kv_e8_width.h"
#include "product/kv_rowscale_persist.h"
// For product::kKvColdPoolStrideBytes: the cold cost model's stride, asserted against this
// file's own cold_slot_stride_for() and ops::kEntropyNvfp4SlotBytes in one static_assert.
#include "product/kv_tier_formats.h"
// For product::kKvBitBudgetColdSlotBytes: the ladder's cold grid point, the THIRD statement
// of the same record and the one nothing else in the tree could check. Host-only/std-only,
// so this costs the engine target nothing but the include.
#include "product/kv_bit_budget.h"

#include <algorithm>
#include <cstdio>
#include <charconv>
#include <cstdlib>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
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
// (NVFP4, FP8_E4M3FN, ISO4E) must carry quant_group 16 with E4M3 group scales,
// while I8/Rk4v4Kv carry 64-wide FP16 scales.
std::int32_t kv_layer_quant_group(DType dtype) {
    switch (dtype) {
    case DType::I8:
    case DType::E8Kv:
    // The narrow e8 widths narrow the K CODE plane only: V and the scale plane are the
    // W4 ones, so the group is 64 here too. They have to be named rather than left to the
    // default arm, which answers 0 -- and 0 is not a wrong group, it is the divisor of the
    // `head_dim % group` check that runs before the geometry below.
    case DType::E8K3Kv:
    case DType::E8K2Kv:
        return kKvInt8QuantGroup;
    case DType::FP8_E4M3FN:
        return kKvFp8QuantGroup;
    case DType::NVFP4:
    // ISO4E shares the NVFP4 plane geometry -- two codes per byte and per-16
    // E4M3FN scales -- and only swaps the nibble decoder, so it takes the same
    // group. Leaving ISO4E out of this table used to resolve to that group only
    // by accident, through the final `else` arm of the old ternary chain.
    case DType::ISO3:
        return kNvfp4KvQuantGroup;
    default:
        return 0;
    }
}

// The tiers whose plane pair is (codes, group scales) rather than a bare 16-bit plane --
// the plane INVENTORY question, next to the group WIDTH question above because a tier has
// to answer both to be allocated. The two layer views below read this to decide whether
// k_scale_pages/v_scale_pages are planes or empty Tensors, so a tier that owns scale
// planes and is missing here publishes NULLS over a geometry plan_cache() reserved both
// for, which is a view/geometry disagreement the allocator cannot see.
[[nodiscard]] constexpr bool kv_layer_has_scale_plane(DType dtype) noexcept {
    return dtype == DType::I8 || dtype == DType::FP8_E4M3FN || dtype == DType::NVFP4 ||
           dtype == DType::E8Kv || dtype == DType::E8K3Kv || dtype == DType::E8K2Kv ||
           dtype == DType::ISO3;
}

// V storage for a resolved layer dtype. The NVFP4 tier stores V in the ISO4E
// codec (a semantic split inside the same planes); an ISO4E layer keeps ISO4E V,
// which is exactly what gqa_attention_decode_iso3.cuh reads and writes back.
//
// SEPARATION: the NVFP4-tier V codec is now selectable. `codec` only changes
// what the producers encode and the consumers decode -- both codecs use the
// same two-codes-per-byte plane with per-16 E4M3FN scales, so no geometry,
// allocation or launcher change follows from it. E2M1 resolves an NVFP4 layer
// to NVFP4 V, i.e. the `v_dtype == ISO4E` tests in the launchers fall through to
// the <NVFP4, NVFP4> kernel that already exists. Plain ISO4E layers keep ISO4E V
// either way: their K is ISO4E too, and the knob is documented as an NVFP4-tier
// switch only.
DType kv_layer_v_dtype(DType dtype, KvVCodec codec) {
    if (dtype != DType::NVFP4) { return dtype; }
    return codec == KvVCodec::E2M1 ? DType::NVFP4 : DType::ISO3;
}

// SEPARATION guard. Two mechanisms hardcode ISO4E V and do NOT read v_dtype, so
// an E2M1 V plane would be silently mis-decoded by them:
//   * V residual planes (gqa_attention_prefill_fill_nvfp4k_iso3v_kernel writes
//     ISO4E; the prefill and decode producers only decode the residual on the
//     VVDType == ISO4E / Iso3V branch);
//   * the entropy cold pool for NVFP4 layers (program_impl.h requantizes V with
//     Iso4eVG16 unconditionally).
// Refuse loudly instead of approximating -- the repository's own idiom.
//
// SEPARATION (domain): the FIRST thing checked is that the switch can reach a
// kernel at all. kv_layer_v_dtype() below is the identity for every dtype other
// than NVFP4, so on a stack with no NVFP4 layer `--kv-v-codec e2m1` used to
// parse, validate, get reported as accepted and change nothing. That is the
// repository's definition of fake completeness, so it is now refused with the
// missing part named (product/kv_component_switch.h).
//
// SEPARATION (domain), second half: the two refusals below must resolve a BF16 per-layer
// slot exactly as plan_cache() does -- selected = (override == BF16) ? kv_dtype : override,
// decoder_state.cpp layer_dtype() and layouts_impl.h:1464 -- because the layer that reaches
// the kernels is the RESOLVED one. Comparing the raw override table made both refusals walk
// past their own case: with the global dtype on nvfp4 and an all-BF16 (or partly BF16)
// override table every layer IS an NVFP4 layer, so `--kv-v-codec e2m1` there really did put
// an E2M1 V plane under a V residual plane / into the entropy cold pool, both of which
// decode ISO4E only -- V was then silently mis-decoded, which is the exact failure this guard
// exists to prevent. The domain check above already ran on RESOLVED dtypes, which is why it
// accepted the configuration while the loops below found nothing to refuse: the two halves
// of one guard disagreed about the same layers.
//
// BF16-mask: `layer_dtypes_set` is plan_cache()'s "was this slot written?" mask, and it is
// threaded through BOTH halves of this guard plus the domain predicate, because a slot the
// spec wrote as BF16 is a real BF16 layer and not an NVFP4 one -- the same resolution
// plan_cache() now performs (kv_resolve_slot_dtype). Without it the new configuration would
// re-open the hole above from the other side: with `--kv-dtype nvfp4 --kv-layer-storage
// 0-11:bf16`, layers 0..11 are NVFP4 when the mask is ignored and BF16 when it is honoured.
void kv_v_codec_check(KvVCodec codec, DType kv_dtype, std::span<const DType> layer_dtypes,
                      std::span<const bool> layer_dtypes_set,
                      std::span<const bool> layer_residual, std::uint32_t layers,
                      std::uint32_t max_cold_pages) {
    if (codec == KvVCodec::Iso3) { return; }
    {
        const std::size_t reach_layers = std::min<std::size_t>(layer_dtypes.size(), layers);
        const std::string reach = ninfer::product::kv_component_switch_domain_error(
            ninfer::product::KvComponentSwitch::VCodec, kv_dtype,
            layer_dtypes.first(reach_layers),
            layer_dtypes_set.first(
                std::min<std::size_t>(layer_dtypes_set.size(), reach_layers)));
        if (!reach.empty()) { throw std::invalid_argument(reach); }
    }
    // The resolved tier of one layer, with plan_cache()'s inheritance rule. An empty table
    // inherits the global dtype wholesale; a layer past the end of a short table is treated
    // the same way rather than indexed out of bounds. A slot the spec WROTE as BF16 stays
    // BF16, so it is not an NVFP4 layer and needs no refusal.
    const auto resolved_layer_dtype = [&](std::uint32_t layer) {
        const DType override_dtype =
            layer < layer_dtypes.size() ? layer_dtypes[layer] : DType::BF16;
        const bool slot_explicit =
            layer < layer_dtypes_set.size() && layer_dtypes_set[layer];
        return ninfer::product::kv_resolve_slot_dtype(kv_dtype, override_dtype, slot_explicit);
    };
    for (std::uint32_t i = 0; i < layers; ++i) {
        if (resolved_layer_dtype(i) != DType::NVFP4) { continue; }
        if (i < layer_residual.size() && layer_residual[i]) {
            throw std::invalid_argument(
                "kv-v-codec e2m1: NVFP4 layer " + std::to_string(i) +
                " keeps a V residual plane, and the residual codec is ISO4E-only; "
                "disable --kv-residual-layers or keep --kv-v-codec iso4e");
        }
    }
    if (max_cold_pages != 0) {
        for (std::uint32_t i = 0; i < layers; ++i) {
            if (resolved_layer_dtype(i) == DType::NVFP4) {
                throw std::invalid_argument(
                    "kv-v-codec e2m1: the cold pool requires ISO4E V for NVFP4 layers "
                    "(the eviction requant is Iso4eVG16); use --cold-policy none or keep "
                    "--kv-v-codec iso4e");
            }
        }
    }
}

// Diagnostic metadata only (PagedKVLayerView::v_quant_group): published by the
// tiers whose V plane is an E4M3 group-scale plane.
std::int32_t kv_layer_v_quant_group(DType dtype) {
    return dtype == DType::NVFP4 || dtype == DType::ISO3 ? kNvfp4KvQuantGroup : 0;
}

// ==== L26 prototype: per-layer KV PLANE DISCARD =============================
// NINFER_KV_DROP_LAYERS="0,3,7" or "2-5" makes plan_cache() push NO planes for
// those FULL-ATTENTION indices, so the device KV page pool really shrinks by
// (that layer's plane bytes x physical pages): each plane is planned as its OWN
// tensor, see the geometry loop below. Unset / empty => the table is all-false
// and every path is byte-identical to the baseline. A measurement instrument,
// not a product knob.
//
// l26c instrument fixes, each with a negative control that turns red:
//   D1  "A-B" is a RANGE. The old strtol() read "0-15" as the single layer 0.
//   D2  a token that is not a pure decimal (or A-B) is an ERROR naming the
//       variable and the token. The old strtol() read "abc"/"x"/"7x" as 0 and
//       silently discarded layer 0.
//   D3  the spec is passed INTO plan_cache, so the MTP cache -- planned with an
//       empty spec -- never loses its only layer. The old code re-read the env
//       var inside every cache, so any list containing 0 killed the whole run
//       with "Paged KV geometry has no planes" and text layer 0 was unusable.
//   D4  discarding every layer is refused HERE, by name, instead of surfacing as
//       a generic geometry error from paged_kv_cache.cpp.
//   D5  a discarded layer is published as DISCARDED by PagedKVCache::layer_is_dropped()
//       (and KvCacheStorage::Dropped in the per-layer reflection) instead of showing the
//       view's default BF16. Deliberately kept OFF the view structs: the ops API takes
//       PagedKVLayerView/PagedKVBatchLayerView BY VALUE, so adding a field there would
//       change their size and silently break any prebuilt libninfer_ops.a.
struct KvLayerDropSpec {
    std::array<bool, 64> table{};
    std::uint32_t count = 0;

    [[nodiscard]] bool any() const noexcept { return count != 0; }
    [[nodiscard]] bool dropped(std::uint32_t layer) const noexcept {
        return layer < table.size() && table[layer];
    }
};

// Strict parser. `layers` is the full-attention layer count of THIS cache and is
// the domain a token must land in: an index this cache cannot address is a typo,
// not a no-op.
[[nodiscard]] KvLayerDropSpec parse_kv_layer_drop(const char* env_name, std::uint32_t layers) {
    KvLayerDropSpec spec;
    const std::string var(env_name);
    const char* const text = std::getenv(env_name);
    if (text == nullptr || *text == '\0') { return spec; }
    const std::string_view all(text);
    const auto parse_index = [&](std::string_view token) -> std::uint32_t {
        if (token.empty()) {
            throw std::invalid_argument(var + ": empty layer index in \"" + std::string(text) +
                                        "\"");
        }
        std::uint32_t value  = 0;
        const char* first    = token.data();
        const char* const last = token.data() + token.size();
        const auto parsed    = std::from_chars(first, last, value);
        if (parsed.ec != std::errc{} || parsed.ptr != last) {
            throw std::invalid_argument(var + ": \"" + std::string(token) +
                                        "\" is not a decimal layer index (spec \"" +
                                        std::string(text) + "\")");
        }
        if (value >= layers) {
            throw std::invalid_argument(var + ": layer index " + std::to_string(value) +
                                        " is out of range for this cache (" +
                                        std::to_string(layers) + " full-attention layers; spec \"" +
                                        std::string(text) + "\")");
        }
        return value;
    };
    const auto add = [&](std::uint32_t index) {
        if (!spec.table[index]) {
            spec.table[index] = true;
            ++spec.count;
        }
    };
    std::size_t cursor = 0;
    while (cursor <= all.size()) {
        const std::size_t comma = all.find(',', cursor);
        const std::string_view token =
            all.substr(cursor, comma == std::string_view::npos ? all.size() - cursor
                                                               : comma - cursor);
        if (!token.empty()) {
            const std::size_t dash = token.find('-');
            if (dash == std::string_view::npos) {
                add(parse_index(token));
            } else {
                const std::uint32_t range_first = parse_index(token.substr(0, dash));
                const std::uint32_t range_last  = parse_index(token.substr(dash + 1));
                if (range_first > range_last) {
                    throw std::invalid_argument(var + ": reversed range \"" +
                                                std::string(token) + "\" (spec \"" +
                                                std::string(text) + "\")");
                }
                for (std::uint32_t index = range_first; index <= range_last; ++index) { add(index); }
            }
        }
        if (comma == std::string_view::npos) { break; }
        cursor = comma + 1;
    }
    if (spec.count == 0) {
        // D2 (continued): a NON-EMPTY spec that names no layer at all ("," / ",,") is
        // a typo. Empty tokens are skipped so a trailing comma is tolerated, but if
        // that leaves nothing, silence is exactly the failure mode D2 is about.
        throw std::invalid_argument(var + ": \"" + std::string(text) +
                                    "\" names no layer (empty/blank entries only)");
    }
    if (spec.count == layers) {
        // D4: the width=0 endpoint of the discard curve. The engine cannot build a
        // pool with no planes at all (paged_kv_cache.cpp rejects an empty plane
        // inventory), so refuse it here with a message that names the variable
        // instead of letting a generic "geometry has no planes" escape.
        throw std::invalid_argument(
            var + ": discarding all " + std::to_string(layers) +
            " full-attention layers leaves the KV page pool with no planes, which the "
            "engine cannot build; drop at least one layer fewer (the width=0 endpoint of "
            "the discard curve is not reachable by this instrument, by design).");
    }
    return spec;
}
// ===========================================================================
// `layer_dtypes_set` is the "was this slot written?" mask of DecoderStateSpec
// (see EngineOptions::kv_layer_storage_set). It is deliberately NOT defaulted:
// every caller must state whether it has a mask, so a new call site cannot
// silently fall back to the inheritance-only rule and drop the operator's
// per-layer BF16 request -- the exact class of silent hole this file keeps
// removing. Pass `{}` for "no slot was written".
PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                              std::int32_t kv_heads, std::int32_t head_dim, DType dtype,
                              std::int32_t quant_group, std::span<const DType> layer_dtypes,
                              std::span<const bool> layer_dtypes_set,
                              std::span<const bool> layer_residual,
                              std::span<const std::uint32_t> layer_windows,
                              std::int32_t table_rows, std::uint32_t physical_page_groups,
                              const KvLayerDropSpec& layer_drop) {
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
    if (!layer_dtypes_set.empty() && layer_dtypes_set.size() < layers) {
        throw std::invalid_argument(
            "Paged KV per-layer dtype mask is shorter than the layer count");
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
    // Per-layer resolution: a BF16 entry inherits the global dtype UNLESS the spec
    // actually wrote that slot, in which case the entry is authoritative -- that is
    // the per-layer BF16 request (kv_resolve_slot_dtype, which is the same rule
    // this lambda used to spell inline). `slot_explicit` false is the historical
    // behaviour, bit-for-bit, and it is the default for every slot of the mask.
    // Accepted per-layer storages are the quantized codecs with their native group.
    const auto layer_dtype = [&](std::uint32_t layer) {
        const DType override_dtype = layer_dtypes.empty() ? DType::BF16 : layer_dtypes[layer];
        const bool slot_explicit = layer < layer_dtypes_set.size() && layer_dtypes_set[layer];
        const DType selected =
            ninfer::product::kv_resolve_slot_dtype(dtype, override_dtype, slot_explicit);
        if (selected != DType::BF16 && selected != DType::I8 && selected != DType::NVFP4 &&
            selected != DType::FP8_E4M3FN && selected != DType::E8Kv &&
            selected != DType::E8K3Kv && selected != DType::E8K2Kv &&
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
    std::array<bool, 64> dropped_flags{};
    std::uint32_t dropped_count = 0;
    std::uint32_t plane_cursor = 0;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const DType selected = layer_dtype(layer);
        stored[layer]        = selected;
        window_flags[layer]  = layer_windows.empty() ? 0U : layer_windows[layer];
        plane_base[layer]    = plane_cursor;
        // L26 prototype: a dropped layer contributes ZERO planes to the pool. Its
        // plane_base therefore aliases the next layer's, which is why every reader
        // has to go through layer_view()/batch_layer_view() (both guarded below).
        if (layer_drop.dropped(layer)) {
            // L26 prototype: a discarded layer contributes ZERO planes to the pool.
            // Its plane_base therefore aliases the next layer's, which is why every
            // reader has to go through layer_view()/batch_layer_view().
            dropped_flags[layer] = true;
            ++dropped_count;
            continue;
        }
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
            // Rk4v4 tier: packed 4-bit E8-lattice K codes + i4 V codes (two per
            // byte) with per-64 FP16 scales (int8-kernel path).
            geometry.planes.push_back({DType::U8, head_dim / 2, kv_heads, 256});
            geometry.planes.push_back({DType::U8, head_dim / 2, kv_heads, 256});
            geometry.planes.push_back({DType::FP16, head_dim / group, kv_heads, 256});
            geometry.planes.push_back({DType::FP16, head_dim / group, kv_heads, 256});
        } else if (selected == DType::E8K3Kv || selected == DType::E8K2Kv) {
            // Rk3v4/Rk2v4 tiers: the SAME K+V plane pair and the same per-64 FP16 scale
            // plane as Rk4v4 above -- the family narrows the K code plane and nothing
            // else -- but the K row is this width's own: 8 elements into 3 bytes at W3
            // and into 2 at W2, i.e. 6656 B and 4608 B per head-page at head_dim 256
            // (product/kv_e8_width.h, where the ladder row moves by half a bit per
            // element because only K moves). The row is spelled through that header's
            // own per-8 constant and scaled by THIS model's head_dim, because the
            // header's byte counts are pinned to the reference 256: a 128-wide d256
            // model must get 48/32 here or the plane is sized for another model.
            // `head_dim % group` above already forces the 8-alignment the /8 needs,
            // since group is 64 for every e8 tier.
            const product::E8KvWidth width = selected == DType::E8K3Kv
                                                 ? product::E8KvWidth::W3
                                                 : product::E8KvWidth::W2;
            const std::int32_t k_code_row =
                head_dim / 8 * product::e8_kv_code_bytes_per_8(width);
            geometry.planes.push_back({DType::U8, k_code_row, kv_heads, 256});
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
            // NVFP4/ISO4E tier: identical plane geometry. K carries E2M1 (NVFP4)
            // or sign-magnitude ISO4E nibbles, both over per-16 E4M3FN scales,
            // and V is ISO4E either way (semantic split via v_dtype, no extra
            // payload). Only NVFP4 layers add the second-stage residual plane
            // set (K/V codes + scales): the residual is read by
            // gqa_attention_decode_nvfp4.cuh and the DType::NVFP4 prefill arm,
            // while gqa_attention_decode_iso3.cuh and the DType::ISO3 prefill
            // arm (which passes nullptrs, gqa_attention_prefill.cu:186-191)
            // never touch a residual plane. Gating it here -- not only in the
            // layer view -- keeps the plane count pushed below in step with
            // plane_cursor, so an ISO4E layer listed in --kv-residual-layers
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
        .layer_dropped = dropped_flags,
        .layer_dropped_count = dropped_count,
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
// RETRACTED 2026-09-18. The band stated in the paragraph above -- and repeated at
// "the ceiling that pays (2.60 b/c, the top of the 2.0-2.6 band ...)" and at "2.60
// is the rate the requantizer MEASURES (2.0-2.6 bits/code on real frames)" below --
// does NOT exist. The instrument is dl/ransceil/rans_probe.cu (see its REPORT.md):
// it includes THIS tree's own kernel headers and runs the REAL
// entropy_nvfp4_slot_encode_kernel on real engine-dumped .kvc planes, and on Qwen3.8
// 27B it measures 3.31-3.86 bits/code (K 3.8455 mean, worst half 3.8613; V 3.3133)
// requantized, against 3.64-3.95 for the g16 storage codes it replaces. The worst K
// stream is 259 B = 4.0469 bits/symbol, ABOVE the 4-bit nibble width.
//
// So the conclusion of the paragraph above is superseded as well, not just its rate:
// at the ceiling this file now pins (404, see the block below) the cold record is
// 9632 B, which is WIDER than the 9216 B nvfp4 plane pair it replaces -- it is a
// device-memory COST, not a 27% saving. Nothing in the paragraph above, nor either of
// its two echoes, may be used to re-derive a ceiling: at 2.60 bits/code the codec
// validates 0 of 64 K slots and produces nothing at all.
//
// REASONING CORRECTED 2026-09-18 (LONG1M). The retraction above says the NUMBERS are
// wrong; a reader must also know
// the ARGUMENT is wrong, because the argument -- not the number -- is what a future ceiling
// gets re-derived from. The paragraph above infers in three steps and all three fail:
//   * "the requantizer measures 2.0-2.6 bits/code ... which is the lever that lets the rANS
//     record fall BELOW the nvfp4 resident plane" -- there is no such lever. Both measured
//     rates sit ABOVE the 4-bit nibble width: the requantized codes measure 3.31-3.86
//     bits/code (K 3.8196 mean on the long-context instrument, 3.8455 on the predecessor's
//     row scale; V 3.3424), against 3.64-3.95 for the g16 storage codes they replace. A
//     lever that would have to point DOWN to explain a saving points UP instead.
//   * "the rANS record fall BELOW the nvfp4 resident plane (9216 B)" -- the plane's own byte
//     parity is not a break-even this codec can sit at. At 9216 B the per-stream budget is
//     (9216 - 320 - 1024) / 32 = 246 B = 3.84 bits/code, and real K streams exceed it on
//     512 of 512 pages: that record validates 0 of 2048 K slots at 64k context and at 130k
//     context alike, and 0 of 64 on the earlier 16-page sample. Parity with the plane is a
//     DEAD WIDTH for this codec, not a boundary the record may approach.
//   * "at a 2.6 bits/code ceiling the record is 6688 B, 27% smaller than the plane it
//     replaces" -- this is not merely stale, its SIGN is inverted, and that is the
//     load-bearing error. Below the 4-bit no-expansion bound the payload stops being the
//     whole code plane and becomes its own budgeted size, so the record shrinks only because
//     its budget shrinks -- and a budget that cannot hold a real page carries no page. At
//     2.60 bits/code the codec is INERT (0 of 2048 K slots): it compresses nothing while the
//     ledger prices it. The record that DOES encode is 9632 B (b=404), which is 416 B WIDER
//     than the 9216 B pair it replaces. There is therefore no width at which the nvfp4 K
//     plane both encodes and saves device memory, and "27% smaller" cannot be recovered by
//     choosing a different ceiling: it is unreachable in this geometry, not unrealised.
// What still stands in that paragraph, and is the basis of the derivation below: the plane
// geometry (64 * head_dim/2 + 64 * head_dim/16 = 9216 B at head_dim 256) and the statement
// that the rANS stream geometry follows the decode block shape. No record size and no bit
// rate may be carried forward from it.
// kColdSlotRansBitsPerCodeX100 is the single place that ceiling is chosen. The
// unit is HUNDREDTHS of a bit per code, not whole bits, and that is load-bearing:
// the ceiling that pays (2.60 b/c, the top of the 2.0-2.6 band the requantizer
// measures on real frames) is not an integer, and every integer value lands on a
// DIFFERENT record -- `int bits_per_code = 2` gives 5440 B, `= 3` gives 7488 B,
// neither of which is the 6688 B the measurement asks for. Chosen in whole bits
// the ceiling can only be 2, 3 or 4 bits/code, so the one value the evidence
// supports is simply unrepresentable.
//
// REASONING CORRECTED 2026-09-18 (LONG1M). The CONCLUSION of the paragraph above
// survives; its PREMISE does not, and the
// difference matters because the premise is the part that is quoted elsewhere.
//   * Dead premise: "the ceiling that pays (2.60 b/c, the top of the 2.0-2.6 band the
//     requantizer measures on real frames) is not an integer". The band is retracted above,
//     and the ceiling that pays is 404 hundredths -- 4.04 bits/code.
//   * Why the unit is STILL hundredths, and now for a better reason: 404 is not chosen from
//     any band, it is INVERTED from a measured byte count. The encoder takes its per-stream
//     budget from the slot stride, so the stride must cover the largest stream the codec was
//     measured to produce (259 B per 512-symbol stream, on the K plane). ceil(512 * b / 800)
//     >= 259 has b = 404 as its least integer solution -- 403 gives 258 B, one byte short.
//     The requirement is a fraction of a bit per code because 259 bytes over 512 symbols IS
//     a fraction of a bit per code, so the fixed-point unit, the integer ceil and the
//     `denominator - 1` hazard described below all remain load-bearing. Only the sentence
//     that says WHY the unit is needed is replaced.
//   * The whole-bit contrast, re-derived for the record that is actually pinned: a whole-bit
//     ceiling can only be 2, 3 or 4 bits/code, which give 5440 B, 7488 B and 9536 B, and
//     NONE of them is the 9632 B this file pins. So the paragraph's conclusion -- that the
//     one value the evidence supports is unrepresentable in whole bits -- is unchanged. The
//     record it must name is 9632 B; the 6688 B it names is the record of a ceiling that
//     validates nothing, and 5440 B / 7488 B remain correct as the whole-bit records.
//
// The derivation below is therefore fixed-point INTEGER arithmetic, not floating
// point: the ceil is done by adding `denominator - 1` before dividing, so there is
// no rounding mode to get wrong and no truncate-vs-ceil hazard. A `double`
// ceiling with a plain `/8` would silently truncate 166.4 to 166, i.e. run the
// slot at 2.59 b/c while the constant says 2.60.
//
// The ceiling is SAFE to tighten: an rANS stream that overflows its budget clears
// the slot's valid flag and the page stays hot (entropy_nvfp4_slot_kernels.cuh
// pass A/B), so an over-tight ceiling costs hit rate, never correctness.
inline constexpr std::int32_t kColdSlotNibbleBits    = 4;   // E2M1 symbol width, bits
inline constexpr std::int32_t kColdSlotBitsScale     = 100; // the ceiling's fixed point
inline constexpr std::int32_t kColdSlotRansBitsPerCodeX100 = 404; // 4.04 bits/code: the measured minimum, NOT a preference -- see below
//
// 260 -> 404, 2026-09-18, and the reason is a MEASUREMENT that the previous value did not have.
// The encoder derives its per-stream budget from the SLOT STRIDE at runtime and does not clamp
// (ops/kernel/entropy_nvfp4_slot_kernels.cuh pass A/B: budget = (slot_bytes - 320 - 1024) / 32),
// so the stride IS the budget, and a page's real rANS streams must fit inside it or the kernel
// clears the slot's valid flag and the page stays hot. Measured on the real encoder over the 2048
// K streams of 16 engine-dumped Qwen3.8 27B pages (dl/ransceil/rans_probe.cu, which includes THIS
// tree's kernei header; see dl/ransceil/REPORT.md):
//
//   K, per 512-symbol rANS stream: min 243  p50 250  p90 252  p99 254  max 259  mean 249.61 B
//   V, same geometry:             min 191  p50 216  p90 224  p99 231  max 239  mean 215.56 B
//   hit rate by record:  6688 B -> K 0/64  V 0/64   (167 B budget: THE CODEC PRODUCES NOTHING)
//                       9216 B -> K 0/64  V 64/64   (246 B: byte parity with the nvfp4 plane pair)
//                       9408 B -> K 12/64 V 64/64   (252 B)
//                       9536 B -> K 61/64 V 64/64   (256 B: the 4.00 no-expansion record)
//                       9632 B -> K 64/64 V 64/64   (259 B: THIS value)
//
// The K plane binds because one stride serves both planes of a layer. 259 B over 512 symbols is
// 4.0469 bits/symbol -- ABOVE the 4-bit nibble width -- because the requantized K codes are
// near-uniform and the rANS coder's own table overhead puts the worst stream 3 bytes over the
// uncompressed width. So the ceiling that carries a page is ABOVE the no-expansion bound, and the
// old "< 400" invariant below was not a safety property: it forbade exactly the region the data
// lives in. At 2.60 bits/code the codec is inert, which is why this moved.
//
// THE PRICE, stated here because it is the point and must not be discovered later: 9632 B is WIDER
// than the 9216 B nvfp4 plane pair the cold page would give back, so at the ceiling that encodes,
// the cold rANS tier COSTS device memory on every class that can reach it (nvfp4/iso4e +416 B,
// rk4v4 +928 B per head-page). "cold is cheaper than every hot tier" was true only at 6688 -- the
// price of a slot that never validates, and a slot that never validates has no price. Pinned in
// product/kv_tier_formats.h.
inline constexpr std::int32_t kColdSlotRansMeasuredMaxStreamBytes = 259; // K, of the instrument above
inline constexpr std::int32_t kColdSlotRansMeasuredMaxStreamBytesV = 239; // V, same instrument
inline constexpr std::int32_t kColdSlotRansMeasuredKHitSlotsAt2Bit = 0;   // of 64, at 167 B/stream
inline constexpr std::int32_t kColdSlotRansMeasuredKHitSlotsAt4Bit = 61;  // of 64, at 256 B/stream
inline constexpr std::int32_t kColdSlotRansMeasuredKHitSlots = 64;        // of 64, at 259 B/stream
// The least ceiling (this file's hundredths of a bit) whose budget covers the measured maximum:
// ceil(512 * b / 800) >= 259 has b = 404 as its least integer solution (403 gives 258 B). DERIVED
// from the measured bytes, never hand-entered, so the two cannot drift.
inline constexpr std::int32_t kColdSlotRansRequiredX100 =
    (kColdSlotRansMeasuredMaxStreamBytes - 1) * kColdSlotBitsScale * 8 / 512 + 1;
//
// ⚠️ THE DEBT THIS VALUE CARRIES, and the reason it is stated here rather than assumed.
// This file's own rule is that the ceiling "must be a measured decision, not a default"
// (git show HEAD:.../decoder_state.cpp:275-279, kept verbatim in this block's history):
// tightening it costs HIT RATE, never correctness -- an rANS stream that overflows its
// 167 B budget clears the slot's valid flag and the page stays hot -- and the size of that
// cost is the fraction of pages whose per-stream rANS size exceeds 167 B. THAT FRACTION IS
// COUNTED NOWHERE. The whole tree carries exactly one signal for it: the `[cold] page N
// INVALID ...` line in program_impl.h (the validity_logged guard collapses a whole pass to
// one line per layer, so it cannot be tallied from the log either). 2.60 is the rate the
// requantizer MEASURES (2.0-2.6 bits/code on real frames), and 6688 B is inside every
// break-even in the COLD CODEC RULE (3.84 b/c for the 9216 B plane pair, 3.59 b/c for
// rk4v4's 8704 B one), so the value is defensible on the code-rate evidence; what is still
// owed is the FALLBACK count, which needs a slot_valid==0 tally this tree does not have.
//
// REASONING CORRECTED 2026-09-18 (LONG1M). This is the only correction to this DEBT
// block. It replaces the block's DEFENCE,
// not its debt: the debt (a hit-rate cost that is counted nowhere) is still owed, and so is
// the FALLBACK tally named in the line above. What cannot stand is the sentence "2.60 is the
// rate the requantizer MEASURES (2.0-2.6 bits/code on real frames), and 6688 B is inside
// every break-even in the ... CODEC RULE ..., so the value is defensible on the code-rate
// evidence". That sentence is false in three separate ways:
//   * "2.60 is the rate the requantizer MEASURES" -- retracted above. What was measured is
//     3.31-3.86 bits/code requantized (K 3.8196 mean long-context, 3.8455 on the
//     predecessor's row scale; V 3.3424) against 3.64-3.95 for the g16 codes replaced.
//   * "6688 B is inside every break-even" -- the record under discussion is 9632 B, and the
//     comparison now points the OTHER WAY. The break-even ceilings are still the ones this
//     block states (3.84 b/c for the 9216 B plane pair, 3.59 b/c for rk4v4's smaller one),
//     but the pinned ceiling is 4.04 b/c, which is ABOVE both. "Inside every break-even" was
//     therefore true only of a ceiling that validates nothing; at the ceiling that encodes,
//     the cold rANS tier does NOT break even on any class -- it COSTS device memory
//     (nvfp4/iso4e +416 B, rk4v4 +928 B per head-page; the "THE PRICE" paragraph of the block
//     below states this in full).
//   * "defensible on the code-rate evidence" -- the defence is no longer a code-rate
//     preference. It is a measurement of the encoder's own maximum stream: the value moved
//     because at the previous ceiling the codec produced NOTHING on real K pages.
// This block's own `167 B` is the OLD budget. The enforced budget is the stride's own,
// (9632 - 320 - 1024) / 32 = 259 B, so the debt described below is the fraction of pages
// whose per-stream rANS size exceeds 259 B -- not 167 B. The structure of the debt is
// unchanged, and the sentence that follows it (no tally exists for that fraction) is the
// part of this block that is still exactly true.
static_assert(kColdSlotRansRequiredX100 == 404,
              "the measured requirement (259 B per 512-symbol stream) no longer inverts to 404 "
              "hundredths of a bit: re-derive kColdSlotRansMeasuredMaxStreamBytes from the "
              "instrument (dl/ransceil/rans_probe.cu) before changing either");
static_assert(kColdSlotRansMeasuredMaxStreamBytes > kColdSlotNibbleBits * kColdSlotBitsScale / 8,
              "the measured K requirement is at or below the 4-bit nibble width (256 B per 512 "
              "symbols): if the requantizer's output really did fit in the no-expansion record, "
              "the 2026-09-18 measurement in dl/ransceil/REPORT.md (max 259 B) is stale and the "
              "ceiling, the stride and the cold tier's PRICE all have to be re-derived together");
// The ceiling must COVER the measurement. This is the invariant that replaces the old
// `kColdSlotRansBitsPerCodeX100 < 4 * 100` UPPER bound, which was removed for two reasons:
//   * its stated reason was "at or above 4.00 b/c the record is strictly wider than the nvfp4
//     resident plane it replaces". But the record at its OWN pinned value (3.99 b/c -> 9536 B)
//     is already 320 B wider than that 9216 B plane. It never gated what it claimed to gate, so
//     nothing was protected by keeping it.
//   * the measurement puts the requirement ABOVE it (4.0469 bits/symbol on the binding plane),
//     so the bound forbade exactly the region in which the codec works -- 0 of 64 slots at
//     2.60 b/c, 61 of 64 at 3.99, 64 of 64 at 4.04.
// The bound that DOES have content is the lower one, and it is what stops the shape of defect
// this file is recovering from: a ceiling nobody measured, justified by a band that no
// instrument produced.
static_assert(kColdSlotRansBitsPerCodeX100 >= kColdSlotRansRequiredX100,
              "the rANS ceiling's per-stream budget does not cover the largest stream the real "
              "encoder was measured to produce: the encoder clears slot_valid on real K pages "
              "and the cold pool compresses NOTHING while the ledger prices it. Raise "
              "kColdSlotRansBitsPerCodeX100 to at least kColdSlotRansRequiredX100 and re-derive "
              "product/kv_tier_formats.h (kKvColdRansStreamBytes, kKvColdPoolStrideBytes and the "
              "cold tier's SIGN) together with product/kv_bit_budget.h");
static_assert(kColdSlotRansBitsPerCodeX100 > 0,
              "the rANS ceiling must be positive");
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
// The rANS header is spelled out, NOT recovered the same way. Recovering it works
// only while the record is `header + the WHOLE code plane + scale tail`, which is
// exactly the 4-bit no-expansion bound. Below it the payload is its own budgeted
// size, so `stride - code_plane - scale_plane` stops being the header: at
// 2.60 bits/code it is 6688 - 8192 - 1024 = -2528, a NEGATIVE header that cancels
// the compressed payload back out and derives a 3840 B record. That record fits
// the pool, is below the nvfp4 plane, and passes every byte verdict -- while the
// encoder's own budget, taken from the same stride, comes out at 78 B per stream
// and overflows on every page, clearing every valid flag and leaving the cold tier
// silently inert while the report advertises a saving. The literal below is the
// real header, and ops/kernel/entropy_nvfp4_slot.cuh:61 already pins
// sizeof(EntropyNvfp4SlotHeader) to it; the assert at the bottom of this block
// pins that this literal is what the ops stride implies.
inline constexpr std::int32_t kColdSlotRansHeaderBytes = 320;
static_assert(kColdSlotRansHeaderBytes == 320,
              "the rANS header must match sizeof(EntropyNvfp4SlotHeader)");

enum class ColdSlotCodec : std::uint8_t { None, Int8Raw, Nvfp4Rans };

// Which cold-slot codec a resolved layer dtype feeds; mirrors the device-side
// dispatch in enqueue_cold_compressions (program_impl.h): int8 -> raw slot,
// nvfp4 -> rANS slot, everything else has no cold codec at all.
[[nodiscard]] constexpr ColdSlotCodec cold_slot_codec_of(DType dtype) noexcept {
    switch (dtype) {
    case DType::I8: return ColdSlotCodec::Int8Raw;
    case DType::NVFP4: return ColdSlotCodec::Nvfp4Rans;
    // BF16-COLD-LAND A1. A bf16 layer feeds the SAME raw slot the int8 tier does: its
    // resident plane (head_dim x tokens bf16 = 32768 B/head-page, and no scale plane at
    // all) is requantized to packed E2M1 nibbles plus one E4M3FN scale per 64 channels
    // by entropy_cold_requant's Bf16G64 arm, and cold_i8_slot_pack_kernel writes the
    // fixed 9232 B record. There is no entropy budget on this path -- the raw slot is a
    // fixed layout with no overflow case -- so the rANS ceiling
    // kColdSlotRansBitsPerCodeX100 is not involved and this does not re-price any other
    // dtype. Without this arm a bf16 stack has no cold codec, and a bf16 stack is what
    // the MTP tree verify needs (its per-column masks exist only on the bf16 routes:
    // gqa_attention.cpp refuses a quantized KV tier by name), so the tree and the cold
    // tier could not both be on at once.
    case DType::BF16: return ColdSlotCodec::Int8Raw;
    // BF16-COLD-LAND E1. Same record, same slot, for the Rk4v4 tier -- and for a stronger
    // reason than bf16: an rk4v4 layer's K/V planes already ARE packed 4-bit codes (one
    // nibble per element, 128 B per 64-token row) with an fp16 scale per 64 channels
    // (gqa_attention_decode_i8.cuh's Rk4v4 arm), so the cold record holds those codes
    // VERBATIM (entropy_cold_requant's Rk4v4KvG64 arm) and only the scale is requantized.
    // Without a codec here, ONE rk4v4 layer is enough to leave the pool Closed and retire
    // nothing (the page is all-or-nothing), and the 27B target's factory per-layer table
    // puts rk4v4 on six of its sixteen full-attention layers.
    case DType::E8Kv: return ColdSlotCodec::Int8Raw;
    default: return ColdSlotCodec::None;
    }
}

[[nodiscard]] constexpr std::int32_t cold_slot_round_up(std::int32_t value,
                                                        std::int32_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

// Record stride of one codec at `bits_per_code_x100` hundredths of a bit of
// entropy per E2M1 symbol (0 for the raw codec, which does not use the ceiling).
[[nodiscard]] constexpr std::int32_t cold_slot_stride_bytes(ColdSlotCodec codec,
                                                            std::int32_t head_dim,
                                                            std::int32_t page_tokens,
                                                            std::int32_t bits_per_code_x100) {
    const std::int32_t code_plane  = page_tokens * (head_dim / 2);
    const std::int32_t scale_plane = page_tokens * (head_dim / kColdSlotScaleGroup);
    // rANS stream geometry: one stream per thread of a half-page decode group,
    // each covering one 16th of the code plane; the budget is the entropy bound
    // at `bits_per_code` per E2M1 symbol.
    const std::int32_t stream_symbols = (code_plane * 2) / kColdSlotRansStreams;
    // ceil(stream_symbols * bits_per_code / 8) in fixed point. The budget this
    // produces is also what the encoder enforces, but only indirectly: the kernel
    // re-derives it from the stride as (slot_bytes - header - scales) / streams
    // (entropy_nvfp4_slot_kernels.cuh pass A), and the two agree exactly as long
    // as the record is header + streams * budget + scales -- which it is, at every
    // ceiling, because 32 * budget is a multiple of the 16-byte alignment.
    const std::int32_t stream_budget = (stream_symbols * bits_per_code_x100 +
                                        kColdSlotBitsScale * 8 - 1) /
                                       (kColdSlotBitsScale * 8);
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
                                  kColdSlotRansBitsPerCodeX100);
}

// The two published record sizes must FALL OUT of the derivation at the geometry
// the codecs are written for. If either codec's header, stream count or scale
// tail changes, this fails the build instead of silently mis-sizing every pool.
static_assert(cold_slot_stride_for(DType::I8, kColdSlotRefHeadDim, kPagedKVPageSize) ==
                  ops::kColdI8SlotBytes,
              "int8 cold-slot derivation drifted from ops::kColdI8SlotBytes");
// BF16-COLD-LAND A1: the bf16 record must BE the int8 raw record, because both the
// reader (gqa_attention_decode_bf16.cuh) and the writer (cold_i8_slot_pack_raw) use
// that slot's fixed header/code/scale geometry. If this ever drifted, bf16 cold pages
// would be decoded off the end of their own record.
static_assert(cold_slot_stride_for(DType::BF16, kColdSlotRefHeadDim, kPagedKVPageSize) ==
                  ops::kColdI8SlotBytes,
              "bf16 cold slot must be the int8 raw record (BF16-COLD-LAND A1)");
// BF16-COLD-LAND E1: same for the rk4v4 tier. The reader is the Rk4v4 arm of the i8 decode
// kernel and the writer is cold_i8_slot_pack_raw, both of which use the raw record's
// fixed header/code/scale geometry.
static_assert(cold_slot_stride_for(DType::E8Kv, kColdSlotRefHeadDim, kPagedKVPageSize) ==
                  ops::kColdI8SlotBytes,
              "rk4v4 cold slot must be the int8 raw record (BF16-COLD-LAND E1)");
static_assert(cold_slot_stride_for(DType::NVFP4, kColdSlotRefHeadDim, kPagedKVPageSize) ==
                  ops::kEntropyNvfp4SlotBytes,
              "nvfp4 cold-slot derivation drifted from ops::kEntropyNvfp4SlotBytes");
// The ONE place the three independent statements of the same number meet: this file's
// derivation (the stride the arena is allocated at), product/kv_tier_formats.h's own
// integer ceil (the stride the COLD CODEC RULE prices, kept cuda-free so it repeats the
// arithmetic instead of including the ops header) and product/kv_bit_budget.h's
// kKvBitBudgetColdSlotBytes (the ladder's cold grid point). Before these asserts each of
// the three could be moved alone, and the cold tier's cost model could price a slot the
// allocator never sizes.
static_assert(product::kKvColdPoolStrideBytes == ops::kEntropyNvfp4SlotBytes,
              "the cold cost model and the ops slot geometry disagree: re-derive "
              "product/kv_tier_formats.h (kKvColdPoolStrideBytes) and "
              "product/kv_bit_budget.h (kKvBitBudgetColdSlotBytes) with this file");
// ASSERTATOM. The THIRD statement, and until now the unchecked one. This comment used to
// claim all three met here while the assert above covered only two -- and the untethered
// one is exactly the one that drifted: on 2026-09-18 the working tree carried
// kKvBitBudgetColdSlotBytes = 6688 in product/kv_bit_budget.h while this file derived 9536
// (kColdSlotRansBitsPerCodeX100 was 399) and ops::kEntropyNvfp4SlotBytes was 9536, so the
// arena allocated 9536 B per (page, kv_head, plane) and the ladder priced 327 b/el
// (== 6688 B) for it. Both files compiled. Neither was self-contradictory: kv_bit_budget.h
// only ever compared its constant against its OWN hand-entered literals (the 327 grid point
// at :389 and the 2544 difference at :407), so ANY self-consistent world satisfied it --
// 9536+466+(-304) compiles exactly as cleanly as 6688+327+2544. The defence cannot live in
// that file (host-only, and the authority is an ops header that needs cuda_runtime.h); it
// has to live in the one TU that can see both, which is this one. The authority is the
// literal the arena is allocated from, ops::kEntropyNvfp4SlotBytes, and this assert is what
// makes it one authority with two mirrors rather than three peers.
static_assert(product::kKvBitBudgetColdSlotBytes == ops::kEntropyNvfp4SlotBytes,
              "the ladder's cold grid point and the ops slot geometry disagree: "
              "product/kv_bit_budget.h kKvBitBudgetColdSlotBytes must equal "
              "ops::kEntropyNvfp4SlotBytes, the record cold_slot_stride_for() above hands the "
              "arena -- otherwise the DP prices a slot (and the audit at "
              "kv_bit_budget.h kv_bit_budget_head_page_bytes sums one) that is never allocated, "
              "and the cold tier's sign can invert with no build error.\n"
              "RECONCILE PATH (2026-09-18, COLDCEIL): kKvBitBudgetColdSlotBytes is no longer "
              "a literal -- it is DERIVED from product/kv_tier_formats.h kKvColdPoolStrideBytes, "
              "so this assert can only fire if that derivation is removed or that file moves "
              "alone. Record the before/after sha16 of BOTH files and reconcile them together "
              "(and re-derive kKvBitBudgetColdBitsX100, whose grid point follows the record). "
              "See dl/coldceil/REPORT.md");
// ASSERTATOM. The SAME hole, one constant over: kv_bit_budget.h:403 asserts
// `kKvBitBudgetColdSlotBytesInt8Raw == 9232` under the message "int8 raw cold record ==
// ops::kColdI8SlotBytes" -- i.e. it names the authority and then checks a hand-entered
// literal, exactly as the cold-slot assert above did. It is numerically correct today, so
// it is latent rather than live, but it is the same defect class and it costs nothing to
// close: the raw record's authority is the ops literal the int8 arena is allocated from, and
// the three asserts above (I8 / BF16 / E8Kv == ops::kColdI8SlotBytes) already pin it from
// side. Note this is deliberately NOT an assert that the two cold records are related to
// each other -- kv_bit_budget.h:407 does that (2544 == 9232 - 6688) and must keep moving
// with the ceiling.
static_assert(product::kKvBitBudgetColdSlotBytesInt8Raw == ops::kColdI8SlotBytes,
              "product/kv_bit_budget.h kKvBitBudgetColdSlotBytesInt8Raw is not the int8 raw "
              "record the arena allocates (ops::kColdI8SlotBytes): the ledger's second cold "
              "record has drifted from the codec that writes it");
// The ceiling itself, pinned. This is the number the whole cold-vs-resident byte
// table hangs off (product/kv_tier_formats.h carries the derivation), and moving
// it silently would re-price every cold plan in the tree: at the 4-bit bound the
// rANS record is 9536 B and every nvfp4-shaped layer COSTS memory, at 2.60 it is
// 6688 B and they all SAVE. The break-even ceilings are 3.84 b/c for the nvfp4
// plane pair and 3.59 b/c for rk4v4's smaller one, so 2.60 is inside both.
static_assert(kColdSlotRansBitsPerCodeX100 == kColdSlotRansRequiredX100,
              "the rANS ceiling moved off the measured minimum: re-derive the cold-vs-resident "
              "byte table in product/kv_tier_formats.h (kv_cold_class_bytes_of, "
              "kKvColdPoolStrideBytes) and the cold tier in product/kv_bit_budget.h "
              "(kKvBitBudgetColdSlotBytes, kKvBitBudgetColdBitsX100) with this file. A ceiling "
              "BELOW 404 leaves real K pages unencodable (0/64 at 2.60 b/c); a ceiling ABOVE it "
              "buys nothing the measurement asks for -- 404 already carries 64 of 64 -- and only "
              "widens a record that is ALREADY 416 B wider than the nvfp4 plane pair it replaces");
// The reference stream budget, DERIVED from the ceiling rather than spelled as a literal
// (the literal 256 satisfied both 3.99 and 4.00, so it could not report a ceiling move):
// ceil(512 codes x 2.60 / 8) == 167 B. product/kv_tier_formats.h computes the same value
// with its own integer ceil and static_asserts 167, and ops::kEntropyNvfp4SlotBytes is the
// record this budget produces.
inline constexpr std::int32_t kColdSlotRansRefStreamBudgetBytes =
    (kColdSlotRefCodePlane * 2 / kColdSlotRansStreams * kColdSlotRansBitsPerCodeX100 +
     kColdSlotBitsScale * 8 - 1) /
    (kColdSlotBitsScale * 8);
static_assert(kColdSlotRansRefStreamBudgetBytes == kColdSlotRansMeasuredMaxStreamBytes,
              "the record's per-stream budget is not the measured maximum: it is DERIVED from the "
              "ceiling and must EQUAL the largest stream the instrument produced (259 B), which is "
              "the only reason the value above is 404 and not 260. If this moved, re-derive "
              "kKvColdRansMeasuredMaxStreamBytes, kKvColdRansStreamBytes / "
              "kKvColdPoolStrideBytes in product/kv_tier_formats.h and "
              "kKvBitBudgetColdSlotBytes in product/kv_bit_budget.h together with this file");
static_assert(cold_slot_stride_for(DType::NVFP4, kColdSlotRefHeadDim, kPagedKVPageSize) ==
                  kColdSlotRansHeaderBytes +
                      kColdSlotRansStreams * kColdSlotRansRefStreamBudgetBytes +
                      kColdSlotRefScalePlane,
              "the 4.04 b/c record must be 320 B header + 32 x 259 B + 1024 B scales = "
              "9632 B, and 9632 > 9216 is why the cold rANS tier COSTS on the 4-bit "
              "classes at the only ceiling that encodes");
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

// N3 runtime loop: which KV tiers can use the row scale at all. The scale is
// applied by the NVFP4 K/V kernels and nowhere else
// (gqa_attention_decode_nvfp4.cuh:330 on the K write and :489 on the Q read,
// gqa_attention_prefill_nvfp4.cuh:499/720/1052). A stack with no NVFP4 layer
// therefore has nothing to calibrate, and the loop must not arm a capture for
// it.
//
// SEPARATION: the predicate is now the same one the switch guard uses
// (product/kv_component_switch.h), so "the calibration loop has nothing to do"
// and "--kv-row-scale cannot take effect" can never disagree. The layer count is
// clamped to the 64-wide table here as well: the span constructed at the call
// site below is validated by plan_cache(), which runs later.
// The RESOLVED per-layer KV store, compressed into "0,1,3:rk4v4 2,4-15:nvfp4", using the
// same rule plan_cache() applies (a BF16 table entry inherits the global dtype). Any
// diagnostic that speaks about "the KV store" must read this and not spec.kv_dtype:
// with the registered 27b default table the global dtype is bf16 while the layers are
// rk4v4/nvfp4, which is how the row-scale message came to claim "bf16 KV" on an int8 run.
std::string kv_store_spec_text(const DecoderStateSpec& spec,
                               std::span<const DType> layer_dtypes,
                               std::span<const bool> layer_dtypes_set) {
    const std::size_t layers =
        std::min<std::size_t>(spec.full_attention_layers, layer_dtypes.size());
    // The RESOLVED tier of one layer, mask included: a slot the spec wrote as BF16
    // is BF16 here, exactly as plan_cache() builds it (kv_resolve_slot_dtype). A
    // diagnostic that said "nvfp4" for a layer the pool built as BF16 would be the
    // same class of false report this function exists to remove.
    const auto resolved_at = [&](std::size_t layer) {
        const bool slot_explicit =
            layer < layer_dtypes_set.size() && layer_dtypes_set[layer];
        return ninfer::product::kv_resolve_slot_dtype(spec.kv_dtype, layer_dtypes[layer],
                                                      slot_explicit);
    };
    const auto name_of = [](DType dtype) -> std::string_view {
        switch (dtype) {
        case DType::BF16:
            return "bf16";
        case DType::I8:
            return "int8";
        case DType::FP8_E4M3FN:
            return "fp8-e4m3";
        case DType::NVFP4:
            return "nvfp4";
        case DType::ISO3:
            return "iso4e";
        case DType::E8Kv:
            return "rk4v4";
        // The two narrow widths, in the ladder's own tokens (product/kv_bit_budget.h
        // kKvBitBudgetTiers spec_name), so the resolved-store line does not answer "?"
        // for a layer this file now lays out.
        case DType::E8K3Kv:
            return "rk3v4";
        case DType::E8K2Kv:
            return "rk2v4";
        default:
            return "?";
        }
    };
    std::string out;
    for (std::size_t first = 0; first < layers;) {
        const DType resolved = resolved_at(first);
        std::size_t last = first;
        while (last + 1 < layers && resolved_at(last + 1) == resolved) {
            ++last;
        }
        if (!out.empty()) { out += ' '; }
        out += std::to_string(first);
        if (last != first) {
            out += '-';
            out += std::to_string(last);
        }
        out += ':';
        out += name_of(resolved);
        first = last + 1;
    }
    return out.empty() ? std::string("none") : out;
}

bool rowscale_domain_active(const DecoderStateSpec& spec) {
    const std::uint32_t layers =
        std::min<std::uint32_t>(spec.full_attention_layers, kPagedKVCacheMaxLayers);
    const std::span<const DType> layer_dtypes(spec.layer_kv_dtypes.data(), layers);
    const std::span<const bool> layer_dtypes_set(spec.layer_kv_dtypes_set.data(), layers);
    return ninfer::product::kv_component_switch_domain_active(
        ninfer::product::KvComponentSwitch::RowScale, spec.kv_dtype, layer_dtypes,
        layer_dtypes_set);
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;

    // S28: apply the row-scale sidecar before any KV write path can observe the
    // table. Env-gated (NINFER_KV_ROWSCALE); unset => false, nothing happens.
    // SEPARATION: the switch is three-state -- auto (baked table, the default),
    // off (kernel-side identity, no file needed) and <path>. The explicit spec
    // from --kv-row-scale wins over the environment, which wins over the loop.
    //
    // N3 runtime loop: the one state that had no owner -- mode auto with neither
    // an explicit spec nor NINFER_KV_ROWSCALE -- is decided by the persisted table
    // next to the artifact (product/kv_rowscale_persist.h). A table that is
    // present, parsed, carries this producer's schema and this run's KV
    // configuration fingerprint, matches the live geometry and is not older than
    // the artifact is LOADED (through the unchanged loader) and the capture is
    // skipped; anything else captures this run and writes the table at the end of
    // it. The other three cases are byte-identical to what they were.
    const std::uint32_t rowscale_layers =
        static_cast<std::uint32_t>(spec.full_attention_layers);
    const std::uint32_t rowscale_heads = static_cast<std::uint32_t>(spec.kv_heads);
    const std::uint32_t rowscale_dim = static_cast<std::uint32_t>(spec.attention_head_dim);
    ninfer::ops::KvRowScaleMode rowscale_mode = ninfer::ops::KvRowScaleMode::Auto;
    {
        std::string mode_path;
        std::string mode_err;
        (void)ninfer::ops::kv_rowscale_mode_from_spec(spec.kv_row_scale_spec, rowscale_mode,
                                                     mode_path, mode_err);
    }
    const bool rowscale_env_present = std::getenv("NINFER_KV_ROWSCALE") != nullptr;

    const std::span<const DType> layer_dtypes(spec.layer_kv_dtypes.data(),
                                              spec.full_attention_layers);
    // The "was this slot written?" mask, same length as the table above. Built
    // unconditionally: an all-false mask is exactly the pre-mask rule, so a spec
    // that carries no mask behaves as before at every consumer below.
    const std::span<const bool> layer_dtypes_set(spec.layer_kv_dtypes_set.data(),
                                                 spec.full_attention_layers);
    const std::span<const bool> layer_residual(spec.layer_residual.data(),
                                               spec.full_attention_layers);
    const std::span<const std::uint32_t> layer_windows(spec.layer_sliding_windows.data(),
                                                       spec.full_attention_layers);

    // SEPARATION: a component switch that cannot reach a kernel is REFUSED
    // rather than accepted-and-ignored, and it is refused BEFORE the two
    // descriptors below are written: a rejected run must not leave process-global
    // device state half-switched for the next engine in the same process.
    // product/kv_component_switch.h holds the domains and the evidence.
    if (rowscale_mode != ninfer::ops::KvRowScaleMode::Auto || rowscale_env_present) {
        const std::string reach = ninfer::product::kv_component_switch_domain_error(
            ninfer::product::KvComponentSwitch::RowScale, spec.kv_dtype, layer_dtypes,
            layer_dtypes_set);
        if (!reach.empty()) { throw std::invalid_argument(reach); }
    }
    if (rowscale_mode != ninfer::ops::KvRowScaleMode::Auto || rowscale_env_present) {
        if (spec.kv_row_scale_spec.empty()) {
            (void)ninfer::ops::kv_rowscale_sidecar_apply_from_env(
                rowscale_layers, rowscale_heads, rowscale_dim,
                0);  // model artifact hash: unspecified (the table's own stamp only)
        } else {
            (void)ninfer::ops::kv_rowscale_sidecar_apply_spec(spec.kv_row_scale_spec, rowscale_layers,
                                                             rowscale_heads, rowscale_dim, 0);
        }
    } else {
        const ninfer::product::KvRowScaleResolution rowscale =
            ninfer::product::kv_rowscale_persist_resolve(rowscale_layers, rowscale_heads,
                                                         rowscale_dim,
                                                         rowscale_domain_active(spec),
                                                         kv_store_spec_text(spec, layer_dtypes,
                                                                            layer_dtypes_set));
        if (!rowscale.report.empty()) {
            std::fprintf(stderr, "%s\n", rowscale.report.c_str());
        }
        if (rowscale.plan == ninfer::product::KvRowScalePlan::UsePersisted) {
            (void)ninfer::ops::kv_rowscale_sidecar_apply_spec(rowscale.apply_spec, rowscale_layers,
                                                             rowscale_heads, rowscale_dim, 0);
        } else if (!spec.kv_row_scale_spec.empty()) {
            // An EXPLICIT "auto" keeps its documented meaning even while a capture
            // is being armed: restore the baked geometry, so a previous engine's
            // "off" cannot leak into this one.
            (void)ninfer::ops::kv_rowscale_sidecar_apply_spec("auto", rowscale_layers,
                                                             rowscale_heads, rowscale_dim, 0);
        }
        // Otherwise the descriptor is left alone: the run uses the baked table,
        // which is exactly what mode auto meant before the loop existed.
    }
    // SEPARATION: the SO(4) rotation gate, applied at the same commit point as
    // the row scale so no KV write path can observe a half-separated module.
    // --kv-rotation wins over NINFER_KV_ROTATION; both are state-based (the
    // enabled state is written back explicitly, see kv_rotation_apply_spec).
    // The request is resolved first WITHOUT touching device state, so the guard
    // below can refuse a request that no tier can honour before the upload
    // happens. An unparsable NINFER_KV_ROTATION deliberately leaves the request
    // at "not off": kv_rotation_apply_from_env() below is what reports it, and
    // that error text must not be pre-empted here.
    {
        bool rotation_off = spec.kv_rotation_off;
        if (!rotation_off) {
            if (const char* const spec_env = std::getenv("NINFER_KV_ROTATION")) {
                ninfer::ops::KvRotationMode mode = ninfer::ops::KvRotationMode::Auto;
                std::string spec_err;
                if (ninfer::ops::kv_rotation_mode_from_spec(spec_env, mode, spec_err)) {
                    rotation_off = mode == ninfer::ops::KvRotationMode::Off;
                }
            }
        }
        if (rotation_off) {
            const std::string reach = ninfer::product::kv_component_switch_domain_error(
                ninfer::product::KvComponentSwitch::Rotation, spec.kv_dtype, layer_dtypes,
                layer_dtypes_set);
            if (!reach.empty()) { throw std::invalid_argument(reach); }
        }
    }
    if (spec.kv_rotation_off) {
        (void)ninfer::ops::kv_rotation_apply_spec("off");
    } else {
        (void)ninfer::ops::kv_rotation_apply_from_env();
    }
    // SEPARATION: refuse an E2M1 V plane on a layer whose V is decoded by a
    // mechanism that only knows ISO4E (residual plane / cold pool), and refuse it
    // outright when no layer is on the NVFP4 tier at all (the switch could not
    // take effect).
    kv_v_codec_check(spec.kv_v_codec, spec.kv_dtype, layer_dtypes, layer_dtypes_set,
                     layer_residual, spec.full_attention_layers, spec.max_cold_pages);
    // L26 instrument (D3): the discard table is parsed ONCE, against the TEXT
    // cache's own full-attention count, and handed to the text plan only. The MTP
    // cache below is planned with an empty spec, so its single layer stays usable.
    const KvLayerDropSpec text_layer_drop =
        parse_kv_layer_drop("NINFER_KV_DROP_LAYERS", spec.full_attention_layers);
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                layer_dtypes, layer_dtypes_set, layer_residual, layer_windows,
                                spec.kv_table_rows, spec.text_physical_page_groups,
                                text_layer_drop);
    layout.text_kv.kv_v_codec = spec.kv_v_codec;
    if (spec.enable_mtp) {
        // MTP layers carry no per-layer table, so an empty table AND an empty mask:
        // every MTP layer is the resolved global dtype.
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                   {}, {}, {}, {}, spec.kv_table_rows,
                                   spec.mtp_physical_page_groups, KvLayerDropSpec{});
        // MTP layers inherit the resolved global dtype: plan_cache() is handed an empty
        // span and resolves every slot to the pool-wide dtype, so the table it returns is
        // authoritative for them too. (This used to read "...through
        // layer_dtypes_.empty()", which is a constant false on std::array<DType, 64>.)
        layout.mtp_kv->kv_v_codec = spec.kv_v_codec;
    }
    // Cold pool slots: one record per (page, kv_head, K|V plane) and one tensor
    // per layer. The record stride is the LAYER's codec width, derived from the
    // attention geometry above: the INT8 tier packs requantized E2M1 nibbles
    // into 9232 B raw records, the NVFP4 tier rANS-encodes the same plane pair
    // into 9632 B records (320 B header + 32 x 259 B stream budget at the measured
    // 4.04 bits/code ceiling + the 1024 B scale tail). Sizing every layer for the rANS
    // maximum used to charge an all-int8 pool 304 B per (page, head, plane) it
    // could never use -- and at the measured ceiling the rANS record is now the
    // WIDER of the two, by 400 B, which is why the ladder's single cold grid point
    // cannot price both (product/kv_bit_budget.h kKvBitBudgetColdSlotBytesInt8Raw).
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
      layer_plane_base_(layout.layer_plane_base),
      layer_dropped_(layout.layer_dropped), dropped_layers_(layout.layer_dropped_count),
      kv_v_codec_(layout.kv_v_codec) {
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
    // L26 prototype: a dropped layer owns no planes. Hand back an EMPTY view so no
    // stray reader can silently read the NEXT layer planes (the plane base of a
    // dropped layer aliases its successor). Designated initialisers follow the
    // declaration order of PagedKVLayerView.
    if (layer_dropped_[layer]) {
        return PagedKVLayerView{.block_table  = block_table,
                                .head_dim     = head_dim_,
                                .num_kv_heads = kv_heads_,
                                .layer_index  = static_cast<std::int32_t>(layer)};
    }
    // The scaled/stride decision follows the layer's resolved dtype so a
    // per-layer table (PR1) can mix quantized and BF16 layers in one pool.
    // layer_dtypes_ is std::array<DType, 64>, so `.empty()` is a CONSTANT false and the
    // `dtype_` fallback could never be taken. It was never needed either: plan_cache()
    // resolves the global inheritance itself (kv_resolve_slot_dtype,
    // decoder_state.cpp:319-330), writes every slot (decoder_state.cpp:353,360-362), and
    // THROWS on a table shorter than the layer count (decoder_state.cpp:296-298). See the
    // note on the table declarations in decoder_state.h before restoring any fallback.
    const DType layer_dtype = layer_dtypes_[layer];
    const bool scaled = kv_layer_has_scale_plane(layer_dtype);
    // Same constant-false guard: layer_plane_base_ is std::array<std::uint32_t, 64>, and
    // plan_cache() writes plane_base[layer] for EVERY layer (decoder_state.cpp:364) before
    // any dropped layer is skipped, so the `layer * (scaled ? 4 : 2)` fallback was both
    // unreachable and WRONG for a mixed-tier pool: it ignores the per-layer stride.
    const std::size_t base = layer_plane_base_[layer];
    // layer_residual_ is std::array<bool, 64>: `.empty()` is a constant false, so that
    // conjunct never fired. The real bound is `layer < layers_`, and it stays.
    const bool residual =
        layer < layers_ && layer_residual_[layer] && layer_dtype == DType::NVFP4;
    // E3: per-layer SWA window (0 = full attention; every kernel guards on
    // sliding_window > 0, so absent/zero entries are inert).
    // layer_sliding_windows_ is std::array<std::uint32_t, 64>: `.empty()` is a constant
    // false, so that conjunct never fired. plan_cache() writes window_flags[layer] for
    // every layer (decoder_state.cpp:363); the real bound is `layer < layers_`.
    const std::uint32_t window =
        layer < layers_ ? layer_sliding_windows_[layer] : 0U;
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
        .dtype         = layer_dtypes_[layer],
        .quant_group   = kv_layer_quant_group(layer_dtypes_[layer]),
        .v_dtype       = kv_layer_v_dtype(layer_dtypes_[layer], kv_v_codec_),
        .v_quant_group = kv_layer_v_quant_group(layer_dtypes_[layer]),
        .sliding_window_tokens = window,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    // L26 prototype: empty view for a dropped layer (see layer_view above).
    if (layer_dropped_[layer]) {
        return PagedKVBatchLayerView{.block_tables = execution_tables_.matrix(),
                                     .head_dim     = head_dim_,
                                     .num_kv_heads = kv_heads_,
                                     .layer_index  = static_cast<std::int32_t>(layer)};
    }
    // The scaled/stride decision follows the layer's resolved dtype so a
    // per-layer table (PR1) can mix quantized and BF16 layers in one pool.
    // layer_dtypes_ is std::array<DType, 64>, so `.empty()` is a CONSTANT false and the
    // `dtype_` fallback could never be taken. It was never needed either: plan_cache()
    // resolves the global inheritance itself (kv_resolve_slot_dtype,
    // decoder_state.cpp:319-330), writes every slot (decoder_state.cpp:353,360-362), and
    // THROWS on a table shorter than the layer count (decoder_state.cpp:296-298). See the
    // note on the table declarations in decoder_state.h before restoring any fallback.
    const DType layer_dtype = layer_dtypes_[layer];
    const bool scaled = kv_layer_has_scale_plane(layer_dtype);
    // Same constant-false guard: layer_plane_base_ is std::array<std::uint32_t, 64>, and
    // plan_cache() writes plane_base[layer] for EVERY layer (decoder_state.cpp:364) before
    // any dropped layer is skipped, so the `layer * (scaled ? 4 : 2)` fallback was both
    // unreachable and WRONG for a mixed-tier pool: it ignores the per-layer stride.
    const std::size_t base = layer_plane_base_[layer];
    // layer_residual_ is std::array<bool, 64>: `.empty()` is a constant false, so that
    // conjunct never fired. The real bound is `layer < layers_`, and it stays.
    const bool residual =
        layer < layers_ && layer_residual_[layer] && layer_dtype == DType::NVFP4;
    // E3: per-layer SWA window (0 = full attention; every kernel guards on
    // sliding_window > 0, so absent/zero entries are inert).
    // layer_sliding_windows_ is std::array<std::uint32_t, 64>: `.empty()` is a constant
    // false, so that conjunct never fired. plan_cache() writes window_flags[layer] for
    // every layer (decoder_state.cpp:363); the real bound is `layer < layers_`.
    const std::uint32_t window =
        layer < layers_ ? layer_sliding_windows_[layer] : 0U;
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
        .dtype         = layer_dtypes_[layer],
        .quant_group   = kv_layer_quant_group(layer_dtypes_[layer]),
        .v_dtype       = kv_layer_v_dtype(layer_dtypes_[layer], kv_v_codec_),
        .v_quant_group = kv_layer_v_quant_group(layer_dtypes_[layer]),
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
