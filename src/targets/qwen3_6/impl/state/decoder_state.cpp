#include <ninfer/targets/qwen3_6/decoder_state.h>
#include "ninfer/ops/cold_i8.h"
#include "ninfer/ops/entropy_nvfp4_slot.h"

#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                              std::int32_t kv_heads, std::int32_t head_dim, DType dtype,
                              std::int32_t quant_group, std::span<const DType> layer_dtypes,
                              std::span<const bool> layer_residual,
                              std::int32_t table_rows, std::uint32_t physical_page_groups) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    if (!layer_dtypes.empty() && layer_dtypes.size() < layers) {
        throw std::invalid_argument("Paged KV per-layer dtype table is shorter than the layer count");
    }
    if (!layer_residual.empty() && layer_residual.size() < layers) {
        throw std::invalid_argument("Paged KV per-layer residual table is shorter than the layer count");
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
            selected != DType::FP8_E4M3FN && selected != DType::E8Kv) {
            throw std::invalid_argument("Paged KV per-layer dtype is invalid");
        }
        return selected;
    };
    const auto layer_quant_group = [&](DType selected) {
        return selected == DType::I8
                   ? kKvInt8QuantGroup
                   : (selected == DType::BF16
                          ? 0
                          : (selected == DType::FP8_E4M3FN
                                 ? kKvFp8QuantGroup
                                 : (selected == DType::E8Kv ? kKvInt8QuantGroup
                                                            : kNvfp4KvQuantGroup)));
    };
    (void)quant_group;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const DType selected = layer_dtype(layer);
        if (selected != DType::BF16) {
            const std::int32_t group = layer_quant_group(selected);
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
    std::array<DType, 16> stored{};
    std::array<bool, 16> residual_flags{};
    std::array<std::uint32_t, 16> plane_base{};
    std::uint32_t plane_cursor = 0;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const DType selected = layer_dtype(layer);
        stored[layer]        = selected;
        plane_base[layer]    = plane_cursor;
        const std::int32_t group = layer_quant_group(selected);
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
            // FP8 per-group scales are FP16 in the production codecs; the
            // attention kernels require FP16 scale planes.
            geometry.planes.push_back({DType::FP16, head_dim / group, kv_heads, 256});
            geometry.planes.push_back({DType::FP16, head_dim / group, kv_heads, 256});
        } else {
            // NVFP4 tier: K keeps E2M1 packed codes with per-16 E4M3FN scales;
            // V stores ISO3 sign-magnitude nibbles in the same plane geometry
            // (semantic split via v_dtype, no extra payload). Enabled layers
            // add a second-stage residual plane set (K/V codes + scales).
            const bool residual = layer_has_residual(layer);
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
        plane_cursor += selected == DType::BF16 ? 2U : (layer_has_residual(layer) && selected == DType::NVFP4 ? 8U : 4U);
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
        .layer_plane_base = plane_base,
    };
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    const std::span<const DType> layer_dtypes(spec.layer_kv_dtypes.data(),
                                              spec.full_attention_layers);
    const std::span<const bool> layer_residual(spec.layer_residual.data(),
                                               spec.full_attention_layers);
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                layer_dtypes, layer_residual, spec.kv_table_rows,
                                spec.text_physical_page_groups);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                   {}, {}, spec.kv_table_rows, spec.mtp_physical_page_groups);
    }
    // Cold pool slots: one fixed-size slot per (page, kv_head, plane). The
    // INT8 tier packs requantized E2M1 nibbles into 9232 B raw slots; the
    // NVFP4 tier rANS-encodes the same geometry into slots with a 320 B
    // header, 32 x 256 B stream budget, and a 1024 B scale tail. One buffer
    // size serves both: 9536 B covers the rANS max; the raw I8 layout keeps
    // its offsets from slot start and simply leaves the tail unused.
    const std::int32_t slot_bytes = ops::kEntropyNvfp4SlotBytes;
    if (spec.max_cold_pages != 0) {
        const std::uint32_t cold_pages = spec.max_cold_pages;
        layout.text_kv.slot_bytes = slot_bytes;
        layout.text_kv.max_cold_pages  = spec.max_cold_pages;
        for (std::uint32_t layer = 0; layer < spec.full_attention_layers; ++layer) {
            layout.text_kv.cold_slots[layer] = builder.add_tensor(
                DType::U8, {slot_bytes, static_cast<std::uint32_t>(spec.kv_heads),
                             2, cold_pages},
                256, "cold slots L" + std::to_string(layer));
            layout.text_kv.cold_slot_valid[layer] = builder.add_tensor(
                DType::I32, {static_cast<std::uint32_t>(spec.kv_heads), 2, cold_pages}, 256,
                "cold slot valid L" + std::to_string(layer));
        }
    }
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pages_(backing, layout.pages), execution_tables_(backing, layout.execution_tables, pages_),
      layers_(layout.layers), max_context_(layout.max_context), kv_heads_(layout.kv_heads),
      head_dim_(layout.head_dim), dtype_(layout.dtype), quant_group_(layout.quant_group),
      slot_bytes_(layout.slot_bytes), max_cold_pages_(layout.max_cold_pages),
      layer_dtypes_(layout.layer_dtypes), layer_residual_(layout.layer_residual),
      layer_plane_base_(layout.layer_plane_base) {
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
                        layer_dtype == DType::NVFP4 || layer_dtype == DType::E8Kv;
    const std::size_t base   = layer_plane_base_.empty()
                                    ? static_cast<std::size_t>(layer) * (scaled ? 4ULL : 2ULL)
                                    : layer_plane_base_[layer];
    const bool residual = layer < layers_ && !layer_residual_.empty() &&
                          layer_residual_[layer] && layer_dtype == DType::NVFP4;
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
        .slot_bytes = slot_bytes_,
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .layer_index   = static_cast<std::int32_t>(layer),
        .dtype         = layer_dtypes_.empty() ? dtype_ : layer_dtypes_[layer],
        .quant_group   = layer_dtypes_.empty()
                             ? quant_group_
                             : (layer_dtypes_[layer] == DType::I8
                                    ? kKvInt8QuantGroup
                                    : (layer_dtypes_[layer] == DType::E8Kv
                                           ? kKvInt8QuantGroup
                                           : (layer_dtypes_[layer] == DType::FP8_E4M3FN
                                                  ? kKvFp8QuantGroup
                                                  : (layer_dtypes_[layer] == DType::NVFP4
                                                         ? kNvfp4KvQuantGroup
                                                         : 0)))),
        .v_dtype       = layer_dtypes_.empty()
                             ? dtype_
                             : (layer_dtypes_[layer] == DType::NVFP4 ? DType::ISO3
                                                                   : layer_dtypes_[layer]),
        .v_quant_group = layer_dtypes_.empty()
                             ? quant_group_
                             : (layer_dtypes_[layer] == DType::NVFP4 ? kNvfp4KvQuantGroup
                                                                   : 0),
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    // The scaled/stride decision follows the layer's resolved dtype so a
    // per-layer table (PR1) can mix quantized and BF16 layers in one pool.
    const DType layer_dtype =
        layer_dtypes_.empty() ? dtype_ : layer_dtypes_[layer];
    const bool scaled = layer_dtype == DType::I8 || layer_dtype == DType::FP8_E4M3FN ||
                        layer_dtype == DType::NVFP4 || layer_dtype == DType::E8Kv;
    const std::size_t base   = layer_plane_base_.empty()
                                    ? static_cast<std::size_t>(layer) * (scaled ? 4ULL : 2ULL)
                                    : layer_plane_base_[layer];
    const bool residual = layer < layers_ && !layer_residual_.empty() &&
                          layer_residual_[layer] && layer_dtype == DType::NVFP4;
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
        .slot_bytes = slot_bytes_,
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .layer_index   = static_cast<std::int32_t>(layer),
        .dtype         = layer_dtypes_.empty() ? dtype_ : layer_dtypes_[layer],
        .quant_group   = layer_dtypes_.empty()
                             ? quant_group_
                             : (layer_dtypes_[layer] == DType::I8
                                    ? kKvInt8QuantGroup
                                    : (layer_dtypes_[layer] == DType::E8Kv
                                           ? kKvInt8QuantGroup
                                           : (layer_dtypes_[layer] == DType::FP8_E4M3FN
                                                  ? kKvFp8QuantGroup
                                                  : (layer_dtypes_[layer] == DType::NVFP4
                                                         ? kNvfp4KvQuantGroup
                                                         : 0)))),
        .v_dtype       = layer_dtypes_.empty()
                             ? dtype_
                             : (layer_dtypes_[layer] == DType::NVFP4 ? DType::ISO3
                                                                   : layer_dtypes_[layer]),
        .v_quant_group = layer_dtypes_.empty()
                             ? quant_group_
                             : (layer_dtypes_[layer] == DType::NVFP4 ? kNvfp4KvQuantGroup
                                                                   : 0),
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
