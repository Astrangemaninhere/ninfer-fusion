#pragma once

#include "core/layout.h"
#include "core/paged_kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::targets::qwen3_6 {

inline constexpr std::int32_t kKvInt8QuantGroup = 64;
inline constexpr std::int32_t kKvFp8QuantGroup  = 256;
inline constexpr std::int32_t kNvfp4KvQuantGroup = 16;

struct DecoderStateSpec {
    std::uint32_t full_attention_layers     = 0;
    std::uint32_t mtp_layers                = 0;
    std::uint32_t capacity                  = 0;
    std::int32_t kv_heads                   = 0;
    std::int32_t attention_head_dim         = 0;
    DType kv_dtype                          = DType::BF16;
    std::int32_t kv_quant_group             = 0;
    // Per-layer storage table indexed by full-attention layer order.
    // BFloat16 entries inherit kv_dtype; empty (all-BF16) inherits wholesale.
    std::array<DType, 16> layer_kv_dtypes{};
    // NVFP4 layers that keep a residual plane (indexed like layer_kv_dtypes).
    std::array<bool, 16> layer_residual{};
    bool enable_mtp                         = false;
    std::int32_t kv_table_rows              = 1;
    std::uint32_t text_physical_page_groups = 0;
    std::uint32_t mtp_physical_page_groups  = 0;
    // Entropy-coded cold pool capacity in pages; 0 disables the pool.
    std::uint32_t max_cold_pages            = 0;
};

struct PagedKVCacheLayout {
    DeviceKVPagePoolLayout pages;
    KVExecutionTableLayout execution_tables;
    std::uint32_t layers      = 0;
    std::uint32_t max_context = 0;
    std::int32_t kv_heads     = 0;
    std::int32_t head_dim     = 0;
    DType dtype               = DType::BF16;
    std::int32_t quant_group  = 0;
    // Cold slots per layer: [slot_bytes, kv_heads, 2, max_cold_pages]
    // plus an I32 validity plane of [kv_heads, 2, max_cold_pages].
    std::array<TensorRegion, 16> cold_slots;
    std::array<TensorRegion, 16> cold_slot_valid;
    std::int32_t slot_bytes = 0;
    std::uint32_t max_cold_pages = 0;
    // Resolved per-layer storage (one entry per full-attention layer).
    std::array<DType, 16> layer_dtypes{};
    // Layers that keep an NVFP4 residual plane (base + 4) in the page
    // geometry; empty disables residual planes.
    std::array<bool, 16> layer_residual{};
    // Plane offset of each layer in the page geometry (prefix sums over
    // per-layer plane counts; mixed BF16/quantized tables have unequal
    // strides).
    std::array<std::uint32_t, 16> layer_plane_base{};

    [[nodiscard]] std::size_t payload_bytes() const noexcept { return pages.payload_bytes(); }
};

class PagedKVCache;

class PagedKVCacheView {
public:
    PagedKVCacheView() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return cache_ != nullptr; }

    [[nodiscard]] std::uint32_t max_context() const noexcept;

    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer) const;

private:
    friend class PagedKVCache;
    PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept;

    const PagedKVCache* cache_ = nullptr;
    Tensor block_table_;
};

class PagedKVCache {
public:
    PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout);

    PagedKVCache(const PagedKVCache&)            = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    PagedKVCache(PagedKVCache&&)                 = delete;
    PagedKVCache& operator=(PagedKVCache&&)      = delete;

        // Cold-slot pool: fixed raw slots per (layer, kv_head, plane).
    [[nodiscard]] std::int32_t cold_slot_bytes() const noexcept { return slot_bytes_; }
    [[nodiscard]] std::int32_t slot_bytes() const noexcept { return slot_bytes_; }
    [[nodiscard]] std::uint32_t max_cold_pages() const noexcept { return max_cold_pages_; }
    std::int32_t allocate_cold_slot() noexcept;
    void release_cold_slot(std::int32_t slot) noexcept;

[[nodiscard]] std::uint32_t max_context() const noexcept { return max_context_; }

    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }

    [[nodiscard]] DeviceKVPagePool& page_pool() noexcept { return pages_; }

    [[nodiscard]] const DeviceKVPagePool& page_pool() const noexcept { return pages_; }

    [[nodiscard]] KVExecutionTablePool& execution_tables() noexcept { return execution_tables_; }

    [[nodiscard]] const KVExecutionTablePool& execution_tables() const noexcept {
        return execution_tables_;
    }

    [[nodiscard]] PagedKVCacheView execution_view(const KVExecutionRowLease& row) const;

    [[nodiscard]] PagedKVBatchLayerView batch_layer_view(std::uint32_t layer) const;

    [[nodiscard]] Tensor cold_slot_valid(std::uint32_t layer) const noexcept {
        return layer < layers_ ? cold_slot_valid_[layer] : Tensor{};
    }

private:
    friend class PagedKVCacheView;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer, Tensor block_table) const;

    DeviceKVPagePool pages_;
    KVExecutionTablePool execution_tables_;
    std::uint32_t layers_      = 0;
    std::uint32_t max_context_ = 0;
    std::int32_t kv_heads_     = 0;

    std::int32_t head_dim_     = 0;
    DType dtype_               = DType::BF16;
    std::int32_t quant_group_  = 0;
    std::array<DType, 16> layer_dtypes_{};
    std::array<bool, 16> layer_residual_{};
    std::array<std::uint32_t, 16> layer_plane_base_{};
    std::array<Tensor, 16> cold_slots_;
    std::array<Tensor, 16> cold_slot_valid_;
    std::int32_t slot_bytes_ = 0;
    std::uint32_t max_cold_pages_ = 0;
    std::vector<std::uint8_t> cold_slot_used_;
};

struct DecoderStateLayout {
    PagedKVCacheLayout text_kv;
    std::optional<PagedKVCacheLayout> mtp_kv;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept;
};

[[nodiscard]] DecoderStateLayout plan_decoder_state(LayoutBuilder& builder,
                                                    const DecoderStateSpec& spec);

struct DecoderState {
    PagedKVCache text_kv;
    std::optional<PagedKVCache> mtp_kv;

    DecoderState(DeviceSpan backing, const DecoderStateLayout& layout);

    [[nodiscard]] PagedKVCache* mtp_cache() noexcept;
    [[nodiscard]] const PagedKVCache* mtp_cache() const noexcept;
};

} // namespace ninfer::targets::qwen3_6
