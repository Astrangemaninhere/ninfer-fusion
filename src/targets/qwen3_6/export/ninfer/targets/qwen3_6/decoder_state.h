#pragma once

#include "core/layout.h"
#include "core/paged_kv_cache.h"

#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>

namespace ninfer::targets::qwen3_6 {

inline constexpr std::int32_t kKvInt8QuantGroup = 64;
inline constexpr std::int32_t kKvFp8QuantGroup = 16;
inline constexpr std::int32_t kNvfp4KvQuantGroup = 16;
// Per-layer KV tables (PagedKVCacheLayout / PagedKVCache / the attention
// kernels' constant tables) are 64 slots wide; every supported model stays
// below it and the planner rejects anything larger instead of indexing past
// the arrays.
inline constexpr std::uint32_t kPagedKVCacheMaxLayers = 64;

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
    std::array<DType, 64> layer_kv_dtypes{};
    // NVFP4 layers that keep a residual plane (indexed like layer_kv_dtypes).
    std::array<bool, 64> layer_residual{};
    // SWA layers' attention window in tokens, indexed like layer_kv_dtypes.
    // 0 = no window (full attention) -- the value every consumer guards on.
    std::array<std::uint32_t, 64> layer_sliding_windows{};
    // SEPARATION: codec of the V plane on the NVFP4 tier. Iso3 keeps the engine
    // default (V stored as ISO3 sign-magnitude INT3); E2M1 is the ablation.
    // Both share one plane geometry, so this only changes what the producers
    // encode and the consumers decode.
    KvVCodec kv_v_codec                     = KvVCodec::Iso3;
    // SEPARATION: the two STATE-BASED component switches. Both are
    // committed to device state at the same single point as kv_v_codec
    // (plan_decoder_state). The upstream *_explicit resolution happens in
    // layouts_impl.h, so the spec carries the final decision, not the flag.
    bool kv_rotation_off                    = false;
    std::string kv_row_scale_spec;
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
    // Cold slots per layer: [record_stride, kv_heads, 2, max_cold_pages]
    // plus an I32 validity plane of [kv_heads, 2, max_cold_pages].
    // Per-layer cold slots; sized like kKvLayerStorageSlots (a family member may
    // exceed the 16 full-attention layers of the smallest variant).
    std::array<TensorRegion, 64> cold_slots{};
    std::array<TensorRegion, 64> cold_slot_valid{};
    // Cold-slot record stride PER LAYER, in bytes; one record is one
    // (page, kv_head, K|V) plane set. Each layer's record is exactly as wide as
    // the codec its resolved dtype feeds -- 9232 B for the int8 raw slot, 9536 B
    // for the nvfp4 rANS slot -- so a mixed stack is not charged the widest codec
    // on every layer. Indexed like layer_dtypes; all-zero when the pool is off
    // (decoder_state.cpp derives it from the attention geometry).
    std::array<std::int32_t, 64> layer_slot_bytes{};
    // Widest record in the pool (the codec default). Diagnostics and upper
    // bound only: every record access must use the per-layer stride above.
    std::int32_t slot_bytes = 0;
    std::uint32_t max_cold_pages = 0;
    // Resolved per-layer storage (one entry per full-attention layer).
    std::array<DType, 64> layer_dtypes{};
    // Layers that keep an NVFP4 residual plane (base + 4) in the page
    // geometry; empty disables residual planes.
    std::array<bool, 64> layer_residual{};
    // Resolved per-layer SWA window in tokens (0 = full attention).
    std::array<std::uint32_t, 64> layer_sliding_windows{};
    // Plane offset of each layer in the page geometry (prefix sums over
    // per-layer plane counts; mixed BF16/quantized tables have unequal
    // strides).
    std::array<std::uint32_t, 64> layer_plane_base{};
    // SEPARATION: V codec of the NVFP4 tier (see DecoderStateSpec::kv_v_codec).
    // Carried on the layout so PagedKVCache can publish it as v_dtype without
    // re-deriving it from a global option.
    KvVCodec kv_v_codec = KvVCodec::Iso3;

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

        // Cold-slot pool: fixed records per (layer, kv_head, plane). The pool
        // reserves one record per layer and sizes each layer's record by the
        // codec that layer's dtype feeds; slot_bytes()/cold_slot_bytes() report
        // the widest record (diagnostics only).
    [[nodiscard]] std::int32_t cold_slot_bytes() const noexcept { return slot_bytes_; }
    [[nodiscard]] std::int32_t slot_bytes() const noexcept { return slot_bytes_; }
    // Cold-slot record stride of one layer (0 when the cold pool is disabled).
    // This -- not slot_bytes() -- is the stride every cold record access uses.
    [[nodiscard]] std::int32_t layer_slot_bytes(std::uint32_t layer) const noexcept {
        return layer < layers_ && layer_slot_bytes_[layer] != 0 ? layer_slot_bytes_[layer]
                                                                : slot_bytes_;
    }
    [[nodiscard]] std::uint32_t max_cold_pages() const noexcept { return max_cold_pages_; }
    std::int32_t allocate_cold_slot() noexcept;
    void release_cold_slot(std::int32_t slot) noexcept;

    // Per-layer sliding windows (0 = full attention, i.e. the layer reads every
    // committed token), sized to the layer count. The Cold Host tier's read-free
    // predicate needs it: a page a full-attention layer may still read can never
    // leave the device, so the tier reports itself inert instead of reserving
    // host memory it cannot use.
    [[nodiscard]] std::span<const std::uint32_t> layer_sliding_windows() const noexcept {
        return std::span<const std::uint32_t>(layer_sliding_windows_.data(), layers_);
    }

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
    std::array<DType, 64> layer_dtypes_{};
    std::array<bool, 64> layer_residual_{};
    std::array<std::uint32_t, 64> layer_sliding_windows_{};
    std::array<std::uint32_t, 64> layer_plane_base_{};
    // SEPARATION: V codec of the NVFP4 tier (PagedKVCacheLayout::kv_v_codec).
    KvVCodec kv_v_codec_ = KvVCodec::Iso3;
    std::array<Tensor, 64> cold_slots_;
    std::array<Tensor, 64> cold_slot_valid_;
    std::int32_t slot_bytes_ = 0;
    std::array<std::int32_t, 64> layer_slot_bytes_{};
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
