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
    // SEPARATION: codec of the V plane on the NVFP4 tier. Iso4e keeps the engine
    // default (V stored as ISO4E sign-magnitude INT3); E2M1 is the ablation.
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
    // WHICH layers the per-layer spec actually WROTE (see
    // EngineOptions::kv_layer_storage_set), for layer_kv_dtypes above.
    // layer_kv_dtypes_set[L] true makes layer_kv_dtypes[L] authoritative even when
    // it is DType::BF16 -- that is the only way to ask for a REAL BF16 layer under
    // a quantized kv_dtype. All-false (the default) is the pre-mask rule: a BF16
    // entry inherits kv_dtype. Read by plan_cache() and by the three switch/gate
    // predicates that must resolve a layer the same way it does.
    // LAST MEMBER ON PURPOSE: this header is an export (embedders and the target
    // TUs compile against it), so a field appended at the end cannot move any
    // existing field's offset and an object built before this change still reads
    // everything it knows where it expects it.
    std::array<bool, 64> layer_kv_dtypes_set{};
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
    // the codec its resolved dtype feeds -- 9232 B for the int8 raw slot, 6688 B
    // for the nvfp4 rANS slot at the 2.60 bits/code ceiling (9536 B at the 4-bit
    // no-expansion bound) -- so a mixed stack is not charged the widest codec on
    // every layer. Indexed like layer_dtypes; all-zero when the pool is off
    // (decoder_state.cpp derives it from the attention geometry).
    std::array<std::int32_t, 64> layer_slot_bytes{};
    // Widest record in the pool (the codec default). Diagnostics and upper
    // bound only: every record access must use the per-layer stride above.
    std::int32_t slot_bytes = 0;
    std::uint32_t max_cold_pages = 0;
    // Resolved per-layer storage (one entry per full-attention layer).
    // THE FOUR TABLES BELOW ARE ALWAYS FULLY POPULATED, AND THEY ARE AUTHORITATIVE.
    // plan_cache() resolves the pool-wide dtype inheritance itself before storing
    // (kv_resolve_slot_dtype), writes every layer of every one of the four, and throws on
    // a table shorter than the layer count; the only producer of a PagedKVCacheLayout is
    // plan_cache(). There is therefore NO "table absent" state, and
    // `layer_dtypes_.empty()`, `layer_plane_base_.empty()`, `layer_residual_.empty()` and
    // `layer_sliding_windows_.empty()` are all CONSTANT FALSE (std::array<T,N>::empty()
    // returns false -- it is a size constant, not a query). A fallback guarded by one of
    // them is unreachable, and would silently IGNORE the per-layer resolution the layout
    // already performed. layer_view()/batch_layer_view() read these unconditionally for
    // that reason and pin `layer < layers_` (the one bound that is real) separately.
    std::array<DType, 64> layer_dtypes{};
    // Layers that keep an NVFP4 residual plane (base + 4) in the page geometry. This
    // table is 64 wide and every full-attention layer is written, so there is no
    // "table absent" state: a layer with no residual plane stores FALSE. (It used to say
    // "empty disables residual planes", which a std::array cannot express and which was
    // never how the pool worked.)
    std::array<bool, 64> layer_residual{};
    // Resolved per-layer SWA window in tokens (0 = full attention).
    std::array<std::uint32_t, 64> layer_sliding_windows{};
    // Plane offset of each layer in the page geometry (prefix sums over
    // per-layer plane counts; mixed BF16/quantized tables have unequal
    // strides).
    std::array<std::uint32_t, 64> layer_plane_base{};
    // L26 instrument: full-attention indices whose planes were DISCARDED by
    // NINFER_KV_DROP_LAYERS. Carried on the LAYOUT, not read from the
    // environment at each use, so the text cache can own the table while the MTP
    // cache -- planned with an empty spec -- keeps every one of its layers (D3).
    std::array<bool, 64> layer_dropped{};
    // Number of set entries in layer_dropped (carried so a reader does not have to
    // popcount the table back out).
    std::uint32_t layer_dropped_count = 0;
    // SEPARATION: V codec of the NVFP4 tier (see DecoderStateSpec::kv_v_codec).
    // Carried on the layout so PagedKVCache can publish it as v_dtype without
    // re-deriving it from a global option.
    KvVCodec kv_v_codec = KvVCodec::Iso3;
    // [F1255 kvaxisA] ⭐⭐ THE THIRD AXIS, AS THE LAYOUT STATES IT.
    // A layer that carries a SECOND plane set carries the pages of its narrow class there; the
    // first set is then sized for the REST of the pool's pages. `layer_narrow_base[L]` is the plane
    // index the narrow set starts at (0 = this layer has no narrow class, the pre-image shape) and
    // `layer_narrow_pages[L]` is how many pages it holds. The class of a given (block, layer) CELL
    // is carried by that layer's own block-table row, so the unit of the decision is the cell.
    // APPENDED, so no existing field's offset moves.
    std::array<std::uint32_t, 64> layer_narrow_base{};
    std::array<std::uint32_t, 64> layer_narrow_pages{};
    // Pool-wide resolved narrow capacity (0 = the knob is unset = the pre-image pool). One number
    // rather than a table because `plan_cache` resolves ONE knob for every layer it applies to.
    std::uint32_t narrow_pages_per_layer = 0;

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

    // [F1255 kvaxisA] THE THIRD AXIS, READABLE. The narrow plane set of one layer, and the pool-wide
    // resolved capacity. Both are the LAYOUT's own numbers; a reader must not re-derive them from
    // the knob, or the reading and the storage could drift.
    [[nodiscard]] std::uint32_t layer_narrow_base(std::uint32_t layer) const noexcept {
        return layer < layers_ ? layer_narrow_base_[layer] : 0U;
    }
    [[nodiscard]] std::uint32_t layer_narrow_pages(std::uint32_t layer) const noexcept {
        return layer < layers_ ? layer_narrow_pages_[layer] : 0U;
    }
    [[nodiscard]] std::uint32_t narrow_pages_per_layer() const noexcept {
        return narrow_pages_per_layer_;
    }
    [[nodiscard]] std::uint32_t resident_page_capacity() const noexcept {
        return pages_.capacity_pages();
    }

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

    // L26 instrument: TRUE when this cache holds NO planes for `layer` (its
    // storage was discarded). Per-INSTANCE state, so a cache planned with an
    // empty drop spec -- the MTP cache -- answers false for every layer, which
    // is what makes dropping text layer 0 legal again (D3).
    [[nodiscard]] bool layer_is_dropped(std::uint32_t layer) const noexcept {
        return layer < layers_ && layer < layer_dropped_.size() && layer_dropped_[layer];
    }
    [[nodiscard]] std::uint32_t dropped_layer_count() const noexcept { return dropped_layers_; }

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
    // Pool-wide defaults carried from the layout: the values plan_cache() RESOLVES FROM,
    // not an alternative source of truth. Since the sixteen dead
    // `layer_dtypes_.empty()`-style ternaries were removed from decoder_state.cpp there is
    // no reader left in this class, and a future reader that wants a layer's dtype must
    // use layer_dtypes_[layer], never this.
    DType dtype_               = DType::BF16;
    std::int32_t quant_group_  = 0;
    std::array<DType, 64> layer_dtypes_{};
    std::array<bool, 64> layer_residual_{};
    std::array<std::uint32_t, 64> layer_sliding_windows_{};
    std::array<std::uint32_t, 64> layer_plane_base_{};
    // [F1255 kvaxisA] the third axis, carried on the OBJECT as well as the layout.
    std::array<std::uint32_t, 64> layer_narrow_base_{};
    std::array<std::uint32_t, 64> layer_narrow_pages_{};
    std::uint32_t narrow_pages_per_layer_ = 0;
    // L26 instrument: PagedKVCacheLayout::layer_dropped, plus its popcount.
    std::array<bool, 64> layer_dropped_{};
    std::uint32_t dropped_layers_ = 0;
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
