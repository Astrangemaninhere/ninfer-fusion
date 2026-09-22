// ninfer::ops::detail - small-T GQA launch path.
// Cut B: the per-codec kernel instantiation cascades moved into
// gqa_attention_decode_{i8,nvfp4,fp8,iso3,bf16}.cu, so this TU no longer includes
// gqa_attention_decode_partial.cuh and carries only the runtime dtype chain below plus the
// split-reduction kernel. Relocated verbatim from gqa_attention_decode.cu.
#include "ops/launcher/gqa_attention.h"
#include "ops/launcher/gqa_attention_decode_split.h"
#include "ops/launcher/gqa_attention_decode_tiers.h"

#include "core/device.h" // CUDA_CHECK

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

// FIX-A item 3/4 instrument and contract switch. The canonical copy of the rationale lives in
// ops/launcher/gqa_attention_decode_partial.cuh; this TU does not include that header (it is
// the tier cascade the per-codec TUs include), so the two env readers are repeated here.
inline bool gqa_splitdbg_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("NINFER_SPLITDBG");
        return env != nullptr && *env != '\0' && !(env[0] == '0' && env[1] == '\0');
    }();
    return enabled;
}

// Same length gate as the canonical copy in gqa_attention_decode_partial.cuh.
inline bool gqa_splitdbg_allow() {
    if (!gqa_splitdbg_enabled()) { return false; }
    static const std::size_t cap = [] {
        const char* env = std::getenv("NINFER_SPLITDBG_MAX");
        if (env == nullptr || *env == '\0') { return std::size_t{512}; }
        const long value = std::strtol(env, nullptr, 10);
        return value <= 0 ? std::size_t{0} : static_cast<std::size_t>(value);
    }();
    static std::atomic<std::size_t> used{0};
    return used.fetch_add(1) < cap;
}

// FIX-A pin test. `logical_capacity` and `invocation.full_width` are the two fields of this
// launch that still move with the draft window k and that reach the partial kernel -- the
// split grid is pinned (split_units from the envelope's split_reference_keys) and the tile
// ladder keys on invocation.width alone (gqa_attention_decode_bf16.cu:26-59). The k<=7
// identity shows the two fields CAN differ while the result does not move, so pinning them is
// what turns 'a difference' into 'the carrier'. Two independent flags, so they bisect:
//   NINFER_SPLIT_PARITY=1         logical_capacity := round_base + column_begin + width
//                                 (the window the batch-1 decode of this launch's LAST column
//                                 would carry, i.e. the per-column window FIX-A item 2 asks for)
//   NINFER_SPLIT_PARITY_WIDTH=1   full_width := width (the launch is its own round)
// Both are off by default: with them off the launch is byte-identical to the pre-fix tree.
inline bool gqa_split_parity() {
    static const bool on = [] {
        const char* env = std::getenv("NINFER_SPLIT_PARITY");
        return env != nullptr && *env != '\0' && !(env[0] == '0' && env[1] == '\0');
    }();
    return on;
}

inline bool gqa_split_parity_width() {
    static const bool on = [] {
        const char* env = std::getenv("NINFER_SPLIT_PARITY_WIDTH");
        return env != nullptr && *env != '\0' && !(env[0] == '0' && env[1] == '\0');
    }();
    return on;
}

inline bool gqa_verify_exact() {
    // UNSET IS LEGACY, and it has to be. The device-side publisher of this same switch
    // reads the VALUE and not its absence (src/ops/kernel/gqa_attention_decode.cuh:69-70
    // tests `env[0] == '1' && env[1] == '\0'`, so unset means 0 there), and an ordinary
    // --spec none decode legitimately carries no pin at all. A host half that defaulted to
    // strict therefore refused EVERY arm -- including --spec none -- with rc=1 while the
    // device half reported the legacy arm. Unset and 0 are the legacy behaviour; the
    // strict arm is OPT-IN via NINFER_VERIFY_EXACT=1, where it acts as a guard that makes
    // an unpinned launch loud instead of silently partitioned from the live window.
    static const bool strict = [] {
        const char* env = std::getenv("NINFER_VERIFY_EXACT");
        return env != nullptr && env[0] == '1' && env[1] == '\0';
    }();
    return strict;
}

PagedKVBatchLayerView single_row_batch_view(const PagedKVLayerView& cache) {
    return {
        .k_pages       = cache.k_pages,
        .v_pages       = cache.v_pages,
        .k_scale_pages = cache.k_scale_pages,
        .v_scale_pages = cache.v_scale_pages,
        .k_residual_pages = cache.k_residual_pages,
        .k_residual_scale_pages = cache.k_residual_scale_pages,
        .v_residual_pages = cache.v_residual_pages,
        .v_residual_scale_pages = cache.v_residual_scale_pages,
        .block_tables  = cache.block_table.view({cache.block_table.ne[0], 1}),
        .cold_slots    = cache.cold_slots,
        .cold_slot_valid = cache.cold_slot_valid,
        .slot_bytes    = cache.slot_bytes,
        .head_dim      = cache.head_dim,
        .num_kv_heads  = cache.num_kv_heads,
        .layer_index   = cache.layer_index,
        .dtype         = cache.dtype,
        .quant_group   = cache.quant_group,
        .v_dtype       = cache.v_dtype,
        .v_quant_group = cache.v_quant_group,
        .sliding_window_tokens = cache.sliding_window_tokens,
    };
}

} // namespace

template <typename Geometry, typename CacheInput>
void gqa_attention_small_t_launch_for(const Tensor& q, CacheInput input, const Tensor& pos,
                                      float scale, PagedKVBatchLayerView cache,
                                      const GqaSmallTInvocation& invocation,
                                      GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                      Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                      cudaStream_t stream) {
    const auto logical_capacity = static_cast<std::int32_t>(envelope.max_visible_keys);
    // FIX-A pin test (NINFER_SPLIT_PARITY): the window the batch-1 decode of THIS launch's last
    // column would carry. The round's column 0 sits (full_width - 1) below
    // envelope.max_visible_keys - 1, so the last column of a launch that starts at
    // column_begin and covers `width` columns sits at
    //     round_base + column_begin + width - 1,   round_base = max_visible_keys - full_width + 1
    // and its own visible-key count is one more than that. For a batch-1 decode
    // (full_width == width, column_begin == 0) this is max_visible_keys itself, so the pin is a
    // no-op there and the decode arm cannot move.
    const std::int32_t parity_capacity =
        static_cast<std::int32_t>(envelope.max_visible_keys) - invocation.full_width + 1 +
        invocation.column_begin + invocation.width;
    const std::int32_t kernel_capacity =
        gqa_split_parity() && parity_capacity > 0 ? parity_capacity : logical_capacity;
    // Split reference: a constant of the produced execution graph (the sequence key
    // capacity). Batch-1 decode and the MTP/DFlash verify of the same request carry the
    // same value, so a verify column reduces its keys in the same order as the decode
    // row it replaces. It also replaces the live window in the I8 tile schedule, which
    // is the second way a launch width used to reach the exact result.
    const auto split_reference       = gqa_small_t_split_reference(envelope);
    // FIX-A item 4 - NINFER_VERIFY_EXACT=1 (OPT-IN, unset and 0 are the legacy arm) turns
    // on the strict contract. A split-KV small-T launch whose envelope does not PIN
    // split_reference_keys (== 0) derives its split grid from the LIVE window instead, i.e.
    // from how far the sequence has advanced and how wide the draft window is -- exactly
    // what include/ninfer/ops/gqa_attention.h:18-25 forbids for a row that must reduce like
    // the batch-1 decode of itself. Under the strict arm that launch is refused BY NAME
    // instead of being silently partitioned differently from that decode; the arm is a
    // GUARD that makes the defect loud, not the fix itself. The fix is the pin at the
    // envelope's source (see the pinned sites in program_impl.h / layouts_impl.h).
    const bool verify_exact = gqa_verify_exact();
    if (verify_exact && envelope.split_reference_keys == 0) {
        // SCOPE. The guard is for the class the pin exists for: a MULTI-COLUMN verify round,
        // i.e. a chunked launch (full_width > width) whose envelope is the round style
        // (min_visible_keys == 1). It must NOT fire on the other two unpinned classes the tree
        // still has, because neither is a discriminator: the prefill chunk envelope
        // (`{visible, visible}`, min == max, full_width == width) is present and identical in
        // EVERY arm, including --spec none, and the MTP draft bridge/AR envelopes are the
        // per-round draft attention. A check that refuses every arm makes the engine
        // un-runnable in strict mode without telling us anything, so those classes get a
        // capped loud warning instead of a throw, and only the verify class is refused.
        const bool round_verify_class =
            envelope.min_visible_keys == 1 && invocation.full_width > invocation.width;
        if (round_verify_class) {
            throw std::invalid_argument(
                "gqa_attention small-T split-KV verify launch: the execution envelope does "
                "not pin split_reference_keys, so this verify round cannot reduce its keys "
                "like the batch-1 decode of the same row; unset NINFER_VERIFY_EXACT (or set "
                "it to 0) to run the legacy window-driven partition");
        }
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 8) {
            std::fprintf(stderr,
                         "[verify-exact] WARNING: unpinned split-KV small-T launch "
                         "(route=%s width=%d full_width=%d min_visible=%u max_visible=%u): "
                         "its split grid follows the live window; this class is not the "
                         "multi-column verify round the pin is for, so it is reported and "
                         "not refused\n",
                         (invocation.column_begin != 0 || invocation.width < invocation.full_width)
                             ? "chunked_small_t"
                             : "small_t",
                         invocation.width, invocation.full_width, envelope.min_visible_keys,
                         envelope.max_visible_keys);
        }
    }
    // FIX-A item 2: the tile schedule's window input. Under strict it is the PINNED
    // reference, a constant of the produced graph; the legacy arm reproduces the old
    // behaviour and hands the tiers envelope.max_visible_keys, which for an MTP verify is
    // frontier + k + 1 and therefore moves with the draft window k.
    // In this arm every envelope of the split-KV small-T family carries the pin, so this
    // expression is byte-identical to the pre-fix tree: the schedule window is the pin. The
    // formula is written out because the fallback is the defect: an unpinned envelope has no
    // width-independent window to offer, and strict mode refuses that launch above instead of
    // silently taking the live one.
    const auto implementation_window =
        envelope.split_reference_keys != 0
            ? split_reference
            : static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto split_units           = gqa_small_t_split_units<Geometry>(split_reference);
    const auto splits =
        gqa_small_t_launch_capacity<Geometry>(envelope, invocation.width, cache.dtype);
    if (gqa_splitdbg_allow()) {
        std::fprintf(stderr,
                     "[splitdbg] launch route=%s dtype=%d width=%d full_width=%d "
                     "column_begin=%d batch=%d min_visible=%u max_visible=%u pin=%u "
                     "split_reference=%d schedule_window=%d splits=%d split_units=%d "
                     "logical_capacity=%d kernel_capacity=%d parity=%d parity_width=%d "
                     "verify_exact=%d q_heads=%d kv_heads=%d\n",
                     (invocation.column_begin != 0 || invocation.width < invocation.full_width)
                         ? "chunked_small_t"
                         : "small_t",
                     static_cast<int>(cache.dtype), invocation.width, invocation.full_width,
                     invocation.column_begin, invocation.batch_size, envelope.min_visible_keys,
                     envelope.max_visible_keys, envelope.split_reference_keys,
                     split_reference, implementation_window, splits, split_units,
                     logical_capacity, kernel_capacity, gqa_split_parity() ? 1 : 0,
                     gqa_split_parity_width() ? 1 : 0, verify_exact ? 1 : 0, Geometry::QHeads,
                     Geometry::KVHeads);
    }

    // Cut B: one runtime dtype chain instead of the pre-split
    // NINFER_GQA_SMALL_T_DISPATCH expansion. The per-codec kernel instantiations moved into
    // gqa_attention_decode_{i8,nvfp4,fp8,iso3,bf16}.cu and the E8 tier was already in its own
    // TU. The dtype set, the arguments passed, and the width ladder each tier then applies are
    // the ones this TU's macro used -- only the instantiation site moved. DType::NVFP4 is no
    // longer split on v_dtype here: the nvfp4 tier re-keys that to its Iso3V kernel variant,
    // which is the same accept set with one dispatch site fewer.
    if (cache.dtype == DType::I8) {
        gqa_attention_decode_i8_launch(q, input, pos, scale, cache, invocation, kernel_capacity,
                                       implementation_window, splits, partial_acc, partial_m,
                                       partial_l, stream);
    } else if (cache.dtype == DType::E8Kv) {
        gqa_attention_decode_e8_launch(q, input, pos, scale, cache, invocation, kernel_capacity,
                                       implementation_window, splits, partial_acc, partial_m,
                                       partial_l, stream);
    } else if (cache.dtype == DType::NVFP4) {
        gqa_attention_decode_nvfp4_launch(q, input, pos, scale, cache, invocation,
                                          kernel_capacity, implementation_window, splits,
                                          partial_acc, partial_m, partial_l, stream);
    } else if (cache.dtype == DType::FP8_E4M3FN) {
        gqa_attention_decode_fp8_launch(q, input, pos, scale, cache, invocation, kernel_capacity,
                                        implementation_window, splits, partial_acc, partial_m,
                                        partial_l, stream);
    } else if (cache.dtype == DType::ISO3) {
        gqa_attention_decode_iso3_launch(q, input, pos, scale, cache, invocation,
                                         kernel_capacity, implementation_window, splits,
                                         partial_acc, partial_m, partial_l, stream);
    } else {
        gqa_attention_decode_bf16_launch(q, input, pos, scale, cache, invocation,
                                         kernel_capacity, implementation_window, splits,
                                         partial_acc, partial_m, partial_l, stream);
    }

    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = 64;
    const dim3 reduce_grid(Geometry::QHeads, div_up(kGqaHeadDim, kDChunk),
                           invocation.width * invocation.batch_size);
    const auto launch_reduce = [&]<bool Int8, bool MultiBatch, bool Masked, bool Offset>() {
        gqa_attention_small_t_reduce_output_kernel<Geometry, kDChunk, Int8, MultiBatch, Masked,
                                                   Offset>
            <<<reduce_grid, kReduceBlock, 0, stream>>>(
                static_cast<const float*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(pos.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.width, invocation.full_width, invocation.column_begin,
                invocation.batch_size, splits, split_units,
                static_cast<std::int32_t>(cache.sliding_window_tokens),
                static_cast<__nv_bfloat16*>(out.data));
    };
    const bool masked         = invocation.valid_columns != nullptr;
    const auto launch_profile = [&]<bool Int8, bool MultiBatch, bool Masked>() {
        if (invocation.column_begin == 0) {
            launch_reduce.template operator()<Int8, MultiBatch, Masked, false>();
        } else {
            launch_reduce.template operator()<Int8, MultiBatch, Masked, true>();
        }
    };
    const auto launch_for_dtype = [&]<bool Int8>() {
        if (invocation.batch_size == 1) {
            if (masked) {
                launch_profile.template operator()<Int8, false, true>();
            } else {
                launch_profile.template operator()<Int8, false, false>();
            }
        } else if (masked) {
            launch_profile.template operator()<Int8, true, true>();
        } else {
            launch_profile.template operator()<Int8, true, false>();
        }
    };
    // FP8_E4M3FN and ISO3 are quantized but deliberately use the BF16
    // (Int8=false) reducer path: gqa_small_t_split_count falls through to the
    // generic BF16 policy for both dtypes, and their partial kernels compute
    // active splits with gqa_small_t_active_splits<Geometry,false>. The
    // Int8=true path would apply the I8 token-5/6 active-split specializations
    // and disagree with the launch.
    if (cache.dtype == DType::I8 || cache.dtype == DType::NVFP4) {
        launch_for_dtype.template operator()<true>();
    } else {
        launch_for_dtype.template operator()<false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& pos, const Tensor& valid_columns,
                                  const Tensor& column_masks, const Tensor& table_rows,
                                  float scale, PagedKVBatchLayerView cache,
                                  GqaExecutionEnvelope envelope, std::int32_t column_begin,
                                  std::int32_t width, Tensor& partial_acc, Tensor& partial_m,
                                  Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    const GqaAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                               static_cast<const __nv_bfloat16*>(v.data)};
    // FIX-A pin test (NINFER_SPLIT_PARITY_WIDTH): full_width is the round's column count
    // (k + 1 on an MTP verify, 1 on a batch-1 decode). Under the pin a launch is told it is
    // its own round, which is the decode-equivalent value; for this configuration the field
    // reaches the partial kernel only through the MultiBatch column offset and the per-column
    // ancestor-mask index, both inert at batch == 1 with column_masks == nullptr, so whether
    // the ids move under the pin is a direct measurement of whether it is inert.
    const std::int32_t full_width = gqa_split_parity_width() ? width : q.ne[2];
    const GqaSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .column_masks  = column_masks.data == nullptr ? nullptr : &column_masks,
        .table_rows    = &table_rows,
        .full_width    = full_width,
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
    };
    if (q.ne[1] == Gqa27Geometry::QHeads && q.ne[0] == Gqa27Geometry::HeadDim) {
        gqa_attention_small_t_launch_for<Gqa27Geometry>(q, input, pos, scale, cache, invocation,
                                                        envelope, partial_acc, partial_m, partial_l,
                                                        out, stream);
        return;
    }
    if (q.ne[1] == GqaMuseGeometry::QHeads && q.ne[0] == GqaMuseGeometry::HeadDim) {
        gqa_attention_small_t_launch_for<GqaMuseGeometry>(q, input, pos, scale, cache, invocation,
                                                          envelope, partial_acc, partial_m,
                                                          partial_l, out, stream);
        return;
    }
    if (q.ne[1] == Gqa35Geometry::QHeads && q.ne[0] == Gqa35Geometry::HeadDim) {
        gqa_attention_small_t_launch_for<Gqa35Geometry>(q, input, pos, scale, cache, invocation,
                                                        envelope, partial_acc, partial_m,
                                                        partial_l, out, stream);
        return;
    }
    throw std::invalid_argument(
        "gqa_attention_small_t_launch: unsupported query-head geometry (" +
        std::to_string(q.ne[1]) + " q-heads); registered: " +
        std::to_string(Gqa27Geometry::QHeads) + "/" +
        std::to_string(GqaMuseGeometry::QHeads) + "/" +
        std::to_string(Gqa35Geometry::QHeads));
}

void gqa_attention_cached_small_t_launch(const Tensor& q, const Tensor& pos, float scale,
                                         const PagedKVLayerView& cache,
                                         GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                         Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                         cudaStream_t stream) {
    const GqaCachedInput input{};
    const GqaSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size    = 1,
    };
    const PagedKVBatchLayerView batch_cache = single_row_batch_view(cache);
    if (q.ne[1] == Gqa27Geometry::QHeads && q.ne[0] == Gqa27Geometry::HeadDim) {
        gqa_attention_small_t_launch_for<Gqa27Geometry>(q, input, pos, scale, batch_cache,
                                                        invocation, envelope, partial_acc,
                                                        partial_m, partial_l, out, stream);
        return;
    }
    if (q.ne[1] == GqaMuseGeometry::QHeads && q.ne[0] == GqaMuseGeometry::HeadDim) {
        gqa_attention_small_t_launch_for<GqaMuseGeometry>(q, input, pos, scale, batch_cache,
                                                          invocation, envelope, partial_acc,
                                                          partial_m, partial_l, out, stream);
        return;
    }
    gqa_attention_small_t_launch_for<Gqa35Geometry>(q, input, pos, scale, batch_cache, invocation,
                                                    envelope, partial_acc, partial_m, partial_l,
                                                    out, stream);
}

} // namespace ninfer::ops::detail
