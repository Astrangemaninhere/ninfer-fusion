#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "core/dtype.h"
#include "core/nvtx.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/dspark_markov_argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/swa.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

void require_dflash_state(const PrefillContext& state) {
    if (state.dflash == nullptr || !state.execution.model.dflash.has_value()) {
        throw std::logic_error("DFlash schedule requires DFlash weights and state");
    }
}

DFlashPersistentState& dflash_state(PrefillContext& state) {
    require_dflash_state(state);
    return *state.dflash;
}

DFlashPersistentState& dflash_state(DFlashBatchContext& state) { return state.dflash; }

DFlashPersistentState& dflash_state(DFlashAppendContext& state) { return state.dflash; }

template <class V>
DFlashFeatureSink prefill_feature_sink_impl(PrefillContext& state,
                                            DFlashFeatureSink::PrefillConsumer consume_prefill) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        require_dflash_state(state);
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .features        = &dflash_state(state).prefill_features,
            .positions       = &dflash_state(state).prefill_positions,
            .layers          = std::span<const int>(Config::target_feature_layers),
            .consume_prefill = std::move(consume_prefill),
        };
    }
}

template <class V>
DFlashFeatureSink batch_feature_sink_impl(DFlashBatchContext& state, const Tensor& lanes,
                                          const Tensor& valid_columns, std::int32_t width,
                                          std::int32_t batch_size) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .batch_features      = &dflash_state(state).pending_features,
            .batch_lanes         = &lanes,
            .batch_valid_columns = &valid_columns,
            .batch_width         = width,
            .batch_size          = batch_size,
            .layers              = std::span<const int>(Config::target_feature_layers),
        };
    }
}

template <class V, class Context>
void append_context_impl(Context& state, const Tensor& features, const Tensor& positions,
                         const Tensor& commit_counts, const Tensor& lanes, const Tensor& table_rows,
                         ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash context append is unavailable for this target");
    } else {
        using Config               = typename V::DFlashConfig;
        const std::int32_t width   = features.ne[1];
        const std::int32_t batch   = features.ne[2];
        const std::int32_t columns = width * batch;
        nvtx::ScopedRange append_range(nvtx::Name::DFlashContextAppend, nvtx::Category::DFlash,
                                       static_cast<std::uint64_t>(columns));
        if (width <= 0 || batch <= 0 || features.dtype != DType::BF16 ||
            features.ne[0] != Config::feature_rows || features.ne[3] != 1 ||
            positions.dtype != DType::I32 || positions.ne[0] != width || positions.ne[1] != batch ||
            commit_counts.dtype != DType::I32 || commit_counts.ne[0] != batch ||
            lanes.dtype != DType::I32 || lanes.ne[0] != batch || table_rows.dtype != DType::I32 ||
            table_rows.ne[0] != batch) {
            throw std::invalid_argument("DFlash context append inputs are invalid");
        }
        const bool replace_local_window =
            !Config::full_only && batch == 1 && width > Config::local_capacity;
        if (replace_local_window && (envelope.min_count != static_cast<std::uint32_t>(width) ||
                                     envelope.max_count != static_cast<std::uint32_t>(width))) {
            throw std::invalid_argument(
                "DFlash oversized local append requires an exact full-prefix commit");
        }
        const int local_offset = replace_local_window ? width - Config::local_capacity : 0;
        const int local_width  = replace_local_window ? Config::local_capacity : width;
        const ops::KVCacheAppendPrefixExecutionEnvelope local_envelope{
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.min_count,
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.max_count,
        };
        Tensor local_counts = commit_counts;
        if (replace_local_window) {
            if (!state.execution.io.dflash_prefill) {
                throw std::logic_error("DFlash prefill count storage is unavailable");
            }
            local_counts = state.execution.io.dflash_prefill->produced_count;
            ops::set_i32_scalar(local_counts, Config::local_capacity,
                                state.execution.device.stream);
        }

        const auto context_roots =
            workspace_recipe::dflash_context<Config>(state.execution.work, columns);
        Tensor projected = context_roots.projected;
        ops::linear(features.view({Config::feature_rows, columns}),
                    state.execution.model.dflash->feature_projection, projected,
                    state.execution.device.stream);
        Tensor context = context_roots.normalized;
        ops::rmsnorm(projected, state.execution.model.dflash->context_norm, Config::rms_epsilon,
                     false, context, state.execution.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            auto layer_scope = state.execution.work.scope();
            const auto& weight =
                state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));
            const bool local_layer = !Config::full_only && layer < Config::local_layers;
            const int full_index   = Config::full_only ? layer : layer - Config::local_layers;
            const int layer_width   = local_layer ? local_width : width;
            const int layer_columns = layer_width * batch;
            Tensor layer_context    = local_layer && replace_local_window
                                          ? context.slice(1, local_offset, local_width)
                                          : context;
            Tensor layer_positions  = local_layer && replace_local_window
                                          ? positions.slice(0, local_offset, local_width)
                                          : positions;
            auto layer_roots =
                workspace_recipe::dflash_context_layer<Config>(state.execution.work, layer_columns);
            Tensor key_raw =
                layer_roots.key_raw.view({Config::head_dim, Config::kv_heads, layer_columns});
            Tensor value =
                layer_roots.value.view({Config::head_dim, Config::kv_heads, layer_columns});
            Tensor key_flat   = key_raw.view({Config::kv_size, layer_columns});
            Tensor value_flat = value.view({Config::kv_size, layer_columns});
            if constexpr (Config::bf16_weights) {
                ops::linear(layer_context, weight.context_key, key_flat,
                            state.execution.device.stream);
                ops::linear(layer_context, weight.context_value, value_flat,
                            state.execution.device.stream);
            } else {
                ops::linear_pair(layer_context, weight.context_key, weight.context_value, key_flat,
                                 value_flat, state.execution.device.stream);
            }
            Tensor key = layer_roots.key.view({Config::head_dim, Config::kv_heads, layer_columns});
            ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                         state.execution.device.stream);
            ops::rope(layer_positions.view({layer_columns}), Config::head_dim, Config::rope_theta,
                      key, state.execution.device.stream);
            Tensor key_batch = key.view({Config::head_dim, Config::kv_heads, layer_width, batch});
            Tensor value_batch =
                value.view({Config::head_dim, Config::kv_heads, layer_width, batch});
            Tensor position_batch = layer_positions.view({layer_width, batch});
            if (local_layer) {
                ops::kv_cache_append_prefix(
                    key_batch, value_batch, position_batch, local_counts, lanes, local_envelope,
                    dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                    Config::local_window, state.execution.device.stream);
            } else {
                ops::kv_cache_append_prefix(
                    key_batch, value_batch, position_batch, commit_counts, table_rows, envelope,
                    dflash_state(state).full_batch_layer(static_cast<std::uint32_t>(full_index)),
                    state.execution.device.stream);
            }
        }
    }
}

template <class V>
void propose_batch_impl(DFlashBatchContext& state, qwen3_6::DFlashDecodeState& frame,
                        std::int32_t batch_size, std::uint32_t k, DFlashEnvelopes envelopes) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash proposal is unavailable for this target");
    } else {
        using Config               = typename V::DFlashConfig;
        const std::int32_t width   = static_cast<std::int32_t>(k) + 1;
        const std::int32_t columns = width * batch_size;
        nvtx::ScopedRange proposal_range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                                         static_cast<std::uint64_t>(columns));
        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor valid_columns      = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor lanes              = frame.active_lanes.slice(0, 0, batch_size);
        Tensor full_rows          = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor ids                = frame.proposal_ids.slice(1, 0, batch_size);
        Tensor positions          = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor drafts             = frame.draft_tokens.slice(1, 0, batch_size);

        // DSpark mirrors HF spec_generate: the draft block has exactly `k`
        // noise rows (anchor + k-1 masks) and predicts k rows, while the
        // target verify consumes width = k+1 (including the bonus token).
        // The legacy W8 DFlash path keeps its own width semantics.
        state.execution.work.reset();
        Tensor attention_valid = valid_columns;
        if constexpr (Config::bf16_weights) {
            attention_valid = state.execution.work.alloc(DType::I32, {batch_size});
            // DSpark keeps exactly k live draft columns, but this table is also the
            // target verify's cache/rope position table, and that contract is
            // target_valid_columns = extent + 1 = width live columns. Build the table
            // at width so its last column sits at frontier + k instead of repeating
            // the position of column k-1 (which also aliases its KV cache slot).
            ops::set_i32_scalar(attention_valid, width, state.execution.device.stream);
        }

        ops::prepare_masked_block(anchors, frontiers, attention_valid, Config::mask_token, ids,
                                  positions, state.execution.device.stream);
        // The table stays at `width` (= k + 1) live columns. prepare_masked_block only
        // reads it -- it ramps positions to frontier + k -- and never writes it, so
        // lowering it to k hides column k from its own KV slot while column k is exactly
        // the last hidden the proposal consumes (source_column_offset = 1). The sibling
        // DFlash2 implementation keeps width for the same reason (dflash2_impl.h:190).
        Tensor residual = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
        ops::embedding(ids.view({columns}), state.execution.model.token_embedding, residual,
                       state.execution.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash,
                                          static_cast<std::uint64_t>(layer));
            const auto& weight =
                state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));
            {
                nvtx::ScopedRange attention_range(nvtx::Name::DFlashAttention,
                                                  nvtx::Category::Attention,
                                                  static_cast<std::uint64_t>(layer));
                auto attention_scope = state.execution.work.scope();
                auto roots =
                    workspace_recipe::dflash_attention<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.input_norm, Config::rms_epsilon, false, roots.hidden,
                             state.execution.device.stream);
                Tensor query_raw =
                    roots.query_raw.view({Config::head_dim, Config::query_heads, columns});
                Tensor key_raw = roots.key_raw.view({Config::head_dim, Config::kv_heads, columns});
                Tensor value   = roots.value.view({Config::head_dim, Config::kv_heads, columns});
                Tensor query_flat = query_raw.view({Config::query_size, columns});
                Tensor key_flat   = key_raw.view({Config::kv_size, columns});
                Tensor value_flat = value.view({Config::kv_size, columns});
                if constexpr (Config::bf16_weights) {
                    // The fused [query_size + 2*kv_size, K] object is stored as
                    // contiguous Q/K/V row groups. Linear writes each token's
                    // N outputs contiguously, so a fused output buffer would
                    // interleave Q, K, and V per token; split the weight rows
                    // and run three projections into the per-token buffers.
                    const Weight qkv_weight   = weight.query_key_value;
                    const std::size_t row_len = static_cast<std::size_t>(qkv_weight.k) *
                                                dtype_size(DType::BF16);
                    Weight query_weight = qkv_weight;
                    query_weight.n      = Config::query_size;
                    Weight key_weight   = qkv_weight;
                    key_weight.n        = Config::kv_size;
                    key_weight.qdata    = static_cast<const std::byte*>(qkv_weight.qdata) +
                                       static_cast<std::size_t>(Config::query_size) * row_len;
                    Weight value_weight = qkv_weight;
                    value_weight.n      = Config::kv_size;
                    value_weight.qdata =
                        static_cast<const std::byte*>(qkv_weight.qdata) +
                        static_cast<std::size_t>(Config::query_size + Config::kv_size) * row_len;
                    ops::linear(roots.hidden, query_weight, query_flat,
                                state.execution.device.stream);
                    ops::linear(roots.hidden, key_weight, key_flat,
                                state.execution.device.stream);
                    ops::linear(roots.hidden, value_weight, value_flat,
                                state.execution.device.stream);
                } else {
                    ops::attn_input_proj(roots.hidden, weight.query_key_value, query_flat,
                                         key_flat, value_flat, state.execution.device.stream);
                }
                Tensor query = roots.query.view({Config::head_dim, Config::query_heads, columns});
                Tensor key   = roots.key.view({Config::head_dim, Config::kv_heads, columns});
                ops::rmsnorm(query_raw, weight.query_norm, Config::rms_epsilon, false, query,
                             state.execution.device.stream);
                ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                             state.execution.device.stream);
                ops::rope(positions.view({columns}), Config::head_dim, Config::rope_theta, query,
                          key, state.execution.device.stream);
                Tensor query_batch =
                    query.view({Config::head_dim, Config::query_heads, width, batch_size});
                Tensor key_batch =
                    key.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor value_batch =
                    value.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor attention_batch = roots.attention.view(
                    {Config::head_dim, Config::query_heads, width, batch_size});
                const bool local_layer = !Config::full_only && layer < Config::local_layers;
                const int full_index   = Config::full_only ? layer : layer - Config::local_layers;
                if (local_layer) {
                    ops::swa(query_batch, key_batch, value_batch, positions, attention_valid,
                             lanes, Config::attention_scale,
                             dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                             envelopes.local, Config::local_window, state.execution.work,
                             attention_batch, state.execution.device.stream);
                } else {
                    if constexpr (Config::query_heads == 32 && Config::kv_heads == 8) {
                        ops::bidirectional_gqa_attention(
                            query_batch, key_batch, value_batch, frontiers, attention_valid,
                            full_rows, Config::attention_scale,
                            dflash_state(state).full_batch_layer(
                                static_cast<std::uint32_t>(full_index)),
                            envelopes.full, state.execution.work, attention_batch,
                            state.execution.device.stream);
                    } else if constexpr (Config::query_heads == 40 && Config::kv_heads == 8) {
                        // DSpark GQA group is 5 (head h -> KV head h/5), while the
                        // 32/8 bidirectional op uses group 4. Two 32-head calls
                        // cover the 40 real heads: call A places real heads
                        // 5g..5g+3 in padded slots 4g..4g+3 (both map to KV head
                        // g), and call B places real head 5g+4 in padded slot 4g.
                        // Unused padded slots are never copied back, so their
                        // stale query rows do not affect the result.
                        const auto full_context = dflash_state(state).full_batch_layer(
                            static_cast<std::uint32_t>(full_index));
                        Tensor padded_query =
                            roots.padded_query.view({Config::head_dim, 32, width, batch_size});
                        Tensor padded_attention =
                            roots.padded_attention.view({Config::head_dim, 32, width, batch_size});
                        const std::size_t elem_bytes = dtype_size(DType::BF16);
                        const std::size_t head_bytes =
                            static_cast<std::size_t>(Config::head_dim) * elem_bytes;
                        const std::size_t token_rows =
                            static_cast<std::size_t>(width) * static_cast<std::size_t>(batch_size);
                        const auto copy_heads = [&](const Tensor& dst, std::int32_t dst_head,
                                                    std::int32_t count, const Tensor& src,
                                                    std::int32_t src_head) {
                            const std::size_t dst_pitch =
                                static_cast<std::size_t>(Config::head_dim) * dst.ne[1] * elem_bytes;
                            const std::size_t src_pitch =
                                static_cast<std::size_t>(Config::head_dim) * src.ne[1] * elem_bytes;
                            void* dst_ptr = static_cast<std::byte*>(dst.data) +
                                            static_cast<std::size_t>(dst_head) * head_bytes;
                            const void* src_ptr =
                                static_cast<const std::byte*>(src.data) +
                                static_cast<std::size_t>(src_head) * head_bytes;
                            CUDA_CHECK(cudaMemcpy2DAsync(
                                dst_ptr, dst_pitch, src_ptr, src_pitch,
                                head_bytes * static_cast<std::size_t>(count), token_rows,
                                cudaMemcpyDeviceToDevice, state.execution.device.stream));
                        };
                        const auto run_attention = [&]() {
                            ops::bidirectional_gqa_attention(
                                padded_query, key_batch, value_batch, frontiers, attention_valid,
                                full_rows, Config::attention_scale, full_context, envelopes.full,
                                state.execution.work, padded_attention,
                                state.execution.device.stream);
                        };
                        {
                            auto call_scope = state.execution.work.scope();
                            for (std::int32_t group = 0; group < 8; ++group) {
                                copy_heads(padded_query, group * 4, 4, query_batch, group * 5);
                            }
                            run_attention();
                            for (std::int32_t group = 0; group < 8; ++group) {
                                copy_heads(attention_batch, group * 5, 4, padded_attention,
                                           group * 4);
                            }
                        }
                        {
                            auto call_scope = state.execution.work.scope();
                            for (std::int32_t group = 0; group < 8; ++group) {
                                copy_heads(padded_query, group * 4, 1, query_batch, group * 5 + 4);
                            }
                            run_attention();
                            for (std::int32_t group = 0; group < 8; ++group) {
                                copy_heads(attention_batch, group * 5 + 4, 1, padded_attention,
                                           group * 4);
                            }
                        }
                    } else {
                        static_assert(Config::query_heads == 32 && Config::kv_heads == 8,
                                      "unsupported DFlash attention geometry");
                    }
                }
                if constexpr (Config::bf16_weights) {
                    Tensor delta = roots.attention_delta.view({Config::hidden, columns});
                    ops::linear(roots.attention.view({Config::query_size, columns}),
                                weight.attention_output, delta, state.execution.device.stream);
                    ops::residual_add(delta, residual, state.execution.device.stream);
                } else {
                    ops::linear_add(roots.attention.view({Config::query_size, columns}),
                                    weight.attention_output, residual, state.execution.work,
                                    state.execution.device.stream);
                }
            }
            {
                nvtx::ScopedRange mlp_range(nvtx::Name::DFlashMlp, nvtx::Category::PostMixer,
                                            static_cast<std::uint64_t>(layer));
                auto mlp_scope = state.execution.work.scope();
                auto roots = workspace_recipe::dflash_mlp<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.post_attention_norm, Config::rms_epsilon, false,
                             roots.hidden, state.execution.device.stream);
                if constexpr (Config::bf16_weights) {
                    Tensor gate_up = roots.gate_up.view({2 * Config::intermediate, columns});
                    ops::linear(roots.hidden, weight.gate_up, gate_up,
                                state.execution.device.stream);
                    ops::silu_mul(gate_up.slice(0, 0, Config::intermediate),
                                  gate_up.slice(0, Config::intermediate, Config::intermediate),
                                  roots.intermediate, state.execution.device.stream);
                    Tensor delta = roots.delta.view({Config::hidden, columns});
                    ops::linear(roots.intermediate, weight.down, delta,
                                state.execution.device.stream);
                    ops::residual_add(delta, residual, state.execution.device.stream);
                } else {
                    ops::linear_swiglu(roots.hidden, weight.gate_up, roots.intermediate,
                                       state.execution.work, state.execution.device.stream);
                    ops::linear_add(roots.intermediate, weight.down, residual, state.execution.work,
                                    state.execution.device.stream);
                }
            }
        }

        Tensor packed = state.execution.work.alloc(
            DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
        const std::size_t element_bytes = dtype_size(DType::BF16);
        const std::size_t row_bytes =
            static_cast<std::size_t>(Config::hidden) * static_cast<std::size_t>(k) * element_bytes;
        const std::size_t source_pitch =
            static_cast<std::size_t>(Config::hidden) * width * element_bytes;
        const auto* source = static_cast<const std::byte*>(residual.data);
        // The trainer pairs block column p (position a+p) with token x_{a+p} and only
        // consumes `out[:, 1:]` (train_dspark.py:168-169,428-432): column 0 is the anchor
        // and never part of the proposal. Taking columns 0..k-1 here therefore shifted the
        // whole draft block by one column for bf16 drafts (the dspark artifact is bf16),
        // which is what collapsed its position-0 acceptance. Skip the anchor column always;
        // the offset stays a named constant so it can become an artifact property later.
        constexpr std::size_t source_column_offset = 1;
        source += source_column_offset * static_cast<std::size_t>(Config::hidden) * element_bytes;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.data, row_bytes, source, source_pitch, row_bytes,
                                     static_cast<std::size_t>(batch_size), cudaMemcpyDeviceToDevice,
                                     state.execution.device.stream));
        Tensor proposal_hidden = state.execution.work.alloc(
            DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
        ops::rmsnorm(packed, state.execution.model.dflash->final_norm, Config::rms_epsilon, false,
                     proposal_hidden, state.execution.device.stream);
        Tensor flat_drafts = drafts.view({static_cast<std::int32_t>(k) * batch_size});
        if (state.execution.proposal_head == ProposalHead::Full) {
            Tensor logits = state.execution.work.alloc(
                DType::BF16, {TextConfig::output_rows, static_cast<std::int32_t>(k) * batch_size});
            ops::linear(proposal_hidden, state.execution.model.output_head, logits,
                        state.execution.device.stream);
            if (state.execution.model.dflash->markov_w1.has_value() &&
                state.execution.model.dflash->markov_w2.has_value()) {
                Tensor best_value = state.execution.work.alloc(DType::I32, {batch_size});
                Tensor best_index = state.execution.work.alloc(DType::I32, {batch_size});
                Tensor extents    = state.execution.work.alloc(DType::I32, {batch_size});
                CUDA_CHECK(cudaMemcpyAsync(
                    extents.data, frame.proposal_extents.slice(0, 0, batch_size).data,
                    static_cast<std::size_t>(batch_size) * sizeof(std::int32_t),
                    cudaMemcpyDeviceToDevice, state.execution.device.stream));
                ops::dspark_markov_argmax(logits, *state.execution.model.dflash->markov_w1,
                                          *state.execution.model.dflash->markov_w2, anchors,
                                          flat_drafts, best_value, best_index, extents,
                                          state.svip_entropy_threshold,
                                          state.execution.device.stream);
                CUDA_CHECK(cudaMemcpyAsync(
                    frame.proposal_extents.slice(0, 0, batch_size).data, extents.data,
                    static_cast<std::size_t>(batch_size) * sizeof(std::int32_t),
                    cudaMemcpyDeviceToDevice, state.execution.device.stream));
            } else {
                ops::argmax(logits, flat_drafts, TextConfig::token_domain,
                            state.execution.device.stream);
            }
        } else {
            if (!state.execution.model.optimized_proposal.has_value()) {
                throw std::logic_error("optimized DFlash proposal head is unavailable");
            }
            const auto& proposal = *state.execution.model.optimized_proposal;
            Tensor logits        = state.execution.work.alloc(
                DType::BF16, {V::draft_head_rows, static_cast<std::int32_t>(k) * batch_size});
            ops::linear(proposal_hidden, proposal.head, logits, state.execution.device.stream);
            ops::argmax(logits, flat_drafts, V::draft_head_rows, state.execution.device.stream);
            ops::proposal_remap_token_ids(flat_drafts,
                                          static_cast<const std::int32_t*>(proposal.token_ids.data),
                                          V::draft_head_rows, state.execution.device.stream);
        }
        state.execution.work.reset();
    }
}

auto dflash_decode_batch_body(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              DFlashEnvelopes envelopes,
                              ops::GqaExecutionEnvelope target_envelope) {
    return [&state, batch_size, k, envelopes, target_envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kDFlashDecodeMaximumDrafts) {
            throw std::logic_error("DFlash decode batch state is incomplete");
        }
        qwen3_6::DFlashDecodeState& frame = state.frame;
        const std::int32_t width          = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor context_starts     = frame.context_frontiers.slice(0, 0, batch_size);
        Tensor extents            = frame.proposal_extents.slice(0, 0, batch_size);
        Tensor valid_columns      = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor text_rows          = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor dflash_rows        = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor active_lanes       = frame.active_lanes.slice(0, 0, batch_size);
        Tensor state_sources      = frame.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor append_positions   = frame.append_positions.slice(1, 0, batch_size);
        Tensor append_counts      = frame.append_counts.slice(0, 0, batch_size);
        Tensor drafts             = frame.draft_tokens.slice(1, 0, batch_size);
        Tensor verify_ids         = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions   = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor target_tokens      = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits      = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden      = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden    = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens    = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts    = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted           = frame.accepted_drafts.slice(0, 0, batch_size);

        state.execution.work.reset();
        Tensor compact_features = state.execution.work.alloc(
            DType::BF16, {Variant::DFlashConfig::feature_rows, width, batch_size});
        ops::prepare_ragged_prefix(dflash_state(state).pending_features, active_lanes,
                                   context_starts, frontiers, compact_features, append_positions,
                                   append_counts, state.execution.device.stream);
        append_context_impl<Variant>(state, compact_features, append_positions, append_counts,
                                     active_lanes, dflash_rows, envelopes.append);

        propose_batch_impl<Variant>(state, frame, batch_size, k, envelopes);
        ops::speculative_prepare_verify_ids(anchors, drafts, extents, verify_ids,
                                            state.execution.device.stream);

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache);
        DFlashFeatureSink sink =
            batch_feature_sink_impl<Variant>(state, active_lanes, valid_columns, width, batch_size);
        {
            nvtx::ScopedRange target_range(nvtx::Name::DecodeDFlashTarget, nvtx::Category::DFlash,
                                           static_cast<std::uint64_t>(width) * batch_size);
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 TargetVerifyFrameView{
                                     .ids                     = verify_ids,
                                     .cache_positions         = target_positions,
                                     .rope_positions          = target_positions,
                                     .valid_columns           = valid_columns,
                                     .kv_table_rows           = text_rows,
                                     .state_source_slots      = state_sources,
                                     .state_destination_slots = state_destinations,
                                     .target_hidden           = target_hidden,
                                     .target_logits           = target_logits,
                                     .target_tokens           = target_tokens,
                                     .drafts                  = drafts,
                                     .current_extents         = extents,
                                     .frontiers               = frontiers,
                                     .anchors                 = anchors,
                                     .licensed_tokens         = licensed_tokens,
                                     .licensed_counts         = licensed_counts,
                                     .accepted_drafts         = accepted,
                                     .selected_hidden         = selected_hidden,
                                     .replay_records          = state.execution.replay_records,
                                     .sampling                = frame.sampling,
                                     .feature_sink            = &sink,
                                 },
                                 target_envelope);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            frame.egress_proposal_extents.slice(0, 0, batch_size).data, extents.data,
            static_cast<std::size_t>(batch_size) * sizeof(std::int32_t), cudaMemcpyDeviceToDevice,
            state.execution.device.stream));
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::DFlashDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

} // namespace

DFlashFeatureSink dflash_feature_sink(PrefillContext& state,
                                      DFlashFeatureSink::PrefillConsumer consume_prefill) {
    return prefill_feature_sink_impl<Variant>(state, std::move(consume_prefill));
}

void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::GqaExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition) {
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    capture_graph(state, definition, body);
}

void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes,
                         ops::GqaExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable) {
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
