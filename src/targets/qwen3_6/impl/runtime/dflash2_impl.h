#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "core/dtype.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/dflash2_grouped_conv.h"
#include "ninfer/ops/dflash2_selector.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
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
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

DFlash2PersistentState& dflash2_state(PrefillContext& state) {
    if (state.dflash2 == nullptr) {
        throw std::logic_error("DFlash2 schedule requires DFlash2 weights and state");
    }
    return *state.dflash2;
}

DFlash2PersistentState& dflash2_state(DFlash2BatchContext& state) { return state.dflash2; }

DFlash2PersistentState& dflash2_state(DFlash2AppendContext& state) { return state.dflash2; }

Weight conv_side_weight(const Weight& base, std::int32_t side, std::int32_t taps) {
    Weight out      = base;
    out.qdata       = static_cast<const std::byte*>(base.qdata) +
                static_cast<std::size_t>(side) * static_cast<std::size_t>(taps) *
                    static_cast<std::size_t>(base.k) * dtype_size(DType::BF16);
    out.n           = taps;
    return out;
}

template <class V>
DFlashFeatureSink dflash2_prefill_feature_sink_impl(
    PrefillContext& state, DFlashFeatureSink::PrefillConsumer consume_prefill) {
    if constexpr (!V::supports_dflash2) {
        throw std::logic_error("DFlash2 feature capture is unavailable for this target");
    } else {
        using Config = typename V::DFlash2Config;
        return DFlashFeatureSink{
            .features        = &dflash2_state(state).prefill_features,
            .positions       = &dflash2_state(state).prefill_positions,
            .layers          = std::span<const int>(Config::target_feature_layers),
            .consume_prefill = std::move(consume_prefill),
        };
    }
}

template <class V>
DFlashFeatureSink dflash2_batch_feature_sink_impl(DFlash2BatchContext& state, const Tensor& lanes,
                                                  const Tensor& valid_columns,
                                                  std::int32_t width, std::int32_t batch_size) {
    if constexpr (!V::supports_dflash2) {
        throw std::logic_error("DFlash2 feature capture is unavailable for this target");
    } else {
        using Config = typename V::DFlash2Config;
        return DFlashFeatureSink{
            .batch_features      = &dflash2_state(state).pending_features,
            .batch_lanes         = &lanes,
            .batch_valid_columns = &valid_columns,
            .batch_width         = width,
            .batch_size          = batch_size,
            .layers              = std::span<const int>(Config::target_feature_layers),
        };
    }
}

void append_context_impl(DFlash2AppendContext& state, const Tensor& features,
                         const Tensor& positions, const Tensor& commit_counts,
                         const Tensor& lanes, const Tensor& table_rows,
                         ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    using Config            = DFlash2Config;
    const std::int32_t width = features.ne[1];
    const std::int32_t batch = features.ne[2];
    const std::int32_t columns = width * batch;
    if (width <= 0 || batch <= 0 || features.dtype != DType::BF16 ||
        features.ne[0] != Config::feature_rows || positions.dtype != DType::I32 ||
        positions.ne[0] != width || positions.ne[1] != batch ||
        commit_counts.dtype != DType::I32 || commit_counts.ne[0] != batch ||
        lanes.dtype != DType::I32 || lanes.ne[0] != batch) {
        throw std::invalid_argument("DFlash2 context append inputs are invalid");
    }
    if (!state.execution.model.dflash2.has_value()) {
        throw std::logic_error("DFlash2 weights are unavailable");
    }
    const auto& dflash2 = *state.execution.model.dflash2;

    // The cyclic draft cache retains only the last `local_capacity` absolute
    // positions. An oversized prefill chunk can therefore skip its older
    // columns and commit just the trailing window.
    const bool replace_local_window = batch == 1 && width > Config::local_capacity;
    const int local_offset = replace_local_window ? width - Config::local_capacity : 0;
    const int local_width  = replace_local_window ? Config::local_capacity : width;
    Tensor local_counts    = commit_counts;
    ops::KVCacheAppendPrefixExecutionEnvelope local_envelope = envelope;
    if (replace_local_window) {
        if (!state.execution.io.dflash_prefill) {
            throw std::logic_error("DFlash2 prefill count storage is unavailable");
        }
        local_counts = state.execution.io.dflash_prefill->produced_count;
        ops::set_i32_scalar(local_counts, Config::local_capacity,
                            state.execution.device.stream);
        local_envelope = {static_cast<std::uint32_t>(Config::local_capacity),
                          static_cast<std::uint32_t>(Config::local_capacity)};
    }

    auto context_roots = workspace_recipe::dflash_context<Config>(state.execution.work, columns);
    ops::linear(features.view({Config::feature_rows, columns}), dflash2.feature_projection,
                context_roots.projected, state.execution.device.stream);
    Tensor context_full = context_roots.normalized;
    ops::rmsnorm(context_roots.projected, dflash2.context_norm, Config::rms_epsilon, false,
                 context_full, state.execution.device.stream);
    Tensor context = replace_local_window
                         ? context_full.slice(1, local_offset, local_width)
                         : context_full;
    Tensor local_positions = replace_local_window
                                 ? positions.slice(0, local_offset, local_width)
                                 : positions;

    for (int layer = 0; layer < Config::layers; ++layer) {
        auto layer_scope = state.execution.work.scope();
        const auto& weight =
            dflash2.layers.at(static_cast<std::size_t>(layer));
        auto layer_roots =
            workspace_recipe::dflash_context_layer<Config>(state.execution.work, local_width * batch);
        Tensor key_raw =
            layer_roots.key_raw.view({Config::head_dim, Config::kv_heads, local_width * batch});
        Tensor value = layer_roots.value.view({Config::head_dim, Config::kv_heads, local_width * batch});
        Tensor key_flat   = key_raw.view({Config::kv_size, local_width * batch});
        Tensor value_flat = value.view({Config::kv_size, local_width * batch});
        ops::linear(context, weight.context_key, key_flat, state.execution.device.stream);
        ops::linear(context, weight.context_value, value_flat, state.execution.device.stream);
        Tensor key = layer_roots.key.view({Config::head_dim, Config::kv_heads, local_width * batch});
        ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                     state.execution.device.stream);
        ops::rope(local_positions.view({local_width * batch}), Config::head_dim,
                  Config::rope_theta, key, state.execution.device.stream);
        Tensor key_batch = key.view({Config::head_dim, Config::kv_heads, local_width, batch});
        Tensor value_batch = value.view({Config::head_dim, Config::kv_heads, local_width, batch});
        Tensor position_batch = local_positions.view({local_width, batch});
        ops::kv_cache_append_prefix(key_batch, value_batch, position_batch, local_counts, lanes,
                                    local_envelope,
                                    dflash2_state(state).local_layer(
                                        static_cast<std::uint32_t>(layer)),
                                    Config::local_window, state.execution.device.stream);
    }
}

void propose_batch_impl(DFlash2BatchContext& state, qwen3_6::DFlashDecodeState& frame,
                        std::int32_t batch_size, std::uint32_t k, DFlash2Envelopes envelopes) {
    using Config             = DFlash2Config;
    const std::int32_t width = static_cast<std::int32_t>(k) + 1;
    const std::int32_t columns = width * batch_size;
    Tensor anchors           = frame.anchors.slice(0, 0, batch_size);
    Tensor frontiers         = frame.execution_frontiers.slice(0, 0, batch_size);
    Tensor valid_columns     = frame.target_valid_columns.slice(0, 0, batch_size);
    Tensor lanes             = frame.active_lanes.slice(0, 0, batch_size);
    Tensor ids               = frame.proposal_ids.slice(1, 0, batch_size);
    Tensor positions         = frame.proposal_positions.slice(1, 0, batch_size);
    Tensor drafts            = frame.draft_tokens.slice(1, 0, batch_size);
    if (!state.execution.model.dflash2.has_value()) {
        throw std::logic_error("DFlash2 weights are unavailable");
    }
    const auto& dflash2 = *state.execution.model.dflash2;

    state.execution.work.reset();
    Tensor attention_valid = state.execution.work.alloc(DType::I32, {batch_size});
    ops::set_i32_scalar(attention_valid, static_cast<std::int32_t>(width),
                        state.execution.device.stream);
    (void)valid_columns;

    ops::prepare_masked_block(anchors, frontiers, attention_valid, Config::mask_token, ids,
                              positions, state.execution.device.stream);
    Tensor residual = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
    ops::embedding(ids.view({columns}), state.execution.model.token_embedding, residual,
                   state.execution.device.stream);

    for (int layer = 0; layer < Config::layers; ++layer) {
        const auto& weight = dflash2.layers.at(static_cast<std::size_t>(layer));
        {
            auto attention_scope = state.execution.work.scope();
            auto roots =
                workspace_recipe::dflash_attention<Config>(state.execution.work, columns);
            ops::rmsnorm(residual, weight.input_norm, Config::rms_epsilon, false, roots.hidden,
                         state.execution.device.stream);

            Tensor conv_coefficients = state.execution.work.alloc(DType::BF16, {1280, columns});
            ops::linear(roots.hidden, weight.attention_conv_projection, conv_coefficients,
                        state.execution.device.stream);
            Tensor conv_hidden = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
            ops::dflash2_grouped_conv(
                roots.hidden, conv_coefficients,
                conv_side_weight(weight.attention_conv_base, 0, Config::conv_kernel_size),
                static_cast<std::int32_t>(k + 1), Config::conv_group_size,
                Config::conv_kernel_size, 0, conv_hidden, state.execution.device.stream);

            Tensor query_raw = roots.query_raw.view({Config::head_dim, Config::query_heads, columns});
            Tensor key_raw   = roots.key_raw.view({Config::head_dim, Config::kv_heads, columns});
            Tensor value     = roots.value.view({Config::head_dim, Config::kv_heads, columns});
            Tensor query_flat = query_raw.view({Config::query_size, columns});
            Tensor key_flat   = key_raw.view({Config::kv_size, columns});
            Tensor value_flat = value.view({Config::kv_size, columns});
            const Weight qkv_weight = weight.query_key_value;
            const std::size_t row_len = static_cast<std::size_t>(qkv_weight.k) * dtype_size(DType::BF16);
            Weight query_weight = qkv_weight;
            query_weight.n      = Config::query_size;
            Weight key_weight   = qkv_weight;
            key_weight.n        = Config::kv_size;
            key_weight.qdata    = static_cast<const std::byte*>(qkv_weight.qdata) +
                                static_cast<std::size_t>(Config::query_size) * row_len;
            Weight value_weight = qkv_weight;
            value_weight.n      = Config::kv_size;
            value_weight.qdata  = static_cast<const std::byte*>(qkv_weight.qdata) +
                                 static_cast<std::size_t>(Config::query_size + Config::kv_size) *
                                     row_len;
            ops::linear(conv_hidden, query_weight, query_flat, state.execution.device.stream);
            ops::linear(conv_hidden, key_weight, key_flat, state.execution.device.stream);
            ops::linear(conv_hidden, value_weight, value_flat, state.execution.device.stream);

            Tensor query = roots.query.view({Config::head_dim, Config::query_heads, columns});
            Tensor key   = roots.key.view({Config::head_dim, Config::kv_heads, columns});
            ops::rmsnorm(query_raw, weight.query_norm, Config::rms_epsilon, false, query,
                         state.execution.device.stream);
            ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                         state.execution.device.stream);
            ops::rope(positions.view({columns}), Config::head_dim, Config::rope_theta, query, key,
                      state.execution.device.stream);
            Tensor query_batch =
                query.view({Config::head_dim, Config::query_heads, width, batch_size});
            Tensor key_batch =
                key.view({Config::head_dim, Config::kv_heads, width, batch_size});
            Tensor value_batch =
                value.view({Config::head_dim, Config::kv_heads, width, batch_size});
            Tensor attention_batch = roots.attention.view(
                {Config::head_dim, Config::query_heads, width, batch_size});
            ops::swa(query_batch, key_batch, value_batch, positions, attention_valid, lanes,
                     Config::attention_scale,
                     dflash2_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                     envelopes.local, Config::local_window, state.execution.work, attention_batch,
                     state.execution.device.stream);

            Tensor delta = roots.attention_delta.view({Config::hidden, columns});
            ops::linear(roots.attention.view({Config::query_size, columns}),
                        weight.attention_output, delta, state.execution.device.stream);
            Tensor conv_finish = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
            ops::dflash2_grouped_conv(
                delta, conv_coefficients,
                conv_side_weight(weight.attention_conv_base, 1, Config::conv_kernel_size),
                static_cast<std::int32_t>(k + 1), Config::conv_group_size,
                Config::conv_kernel_size, 1, conv_finish, state.execution.device.stream);
            ops::residual_add(conv_finish, residual, state.execution.device.stream);
        }
        {
            auto mlp_scope = state.execution.work.scope();
            auto roots = workspace_recipe::dflash_mlp<Config>(state.execution.work, columns);
            ops::rmsnorm(residual, weight.post_attention_norm, Config::rms_epsilon, false,
                         roots.hidden, state.execution.device.stream);

            Tensor conv_coefficients = state.execution.work.alloc(DType::BF16, {1280, columns});
            ops::linear(roots.hidden, weight.mlp_conv_projection, conv_coefficients,
                        state.execution.device.stream);
            Tensor conv_hidden = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
            ops::dflash2_grouped_conv(
                roots.hidden, conv_coefficients,
                conv_side_weight(weight.mlp_conv_base, 0, Config::conv_kernel_size),
                static_cast<std::int32_t>(k + 1), Config::conv_group_size,
                Config::conv_kernel_size, 0, conv_hidden, state.execution.device.stream);

            Tensor gate_up = roots.gate_up.view({2 * Config::intermediate, columns});
            ops::linear(conv_hidden, weight.gate_up, gate_up, state.execution.device.stream);
            ops::silu_mul(gate_up.slice(0, 0, Config::intermediate),
                          gate_up.slice(0, Config::intermediate, Config::intermediate),
                          roots.intermediate, state.execution.device.stream);
            Tensor delta = roots.delta.view({Config::hidden, columns});
            ops::linear(roots.intermediate, weight.down, delta, state.execution.device.stream);
            Tensor conv_finish = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
            ops::dflash2_grouped_conv(
                delta, conv_coefficients,
                conv_side_weight(weight.mlp_conv_base, 1, Config::conv_kernel_size),
                static_cast<std::int32_t>(k + 1), Config::conv_group_size,
                Config::conv_kernel_size, 1, conv_finish, state.execution.device.stream);
            ops::residual_add(conv_finish, residual, state.execution.device.stream);
        }
    }

    Tensor packed = state.execution.work.alloc(
        DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t row_bytes =
        static_cast<std::size_t>(Config::hidden) * static_cast<std::size_t>(k) * element_bytes;
    const std::size_t source_pitch =
        static_cast<std::size_t>(Config::hidden) * width * element_bytes;
    // DFlash2 predicts the seven masked columns (1..7); the bonus token stays
    // at column 0 and is not part of the proposal block.
    const auto* source = static_cast<const std::byte*>(residual.data) +
                         static_cast<std::size_t>(Config::hidden) * element_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(packed.data, row_bytes, source, source_pitch, row_bytes,
                                 static_cast<std::size_t>(batch_size), cudaMemcpyDeviceToDevice,
                                 state.execution.device.stream));
    Tensor proposal_hidden = state.execution.work.alloc(
        DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
    ops::rmsnorm(packed, dflash2.final_norm, Config::rms_epsilon, false, proposal_hidden,
                 state.execution.device.stream);

    Tensor flat_drafts = drafts.view({static_cast<std::int32_t>(k) * batch_size});
    Tensor logits = state.execution.work.alloc(
        DType::BF16, {TextConfig::output_rows, static_cast<std::int32_t>(k) * batch_size});
    ops::linear(proposal_hidden, state.execution.model.output_head, logits,
                state.execution.device.stream);
    Tensor projected = state.execution.work.alloc(
        DType::BF16, {Config::selector_rank, static_cast<std::int32_t>(k) * batch_size});
    ops::linear(proposal_hidden, dflash2.selector_hidden_projection, projected,
                state.execution.device.stream);
    Tensor candidates = state.execution.work.alloc(
        DType::I32, {batch_size, Config::block_drafts, Config::selector_top_k});
    Tensor unary = state.execution.work.alloc(
        DType::FP32, {batch_size, Config::block_drafts, Config::selector_top_k});
    Tensor scores = state.execution.work.alloc(
        DType::FP32,
        {batch_size, Config::block_drafts, Config::selector_top_k, Config::selector_top_k});
    ops::dflash2_selector(logits, projected, dflash2.selector_predecessor_codebook,
                          dflash2.selector_successor_codebook, anchors, candidates, unary, scores,
                          flat_drafts, Config::block_drafts, Config::selector_top_k,
                          state.execution.device.stream);
    state.execution.work.reset();
}

auto dflash2_decode_batch_body(DFlash2BatchContext& state, std::int32_t batch_size,
                               std::uint32_t k, DFlash2Envelopes envelopes,
                               ops::GqaExecutionEnvelope target_envelope) {
    return [&state, batch_size, k, envelopes, target_envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kDFlashDecodeMaximumDrafts || k != DFlash2Config::block_drafts) {
            throw std::logic_error("DFlash2 decode batch state is incomplete");
        }
        qwen3_6::DFlashDecodeState& frame = state.frame;
        const std::int32_t width          = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        Tensor anchors          = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers        = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor context_starts   = frame.context_frontiers.slice(0, 0, batch_size);
        Tensor extents          = frame.proposal_extents.slice(0, 0, batch_size);
        Tensor valid_columns    = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor text_rows        = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor lanes            = frame.active_lanes.slice(0, 0, batch_size);
        Tensor state_sources    = frame.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor append_positions = frame.append_positions.slice(1, 0, batch_size);
        Tensor append_counts    = frame.append_counts.slice(0, 0, batch_size);
        Tensor drafts           = frame.draft_tokens.slice(1, 0, batch_size);
        Tensor verify_ids       = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor target_tokens    = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits    = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden    = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden  = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens  = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts  = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted         = frame.accepted_drafts.slice(0, 0, batch_size);

        state.execution.work.reset();
        Tensor compact_features = state.execution.work.alloc(
            DType::BF16, {DFlash2Config::feature_rows, width, batch_size});
        ops::prepare_ragged_prefix(dflash2_state(state).pending_features, lanes, context_starts,
                                   frontiers, compact_features, append_positions, append_counts,
                                   state.execution.device.stream);
        DFlash2AppendContext append_state{.execution = state.execution,
                                          .dflash2   = state.dflash2};
        append_context_impl(append_state, compact_features, append_positions, append_counts, lanes,
                            text_rows, envelopes.append);

        propose_batch_impl(state, frame, batch_size, k, envelopes);
        ops::speculative_prepare_verify_ids(anchors, drafts, extents, verify_ids,
                                            state.execution.device.stream);

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache);
        DFlashFeatureSink sink = dflash2_batch_feature_sink_impl<Variant>(
            state, lanes, valid_columns, width, batch_size);
        target_verify_accept(state.execution, state.continuation_hidden_store, card,
                             TargetVerifyFrameView{
                                 .ids             = verify_ids,
                                 .cache_positions = target_positions,
                                 .rope_positions  = target_positions,
                                 .valid_columns   = valid_columns,
                                 .kv_table_rows   = text_rows,
                                 .state_source_slots      = state_sources,
                                 .state_destination_slots = state_destinations,
                                 .target_hidden   = target_hidden,
                                 .target_logits   = target_logits,
                                 .target_tokens   = target_tokens,
                                 .drafts          = drafts,
                                 .current_extents = extents,
                                 .frontiers       = frontiers,
                                 .anchors         = anchors,
                                 .licensed_tokens = licensed_tokens,
                                 .licensed_counts = licensed_counts,
                                 .accepted_drafts = accepted,
                                 .selected_hidden = selected_hidden,
                                 .replay_records  = state.execution.replay_records,
                                 .sampling        = frame.sampling,
                                 .feature_sink    = &sink,
                             },
                             target_envelope);
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

DFlashFeatureSink dflash2_feature_sink(PrefillContext& state,
                                       DFlashFeatureSink::PrefillConsumer consume_prefill) {
    return dflash2_prefill_feature_sink_impl<Variant>(state, std::move(consume_prefill));
}

void dflash2_append_context(DFlash2AppendContext& state, const Tensor& features,
                            const Tensor& positions, const Tensor& commit_counts,
                            const Tensor& lanes, const Tensor& table_rows,
                            ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl(state, features, positions, commit_counts, lanes, table_rows, envelope);
}

void dflash2_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                            const Tensor& commit_counts, const Tensor& lanes,
                            const Tensor& table_rows,
                            ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    if (state.dflash2 == nullptr) {
        throw std::logic_error("DFlash2 context append requires DFlash2 state");
    }
    DFlash2AppendContext context{.execution = state.execution, .dflash2 = *state.dflash2};
    append_context_impl(context, features, positions, commit_counts, lanes, table_rows, envelope);
}

void capture_dflash2_decode_batch(DFlash2BatchContext& state, std::int32_t batch_size,
                                  std::uint32_t k, DFlash2Envelopes envelopes,
                                  ops::GqaExecutionEnvelope target_envelope,
                                  DecodeGraphDefinition& definition) {
    auto body = dflash2_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    capture_graph(state, definition, body);
}

void dflash2_decode_batch(DFlash2BatchContext& state, std::int32_t batch_size, std::uint32_t k,
                          DFlash2Envelopes envelopes, ops::GqaExecutionEnvelope target_envelope,
                          DecodeGraphExecutable* executable) {
    auto body = dflash2_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
