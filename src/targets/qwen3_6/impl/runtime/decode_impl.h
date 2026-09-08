#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

auto ordinary_batch_body(OrdinaryBatchContext& state, std::int32_t batch_size,
                         ops::GqaExecutionEnvelope envelope, std::int32_t segment = 0,
                         std::int32_t segments = 1) {
    return [&state, batch_size, envelope, segment, segments] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency)) {
            throw std::logic_error("ordinary decode batch state is incomplete");
        }

        qwen3_6::OrdinaryDecodeState& ordinary = state.frame;
        CUDA_CHECK(cudaMemcpyAsync(ordinary.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::OrdinaryDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache);
        card.set_graph_segment(segment, segments);

        Tensor tokens             = ordinary.tokens.slice(0, 0, batch_size);
                Tensor cache_positions    = ordinary.cache_positions.slice(0, 0, batch_size);
                Tensor rope_positions     = ordinary.rope_positions.slice(0, 0, batch_size);
        Tensor kv_rows            = ordinary.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources      = ordinary.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = ordinary.state_destination_slots.slice(0, 0, batch_size);
        Tensor hidden             = ordinary.hidden.slice(1, 0, batch_size);
        Tensor logits             = ordinary.logits.slice(1, 0, batch_size);
        Tensor sampled            = ordinary.sampled_tokens.slice(0, 0, batch_size);

        card.ordinary_decode_batch(tokens, cache_positions, rope_positions, kv_rows, state_sources,
                                   state_destinations, envelope, hidden, logits);
        if (segment + 1 == segments) {
            ops::scatter(hidden, state_destinations, state.continuation_hidden_store,
                         state.execution.device.stream);
            
            ops::sample(logits, sampled, TextConfig::token_domain, ordinary.sampling,
                        cache_positions, ops::kSamplePurposeDecode, state.execution.work,
                        state.execution.device.stream);
            CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, ordinary.egress.data,
                                       sizeof(qwen3_6::OrdinaryDecodeEgress),
                                       cudaMemcpyDeviceToHost, state.execution.device.stream));
            
        }
    };
}

} // namespace

void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::GqaExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition) {
    // Segmented capture: splitting the decode round into a few graphs lets the
    // driver submit segment k's nodes while segment k-1 is still executing,
    // hiding the per-node launch cost (~0.9us/node) that otherwise idles the GPU.
    constexpr int kSegments = 1;
    if (kSegments <= 1) {
        auto body = ordinary_batch_body(state, batch_size, envelope);
        capture_graph(state, definition, body);
        return;
    }
    std::vector<std::function<void()>> bodies;
    bodies.reserve(kSegments);
    for (int s = 0; s < kSegments; ++s) {
        bodies.push_back(ordinary_batch_body(state, batch_size, envelope, s, kSegments));
    }
    state.execution.work.reset();
    definition.capture_segments(state.execution.device.stream, bodies);
}

void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::GqaExecutionEnvelope envelope,
                           DecodeGraphExecutable* executable) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
