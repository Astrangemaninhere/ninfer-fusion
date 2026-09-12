#include <cuda_bf16.h>
#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/vocab_topk16.h"
#include "ninfer/ops/sampling.h"
namespace { constexpr int kDumpTopK = 16; }

#include "ninfer/ops/scalar.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

DFlashFeatureSink make_dflash_prefill_sink(PrefillContext& state) {
    if (state.execution.model.features.dflash2()) {
        if (!state.execution.io.dflash_decode || state.dflash2_host_ingress == nullptr) {
            throw std::logic_error("DFlash2 prefill controls are unavailable");
        }
        return dflash2_feature_sink(
            state, [&state](const Tensor& features, const Tensor& positions,
                            bool rewrite_checkpoint) {
                auto& frame  = *state.execution.io.dflash_decode;
                Tensor count = frame.append_counts.slice(0, 0, 1);
                Tensor lane  = frame.state_destination_slots.slice(0, 0, 1);
                Tensor row   = frame.dflash_kv_table_rows.slice(0, 0, 1);
                ops::set_i32_scalar(count, features.ne[1], state.execution.device.stream);
                const auto exact = static_cast<std::uint32_t>(features.ne[1]);
                dflash2_append_context(state, features, positions, count, lane, row, {exact, exact});
                if (rewrite_checkpoint) {
                    state.dflash2->save_rewrite_checkpoint(
                        state.dflash2_host_ingress->active_lanes[0],
                        state.execution.device.stream);
                }
            });
    }
    if (!state.execution.io.dflash_decode || state.dflash_host_ingress == nullptr) {
        throw std::logic_error("DFlash prefill controls are unavailable");
    }
    return dflash_feature_sink(
        state, [&state](const Tensor& features, const Tensor& positions, bool rewrite_checkpoint) {
            auto& frame  = *state.execution.io.dflash_decode;
            Tensor count = frame.append_counts.slice(0, 0, 1);
            Tensor lane  = frame.state_destination_slots.slice(0, 0, 1);
            Tensor row   = frame.dflash_kv_table_rows.slice(0, 0, 1);
            ops::set_i32_scalar(count, features.ne[1], state.execution.device.stream);
            const auto exact = static_cast<std::uint32_t>(features.ne[1]);
            dflash_append_context(state, features, positions, count, lane, row, {exact, exact});
            (void)rewrite_checkpoint;
        });
}

} // namespace

void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t state_source_slot,
                         std::int32_t state_destination_slot, std::uint32_t mtp_proposal_extent) {
    card.set_sampling(sampling);
    card.set_linear_state_slots(state_source_slot, state_destination_slot);
    card.set_gdn_state_action(GdnStateAction::UpdateInPlace, nullptr);
    card.set_mtp_proposal_extent(mtp_proposal_extent);
    if (execution.proposal_head == ProposalHead::Full) {
        card.set_proposal_head(nullptr, nullptr, 0);
        return;
    }
    if (card.proposal_head() == nullptr || card.proposal_head_ids() == nullptr ||
        card.proposal_head_n() <= 0) {
        throw std::runtime_error("optimized proposal head is unavailable");
    }
}

// Hidden-state dump mode (DFlash2 finetune data collection): when
// NINFER_HS_DUMP_DIR is set, every dflash/dflash2 prefill chunk appends a raw
// record file (chunk_%06d.bin): magic "NHS1", i32 tokens, i32 ids[tokens],
// u16 bf16 feat[feature_rows x tokens] (five target layers concatenated in
// capture order), u16 bf16 last[hidden x tokens] (final-norm **input**, i.e.
// the hidden *before* text/final_norm is applied; consumers must apply it
// themselves. Verified empirically: RMS(last)=1.87 > max|final_norm|=1.71,
// while RMS(rmsnorm(last))=0.97 ~ RMS(final_norm)=0.95.)
void dump_prefill_chunk(const PrefillContext& state, std::span<const TokenId> ids,
                        std::uint32_t chunk_base, std::int32_t tokens, cudaStream_t stream) {
    static const char* dump_dir = std::getenv("NINFER_HS_DUMP_DIR");
    if (dump_dir == nullptr || *dump_dir == '\0' || tokens <= 0) { return; }
    static std::atomic<std::uint32_t> counter{0};
    const Tensor* features = nullptr;
    if (state.dflash != nullptr) { features = &state.dflash->prefill_features; }
    if (state.dflash2 != nullptr) { features = &state.dflash2->prefill_features; }
    if (features == nullptr || features->dtype != DType::BF16 || tokens > features->ne[1] ||
        state.execution.prefill_hidden.ne[1] < tokens) {
        throw std::logic_error("HS dump prefill buffers are invalid");
    }
    const std::int32_t feature_rows = features->ne[0];
    const std::size_t token_bytes =
        static_cast<std::size_t>(tokens) * sizeof(std::int32_t);
    const std::size_t feat_bytes =
        static_cast<std::size_t>(feature_rows) * static_cast<std::size_t>(tokens) * 2;
    const std::size_t last_bytes =
        static_cast<std::size_t>(state.execution.prefill_hidden.ne[0]) *
        static_cast<std::size_t>(tokens) * 2;
    std::vector<std::int32_t> ids_host(static_cast<std::size_t>(tokens));
    for (std::int32_t i = 0; i < tokens; ++i) {
        ids_host[static_cast<std::size_t>(i)] = ids[chunk_base + static_cast<std::size_t>(i)];
    }
    std::vector<std::byte> feat_host(feat_bytes);
    std::vector<std::byte> last_host(last_bytes);
    CUDA_CHECK(cudaMemcpyAsync(feat_host.data(), features->data, feat_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(last_host.data(), state.execution.prefill_hidden.data, last_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::uint32_t id = counter.fetch_add(1);
    const std::string path =
        std::string(dump_dir) + "/chunk_" + std::to_string(id / 1000000 % 10) +
        std::to_string(id / 100000 % 10) + std::to_string(id / 10000 % 10) +
        std::to_string(id / 1000 % 10) + std::to_string(id / 100 % 10) +
        std::to_string(id / 10 % 10) + std::to_string(id % 10) + ".bin";
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) { throw std::runtime_error("HS dump cannot open " + path); }
    const std::uint32_t magic = 0x4E485331U;
    std::fwrite(&magic, sizeof(magic), 1, file);
    std::fwrite(&tokens, sizeof(tokens), 1, file);
    std::fwrite(ids_host.data(), sizeof(std::int32_t), ids_host.size(), file);
    std::fwrite(feat_host.data(), 1, feat_host.size(), file);
    std::fwrite(last_host.data(), 1, last_host.size(), file);
    // Validation mode (NINFER_HS_DUMP_TOPK=1): engine-internal ground truth —
    // per-column argmax over the full vocab using the engine's own head on the
    // post-norm hidden, appended as i32[tokens]. Offline tools compare this to
    // ids[pos+1] to decide whether dump semantics or offline head decode is
    // responsible for any teacher misalignment.
    if (std::getenv("NINFER_HS_DUMP_TOPK") != nullptr && tokens <= 1024) {
        const std::int32_t vocab = state.execution.model.output_head.n;
        Tensor xwin = state.execution.prefill_hidden.slice(1, 0, tokens);
        Tensor normed = state.execution.work.alloc(DType::BF16,
                                                   {TextConfig::hidden, tokens});
        // The [vocab, tokens] logits tensor does not fit the prefill workspace
        // arena (it is sized for the chunk's activation buffers), so this
        // diagnostic path owns a temporary device buffer instead.
        const std::size_t lg_bytes =
            static_cast<std::size_t>(vocab) * static_cast<std::size_t>(tokens) * 2;
        void* logits_dev = nullptr;
        CUDA_CHECK(cudaMalloc(&logits_dev, lg_bytes));
        Tensor logits(logits_dev, DType::BF16, {vocab, tokens});
        ops::rmsnorm(xwin, state.execution.model.final_norm,
                     TextConfig::rms_epsilon, false, normed, stream);
        ops::linear(normed, state.execution.model.output_head, logits, stream);
        kCfg.apply_final_logit_policy(logits, stream);
        std::vector<std::uint16_t> lg_host(lg_bytes / 2);
        CUDA_CHECK(cudaMemcpyAsync(lg_host.data(), logits.data, lg_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(logits_dev));
        // Host top-16 per token column (ties broken by smaller id).
        constexpr int kDumpTopK = 16;
        std::vector<std::int32_t> top1_host(static_cast<std::size_t>(tokens));
        std::vector<std::int32_t> ids16(static_cast<std::size_t>(tokens * kDumpTopK));
        std::vector<std::uint16_t> vals16(static_cast<std::size_t>(tokens * kDumpTopK));
        for (int t = 0; t < tokens; ++t) {
            float topv[kDumpTopK];
            int topi[kDumpTopK];
            for (int k = 0; k < kDumpTopK; ++k) { topv[k] = -1e30f; topi[k] = -1; }
            const std::uint16_t* column = lg_host.data() + static_cast<std::size_t>(t) * vocab;
            for (int v = 0; v < vocab; ++v) {
                const std::uint32_t bits = static_cast<std::uint32_t>(column[v]) << 16;
                float f;
                std::memcpy(&f, &bits, 4);
                if (f <= topv[kDumpTopK - 1]) { continue; }
                for (int k = 0; k < kDumpTopK; ++k) {
                    if (f > topv[k] || (f == topv[k] && v < topi[k])) {
                        for (int j = kDumpTopK - 1; j > k; --j) {
                            topv[j] = topv[j - 1];
                            topi[j] = topi[j - 1];
                        }
                        topv[k] = f;
                        topi[k] = v;
                        break;
                    }
                }
            }
            top1_host[static_cast<std::size_t>(t)] = topi[0];
            for (int k = 0; k < kDumpTopK; ++k) {
                ids16[static_cast<std::size_t>(t * kDumpTopK + k)] = topi[k];
                vals16[static_cast<std::size_t>(t * kDumpTopK + k)] =
                    static_cast<std::uint16_t>(
                        std::bit_cast<std::uint32_t>(topv[k]) >> 16);
            }
        }
        std::fwrite(top1_host.data(), sizeof(std::int32_t), top1_host.size(), file);
        const std::int32_t marker = kDumpTopK;
        std::fwrite(&marker, sizeof(marker), 1, file);
        std::fwrite(ids16.data(), sizeof(std::int32_t), ids16.size(), file);
        std::fwrite(vals16.data(), sizeof(std::uint16_t), vals16.size(), file);
    }
    std::fclose(file);
}

// KV cache forensics (NINFER_KVDUMP_DIR): after every prefill chunk, dump the
// block tables and the raw KV planes of the layers named in
// NINFER_KVDUMP_LAYERS (default "15,16", cap pages via NINFER_KVDUMP_PAGES).
// Host tools then decode the stored codes/scales exactly as the attention
// kernels would, which separates a bad write from a bad read (_TODO.md 104).
void dump_kv_cache(const PrefillContext& state, std::int32_t tokens, cudaStream_t stream) {
    const char* dump_dir = kvdump_dir();
    if (dump_dir == nullptr || tokens <= 0) { return; }
    std::vector<std::uint32_t> layers;
    if (const char* layers_env = std::getenv("NINFER_KVDUMP_LAYERS");
        layers_env != nullptr && *layers_env != '\0') {
        const std::string list(layers_env);
        std::size_t pos = 0;
        while (pos <= list.size()) {
            const std::size_t comma = list.find(',', pos);
            const std::string token =
                list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            if (!token.empty()) {
                layers.push_back(static_cast<std::uint32_t>(std::atoi(token.c_str())));
            }
            if (comma == std::string::npos) { break; }
            pos = comma + 1;
        }
    }
    if (layers.empty()) { layers = {15U, 16U}; }
    const int max_pages = [&] {
        const char* env = std::getenv("NINFER_KVDUMP_PAGES");
        return env != nullptr ? std::atoi(env) : 4;
    }();
    static std::atomic<std::uint32_t> counter{0};
    const std::uint32_t dump_id = counter.fetch_add(1);
    const std::string base =
        std::string(dump_dir) + "/kvc_" + std::to_string(dump_id) + "_t" + std::to_string(tokens);
    const Tensor tables = state.text_cache.execution_tables().matrix();
    kvdump_dump_tensor(stream, tables, 0, base + "_bt.bin");
    std::FILE* table_meta = std::fopen((base + "_bt_meta.txt").c_str(), "w");
    if (table_meta != nullptr) {
        std::fprintf(table_meta, "tokens=%d\n", tokens);
        kvdump_describe(table_meta, "block_tables", tables);
        std::fclose(table_meta);
    }
    for (const std::uint32_t layer : layers) {
        const PagedKVBatchLayerView view = state.text_cache.batch_layer_view(layer);
        const std::string prefix = base + "_L" + std::to_string(layer);
        kvdump_dump_tensor(stream, view.k_pages, max_pages, prefix + "_k.bin");
        kvdump_dump_tensor(stream, view.v_pages, max_pages, prefix + "_v.bin");
        kvdump_dump_tensor(stream, view.k_scale_pages, max_pages, prefix + "_ks.bin");
        kvdump_dump_tensor(stream, view.v_scale_pages, max_pages, prefix + "_vs.bin");
        kvdump_dump_tensor(stream, view.k_residual_pages, max_pages, prefix + "_kr.bin");
        kvdump_dump_tensor(stream, view.v_residual_pages, max_pages, prefix + "_vr.bin");
        std::FILE* meta = std::fopen((prefix + "_meta.txt").c_str(), "w");
        if (meta == nullptr) { continue; }
        std::fprintf(meta,
                     "layer=%u dtype=%d v_dtype=%d quant_group=%d v_quant_group=%d head_dim=%d "
                     "num_kv_heads=%d slot_bytes=%d cold_slot_bytes=%d layer_index=%d tokens=%d\n",
                     layer, static_cast<int>(view.dtype), static_cast<int>(view.v_dtype),
                     view.quant_group, view.v_quant_group, view.head_dim, view.num_kv_heads,
                     view.slot_bytes, view.cold_slot_bytes, view.layer_index, tokens);
        kvdump_describe(meta, "k", view.k_pages);
        kvdump_describe(meta, "v", view.v_pages);
        kvdump_describe(meta, "ks", view.k_scale_pages);
        kvdump_describe(meta, "vs", view.v_scale_pages);
        // Raw device addresses: two layers sharing bytes (e8 plane aliasing,
        // _TODO.md 96) show up as overlapping [data, data+bytes) ranges.
        std::fprintf(meta, "addr k=%p v=%p ks=%p vs=%p bytes k=%zu v=%zu ks=%zu vs=%zu\n",
                     view.k_pages.data, view.v_pages.data, view.k_scale_pages.data,
                     view.v_scale_pages.data, view.k_pages.bytes(), view.v_pages.bytes(),
                     view.k_scale_pages.bytes(), view.v_scale_pages.bytes());
        std::fclose(meta);
    }
}

PrefillChunkResult prefill_text_chunk(PrefillContext& state, std::span<const TokenId> ids,
                                      std::uint32_t nominal_length,
                                      std::optional<std::uint32_t> split_frontier,
                                      bool finalize_at_end) {
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_prefill_split_frontier(split_frontier ? static_cast<std::int64_t>(*split_frontier)
                                                   : -1);
    const std::span<const int> prompt(ids.data(), ids.size());
    PrefillChunkResult result;
    if (state.dflash != nullptr || state.dflash2 != nullptr) {
        DFlashFeatureSink sink = make_dflash_prefill_sink(state);
        result = card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end,
                                    sink);
    } else {
        result = card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end);
    }
    if (std::getenv("NINFER_HS_DUMP_DIR") != nullptr &&
        (state.dflash != nullptr || state.dflash2 != nullptr)) {
        dump_prefill_chunk(state, ids, state.text_kv_base,
                           static_cast<std::int32_t>(result.processed_tokens),
                           state.execution.device.stream);
    }
    dump_kv_cache(state, static_cast<std::int32_t>(result.processed_tokens),
                  state.execution.device.stream);
    return result;
}

PrefillChunkResult prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                                            VisionPrefillSession& vision,
                                            std::uint32_t nominal_length,
                                            std::optional<std::uint32_t> split_frontier,
                                            bool finalize_at_end) {
    if (state.dflash != nullptr || state.dflash2 != nullptr) {
        throw std::logic_error("DFlash staged multimodal prefill is unavailable");
    }
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_prefill_split_frontier(split_frontier ? static_cast<std::int64_t>(*split_frontier)
                                                   : -1);
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, vision, finalize_at_end);
}

void mtp_bridge_multimodal(PrefillContext& state, const PreparedPromptData& prompt,
                           VisionPrefillSession& vision, const MtpBridgeInput& bridge) {
    if (!state.mtp_kv.valid() || bridge.previous_hidden == nullptr || state.text_kv_base == 0 ||
        bridge.position < 0 ||
        static_cast<std::uint32_t>(bridge.position) + 1 != state.text_kv_base) {
        throw std::logic_error("multimodal MTP bridge does not match the reusable frontier");
    }

    Tensor bridge_token = state.execution.io.mtp->target_input_ids.slice(0, 0, 1);
    const TokenId token = prompt.token_ids[state.text_kv_base];
    CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token), cudaMemcpyHostToDevice,
                               state.execution.device.stream));

    Tensor visual_embedding;
    const Tensor* composed_embedding = nullptr;
    if (prompt.token_types[state.text_kv_base] != 0) {
        const VisionChunk chunk = vision.prepare_chunk(state.text_kv_base, 1);
        if (chunk.control == nullptr) {
            throw std::logic_error("visual MTP bridge has no encoded Vision item");
        }
        const auto& scatter = chunk.control->scatter_indices;
        const auto column   = std::lower_bound(scatter.begin(), scatter.end(),
                                               static_cast<std::int32_t>(state.text_kv_base));
        if (column == scatter.end() || *column != static_cast<std::int32_t>(state.text_kv_base) ||
            static_cast<std::uint8_t>(chunk.control->modality) !=
                prompt.token_types[state.text_kv_base]) {
            throw std::logic_error("visual MTP bridge does not match Vision scatter metadata");
        }
        visual_embedding =
            chunk.embeddings.slice(1, static_cast<std::int32_t>(column - scatter.begin()), 1);
        composed_embedding = &visual_embedding;
    }

    mtp_bridge_and_propose(state, bridge_token, *bridge.previous_hidden, bridge.position,
                           bridge.rope_position, false, composed_embedding);
}

void sample_from_hidden(PrefillContext& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose) {
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != TextConfig::hidden || hidden.ne[1] != 1 ||
        hidden.ne[2] != 1 || hidden.ne[3] != 1 || hidden.data == nullptr) {
        throw std::invalid_argument("sample_from_hidden requires BF16 [hidden,1]");
    }
    state.execution.work.reset();
    Tensor logits = state.execution.io.logits.slice(1, 0, 1);
    ops::linear(hidden, state.execution.model.output_head, logits, state.execution.device.stream);
    kCfg.apply_final_logit_policy(logits, state.execution.device.stream);
    CUDA_CHECK(cudaMemcpyAsync(state.execution.io.pos.data, &absolute_position,
                               sizeof(absolute_position), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    ops::sample(logits, state.execution.io.token, TextConfig::token_domain, state.sampling,
                state.execution.io.pos, purpose, state.execution.work,
                state.execution.device.stream);
    state.execution.work.reset();
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
