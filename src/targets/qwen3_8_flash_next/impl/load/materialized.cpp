// Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
#include "targets/qwen3_8_flash_next/impl/load/materialized.h"

#include "artifact/typed_binding.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "targets/qwen3_8_flash_next/impl/load/quantize_nvfp4_expert_bank.h"
#include "targets/qwen3_8_flash_next/impl/load/quantize_output_head.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

std::vector<std::int32_t> load_flash_next_draft_shortlist(std::uint32_t K) {
    std::vector<std::int32_t> shortlist(K);
    const std::string i32_name = (K == 32'768) ? "shortlist_32k.i32" : ((K == 65'536) ? "shortlist_65k.i32" : "");
    std::vector<std::string> candidate_paths;
    if (const char* env_path = std::getenv("NINFER_RANKING_PATH"); env_path != nullptr && *env_path != '\0') {
        candidate_paths.emplace_back(env_path);
    }
    if (!i32_name.empty()) {
        candidate_paths.push_back("tools/freq_corpus/fixtures/ranking/" + i32_name);
        candidate_paths.push_back("../tools/freq_corpus/fixtures/ranking/" + i32_name);
        candidate_paths.push_back("E:/NInfer.gemini2/tools/freq_corpus/fixtures/ranking/" + i32_name);
    }
    candidate_paths.push_back("tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64");
    candidate_paths.push_back("../tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64");
    candidate_paths.push_back("E:/NInfer.gemini2/tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64");

    for (const auto& path_str : candidate_paths) {
        std::ifstream file(path_str, std::ios::binary);
        if (!file.is_open()) { continue; }

        file.seekg(0, std::ios::end);
        const auto file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (file_size >= static_cast<std::streamoff>(K * sizeof(std::int32_t)) &&
            path_str.ends_with(".i32")) {
            file.read(reinterpret_cast<char*>(shortlist.data()), K * sizeof(std::int32_t));
            if (file.gcount() == static_cast<std::streamsize>(K * sizeof(std::int32_t))) {
                return shortlist;
            }
        }

        constexpr std::size_t kVocab = 248'320;
        if (file_size >= static_cast<std::streamoff>(kVocab * sizeof(std::int64_t))) {
            std::vector<std::int64_t> counts(kVocab);
            file.read(reinterpret_cast<char*>(counts.data()), kVocab * sizeof(std::int64_t));
            if (file.gcount() == static_cast<std::streamsize>(kVocab * sizeof(std::int64_t))) {
                std::vector<std::int32_t> indices(kVocab);
                std::iota(indices.begin(), indices.end(), 0);
                std::stable_sort(indices.begin(), indices.end(), [&](std::int32_t a, std::int32_t b) {
                    return counts[a] > counts[b];
                });
                std::copy_n(indices.begin(), K, shortlist.begin());
                return shortlist;
            }
        }
    }

    std::iota(shortlist.begin(), shortlist.end(), 0);
    return shortlist;
}

using artifact::NumericFormat;

Tensor bf16_tensor(const artifact::MaterializedArtifact& backing, artifact::ObjectHandle handle,
                   std::initializer_list<std::int32_t> shape) {
    return artifact::materialized_tensor(backing, handle, NumericFormat::BF16, shape);
}

Weight bf16_weight(const artifact::MaterializedArtifact& backing, artifact::ObjectHandle handle,
                   std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(backing, handle, NumericFormat::BF16, rows, columns);
}

Weight fp8_weight(const artifact::MaterializedArtifact& backing, artifact::ObjectHandle handle,
                  std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(backing, handle, NumericFormat::FP8_E4M3FN_ROW_F32S, rows,
                                         columns);
}

// Host-mapped BF16 payload of a tensor bound with retain_mapped_tensor. The size check is the
// only guard against quantizing a differently shaped tensor: the mapping is raw file bytes.
std::span<const std::byte> mapped_bf16_rows(const artifact::MaterializedArtifact& backing,
                                            artifact::ObjectHandle handle, std::int32_t rows,
                                            std::int32_t columns) {
    const std::span<const std::byte> bytes = backing.mapped_tensor_bytes(handle);
    const std::size_t expected =
        static_cast<std::size_t>(rows) * columns * sizeof(std::uint16_t);
    if (bytes.size() != expected) {
        throw std::logic_error("Flash-Next mapped BF16 tensor has an unexpected payload size");
    }
    return bytes;
}

HyperConnectionWeights load_hyper(const HyperConnectionPlan& plan,
                                  const artifact::MaterializedArtifact& backing) {
    return {
        .block_inject   = bf16_weight(backing, plan.block_inject, 4, 10'240),
        .norm           = bf16_tensor(backing, plan.norm, {10'240}),
        .input_mix_down = bf16_weight(backing, plan.input_mix_down, 320, 10'240),
        .input_mix_up   = bf16_weight(backing, plan.input_mix_up, 10'240, 320),
    };
}

HyperMixerWeights load_mixer(const HyperMixerPlan& plan,
                             const artifact::MaterializedArtifact& backing) {
    return {
        .norm           = bf16_tensor(backing, plan.norm, {10'240}),
        .input_mix_down = bf16_weight(backing, plan.input_mix_down, 320, 10'240),
        .input_mix_up   = bf16_weight(backing, plan.input_mix_up, 10'240, 320),
    };
}

MoeWeights load_moe(const MoePlan& plan, const artifact::MaterializedArtifact& backing) {
    return {
        .router             = bf16_weight(backing, plan.router, 512, 2'560),
        .shared_down        = bf16_weight(backing, plan.shared_down, 2'560, 640),
        .shared_gate        = bf16_weight(backing, plan.shared_gate, 640, 2'560),
        .shared_up          = bf16_weight(backing, plan.shared_up, 640, 2'560),
        .shared_gate_weight = bf16_weight(backing, plan.shared_gate_weight, 1, 2'560),
        .expert_gate_up =
            materialized_nvfp4_expert_bank_view(backing, plan.expert_gate_up, 512, 1'280, 2'560),
        .expert_down =
            materialized_nvfp4_expert_bank_view(backing, plan.expert_down, 512, 2'560, 640),
    };
}

GdnWeights load_gdn(const GdnPlan& plan, const artifact::MaterializedArtifact& backing) {
    return {
        .a_log             = bf16_tensor(backing, plan.a_log, {48}),
        .convolution       = bf16_tensor(backing, plan.convolution, {10'240, 4}),
        .dt_bias           = bf16_tensor(backing, plan.dt_bias, {48}),
        .a_b_projection    = bf16_weight(backing, plan.a_b_projection, 96, 2'560),
        .norm              = bf16_tensor(backing, plan.norm, {128}),
        .query_key_value_z = fp8_weight(backing, plan.query_key_value_z, 16'384, 2'560),
        .output            = fp8_weight(backing, plan.output, 2'560, 6'144),
    };
}

AttentionWeights load_attention(const AttentionPlan& plan,
                                const artifact::MaterializedArtifact& backing) {
    return {
        .indexer_query_key    = bf16_weight(backing, plan.indexer_query_key, 640, 2'560),
        .indexer_key_norm     = bf16_tensor(backing, plan.indexer_key_norm, {128}),
        .indexer_query_norm   = bf16_tensor(backing, plan.indexer_query_norm, {128}),
        .key_norm             = bf16_tensor(backing, plan.key_norm, {256}),
        .query_norm           = bf16_tensor(backing, plan.query_norm, {256}),
        .query_gate_key_value = fp8_weight(backing, plan.query_gate_key_value, 13'312, 2'560),
        .output               = fp8_weight(backing, plan.output, 2'560, 6'144),
    };
}

PleWeights load_ple(const PlePlan& plan, const artifact::MaterializedArtifact& backing) {
    PleWeights out{
        .convolution      = bf16_tensor(backing, plan.convolution, {10'240, 4}),
        .key_projection   = bf16_weight(backing, plan.key_projection, 10'240, 2'560),
        .conv_norm        = bf16_tensor(backing, plan.conv_norm, {10'240}),
        .key_norm         = bf16_tensor(backing, plan.key_norm, {10'240}),
        .query_norm       = bf16_tensor(backing, plan.query_norm, {10'240}),
        .value_projection = bf16_weight(backing, plan.value_projection, 2'560, 2'560),
    };
    for (std::size_t shard = 0; shard < out.table.shards.size(); ++shard) {
        out.table.shards[shard] =
            make_ple_shard_view(backing.mapped_tensor_bytes(plan.shards[shard]));
    }
    return out;
}

VisionModelView load_vision(const VisionPlan& plan, const artifact::MaterializedArtifact& backing) {
    VisionModelView out;
    out.patch_embedding      = bf16_weight(backing, plan.patch_embedding, 1'152, 1'536);
    out.patch_embedding_bias = bf16_tensor(backing, plan.patch_embedding_bias, {1'152});
    out.position_embedding   = bf16_tensor(backing, plan.position_embedding, {2'304, 1'152});

    for (std::size_t layer = 0; layer < plan.layers.size(); ++layer) {
        const auto& src   = plan.layers[layer];
        out.layers[layer] = {
            .qkv          = bf16_weight(backing, src.qkv, 3'456, 1'152),
            .qkv_bias     = bf16_tensor(backing, src.qkv_bias, {3'456}),
            .output       = bf16_weight(backing, src.output, 1'152, 1'152),
            .output_bias  = bf16_tensor(backing, src.output_bias, {1'152}),
            .fc1          = bf16_weight(backing, src.fc1, 4'304, 1'152),
            .fc1_bias     = bf16_tensor(backing, src.fc1_bias, {4'304}),
            .fc2          = bf16_weight(backing, src.fc2, 1'152, 4'304),
            .fc2_bias     = bf16_tensor(backing, src.fc2_bias, {1'152}),
            .norm1_weight = bf16_tensor(backing, src.norm1_weight, {1'152}),
            .norm1_bias   = bf16_tensor(backing, src.norm1_bias, {1'152}),
            .norm2_weight = bf16_tensor(backing, src.norm2_weight, {1'152}),
            .norm2_bias   = bf16_tensor(backing, src.norm2_bias, {1'152}),
        };
    }

    out.merger_fc1         = bf16_weight(backing, plan.merger_fc1, 4'608, 4'608);
    out.merger_fc1_bias    = bf16_tensor(backing, plan.merger_fc1_bias, {4'608});
    out.merger_fc2         = bf16_weight(backing, plan.merger_fc2, 2'560, 4'608);
    out.merger_fc2_bias    = bf16_tensor(backing, plan.merger_fc2_bias, {2'560});
    out.merger_norm_weight = bf16_tensor(backing, plan.merger_norm_weight, {1'152});
    out.merger_norm_bias   = bf16_tensor(backing, plan.merger_norm_bias, {1'152});
    return out;
}

AttentionWeights load_mtp_attention(const AttentionPlan& plan,
                                    const artifact::MaterializedArtifact& backing) {
    return {
        .indexer_query_key    = bf16_weight(backing, plan.indexer_query_key, 640, 2'560),
        .indexer_key_norm     = bf16_tensor(backing, plan.indexer_key_norm, {128}),
        .indexer_query_norm   = bf16_tensor(backing, plan.indexer_query_norm, {128}),
        .key_norm             = bf16_tensor(backing, plan.key_norm, {256}),
        .query_norm           = bf16_tensor(backing, plan.query_norm, {256}),
        .query_gate_key_value = bf16_weight(backing, plan.query_gate_key_value, 13'312, 2'560),
        .output               = bf16_weight(backing, plan.output, 2'560, 6'144),
    };
}

MoeWeights load_mtp_moe_legacy(const MoePlan& plan, const artifact::MaterializedArtifact& backing,
                               void* expert_gate_up_nvfp4, void* expert_down_nvfp4) {
    const std::size_t gate_bytes = flash_next_nvfp4_expert_bank_payload_bytes(512, 1'280, 2'560);
    const std::size_t down_bytes = flash_next_nvfp4_expert_bank_payload_bytes(512, 2'560, 640);
    return {
        .router             = bf16_weight(backing, plan.router, 512, 2'560),
        .shared_down        = bf16_weight(backing, plan.shared_down, 2'560, 640),
        .shared_gate        = bf16_weight(backing, plan.shared_gate, 640, 2'560),
        .shared_up          = bf16_weight(backing, plan.shared_up, 640, 2'560),
        .shared_gate_weight = bf16_weight(backing, plan.shared_gate_weight, 1, 2'560),
        .expert_gate_up =
            make_nvfp4_expert_bank_view(expert_gate_up_nvfp4, gate_bytes, 512, 1'280, 2'560),
        .expert_down =
            make_nvfp4_expert_bank_view(expert_down_nvfp4, down_bytes, 512, 2'560, 640),
    };
}

MtpModelView load_mtp(const MtpPlan& plan, const artifact::MaterializedArtifact& backing,
                      void* expert_gate_up_nvfp4 = nullptr, void* expert_down_nvfp4 = nullptr) {
    return {
        .embedding_projection = bf16_weight(backing, plan.embedding_projection, 2'560, 2'560),
        .hidden_projection    = bf16_weight(backing, plan.hidden_projection, 2'560, 2'560),
        .mixer                = load_mixer(plan.mixer, backing),
        .attention_hyper      = load_hyper(plan.attention_hyper, backing),
        .attention            = load_mtp_attention(plan.attention, backing),
        .mlp_hyper            = load_hyper(plan.mlp_hyper, backing),
        .moe                  = plan.moe.experts_nvfp4
                                    ? load_moe(plan.moe, backing)
                                    : load_mtp_moe_legacy(plan.moe, backing, expert_gate_up_nvfp4,
                                                          expert_down_nvfp4),
        .embedding_norm       = bf16_tensor(backing, plan.embedding_norm, {2'560}),
        .hidden_norm          = bf16_tensor(backing, plan.hidden_norm, {10'240}),
    };
}

} // namespace

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized,
                                 bool quantize_output_head_fp8,
                                 bool quantize_token_embedding_fp8)
    : backing(std::move(materialized)) {
    frontend = qwen3_6::take_frontend_resources(backing, plan.frontend);

    text.weights_arena = &backing.device_arena();

    // plan.features carries the *binding* decision made in bind_artifact: when its FP8 flag is set
    // the BF16 tensor was retained in the file mapping and must be quantized from the host span.
    // The constructor arguments stay authoritative for StandaloneLoadedModel, which always binds
    // on the device and therefore quantizes device-to-device (no VRAM saving, same numerics).
    const bool embedding_mapped = plan.features.quantize_token_embedding_fp8;
    const bool head_mapped      = plan.features.quantize_output_head_fp8;

    if (embedding_mapped) {
        token_embedding_fp8 = DeviceBuffer(flash_next_fp8_head_payload_bytes(248'320, 2'560));
        quantize_bf16_rows_to_fp8_e4m3_row_f32s(
            mapped_bf16_rows(backing, plan.token_embedding, 248'320, 2'560).data(),
            token_embedding_fp8, text.token_embedding, 248'320, 2'560, cudaStream_t{});
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        text.token_embedding = bf16_weight(backing, plan.token_embedding, 248'320, 2'560);
        if (quantize_token_embedding_fp8) {
            token_embedding_fp8 = DeviceBuffer(flash_next_fp8_head_payload_bytes(248'320, 2'560));
            Weight fp8_embedding{};
            quantize_bf16_head_to_fp8_e4m3_row_f32s(text.token_embedding, token_embedding_fp8,
                                                    fp8_embedding, 248'320, 2'560, cudaStream_t{});
            CUDA_CHECK(cudaDeviceSynchronize());
            text.token_embedding = fp8_embedding;
        }
    }
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < plan.text_layers.size(); ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        TextLayerWeights& target    = text.layers[layer];
        target.attention_hyper      = load_hyper(source.attention_hyper, backing);
        target.moe                  = load_moe(source.moe, backing);
        target.mlp_hyper            = load_hyper(source.mlp_hyper, backing);
        if (source.is_full_attention) {
            text.full_attention.at(full_index++) = load_attention(source.attention, backing);
        } else {
            text.gdn.at(gdn_index++) = load_gdn(source.gdn, backing);
        }
    }
    if (full_index != text.full_attention.size() || gdn_index != text.gdn.size()) {
        throw std::logic_error("Flash-Next Text topology materialization is incomplete");
    }
    text.ple = load_ple(plan.ple, backing);

    // Mapped head: the BF16 rows stay in the file mapping, so both the FP8 head and the optimized
    // proposal head's shortlist rows are produced from `mapped_head` instead of a device view.
    std::span<const std::byte> mapped_head;
    Weight raw_bf16_head{};
    if (head_mapped) {
        mapped_head     = mapped_bf16_rows(backing, plan.output_head, 248'320, 2'560);
        output_head_fp8 = DeviceBuffer(flash_next_fp8_output_head_payload_bytes());
        quantize_bf16_rows_to_fp8_e4m3_row_f32s(mapped_head.data(), output_head_fp8,
                                                text.output_head, 248'320, 2'560, cudaStream_t{});
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        raw_bf16_head    = bf16_weight(backing, plan.output_head, 248'320, 2'560);
        text.output_head = raw_bf16_head;
        if (quantize_output_head_fp8) {
            output_head_fp8 = DeviceBuffer(flash_next_fp8_output_head_payload_bytes());
            Weight fp8_head{};
            quantize_bf16_output_head_to_fp8_e4m3_row_f32s(raw_bf16_head, output_head_fp8, fp8_head,
                                                           cudaStream_t{});
            CUDA_CHECK(cudaDeviceSynchronize());
            text.output_head = fp8_head;
        }
    }

    if (plan.features.proposal_head == ProposalHead::Optimized) {
        const std::uint32_t K =
            plan.features.draft_head_rows > 0 ? plan.features.draft_head_rows : 32'768;
        const auto shortlist_ids = load_flash_next_draft_shortlist(K);

        proposal_token_ids = DeviceBuffer(K * sizeof(std::int32_t));
        proposal_token_ids.copy_from_host(shortlist_ids.data(), K * sizeof(std::int32_t));

        proposal_head_payload =
            DeviceBuffer(static_cast<std::size_t>(K) * 2'560 * sizeof(std::uint16_t));
        Weight bf16_proposal_head{};
        if (head_mapped) {
            gather_head_rows_bf16_from_host(mapped_head, shortlist_ids.data(),
                                            static_cast<std::int32_t>(K), 248'320, 2'560,
                                            proposal_head_payload, bf16_proposal_head,
                                            cudaStream_t{});
        } else {
            gather_head_rows_bf16(raw_bf16_head,
                                  static_cast<const std::int32_t*>(proposal_token_ids.p),
                                  static_cast<std::int32_t>(K), 2'560, proposal_head_payload,
                                  bf16_proposal_head, cudaStream_t{});
        }

        Weight final_proposal_head = bf16_proposal_head;
        if (quantize_output_head_fp8 || head_mapped) {
            proposal_head_fp8 = DeviceBuffer(flash_next_fp8_head_payload_bytes(static_cast<std::int32_t>(K), 2'560));
            Weight fp8_proposal_head{};
            quantize_bf16_head_to_fp8_e4m3_row_f32s(bf16_proposal_head, proposal_head_fp8,
                                                    fp8_proposal_head, static_cast<std::int32_t>(K),
                                                    2'560, cudaStream_t{});
            final_proposal_head = fp8_proposal_head;
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        OptimizedProposalWeights prop{};
        prop.head      = final_proposal_head;
        prop.token_ids = Tensor(proposal_token_ids.p, DType::I32, {static_cast<std::int32_t>(K)});
        text.proposal  = prop;
    }

    text.final_mixer = load_mixer(plan.final_mixer, backing);

    if (plan.features.mtp) {
        if (plan.mtp.moe.experts_nvfp4) {
            text.mtp = load_mtp(plan.mtp, backing);
        } else {
            mtp_expert_gate_up_nvfp4 =
                DeviceBuffer(flash_next_nvfp4_expert_bank_payload_bytes(512, 1'280, 2'560));
            mtp_expert_down_nvfp4 =
                DeviceBuffer(flash_next_nvfp4_expert_bank_payload_bytes(512, 2'560, 640));

            const auto gate_up_bytes = backing.mapped_tensor_bytes(plan.mtp.moe.expert_gate_up);
            const auto down_bytes    = backing.mapped_tensor_bytes(plan.mtp.moe.expert_down);

            quantize_bf16_expert_bank_to_nvfp4(gate_up_bytes.data(), mtp_expert_gate_up_nvfp4.p, 512,
                                               1'280, 2'560, cudaStream_t{});
            quantize_bf16_expert_bank_to_nvfp4(down_bytes.data(), mtp_expert_down_nvfp4.p, 512, 2'560,
                                               640, cudaStream_t{});
            CUDA_CHECK(cudaDeviceSynchronize());

            text.mtp = load_mtp(plan.mtp, backing, mtp_expert_gate_up_nvfp4.p,
                                mtp_expert_down_nvfp4.p);
        }
    }
    if (plan.features.vision) { vision = load_vision(plan.vision, backing); }

    // PLE is a random-access host weight table. Leaving its mapped pages cold makes
    // the first code prompt pay thousands of synchronous disk faults per token batch.
    // Warm only these persistent host weights after device materialization, before ready.
    const auto warm_started = std::chrono::steady_clock::now();
    std::size_t host_bytes = 0;
    for (const auto& shard : text.ple.table.shards) {
        warm_readonly_host_memory(shard.codes);
        warm_readonly_host_memory(shard.scales);
        host_bytes += shard.codes.size() + shard.scales.size();
    }
    std::fprintf(stderr, "flash_next host_ple_warm bytes=%zu duration_ms=%.3f\n", host_bytes,
                 std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - warm_started).count());
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
