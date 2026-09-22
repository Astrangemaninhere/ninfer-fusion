// Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/ple_index.h"

#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

using artifact::NumericFormat;
using artifact::ObjectHandle;
using artifact::StorageLayout;

constexpr auto kBf16Layout   = StorageLayout::ContiguousLeV1;
constexpr auto kFp8Layout    = StorageLayout::RowScaleF32V1;
constexpr auto kExpertLayout = StorageLayout::ExpertBlockScaleK16M128x4V1;
constexpr auto kPleLayout    = StorageLayout::PackedU4G16V1;

ObjectHandle bind_device(artifact::Binder& binder, std::string_view name, NumericFormat format,
                         StorageLayout layout, std::initializer_list<std::uint64_t> shape) {
    const ObjectHandle handle = binder.require_tensor(name, format, layout, shape);
    binder.materialize_on_device(handle);
    return handle;
}

ObjectHandle bind_optional_device(artifact::Binder& binder, std::string_view name,
                                  NumericFormat format, StorageLayout layout,
                                  std::initializer_list<std::uint64_t> shape, bool enabled) {
    const ObjectHandle handle = binder.require_tensor(name, format, layout, shape);
    if (enabled) {
        binder.materialize_on_device(handle);
    } else {
        binder.validate_only(handle);
    }
    return handle;
}

ObjectHandle bind_mapped(artifact::Binder& binder, std::string_view name, NumericFormat format,
                         StorageLayout layout, std::initializer_list<std::uint64_t> shape) {
    const ObjectHandle handle = binder.require_tensor(name, format, layout, shape);
    binder.retain_mapped_tensor(handle);
    return handle;
}

template <std::size_t Size>
void require_i64_values(const artifact::Binder& binder, ObjectHandle handle,
                        const std::array<std::int64_t, Size>& expected, std::string_view label) {
    const std::span bytes = binder.payload(handle).data;
    if (bytes.size() != Size * sizeof(std::int64_t)) {
        throw artifact::ArtifactError(std::string(label) + ": invalid I64 payload size");
    }
    for (std::size_t index = 0; index < Size; ++index) {
        std::uint64_t bits = 0;
        for (std::size_t byte = 0; byte < sizeof(std::int64_t); ++byte) {
            bits |= static_cast<std::uint64_t>(
                        std::to_integer<std::uint8_t>(bytes[index * sizeof(std::int64_t) + byte]))
                    << (byte * 8U);
        }
        if (std::bit_cast<std::int64_t>(bits) != expected[index]) {
            throw artifact::ArtifactError(std::string(label) + ": value does not match target");
        }
    }
}


HyperConnectionPlan bind_hyper(artifact::Binder& binder, const std::string& prefix,
                               bool enabled = true) {
    const auto bind = [&](std::string_view suffix, std::initializer_list<std::uint64_t> shape) {
        return bind_optional_device(binder, prefix + std::string(suffix), NumericFormat::BF16,
                                    kBf16Layout, shape, enabled);
    };
    return {
        .block_inject   = bind("block_inject", {4, 10'240}),
        .norm           = bind("norm", {10'240}),
        .input_mix_down = bind("input_mix/down", {320, 10'240}),
        .input_mix_up   = bind("input_mix/up", {10'240, 320}),
    };
}

HyperMixerPlan bind_mixer(artifact::Binder& binder, const std::string& prefix,
                          bool enabled = true) {
    const auto bind = [&](std::string_view suffix, std::initializer_list<std::uint64_t> shape) {
        return bind_optional_device(binder, prefix + std::string(suffix), NumericFormat::BF16,
                                    kBf16Layout, shape, enabled);
    };
    return {
        .norm           = bind("norm", {10'240}),
        .input_mix_down = bind("input_mix/down", {320, 10'240}),
        .input_mix_up   = bind("input_mix/up", {10'240, 320}),
    };
}

artifact::ObjectHandle bind_optional_host(artifact::Binder& binder, std::string_view name,
                                          NumericFormat format, StorageLayout layout,
                                          std::initializer_list<std::uint64_t> shape, bool enabled) {
    // Every artifact object must be consumed by the target even when its feature is off,
    // exactly as bind_optional_device does; otherwise Binder::finish() rejects the artifact
    // ("artifact object was not consumed by the selected target"). Found in production
    // window 5 on the real artifact: sequence 12b returned early here and every serve failed.
    const artifact::ObjectHandle handle = binder.require_tensor(name, format, layout, shape);
    if (enabled) {
        binder.retain_mapped_tensor(handle); // tensors are file-mapped on the host (like the PLE shards); retain_on_host is for resources only
    } else {
        binder.validate_only(handle);
    }
    return handle;
}

MoePlan bind_moe(artifact::Binder& binder, const std::string& prefix, NumericFormat expert_format,
                 bool enabled = true, bool retain_experts_on_host = false) {
    const auto bf16 = [&](std::string_view suffix, std::initializer_list<std::uint64_t> shape) {
        return bind_optional_device(binder, prefix + std::string(suffix), NumericFormat::BF16,
                                    kBf16Layout, shape, enabled);
    };
    const StorageLayout expert_layout =
        expert_format == NumericFormat::NVFP4 ? kExpertLayout : kBf16Layout;
    const auto bind_expert = [&](std::string_view suffix, std::initializer_list<std::uint64_t> shape) {
        if (retain_experts_on_host) {
            return bind_optional_host(binder, prefix + std::string(suffix), expert_format,
                                      expert_layout, shape, enabled);
        }
        return bind_optional_device(binder, prefix + std::string(suffix), expert_format,
                                    expert_layout, shape, enabled);
    };
    return {
        .router             = bf16("router", {512, 2'560}),
        .shared_down        = bf16("shared_expert/down", {2'560, 640}),
        .shared_gate        = bf16("shared_expert/gate", {640, 2'560}),
        .shared_up          = bf16("shared_expert/up", {640, 2'560}),
        .shared_gate_weight = bf16("shared_expert_gate", {1, 2'560}),
        .expert_gate_up     = bind_expert("experts/gate_up", {512, 1'280, 2'560}),
        .expert_down        = bind_expert("experts/down", {512, 2'560, 640}),
        .experts_nvfp4      = (expert_format == NumericFormat::NVFP4),
    };
}

GdnPlan bind_gdn(artifact::Binder& binder, const std::string& prefix) {
    return {
        .a_log = bind_device(binder, prefix + "a_log", NumericFormat::BF16, kBf16Layout, {48}),
        .convolution = bind_device(binder, prefix + "convolution", NumericFormat::BF16, kBf16Layout,
                                   {4, 10'240}),
        .dt_bias = bind_device(binder, prefix + "dt_bias", NumericFormat::BF16, kBf16Layout, {48}),
        .a_b_projection = bind_device(binder, prefix + "a_b_projection", NumericFormat::BF16,
                                      kBf16Layout, {96, 2'560}),
        .norm = bind_device(binder, prefix + "norm", NumericFormat::BF16, kBf16Layout, {128}),
        .query_key_value_z =
            bind_device(binder, prefix + "query_key_value_z", NumericFormat::FP8_E4M3FN_ROW_F32S,
                        kFp8Layout, {16'384, 2'560}),
        .output = bind_device(binder, prefix + "output", NumericFormat::FP8_E4M3FN_ROW_F32S,
                              kFp8Layout, {2'560, 6'144}),
    };
}

AttentionPlan bind_attention(artifact::Binder& binder, const std::string& prefix,
                             NumericFormat projection_format, bool enabled = true) {
    const StorageLayout projection_layout =
        projection_format == NumericFormat::BF16 ? kBf16Layout : kFp8Layout;
    const auto bind = [&](std::string_view suffix, NumericFormat format, StorageLayout layout,
                          std::initializer_list<std::uint64_t> shape) {
        return bind_optional_device(binder, prefix + std::string(suffix), format, layout, shape,
                                    enabled);
    };
    return {
        .indexer_query_key =
            bind("indexer/query_key", NumericFormat::BF16, kBf16Layout, {640, 2'560}),
        .indexer_key_norm   = bind("indexer/key_norm", NumericFormat::BF16, kBf16Layout, {128}),
        .indexer_query_norm = bind("indexer/query_norm", NumericFormat::BF16, kBf16Layout, {128}),
        .key_norm           = bind("key_norm", NumericFormat::BF16, kBf16Layout, {256}),
        .query_norm         = bind("query_norm", NumericFormat::BF16, kBf16Layout, {256}),
        .query_gate_key_value =
            bind("query_gate_key_value", projection_format, projection_layout, {13'312, 2'560}),
        .output = bind("output", projection_format, projection_layout, {2'560, 6'144}),
    };
}

PlePlan bind_ple(artifact::Binder& binder) {
    const std::string prefix = "text/layers/1/ple/";
    PlePlan out{
        .convolution = bind_device(binder, prefix + "convolution", NumericFormat::BF16, kBf16Layout,
                                   {4, 10'240}),
        .key_projection = bind_device(binder, prefix + "key_projection", NumericFormat::BF16,
                                      kBf16Layout, {10'240, 2'560}),
        .conv_norm =
            bind_device(binder, prefix + "conv_norm", NumericFormat::BF16, kBf16Layout, {10'240}),
        .key_norm =
            bind_device(binder, prefix + "key_norm", NumericFormat::BF16, kBf16Layout, {10'240}),
        .query_norm =
            bind_device(binder, prefix + "query_norm", NumericFormat::BF16, kBf16Layout, {10'240}),
        .value_projection   = bind_device(binder, prefix + "value_projection", NumericFormat::BF16,
                                          kBf16Layout, {2'560, 2'560}),
        .layer_multipliers  = bind_mapped(binder, prefix + "embedding/layer_multipliers",
                                          NumericFormat::I64, kBf16Layout, {3}),
        .ngram_head_offsets = bind_mapped(binder, prefix + "embedding/ngram_head_offsets",
                                          NumericFormat::I64, kBf16Layout, {16}),
        .ngram_head_vocab_sizes = bind_mapped(binder, prefix + "embedding/ngram_head_vocab_sizes",
                                              NumericFormat::I64, kBf16Layout, {16}),
    };
    for (std::size_t shard = 0; shard < out.shards.size(); ++shard) {
        out.shards[shard] =
            bind_mapped(binder, prefix + "embedding/shards/" + std::to_string(shard),
                        NumericFormat::U4Z8G16_F16S, kPleLayout, {2'500'012, 160});
    }
    require_i64_values(binder, out.layer_multipliers, kPleLayerMultipliers, "PLE multipliers");
    require_i64_values(binder, out.ngram_head_offsets, kPleHeadOffsets, "PLE head offsets");
    require_i64_values(binder, out.ngram_head_vocab_sizes, kPleHeadVocabSizes,
                       "PLE head vocabularies");
    return out;
}

MtpPlan bind_mtp(artifact::Binder& binder, bool enabled) {
    MtpPlan out;
    out.embedding_projection =
        bind_optional_device(binder, "mtp/embedding_projection", NumericFormat::BF16, kBf16Layout,
                             {2'560, 2'560}, enabled);
    out.hidden_projection = bind_optional_device(
        binder, "mtp/hidden_projection", NumericFormat::BF16, kBf16Layout, {2'560, 2'560}, enabled);
    out.mixer           = bind_mixer(binder, "mtp/hyper_connection/", enabled);
    out.attention_hyper = bind_hyper(binder, "mtp/layer/attention/hyper_connection/", enabled);

    // Format-tolerant MTP MoE expert binding:
    // Support both new NVFP4 spliced artifacts and legacy BF16 artifacts seamlessly.
    NumericFormat mtp_expert_format = NumericFormat::NVFP4;
    bool retain_on_host             = false;
    const auto* gate_up_desc        = binder.reader().find("mtp/layer/mlp/experts/gate_up");
    if (gate_up_desc != nullptr) {
        if (const auto* tensor = std::get_if<artifact::TensorDescriptor>(gate_up_desc)) {
            if (tensor->format == artifact::NumericFormat::BF16) {
                mtp_expert_format = artifact::NumericFormat::BF16;
                retain_on_host    = true;
            }
        }
    }
    out.moe = bind_moe(binder, "mtp/layer/mlp/", mtp_expert_format, enabled, retain_on_host);

    out.mlp_hyper       = bind_hyper(binder, "mtp/layer/mlp/hyper_connection/", enabled);
    out.attention = bind_attention(binder, "mtp/layer/attention/", NumericFormat::BF16, enabled);
    out.embedding_norm = bind_optional_device(binder, "mtp/embedding_norm", NumericFormat::BF16,
                                              kBf16Layout, {2'560}, enabled);
    out.hidden_norm    = bind_optional_device(binder, "mtp/hidden_norm", NumericFormat::BF16,
                                              kBf16Layout, {10'240}, enabled);
    return out;
}

VisionPlan bind_vision(artifact::Binder& binder, bool enabled) {
    const auto bind = [&](const std::string& name, std::initializer_list<std::uint64_t> shape) {
        return bind_optional_device(binder, name, NumericFormat::BF16, kBf16Layout, shape, enabled);
    };
    VisionPlan out;
    out.patch_embedding      = bind("vision/patch_embedding", {1'152, 1'536});
    out.patch_embedding_bias = bind("vision/patch_embedding_bias", {1'152});
    out.position_embedding   = bind("vision/position_embedding", {2'304, 1'152});
    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
        out.layers[layer]        = {
                   .qkv          = bind(prefix + "attention/qkv", {3'456, 1'152}),
                   .qkv_bias     = bind(prefix + "attention/qkv_bias", {3'456}),
                   .output       = bind(prefix + "attention/output", {1'152, 1'152}),
                   .output_bias  = bind(prefix + "attention/output_bias", {1'152}),
                   .fc1          = bind(prefix + "mlp/fc1", {4'304, 1'152}),
                   .fc1_bias     = bind(prefix + "mlp/fc1_bias", {4'304}),
                   .fc2          = bind(prefix + "mlp/fc2", {1'152, 4'304}),
                   .fc2_bias     = bind(prefix + "mlp/fc2_bias", {1'152}),
                   .norm1_weight = bind(prefix + "norm1/weight", {1'152}),
                   .norm1_bias   = bind(prefix + "norm1/bias", {1'152}),
                   .norm2_weight = bind(prefix + "norm2/weight", {1'152}),
                   .norm2_bias   = bind(prefix + "norm2/bias", {1'152}),
        };
    }
    out.merger_fc1         = bind("vision/merger/fc1", {4'608, 4'608});
    out.merger_fc1_bias    = bind("vision/merger/fc1_bias", {4'608});
    out.merger_fc2         = bind("vision/merger/fc2", {2'560, 4'608});
    out.merger_fc2_bias    = bind("vision/merger/fc2_bias", {2'560});
    out.merger_norm_weight = bind("vision/merger/norm/weight", {1'152});
    out.merger_norm_bias   = bind("vision/merger/norm/bias", {1'152});
    return out;
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, LoadFeatures features) {
    ArtifactLoadPlan out;
    BindingPlan& plan    = out.bindings;
    plan.features        = features;
    plan.frontend        = qwen3_6::bind_frontend_resources(binder);
    // FP8 at load: keep the BF16 payload in the file mapping (retain_mapped_tensor adds zero
    // device bytes) and let LoadedModelData quantize from the host span. Uploading it first would
    // leave the BF16 copy pinned in the artifact's bump arena, which has no per-object free.
    plan.token_embedding =
        features.quantize_token_embedding_fp8
            ? bind_mapped(binder, "text/token_embedding", NumericFormat::BF16, kBf16Layout,
                          {248'320, 2'560})
            : bind_device(binder, "text/token_embedding", NumericFormat::BF16, kBf16Layout,
                          {248'320, 2'560});

    for (std::size_t layer = 0; layer < plan.text_layers.size(); ++layer) {
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        TextLayerPlan& target    = plan.text_layers[layer];
        target.attention_hyper   = bind_hyper(binder, prefix + "attention/hyper_connection/");
        target.moe               = bind_moe(binder, prefix + "mlp/", NumericFormat::NVFP4);
        target.mlp_hyper         = bind_hyper(binder, prefix + "mlp/hyper_connection/");
        target.is_full_attention = layer >= 3 && (layer - 3) % 4 == 0;
        if (target.is_full_attention) {
            target.attention =
                bind_attention(binder, prefix + "attention/", NumericFormat::FP8_E4M3FN_ROW_F32S);
        } else {
            target.gdn = bind_gdn(binder, prefix + "gdn/");
        }
    }

    plan.ple         = bind_ple(binder);
    plan.output_head = features.quantize_output_head_fp8
                           ? bind_mapped(binder, "text/output_head", NumericFormat::BF16,
                                         kBf16Layout, {248'320, 2'560})
                           : bind_device(binder, "text/output_head", NumericFormat::BF16,
                                         kBf16Layout, {248'320, 2'560});
    plan.final_mixer    = bind_mixer(binder, "text/hyper_connection/");
    plan.mtp            = bind_mtp(binder, features.mtp);
    plan.vision         = bind_vision(binder, features.vision);
    out.materialization = binder.finish();
    return out;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
