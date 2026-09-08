#pragma once

#include "targets/muse_glimmer_30b/impl/config.h"
#include <ninfer/targets/muse_glimmer_30b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/model_view.h>
#include <ninfer/targets/qwen3_6/startup_features.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>

namespace ninfer::targets::muse_glimmer_30b::detail {

inline constexpr std::size_t kTextLayers          = 52;
inline constexpr std::size_t kFullAttentionLayers = 52;
inline constexpr std::size_t kGdnLayers           = 0;

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format          = artifact::NumericFormat::BF16;
    std::uint32_t weight_scale_divisor_bits = 0;
    std::uint32_t input_scale_divisor_bits  = 0;
};

struct MlpPlan {
    WeightPlan gate;
    WeightPlan up;
    WeightPlan down;
    // Double-norm layer graph (Muse): post_feedforward_layernorm applied to
    // the MLP output before the residual add. Unused by single-norm plans.
    artifact::ObjectHandle mlp_out_norm;
};

struct SplitAttentionProjectionPlan {
    WeightPlan query_key;
    WeightPlan gate_value;
};

struct FusedAttentionProjectionPlan {
    WeightPlan query_key_gate_value;
};

struct MuseAttentionProjectionPlan {
    WeightPlan query;
    WeightPlan key;
    WeightPlan gate;
    WeightPlan value;
};

struct FullAttentionPlan {
    // Muse: 4 独立投影 (引擎 fp8 attn_input_proj 为 qwen 固定几何, 拆分走
    // 通用 linear 内核).
    MuseAttentionProjectionPlan projection;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan output;
};

struct SplitGdnInputProjectionPlan {
    WeightPlan query_key;
    WeightPlan value_z;
};

struct FusedGdnInputProjectionPlan {
    WeightPlan query_key_value_z;
};

struct SplitGdnControlProjectionPlan {
    WeightPlan a_projection;
    WeightPlan b_projection;
};

struct FusedGdnControlProjectionPlan {
    WeightPlan a_b_projection;
};

using GdnControlProjectionPlan =
    std::variant<SplitGdnControlProjectionPlan, FusedGdnControlProjectionPlan>;

struct GdnPlan {
    artifact::ObjectHandle a_log;
    artifact::ObjectHandle dt_bias;
    artifact::ObjectHandle convolution;
    GdnControlProjectionPlan control_projection;
    std::variant<SplitGdnInputProjectionPlan, FusedGdnInputProjectionPlan> input_projection;
    artifact::ObjectHandle norm;
    WeightPlan output;
};

struct TextLayerPlan {
    artifact::ObjectHandle input_norm;
    FullAttentionPlan attention{};
    GdnPlan gdn{};
    bool is_full_attention = false;
    // Double-norm layer graph (Muse): post_attention_layernorm on the
    // attention output before the residual add.
    artifact::ObjectHandle post_attn_out_norm;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
};

struct MtpPlan {
    artifact::ObjectHandle input_projection;
    artifact::ObjectHandle embedding_norm;
    artifact::ObjectHandle hidden_norm;
    artifact::ObjectHandle input_norm;
    artifact::ObjectHandle query_key_gate_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle output;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
    artifact::ObjectHandle final_norm;
};

struct DFlashLayerPlan {
    artifact::ObjectHandle input_norm;
    WeightPlan query_key_value;
    WeightPlan context_key;
    WeightPlan context_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan attention_output;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
};

struct DFlashPlan {
    WeightPlan feature_projection;
    artifact::ObjectHandle context_norm;
    std::array<DFlashLayerPlan, DFlashConfig::layers> layers;
    artifact::ObjectHandle final_norm;
    WeightPlan markov_w1;
    WeightPlan markov_w2;
};

struct DFlash2LayerPlan {
    artifact::ObjectHandle input_norm;
    WeightPlan query_key_value;
    WeightPlan context_key;
    WeightPlan context_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan attention_output;
    WeightPlan attention_conv_base;
    WeightPlan attention_conv_projection;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
    WeightPlan mlp_conv_base;
    WeightPlan mlp_conv_projection;
};

struct DFlash2Plan {
    WeightPlan feature_projection;
    artifact::ObjectHandle context_norm;
    std::array<DFlash2LayerPlan, DFlash2Config::layers> layers;
    artifact::ObjectHandle final_norm;
    WeightPlan selector_hidden_projection;
    WeightPlan selector_predecessor_codebook;
    WeightPlan selector_successor_codebook;
};

struct BindingPlan {
    qwen3_6::FrontendResourcePlan frontend;
    qwen3_6::StartupFeatures features;

    WeightPlan token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    artifact::ObjectHandle final_norm;
    WeightPlan output_head;
    artifact::ObjectHandle draft_head;
    artifact::ObjectHandle draft_head_token_ids;
    MtpPlan mtp;
    DFlashPlan dflash;
    DFlash2Plan dflash2;

    qwen3_6::VisionBackbonePlan vision_backbone;
    qwen3_6::VisionMergerInputPlan vision_merger_input;
    artifact::ObjectHandle vision_merger_fc2;
    artifact::ObjectHandle vision_merger_fc2_bias;
    qwen3_6::VisionMergerNormPlan vision_merger_norm;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features);

struct DensePostMixerPayload {
    Weight gate;
    Weight up;
    Weight down;
};

struct MuseAttentionProjectionPayload {
    Weight query;
    Weight key;
    Weight gate;
    Weight value;
};

using FullAttentionProjectionPayload = MuseAttentionProjectionPayload;

struct SplitGdnInputProjectionPayload {
    Weight query_key;
    Weight value_z;
};

struct FusedGdnInputProjectionPayload {
    Weight query_key_value_z;
};

using GdnInputProjectionPayload =
    std::variant<SplitGdnInputProjectionPayload, FusedGdnInputProjectionPayload>;

struct SplitGdnControlProjectionPayload {
    Weight a_projection;
    Weight b_projection;
};

struct FusedGdnControlProjectionPayload {
    Weight a_b_projection;
};

using GdnControlProjectionPayload =
    std::variant<SplitGdnControlProjectionPayload, FusedGdnControlProjectionPayload>;

struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
    GdnControlProjectionPayload control_projection;
    GdnInputProjectionPayload input_projection;
};

struct MtpAttentionPayload {
    Weight packed;
    Weight query;
    Weight key;
    Weight output_gate;
    Weight value;
};

using RuntimeModelView =
    qwen3_6::ModelView<FullAttentionProjectionPayload, GdnProjectionPayload, DensePostMixerPayload,
                       MtpAttentionPayload, DensePostMixerPayload, qwen3_6::DFlashWeights<DFlashConfig::layers>,
                       qwen3_6::DFlash2Weights<DFlash2Config::layers>, kFullAttentionLayers,
                       kGdnLayers>;
using FullAttentionWeights = RuntimeModelView::FullLayer;
using GdnWeights           = RuntimeModelView::GdnLayer;
using MtpWeights           = RuntimeModelView::MtpLayer;
using DFlashWeights        = RuntimeModelView::DFlash;
using DFlashLayerWeights   = qwen3_6::DFlashLayerWeights;
using DFlash2Weights       = RuntimeModelView::DFlash2;
using DFlash2LayerWeights  = qwen3_6::DFlash2LayerWeights;

class LoadedModelData {
public:
    LoadedModelData(WeightsProfile weights_profile, BindingPlan plan,
                    artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    qwen3_6::FrontendResources frontend;
    RuntimeModelView runtime;
};

class LoadedModel::Impl {
public:
    Impl(WeightsProfile weights_profile_in, BindingPlan plan,
         artifact::MaterializedArtifact materialized)
        : weights_profile(weights_profile_in),
          data(weights_profile_in, std::move(plan), std::move(materialized)) {}

    WeightsProfile weights_profile;
    LoadedModelData data;
};

} // namespace ninfer::targets::muse_glimmer_30b::detail
