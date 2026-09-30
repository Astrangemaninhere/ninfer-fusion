#pragma once

// src/targets/spark_x2_5_4b/impl/load/bindings.h
//
// The artifact object plan and the runtime payload structs.  Shape copied from
// src/targets/muse_glimmer_30b/impl/load/bindings.h -- the sibling that is also
// "pure softmax, gdn = 0, no MTP, no draft" -- with the FOUR places Spark differs,
// each marked `SPARK:` below.
//
// THIS FILE IS WHERE THE ARTIFACT'S OBJECT NAMES ARE PINNED.  dl/sparkintel's
// REPORT section A5 named exactly this as open: "the object names in recipe.py are
// the engine's *roles*; the storage layout of this target's objects cannot be
// pinned until a target owns them".  This header plus impl/load/bindings.cpp is that
// ownership.  The consequence is a coupling the report states plainly: once these
// names land, tools/convert/spark_x2_5_4b/recipe.py's LEAF_ROLES table has to be
// re-pointed at them, and that is companion land (2) of three.

#include "targets/spark_x2_5_4b/impl/config.h"
#include <ninfer/targets/spark_x2_5_4b/package.h>
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

namespace ninfer::targets::spark_x2_5_4b::detail {

inline constexpr std::size_t kTextLayers          = 36;
// All 36, not 9.  This is the number the KV plan is sized by
// (decoder_state.cpp:164) and it must equal the number of times run_layers walks a
// layer; the sliding/full distinction is carried by TextConfig::layer_kind and, for
// now, by nothing else.  See config.h's is_full_attention() comment.
inline constexpr std::size_t kFullAttentionLayers = 36;
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
};

// SPARK (1 of 4): four INDEPENDENT projections.  The checkpoint ships one fused
// `self_attn.q_k_v_proj` of 6144 = 4096 + 1024 + 1024 rows and one separate
// `self_attn.g_proj` of 16 rows, and the engine's fused attn_input_proj op is a
// fixed-geometry op for the qwen shapes ([14336,5120] in, q/gate [6144,T], k/v
// [1024,T]; include/ninfer/ops/attn_input_proj.h:18-50), so Spark cannot use it.
// Same resolution muse took (muse_glimmer_30b/impl/load/bindings.h:51-56): split
// into independent ops::linear calls.  The converter is the half that splits the
// one source tensor into three objects.
struct SparkAttentionProjectionPlan {
    WeightPlan query;  // [query_size, hidden] = [4096, 2560]
    WeightPlan key;    // [kv_size, hidden]    = [1024, 2560]
    WeightPlan value;  // [kv_size, hidden]    = [1024, 2560]
    // SPARK (2 of 4): the headwise output gate.  Rows = query_heads = 16, NOT
    // query_size = 4096.  dl/sparkintel's `volume` gate caught this the hard way --
    // the first plan had it at [4096, 2560] and the derived parameter count came out
    // 376,012,800 = 36 * (4096 - 16) * 2560 too high against the index's own
    // `total_parameters`.  One scalar per head is the measured shape.
    WeightPlan gate;   // [attention_gate_rows, hidden] = [16, 2560]
};

struct FullAttentionPlan {
    SparkAttentionProjectionPlan projection;
    // SPARK (3 of 4): NO query_norm / key_norm objects, and not by omission --
    // the checkpoint has no q_norm/k_norm tensor in any of its 36 layers (the index's
    // 290 keys are exactly 8 leaves per layer plus embedding plus final norm).
    // This is where the muse precedent must NOT be copied: muse binds an all-ones
    // vector because its reference normalises q and k WITHOUT a scale, so the shared
    // rmsnorm path is numerically exact for it
    // (muse_glimmer_30b/impl/load/bindings.cpp:223-229).  Spark's reference does not
    // normalise q or k at all (modeling_spark.py goes straight from the projection to
    // rope), so an all-ones vector would apply a normalisation the checkpoint never
    // applies -- a silent wrong number, not a shortcut.  The plan therefore carries
    // no such object and `qk_norm_absent` records why the load path refuses.
    bool qk_norm_absent = true;
    WeightPlan output;  // [hidden, query_size] = [2560, 4096]
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

// Kept whole only because the shared RuntimeModelView is parameterised on a
// GdnProjectionPayload type; kGdnLayers is 0, so nothing below is ever bound.
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
    // SPARK (4 of 4): SINGLE-NORM, and the name of the one norm differs from muse's
    // by more than spelling.  The checkpoint's per-layer norms are
    // `input_layernorm` (before attention) and `post_attention_layernorm` (before the
    // MLP) -- there is no `post_attention_layernorm` applied to the ATTENTION OUTPUT
    // and no `post_feedforward_layernorm` at all (muse has both, plus a
    // pre_feedforward one; muse_glimmer_30b/impl/load/bindings.h:98-108, :236-252).
    // So Spark's pre-MLP norm is bound here as `post_attention_norm`, and the two
    // double-norm fields deliberately do not exist: TextConfig declares neither
    // attn_out_post_norm() nor mlp_out_post_norm(), which makes the family's
    // double_norm_attn_impl / double_norm_mlp_impl detection idioms compile-time
    // false (text_context.h:193-206) and the two leaves unreachable.
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
};

// Present only because BindingPlan carries it in the sibling targets; kGdnLayers and
// mtp_layers are 0, so no instance is ever populated.
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

struct BindingPlan {
    qwen3_6::FrontendResourcePlan frontend;
    qwen3_6::StartupFeatures features;

    WeightPlan token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    artifact::ObjectHandle final_norm;
    // The materialised copy of the embedding.  The converter's recipe.tie_decision()
    // is what puts it there; this plan reads it as an independent object, which is
    // what makes the tie irrelevant to the engine.
    WeightPlan output_head;
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

struct SparkAttentionProjectionPayload {
    Weight query;
    Weight key;
    // One scalar per head: the family's sigmoid_mul consumer is the headwise-scalar
    // route, and the payload carries the shape that selects it.
    Weight gate;
    Weight value;
};

using FullAttentionProjectionPayload = SparkAttentionProjectionPayload;

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
                       MtpAttentionPayload, DensePostMixerPayload,
                       qwen3_6::DFlashWeights<DFlashConfig::layers>,
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

} // namespace ninfer::targets::spark_x2_5_4b::detail
