#include "targets/qwen3_5_9b/impl/load/bindings.h"

#include "artifact/typed_binding.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <variant>

// ---------------------------------------------------------------------------
// Object plan of the qwen3_5_9b artifact.
//
// Naming, layout, and the numeric-format vocabulary are the qwen3_6 family's; every
// shape is derived from `TextConfig`, and the layer kind comes from the family's own
// topology header (`qwen3_6::is_full_attention_layer`), so the loader and the runtime
// read one declaration of the layer schedule instead of two.
//
// The numeric formats are the ones the family's groupwise-int artifact is known to
// run with (Q4G64_F16S for the fused gate/up and the q/k input projections,
// Q5G64_F16S for the outputs and for the v/z half, Q6G64_F16S for the vocabulary
// matrices, BF16 for norms and the small GDN control/convolution tensors, FP32 for
// a_log/dt_bias).  They are NOT derived from the source GGUF's ggml types: the
// converter dequantises Q4_K/Q6_K/F32 to fp32 and re-encodes into this vocabulary,
// so an artifact of this package always declares exactly these formats and this
// loader can therefore require them by name.
// ---------------------------------------------------------------------------

namespace ninfer::targets::qwen3_5_9b::detail {
namespace {

using artifact::NumericFormat;

constexpr NumericFormat kEmbedFormat  = NumericFormat::Q6G64_F16S;
constexpr NumericFormat kHeadFormat   = NumericFormat::Q6G64_F16S;
constexpr NumericFormat kProjection4  = NumericFormat::Q4G64_F16S;
constexpr NumericFormat kProjection5  = NumericFormat::Q5G64_F16S;

// Layer topology is declared once, by the family's own topology header, and
// TextConfig::is_full_attention is the runtime's reading of that same declaration.
bool is_full_layer(std::size_t layer) {
    return qwen3_6::is_full_attention_layer(static_cast<std::int32_t>(layer));
}

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape) {
    return WeightPlan{.object = artifact::bind_device_tensor(binder, name, format, shape),
                      .format = format};
}

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns);
}

// A row view of a fused row-split weight.  The strides follow the row-split-k128-v1
// layout: one low plane (4 bits per element), an optional high plane (1 bit for Q5,
// 2 bits for Q6), and a per-group fp16 scale plane, each with its own row stride.
Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        block.layout != QuantLayout::RowSplit) {
        throw std::logic_error("invalid target row view");
    }
    const std::uint64_t groups     = static_cast<std::uint64_t>(block.padded_shape[1] / block.group);
    const std::uint64_t low_group  = 32;
    const std::uint64_t high_group = block.qtype == QType::Q5G64_F16S   ? 8
                                     : block.qtype == QType::Q6G64_F16S ? 16
                                                                        : 0;
    const std::uint64_t low_row   = groups * low_group;
    const std::uint64_t high_row  = groups * high_group;
    const std::uint64_t scale_row = groups * 2;
    Weight out                    = block;
    out.qdata                     = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * low_row;
    out.qhigh = high_group == 0 ? nullptr
                                : static_cast<const std::byte*>(block.qhigh) +
                                      static_cast<std::uint64_t>(row_begin) * high_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized) {
    DensePostMixerPayload out;
    out.gate_up = materialized_weight(materialized, plan.gate_up, 2 * TextConfig::intermediate,
                                      TextConfig::hidden);
    out.down = materialized_weight(materialized, plan.down, TextConfig::hidden,
                                   TextConfig::intermediate);
    return out;
}

FullAttentionProjectionPayload
load_attention_projection(const FullAttentionPlan& plan,
                          const artifact::MaterializedArtifact& materialized) {
    if (const auto* split = std::get_if<SplitAttentionProjectionPlan>(&plan.projection)) {
        return SplitAttentionProjectionPayload{
            .query_key = materialized_weight(materialized, split->query_key,
                                             TextConfig::query_size + TextConfig::kv_size,
                                             TextConfig::hidden),
            .gate_value = materialized_weight(materialized, split->gate_value,
                                              TextConfig::query_size + TextConfig::kv_size,
                                              TextConfig::hidden),
        };
    }
    const auto& fused = std::get<FusedAttentionProjectionPlan>(plan.projection);
    return FusedAttentionProjectionPayload{
        .query_key_gate_value =
            materialized_weight(materialized, fused.query_key_gate_value,
                                TextConfig::query_projection_rows + 2 * TextConfig::kv_size,
                                TextConfig::hidden),
    };
}

GdnInputProjectionPayload
load_gdn_input_projection(const GdnPlan& plan, const artifact::MaterializedArtifact& materialized) {
    if (const auto* split = std::get_if<SplitGdnInputProjectionPlan>(&plan.input_projection)) {
        return SplitGdnInputProjectionPayload{
            .query_key = materialized_weight(materialized, split->query_key, 2 * TextConfig::key_dim,
                                             TextConfig::hidden),
            .value_z = materialized_weight(materialized, split->value_z, 2 * TextConfig::value_dim,
                                           TextConfig::hidden),
        };
    }
    const auto& fused = std::get<FusedGdnInputProjectionPlan>(plan.input_projection);
    return FusedGdnInputProjectionPayload{
        .query_key_value_z =
            materialized_weight(materialized, fused.query_key_value_z,
                                2 * TextConfig::key_dim + 2 * TextConfig::value_dim,
                                TextConfig::hidden),
    };
}

GdnControlProjectionPayload
load_gdn_control_projection(const GdnPlan& plan,
                            const artifact::MaterializedArtifact& materialized) {
    if (const auto* split = std::get_if<SplitGdnControlProjectionPlan>(&plan.control_projection)) {
        return SplitGdnControlProjectionPayload{
            .a_projection = materialized_weight(materialized, split->a_projection,
                                                TextConfig::gdn_value_heads, TextConfig::hidden),
            .b_projection = materialized_weight(materialized, split->b_projection,
                                                TextConfig::gdn_value_heads, TextConfig::hidden),
        };
    }
    const auto& fused = std::get<FusedGdnControlProjectionPlan>(plan.control_projection);
    return FusedGdnControlProjectionPayload{
        .a_b_projection = materialized_weight(materialized, fused.a_b_projection,
                                              2 * TextConfig::gdn_value_heads, TextConfig::hidden),
    };
}

void bind_text_layers(artifact::Binder& binder, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16,
                                                                {TextConfig::hidden});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = SplitAttentionProjectionPlan{
                .query_key = bind_weight(binder, prefix + "attention/query_key", kProjection4,
                                         {TextConfig::query_size + TextConfig::kv_size,
                                          TextConfig::hidden}),
                .gate_value = bind_weight(binder, prefix + "attention/gate_value", kProjection5,
                                          {TextConfig::query_size + TextConfig::kv_size,
                                           TextConfig::hidden}),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16,
                {TextConfig::head_dim});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {TextConfig::head_dim});
            target.attention.output =
                bind_weight(binder, prefix + "attention/output", kProjection5,
                            {TextConfig::hidden, TextConfig::query_size});
        } else {
            target.gdn.a_log = artifact::bind_device_tensor(
                binder, prefix + "gdn/a_log", NumericFormat::FP32,
                {static_cast<std::uint64_t>(TextConfig::gdn_value_heads)});
            target.gdn.dt_bias = artifact::bind_device_tensor(
                binder, prefix + "gdn/dt_bias", NumericFormat::FP32,
                {static_cast<std::uint64_t>(TextConfig::gdn_value_heads)});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16,
                {static_cast<std::uint64_t>(TextConfig::gdn_conv_kernel),
                 static_cast<std::uint64_t>(TextConfig::convolution_dim)});
            target.gdn.control_projection = SplitGdnControlProjectionPlan{
                .a_projection = bind_weight(binder, prefix + "gdn/a_projection",
                                            NumericFormat::BF16,
                                            {TextConfig::gdn_value_heads, TextConfig::hidden}),
                .b_projection = bind_weight(binder, prefix + "gdn/b_projection",
                                            NumericFormat::BF16,
                                            {TextConfig::gdn_value_heads, TextConfig::hidden}),
            };
            target.gdn.input_projection = SplitGdnInputProjectionPlan{
                .query_key =
                    bind_weight(binder, prefix + "gdn/query_key", kProjection4,
                                {2 * TextConfig::key_dim, TextConfig::hidden}),
                .value_z = bind_weight(binder, prefix + "gdn/value_z", kProjection5,
                                       {2 * TextConfig::value_dim, TextConfig::hidden}),
            };
            target.gdn.norm = artifact::bind_device_tensor(
                binder, prefix + "gdn/norm", NumericFormat::BF16,
                {static_cast<std::uint64_t>(TextConfig::gdn_value_head_dim)});
            target.gdn.output = bind_weight(binder, prefix + "gdn/output", kProjection5,
                                            {TextConfig::hidden, TextConfig::value_dim});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {TextConfig::hidden});
        target.mlp.gate_up = bind_weight(binder, prefix + "mlp/gate_up", kProjection4,
                                         {2 * TextConfig::intermediate, TextConfig::hidden});
        target.mlp.down   = bind_weight(binder, prefix + "mlp/down", kProjection5,
                                        {TextConfig::hidden, TextConfig::intermediate});
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.frontend     = qwen3_6::bind_frontend_resources(binder);
    out.features     = features;

    out.token_embedding = bind_weight(binder, "text/token_embedding", kEmbedFormat,
                                      {TextConfig::output_rows, TextConfig::hidden});
    bind_text_layers(binder, out);
    out.final_norm = artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16,
                                                  {TextConfig::hidden});
    out.output_head = bind_weight(binder, "text/output_head", kHeadFormat,
                                  {TextConfig::output_rows, TextConfig::hidden});

    // The MTP block is the checkpoint's own nextn block, and WHETHER THE ARTIFACT HAS ONE is
    // a fact about the source GGUF, not about the run: a source with no
    // `nextn_predict_layers` has no `blk.<layers>.nextn.*` tensor, so its artifact carries no
    // `mtp/*` object at all, and requiring the 12 objects would refuse an artifact that is
    // complete for what it was written from.  The block is therefore bound from the artifact's
    // OWN declaration -- `Binder::find_tensor` reads the object table and consumes nothing:
    //   * all present: bound exactly as before, Device when the resolved run selects MTP and
    //     ValidateOnly otherwise (name, format and shape are still validated, so the artifact
    //     cannot quietly change its numeric formats or shapes under this binding);
    //   * none present: no handle, `out.mtp.declared == false`, and the plan records it;
    //   * some present: refused by name -- a half-written draft block is a broken artifact.
    // The identity's weights flavour is the second statement of the same fact
    // (`WeightsProfile::GgufKquant` vs `GgufKquantNoMtp`, the flavour
    // `resolved_auto_speculative` resolves `--spec auto` from), and the two are cross-checked
    // below so they cannot drift apart.
    const artifact::TensorPlacement mtp_placement =
        features.mtp() ? artifact::TensorPlacement::Device
                       : artifact::TensorPlacement::ValidateOnly;
    std::size_t mtp_found     = 0;
    std::size_t mtp_absent    = 0;
    std::string_view mtp_first_absent;
    const auto bind_mtp = [&](std::string_view name, NumericFormat format,
                              std::initializer_list<std::uint64_t> shape) {
        if (binder.find_tensor(name) == nullptr) {
            if (mtp_absent == 0) { mtp_first_absent = name; }
            ++mtp_absent;
            return artifact::ObjectHandle{};   // unused: `declared` stays false
        }
        ++mtp_found;
        return artifact::bind_tensor(binder, name, format, shape, mtp_placement);
    };
    out.mtp.input_projection = bind_mtp("mtp/input_projection", kProjection4,
                                        {TextConfig::hidden, TextConfig::mtp_input_rows});
    out.mtp.embedding_norm = bind_mtp("mtp/embedding_norm", NumericFormat::BF16,
                                      {TextConfig::hidden});
    out.mtp.hidden_norm =
        bind_mtp("mtp/hidden_norm", NumericFormat::BF16, {TextConfig::hidden});
    out.mtp.input_norm = bind_mtp("mtp/layer/input_norm", NumericFormat::BF16,
                                  {TextConfig::hidden});
    out.mtp.query_key_gate_value =
        bind_mtp("mtp/layer/attention/query_key_gate_value", kProjection4,
                 {TextConfig::mtp_attention_input_rows, TextConfig::hidden});
    out.mtp.query_norm =
        bind_mtp("mtp/layer/attention/query_norm", NumericFormat::BF16, {TextConfig::head_dim});
    out.mtp.key_norm =
        bind_mtp("mtp/layer/attention/key_norm", NumericFormat::BF16, {TextConfig::head_dim});
    out.mtp.output = bind_mtp("mtp/layer/attention/output", kProjection5,
                              {TextConfig::hidden, TextConfig::query_size});
    out.mtp.post_attention_norm =
        bind_mtp("mtp/layer/post_attention_norm", NumericFormat::BF16, {TextConfig::hidden});
    out.mtp.mlp.gate_up =
        WeightPlan{.object = bind_mtp("mtp/layer/mlp/gate_up", kProjection4,
                                      {TextConfig::mtp_mlp_gate_up_rows, TextConfig::hidden}),
                   .format = kProjection4};
    out.mtp.mlp.down = WeightPlan{.object = bind_mtp("mtp/layer/mlp/down", kProjection5,
                                                     {TextConfig::hidden,
                                                      TextConfig::intermediate}),
                                  .format = kProjection5};
    out.mtp.final_norm =
        bind_mtp("mtp/final_norm", NumericFormat::BF16, {TextConfig::hidden});

    if (mtp_found != 0 && mtp_absent != 0) {
        throw std::invalid_argument(
            "this artifact declares only part of its mtp draft block: " +
            std::to_string(mtp_found) + " of its objects are present and " +
            std::to_string(mtp_absent) + " are absent (first absent: " +
            std::string(mtp_first_absent) +
            ").  A draft block is all or nothing; a partial one is a converter that stopped "
            "part-way or two artifacts mixed together, and neither can be run.");
    }
    out.mtp.declared = mtp_found != 0;
    // The artifact states the same fact twice (its identity's weights flavour and its object
    // table).  Both are read here, so a disagreement is a red load rather than a run that
    // drafts from objects one of the two statements says are not there.
    const bool identity_declares_mtp = weights_profile == WeightsProfile::GgufKquant;
    if (out.mtp.declared != identity_declares_mtp) {
        throw std::invalid_argument(
            std::string("this artifact's identity says ") +
            (identity_declares_mtp ? "it has a draft block" : "it has no draft block") +
            " (weights flavour " +
            (identity_declares_mtp ? "gguf-kquant" : "gguf-kquant-nomtp") +
            ") but its object table says the opposite: " +
            (out.mtp.declared ? "the mtp/* objects are present" : "no mtp/* object is present") +
            ".  The identity and the object table are two statements of one fact; re-convert "
            "the artifact rather than trusting either one alone.");
    }
    if (features.mtp() && !out.mtp.declared) {
        throw std::invalid_argument(
            "this artifact declares no mtp draft block: its source GGUF had no "
            "nextn_predict_layers block, so there is nothing for the resolved --spec mtp to "
            "draft with.  Drop --spec mtp (or use --spec none); text generation is unaffected.");
    }

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(WeightsProfile weights_profile, BindingPlan plan,
                                 artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    frontend = qwen3_6::take_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    // W13: publish the artifact's weight-offload runtime to the layer-boundary hook.
    // Null unless a host budget was set, so the hook stays a no-op by default.
    runtime.backing.bind(backing.weight_residency());
    runtime.features      = plan.features;
    auto& token_embedding = runtime.token_embedding;
    auto& full_layers     = runtime.full_layers;
    auto& gdn_layers      = runtime.gdn_layers;
    auto& final_norm      = runtime.final_norm;
    auto& output_head     = runtime.output_head;

    token_embedding = materialized_weight(backing, plan.token_embedding, TextConfig::output_rows,
                                          TextConfig::hidden);
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm            = artifact::materialized_tensor(
                backing, source.input_norm, NumericFormat::BF16, {TextConfig::hidden});
            target.projection = load_attention_projection(source.attention, backing);
            target.query_norm = artifact::materialized_tensor(
                backing, source.attention.query_norm, NumericFormat::BF16, {TextConfig::head_dim});
            target.key_norm = artifact::materialized_tensor(
                backing, source.attention.key_norm, NumericFormat::BF16, {TextConfig::head_dim});
            target.output = materialized_weight(backing, source.attention.output, TextConfig::hidden,
                                                TextConfig::query_size);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
            target.post_mixer = load_mlp(source.mlp, backing);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm  = artifact::materialized_tensor(backing, source.input_norm,
                                                               NumericFormat::BF16,
                                                               {TextConfig::hidden});
            target.projection.a_log = artifact::materialized_tensor(
                backing, source.gdn.a_log, NumericFormat::FP32, {TextConfig::gdn_value_heads});
            target.projection.dt_bias = artifact::materialized_tensor(
                backing, source.gdn.dt_bias, NumericFormat::FP32, {TextConfig::gdn_value_heads});
            // The convolution is stored (kernel, channels) and consumed (channels, kernel),
            // matching the family's convention; the payload bytes are unchanged.
            target.convolution = artifact::materialized_tensor(
                backing, source.gdn.convolution, NumericFormat::BF16,
                {TextConfig::convolution_dim, TextConfig::gdn_conv_kernel});
            target.projection.control_projection = load_gdn_control_projection(source.gdn, backing);
            target.projection.input_projection   = load_gdn_input_projection(source.gdn, backing);
            target.norm = artifact::materialized_tensor(
                backing, source.gdn.norm, NumericFormat::BF16,
                {TextConfig::gdn_value_head_dim});
            target.output = materialized_weight(backing, source.gdn.output, TextConfig::hidden,
                                                TextConfig::value_dim);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
            target.post_mixer = load_mlp(source.mlp, backing);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("text topology binding is incomplete");
    }
    final_norm = artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16,
                                               {TextConfig::hidden});
    output_head = materialized_weight(backing, plan.output_head, TextConfig::output_rows,
                                      TextConfig::hidden);

    // Both halves of one fact, spelled out: `bind_artifact` refuses `features.mtp()` on an
    // artifact with no draft block, so a run that reaches here with a draft selected has the 12
    // objects in hand -- and the handles in `plan.mtp` are meaningless when they are not.
    if (plan.features.mtp() && plan.mtp.declared) {
        auto& mtp = runtime.mtp.emplace();
        mtp.input_projection =
            artifact::materialized_weight(backing, plan.mtp.input_projection, kProjection4,
                                          TextConfig::hidden, TextConfig::mtp_input_rows);
        mtp.embedding_norm   = artifact::materialized_tensor(
            backing, plan.mtp.embedding_norm, NumericFormat::BF16, {TextConfig::hidden});
        mtp.hidden_norm = artifact::materialized_tensor(backing, plan.mtp.hidden_norm,
                                                        NumericFormat::BF16, {TextConfig::hidden});
        mtp.input_norm  = artifact::materialized_tensor(backing, plan.mtp.input_norm,
                                                        NumericFormat::BF16, {TextConfig::hidden});
        mtp.attention.packed = artifact::materialized_weight(
            backing, plan.mtp.query_key_gate_value, kProjection4,
            TextConfig::mtp_attention_input_rows, TextConfig::hidden);
        mtp.attention.query =
            row_view(mtp.attention.packed, 0, TextConfig::query_size);
        mtp.attention.key =
            row_view(mtp.attention.packed, TextConfig::query_size, TextConfig::kv_size);
        mtp.attention.output_gate = row_view(mtp.attention.packed,
                                             TextConfig::query_size + TextConfig::kv_size,
                                             TextConfig::query_size);
        mtp.attention.value = row_view(
            mtp.attention.packed, TextConfig::query_projection_rows + TextConfig::kv_size,
            TextConfig::kv_size);
        mtp.query_norm = artifact::materialized_tensor(
            backing, plan.mtp.query_norm, NumericFormat::BF16, {TextConfig::head_dim});
        mtp.key_norm = artifact::materialized_tensor(backing, plan.mtp.key_norm,
                                                     NumericFormat::BF16, {TextConfig::head_dim});
        mtp.output = artifact::materialized_weight(backing, plan.mtp.output, kProjection5,
                                                   TextConfig::hidden, TextConfig::query_size);
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
        mtp.post_mixer = load_mlp(plan.mtp.mlp, backing);
        mtp.final_norm = artifact::materialized_tensor(backing, plan.mtp.final_norm,
                                                       NumericFormat::BF16, {TextConfig::hidden});
    }

    // No vision tower: this package binds none and plan_load refuses --vision by name,
    // so plan.features.vision is false whenever this constructor runs.
    (void)weights_profile;
}

} // namespace ninfer::targets::qwen3_5_9b::detail
