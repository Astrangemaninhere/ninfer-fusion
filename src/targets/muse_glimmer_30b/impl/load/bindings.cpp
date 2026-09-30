// ninfer::targets::muse_glimmer_30b - artifact binding (generated from the
// qwen3_6_27b template by tools/archkit; Muse layout: 52 softmax layers
// (13 full + 39 sliding, all bound as full for the acceptance phase), fused
// q|k|gate|v projections, double-norm layer graph, no draft backends).
#include "targets/muse_glimmer_30b/impl/load/bindings.h"

#include "artifact/typed_binding.h"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <span>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::targets::muse_glimmer_30b::detail {
namespace {

using artifact::NumericFormat;

constexpr NumericFormat kWeightFormat = NumericFormat::FP8_E4M3FN_ROW_BF16S;

std::uint32_t read_u32_le(std::span<const std::byte> bytes, std::uint64_t offset,
                          std::string_view label) {
    if (offset + 4 > bytes.size()) {
        throw std::runtime_error(std::string(label) + ": payload is too short");
    }
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        value |= static_cast<std::uint32_t>(bytes[offset + static_cast<std::size_t>(i)]) << (8 * i);
    }
    return value;
}

void require_positive_finite(std::uint32_t bits, std::string_view label) {
    const float value = [&] {
        float v;
        std::memcpy(&v, &bits, 4);
        return v;
    }();
    if (!(value > 0.0f) || !std::isfinite(value)) {
        throw std::runtime_error(std::string(label) + " divisor must be positive and finite");
    }
}

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape) {
    return WeightPlan{.object = artifact::bind_device_tensor(binder, name, format, shape),
                      .format = format};
}

WeightPlan bind_nvfp4_weight(artifact::Binder& binder, const std::string& name,
                                   std::uint64_t rows, std::uint64_t columns) {
    const std::array<std::uint64_t, 2> shape = {rows, columns};
    const artifact::ObjectHandle parent =
        binder.require_tensor(name, NumericFormat::NVFP4,
                              artifact::StorageLayout::BlockScaleK16M128x4V1, shape);
    binder.materialize_on_device(parent);
    const artifact::ObjectHandle input_divisor =
        artifact::bind_tensor(binder, name + "/input_scale_divisor", NumericFormat::FP32, {},
                              artifact::TensorPlacement::ValidateOnly);
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const std::uint32_t weight_bits =
        read_u32_le(binder.payload(parent).data, geometry.weight_divisor_offset, name);
    const std::uint32_t input_bits =
        read_u32_le(binder.payload(input_divisor).data, 0, name + " input divisor");
    require_positive_finite(weight_bits, name);
    require_positive_finite(input_bits, name + " input divisor");
    return WeightPlan{.object                    = parent,
                      .format                    = NumericFormat::NVFP4,
                      .weight_scale_divisor_bits = weight_bits,
                      .input_scale_divisor_bits  = input_bits};
}

// The MLP precision is the artifact's own declaration.  The converter derives it
// from the source tensor's dtype (tools/convert/muse_glimmer_30b/convert.py:
// `mlp_layer_fmt`) and the artifact stores exactly what it derived; the table of
// layer numbers that used to live here ("gate/up fp8 on 5..11, down fp8 on
// 1..12") was a second, independently maintained copy of that decision, and the
// loader and the artifact would disagree the moment either moved.  Reading the
// declared format keeps the registered pair {FP8_E4M3FN_ROW_BF16S, NVFP4} as the
// contract while letting the two sides agree by construction.
WeightPlan bind_mlp_weight(artifact::Binder& binder, const std::string& prefix,
                           const char* which, std::uint64_t rows, std::uint64_t columns) {
    const std::string name                      = prefix + which;
    const artifact::TensorDescriptor* declared  = binder.find_tensor(name);
    if (declared == nullptr) {
        throw artifact::ArtifactError("required artifact tensor is missing: " + name);
    }
    if (declared->format == NumericFormat::FP8_E4M3FN_ROW_BF16S) {
        return bind_weight(binder, name, NumericFormat::FP8_E4M3FN_ROW_BF16S, {rows, columns});
    }
    if (declared->format != NumericFormat::NVFP4) {
        throw artifact::ArtifactError(name + " is declared " +
                                      std::string(artifact::format_name(declared->format)) +
                                      ", but this projection is registered as FP8 or NVFP4");
    }
    // NVFP4 block-scale: the paired divisors are read OUT of the artifact, not
    // assumed.  This branch used to return `WeightPlan{.object = parent,
    // .format = NumericFormat::NVFP4}` with both divisor bit-fields left at their
    // default 0, and `materialized_weight` bit-casts those zeros into
    // Weight::weight_scale_divisor / input_scale_divisor -- which the shared
    // validator rejects: `validate_nvfp4_weight`
    // (src/ops/linear/nvfp4/nvfp4_format.cpp:61-62) requires both to be finite and
    // > 0.  Measured on models/muse_glimmer_30b_nvfp4.ninfer: every NVFP4 MLP weight
    // failed at prefill with `[nvfp4] invalid weight n=19968 k=6656 ... wsdiv=0
    // isdiv=0` / `nvfp4 linear: invalid NVFP4 weight`, even though the artifact
    // carried the right words (mlp/gate divisor 9252.1406 at payload offset
    // 74760192, input divisor 1.0).  Same extraction as bind_nvfp4_weight above,
    // which is why the head path never had this defect.
    const std::array<std::uint64_t, 2> shape = {rows, columns};
    const artifact::ObjectHandle parent =
        binder.require_tensor(name, NumericFormat::NVFP4,
                              artifact::StorageLayout::BlockScaleK16M128x4V1, shape);
    binder.materialize_on_device(parent);
    const artifact::ObjectHandle input_divisor =
        artifact::bind_tensor(binder, name + "/input_scale_divisor", NumericFormat::FP32, {},
                              artifact::TensorPlacement::ValidateOnly);
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const std::uint32_t weight_bits =
        read_u32_le(binder.payload(parent).data, geometry.weight_divisor_offset, name);
    const std::uint32_t input_bits =
        read_u32_le(binder.payload(input_divisor).data, 0, name + " input divisor");
    require_positive_finite(weight_bits, name);
    require_positive_finite(input_bits, name + " input divisor");
    return WeightPlan{.object                    = parent,
                      .format                    = NumericFormat::NVFP4,
                      .weight_scale_divisor_bits = weight_bits,
                      .input_scale_divisor_bits  = input_bits};
}

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    if (plan.format != NumericFormat::NVFP4) {
        return artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns);
    }
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(plan.object));
    if (bytes == nullptr) {
        throw std::logic_error("NVFP4 weight object was not materialized on device");
    }
    Weight out{};
    out.payload              = bytes;
    out.payload_bytes        = geometry.encoded_bytes;
    out.qtype                = QType::NVFP4;
    out.group_size           = 16;
    out.ndim                 = 2;
    out.qdata                = bytes;
    out.scales               = bytes + geometry.scale_plane_offset;
    out.n                    = rows;
    out.k                    = columns;
    out.group                = 16;
    out.layout               = QuantLayout::BlockScaleK16M128x4;
    out.scale_dtype          = DType::FP8_E4M3FN;
    out.shape[0]             = rows;
    out.shape[1]             = columns;
    out.padded_shape[0]      = rows;
    out.padded_shape[1]      = columns;
    out.weight_scale_divisor = std::bit_cast<float>(plan.weight_scale_divisor_bits);
    out.input_scale_divisor  = std::bit_cast<float>(plan.input_scale_divisor_bits);
    return out;
}


DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized) {
    DensePostMixerPayload out;
    out.gate = materialized_weight(materialized, plan.gate, TextConfig::intermediate,
                                   TextConfig::hidden);
    out.up   = materialized_weight(materialized, plan.up, TextConfig::intermediate,
                                   TextConfig::hidden);
    out.down = materialized_weight(materialized, plan.down, TextConfig::hidden,
                                   TextConfig::intermediate);
    return out;
}

FullAttentionProjectionPayload
load_attention_projection(const FullAttentionPlan& plan,
                          const artifact::MaterializedArtifact& materialized) {
    const auto& pr = plan.projection;
    return MuseAttentionProjectionPayload{
        .query = materialized_weight(materialized, pr.query, TextConfig::query_size,
                                     TextConfig::hidden),
        .key   = materialized_weight(materialized, pr.key, TextConfig::kv_size,
                                     TextConfig::hidden),
        .gate  = materialized_weight(materialized, pr.gate, TextConfig::query_size,
                                     TextConfig::hidden),
        .value = materialized_weight(materialized, pr.value, TextConfig::kv_size,
                                     TextConfig::hidden),
    };
}

void bind_text_layers(artifact::Binder& binder, BindingPlan& out) {
    // All 52 Muse layers are softmax attention; the 39 sliding layers run as
    // full for the acceptance phase (short contexts: window is not clipped).
    //
    // F1227 -- THE TWO AXES, NAMED, BECAUSE THE FLAG BELOW IS ONLY ONE OF THEM.
    // `is_full_attention` is the runtime's full-vs-GDN axis
    // (config.h:65-70, `layer_kind[layer] != 2`): it selects WHICH LEAF carries the layer.
    // Muse has no GDN at all (`kGdnScale == 0.0F`, config.h:140), so on this target the
    // axis is degenerate -- TRUE for all 52 -- and `is_full_attention(layer)` is the
    // honest spelling of that fact, not a per-layer decision.
    //
    // The WINDOW is the OTHER axis: `is_swa_attention` (config.h:71-73, 39 of 52) is what
    // the KV plan keys on (the `is_swa_attention` loop in
    // targets/qwen3_6/impl/runtime/layouts_impl.h), and it is where the declared 2048
    // lives. The two axes coinciding at or below 2048 TOKENS -- the acceptance-phase
    // regime -- is exactly what makes the all-full mapping exact, and it is now REPORTED
    // with its number by the plan (`[kv-regime]`, same file) instead of only asserted
    // here in prose. Above that number this file's mapping is a topology the model does
    // not have; see the guard below, which checks the axis it can see and names the layer.
    static_assert(TextConfig::swa_attention_layers() == 39,
                  "F1227: the acceptance-phase mapping is 'all 52 full + 39 declared "
                  "sliding'; if the sliding count moves, the regime number and this "
                  "file's comment moved with it");
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16,
                                                                {TextConfig::hidden});
        // F1227: read the target's own accessor instead of restating its answer. Today
        // this is TRUE on all 52 (the axis is degenerate, see above); reading it keeps
        // the binding and the declaration ONE fact, so a future kind-2 layer cannot be
        // bound full by a hardcode that nothing re-reads.
        target.is_full_attention = TextConfig::is_full_attention(static_cast<int>(layer));
        target.attention.projection = MuseAttentionProjectionPlan{
            .query = bind_weight(binder, prefix + "attention/query", kWeightFormat,
                                 {TextConfig::query_size, TextConfig::hidden}),
            .key   = bind_weight(binder, prefix + "attention/key", kWeightFormat,
                                 {TextConfig::kv_size, TextConfig::hidden}),
            .gate  = bind_weight(binder, prefix + "attention/gate", kWeightFormat,
                                 {TextConfig::query_size, TextConfig::hidden}),
            .value = bind_weight(binder, prefix + "attention/value", kWeightFormat,
                                 {TextConfig::kv_size, TextConfig::hidden}),
        };
        // Muse qk-norm has no scale parameters (with_scale=false): the
        // converter writes all-ones weight vectors so the shared rmsnorm path
        // stays uniform.
        target.attention.query_norm = artifact::bind_device_tensor(
            binder, prefix + "attention/query_norm", NumericFormat::BF16, {TextConfig::head_dim});
        target.attention.key_norm = artifact::bind_device_tensor(
            binder, prefix + "attention/key_norm", NumericFormat::BF16, {TextConfig::head_dim});
        target.attention.output =
            bind_weight(binder, prefix + "attention/output", kWeightFormat,
                        {TextConfig::hidden, TextConfig::query_size});
        // Double-norm layer graph: post_attention_layernorm (attention
        // output, before residual) and pre_feedforward_layernorm (MLP input)
        // and post_feedforward_layernorm (MLP output, before residual).
        target.post_attn_out_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_layernorm", NumericFormat::BF16,
            {TextConfig::hidden});
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "pre_feedforward_layernorm", NumericFormat::BF16,
            {TextConfig::hidden});
        target.mlp.gate =
            bind_mlp_weight(binder, prefix, "mlp/gate", TextConfig::intermediate,
                            TextConfig::hidden);
        target.mlp.up =
            bind_mlp_weight(binder, prefix, "mlp/up", TextConfig::intermediate, TextConfig::hidden);
        target.mlp.down =
            bind_mlp_weight(binder, prefix, "mlp/down", TextConfig::hidden,
                            TextConfig::intermediate);
        target.mlp.mlp_out_norm = artifact::bind_device_tensor(
            binder, prefix + "post_feedforward_layernorm", NumericFormat::BF16,
            {TextConfig::hidden});
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.frontend     = qwen3_6::bind_frontend_resources(binder);
    out.features     = features;

    out.token_embedding =
        bind_weight(binder, "text/token_embedding", kWeightFormat,
                    {TextConfig::output_rows, TextConfig::hidden});
    bind_text_layers(binder, out);
    out.final_norm = artifact::bind_device_tensor(binder, "text/final_norm",
                                                  NumericFormat::BF16, {TextConfig::hidden});
    // NVFP4 packed head with paired input divisor; the divisor bits live in the
    // payload tail and must be captured for runtime validation (same as MLP).
    out.output_head = bind_nvfp4_weight(binder, "text/output_head",
                                        static_cast<std::uint64_t>(TextConfig::output_rows),
                                        static_cast<std::uint64_t>(TextConfig::hidden));
    (void)weights_profile;
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
    auto& final_norm      = runtime.final_norm;
    auto& output_head     = runtime.output_head;

    token_embedding = materialized_weight(backing, plan.token_embedding, TextConfig::output_rows,
                                          TextConfig::hidden);
    std::size_t full_index = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (!source.is_full_attention) {
            // F1227: THIS GUARD CHECKS THE FULL-vs-GDN AXIS AND ONLY IT. It used to say
            // "must be all-full in the acceptance phase", which read as if the WINDOW were
            // being enforced here; it is not and cannot be -- the window is the other axis
            // (`TextConfig::is_swa_attention`, 39 layers) and it is carried by the KV plan,
            // whose `[kv-window]` / `[kv-regime]` lines report what is actually in force.
            // What this refuses is a layer routed to some leaf this runtime has no slot
            // for: `full_layers` is the ONLY binding table muse has (there is no swa leaf,
            // and `kGdnScale == 0.0F` means no GDN leaf either), so a non-full layer would
            // be silently DROPPED rather than bound. Named per layer instead of per file:
            throw std::logic_error(
                "muse text layer " + std::to_string(layer) +
                " is not on the full-attention axis, and this runtime binds only that leaf "
                "(no swa and no gdn binding table exists for this target). The acceptance-phase "
                "mapping is all 52 layers on the full leaf, with the declared 2048-token "
                "window carried by the KV plan; see the [kv-regime] report for the regime "
                "this mapping is exact in (F1227)");
        }
        FullAttentionWeights& target = full_layers.at(full_index++);
        target.input_norm            = artifact::materialized_tensor(
            backing, source.input_norm, NumericFormat::BF16, {TextConfig::hidden});
        target.projection = load_attention_projection(source.attention, backing);
        target.query_norm = artifact::materialized_tensor(
            backing, source.attention.query_norm, NumericFormat::BF16, {TextConfig::head_dim});
        target.key_norm = artifact::materialized_tensor(
            backing, source.attention.key_norm, NumericFormat::BF16, {TextConfig::head_dim});
        target.output = materialized_weight(backing, source.attention.output,
                                            TextConfig::hidden, TextConfig::query_size);
        target.post_attn_out_norm = artifact::materialized_tensor(
            backing, source.post_attn_out_norm, NumericFormat::BF16, {TextConfig::hidden});
        target.post_attention_norm = artifact::materialized_tensor(
            backing, source.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
        target.post_mlp_out_norm = artifact::materialized_tensor(
            backing, source.mlp.mlp_out_norm, NumericFormat::BF16, {TextConfig::hidden});
        target.post_mixer = load_mlp(source.mlp, backing);
    }
    if (full_index != full_layers.size()) {
        throw std::logic_error("text topology binding is incomplete");
    }
    final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16,
                                      {TextConfig::hidden});
    output_head = materialized_weight(backing, plan.output_head, TextConfig::output_rows,
                                      TextConfig::hidden);
    (void)weights_profile;
}

} // namespace ninfer::targets::muse_glimmer_30b::detail
