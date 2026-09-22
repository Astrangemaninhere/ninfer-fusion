#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include "artifact/typed_binding.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

using artifact::NumericFormat;

// Layer topology is declared once, by the family's own topology header, and
// TextConfig::is_full_attention is the runtime's reading of that same
// declaration.  Spelling the interval and its offset out again here made the
// loader a second, independent copy of the rule: the loader and the runtime
// would bind different layer families the moment either copy moved.
bool is_full_layer(std::size_t layer) {
    return qwen3_6::is_full_attention_layer(static_cast<std::int32_t>(layer));
}

NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
        return NumericFormat::Q6G64_F16S;
    case WeightsProfile::Qwen38GroupwiseInt:
    case WeightsProfile::Qwen36Nvfp4:
        return NumericFormat::W8G32_F16S;
    case WeightsProfile::Qwen38Nvfp4:
    case WeightsProfile::Qwen38Nvfp4Dspark:
    case WeightsProfile::Qwen38Nvfp4DFlash2:
    case WeightsProfile::Qwen38Nvfp4ModelOpt:
        return NumericFormat::FP8_E4M3FN_ROW_BF16S;
    case WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head:
        return NumericFormat::BF16;
    }
    throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
}

std::uint32_t read_u32_le(std::span<const std::byte> bytes, std::uint64_t offset,
                          std::string_view label) {
    if (offset > bytes.size() || bytes.size() - static_cast<std::size_t>(offset) < 4) {
        throw artifact::ArtifactError(std::string(label) + ": FP32 word is outside payload");
    }
    const std::byte* value = bytes.data() + static_cast<std::size_t>(offset);
    return std::to_integer<std::uint32_t>(value[0]) |
           (std::to_integer<std::uint32_t>(value[1]) << 8U) |
           (std::to_integer<std::uint32_t>(value[2]) << 16U) |
           (std::to_integer<std::uint32_t>(value[3]) << 24U);
}

void require_positive_finite(std::uint32_t bits, std::string_view label) {
    const float value = std::bit_cast<float>(bits);
    if (!std::isfinite(value) || value <= 0.0F) {
        throw artifact::ArtifactError(std::string(label) + ": divisor must be finite and positive");
    }
}

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape) {
    if (format == NumericFormat::NVFP4) {
        throw std::logic_error("NVFP4 weight requires a paired input divisor");
    }
    return WeightPlan{.object = artifact::bind_device_tensor(binder, name, format, shape),
                      .format = format};
}

WeightPlan bind_nvfp4_weight(artifact::Binder& binder, std::string_view name, std::int32_t rows,
                             std::int32_t columns, std::string_view input_divisor_name) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::ObjectHandle parent      = binder.require_tensor(
        name, NumericFormat::NVFP4, artifact::StorageLayout::BlockScaleK16M128x4V1, shape);
    binder.materialize_on_device(parent);

    const artifact::ObjectHandle input_divisor =
        artifact::bind_tensor(binder, input_divisor_name, NumericFormat::FP32, {},
                              artifact::TensorPlacement::ValidateOnly);
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const std::uint32_t weight_bits =
        read_u32_le(binder.payload(parent).data, geometry.weight_divisor_offset, name);
    const std::uint32_t input_bits =
        read_u32_le(binder.payload(input_divisor).data, 0, input_divisor_name);
    require_positive_finite(weight_bits, name);
    require_positive_finite(input_bits, input_divisor_name);
    return WeightPlan{.object                    = parent,
                      .format                    = NumericFormat::NVFP4,
                      .weight_scale_divisor_bits = weight_bits,
                      .input_scale_divisor_bits  = input_bits};
}

// The format of a projection is the artifact's own declaration, not a function of the
// layer index.  The registered object plan of a mixed-precision source (NVFP4 below
// layer 56 of the family, FP8 above it) describes exactly one artifact; a later
// artifact of the same source declares the stored precision per object, and the two
// disagree layer by layer.  Reading the declaration keeps the NVFP4 contract intact -
// NVFP4 still has to arrive with its paired input divisor - while letting an object
// that is stored as FP8 bind as FP8.
NumericFormat declared_weight_format(const artifact::Binder& binder, std::string_view name) {
    const artifact::TensorDescriptor* tensor = binder.find_tensor(name);
    if (tensor == nullptr) {
        throw artifact::ArtifactError("required artifact tensor is missing: " +
                                      std::string(name));
    }
    return tensor->format;
}

WeightPlan bind_declared_weight(artifact::Binder& binder, std::string_view name,
                                std::string_view input_divisor_name,
                                std::initializer_list<std::uint64_t> shape) {
    const NumericFormat format = declared_weight_format(binder, name);
    if (format == NumericFormat::NVFP4) {
        if (shape.size() != 2) {
            throw artifact::ArtifactError(std::string(name) +
                                          ": an NVFP4 weight is a rank-two object");
        }
        const std::uint64_t* dims = shape.begin();
        return bind_nvfp4_weight(binder, name, static_cast<std::int32_t>(dims[0]),
                                 static_cast<std::int32_t>(dims[1]), input_divisor_name);
    }
    if (format != NumericFormat::FP8_E4M3FN_ROW_BF16S) {
        // The registered contract for this object is exactly these two formats.  A
        // declaration outside the pair is a different artifact, not a different
        // spelling of the registered one, so it is still refused.
        throw artifact::ArtifactError(std::string(name) + " is declared " +
                                      std::string(artifact::format_name(format)) +
                                      ", but this projection is registered as NVFP4 or "
                                      "FP8_E4M3FN_ROW_BF16S");
    }
    return bind_weight(binder, name, format, shape);
}

// The additive Qwen3.6-27B NVFP4 source stores three families of projection
// (attention input, attention output, GDN output) in two precisions: a
// full-precision BF16 copy for the layers its own declaration names, and an
// NVFP4 group with a paired input divisor for the rest.  The predicates that
// used to spell those layer numbers here were a second copy of that
// declaration, and only the artifact has to agree with itself, so read the
// precision off the object.  The registered pair stays the contract: nothing
// outside BF16/NVFP4 may bind through this path, and the paired input divisor
// is bound exactly where the parent is NVFP4, which is also where the object
// plan writes one.
WeightPlan bind_declared_direct_or_nvfp4_weight(artifact::Binder& binder, std::string_view name,
                                                std::string_view input_divisor_name,
                                                std::initializer_list<std::uint64_t> shape) {
    const NumericFormat format = declared_weight_format(binder, name);
    if (format == NumericFormat::NVFP4) {
        if (shape.size() != 2) {
            throw artifact::ArtifactError(std::string(name) +
                                          ": an NVFP4 weight is a rank-two object");
        }
        const std::uint64_t* dims = shape.begin();
        return bind_nvfp4_weight(binder, name, static_cast<std::int32_t>(dims[0]),
                                 static_cast<std::int32_t>(dims[1]), input_divisor_name);
    }
    if (format != NumericFormat::BF16) {
        throw artifact::ArtifactError(std::string(name) + " is declared " +
                                      std::string(artifact::format_name(format)) +
                                      ", but this projection is registered as BF16 or NVFP4");
    }
    return bind_weight(binder, name, format, shape);
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

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        block.layout != QuantLayout::RowSplit) {
        throw std::logic_error("invalid target row view");
    }
    const std::uint64_t groups    = static_cast<std::uint64_t>(block.padded_shape[1] / block.group);
    const std::uint64_t low_group = 32;
    const std::uint64_t high_group = block.qtype == QType::Q5G64_F16S   ? 8
                                     : block.qtype == QType::Q6G64_F16S ? 16
                                                                        : 0;
    const std::uint64_t low_row    = groups * low_group;
    const std::uint64_t high_row   = groups * high_group;
    const std::uint64_t scale_row  = groups * 2;
    Weight out                     = block;
    out.qdata                      = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * low_row;
    out.qhigh  = high_group == 0 ? nullptr
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
    out.gate_up = materialized_weight(materialized, plan.gate_up, 34816, 5120);
    out.down    = materialized_weight(materialized, plan.down, 5120, 17408);
    return out;
}

FullAttentionProjectionPayload
load_attention_projection(const FullAttentionPlan& plan,
                          const artifact::MaterializedArtifact& materialized) {
    if (const auto* split = std::get_if<SplitAttentionProjectionPlan>(&plan.projection)) {
        return SplitAttentionProjectionPayload{
            .query_key  = materialized_weight(materialized, split->query_key, 7168, 5120),
            .gate_value = materialized_weight(materialized, split->gate_value, 7168, 5120),
        };
    }
    const auto& fused = std::get<FusedAttentionProjectionPlan>(plan.projection);
    return FusedAttentionProjectionPayload{
        .query_key_gate_value =
            materialized_weight(materialized, fused.query_key_gate_value, 14336, 5120),
    };
}

GdnInputProjectionPayload
load_gdn_input_projection(const GdnPlan& plan, const artifact::MaterializedArtifact& materialized) {
    if (const auto* split = std::get_if<SplitGdnInputProjectionPlan>(&plan.input_projection)) {
        return SplitGdnInputProjectionPayload{
            .query_key = materialized_weight(materialized, split->query_key, 4096, 5120),
            .value_z   = materialized_weight(materialized, split->value_z, 12288, 5120),
        };
    }
    const auto& fused = std::get<FusedGdnInputProjectionPlan>(plan.input_projection);
    return FusedGdnInputProjectionPayload{
        .query_key_value_z =
            materialized_weight(materialized, fused.query_key_value_z, 16384, 5120),
    };
}

GdnControlProjectionPayload
load_gdn_control_projection(const GdnPlan& plan,
                            const artifact::MaterializedArtifact& materialized) {
    if (const auto* split = std::get_if<SplitGdnControlProjectionPlan>(&plan.control_projection)) {
        return SplitGdnControlProjectionPayload{
            .a_projection = materialized_weight(materialized, split->a_projection, 48, 5120),
            .b_projection = materialized_weight(materialized, split->b_projection, 48, 5120),
        };
    }
    const auto& fused = std::get<FusedGdnControlProjectionPlan>(plan.control_projection);
    return FusedGdnControlProjectionPayload{
        .a_b_projection = materialized_weight(materialized, fused.a_b_projection, 96, 5120),
    };
}

void bind_groupwise_text_layers(artifact::Binder& binder, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = SplitAttentionProjectionPlan{
                .query_key  = bind_weight(binder, prefix + "attention/query_key",
                                          NumericFormat::Q4G64_F16S, {7168, 5120}),
                .gate_value = bind_weight(binder, prefix + "attention/gate_value",
                                          NumericFormat::Q5G64_F16S, {7168, 5120}),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output = bind_weight(binder, prefix + "attention/output",
                                                  NumericFormat::Q5G64_F16S, {5120, 6144});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {48});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {48});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240});
            target.gdn.control_projection = SplitGdnControlProjectionPlan{
                .a_projection = bind_weight(binder, prefix + "gdn/a_projection",
                                            NumericFormat::BF16, {48, 5120}),
                .b_projection = bind_weight(binder, prefix + "gdn/b_projection",
                                            NumericFormat::BF16, {48, 5120}),
            };
            target.gdn.input_projection = SplitGdnInputProjectionPlan{
                .query_key = bind_weight(binder, prefix + "gdn/query_key",
                                         NumericFormat::Q4G64_F16S, {4096, 5120}),
                .value_z   = bind_weight(binder, prefix + "gdn/value_z", NumericFormat::Q5G64_F16S,
                                         {12288, 5120}),
            };
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {128});
            target.gdn.output =
                bind_weight(binder, prefix + "gdn/output", NumericFormat::Q5G64_F16S, {5120, 6144});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        target.mlp.gate_up =
            bind_weight(binder, prefix + "mlp/gate_up", NumericFormat::Q4G64_F16S, {34816, 5120});
        target.mlp.down =
            bind_weight(binder, prefix + "mlp/down", NumericFormat::Q5G64_F16S, {5120, 17408});
    }
}

void bind_nvfp4_text_layers(artifact::Binder& binder, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = FusedAttentionProjectionPlan{
                .query_key_gate_value = bind_declared_direct_or_nvfp4_weight(
                    binder, prefix + "attention/query_key_gate_value",
                    prefix + "attention/input_projection/input_scale_divisor", {14336, 5120}),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output = bind_declared_direct_or_nvfp4_weight(
                binder, prefix + "attention/output",
                prefix + "attention/output_projection/input_scale_divisor", {5120, 6144});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {48});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {48});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240});
            target.gdn.control_projection = SplitGdnControlProjectionPlan{
                .a_projection = bind_weight(binder, prefix + "gdn/a_projection",
                                            NumericFormat::BF16, {48, 5120}),
                .b_projection = bind_weight(binder, prefix + "gdn/b_projection",
                                            NumericFormat::BF16, {48, 5120}),
            };
            target.gdn.input_projection = FusedGdnInputProjectionPlan{
                .query_key_value_z =
                    bind_nvfp4_weight(binder, prefix + "gdn/query_key_value_z", 16384, 5120,
                                      prefix + "gdn/input_projection/input_scale_divisor"),
            };
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {128});
            target.gdn.output = bind_declared_direct_or_nvfp4_weight(
                binder, prefix + "gdn/output", prefix + "gdn/output_projection/input_scale_divisor",
                {5120, 6144});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        target.mlp.gate_up =
            bind_nvfp4_weight(binder, prefix + "mlp/gate_up", 34816, 5120,
                              prefix + "mlp/gate_up_projection/input_scale_divisor");
        target.mlp.down = bind_nvfp4_weight(binder, prefix + "mlp/down", 5120, 17408,
                                            prefix + "mlp/down_projection/input_scale_divisor");
    }
}

void bind_qwen38_nvfp4_text_layers(artifact::Binder& binder, BindingPlan& out) {
    constexpr NumericFormat kFp8 = NumericFormat::FP8_E4M3FN_ROW_BF16S;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = FusedAttentionProjectionPlan{
                .query_key_gate_value = bind_weight(
                    binder, prefix + "attention/query_key_gate_value", kFp8, {14336, 5120}),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output =
                bind_weight(binder, prefix + "attention/output", kFp8, {5120, 6144});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {48});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {48});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240});
            target.gdn.control_projection = FusedGdnControlProjectionPlan{
                .a_b_projection = bind_weight(binder, prefix + "gdn/a_b_projection",
                                              NumericFormat::BF16, {96, 5120}),
            };
            target.gdn.input_projection = FusedGdnInputProjectionPlan{
                .query_key_value_z =
                    bind_weight(binder, prefix + "gdn/query_key_value_z", kFp8, {16384, 5120}),
            };
            target.gdn.norm   = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                             NumericFormat::BF16, {128});
            target.gdn.output = bind_weight(binder, prefix + "gdn/output", kFp8, {5120, 6144});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        // gate_up and down carry their own declared precision, and they do not agree
        // with each other: a layer whose down is stored FP8 can have an NVFP4 gate_up.
        // The paired input divisor is bound only where the parent is NVFP4, which is
        // also where the object plan writes one.
        target.mlp.gate_up =
            bind_declared_weight(binder, prefix + "mlp/gate_up",
                                 prefix + "mlp/gate_up_projection/input_scale_divisor",
                                 {34816, 5120});
        target.mlp.down = bind_declared_weight(binder, prefix + "mlp/down",
                                               prefix + "mlp/down_projection/input_scale_divisor",
                                               {5120, 17408});
    }
}

void validate_draft_ids(const artifact::Binder& binder, artifact::ObjectHandle handle) {
    constexpr std::size_t kDraftVocab     = 131072;
    constexpr std::size_t kTokenizerVocab = 248077;
    const auto bytes                      = binder.payload(handle).data;
    std::vector<bool> seen(kTokenizerVocab, false);
    for (std::size_t i = 0; i < kDraftVocab; ++i) {
        const std::byte* value = bytes.data() + i * sizeof(std::uint32_t);
        const std::uint32_t id = std::to_integer<std::uint32_t>(value[0]) |
                                 (std::to_integer<std::uint32_t>(value[1]) << 8U) |
                                 (std::to_integer<std::uint32_t>(value[2]) << 16U) |
                                 (std::to_integer<std::uint32_t>(value[3]) << 24U);
        if (id >= kTokenizerVocab) {
            throw artifact::ArtifactError("draft-head token id is outside tokenizer domain");
        }
        if (seen[id]) { throw artifact::ArtifactError("draft-head token ids are not unique"); }
        seen[id] = true;
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.frontend     = qwen3_6::bind_frontend_resources(binder);
    out.features     = features;

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);
    out.token_embedding =
        bind_weight(binder, "text/token_embedding", vocabulary_format, {248320, 5120});
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        bind_groupwise_text_layers(binder, out);
        break;
    case WeightsProfile::Qwen36Nvfp4:
        bind_nvfp4_text_layers(binder, out);
        break;
    case WeightsProfile::Qwen38Nvfp4:
        bind_qwen38_nvfp4_text_layers(binder, out);
        break;
    case WeightsProfile::Qwen38Nvfp4Dspark:
    case WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head:
    case WeightsProfile::Qwen38Nvfp4DFlash2:
    case WeightsProfile::Qwen38Nvfp4ModelOpt:
        bind_qwen38_nvfp4_text_layers(binder, out);
        break;
    default:
        throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
    }
    out.final_norm =
        artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16, {5120});
    out.output_head = bind_weight(binder, "text/output_head", vocabulary_format, {248320, 5120});
    // ---- the OPTIONAL groups -----------------------------------------------------------
    // Whether an artifact carries a draft head, an MTP block and a vision tower is a fact about
    // the RECIPE that wrote it, not about the weights flavour.  The registered `groupwise-int`
    // flavour has two products on this box: the full closure (1124 objects,
    // /home/user/models/qwen3_8_27b_nvfp4.ninfer) and the text-core recipe's 777 objects
    // (/var/tmp/target27/full_q1.ninfer), which declares no mtp/*, no text/draft_head* and no
    // vision/* object at all.  Requiring all of them of every artifact of the flavour is a
    // requirement the text-core product cannot answer, and it was the reason it could not load.
    //
    // Each group is therefore read from the artifact's OWN object table (`Binder::find_tensor`
    // reads the directory and consumes nothing):
    //   * every member present: bound exactly as before -- Device when the resolved run selects
    //     it, ValidateOnly otherwise, with name, format, shape and layout still checked by
    //     `Binder::require_tensor`;
    //   * none present: skipped, and the plan records `*_declared == false`;
    //   * some present: refused by name -- a half-written group is a broken artifact.
    // A run that SELECTS a group the artifact does not declare is refused by name below, at the
    // one place where both facts are in hand.
    struct GroupPresence {
        std::size_t present = 0;
        std::size_t absent  = 0;
        std::string_view first_absent;
        [[nodiscard]] bool complete() const noexcept { return absent == 0; }
        [[nodiscard]] bool partial() const noexcept { return present != 0 && absent != 0; }
        void note(std::string_view name, bool found) {
            if (found) {
                ++present;
                return;
            }
            if (absent == 0) { first_absent = name; }
            ++absent;
        }
    };
    const auto declares = [&](std::string_view name) {
        return binder.find_tensor(name) != nullptr;
    };

    // ---- draft head (2 objects) ----
    GroupPresence draft_head_group;
    draft_head_group.note("text/draft_head", declares("text/draft_head"));
    draft_head_group.note("text/draft_head_token_ids", declares("text/draft_head_token_ids"));
    if (draft_head_group.partial()) {
        throw std::invalid_argument(
            "this artifact declares only part of its draft head: " +
            std::to_string(draft_head_group.present) + " of its 2 objects are present and " +
            std::to_string(draft_head_group.absent) + " are absent (first absent: " +
            std::string(draft_head_group.first_absent) +
            ").  A draft head is all or nothing: fix the converter, do not trust half of it.");
    }
    out.draft_head_declared = draft_head_group.complete();
    if (out.draft_head_declared) {
        const artifact::TensorPlacement proposal_placement =
            features.optimized_proposal() ? artifact::TensorPlacement::Device
                                          : artifact::TensorPlacement::ValidateOnly;
        out.draft_head = artifact::bind_tensor(binder, "text/draft_head",
                                               NumericFormat::Q4G64_F16S, {131072, 5120},
                                               proposal_placement);
        out.draft_head_token_ids = artifact::bind_tensor(
            binder, "text/draft_head_token_ids", NumericFormat::I32, {131072},
            proposal_placement);
        validate_draft_ids(binder, out.draft_head_token_ids);
    } else if (features.optimized_proposal()) {
        throw std::invalid_argument(
            "this artifact declares no draft head (neither text/draft_head nor "
            "text/draft_head_token_ids is in it): its recipe wrote the text core only, so the "
            "resolved optimized proposal head has nothing to propose with.  Drop --lm-head-draft "
            "(or whatever spelling selected that head); text generation is unaffected.");
    }

    const artifact::TensorPlacement mtp_placement = features.mtp()
                                                        ? artifact::TensorPlacement::Device
                                                        : artifact::TensorPlacement::ValidateOnly;
    GroupPresence mtp_group;
    const auto bind_mtp                           = [&](std::string_view name, NumericFormat format,
                              std::initializer_list<std::uint64_t> shape) {
        const bool found = declares(name);
        mtp_group.note(name, found);
        // Unused when the group is absent: `out.mtp_declared` is then false and nothing reads
        // these handles (the materialization below is gated on both flags).
        return found ? artifact::bind_tensor(binder, name, format, shape, mtp_placement)
                     : artifact::ObjectHandle{};
    };
    out.mtp.input_projection =
        bind_mtp("mtp/input_projection", NumericFormat::W8G32_F16S, {5120, 10240});
    out.mtp.embedding_norm       = bind_mtp("mtp/embedding_norm", NumericFormat::BF16, {5120});
    out.mtp.hidden_norm          = bind_mtp("mtp/hidden_norm", NumericFormat::BF16, {5120});
    out.mtp.input_norm           = bind_mtp("mtp/layer/input_norm", NumericFormat::BF16, {5120});
    out.mtp.query_key_gate_value = bind_mtp("mtp/layer/attention/query_key_gate_value",
                                            NumericFormat::W8G32_F16S, {14336, 5120});
    out.mtp.query_norm = bind_mtp("mtp/layer/attention/query_norm", NumericFormat::BF16, {256});
    out.mtp.key_norm   = bind_mtp("mtp/layer/attention/key_norm", NumericFormat::BF16, {256});
    out.mtp.output =
        bind_mtp("mtp/layer/attention/output", NumericFormat::W8G32_F16S, {5120, 6144});
    out.mtp.post_attention_norm =
        bind_mtp("mtp/layer/post_attention_norm", NumericFormat::BF16, {5120});
    out.mtp.mlp.gate_up = WeightPlan{
        .object = bind_mtp("mtp/layer/mlp/gate_up", NumericFormat::W8G32_F16S, {34816, 5120}),
        .format = NumericFormat::W8G32_F16S};
    out.mtp.mlp.down = WeightPlan{
        .object = bind_mtp("mtp/layer/mlp/down", NumericFormat::W8G32_F16S, {5120, 17408}),
        .format = NumericFormat::W8G32_F16S};
    out.mtp.final_norm = bind_mtp("mtp/final_norm", NumericFormat::BF16, {5120});
    if (mtp_group.partial()) {
        throw std::invalid_argument(
            "this artifact declares only part of its mtp block: " +
            std::to_string(mtp_group.present) + " of its 12 objects are present and " +
            std::to_string(mtp_group.absent) + " are absent (first absent: " +
            std::string(mtp_group.first_absent) +
            ").  A draft block is all or nothing: fix the converter, do not trust half of it.");
    }
    out.mtp_declared = mtp_group.complete();
    if (features.mtp() && !out.mtp_declared) {
        throw std::invalid_argument(
            "this artifact declares no mtp draft block (no mtp/* object), so there is nothing "
            "for the resolved MTP backend to draft with.  Its recipe wrote the text core only.  "
            "Pass --spec none (or convert a source/recipe that carries the draft block); text "
            "generation is unaffected.");
    }

    if (weights_profile == WeightsProfile::Qwen38Nvfp4Dspark) {
        const artifact::TensorPlacement dflash_placement =
            features.dflash() ? artifact::TensorPlacement::Device
                              : artifact::TensorPlacement::ValidateOnly;
        const auto bind_dflash = [&](std::string_view name, NumericFormat format,
                                     std::initializer_list<std::uint64_t> shape) {
            return artifact::bind_tensor(binder, name, format, shape, dflash_placement);
        };
        const auto bind_dflash_weight = [&](std::string_view name, NumericFormat format,
                                            std::initializer_list<std::uint64_t> shape) {
            return WeightPlan{.object = bind_dflash(name, format, shape), .format = format};
        };
        out.dflash.feature_projection = bind_dflash_weight("dflash/feature_projection",
                                                           NumericFormat::BF16, {5120, 25600});
        out.dflash.context_norm =
            bind_dflash("dflash/context_norm", NumericFormat::BF16, {5120});
        for (std::size_t layer = 0; layer < DFlashConfig::layers; ++layer) {
            DFlashLayerPlan& target  = out.dflash.layers[layer];
            const std::string prefix = "dflash/layers/" + std::to_string(layer) + "/";
            target.input_norm = bind_dflash(prefix + "input_norm", NumericFormat::BF16, {5120});
            target.query_key_value = bind_dflash_weight(prefix + "attention/query_key_value",
                                                        NumericFormat::BF16, {7168, 5120});
            target.context_key = bind_dflash_weight(prefix + "attention/context_key",
                                                    NumericFormat::BF16, {1024, 5120});
            target.context_value = bind_dflash_weight(prefix + "attention/context_value",
                                                      NumericFormat::BF16, {1024, 5120});
            target.query_norm =
                bind_dflash(prefix + "attention/query_norm", NumericFormat::BF16, {128});
            target.key_norm =
                bind_dflash(prefix + "attention/key_norm", NumericFormat::BF16, {128});
            target.attention_output = bind_dflash_weight(prefix + "attention/output",
                                                         NumericFormat::BF16, {5120, 5120});
            target.post_attention_norm =
                bind_dflash(prefix + "post_attention_norm", NumericFormat::BF16, {5120});
            target.mlp.gate_up = bind_dflash_weight(prefix + "mlp/gate_up", NumericFormat::BF16,
                                                    {20480, 5120});
            target.mlp.down = bind_dflash_weight(prefix + "mlp/down", NumericFormat::BF16,
                                                 {5120, 10240});
        }
        out.dflash.final_norm = bind_dflash("dflash/final_norm", NumericFormat::BF16, {5120});
        out.dflash.markov_w1 = bind_dflash_weight("dflash/markov_w1", NumericFormat::BF16,
                                                  {248320, 256});
        out.dflash.markov_w2 = bind_dflash_weight("dflash/markov_w2", NumericFormat::BF16,
                                                  {248320, 256});
    }

    if (weights_profile == WeightsProfile::Qwen38Nvfp4DFlash2 ||
        weights_profile == WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head) {
        const artifact::TensorPlacement dflash2_placement =
            features.dflash2() ? artifact::TensorPlacement::Device
                               : artifact::TensorPlacement::ValidateOnly;
        const auto bind_dflash2 = [&](std::string_view name, NumericFormat format,
                                      std::initializer_list<std::uint64_t> shape) {
            return artifact::bind_tensor(binder, name, format, shape, dflash2_placement);
        };
        const auto bind_dflash2_weight = [&](std::string_view name, NumericFormat format,
                                             std::initializer_list<std::uint64_t> shape) {
            return WeightPlan{.object = bind_dflash2(name, format, shape), .format = format};
        };
        out.dflash2.feature_projection = bind_dflash2_weight("dflash2/feature_projection",
                                                             NumericFormat::BF16, {5120, 25600});
        out.dflash2.context_norm =
            bind_dflash2("dflash2/context_norm", NumericFormat::BF16, {5120});
        for (std::size_t layer = 0; layer < DFlash2Config::layers; ++layer) {
            DFlash2LayerPlan& target  = out.dflash2.layers[layer];
            const std::string prefix = "dflash2/layers/" + std::to_string(layer) + "/";
            target.input_norm = bind_dflash2(prefix + "input_norm", NumericFormat::BF16, {5120});
            target.query_key_value = bind_dflash2_weight(
                prefix + "attention/query_key_value", NumericFormat::BF16, {6144, 5120});
            target.context_key = bind_dflash2_weight(prefix + "attention/context_key",
                                                     NumericFormat::BF16, {1024, 5120});
            target.context_value = bind_dflash2_weight(prefix + "attention/context_value",
                                                       NumericFormat::BF16, {1024, 5120});
            target.query_norm =
                bind_dflash2(prefix + "attention/query_norm", NumericFormat::BF16, {128});
            target.key_norm =
                bind_dflash2(prefix + "attention/key_norm", NumericFormat::BF16, {128});
            target.attention_output = bind_dflash2_weight(prefix + "attention/output",
                                                          NumericFormat::BF16, {5120, 4096});
            target.attention_conv_base = WeightPlan{
                .object = bind_dflash2(prefix + "attention_conv/base_kernel", NumericFormat::BF16,
                                       {2, 2, 5120}),
                .format = NumericFormat::BF16};
            target.attention_conv_projection = bind_dflash2_weight(
                prefix + "attention_conv/kernel_projection", NumericFormat::BF16, {1280, 5120});
            target.post_attention_norm =
                bind_dflash2(prefix + "post_attention_norm", NumericFormat::BF16, {5120});
            target.mlp.gate_up = bind_dflash2_weight(prefix + "mlp/gate_up", NumericFormat::BF16,
                                                     {34816, 5120});
            target.mlp.down =
                bind_dflash2_weight(prefix + "mlp/down", NumericFormat::BF16, {5120, 17408});
            target.mlp_conv_base = WeightPlan{
                .object = bind_dflash2(prefix + "mlp_conv/base_kernel", NumericFormat::BF16,
                                       {2, 2, 5120}),
                .format = NumericFormat::BF16};
            target.mlp_conv_projection = bind_dflash2_weight(
                prefix + "mlp_conv/kernel_projection", NumericFormat::BF16, {1280, 5120});
        }
        out.dflash2.final_norm = bind_dflash2("dflash2/final_norm", NumericFormat::BF16, {5120});
        out.dflash2.selector_hidden_projection = bind_dflash2_weight(
            "dflash2/candidate_selector/hidden_projection", NumericFormat::BF16, {256, 5120});
        out.dflash2.selector_predecessor_codebook = bind_dflash2_weight(
            "dflash2/candidate_selector/predecessor_codebook", NumericFormat::BF16, {248320, 256});
        out.dflash2.selector_successor_codebook = bind_dflash2_weight(
            "dflash2/candidate_selector/successor_codebook", NumericFormat::BF16, {248320, 256});
    }

    // ---- vision tower (334 objects: patch embedding, merger, 324 layer objects) ----
    GroupPresence vision_group;
    for (const std::string_view anchor : {"vision/patch_embedding", "vision/merger/fc1",
                                          "vision/merger/fc2", "vision/merger/fc2_bias",
                                          "vision/merger/norm/weight"}) {
        vision_group.note(anchor, declares(anchor));
    }
    if (vision_group.partial()) {
        throw std::invalid_argument(
            "this artifact declares only part of its vision tower: " +
            std::to_string(vision_group.present) + " of the 5 entry objects are present and " +
            std::to_string(vision_group.absent) + " are absent (first absent: " +
            std::string(vision_group.first_absent) +
            ").  A tower is all or nothing: fix the converter, do not trust half of it.");
    }
    out.vision_declared = vision_group.complete();
    if (out.vision_declared) {
        const artifact::TensorPlacement vision_placement =
            features.vision ? artifact::TensorPlacement::Device
                            : artifact::TensorPlacement::ValidateOnly;
        out.vision_backbone     = qwen3_6::bind_vision_backbone(binder, vision_placement);
        out.vision_merger_input = qwen3_6::bind_vision_merger_input(binder, vision_placement);
        out.vision_merger_fc2   = artifact::bind_tensor(
            binder, "vision/merger/fc2", NumericFormat::W8G32_F16S, {5120, 4608},
            vision_placement);
        out.vision_merger_fc2_bias = artifact::bind_tensor(
            binder, "vision/merger/fc2_bias", NumericFormat::BF16, {5120}, vision_placement);
        out.vision_merger_norm = qwen3_6::bind_vision_merger_norm(binder, vision_placement);
    } else if (features.vision) {
        throw std::invalid_argument(
            "this artifact declares no vision tower (no vision/* object at all): its recipe "
            "wrote the text core only, so --vision has nothing to load.  Drop --vision; text "
            "generation is unaffected.");
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

    token_embedding        = materialized_weight(backing, plan.token_embedding, 248320, 5120);
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm            = artifact::materialized_tensor(backing, source.input_norm,
                                                                         NumericFormat::BF16, {5120});
            target.projection            = load_attention_projection(source.attention, backing);
            target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                              NumericFormat::BF16, {256});
            target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                              NumericFormat::BF16, {256});
            target.output     = materialized_weight(backing, source.attention.output, 5120, 6144);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {5120});
            target.post_mixer = load_mlp(source.mlp, backing);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm  = artifact::materialized_tensor(backing, source.input_norm,
                                                               NumericFormat::BF16, {5120});
            target.projection.a_log =
                artifact::materialized_tensor(backing, source.gdn.a_log, NumericFormat::FP32, {48});
            target.projection.dt_bias = artifact::materialized_tensor(backing, source.gdn.dt_bias,
                                                                      NumericFormat::FP32, {48});
            target.convolution = artifact::materialized_tensor(backing, source.gdn.convolution,
                                                               NumericFormat::BF16, {10240, 4});
            target.projection.control_projection = load_gdn_control_projection(source.gdn, backing);
            target.projection.input_projection   = load_gdn_input_projection(source.gdn, backing);
            target.norm =
                artifact::materialized_tensor(backing, source.gdn.norm, NumericFormat::BF16, {128});
            target.output = materialized_weight(backing, source.gdn.output, 5120, 6144);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {5120});
            target.post_mixer = load_mlp(source.mlp, backing);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("text topology binding is incomplete");
    }
    final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16, {5120});
    output_head = materialized_weight(backing, plan.output_head, 248320, 5120);
    // `*_declared` and the feature are one fact twice: `bind_artifact` refuses a selected
    // feature whose group the artifact does not declare, so a run that reaches here has the
    // objects in hand -- and the handles are meaningless when it does not.
    if (plan.features.optimized_proposal() && plan.draft_head_declared) {
        auto& proposal     = runtime.optimized_proposal.emplace();
        proposal.head      = artifact::materialized_weight(backing, plan.draft_head,
                                                           NumericFormat::Q4G64_F16S, 131072, 5120);
        proposal.token_ids = artifact::materialized_tensor(backing, plan.draft_head_token_ids,
                                                           NumericFormat::I32, {131072});
    }

    if (plan.features.mtp() && plan.mtp_declared) {
        auto& mtp            = runtime.mtp.emplace();
        mtp.input_projection = artifact::materialized_weight(
            backing, plan.mtp.input_projection, NumericFormat::W8G32_F16S, 5120, 10240);
        mtp.embedding_norm   = artifact::materialized_tensor(backing, plan.mtp.embedding_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.hidden_norm      = artifact::materialized_tensor(backing, plan.mtp.hidden_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.input_norm       = artifact::materialized_tensor(backing, plan.mtp.input_norm,
                                                             NumericFormat::BF16, {5120});
        mtp.attention.packed = artifact::materialized_weight(
            backing, plan.mtp.query_key_gate_value, NumericFormat::W8G32_F16S, 14336, 5120);
        mtp.attention.query       = row_view(mtp.attention.packed, 0, 6144);
        mtp.attention.key         = row_view(mtp.attention.packed, 6144, 1024);
        mtp.attention.output_gate = row_view(mtp.attention.packed, 7168, 6144);
        mtp.attention.value       = row_view(mtp.attention.packed, 13312, 1024);
        mtp.query_norm =
            artifact::materialized_tensor(backing, plan.mtp.query_norm, NumericFormat::BF16, {256});
        mtp.key_norm =
            artifact::materialized_tensor(backing, plan.mtp.key_norm, NumericFormat::BF16, {256});
        mtp.output              = artifact::materialized_weight(backing, plan.mtp.output,
                                                                NumericFormat::W8G32_F16S, 5120, 6144);
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {5120});
        mtp.post_mixer = load_mlp(plan.mtp.mlp, backing);
        mtp.final_norm = artifact::materialized_tensor(backing, plan.mtp.final_norm,
                                                       NumericFormat::BF16, {5120});
    }

    if (plan.features.dflash()) {
        DFlashWeights& target     = runtime.dflash.emplace();
        target.feature_projection = materialized_weight(backing, plan.dflash.feature_projection,
                                                        5120, 25600);
        target.context_norm = artifact::materialized_tensor(backing, plan.dflash.context_norm,
                                                            NumericFormat::BF16, {5120});
        for (std::size_t layer = 0; layer < DFlashConfig::layers; ++layer) {
            const DFlashLayerPlan& source = plan.dflash.layers[layer];
            DFlashLayerWeights& weights   = target.layers[layer];
            weights.input_norm      = artifact::materialized_tensor(backing, source.input_norm,
                                                                    NumericFormat::BF16, {5120});
            weights.query_key_value = materialized_weight(backing, source.query_key_value,
                                                          7168, 5120);
            weights.context_key = materialized_weight(backing, source.context_key, 1024, 5120);
            weights.context_value =
                materialized_weight(backing, source.context_value, 1024, 5120);
            weights.query_norm    = artifact::materialized_tensor(backing, source.query_norm,
                                                                  NumericFormat::BF16, {128});
            weights.key_norm =
                artifact::materialized_tensor(backing, source.key_norm, NumericFormat::BF16, {128});
            weights.attention_output =
                materialized_weight(backing, source.attention_output, 5120, 5120);
            weights.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {5120});
            weights.gate_up = materialized_weight(backing, source.mlp.gate_up, 20480, 5120);
            weights.down    = materialized_weight(backing, source.mlp.down, 5120, 10240);
        }
        target.final_norm = artifact::materialized_tensor(backing, plan.dflash.final_norm,
                                                          NumericFormat::BF16, {5120});
        if (weights_profile == WeightsProfile::Qwen38Nvfp4Dspark) {
            target.markov_w1 = materialized_weight(backing, plan.dflash.markov_w1, 248320, 256);
            target.markov_w2 = materialized_weight(backing, plan.dflash.markov_w2, 248320, 256);
        }
    }

    if (plan.features.dflash2()) {
        DFlash2Weights& target      = runtime.dflash2.emplace();
        target.feature_projection  = materialized_weight(backing, plan.dflash2.feature_projection,
                                                         5120, 25600);
        target.context_norm = artifact::materialized_tensor(backing, plan.dflash2.context_norm,
                                                            NumericFormat::BF16, {5120});
        for (std::size_t layer = 0; layer < DFlash2Config::layers; ++layer) {
            const DFlash2LayerPlan& source = plan.dflash2.layers[layer];
            DFlash2LayerWeights& weights   = target.layers[layer];
            weights.input_norm      = artifact::materialized_tensor(backing, source.input_norm,
                                                                    NumericFormat::BF16, {5120});
            weights.query_key_value = materialized_weight(backing, source.query_key_value, 6144, 5120);
            weights.context_key = materialized_weight(backing, source.context_key, 1024, 5120);
            weights.context_value =
                materialized_weight(backing, source.context_value, 1024, 5120);
            weights.query_norm    = artifact::materialized_tensor(backing, source.query_norm,
                                                                  NumericFormat::BF16, {128});
            weights.key_norm =
                artifact::materialized_tensor(backing, source.key_norm, NumericFormat::BF16, {128});
            weights.attention_output =
                materialized_weight(backing, source.attention_output, 5120, 4096);
            weights.attention_conv_base = artifact::materialized_weight(
                backing, source.attention_conv_base.object, NumericFormat::BF16, 4, 5120);
            weights.attention_conv_projection =
                materialized_weight(backing, source.attention_conv_projection, 1280, 5120);
            weights.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {5120});
            weights.gate_up = materialized_weight(backing, source.mlp.gate_up, 34816, 5120);
            weights.down    = materialized_weight(backing, source.mlp.down, 5120, 17408);
            weights.mlp_conv_base = artifact::materialized_weight(
                backing, source.mlp_conv_base.object, NumericFormat::BF16, 4, 5120);
            weights.mlp_conv_projection =
                materialized_weight(backing, source.mlp_conv_projection, 1280, 5120);
        }
        target.final_norm = artifact::materialized_tensor(backing, plan.dflash2.final_norm,
                                                          NumericFormat::BF16, {5120});
        target.selector_hidden_projection = materialized_weight(
            backing, plan.dflash2.selector_hidden_projection, 256, 5120);
        target.selector_predecessor_codebook = materialized_weight(
            backing, plan.dflash2.selector_predecessor_codebook, 248320, 256);
        target.selector_successor_codebook = materialized_weight(
            backing, plan.dflash2.selector_successor_codebook, 248320, 256);
    }

    if (plan.features.vision && plan.vision_declared) {
        auto& vision  = runtime.vision.emplace();
        vision.common = qwen3_6::materialize_vision_common(
            backing, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm);
        vision.merger_fc2      = artifact::materialized_weight(backing, plan.vision_merger_fc2,
                                                               NumericFormat::W8G32_F16S, 5120, 4608);
        vision.merger_fc2_bias = artifact::materialized_tensor(backing, plan.vision_merger_fc2_bias,
                                                               NumericFormat::BF16, {5120});
    }
}

} // namespace ninfer::targets::qwen3_6_27b::detail
