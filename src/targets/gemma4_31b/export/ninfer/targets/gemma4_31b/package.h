#pragma once

// src/targets/gemma4_31b/export/ninfer/targets/gemma4_31b/package.h
//
// Gemma-4-31B: identity header + geometry cross-check. There is no Package,
// LoadedModel, Frontend, SequencePlan or Program here, because this target has
// no load path yet; the two measured blockers are named in kRuntimeBlockers and
// pinned as static_asserts in targets/gemma4_31b/impl/config.h.
//
// Shape of this header is src/targets/qwen4_exp/export/ninfer/targets/qwen4_exp/package.h,
// the tree's existing "identity is recognized but the runtime does not exist
// yet" target, and its stage (a) reasoning applies unchanged:
//
//   * src/targets/registry.cpp runs the table lookup at :520-525, and only after
//     it misses does anything here get a chance to speak. So every OTHER model id
//     takes exactly the code path it took before this target existed, including
//     `declared_model_ids()` and the NINFER_REGISTRY_DUMP output (this target
//     adds no kTargetRegistrations row -- it has no Instance adapter to name, and
//     a row without one fails at link time by construction,
//     src/CMakeLists.txt:626-632).
//   * an artifact manifest carries no semantic geometry (src/artifact/reader.cpp's
//     parse_tensor knows name/kind/shape/format/layout/offset/bytes only), so the
//     TENSOR SHAPES ARE THE ARTIFACT SIDE'S ONLY GEOMETRY SOURCE. impl/config.h's
//     constants are the spec side's expectation; every quantity below is derived
//     from one named tensor and compared against the constant, and a mismatch
//     throws naming BOTH numbers. The constants are never silently trusted.

#include "artifact/reader.h"
#include "targets/gemma4_31b/impl/config.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::targets::gemma4_31b {

inline constexpr std::string_view kModelId      = "gemma4-31b";
inline constexpr std::string_view kWeightsId    = "nvfp4";
inline constexpr std::string_view kTargetKey    = "gemma4_31b";
inline constexpr std::string_view kRuntimeStage = "a-identity-only";

// The two items that keep this target from having a load path, each with the
// file that owns it. Carried in the refusal text so the refusal names the work
// instead of only the absence.
inline constexpr std::string_view kRuntimeBlockers =
    "(1) GQA geometry: the 10 full_attention layers are 32 q / 4 kv / head_dim 512 and the "
    "50 sliding ones are 32 q / 16 kv / head_dim 256; src/ops/kernel/"
    "gqa_attention_geometry.cuh:15 admits only head_dim 128 or 256 and its alias list is "
    "{<24,4,256>,<16,2,256>,<32,2,128>,<16,4,256>}, so neither triple is instantiable -- "
    "tools/convert/gemma4_31b/convert.py:154-155. "
    "(2) per-layer rotary: rotary_dim is 256 on the sliding layers and 128 on the full ones "
    "while the shared TextConfig has a single rotary_dim slot -- tools/archkit/"
    "gen_full_target.py:78-82 refuses to emit a config.h for this spec for exactly this "
    "reason -- convert.py:156-158. Two further hooks are named without a site: the per-layer "
    "layer_scalar (all 60 layers, multiply point undecided, "
    "tools/archkit/gemma_engine_plan.md:95-96) and the sliding window, which a 50-of-60 "
    "non-zero table leaves inert rather than honoured "
    "(src/product/kv_component_switch.h:269-270, :314-317).";

namespace detail {

// One named tensor's shape. Missing, or not a tensor, throws naming the object
// and the quantity -- this is the contract face the converter must meet.
inline std::vector<std::uint64_t> tensor_shape(const artifact::Reader& reader,
                                               std::string_view name,
                                               std::string_view quantity) {
    const artifact::ObjectDescriptor* object = reader.find(name);
    if (object == nullptr) {
        throw std::runtime_error(
            "gemma4-31b stage (a): artifact has no tensor '" + std::string(name) +
            "', the only shape source for " + std::string(quantity) +
            " - tools/convert/gemma4_31b/inventory.py plans it, so the writer must emit it; "
            "refusing to guess");
    }
    const auto* tensor = std::get_if<artifact::TensorDescriptor>(object);
    if (tensor == nullptr) {
        throw std::runtime_error(
            "gemma4-31b stage (a): artifact object '" + std::string(name) +
            "' is not a tensor, so " + std::string(quantity) + " is not derivable; "
            "refusing to guess");
    }
    return tensor->shape;
}

inline void expect_dim(std::string_view quantity, std::uint64_t from_artifact,
                       std::uint64_t from_config, std::string_view tensor_name) {
    if (from_artifact != from_config) {
        throw std::runtime_error(
            "gemma4-31b stage (a): geometry mismatch for " + std::string(quantity) +
            ": artifact tensor '" + std::string(tensor_name) + "' says " +
            std::to_string(from_artifact) + ", targets/gemma4_31b/impl/config.h says " +
            std::to_string(from_config) + "; refusing to load");
    }
}

} // namespace detail

// Returns one PASS verdict line, to be spliced into the refusal message that
// src/targets/registry.cpp raises. Throws std::runtime_error naming both sides
// on the first quantity that disagrees or cannot be derived.
//
// Tensor -> quantity, with the plan object each name comes from
// (tools/convert/gemma4_31b/inventory.py:175-222 builds exactly these names):
//   text/token_embedding          (vocab, hidden)        -> vocab, hidden
//   text/layers/0/attention/query (8192, 5376)           -> sliding q rows, hidden
//   text/layers/0/attention/key   (4096, 5376)           -> sliding kv rows, hidden
//   text/layers/5/attention/query (16384, 5376)          -> full q rows, hidden
//   text/layers/5/attention/key   (2048, 5376)           -> full kv rows, hidden
//   text/layers/0/mlp/gate        (21504, 5376)          -> intermediate, hidden
//   text/layers/0/input_norm      (5376,)                -> hidden
//   text/layers/0/layer_scalar    (1,)                   -> the per-layer scalar object exists
//   text/layers/<N>/*             max N + 1              -> layers
// Layer 0 is a sliding layer and layer 5 is the first full one
// (impl/config.h full_attention_slots()[0] == 5), so the two geometries are
// both reachable, and layer 0 is also the layer inventory.py:25 measures its
// packed q_proj on.
[[nodiscard]] std::string validate_stage_a_geometry(const artifact::Reader& reader);

} // namespace ninfer::targets::gemma4_31b
