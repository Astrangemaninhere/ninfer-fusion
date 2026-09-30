// src/targets/spark_x2_5_4b/impl/load/bindings.cpp
//
// The artifact object plan for Spark-X2.5-4B: one function per layer, one function
// per object kind, and the runtime-view fill that publishes them to the family.
// Derived from src/targets/muse_glimmer_30b/impl/load/bindings.cpp; the differences
// are marked `SPARK:` and all four of them are consequences of the checkpoint, not
// of taste.
//
// ⚠ THIS FILE PINS THE OBJECT NAMES THE CONVERTER MUST EMIT.  dl/sparkintel's
// REPORT section A5 recorded the gap in exactly these words: "the object names in
// recipe.py are the engine's *roles*; the storage layout of this target's objects
// cannot be pinned until a target owns them".  The table of the mapping, read off
// this file and off tools/convert/spark_x2_5_4b/recipe.py:77-86, is in
// dl/sparktarget/REPORT.md section 4.  The converter-side edit is companion land (2).

#include "targets/spark_x2_5_4b/impl/load/bindings.h"

#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::targets::spark_x2_5_4b::detail {
namespace {

using artifact::NumericFormat;

// SPARK: one format for every weight, and it is BF16.  The artifact the converter
// plans is 290 objects, all BF16, all contiguous-le-v1
// (tools/convert/spark_x2_5_4b: `formats: {'BF16': 290}`, `layouts:
// {'contiguous-le-v1': 290}`), because S54 section 7 clause 2 ruled BF16 for v1 --
// Spark ships no quantisation products and no new architecture has a registered
// quantised geometry.  The sibling targets use FP8_E4M3FN_ROW_BF16S here; copying
// that value would make every object unresolvable, and the check below turns that
// into a named failure rather than a lookup miss.
constexpr NumericFormat kWeightFormat = NumericFormat::BF16;

// A declared format this target does not consume is refused at BIND time, with both
// sides named.  The alternative -- letting the binder fail later with "required
// artifact tensor is missing" -- would be a true statement about the wrong thing,
// because the tensor IS there and it is the FORMAT that is not this target's.
void require_bf16(const artifact::TensorDescriptor& declared, std::string_view name) {
    if (declared.format != kWeightFormat) {
        throw artifact::ArtifactError(std::string(name) + " is declared " +
                                      std::string(artifact::format_name(declared.format)) +
                                      ", but spark_x2_5_4b v1 consumes BF16 only");
    }
}

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape) {
    return WeightPlan{.object = artifact::bind_device_tensor(binder, name, format, shape),
                      .format = format};
}

// The three tensor kinds this target has, each a thin named wrapper so the call sites
// read as what they are rather than as repeated argument triples.
WeightPlan bind_matrix(artifact::Binder& binder, const std::string& name, std::uint64_t rows,
                       std::uint64_t columns) {
    const artifact::TensorDescriptor* declared = binder.find_tensor(name);
    if (declared == nullptr) {
        throw artifact::ArtifactError("required artifact tensor is missing: " + name);
    }
    require_bf16(*declared, name);
    return bind_weight(binder, name, kWeightFormat, {rows, columns});
}

artifact::ObjectHandle bind_vector(artifact::Binder& binder, const std::string& name,
                                  std::uint64_t width) {
    const artifact::TensorDescriptor* declared = binder.find_tensor(name);
    if (declared == nullptr) {
        throw artifact::ArtifactError("required artifact tensor is missing: " + name);
    }
    require_bf16(*declared, name);
    return artifact::bind_device_tensor(binder, name, kWeightFormat, {width});
}

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns);
}

// --------------------------------------------------------------------------
// SPARK: one layer, ten objects, and the ten names are the contract.
//
// What the CHECKPOINT ships per layer (8 source leaves, measured in
// models/Spark-X2.5-4B/model.safetensors.index.json) and what the engine binds:
//
//   model.layers.N.input_layernorm.weight   -> text/layers/N/input_norm
//   model.layers.N.self_attn.q_k_v_proj.weight
//       one fused [6144, 2560] tensor, split by the converter into
//                                          -> text/layers/N/attention/query  [4096, 2560]
//                                          -> text/layers/N/attention/key    [1024, 2560]
//                                          -> text/layers/N/attention/value  [1024, 2560]
//   model.layers.N.self_attn.g_proj.weight  -> text/layers/N/attention/gate   [16, 2560]
//   model.layers.N.self_attn.out_proj.weight
//       [2560, 4096], note the contraction axis   -> text/layers/N/attention/output
//   model.layers.N.post_attention_layernorm.weight
//                                           -> text/layers/N/post_attention_norm
//   model.layers.N.mlp.gate_proj.weight     -> text/layers/N/mlp/gate        [10240, 2560]
//   model.layers.N.mlp.up_proj.weight       -> text/layers/N/mlp/up          [10240, 2560]
//   model.layers.N.mlp.down_proj.weight     -> text/layers/N/mlp/down        [2560, 10240]
//
// TWO THINGS ARE ABSENT AND BOTH ARE MEASURED, NOT FORGOTTEN:
//
//   * there is NO `attention/query_norm` or `attention/key_norm`.  The checkpoint has
//     no q_norm/k_norm tensor in any of its 36 layers, and the engine's
//     `FullAttentionWeights::query_norm` / `key_norm` members therefore have nothing
//     to be bound to.  See the LoadedModelData constructor at the bottom of this file
//     for what happens instead of binding an all-ones vector.
//   * there is NO `post_attn_out_norm` and no MLP-output norm.  Spark normalises
//     twice per layer, not four times: input_layernorm before attention and
//     post_attention_layernorm before the MLP.  TextConfig declares neither
//     attn_out_post_norm() nor mlp_out_post_norm(), so the family's double-norm
//     branches are compile-time false.
//
// The branch below on layer_kind is a real one, not a formality: kFullAttentionLayers
// is 36 here, so every layer takes the same path -- which is the point, and the
// static_assert is what keeps it that way.  If a future Spark revision made the
// geometry heterogeneous this file would need two plans, exactly as gemma4_31b's
// config.h:287-296 pins.
static_assert(TextConfig::full_attention_layers() == static_cast<int>(kTextLayers),
              "Spark binds one plan shape per layer because its geometry is flat; a "
              "heterogeneous revision needs a per-kind plan, not a flattened one");

void bind_text_layers(artifact::Binder& binder, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = bind_vector(binder, prefix + "input_norm", TextConfig::hidden);
        target.is_full_attention = TextConfig::is_full_attention(static_cast<int>(layer));
        target.attention.projection = SparkAttentionProjectionPlan{
            .query = bind_matrix(binder, prefix + "attention/query", TextConfig::query_size,
                                 TextConfig::hidden),
            .key   = bind_matrix(binder, prefix + "attention/key", TextConfig::kv_size,
                                 TextConfig::hidden),
            .value = bind_matrix(binder, prefix + "attention/value", TextConfig::kv_size,
                                 TextConfig::hidden),
            // 16 rows, not 4096: the headwise output gate.  dl/sparkintel's `volume`
            // gate is what established the shape -- the first plan had [4096, 2560]
            // here and the derived parameter count came out exactly
            // 36 * (4096 - 16) * 2560 = 376,012,800 too high against the index's own
            // total_parameters.  A wrong number here is not a rounding error; it is a
            // whole extra projection's worth of weights.
            .gate  = bind_matrix(binder, prefix + "attention/gate", TextConfig::attention_gate_rows,
                                 TextConfig::hidden),
        };
        target.attention.qk_norm_absent = true;
        target.attention.output =
            bind_matrix(binder, prefix + "attention/output", TextConfig::hidden,
                        TextConfig::query_size);
        // Spark's post_attention_layernorm is the PRE-MLP norm.  It is bound to
        // `post_attention_norm` for the same reason muse does -- that is the field the
        // shared runtime reads before the mixer -- and NOT to post_attn_out_norm,
        // which this architecture does not have.
        target.post_attention_norm =
            bind_vector(binder, prefix + "post_attention_layernorm", TextConfig::hidden);
        target.mlp.gate = bind_matrix(binder, prefix + "mlp/gate", TextConfig::intermediate,
                                      TextConfig::hidden);
        target.mlp.up   = bind_matrix(binder, prefix + "mlp/up", TextConfig::intermediate,
                                      TextConfig::hidden);
        target.mlp.down = bind_matrix(binder, prefix + "mlp/down", TextConfig::hidden,
                                      TextConfig::intermediate);
    }
}

// The three MLP matrices, in the shapes the payload's ops::linear calls declare
// (impl/variant.cpp post_mixer: gate and up are [intermediate, cols], down is
// [hidden, intermediate]).
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

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.frontend     = qwen3_6::bind_frontend_resources(binder);
    out.features     = features;

    out.token_embedding =
        bind_matrix(binder, "text/token_embedding", TextConfig::output_rows, TextConfig::hidden);
    bind_text_layers(binder, out);
    out.final_norm =
        bind_vector(binder, "text/final_norm", TextConfig::hidden);
    // The materialised head.  `tie_word_embeddings` is true and the index ships no
    // lm_head.weight, so this object's BYTES are the embedding's, written by
    // dl/sparkintel's recipe.tie_decision() with the orientation it reported
    // (identity: the embedding header is already (131072, 2560)).  The engine binds it
    // as an independent object exactly as the 27b precedent does, which is why the tie
    // never reaches a kernel.
    //
    // It is also the one object whose row count is worth a second look, because
    // Spark's domain is 131072 and 131072 % 128 == 0, so no zero-row padding appears:
    // TextConfig::output_rows == TextConfig::token_domain and both are asserted in
    // impl/config.h.
    out.output_head = bind_matrix(binder, "text/output_head", TextConfig::output_rows,
                                  TextConfig::hidden);
    (void)weights_profile;
    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(WeightsProfile weights_profile, BindingPlan plan,
                                 artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    // ------------------------------------------------------------------
    // ⛔ THE REFUSAL, AND WHY IT IS HERE AND NOT IN A static_assert.
    //
    // Everything above this point is correct and testable: the plan names ten objects
    // per layer with the measured shapes, and a structural test can check the counts
    // and the geometry against a real artifact.  Everything below it fills the runtime
    // view the family's `attn_mix` reads, and THREE of the things attn_mix does
    // unconditionally are wrong for this checkpoint, with ONE family-side fix each:
    //
    //   (A) text_context_impl.h:1204-1205 normalise q and k:
    //           ops::rmsnorm(q, *w.q_norm, kCfg.rms_eps, true, qn, s);
    //           ops::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, kn, s);
    //       Spark's reference normalises NEITHER, and the checkpoint has no such
    //       tensor.  `FullAttentionWeights::query_norm` is a `Tensor` by value
    //       (model_view.h:55-56) and `FullLayerW` takes its ADDRESS unconditionally
    //       (text_context_impl.h:593-594), so there is no "leave it unset" that means
    //       "skip" -- an empty Tensor is still addressable and `qn` would be left
    //       holding whatever the arena had.  The muse precedent (bind an all-ones
    //       vector so the shared rmsnorm path is exact when the reference normalises
    //       WITHOUT a scale) is NOT applicable: for Spark it would apply a
    //       normalisation the checkpoint never applies.  Needs a target-visible gate:
    //       `TextConfig::qk_norm_enabled()`, which impl/config.h already declares as
    //       false and which nothing in the tree reads today (grep for it finds one
    //       definition and zero call sites).
    //
    //   (B) text_context_impl.h:1212-1214 rotate with ONE width and ONE base:
    //           ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);
    //       Spark needs 256 @ 10000 on the 27 sliding layers and 64 @ 5000000 on the 9
    //       full ones.  Both tables are in impl/config.h; `ModelConfig::rope_theta_at`
    //       exists (text_context.h:108) and is called from NOTHING, and there is no
    //       `rotary_dim_at` hook at all -- the slot is a single constexpr
    //       (text_context.h:44).  9 of 36 layers would rotate 256 where the checkpoint
    //       rotates 64.
    //
    //   (C) text_context_impl.h:1180 then :1284 gate the attention output with a
    //       {head_dim, n_q, T} view of the gate buffer, and ops::sigmoid_mul decides
    //       its headwise route BY SHAPE (src/ops/wrapper/sigmoid_mul.cpp:222-225).
    //       Spark's gate is 16 rows, not 4096.  See the long note in
    //       impl/variant.cpp's attention_projection: with the family as it stands, the
    //       per-element route ACCEPTS the pair and multiplies by rows this target never
    //       wrote.  Silent wrong number.
    //
    // A failing static_assert would be the wrong instrument for (A)-(C): it would take
    // the whole `ninfer_engine` archive down, and these three TUs must compile for the
    // build wiring to be real (that is the whole point of landing this target before
    // its runtime).  So the refusal is a runtime one, at the single site that would
    // otherwise publish a wrong view, and it names all three hooks with their file and
    // line.  It is ONE call to delete on the day companion land (1) lands.
    // COMPANION LAND (1) OF THREE HAS LANDED -- this was the ONE call to delete.
    // dl/sparkhooks took the family side: text_context.h now carries the three
    // target-visible hooks (qk_norm / per_layer_rope + rotary_dim_at / gate_rows +
    // headwise_gate, with a namespace-scope static_assert tying the headwise
    // declaration to the shape src/ops/wrapper/sigmoid_mul.cpp:34-37 dispatches on)
    // and text_context_impl.h reads them at the three sites named above
    // (:1204-1205, :1212-1214, :1180/:1184).  Nothing became silent: the arm that
    // used to fall through to the PER-ELEMENT gate route cannot be taken now,
    // because this target's own declarations (`headwise_attn_output_gate_enabled`,
    // `attention_gate_rows = query_heads = 16`) are what select the 2-D view, and
    // the two declarations are cross-checked by a compile-time static_assert.

    // Reached now that the refusal is gone.  Note what is deliberately NOT assigned:
    // query_norm and key_norm stay default-constructed (empty Tensors), which is
    // exactly right, because TextConfig::qk_norm_enabled() == false makes
    // text_context_impl.h's q/k rmsnorm a discarded `if constexpr` branch.  Their
    // addresses are still taken at text_context_impl.h:593-594, and they are never
    // dereferenced.
    frontend = qwen3_6::take_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
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
            throw std::logic_error("spark_x2_5_4b text topology must be all-full: the "
                                   "family's run_layers has no third attention branch");
        }
        FullAttentionWeights& target = full_layers.at(full_index++);
        target.input_norm            = artifact::materialized_tensor(
            backing, source.input_norm, NumericFormat::BF16, {TextConfig::hidden});
        target.projection = FullAttentionProjectionPayload{
            .query = materialized_weight(backing, source.attention.projection.query,
                                         TextConfig::query_size, TextConfig::hidden),
            .key   = materialized_weight(backing, source.attention.projection.key,
                                         TextConfig::kv_size, TextConfig::hidden),
            .gate  = materialized_weight(backing, source.attention.projection.gate,
                                         TextConfig::attention_gate_rows, TextConfig::hidden),
            .value = materialized_weight(backing, source.attention.projection.value,
                                         TextConfig::kv_size, TextConfig::hidden),
        };
        target.output = materialized_weight(backing, source.attention.output, TextConfig::hidden,
                                            TextConfig::query_size);
        target.post_attention_norm = artifact::materialized_tensor(
            backing, source.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
        target.post_mixer = load_mlp(source.mlp, backing);
    }
    if (full_index != full_layers.size()) {
        throw std::logic_error("text topology binding is incomplete");
    }
    final_norm = artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16,
                                               {TextConfig::hidden});
    output_head = materialized_weight(backing, plan.output_head, TextConfig::output_rows,
                                      TextConfig::hidden);
    (void)weights_profile;
}

} // namespace ninfer::targets::spark_x2_5_4b::detail

