#pragma once

// src/targets/spark_x2_5_4b/impl/config.h
//
// The complete `TextConfig` surface for Spark-X2.5-4B.  This is the file the
// message that opened this line called "the C++ half": every number below is read
// off the checkpoint or off a tree file that already landed, and every derived
// quantity is pinned by a static_assert with a SECOND, independent source, so a
// drift fails the build instead of reaching a kernel.
//
// PROVENANCE, per group
// ---------------------
//   * geometry, layer schedule, per-kind rope:
//     tools/convert/spark_x2_5_4b/inventory.py (landed by dl/sparkintel, sha256
//     68d9cd19d4b6227b243e26ff9510a596ffad3dfaf49e6e5c2787a91bb487095b) and
//     models/Spark-X2.5-4B/config.json (sha256
//     767161ece8ed2891e44345afdf98d29c1cacb964116cb6db7d20dc1c757490ad).
//   * the catalogued adapt output this file supersedes:
//     tools/archkit/out/spark-x2.5-4b/config.h (949 B, sha256
//     d5f8c218a553088852f450ec9533a88196c718ee508c6d6427343fe616837b64).
//     THAT FILE IS NOT USABLE AS A TARGET CONFIG and this file is the replacement,
//     for four measured reasons, all of them named in dl/sparktarget/REPORT.md:
//       (1) it is the `adapt.py` v4 skeleton, not the family TextConfig: it has no
//           output_rows / token_domain / rotary_dim / key_dim / value_dim /
//           convolution_dim / query_size / kv_size / mtp_* / is_full_attention() /
//           full_attention_index() / gdn_index(), and no type for the DFlash pair;
//       (2) its `full_attention_layers()` returns 9 and its `gdn_layers()` returns 0,
//           which sum to 9, not 36 -- the family's `run_layers` is a full/gdn
//           dichotomy, so 27 layers would silently never run;
//       (3) it has no per-layer table at all, so the 27 sliding layers and the 9
//           full ones cannot be told apart;
//       (4) `spark-x2.5-4b` carries a '.', which is not a legal C++ identifier.
//           adapt.py:343 now folds '.' and '-' to '_' (S54 section 7 clause 1), so
//           the namespace below is the generated one -- but the generated file is
//           still (1)-(3).
//   * the encoder half of the same schema: tools/convert/spark_x2_5_4b/convert.py
//     (sha256 f94c220ef7c325ddc7530a0233784330d9e1565da22a1803fb6a8189767804c4),
//     which is where the front door cross-checks these same numbers.
//
// THE ONE STRUCTURAL FACT THAT MAKES SPARK EASIER THAN GEMMA-4-31B
// ---------------------------------------------------------------
// All 36 layers share ONE attention geometry: 16 query heads, 4 KV heads,
// head_dim 256.  src/ops/kernel/gqa_attention_geometry.cuh:362 already carries the
// alias this needs --
//
//     using Gqa16x4Geometry = GqaGeometry<16, 4, 1>;
//
// (with :344 measuring that its KVHeads is 4, the same count Gqa27Geometry has, so
// DecodeSplitScale 1 keeps the same CTA count).  So unlike gemma4_31b -- whose
// config.h must pin two different head widths as STATIC ASSERTIONS OF THE OBSTACLE
// (src/targets/gemma4_31b/impl/config.h:287-296) and which therefore ships no
// runtime at all -- a flattened TextConfig CAN represent Spark's attention exactly,
// and the single `query_heads`/`kv_heads`/`head_dim` below are the truth for every
// layer rather than a majority that is wrong on nine of them.
//
// WHAT IS *NOT* FLAT, AND THE FAMILY HOOKS THAT DO NOT EXIST YET
// -------------------------------------------------------------
// Exactly one thing varies by layer kind and it is the rope width and its base:
// the 27 sliding layers rotate all 256 dimensions at theta 10000, the 9 full layers
// rotate 64 at theta 5000000 (partial_rotary_factor 0.25 applies to the full group
// only).  The family runtime has ONE `rotary_dim` slot and it reads it at ONE call
// site.  Both halves are measured:
//
//   * the slot:      src/targets/qwen3_6/impl/runtime/text_context.h:44
//                      `static constexpr int rotary_dim = TextConfig::rotary_dim;`
//   * the call site: src/targets/qwen3_6/impl/runtime/text_context_impl.h:1212-1214
//                      `ops::rope_yarn4(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, ...)`
//                      `ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, ...)`
//
// `ModelConfig::rope_theta_at(layer)` DOES exist (text_context.h:108) and is NOT
// called from any of the two rope providers this target reaches -- so the per-layer
// theta table below is DECLARED AND UNCONSUMED, which is the same standing
// gemma4_31b's per-kind accessors have ("THE SURFACE THE FAMILY DOES NOT READ",
// gemma4_31b/impl/config.h:199-203).  Declaring it here is what makes the numbers
// single-sourced today; reading it is a family-side change, named in this line's
// report as companion land (1) of three.
//
// This file therefore states the table AND states that it is not read.  It does not
// pick one of the two rope widths and call it the truth: `rotary_dim` below is the
// 27-layer majority value, and the 9 full layers are WRONG under it.  That is why
// src/targets/spark_x2_5_4b/impl/package.cpp refuses at plan_load instead of
// loading numbers that are silently wrong on a quarter of the stack.

#include <array>
#include <cstdint>

#include <ninfer/targets/qwen3_6/hybrid_topology.h>
#include <ninfer/targets/qwen3_6/vision.h>

namespace ninfer::targets::spark_x2_5_4b::detail {

// Layer kind vocabulary, the numbering the family's own emitters use
// (tools/archkit/gen_full_target.py:21-23) and the two 52-layer precedents share
// (src/targets/muse_glimmer_30b/impl/config.h:57-59).
inline constexpr int kKindFull    = 0;
inline constexpr int kKindSliding = 1;
inline constexpr int kKindGdn     = 2;

struct TextConfig {
    // --------------------------------------------------------------------- //
    // shared geometry -- one attention geometry for all 36 layers
    // --------------------------------------------------------------------- //
    static constexpr int hidden       = 2560;    // config.json hidden_size
    static constexpr int layers       = 36;      // config.json num_hidden_layers
    static constexpr int intermediate = 10240;   // config.json intermediate_size

    // Output rows and sampling domain are the same number here, and that is a
    // measured property, not a coincidence of rounding: Spark's tokenizer domain is
    // exactly 131072 and 131072 % 128 == 0, so no zero-row padding is needed for the
    // NVFP4 block-scale head layout that forces the split on muse (202112 artifact
    // rows vs 202048 domain, muse_glimmer_30b/impl/config.h:18-21).  Stated as a
    // static_assert below so the day it stops holding it stops holding loudly.
    static constexpr int output_rows  = 131072;
    static constexpr int token_domain = 131072;

    // Pure softmax family: no GDN, no convolution, no MTP, no draft head.  The slots
    // are kept non-zero because the shared ModelConfig surface reads them
    // (text_context.h:95-98 map them unconditionally) and the pool planner's
    // non-zero-geometry requirement is what made muse fill 1 rather than 0
    // (muse_glimmer_30b/impl/config.h:23-29, src/targets/.../layouts_impl.h:159-166
    // uses max(1, ...) either way).
    static constexpr int gdn_conv_kernel      = 2;
    static constexpr int gdn_conv_state_width = 1;
    static constexpr int gdn_key_heads        = 1;
    static constexpr int gdn_key_head_dim     = 128;
    static constexpr int gdn_value_heads      = 1;
    static constexpr int gdn_value_head_dim   = 128;

    static constexpr int query_heads = 16;   // config.json num_attention_heads
    static constexpr int kv_heads    = 4;    // config.json num_key_value_heads
    static constexpr int head_dim    = 256;  // config.json head_dim (explicit)
    // THE SINGLE SLOT, and the 27-of-36 majority value.  See the file header: the 9
    // full layers need 64 and there is no hook that would let this file say so.
    static constexpr int rotary_dim  = 256;

    static constexpr float rms_epsilon   = 1e-06F;  // config.json rms_norm_eps
    static constexpr float rope_theta    = 10000.0F;  // sliding group's base; majority slot
    // config.json declares neither final_logit_softcapping nor an output multiplier;
    // the archkit adapt output agrees (both 1.0 / 0.0).  Declaring 0.0F here keeps
    // ModelConfig::apply_final_logit_policy a compile-time no-op
    // (text_context.h:183-187), which is the measured truth, rather than leaving it
    // to a default.
    static constexpr float final_logit_softcapping = 0.0F;

    // config.json tie_word_embeddings = true, and the index ships no lm_head.weight,
    // so the head has to be materialised at conversion time.  dl/sparkintel's
    // recipe.tie_decision() is that rule and it measured the orientation as identity
    // (the embedding header is already (131072, 2560) == the head object shape).
    // Declared here as the fact the registry cross-check reads; the materialisation
    // itself is the converter's, which is why nothing in this target's bindings.cpp
    // reads a `lm_head` object.
    static constexpr bool tie_word_embeddings = true;

    // config.json hidden_act = "gelu" -- the EXACT erf form, not the tanh
    // approximation.  This is a leaf-level property, and the leaf that consumes it
    // needs an op the tree now has: include/ninfer/ops/gelu_mul.h:27,
    // `ideal[i] = gelu(gate[i]) * up[i]`, exact erf, "every registered gated
    // architecture declares hidden_act = gelu".  See impl/variant.cpp post_mixer.
    static constexpr const char* hidden_activation = "gelu";

    // config.json attention_bias = false and mlp_bias = false: no projection in this
    // checkpoint carries a bias row, which is what lets every leaf below be a bare
    // ops::linear with no add_bias.
    static constexpr bool attention_bias = false;
    static constexpr bool mlp_bias       = false;

    // config.json headwise_attn_output_gate = true, gate_attn_act_mode = "sigmoid":
    // self_attn.g_proj is [query_heads, hidden] -- ONE SCALAR PER HEAD, not a
    // per-channel projection.  The measured shape is (16, 2560) and the engine
    // consumer is ops::sigmoid_mul's headwise-scalar route
    // (include/ninfer/ops/sigmoid_mul.h:17-27, selected by shape in
    // src/ops/wrapper/sigmoid_mul.cpp:222-225).
    static constexpr bool headwise_attn_output_gate = true;
    // Rows of the gate object, i.e. what the headwise broadcast is over.  This is a
    // NEW derived quantity with no sibling: every other target in this tree gates
    // per channel, so its gate rows are query_size.  16 != 4096 is the whole
    // difference and it is a family-side view change, named in the report.
    static constexpr int attention_gate_rows = query_heads;

    // --------------------------------------------------------------------- //
    // per-kind rope -- the tables this file owns and the family does not read
    // --------------------------------------------------------------------- //
    // inventory.py: ROTARY_DIM_BY_KIND = {sliding: 256, full: 64} and
    // ROPE_THETA_BY_KIND = {sliding: 10000.0, full: 5000000.0}; the 0.25
    // partial_rotary_factor in config.json applies to the full group only, and
    // 0.25 * 256 = 64.
    static constexpr int sliding_rotary_dim  = 256;
    static constexpr int full_rotary_dim     = 64;
    static constexpr float sliding_rope_theta = 10000.0F;
    static constexpr float full_rope_theta    = 5000000.0F;
    // config.json sliding_window = 512, and only the sliding group is windowed;
    // the 9 full layers are global by definition and declare 0 (the gemma4_31b
    // convention, config.h:232-234).
    static constexpr int sliding_window = 512;

    // --------------------------------------------------------------------- //
    // the layer schedule.  0 = full, 1 = sliding.  config.json layer_types is the
    // repeating "3 sliding + 1 full" cycle, so the full layers are 3, 7, 11, ... 35
    // (nine of them) -- i.e. every layer with layer % 4 == 3, which is the
    // independent rule the first static_assert below checks the literal table
    // against.
    // --------------------------------------------------------------------- //
    static constexpr std::array<int, 36> layer_kind{
        1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,
        1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0};

    // Per-layer rope base, in the shape muse_glimmer_30b/impl/config.h:62 uses.  A
    // 0.0F entry would mean NoPE in that file's convention; Spark has NO NoPE layer
    // -- every one of the 36 rotates something -- so the NoPE assert of that file is
    // deliberately NOT copied and the equivalent check below asserts the opposite
    // (36 non-zero entries, in two distinct values).
    static constexpr std::array<float, 36> layer_rope_theta{
        10000.0F,   10000.0F,   10000.0F,   5000000.0F, 10000.0F,   10000.0F,
        10000.0F,   5000000.0F, 10000.0F,   10000.0F,   10000.0F,   5000000.0F,
        10000.0F,   10000.0F,   10000.0F,   5000000.0F, 10000.0F,   10000.0F,
        10000.0F,   5000000.0F, 10000.0F,   10000.0F,   10000.0F,   5000000.0F,
        10000.0F,   10000.0F,   10000.0F,   5000000.0F, 10000.0F,   10000.0F,
        10000.0F,   5000000.0F, 10000.0F,   10000.0F,   10000.0F,   5000000.0F};

    // Per-layer rope width, the quantity with no family slot at all.
    static constexpr std::array<int, 36> layer_rotary_dim{
        256, 256, 256, 64, 256, 256, 256, 64, 256, 256, 256, 64, 256, 256, 256, 64, 256, 256,
        256, 64,  256, 256, 256, 64, 256, 256, 256, 64, 256, 256, 256, 64, 256, 256, 256, 64};

    // `layer_of_full_index` is the identity for Spark because EVERY layer is a full
    // layer as far as the family's full/gdn dichotomy is concerned -- the sliding
    // ones run as full until the window is wired (the same standing muse's 39
    // sliding layers have, muse_glimmer_30b/impl/config.h:66-68).  The table is
    // written out rather than left to fidx == layer so that the bijection assert
    // below has something to check.
    static constexpr std::array<int, 36> kFullLayers{
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};

    // --------------------------------------------------------------------- //
    // derived widths.  Exact, because the geometry is flat.
    // --------------------------------------------------------------------- //
    static constexpr int key_dim               = gdn_key_heads * gdn_key_head_dim;
    static constexpr int value_dim             = gdn_value_heads * gdn_value_head_dim;
    static constexpr int convolution_dim       = 2 * key_dim + value_dim;
    static constexpr int query_size            = query_heads * head_dim;   // 4096
    static constexpr int kv_size               = kv_heads * head_dim;      // 1024
    static constexpr int query_projection_rows = 2 * query_size;           // shared surface

    static constexpr int mtp_layers               = 0;
    static constexpr int mtp_input_rows           = 0;
    static constexpr int mtp_attention_input_rows = 0;
    static constexpr int mtp_mlp_gate_up_rows     = 0;

    // --------------------------------------------------------------------- //
    // the family accessors
    // --------------------------------------------------------------------- //
    [[nodiscard]] static constexpr bool is_full_attention(int layer) {
        // TRUE FOR ALL 36.  The family has exactly two branches, full and gdn
        // (text_context_impl.h run_layers), and `full_attention_layers()` is what
        // sizes the KV plan (decoder_state.cpp:164).  Answering 9 here would plan KV
        // for 9 layers while the loop walks 36 -- the defect tools/archkit's
        // generated config.h has, and the reason it is not used.
        return layer_kind[static_cast<std::size_t>(layer)] != kKindGdn;
    }
    [[nodiscard]] static constexpr bool is_swa_attention(int layer) {
        return layer_kind[static_cast<std::size_t>(layer)] == kKindSliding;
    }
    // The gate that does not exist yet.  Declared false so the fact is in one place
    // and a reader cannot think q/k are normalized; UNCONSUMED today --
    // text_context_impl.h:1204-1205 normalises q and k unconditionally, and the
    // checkpoint has no q_norm / k_norm tensor in any of its 36 layers.
    [[nodiscard]] static constexpr bool qk_norm_enabled() { return false; }
    [[nodiscard]] static constexpr bool qk_norm_tensors_present() { return false; }
    [[nodiscard]] static constexpr bool per_layer_scalar() { return false; }
    [[nodiscard]] static constexpr bool headwise_attn_output_gate_enabled() { return true; }
    [[nodiscard]] static constexpr bool final_logit_softcapping_enabled() {
        return final_logit_softcapping > 0.0F;
    }
    [[nodiscard]] static constexpr int full_attention_layers() {
        int count = 0;
        for (const int kind : layer_kind) {
            if (kind != kKindGdn) { ++count; }
        }
        return count;
    }
    [[nodiscard]] static constexpr int swa_attention_layers() {
        int count = 0;
        for (const int kind : layer_kind) {
            if (kind == kKindSliding) { ++count; }
        }
        return count;
    }
    [[nodiscard]] static constexpr int gdn_layers() {
        int count = 0;
        for (const int kind : layer_kind) {
            if (kind == kKindGdn) { ++count; }
        }
        return count;
    }
    [[nodiscard]] static constexpr int swa_attention_index(int layer) {
        int count = 0;
        for (int i = 0; i < layer; ++i) { count += is_swa_attention(i) ? 1 : 0; }
        return count;
    }
    [[nodiscard]] static constexpr int full_attention_index(int layer) {
        int count = 0;
        for (int i = 0; i < layer; ++i) { count += is_full_attention(i) ? 1 : 0; }
        return count;
    }
    [[nodiscard]] static constexpr int gdn_index(int) { return 0; }
    [[nodiscard]] static constexpr int layer_of_full_index(int fidx) {
        return kFullLayers[static_cast<std::size_t>(fidx)];
    }
    [[nodiscard]] static constexpr float rope_theta_at(int layer) {
        return layer_rope_theta[static_cast<std::size_t>(layer)];
    }
    [[nodiscard]] static constexpr int rotary_dim_at(int layer) {
        return layer_rotary_dim[static_cast<std::size_t>(layer)];
    }
    // 0 on the 9 global layers, the window on the 27 windowed ones -- the shape
    // gemma4_31b/impl/config.h:232-234 established.  INERT, and stated as measured:
    // an all-non-zero table on a tier that does not read `sliding_window_tokens` is
    // refused by name (src/product/kv_component_switch.h:269-270, :314-317), and
    // src/core/paged_kv_cache.h's field is never assigned on the main path, so this
    // table neither releases a page nor bounds attention.  It is here so that when
    // the window lands, the numbers are already single-sourced.
    [[nodiscard]] static constexpr std::uint32_t sliding_window_at(int layer) {
        return is_swa_attention(layer) ? static_cast<std::uint32_t>(sliding_window) : 0U;
    }
};

// --------------------------------------------------------------------------- //
// Compile-time self-checks.  Each names a second source for the same fact.
// --------------------------------------------------------------------------- //

static_assert(TextConfig::layer_kind.size() == TextConfig::layers);
static_assert(TextConfig::layer_rope_theta.size() == TextConfig::layers);
static_assert(TextConfig::layer_rotary_dim.size() == TextConfig::layers);
static_assert(TextConfig::full_attention_layers() == 36);  // all 36 run as full attention
static_assert(TextConfig::swa_attention_layers() == 27);   // front door: {'sliding': 27, 'full': 9}
static_assert(TextConfig::gdn_layers() == 0);              // inventory: all 36 layers are softmax

// the literal schedule vs the arithmetic rule that generates the same slots
static_assert([] {
    for (int layer = 0; layer < TextConfig::layers; ++layer) {
        const bool by_rule = (layer % 4) == 3;
        if (by_rule != (TextConfig::layer_kind[static_cast<std::size_t>(layer)] == kKindFull)) {
            return false;
        }
    }
    return true;
}());

// the literal theta table vs the per-kind constants, layer by layer
static_assert([] {
    for (int layer = 0; layer < TextConfig::layers; ++layer) {
        const float want = TextConfig::is_swa_attention(layer) ? TextConfig::sliding_rope_theta
                                                               : TextConfig::full_rope_theta;
        if (TextConfig::layer_rope_theta[static_cast<std::size_t>(layer)] != want) { return false; }
    }
    return true;
}());

// the literal rotary table vs the per-kind constants, and the two facts the single
// slot cannot carry
static_assert([] {
    for (int layer = 0; layer < TextConfig::layers; ++layer) {
        const int want = TextConfig::is_swa_attention(layer) ? TextConfig::sliding_rotary_dim
                                                             : TextConfig::full_rotary_dim;
        if (TextConfig::layer_rotary_dim[static_cast<std::size_t>(layer)] != want) { return false; }
    }
    return true;
}());
// the partial-rotary arithmetic: 0.25 * head_dim == the full group's width
static_assert(TextConfig::full_rotary_dim * 4 == TextConfig::head_dim);
static_assert(TextConfig::sliding_rotary_dim == TextConfig::head_dim);
// NoPE is absent, which is the opposite of the family's muse precedent and is
// asserted rather than commented because muse's config.h asserts the other thing.
static_assert([] {
    int nonzero = 0;
    for (const float theta : TextConfig::layer_rope_theta) {
        if (theta > 0.0F) { ++nonzero; }
    }
    return nonzero == TextConfig::layers;
}());
// the two rope widths really are two, and the single slot really is one of them
static_assert(TextConfig::rotary_dim_at(0) == 256);
static_assert(TextConfig::rotary_dim_at(3) == 64);
static_assert(TextConfig::rotary_dim_at(0) != TextConfig::rotary_dim_at(3));
static_assert(TextConfig::rotary_dim == TextConfig::rotary_dim_at(0));

// the window table: 27 non-zero, 9 zero
static_assert([] {
    std::uint32_t windowed = 0;
    std::uint32_t global   = 0;
    for (int layer = 0; layer < TextConfig::layers; ++layer) {
        if (TextConfig::sliding_window_at(layer) == 0U) { ++global; } else { ++windowed; }
    }
    return windowed == 27U && global == 9U;
}());

// full-slot bijection: layer_of_full_index(fidx) is a full layer and its index agrees
static_assert([] {
    for (int f = 0; f < TextConfig::full_attention_layers(); ++f) {
        if (!TextConfig::is_full_attention(TextConfig::layer_of_full_index(f))) { return false; }
        if (TextConfig::full_attention_index(TextConfig::layer_of_full_index(f)) != f) {
            return false;
        }
    }
    return true;
}());

// rows, the quantities the artifact's own object shapes are compared against
static_assert(TextConfig::query_size == 4096);
static_assert(TextConfig::kv_size == 1024);
static_assert(TextConfig::attention_gate_rows == 16);
static_assert(TextConfig::attention_gate_rows != TextConfig::query_size);
// the fused q_k_v_proj the checkpoint ships is one tensor with 6144 rows, which is
// exactly query + key + value -- this is why no row padding appears anywhere in this
// target's bindings, and it is a check rather than a comment because the converter
// splits that one tensor into three objects.
static_assert(TextConfig::query_size + 2 * TextConfig::kv_size == 6144);

// token domain: no zero-row padding is needed for the head
static_assert(TextConfig::output_rows == TextConfig::token_domain);
static_assert(TextConfig::token_domain == 131072);
static_assert(TextConfig::token_domain % 128 == 0);

// THE ONE BLOCKER, PINNED SO THE BUILD NAMES IT RATHER THAN A KERNEL FEELING IT.
// Rotary width is per layer kind and the family has a single slot
// (text_context.h:44, read at text_context_impl.h:1212-1214).  Written as an assert
// on the obstacle, in the shape gemma4_31b/impl/config.h:298-302 uses, so that the
// day someone "fixes" the single slot by picking one value, the build states what
// was lost.  The refusal that keeps this from being a silent wrong number is in
// impl/package.cpp, not here: a failing static_assert would take the whole engine
// down, and the three TUs must compile for the build wiring to be real.
static_assert(TextConfig::rotary_dim_at(0) != TextConfig::rotary_dim_at(3),
              "Spark's rope width is per layer kind; the shared TextConfig rotary_dim "
              "slot and the two rope call sites in text_context_impl.h carry ONE value, "
              "so the 9 full layers would rotate 256 where the checkpoint rotates 64");

} // namespace ninfer::targets::spark_x2_5_4b::detail

// --------------------------------------------------------------------------- //
// family-level constants.  Each one named for the quantity it is, and each one
// derived from a number above rather than retyped.
// --------------------------------------------------------------------------- //
namespace ninfer::targets::spark_x2_5_4b::detail {

// 1 / sqrt(head_dim) * qk_scale_factor, with config.json's qk_scale_factor absent
// (the archkit adapt output reads it as 1.0).  Spark carries no qk-norm and no
// softcap, so this is the whole scaling: 1/16 = 0.0625, which is the same value
// Gqa27Geometry's target uses.
inline constexpr float kAttentionScale = 0.0625F;
inline constexpr float kGdnScale       = 0.0F;
// 131072 % 128 == 0, so the head's row count needs no alignment padding; the value
// is stated because text_context.h's prefill alignment is a real scheduler bound,
// not because this target rounds anything.
inline constexpr std::uint32_t kPrefillChunkAlignment = 128;
// Inert: there is no draft backend on this target (the index has no MTP head and no
// draft head, and the family's own DFlash/DFlash2 placeholder structs below are
// `supported = false`).  Kept at muse's values so the two 52/36-layer precedents
// differ only where the checkpoint differs.
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 5;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 7;
// config.json max_position_embeddings is 1048576 and the sliding window is 512.
// Neither is claimed: v1 takes the same provisional value muse took, because the engine's own
// visible-key ceiling is 1'010'000 (src/ops/... gqa_attention.h) and the window is
// not wired (S54 section 7 clause 3: v1 acceptance is a context <= 512, where a
// 512-token window is semantically identical to full attention).
inline constexpr std::uint32_t kNativeContext = 131072;

// Draft backend placeholders.  Spark has neither; the shared family surface reads
// these types, so they exist with `supported = false` and zero geometry -- the same
// shape muse_glimmer_30b/impl/config.h:133-192 uses, and the reason nothing in this
// target's variant.cpp has to implement an MTP leaf.
struct DFlashConfig {
    static constexpr bool supported     = false;
    static constexpr bool bf16_weights  = false;
    static constexpr int layers         = 0;
    static constexpr bool full_only     = true;
    static constexpr int local_layers   = 0;
    static constexpr int full_layers    = 0;
    static constexpr int feature_layers = 0;
    static constexpr int feature_rows   = 0;
    static constexpr int hidden         = TextConfig::hidden;
    static constexpr int intermediate   = TextConfig::intermediate;
    static constexpr int query_heads    = 0;
    static constexpr int kv_heads       = 0;
    static constexpr int head_dim       = 0;
    static constexpr int query_size     = 0;
    static constexpr int kv_size        = 0;
    static constexpr int local_capacity = 0;
    static constexpr std::uint32_t local_window = 0;
    static constexpr int mask_token     = 0;
    static constexpr float rms_epsilon  = TextConfig::rms_epsilon;
    static constexpr float rope_theta   = TextConfig::rope_theta;
    static constexpr float attention_scale = 0.0F;
    static constexpr float svip_entropy_threshold = 0.0F;
    static constexpr std::array<int, 1> target_feature_layers{0};
};

struct DFlash2Config {
    static constexpr bool supported     = false;
    static constexpr bool bf16_weights  = false;
    static constexpr int layers         = 0;
    static constexpr bool full_only     = true;
    static constexpr int local_layers   = 0;
    static constexpr int full_layers    = 0;
    static constexpr int feature_layers = 0;
    static constexpr int feature_rows   = 0;
    static constexpr int hidden         = TextConfig::hidden;
    static constexpr int intermediate   = TextConfig::intermediate;
    static constexpr int query_heads    = 0;
    static constexpr int kv_heads       = 0;
    static constexpr int head_dim       = 0;
    static constexpr int query_size     = 0;
    static constexpr int kv_size        = 0;
    static constexpr int local_capacity = 0;
    static constexpr std::uint32_t local_window = 0;
    static constexpr int mask_token     = 0;
    static constexpr float rms_epsilon  = TextConfig::rms_epsilon;
    static constexpr float rope_theta   = TextConfig::rope_theta;
    static constexpr float attention_scale = 0.0F;
    static constexpr int block_drafts   = 0;
    static constexpr int conv_group_size = 0;
    static constexpr int conv_kernel_size = 0;
    static constexpr int selector_rank   = 0;
    static constexpr int selector_top_k  = 0;
    // Rows of the checkpoint's dedicated proposal head.  Spark ships none: the
    // index's 290 tensors contain no draft head (dl/sparkintel/REPORT.md section 1.2
    // measured exactly the 288 per-layer leaves plus embedding plus final norm), so
    // this is an inert placeholder that must stay non-zero to match the shared
    // skeleton's own requirement -- and it is the SAME number as the token domain,
    // which is why it is spelled that way rather than as a literal.
    static constexpr int draft_head_rows = TextConfig::token_domain;
    static constexpr std::array<int, 1> target_feature_layers{0};
};

// Vision placeholder.  Spark-X2.5-4B is text-only, and the converter says so in the
// front-door contract itself: tools/convert/spark_x2_5_4b/convert.py:72
// `SUPPLIES_FRONTEND_RESOURCES = False`.  The type exists only because the shared
// vision_context has compile-time member accesses that must resolve; with
// `features.vision == false` it is never instantiated at run time.  ⚠ See the
// report's companion-land (3): a Package whose frontend plan is empty while the
// family's make_frontend expects bound resources is the one seam in this target that
// the converter side has to close, and it is named there rather than assumed here.
struct VisionConfig : qwen3_6::VisionBackboneConfig {
    static constexpr int output_hidden = TextConfig::hidden;
};

} // namespace ninfer::targets::spark_x2_5_4b::detail
