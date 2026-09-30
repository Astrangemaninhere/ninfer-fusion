#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/attn_output_gate_route.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.


#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/stage_plan.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/logit_policy.h"
#include "ninfer/ops/softmax_attention.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6/round_state.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

// Target-private compatibility vocabulary for the mechanically preserved fixed schedule. It is
// data-only: TextContext is constructed on the stack for one schedule recording/execution and owns
// neither weights nor device state.
struct ModelConfig {
    static constexpr int hidden              = TextConfig::hidden;
    static constexpr int n_layers            = TextConfig::layers;
    static constexpr int intermediate        = TextConfig::intermediate;
    static constexpr int vocab               = TextConfig::output_rows;
    static constexpr int token_domain        = TextConfig::token_domain;
    static constexpr int gdn_k_heads         = TextConfig::gdn_key_heads;
    static constexpr int gdn_k_dim           = TextConfig::gdn_key_head_dim;
    static constexpr int gdn_v_heads         = TextConfig::gdn_value_heads;
    static constexpr int gdn_v_dim           = TextConfig::gdn_value_head_dim;
    static constexpr int n_q                 = TextConfig::query_heads;
    static constexpr int n_kv                = TextConfig::kv_heads;
    static constexpr int head_dim            = TextConfig::head_dim;
    static constexpr int rotary_dim          = TextConfig::rotary_dim;
    static constexpr int key_dim             = TextConfig::key_dim;
    static constexpr int value_dim           = TextConfig::value_dim;
    static constexpr int conv_dim            = TextConfig::convolution_dim;
    static constexpr int q_size              = TextConfig::query_size;
    static constexpr int kv_size             = TextConfig::kv_size;
    static constexpr int mtp_fc_in           = TextConfig::mtp_input_rows;
    static constexpr int mtp_attn_in         = TextConfig::mtp_attention_input_rows;
    static constexpr int mtp_mlp_gateup_rows = TextConfig::mtp_mlp_gate_up_rows;
    static constexpr float rms_eps           = TextConfig::rms_epsilon;
    static constexpr float rope_theta        = TextConfig::rope_theta;
    static constexpr int mtp_layers          = TextConfig::mtp_layers;

    [[nodiscard]] static constexpr bool is_full(int layer) {
        return TextConfig::is_full_attention(layer);
    }

    [[nodiscard]] static constexpr int n_full() { return TextConfig::full_attention_layers(); }

    [[nodiscard]] static constexpr int n_gdn() { return TextConfig::gdn_layers(); }

    [[nodiscard]] static constexpr int full_idx(int layer) {
        return TextConfig::full_attention_index(layer);
    }

    [[nodiscard]] static constexpr int gdn_idx(int layer) { return TextConfig::gdn_index(layer); }

    // Generic architecture knobs: detection-idiom defaults keep existing
    // behavior when an arch TextConfig does not declare them; arch configs
    // that declare per-layer rope theta / output multiplier / softcap are
    // honored automatically (auto-adaptation contract).
    template <class TC>
    static constexpr float rope_theta_at_impl(int layer) {
        if constexpr (requires { TC::rope_theta_at(0); }) {
            return TC::rope_theta_at(layer);
        } else {
            return TC::rope_theta;
        }
    }
    template <class TC>
    static constexpr int layer_of_full_impl(int fidx) {
        if constexpr (requires { TC::layer_of_full_index(0); }) {
            return TC::layer_of_full_index(fidx);
        } else {
            return fidx * qwen3_6::kHybridAttentionInterval +
                   (qwen3_6::kHybridAttentionInterval - 1);
        }
    }
    template <class TC>
    static constexpr float output_multiplier_impl() {
        if constexpr (requires { TC::output_multiplier(); }) {
            return TC::output_multiplier();
        } else {
            return 1.0f;
        }
    }
    template <class TC>
    static constexpr float softcap_impl() {
        if constexpr (requires { TC::final_logit_softcapping; }) {
            return TC::final_logit_softcapping;
        } else {
            return 0.0f;
        }
    }
    // ------------------------------------------------------------------ //
    // The three hooks the Spark-X2.5-4B artifact needs and nothing read.
    // Named, target-visible, and opt-in: every one is a compile-time NO-OP for an
    // arch that does not declare it, so the four other targets that share this
    // runtime (muse_glimmer_30b / qwen3_6_27b / qwen3_6_35b_a3b / qwen3_5_9b) keep
    // the exact call sequence and the exact constants.  The readings that prove it
    // are dl/sparkhooks/logs/c01_compile.log; the reason each hook exists is
    // dl/sparktarget/REPORT.md section 5 (A), (B), (C).
    // ------------------------------------------------------------------ //
    //
    // (A) q/k rmsnorm at the attention entry is UNCONDITIONAL today
    //     (text_context_impl.h:1204-1205).  Where a checkpoint has no q_norm/k_norm
    //     object AT ALL, an all-ones stand-in is not the fix: rmsnorm still divides
    //     by the RMS whatever the weight is.  `FullAttentionWeights::query_norm` is
    //     a Tensor BY VALUE and `FullLayerW` takes its ADDRESS unconditionally
    //     (text_context_impl.h:593-594), so there is no "leave it unset" that means
    //     "skip".  Muse declares qk_norm_enabled() == true
    //     (muse_glimmer_30b/impl/config.h:74) and the qwen3 family declares nothing,
    //     so THE DEFAULT IS true and only an arch that says `false` skips.
    //     spark_x2_5_4b says false (impl/config.h:268), and its checkpoint backs it:
    //     no q_norm/k_norm key in config.json, no such object among the artifact's
    //     363 tensors.
    template <class TC>
    static constexpr bool qk_norm_impl() {
        if constexpr (requires { TC::qk_norm_enabled(); }) {
            return TC::qk_norm_enabled();
        } else {
            return true;
        }
    }
    [[nodiscard]] static constexpr bool qk_norm() { return qk_norm_impl<TextConfig>(); }
    //
    // (B) rope width and base are single-valued today: one `rotary_dim` constexpr
    //     slot (this file, :44) and one `rope_theta` (this file, :54), read at one
    //     call site (text_context_impl.h:1212-1214).  `rope_theta_at` above is
    //     defined and called from NOWHERE.  Spark rotates 27 layers at 256 @ 10000
    //     and 9 at 64 @ 5000000 -- and those are NOT copied from a sibling, they are
    //     the checkpoint's own declaration: config.json `rope_parameters`
    //     {sliding_attention: partial_rotary_factor 1.0, rope_theta 10000} and
    //     {full_attention: partial_rotary_factor 0.25, rope_theta 5000000}, against
    //     `head_dim: 256` -- i.e. 1.0 x 256 = 256 and 0.25 x 256 = 64.
    //
    //     THE TRIGGER IS `rotary_dim_at`, NOT `requires { TC::rope_theta_at(0); }`.
    //     muse_glimmer_30b -- which HAS a runtime -- declares rope_theta_at
    //     (muse_glimmer_30b/impl/config.h:97) and returns 0.0F on 13 of its 52
    //     layers (NoPE; counted and asserted at muse_glimmer_30b/impl/config.h:118-124).
    //     Consuming that accessor on sight would hand those 13 layers theta == 0.0F,
    //     and ops::rope refuses a non-positive theta by name
    //     (src/ops/wrapper/rope.cpp:66-68): a hard regression on Muse.  `rotary_dim_at`
    //     has no such holder -- the only two archs that declare it are gemma4_31b
    //     (identity-only, no runtime; dl/heritage/REPORT.md 3.4 row 1) and
    //     spark_x2_5_4b.
    // ---- (B) THE PER-LAYER TRIGGER TESTS THE DECLARATION, NOT A PROXY FOR IT ----
    // PRE  was `requires { TC::rotary_dim_at(0); }` ALONE, and muse_glimmer_30b -- which HAS a
    // runtime -- declares `rope_theta_at` and NOT `rotary_dim_at`, so the trigger read FALSE and
    // `rope_theta_for()` handed EVERY layer the global `TextConfig::rope_theta`. On muse's 13
    // NoPE layers (its own config.h asserts `rope_theta_at(l) == 0.0F` for exactly 13 of 52)
    // that rotated a layer the model declares MUST NOT be rotated, by a theta the arch never
    // declared for it: rc=0 and a wrong number, silently. The trigger now reads BOTH accessors,
    // because either one is the arch declaring that this is a per-layer model, and
    // `rope_is_nope()` below carries the 0.0F case to its own leaf instead of to ops::rope --
    // which refuses a non-positive theta by name (src/ops/wrapper/rope.cpp:66-68), so the NoPE
    // path had never been drivable at all.
    template <class TC>
    static constexpr bool per_layer_rope_impl() {
        return requires { TC::rotary_dim_at(0); } || requires { TC::rope_theta_at(0); };
    }
    [[nodiscard]] static constexpr bool per_layer_rope() {
        return per_layer_rope_impl<TextConfig>();
    }
    // Whether THIS arch declares a per-layer theta at all. Only such an arch can have a NoPE
    // layer, and the skip below is compiled only for it, so every other arch keeps the exact
    // call sequence it had.
    template <class TC>
    static constexpr bool rope_declares_theta_impl() { return requires { TC::rope_theta_at(0); }; }
    [[nodiscard]] static constexpr bool rope_declares_theta() {
        return rope_declares_theta_impl<TextConfig>();
    }
    // THE NoPE GATE. A layer whose DECLARED theta is 0.0F is a layer the reference derivation
    // passes NO position embeddings for (`position_embeddings if layer_rope_theta[i] else None`):
    // it must receive no rotation at all, which is a different thing from being rotated by the
    // global default. The population is asserted by the arch's own static_assert (muse: 13).
    template <class TC>
    static constexpr bool rope_is_nope_impl(int fidx) {
        if constexpr (requires { TC::rope_theta_at(0); }) {
            return per_layer_rope_impl<TC>() &&
                   TC::rope_theta_at(layer_of_full_impl<TC>(fidx)) == 0.0F;
        }
        return false;
    }
    [[nodiscard]] static constexpr bool rope_is_nope(int fidx) {
        return rope_is_nope_impl<TextConfig>(fidx);
    }
    // Same detection idiom as rope_theta_at_impl above: an arch that declares the
    // accessor gets it; every other arch keeps the single TextConfig::rotary_dim
    // slot it has always used.
    template <class TC>
    static constexpr int rotary_dim_at_impl(int layer) {
        if constexpr (requires { TC::rotary_dim_at(0); }) {
            return TC::rotary_dim_at(layer);
        } else {
            return TC::rotary_dim;
        }
    }
    [[nodiscard]] static constexpr int rotary_dim_at(int layer) {
        return rotary_dim_at_impl<TextConfig>(layer);
    }
    // `fidx` is a FULL-layer index -- the coordinate `layer_of_full` and the call
    // site use -- while the per-kind tables are indexed by MODEL layer, so the
    // composition goes through layer_of_full and never through fidx directly.
    // When the arch does not opt in these are the two constants the call site used
    // before, and `if constexpr` means the per-layer branch is not instantiated.
    [[nodiscard]] static constexpr int rope_dim_for(int fidx) {
        if constexpr (per_layer_rope()) {
            return rotary_dim_at(layer_of_full(fidx));
        } else {
            return TextConfig::rotary_dim;
        }
    }
    [[nodiscard]] static constexpr float rope_theta_for(int fidx) {
        if constexpr (per_layer_rope()) {
            return rope_theta_at(layer_of_full(fidx));
        } else {
            return TextConfig::rope_theta;
        }
    }
    //
    // (C) the attention-output gate.  The family hands it to the leaf as a
    //     {head_dim, n_q, T} view flattened to {q_size, T}, and then to
    //     ops::sigmoid_mul (text_context_impl.h:1284).  sigmoid_mul picks its
    //     headwise route BY SHAPE -- `headwise_gate_shape`, src/ops/wrapper/
    //     sigmoid_mul.cpp:34-37, dispatched at :48 -- and the contract it serves is
    //     stated in src/ops/launcher/sigmoid_gate_mul.h:16-19: x is [head_dim, H, T]
    //     and gate is [H, T], one sigmoid per (head, token) broadcast over the
    //     head_dim axis.  Spark's gate is 16 rows, not 4096 (`headwise_attn_output_
    //     gate: true`, `gate_attn_act_mode: "sigmoid"`, `num_attention_heads: 16`),
    //     and 256 == 16 is FALSE, so a {256,16,T} gate falls through to the
    //     PER-ELEMENT route -- which ACCEPTS the pair and multiplies `a` by gate rows
    //     16..4095, numbers this target never wrote.  Silent wrong number.  The
    //     routing decision IS the shape, so the family must hand a headwise gate over
    //     2-D, and the two declarations are cross-checked at namespace scope just
    //     below the class (see the static_assert after kCfg).
    template <class TC>
    static constexpr int gate_rows_impl() {
        if constexpr (requires { TC::attention_gate_rows; }) {
            return TC::attention_gate_rows;
        } else {
            return TC::query_size;
        }
    }
    [[nodiscard]] static constexpr int gate_rows() { return gate_rows_impl<TextConfig>(); }
    // ---- (C') THE GATE IS DECIDED BY THE ARCH'S OWN DECLARATIONS, NOT BY A SHAPE ----
    // `headwise_gate_declared_impl` / `gate_rows_declared_impl` exist so that "the arch declares
    // false" and "the arch declares NOTHING" are DISTINGUISHABLE FACTS: that difference is the
    // whole defect this reader closes (impl/runtime/attn_output_gate_route.h (A)). The route is
    // read from the table in that header by the arch's DECLARED GATE TRIPLE; a triple with no row
    // is REFUSED BY NAME by the namespace-scope static_assert below and, at runtime, by
    // `gate_route_refusal()`.
    template <class TC>
    static constexpr bool headwise_gate_declared_impl() {
        return requires { TC::headwise_attn_output_gate_enabled(); };
    }
    template <class TC>
    static constexpr bool gate_rows_declared_impl() { return requires { TC::attention_gate_rows; }; }
    template <class TC>
    static constexpr bool headwise_gate_impl() {
        if constexpr (headwise_gate_declared_impl<TC>()) {
            return TC::headwise_attn_output_gate_enabled();
        } else {
            return false;
        }
    }
    [[nodiscard]] static constexpr bool headwise_gate_declared() {
        return headwise_gate_declared_impl<TextConfig>();
    }
    [[nodiscard]] static constexpr bool gate_rows_declared() {
        return gate_rows_declared_impl<TextConfig>();
    }
    [[nodiscard]] static constexpr attn_output_gate_route::GateRoute gate_route() {
        return attn_output_gate_route::gate_route_for(
            headwise_gate_declared_impl<TextConfig>(), headwise_gate_impl<TextConfig>(),
            gate_rows_declared_impl<TextConfig>(), gate_rows_impl<TextConfig>(),
            TextConfig::query_size);
    }
    [[nodiscard]] static constexpr bool headwise_gate() {
        return gate_route() == attn_output_gate_route::GateRoute::Headwise;
    }
    [[nodiscard]] static std::string gate_route_refusal() {
        return attn_output_gate_route::attn_output_gate_refusal(
            headwise_gate_declared_impl<TextConfig>(), headwise_gate_impl<TextConfig>(),
            gate_rows_declared_impl<TextConfig>(), gate_rows_impl<TextConfig>(),
            TextConfig::query_size);
    }
    [[nodiscard]] static constexpr float rope_theta_at(int layer) {
        return rope_theta_at_impl<TextConfig>(layer);
    }
    [[nodiscard]] static constexpr int layer_of_full(int fidx) {
        return layer_of_full_impl<TextConfig>(fidx);
    }
    [[nodiscard]] static constexpr float output_multiplier() {
        return output_multiplier_impl<TextConfig>();
    }
    [[nodiscard]] static constexpr float final_logit_softcapping() {
        return softcap_impl<TextConfig>();
    }

    // Auto-adaptation contract: every lm_head logits production site applies
    // the architecture's final-logit policy (output_multiplier, then tanh
    // softcap when final_logit_softcapping > 0). Compile-time no-op for
    // architectures that declare neither (qwen3 family keeps zero overhead).
    static void apply_final_logit_policy(Tensor& logits, cudaStream_t stream) {
        if constexpr (final_logit_softcapping() > 0.0F || output_multiplier() != 1.0F) {
            ops::logit_policy(logits, output_multiplier(), final_logit_softcapping(), stream);
        }
    }

    // Auto-adaptation: double-norm layer graphs (e.g. Muse: the attention
    // output is normalized again before the residual add, and the MLP output
    // likewise). Compile-time no-ops for single-norm architectures (qwen3
    // family keeps the exact previous call sequence).
    template <class TC>
    static constexpr bool double_norm_attn_impl() {
        if constexpr (requires { TC::attn_out_post_norm(); }) {
            return TC::attn_out_post_norm();
        }
        return false;
    }
    template <class TC>
    static constexpr bool double_norm_mlp_impl() {
        if constexpr (requires { TC::mlp_out_post_norm(); }) {
            return TC::mlp_out_post_norm();
        }
        return false;
    }
    template <class TC>
    static constexpr float post_eps_impl() {
        if constexpr (requires { TC::post_norm_eps; }) {
            return TC::post_norm_eps;
        }
        return TC::rms_epsilon;
    }
    [[nodiscard]] static constexpr bool attn_out_double_norm() {
        return double_norm_attn_impl<TextConfig>();
    }
    [[nodiscard]] static constexpr bool mlp_out_double_norm() {
        return double_norm_mlp_impl<TextConfig>();
    }
    [[nodiscard]] static constexpr float post_norm_epsilon() {
        return post_eps_impl<TextConfig>();
    }
};

inline constexpr ModelConfig kCfg{};
// THE BRIDGE TO ops::sigmoid_mul IS A SHAPE, AND THE ROUTE IS NOT: sigmoid_mul chooses by shape
// (src/ops/wrapper/sigmoid_mul.cpp's `headwise_gate_shape`, then the per-element loop), so an arch
// that says "headwise" while leaving its gate query_size wide would silently take the PER-ELEMENT
// route and read gate rows it never wrote. Refused at compile time, at namespace scope because the
// class is not complete inside its own definition (see the two rejected placements recorded in
// dl/sparkhooks/sh/b01_patch.py).
//
// * THIS FIRST ASSERT IS THE PRE GUARD, INVERTED, AND THE INVERSION IS THE FIX (accfix F910). *
// PRE  was  `!headwise_gate() || gate_rows() != query_size`, whose arming condition IS the
// accessor that may be missing: an arch that OMITS `headwise_attn_output_gate_enabled()` reads
// `false`, the guard is satisfied VACUOUSLY, and the omission DISARMS THE GUARD IT EXISTS FOR.
// POST arms it on the DECLARATION OF THE GATE instead: an arch that declares a row count the
// per-element route cannot accept must be able to say why, and if it cannot, the build stops here
// with the arch's own declaration in the message.
static_assert(!ModelConfig::gate_rows_declared() ||
                  ModelConfig::gate_rows() == TextConfig::query_size ||
                  ModelConfig::headwise_gate(),
              "this arch DECLARES attention_gate_rows with a value the per-element route cannot "
              "accept, which IS a headwise gate, but it does not declare "
              "headwise_attn_output_gate_enabled(). Add that declaration, or declare the gate row "
              "count as query_size. THE GUARD IS ARMED BY THE GATE'S OWN DECLARATION, NOT BY THE "
              "DECLARATION OF HEADWISE-NESS: the PRE guard's arming condition was the very "
              "accessor whose omission it existed to catch.");
// And the PRE clause is KEPT -- a headwise gate must still have a row count the per-element route
// cannot accept, or the two legal shapes would both reach it.
static_assert(!ModelConfig::headwise_gate() ||
                  ModelConfig::gate_rows() != TextConfig::query_size,
              "a headwise output gate must have a row count the per-element route "
              "cannot accept, or src/ops/wrapper/sigmoid_mul.cpp:34-37 routes it "
              "elementwise and reads gate rows the target never wrote");
// THE REGISTRATION. The identity is the DECLARED GATE TRIPLE and not the geometry -- (16 q / 4 kv /
// 256) is both spark_x2_5_4b (headwise) and qwen3_5_9b (per-channel). A triple with no row is
// REFUSED HERE, by name, with the arch's own declaration and every registered row in the message.
static_assert(ModelConfig::gate_route() != attn_output_gate_route::GateRoute::Unregistered,
              "attn output gate: this arch's DECLARED GATE TRIPLE has no registered row (see "
              "impl/runtime/attn_output_gate_route.h). Before this seam an absent headwise "
              "accessor took a SILENT DEFAULT of false and the launch returned rc=0 with a "
              "different answer; the resting state is now a refusal that names the fingerprint, "
              "so add the arch's row to kRegisteredAttnOutputGates (with the arch's own "
              "impl/config.h as its evidence) or fix the declarations it disagrees with.");
inline constexpr float kAttnScale                     = kAttentionScale;
inline constexpr std::uint32_t kPrefillChunkAlignment = 128;

struct MlpW {
    const MlpWeights* payload = nullptr;
    // Double-norm layer graphs: normalization applied to the MLP output
    // before the residual add (Muse post_feedforward_layernorm). Null for
    // single-norm architectures (qwen3 family).
    const Tensor* post_ff_norm = nullptr;
};

struct FullLayerW {
    const Tensor* input_norm                         = nullptr;
    const FullAttentionProjectionWeights* projection = nullptr;
    const Weight* o_proj                             = nullptr;
    const Tensor* q_norm                             = nullptr;
    const Tensor* k_norm                             = nullptr;
    const Tensor* post_attn_norm                     = nullptr;
    // Double-norm layer graphs: normalization applied to the attention
    // output (o_proj result) before the residual add (Muse
    // post_attention_layernorm). Null for single-norm architectures.
    const Tensor* post_attn_out_norm                 = nullptr;
    MlpW mlp;
};

struct GdnLayerW {
    const Tensor* input_norm               = nullptr;
    const GdnProjectionWeights* projection = nullptr;
    const Tensor* conv1d                   = nullptr;
    const Tensor* gdn_norm                 = nullptr;
    const Weight* out_proj                 = nullptr;
    const Tensor* post_attn_norm           = nullptr;
    MlpW mlp;
};

struct MtpW {
    const MtpWeights* payload           = nullptr;
    const Weight* fc                    = nullptr;
    const Tensor* pre_fc_norm_embedding = nullptr;
    const Tensor* pre_fc_norm_hidden    = nullptr;
    const Tensor* input_norm            = nullptr;
    const Tensor* q_norm                = nullptr;
    const Tensor* k_norm                = nullptr;
    const Weight* o_proj                = nullptr;
    const Tensor* post_attn_norm        = nullptr;
    const Tensor* norm                  = nullptr;
};

using Phase = qwen3_6::TextPhase;

enum class GdnStateAction : std::uint8_t {
    UpdateInPlace,
    RecordForReplay,
};

struct NullTap {
    static constexpr bool enabled = false;
};

struct PrefillChunkResult {
    std::uint32_t processed_tokens = 0;
    bool finalized                 = false;
    runtime::ExecutionTiming timing;
};

struct DFlashFeatureSink {
    static constexpr bool enabled = true;
    using PrefillConsumer         = std::function<void(const Tensor&, const Tensor&, bool)>;

    Tensor* features                  = nullptr;
    Tensor* positions                 = nullptr;
    Tensor* batch_features            = nullptr;
    const Tensor* batch_lanes         = nullptr;
    const Tensor* batch_valid_columns = nullptr;
    std::int32_t batch_width          = 0;
    std::int32_t batch_size           = 0;
    std::span<const int> layers;
    PrefillConsumer consume_prefill;
    std::uint32_t captured_mask = 0;
    std::int32_t active_tokens  = 0;

    void begin(const Tensor& value);
    void capture_layer(int layer, const Tensor& value, cudaStream_t stream);
    void capture_positions(const Tensor& source, cudaStream_t stream);
    void consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint);
};

class VisionPrefillSession;

class TextContext {
public:
    // --stage-layers SPEC / --stage-handoff DIR / --stage-handoff-cut. THE entry point of the
    // layer-range surface: parses the raw spec with core/stage_plan.h (the one parser), then
    // refuses BY NAME -- before a single layer is walked -- a spec that is mis-shaped
    // (refused-stage-layers), that is not a cover of this model's text layers
    // (refused-stage-layers-partition), or that is not a shard of the world the RANK AXIS
    // derives for THIS model's geometry (refused-stage-layers-axis, decided by plan_shards()
    // itself rather than by a second derivation of the balanced split). `spec` empty is the
    // flag-absent state and is a no-op.
    //
    // It does NOT call validate_virtual_request() and it does not move that guard's answer:
    // for axis=pp that guard returns active = 0 and STAYS 0. This makes pp REACHABLE, not
    // universally SUPPORTED -- the run shapes a partial range cannot carry are refused below
    // in core/stage_plan.h's stage_layers_run_shape_refusal().
    void set_stage_layers_spec(std::string_view spec, std::string handoff_dir, bool handoff_cut);
    void set_graph_segment(std::int32_t segment, std::int32_t segments) noexcept {
        active_graph_segment_  = segment;
        active_graph_segments_ = segments;
    }
    TextContext(DeviceContext& ctx, const LoadedModelData& weights, WorkspaceArena& work,
                qwen3_6::PagedKVCacheView kv, LinearAttentionStatePool& state,
                qwen3_6::RoundState& io, Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                std::uint32_t text_kv_base,
                qwen3_6::PagedKVCacheView mtp_kv           = qwen3_6::PagedKVCacheView(),
                const qwen3_6::PagedKVCache* batch_text_kv = nullptr,
                const qwen3_6::PagedKVCache* batch_mtp_kv  = nullptr);
    ~TextContext();

    TextContext(const TextContext&)            = delete;
    TextContext& operator=(const TextContext&) = delete;

    void set_proposal_head(const Weight* weight, const std::int32_t* ids, int count) noexcept {
        proposal_head_     = weight;
        proposal_head_ids_ = ids;
        proposal_head_n_   = count;
    }

    void set_sampling(const ops::SamplingConfig* config) noexcept { sampling_config_ = config; }

    void set_prefill_split_frontier(std::int64_t position) noexcept {
        prefill_split_frontier_ = position;
    }

    void set_rewrite_checkpoint_hidden_output(Tensor* output) noexcept {
        rewrite_checkpoint_hidden_output_ = output;
    }

    void set_mtp_proposal_extent(std::uint32_t extent) noexcept { mtp_proposal_extent_ = extent; }

    // --draft-tree L,d (L > 1): the tree shape THIS prefill's proposal loop must also publish a
    // lattice for (MtpPrefillState::lattice_ids), beside the per-depth draft token it already
    // produces. The decode loop's own lattice (MtpDecodeEgress::next_proposal_ids) is published by
    // mtp_impl.h; the round-1 lattice has to come from the PREFILL proposal block, which is
    // TextContext::prefill_chunk -- see extract_mtp_proposal_lattice(). paths <= 1 / depth == 0 is
    // the chain spelling and launches no extra work at all, which keeps --draft-tokens
    // bit-identical.
    void set_mtp_tree_shape(std::uint32_t paths, std::uint32_t depth) noexcept {
        mtp_tree_paths_ = paths;
        mtp_tree_depth_ = depth;
    }

    void set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot);
    void set_gdn_state_action(GdnStateAction action, const GdnReplayRecords* replay_records);

    [[nodiscard]] const Weight* proposal_head() const noexcept { return proposal_head_; }

    [[nodiscard]] const std::int32_t* proposal_head_ids() const noexcept {
        return proposal_head_ids_;
    }

    [[nodiscard]] int proposal_head_n() const noexcept { return proposal_head_n_; }

    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    [[nodiscard]] PrefillChunkResult
    prefill_chunk(const qwen3_6::PreparedPromptData& input, std::uint32_t begin,
                  std::uint32_t nominal_length, VisionPrefillSession& vision, bool finalize_at_end);
    void ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                               const Tensor& rope_positions, const Tensor& kv_table_rows,
                               const Tensor& linear_state_source_slots,
                               const Tensor& linear_state_destination_slots,
                               ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                               Tensor& logits);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& column_masks, const Tensor& kv_table_rows,
                             const Tensor& linear_state_source_slots,
                             ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& column_masks, const Tensor& kv_table_rows,
                             const Tensor& linear_state_source_slots,
                             ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens, DFlashFeatureSink& sink);
    void mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                  const Tensor& cache_positions, const Tensor& rope_positions,
                                  const Tensor& valid_columns, const Tensor& kv_table_rows,
                                  ops::GqaExecutionEnvelope envelope,
                                  Tensor& mtp_hidden);
    void mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens);
    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           ops::GqaExecutionEnvelope envelope, Tensor& mtp_hidden,
                           int logits_column, Tensor* logits, Tensor* draft_token,
                           const Tensor* explicit_rope_positions = nullptr,
                           const Tensor* input_embeddings        = nullptr);
    void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                             const Tensor& position, ops::GqaExecutionEnvelope envelope,
                             Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token);
    // --draft-tree L,d (L > 1): publish the depth-`depth` top-L candidate rows of the SAME logits
    // window that just produced the depth-`depth` draft token, into the prefill frame's
    // lattice_ids. A no-op for the chain spelling and for a depth past the tree's own depth; the
    // definition states why this belongs to the proposal block and not to the bridge.
    void extract_mtp_proposal_lattice(Tensor& logits, std::uint32_t depth);

private:
    void bind();

    [[nodiscard]] bool mtp_enabled() const noexcept {
        return mtp_kv_.valid() || batch_mtp_kv_ != nullptr;
    }

    [[nodiscard]] const MtpW& mtp_weights() const;
    void attn_mix(const FullLayerW& weights, Tensor& x, int index, Phase phase);
    void gdn_mix(const GdnLayerW& weights, Tensor& x, int index, Phase phase);
    void mlp_tail(const Tensor* post_norm, const MlpW& weights, Tensor& x, Phase phase);
    void run_layers(Tensor& x, Phase phase);
    template <class Tap>
    void run_layers(Tensor& x, Phase phase, Tap& tap);
    template <class Tap>
    void target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                  const Tensor& rope_positions, const Tensor& valid_columns,
                                  const Tensor& column_masks, const Tensor& kv_table_rows,
                                  const Tensor& linear_state_source_slots,
                                  ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                  Tensor& logits, Tensor& target_tokens, Tap& tap);

    void mtp_forward_stem(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                          Tensor& x, Tensor& ah);
    void mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::GqaExecutionEnvelope envelope, Tensor& mtp_hidden);
    void mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::GqaExecutionEnvelope envelope, Tensor& mtp_hidden,
                          const Tensor* input_embeddings);
    void mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                           const Tensor& positions, const Tensor& rope_positions,
                           ops::GqaExecutionEnvelope envelope, bool final_chunk,
                           Tensor* final_hidden, Tensor* logits, Tensor* draft_token);
    void proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens);

    struct MultimodalPrefill {
        std::span<const int> token_ids;
        std::span<const std::int32_t> positions;
        VisionPrefillSession* vision = nullptr;
        std::uint32_t begin          = 0;
        std::int32_t rope_delta      = 0;
    };

    struct TextPrefill {
        std::span<const int> token_ids;
        std::uint32_t begin = 0;
    };

    template <class Tap>
    [[nodiscard]] PrefillChunkResult
    prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                 const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end);
    DeviceContext& ctx_;
    const LoadedModelData& weights_;
    WorkspaceArena& work_;
    qwen3_6::PagedKVCacheView kv_;
    qwen3_6::PagedKVCacheView mtp_kv_;
    const qwen3_6::PagedKVCache* batch_text_kv_ = nullptr;
    const qwen3_6::PagedKVCache* batch_mtp_kv_  = nullptr;
    LinearAttentionStatePool& state_;
    qwen3_6::RoundState& io_;
    Tensor& prefill_hidden_;
    std::uint32_t prefill_chunk_;
    std::uint32_t text_kv_base_;
    const Tensor* active_cache_positions_                                          = nullptr;
    const Tensor* active_rope_positions_                                           = nullptr;
    const Tensor* active_kv_table_rows_                                            = nullptr;
    const Tensor* active_linear_state_source_slots_                                = nullptr;
    const Tensor* active_linear_state_destination_slots_                           = nullptr;
    const Tensor* active_valid_columns_                                            = nullptr;
    // M1: the per-column ancestor masks of a tree verify round. Null = no tree this round.
    const Tensor* active_column_masks_                                             = nullptr;
    const Tensor* active_backend_kv_table_rows_                                    = nullptr;
    const ops::GqaExecutionEnvelope* active_causal_attention_envelope_ = nullptr;
    std::int32_t active_sequence_batch_                                            = 0;
    std::int32_t active_sequence_width_                                            = 0;
    // Optional layer window for segmented graph capture. last < first disables.
    std::int32_t active_layer_first_                                               = 0;
    std::int32_t active_layer_last_                                                = -1;
    // --stage-layers: the parsed stage partition. `requested == false` (the default, the
    // flag-absent state) makes the layer walk in run_layers() exactly `0 .. kCfg.n_layers - 1`,
    // i.e. byte-for-byte the single walk it was before this surface existed.
    multi::StagePlan stage_plan_;
    std::string stage_handoff_dir_;
    bool stage_handoff_cut_ = false;
    // The seam's payload, written by the producing stage and read by the consuming one: raw
    // bytes of the hidden tensor plus a header carrying a magic, the producing layer, the
    // element count and an FNV-1a, so a payload left over from another prompt is DETECTABLE
    // rather than silently consumed.
    void stage_handoff_read(Tensor& x);
    void stage_handoff_write(Tensor& x, int layer_last);
    // Segmented decode state: when active_graph_segments_ > 1 each
    // ordinary_decode_batch call executes only its slice of the forward pass.
    std::int32_t active_graph_segment_                                             = 0;
    std::int32_t active_graph_segments_                                            = 1;
    std::int32_t rope_delta_                                                       = 0;
    std::int32_t linear_state_source_slot_                                         = 0;
    std::int32_t linear_state_destination_slot_                                    = 0;
    GdnStateAction gdn_state_action_          = GdnStateAction::UpdateInPlace;
    const GdnReplayRecords* replay_records_   = nullptr;
    std::int64_t prefill_split_frontier_      = -1;
    Tensor* rewrite_checkpoint_hidden_output_ = nullptr;
    std::uint32_t mtp_proposal_extent_        = 0;
    // --draft-tree L,d (L > 1): see set_mtp_tree_shape(). Read only by extract_mtp_proposal_lattice().
    std::uint32_t mtp_tree_paths_             = 0;
    std::uint32_t mtp_tree_depth_             = 0;

    const Weight* embed_                        = nullptr;
    const Tensor* final_norm_                   = nullptr;
    const Weight* lm_head_                      = nullptr;
    const Weight* proposal_head_                = nullptr;
    const std::int32_t* proposal_head_ids_      = nullptr;
    int proposal_head_n_                        = 0;
    const ops::SamplingConfig* sampling_config_ = nullptr;
    MtpW mtp_;
    std::array<FullLayerW, TextConfig::full_attention_layers()> full_{};
    std::array<GdnLayerW, TextConfig::gdn_layers()> gdn_{};
    std::array<Weight, TextConfig::gdn_layers()> gdn_in_a_{};
    std::array<Weight, TextConfig::gdn_layers()> gdn_in_b_{};
    std::array<Tensor, TextConfig::gdn_layers()> gdn_conv1d_views_{};
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule

