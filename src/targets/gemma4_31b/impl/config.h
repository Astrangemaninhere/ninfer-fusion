#pragma once

// src/targets/gemma4_31b/impl/config.h
//
// Measured geometry for Gemma-4-31B. Nothing here is inferred from another
// model's layout; every constant names the checkpoint artifact or the tree file
// it was read off, and every derived quantity is pinned by a static_assert that
// has a second, independent source (the same idiom
// src/targets/qwen4_exp/impl/config.h:127-159 uses).
//
// Provenance, per group
// ---------------------
//   * geometry, schedule, kinds, rope:
//     tools/convert/gemma4_31b/inventory.py:78-111
//     (read off data/gemma4-31B/model.safetensors' own 2418-tensor header and
//     tools/archkit/specs/gemma4-31b_spec.json:20-95).
//   * the front door's config agreement check, which is the other side of these
//     numbers: tools/convert/gemma4_31b/convert.py:96-135 (_ROOT_CONFIG,
//     _TEXT_GEOMETRY) and :186-224 (the layer_types and rope_parameters gates).
//   * the two attention geometries are stated verbatim in
//     tools/convert/gemma4_31b/convert.py:43-45:
//       "two attention geometries in one checkpoint (32/16/256 on the 50 sliding
//        layers, 32/4/512 on the full ones). Neither triple is in the engine's
//        GQA registry"
//
// WHY config.h CARRIES PER-KIND TABLES AND NOT ONE FLATTENED GEOMETRY
// ------------------------------------------------------------------
// `TextConfig` in the sibling targets has exactly one `query_heads`, one
// `kv_heads`, one `head_dim` and one `rotary_dim`
// (src/targets/muse_glimmer_30b/impl/config.h:31-34). Gemma-4-31B needs two of
// each inside one checkpoint, so a flattened TextConfig cannot represent it and
// the two static_asserts at the bottom of this file fail the BUILD the day a
// flattened table is written here. That is deliberate: the alternative is a
// target that loads and then computes the ten full layers with the sliding
// layers' head width, which is a silent wrong number, not a refusal.
//
// This is also why tools/archkit/gen_full_target.py refuses to emit a config.h
// for this spec at all (its :78-82 raises when partial_rotary_factor is declared
// per layer kind while TextConfig has a single rotary_dim; the report is
// `per layer kind (...) but TextConfig has a single rotary_dim`).

#include <array>
#include <cstdint>

namespace ninfer::targets::gemma4_31b::detail {

// Layer kind vocabulary, the kind numbering the family's emitters already use
// (tools/archkit/gen_full_target.py:21-23).
inline constexpr int kKindFull    = 0;
inline constexpr int kKindSliding = 1;
inline constexpr int kKindGdn     = 2;

struct TextConfig {
    // --------------------------------------------------------------------- //
    // shared geometry
    // --------------------------------------------------------------------- //
    static constexpr int hidden       = 5376;      // inventory.py:78
    static constexpr int layers       = 60;        // inventory.py:79, spec:11
    static constexpr int intermediate = 21504;     // inventory.py:80
    static constexpr int vocab        = 262144;    // inventory.py:81
    static constexpr int max_ctx      = 262144;    // inventory.py:82
    static constexpr float rms_epsilon = 1e-06F;   // inventory.py:83
    static constexpr float final_logit_softcapping = 30.0F;   // inventory.py:86
    // inventory.py:85 `TIE_WORD_EMBEDDINGS = True`; the source ships no lm_head
    // tensor, so the output head is a materialised copy of the embedding
    // (inventory.py:36-39, tools/archkit/gemma_engine_plan.md:56-58).
    static constexpr bool tie_word_embeddings = true;
    // convert.py:124 HIDDEN_ACTIVATION. The family's post-mixer leaf is the
    // swiglu/gelu body, so this is a leaf-level property, declared here so it is
    // not re-derived from a checkpoint later.
    static constexpr const char* hidden_activation = "gelu_pytorch_tanh";
    // convert.py:125 `attention_bias: False` -- the quantised projections carry
    // no bias rows.
    static constexpr bool attention_bias = false;

    // --------------------------------------------------------------------- //
    // the 50 sliding_attention layers (inventory.py:89-92, :104-111)
    // --------------------------------------------------------------------- //
    static constexpr int sliding_query_heads = 32;
    static constexpr int sliding_kv_heads    = 16;
    static constexpr int sliding_head_dim    = 256;
    static constexpr int sliding_rotary_dim  = 256;     // ROTARY_DIM_BY_KIND["sliding"]
    static constexpr float sliding_rope_theta = 10000.0F;  // spec:90-93, rope_type default
    static constexpr int sliding_window        = 1024;     // inventory.py:84

    // --------------------------------------------------------------------- //
    // the 10 full_attention layers -- a SECOND geometry in the same checkpoint
    // (inventory.py:93-95, :104-111; convert.py:43-45, :52-55)
    // --------------------------------------------------------------------- //
    static constexpr int full_query_heads = 32;
    static constexpr int full_kv_heads    = 4;
    static constexpr int full_head_dim    = 512;
    static constexpr int full_rotary_dim  = 128;       // 0.25 * global_head_dim, spec:86
    static constexpr float full_rope_theta = 1000000.0F;  // spec:85-88, rope_type proportional

    // `attention_k_eq_v` is true, so the ten full layers ship no v_proj at all
    // (inventory.py:27-29 measured: 410 quantised linears = 60x6 + 50).
    static constexpr bool attention_k_eq_v = true;

    // --------------------------------------------------------------------- //
    // layer schedule. 0 = full, 1 = sliding. Verbatim from
    // inventory.py:97-98 FULL_ATTENTION_LAYERS = (5, 11, 17, ..., 59), which the
    // front door independently enforces against the checkpoint's own
    // `text_config.layer_types` (convert.py:186-199).
    // --------------------------------------------------------------------- //
    static constexpr std::array<int, 60> layer_kind{
        1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
        1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0};

    [[nodiscard]] static constexpr std::array<int, 10> full_attention_slots() noexcept {
        return {5, 11, 17, 23, 29, 35, 41, 47, 53, 59};
    }
    [[nodiscard]] static constexpr bool is_full_attention(int layer) noexcept {
        return layer_kind[static_cast<std::size_t>(layer)] == kKindFull;
    }
    [[nodiscard]] static constexpr bool is_swa_attention(int layer) noexcept {
        return layer_kind[static_cast<std::size_t>(layer)] == kKindSliding;
    }
    [[nodiscard]] static constexpr int full_attention_layers() noexcept {
        int count = 0;
        for (const int kind : layer_kind) {
            if (kind == kKindFull) { ++count; }
        }
        return count;
    }
    [[nodiscard]] static constexpr int swa_attention_layers() noexcept {
        int count = 0;
        for (const int kind : layer_kind) {
            if (kind == kKindSliding) { ++count; }
        }
        return count;
    }
    [[nodiscard]] static constexpr int gdn_layers() noexcept {
        int count = 0;
        for (const int kind : layer_kind) {
            if (kind == kKindGdn) { ++count; }
        }
        return count;
    }
    [[nodiscard]] static constexpr int full_attention_index(int layer) noexcept {
        int count = 0;
        for (int i = 0; i < layer; ++i) { count += is_full_attention(i) ? 1 : 0; }
        return count;
    }
    [[nodiscard]] static constexpr int swa_attention_index(int layer) noexcept {
        int count = 0;
        for (int i = 0; i < layer; ++i) { count += is_swa_attention(i) ? 1 : 0; }
        return count;
    }
    [[nodiscard]] static constexpr int gdn_index(int) noexcept { return 0; }

    // --------------------------------------------------------------------- //
    // per-kind accessors. THESE ARE THE SURFACE THE FAMILY DOES NOT READ: a
    // variant body would have to call them per layer, which is exactly the
    // engine hook convert.py:156-158 names as WORK_ITEM "per-layer rotary".
    // --------------------------------------------------------------------- //
    [[nodiscard]] static constexpr int head_dim_at(int layer) noexcept {
        return is_full_attention(layer) ? full_head_dim : sliding_head_dim;
    }
    [[nodiscard]] static constexpr int query_heads_at(int layer) noexcept {
        return is_full_attention(layer) ? full_query_heads : sliding_query_heads;
    }
    [[nodiscard]] static constexpr int kv_heads_at(int layer) noexcept {
        return is_full_attention(layer) ? full_kv_heads : sliding_kv_heads;
    }
    [[nodiscard]] static constexpr int query_rows_at(int layer) noexcept {
        return query_heads_at(layer) * head_dim_at(layer);
    }
    [[nodiscard]] static constexpr int kv_rows_at(int layer) noexcept {
        return kv_heads_at(layer) * head_dim_at(layer);
    }
    [[nodiscard]] static constexpr int rotary_dim_at(int layer) noexcept {
        return is_full_attention(layer) ? full_rotary_dim : sliding_rotary_dim;
    }
    [[nodiscard]] static constexpr float rope_theta_at(int layer) noexcept {
        return is_full_attention(layer) ? full_rope_theta : sliding_rope_theta;
    }
    // Sliding layers carry the window; the ten full layers are global by
    // definition and declare 0. A table with a zero entry is inert at the
    // Cold Host tier (src/product/kv_component_switch.h:314-317 requires every
    // entry non-zero before it refuses, and :269-270 records that
    // muse_glimmer_30b's 39-of-52 table is inert for the same reason), so this
    // declaration cannot release a page -- and equally cannot bound attention on
    // the windowed layers. See the "window" note in the static_assert block.
    [[nodiscard]] static constexpr std::uint32_t sliding_window_at(int layer) noexcept {
        return is_full_attention(layer) ? 0U : static_cast<std::uint32_t>(sliding_window);
    }
};

// --------------------------------------------------------------------------- //
// Compile-time self-checks. Each has a second source, so a drift in one place
// fails the build instead of reaching a kernel.
// --------------------------------------------------------------------------- //

// counts: table vs the two independent statements of the schedule.
static_assert(TextConfig::layer_kind.size() == TextConfig::layers);
static_assert(TextConfig::full_attention_layers() == 10);   // convert.py:198 "(10 full, 50 sliding)"
static_assert(TextConfig::swa_attention_layers() == 50);    // convert.py:198
static_assert(TextConfig::gdn_layers() == 0);               // inventory.py:42 "all 60 layers are softmax"

// the literal table vs the arithmetic rule that generates the same slots
// (full_attention_slots() == 5, 11, ..., 59 == every layer with layer % 6 == 5).
static_assert([] {
    for (int layer = 0; layer < TextConfig::layers; ++layer) {
        const bool by_rule = (layer % 6) == 5;
        if (by_rule != TextConfig::is_full_attention(layer)) { return false; }
    }
    return true;
}());

// the literal table vs the exported slot tuple (inventory.py:98).
static_assert([] {
    for (int index = 0; index < static_cast<int>(TextConfig::full_attention_slots().size());
         ++index) {
        const int layer = TextConfig::full_attention_slots()[static_cast<std::size_t>(index)];
        if (!TextConfig::is_full_attention(layer)) { return false; }
        if (TextConfig::full_attention_index(layer) != index) { return false; }
    }
    return true;
}());

// row counts, the quantities the artifact's own tensor shapes are compared
// against (impl/package.cpp). 8192 / 4096 / 16384 / 2048 are the four numbers
// inventory.py:20-26 states as measured (q_proj packed [8192,2688] on layer 0
// and [16384,2688] on layer 5).
static_assert(TextConfig::query_rows_at(0) == 8192);
static_assert(TextConfig::kv_rows_at(0) == 4096);
static_assert(TextConfig::query_rows_at(5) == 16384);
static_assert(TextConfig::kv_rows_at(5) == 2048);
static_assert(TextConfig::full_attention_index(5) == 0);
static_assert(TextConfig::swa_attention_index(5) == 5);

// --------------------------------------------------------------------------- //
// THE TWO BLOCKERS, PINNED SO THE BUILD NAMES THEM
// --------------------------------------------------------------------------- //
// Written as asserts on the *obstacle* rather than as comments: the day someone
// flattens these into one geometry, the build states what was lost instead of
// producing a target that computes ten layers at the wrong width.
//
// BLOCKER 1 -- the full layers' head width cannot be expressed by any kernel.
// src/ops/kernel/gqa_attention_geometry.cuh:15 is
//   static_assert(HeadDimValue == 128 || HeadDimValue == 256);
// and its alias list (:25-32) is exactly {<24,4,256>, <16,2,256>, <32,2,128>,
// <16,4,256>}. Gemma-4-31B needs <32,16,256> for the 50 sliding layers and
// <32,4,512> for the 10 full ones; convert.py:47-51 prints the refusal verbatim.
static_assert(TextConfig::sliding_head_dim == 256);
static_assert(TextConfig::full_head_dim == 512);
static_assert(TextConfig::sliding_head_dim != TextConfig::full_head_dim);
static_assert(TextConfig::sliding_kv_heads != TextConfig::full_kv_heads);

// BLOCKER 2 -- rotary width is per layer kind, and the shared TextConfig has one
// rotary_dim slot (src/targets/muse_glimmer_30b/impl/config.h:34).
static_assert(TextConfig::rotary_dim_at(0) == 256);
static_assert(TextConfig::rotary_dim_at(5) == 128);
static_assert(TextConfig::rotary_dim_at(0) != TextConfig::rotary_dim_at(5));

// THE WINDOW, stated as measured rather than as an intention: the table is 50
// non-zero entries and 10 zeros. Per src/product/kv_component_switch.h:269-270
// an all-non-zero table on a tier that does not read `sliding_window_tokens` is
// refused BY NAME, and a table with any zero is inert. So this table is neither
// refused nor honoured -- the window never bounds attention on the 50 layers
// that declare it. That is the same standing as muse_glimmer_30b
// (tools/archkit/gemma_engine_plan.md:81 TODO-1; muse's config.h:66-68 records
// its 39 sliding layers running as full until the window is wired).
static_assert(TextConfig::sliding_window_at(0) == 1024U);
static_assert(TextConfig::sliding_window_at(5) == 0U);
static_assert([] {
    std::uint32_t windowed = 0;
    std::uint32_t global   = 0;
    for (int layer = 0; layer < TextConfig::layers; ++layer) {
        if (TextConfig::sliding_window_at(layer) == 0U) { ++global; } else { ++windowed; }
    }
    return windowed == 50U && global == 10U;
}());

} // namespace ninfer::targets::gemma4_31b::detail
