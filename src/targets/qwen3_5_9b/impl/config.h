#pragma once

#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/hybrid_topology.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include <array>
#include <cstdint>

// ---------------------------------------------------------------------------
// qwen3_5_9b -- the Qwen3.5-family GDN-hybrid text stack as published in GGUF.
//
// This package exists because a real GGUF declares `general.architecture = qwen35`
// and, before it, no registered target claimed the artifact identity its converter
// produces: the loader answered "no registered target", which is a statement about
// the *registry*, not about the weights.  Every number below is read off the real
// `Ornith-1.5-9B-Q4_K_M.gguf` metadata (5,780,090,816 B), not guessed:
//
//   qwen35.embedding_length            = 4096      -> hidden
//   qwen35.block_count                 = 33        -> 32 main + 1 nextn/draft
//   qwen35.nextn_predict_layers        = 1         -> layers = 32, mtp_layers = 1
//   qwen35.feed_forward_length         = 12288     -> intermediate
//   qwen35.attention.head_count        = 16        -> query_heads
//   qwen35.attention.head_count_kv     = 4         -> kv_heads
//   qwen35.attention.key_length        = 256       -> head_dim
//   qwen35.rope.dimension_count        = 64        -> rotary_dim
//   qwen35.full_attention_interval     = 4         -> 8 full + 24 gdn over 32 layers
//   qwen35.ssm.conv_kernel             = 4         -> gdn_conv_kernel
//   qwen35.ssm.group_count             = 16        -> gdn_key_heads
//   qwen35.ssm.state_size              = 128       -> gdn_{key,value}_head_dim
//   qwen35.ssm.time_step_rank          = 32        -> gdn_value_heads
//   qwen35.attention.layer_norm_rms_epsilon = 1e-6
//   qwen35.rope.freq_base              = 1e7
//   qwen35.context_length              = 262144
//   len(tokenizer.ggml.tokens)         = 248320
//
// The GDN input width check is on the names, not on a doc: GGUF `blk.N.ssm_conv1d`
// is (conv_kernel, channels) = (4, 8192) and 2*key_dim + value_dim = 2*2048 + 4096
// = 8192, so key_dim = 16*128 and value_dim = 32*128 are the only assignment that
// fits the published tensor.  Likewise `blk.N.attn_qkv` is (8192, 4096) =
// q(2048) + k(2048) + v(4096).
//
// The token domain is the family constant rather than the GGUF vocab size, and that
// is a deliberate, measured choice: `frontend.cpp validate_registered_tokenizer`
// requires the artifact's tokenizer to expose *exactly* `options.token_domain`
// token ids, and the only Qwen3.5/3.6/3.8 sidecar on this machine exposes 248,077
// (248,044 BPE entries + 33 added tokens).  `tokenizer.ggml.tokens`'s first 248,044
// entries were compared one by one against that sidecar's `model.vocab` and are
// byte-identical (0 mismatches), so the family sidecar *is* this model's tokenizer
// for ids 0..248,043.  The remaining 276 rows are the special/added tail
// (`<|endoftext|>`, `<|im_start|>`, ... and `[PAD<id>]` fillers); the head is still
// sized 248,320 rows so those rows exist in the artifact, and ids >= 248,077 stay
// outside the sampling domain exactly as they do for qwen3.6-27b.
// ---------------------------------------------------------------------------

namespace ninfer::targets::qwen3_5_9b::detail {

struct TextConfig {
    // The shared window wiring in layouts_impl.h binds TextConfig::sliding_window +
    // TextConfig::is_swa_attention when a variant declares them.  This family is
    // full attention on every full-attention layer (the GDN layers are not sliding
    // window at all), so the no-op values are declared explicitly.
    static constexpr int sliding_window = 0;
    [[nodiscard]] static constexpr bool is_swa_attention(int /*layer*/) { return false; }

    static constexpr int hidden       = 4096;
    static constexpr int layers       = 32;
    static constexpr int intermediate = 12288;

    // The output matrix keeps the checkpoint's 248,320 rows; only ids in
    // [0, token_domain) are tokenizer-addressable and valid sampling results.
    static constexpr int output_rows  = 248320;
    static constexpr int token_domain = static_cast<int>(qwen3_6::kTokenDomain);

    static constexpr int gdn_conv_kernel      = 4;
    static constexpr int gdn_conv_state_width = gdn_conv_kernel - 1;
    static constexpr int gdn_key_heads        = 16;
    static constexpr int gdn_key_head_dim     = 128;
    static constexpr int gdn_value_heads      = 32;
    static constexpr int gdn_value_head_dim   = 128;

    static constexpr int query_heads = 16;
    static constexpr int kv_heads    = 4;
    static constexpr int head_dim    = 256;
    static constexpr int rotary_dim  = 64;

    static constexpr int full_attention_interval = qwen3_6::kHybridAttentionInterval;
    static constexpr float rms_epsilon           = 1.0e-6F;
    static constexpr float rope_theta            = 1.0e7F;

    static constexpr int key_dim               = gdn_key_heads * gdn_key_head_dim;
    static constexpr int value_dim             = gdn_value_heads * gdn_value_head_dim;
    static constexpr int convolution_dim       = 2 * key_dim + value_dim;
    static constexpr int query_size            = query_heads * head_dim;
    static constexpr int kv_size               = kv_heads * head_dim;
    static constexpr int query_projection_rows = 2 * query_size;

    static constexpr int mtp_layers               = 1;
    static constexpr int mtp_input_rows           = 2 * hidden;
    static constexpr int mtp_attention_input_rows = 2 * query_size + 2 * kv_size;
    static constexpr int mtp_mlp_gate_up_rows     = 2 * intermediate;

    [[nodiscard]] static constexpr bool is_full_attention(int layer) {
        return qwen3_6::is_full_attention_layer(layer);
    }

    [[nodiscard]] static constexpr int full_attention_layers() {
        return qwen3_6::full_attention_layers(layers);
    }

    [[nodiscard]] static constexpr int gdn_layers() { return qwen3_6::gdn_layers(layers); }

    [[nodiscard]] static constexpr int full_attention_index(int layer) {
        return qwen3_6::full_attention_index(layer);
    }

    [[nodiscard]] static constexpr int gdn_index(int layer) { return qwen3_6::gdn_index(layer); }
};

static_assert(TextConfig::full_attention_layers() == 8);
static_assert(TextConfig::gdn_layers() == 24);
// The three identities that pin the GDN geometry to the published tensors.
static_assert(TextConfig::convolution_dim == 8192);
static_assert(TextConfig::key_dim == 2048);
static_assert(TextConfig::value_dim == 4096);
static_assert(TextConfig::query_size == 4096);

struct VisionConfig : qwen3_6::VisionBackboneConfig {
    static constexpr int output_hidden = TextConfig::hidden;
};

// This checkpoint ships no DSpark draft.  The family runtime still needs concrete
// config types for its compile-time aliases and recipe instantiations, so the two
// are mirrors of the family shape with `supported = false`; the loader never binds
// a dflash/dflash2 object and `resolved_auto_speculative` never selects one.
struct DFlashConfig {
    static constexpr bool supported             = false;
    static constexpr bool bf16_weights          = true;
    static constexpr int layers                 = 1;
    static constexpr bool full_only             = true;
    static constexpr int local_layers           = 1;
    static constexpr int full_layers            = full_only ? layers : layers - local_layers;
    static constexpr int feature_layers         = 1;
    static constexpr int feature_rows           = feature_layers * TextConfig::hidden;
    static constexpr int hidden                 = TextConfig::hidden;
    static constexpr int intermediate           = TextConfig::intermediate;
    static constexpr int query_heads            = 16;
    static constexpr int kv_heads               = 4;
    static constexpr int head_dim               = 256;
    static constexpr int query_size             = query_heads * head_dim;
    static constexpr int kv_size                = kv_heads * head_dim;
    static constexpr int local_capacity         = 4096;
    static constexpr std::uint32_t local_window = 4096;
    static constexpr int mask_token             = 248077;
    static constexpr float rms_epsilon          = 1.0e-6F;
    static constexpr float rope_theta           = 1.0e7F;
    static constexpr float attention_scale      = 0.0625F;
    static constexpr float svip_entropy_threshold = 2.5F;
    static constexpr std::array<int, feature_layers> target_feature_layers{3};
};

struct DFlash2Config {
    static constexpr bool supported             = false;
    static constexpr bool bf16_weights          = true;
    static constexpr int layers                 = 1;
    static constexpr bool full_only             = false;
    static constexpr int local_layers           = 1;
    static constexpr int full_layers            = 0;
    static constexpr int feature_layers         = 1;
    static constexpr int feature_rows           = feature_layers * TextConfig::hidden;
    static constexpr int hidden                 = TextConfig::hidden;
    static constexpr int intermediate           = TextConfig::intermediate;
    static constexpr int query_heads            = 16;
    static constexpr int kv_heads               = 4;
    static constexpr int head_dim               = 256;
    static constexpr int query_size             = query_heads * head_dim;
    static constexpr int kv_size                = kv_heads * head_dim;
    static constexpr int local_capacity         = 2048;
    static constexpr std::uint32_t local_window = 2048;
    static constexpr int mask_token             = 248070;
    static constexpr float rms_epsilon          = 1.0e-6F;
    static constexpr float rope_theta           = 1.0e7F;
    static constexpr float attention_scale      = 0.0625F;
    static constexpr int block_drafts           = 7;
    static constexpr int conv_group_size        = 16;
    static constexpr int conv_kernel_size       = 2;
    static constexpr int selector_rank          = 256;
    static constexpr int selector_top_k         = 16;
    static constexpr int draft_head_rows        = 131072;
    static constexpr std::array<int, feature_layers> target_feature_layers{3};
};

inline constexpr float kAttentionScale                = 0.0625F;  // 1/sqrt(head_dim 256)
inline constexpr float kGdnScale                      = 0.08838834764831845F;  // 1/sqrt(128)
inline constexpr std::uint32_t kPrefillChunkAlignment = 128;
// 5 is the family floor for the adaptive draft window; this target carries no
// measured width optimisation of its own, so it takes the floor rather than the
// 27B's measured 15.
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 5;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 1;
inline constexpr std::uint32_t kNativeContext            = 262144;

} // namespace ninfer::targets::qwen3_5_9b::detail
