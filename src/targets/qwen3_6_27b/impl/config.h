#pragma once

#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/hybrid_topology.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include <array>
#include <cstdint>

namespace ninfer::targets::qwen3_6_27b::detail {

struct TextConfig {
    static constexpr int hidden       = 5120;
    static constexpr int layers       = 64;
    static constexpr int intermediate = 17408;

    // The output matrix is padded for the selected kernels. Only token IDs in
    // [0, token_domain) are tokenizer-addressable and valid sampling results.
    static constexpr int output_rows  = 248320;
    static constexpr int token_domain = static_cast<int>(qwen3_6::kTokenDomain);

    static constexpr int gdn_conv_kernel      = 4;
    static constexpr int gdn_conv_state_width = gdn_conv_kernel - 1;
    static constexpr int gdn_key_heads        = 16;
    static constexpr int gdn_key_head_dim     = 128;
    static constexpr int gdn_value_heads      = 48;
    static constexpr int gdn_value_head_dim   = 128;

    static constexpr int query_heads = 24;
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

static_assert(TextConfig::full_attention_layers() == 16);
static_assert(TextConfig::gdn_layers() == 48);

struct VisionConfig : qwen3_6::VisionBackboneConfig {
    static constexpr int output_hidden = TextConfig::hidden;
};

struct DFlashConfig {
    static constexpr bool supported     = true;
    // DSpark draft weights ship as BF16 (not W8 like the 35B DFlash draft),
    // so the runtime uses the generic BF16 MMA GEMM path for projections.
    static constexpr bool bf16_weights  = true;
    static constexpr int layers         = 5;
    // DSpark has no sliding-window local layers: every layer uses the full
    // dual-source (target-feature context + noise rows) attention path.
    // A single legacy local cyclic layer is still allocated because the
    // cyclic-cache layout requires layers >= 1; full_only keeps it unused.
    static constexpr bool full_only     = true;
    static constexpr int local_layers   = 1;
    static constexpr int full_layers    = full_only ? layers : layers - local_layers;
    static constexpr int feature_layers = 5;
    static constexpr int feature_rows   = feature_layers * TextConfig::hidden;
    static constexpr int hidden         = TextConfig::hidden;
    static constexpr int intermediate   = 10240;
    static constexpr int query_heads    = 40;
    static constexpr int kv_heads       = 8;
    static constexpr int head_dim       = 128;
    static constexpr int query_size     = query_heads * head_dim;
    static constexpr int kv_size        = kv_heads * head_dim;
    static constexpr int local_capacity = 4096;
    static constexpr std::uint32_t local_window = 4096;
    static constexpr int mask_token     = 248077;
    static constexpr float rms_epsilon  = 1.0e-6F;
    static constexpr float rope_theta   = 1.0e7F;
    static constexpr float attention_scale = 0.08838834764831845F;
    // SVIP self-verification length policy: stop drafting after the first
    // position whose base-logit entropy exceeds threshold^2.
    static constexpr float svip_entropy_threshold = 2.5F;
    static constexpr std::array<int, feature_layers> target_feature_layers{4, 16, 28, 40, 52};
};

// DFlash2 block-diffusion draft: five sliding-window (2048) non-causal layers,
// grouped convolutions, and a rank-256 candidate selector with a 16-way path
// walk. The block always drafts seven tokens after the bonus token.
struct DFlash2Config {
    static constexpr bool supported     = true;
    static constexpr bool bf16_weights  = true;
    static constexpr int layers         = 5;
    static constexpr bool full_only     = false;
    static constexpr int local_layers   = 5;
    static constexpr int full_layers    = full_only ? layers : layers - local_layers;
    static constexpr int feature_layers = 5;
    static constexpr int feature_rows   = feature_layers * TextConfig::hidden;
    static constexpr int hidden         = TextConfig::hidden;
    static constexpr int intermediate   = 17408;
    static constexpr int query_heads    = 32;
    static constexpr int kv_heads       = 8;
    static constexpr int head_dim       = 128;
    static constexpr int query_size     = query_heads * head_dim;
    static constexpr int kv_size        = kv_heads * head_dim;
    static constexpr int local_capacity = 2048;
    static constexpr std::uint32_t local_window = 2048;
    static constexpr int mask_token     = 248070;
    static constexpr float rms_epsilon  = 1.0e-6F;
    static constexpr float rope_theta   = 1.0e7F;
    static constexpr float attention_scale = 0.08838834764831845F;
    static constexpr int block_drafts   = 7;
    static constexpr int conv_group_size = 16;
    static constexpr int conv_kernel_size = 2;
    static constexpr int selector_rank   = 256;
    static constexpr int selector_top_k  = 16;
    static constexpr std::array<int, feature_layers> target_feature_layers{5, 19, 33, 47, 61};
};

inline constexpr float kAttentionScale                   = 0.0625F;
inline constexpr float kGdnScale                         = 0.08838834764831845F;
inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 5;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 7;
inline constexpr std::uint32_t kNativeContext            = 262144;

} // namespace ninfer::targets::qwen3_6_27b::detail
