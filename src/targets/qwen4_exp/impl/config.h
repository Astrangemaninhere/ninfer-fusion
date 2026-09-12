// Generated for FlashNext (Qwen4Exp) P0 skeleton via tools/archkit.
// model_type=qwen4_exp; MoE 512x10 + PLE ngram + MTP ——
// MoE/PLE/MTP 算子为 new_op 工作包 (见 manifest / _flashnext_plan.md), 本骨架先立几何与注册。
#pragma once
#include <array>
namespace ninfer::targets::qwen4_exp::detail {

struct TextConfig {
    static constexpr int hidden = 2560;
    static constexpr int layers = 48;
    static constexpr int query_heads = 24;
    static constexpr int kv_heads = 2;
    static constexpr int head_dim = 256;
    static constexpr int intermediate = None;
    static constexpr int vocab = 248320;
    static constexpr int max_ctx = 262144;
    static constexpr float rms_epsilon = 1e-06f;
    static constexpr int sliding_window = 0;
    static constexpr float qk_scale_factor = 1.0f;
    static constexpr float output_multiplier = 1.0f;
    static constexpr float final_logit_softcapping = 0.0f;

    // 层类型表: 0=full, 1=swa, 2=linear(gdn)。MoE 混布信息在 layer_types 原表。
    static constexpr std::array<int, 48> layer_kind{2,2,2,0,2,2,2,0,2,2,2,0,2,2,2,0,2,2,2,0,2,2,2,0,2,2,2,0,2,2,2,0,2,2,2,0,2,2,2,0,2,2,2,0,2,2,2,0};

    static constexpr int full_attention_layers() {
        int c = 0;
        for (int k : layer_kind) { if (k == 0) { c += 1; } }
        return c;
    }
    static constexpr int swa_attention_layers() {
        int c = 0;
        for (int k : layer_kind) { if (k == 1) { c += 1; } }
        return c;
    }
    static constexpr int gdn_layers() {
        int c = 0;
        for (int k : layer_kind) { if (k == 2) { c += 1; } }
        return c;
    }
};

// MoE 工作包参数 (P2 启用): 512 experts, top-10,
// moe_intermediate=640, shared_expert=共享专家见上游.
// PLE ngram: ngram_size=3, vocab_base=20000000,
// heads_per_ngram=8.
// MTP: mtp_num_hidden_layers=1.
} // namespace ninfer::targets::qwen4_exp::detail
