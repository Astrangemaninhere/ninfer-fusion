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
    // 该目标没有 dense FFN：48 层每层都是 MoE（契约 flashnext_bindings.py:116-141 逐层
    // 发射 layer.{i}.moe.*；全树没有 layer.{i}.mlp.{gate,up,down}），所以这个宽度是
    // **MoE 的 MLP 宽度**（每专家 640），不是 DensePostMixerPayload 那条 dense 叶子的
    // 宽度 —— 阶段 (c) 的 MoE 叶是它唯一的消费者（见下方 MoEConfig）。
    //
    // 出处：tools/archkit/specs/qwen4_exp_spec.json 的 moe.moe_intermediate_size=640
    // （该 spec 由官方 config fn_official.json 派生），routed 与 shared 专家同为 640。
    // 这是 **期望值而不是被信任的常量**：artifact 侧唯一的几何来源是张量形状，装载期
    // validate_stage_a_geometry()（export 头）从 layer.0.moe.e0.down 反推并与本常量
    // 断言相等，不符即抛错并列印两侧数值。
    //
    // 历史：生成器曾在此吐 `None`（非法 C++，S33 记录）；随后被换成 -1 当"毒值"占位
    // （理由：0 会让"维数=0"静默过关）。本补丁按 B_s37（M 共识）把期望值落为 640，并把
    // "没有 dense FFN"显式写成 has_dense_ffn=false，而不是靠一个毒值让消费者炸掉。
    static constexpr int intermediate = 640;
    static constexpr bool has_dense_ffn = false;
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

// ---------------------------------------------------------------------------
// 算子工作包参数（S37 阶段 a：把文件尾注释升级成真常量）。
//
// 值的出处 = tools/archkit/specs/qwen4_exp_spec.json 对应段（由官方 fn_official.json
// 派生）。结构体与成员名对齐 tools/archkit/gen_target.py 的发射器（参考产物
// tools/archkit/out/qwen4_exp/config.h），所以将来用生成器重吐同一份 config.h 是收敛
// 而不是冲突。派生量的命名对齐真正的消费者：契约文本
// tools/archkit/flashnext_bindings.py:84-113 与 src/ops/ple/ple_layout.h:44-59。
// ---------------------------------------------------------------------------

struct MoEConfig { // spec "moe"
    static constexpr bool enabled                        = true;
    static constexpr int num_experts                     = 512;   // moe.num_experts
    static constexpr int num_experts_per_tok             = 10;    // moe.num_experts_per_tok
    static constexpr int moe_intermediate_size           = 640;   // moe.moe_intermediate_size
    static constexpr int shared_expert_intermediate_size = 640;   // moe.shared_expert_intermediate_size
    static constexpr float router_aux_loss_coef          = 0.001F; // moe.router_aux_loss_coef
    // 每一层都有 MoE 块（契约 :116-141 是 for i in range(N_LAYERS)），没有 dense 兜底。
    static constexpr int layers_with_moe                 = TextConfig::layers;
};

struct GDNConfig { // spec "gdn" —— 36 个 linear_attention 层（0,1,2,4,5,6,...）
    static constexpr int linear_conv_kernel_dim   = 4;     // gdn.linear_conv_kernel_dim
    static constexpr int linear_key_head_dim      = 128;   // gdn.linear_key_head_dim
    static constexpr int linear_num_key_heads     = 16;    // gdn.linear_num_key_heads
    static constexpr int linear_value_head_dim    = 128;   // gdn.linear_value_head_dim
    static constexpr int linear_num_value_heads   = 48;    // gdn.linear_num_value_heads
    static constexpr const char* output_gate_type = "sigmoid"; // gdn.output_gate_type
    static constexpr int key_dim   = linear_num_key_heads * linear_key_head_dim;     // 2048
    static constexpr int value_dim = linear_num_value_heads * linear_value_head_dim; // 6144
};

struct PLEConfig { // spec "ple"
    static constexpr int ngram_size            = 3;        // ple.ngram_size
    static constexpr int heads_per_ngram       = 8;        // ple.heads_per_ngram
    static constexpr int ngram_vocab_size_base = 20000000; // ple.ngram_vocab_size_base
    static constexpr int conv_kernel_size      = 4;        // ple.ple_conv_kernel_size
    static constexpr int embed_dim             = 2560;     // ple.ple_embed_dim（== hidden）
    static constexpr int split_ngram_parts     = 128;      // ple.split_ngram_parts（侧车 128 逻辑分片）
    // ⚠ 下标基准：spec 的 ple_layer_ids=[2] 是 **1-based** 的层序号，0-based 层号是 1；
    // checkpoint 键因此是 model.layers.1.ple.*（B_s26 契约 rebase 改指；ple_runtime.h:34）。
    // flashnext_bindings.py:143 同样注明 "sits at layer id 2 per spec; 1-based docs -> 2"。
    // 别把 2 当 0-based 用 —— 那是 PLE 残差栈挂错层的经典错法。
    static constexpr int ple_layer_index_1based = 2;       // ple.ple_layer_ids[0]
    static constexpr int ple_layer_index_0based = ple_layer_index_1based - 1;
    // 两个 head 类（bigram 0..7 + trigram 8..15），见 ple_layout.h:8-9 与 :49-51。
    static constexpr int n_heads                 = 2 * heads_per_ngram;         // 16
    static constexpr int embedding_row_dimension = embed_dim / n_heads;         // 160
    static constexpr int row_stride_bytes        = 2 * embedding_row_dimension; // BF16 -> 320
};

struct MtpConfig { // spec "mtp"
    static constexpr int mtp_layers            = 1;      // mtp.mtp_num_hidden_layers
    static constexpr bool dedicated_embeddings = false;  // mtp.mtp_use_dedicated_embeddings
};

struct IndexerConfig { // spec "indexer" —— QSA 层的稀疏索引器
    static constexpr int budget         = 2048; // indexer.indexer_budget
    static constexpr int compress_ratio = 4;    // indexer.indexer_compress_ratio
    static constexpr int head_dim       = 128;  // indexer.indexer_head_dim
    static constexpr int n_heads        = 4;    // indexer.indexer_n_heads
    static constexpr int kv_heads       = 1;    // indexer.indexer_kv_heads
};

struct HyperConnectionConfig { // spec "indexer" —— 超稀疏头压缩（hc）
    static constexpr int count   = 4;   // indexer.hc_count
    static constexpr int lowrank = 320; // indexer.hc_lowrank
};

// ---------------------------------------------------------------------------
// 编译期自检：每条断言都有第二个独立来源（spec 常量 vs 契约/消费端几何），任一漂移即
// 编译失败。放在这里而不是运行时，是因为这里能白拿一次构建门。
// ---------------------------------------------------------------------------
static_assert(TextConfig::layers == 48);
static_assert(TextConfig::full_attention_layers() == 12);
static_assert(TextConfig::gdn_layers() == 36);
// 层型表与契约的 QSA 层表逐层一致（flashnext_bindings.py:36-38：QSA_LAYERS = range(3,48,4)）。
static_assert([] {
    for (int layer = 0; layer < TextConfig::layers; ++layer) {
        const bool qsa = layer >= 3 && (layer - 3) % 4 == 0;
        if (qsa != (TextConfig::layer_kind[static_cast<std::size_t>(layer)] == 0)) {
            return false;
        }
    }
    return true;
}());
// MoE 覆盖全层，且 TextConfig::intermediate 就是每专家的 MLP 宽度。
static_assert(MoEConfig::moe_intermediate_size == TextConfig::intermediate);
static_assert(MoEConfig::shared_expert_intermediate_size == MoEConfig::moe_intermediate_size);
// GDN 几何 vs 契约：layer.{i}.gdn.in_qkv 声明 [2560, K+V]（K=16x128, V=48x128）。
static_assert(GDNConfig::key_dim == 2048);
static_assert(GDNConfig::value_dim == 6144);
static_assert(GDNConfig::key_dim + GDNConfig::value_dim == 8192);
// PLE 几何 vs 侧车消费端（ple_layout.h:49-51：n_heads=16, row_dim=160, stride=320B）。
static_assert(PLEConfig::embed_dim == TextConfig::hidden);
static_assert(PLEConfig::n_heads == 16);
static_assert(PLEConfig::embedding_row_dimension == 160);
static_assert(PLEConfig::row_stride_bytes == 320);
static_assert(PLEConfig::n_heads * PLEConfig::embedding_row_dimension == TextConfig::hidden);
// 索引器 / hc 几何 vs 契约：qsa.idx_wq 声明 [2560,512]（=4x128）, qsa.hc.*.down 声明 [2560,320]。
static_assert(IndexerConfig::n_heads * IndexerConfig::head_dim == 512);
static_assert(HyperConnectionConfig::lowrank == 320);
} // namespace ninfer::targets::qwen4_exp::detail
