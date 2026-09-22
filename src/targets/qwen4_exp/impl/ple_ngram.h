#pragma once

// N6：qwen4_exp PLE n-gram 阶段的**目标侧、纯 host** 一半。
//
// 已经存在（均已实证）：
//   * ops/ple/ple_layout.{h,cpp} —— 行公式 + 行号 -> (物理文件, 字节偏移)。它的
//     derive_rows() 是**权威**：吃 (tokens, prevs, eos)，自己在内部做 EOS 截断
//     （ple_layout.cpp:122-145：ctx[s] 从近扫到远，遇到 eos/负数后更远的全填 eos）。
//   * ops/ple/ple_table.{h,cu} —— SSD 真表 gather。gather_phase() 按**列**索引 prevs
//     （ple_table.cu:91-95：prevs.data() + column*n_prev，column 遍历被喂进来的整批），
//     所以调用者必须为**每一个被喂进来的列**交出 (ngram_size-1) 个 id —— 包括
//     Decode/Verify 阶段随后会丢掉的那些列（ple_stage.h:12-17）。
//   * impl/ple_runtime.h —— CUDA 侧接缝（PleTable 所有权 + 相位 gather 入口）。它把
//     契约写清了，但树里**没有任何东西生产 (tokens, prevs, eos) 这个三元组**：现存的两个
//     调用点都是手填 prevs = eos（tests/test_ple_layout.cpp:96 与 :176），而
//     tools/ple_reference.py 的 replay 子命令正是为这件事加的（ple_reference.py:138-140：
//     "The existing paths only ever run with prevs == {eos, eos}, so the bigram/trigram
//     mixers never see a real predecessor"）。
//
// 本文件就是那个生产者，别的都不是。它是 PLE 消费链上**既不依赖 MoE、也不依赖 qwen3_6
// 家族运行时**的那一件，因此可以纯 host、无 CUDA、无 GPU、无 artifact 地编译与验证。
//
// 它编码的契约（未来任何调用者唯一不能搞错的一件事）：
//   `prevs` = 每个列的 (ngram_size-1) 个**已提交**前驱，**最旧在前**、token-major
//   （ple_runtime.h:29-35）。**不存在**的前驱（序列开头，或当前 chunk 之前的列）读作
//   EOS（ple_layout.h:81-87）。**真的**是 EOS 记号的前驱要**原样传下去**：截断归
//   derive_rows() 所有（ple_layout.cpp:126-132）。在这里再实现一遍截断，等于给一个已经
//   存在的权威再造一份会静默漂移的副本。
//   本文件不决定哪个相位欠哪些列（那是 ple_phase_window 的事），也不决定行号是什么
//   意思（那是 PleLayout 的事）。

#include "ops/ple/ple_stage.h"
#include "targets/qwen4_exp/impl/config.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen4_exp::detail {

// 一条序列的已提交 token 历史 -> 下一批的 prevs。
//
// 生命周期：**一个序列一个实例**（n-gram 窗口是逐序列状态，不是全局的）。引擎的
// 单序列文本路径一个；将来多序列并发时每序列一个。
// 线程：不加锁，调用者保证同一条序列不被并发驱动。
class PleNgramWindow {
public:
    PleNgramWindow(std::uint32_t ngram_size, std::int32_t eos)
        : ngram_size_(ngram_size), eos_(eos) {
        if (ngram_size < 2) {
            throw std::invalid_argument("PLE n-gram window needs ngram_size >= 2");
        }
        history_.reserve(prevs_per_token(ngram_size));
    }

    // 算 tokens 这一批的 prevs，**不改**历史。prevs_out 必须是
    // prevs_per_token(ngram_size) * tokens.size() 个 id，token-major：列 i 的
    // (ngram_size-1) 个 id 落在 [i*n_prev, (i+1)*n_prev)，最旧在前。尺寸不符即抛。
    //
    // 为什么单独暴露：**投机 Verify** 一次喂 K+1 列，只收最后一列；被拒的草稿列绝不能进
    // 历史。于是调用者先 fill_prevs 拿到全部被喂列的 id，等接受结果出来再
    // commit(被接受的 token)。顺序搞反（先 commit 后 fill）会**静默算错**、不会崩，
    // 所以常规路径走 step()。
    void fill_prevs(std::span<const std::int32_t> tokens,
                    std::span<std::int32_t> prevs_out) const;

    // 把这一批记为已提交（成为下一批的历史）。历史只保留最近 ngram_size-1 个 token。
    void commit(std::span<const std::int32_t> tokens);

    // Prefill / Decode 的常规一步：先 fill_prevs、再 commit。
    void step(std::span<const std::int32_t> tokens, std::span<std::int32_t> prevs_out);

    // 新序列：历史清空（下一次 step 的前驱全是 eos）。
    void reset() noexcept { history_.clear(); }

    [[nodiscard]] std::uint32_t ngram_size() const noexcept { return ngram_size_; }
    [[nodiscard]] std::int32_t eos() const noexcept { return eos_; }
    [[nodiscard]] std::size_t committed() const noexcept { return history_.size(); }

    // 每列欠几个前驱 = ngram_size-1。契约里 prevs 的长度就用它算。
    [[nodiscard]] static std::size_t prevs_per_token(std::uint32_t ngram_size) noexcept {
        return ngram_size - 1;
    }

private:
    std::uint32_t ngram_size_;
    std::int32_t eos_;
    std::vector<std::int32_t> history_; // 最多 ngram_size-1 个，最旧在前
};

// arch-spec 的 "ple" 块（tools/archkit/specs/qwen4_exp_spec.json）作为**运行时数据**
// （ops/ple/ple_stage.h:157 的 PleStageDecl）。
//
// 为什么不用 PleStageDecl::from_spec_file()：它能读的那部分（ngram_size /
// heads_per_ngram / ple_embed_dim）本函数照抄 config.h 的编译期常量（同一份来源，多一次
// 运行时 JSON 解析没有收益，且 config.h 已有 static_assert 钉住几何），而它**读不出的
// 两件事**只能由这里补：
//   * 0-based 层号：spec 的 ple_layer_ids=[2] 是 **1-based**（flashnext_bindings.py:167
//     "sits at layer id 2 per spec; 1-based docs -> 2"），而引擎的层走查是 0-based，所以
//     这一级挂在 **layer 1** —— 与 checkpoint 键 rebase 成 model.layers.1.ple.* 是同一次
//     决定（impl/config.h:97-102）。把 2 当 0-based 用就是把 PLE 残差栈挂错层的经典错法。
//   * eos：spec 里**没有** eos_token_id（已核：全文件无 "eos" 键），而 qwen4_exp 今天没有
//     frontend（export 头注释：运行时类型都还不存在），所以由调用者传入。0 表示"还不知道"，
//     **原样保留不猜**。
[[nodiscard]] ops::ple::PleStageDecl ple_stage_declaration(std::uint32_t eos_token_id);

// 编译期自检：声明侧的 head 数（ple_stage.h:168-170 的 (ngram-1)*heads_per_ngram）必须
// 等于侧车/几何侧的 n_heads（config.h:104 的 2*heads_per_ngram = 8 bigram + 8 trigram）。
// 两个来源独立：一个是阶段声明公式，一个是侧车 16 head 的几何。哪天真把 ngram_size 调成 2
// 而 n_heads 留在 16，这里先炸。
static_assert((PLEConfig::ngram_size - 1) * PLEConfig::heads_per_ngram == PLEConfig::n_heads,
              "PLE declaration head count disagrees with the sidecar geometry (config.h)");

} // namespace ninfer::targets::qwen4_exp::detail
