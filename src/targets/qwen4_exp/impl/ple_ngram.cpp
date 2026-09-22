// N6：ple_ngram.h 的实现。纯 host：不 include 任何 CUDA 头，因此这个 TU 可以只用
// g++ -fsyntax-only 验证（见 report.md 的验证命令），也是本目录里第一个由
// src/targets/qwen4_exp/CMakeLists.txt 自己拥有的实现源。
#include "targets/qwen4_exp/impl/ple_ngram.h"

#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen4_exp::detail {

void PleNgramWindow::fill_prevs(std::span<const std::int32_t> tokens,
                               std::span<std::int32_t> prevs_out) const {
    const std::size_t n_prev = prevs_per_token(ngram_size_);
    if (prevs_out.size() != n_prev * tokens.size()) {
        throw std::invalid_argument(
            "PLE n-gram window: prevs must hold (ngram_size-1)*n_tokens ids (" +
            std::to_string(prevs_out.size()) + " given for " +
            std::to_string(tokens.size()) + " tokens)");
    }
    // 把历史与本批连起来看：绝对下标 [0, history_.size()) 是历史（最旧在前），
    // [history_.size(), history_.size()+tokens.size()) 是本批。列 i 的第 (s+1) 个前驱的
    // 绝对下标 = base + i - (s+1)（s = 0 是最近的）：
    //   >= base  -> 本批内，取 tokens[absolute - base]
    //   [0,base) -> 历史内，取 history_[absolute]
    //   < 0      -> 不存在，读作 eos
    const std::ptrdiff_t base = static_cast<std::ptrdiff_t>(history_.size());
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        for (std::size_t s = 0; s < n_prev; ++s) {
            const std::ptrdiff_t absolute =
                base + static_cast<std::ptrdiff_t>(i) - static_cast<std::ptrdiff_t>(s) - 1;
            std::int32_t id = eos_;
            if (absolute >= base) {
                id = tokens[static_cast<std::size_t>(absolute - base)];
            } else if (absolute >= 0) {
                id = history_[static_cast<std::size_t>(absolute)];
            }
            prevs_out[i * n_prev + s] = id;
        }
    }
}

void PleNgramWindow::commit(std::span<const std::int32_t> tokens) {
    const std::size_t keep = prevs_per_token(ngram_size_);
    for (const std::int32_t token : tokens) {
        if (history_.size() == keep) { history_.erase(history_.begin()); }
        history_.push_back(token);
    }
}

void PleNgramWindow::step(std::span<const std::int32_t> tokens,
                          std::span<std::int32_t> prevs_out) {
    fill_prevs(tokens, prevs_out);
    commit(tokens);
}

ops::ple::PleStageDecl ple_stage_declaration(std::uint32_t eos_token_id) {
    using PC = PLEConfig;
    ops::ple::PleStageDecl decl;
    decl.present         = true;
    decl.ngram_size      = static_cast<std::uint32_t>(PC::ngram_size);
    decl.heads_per_ngram = static_cast<std::uint32_t>(PC::heads_per_ngram);
    decl.embed_dim       = static_cast<std::uint32_t>(PC::embed_dim);
    decl.eos_token_id    = eos_token_id;
    // spec 是 1-based，引擎是 0-based（见头文件里的出处）。
    decl.layer_ids = {static_cast<std::uint32_t>(PC::ple_layer_index_0based)};
    // ple_layout.cpp 的结构检查（ngram_size >= 2、heads_per_ngram > 0、embed_dim > 0）。
    decl.validate();
    return decl;
}

} // namespace ninfer::targets::qwen4_exp::detail
