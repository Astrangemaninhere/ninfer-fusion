#!/usr/bin/env python3
"""给 dflash2_selector 贯通 token-domain 过滤参数。
契约（上游 docs/maintainer/qwen3.8-27b-dflash2.md §6.1）：
  "full route 的 Head 是 text/output_head [248320,5120]，只允许 token ids 0..248076 进入 top-16"
我们的 top-K 却扫到 vocab=248320。本补丁加一个 `domain` 参数（= TextConfig::token_domain）。
"""
import pathlib, sys

R = pathlib.Path("/home/user/ninfer-fusion")

EDITS = [
    # 1) 公开头
    ("include/ninfer/ops/dflash2_selector.h",
     [" * The registered domain is V=248320, R=256, S=7, B=1..8, K=16. Inputs and",
      " * weights are unchanged and the Op owns no workspace or persistent state.",
      " */",
      "void dflash2_selector(const Tensor& unary_logits, const Tensor& projected_hidden,",
      "                      const Weight& predecessor_codebook, const Weight& successor_codebook,",
      "                      const Tensor& anchors, Tensor& candidates, Tensor& unary, Tensor& scores,",
      "                      Tensor& drafts, const ops::SamplingConfig* configs, Tensor& candidate_ids,",
      "                      Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,",
      "                      cudaStream_t stream);"],
     [" * The registered domain is V=248320, R=256, S=7, B=1..8, K=16. `domain` bounds the",
      " * token ids admitted into the top-K candidate set (the contract only allows ids",
      " * `0..token_domain-1`; 248077 for this checkpoint). Inputs and weights are unchanged",
      " * and the Op owns no workspace or persistent state.",
      " */",
      "void dflash2_selector(const Tensor& unary_logits, const Tensor& projected_hidden,",
      "                      const Weight& predecessor_codebook, const Weight& successor_codebook,",
      "                      const Tensor& anchors, Tensor& candidates, Tensor& unary, Tensor& scores,",
      "                      Tensor& drafts, const ops::SamplingConfig* configs, Tensor& candidate_ids,",
      "                      Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,",
      "                      std::int32_t domain, cudaStream_t stream);"]),
    # 2) launcher 头
    ("src/ops/launcher/dflash2_selector.h",
     ["                              Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,",
      "                              cudaStream_t stream);"],
     ["                              Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,",
      "                              std::int32_t domain, cudaStream_t stream);"]),
    # 3) launcher 实现
    ("src/ops/launcher/dflash2_selector.cu",
     ["                              Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,",
      "                              cudaStream_t stream) {"],
     ["                              Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,",
      "                              std::int32_t domain, cudaStream_t stream) {"]),
    ("src/ops/launcher/dflash2_selector.cu",
     ["        static_cast<std::int32_t*>(candidates.data), static_cast<float*>(unary.data),",
      "        unary_logits.ne[0], batch, steps, columns);"],
     ["        static_cast<std::int32_t*>(candidates.data), static_cast<float*>(unary.data),",
      "        unary_logits.ne[0], domain, batch, steps, columns);"]),
    # 4) kernel：签名 + 扫描上界
    ("src/ops/kernel/dflash2_selector.cuh",
     ["    float* __restrict__ unary, int vocab, int batch, int steps, int columns) {"],
     ["    float* __restrict__ unary, int vocab, int domain, int batch, int steps, int columns) {"]),
    ("src/ops/kernel/dflash2_selector.cuh",
     ["    for (int v = tid; v < vocab; v += Block) {"],
     ["    // Contract: only token ids [0, domain) may enter the top-K; ids at or above the",
      "    // token domain (248077 here) are not addressable sampling results.",
      "    for (int v = tid; v < domain; v += Block) {"]),
    # 5) wrapper：参数 + 校验 + 转发
    ("src/ops/wrapper/dflash2_selector.cpp",
     ["                      Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,",
      "                      cudaStream_t stream) {"],
     ["                      Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,",
      "                      std::int32_t domain, cudaStream_t stream) {"]),
    ("src/ops/wrapper/dflash2_selector.cpp",
     ["    if (vocab != 248320 || rank != detail::kDflash2SelectorRank || steps < 1 ||",
      "        steps > 15 || top_k != detail::kDflash2SelectorTopK || batch < 1 || batch > 8 ||",
      "        tokens != steps * batch ||",
      "        projected_hidden.ne[1] != tokens) {",
      "        invalid(",
      "            \"dflash2_selector: registered domain is V=248320, R=256, S=1..15, B=1..8, K=16\");",
      "    }"],
     ["    if (vocab != 248320 || rank != detail::kDflash2SelectorRank || steps < 1 ||",
      "        steps > 15 || top_k != detail::kDflash2SelectorTopK || batch < 1 || batch > 8 ||",
      "        tokens != steps * batch || domain < 1 || domain > vocab ||",
      "        projected_hidden.ne[1] != tokens) {",
      "        invalid(",
      "            \"dflash2_selector: registered domain is V=248320, R=256, S=1..15, B=1..8, \"",
      "            \"K=16, domain in [1,V]\");",
      "    }"]),
    ("src/ops/wrapper/dflash2_selector.cpp",
     ["    detail::dflash2_selector_launch(unary_logits, projected_hidden, predecessor_codebook,",
      "                                    successor_codebook, anchors, candidates, unary, scores, drafts,",
      "                                    configs, candidate_ids, candidate_probs, steps, top_k, stream);"],
     ["    detail::dflash2_selector_launch(unary_logits, projected_hidden, predecessor_codebook,",
      "                                    successor_codebook, anchors, candidates, unary, scores, drafts,",
      "                                    configs, candidate_ids, candidate_probs, steps, top_k, domain,",
      "                                    stream);"]),
    # 6) 调用点：传 TextConfig::token_domain
    ("src/targets/qwen3_6/impl/runtime/dflash2_impl.h",
     ["                          frame.draft_candidate_probs, steps, Config::selector_top_k,",
      "                          state.execution.device.stream);"],
     ["                          frame.draft_candidate_probs, steps, Config::selector_top_k,",
      "                          TextConfig::token_domain, state.execution.device.stream);"]),
]

ok = True
for rel, old_lines, new_lines in EDITS:
    p = R / rel
    s = p.read_text(encoding="utf-8", errors="surrogateescape")
    old = "\n".join(old_lines)
    new = "\n".join(new_lines)
    n = s.count(old)
    if n != 1:
        print("FAIL %-58s 匹配 %d 次（期望 1）" % (rel, n))
        ok = False
        continue
    p.write_text(s.replace(old, new), encoding="utf-8", errors="surrogateescape")
    print("OK   %s" % rel)

print()
print("全部成功" if ok else "有失败项，未编译")
sys.exit(0 if ok else 1)
