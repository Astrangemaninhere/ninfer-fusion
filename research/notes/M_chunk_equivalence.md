
## E1 判别：plain 输出是否随 --prefill-chunk 变化 (2026-09-10 19:08)

- chunk 128 vs 2048: IDENTICAL (48 tokens)
- chunk 512 vs 2048: IDENTICAL (48 tokens)

判读：**plain + 贪心**下输出若随 chunk 变化 ⇒ 引擎的贪心结果依赖 batching/分块
（即 kernel 实例化带来的数值差异足以翻转 argmax）⇒ 投机 verify（T=width≠1）与 plain（T=1）本就不可能逐 token 一致，
接受率的上限被**实现数值差**封住，而不是草稿质量。若完全相同 ⇒ 排除该假设，转查 dflash2 的 rope_delta 与 KV 量化路径。
