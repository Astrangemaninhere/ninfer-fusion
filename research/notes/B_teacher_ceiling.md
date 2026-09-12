# B 线：教师天花板表（引擎 argmax 与语料 next-token 的重合率，按源分类）

- 脚本（独立实现，未复用 `_data_mix.py`）: `_collab/B_teacher_ceiling_run.py`
- 复现命令（CPU-only，~1 min）:
  `wsl.exe -e python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/B_teacher_ceiling_run.py`
- 数据: `data/hs_cache_topk2` 480 个 npz 全量 + `hs_corpus.jsonl` 的 `src` 前缀分类；
  天花板 = `ids16[t,0] == tokens[t+1]` 的占比（t 取 ids16/tokens 对齐全集，705,740 个位置）。

## 表

| source | files | positions | top1 | top4 | top16 |
|---|---|---|---|---|---|
| rag | 220 | 519,448 | **0.1808** | 0.5062 | 0.6635 |
| code | 140 | 148,948 | **0.4742** | 0.6592 | 0.7439 |
| qa | 120 | 37,344 | **0.3582** | 0.7031 | 0.8344 |
| TOTAL | 480 | 705,740 | **0.2521** | 0.5489 | — |

与 A 的 `_data_mix.py` 数字吻合（rag 0.181/0.506/0.664，code 0.474，qa 0.358，
TOTAL 706,220 token——差 = 每文件 1 个位置，A 计 token 数、我计有效位置数）。

## 判读（对 S4 `--teacher` 口径的意义）

1. **rag（73.6% 的数据）语料口径上限只有 0.18**：无论 draft 多好，rag 上
   语料 hit@1 ≤ 0.1808 —— 这就是为什么 W9 门必须用 teacher 口径打分，
   语料口径是诊断/下界。qa/code 的语料口径上限 0.36/0.47 也不高。
2. **混合语料口径的理论上限 0.2521**：如果 draft 完美复制教师 argmax，
   全量语料 hit@1 也就是 0.25 —— 与 09-01 实测 zh-wiki greedy 23.6% 几乎重合
   （那批权重的教师≈同分布），旁证旧 23.6% 主要是口径问题而非 draft 全坏。
3. top4/top16 列给出树口径的天花板参照：教师自己 top-4 召回 rag 只有 0.506，
   code 0.659，qa 0.703 —— hit@4 的现实上限离 1 很远，W9 门
   （teacher hit@1 ≥ 0.35 且 hit@4−hit@1 ≥ 0.12）应对照这些数评估松紧。
