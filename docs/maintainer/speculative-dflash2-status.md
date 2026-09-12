# DFlash2 草稿后端：现状与支持级别（2026-09-12）

状态：**低支持 / 不推荐**。MTP 后端是当前推荐路径。欢迎提交 PR。

## 结论（一句话）

同一 artifact 上，**MTP 健康、DFlash2 在中文上塌陷**；官方参考实现（vLLM 0.29 自带
`DFlash2DraftModel`）在**同一草稿**上比我们高，但**其绝对值也偏低**。因此我们把
DFlash2 降为"低支持"，把工程投入转回 MTP 与其它未落地项。

## 实测数字（同一条中文 prompt，greedy）

| 后端 | 接受率 | 位置剖面 | AL (tok/round) | decode |
|---|---|---|---|---|
| `--spec mtp --draft-tokens 3 --lm-head-draft` | **37.88%** | `32,14,4` | 2.11 | 104 tok/s |
| `--spec dflash2`（默认宽度） | **4.81%** | `20,3,0,0,0,0,0` | 1.32 | 44 tok/s |
| plain（无投机） | — | — | — | 59 tok/s |

跨语言（同引擎、同草稿、只换 prompt）：

| prompt | MTP α₁ | DFlash2 α₁ | DFlash2 AL |
|---|---|---|---|
| 中文 | 0.711 | **0.282** | 1.32 |
| 英文 | 0.737 | 0.667 | 2.88 |
| 代码 | 0.889 | **0.933** | **6.33** |

⇒ DFlash2 **在英文/代码上是好的**（代码 AL 6.33、221 tok/s），**只有中文塌**；
而 MTP 三语皆健康。按 1Cat 的资料（`tools/archkit/_GPU_MATRIX.md`："greedy MTP
+10~25 接受点"），MTP 的实测值已明显高于该口径。

## 官方参考实现对照（决定性）

vLLM 0.29.0 自带 DFlash2 + 同一草稿 `draft_dflash2_ref`，greedy，block_size 8：
中文 AL **1.860**（整块接受率 12.28%）、逐位 `45.8/18.7/9.3/4.7/3.7/1.9/1.9 %`。

- 我们（AL 1.32 / 第 3 位起归零）**确实低于**参考 ⇒ 引擎侧有待查缺陷；
- 但参考**本身也不高**（pos-0 仅 45.8%，英文 80.5%）⇒ 低接受率有相当部分是
  这份草稿+目标模型在中文上的固有弱点。

## 已定位的异常（供 PR 接手）

1. **深度 ≥2 的接受率精确为零**：71 轮、355 次机会、深度 2–6 全为 0，而参考在
   同样深度有 1.9–9.3% ⇒ 我们的深度 ≥2 **不是更弱，而是功能性死亡**。
2. **草稿的 unary top-1 在深度间"粘住"**：相邻深度相同率 **56.9%**、最大游程 5；
   参考仅 **9.5%**、最大游程 2（147 个深度槽零游程 ≥3）⇒ 差 **6 倍**。
   表现为"上下文没起作用、退回高频先验"。
3. **未验证的唯一环节**：草稿的上下文/特征保真度
   （`features → feature_projection → context_norm → context_key/value → rope → kv_cache_append_prefix`）。
   已排除：selector 算术（与参考逐项一致、codebook 字节级全等）、KV 精度、
   头精度（BF16 头 A/B 完全相同）、块位置/RoPE、mask_token。

## 可用的诊断工具（已落地，可复用）

- `NINFER_DF2SCORES=1`：落盘 12 组 `scores(16×16)/cand/unary/front/anch/logits/ph/proj`
  （`--no-cuda-graph` 必需；写入 `dl/df2scores_*.bin`）。`front`+`anch` 是唯一对齐键。
- `NINFER_DF2FEAT=1`、`NINFER_DF2SEL=1`、`NINFER_DF2DBG=1`：既有探针。
- 参考侧插桩：`/home/user/_bak_df2ref/` 备份 + `speculator.py` 的 env 门控 dump
  （`DF2REF_DUMP=1`），回滚：`cp /home/user/_bak_df2ref/speculator.py.orig <vllm>/v1/worker/gpu/spec_decode/dflash2/speculator.py`。

## 不是 DFlash2 的问题（澄清）

- `--lm-head-draft` 对 **MTP 是 no-op**（MTP 不读 `proposal_head`），只对 DFlash2 是真切换。
- KV 两级残差面**在出厂二进制里不可达**（唯一置位点在 `engine.cpp:542-555`，而 CLI 无该
  flag、serve 侧不转发）⇒ `--kv-dtype nvfp4` 实跑单平面；KV 精度对 DFlash2 的影响 ≈0.03pp。
- 热窗在 36-token prompt 上**无影响**（4 臂全同）；**长上下文版本尚未检验**。

## 要做什么（优先级）

1. **MTP + 树 + SVIP**（推荐主线）：SVIP 是熵驱动的草稿截断（`mtp_impl.h:156-171`），
   目前只接 MTP/DFlash-v1；把"熵 → 逐档宽度"接起来即"树 + SVIP"，可把 16 列 verify 预算
   花在高熵深度上（估计 b=2 ⇒ AL 2.11→2.85，b=4 ⇒ 3.55；假设兄弟独立，属乐观估计）。
2. **长上下文版本的热窗检验**（8–16k prompt + 小窗口）。
3. **DFlash2 的上下文保真度校验**（上面第 3 条），这是唯一未验证环节。
