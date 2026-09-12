# R8：草稿链路参数审计（对照 ckpt 自带 config）+ vLLM 对照数字核实（2026-09-11 14:2X）

## 一、最关键的结论：dflash2 的**声明型**超参与 ckpt 自带 config **逐项一致**
权威来源：`data/draft_dflash2_ref/config.json`（`architectures: DFlash2DraftModel`，即文档健康数字对应的
参照草稿，Sep 11 12:32 已在盘上）。对照我们引擎硬编码的 `DFlash2Config`
（`src/targets/qwen3_6_27b/impl/config.h`）：

| 参数 | ckpt config（`dflash_config`） | 引擎 `DFlash2Config` | |
|---|---|---|---|
| `mask_token_id` | 248070 | 248070 | ✓ |
| `selector_rank` | 256 | 256 | ✓ |
| `selector_top_k` | 16 | 16 | ✓ |
| `conv_group_size` | 16 | 16 | ✓ |
| `conv_kernel_size` | 2 | 2 | ✓ |
| `block_size`（= W = K+1） | 8 | `block_drafts`=7 → W=8 | ✓ |
| `target_layer_ids` | [5,19,33,47,61] | `target_feature_layers`=[5,19,33,47,61] | ✓ |
| `num_attention_heads` | 32 | `query_heads`=32 | ✓ |
| `num_key_value_heads` | 8 | `kv_heads`=8 | ✓ |
| `head_dim` | 128 | 128 | ✓ |
| `hidden_size` / `intermediate_size` | 5120 / 17408 | 5120 / 17408 | ✓ |
| `num_hidden_layers` | 5 | `layers`=5 | ✓ |
| `rope_parameters.rope_theta` | 1e7 | 1e7 | ✓ |
| `sliding_window` / `use_sliding_window` | 2048 / true | `local_window`=2048 | ✓ |
| `layer_types` | 全 sliding_attention | —— | ✓ |
| `is_causal` | false | 块注意力非因果（swa） | ✓ |
| `input_embedding_scale` / `output_multiplier` | 未声明 → 默认 1 | 未施加（且 qwen3 的 policy 为 no-op） | ✓ |

⇒ **"dflash2 声明的草稿超参传错"这一假设被证伪**（至少对这些键）。剩下的错只能在**未声明的约定**
（块的语义、context K/V 的 positions、feature 拼接顺序）或 **verify/状态侧** —— 而后者今天已被实测出真偏差（见三）。

## 二、vLLM 对照数字核实（用户记忆的那组）
盘上可查的 vLLM 运行（`dl/clean_run.log`、`dl/anchor_rev.log`）配置是：

```json
{"method": "dspark", "model": ".../data/draft_model", "num_speculative_tokens": 7,
 "draft_sample_method": "greedy"}
```
指标：
```
spec_decode_num_drafts_total                 94
spec_decode_num_draft_tokens_total           658
spec_decode_num_accepted_tokens_total         1.0
per_pos: position0 = 1.0, position1..6 = 0.0
```
**⇒ 记录里那次（含干净环境）vLLM 是塌陷（1/658），不是"一切正常"**；而且 stock vLLM 没有 dflash2，
它走的是 **dspark** 路径、用的是 `data/draft_model`（与 我们 artifact 的 dspark 源
`models/qwen3.8-27b-dspark-zh/` 不是同一份，后者目录里**连 config.json 都没有**）。
若"正常"的那组在别的日志里（例如 vLLM 引擎自报的 acceptance length 行），请指给我文件名/行，
我立刻核。**空结果不能当证据**（`M_patchA_effect.md` §5 的同一条纪律）。

### 顺带发现：dspark 侧存在**真实**的参数不一致风险
`data/draft_model/config.json`（`Qwen3DSparkModel`）声明：

| 参数 | 该 ckpt | 我们 `DFlashConfig`（摘要记录） |
|---|---|---|
| `mask_token_id` | **190221** | 248077 |
| `target_layer_ids` | **[1,14,29,44,57]** | {4,16,28,40,52} |
| `rope_parameters.rope_theta` | **1e6** | 1e7 |
| `num_attention_heads` | **40** | 32 |

dspark 也塌在位置 1（`[17,1,0,…]`）。但这些差异是否成立取决于"我们 artifact 的 dspark 到底出自哪份 ckpt"——
`models/qwen3.8-27b-dspark-zh/` 只有 `model.safetensors`、**没有 config**，所以引擎的 dspark 常量目前
**没有可核对的权威来源**。这是一处该补的账（把 ckpt config 一起归档，或让引擎读 ckpt config）。

## 三、今天测出的**真参数/状态偏差**（接受率上限，与草稿无关）
同一 prompt、同一参数：
```
zh_plain vs zh_dflash2 : DIFFER at 29/96     （plain [...97844, 129775...] / spec [...97844, 95966, 129775...]）
num_plain vs num_dflash2: IDENTICAL (96 tok)
```
K 扫描（隔离机制）：
```
K=1: DIFFER at 62/96
K=3: DIFFER at 62/96   ← 与 K=1 同位置同 token
K=7: DIFFER at 29/96
```
⇒ K=1 就偏、且 K=1 与 K=3 同位置 ⇒ **不是"后续 draft 列被前面的列看到"的因果掩码污染**，
而是**与草稿无关的每轮状态/参数偏差**（paged KV 槽写入/回滚、GDN 循环状态 replay、
anchor/bonus 记账），随轮次缓慢漂移，只在分布接近（散文）处翻 argmax；可复制文本上不显现。
这正是契约 §7"无损"要求被破坏之处，也是 9-10 `M_patchA_effect.md` §3 遗留的同一症状。

## 四、下一步（按性价比排序）
1. **验证/状态链路**：审 `speculative_accept_greedy_drafts` 的列→licensed 映射、
   `speculative_select_accepted_hidden` + `scatter(state_destination_slots, continuation_hidden_store)`
   对**被拒列**的状态回滚、以及 GDN `RecordForReplay` 的 replay 一致性；判据 = spec 流与 plain 逐位一致。
   跨后端交叉验证：`--spec mtp --draft-tokens 3` 在同一 prompt 上是否也在同一 token 位置偏（若同位置 ⇒ 共享路径）。
2. **未声明约定的三项**（config 不管，只能对契约/上游）：块语义、context K/V 的绝对 positions、
   feature 拼接顺序 [5,19,33,47,61]（后两项已与上游实现逐条对上，前一项已对 §5 逐条对上）。
3. **参照草稿 A/B（一次定生死，需你点头）**：`data/draft_dflash2_ref/` 已在盘上（12:32 下载），
   用 `patch_dflash2.py --ckpt` 那套把它打成 artifact 跑一次：
   - 若接受率回到健康带（AL≥3）⇒ 引擎/verify 路被开脱，问题全在 `step_006000`（训练配方）；
   - 若仍 5% ⇒ 引擎/verify 侧确有问题（与第 1 项结论互相印证）。
   你之前说"别下参照草稿"，所以我没有自行使用；它已在本地，是否拿来做这次 A/B 你定。
