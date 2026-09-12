# B_s21_contract_fix.md — S21: FlashNext 契约 × 真实 checkpoint 名单对齐

日期 2026-09-09 · B · 改动文件:
- `ninfer-fusion-repo/tools/archkit/flashnext_bindings.py`: 新增 `canonical_source_key()`
  (唯一规范化点: `model.language_model.`→`model.`, 74,520 条 alias 未动) + 重写 `audit()`
  (规范化匹配 / NVFP4 伴生派生 / 残余分桶 / 影子匹配暴露 / 会计恒等式; 精确索引 O(1))。
- `ninfer-fusion-repo/tools/archkit/flashnext_convert.py`: `SourceIndex` 走规范化;
  `verify_plan()` 伴生/残余分类 (无静默丢弃, `accounting_ok`); `self_test()` 主检查改为
  消费真实名单 (`--real-names`), 旧自指检查降级为 [smoke] 标签; CLI 增 `--real-names`。

## 数字 (复算命令 + 原始输出)

```
python3 _collab/_b_tmp/b_s21_probe.py        # (探针已并入本报告, 命令可复现)
names: 296475
RAW (no canon): matched keys=1 engines=1 / 74520 entries          ← 独立复现 M 的 1/74520
canonical:     matched_sources=74210  engines_covered=74174/74520  ← 与 M 的 strip_lm 一致
quant_companions=221184 (=73728 专家四元组 × 3 伴生)
missing_count=346   unmatched(残余)=1081   accounting_ok=true
74,210 + 221,184 + 1,081 = 296,475 ✓ 每键必居其一
```

## 残余 1,081 逐项 (residue_by_bucket, `other` 桶为空 — 全部有解释)

| 桶 | 数 | 解释 / 去向 |
|---|---|---|
| quant companions | 221,184 | 48 层×512 专家×{weight_scale,weight_scale_2,input_scale}; 契约不立条目, 转换器随基张量打包 (派生已实现于两文件) |
| visual | 333 | `model.visual.*` 视觉塔; 引擎 text-first, 同 lfm2 vision new_op (跳过可, 须显式声明) |
| attn_hyper_connection | 192 | 48 层×4 键; 该架构用 hyper-connection **替代逐层残差范数** — 契约完全没有此板块 |
| mlp_hyper_connection | 192 | 同上 (MLP 侧) |
| hyper_connection_mixer | 3 | `model.language_model.hyper_connection_mixer.*` 顶层 mixer (含 hc_norm = output_norm 的现实体) |
| ple_table_shards | 132 | PLE 表在真 ckpt 是 **128 分片** + weight_scale + 3 元数据 (contract 以为整块 20M×160) |
| gdn_in_proj_extras | 108 | 36 层×{in_proj_a,in_proj_b,in_proj_z} = beta/gate 的真实键 (契约猜成 beta.weight/gate_proj.weight) |
| shared_expert_gate | 48 | 共享专家路由门, 契约无对应引擎 |
| indexer_fused | 36 | 12 QSA 层×{index_qk_proj(融合),q_layernorm,k_layernorm}; 契约猜 wq_b/wk/k_norm 三件套 |
| mtp | 31 | 真实 MTP: fc 拆 fc_embedding/fc_hidden + 整层 self_attn+MoE(融合专家) + pre_fc_norm×2 + mixer |
| ple_misc | 6 | `layers.1.ple.{conv1d,key_proj,value_proj,norm_conv,norm_key,norm_query}` — 位置与命名全偏 |

## 缺源引擎清单 (任务 5 — canonical 化后仍无源, 共 346)

| 引擎族 | 数 | 真实候选键 (需引擎侧定语义后立契约行) |
|---|---|---|
| layer.N.{attn,ffn}_norm | 96 | 无逐层范数; 候选 = {attn,mlp}_hyper_connection.hc_norm.weight |
| layer.Q.qsa.hc.{0-3}.{down,up} | 96 | 不存在 (契约把 hyper_connection 误读成 hierarchical compression); 应删行或重指 |
| gdn.conv_bias | 36 | 真实无 conv1d.bias → 引擎槽应改可选 |
| gdn.beta | 36 | 候选 in_proj_a.weight / in_proj_b.weight (语义待引擎定) |
| gdn.gate | 36 | 候选 in_proj_z.weight (z=sigmoid 输出门) |
| qsa.idx_{wq,wk,norm} | 36 | 融合 index_qk_proj.weight + q/k_layernorm.weight (需拆/并变换) |
| ple.* | 6 | layers.1.ple.* (表=128 分片拼接; gate_query 现实体疑为 norm_query) |
| mtp.{fc,norm,head_norm} | 3 | fc_embedding+fc_hidden 拆分 + pre_fc_norm_{embedding,hidden} |
| output_norm | 1 | hyper_connection_mixer.hc_norm.weight |

**影子匹配 (审计新增暴露)**: 36 个 `A_log` 键经 alias[1] 落到 `gdn.dt_bias` — 两个语义不同的
张量被一条契约项吞并 (引擎 GdnWeights 有 a_log/dt_bias 两槽, converter 头注已预告需拆项)。
拆项前 converter 侧 A_log 计入残余 (converter 残余 1,117 = 1,081+36, missing 345+1 sidecar
ple.table = 346 — 两套口径自洽)。

## 20+ 点检 (任务附加项: 命中必须落在**正确**引擎上) — ALL OK (21/21)
```
embed_tokens.weight -> token_embd            lm_head.weight -> output_head
0.linear_attn.in_proj_qkv -> layer.0.gdn.in_qkv   (0∈GDN)   2.linear_attn.in_proj_qkv -> layer.2.gdn.in_qkv
0.linear_attn.conv1d -> layer.0.gdn.conv     0.linear_attn.norm -> layer.0.gdn.norm
0.linear_attn.out_proj -> layer.0.gdn.out    0.linear_attn.dt_bias -> layer.0.gdn.dt_bias (alias[0])
0.linear_attn.A_log -> layer.0.gdn.dt_bias   (SHADOW, 有意暴露)
3.self_attn.{q,k_norm,o}_proj -> layer.3.qsa.{q,k_norm,o}   (3∈QSA)
3.self_attn.indexer.index_qk_proj -> 无命中 (融合键, 契约三件套不匹配 = 正确不误中)
0.mlp.gate -> layer.0.moe.router             0.mlp.shared_expert.gate_proj -> sh_gate (非 e{x}.gate!)
0.mlp.experts.511.down_proj -> layer.0.moe.e511.down          47.mlp.experts.0.gate_proj -> layer.47.moe.e0.gate
0.mlp.shared_expert_gate / 1.ple.key_proj / mtp.fc_embedding / visual.* -> 无命中 (= 缺口, 正确可见)
```
未放松任何 alias 提升命中数; 契约别名全为字面名 (149,184 个, 0 碰撞)。

## 下载完成后的实操配方 (coordinator 追加要求)
设 `M=<下载目录>` (含 `model.safetensors.index.json` 与全部分片), 在
`/mnt/c/Users/User/Documents/ziqinzhang` 下按序:

```bash
# 0) 分片齐全预检 (只要文件存在性, 不读权重)
python3 -c "import json,os;M='$M';m=json.load(open(M+'/model.safetensors.index.json'))['weight_map'];s=sorted(set(m.values()));miss=[f for f in s if not os.path.exists(os.path.join(M,f))];print(len(s),'shards, missing',len(miss),miss[:3])"
# 期望: N shards, missing 0。缺片 → 下完再继续 (步骤 4 会 FileNotFoundError)。

# 1) 契约 × 现场 index 双向审计 (仅名字)                                  [无需权重]
python3 ninfer-fusion-repo/tools/archkit/flashnext_bindings.py --audit $M/model.safetensors.index.json
# 期望: matched_sources=74210, engines_covered=74174, quant_companions=221184,
#       missing_count=346, accounting_ok=true; exit code = 1 (complete:false = 真缺口, 非跑挂)。
# 若 matched != 74210 → checkpoint 版本与实测名单不同 → 停, 重测, 勿继续。

# 2) 回归门 (钉死 M 名单的固定事实)                                       [无需权重, 需 CPU torch]
python3 ninfer-fusion-repo/tools/archkit/flashnext_convert.py --self-test --real-names _collab/M_flashnext_names.txt
# 期望: [self-test][real] names=296475 resolved=74174 quant_companions=221184; [self-test] PASS; exit 0。

# 3) 计划落盘                                                             [无需 checkpoint]
python3 ninfer-fusion-repo/tools/archkit/flashnext_convert.py --plan --out _collab/B_s21_plan.json
# 期望: plan: 74520 entries, 73800 pending, 1 sidecar。

# 4) 首个需要真实张量文件的步骤: 形状双向核对 (开每个分片头, 只读 shape 元数据)
python3 ninfer-fusion-repo/tools/archkit/flashnext_convert.py --checkpoint $M
# 期望: missing_count=345(+ple.table sidecar), quant_companions=221184, residue=1117;
#       mismatch_count = 新的可行动清单 (契约猜测形状 vs 真 NVFP4 张量; 专家 .weight 形状
#       会揭示打包布局)。exit 1 直到 346 行缺口回填。
# WATCH: FileNotFoundError = 下载未完; mismatch 数暴增 = 布局假设错, 不是数据坏。
```
**边界声明**: 步骤 1–3 只需名字/元数据; 步骤 4 需真实分片文件 (仅 header, 不载权重);
**真实权重数据读取与 artifact 写出尚不存在** — `flashnext_convert.py` 是骨架, 产出 .ninfer
还需: (a) 346 行契约回填 (上表逐行已给真实候选键), (b) artifact writer + NVFP4 打包变换,
(c) 128 分片 PLE 表拼接进 sidecar。建议拆给 S22 (契约回填) / S23 (writer)。
真实接受率/对拍要在 artifact 之后 (GPU 窗口)。

## 复现
探针 `_collab/_b_tmp/b_s21_probe.py` 与日志 `_b_s21_probe.log`/`_b_s21_selftest.log` 已清理;
上述命令 1/2 即固化数字 (audit JSON 全文含 shadow 36 条与 missing_by_family)。
py_compile 两文件通过; `--self-test` 无 `--real-names` 时打印降级说明。
