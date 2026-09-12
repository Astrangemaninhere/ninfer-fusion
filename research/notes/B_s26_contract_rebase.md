# B_s26_contract_rebase.md — S26: FlashNext 契约 346 缺源行 rebase

日期 2026-09-09 · B · 改动: `flashnext_bindings.py` (契约区重建) + `flashnext_convert.py`
(规则/形状/计划/自检引脚随动)。别名零放宽: 所有变更 = 删除(张量不存在) / 改指(真实键) /
新增(忠实命名的真实张量行)。

## 结果 (audit × 真实名单, 296,475 键)

| 量 | S21 (rebase 前) | S26 (rebase 后) |
|---|---|---|
| 契约条目 | 74,520 | **74,804** (−306 删 / +590 增) |
| engines_covered / missing | 74,174 / **346** | **74,804 / 0** |
| 消费键 matched | 74,210 (含 36 影子) | **74,931** (影子 0 语义并吞) |
| NVFP4 伴生 | 221,184 | 221,184 (不变) |
| 残余 | 1,081 (10 桶) | **360** = visual 333 + mtp 27 (`other`=0) |
| 会计恒等 | 74,210+221,184+1,081=296,475 | **74,931+221,184+360=296,475** ✓ |
| 完备 | false | false (残余=显式文档化, 见下) |

复现: `python3 ninfer-fusion-repo/tools/archkit/flashnext_bindings.py --audit-gguf _collab/M_flashnext_names.txt`
(exit 1 = complete:false 仅因文档化残余; missing_count=0)
`python3 ninfer-fusion-repo/tools/archkit/flashnext_convert.py --self-test --real-names _collab/M_flashnext_names.txt` → exit 0,
`[self-test][real] names=296475 resolved=74799 missing=0 quant_companions=221184 residue=360`。

## 逐项变更 (每行都引真实键; 全部在 M_flashnext_names.txt 中验证存在)

**删除 306 行 — 张量在清单中不存在 (rule 1: 不发明别名):**
- 96× `layer.{i}.{attn,ffn}_norm` — 全清单无 `input_layernorm/post_attention_layernorm`;
  该架构用 hyper-connection 替代逐层残差范数 (真键 = {attn,mlp}_hyper_connection.*, 每层 8 键全量在清单)。
- 96× `layer.{Q}.qsa.hc.{0-3}.{down,up}` — "hierarchical compression" 是对 hyper_connection 的误读,
  对应键不存在; 真实 hc 键已在新增行中按本名建模。
- 36× `gdn.conv_bias` — 无任何 `conv1d.bias` 键。
- 36× `gdn.beta` (猜测别名 beta.weight/b_proj.weight 不存在) → 由 name-faithful 的
  in_proj_a / in_proj_b 两行替代 (a/b 槽位对应关系非名字可定, 留引擎侧)。
- 36× 旧 `idx_wq/idx_wk/idx_norm` (wq_b/wk/k_norm 不存在) → 3 换 3 name-faithful:
  `idx_qk`←`self_attn.indexer.index_qk_proj.weight` (融合张量不能拆成两条源),
  `idx_q_norm`←`indexer.q_layernorm.weight`, `idx_k_norm`←`indexer.k_layernorm.weight`。
- 2× `ple.norm`(单范数猜测→真实三范数) 与 `ple.gate_query`(门张量不存在; norm_query 是范数不是门)。
- 3× `mtp.fc/norm/head_norm` (猜测键不存在) → 真实: fc 拆 `fc_embedding`+`fc_hidden`,
  双 pre-norm `pre_fc_norm_{embedding,hidden}`; 无 head 终范数。
- 1× `output_norm` — 无 `model.norm.weight`; 唯一顶层范数族张量是 mixer 的 hc_norm (见新增, 语义留引擎定)。

**改指 (行保留, 别名换成引证真实键):**
- `gdn.dt_bias` ← `dt_bias` (删掉 A_log 别名 — S21 影子并吞根除, shadow 里 0 条 A_log)
- 新增 `gdn.a_log` ← `A_log` (引擎 GdnWeights a_log 槽)
- `gdn.gate` ← `in_proj_z.weight` (z = GDN 族输出门惯例; 旧猜 gate_proj.weight 不存在)
- `ple.conv/key/value` ← `model.layers.1.ple.{conv1d,key_proj,value_proj}.weight`
- `ple.table` ← 128 个 `...ngram_embedding.shard_{0..127}.weight` 别名 (sidecar 构建器须拼接)

**新增 590 行 (全部 name-faithful, 维度未知者标 PENDING_SHAPE 共 460):**
- 384 hyper-connection: 每层 `{attn,mlp}_hc.{inject,norm,mix_down,mix_up}` ×48
- 108 GDN: `a_log` 36 + `in_proj_a` 36 + `in_proj_b` 36
- 48 `moe.shared_expert_gate` ← `mlp.shared_expert_gate.weight` (原残余)
- 7 PLE: `table_scale`(表级 weight_scale, sidecar) + `meta.{layer_multipliers,ngram_heads_offsets,ngram_heads_vocab_sizes}` + `norm_{key,query,conv}`
- 3 顶层 `hc_mixer.{norm,mix_down,mix_up}` ← `model.hyper_connection_mixer.*`
- 4 MTP 顶部 (见上)
别名唯一性: 148,853 个别名 0 碰撞 (self-test 1b)。

## 残余 360 — 为什么不从名字关闭 (rule 3 允许的余量)
- **visual 333**: 视觉塔; 引擎 text-first 政策 (同 lfm2/falcon 的 vision new_op 路径), 显式跳过。
- **mtp 27**: `mtp.layers.0.*` 完整草稿层 24 键 (self_attn + 融合专家 MoE `experts.{down,gate_up}_proj`
  + hyper-connections) + `mtp.hyper_connection_mixer.*` 3 键 — 需要引擎侧 MTP 层设计 (草稿头几何)
  才能定行, 名字层面无法忠实映射; 全部键已在清单清点, 不静默。

## 转换器随动
规则表/`artifact_shape` 按新引擎族重建; 未知维度族 → `(0,)` 哨兵 + `PENDING_SHAPE`(460 行,
`--checkpoint` 读真张量头回填, S27); `build_plan` 对 PENDING_SHAPE 不生成形状断言;
sidecar 行消费**全部**存在的别名 (ple.table 128 分片不再误报计划外); `_sample_engines` 更新;
self-test 引脚: entries=74804, sidecars=5, PENDING_SHAPE=460, real: missing=0/resolved=74799/
companions=221184/residue=360。

## 下载后配方 (更新版, 取代 B_s21_contract_fix.md 的数字)
```bash
# 0) 分片齐全预检 (同 S21)
# 1) 契约审计:  python3 ninfer-fusion-repo/tools/archkit/flashnext_bindings.py --audit $M/model.safetensors.index.json
#    新期望: matched_sources=74931, engines_covered=74804, missing_count=0,
#            quant_companions=221184, residue={visual:333, mtp:27}, accounting_ok=true;
#    exit 1 仅因 complete:false(文档化残余)。matched≠74931 → 版本不同, 停。
# 2) 回归门:    python3 ninfer-fusion-repo/tools/archkit/flashnext_convert.py --self-test \
#                --real-names _collab/M_flashnext_names.txt   → [self-test] PASS, exit 0
# 3) 计划:      python3 .../flashnext_convert.py --plan --out _collab/B_s26_plan.json
#    新期望: plan: 74804 entries, 74224 pending, 5 sidecar
# 4) 形状核对 (首个需真分片文件的步骤; 现在同时回填 460 个 PENDING_SHAPE 维度):
#    python3 .../flashnext_convert.py --checkpoint $M
#    期望: missing=0, residue=360; mismatch_count = 形状层新发现 (可行动清单)。
#    WATCH: FileNotFoundError=下载未完; ple.table 128 分片应全被 sidecar 分支认领。
```
S27 (writer 路径, 未开始): 真权重读取 → NVFP4 打包 (221,184 伴生随基) → 128 分片 PLE 拼接
sidecar → artifact 写出。本产物未动。

## 附记
- audit 的 shadow_matches 语义更新: 127 条 = ple.table 分片经非首别名命中 (设计使然, 单逻辑
  sidecar 表); 语义并吞型影子 (A_log) 为 0。
- py_compile 两文件通过; 未构建、未碰 GPU; 临时日志 `_b_s26_*.log` 已清理。
