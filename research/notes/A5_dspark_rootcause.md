# A5 · DSpark（DFlash v1）11.22% 接受率：H1 的裁决 + 两条剩余候选与判别实验

本回合**只读分析 + 补丁草案**：未改任何 `src/**`（WSL 树 `dflash_impl.h` 前后 md5 均为
`02cb3bc9b511a5c2341d59a47f614931`）、未编译、未开 nvcc/ptxas、未跑 GPU/引擎；未写 `tools/**`。
新增只读工具（本回合实际执行过、结果在 §3.1）：
`_collab/A5_weights_audit.py`（artifact ⇄ checkpoint 逐字节审计）、`A5_parse_round0.py`（探针读数器）、
生成器 `A5_mkprobe.py` / `A5_mkblockrows.py`，补丁 `A5_round_probe.diff`、`A5_block_rows.diff`
（两者 `patch -p1 --dry-run` 均 **rc=0**，见 §4）。

> **行号说明（重要）**：E5/E6 文档里的 `text_context_impl.h:1221/1244` 是**旧行号**。当前
> `/home/user/ninfer-fusion`（= 真正在编译的树，且 E6 补丁**已落地**：`dflash_impl.h` md5
> `02cb3bc9…`，Windows 镜像 `ninfer-fusion-repo` 仍是落地前的 `60a51d1c…`）里两个 tap 点在
> **`:1227`（full 层）与 `:1250`（GDN 层）**；引用行号时请以本文件为准，并对旧文档做 46 行偏移。

---

## 0. 结论速览

1. **H1（"tap 层号/tap 点错"）在代码级被排除**：引擎 tap 抓的是**该层 post-MLP residual**
   （等价于 HF `hidden_states[layer+1]`），而 DSpark 参考实现消费的正是 `hidden_states[i+1]`
   （`i ∈ target_layer_ids`）。两边**同一条约定**，层号也对得上（`[4,16,28,40,52]` ↔ 27B
   `DFlashConfig::target_feature_layers`）。证据链见 §1.1–§1.3；**残余风险**是"上游训练侧是否也用
   `[i+1]`"只能用一次 dump 数值复核（§1.4），不能只靠读码终结。
2. **本回合新做的两步离线取证**（都不需要 GPU，也都**不需要 target 权重**）：
   - **草稿权重逐字节审计**：正在服务的那份 artifact 的 **55/55 个 `dflash/*` 张量与 checkpoint
     逐字节相同**（含 `markov_w1/w2` **未互换**、`query_key_value == [q;k;v]`、`gate_up == [gate;up]`、
     `context_key←k_proj / context_value←v_proj` **未互换**）。⇒ "权重绑定/换位错"整族假设被排除。
   - **层号溯源**：dspark `target_layer_ids [4,16,28,40,52]`（`tmp/dspark-config.json:19-25`）
     ↔ `src/targets/qwen3_6_27b/impl/config.h:113` 完全一致；DFlash2 的 `{5,19,33,47,61}`
     （`config.h:146`）与它自己的训练脚本注释一致（`data/df2pilot/train_dflash2.py:43`：
     `TARGET_LAYER_IDS = [5, 19, 33, 47, 61]  # collect layers (matches engine DFlash2Config)`）。
3. **排出的下一个最可能原因（两条，均 DSpark 专属、均不伤 DFlash2）**：
   - **候选 A（首选）草稿侧 RoPE 约定**：checkpoint 声明 `rope_type=yarn, factor=32,
     original_max_position_embeddings=8192`（`tmp/dspark-config.json:54-68`），参考实现照办
     （`MuseGlimmerAssistantRotaryEmbedding`，见 §1.2）；而引擎对草稿的 block 与 context 两条
     rope 都传 `Config::rope_theta`（纯 rope：`dflash_impl.h:177`、`:304`），target 侧也只有
     `--yarn` 时才切 **factor-4** 的 `ops::rope_yarn4`（`rope.h:44-53`、`rope.cpp:143-153`）
     ⇒ dspark 需要的 factor-32 引擎里**根本没有**。
   - **候选 B 次选）block 行→verify 列约定**：引擎 bf16 分支把**第 0 行（anchor 行）**的 hidden
     当成第 0 个草稿（`dflash_impl.h:447-451`，`source_column_offset = 0`），而 HF 参考取
     `head(last_hidden_state[:, 1:])`（`candidate_generator.py:1694-1697`）、引擎自己的 W8 DFlash
     分支也取 rows 1..k ⇒ 草稿"预测位置 p 却按 p+1 校验"，**一路整体错位**。
4. **判别实验很便宜**：一个 10 秒 GPU dump（`_collab/A5_round_probe.diff` 已备好，零成本 no-op）
   + 纯 CPU 复算（用本地 checkpoint），按 **(rope: 纯 rope / yarn-32) × (rows: 0..k-1 / 1..k)**
   四种组合各算一遍，能复现引擎 drafts/logits 的那个组合即真约定（§2.3）。

---

## 1. 任务 1：把 H1 变成可判定

### 1.1 引擎 tap：抓的是哪个张量（代码级）

| 事实 | 位置 |
|---|---|
| full 层：`attn_mix(...)` → `mlp_tail(full.post_attn_norm, full.mlp, x, ph)` → **立刻** `tap.capture_layer(layer, x, …)` | `text_context_impl.h:1226-1227` |
| GDN 层：同样顺序 | `text_context_impl.h:1249-1250` |
| `mlp_tail` = `rmsnorm(x, post_attn_norm, …) → h`，再 `Variant::post_mixer(h, mlp, x, ph)` | `text_context_impl.h:1187-1193` |
| 27B 的 `post_mixer` = `linear_swiglu(h, gate_up) → activation`，再 **`linear_add(activation, down, residual)`**（即 **`x ← x + down·silu(gate_up·rmsnorm(x))`，原地写回 `x`**） | `src/targets/qwen3_6_27b/impl/variant.cpp:369-377` |
| `run_layers` 用**同一个 `Tensor& x`** 串起所有层（tap 后 x 就是下一层的输入） | `text_context_impl.h:1144-1203` |
| `capture_layer` 只保留 `layers = DFlashConfig::target_feature_layers` 里的层，按**升序**索引写入 `feat[index·hidden …]` | `text_context_impl.h:276-307`、`dflash_impl.h:59-64/76-83` |

⇒ **引擎 tap 的张量 = 第 `layer` 个 decoder layer 的 output residual stream（post-MLP）**。
在 HF 约定下（`hidden_states[0]` = embedding 输出）这就是 **`hidden_states[layer+1]`**。

### 1.2 参考侧（HF / vLLM）恰好消费同一项

- **HF 官方 DFlash/DFlash2 候选生成器**（本地 `transformers 5.8.1`，正是 dspark checkpoint 的
  `transformers_version`）：
  `context_hidden_states = cat([model_outputs.hidden_states[i + 1][:, :num_last] for i in target_layer_ids])`
  —— `transformers/generation/candidate_generator.py:1651-1657`。
- `hidden_states` 的构造：`transformers/utils/output_capturing.py:112-119`
  （`capture_initial_hidden_state=True` ⇒ 第一个被 hook 的层的**输入**先入队 = embedding 输出，
  其后每层输出依次入队；`tie_last_hidden_states` 只覆盖最后一项）⇒ **`hidden_states[i+1]` = 第 i 层输出**。
- **vLLM 侧**同样把 `target_layer_ids` 当 EAGLE3-style aux 层用：`qwen3_dflash.py:98-106`
  （`_get_dflash_fc_input_size`）、`:768-816`（`combine_hidden_states` = `fc`）、`:592-663`
  （`precompute_and_store_context_kv`：**先 `fc` 再 `hidden_norm`（`output_norm_enc`）再 k/v 投影**）。
- **HF 助手段模型**（`models/muse_glimmer_assistant/modeling_muse_glimmer_assistant.py:295-312`）
  给出本引擎逐项对应的另一条独立证据：`encoder = fc → output_norm_enc`（= 引擎
  `feature_projection → context_norm`，`dflash_impl.h:136-141`）；**同一份 `k_proj/v_proj` 同时作用于
  context 与 block**（`:171-176` `kv_hidden_states = cat([context_hidden_states, hidden_states], dim=1)`）
  ⇒ 引擎 `context_key/context_value` 必须等于 checkpoint 的 `k_proj/v_proj`（§3.1 已按字节证实）；
  block 行双向、context 行因果（`:433-441`）、q/k RMSNorm、RoPE 在 q 与 k 两侧都上（`:199-201`）。

**⇒ 引擎 tap 语义 == HF `hidden_states[i+1]` == DSpark 参考消费的那一项；层号列表也与 checkpoint
config 一致。H1 的"错一层 / 抓错张量"版本因此不成立。**

残余风险（**待证实**，只能用 §1.4 的数值复核终结）：
1. 上游**训练**代码是否也用 `[i+1]`（inference 侧 HF `5.8.1` 与移植文档 `DSPARK-ADAPTATION.md:57-59`
   的 "offset=1" 说法一致，但这只是"参考实现"级证据）；
2. 引擎 tap 的 `x` 是否逐位等于 HF 的 `hidden_states[i+1]`（数值等价只在 target 自身正确时成立；
   见候选 A：target 与 draft 的 rope 约定都可能与 HF 不一致）。

### 1.3 离线可验证做法（E-2 的精确化：命令 + 判据 + 输入文件 + 缺口）

**E-2a（~10 s GPU，单独跑，不与你现在的训练抢显存；这是唯一需要 GPU 的一步）**
```bash
BIN=/home/user/ninfer-fusion/build/apps/ninfer
P=$(head -c 1500 /home/user/spec_prompt_base.txt)          # 与 E-1 完全同一个 prompt
mkdir -p /tmp/hsd
NINFER_HS_DUMP_DIR=/tmp/hsd NINFER_HS_DUMP_TOPK=1 \
  $BIN /home/user/models/qwen3_8_27b_nvfp4_dspark.ninfer \
  --prompt "$P" --max-new 1 --greedy --no-thinking --no-cuda-graph \
  --spec dflash --draft-tokens 7
# 产物 /tmp/hsd/chunk_0000000.bin（magic NHS1）：
#   i32 tokens | i32 ids[tokens] | u16 feat[25600×tokens]（5 层按捕获顺序）| u16 last[5120×tokens]
#   （NINFER_HS_DUMP_TOPK=1 时再追加 i32 top1[tokens]；写入点 text_prefill_impl.h:90-181）
```
**输入文件（比对所需，缺一不可）**
| 文件 | 状态 |
|---|---|
| dspark 草稿 checkpoint `models/qwen3.8-27b-dspark-zh/model.safetensors`（2.72 GB，62 张量） | ✅ 在 Windows 侧，且已验证 artifact 用的就是它（§3.1） |
| 引擎 dump（上面 10 秒产物） | ⏳ 需 GPU 跑一次 |
| **完整 target HF 权重**（`Qwen/Qwen3.8-27B`，bf16 ≈54 GB / fp8 ≈27 GB） | ❌ **不在本机**：`data/Qwen3.8-27B` 只有 `model.embed_tokens.weight` + `lm_head.weight`（5.08 GB，`model.safetensors.index.json` 只列这 2 项）⇒ **E-2 原方案（与 HF 逐层 cos）当前跑不了**，必须先补权重，或改走 E-2b |

**E-2b（不需要 target 权重，纯 CPU，用 §2.4 的探针 dump）**：dump 出来的
`feat` 是引擎喂给草稿的**原样** 25600 维输入，把它当"已知输入"在 CPU 上复算草稿前向（fc →
context_norm → 5 层 dual-source attention → final_norm → 复用 target head），**与同一轮引擎自己产出的
`drafts`/`logits` 逐位比**。这既不需要 target 权重，也不依赖"HF 是不是对的"这一前提，
还能**同时**分辨候选 A（rope 约定）与候选 B（行约定）——见 §2.3。
数值复算需要的两块输入由此凑齐：**context 特征**（`NINFER_HS_DUMP_DIR` 的 prefill dump）
+ **第 0 轮 block/positions/anchors/drafts/logits**（`_collab/A5_round_probe.diff`）。

**判据（E-2a 若补上权重）**
- 五层 `cos(feat[5L·5120:(5L+1)·5120), hidden_states[L+1])>0.999` ⇒ tap 语义对（H1 死）；
- 另做 **±1 位移对照**：`cos(feat[p], hs[i+1][p±1])`——"错一层"与"错一个 token"必须分开判，
  只比同索引会把后者误判成"数值精度问题"；
- `feat` 与 `hidden_states[i]`（错一层）或与 post-attention hidden 对上 ⇒ H1 成立。

**没有 GPU 时还能做的（本回合已做完）**
1. **config/形状/命名溯源**：`target_layer_ids` 与引擎 `target_feature_layers` 一致（§0.2）；
2. **artifact ⇄ checkpoint 逐字节审计**（§3.1）——把"权重绑定错"整族排除；
3. **本地 checkpoint 的 rope 参数**（`tmp/dspark-config.json:54-68`）与引擎草稿 rope 调用点
   （`dflash_impl.h:177/304`）对照 —— 候选 A 就是这一步推出来的；
4. **离线 YaRN 频表复算**（建议下步，纯 CPU）：用 HF `ROPE_INIT_FUNCTIONS["yarn"]` 复算
   factor-32 的 `inv_freq`/`attention_scaling`，与引擎 `Config::rope_theta=1e7` 的纯 rope 比较，
   给出"位置 p 处逐维角度误差"的定量曲线（**待证实**：本回合没算，预算用在 §3.1 的审计上）。

### 1.4 位置剖面判读（任务 3）

| 臂 | `accepted by pos` | rounds | p0 | p1\|p0 | p2\|p1 | p3\|p2 |
|---|---|---|---|---|---|---|
| dspark | 15,5,2,0,0,0,0 | 56 | **26.8%** | 33.3% | 40.0% | – |
| dflash2 | 23,11,3,1,0,0,0 | 55 | 41.8% | 47.8% | 27.3% | 33.3% |
| MTP | 23,12,8 | 43 | 53.5% | 52.2% | 66.7% | – |

判读（结论先行）：
1. **与 H1 的"广义版"（输入链/上下文侧的语义或角度不对）一致，与"局部块约定错"矛盾。**
   若病根在 block 内部约定（候选 B：整路错位一行/一列），位置 0 会近乎归零（那一路的"预测对象"
   根本不是位置 F+1 的 token），实测 26.8% 不支持它当**主因**；若病根是"块内位置/rope 随位置恶化"，
   应看到 p0 ≥ p1 ≥ p2 的快速衰减，实测（在样本量允许的范围内）也不是。
   而"上下文侧"的病（特征语义、context K/V 的角度）对**所有位置同等压降**（每行的 q 都要看整段
   context，context 才是主导项，块内 7 行之间的相对角差本来就只有几个位置的距离）⇒ 形态是
   **p0 整体低于健康对照、且各位置大致同量级**——这正是观测到的样子（26.8% vs 41.8%/53.5%）。
2. **p0<p1<p2 的"回升"不要过度解读**：`n1=5`、`n2=2` 的样本下，p1、p2 的二项标准差约
   ±12 / ±22 个百分点 ⇒ 与"平坦"不可区分；把一个 n=2 的 40% 当成"位置 2 更好"是过读。
   真正稳健的信号只有一个：**p0 只有 dflash2 的 0.64×、MTP 的 0.50×**，且**没有位置衰减**。
3. 与历史记录的自洽性：候选 B（整路错位）**能解释**移植文档里 "k=1 时 0%"（`DSPARK-ADAPTATION.md:151-156`）
   ——`k=1` 时引擎只拿出 **第 0 行（anchor 行）** 的 hidden 当草稿，而 anchor 行在参考约定里
   **不是**一个草稿行；但注意该 0% 在移植文档里同时出现在块语义修改**前后**，所以它**既不能证实
   也不能否证**候选 B ⇒ 落到 §2.3 的实验。

---

## 2. 修法

### 2.1 候选 A（首选）：草稿侧 RoPE 约定

- 证据：checkpoint `tmp/dspark-config.json:54-68` 同时声明 `rope_parameters` 与 `rope_scaling`
  = `{rope_type: yarn, factor: 32, original_max_position_embeddings: 8192, beta_fast: 32, beta_slow: 1}`，
  且 `rope_theta: 1e7`；参考实现的 rotary 直接读它（`modeling_muse_glimmer_assistant.py:314-368`：
  `rope_type != "default" ⇒ ROPE_INIT_FUNCTIONS[rope_type]`，即真 YaRN，含 `attention_scaling`）。
- 引擎侧：草稿 block 的 q/k rope 与 context K 的 rope **都**用纯 `Config::rope_theta`
  （`dflash_impl.h:177`（append 侧）、`:304`（propose 侧））；target 侧只有 `--yarn` 才走
  `ops::rope_yarn4`（`text_context_impl.h:504-505/618-619/662-663/1000-1001`），而 yarn4 =
  **factor 4**、`attention_scaling=1.1386`（`include/ninfer/ops/rope.h:44-53`、
  `src/ops/wrapper/rope.cpp:143-153`：`dflash_yarn4` 域恰好是 `rotary_dim=128 & head_dim=128`，
  正是草稿的几何）⇒ **factor-32 的 YaRN 引擎里没有实现**，与 checkpoint 无论如何都对不上。
- 为什么**只伤 dspark、不伤 DFlash2（也不伤 MTP/qwen 基线）**：DFlash2 草稿是我们自己在
  `data/df2pilot/train_dflash2.py` 里训的，训练用的 rope 是**硬编码纯 rope**（`:41` `ROPE_THETA=1e7`、
  `:79-88` `apply_rope`），且训练特征/教师都取自引擎自己的 dump（`:43` 注释、`data/hs_cache/seq_*.npz`
  的 `feat/last/ids16/vals16`）⇒ 训练与推理**同一套（错）约定**，自洽免疫。
  MTP/W8-DFlash 同理走各自已被验证的路径。**dspark 是唯一"约定由外部 checkpoint 决定"的臂。**
- 影响面：修复只改**草稿**侧的 rope 选择（以及可选地让 target 侧按 checkpoint 的 factor 走），
  对 DFlash2/MTP/W8/qwen **零影响**——只要把新参数挂在 `DFlashConfig`（27B 的 dspark）上，
  并在 `if constexpr (Config::rope_type == ...)` 里分派（默认 = 现状）。
- 复杂度警示：真正做对需要新增一个 **factor 可配的 YaRN rope op**（现在是静态 yarn4），
  属于 `src/ops/**` 改动 ⇒ 本回合**只给判别实验与探针，不给这个内核的盲写补丁**（不确定性太高）。

### 2.2 候选 B（次选）：block 行 → verify 列约定

- 引擎 bf16 分支：`source_column_offset = 0`（`dflash_impl.h:439-455`）⇒ `drafts[0..k-1]` = 第
  0..k-1 行（第 0 行 = anchor 行）的 hidden，随后 verify 把 `drafts[m]` 放在**位置 F+1+m**
  （列约定见 E6 文档 §1 表）。
- 参考侧：`candidate_logits = main_model_output_embeddings(outputs.last_hidden_state[:, 1:])`
  （`candidate_generator.py:1694-1697`）⇒ **anchor 行不进 proposal**；引擎自己的 W8 DFlash
  也是 `source_column_offset = 1`（同一段代码的 `if constexpr (!Config::bf16_weights)` 分支）。
- 这一处正是移植时被**主动改掉**的地方（`DSPARK-ADAPTATION.md:152-156`：把 "width=k+1 noise rows
  取 rows 1..k" 改成 "V=k、pack 行 0..k-1"）⇒ 若参考读法反了，这一改就是把整路错位一行。
- 判别指纹：候选 B 成立 ⇒ **`--spec dflash --draft-tokens 1` 的接受率≈0**（只有 anchor 行一路）
  且 p0 应显著低于 p1/p2。实测 p0=26.8% 最低、但**不是 0** ⇒ 若 §2.3 证明 B 成立，则
  "k=1 的 0%" 是它的直接指纹，而 p0 的非零来自"anchor 行自身预测"的偶然命中。
- 补丁草案（**给出来了**）：`_collab/A5_block_rows.diff`（2 hunk / +5 −12，`patch -p1 --dry-run` rc=0）。
  它只动 `bf16_weights` 分支：① 删掉 E6 补丁的第二半（把 `attention_valid` 从 `width` 复位成 `k`
  那段），使草稿块自己保持 `width=k+1` 个 live 列（行 1..k 成为 frontier+1..frontier+k 上的 k 个
  mask 行）；② 把 pack 偏移恢复为 1。W8/DFlash2/Muse 分支**一字节不变**。

### 2.3 判别实验（一次 10 秒 GPU + 纯 CPU，同时裁决 A 与 B）

1. 打上 `_collab/A5_round_probe.diff`（env 未设时零成本；`NINFER_DSPARK_DUMP_DIR` 设了就只在
   **第 0 轮**写一个 `dspark_round0.bin`）。
2. 跑一条与 E-1 相同的 dspark arm，令 `NINFER_DSPARK_DUMP_DIR=/tmp/dsp NINFER_HS_DUMP_DIR=/tmp/hsd`
   （前者给第 0 轮的 block/positions/anchors/drafts/base-logits，后者给第 0 轮的 context 特征）。
3. CPU 复算（Windows 侧即可，形状/命名映射已由 §3.1 的字节审计钉死）：`python _collab/A5_parse_round0.py`
   读 dump → 用 `models/qwen3.8-27b-dspark-zh/model.safetensors` 复算草稿前向，按
   **(rope: 纯 rope / yarn-32) × (rows: 0..k-1 / 1..k)** 四种组合各算一遍。
4. 判据：**能逐位复现引擎 `drafts`（以及 base logits 的 argmax 序列）的那个组合就是真约定**；
   若"rows 1..k + yarn-32"复现 ⇒ A、B **都**是 bug（分别修）；若只有其一 ⇒ 只修其一；
   若四种都对不上 ⇒ 病在 target 侧特征（回到 §1.4 的 E-2a，需要补 target 权重）。

### 2.4 补丁草案（都不落地）

- `_collab/A5_round_probe.diff`：**诊断探针**（A5_round_probe）。2 hunk / +74 / −0；
  `patch -p1 --dry-run` **rc=0**；env 未设时零开销 no-op；不改变任何既有路径的数学。
  它把 `dflash_impl.h` 第 0 轮的输入与输出落盘，使 §2.3 的离线复算成为可能（这正是
  `DSPARK-ADAPTATION.md:157-158` 留给下一步的那件事，现在有确定性写法了）。
- `_collab/A5_block_rows.diff`：**候选 B 的修法草案**（不是落地建议，先做 §2.3）。
  2 hunk / +5 / −12；`patch -p1 --dry-run` **rc=0**；两处编辑都在 `if constexpr
  (Config::bf16_weights)` 内 ⇒ 对 W8 DFlash（35B）、DFlash2、Muse、qwen 基线**零影响**。

---

## 3. 本轮实际取到的离线证据（可复现）

### 3.1 artifact ⇄ checkpoint 逐字节审计（本回合实跑）

```
python _collab/A5_weights_audit.py                                                   # 服务用 v2 副本
python _collab/A5_weights_audit.py "\\wsl$\Ubuntu\home\user\models\qwen3_8_27b_nvfp4_dspark.ninfer"
```
两次输出一致：
```
  1:1 fc == fc.weight                            OK
  1:1 hidden_norm == hidden_norm.weight          OK
  1:1 norm == norm.weight                        OK
  markov: w1<-w1 & w2<-w2 (identity)             OK identity(w1=True,w2=True) swap(w1=False,w2=False)
  layers: norms/o_proj/down (6 per layer)        OK
  layers: gate_up == [gate; up]                  OK
  layers: query_key_value == [q; k; v]           OK
  layers: context_key<-k_proj, context_value<-v_proj OK
RESULT: all dflash/* tensors are byte-identical to the nominal checkpoint tensors
```
（55 个 `dflash/*` 张量、5 层 + markov 全查。实现细节：artifact 的 `objects[].offset` 是相对
**4096 对齐后的 payload 起点**（v2 上 = 188416 = 46×4096，不是 `16+json_len=185583`），
safetensors 的数据段起点是 `8+header_len`；两者都用"在另一容器里全局搜到模式"的方式实测确认，
脚本里写了结论。）

### 3.2 已排除项汇总（本回合新增的部分）

| 假设 | 状态 | 依据 |
|---|---|---|
| tap 层号错（`hidden_states[L]` vs `[L+1]`） | **排除**（读码级） | §1.1–§1.3；残余风险=上游训练侧，待 E-2 |
| 草稿权重绑定错（k/v 互换、markov 互换、qkv 行序、装错 checkpoint） | **排除**（字节级，实测） | §3.1 |
| artifact 选错（服务的那份 ≠ 审计的那份） | **排除**（对象集 + dflash 载荷字节，实测） | §3.1 第二行命令 |
| dspark 层号列表与 config 不一致 | **排除** | `tmp/dspark-config.json:19-25` ↔ `config.h:113` |
| verify 位置表少一列 | 已由 E6 落地 | `dflash_impl.h:226-242`（当前树已含，md5 `02cb3bc9…`） |
| **草稿/target 的 rope 约定（YaRN factor 32）** | **存活（首选）** | §2.1 |
| **block 行→verify 列约定（rows 0..k-1 vs 1..k）** | **存活（次选）** | §2.2 |

---

## 5. 补丁原文（两个补丁都对当前编译树 `patch -p1 --dry-run` rc=0）

### 5.1 `_collab/A5_round_probe.diff` — 诊断探针（2 hunk / +74 / −0）

用途：`NINFER_DSPARK_DUMP_DIR=<dir>` 时把**第 0 个 DSpark decode 轮**的 block ids/positions、
proposals、anchors/frontiers/extents 与**未加 Markov 的 base logits** 落盘成 `dspark_round0.bin`
（读数器 `_collab/A5_parse_round0.py`）；env 未设时零开销、零行为变化。

```diff
--- a/src/targets/qwen3_6/impl/runtime/dflash_impl.h
+++ b/src/targets/qwen3_6/impl/runtime/dflash_impl.h
@@ -29,6 +29,76 @@
 #include <cstddef>
 #include <stdexcept>
 #include <utility>
+#include <atomic>
+#include <cstdio>
+#include <vector>
+
+// A5 diagnostic probe (NINFER_DSPARK_DUMP_DIR): dump the first DSpark draft
+// round's block, positions, proposed tokens and base logits so the draft can be
+// recomputed offline against the shipped checkpoint. Unset = no work at all.
+inline void dspark_probe_dump(const Tensor& ids, const Tensor& positions,
+                              const Tensor& drafts, const Tensor& anchors,
+                              const Tensor& frontiers, const Tensor& extents,
+                              const Tensor& logits, std::int32_t width, std::int32_t k,
+                              std::int32_t batch, cudaStream_t stream) {
+    static const char* dir = std::getenv("NINFER_DSPARK_DUMP_DIR");
+    static std::atomic<std::uint32_t> counter{0};
+    if (dir == nullptr || *dir == '\0' || width <= 0 || k <= 0 || batch <= 0) { return; }
+    if (counter.fetch_add(1) >= 1) { return; } // probe scope: the first round only
+    const std::int32_t vocab     = logits.ne[0];
+    const std::size_t block_cols = static_cast<std::size_t>(width) * batch;
+    const std::size_t draft_cols = static_cast<std::size_t>(k) * batch;
+    const std::size_t logit_bytes = static_cast<std::size_t>(vocab) * draft_cols * 2;
+    std::vector<std::int32_t> ids_host(block_cols);
+    std::vector<std::int32_t> positions_host(block_cols);
+    std::vector<std::int32_t> drafts_host(draft_cols);
+    std::vector<std::int32_t> anchors_host(static_cast<std::size_t>(batch));
+    std::vector<std::int32_t> frontiers_host(static_cast<std::size_t>(batch));
+    std::vector<std::int32_t> extents_host(static_cast<std::size_t>(batch));
+    std::vector<std::uint16_t> logits_host(logit_bytes / 2);
+    // The block tensors are views of wider round-state buffers, so copy them row
+    // by row with their own pitch instead of assuming contiguity.
+    const auto copy_rows = [&](const void* device, std::size_t row_bytes, std::size_t pitch,
+                               std::size_t rows, void* host) {
+        CUDA_CHECK(cudaMemcpy2DAsync(host, row_bytes, device, pitch, row_bytes, rows,
+                                     cudaMemcpyDeviceToHost, stream));
+    };
+    copy_rows(ids.data, static_cast<std::size_t>(batch) * sizeof(std::int32_t), ids.nb[1],
+              static_cast<std::size_t>(width), ids_host.data());
+    copy_rows(positions.data, static_cast<std::size_t>(batch) * sizeof(std::int32_t),
+              positions.nb[1], static_cast<std::size_t>(width), positions_host.data());
+    copy_rows(drafts.data, static_cast<std::size_t>(batch) * sizeof(std::int32_t),
+              drafts.nb[1], static_cast<std::size_t>(k), drafts_host.data());
+    copy_rows(logits.data, static_cast<std::size_t>(k) * batch * 2, logits.nb[1],
+              static_cast<std::size_t>(vocab), logits_host.data());
+    CUDA_CHECK(cudaMemcpyAsync(anchors_host.data(), anchors.data,
+                               static_cast<std::size_t>(batch) * sizeof(std::int32_t),
+                               cudaMemcpyDeviceToHost, stream));
+    CUDA_CHECK(cudaMemcpyAsync(frontiers_host.data(), frontiers.data,
+                               static_cast<std::size_t>(batch) * sizeof(std::int32_t),
+                               cudaMemcpyDeviceToHost, stream));
+    CUDA_CHECK(cudaMemcpyAsync(extents_host.data(), extents.data,
+                               static_cast<std::size_t>(batch) * sizeof(std::int32_t),
+                               cudaMemcpyDeviceToHost, stream));
+    CUDA_CHECK(cudaStreamSynchronize(stream));
+    const std::string path = std::string(dir) + "/dspark_round0.bin";
+    std::FILE* file        = std::fopen(path.c_str(), "wb");
+    if (file == nullptr) { throw std::runtime_error("DSpark probe cannot open " + path); }
+    const std::uint32_t magic = 0x314B5344U; // "DSK1"
+    std::fwrite(&magic, sizeof(magic), 1, file);
+    std::fwrite(&width, sizeof(width), 1, file);
+    std::fwrite(&k, sizeof(k), 1, file);
+    std::fwrite(&batch, sizeof(batch), 1, file);
+    std::fwrite(&vocab, sizeof(vocab), 1, file);
+    std::fwrite(ids_host.data(), sizeof(std::int32_t), ids_host.size(), file);
+    std::fwrite(positions_host.data(), sizeof(std::int32_t), positions_host.size(), file);
+    std::fwrite(drafts_host.data(), sizeof(std::int32_t), drafts_host.size(), file);
+    std::fwrite(anchors_host.data(), sizeof(std::int32_t), anchors_host.size(), file);
+    std::fwrite(frontiers_host.data(), sizeof(std::int32_t), frontiers_host.size(), file);
+    std::fwrite(extents_host.data(), sizeof(std::int32_t), extents_host.size(), file);
+    std::fwrite(logits_host.data(), sizeof(std::uint16_t), logits_host.size(), file);
+    std::fclose(file);
+}
 
 namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
 namespace {
@@ -463,6 +533,10 @@
                 DType::BF16, {TextConfig::output_rows, static_cast<std::int32_t>(k) * batch_size});
             ops::linear(proposal_hidden, state.execution.model.output_head, logits,
                         state.execution.device.stream);
+            dspark_probe_dump(ids, positions, flat_drafts, anchors, frontiers,
+                              frame.proposal_extents.slice(0, 0, batch_size), logits, width,
+                              static_cast<std::int32_t>(k), batch_size,
+                              state.execution.device.stream);
             if (state.execution.model.dflash->markov_w1.has_value() &&
                 state.execution.model.dflash->markov_w2.has_value()) {
                 Tensor best_value = state.execution.work.alloc(DType::I32, {batch_size});
```

### 5.2 `_collab/A5_block_rows.diff` — 候选 B 修法草案（2 hunk / +5 / −12）

用途：把 DSpark(bf16) 的 block 行→verify 列约定恢复成"rows 1..k"（= HF 参考与引擎自有 W8 分支的
约定）。**先做 §2.3 的判别实验再决定是否落地**，它与已落地的 E6 补丁不同方向，不可同时生效。

```diff
--- a/src/targets/qwen3_6/impl/runtime/dflash_impl.h
+++ b/src/targets/qwen3_6/impl/runtime/dflash_impl.h
@@ -233,13 +233,6 @@
 
         ops::prepare_masked_block(anchors, frontiers, attention_valid, Config::mask_token, ids,
                                   positions, state.execution.device.stream);
-        if constexpr (Config::bf16_weights) {
-            // Restore the draft block's own k live columns for the attention below;
-            // columns 0..k-1 of the table already hold frontier+0..frontier+k-1 and are
-            // not affected by the width above.
-            ops::set_i32_scalar(attention_valid, static_cast<std::int32_t>(k),
-                                state.execution.device.stream);
-        }
         Tensor residual = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
         ops::embedding(ids.view({columns}), state.execution.model.token_embedding, residual,
                        state.execution.device.stream);
@@ -444,11 +437,11 @@
         const std::size_t source_pitch =
             static_cast<std::size_t>(Config::hidden) * width * element_bytes;
         const auto* source = static_cast<const std::byte*>(residual.data);
-        std::size_t source_column_offset = 0;
-        if constexpr (!Config::bf16_weights) {
-            // Legacy DFlash keeps the anchor column out of the proposal rows.
-            source_column_offset = 1;
-        }
+        // The checkpoint's block is [anchor, mask...] and the proposal rows must be
+        // each row's prediction for its own position, so the anchor column never
+        // enters them -- the same contract as the legacy W8 DFlash path and as the
+        // HF reference (`head(last_hidden_state[:, 1:])`).
+        const std::size_t source_column_offset = 1;
         source += source_column_offset * static_cast<std::size_t>(Config::hidden) * element_bytes;
         CUDA_CHECK(cudaMemcpy2DAsync(packed.data, row_bytes, source, source_pitch, row_bytes,
                                      static_cast<std::size_t>(batch_size), cudaMemcpyDeviceToDevice,
```

---

### 5.3 复现 dry-run

```bash
wsl.exe -e bash -c "cd /home/user/ninfer-fusion && patch -p1 --dry-run < \
  /mnt/c/Users/User/Documents/ziqinzhang/_collab/A5_round_probe.diff; echo rc=$?"
# 期望：checking file src/targets/qwen3_6/impl/runtime/dflash_impl.h / rc=0
wsl.exe -e bash -c "cd /home/user/ninfer-fusion && patch -p1 --dry-run < \
  /mnt/c/Users/User/Documents/ziqinzhang/_collab/A5_block_rows.diff; echo rc=$?"
# 期望：同上 rc=0
# 本回合两次执行后 `md5sum src/targets/qwen3_6/impl/runtime/dflash_impl.h`
#   = 02cb3bc9b511a5c2341d59a47f614931（与执行前一致，未落盘）
```


---

## 4. 诚实标注 / 未做项

- **没有任何 GPU 结果**：本文件所有"实测"仅指**离线 CPU/磁盘取证**（§3.1 的字节审计）与读码推导；
  涉及引擎行为的一律标为推断或"待证实"。
- 未做：编译/语法检查（硬约束）、`nvcc -fsyntax-only`、YaRN 频表离线复算、探针其实跑一次。
  两个 diff 的 C++ 合法性只有 `patch -p1 --dry-run rc=0` + 锚点唯一性 + 括号人工核对 ⇒ **待编译确认**。
- `A5_parse_round0.py` 只用**合成记录**做过冒烟（`width=8,k=7,batch=1,vocab=5` 的假 dump，
  字段解码与分段正确）；真 dump 未跑。
- HF 的 block/position 记账（`noise_position_ids`、`[:, 1:]` 的取舍）我**没有**跑过 HF，
  结论来自读 `transformers 5.8.1` 源码；若上游训练侧与 HF 推理侧约定不同，候选 B 的"参考侧"
  论证需要重估（这也正是 §2.3 用"复现引擎自己输出"来裁决、而不依赖 HF 的原因）。
- 与 E6 的关系：E6 的补丁**已经在树里**，本文件 §2.2 的候选 B 补丁是**在它之上的反向改动**，
  两者不可同时落地（先跑 §2.3 再决定保留哪一边）。

---

## 附注（2026-09-12）：§1.3 那套 `feat[5L*5120:(5L+1)*5120]` 切法的适用边界

**该切法只在 prefill dump 上成立，在 decode dump 上会给出错误结论。**

原因：`features` 是 `[D=5*hidden=25600, width, batch]`，而引擎 `Tensor` 的 **`nb[0]` 才是连续轴**
（`src/core/tensor.cpp:51-58 set_contiguous_strides`），即 **D 是最快轴**。因此内存里的排布是
"按列（位置）连续"，一列 = 一个位置的 25600 个值（5 个 tap 各 5120）。而 decode 轮次里
`prepare_ragged_prefix`（`src/ops/kernel/prepare_ragged_prefix.cuh:17-26`，`live = column < count`，
`count = ends - starts`）**只填 live 列、其余列整列零填充**，且 decode 时
`count = 1 + accepted = 1`（接受率≈0 时）⇒ 非零内容只有第 0 列。

若按 row-major `(25600, W)` 读并沿行切 5 段，就会看到"第 0 段有值、后 4 段为零"，
从而误判为"只有 tap0 被写入"。**判别式**：正确几何下非零元素应为 `live*25600` 个连续值
（decode 时 live=1 ⇒ 恰好 25600 个、offset [0,25599]）；错误读法下"每列非零数 = D/W"（如 3200），
而若真为 tap0 独有则应为 5120。

**正确读法**：`v = np.fromfile(f, dtype='<f2').reshape(W*batch, 25600)`，
tap i 取 `v[row, i*5120:(i+1)*5120]`；decode 时 tap 范数示例（call5）
`[102.36, 118.58, 119.42, 127.74, 149.55]`（单调递增 = 层深递增 ⇒ 5 层都在）。
