# S47 · dspark(DFlash v1) ingress/verify 审计 + 10.3% 接受率归因

回答 M 的三问。全部 CPU 侧（读码 + 读 artifact 头 + 读日志），未写 `src/**`、未开 nvcc、未用 GPU。
本回合新增的两个只读探针：`_collab/E5_s47_dump.py`（artifact 头部对象表）、`E5_s47_cmp.py`（v1/v2/已服务文件的对象集对比）。

---

## 0. 一句话结论（先说结论，证据在下）

1. **ingress 没有 DFlash2 那种漏填**：DFlash v1 的 decode 轮把 `DFlashDecodeIngress` 的 11 个字段**逐个填满**
   （`program_impl.h:12164-12179`），`state_source_slots/state_destination_slots` 都在（12177/12178，走的是与
   MTP `:11989` 同一个 `state_selectors()`，`program_impl.h:10028-10034`）。`:11414` 那条是**另一条路径**
   （`enqueue_dflash_context_append`，`11378-11449`）——它只服务 context append，不进 decode 轮。
2. **verify/accept 与 MTP/DFlash2 完全同一条**：三个后端都调 `speculative_target_impl.h:9` 的
   `target_verify_accept()`（DFlash 在 `dflash_impl.h:553`），state slot 的读（`:18/:23`）与写（`:36`）语义一致；
   接受判定是同一个 op（`:27` → `ops::speculative_accept_greedy_drafts`）。**没有第二条判定路径。**
3. **10.3% 不是"缺 markov 头"**（这点先排除掉，省得走弯路）：今天服务的那份 artifact **带 markov**
   （`/home/user/models/qwen3_8_27b_nvfp4_dspark.ninfer` = 24,316,119,552 B，含 `dflash/markov_w1|w2`；
   与 Windows 侧 `*_dspark_v2.ninfer` 对象集逐字段相同）。引擎里"没有 markov 就静默退回 `ops::argmax`"
   的岔路（`dflash_impl.h:455-474`）**今天没有被走到**。
   真正可疑的是**草稿输入链（目标特征/上下文 → 草稿 hidden）**：DSpark 的草稿是在上游按 HF/ModelOpt 的
   `hidden_states` 约定训练的，而 DFlash2 的草稿是**在本引擎自己的 tap 上**训练的（`NINFER_HS_DUMP_DIR` 的
   NHS1 dump → `hs_cache` → `train_dflash2.py`）⇒ **tap 约定即使错，也伤不到 DFlash2，只伤 DSpark**。
   这正是 08-26 移植文档里**唯一未打勾的那条**（`DSPARK-ADAPTATION.md:151-158`：块语义修好后 k=1 仍 0%，
   结论"剩余错误在特征/上下文→草稿 hidden 的执行链"，下一步=按 safetensors 逐层复算）。
4. 另外找到一条**真 off-by-one**（任务书里预言的那类）：DFlash 的 verify **复用了草稿块的 position 表**，
   而该表是按 `V=k` 造的，比 verify 自己的 `valid_columns=extent+1=k+1` **少一位** ⇒ 最后一路 verify
   列落在重复位置上。行号与影响面见 §1.3。

---

## 1. 问 1：dspark 的 ingress 每个字段都填了吗？

### 1.1 结构体与填充点
`DFlashDecodeIngress` 共 11 个字段（`src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h:74-86`）：
anchors / execution_frontiers / context_frontiers / proposal_extents / target_valid_columns /
text_kv_table_rows / dflash_kv_table_rows / active_lanes / state_source_slots / state_destination_slots / sampling。

decode 轮的实际填充（`ProgramImplCore::decode_dflash_batch`，`program_impl.h:12155-12181`）：

| 字段 | 行 |
|---|---|
| anchors | `12164` |
| execution_frontiers | `12165` |
| context_frontiers | `12167` |
| proposal_extents | `12169` |
| target_valid_columns | `12170` |
| text_kv_table_rows | `12171` |
| dflash_kv_table_rows | `12173` |
| active_lanes | `12175` |
| state_source_slots | `12177`（`state_selectors(sequence).source`） |
| state_destination_slots | `12178`（`.destination`） |
| sampling | `12179` |

**11/11 全填**，且 `state_selectors()` 与 MTP（`11988-11990`）是同一个函数（`10028-10034`，
source=`sequence.state.read`，destination=`sequence.state.write`）。DFlash2 的那类漏填（M 线索里
`12408-12411` 的修复注释）**在 DFlash v1 上不存在**。

注意：decode 轮**没有**先 `*dflash_host_ingress = {}`（对比 `11389`、`8723`、`9719` 都先清零）。
未写到的尾行 `[lanes.size(), kMaximumConcurrency)` 会保留上一轮的值；设备侧按"exact-B 前缀"消费
（`round_state.h:29-30` 明写），尾行不进数学 ⇒ 无害（但若将来有人改成读整批，这里是个雷）。

### 1.2 `:11414` 是另一条路径（不是第二个 decode 填充点）
`ProgramImplCore::enqueue_dflash_context_append`（`program_impl.h:11378-11449`）：
- `11389` 先整struct清零；`11406-11409` 填 context/execution frontier，`11410` 填 dflash kv 行，
  `11412` 填 active_lanes，`11414-11415` 填 state slots（**就是 M 看到的那两行**）；
- 它**不跑 decode 轮**：`11446` 调的是 `schedule::dflash_append_context(...)`（append 只吃
  features/positions/counts/lanes/table_rows 这些实参，根本不再读 ingress）；
- 调用者：forced-token 路径（`8700`）与 prefill 收尾——两处都是"把草稿上下文补齐"，不是验证。

另有两条写 `DFlashDecodeIngress` 的地方，同样是**零初始化 + 只填子集**，且都只被 prefill 侧消费：
`append_forced_tokens`（`8723-8732`）与 `start_sequence`（`9719-9728`）。
⇒ 结论：**dspark 的 decode 轮不存在第二处漏填**；`state slots` 全程有效。

### 1.3 但有一处**真 off-by-one（位置表）**——与 DFlash2/MTP 结构性不同
- `proposal_positions` 由 `prepare_masked_block(anchors, frontiers, attention_valid, mask_token, ids, positions)`
  生成（`dflash_impl.h:230-231`），而 bf16 分支把 `attention_valid` 置为 **k**（`223-227`）。
  该 op 的定义（`include/ninfer/ops/prepare_masked_block.h`）：`positions[i] = frontier + min(i, V-1)`，
  V=`valid_columns`=k ⇒ 列 `i=0..k-1` 在 `frontier..frontier+k-1`，**第 k 列（物理尾列）重复 `frontier+k-1`**。
- 但 **verify 用的是同一张表**（`dflash_impl.h:522` + `556-557`：`.cache_positions = target_positions,
  .rope_positions = target_positions`），而 verify 的 `valid_columns = extent+1 = k+1 = 8`（`12170`）。
  对比 MTP：verify 的位置来自 `speculative_prepare_verify_inputs`（`positions[j] = frontier + min(j, Pcur)`，
  Pcur=k）⇒ 第 k 列 = `frontier+k`（`include/ninfer/ops/speculative_round.h:18-42`）。DFlash2 则把
  `attention_valid` 置为 **width**（`dflash2_impl.h:187-189`）⇒ 也不缺位。
- 后果（可证伪、有界）：① 第 `k` 路草稿（`drafts[k-1]`）是在"和上一列同一位置"上被验证的 —— 该位置的
  target logits 与上一列相同 ⇒ **这一路不可能被接受**（acceptance 上界 k-1）；② 该列的 KV 写入落在
  `frontier+k-1`，与上一列**同址重复写** ⇒ 只有当本轮接受 ≥ k-1 路时才会污染已提交位置的 KV。
  以今天 10.3%、EAL 1.38 的水平，这条**不足以单独解释 5× 缺口**，但它就是任务书预言的"engine-side
  off-by-one"，且在任何接受率数字可信之前必须修（建议：verify 侧另用一张 V=extent+1 的表，或把
  `attention_valid` 传成 `extent+1`，让两块语义各自对齐）。
- 顺带记录：`DFlashDecodeIngress` 里**没有** `target_rope_positions` / `rope_deltas`（MTP 有，
  `round_state.h:54,59`）。DFlash 的 verify 因此**不加** `sequence.rope_delta`（MTP 在 `11980-11982` 加）。
  今天两次运行都没开 `--yarn`，`rope_delta` 预期为 0（**待证实**：`rope_delta` 来自
  `staged.prompt.rope_delta`，`program_impl.h:9701`）；一旦走 YaRN/长上下文，这条与"草稿 rope 用的是
  纯 `rope_theta`"（`dflash_impl.h:293-294`，而 checkpoint 的 `rope_type=yarn, factor=32`，见
  `tmp/dspark-config.json`）合起来会让草稿和 verify 同时跑在**错的旋转角度**上。

---

## 2. 问 2：dspark 的 verify 是否也消费 state slots / 是否同一套 accept？

**是，完全同一套。**
- 调用链：`dflash_impl.h:553` `target_verify_accept(...)` → `speculative_target_impl.h:9-38`：
  - state 读：`card.target_verify_batch(..., frame.state_source_slots, ...)`（`:17-20` / `:22-25`）；
  - 接受：`ops::speculative_accept_greedy_drafts(...)`（`:27`，df 与 mtp/df2 逐字同一调用）；
  - 续写：`ops::scatter(selected_hidden, frame.state_destination_slots, continuation_hidden_store)`（`:36`）。
- DFlash 传给该 frame 的 `state_sources/state_destinations` 直接来自 ingress（`dflash_impl.h:516-517`，
  即 §1.1 的 12177/12178）。
- 三个后端**唯一的差别是喂进去的输入**，而不是判定规则。与 dspark 有关的两个"口径差"（不是 bug，但比较时必须知道）：
  1. `current_extents`：DFlash 用**egress** 的 `proposal_extents`（= 被 SVIP 截断后的实现值，
     `program_impl.h:12215-12218`），MTP 用 **ingress** 的 `current_extents`（`:12040-12041`）；
  2. DFlash 的 `drafts` 来自 `speculative_prepare_verify_ids`（`dflash_impl.h:541`，anchor 占列 0），
     DFlash2 额外带 selector 的 `draft_candidate_ids/probs`（走 `u < min(1,p/q)` 分支）；greedy 下 DSpark
     与 MTP 都走"与目标 argmax 逐位相符"的同一分支 ⇒ **今天三臂的接受率可直接比**。
- 因此 10.3% vs 50.9% vs 42.6% **不可能**是判定规则差异：只能是**喂给判定的草稿 token/位置/目标 logits** 不同。

---

## 3. 问 3：10.3% 的可证伪最小实验 + 目前已有证据

### 3.1 已有证据（都在本回合 CPU 内取证）

| # | 事实 | 证据 |
|---|---|---|
| E-1 | 今天服务的是**带 markov** 的 artifact；引擎"无 markov 静默 argmax"岔路未被走到 | `E5_s47_cmp.py`：`/home/user/models/qwen3_8_27b_nvfp4_dspark.ninfer` = 24,316,119,552 B，对象集与 win `*_dspark_v2.ninfer` **完全一致**（含 `dflash/markov_w1|w2 [248320,256] BF16`）；win `*_dspark.ninfer` = 24,061,839,872 B **缺**这两项（v1=v2 去掉 markov，其余对象形状/格式逐个相同）。`_spec_4way_v2.sh:73` 服务的就是 `$M/qwen3_8_27b_nvfp4_dspark.ninfer`。引擎岔路在 `dflash_impl.h:455-474`。 |
| E-2 | 引擎 DSpark 配置与 checkpoint config 逐项一致（层数/GQA/intermediate/mask/特征层/markov rank） | `tmp/dspark-config.json`（`target_layer_ids [4,16,28,40,52]`, `block_size 7`, `mask_token_id 248077`, `40/8`, `10240`）↔ `qwen3_6_27b/impl/config.h:74-106`；binder 期望的 markov 形状 `[248320,256]` 见 `load/bindings.cpp:523-526,727-730` |
| E-3 | markov 的数学与上游参考实现一致（`bias(v)=Σ_r w1[p,r]·w2[v,r]`，p=anchor/前一草稿） | `/home/user/qwen3_dspark.py:43-109`（`markov_w1`=Embedding(target vocab)，`markov_w2`=LMHead(draft vocab)）↔ `include/ninfer/ops/dspark_markov_argmax.h` 的公式 |
| E-4 | 40Q/8KV 桥接的 head→KV 组映射**逐项正确**（不是元凶） | `dflash_impl.h:339-381`：真实头 `5g+r`(r≤3) → padded 槽 `4g+r` → KV 头 `(4g+r)/4=g` ✓；真实头 `5g+4` → padded 槽 `4g` → KV 头 `g` ✓；`copy_heads` 的 pitch=`head_dim·ne[1]·2`、行长=`head_bytes·count`、token_rows=`width·batch` ✓（padded_query 为 `[128,32,W,B]`，dst.ne[1]=32 ⇒ 列间 stride 正确） |
| E-5 | 草稿块的第 k 物理列**不参与**键集合（不会污染草稿注意力） | `include/ninfer/ops/bidirectional_gqa_attention.h`：`keys = context[0,L) 后接 live query rows [0,V)`，`i>=V` 为 inert tail 且输出置零 ⇒ 该列只浪费算力 |
| E-6 | tap 语义 = **该层 post-MLP 的 residual**（即 HF `hidden_states[layer+1]`），与移植文档"offset=1"的说法一致，但**从未数值验证** | `text_context_impl.h:1221`（full 层）与 `:1244`（GDN 层）：`mlp_tail(...)` 之后立即 `tap.capture_layer(layer, x, ...)`；移植文档自述 `DSPARK-ADAPTATION.md:57-59` 与未打勾的 P1 `:151-158` |
| E-7 | **DFlash2 的草稿是在引擎自己的 tap 上训练的** ⇒ 上述 tap 隐患对它是免疫的 | `text_prefill_impl.h:85-160`（`NINFER_HS_DUMP_DIR` → NHS1：`feat[feature_rows×T]` = 5 个目标层**按捕获顺序**拼接 + `last[hidden×T]`）→ `hs_cache` → `train_dflash2.py`；`_feat_probe.sh` 也是围绕这条链做的探针 |
| E-8 | 今天四臂的**同口径**数字（同一 verify/accept 核）：dspark 10.3%(zh)/19.0%(num) vs MTP 42.6%/70.4% vs dflash2 50.9% | `/home/user/s4w_dspark.log`（zh：`dflash 1.38tok/round (10.3%)`，E2/S44 文档独立复核过同一行；计数 prompt 同臂为 `2.33/19.0%`）、`s4v_dspark.log`（`1.42/11.5%`）、`/home/user/s4w_plain_mtp.log` 与 `s4v_plain_mtp.log`（`mtp 2.28/42.6%`、`3.11/70.4%`）、`dl/df2_final.log`（`dflash2 4.55/50.9%`） |
| E-9 | "计数缺失"是误判：**计数早在行里**（无标签），且 CLI 还打印**逐位置**接受直方图 | `src/serve/request_log.cpp:438-456`（`speculative=<backend> X.XXtok/round (NN.N%)`，行尾 `:591`）与 `:298-306`（JSON 含 `accepted_per_position`）；`apps/cli/main.cpp:219-245`（`drafted/accepted/acceptance rate/acceptance length/accepted by pos`）。M 的扫描落空是因为 `_spec_4way.sh:55` 把汇总行 `cut -c1-230` 截断了，且字段里没有 "acceptance" 这个词（E2/S44 补丁就是给它加标签） |

### 3.2 假设排序（带"若成立的预测"）

| 排序 | 假设 | 支持 | 证伪/预测 |
|---|---|---|---|
| **H1（首选）** | **草稿输入链错位**：目标特征/上下文 → 草稿 hidden 这一段的语义与 checkpoint 训练时不一致（层号偏移、tap 点（post-MLP vs 层输入/post-attention）、或拼接顺序），DSpark 被暴露、DFlash2 免疫 | E-6/E-7；上游移植文档自己把结论停在这里（"剩余错误在特征/上下文→草稿 hidden 的执行链"，`DSPARK-ADAPTATION.md:156-158`）；且**连最可预测的计数 prompt 也只有 19.0%**（健康草稿在那种 prompt 上应接近全接受） | 若成立：位置 0 的接受率就低（≈10-20%），且 HS dump 与 HF 参考对不上（cos 明显 <0.999 或整齐地错一层） |
| **H2** | **草稿/verify 的位置与旋转**（§1.3 的少一位 + DFlash 无 `rope_delta` + checkpoint 声明 YaRN 而引擎用纯 `rope_theta`） | 代码实证；与旧曲线"DSpark 2k 65.8 → 15k 35.3"的随上下文退化一致（`spec_decision.h:1-24` 注释里的历史表） | 若成立：位置 0 尚可、**位置 ≥1 快速衰减**，且**上下文越长越差**；打开 `--yarn` 会同时改变 target/draft 两侧（可做差） |
| **H3** | 草稿执行成本（5 层全上下文双向注意力 + 全输出头 248320×5120 每轮各读一遍）把"每轮只多 0.38 token"的收益吃光 ⇒ 净速度低于 baseline | `dflash_impl.h:200-430`（每轮 5 层 + `linear(proposal_hidden, output_head)` `:472-473` 或 `:485`）；dspark `full_only=true` ⇒ 每层都看全上下文（`config.h:84-86`） | 这条解释"慢"，**不解释"接受率低"**；修好 H1/H2 后它只影响加速比上限 |
| **H4** | markov 头绑定/调用错（静默 argmax、形状对但权重错位、w1/w2 角色互换） | 形状与数学都对（E-3）⇒ 只剩"权重内容"层面 | 若成立：位置 0（用 anchor 作 p）与位置 ≥1 的差异不明显，且复算 Python 时 `logits+bias` 与引擎不一致 |
| **已排除** | ingress 漏填（§1.1）、accept 判定分叉（§2）、缺 markov 的 artifact（E-1）、config 不匹配（E-2）、40Q 桥（E-4）、物理尾列污染（E-5） | — | — |

### 3.3 最便宜的可判别实验（按代价排序；E-1 不需要 GPU，E-2/E-3 各 1 次短 GPU 运行）

**E-1（首选，零改动、零 GPU）——"逐位置接受直方图"先切一刀**
用 CLI（不是 serve）跑两臂，读 `accepted by pos`：
```bash
# dspark（服务的那份，带 markov）
/home/user/ninfer-fusion/build/apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dspark.ninfer \
  --prompt "$(cat /home/user/spec_prompt_base.txt | head -c 1500)" --max-new 96 --greedy --no-thinking \
  --no-cuda-graph --kv-dtype bf16 --kv-layer-storage all:bf16 --max-context 4096 --kv-capacity 4096 \
  --spec dflash --draft-tokens 7 2>&1 | grep -E 'acceptance rate|acceptance length|accepted by pos'
# 对照 1：同一 prompt 换 --spec mtp --draft-tokens 3（健康对照）
# 对照 2：同 prompt 换 dflash2 artifact + --spec dflash2（健康对照，50.9%）
```
判据（每臂一行数字即可定案）：
- `accepted by pos` 形状是 `[n0,n1,...]`（n_i = 至少接受 i+1 路的轮数）⇒ 条件接受率 `p_i = n_i / n_{i-1}`。
  - `p_0` ≈ 0.15-0.2（dspark）而 MTP/dflash2 的 `p_0` ≥ 0.5 ⇒ **H1 成立**（草稿的输入/首 token 就已经错）。
  - `p_0` 与对照同量级、但 `p_1/p_0 ≤ 0.2` 且之后迅速归零 ⇒ **H2/H4**（首 token 对、链断）⇒ 下一步查
    markov 链与位置表（§1.3 的少一位正好砍掉最后一路）。
- 与 `--draft-tokens 1` 的那一千次（历史上 k=1 给 0%，`DSPARK-ADAPTATION.md:151-156`）对比，可以直接把
  "位置 0 的绝对水平"钉死。

**E-2（决定性、~10 s GPU + CPU，验证 H1 的 tap 语义）——引擎 tap ↔ HF 参考逐层对齐**
```bash
# ① GPU（短）：让引擎把"喂给草稿的那份特征"原样吐出来（NHS1 格式，现成机制）
NINFER_HS_DUMP_DIR=/tmp/hsd NINFER_HS_DUMP_TOPK=1 \
  /home/user/ninfer-fusion/build/apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dspark.ninfer \
  --prompt "<与 E-1 完全相同的 prompt>" --max-new 1 --greedy --no-thinking --no-cuda-graph \
  --spec dflash --draft-tokens 7
# 产物 /tmp/hsd/chunk_000000.bin：magic NHS1, i32 tokens, i32 ids[tokens],
#   u16 feat[25600 × tokens]（5 层按捕获顺序拼接）, u16 last[5120 × tokens], i32 argmax[tokens]
# ② CPU（Windows 侧 python + 已有的 data/Qwen3.8-27B 权重；eval_ddtree.py 已在用同一套权重）：
#   对同一 prompt 取 HF hidden_states[L+1]，L∈{4,16,28,40,52}，与 feat[5L·5120:(5L+1)·5120) 逐层比
#   （cos / max|Δ|），并把 last+output_head 的 argmax 与 ids[pos+1] 对齐（NHS1 自带这条自检的用途）。
```
判据：五层 cos 全 >0.999 ⇒ **H1 死**（tap 语义对），转去打草稿执行链（H2/H4：把
`dflash_impl.h:531-540` 的 `compact_features` 与 `:470-474` 的 proposal logits 各 dump 一份，
用 `models/qwen3.8-27b-dspark-zh/model.safetensors`（62 张量）在 Python 里逐层复算 —— 首个发散的 op 就是元凶，
这正是移植文档留的下一步）；若 feat 与 `hidden_states[L]`（错一层）或与 post-attention hidden 对上 ⇒
**H1 成立**，修 `config.h:105` 的 `target_feature_layers` 或 tap 点，然后用同一条 dump 复验 + 重测接受率。

**E-3（CPU-only 的旁证）**：`NINFER_HS_DUMP_TOPK=1` 写出的 `argmax[tokens]` 与 `ids[pos+1]` 的比对
（`text_prefill_impl.h:135-160` 的注释就是为这类对齐问题准备的）能在**没有 draft** 的情况下确认
"特征/位置是否与下一个 token 对齐"。

### 3.4 我建议的下一步（按序）
1. 修 §1.3 的 verify 位置表少一位（独立小补丁：verify 用 V=`extent+1` 的表；MTP/DFlash2 语义各自对齐即可），
   并在验收里加一条"`drafted` 那一路不被结构性砍掉"的不变量。
2. 跑 E-1（三条命令，约 3 分钟 GPU）拿 `accepted by pos` 剖面 —— 这是"归因到 H1 还是 H2"的分水岭。
3. 按 E-1 的结论走 E-2（tap 对齐）或草稿逐层复算（H2/H4）。
4. 在此之前，任何"dspark 慢"的调参（draft-tokens、SVIP 阈值）都不必做：今天的扫描（d1 65.4 / d3 80.5 /
   d7 70.0 tok/s）已经说明宽度不是瓶颈（`dl/sweep.log`）。

### 3.5 诚实标注
- §1.3 的"最后一路不可能被接受"是**从代码语义推出的**（positions 重复 + accept 核逐位比 argmax），
  未在 GPU 上实测；但 `accepted by pos` 的最后一位若恒为 0 就是它的直接指纹。
- "DFlash 不加 `rope_delta`"是否真的与今天无关，取决于 `sequence.rope_delta` 在未开 `--yarn` 时为 0（待证实，
  一行即可确认：`program_impl.h:9701` 的取值来源）。
- H1 的"DFlash2 免疫"论证依赖"dflash2 草稿训练数据确实取自引擎 NHS1 tap"，证据是
  `text_prefill_impl.h:85-89` 与 `_feat_probe.sh` 的注释链；若训练侧另有 tap（例如取自 torch 侧），
  这条对称性论证需要重估。
