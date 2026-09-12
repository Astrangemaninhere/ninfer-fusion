# S47 · DFlash(dspark) 为什么慢/为什么接受率只有 10.3%（与 MTP 健康对照逐条对比）

本回合：CPU only（读码 + 读 artifact 头 + 读日志 + 只读小脚本），未写 `src/**`、未开 nvcc/ptxas、未用 GPU。
姊妹产物：`_collab/E5_s47_dspark_ingress.md`（ingress/verify 审计 + 三问答复，本文件不重复其细节）。
探针：`_collab/E5_s47_dump.py`、`E5_s47_cmp.py`（只读 artifact 头部：`magic[0:8]`、`u64 json_len@8`、JSON@16）。

今天四臂的同口径数字（serve 的 request-done 行，**接受率字段一直都在**，只是没标签、且汇总脚本把它截断了）：

| 臂 | 宽度 k | decode | 接受长度 | 接受率 | 日志 |
|---|---|---|---|---|---|
| plain（无 spec） | – | 54.8 / 54.3 | – | `off` | `dl/s4v2.log` |
| plain + MTP | 3 | 94.0 / 119.4 | 2.28 / 3.11 tok/round | **42.6% / 70.4%** | `/home/user/s4w_plain_mtp.log`、`s4v_plain_mtp.log` |
| dspark（DFlash v1） | 7 | 44.7 / 46.1 (70.1) | 1.38 / 1.42 (2.33) | **10.3% / 11.5% (19.0%)** | `/home/user/s4w_dspark.log`、`s4v_dspark.log`、`s4v2` |
| dflash2 | 7 | 141.6（计数 prompt） | 4.55 | **50.9%** | `dl/df2_final.log` |

⇒ 同样是宽度 7、**同一个 verify/accept 内核**，dspark 的接受率是 dflash2 的 1/5；而宽度扫描（`dl/sweep.log`：
dflash d1 65.4 / d3 80.5 / d7 70.0 tok/s）说明**宽度不是瓶颈**。所以问题在"喂进判定的草稿/位置/目标分布"。

---

## 1. 两条轮次机制逐项对比（每条带 file:line）

### 1.1 草稿宽度与 `--draft-tokens` 默认值
- CLI 校验：`src/product/speculative_options.h:35-69` —— `--spec mtp` 要求 `[1,5]`；`--spec dflash` `[1,15]`；
  `--spec dflash2` **只接受 0 或 7**（固定 7 草稿块）。
- `--spec auto` 的解析：`src/targets/qwen3_6_27b/impl/package.cpp:114-136` —— dflash2 产物 → `DFlash2` + `7`；
  dspark 产物 → `DFlash` + `7`；其余 → `Mtp` + `3`。**今天 dspark 臂就是 k=7（verify 宽 8），MTP 臂 k=3（宽 4）**。
- 每轮实现宽度：DFlash `extent = min({draft_window, max_by_budget, capacity - frontier - 1})`
  （`program_impl.h:12162-12163`）；MTP 多一道"上一轮留下的草稿数"上限
  `min({sequence.mtp_draft_count, draft_window, ...})`（`:11966-11968`）。

### 1.2 一轮的顺序：MTP 是流水线，DFlash 是同轮
- **MTP**（`mtp_impl.h:79-229` + `program_impl.h:11900-12070`）：先 verify（宽 k+1），**再**草稿：
  1 个 MTP 层（`TextConfig::mtp_layers = 1`，`qwen3_6_27b/impl/config.h:44`）跑对齐 hidden，
  然后 k-1 次 `mtp_forward_decode_batch` + `mtp_propose_batch`（`mtp_impl.h:140-176`）。
  草稿**跨轮携带**（`sequence.mtp_drafts` / `next_extents`，入口 `:11975-11978`；语义见
  `include/ninfer/ops/mtp_round.h`：`next_extents = min(K, budget-1, context)`）⇒ 第 r 轮验的是第 r-1 轮产出的草稿。
- **DFlash**（`dflash_impl.h:495-580`）：同轮内 先 append 草稿上下文（`534-538`）→ propose 5 层（`540`）→
  拼 verify ids（`541`）→ target verify（`553-572`）→ 拷 egress。**没有跨轮草稿**。

### 1.3 草稿每轮都跑吗？会不会被跳过？
- MTP：只要 `extent>0` 就跑（草稿阶段无条件）；`extent==0` 只在预算 ≤1 或上下文满时出现
  （`:11963-11968`），并计 `fallback_steps`（`:12042-12051`）。**没有接受率闸门。**
- DFlash：同样没有接受率闸门；唯一的数据相关收缩是 SVIP 熵截断（`dspark_markov_argmax` 写回
  `proposal_extents`，环境变量 `NINFER_DFLASH_SVIP_THRESHOLD`，`program_impl.h:607`；配置默认 2.5，
  `config.h:104`）。实测 drafted/round 均值 ≈ 名义值的 93%（d4 时 3.72/4），**几乎不截**。
- DFlash2 才有额外闸门：接受率地板 `dflash2_acceptance_too_low`（`spec_decision.h:98-109`）→ `extent=0`
  的 dense 轮，以及前沿降档 `kSpecDemoteTokens`（`spec_decision.h:29-33`）。⇒ MTP/DFlash 的 `fallback_steps`
  今天都是 0，三臂的接受率是"纯草稿质量"。

### 1.4 一次草稿步相对一次目标步的成本（用今天 artifact 的真实形状算）
- MTP：`mtp/*` = `input_projection [5120,10240]`（W8）、`layer/attention/query_key_gate_value [14336,5120]`、
  `attention/output [5120,6144]`、`mlp/gate_up [34816,5120]`、`mlp/down [5120,17408]`；单层 ≈ 0.34G 参数，
  但**一轮要串行跑 k=3 次**（每次都重读该层权重），外加每步一次 proposal 头。
- DFlash（dspark，全 BF16）：`feature_projection [5120,25600]`=131M（262 MB/轮）+ 5 层 ×
  (`query_key_value [7168,5120]` 36.7M + `context_key/value [1024,5120]`×2 + `attention/output [5120,5120]` 26.2M
  + `mlp/gate_up [20480,5120]` 105M + `mlp/down [5120,10240]` 52.4M ≈ 230M) ≈ **1.15G 参数 = 2.3 GB/轮**，
  再加 `dflash/markov_w1|w2`（各 127 MB）与**每轮读一遍完整输出头**（`dflash_impl.h:472-473`；头是
  `[248320,5120]`，BF16 ≈ 2.5 GB）⇒ 草稿侧 ≈ **5 GB/轮**，约等于目标一次 dense 步权重流量的 1/3。
  目标 verify 本身还要再读一遍模型+头（宽 8 列，权重只读一次）—— 这就是"每轮花 ≈1.3 个 dense 步，
  只换回 1+EAL 个 token"的经济学：EAL = 1.38 时净亏（实测 46.1 < 54.8）；EAL = 2.28（MTP）时净赚 1.7×。
- 另外 DSpark 的草稿注意力是**每层都看全上下文**（`full_only=true`，`27b/impl/config.h:84-86`；调用点
  `dflash_impl.h:313-320`），DFlash2 是 5 层全局部窗（2048，`config.h:115-128`，调用点
  `dflash2_impl.h:256-260`）⇒ 上下文越长，DSpark 的草稿越贵（与历史曲线 "2k 65.8 → 15k 35.3" 一致）。

### 1.5 verify 怎么决定接受
- 三后端**同一个实现**：`src/targets/qwen3_6/impl/runtime/speculative_target_impl.h:9-38`（ops 调用在 `:27`）
  → `ops::speculative_accept_greedy_drafts`（`include/ninfer/ops/speculative_round.h:53-99`）：greedy 下
  "接受与逐列 penalty-adjusted argmax 逐位相符的最长前缀，在首个不匹配（或 bonus）列提交该 argmax"；
  仅当 `draft_ids/draft_probs` 非空（DFlash2 的 selector）才升级为 `u < min(1, p_i/q_i)`。
  列约定三臂一致：列 0 = anchor，`drafts[j-1]` 落在列 j（`speculative_round.h:18-51`）；温度 0 ⇒ 纯 argmax 快速路径。
- 逐轮计数（归因用的口径）：MTP `program_impl.h:12043-12049`（`pcur` = ingress extent）、
  DFlash `:12232-12238`（extent = **egress**，即 SVIP 截断后的实现值）、DFlash2 `:12459-12465`。
- **差异只在"喂什么"，不在"怎么判"**：DFlash 把**草稿块自己的 position 表**同时当 verify 的
  `cache_positions` 与 `rope_positions`（`dflash_impl.h:522,556-557`）且不加 `rope_delta`；MTP 用
  `speculative_prepare_verify_inputs` 造表并在 host 侧加 `sequence.rope_delta`（`program_impl.h:11980-11982`）。
  详见姊妹文件的 §1.3（那里还有一条"最后一路被重复位置砍掉"的实证 off-by-one）。

---

## 2. 假设排序（针对"net slower than no speculation"）

| 排序 | 假设 | 现状与预测 |
|---|---|---|
| **H1** | **草稿输入链（目标特征/上下文 → 草稿 hidden）与 checkpoint 训练约定不一致** —— 上游 DSpark 按 HF `hidden_states` 训练，本引擎 tap 是"该层 post-MLP residual"（`text_context_impl.h:1221/1244`）；DFlash2 的草稿是在**本引擎自己的 tap**（`NINFER_HS_DUMP_DIR` 的 NHS1 dump）上训练的 ⇒ 对 tap 约定免疫 | 唯一被上游移植文档自己标为未解决的点（`DSPARK-ADAPTATION.md:151-158`）；预测：**位置 0 的接受率就已很低**，连可预测的计数 prompt 也只有 19.0% |
| **H2** | **草稿/verify 的位置与旋转**：verify 位置表比自身 `valid_columns` 少一位（`dflash_impl.h:223-231` vs `12170`）；DFlash 不带 `rope_delta`；checkpoint 声明 `rope_type=yarn, factor 32`（`tmp/dspark-config.json`）而草稿用纯 `rope_theta`（`dflash_impl.h:293-294`） | 预测：位置 0 尚可、**位置 ≥1 快速衰减**且**上下文越长越差**（与历史 2k→15k 退化曲线一致） |
| **H3** | **草稿成本 > 接受前缀收益**（§1.4：≈5 GB/轮 + 全上下文注意力 + 全词表头×2） | 解释"慢"，不解释"接受率低"；H1/H2 修好后它决定加速比上限 |
| **H4** | **markov 头**：静默退回 argmax（`dflash_impl.h:455-474`）或权重内容/角色错 | 静默退回**已排除**（今天服务的 artifact 带 `dflash/markov_w1|w2`，`E5_s47_cmp.py` 实证）；数学与上游一致（`/home/user/qwen3_dspark.py:43-109`）⇒ 只剩"权重内容"层面，需复算才能定 |
| 排除 | ingress 漏填 / accept 分叉 / artifact 选错 / config 不匹配 / 40Q-8KV 桥 / 物理尾列污染 | 逐条证据见 `E5_s47_dspark_ingress.md` §1、§2 与 §3.1（E-1..E-5） |

---

## 3. 最便宜的可判别实验（现成 serve/CLI，**不需要 E2/S44 的计数器补丁**）

**为什么不需要计数器**：接受率与逐位置直方图**早就在**——
serve：`src/serve/request_log.cpp:438-456`（`speculative=<backend> X.XXtok/round (NN.N%)`，行尾 `:591`）
与 JSON `:298-306`（含 `accepted_per_position`）；CLI：`apps/cli/main.cpp:219-245`
（`drafted / accepted / acceptance rate / acceptance length / accepted by pos`）。
今天"扫不到 acceptance"的真因是：字段没有标签 + `_spec_4way.sh:55` 把汇总行 `cut -c1-230` 截断
（完整行仍在 `/home/user/s4w_*.log`）。E2/S44 的补丁只是加标签/修 CLI 的 DFlash2 错标，不是新增计数。

**E-1（首选，3 条命令，~3 分钟 GPU）——先切"输入链错"还是"链断"**
```bash
BIN=/home/user/ninfer-fusion/build/apps/ninfer
P=$(head -c 1500 /home/user/spec_prompt_base.txt)
# ① dspark（服务的那份，含 markov）
$BIN /home/user/models/qwen3_8_27b_nvfp4_dspark.ninfer --prompt "$P" --max-new 96 --greedy \
     --no-thinking --no-cuda-graph --kv-dtype bf16 --kv-layer-storage all:bf16 \
     --max-context 4096 --kv-capacity 4096 --spec dflash --draft-tokens 7 \
     2>&1 | grep -E 'acceptance rate|acceptance length|accepted by pos'
# ② 健康对照 A：同 prompt、同 artifact 换 --spec mtp --draft-tokens 3
# ③ 健康对照 B：dflash2 artifact + --spec dflash2
```
判读：`accepted by pos` 是前缀直方图 `[n0,n1,...]`，条件接受率 `p_i = n_i/n_{i-1}`。
- dspark 的 `p_0` 明显低于对照（对照 `p_0` ≥0.5）⇒ **H1**（草稿首 token 就已经错）→ 走 E-2。
- dspark 的 `p_0` 与对照同量级而 `p_1` 起崩塌 ⇒ **H2/H4**（链断）→ 查 markov 链与位置表（§1.3 的少一位正好砍掉最后一路）。
- 顺带把 `--draft-tokens 1` 跑一遍：它就是"位置 0 的绝对接受率"（历史记录是 0%，`DSPARK-ADAPTATION.md:151-156`）。

**E-2（决定性，~10 s GPU + 纯 CPU 比对）——tap 语义：引擎特征 ↔ HF 参考**
```bash
NINFER_HS_DUMP_DIR=/tmp/hsd NINFER_HS_DUMP_TOPK=1 $BIN /home/user/models/qwen3_8_27b_nvfp4_dspark.ninfer \
   --prompt "$P" --max-new 1 --greedy --no-thinking --no-cuda-graph --spec dflash --draft-tokens 7
# → /tmp/hsd/chunk_000000.bin：NHS1 | tokens | ids[] | feat[25600×T]（5 层按捕获顺序）| last[5120×T] | argmax[]
# CPU：Windows 侧 python(已有 data/Qwen3.8-27B 权重) 取 hidden_states[L+1], L∈{4,16,28,40,52}
#      与 feat[5L·5120:(5L+1)·5120) 逐层 cosine/max|Δ|；last 的 argmax 对 ids[pos+1] 做对齐自检
```
判据：五层 cos>0.999 ⇒ F1 死（tap 对），转去打草稿执行链：dump `compact_features`（`dflash_impl.h:531-540`）
与 proposal logits（`:470-474`），用 `models/qwen3.8-27b-dspark-zh/model.safetensors`（62 张量）在 Python 里
逐层复算，**首个发散的 op 即元凶**（移植文档留的正是这一步，`DSPARK-ADAPTATION.md:157-158`）。
若 feat 与 `hidden_states[L]`（错一层）或 post-attention hidden 对上 ⇒ H1 成立，改
`qwen3_6_27b/impl/config.h:105` 的 `target_feature_layers` 或 tap 点，一位之改 + 同 dump 复验。

**E-3（CPU-only 旁证）**：E-2 产出的 `argmax[]` 与 `ids[pos+1]` 的比对（`text_prefill_impl.h:135-160`
的注释就是为此写的）能在没有草稿参与时确认"特征/位置 ↔ 下一 token"的对齐。

---

## 4. 缺失的那项测量 & 一条修法建议
- **不是缺计数**（§3 开头）：三后端的 `drafted/accepted/rounds/fallback` 都有，缺的是"带标签"（E2/S44 在做）
  和"别把行截断"（`_spec_4way.sh:55` 的 `cut -c1-230`）。
- 真正需要补的一条**不变量**（本回合新提）：DFlash 的 verify 位置表必须与 `target_valid_columns` 同宽
  （§1.3），否则"最后一路结构性不可接受"会被误读成草稿质量——建议修完后加一条断言/直方图检查：
  `accepted by pos` 的第 k-1 位恒为 0 ⇒ 位置表 bug 回归。
- 排序建议（给 M）：① 修 verify 位置表少一位（独立小补丁）；② 跑 E-1 拿 `accepted by pos` 剖面定 H1/H2；
  ③ 按结论走 E-2（tap 对齐）或草稿逐层复算；④ 在此之前不做任何 dspark 调参（宽度已证明不是瓶颈）。
