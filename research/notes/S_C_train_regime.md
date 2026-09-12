# S_C：训练口径上限（`train_dflash2.py`）——M1/M2 独立复核 + retrain 可证伪预期

> 硬约束遵守声明：只读分析；未改任何源码；未编译、未跑引擎/GPU；只写本文件（Scratch 探针见 §9）。
> 所有"实测"行都是我本人跑的只读 CPU numpy 检查；推断一律标注（推断）。

## 0. 结论速览

| 条目 | 判定 | 关键依据 |
|---|---|---|
| **M1 目标行 off-by-one** | **成立。shift 必须 = 0**；legacy 硬编码 +1 = "把列 i 训成预测 tok[a+2+i]" | 代码 `train_dflash2.py:227 / :454 / :457` + 参照 `speculator.py:1189-1190`、`qwen3_dflash2.py:228`；我把"ids16 是下一 token 行"钉在**训练用那份 cache** 上：0.2420 vs 0.0023（§2.2） |
| **M2 真 token vs mask** | **成立。legacy 就是"把答案当输入"（label leak）+ mask embedding 梯度恒 0** | 代码 `:309 / :451-452`；legacy 配方有源码快照直接证据（`data/df2pilot/train_dflash2.py:420,425`）；全量 cache 706,220 位置里 mask id **0** 次（§3.2） |
| A2 的 M1/M2 两条主张 | **前提我都独立复核通过**；但 **M1 从未被实验测过**（A/B 空跑） | §2.4、§6 |
| "引擎已把该 ckpt 榨干"（R12 E1） | **该论据不成立**（比了两个不同草稿）；E2/E3/E4/E5 不受影响 | §7 |
| 新增阻碍项 | **live artifact 是 08-26 的旧草稿，任何 Sept ckpt 从未进过引擎** | §7（mtime/size/无 `*_tuned`/导出报 `ModuleNotFoundError: tools`） |
| retrain 方向 | `_train_df2_shift0.bat` 的口径（mask + shift 0）**方向正确**，但需先补 `--out-dir`、先修导出链、先做 §6 的便宜判定 | §4/§6/§8 |

## 1. 复核范围与方法（可复现）

- 读：`train_dflash2.py`（根，491 行，mtime 2026-09-10 19:34）、`data/df2pilot/train_dflash2.py`（09-03 快照）、
  参照 `1Cat-vLLM/.../qwen3_dflash2.py`、`1Cat-vLLM/.../dflash2/speculator.py`、fork 的
  `src/targets/qwen3_6_27b/impl/config.h`、`src/targets/qwen3_6/impl/runtime/dflash2_impl.h`。
- 实测（只读 npz，CPU，无 GPU）：`data/hs_cache_topk2`（480 个 `seq_*.npz`）。
  脚本：`_collab/_tmp/sc_fullchk.py`、`_collab/_tmp/sc_ids16chk.py`、`_collab/_tmp/sc_maskchk.py`；
  均已跑通，命令：`"C:\Program Files\Python312\python.exe" <脚本>`。
- 不做：不重跑 A2 的两口径离线评测（需 GPU/torch），因此 §5 的阈值是**判定线**而非预测量。

## 2. M1（teacher 目标行错一位）——复核成立

### 2.1 三条代码依据（列语义 → 目标行）
1. `train_dflash2.py:227` `return lm_head(out[:, 1:]), out[:, 1:]`；loss 只算 `B-1=7` 行、等权
   `:454 w = torch.ones(1, B - 1)`。⇒ 返回行 `i` ↔ block slot `1+i` ↔ 绝对位置 `a+1+i`。
2. `train_dflash2.py:457-458` `t16_ids = ids16[a + target_shift : a + B - 1 + target_shift]`
   ⇒ 行 `i` 的教师 = `ids16[a + target_shift + i]`。
3. 引擎/参照的列语义（独立于训练脚本）：`speculator.py:1189-1190`
   `num_sample = num_reqs * self.draft_block` + `hidden_states = last_hidden_states[sample_indices[:num_sample]].view(num_reqs, draft_block, -1)`
   —— 采样的是 **anchor 之外的 K 列**（`draft_tokens[:num_reqs, :self.draft_block]`，`:1222`）；模型侧 conv 的
   `block_size = 1 + get_dflash_model_draft_tokens(...)`（`qwen3_dflash2.py:228`）
   ⇒ **第 i 个 draft token 就是位置 a+1+i 的 token**（列 i 的答案 = `tok[a+1+i]`）。
⇒ 列 `i` 的答案 token 是 `tok[a+1+i]`，其教师行是 `ids16[a+i]` ⇒ **shift = 0 才对齐**；`shift=1`（legacy 默认）
   把列 i 训成预测 `tok[a+2+i]`，即整块"目标晚一行"。

### 2.2 我自己的独立实测（钉在**训练用** cache 上）
`data/hs_cache_topk2`，40 文件 / 62,033 位置（`files[::12]`）：

| 量 | 我实测 | A2 旧值（不同样本） |
|---|---|---|
| `P(ids16[t,0] == tok[t+1])` | **0.2420** (15013/62033) | 0.2904 |
| `P(ids16[t,0] == tok[t])` | **0.0023** (140/62033) | 0.0008 |
| `top1` 字段 | 与 `ids16[:,0]` 逐位相同（冗余字段） | — |
| npz keys | `tokens, feat, last, prompt_len, top1, ids16, vals16` | — |

⇒ `ids16[t]` 是"位置 t 的**下一** token 分布"= 实测结论（非 docstring 推断）。A2 的 0.2904/0.2594 与我的
0.2420 同量级（样本/规模不同），**M1 的前提成立**。

### 2.3 A2 的指纹（我的读法，标注为推断）
A2 §Q1：同一 ckpt、**同一（训练器）输入口径**下，草稿对"错位目标 `ids16[a+1,0]`"命中 **0.2504**，
对"正确答案 `ids16[a,0]`"只有 **0.1708** ⇒ 模型更贴自己的**错位**目标，且 0.25 已贴到教师天花板
（0.242~0.29）附近。**这是 shift=1 的直接指纹**。→ 由此得到 §5 的 P2 判定线。

### 2.4 现状：这条主张**从未被实验验证**
`data/_ab_shift0` 与 `data/_ab_shift1` 都是**空目录**；`dl/shift_ab.log`（09-10 14:15）只有 4 行启动输出
（停在 `lm_head ... embed ...`），**没有 step 行、没有 `SHIFT_AB_DONE`** ⇒ `_df2_shift_ab.bat` 空跑。
⇒ "shift 是主因"目前**只有代码语义 + 指纹**支撑，没有 A/B 数据。

## 3. M2（真 token vs mask）——复核成立

### 3.1 代码 + 历史源码快照（双重证据）
- 现状（09-10 19:34 之后）：`:309 --mask-block` 默认 **True**，`:451-452 block[0, 1:] = MASK_ID`（=248077，`:42`）。
- legacy 路径：`:310 --no-mask-block` 保留 `block = tok[a:a+B]` ⇒ **列 1..7 的输入就是列 1..7 的答案**（label leak），
  且 mask embedding 行**永远拿不到梯度**。
- **历史配方有源码直接证据**：`data/df2pilot/train_dflash2.py`（09-03 22:17 快照）
  - `:420 block = tok[a:a + B]`（真 token，无 mask）
  - `:425 t16_ids = ids16[a + 1:a + B]`（**硬编码 +1**，当时还没有 `--target-shift` 参数）
  ⇒ "legacy = 真 token + shift 1"是从源码读出来的，不是从注释推断的。

### 3.2 我自己的独立实测（全量 cache）
`data/hs_cache_topk2` **全部 480 文件 / 706,220 位置**：

| 量 | 实测 |
|---|---|
| `248077`（MASK_ID，dflash2 path）出现次数 | **0** |
| `>= 248047` 的 id 出现次数 | **0** |
| token id 域 | **[0, 248046]** |
| `prompt_len` | **480/480 都是 0** |

⇒ 真 token 配方下 MASK_ID **一次都没被喂过**（仍是初始化值），而推理时**每轮 7/8 列**都喂它。
另：`prompt_len` 全 0 ⇒ `a_lo = max(BLOCK_SIZE, pl-1) = 8`（`:398`），锚点从位置 8 起在**整段正文内部**任意采样，
不是"解码前沿"分布（轻微 train/infer 错配 + 数据卫生项，见 §4）。

### 3.3 引擎侧 mask id 核对（**这是重点，别改错**）
fork `/home/user/ninfer-fusion/src/targets/qwen3_6_27b/impl/config.h`：
`:106 mask_token = 248077`（DFlash2 路径，调用点 `dflash2_impl.h:192`
`ops::prepare_masked_block(anchors, frontiers, attention_valid, Config::mask_token, ids, ...)`）；
`:137 = 248070`（另一条 = dspark 路径，`dflash_impl.h:234`）。`ninfer-upstream` 镜像 27b 那份是 248070（≠ live）。
⇒ trainer 的 `MASK_ID = 248077` **与 live 引擎一致**。
⇒ 09-11 的 mask-id 实验（改 190221）**已回退**：`dl/revert_maskid.log`（106 已回 248077，build rc=0）、
`dl/maskid_exp.log`（dspark K=7：248077→7.38% vs 190221→6.83%，"无改善"）⇒ **mask id 不要动**，trainer 那边也不要动。

### 3.4 A2 的量化（引用）
同 ckpt、679 anchors：engine 口径（mask 块）pos0 = **0.0309 (21/679)** @1200、**0.0427** @1800；
训练器口径（真 token 块）= **0.1708 / 0.1222** ⇒ **5.5× 塌陷**，方向与"mask 列 OOD"一致。

## 4. 引擎要匹配该 ckpt：需要的输入口径（checklist）

| 项 | 引擎现状 | trainer 需要 | 判定 |
|---|---|---|---|
| 列 0 | 真 anchor | 真 anchor（`:450`） | 已一致 |
| 列 1..7 | `Config::mask_token`=248077 | `MASK_ID`=248077（`:452`） | 已一致（`--mask-block` 默认 True） |
| 位置 | `F+min(i,V-1)`（R12 E5） | `arange(a, a+B)`（`:453`） | 一致（W=B=8 时） |
| 目标行 | —— | `--target-shift` **必须 0** | **必须传**（legacy 1 = 错位） |
| 目标分布 | selector top-16 候选 | 教师 top-16（`:457-467`，截断重归一） | 一致 |
| 损失形状 | —— | 7 列**等权**（`:454`）、**逐列独立** softmax（无链式/联合项） | 结构性事实：只优化"按列边际命中"，不优化链 |
| 块内 conv | 参照按 `position % block_size` 复位（`qwen3_dflash2.py:112-118`） | `torch.cat(zeros)` 复位（`:105-110`） | 一致 |
| 上下文宽度 | 引擎最多 2048 | `--max-ctx 128` | **错配**（A2 §8b）⇒ retrain 顺手 A/B 128 vs 512/2048 |
| anchor 分布 | 解码前沿（有真实前缀） | `prompt_len` 全 0 ⇒ 位置 8 起任意采样 | 轻微错配（记录即可） |

**结论句**：引擎侧**没有**"改成喂真 token"的选项（未来 token 未知）⇒ 唯一出路是 retrain。
现有 `_train_df2_shift0.bat` 的口径（mask + shift 0）**方向正确**；需要补的只有：`--out-dir`（§8）、
ctx 的 A/B、以及**先修导出链**（§7）——否则新 ckpt 根本到不了引擎。

## 5. retrain 后**可证伪**的预期（每条给阈值 + 判定仪器）

基准（现 ckpt / Sept 配方）：离线 mask 口径 pos0 = 0.0309@1200、0.0427@1800；真 token 口径 0.1708/0.1222；
教师天花板 = **0.2420**（我）/ 0.2594（A2）；引擎 CLI 剖面 `[8,0,0,0,0,0,0]`（现 ≈1.03 tok/round、接受率 4.5%）。
（注：用 `--target-shift` 逐列对齐要看 `_stepwise_audit2.py` / `_df2_blame2.py` 的逐列 blame 输出。）

- **P1（M2 是否主因）**：同一 A2 评测器（`A2_draft_eval.py`，679 anchors）用 **mask 块**输入，新 ckpt 的 pos0 hit
  必须 **≥ 0.15**（≥3.5× 当前 0.0427），且 ≥ 旧 ckpt 真 token 口径值 0.1708 的一半以上。
  **证伪线**：仍 ≤ 0.06 ⇒ M2 不是限制器，转向容量/数据量/loss/优化。
- **P2（M1 是否主因）**：固定 `--mask-block`，只差 `--target-shift {0,1}` 两臂（300–600 步即可看方向）。
  预测 arm0 的 pos0 hit − arm1 **≥ +0.03 绝对**，且 arm0 ≥ arm1 在 7 列中 **≥5 列**（仪器：逐列 blame）。
  **证伪线**：arm1 ≥ arm0 ⇒ shift 不是主因，§2.3 的 0.2504>0.1708 指纹要另找解释。
- **P3（剖面形状，最终判据）**：新 ckpt 出 artifact 后，`ninfer --spec dflash2 --greedy` 的 `accepted by pos`
  **不得再是** `[x,0,0,0,0,0,0]`；预测 **p1/p0 ≥ 0.5**、**p2/p1 ≥ 0.4**、**AL ≥ 1.8 tok/round**（现 ~1.03–1.35），
  剖面按 ~0.5–0.7 几何比衰减（因为 loss 等权 7 列、且只有边际目标，链式衰减是必然）。
  **证伪线**：仍 `[x,0,…]` 或 AL ≤1.2 ⇒ 口径不是主因，才回头查 capacity/数据/链式目标。
- **P4（口径哨兵，防"偷偷又用真 token"）**：任何"修复后"的 pos0 **不得**超过教师自身天花板 0.242；
  若 > 0.35 ⇒ 先查评测器是不是又喂了真 token（`mask_block` 是否真开）。
- **P5（别把 6000 步当必需）**：A2 §8 已量化现配方 6000 步 ≈ 一个 epoch 的 13%；
  预测收益主要落在前 ~1000 步，step6000 − step2000 的 pos0 差 **< 0.03** ⇒ 先跑 600–1000 步看 P1 方向。

## 6. 今天就能做的"便宜判定"（不需要重训 6000 步）

**(A) 用已存在的"新配方"样本做 R 比（0 GPU，只需 CPU 评测器）**
`data/dflash2_ckpts/step_000100.pt` / `step_000200.pt` 的 mtime 是 **09-10 23:32 / 23:36**，
**晚于** `train_dflash2.py` 的 19:34 改动 ⇒ 它们是**唯一存在的 mask+shift0 配方 ckpt**（日志 `dl/train-dflash2.log`
23:26:50 起从 step 1 重跑、23:36:55 存 step_200 后终止）。
把评测器分别指向 `step_000200`（mask+shift0, 200 步）与 `step_001200`（legacy），**各测两种输入口径**，
比较"塌陷比" `R = hit(mask 口径) / hit(真 token 口径)`：
- 旧 ckpt：R ≈ 0.0309/0.1708 = **0.18**；
- **预测新配方 ckpt 的 R ≥ 0.6**（输入不再是 OOD）。R 是相对量，**抗步数不足**，200 步就能看方向。
- 若 step_000200 的 R 仍 ~0.2 ⇒ M2 的"输入口径"解释被证伪（或 200 步不足以学到 mask）——先解释再花 6000 步。

**(B) 直接测"live artifact 里那份草稿"的 R（最便宜、最能定案）**
`data/draft_model/model.safetensors`（7.6 GB，**08-17**）与 `data/draft_checkpoints/ckpt_*.pt`（08-16/17，3.8 GB）
是 08 期的草稿导出，**08-26 artifact 的草稿极可能来自这一系**（时间线唯一吻合；`data/df2pilot` 是 09-03，太晚）。
把评测器指向它、同样跑两种口径求 R：
- 若 R ≈ 0.18（也 mask-naive）⇒ 说明 live 与 Sept ckpt 同病，那"live p0 26–35%"与"Sept ckpt 3%"的落差就是
  **评测口径/harness 差异**，M1/M2 无法解释 live 的数字 ⇒ 整条"ckpt 口径"论证要重估；
- 若 R ≈ 1（mask in-distribution）⇒ live 草稿是**另一种配方/来源**，Sept 的 5.5× 塌陷才是异常 ⇒ M2 就是修法。
无论哪个结果，这一步都比 retrain 便宜且能定案。

## 7. 阻碍性发现：live artifact ≠ 任何 Sept ckpt（先修导出链，否则 retrain 无意义）

全部为文件级事实，无推测：
1. `/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer`，25,446,373,888 B，mtime **2026-08-26 12:24:08**；
   `models/Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2/qwen3_8_27b_nvfp4_dflash2.ninfer`（08-26 11:42，同 size）是源。
2. 全树只有这一个 dflash2 artifact（`find /home/user -maxdepth 2`、`data/`、`models/` 均**无**
   `*_tuned.ninfer` / `*w9s1200*`）；根目录 `.sh` 里该路径出现 **29 处**，包括验收脚本
   `_verify_df2head.sh:16`、`_ga_check.sh:14`、`_ga_ksweep.sh:14`。
3. 唯一一次导出尝试 `dl/df2_w9_export.log`（09-10 15:45）**失败**：`patch_dflash2.py` 与 `verify_patch.py` 都是
   `ModuleNotFoundError: No module named 'tools'`（必须在 repo 根跑/设 `PYTHONPATH`）；随后 serve 报
   `open /home/user/models/qwen3_8_27b_nvfp4_dflash2_w9s1200.ninfer: No such file or directory`；
   `dl/` 全部日志里**没有** `mapping ok` / `written:` 字符串 ⇒ **从未 patch 成功**。

三个后果：
- **(a)** 今天引擎上的接受率数字（`[8,0,0,0,0,0,0]`、4.5%、`accepted by pos`）说的是 **08-26 那份草稿**，
  其训练配方**不在本树里**；§6(B) 是唯一不动 GPU 就能判它的路径。
- **(b)** R12 的 E1（"引擎 p0 26–35% ≥ 该 ckpt 离线上限"）把**两个不同草稿**放在一起比：
  A2 离线评的是 `step_001200/1800`（Sept 配方），引擎跑的是 08-26 草稿 ⇒ **该论据不足以支撑"引擎没丢好草稿"**
  （E2/E3/E4/E5 的引擎保真结论不受影响，它们靠的是引擎自洽性）。
- **(c)** `_collab/M_df2_serve_measure.md`（09-10 15:25，用的是 `C_artifact_manifest.md` 明确标注"旧路径，勿用"的
  `~/ninfer/build`）在同一 artifact 上记到 `dflash2 4.55tok/round (50.9%)`、decode 141.6 vs baseline 54.8 tok/s；
  而今天 `~/ninfer-fusion/build` 的 CLI 记到 `[8,0,0,0,0,0,0]`/≈1.03 tok/round。**两者不可能同时成立**
  ⇒ retrain 之前先把"旧 binary vs 新 binary 在同 artifact 同 prompt 上的差异"钉死，否则新 ckpt 的收益会被这个混淆项吞掉。

## 8. ckpt 目录的**配方污染**（retrain 前必须处理）

- `data/dflash2_ckpts/` 混了**两套配方**的 ckpt：
  - `step_000100.pt` / `step_000200.pt`（09-10 **23:32/23:36**，晚于 19:34 改动）⇒ mask+shift0；
  - `step_000300..001900`（09-10 **12:56–18:56**，早于改动）⇒ legacy 真 token + shift1（= A2 评的那批）。
- `--resume` 用 `sorted(glob('step_*.pt'))[-1]`（`train_dflash2.py:360-364`）⇒ **现在 resume 会挑到 `step_001900`
  = legacy 错位权重**，新旧配方混训（且注释里只有 `_orig_mod.` 前缀处理，没有配方校验）。
- 而 `_train_df2_shift0.bat` **没写 `--out-dir`**（`:9`）⇒ 会写回同一目录、覆盖 `step_000100.pt`，步号再与 legacy 错位。
- 建议（不改源码）：retrain 显式 `--out-dir data\dflash2_ckpts_mask_s0`（或先把 legacy 那 17 个文件移走），
  并在 bat 注释里写明"本目录含两套配方"。

## 9. 边界声明 + Scratch

- 未验证 08-26 artifact 内草稿的**实际**训练配方（需要 Aug 版 `train_dflash2.py`，树里没有；`data/df2pilot/train_dflash2.py` 是 09-03 且**没有** mask/shift 开关）。
- 未重跑 A2 的两口径离线评测（需 GPU），§5 的数值是**判定线**；§6 的两个 R 比是推荐的低成本判定。
- Scratch（可复现，均在 `_collab/_tmp/`）：`sc_fullchk.py`（全量 cache：mask id / id 域 / prompt_len）、
  `sc_ids16chk.py`（ids16 行语义 0.2420/0.0023）、`sc_maskchk.py`（40 文件抽样）。
