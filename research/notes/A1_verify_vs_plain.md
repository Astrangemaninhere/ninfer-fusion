# A1 · verify 与 plain 为什么不等价：把候选差异穷举成可实验的表

**范围**：只读代码分析。未改 `src/**`、未编译、未跑 GPU、未碰 `tools/**`。
**被测目标**：`/home/user/models/qwen3_8_27b_nvfp4.ninfer` = `qwen3_6_27b` 家族
（64 层 = 16 GQA + 48 GDN，hidden 5120，`output_rows=248320`，`token_domain=248077`，
`rotary_dim=64`，`rope_theta=1e7`，`sliding_window=0`）—
证据 `src/targets/qwen3_6_27b/impl/config.h:18-19,27-28,40,44` + `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/frontend.h:16`。

**读法**：§1 = 已有证据、可以直接排除的怀疑（别在这里花时间）；§2 = 候选差异主表（任务书要的那张表）；
§3 = 最省事的判别序列；§4 = 与"接受率"的关系与边界。

---

## 0. 先把账本对齐（后面每一行判断都靠它）

两条路径在**同一个锚点**上的语义是同一个（这就是 `frontier` 账本不变式）：

| 量 | plain 普通解码 | verify 第 0 列 |
|---|---|---|
| 输入 token | `sequence.ledger.back()`（`program_impl.h:11823`） | `verify_ids[0]=anchors[0]=ledger.back()`（`program_impl.h:11969` + `speculative_round.cuh:31-32`） |
| cache/因果位置 | `frontier`（`program_impl.h:11824`） | `positions[0]=base_frontiers[0]=frontier`（`speculative_round.cuh:33` + `program_impl.h:11970`） |
| RoPE 位置 | `frontier + rope_delta`（`program_impl.h:11826-11827`） | `frontier + min(0,extent) + rope_delta`（`program_impl.h:11979-11983`） |
| 可见 KV 集合 | `[0, frontier]`（A1 契约 `include/ninfer/ops/gqa_attention.h`：`0<=x<=positions[j,b]`，含本步新写 key） | 同 |
| 线性注意力输入状态 | `state_selectors(sequence).source`（`program_impl.h:11831`） | 同一个 `selectors.source`（`program_impl.h:11989`） |

**⇒ 任务书 #5（anchor 重喂 vs prefill 最后位置 hidden）与 #2（位置语义）在语义层已经被排除**：
第 0 列的 (token, 位置, RoPE, 可见 KV, 状态槽) 五件套两条路径逐一相同。
剩下所有可能只能落在**数值层**（同一个数学用不同 kernel/切分）或**观测层**（看不到中间量）。

账本细节（用于自查其它轮次）：anchor 是"已发布但 KV 还没写"的待定 token，其 K/V 由**消费它的那一轮**在列 0
写入（`gqa_attention` 边写边算），`frontier` 即 anchor 的位置；每轮写 KV 的列数 = `valid=extent+1`
（`program_impl.h:11974`），发布 token 数 = `accepted+1`，`lengths += produced`
（`speculative_round.cuh:141-146`）⇒ KV 列数与 frontier 增量自洽。
GDN 侧同账：`commit_columns=committed=accepted+1`（`program_impl.h:9847,9857` + `include/ninfer/ops/gdn_replay.h`），
fold 的 source/destination 与 plain 同源（`program_impl.h:9853`）。

---

## 1. 已证实"相同/无关"（有证据，直接排除）

| # | 结论 | 证据（file:line） | 为什么可以排除 |
|---|---|---|---|
| E1 | **argmax 的 tie-break 与 reduce 顺序不可能造成差异** | `src/ops/kernel/argmax.cuh:22-25`（`value>best \|\| (value==best && index<best)`）；`src/ops/kernel/sampling_device.cuh:76-78`（`sampling_better` 同序）；`sampling_device.cuh:85-97`（key=(有序浮点<<32)\|(0xffffffff-idx)，双射无精度损失）；plain 的 greedy 路由 `src/ops/kernel/sampling.cuh:27-64` / 多块路由 `sampling.cuh:187-216`；verify 的 argmax 两条路由 `src/ops/launcher/argmax.cu:48-57`（直接归约）与 `:62-79`（memset 0 + atomicCAS 收敛） | 两个实现都是**同一个全序**（值降序、并列取小索引）上的最大值；全序的最大值唯一 ⇒ **与归约顺序、线程映射、并列处理都无关**。只有在 logits 本身不同的前提下才可能给出不同 token。任务书 #4 到此为止。 |
| E2 | `token_domain` 掩码一致 | plain `decode_impl.h:47`（`TextConfig::token_domain`）；verify `text_context_impl.h:888`（`kCfg.token_domain`=同值）；`speculative_round.cpp:127-131` 只要求 `token_domain<=logits.ne[0]` | 两条路径都在 `[0,248077)` 上取 argmax，`output_rows=248320` 只当物理行数 |
| E3 | `apply_final_logit_policy` 对 qwen 是编译期 no-op | `text_context.h:93-99`（未声明 `output_multiplier()`→1.0）、`:100-107`（未声明软上限→0.0）、`:125-129`（`if constexpr` 才发射）；qwen3_6_27b 未声明这两个字段（`config.h` 全文） | verify 补不补它都不改变数值；plain 侧根本没调（`text_context_impl.h:820-821`）也不产生差异 |
| E4 | **无效尾列不会污染有效列**（"clamp 位置覆写"嫌疑排除） | 写回循环用 `valid_tokens` 限界：`src/ops/kernel/gqa_attention_decode_bf16.cuh:59-63`（`valid_columns[b]-column_begin`）+ `:148-166`（`chunk < valid_tokens*(D/8)`）；读路径同样 `:230-233` | `valid=extent+1` 之外的列既不写 KV 也不参与计算 |
| E5 | KV 读集合第 0 列一致 | A1 契约 `include/ninfer/ops/gqa_attention.h`（masked 形式：前缀 `[0,valid_columns[b])` 有效、位置在该前缀内递进；`score` 只对 `0<=x<=p` 求和）；`gqa_attention_decode_bf16.cuh:117-135`（`window=last_pos+1` 决定键区间） | 第 0 列的 `p=frontier` 两条路径相同 ⇒ 可见键数相同；`valid_columns` 只决定"哪些列被算/被写" |
| E6 | 自己的 key 都从**未量化**的 input 读，历史 key 都从**量化** cache 读（规则相同） | `gqa_attention_decode_bf16.cuh:229-243`：`from_new = new_token in [0,valid_tokens)` → 读 `input.k/v`，否则读 `cache_k/v` | 第 0 列自己那格在两条路径都命中 `from_new` ⇒ 处理一致（"verify 用未量化 local 键而 plain 用量化键"这种错位**不存在**） |
| E7 | RoPE 同参同位置 | plain/verify 都走 `text_context_impl.h:999-1004`（`ops::rope`/`rope_yarn4(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn)`），只用位置与静态常量 | `rotary_dim/theta` 无分支差异；第 0 列位置相同（§0） |
| E8 | penalty 分支不产生不一致 | `speculative_round.cuh:187-216`（无惩罚 fast path 用裸 argmax；有惩罚时在 accept 内核内用 `sampling_adjusted_logit(+overlay)` 重算 argmax）；plain 侧 `sampling.cuh:32-48` 同样在惩罚下 adjusted | `temperature=0` 且无惩罚时两条路径都是"裸 logits 的精确 argmax"；有惩罚时 accept 内核自己重算的也是 adjusted argmax，与 plain 一致 |
| E9 | 折叠列数与提交 KV 列数一致（状态账本不是嫌疑） | `program_impl.h:9847-9861`（`commit_columns=accepted_tokens`）、`:9868`（fold）、`include/ninfer/ops/gdn_replay.h`（"consume records `[0,commit_columns)`"）；`recurrent.cuh:686-697`（fold 内核 `valid=commit` 从 0 推进） | 折叠的列数 = KV 已提交列数 ⇒ 状态与 frontier 同步；若这里错会**立刻**每轮都错（与 `mtp3 num` 全程逐字相同矛盾） |

---

## 2. 候选差异主表

> 「判别实验」一列注明是否只需 CLI（纯命令行、不需改码）／CPU（纯文本/grep）／需窗口（要打补丁+批编译）。
> 「先验」= 我的排序（高→低），不代表已验证。

| # | 候选差异 | file:line 证据 | 为什么会导致 token 不同 | 判别实验 | 先验 |
|---|---|---|---|---|---|
| D1 | **T 维变化 ⇒ 同一数学换 kernel 实例化（注意力）**：plain 每步 T=1，verify 一步 T=`width`；attention 的 `TokenTile`/split 数都由 T 与 envelope 窗口决定 | `text_context_impl.h:810`（plain `width=1`）vs `:827-873`（verify `width*batch`）；`gqa_attention_decode_impl.cuh:68-102`（`gqa_small_t_split_count(window,tokens,dtype)` **显式按 tokens 分档**）、`:104-120`（容量取 envelope 区间最大值）、`:123-190`/`:220-271`（Kernel 模板按 `TokenTile/WarpsPerCta/Masked` 实例化）；envelope：plain `program_impl.h:11800,11816`（点区间）vs verify `program_impl.h:594`（`{1, max_frontier+k+1}`，由 `mtp_impl.h:150` 传入）；KV 分块参数由 `gqa_attention_decode_bf16.cuh:124-135` 按 window 重算 | 同一个 query 对同一段 KV，**softmax 的 (m,l,acc) 分块归约树与 MMA 的 K 方向累加顺序都不同** ⇒ 输出不保证逐位相同。16 个 GQA 层各引入一次 | **纯 CLI**：同一 prompt、同一产物、`--no-spec`，比较 `--prefill-chunk 128` 与 `--prefill-chunk 2048`（`apps/cli/options.cpp:138`，须 128 的倍数，`:231`）的首 ~16 个 `--print-token-ids`。不同 ⇒ 引擎贪心输出不随 T 不变，D1/D2 成立 | **最高** |
| D2 | **所有 linear/GEMM 的 N 维变化**（含 lm_head）：同一权重、同一列向量，verify 是 `N=width` 的 GEMM，plain 是 `N=1` | `text_context_impl.h:820`（plain `ops::linear(hidden,lm_head,logits)`，`logits` 为 `[vocab,batch]`）vs `:881`（verify `flat_logits [vocab,width*batch]`）；层内投影同理（`text_context_impl.h:980-990,1074,1183`） | GEMM 的 tiling 由 N/M 决定 ⇒ 同一输出元素的 K 归约可能分批不同 ⇒ 次 ULP 差异（NVFP4 权重还要过 dequant 路径）。**待证实**：我没读 `src/ops/linear/nvfp4/*` 的路由表，不能断言 N=1 与 N=width 一定选不同 kernel | 与 D1 同一实验（prefill-chunk 扫描）即可同时覆盖；要定位则**CPU** 读 `src/ops/linear/nvfp4/` 的路由条件 | 高 |
| D3 | **KV 量化的"放大+永久化"效应**：K/V 用 BF16 算完后按行标定量化（scale 由 `max|x|` 决定）写进 cache，后续所有轮从量化 cache 读历史键 | 写：`gqa_attention_decode_bf16.cuh:148-166`；读历史：`:229-243` 的 else 分支；量化契约见 `include/ninfer/ops/gqa_attention.h` 的 INT8-G64 段（`a=max|x| → scale=RNE(a/127) → code=round(x*inv)`） | 隐藏态差 1 ULP ⇒ 可能跨过量化边界 ⇒ 该位置的 cache 码与 plain 不同 ⇒ **此后每一轮读到的历史 KV 都不同**，两条轨迹不可逆分叉。这正好解释"前 31 个字符全同、第 32 个才分叉"的形态 | **需窗口**：在 verify 与 plain 各 dump 同一位置的 logits + 该位置 cache 码（plain 已有 `debug_head_probe` 机制，见 D10） | 高 |
| D4 | **GDN（48 层）recurrent 走 record 内核而非 batch_update 内核** | `recurrent.cuh:665-673`（`recurrent_batch_update_kernel` **硬编码 `valid=1`**）vs `:675-684`（`recurrent_record_kernel` 用 `active_columns`）；调用侧 `text_context_impl.h:1113-1127`（record）/:1163-1168（batch_update）；`speculative_target_impl.h:15`（verify 强制 `RecordForReplay`） | 逐列数学**同源**（`RecordEffects::publish_output` 直接调用 `OutputEffects::publish_output`，`recurrent.cuh:207`）⇒ 第 0 列在输入相同时应逐位相同；真正风险是"输入来自 D2 的 T 相关 GEMM" ⇒ 本质退回 D2 | 与 D1 同一 CLI 实验；要独立隔离则**需窗口**：把 plain 也切成 `RecordForReplay` 对照（或 verify 改 `UpdateInPlace`+width=1 逐列） | 中低 |
| D5 | **GDN 的 conv 分支 masked/dense 差异** | `text_context_impl.h:1118-1127`（verify 传 `valid`）vs plain 传空 Tensor；`src/ops/wrapper/causal_conv1d_silu.cpp:342-377`（`masked = valid_columns.data != nullptr`） | 同源代码、第 0 列应一致；只有 verify 在 width>1 时传 dense 才会把 clamp 尾列当有效列（当前不会）。**待证实**：masked/dense 两实例化的列 0 是否真的逐位一致 | **CPU**：读 wrapper + 内核确认；端到端靠 D1 实验覆盖 | 低 |
| D6 | **dflash2 的目标 verify 少了 `rope_delta`** | `dflash2_impl.h:410-411`：`.cache_positions = target_positions, .rope_positions = target_positions`，而 `target_positions=frame.proposal_positions`（`:377`）由 `ops::prepare_masked_block` 生成 = `lengths[b]+min(i,valid-1)`（`src/ops/kernel/prepare_masked_block.cuh:18`，`valid=width`，`dflash2_impl.h:187-189`）；对照：plain 加 delta（`program_impl.h:11826-11827`）、MTP 加 delta（`program_impl.h:11979-11983`） | 若 `sequence.rope_delta != 0`，dflash2 路径**整列** RoPE 角度都偏移一个常量 ⇒ 系统性（不是舍入）错误，且会随上下文增长放大 | **CPU**：追 `sequence.rope_delta` 的赋值链，确认是否恒 0（若恒 0 则本行作废）；**CLI** 侧可看 `target_rope` 与位置表是否同值 | **中高**（dflash2 是被点名的失败后端，且成本极低，应最先证实） |
| D7 | **MTP 的 AR 草稿步有效位**：`valid=0` 的列输出被置零，而 `ar_hidden` 可能取自这些列 | `src/ops/kernel/mtp_round.cuh:47`（`ar_valid_columns[s]= s+1<next ? 1:0`）；消费点 `mtp_impl.h:196-202`（`mtp_forward_decode_batch(..., valid, ...)`）；置零行为 `gqa_attention_decode_bf16.cuh:84-115`（`write_neutral`：`partial_m=-inf, acc=0`）；随后 `mtp_impl.h:181-186`（`speculative_select_accepted_hidden` → `mtp_propose_batch` → draft0） | 若某轮 `accepted` 落在 `valid=0` 的列上，下一轮的 draft0 会是垃圾 token ⇒ **直接压低 pos0 接受率**（与"接受率只有 ~50%"的形态吻合），但不必然造成 verify≠plain | **需窗口**（dump `accepted` 与 `ar_valid_columns` 的关系）或**CLI** 扫 `--draft-tokens 1..5` 看接受率剖面是否随 k 断崖 | 中（这条是"接受率"侧，不是"等价性"侧） |
| D8 | **历史 KV 的冷页/低精度档只在部分实例化里生效** | 冷槽分支：`gqa_attention_decode_nvfp4.cuh:132-133,512-523`（`cold_k_slots/cold_v_slots`）、`gqa_attention_decode_impl.cuh:231-241`（i8 tiled 的 `cold_k_i8/cold_v_i8`）；`cold_slots` 来自缓存视图 | 若某个 `TokenTile` 实例化漏掉冷页分支（或冷页在另一 codec 下解读），长上文的历史键读到的值就不同 ⇒ 系统性偏差，长上下文才暴露 | **CPU**：grep 实例化表逐个确认是否都带冷页参数；**CLI**：把 `--max-context` 拉大做长文对照 | 中 |
| D9 | **verify 的 envelope 下界用 `1` 而不是 `frontier+1`** | `program_impl.h:594`（`out.target_verify = {1, visible(max_frontier+k+1)}`）vs plain `program_impl.h:11800,11816`（`{frontier+1,frontier+1}` 或图档区间）；消费点 `gqa_attention_decode_impl.cuh:104-120`（`include(envelope.min_visible_keys)`） | 某些 route 的 split 容量按区间**上界**取值 ⇒ 图内/图外、plain/verify 的 split 数可能不同（= D1 的一个具体来源）；**待证实**：注意力内核当前用 `window=last_pos+1`（`gqa_attention_decode_bf16.cuh:124-127`）而不是 envelope ⇒ 目前可能无害 | **CLI**：自洽性基线（plain×2、spec×2 必须逐 id 相同）+ **CPU** 读 envelope 的全部消费点 | 低-中 |
| D10 | **verify 没有任何 head probe（观测缺口）** | `text_context_impl.h:817-821`（plain 有 `post_embed/post_layers_x/final_hidden/logits` 四处）vs `:873-888`（verify 只有 `tap.begin/capture_positions`，无 probe） | 不是数值差异，而是"为什么现在只能比 token id"的原因；也是 D3/D1 判别实验必须补的一处补丁 | 不需要实验（是实验的前置） | —（建议优先补的观测点） |

---

## 3. 最省事的判别序列（按"成本从低到高"）

全部默认 `temperature=0`（CLI 的 `--greedy`/默认贪心）、**比 token id 不比文本**、GPU 独占。

**E0（CLI，已有脚本）自洽性基线** — `bash /mnt/c/Users/User/Documents/ziqinzhang/_verify_equivalence.sh`
（`plain×2` 与 `dflash2×2` 必须逐 id 完全相同）。
任一不自洽 ⇒ 先查确定性（CUDA 图捕获/envelope/workspace/RNG），等价性讨论作废（可能表现为 D9）。

**E1（CLI，最便宜且决定性；本任务新增，不在现有脚本里）**
同一产物、同一 prompt、`--no-spec`，只换 `--prefill-chunk`：
`--prefill-chunk 128` vs `--prefill-chunk 2048`（可再加 4096），各 `--max-new 16 --print-token-ids`。
- **不同** ⇒ 引擎在本模型上"贪心输出不随 token 分块大小不变" ⇒ **D1/D2 直接证实**，
  verify≠plain 有共同根因；并且意味着接受率里有一部分**不是草稿质量能挽回的**。
- **相同** ⇒ D1/D2 在本模型上弱，火力转向 D6（dflash2 rope delta）→ D3（量化放大）→ D7（草稿有效位）。

**E2（CLI）首分叉位置 vs 草稿宽度** — `plain` / `dflash2` / `mtp3` 各跑一次（现有脚本已覆盖一部分），
再用 `--draft-tokens 1/2/3/5/7` 扫。判读：
首分叉位置随 k 单调提前 ⇒ D1/D3（T 与 KV 长度相关）；
与 k 无关固定在早期 ⇒ D6/D7（固定机制）。

**E3（CLI）上下文长度扫描** — prompt 与 `--max-context` 拉到 4k/8k/16k：
首分叉提前 ⇒ 与历史 KV 长度有关（D3/D8/D9）；完全不变 ⇒ 只与当轮 T 有关（D1）。

**E4（CPU）D6 的一票判定** — 追 `sequence.rope_delta` 赋值链（`program_impl.h:11991` 写 ingress、
`text_context_impl.h:698-700` 用于 MTP；dflash2 侧未加），确认 dflash2 是否恒为 0。
若非 0 ⇒ 这就是 dflash2 那一路的系统性错误，**不必先做任何 GPU 实验**。

**E5（需窗口：一处补丁）dump 同一位置的 logits** — 给 verify 补一个 probe（plain 已有，见 D10），
dump 首轮列 0（也可列 1..k）的 BF16 原始位与 top1/top2 值。三分支：
(a) logits **逐位相同**而 token 不同 ⇒ argmax 实现差异（按 E1 不该发生 ⇒ 重大发现，回头查
`argmax.cu:62-79` 的 memset/atomic 收尾路径）；
(b) logits 只差末 1-2 bit 且 top1≈top2 ⇒ 近并列 + 次 ULP（D1/D2/D3）；
(c) logits 稳定差 ≥1e-3 或顺序错 ⇒ 系统性（D6/D7/状态/冷页）。

**E6（需窗口）T 隔离** — 把 verify 改成"逐列 width=1 跑"（位置/ids 完全不变），与 width=k+1 对比 token id：
不同 ⇒ 坐实 D1（T 相关归约），此时"verify≡plain"只能靠**统一 T 的路由**解决，而不是改草稿。

---

## 4. 与"接受率"的关系（边界说明）

- 本报告的目标是**等价性**（贪心下投机必须逐 token 复现 plain）。上面 D1–D3 是它的候选根因。
- **pos0 接受率 ~53%（目标 ~0.9）是另一个问题**：pos0 的定义是 `draft[0] == target 列 0 argmax`
  （`speculative_round.cuh:130,134`；`accepted by pos` 第 m 位 = drafts[m] 由列 m 判定），
  它主要衡量**草稿**而不是 verify。等价性坏了会让"验收列"本身错（进一步压低接受率），
  但即使 verify≡plain，"53%" 也说明草稿顶 1 命中率只有约一半 ⇒ 还需要一条草稿侧的独立线索（D7 是候选）。
- 我**没有**证据说 verify≠plain 就是接受率低的主因；反过来，`mtp3 num` 全程逐字相同说明等价性缺口
  是"偶发"的（近并列 + 量化边界），它的直接后果是**输出正确性风险**，而接受率的量级问题应另查草稿链
  （E5_s47 的 H1 假设、`mtp_round.cuh:47` 的有效位、`alignment_ids/hidden` 的选择列）。

## 5. 本报告未覆盖 / 待证实清单

1. `src/ops/linear/nvfp4/*` 的 N 维路由（D2 是否真的换 kernel）——未读。
2. 各 `TokenTile` 实例化是否都带冷页分支（D8）——未逐个 grep 完。
3. `envelope` 的全部消费点（D9 是否真有害）——只读了 bf16 小 T 内核。
4. `sequence.rope_delta` 对 dflash2 是否为 0（D6 是否成真）——未追赋值链。
5. `ar_valid_columns` 与 `accepted` 的不变式（D7 是否真能取到 0 列）——未证明可达性。
6. dflash2 的 `frame.proposal_extents` 与 `target_valid_columns` 的一致性（`program_impl.h:12402`）
   与 `dflash2_impl.h:355` 要求 `k == block_drafts` 的交互——只读到表面。
