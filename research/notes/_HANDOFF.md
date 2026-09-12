# HANDOFF：dflash2 接受率 / 周期性偏移（新会话从这里开始）

## 0. 你的角色与铁律（每次回复开头原样复述）
铁律：① 长任务不空等，等就并行推另一项；② 杀进程/端口先查 cmdline；③ 状态同步 `_TODO.md`；
④ 数据/数值/布局问题一律实证，不纸上推测；⑤ 不要等任何东西，等的时候就思考并行；
⑥ **绝不扫盘**——只用已知路径定点读写，或直接向用户要指针；单文件内 grep 允许。
你是 ninfer-fusion 的唯一执行者；用户只做最终验收（"我出方案你自己实施""只要结果验收"）。
方法：遇到问题先分拆 → 派子代理设计论证 + 你自己也做一部分 → 回主 chat 合并。

## 1'. 子代理配置现状（2026-09-11 由用户裁定）
- `C:\Users\User\.zcode\v2\agents-state.json` 的 `builtInModelOverrides` 已**清空**，子代理回落到
  ZCode 内置默认模型（此前我把 general-purpose/Explore 指到 `…/deepseek-flash`，该模型**窗口声明
  与实际能力不匹配**导致子代理一开工就爆、并连带主会话 autocompact thrash）。
- 生效需重启 ZCode（客户端会缓存）：脚本 `C:\Users\User\Documents\ziqinzhang\_restart_zcode.ps1`
  （kill 全部 ZCode 进程 → 5 s → `Start-Process` 重新拉起）。
- 回滚：`dl\agents-state.bak3_myedit_deepseek_flash.json`（= 指向 deepseek-flash 的那版）、
  `dl\agents-state.bak_20260911_091726.json`（= 最早的退役模型版）。
- 纪律：**不要再给内置代理配模型名**，就这样用默认。

## 1. 环境与已知路径（全部已核实存在）
- 构建树（权威）：WSL `/home/user/ninfer-fusion`；CMake+make，二进制 `build/apps/ninfer`；
  增量编译 `cd /home/user/ninfer-fusion/build && make ninfer -j2`（**无头文件依赖跟踪** ⇒ 改 .h 后
  必须 `touch` 其 includer；纯 .cpp/.cu 改动 make 会自动重编）。
- 模型 artifact：`/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer`（另见 `..._dspark.ninfer`、
  `qwen3_8_27b_nvfp4.ninfer` 纯 target）。
- 参照实现（vLLM fork，本机只读）：`C:\Users\User\Documents\ziqinzhang\1Cat-vLLM\`
  （`vllm/model_executor/models/qwen3_dflash2.py`、`vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py`）。
- 契约（权威文档）：`C:\Users\User\Documents\ziqinzhang\ninfer-upstream\docs\maintainer\qwen3.8-27b-dflash2.md`。
- ckpt 自带超参：`C:\Users\User\Documents\ziqinzhang\data\draft_dflash2_ref\config.json`（DFlash2DraftModel，参照草稿）。
- 我写的报告：`_collab\R6…R13`（R12=引擎侧排除、R13=GDN 周期性偏移）；续接快照 `_collab\_resume_after_restart.md`；
  状态与历史全在 `_TODO.md`（末尾是本次追加）。
- 工具脚本（都在 `C:\Users\User\Documents\ziqinzhang\`）：`_ga_check.sh`（spec vs plain 逐位比较）、
  `_ga_ksweep.sh`（K 扫描）、`_stepwise_audit2.py` / `_df2_blame2.py`（逐列 blame）、
  `_verify_df2head.sh`（接受率+G-A）、`_land_gdn_conv.sh`（补丁落地范式）。
- Bash 工具坑：内联引号会被吃 ⇒ **把命令写成 .sh 文件再 `tr -d '\r' < f > /tmp/x.sh && bash /tmp/x.sh`**。

## 2. 已确证的事实（**不要重复推导**）
1. **target 侧健康**：verify 列 0 的 argmax 与无投机贪心流一致 17/20=85%（列 1 在被接受前缀上 6/8=75%）。
2. **walk argmax 曾是真 bug，已修**（F4：`chosen` 恒 0 = 丢边项；修后 `pred` 真走链）。参照实现的
   walk 语义与修后一致（行=previous、列=C_step、previous 链式）。
3. **K=1 就偏、K=1 与 K=3 同位置同 token ⇒ 不是掩码污染**；分歧位置随每轮列数变化（K=1→62、K=3→62、K=7→29）。
4. **MTP 在同一 verify 路径上拿到 `[29,16,5]`**（p1 55%、p2 31%）⇒ verify/轮次机制**能**支撑多 token 链。
5. **已落地但仍未消除偏移**：`src/ops/gdn_input_proj/gdn_conv.cuh:116` 改为
   `s2 = __bfloat162float(__float2bfloat16_rn(p));`（备份 `/home/user/gdnconv_bak/`）。
   落地后 **mtp 的 token 流逐位未变** ⇒ mtp 不走这条 epilogue ⇒ 生效的是**另一条** GDN kernel。
6. **接受率上限归 ckpt**：引擎 p0 26–35% ≥ 该 ckpt 离线上限 ≈25–31%（A2 §Q1）；训练喂真 token
   （mask id 0/4468）+ 默认 `--target-shift 1`，而引擎推理喂 `[anchor, mask×7]` + shift 0 ⇒ 第 1..K 位分布外。
7. 已排除：CUDA graph 回放、KV 精度（bf16）、`rope_delta`（纯文本 0）、logit policy（qwen3 no-op）、
   声明型草稿超参（与 ckpt config 逐项一致）、head 选择（两 route 剖面相同）。
8. 保真项（**已由 S_D 修正，别再按旧说法做**）：BF16 只从两处进入 selector —— `logits`
   （`dflash2_impl.h:342-347`，决定候选集/名次 + unary 数值）与 `projected`（`:350-353`，edge 项）；
   `unary`/`scores` scratch 本来就是 FP32。参照的"FP32"只作用在 **unary 一侧**（SM70 默认开
   `VLLM_SM70_DFLASH2_FP32_LOGITS=1`，走 FP8 top-64 → FP32 重算 → top-16 两段式），
   其 `hidden_projection`/codebook 仍是 bf16 ⇒ `projected` **不必改**。裸改 dtype 会静默出错
   （`linear.cpp:52` 强制 x/out=BF16；nvfp4/fp8 dispatch 还会把输出重打成 BF16），需新 op 做两段式；
   **且它与 spec≠plain 的逐位偏差无关**（selector 只影响草稿），也与接受率上限无关 ⇒ 低优先、不要合批。
   ⇒ 正确的下一批实验见 R15（形状相关数值不等价）与 R14（扰动幅度而非首次翻转点）。

## 3. 待派发的四路子代理（各自独立、只读分析、**禁编译/禁 GPU**、交付到 `_collab\S_*.md`）
派发时把"硬约束"抄进每份 prompt：绝不扫盘；不改源码；不编译不跑引擎；只写一份报告；
返回 ≤15 行结论摘要 + 报告路径。
- **S_A GDN kernel 分派**：`src/ops/gdn_input_proj/{nvfp4,q4_q5,w8,fp8}/` 里按权重格式与 token 数
  分派的代码（T=1 plain vs T=W verify 各走哪条），GDN 递推是"逐 token"还是"分块（chunked）"，
  给出 file:line + 最小对齐方案；并解释为何落地 conv 补丁后 mtp 的流逐位未变。→ `_collab\S_A_gdn_dispatch.md`
- **S_B GDN 状态记录/回滚**：`core/gdn_replay_records.h` + `text_context.h`（`GdnStateAction{UpdateInPlace,RecordForReplay}`）
  + spec 轮次调用点；verify 一次推进 W 列时**被拒列**的状态如何回滚、是否与逐 token 语义严格等价；
  给出可检验的预测。→ `_collab\S_B_gdn_state.md`
- **S_C 训练口径上限**：读 `train_dflash2.py`（根目录）自行验证 A2 的 M1（shift）/M2（真 token vs mask）
  两条主张；说明引擎要匹配该 ckpt 需要什么输入口径；给出 retrain 后**可证伪的预期**（每列接受率）。→ `_collab\S_C_train_regime.md`
- **S_D selector 保真**：BF16 进入候选顺序/分数的确切位置与 FP32 化改动面；
  以及 accept 侧是否消费 `draft_candidate_probs`、greedy 下 softmax vs one-hot 是否有正确性风险。
  → `_collab\S_D_selector_fidelity.md`（给 diff 文本，不要落地）

## 4. 主代理自己留的（GPU/编译独占，串行）
优先级：① 读 S_A/S_B 结论后，在**正确的** GDN kernel 上做 chunked-vs-递推的对齐；
② 验收判据：**spec 流与 plain 逐位一致**（`_ga_check.sh`）且接受率不低于修复前；
③ selector 分数 FP32 化与 ① 合批一次编译；
④ 落地后用 `_verify_df2head.sh` + `_df2_blame2.py` 复测（判据：链式剖面 p1+ 非零、AL≥3、G-A IDENTICAL）。

## 5. 待用户决策（不要自作主张）
- **续训** `_train_df2_shift0.bat`（mask 输入 + shift 0）——会长时间占 GPU，用户说"训练暂时放下"。
- **参照草稿 A/B**：`data\draft_dflash2_ref\` 已在盘（训练过 mask 块），用它打 artifact 跑一次可一次区分
  "引擎 vs ckpt"；用户此前说"别下参照草稿"，需其点头。

## 6. 上下文纪律（本会话被 autocompact 守卫挡过，务必遵守）
- 大文件/大日志**交给子代理读**（输出落在它们上下文里）；
- 自己读一律带小 `limit`（30–40 行），不整份 `cat`、不 dump 整段日志；
- 一切状态即时写 `_TODO.md` / `_collab/`，随时可新开会话续接。

> **R12 更正（R18）**：E1/E2 论据作废（live artifact 是 08-26 草稿，训练 ckpt 从未进过引擎）⇒ "引擎侧已排除" 的定量桥撤销；E3/E4/E5 仍有效。retrain 前先钉死 serve 4.55 vs CLI 1.35 tok/round 的混淆项； 会取到 legacy 的 step_001900 ⇒ 必须显式 。

> **R12 更正（见 R18，权威版）**：E1/E2 论据**作废**——live artifact 是 08-26 草稿、训练 ckpt 从未进过引擎
> ⇒ "引擎侧已排除" 的定量桥撤销（E3/E4/E5 仍有效）。retrain 前先钉死
> serve 路径 4.55 tok/round vs CLI 1.35 tok/round 的混淆项。
> **续训必读**：`data/dflash2_ckpts/` 混两套配方，`--resume` 会取 `sorted()[-1]` = legacy 错位的
> `step_001900` ⇒ 必须显式指定输出目录参数（`--out-dir`），否则接错配方。
> 便宜判定优先：测已有 mask+shift0 的 `step_000100/000200` 与 08-26 草稿的"塌陷比"R（旧≈0.18，
> 新配方预测 ≥0.6），比重训便宜。
