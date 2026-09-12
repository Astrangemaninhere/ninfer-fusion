# R17：首因确认 —— GDN 预归一化的 BF16/FP32 双路（S_A 预测一次命中）（2026-09-11 17:2X）

## 一、可证伪预测的结果（零编译、零投机）
| prompt | `--prefill-chunk 128` vs `4096` |
|---|---|
| 未对齐（≈1200 token） | **第 0 个生成 token 就分叉**（104980 vs 103735） |
| **对齐到 1920 = 15×128** | **整段 IDENTICAL（31 token）** |
脚本 `_align128_ab.sh`（含对齐脚本 `_align_prompt128.py`，tokenizer 直接从 artifact 的
`frontend/tokenizer.json` 取，不加载模型），日志 `dl/align128_ab.log`。

⇒ **S_A 的机理①成立**：`gated_delta_net.cpp:254-261`
—— 只有当该次调用处理了完整 64 块（`T_full>0`）时，q/k 的 `l2norm` 才写进 **BF16 缓冲**
（缓冲 dtype 见 `:187-190`）并令 `recurrent_normalize=false`；`T_full==0`（本次 <64 token）时
改在 **FP32 寄存器**里归一化（`recurrent.cuh:64-70,125,617`）。
⇒ 同一批 token 因"切分方式"不同而拿到 **不同的 k**（`l2norm(q/k)` 的发布精度不同）。

## 二、意义
1. 这是**第三处**"状态/缓冲里存的是发布值、计算里用了更宽累加器"的同族缺陷
   （前两处：`gdn_conv.cuh:116` 的 `s2=p`、fused conv 的 `:99-104`）。
2. 它**只影响形状相关的进料**（不是状态回滚问题，S_B 已证明状态机精确）
   ⇒ 正是 **verify(T=W=8 列) vs plain(T=1 列)** 这类形状切换的差异来源。
3. 由此得到**最小改动清单（S_A 给，按性价比）**：
   ① **删/改 `gated_delta_net.cpp:255-261` 的 BF16 预归一化分支（≈6 行，本次首因）**；
   ② `h_chunk` BF16→FP32（3 处：`chunked/launch.h:42`、`launch.cu:55,68`、`output.cuh:11,155`）；
   ③ `nvfp4_gdn_snapshot_plan.cpp:42-44` 放宽到 `tokens<=16` 分派；
   ④ 残余的"按 T 选 tile/route"（`nvfp4_gdn_input_w4a4.cu:40-58`、attention/conv 按 T 选 kernel）
      只能统一 kernel 家族或接受非 bit-exact —— 建议先修 ①②③ 再复测，否则会把"精度档/发布口径"
      误判为不可控舍入。
4. **回归探针（重要细节）**：S_A 建议把"同 prompt 只改 `--prefill-chunk`"加入验收；
   但本次实测表明它**只在对齐非 128 倍数时灵敏**（对齐后即使缺陷仍在也会通过 ⇒ 假阴性）
   ⇒ 探针必须用**非 128 对齐**的 prompt（例如 1200 token 那版）。

## 三、下一步（主代理车道，串行一次编译）
1. 落地 ①（≈6 行）与 ②（3 处 dtype）；③ 视风险决定是否同批。
2. 复测三件：
   a) `--prefill-chunk 128 vs 4096`（**非对齐** prompt）→ 期望 IDENTICAL；
   b) `_ga_check.sh`（spec vs plain，zh/num）→ 看首次偏离是否后移/消失；
   c) 接受率剖面（`_verify_df2head.sh`）→ 不应退化。
3. 若 ①②③ 后 spec 仍 ≠ plain（预期仍有 ④ 的按 T 选 tile），再立项"verify 的 GDN 走 recurrent 逐 token"
   （S_B §三(a)）或接受非 bit-exact 并在文档里明确 G-A 的适用边界。
