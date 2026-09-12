# R16：S_B 结论 —— 状态机是精确的，残差全在"形状相关的进料/出料"（2026-09-11 17:0X）

## 一、S_B 已裁决（报告 `_collab/S_B_gdn_state.md`）
1. **spec 轮不存在"回滚"**：`RecordForReplay` 下 key/value/gate/conv 只写记录平面、主状态零改动
   （`recurrent.cuh:350-441`）；被拒列 = 从未应用。
2. 只有 committed 前缀（`accepted+1`，anchor 恒提交）经 `gdn_replay_fold` 重放
   （`program_impl.h:9847-9868` → `recurrent.cu:104-128`）。
3. **fold 与"逐 token 推进 commit 次"逐位等价**（同一 `run_recurrent_sequence`、同一记录、同一 FP32 起点）。
4. conv 窗口携带精确（记录存 `bf16(p)`；`publish_final_conv_history` 是 `tail_3` 的逐下标双射 + static_assert）。
5. `commit ≤ valid_columns`、`accepted+1 == count` 有强校验（`program_impl.h:12491-12498`）。
⇒ **结论：状态转移严格等价；不等价的只是"W 列一次"与"1 列一次"两套形状产生的进料/出料。**

## 二、残差清单（全部同一"形状相关数值不等价"家族）
| # | 位置 | 机理 | 量级 | 触发 |
|---|---|---|---|---|
| D1 | `fused conv :99-104` vs `:118` | 第 4 抽头用 FP32 `p`，却把 `bf16(p)` 发布进状态 | ~2e-3 | batch==1 且 W∈{2,3,7..10}（**W=8 ⇒ dflash2 中招；W=4 ⇒ mtp 不中招**） |
| D2 | flat conv vs record 侧 | `acc += w*x`（8 舍入） vs `fmaf` 链（4 舍入） | ~1e-7 | 偶发 LSB |
| D3 ★ | `gated_delta_net.cpp:186-191,256-260` vs `recurrent.cuh:615-631` | chunked 把 `l2norm(q/k)` 发布成 **BF16**；recurrent 用寄存器 **FP32** ⇒ **同 token 不同 k** | 形状相关 | chunked(≥64) vs 单列 |
| D4 | `chunked/launch.h:26-35`（kChunkSize=64） | 工作区 `W/U/v_new/h_chunk` 全 BF16 ⇒ 64 token 粒度已发布值、对切分敏感 | 形状相关 | `--prefill-chunk` 改变切分 |

**D1 与今天落地的 `gdn_conv.cuh:116` 同类**（都是"状态里存的是发布值、计算里用了更宽的累加器"）；
补丁打在 FusedA16 ⇒ 只有 W∈{2,3,7..10} 的 batch==1 受影响 ⇒ **解释了"补丁改变了 dflash2 的流、mtp 逐位未变"**。

## 三、战略性结论（回答"消除偏移"的可行边界）
- 用户 P1 闸门 G-A 要的是 **spec 输出与 plain 逐位一致**。按 S_B：状态机已精确，
  但 **chunked GDN 与 recurrent GDN 的中间量表示不同（D3/D4）** ⇒ 只要 verify 仍走 chunked 形状，
  spec==plain 就**不可能**逐位一致（任何引擎同理，含参照实现）。
- ⇒ 真正可落地的方向只有两条：
  a) **让 verify 的 GDN 逐 token 走 recurrent 路径**（与 plain 完全同算术）⇒ 换来 G-A 逐位一致，
     代价是 verify 里 GDN 段失去 W 列并行；
  b) 保持 chunked，但把 `l2norm(q/k)` 与工作区提升到 FP32（D3/D4 的深改，牵动 chunked 全部 kernel）。
- **先做便宜的一半**：D1 的一行修复（按 `:118` 范式把 `:99` 的 `p` 先 bf16 化）⇒ 消掉 2e-3 那一档；
  它只影响 W∈{2,3,7..10}（dflash2 默认 W=8 命中）。

## 四、下一步（主代理车道，按性价比）
1. **P1 可证伪预测（S_B 给）**：强制走 `MaterializedA16` 后跑 `_ga_check.sh`，
   预测 K=1/K=7 的首次偏离会变、**K=3（W=4，本就 materialized）不变**。若成立 ⇒ D1 确在动数。
2. 落地 D1 一行修复（与其它改动合批一次编译），复测 `_ga_check.sh` 与接受率。
3. 若要求 G-A 逐位一致 ⇒ 按 §三(a) 立项"verify 的 GDN 走 recurrent 逐 token"，单独评估性能代价。
4. 与 S_A/S_C 合并：S_A 定"哪种格式/宽度走哪条 conv 与 GDN 路径"（含打印 `Fp8GdnConvScheduleId` 定案），
   S_C 给接受率上限的口径证据。
