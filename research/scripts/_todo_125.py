#!/usr/bin/env python3
"""状态同步 §125：Muse SWA 回归（我自己造成的）+ 修复 + 待验证；四路脚本 bug；训练已起。"""
import datetime
import pathlib

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
T.open("a", encoding="utf-8").write("""
### 125. Muse SWA 回归（我造成）+ 修复 + 待验证；训练已起（""" + stamp + """）
- **教训**：为了让构建绿灯，我把 `layouts_impl.h` 的 S24 窗口接线整段回退（因为它的 `requires` 守卫在非依赖
  上下文里无效）。但那段接线**同时服务 Muse**（E3 注释写明：Muse 52 层里 39 层 SWA、window 2048）——
  回退等于把 Muse 的窗口表清零 ⇒ `muse_srv_bf16.log` 里 L46..L51 全 NaN（118 行）。
  **回退一个改动前必须查它服务哪些 target**。
- **修复（已落地+已编译）**：按正确形式恢复接线 —— 去掉 `requires` 守卫、直接调用
  `TextConfig::full_attention_layers()/is_swa_attention()/sliding_window`（三个 target config 现在都声明了成员：
  Muse 本来有 2048/true，27b/35b 已补 0/false），`.layer_sliding_windows = layer_windows` 挂回聚合。
  `ninfer-serve` 18:08 重建、三个 variant `.o` 同步更新（补丁 A 也在）。
- **验证状态：待 GPU 窗口**（训练 18:08 已启动并占满显存 31.9/32.6 GB）⇒ 暂**不能**声称 NaN 已解决，只能说
  "已恢复接线并重建，验证待做"。判别点：Muse bf16 起 serve 后 `nan_lines=0` 且 L46+ 不再是 NaN。
- **Muse + nvfp4 是设计性拒绝**：`nvfp4 prefill requires head_dim=256; this geometry is 128`（S45d 守卫）——
  这正是历史上"把 256 内核跑在 128 几何上"的路径被显式拦住，Muse 只能用 bf16/i8 KV；验收脚本已按此设计（nvfp4 记为预期拒绝，不算 pass）。
- **顺手修**：`_spec_4way.sh` 的 `plain_mtp` 档一直 SERVE_FAILED，原因是 `--spec mtp` **必须**带
  `--draft-tokens ∈ [1,5]`（engine 硬约束，失败日志尾部就是 usage）⇒ 已补 `--draft-tokens 3`。
- **训练**：`train_dflash2.py --steps 6000` 于 18:08:22 启动（从 step_001200 续跑，batch 4 / anchors 12 / ctx 128）。
- **判别实验已就绪**：`_verify_equivalence.sh`（`--print-token-ids` 比 token id，含 plain×2 自洽基线、
  dflash2×2 确定性、plain vs dflash2/mtp3 交叉）——等 GPU 窗口（训练暂停或结束后）。
""" )
print("_TODO.md §125 已追加")
