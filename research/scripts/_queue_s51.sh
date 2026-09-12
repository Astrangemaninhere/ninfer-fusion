#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
python3 - <<'PY'
import datetime, pathlib
T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
T.open("a", encoding="utf-8").write("""
### 11. S51 收尾细则（E8 终稿，落地下一次时必须带上，%s）
- **机制与上游不同**：我们树里 **没有** `__fdividef`（全树 0 处）、**没有** `-use_fast_math`（153 个 .cu 条目里 0 个），
  所以我们的缺陷形状相同但机制是"精确 expf 在 x ≤ -88.72284 溢出到 +inf，IEEE `x / inf` 得 -0"；
  补丁保留我们的 `expf` 与 IEEE 除法，只把指数折到不溢出的一侧（所以不是照抄上游 hunk）。
- **同族 2 处**（都在 `src/ops/common/math.cuh`）：`silu`（:13）与 `sigmoid`（:15），diff 两个都改；
  背后是 **66 个 silu 调用点 / 18 文件** 与 11 个 sigmoid 调用点 ⇒ **一处修改修 66 处**（与"硬编码 256 × 81 处"相反）。
  E8 另列了 11 处"无需改"的点及理由（softplus、三处 `__expf` softmax、14 处 `__frcp_rn`、测试 oracle、文档注释、bench 字符串）。
- **穷举证据（g++/glibc + numpy 双算）**：旧式在 [−1000,0) 上静默归零 **1,998,749** 个 float32；
  换算到 bf16 层：**救回 1,145,241 / 回归 0 / 扰动 2,432**（占 1.12e9 非零点的 2.2e-6）；
  而在 **x ∈ [0, 60) 的 1,114,636,288 个 float32 上逐位完全相同**（正半轴零风险）。
- **两处不能照抄上游的说法**：① "两式在共同有定义处完全相同"是**错的**——最大相对差 3.3e-7~5.0e-7（2.8–4.19 ulp）、
  最大绝对差 < 1 ulp(1.0)，仍在 bf16 量子之下但不能说"相同"；② 上游 PR 里的 "−9.6e-37" 复现不出来（实测 −1.0267e-36），引用时用我们的数。
- **落地时必须补一个测试**：现有测试证明不了这个修复（`test_silu_mul.cpp:17` 的 gate 只到 ±12，
  `linear_swiglu_test_common.cpp:37` 对 A4 允许 1.6e-1）⇒ 落 S51 时要同时加"极端负值"回归用例；
  另外 E8 给了"我们 artifact 到底会不会踩到"的一行 `__any_sync` 计数实验（md §6.2）。
- 待证实（E8 自己标注）：sm_120a 上 `expf` 的确切溢出点（无编译/无 GPU，只交叉验证了两个宿主 libm）。
""" % datetime.datetime.now().strftime("%H:%M"))
print("  queue updated with S51 landing requirements")
PY
echo
echo "=== build: which TUs are done / in flight ==="
grep -oE '^\[[ 0-9]+%\] (Building|Linking)[^"]*' /tmp/pa_make_1.log | tail -4
for p in $(pgrep -f bin/nvcc); do echo "  nvcc elapsed $(ps -o etime= -p $p | tr -d ' ')"; break; done
ps -eo pid,rss,etime,cmd --sort=-rss | head -3 | awk '{printf "  %8s %8.0f MB %8s %.48s\n", $1,$2/1024,$3,substr($0,index($0,$4),48)}'
echo "=== fresh objects so far ==="
ls -lt --time-style=+%H:%M /home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher/*.o 2>/dev/null | head -4 | awk '{print "  ", $6, $5, $NF}'
