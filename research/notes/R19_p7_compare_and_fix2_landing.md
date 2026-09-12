# R19：P7 双路对比 + ② 落地实况（单变量无效果）+ 构建事故（2026-09-11 18:3X）

## 一、P7 对比（A 路 vs B 路，同方法独立两遍）
### 求其同（两路一致 / 量级吻合）
| 项 | 结论 |
|---|---|
| **修①**（`gdn_conv.cuh:99`：`p` 先 bf16 化） | **两路逐字一致的唯一实质代码改动**；B 另证发布侧逐位不变（`bf16(bf16(p))==bf16(p)`），修后 fused conv 与 materialized post conv **逐表达式相同 ⇒ 对同 p 逐位等价（是 0，不是 1e-7）** |
| **修②**（`nvfp4_gdn_snapshot_plan.cpp:43`：`tokens<=3` → `<=16`） | 实质一致；B 发现该行**已在树中**（我 15:1X 那次被取消的调用落进去的），B 未回退、只把自己的 hunk 降为注释——处理得体 |
| **量级** | ② 的 A4 激活误差：A 测 L2 9.48%/点积 1.14e-1，B 测 RMS 9.4e-2/点积中位 9.6e-2（A16 分别 0.17%/1.28e-3 与 1.7e-3）⇒ **两路独立吻合 ≈55–57×，属一阶偏差** |
| ① 量级 | A：Δconv/|conv| p90 4.1e-3 ≈ 输出 bf16 步长，~10% 列翻 1 ulp；B：均值 1.4e-3、14.8% 列变、97% 恰 1 个 bf16 步 ⇒ 同性质同量级 |
| **性能** | 两路同判：① 中性；② 权重流量不变、激活 +123 KB/层/轮、**省 1 kernel 启动与 w4a4 中间缓冲**，roofline 下 +0~3%/round ⇒ **必须实测 tok/s**（A 的退路：>3% 就把阈值收到 8） |
| **残余（两路同判，重要）** | T=1 走 gemv、T∈[2,16] 走 small_t ⇒ **不可能逐位一致**（~1e-7，偶发 ±1 bf16 ulp）；batch>1 仍走 compose(A4) ⇒ **验收判据应改为：同精度档 + 同 conv 语义 + 首次偏离后移/消失 + 性能不退化**，而不是逐位 hash 相同 |
### 分析不同（待实验裁决）
- **修③（`gated_delta_net.cpp` 尾块归一化）**：A 判"有机会 IDENTICAL"（`T_full=128`，chunk128/4096 的划分相同，`g_cumsum` 逐 64 块重置、`l2norm` 逐 (head,token) 独立）；**B 判"不可能"**（残余来自 attention/MLP/lm_head 的 T 形状 kernel）⇒ 用 `_align128_ab.sh`（非对齐 prompt）裁决。

## 二、② 落地实况（单变量）：**测不到效果**
- 已落：`nvfp4_gdn_snapshot_plan.cpp:43` = `if (tokens <= 16) { … SmallTFusedA16 }`（备份 `/home/user/fix2_bak/`）。
- 构建成功后的**同一套基准**与修复前**逐项一致**（`zh plain vs df2` 仍 DIFFER@10 同 token；chunk 探针仍 DIFFER@0，
  `104980` vs `103735`）⇒ **② 未改变我们测到的路径**。
- 与 A 路 caveat #2 吻合：`AllowA4` 系源码常量推断；**若 dflash2 verify 的 GDN 输入不走 snapshot plan，② 即 no-op**。
- ⇒ **定案手段（A 路建议）**：打一行 schedule 日志，打印解析出的 `Nvfp4GdnConvScheduleId`；
  同时也需确认 dflash2 verify 实际走哪个 GDN 输入 kernel（snapshot / w4a4 / independent …）。
  在此之前**不要**把 ② 当作已产生收益的修复。

## 三、构建事故与教训（已修复）
- 现象：`make` 报 `Error 127`、`/bin/sh: 1: ccache: not found`，`apps/ninfer` 被删（链接未产出）。
- 原因：构建用 ccache 作 CUDA 编译器启动器（早前 `-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache`），
  而这次 `make` 的 PATH 里没有 `/home/user/.local/bin`（ccache 在那里）。
- **教训（写入纪律）：任何构建都必须 `export PATH=/home/user/.local/bin:$PATH`**；
  我早前成功的构建都在脚本里 export 过，裸 `make` 会 127 并删掉二进制。
- 已修：`export PATH=…` 后 `make ninfer -j2` rc=0，`apps/ninfer` 15:22 恢复（含 ②）。

## 四、下一步
1. **修① 落地**（两路共同核心；我上一版 patcher 因真实行含对齐空格未命中，守卫挡住、树未动）：
   锚点改为子串 `= projected[token];`（唯一）⇒ 替换为 `= __bfloat162float(__float2bfloat16_rn(projected[token]));`；
   落地后跑同一套基准（判据：`_ga_check` 首次偏离后移/消失、chunk 探针、接受率、tok/s 不退化）。
2. **定案 ②**：加一行 schedule 日志，确认 verify 走的 GDN kernel（决定 ② 是有效还是 no-op）。
3. **裁决 ③**：跑 `_align128_ab.sh`（非对齐 + 对齐两组）。
4. 验收口径统一按 §一"残余"栏调整（不要求逐位一致）。
