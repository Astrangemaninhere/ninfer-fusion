# 构建效率问题的结构性修法 (2026-09-10 11:2X)

## 先把账算对
- `make` **不会**"从头开始": 已完成的 `.o` 全部保留。今天三次事故后大部分 TU 的产物仍在，
  所以每一轮实际只需补 2 个 decode TU（这也是为什么 window D/E/F 都能立刻接着跑）。
- 真正的痛点: `src/ops/launcher/gqa_attention_decode.cu` 在**一次 nvcc 调用**里实例化
  7 个 KV 档（I8 / E8Kv / NVFP4 / NVFP4+ISO3 / FP8 / ISO3 / BF16）× 每个 (width, MultiBatch, Masked) 组合
  ⇒ 单个 TU 25–35 分钟，且 `.o` 只在最后落盘 ⇒ **被杀在 30 分钟处 = 那一格全废**。
- 三次死因同一根因: WSL `.wslconfig` 天花板 24GB，而单次编译峰值（cicc + ptxas）就顶满它：
  ① 会话 SIGTERM（外因，11:0X 之前）② 内核 OOM killer（11:09/11:10/11:12，journal 实证）
  ③ WSL VM 崩溃（`Wsl/Service/E_UNEXPECTED`）。

## 已落地的止血（window F/G）
1. `.wslconfig`: memory 24GB → **26GB**, swap 16GB → **32GB**；后台采样 `_memwatch.sh` → `dl/memwatch.log`
   记录整轮峰值曲线（用于证明修法有效，不靠猜）。
2. `NVCC_PREPEND_FLAGS=--split-compile-extended=8`: device 编译分区，ptxas 峰值随分区而非整 TU 增长；
   **无需改 CMake / 源码**（这正是选它的原因——不触发全量重编）。
3. `-j1` + 最多 4 次自动重试：make 从已落地的 `.o` 续跑 ⇒ 一轮被杀不再等于整窗报废。
4. 硬闸门保留：无重链接 ⇒ 不做验证（绝不让陈旧二进制产出绿色结果）。

## 结构性修法（留待下一个构建窗口，按性价比排序）
### L2: ccache 作为 CUDA 编译器入口（已装 4.11.3，`~/.local/bin/ccache`）
- 做法: `cmake -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache .`（在 build 目录重跑配置即可）。
- 收益: 任何**输入未变**的重复编译 ~1 秒（我们的 touch→make、崩溃重试、状态来回切换都属此类）。
- 代价: 冷缓存下首次全量重建（一次性）；需先确认 flags/rsp 变更是否会触发全量重编。
### L3: 把巨 TU 按 KV 档拆开（根治"一格 30 分钟"）
- 现状: 一个 TU = 7 档 × N 个 (width, MultiBatch, Masked) = 数百个 kernel 实例。
- 目标: 拆成 5–7 个 .cu（每档一个），共享同一 dispatch 头，用编译期开关只实例化本档。
- 收益: ① 单格 25–35 分钟 → ~5–8 分钟；② 改某档路径只重编那一格（近期修改全是档级: e8/i8/nvfp4）；
  ③ 峰值随格变小 ⇒ 构建与训练**有望并行**（现在必须串行）。
- 风险/验收: 动 712 行 launcher + CMake 源列表；必须用同一组 Muse/e8 判定复核（解码热路径不能靠猜），
  且要求拆分前后**行为一致**。

## 判定是否有效的证据（三条）
1. `dl/memwatch.log` 峰值曲线全程 `avail` 不为 0 且无 `global_oom`（`journalctl -b | grep -c oom` 增量为 0）。
2. `dl/window_g.log` 出现 `REBUILT OK`，且 `apps/ninfer` mtime 更新。
3. 后续 window 的 `MUSE_VERIFY_PASS/FAIL` 与 `E8_VERDICT=` 行（判定式，不是打印式）。
