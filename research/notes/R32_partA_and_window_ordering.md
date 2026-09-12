# R32：PART-A 结果 + 两个窗口的强制排序（幽灵副本已核实）（2026-09-11 18:0X）

## 一、PART-A（FP32 partial_acc）已就绪
- 产物：`_collab/PART_A_patch.diff`（**21 文件 / 61 hunk / +124 −125**）+ `PART_A_report.md`；
  `patch -p1 --dry-run` **21/21 clean、exit 0、无 .rej、树未改**。
- 改动四层：写侧签名（18 处）→ 打包 store（14 处，`make_float2`）→ 中性初始化（5）→ reduce（5）→
  `DType::BF16→FP32` 分配（5）→ `.data` 转换（19）；**`partial_m/l`、索引/步长、路由全未动** ✓。
- 性能：partial 缓冲 ×2（27B T=1 ctx4096 splits=64：768 KiB→1.5 MiB），per-token 流量 **+≈0.17%**；
  **verify(T=8 走 Prompt，该缓冲 0 字节) ⇒ verify 不可能退化** ✓；smem/占用无恶化（部分档位还少一次
  `__syncthreads` 与一次 BF16 中转）。唯一让步：bf16/fp8/iso3 的 128-bit `int4` store → `float2`
  （保 128-bit 需 +16 KB smem 且扰动占用；回退方案见其报告 §4.1）。

## 二、PART-A 的两条关键新发现
1. **27B 的 KV 默认档不是 BF16**：是 `NVFP4` + 10 层 `E8Kv`（`qwen3_6_27b/impl/variant.cpp:23-39`）
   ⇒ **活路径的 partial 核是 `decode_nvfp4` 与 `decode_i8<E8>`**；`decode_bf16.cuh` 只在 `--kv-dtype bf16` 时激活
   ⇒ **只改 bf16 那条会测不到变化**（本 patch 两条都覆盖 ✓）。这条与 §三的排序一起，是本次能测出效果的前提。
2. `dense/causal_cache`（SmallT 并行族）**不可达、自成闭环**，且其 `small_t_fp8` **早就是 FP32 + `make_float2`**
   —— 正是本次要对齐的形态 ⇒ 不动它是对的（§4.2）。

## 三、幽灵副本：**已核实**，并据此定死两个窗口的排序
实测 `src/ops/launcher/`：
| 文件 | 大小 | 在 CMake？ |
|---|---|---|
| `gqa_attention_decode.cu` | 45.8 KB | **是** |
| `gqa_attention_decode_e8.cu` | 12.5 KB | **是** |
| `gqa_attention_decode_smallt.cu` | 16.5 KB | **否** |
| `gqa_attention_decode_partial.cuh` | 26.5 KB | **否** |
| `gqa_attention_decode_impl.cuh` | 37.2 KB | **否** |
| `gqa_attention_decode_g35.cu` / `_muse.cu` | 2.5 KB 各 | **否** |
⇒ `_smallt.cu` / `_partial.cuh` 的 mtime = **Sep 10 19:57**，即"暂存 TU 拆分"那批产物，**早于今天的 FP32-partial 补丁**
⇒ **它们内部必然仍是 BF16**。**若先落 TU 拆分，会把 BF16 静默接回构建（测出来的将是旧行为）** ✗✗

### 强制排序（两个窗口）
1. **窗口①：先落 FP32 partial**（`PART_A_patch.diff` 的公共核心，或两路合并后的版本）→ 编译 → 复测
   （判据：`verify 列 0 与 plain 一致率 69% → ≥90%`、首次偏离后移/消失、接受率与 tok/s 不退化）。
2. **窗口②：再落 TU 拆分**，且**必须**在同一窗口内：
   - 从**打完补丁后**的 `gqa_attention_decode.cu` / `_e8.cu` **重新生成**拆分产物（或对 `_smallt.cu`/`_partial.cuh`
     施加同样的 FP32-partial 编辑），并**校验无 `bfloat16` partial 残留**（grep 断言）；
   - 否则 §三 的静默回退会发生。

## 四、状态
- PART-B 仍在跑；到齐后按 P7：对比 21 文件的逐点差异 → 取共同核心 → 落地窗口①→ 复测。
