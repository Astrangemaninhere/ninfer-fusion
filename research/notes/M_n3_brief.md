# N3 派发简报 (待 agent 空闲即发; 目标: 把"启动自测→固化→跳过→手动重校"做成可判定功能)

## 实证基础 (已核实, 不要再假设)
- 行标定表: `src/ops/kernel/gqa_isoquant_row_scale.cuh:19`
  `extern __constant__ unsigned short kGqaKvRowScaleDev[16][4][256];` ⇒ **运行时可整表替换**
  (`cudaMemcpyToSymbol`), 不需要改 kernel、不需要重编。
- `KvCalibrationCapture` (`src/targets/qwen3_6/impl/runtime/kv_calibration.h:28`) 是**死代码**:
  全树只有它自己的注释与定义命中; `EngineOptions.kv_calibration_dir` 无消费点。
  记录格式已定义好: 每 (full_layer, chunk) 一个 `.kvc` 文件, 64 字节 header (magic `NINFERKVCAL1`)
  + positions + BF16 K + BF16 V。
- 表只覆盖 **16 层** ⇒ Muse-Glimmer-30B (52 层) 在 `gqa_kv_row_scale()` 里直接返回 identity 1.0
  ⇒ Muse 的旋转域量化**从未标定** (这里是质量缺口, 不只是机制缺口)。

## 交付物 (按可判定性排序, 建议分两轮)
### 轮 1 (机制, 小):
1. 旁车格式 (定稿建议): `magic[16]="NINFERKVRS1" + u32 version + u32 layers + u32 kv_heads + u32 head_dim
   + u32 crc32 + padded to 64B + payload (bf16 words, layers*kv_heads*head_dim*2 bytes)`。
   **必须能表达"identity/未标定"的两态**, 且与现有常量表的排布逐字节兼容
   (验证: 从当前常量 dump → 写旁车 → 读回 → 与原常量 diff 为空)。
2. 引擎侧加载: 启动时若 `--kv-rowscale <path>` 或 `<artifact>.kvrs` 存在且校验通过 ⇒
   `cudaMemcpyToSymbol(kGqaKvRowScaleDev, ...)`; 失败必须**硬报错**, 不允许静默回退 (否则等于假成功)。
3. 判定: 加载旁车后 qwen 路径 32K 掉针与加载前**逐位不变**; 旁车损坏时启动报错并给出行号。
### 轮 2 (闭环, 中):
4. `--kv-calibrate off|auto|force` 与 `--recalibrate`: auto = 旁车缺失才采集; force/recalibrate = 强制重采。
   采集用现有 `KvCalibrationCapture`, 需要把 `kv_calibration_dir` 接到 prefill 路径 (现在是死的)。
5. 采集后调 `tools/calib` 的分析器烘表 → 写旁车 → 记录一行 forensics (路径 + crc + 表统计)。
6. 判定: 首次启动出现 capture+bake 行; 二次启动出现 `skip (table ok, crc=…)`; `--recalibrate` 后重新出现
   capture 行; Muse 用标定表 vs identity 的 32K/57K 掉针对照 (Muse 侧的**新质量收益**)。
## 约束
- 所有引擎侧改动走"补丁 + 一次批处理编译"(见 `_window_h.sh` 的编排), 不要单独开窗口。
- 采集/分析是 CPU 侧, 可在训练运行时做; 只有加载与掉针对照需要 GPU 窗口。
