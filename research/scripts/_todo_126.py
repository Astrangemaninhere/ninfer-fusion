#!/usr/bin/env python3
"""§126：本批落地（S28 补齐 / S51 / Muse verify policy）+ ccache PATH 教训 + Spark 权重完成。"""
import datetime
import pathlib

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
T.open("a", encoding="utf-8").write("""
### 126. 下一批落地（CPU 侧，与训练并行）+ Spark 权重完成（""" + stamp + """）
- **S28 补完（三件套）**：① `gqa_isoquant_row_scale_loader.h` 补 `kv_rowscale_sidecar_apply_from_env`
  声明（签名与 .cu 逐字一致）；② `src/CMakeLists.txt` 把 `ops/kernel/gqa_isoquant_row_scale_loader.cu`
  登记进 `ninfer_ops`（就加在已登记的 `gqa_isoquant_row_scale.cu` 旁边）——**这个 .cu 此前从未被编译过**；
  ③ `decoder_state.cpp` 恢复调用（include 放**文件顶层**、调用用完全限定名 `ninfer::ops::...`，避开当初
  "include 落进 namespace"的老毛病）。⇒ 该特性从"半落地"变成"真进构建"。
- **S51 落地**：应用 `_collab/E8_s51_nvfp4_silu.diff`（`ops::silu/sigmoid` 极端负值归零修复）。
  额外收获：本次修改 `src/CMakeLists.txt` 触发了 CMake 重新配置 ⇒ **整棵树的 TU 重编**，这恰好让
  `math.cuh` 的改动**真正生效**（本树没有头依赖跟踪，改头不重编，见 §123）。
- **Muse verify 契约修复**：在 `target_verify_batch_impl` 的 `lm_head` linear 与 `argmax` 之间补
  `kCfg.apply_final_logit_policy(...)` —— 契约要求"每个 lm_head logits 产生点"都应用；对 qwen 是编译期
  no-op，**对 Muse（softcap 20 / multiplier 0.196）是必需**（否则 verifier 的 argmax 与 plain 不等价）。
- **环境坑（已修）**：`ccache: not found`（Error 127）——本树把 `ccache nvcc` 当 launcher，而 ccache 装在
  `/home/user/.local/bin`；由不同 shell 启动 make 时 PATH 不同。已在重建脚本里显式
  `export PATH="/home/user/.local/bin:$PATH"`。**启动构建必须带这一句。**
- **Spark 权重下载完成**：5 个分片全部到手（`ALL_CHUNKED_DOWNLOADS_FINISHED`，7.7 GB）⇒ 配合已就绪的
  S53（tied head 物化 + 嵌入别名 `model.embedding.weight`），**Spark 的转换侧已具备条件**；
  仅剩引擎侧 3 个 new_op（16Q/4KV@256 几何、逐头输出门、gated GELU MLP）。
- **训练**：18:08 起续跑，`step 1825/7200 loss=1.9348 lr=5.13e-04 steps/s=0.25`，已落 step_001800.pt。
- **明确待补（不许含糊）**：S51 的**极端负值回归用例**——E8 已证明现有测试证明不了它
  （`test_silu_mul.cpp` 的 gate 只到 ±12；criterion 绝对容差 2.0e-5 会掩盖 1e-37 量级的归零），
  新用例必须用**相对判据**并覆盖 −88.7 / −100 / −1000；待 tests 目标编译 + GPU 空闲时运行。
""" )
print("_TODO.md §126 已追加")
