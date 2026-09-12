# 新项目路线图（含模型转换调研）

来源: Neroued/ninfer #142（自动 system 前缀共享）、#143（NVMe 冷层）
反馈者: steve8697（agent 工作负载用户，愿意帮忙实测）

## 1. 自动 system/developer 前缀共享（已实现，已推上游 PR #152）

已实现并修复 bot 发现的 3 个问题（Responses 应用时机、marker 槽、leading run 结尾）。
待改进（issue 作者偏好）:
- [ ] tools+system 合并 frontier（Qwen 模板 tools 渲染在 system 前时）
- [ ] 256-token floor
- [ ] 邀请 steve8697 用 16k sibling probe 实测

## 2. NVMe 冷层（已实现，待对齐反馈）

已实现 `--cold-policy disk`。待改进:
- [ ] shared_stable_prefix 完整往返冷页（StateImage 而非仅 KV 槽）
- [ ] 窄改动模块化（各功能独立开关，默认关）

## 3. 模型转换能力（2026-09-01 调研）

### 已支持
- **HF / vLLM 格式（safetensors）→ .ninfer**：`tools/convert/qwen3_8_27b/convert.py`
  （BF16）与 `convert_nvfp4.py`（NVFP4，需量化源目录）。Qwen3.8-27B 微调变体
  权重名不变 → 可直接转换（待实测一个变体验证）。

### 待开发
- **GGUF（llama.cpp）→ .ninfer**：
  - GGUF v3 解析器已验证（markmonger/Qwen3.6-40B q8_0-v1.gguf =
    1290 tensors，元数据完整解析）
  - 反量化（q8_0：f32 scale + int8 block；q4_k_m 等 → bf16）→
    中间表示 → 走 convert.py 量化
  - 需要 tensor 布局重排（GGUF → safetensors 布局）
- **40B 变体（Qwen3.6-vl-40B 系，含 Deckard 等微调）**：
  - GGUF 元数据实测架构 = **qwen35 族**（与 ninfer 35B-A3B 同族）：
    block_count=97、embedding=5120、Q 头 24、KV 头 4、
    key/value_length=256、ffn=17408、full_attention_interval=4、
    rope base 1e7、context 262144、**Qwen3.6-vl（多模态）**
  - 注册新 target `qwen3_6_40b`：对比 35B-A3B 的 config.h 参数差异
    （层数/ffn/rope base），复用其模板
  - GGUF 输入走反量化链路；HF safetensors 输入（如有）直接 convert

### 工具
- `tools/gui/convert_gui.py`：滑块简约风转换 GUI（量化档位 / KV 精度 / 上下文 / 批量宽度）
- `tools/gui/rag_gui.py`：滑块简约风 RAG 检索 GUI（top-k / 相似度下限 / 片段扩展）

## 4. 待测量数据（编译完成后）
- MTP3 接受率 × KV 配置（默认 10L E8+6L NVFP4 / all-E8 / all-NVFP4 / all-I8）
- DFlash2 接受率（qwen3_8_27b_nvfp4_dflash2.ninfer）
- 多模态（--vision）serve 实测

## 5. 上游观点
- 上游作者: 想自己加功能就 fork，想提 PR 就老老实实提，一次别太多。
- 上游贡献流程: 修 bug → 提 issue 说明 → 微量 PR。
