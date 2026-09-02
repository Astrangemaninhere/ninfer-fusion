# 外部参考调研：1Cat-vLLM 与 sglang-qwen38-dual-3080-patch

调研日期：2026-09-02。

## 1. 1Cat-vLLM（1CatAI/1Cat-vLLM，803 stars）

V100/SM70 专项 vLLM fork：4× V100 16GB 跑 Qwen3.8-27B-NVFP4 + DFlash2 ≈260 tok/s；
FlashNext-NVFP4 + MTP4 ≈138 tok/s；DeepSeek-V4-Flash 也上了（8×V100，65-73 tok/s）。

对我们（5090/SM120）有参考价值的点：

| 技术 | 内容 | 对我们的价值 |
|---|---|---|
| **GQA-packed wide QK/PV**（PR #286） | 6 个 GQA head 打包成更宽的 Tensor-Core GEMM，Attention 有用算力 17.9→60.8 TFLOP/s（3.4x） | **高**：我们 24Q/4KV、head_dim 256，prefill 的 QK/PV 投影可以按同样思路打包，减少小 GEMM 与 launch 次数——正好是 prefill 优化计划的一环 |
| **adaptive lookup q16**（PR #366） | DFlash2 之上叠加 lookup 草稿，重复上下文场景 316 tok/s（3.16 ms TPOT，opt-in） | **高**：与我们 ngram 草稿器（PLE/ngram-mod）路线一致；他们用查表增强 DFlash2 而非替代 |
| NVFP4 on V100 | V100 无 FP8 TC，走反量化/软件路径 | 低（5090 有硬件路径） |
| E4M3/E5M2 KV | 主 KV 用 FP8 | 已有对应层（FP8_E4M3FN），无新东西 |

## 2. sglang-qwen38-dual-3080-patch（kk-pcl，双 3080 20GB，TP=2）

SGLang 0.5.19 小补丁，三项改动：

| 改动 | 细节 | 对我们的价值 |
|---|---|---|
| **Per-KV-Head FP8 静态校准** | 主模型 16 个全注意力层按 KV head 校准 E4M3 scale（静态校准文件）；DFlash2 用逐层标量 FP8 | **中**：我们的 KV 分层粒度已经是 block-scale（NVFP4 g64 / E8Kv g64），更细。可借鉴的是**静态校准流程**——自动选每层 KV 位宽，替代手动 --kv-layer-storage |
| **Draft KV 降精度** | DFlash2 草稿 KV 用 FP8，腾出显存给主模型 KV 池（25.5 万 token 池） | **高，直接可做**：我们 DFlash2 的 draft cache 是 BF16 全上下文（spec_decision 里有 dflash2_context_bytes VRAM 压力检查）。Draft KV 降为 FP8/int8 省显存与带宽，草稿模型对 KV 精度不敏感 |
| **GDN State 槽位保护** | max-mamba-cache-size=20：State 槽位不能省太狠，否则 Attention KV 还在但 State 被淘汰 → 整段 Reprefill | **高，需自查**：Qwen3.8 是 3× GDN + 1× 稀疏混合。我们的 pressure planner 有 StateReplicaResidency/state_slots 管理，需确认 state 淘汰是否也可能触发整段 reprefill（我们称 rewrite/rebuild，有 rebuild_work.h） |
| SM90 以下禁用 Hopper symmetric-memory gather | 避免 CUDA Graph 阶段 SIGFPE | 低（SM120 无关） |

## 3. 结论 / 行动项

1. **DFlash2 Draft KV 降精度（FP8/int8）**：把 draft cache 从 BF16 降到 FP8。
   预期：draft cache 显存减半，长上下文 + DFlash2 的 VRAM 压力缓解；接受率损失小。
2. **GQA-packed QK/PV 打包**：prefill 优化的一环（与 sglang 调研的 kernel 优化合并执行）。
3. **State 淘汰策略自查**：确认 GDN state 淘汰不会触发整段 reprefill（对 pressure planner 的
   state 保留策略做 review；必要时加 state 槽位最低保留水位，类似他们的 max-mamba-cache-size）。
4. **静态 KV 校准流程**（backlog）：用校准集自动生成每层 KV 位宽/scale 配置，替代手调。
5. lookup 草稿（backlog）：与 PLE/ngram 草稿器计划合并。

## 来源

- https://github.com/1CatAI/1Cat-vLLM（README + PR #286/#366/#422）
- https://github.com/kk-pcl/sglang-qwen38-dual-3080-patch（README）
