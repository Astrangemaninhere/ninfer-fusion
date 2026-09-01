# FreeToken 调研：思路能否为我们所用（KV offload RAM/SSD）

调研对象：FlashML-org/FreeToken（MoE 推理引擎，目标把大规模 MoE 跑在边缘设备上：
DeepSeek-V4-Flash 284B、Qwen3.6-35B、GLM-5.2 753B 等）。

## FreeToken 的核心机制

1. **弹性内存管理器（Elastic Memory Manager）**：KV cache 与 expert slots 之间
   按需动态分割同一块 VRAM。上下文增长时 KV 占位变大，expert 缓存收缩；
   反之亦然。避免固定配额造成的碎片浪费。
2. **带宽自适应 CPU-GPU 协同执行**：根据 PCIe/内存带宽与 GPU 算力的比值
   决定哪些计算放 GPU、哪些放 CPU（q* 策略），最大化重叠。
3. **LRU expert 缓存**：hot expert 留在 GPU，冷 expert 卸载到系统内存，
   按访问频率淘汰。
4. **语义感知 KV 缓存（semantic anchor checkpoints）**：agentic 场景下
   上下文被反复编辑/回退时，只重算被编辑的片段，锚点之前的 KV 复用，
   避免全量重算。
5. **NVMe 启动加载**：expert 池从 NVMe 流式加载（7GB/s 量级），
   但**运行期 KV/expert 不落 SSD**。

## 逐条对照我们（ninfer）

| FreeToken 机制 | 我们的现状 | 可借鉴点 |
|---|---|---|
| 弹性内存管理 | KV 有分层表（BF16/INT8/FP8/NVFP4/E8Kv），按层静态配置 | **可借鉴**：KV 预算按请求动态调——长 ctx 请求自动降 KV 位宽、短 ctx 用高精度。已有分层机制，缺的是"运行期按压力切换"策略 |
| 带宽自适应协同 | 无 CPU 计算路径（纯 GPU） | 对 dense 模型意义不大（权重必须驻 GPU）；**对 FlashNext MoE 支持是必经之路**（expert 放 RAM，按带宽调度） |
| LRU expert 缓存 | 无 expert（dense）；35B-A3B 的 sparse_moe 全驻留 | FlashNext（512 experts/层）需要；复用 FreeToken 的 LRU + 预取思路 |
| 语义锚点 KV | 已有 #142/#143 共享前缀 + StateImage 快照往返 | ✓ 已覆盖同类问题；可补：编辑片段后仅重算 delta |
| SSD 使用 | **超出 FreeToken**：我们已有磁盘冷层（ColdPolicy::Disk，KV 页落盘） | 我们的磁盘层设计可以反哺 FlashNext 的 PLE 表（见 RESEARCH-FLASHNEXT.md） |
| KV 卸载 RAM | 已有冷池（CPU RAM 层，异步回填） | ✓ 与 FreeToken 的 offload 同思路 |

## 结论

- **KV 卸载到 RAM/SSD 这条线我们已自研完成**（冷池 + 磁盘层），FreeToken 只是印证了方向正确。
- 真正有价值的新借鉴有两个：
  1. **运行期 KV 预算弹性**（按请求上下文长度动态切换位宽档）——低成本改造，列入 backlog；
  2. **expert offload + 带宽自适应调度**——这是 FlashNext（125B MoE）能在 5090 32GB 上跑起来的前提，
     与 sparse_moe 现有算子结合，是 FlashNext 支持的第二阶段。
- SSD 方向我们比 FreeToken 更激进（他们不落盘 KV），磁盘层已验证，继续保留。
