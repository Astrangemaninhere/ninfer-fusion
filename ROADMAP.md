# 新项目路线图（从上游 issue 反馈整理）

来源: Neroued/ninfer #142（自动 system 前缀共享）、#143（NVMe 冷层）
反馈者: steve8697（agent 工作负载用户，愿意帮忙实测）

## 1. 自动 system/developer 前缀共享（已实现，待改进）

已实现（`--no-auto-system-shared-prefix` 默认开）:
- last-content implicit 写保留
- 额外一个 system/developer frontier 自动候选（DefaultAutomatic evidence）
- serve 层实现，复用引擎现有 shared-prefix 机制

issue 作者确认方向 + 三个非阻塞偏好（改进点）:
- [ ] **tools+system 合并 frontier**: 若 Qwen 模板中 tools 渲染在 system 块之前，
      自动候选应落在合并的 tools+system 前缀之后（agent 客户端真实共享头）。
      实现上需要探测模板渲染顺序（qwen3.6/3.8 模板的 tools 位置）。
- [ ] **256-token floor**: 共享头 < 256 tokens 时跳过额外写（不占 catalog 槽）。
      对齐引擎 shared-prefix value model 的现有下限。
- [ ] 保留双写（已满足）。
- [ ] 邀请 steve8697 用 16k sibling probe 重测（无 breakpoint → 期望 shared_stable_prefix）。

## 2. NVMe 冷层（磁盘层，已实现，待对齐反馈）

已实现（`--cold-policy disk --cold-disk-path`）:
- parked 冷页 spill 到每层文件（slot*stride 偏移）
- 设备槽池复用 + 异步预取 + 恢复

issue 作者反馈（改进点）:
- [ ] **共享前缀必须能完整往返冷页**: catalogued shared_stable_prefix 若不能
      穿过磁盘层，sibling reuse（#142）与冷层不组合。要求 exact identity +
      full StateImage（含 GDN recurrent bundle）存进文件，而不只是压缩 KV 槽。
      当前实现存 compressed slots（KV 页）；需评估把 checkpoint/StateImage
      作为磁盘单元，或保证 KV 页往返后 identity 不变。
- [ ] **窄改动原则**: 新项目应把各功能做成独立可审核的模块/开关
      （默认关闭、可独立开启），避免"一次一个大杂烩"（上游关闭 PR #149 的
      直接原因）。对应到新项目的模块划分与文档。
- [ ] 冷层定位: parked checkpoint only（不提升活跃解码并发），默认关。
- [ ] 参考: vLLM/LMCache、TRT-LLM host offload、NVIDIA ICMSP 的
      Device → Host → NVMe → evict 形状。

## 3. 其他上游观点

- 上游作者: "想自己加功能就 fork，想提 PR 就老老实实提，一次别太多"。
  新项目作为独立补丁集发行（fork 形态），不再向上游堆大 PR。
- 上游贡献流程: 修 bug → 提 issue 说明 → 微量 PR。
