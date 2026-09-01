# Third-party notices（合并补丁项目版权清单）

本项目（下称“NInfer 增强补丁集”）是 **Neroued/ninfer** 的修改与扩展发行版，
随 Apache License 2.0 分发。除上游 ninfer 之外，本项目包含以下第三方贡献，
按 Apache 2.0 §4 保留其版权与归属声明：

## 1. 上游 NInfer（基础代码）

- 来源: https://github.com/Neroued/ninfer
- 许可: Apache License 2.0
- 说明: 本项目的全部基础代码（engine / ops / serve / targets）派生自该仓库。
  修改点见 README 的“与本项目差异”一节。

## 2. E8 lattice / 240-root 压缩 KV codec

- 来源: https://github.com/Neroued/ninfer/pull/35（作者 danielfparkernz，
  分支 feat/compressed-kv-blackwell-5090）
- 原始血统（PR #35 自带 attribution）:
  - UDPSendToFailed/ninfer-4090（Ada / sm_89 fork，Hadamard 旋转 K/V、
    packed int4 V、E8 Conway-Sloane lattice / 240-root codec 数学的源头）
  - 其上游 Don-Chad/ninfer-3090
- 涉及文件:
  - `src/ops/kernel/e8_lattice.cuh`（Hadamard 旋转 + E8 最近格点投影）
  - `src/ops/kernel/e8_root_codec.cuh`（240-root 两级编码）
  - `gqa_kv_hadamard64`（64 维 Sylvester 旋转，位于
    `src/ops/kernel/gqa_attention_kv_quant.cuh`）
- 说明: 本项目的 E8Kv 模式（--kv-dtype e8kv / --kv-layer-storage 中的 e8 层）
  在该实现基础上修改（H64 旋转域与 g64 scale 域对齐、scale 语义修正、
  与分层位宽表/残差平面集成）。

## 3. IsoQuant SO(4) 旋转表（NVFP4 路径）

- 来源: nvfp4rtx 离线校准产物 `isoquant_rot.npy`
  （烘焙为 `src/ops/kernel/gqa_isoquant_rot.cuh` 的 constant memory 表，
  [64][4][4] fp32）
- 说明: 表数据来自 nvfp4rtx 校准流程；如 nvfp4rtx 有单独许可要求，
  以其为准。本项目仅使用其校准数值，未修改表本身。

## 4. 模型权重

- 本项目不含任何模型权重。运行需用户自备已注册的 `.ninfer` artifact
  （例如 lyf/Ostfralla 等社区量化），模型权重适用各自的许可
  （Qwen 模型通常为 Apache 2.0 或 Qwen License）。

## 5. 本项目自研部分（无第三方版权负担）

- PR1 分层位宽表（per-layer KV storage）
- PR2 / PR6 熵编码冷池（rANS slot、NVFP4-tier KV）
- PR3 DFlash2 草稿后端（双向注意力、grouped conv、selector）
- PR4 on-demand CUDA Graph 扩展
- PR5 静态 YaRN factor-4
- NVMe 冷层（ColdPolicy::Disk）
- 自动 system 前缀共享（serve 层）
- represented-scale 补偿等 NVFP4 精度优化
- Windows 原生适配层（如有）

## 许可合规提示

- 保留 Apache 2.0 LICENSE 全文，不删除任何源文件头部的版权/归属注释。
- 分发二进制时随附本文件。
- E8 相关文件保留 PR #35 / ninfer-4090 / ninfer-3090 的 attribution 注释。
