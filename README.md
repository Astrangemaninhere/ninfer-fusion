# NInfer Fusion（增强补丁集）

基于 [Neroued/ninfer](https://github.com/Neroued/ninfer)（Apache-2.0）的
社区增强发行版：把多年积累的 KV 压缩、冷层、投机解码、长上下文能力
合并为单一可维护补丁集，目标是 RTX 5090 / sm_120a 上的最大单卡吞吐。

> 状态: 开发中 — 各功能分支已 rebase 到上游 master（da49c0d），
> 正在收尾 E8Kv 精度修复与残差平面稳定性。

## 与本项目（上游 ninfer）的差异

| 功能 | 开关 | 说明 |
|---|---|---|
| 分层位宽表 | `--kv-layer-storage` | 16 层逐层 BF16/INT8/NVFP4/E8Kv 混合 |
| NVFP4-tier KV | `--kv-dtype nvfp4` | E2M1 K + ISO3 V，原生 mxf4nvf4 QK |
| E8Kv 4-bit 整 KV | `--kv-layer-storage all:e8` | E8-lattice K + i4 V（H64 旋转对齐 g64） |
| 残差平面 | `--kv-residual-layers` | NVFP4 二级残差（K/V 双轨） |
| 熵编码冷池 | `--cold-policy window` | rANS slot 压缩冷页 |
| NVMe 冷层 | `--cold-policy disk` | 冷页 spill 到磁盘（文件槽） |
| DFlash2 草稿 | `--spec dflash2` | 双向注意力草稿后端 + 长度切换 |
| on-demand Graph | `--no-cuda-graph` 等 | 解码 ladder 按需扩展 |
| YaRN factor-4 | `--yarn` | 静态 4x 上下文扩展 |
| 自动前缀共享 | serve 默认 | system/developer 边界自动建共享前缀 |
| represented-scale 补偿 | NVFP4 内建 | E4M3 舍入误差吸收进 code（上游 nvfp4 优化） |

## 构建

### WSL2（当前主平台）

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build -j2
```

### Windows 原生（规划中）

见 `docs/windows-native.md`（MSVC + CUDA 工具链适配，待完成）。

## 一键部署（规划中）

- `install.sh` / `install.ps1`：检测驱动/CUDA、克隆、构建、模型目录约定
- 桌面快捷方式 + 系统服务模式（serve）

## 模型

自备 `.ninfer` artifact（如 lyf / Ostfralla 的 Qwen3.8-27B 系列）。
模型权重适用各自许可，本仓库不附带权重。

## 版权

Apache-2.0。第三方贡献与归属见 [THIRD-PARTY.md](THIRD-PARTY.md)：
E8 codec 血统（PR #35 → ninfer-4090 → ninfer-3090）、IsoQuant 表（nvfp4rtx 校准）。

## 验证

- ppl: `ninfer-perplexity --kv-layer-storage ...`（协议见 docs）
- 报告: profiles/perplexity/…
