# 显存占用参考表（RTX 5090D 32GB 实测）

> 实测方法：ninfer-serve 加载各配置，nvidia-smi 读取（4096 context，WSL2，CUDA 13.3）。
> 模型: Qwen3.8-27B NVFP4（基础 21.5GB artifact）。**启动前对照此表，避免 OOM。**

## 各配置加载后显存

| 配置 | 显存 | 说明 |
|---|---|---|
| 空闲（无进程） | 0.05 GB | — |
| 基础（kv int8） | **20.5 GB** | 模型权重 19.0 GB + KV/上下文 |
| 全 NVFP4 KV（层表 all:nvfp4） | **20.6 GB** | KV 更小，4096 ctx 下差异 <0.1 GB |
| E8 混合（默认 10L E8 + 6L NVFP4） | **20.6 GB** | 同上（KV 差异在长上下文才显著） |
| + vision（多模态） | **20.9 GB** | vision 塔 +0.4 GB |
| + MTP3（草稿 3） | **21.4 GB** | MTP 草稿头 +0.9 GB |
| DFlash2 模型（nvfp4-dflash2） | **20.6 GB** | 基础 +0.1 GB（草稿头较小） |
| DSpark 模型 | — | 未注册 target（`nvfp4-dspark` 身份不被支持） |

## 参考余量（32GB 卡）

| 组合 | 总占用 | 余量 |
|---|---|---|
| 基础 + E8 混合 KV | 20.6 GB | ~11.4 GB |
| 基础 + vision + MTP3 | 21.4 GB | ~10.6 GB |
| 基础 + vision + MTP3 + E8 KV | ~21.5 GB | ~10.5 GB |
| 基础 + 262k context（KV 撑满） | 需单独测 | — |

## 长上下文 KV 增长估算（每 token/head）

| KV 方案 | 字节/head/token |
|---|---|
| int8 | 512 B |
| NVFP4（E2M1 K + ISO3 V） | 288 B |
| E8Kv（E8 K + i4 V） | ≈260 B |

4M token×头 的 KV：int8 每 100k context ≈ 512 B × 100k × 8 头 ≈ 0.4 GB/层组；
E8Kv 约减半。实际容量受 `--max-context` 与页池控制，serve 启动日志会打印
`KV capacity` 与 `free-after-startup`，以日志为准。

## 启动失败排查

- `--kv-dtype nvfp4` 在 serve 解析不支持 → 用 `--kv-dtype int8 --kv-layer-storage all:nvfp4`
- DSpark artifact（`nvfp4-dspark`）未注册 → 等待 target 支持
