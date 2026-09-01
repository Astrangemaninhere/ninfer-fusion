# Prefill 优化计划（借鉴 SGLang 方法论）

目标：把 prefill 吞吐从当前 ~2.0k tok/s（4096 ctx 实测）提升到接近
SGLang 的 ~10k tok/s 量级（长 context / 大 batch 场景）。

## SGLang 的 prefill 关键方法 → 我们的现状与差距

| SGLang 方法 | 我们的现状 | 差距 / 行动 |
|---|---|---|
| RadixAttention（树状前缀共享） | 已有 shared-prefix + #142 自动候选 | ✓ 无需重做 |
| Chunked prefill + 流水 | `--prefill-chunk` 默认 1024，ppl 固定 1024 | 试 2048/4096；ppl 加可调参数 |
| **CUDA graph 化 prefill**（固定 shape chunk 捕获） | **prefill 是否在 graph 内？decode 有 graph，prefill 疑似无** | **最大头：固定 chunk shape 后 graph 捕获，减少 launch 开销** |
| 手写 attention kernel（triton，按 head_dim/tile 调） | E8 走 i8 mma（实测 2083 tok/s）、NVFP4 走 mxf4nvf4（1949） | i8 路径反而更快；NVFP4 的 V 解码/scale 开销可优化 |
| 数据布局 / cp.async 流水 | 已有双缓冲 + cp_async | 试加深流水（多缓冲） |
| Tensor parallel | 单卡 | 不适用 |

## 瓶颈判断（待实测）

- **4096 ctx = 2.0k tok/s**：kernel launch / chunk 间同步主导（小 shape）
- **16k/32k ctx**：若吞吐显著高于 2.0k → compute 密集，SGLang 量级可达
  （大 context 下 kernel 效率提升）
- **大 batch**（多请求并发 prefill）：SGLang 的 10k+ 依赖 batch ——
  serve 的并发 prefill 是另一条路

## 实施步骤（按序）

1. **实测 16k/32k context prefill**（ninfer-allbuild 已编译）——确认瓶颈
2. **CUDA graph 化 prefill**：
   - 找 prefill 的 graph 支持（Engine 的 graph 路径是否含 prefill）
   - 固定 chunk shape（--prefill-chunk）→ 捕获 prefill graph
   - 预期：小 context 的 launch 开销消除
3. **chunk 大小调优**：1024 → 2048/4096 对比（chunk 内 kernel 效率 vs 内存）
4. **ppl 工具加 --prefill-chunk / graph 开关**（测量可调）
5. **NVFP4 prefill 的 V 解码开销优化**（对比 E8 i8 路径为什么更快）
6. **serve 并发 prefill 测量**（batch 场景，SGLang 量级的真正来源）

## 验证协议

- ninfer-perplexity：同一语料，context 4096/16384/32768，四组 KV 配置
- 记录 score rate（tok/s）与 ppl（精度不变性）
- serve：并发 2/4 请求的 prefill 吞吐
