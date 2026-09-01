# Qwen3.8 FlashNext 支持方案（PLE n-gram 表 SSD offload）

调研日期：2026-09-02。参考实现：llama.cpp qwen4exp（PR #27742）、
Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF（ds4-dfm-rs 配套格式）。

## 模型解剖

Qwen3.8-Flash-Next（qwen4exp 架构）：

| 部分 | 规模 | 说明 |
|---|---|---|
| MoE 主干 | ~125B（48 层，512 routed experts/层，top-10，~6B active） | 层型按 3× Gated DeltaNet + 1× QSA（稀疏注意力）交替，QSA 在 [3,7,11,…,47] |
| PLE n-gram 表 | 51.2B（320M 行 × 160 列 BF16 = 95.4 GiB） | 独立于主干的查找表，位于 layer 1（0-based）；128 逻辑分片、4 物理文件 |
| MTP head | 2.6-4B | 草稿头，配合 recurrent state rollback 使用 |
| 上下文 | 262,144 原生（YaRN 1M） | — |

Q5 混合量化后主干 77.5 GiB；PLE BF16 侧车 95.4 GiB（也有 FP8/Q8/IQ4 变体）。

## PLE 表行推导（已从 llama.cpp 源码逐行确认，qwen4exp.cpp:1028-1039）

```
ctx[0] = 当前 token；ctx[1..n-1] = 前驱（EOS 重置窗口，缺失前驱视为 EOS）
bigram  (head 0-7):   mixed = ctx0*m0 ^ ctx1*m1
trigram (head 8-15):  mixed = ctx0*m0 ^ ctx1*m1 ^ ctx2*m2
row[head] = mixed % per_head_vocab_size[head] + per_head_offset[head]
```

- 乘法在 uint64 上自然溢出（取低 64 位），m = [23703573157769, 20109073645365, 8052911324071]
- per_head_vocab_sizes / per_head_offsets 各 16 项（见 PLE manifest）
- 每 token 只 gather 16 行 × 160 列 = 2.5 KB（BF16）→ 工作集极小，SSD 随机读完全可行

## PLE 层结构（不是独立草稿器！）

PLE 输出**残差贡献**而非 logits：gather 16 行 → [2560, T] → key/value 线性变换 →
分组 RMSNorm → gate = sigmoid(sign(s)·sqrt(|s|))（s 为 key·query 点积）→
gated value + 深度可分离因果卷积（kernel×ngram_size 膨胀）→ 加回残差流。

⇒ 不能把 PLE 表单独当草稿模型用。独立草稿器走 llama.cpp ngram-mod 路线
（纯文本统计，零权重）——已列入 backlog。

## SSD runtime contract（ds4 参考实现，我们的目标规格）

1. PLE 表**永不驻留 GPU**（95 GiB 不可入 unified memory）；
2. CPU 侧推导行号（上面公式）；
3. **异步 SSD 预取**（16 worker 线程，双缓冲）；
4. **有界 pinned page cache**（参考值 512 MB，LRU）；
5. **UVA gather**（GPU 直接读已 pin 的 cache 页，省一次 H2D）。

## ninfer 差距盘点

| 能力 | 现状 | FlashNext 需求 |
|---|---|---|
| Gated DeltaNet | ✓ 已有（src/ops/linear_attention/gated_delta_net/） | 直接用 |
| MoE | ✓ 已有 sparse_moe + 35B-A3B target（prefill/decode/small_t 全套） | 路由规模 512 experts、expert offload（RAM）待加 |
| MTP 草稿 | ✓ 已有（mtp_impl / mtp_round） | 复用；加 recurrent state rollback |
| KV 磁盘层 | ✓ 已有（ColdPolicy::Disk） | 不冲突 |
| PLE 层 | ✗ 缺失 | hash + sidecar mmap + conv 融合 |
| QSA 稀疏注意力 | ✗ 缺失 | 块级稀疏 attention |
| per-layer token embd | ✗ 缺失 | GGUF 新概念，需要 reader 支持 |

## 实施路线

- **阶段 A（本次，独立可交付）：PLE sidecar 组件**
  `src/ops/ple/`：行推导 hash（host）+ 4 文件 mmap + 异步预取线程池 +
  有界 pinned LRU cache + UVA gather 内核。附带：
  - manifest 校验工具（对拍 llama.cpp 公式）
  - serve 选项 `--ple-dir/--ple-cache-mb/--ple-workers`
  - 纯 C++ 自测（不依赖模型，用合成表验证行号推导）
- **阶段 B：qwen4exp 主干**：PLE 层接入（hash→gather→gate→conv）、
  QSA 稀疏注意力、512-expert 路由 + expert offload（FreeToken 思路）、
  MTP rollback。工作量大，独立排期。
- **阶段 C：ngram-mod 草稿器**（文本统计，零权重）：接入投机框架，
  对重复文本（代码/JSON/模板）即时生效。

## 参考来源

- llama.cpp PR #27742（qwen4exp 初始实现）及 b10660/b10737 release
- Baekpica Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF（PLE sidecar 格式与 SSD contract）
- ds4-dfm-rs（DwarfStar 续作，DGX Spark 上的 603 tok/s 冷 prefill 实测）
- vLLM PR #54070（VLLM_PLE_DISK_OFFLOAD_DIR）
