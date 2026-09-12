import pathlib
p = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\_TODO.md")
old = p.read_text(encoding="utf-8", errors="replace")
block = """
## 2026-09-11 深夜续 · vLLM+dflash2 加载期内存的**实测结论**（这条线关闭）

### 测量方法（终于做到可归因）
- 用户级 systemd 硬上限：`systemd-run --user --unit=vllmstream -p MemoryMax=14G -p MemorySwapMax=0`
  （`--scope` 会随我的会话被杀，必须用瞬态 **service** 才独立存活）。
- 逐 2 秒记录：cgroup `memory.current` / `memory.peak` / 显存 / 当前阶段 → `dl/stream_trace.log`。
- 宿主侧看门狗（`_watchdog2.ps1`，阈值 10GB、2s 采样）→ `dl/watchdog2.log`，越线自动 `wsl --shutdown`。

### 实测结论（三层归因，全部有数）
1. **吃内存的是"加载期"**：`memGB 2.1→3.9→6.7→13.9→14.0`（6–15 秒内撞上限），
   而阶段始终是 `Loading ... checkpoint shards / Runai Model Streamer` ⇒ **不是 KV、不是 MM、不是 JIT 收尾**。
2. **不是"文件被整份持有"**：`runai_streamer` 确实逐张量读（`843/2387 @420it/s` ✓），
   但主机需求**照样到 14GB** ⇒ 那是 **vLLM 对 2387 个张量的自身处理（NVFP4 重打包/元数据 + `enable_jit_warmup`）**，
   属加载流程固有成本，**换加载器搬不掉** ✗。
   （`fastsafetensors` 在 WSL 里退化为 nogds 普通拷贝：`/dev/nvidia-fs0 not found` ⇒ 也搬不掉 ✗。）
3. **算术封死**：加载期≈14GB + VM 开销 + 页缓存 > 21GB VM ⇒ **21GB 的 VM 装不下 16.7GiB 的权重** ✗。

### 用户两条建议的执行与实测
- **KV 量化 FP8**：已做（`--kv-cache-dtype fp8`，与模型 `kv_cache_quant_algo=FP8` 对齐 ✓）⇒ 对这笔账无影响 ✓（KV 在 512/seqs=1 下只有几十 MB）。
- **"别落主机/只进显存"**：加载器两条路都试了（fastsafetensors ✗、runai_streamer ✓真流式）⇒ **主机需求不变** ✗。

### 硬上限的效果（用户最关心）
`MemoryMax` + 看门狗**有效**：宿主最低到 7.7GB 时看门狗自动关 VM，
**宿主全程未崩** ✓（对比之前 7 次 VM 崩机把宿主内存拉到 2.5GB ✗）。

### 结论
**在这台机器（宿主 31.4GB）上，vLLM 0.29 + 27B NVFP4 + dflash2 只能在原生 Windows 跑**
（Windows 侧不受 VM 上限约束，R2 的 1.87% 即出自那里 ✓）；WSL 路线**因加载期主机需求 > VM 容量而不可行** ✗。
若要走 Windows：需构建支持 dflash2 的 Windows 轮子（数小时 ✗）。
"""
p.write_text(old + block, encoding="utf-8")
print("_TODO.md now", len(p.read_text(encoding='utf-8', errors='replace').splitlines()), "lines")
