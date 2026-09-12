# _GPU_MATRIX.md — 多型号 GPU 支持: 能力分发内核路线 (非门禁)

## 定调 (2026-09-03)

不是用门禁把卡挡在外面 —— 是同一 artifact 按 (sm, weights profile) 在运行时
**分发不同的 GEMM 内核路线**, 每个档位都真正能跑:

| sm | 代表卡 | 张量核 | 内核路线 | 状态 |
|----|--------|--------|----------|------|
| 100/120a | B100/RTX50 | fp4 tcgen05/TMA | nvfp4 W4A4 原生 | 现状(已编译) |
| 90 | H100 | fp8 | fp8 TC 路线 / W8A16 | 待 QPN8 移植 |
| 89 | RTX40 | fp8 | fp8 TC / W8A16 + int8 | 待 QPN8 移植 |
| 86/80 | RTX30/A100 | int8 | groupwise-int(已发布 profile) + W4A16 | profile 已有 |
| 75 | RTX20 | int8 | groupwise-int | profile 已有 |
| 70 | V100 (CUDA12 旧链) | 仅 fp16 mma | **QPN2 W4A16**: 直接跑已发布 NVFP4/FP8 混合权重, 4bit 码内联展开喂 fp16 mma, 无需重量化/无需 fp16 副本 | 待移植 |

## QPN 参考 (Apache-2.0, 已归档 data/v100-skinny)

fork_patches/marlin.py:
- `_qpn_prepack`: 片段序重排(字节等价置换, 不改数值) [tile N/32][group K/16]
- gemm_qpn SIMT 小 M; M>=17 让 marlin; QPN2 = M4..8 几何赢家, 表驱动
  (VLLM_SKINNY_QPN / VLLM_SKINNY_QPN2 开关)
- 移植 = CUDA 重写 prepack + SIMT 核; 跑分参考 qpn8_m_sweep/qpn_race csv

## 已落地 (代码在树)

- CMake 门禁放开: 架构列表 + >= sm_75 校验 (实测 70 拒 / 86;120a 过)
- build_arch.sh 矩阵构建脚本
- 运行期 plan 报错按档位给指引 (不是拦截, 是提示该走哪条路线/哪个 dist)

## 待办 (GPU 回归批)

1. sm 分发点: linear 包装层按 device.sm() + profile 选路线
   (QPN2/QPN8/int8-tc/w4a4-tma), 内核文件按 __CUDA_ARCH__ 守卫共存
2. QPN prepack + SIMT 核 CUDA 移植 (V100 旧链 CUDA12 单独构建)
3. groupwise-int 在 3090/4090 真机回归 (195-203 tok/s 社区基线)
4. fp8 路线 (Ada/Hopper) — W8A16 或独立 fp8 profile

## 增补 2 (2026-09-03): v100-skinny vs 1Cat 差距归因 + 1Cat 多卡路线

### 为什么有人报 v100-skinny 不如 1Cat (v100-skinny 自己文档实锤)
1. NVFP4 半边 M9-16 中段是让步带: QPN8(FP8) 有 MT=2 双 tile 分发盖 M9-16,
   NVFP4 半边中段仍跑第一代 QPN 核 (README kernel-gap 注); M>=17 才让 marlin。
   -> 移植时 NVFP4 中段必须补 MT=2, 不能照抄 v1。
2. FP16-KV 政策: SM70 无高效 fp8 attention, 用 FP8 KV 会强制 scalar paged
   attention (+4.82 ms/round), v1.1 改 FP16 KV -> 显存翻倍, 4x16GB 才舒服。
3. 1Cat 是全家桶: KVarN 注意力/marlin/split-KV verify/前缀缓存/贪心 MTP/
   校准 int4 lm_head/图 PINNING; v100-skinny 只换 linear 核, attention/verify/
   调度仍 stock。与 NInfer 同台 A/B 其实打平 (219.1 vs 214.7 tok/s, 同 README)。
4. 实测优化点 (移植清单): decode-partition pinning 值 0.7-2.6ms/round;
   greedy MTP +10-25 接受点; GDN speculative-state 契约 (21 syncs/70 copies 消除)。

### 1Cat 多卡并行 = vLLM 张量并行 (TP)
- v100-skinny 即 1Cat-vLLM 1.2.2 fork: 权重按 rank 切分 (18.6GB 单 shard 多卡读),
  KV 随注意力头分片, 每 rank 独立 decode/verify 管线;
- 跨卡通信: 1Cat custom all-reduce (fork_patches/custom_all_reduce.py, 抄自 1Cat,
  带 gpu_p2p_access_check; PCIe 无 NVLink 走 custom AR);
- 实测: TP=2 通信约 0.65%; TP=2 长上下文 batch1 = 带宽受限 (第二个内存墙),
  DFLASH_TOKENS=7 at TP>1 (T=15 仅单点); 4xV100 chain-MTP 366 tok/s (Alfinaa)。

### 我们的落地路线 (引擎多设备 = 新大件, GPU 批)
1. 单进程 TP 骨架: 权重切分 (linear N 维按 rank) + AR 原语 (PCIe custom AR 参考
   v100-skinny/1Cat) + KV 头分片 + 每 rank 独立投机管线;
2. 调度/图捕获跨卡化 (现有单卡 DecodeGraph 扩展);
3. QPN 移植时 NVFP4 中段 MT=2 一起做 (堵住让步带);
4. 真机逐项验收 (本机 5090D 单卡回归 + V100 实机/社区对照)。
