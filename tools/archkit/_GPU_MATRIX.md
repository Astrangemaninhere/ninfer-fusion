# _GPU_MATRIX.md — 多型号 GPU 支持: 能力分发内核路线 (非门禁)

## 定调 (2026-09-03)

不是用门禁把卡挡在外面 —— 是同一 artifact 按 (sm, weights profile) 在运行时
**分发不同的 GEMM 内核路线**, 每个档位都真正能跑:

| sm (含特性后缀) | 代表卡 | 张量核 | 内核路线 | 状态 |
|----|--------|--------|----------|------|
| 100a/120a/121a | B100/RTX50 | fp4 tcgen05/TMA | nvfp4 W4A4 原生 | 现状(已编译) |
| 90 | H100 | fp8 | fp8 TC 路线 / W8A16 | 待 QPN8 移植 |
| 89 | RTX40 | fp8 | fp8 TC / W8A16 + int8 | 待 QPN8 移植 |
| 86/80 | RTX30/A100 | int8 | groupwise-int(已发布 profile) + W4A16 | profile 已有 |
| 75 | RTX20 | int8 | groupwise-int | profile 已有 |
| 70 | V100 (CUDA12 旧链) | 仅 fp16 mma | **QPN2 W4A16**: 直接跑已发布 NVFP4/FP8 混合权重, 4bit 码内联展开喂 fp16 mma, 无需重量化/无需 fp16 副本 | 待移植 |
| 50/52/53 | GTX 750/750Ti/9xx (Maxwell) | **无** | 待定: 只有纯 SIMT 路线可走, **本树当前没有任何一条能跑的路线** (见增补 3) | 仅构建配置 (未探针) |
| 60/61/62 | GTX 10xx / P100 / TX2 (Pascal) | **无** | 同上 (sm_61 多一个 `__dp4a`, 但 `__dp4a` 只在 QPN 的中间带上, 整 TU 编不过) | 仅构建配置 (未探针) |

### 这张表的键是 (sm 号 + 特性后缀), 不是 sm 号

2026-09-13 实测 (RTX 5090 D, 同一台机器, 同一份源码): 同为 arch 号 `120`,
`120a` 构建报六种 KV 格式全部 SUPPORTED, `120` 构建拒 fp8/nvfp4。原因: nvcc 只为
'a' (arch-specific) 目标定义 `__CUDA_ARCH_FEAT_SM120_ALL`, fp4/TMA 内核只在那个宏
下存在; `device.sm()` (= major*10+minor) 看不到这个区别。

因此:
- 表里第一行的键写 `100a/120a/121a` —— 只有带 `a` 的构建才有那一行的能力;
  不带 `a` 的同号构建属于"groupwise-int 档", 不是 fp4 档。
- "状态"列说的是**构建配置**, 不是"这块卡能跑什么"。它是路线规划, 没有被引擎
  探针逐条验证过; 任何"能跑/不能跑"的结论都要引擎自己报 (能力探针),
  不能从本表或 sm 号推出来 (见 tools/gui/gpu_compat.py 里被去掉的
  `engine_ready = (sm == 120)`: 那正是"用号代替证据"的缺陷)。

## QPN 参考 (Apache-2.0, 已归档 data/v100-skinny)

fork_patches/marlin.py:
- `_qpn_prepack`: 片段序重排(字节等价置换, 不改数值) [tile N/32][group K/16]
- gemm_qpn SIMT 小 M; M>=17 让 marlin; QPN2 = M4..8 几何赢家, 表驱动
  (VLLM_SKINNY_QPN / VLLM_SKINNY_QPN2 开关)
- 移植 = CUDA 重写 prepack + SIMT 核; 跑分参考 qpn8_m_sweep/qpn_race csv

## 已落地 (代码在树)

- CMake 架构校验: 语法检查在 project() 前, 能力检查在 project() 后 ——
  用编译器自己的 `nvcc --list-gpu-arch` 问"这个工具链还能为哪些 compute
  capability 生成代码" (CUDA 13 起 < 75 的离线 codegen 已被移除, 实测 70 拒),
  语法/能力都对时放过 (实测 86;120a 过)。带 `a` 的 arch 会写进
  `__CUDA_ARCH_FEAT_SM<N>_ALL` 的说明; 不带 `a` 且 >= 100 的 arch 给出明确
  WARNING (该构建没有 fp4/TMA 内核), 不是靠数字猜。
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

## 增补 3 (2026-09-18): pre-75 档位 (Maxwell sm_50/52/53 + Pascal sm_60/61/62)

主人指令: "gtx960接着写, 这玩意要能支持可以说全系列的n卡能跑点ai的都能受惠了" +
"70前面的所有和70一起走旧的"。即把已有 sm_70 的"旧工具链档"扩成整个 pre-75 块。

### 3.1 选择器改在哪 (不是那个装饰常量)

- `tools/archkit/build_arch.sh` 里 `LEGACY_ARCHS=(50 52 53 60 61 62 70)`
  **就是选择器**, `TOOLCHAIN_FILE_OF` 由它生成 (加一个 rung 就自动要求有对应表项,
  有表项而不在集合里会 exit 5)。原来那行
  `NINFER_ARCH_NEEDING_LEGACY_TOOLKIT=70` 保留在**原行号**上做历史标记
  (src/core/format_probe.h:136 按名字引用它), 它不再是选择器。
- 新增 `--rung legacy` / `--list-rungs`, 一次调用构建整块, 每个 arch 仍然各自
  `build-<arch>` / `dist-<arch>`。
- **真实选择器的两个缺陷, 2026-09-18 实测后修掉**:
  1. `cd "$(dirname "$0")/.."` 差一级: dirname 是 `tools/archkit`, 一个 `..` 落到
     `<repo>/tools`, 那里**没有 CMakeLists.txt**, 所以这个脚本从来没成功 configure
     过任何 arch —— 包括它当初为 70 扩的那条路 (`build-sm70/CMakeCache.txt` 里
     `CMAKE_HOME_DIRECTORY=/home/user/ninfer-fusion` 说明那次是手工在 repo 根跑的;
     build-sm70 只有 2 个 .o, 也没有 dist-70)。改成 `readlink -f` + 两级 `..`,
     外加 CMakeLists.txt 哨兵检查 (错了就 exit 6, 不是让 cmake 去报错)。
  2. 分档判断只看 `NINFER_CUDA_TOOLKIT[_<arch>]` 环境变量, 于是文档里那条
     `NINFER_CUDA_TOOLKIT_70=... build_arch.sh 70` 的用法会让整块 legacy 被判成
     **两个档**而拒绝 (70 解析成那个 nvcc, 52 解析成空串); 反过来 `70 86` 不设变量时
     它又放行了一个真跨两个工具链的列表。现在档位键 = 工具链文件 (有就优先) →
     编译器 → "cmake 自己找", `52 86` 实测 exit 3 并列出两个档。
- `CMakeLists.txt` 架构校验里那条历史地板 (`_arch_num LESS 70`) 现在在
  `CMAKE_TOOLCHAIN_FILE` 已给出时**不适用**: 它本来只是"探针答不上来"的替身,
  而带工具链文件时编译器会被 project() 的 TryCompile 真正问一遍。实测后果:
  不带这个修改, `build_arch.sh 50` 用 legacy 工具链文件会死在
  "arch 50 is below the floor ... the compiler was not found"。
- `tools/archkit/cuda128_legacy_toolchain.cmake` (新): 7 个 pre-75 档共用的**一个**
  工具链文件, arch 由 `-DCMAKE_CUDA_ARCHITECTURES` 给 (不再硬编码 70)。
  `cuda128_sm70_toolchain.cmake` 原样留档 (它是 sm_70 报告和 AGENTS.md 引用的实测记录),
  两者是同一套宿主修补。
- **宿主修补本身有个实测缺陷, 已修**: jammy (glibc 2.35) 头文件集**必须只喂 CUDA 编译**
  (`-Xcompiler=-isystem,...`: crt/math_functions.h 的 cospi/sinpi/rsqrt 与 glibc 2.43
  的 noexcept 冲突是真的, 不修则连 CMake 编译器 ID 探测都死在
  `CMakeCUDACompilerId.cu`), **不能喂宿主 C++ 编译**: jammy 那套是残缺集
  (823 个 .h, 有 stdlib.h/math.h, **没有 sys/cdefs.h**), 放到宿主编译前会把
  2.35 的 features.h 和 2.43 的 stdlib.h 配在一起, 每个纯 C++ TU 死在
  `/usr/include/stdlib.h:752: error: expected initializer before '__COLD'`
  (实测: 第一次全树构建 31 个 FAILED 块全是这一个原因)。现在 CUDA 侧带, 宿主侧不带,
  `NINFER_LEGACY_JAMMY_HEADERS_HOST=ON` 只作为逃生门存在 (打开就把 __COLD 故障带回来)。

### 3.2 构建事实 (就是"构建配置", 不是支持)

用 CUDA 12.8 (`/mnt/g/cuda12/tk`, nvcc V12.8.61) + `-ccbin g++-13`:

| arch | configure (385 个 TU) | 逐 TU 编译 | 说明 |
|----|----|----|----|
| 52 | rc=0 (脚本路径与手写路径各一次) | **384/385 跑完, 24 个编不过** (360 过) | 其中 **21 个与 arch 无关** (在飞的 iso3->iso4/e8->rk4v4 改名: `KvVCodec::Iso3` / `DType::SO4E` 不存在; 用 CUDA 13.3 编原生 120a 同样 5 处错误, 已证), **3 个与本档位相关** (下表) |
| 61 | rc=0 | **384/385, 同样的 24 个失败** | 与 sm_52 **逐行相同** (同一批 TU, 同一句首个错误); 61 只多一个 `__dp4a`, 而它所在的那条带整 TU 编不过, 所以 61 相对 52 没有净收益 |
| 60 / 62 | rc=0 (手写路径, 只到 configure) | 未跑 | |
| 50 | rc=0 | 未跑 | 修掉 CMake 地板那条之后才过 |
| 53 | 未测 | 未测 | 本档位唯一连 configure 都没跑过的 rung |
| 70 | rc=0 (新文件与旧 sm70 文件各一次) | 未跑全树 | 旧文件 sha256 未变 |

没跑的那一个 TU 是 `src/ops/launcher/gqa_attention_decode_i8.cu` (树里最大的 tensor-core decode TU):
第一次被 kill 的 sweep 写行写了一半丢了它, 补跑时本机 WSL 挂了 (`Wsl/Service/E_UNEXPECTED`),
所以它的结论**不在本表里**。按结构它应该能编过 (它只用 `ops/common/mma.cuh` 里带守卫的 helper,
不像 prefill 那个 TU 是直接踩 `cvt.bf16x2.f32`)。补跑: `stage/sweep_last_tu2.sh`。

sm_52 的三个档位相关失败 (各带出处):

| TU | 第一个错误 | 性质 |
|----|----|----|
| `src/ops/launcher/gqa_attention_prefill.cu` | ptxas: `Feature '.bf16x2' requires .target sm_75 or higher` / `cvt.bf16x2.f32 requires .target sm_80 or higher` | **不是 pre-75 专有**: 同一 TU 编 sm_70 rc=255 (同样 7129 条), sm_75 rc=255 (4753 条)。attention prefill 在 sm_80 以下根本编不出来 |
| `src/ops/linear/qpn/qpn_host.cu` | `qpn_kernels.cuh(31): error: name must be a namespace name` | **就是本档位的 GEMM 墙**: `nvcuda` 在 sm_70 以下不存在, 所以 `using namespace nvcuda;` 先倒 |
| `src/targets/qwen3_8_flash_next/impl/qsa_indexer_kernels.cu` | `fatal error: cub/device/device_topk.cuh: No such file or directory` | **工具链层面**: 这个头只有 CUDA 13.x 的 CCCL 有 (12.8 的 `include/cub/device/` 无 topk), 所以它进不了任何 legacy dist |

即: 361/385 个 TU 能为 sm_52 编出来, 203 个宿主 TU 全过 (那条 `__COLD` 故障已修)。
失败里只有 1 个是"矩阵想扩的那条 pre-Volta 内核"的墙, 另外两个比本档位更宽。

### 3.3 内核边界: pre-Volta 到底能跑什么 (全部为 file:line 证据)

单指令可用性 (nvcc/ptxas 直接问, 不是设备探针; 见 recon/isaprobe/):

| 指令/特性 | sm_50/52 | sm_61/62 | sm_70 | 出处 |
|----|----|----|----|----|
| `mma.sync.m8n8k4.f16` (QPN 的 tensor 通道) | 拒 (`Feature 'mma' requires .target sm_70 or higher`) | 拒 | 收 | ops/common/mma.cuh:191 |
| `mma.sync.m16n8k16.f16` | 拒 | 拒 | 拒 (要 sm_80) | ops/common/mma.cuh:128 |
| `ldmatrix` | 拒 (要 sm_75) | 拒 | 拒 | ops/common/mma.cuh:100 |
| `cp.async` | 拒 (要 sm_80) | 拒 | 拒 | ops/common/memory.cuh:65 |
| `__dp4a` | **未定义** | 收 | 收 | ops/linear/qpn/qpn_kernels.cuh:581 |
| `${__half}` 算术 (`__hmul2`) | 收 | 收 | 收 | — (sm_50 也编得过) |
| `nvcuda::wmma` 16x16x16 | 命名空间不存在 | 命名空间不存在 | 收 | qpn_kernels.cuh:404 |
| `__grid_constant__` | 拒 ("only allowed for compute_70 or later") | 拒 | 收 | ops/kvarn/attention.cu:292,341 |

由此得到的内核结论:

- **attention 侧没有一条 pre-Volta 路线。** 所有 attention 内核 (decode 6 个变体 +
  prefill 5 个 + `src/ops/softmax_attention/dense/causal_cache/*`) 每个都至少有一处
  mma 和一处 ldmatrix (见 recon/r5_attention_census.txt); 全树搜 `simt|scalar` 在
  attention 里零命中。attention 是模型服务的必需品, 所以**目前"最小可服务路径"在
  Maxwell/Pascal 上不成立, 与 GEMM 侧无关** —— 这是新内核, 不是加个守卫。
- **QPN 家族 (矩阵给 70 定的唯一路线) 在 sm_70 以下整个 TU 编不过**:
  `qpn_kernels.cuh:31 using namespace nvcuda;` 与 `:404` 的 `wmma::fragment`
  (nvcc: "name must be a namespace name"), 加上 `:696`/`:895` 未加守卫的
  `mma.m8n8k4` asm、`:581` 的 `__dp4a`。该文件里 `__CUDA_ARCH__` 命中数为 **0**。
  它的纯 SIMT 通道 `skinny_nvfp4_qpn_simt` (qpn_kernels.cuh:985) 本身只做 half2
  反量化 + FFMA, 是唯一 pre-Volta 可行的 QPN 通道 (M<=3, 正好是 decode 带),
  但被同一个 TU 里的另外三条带拖死。
- **SIMT GEMM 里只有 q5/w8 能真跑**: q5/w8 用 `pipe_copy`/`pipe_commit`/`pipe_wait`
  (cuda_pipeline 内建, sm_80 以下退化为同步拷贝, 不 trap);
  q4/q6 用未加守卫的 `cp_async<16,Cache::cg>` (ops/linear/q4/q4_rowsplit_gemm_simt.cuh:70,72,104
  与 q6 同名行), 在 `__CUDA_ARCH__ < 800` 上**编得过但一启动就 trap**
  (ops/common/memory.cuh:73-76) —— 注意这条对 **sm_70 同样成立**。
- 纯 FFMA 的一批 (rmsnorm/l2norm/layer_norm/rope/embed_gather/sampling/kv_cache append)
  对 tensor-core/async 引用为 0, 任何档位都能跑。

### 3.4 还没做的 (必须明说)

- **dist-52 / dist-61 还不存在**: 全树的 link + device-link + staging 没跑完, 现在有的只是
  逐 TU 编译证据 (上面那张表)。要出货得先 `tools/archkit/build_arch.sh --rung legacy`
  跑一轮 (它会自带 staging), 再补 §3.3 里 attention 那条路。
- **本机没有 Maxwell/Pascal 卡**, 真机探针做不了 (见下一条), 所以 3.2/3.3 全是
  **构建配置 + 汇编器事实**, 按本文件自己的规矩这**不是支持声明**。
- 真要验证需要: 一台 sm_52 (GTX 960) 与一台 sm_61 (GTX 1060) 的机器, 用
  `dist-52`/`dist-61` 跑引擎的能力探针 + 一个真实模型的 decode/prefill 冒烟,
  并**先补 attention 的 SIMT 路线**, 否则探针只会报"该路线不在本构建里"。
- `tools/archkit/probe_formats.sh` / `src/core/format_probe.h` 是另一条线在做,
  未改; 本增补里的 `arch_sim.h` 那条路**明确不算证据**
  (模拟器跑的是 sm_120 cubin, 结论不构成任何 Maxwell/Pascal 支持)。
