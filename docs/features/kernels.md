# 内核族与后端

## 1. 规模（实测）

`src/ops/` 下 **19 个**子目录，逐目录文件数（`find "$d" -type f | wc -l`）：

| 目录 | 文件数 | 目录 | 文件数 |
|---|---|---|---|
| `linear/` | 128 | `launcher/` | 95 |
| `kernel/` | 72 | `softmax_attention/` | 52 |
| `wrapper/` | 45 | `gdn_input_proj/` | 36 |
| `linear_add/` | 29 | `attn_input_proj/` | 28 |
| `linear_swiglu/` | 26 | `kv_cache/` | 21 |
| `linear_attention/` | 16 | `common/` | 14 |
| `kvarn/` | 11 | `sparse_moe/` | 10 |
| `linear_pair/` | 8 | `gdn_gating_proj/` | 5 |
| `kv/` | 5 | `ple/` | 5 |
| `generic/` | 2 | — | — |

合计 **608** 个文件（上表相加）。公开头在 `include/ninfer/ops/`，**63 个**头文件
（实测 `ls include/ninfer/ops/`）。

## 2. 以 KV 档位为索引的核族

同一份注意力能力被拆成「每个档位一组核」，三处各一套：

### 预填（prompt）核 —— `src/ops/softmax_attention/dense/causal_cache/`

| 档位 | 文件 |
|---|---|
| bf16 | `prompt_bf16.cuh` |
| fp8 | `prompt_fp8.cu` / `.cuh` |
| int8 | `prompt_i8.cuh` |
| **hq** | `prompt_hq.cu` / `.cuh` |
| **k8v4** | `prompt_k8v4.cu` / `.cuh` |
| **nvfp4** | `prompt_nvfp4.cu` / `.cuh`，另有非 RDC 变体 `prompt_nvfp4_non_rdc.cu` + `prompt_nvfp4_non_rdc_launch.h` |
| **rk2v4e8** | `prompt_rk2v4e8.cu` / `.cuh` |
| 公共 | `prompt.cu`、`prompt_common.cuh`、`geometry.cuh`、`launch.h`、`causal_softmax_attention.cpp` |

### 小 T decode 核 —— 同目录

`small_t.cu` / `.cuh`、`small_t_bf16.cuh`、`small_t_fp8.cu` / `.cuh`、`small_t_i8.cuh`、
`small_t_hq.cu` / `.cuh`、`small_t_hq_h16.cu`、`small_t_hq_h24.cu`、`small_t_hq_tc_launch.h`、
`small_t_k8v4.cu` / `.cuh`、`small_t_nvfp4.cu` / `.cuh`、`small_t_rk2v4e8.cu` / `.cuh`。

### KV append 与 codec —— `src/ops/kv_cache/`

- `append/`（子目录）承载 append 核族。
- codec 头：`fp8_e4m3_row_codec.cuh`、`int8_g64_codec.cuh`、`nvfp4_group16_codec.cuh`、
  `rk2v4e8_codec.cuh`、`hq_e8_rice_codec.cuh`、`hadamard_d256.cuh`、`d256_profile.h`。

### e8 lattice 与 ISO 参考

`src/ops/kv/`：`e8_lattice_plane_codec.cuh`（码字是 E8 ±1/4 的 8 维格点，见 `kv-compression.md` 第 1 节）、
`iso_codec.h`（Python 参考 `tools/convert/kv_iso_ref.py` 对齐的就是它），以及三个自带测试
`e8_width_codec_test.cpp`、`e8_width_contract_test.cpp`、`iso_codec_test.cpp`。

### KVarn（独立 KV 注意力族）

`src/ops/kvarn/` 11 个文件：`attention.cu`、`codec.cu`、`config.cuh`、`decode.cu` / `.cuh`、
`decode_kernel.cuh`、`hadamard.cuh`、`materialized_prefill.cuh`、`sinkhorn.cuh`、`store.cuh`、
`streaming_prefill.cuh`；头 `include/ninfer/ops/kvarn.h`、`kvarn_attention.h`；测试 `tests/ops/test_kvarn.cpp`。

### 选中块注意力

`include/ninfer/ops/selected_block_attention.h`（3249 B）；
实现 `src/ops/softmax_attention/selected_block/selected_block_attention.cu` + `prefill.cuh`。

## 3. 无 tensor core 的回退路径（这是非 NVIDIA 需求的落点）

| 组件 | 大小 / 行数 | 出处 |
|---|---|---|
| SIMT FFMA 注意力核 | 106707 B | `src/ops/kernel/gqa_attention_simt_ffma.cuh` |
| 该核的 env 门 | `NINFER_ATTENTION_SIMT_FFMA`；未探测时的回退由 `NINFER_ATTENTION_SIMT_FFMA_DEFAULT_ON`（`__CUDA_ARCH_LIST__ < 700`）决定 | 读取点 `src/ops/kernel/gqa_attention_simt_ffma.cuh:298`；说明文本 `src/core/device_capabilities.h:176-177` |
| 通用 rowsplit 回退 | — | `src/ops/generic/rowsplit_generic.cu` / `.h`；env `NINFER_GENERIC_ROWDEC` 读取点 `src/ops/generic/rowsplit_generic.cu:377`；测试 `tests/ops/generic/test_rowsplit_generic_layout.cpp` |
| per-format tensor-core 底线表 | 按格式一行 | `src/core/arch_caps.h`（2366 行）；`--capability-report` 打印的就是它（`apps/cli/options.cpp:207-215`） |

⚠ **这一族的证据强度要写清**：回退核**已落地**（有 e2e 测试与探针），但**没有任何厂商的端到端验证**。
`src/core/arch_caps.h:1739-1741` 自述写这张表的机器只有一张 NVIDIA 卡、完全没有 ROCm 工具链
（没有 `hipcc`、没有 `hipify`、没有 `/opt/rocm`），所以「这个引擎能不能在 gfx906/908/90a/942/1100/1201
上跑」**在这里未被回答，也无法在这里回答**。

## 4. 路由与设备能力

| 组件 | 行数 | 出处 |
|---|---|---|
| 中央 GEMM 路由表 `select_route(sm, format, shape)` | `src/core/kernel_route.h` 983 行；声明 `:191,215,229,866,878,931,936` | 7 条机制路由 × 3 种结局，表驱动自 `arch_caps.h` |
| 设备能力（两道门取代 `sm != 120` 硬门） | `src/core/device_capabilities.h` 721 行 | — |
| 设备探测 | `src/core/device_probe.cu` 719 行 | — |
| 运行期格式探测 | `src/core/format_probe.h` 1587 行 | opt-in 且默认关（该文件 `:1185` 自述） |
| SM 数硬编码审计 | `src/core/device_sm_count.h` 76 行 | 审计文档 `tools/archkit/_SM_COUNT_AUDIT.md`（132 行） |
| 声明式能力表 | `src/targets/declared_capabilities.h` 274 行 | — |

## 5. 架构模拟器（**测试专用**，两个键两个轴）

两个 CMake 选项，**默认都是 OFF**，且默认本身就是产品声明：

| 选项 | 默认 | 作用域与理由 | 出处 |
|---|---|---|---|
| `NINFER_ENABLE_ARCH_SIM` | **OFF** | 把测试专用架构模拟器建进引擎；选项文本自述「a shipping binary cannot be talked into simulating a card it is not by an environment variable alone」 | `CMakeLists.txt:253-254` |
| `NINFER_ENABLE_VENDOR_SIM` | **OFF** | **只门控测试二进制**，因为 `src/` 里没有任何 TU include `vendor_sim.h`（已测量）；选项文本自述「Two axes, two keys」 | `CMakeLists.txt:274-275`；测量陈述 `:261-273` |

`CMakeLists.txt:250-252` 还给出这个选项**此前只存在于注释里**的实测：`-DNINFER_ENABLE_ARCH_SIM=ON`
什么也没做，`NINFER_ARCH_SIM_ENABLED` 只在一个目标上被定义。

相关的还有 `NINFER_ARCH_WARN`（**真的环境变量**：默认下未列出的 compute capability 完全不输出，
设了才把同样的告警发到 stderr，`src/core/arch_caps.h:1718`）与 `src/core/announce_once.h`（136 行，
一次性告警门）。

## 6. 多设备 / 跨 rank（**host-only，无调用路径**）

| 组件 | 行数 | 状态 |
|---|---|---|
| `src/core/shard_plan.h` | 564 | 只有算术；`CHANGELOG.md:33` 自述「host-only and has no call path」并列出所有缺失件（rank plumbing、collectives、peer access、逐 rank 权重加载器、召回记录里的 rank 字段） |
| `src/core/shard_rank_axis.h` | 650 | 同上 |
| `src/core/tp_transport.h` | 1382 | 移植件；donor 声明 `:12` |
| `src/core/and_reduce.h` | 452 | 移植件 |
| `src/core/decode_graph_peer.h` | 363 | 跨 rank 图捕获；`:108,206-207,237-238` 自述因为 gfx906 有文档化的 hang，跨 rank 自旋在捕获里被**当已知 hang 拒绝**，不是风格偏好 |
| `src/core/n_dim_slice.h` | 297 | N 维切片 |
| `src/core/virtual_device.h` | 671 | 虚拟设备/rank |
| 契约文档 | 475 行 | `docs/maintainer/multi-device-and-shard-plan.md` |

`src/core/arch_caps.h:2065-2066` 自述这套线是**惰性的**：`allreduce` 在 `src/CMakeLists.txt` 里 0 命中，
而该文件的源列表是**显式**的（无 GLOB）。

## 7. 其它内核面

| 族 | 位置 |
|---|---|
| GDN（gated delta net，线性注意力） | `src/ops/linear_attention/gated_delta_net/`（`recurrent`、`chunked/*`）；状态 `src/core/linear_attention_state.h`（93 行）；重放记录 `src/core/gdn_replay_records.h`（61 行）；文档 `docs/maintainer/replayssm-gdn.md`（642 行） |
| PLE（Per-Layer Embedding） | `src/ops/ple/`：`ple_layout.{cpp,h}`、`ple_stage.h`、`ple_table.{cu,h}`；env `NINFER_PLE_STATS` 读取点 `src/ops/ple/ple_stage.h:146` |
| QPN（量化点积网络） | `src/ops/linear/qpn/`；IP 门 `NINFER_HAVE_QPN`（`src/CMakeLists.txt` 9 处）；工具链 `tools/archkit/cuda128_sm70_toolchain.cmake` |
| sparse MoE | `src/ops/sparse_moe/`（10 文件）；头 `include/ninfer/ops/sparse_moe.h` |
| Prism fold（把旋转折进权重） | `src/ops/linear/prism_fold/`（`prism_fold.cuh`、`.h`、`prism_fold_launch.cu`、`.h`）；转换侧 `tools/convert/gguf_hadamard.py` 等 |
| DSpark Markov 头 | 核 `src/ops/kernel/dspark_markov_argmax.cuh`；头 `include/ninfer/ops/dspark_markov_argmax.h`；测试 `tests/ops/test_dspark_markov_argmax.cpp` |

## 8. 内核测试面（实测存在的文件）

`tests/ops/` 下与本册相关的：`test_kvarn.cpp`、`test_hq_codec.cu`、`test_hq_retrieval.cu`、
`test_e8_root_codec.cu`、`test_kv_cache_append.cpp`、`test_kv_cache_append_geometry_pin.cpp`、
`test_dflash2_ddtree.cpp`、`test_dflash2_ddtree_beam.cpp`、`test_dspark_markov_argmax.cpp`、
`test_mtp_pack.cpp`、`test_mtp_round.cpp`、`test_speculative_round.cpp`、
`test_cold_i8.cpp`、`test_cold_i8_symbol_pin.cpp`、`test_gqa_decode_split_exact.cu`、
`test_gqa_split_contract.cu`、`tests/ops/gqa_split_geometry_probe.cu`（无 `test_` 前缀）、
`test_gated_delta_net*.cpp`、`test_gdn_replay_fold.cpp`、`test_entropy_cold_requant.cpp`、
`tests/ops/softmax_attention/`、`tests/ops/generic/`、`tests/ops/linear*/`（子目录实测存在）。
