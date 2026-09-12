# 本趟编译的落地集（provenance，2026-09-10 16:33）

判读用：测量出来的每个数字都归属于这一组文件哈希。

| 补丁 | 文件 | md5 | 说明 |
|---|---|---|---|
| 补丁 A | `src/targets/qwen3_6/impl/runtime/program_impl.h` | `0bad2de4fd` | DFlash2 decode ingress 漏填 state_source/destination_slots（M_patchA_*.diff 已应用） |
| E2/S44 | `src/serve/request_log.cpp` | `b2e73b23e2` | 接受率计数标签 spec_drafted/spec_accepted/spec_accept_rate |
| E2/S44 | `apps/cli/main.cpp` | `5465c64dcc` | CLI accepted-by-pos 直方图 |
| E4/S46 | `src/serve/kv_cold_policy.h` | `2b7d33ec0b` | 冷窗 F3b：空闲窗不进 EWMA + 置信度门 |
| E4/S46 | `src/serve/kv_auto_relayout.cpp` | `e4be3ed628` | touched 以强制重编（头文件依赖未跟踪） |
| S45d/E3 | `src/ops/launcher/gqa_attention_prefill.cu` | `50e265e620` | 128 几何逐臂 nvfp4 守卫（256 下逐字节等价） |
| S45c/E3 | `src/serve/serve_options.cpp` | `b0a0a6428e` | --spec usage 文本补 dflash2|auto |
| S45c/E3 | `apps/cli/options.cpp` | `a13315e62c` | 同上（CLI 侧） |
| S48/E6 | `src/targets/qwen3_6/impl/runtime/dflash_impl.h` | `02cb3bc9b5` | dspark verify 位置表 k→k+1 |
| S50/E7 | `src/targets/qwen3_6/impl/runtime/logical_kv_store.h` | `69f2281c30` | KV 覆盖下界语义 ensure_mapped_to_tokens |
| S50/E7 | `src/targets/qwen3_6/impl/runtime/program.h` | `7849a2737c` | 声明改名 ensure_sequence_kv_mapped |
| S50/E7 | `src/targets/qwen3_6/impl/runtime/program_impl.h` | `0bad2de4fd` | 14 处调用点改名（参数逐字节不变） |
| S50/E7 | `tests/targets/qwen3_6/test_context_store.cpp` | `aca0866d02` | 测试侧同步改名 |

## 刻意不在这趟（避免污染补丁 A 归因）

- **S51/E8** `ops::silu`/`sigmoid` 极端负值归零修复（改所有 nvfp4 线性层数值）——下一趟，且需同时加极端负值回归用例。
- **E9/S52** dflash2 可配置草稿宽度 K 的最小切片（改 dflash2 行为）——单独一轮。
- **E7 回归测试** `E7_s50_regression_sketch.diff`（store 级页边界用例）——需要能编 tests 的窗口。
- **E3 的 i8 平面步长 + S36 恢复**（256 下逐字节等价）、**E1 补丁 B**（`attention_valid` 契约）。

