# 导入器与转换工具链

## 1. 通用前门

| 工具 | 是什么 | 出处 |
|---|---|---|
| `import_model.py` | 「把本地模型变成 `.ninfer`」的唯一前门。注册的转换器是**封闭、字节钉死的契约**，各接受什么由前门统一收口 | `tools/convert/import_model.py`（**1141 行**，实测 `wc -l`） |
| `convert_runner.py` | 导入后的全自动转换编排 | `tools/convert/convert_runner.py`（403 行）；测试 `tools/convert/test_convert_runner.py`（575 行） |
| `artifact_diff.py` | 两个 `.ninfer` 的**逐对象**比较；这是导入产物的验收闸——「这个 build 是否与参考一致」 | `tools/convert/artifact_diff.py`（224 行），按退出码判 |
| `compress_probe.py` | 测 `.ninfer` artifact 的**无损**压缩余量 | `tools/convert/compress_probe.py`（988 行） |
| `_COMPRESS_VRAM.md` | 压缩/VRAM 备忘 | `tools/convert/_COMPRESS_VRAM.md`（树内已存在） |

⚠ **上一轮清单给出的行数已过期**：它记 `import_model.py` 为 1097 行，实测 **1141** 行；
记 `gguf_extract.py` 为 628 行，实测一致。以本册的行数（同批实测）为准。

## 2. GGUF 族（一份能力拆成四个文件）

| 文件 | 职责 | 行数 |
|---|---|---|
| `tools/convert/gguf_extract.py` | GGUF → bf16 safetensors（qwen3.5/3.6/3.8 文本族） | 628 |
| `tools/convert/gguf_kquant.py` | ggml K-quant 块格式的**尺寸与反量化** | 820 |
| `tools/convert/gguf_hadamard.py` | Prism/Bonsai 的**旋转契约** | 622 |
| `tools/convert/gguf_fold_back.py` | 旋转过的 GGUF 的「折回权重」路线 | 745 |
| `tools/convert/gguf_fold_route.py` | 「谁施加这个变换」的**唯一决策点** | 641 |
| `tools/convert/gguf_names.py` | GGUF 张量名 → 本树名字的映射 | 323 |

行数全部为本批实测，命令：`find tools/convert -name '*.py' | while read f; do printf '%s %s\n' "$(wc -l < "$f")" "$f"; done`。

## 3. 各族转换器（逐族目录，行数实测）

| 族目录 | 关键文件与行数 |
|---|---|
| `gemma4_31b/` | `convert.py` 438、`inventory.py` 311、`recipe.py` 260（**本册新增的族**） |
| `muse_glimmer_30b/` | `convert.py` 555、`config_pins.py` 210；夹具 `qwen_chat_template.jinja`、`qwen_preprocessor_config.json`、`qwen_video_preprocessor_config.json` |
| `qwen3_5_9b/` | `convert.py` 480、`inventory.py` 269、`check_bindings.py` 373 |
| `qwen3_6/common/` | `recipe.py` 452、`frontend_policy.py` 448、`conversion.py` 259、`inventory.py` 174、`draft_head.py` 168、`official_resources.py` 132 |
| `qwen3_6_27b/` | `verify.py` 648、`recipe_nvfp4.py` 527、`text_core.py` 480、`convert_nvfp4.py` 446、`convert.py` 418、`recipe.py` 327、`verify_nvfp4.py` 321、`inventory_nvfp4.py` 313、`inventory.py` 293、`draft_head.py` 76 |
| `qwen3_6_35b_a3b/` | `convert.py` 636、`recipe.py` 563、`inventory.py` 217、`draft_head.py` 67 |
| `qwen3_8_27b/` | `convert_modelopt.py` **1842**、`recipe_nvfp4.py` 773、`convert_nvfp4.py` 483、`mtp.py` 475、`inventory_nvfp4.py` 472、`convert.py` 301、`fp8_embedding.py` 202、`patch_dflash2.py` 150、`verify_patch.py` 96、`inventory.py` 77；另有 `finalize_dflash2.ps1` |
| `qwen3_8_flash_next/` | `recipe.py` 582、`convert.py` 479、`inventory.py` 364、`source.py` 343、`splice_mtp.py` 190 |
| `qwen4_exp/` | `convert.py` 218 |

### ModelOpt 与 NVFP4 路线

| 项 | 是什么 | 出处 |
|---|---|---|
| ModelOpt 读取器 | Qwen3.8-27B 的 ModelOpt NVFP4 HF checkpoint 的逻辑源读取器 | `tools/convert/dequant/modelopt.py`（533 行） |
| 文档 | `docs/maintainer/modelopt-nvfp4-import.md`（159 行，树内已存在） | 同上 |
| fp8 嵌入头 | 输出/嵌入头的 FP8 量化 | `tools/convert/qwen3_8_27b/fp8_embedding.py`（202 行）；核侧 `src/targets/qwen3_8_flash_next/impl/load/quantize_output_head.cu` |
| LoRA 融合 | LoRA adapter → ninfer artifact 的离线权重合并路线 | `tools/convert/lora/lora_merge.py`（166 行）；说明 `tools/convert/lora/_LORA.md` |
| PLE/ngram SSD sidecar | CPU 侧建 PLE 的 SSD sidecar | `tools/convert/ple_sidecar_build.py`（181 行）；carrier 头 `src/product/ple_sidecar_carrier.h` |
| KV ISO 参考 | ISO4/ISO3 KV 量化的 Python 参考，对齐 `src/ops/kv/iso_codec.h` | `tools/convert/kv_iso_ref.py`（62 行） |

## 4. 架构适配工具包（`tools/archkit/`）

| 工具 | 是什么 | 行数 |
|---|---|---|
| `adapt.py` | 从 spec → `config.h` 的适配 | 546 |
| `adapt_all.py` | 一键全自动适配管线（自述是「验收标准执行入口, v1」） | 497 |
| `archkit_target.py` | 目标骨架生成 | 383 |
| `gen_target.py` / `gen_variant.py` / `gen_full_target.py` / `gen_stubs_v2.py` | 骨架/变体/stub 生成 | 209 / 104 / 365 / 110 |
| `check_geometry.py` / `check_params.py` | **会红的门**（几何覆盖、参数完备），不是提示 | 147 / 179 |
| `kv_budget_mirror.py`（+ `kv_budget_probe.cpp` 242、`kv_budget_regen.sh` 179） | 位预算求解的**离线镜像**，与服务端交叉核对 | 510 |
| `kv_bit_budget.py` | 位预算的离线求值 | 154 |
| `kv_tier_matrix.py` | KV 档位 × 层的标定流水线；结果文档 `docs/maintainer/kv-strategy-matrix.md`（150 行） | 444 |
| `kv_calibrate.py` | 扫描 Rk4v4 覆盖率、输出质量-压缩曲线 | 101 |
| `kv_auto_allocate.py` | 用户给的 K/V 档位 → 逐层存储方案 | 132 |
| `ft_tiers.py` | 从 `ft_stats` 观测生成逐层 KV 方案（服务端消费者 `src/serve/kv_auto_relayout.cpp`） | 67 |
| `topology.py` | 拓扑自动判别 `dense \| moe \| moe_ngram` | 122 |
| `ple_gather_check.py`（+ `ple_gather_test.py` 172） | PLE n-gram 落表核对（manifest audit vs HF 算法 + mmap gather + golden emit） | 417 |
| `flashnext_convert.py`（+ `flashnext_bindings.py` 379） | FlashNext 的契约驱动张量映射器（含自测） | 669 |
| `gguf_spec.py` / `gguf_tensors.py` | GGUF 规格与张量读取 | 284 / 168 |
| `longtest_57k.py` | 57k needle 长上下文验收 | 151 |
| `probe_formats.sh` / `test_probe_provenance.sh` | 格式探测与**出处校验** | 532 / 294 |
| 工具链 | `cuda128_sm70_toolchain.cmake`（75 行）、`cuda128_legacy_toolchain.cmake`（166 行） | — |
| QPN 端口件 | `tools/archkit/qpn_port/`（`device_profile.cu`、`placement_planner.py`、`qpn_prepack_proto.py`） | — |
| 审计文档 | `_ARCHKIT.md` 130、`_AUTOADAPT.md` 168、`_GPU_MATRIX.md` 213、`_MUSE_SEMANTICS.md` 116、`_SM_COUNT_AUDIT.md` 132、`gemma_engine_plan.md` 98 | — |

行数全部为本批实测，命令：`find tools/archkit -maxdepth 1 -type f | while read f; do printf '%s %s\n' "$(wc -l < "$f")" "$f"; done`。

## 5. 测试面

`tests/convert/` 下有各族 Python 单测（`common/` 5 个、`qwen3_6_27b/` 7 个、`qwen3_6_35b_a3b/` 4 个、
`qwen3_8_27b/` 4 个、`qwen4_exp/`、`muse_glimmer_30b/`），另有 `tests/convert/test_gguf_kquant.py`（575 行）
与 `tools/convert/test_convert_runner.py`。`tests/` 下 Python 文件共 **44** 个（实测按后缀计数，
见 `verification.md`）。

## 6. 其它工具面（`tools/`）

`tools/gui/`（转换 GUI / 服务 GUI / RAG GUI / 模型导入 / KV 档位 / GPU 兼容 / i18n）、
`tools/smoke/`（服务契约烟测）、`tools/bench/`（TTFT 战役、rk8v4 质量基准）、
`tools/roofline/serve_roofline.py`、`tools/kv_relayout_test.cpp`、`tools/kv_rowscale_sidecar.py`、
`tools/reference/qwen3_8_flash_next/`（参考实现与 oracle）、`tools/e8_verify/`（死守卫 / 尺度约定 /
UB float cast 守卫）、`tools/tp2/`、`tools/mg3/`、`tools/mg4/`、`tools/mgpu2/`（TP2 探针族）、
`tools/test_kv/`、`tools/calib/analyze_kv.py`、`tools/perplexity/prepare_corpus.py`、
`tools/hbm_bandwidth_probe.cu`、`tools/check_link_health.sh`、`tools/ninfer_serve/openai_responses.py`。
顶层 GUI 入口：`ninfer-gui.py`、`gui_page.html`（仓库根）。
Windows 侧打包：`tools/package/build_exe.ps1` + `tools/package/ninfer-gui.spec`（见
[`unfinished.md`](unfinished.md) 第 P6 条：它打的是 **GUI 外壳**，不含 CUDA 引擎）。
