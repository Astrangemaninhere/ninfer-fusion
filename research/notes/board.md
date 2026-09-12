# 协作看板 (唯一状态源)

| ID | 状态 | 负责人 | 产物 | done 判据 | 验证者 / 命令 |
|---|---|---|---|---|---|
| S1 | DOING | A | `src/ops/kernel/gqa_isoquant_row_scale.cuh` (越界→1.0) | 重建后 Muse headdbg 无 NaN | C: 读 diff 并核对 Muse/qwen 层号范围 |
| S2 | DOING | A | `src/ops/kernel/gqa_attention_prefill_i8.cuh` (旋转先于尺度) | e8 32K 掉针改善 | B: **PASS** (`_collab/B_s2_verify.md`) — 独立复算(全 token/页, 98,304 组) post/pre 中位 1.478/1.506 (A: 1.42/1.54), clamp 0.87%/0.71% (A: 1.03%/0.55%); `wsl.exe -e python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/B_s2_verify_run.py` |
| S3 | TODO | C | `src/ops/kernel/gqa_attention_prefill_i8.cuh` page 内核 (Muse e8: 缺 Hadamard + /127) | 未改；需 Muse+e8 单测后再动 | A: 提供 Muse e8 冒烟命令 |
| S4 | DOING | B | `eval_ddtree.py --teacher` 口径 | `--teacher` 数字可复现且 OFF 路径字节不变 | A: 跑一次并 diff |
| S4a | DONE | B | `eval_ddtree.py --beam`（引擎式 tie-break + path-wise hit@L + GATE 行） | OFF 路径与 `_eval_ddtree_v1.py.bak` diff 零删除行 + py_compile | A: 运行级 diff 因宿主 RAM 紧张(3.7GB) BLOCKED, 结构级已验（`diff _eval_ddtree_v1.py.bak eval_ddtree.py` 无 `<` 行）；随 S4 GPU/内存窗口补跑 |
| S5 | DOING | A | 训练 6000 步 | 到 6000 后 `eval_ddtree --teacher` | B: 独立跑 eval |
| S6 | DONE | C | 三闸门于 step_000100+000200 双跑通过（tuned 副本已删，磁盘净） | patch/verify/round-trip 全 EXIT=0, 73/73 BIT-EXACT | A: 已核对报告数字 |
| S7 | TODO | A | 引擎重建 (-j2) | `apps/ninfer` mtime 更新 | C: 重建完成后跑 `_df2_ab.sh`，`NINFER_BIN=/home/user/ninfer-fusion/build/apps/ninfer`（`~/ninfer/build` 是旧路径勿用） |
| S8 | DONE | C | `_df2_ab.sh`（usage: `bash _df2_ab.sh [A=<artifact>] [B=<artifact>]`，B 默认 `$B_ARTIFACT`；训练存活自动 ABORT exit 3） | `wsl.exe -e bash -c "bash -n /mnt/c/Users/User/Documents/ziqinzhang/_df2_ab.sh && echo SYNTAX_OK"` → SYNTAX_OK | A: 待 S7 重建后实跑 A/B |
| S9 | DONE | C | `_collab/C_artifact_manifest.md`（30 秒清单：三闸门命令+期望字符串+失败日志行+A/B 用法+mtp 误标坑；附：dry-run 对缺层 ckpt 报 `CKPT ERROR` exit 2，多余键静默忽略） | 文末自检原样可跑：bash -n → SYNTAX_OK；Select-String 关键字 Count=7(≥6) | A: Select-String 抽查 3 个 gate 字符串 |
| S10 | DONE | B | `_collab/B_teacher_ceiling.md`（教师天花板：rag 0.1808 / qa 0.3582 / code 0.4742 / TOTAL 0.2521，705,740 位置，含 top4/top16） | 三类占比+总占比可复现 | A: `wsl.exe -e python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/B_teacher_ceiling_run.py`（与 `_data_mix.py` 吻合，差=每文件 1 位置） |
| S13 | DONE | B | `_collab/B_gapfix_verify.md`（adapt.py 缺口报告修复的对抗验证，7 个合成 config + pre-fix sed 副本回归） | 8/8 PASS、0 假阳/假阴：`layer_types=["conv","conv","full_attention"]`→`layer_kinds(conv x2)`；`[]+ssm_cfg`→state_space；MiniCPM5 dense grep=0；pre-fix vs post-fix manifest diff 为空 | A: 报告内单行命令；**观察 O1**: 有 new_op 时 config.h 仍无条件生成且 conv 层被计入 `gdn_layers()`（绕过"不假装成功"的隐患，建议 new_op 时跳过工件） |
| S12 | DONE | A | `ninfer-fusion-repo/tools/archkit/kv_bit_budget.py` + `_collab/A_kvbit_cold.md`（opt-in 冷层成本模型 `--cold-pages N`/`--cold-bytes B`：cold=0 热位/页 9536B/penalty 0.25，精确 cap DP 维；e8 仍先打包保 0-7 验证窗；证据 `_collab/A_kvbit_golden_*.txt`+`A_kvbit_orig.py.bak`） | 无冷旗标路径与改前**逐字节一致**（3 组 golden diff 全空）；冷样本 16L/4bit penalty 3.64→1.89 | B/C: `wsl.exe -e bash -c "cd /mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit && python3 kv_bit_budget.py --layers 16 --bits 4 --cold-bytes 76288 && python3 kv_bit_budget.py --layers 8 --bits 4.5"`（第 2 条输出须与 `_collab/A_kvbit_golden_l8.txt` 第 5 行逐字相同） |
| S14 | DONE | A | `ninfer-fusion-repo/src/product/kv_bit_budget.h`（S12 冷层模型的 C++ 移植：`kv_bit_budget_solve`+`KvBitBudgetSolution`，cold=0 热位/页 9536B（源=`include/ninfer/ops/entropy_nvfp4_slot.h:14`，用于 decoder_state.cpp:179）/penalty 0.25，e8 先打包；`kv_bit_budget_spec` 签名兼容）+ `_collab/A_kvbit_cold_cpp.md`（含中途抓到的真分歧：L=24 C=6 b=4.5，py 4.50/2.90 fp8 系 vs cpp 4.48/2.90 int8 系——根因=Python 浮点累加下的非平局，已按 Python 状态空间+字典插入序+浮点域累加修复） | 纯 g++ 主机网格 **27/27 案例 321 行 0 分歧**（层 16/24/30 × 冷页 0-8 × 预算 10 档 vs Python DP 逐行 diff）；新旧头 cold=0 spec 30/30 一致；无引擎 TU 引用该头（构建树零影响） | B/C: `wsl.exe -e bash -c "bash /mnt/c/Users/User/Documents/ziqinzhang/_collab/A_kvbit_grid.sh"`（期望 `PARITY_CASES=27 ... DIVERGENT_CASES=0` + `ORIG_VS_NEW_COLD0_SPECS_IDENTICAL`；证据 `_collab/A_kvbit_cold_cpp_grid_py.txt` vs `_cpp.txt`） |
| S54 | DONE (只读分析; 未改 src/**; 未编译/未开 nvcc/GPU) | S54 | `_collab/S54_spark_target_plan.md` | Spark-X2.5-4B 新 target 接入计划：逐文件清单(variant/bindings/package/CMake/registry/转换器, 27b↔muse 双样本) + 7 钩子逐条接入点 + 6 步顺序与验收 + 14 项风险。**关键更正**: manifest `hook 7/new_op 0` 偏低——①主注意力 16Q/4KV@D256 未注册且 `q_heads==16` 会命中 35B 分支(`ops/wrapper/gqa_attention.cpp:25-30,263`、`ops/launcher/gqa_attention_decode.cu:64`) ②`spark-x2.5-4b` 生成的命名空间含小数点编译不过(`out/spark-x2.5-4b/config.h:3`←`archkit/adapt.py:343`) ③家族无条件 rmsnorm(q,k) 而 Spark 无 qk-norm(`text_context_impl.h:958-959`, 真索引 290 张量无 q_norm/k_norm) ④逐头门需广播/复制权重、gelu MLP 缺两输入乘算子; 滑窗 W=512 无主模型通路(`sliding_window_tokens` 仅在 iso3/NVFP4 核被读, `paged_kv_cache.h:60` 从不被赋值) | M: 读 md §3.0 判定表与 §6 三问(id 命名/首版量化档/SWA 范围) |
| A1 | DONE (只读分析; 未改 src/**; 未编译; 未跑 GPU) | A1 | `_collab/A1_verify_vs_plain.md` (verify vs plain 候选差异表) | 12 条候选差异(D)+9 条已排除(E)，全部带 file:line。**已排除**: #4 argmax tie-break/归约顺序(两条路径是同**一全序**上的唯一最大值 — `ops/kernel/argmax.cuh:22-25` vs `ops/kernel/sampling_device.cuh:76-78,85-97`；verify 走 `ops::argmax`，plain 走 `ops::sample` 的 greedy 分支 `ops/kernel/sampling.cuh:27-64`)、#2/#5 第 0 列语义(token/位置/RoPE/可见 KV/状态槽五件套逐一相同 — `program_impl.h:11823-11827` vs `:11969-11983` + `speculative_round.cuh:31-34`)、`apply_final_logit_policy`(qwen 编译期 no-op — `text_context.h:93-107,125-129`)、无效尾列污染(写回被 `valid_tokens` 限界 — `gqa_attention_decode_bf16.cuh:59-63,148-166`)。**首选根因(D1/D2)**: T=1(plain) vs T=width(verify) ⇒ 同一数学换 kernel 实例化(attention 的 TokenTile/split 由 tokens+envelope 决定 — `gqa_attention_decode_impl.cuh:68-120`；envelope `program_impl.h:594` `{1,..}` vs `:11800` `{frontier+1,..}`；GEMM 的 N=1 vs N=width)，再被 KV 行标定量化**放大并永久化**(scale=max\|x\| ⇒ 次 ULP 可翻码；读回 `gqa_attention_decode_bf16.cuh:148-166,229-243`)。**另报**: dflash2 目标 verify **可能漏 `rope_delta`**(`dflash2_impl.h:410-411` 把 position 表同时当 rope 表，对照 `program_impl.h:11827`/`:11981`)、MTP AR 有效位(`mtp_round.cuh:47` + `write_neutral`)、冷页实例化覆盖待逐档确认、verify 无 head probe(观测缺口) | M: 读 md §1(排除表，防重复劳动) + §3 **先跑 E1**(纯 CLI: 同产物 `--no-spec --prefill-chunk 128` vs `2048`，比 `--print-token-ids`；不同 ⇒ 坐实 D1/D2) + E4(CPU 追 dflash2 `rope_delta` 赋值链) |
| A4 | DONE (只改 `tools/**`; 未写 src/**; 未编译; 未开 nvcc/GPU; 未启 serve; 未下载权重) | A4 | **活1 GUI**: `tools/gui/serve_gui.py`(投机卡片 + 三个日志解析器 + `spec_stats()` + `GET /api/spec_stats` + `/api/state['spec']`) + `tools/gui/i18n_serve.py`(+19 条 `serve.spec_*`) + `_collab/a4_scratch/`(a4_gui_verify.py / a4_gui_http.py / a4_js_check.js / a4_render.py)。**活2 导入复核**: `tools/archkit/adapt.py`(5 处最小修: 派生 head_dim / 复用 index 张量名 + 新 `layer_facts()` / muP multiplier 白名单+探测器 / conv 文案按 §125 改写 / `layer_structure(parallel attn+ssm)` 探测器) + `_collab/a4_scratch/`(a4_probe_index.py / a4_probe_qknorm.py / a4_runall.py / a4_diff.py / adapt_a4.py + before/pre/after 快照 + pre/after 原始输出)。报告 `_collab/A4_gui_specstats.md` | **活1**: 服务卡显示当前会话投机档位+接受率+草稿/轮数/空转步+接受长度，以及 `23,11,3,1,0,0,0 / rounds=55` 形状的位置剖面(柱高按同图归一、悬停给 次数/每轮概率)；门 `gui_i18n_check.py --only serve_gui.py` → **VERDICT: PASS**(13 源键 / 343 表项 / zh-en 等量)，`serve_gui_selftest.py` → **29 passed, 0 failed**(含与 git HEAD 基线的逐字节对拍)，双语渲染 zh 卡 119 个 CJK / en 卡 0 个 / 0 个遗留 `@@`，node 真跑 `paintSpec()` 4 个载荷 0 异常。**数据源实测**: 引擎**没有**指标端点(`http_server.cpp:364-405` 只有 `/health`+`/v1/*`)⇒ 走日志尾部；端点候选表+10s TTL+0.4s 超时已写在代码里，引擎补端点即自动切。位置剖面只在 `--request-log-jsonl` 的 `accepted_per_position` 与 CLI 汇总行里(serve 文本行不打印)，拿不到时明说"没有直方图"。**活2**: 两模型 `config.h` 均被扣成 `config.h.BLOCKED`(LFM2: `new_op:layer_kinds(conv x22); attn:head_geometry(32q/8kv@64)`；Falcon: `state_space(mamba_*); layer_structure(parallel attn+ssm x44/44); attn:head_geometry(12q/2kv@128)`)，与参考实现对照判为**真缺口**(§122 结论成立)；同时抓到导入器 3 处新漏检/误判并修：① LFM2 无 `head_dim` ⇒ head 几何判定被**静默跳过**(`if q and kv and hd` 遇 None 直接不报) ② config 无 qk-norm 旋钮但 index 带 16 个 `q_layernorm/k_layernorm` 张量 ⇒ 原判 `=absent` **极性相反** ③ Falcon 9 个 muP multiplier **完全没读** | M: 活1 门禁 `cd ninfer-fusion-repo/tools/gui && py -3 gui_i18n_check.py --only serve_gui.py`(期望 `VERDICT: PASS`) + `py -3 serve_gui_selftest.py`(期望 `29 passed, 0 failed`) + 功能 `cd ../../.. && py -3 _collab/a4_scratch/a4_gui_http.py`(期望 `/api/state` keys 含 `spec`、`/api/spec_stats` 200)；活2 复跑 `cd ninfer-fusion-repo && py -3 tools/archkit/adapt.py ../models/LFM2-2.6B-Exp --model-id lfm2-2.6b-exp` 与 `... Falcon-H1R-7B ...`，对照实验 `py -3 ../_collab/a4_scratch/a4_diff.py ../_collab/a4_scratch/pre`(期望：只有 lfm2/falcon 的缺口集变化，minicpm5-1b 与 spark-x2.5-4b 的 `config.h.BLOCKED` 逐字节 `IDENTICAL`)。**未验证**: 无权重的形状复核、位置剖面真数字(GPU 在训练)、`tie_embedding` 别名在 `tools/convert/common/source_map.py:387` 未读(结论未受影响，留给 S53/E10)、`check_params.py` 的 `COVERED_KEYS` 未同步 |
| A3 | DONE (三个 patch 只生成未应用; 未改 `src/**`/`tools/**`; 未编译; 未开 nvcc/ptxas; 未用 GPU) | A3 | `_collab/A3_spark_newops.md`（三项 new_op 逐项：改哪些文件/行 + diff 草案 + 代价算术依据 + 验证方法 + 风险）+ `_collab/A3_spark_head_geometry.diff`（8 文件 +140/-21, 335 行）+ `_collab/A3_spark_headwise_gate.diff`（6 文件 +153/-5, 219 行）+ `_collab/A3_spark_gelu_mul.diff`（8 文件含 6 新增 +384, 427 行）+ 生成器 `_collab/A3_mkpatch.py`（每个锚点断言唯一，漂移即失败）+ 证据 `_collab/a3_scratch/`（活体字节副本 `live/`、dry-run 影子树 `shadows/`、`dryrun_*.txt`、`live_md5{,_after}.txt`、`cmp_md5.py`、`counts.py`） | 三项各自 `patch -p1 --dry-run` **rc=0**（原文见 md §5）。①**头几何 16Q/4KV@256 不在 kernel 实例化域内**（S54 §3.9 的 R2"待证实"⇒ 证实）：i8/E8/NVFP4 的 `Wc` 行 tile 调度表按 group 硬编码，`GroupSize=4` 时 i8 TT=5 落 `else` 臂 `Wc=24` ⇒ `Consumer=12 × PVNt=2 × 8 = 192 ≠ 256` **静默少算 8 个 PV n-tile**，i8 TT=6 的 `Wc∈{24,12,6}` 与 nvfp4 `TT≥5` 的 `Wc=12` 都算出 `PVNt∈{5,10}` **违反 static_assert ⇒ 直接编不过**；补丁=新 `Gqa16x4Geometry`（`DecodeSplitScale=1`，与 27b 同为 4 KV 头 ⇒ CTA 量纲一致）+ `kv_heads_for_q_heads` 改**三元组校验**（KV 头数一律取 `cache.num_kv_heads`：16Q/2KV 与 16Q/4KV 在 D256 上只能靠 KV 头数区分）+ 5 处**隐式回落 35B** 收紧（含活体 `gqa_attention_cached_small_t_launch` 尾部**没有 throw** 那处）+ i8/E8/NVFP4 显式拒绝（`if constexpr` 守卫，不用运行期 throw —— 后者拦不住实例化）；BF16/FP8/ISO3 档按 `(TOKENS,WARPS)` 表接受 group 4（`Wc*16 ≥ tokens*GroupSize` 全满足）。②逐头门控：路线(a)=转换期把 `g_proj` 展开成 `[4096,2560]`（行 `h*256+c` 复制第 h 行）**引擎零改动**（gate 缓冲本就按 `query_size` 行分配、`attn_mix` 本就 `view({head_dim,n_q,T})`+`sigmoid_mul`，且 `(4096,2560)` 与 Spark 的 q 投影同形**本就要注册**），代价 **+754.97 MB BF16 = 权重 8.23 GB 的 +9.2%**（decode 权重带宽同比例）；路线(b)=已落 diff（**扩展 `sigmoid_mul` 家族**：同符号加"逐头标量门"形状分支 + 2 kernel + 1 launcher，**不新增 op 文件、不动 CMake**），额外权重 0，且新分支要求 `gate.ne[0]==x.ne[1]`（即 `n_q==head_dim`）而活体 3 个调用点（16/24/32 q 头 vs 256/256/128 头维）**结构不可达** ⇒ 对 27b/35b/muse 零影响；遗留：`g_proj` 的 `(16,2560)` linear 几何需 padding 到 64 行或拼进 `q_k_v`（`bf16_gemm_mma.cu:17` 要求 `n%64==0`）。③gated GELU：全 op 清单**没有两输入逐元素乘**（只有 `silu_mul`/`sigmoid_mul`/`residual_add`/`linear_add`；`ops::gelu` 是 in-place 单元激活）⇒ 必须新增 `ops::gelu_mul`（五件套，复用 `gelu_one<false>` 精确 erf，不复制公式）；零影响证明=只改 2 行 CMake + 6 个新文件、无既有调用点、既有 op 测试不经过新代码 | M: 跑 `wsl.exe -e bash -lc "python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/A3_mkpatch.py"`（期望三行 `DRYRUN_OK`；生成器只读 `_collab/a3_scratch/live/` 的活体字节副本，不写仓库、不写 tools）；复核 md §2.1(c-3) 的 group-4 调度表（可纯手算复核：`RowTiles=ceil(TT*4/16)` + 取 `Wc` 后验 `(Wc/RowTiles)*PVNtPerWarp*8 == 256` 且 `PVNtPerWarp ∈ {2,4,8,16}`）与 §5 dry-run 原文。**口径警告**：S54 用的镜像 `ninfer-fusion-repo` 与本批基线（活体树 `/home/user/ninfer-fusion`）在本次要动的 6 个文件上漂移（镜像缺 S36 的 `Geometry::HeadDim` 泛化、多出 S23 的 decode TU 拆分、缺 S45d 的 nvfp4 守卫）⇒ 本批 diff 全按活体树生成，S54 的镜像行号不可复用（对照表见 md §1）。**未改动证据**：`a3_scratch/live_md5.txt` vs `live_md5_after.txt`（36 文件）diff 为空 ⇒ `TREE_UNCHANGED` |

## GPU 占用
- 训练：单实例 batch4，0.40 steps/s，ETA 约 13:05 到 6000 步。GPU 窗口需先 kill 训练
  （PowerShell `.Kill()`，且**必须复查 nvidia-smi**：崩溃进程会留占显存），窗口结束用
  `_train_df2_resume.bat` 从最新 checkpoint 续跑。

## 待认领 / 未落地项（详见 `_TODO.md` §117）
- U1 [DONE-A] 全局 `--kv-dtype fp8`：规划器组号 256→16（对齐 kernel 的 `gqa_kv_nvfp4_scale_index` 与 wrapper 校验）。待验：fp8 冒烟。
- U2 [DONE-A] 全局 `--kv-dtype e8`：workspace dtype 白名单补 `DType::E8Kv`。待验：同 U1。
- U3 [DOC-A] `all:bf16` 仍是"未设置"语义：已在 `kv_options.h` 注释写清替代做法（`--kv-dtype bf16`）与将来需要 was-set 掩码。
- U4 [RETRACTED] Muse SWA 按 full 跑是 `config.h:64-70` 明写的阶段设计（E3 再接线），**不是缺口**；
  但上下文 >2048 时其数值含义与训练时不同，长上下文掉针判读要区分。
- U5 [DONE-A] `kPagedKVCacheMaxLayers=64` + `plan_cache` 越界校验（SYNTAX_OK）。
- U6 [TODO] 大件：W2③ fuse_draft/LABD / W7(等 ckpt) / W13 权重卸载 / Windows 移植 W-P1..P6。
- U7 [已改未验] e8 尺度顺序：prefill-fill ✓ / decode ✓（同族第三处）/ Muse page-fill ✗（§116c，需 Muse+e8 冒烟）。
- U8 [LESSON] 本机 31.4GB 内存 + WSL 24GB 上限 ⇒ **构建与训练必须串行**：并行时两边都停摆
  （实测训练 09:04 后完全停、ptxas 拖过 27 分钟）。后续重构建一律走"窗口"：暂停训练→构建/验证→续训。

## 当前窗口（window C，运行中）
暂停训练 → 等构建落地 → `touch` 三个 header 强制补编 decode TU → Muse 验证 (`_muse_verify.sh`)
→ e8 三档对照 (`_e8_postfix.sh`) → `_train_df2_resume.bat` 续训。结果回来由 A 落 board + TODO。

## 用户八问的落地状态（2026-09-10 09:3X 逐条查证）
| 问 | 已有 | 缺 | 状态 |
|---|---|---|---|
| 启动自测→固化→跳过→可重校 | 离线采集 `kv_calibration.h` + 离线烘表(kvcalib-a→常量) | 运行时"跑一遍→固化→跳过→手动重校"流程 | **半截** |
| 输入 bit → KV 温窗自动分配 | 实证矩阵 + DP 分配器 `tools/archkit/kv_bit_budget.py` + `--kv-layer-storage` 可吃输出 | 引擎内"给 bit 数自动分配"入口 | **策略完善/引擎自动化未做** |
| 推广到冷窗 | `--cold-policy none/window/host/disk` + entropy/rANS 冷槽 + `max_cold_pages` | bit 分配器未含冷槽代价模型 | **冷窗可用/未统一分配** |
| 热/温/冷自动动态分配 | FreeToken 三步骨架(能量观测/周期重排/pacing) | 按热度自动决定热冷归属的闭环策略 | **未闭环** |
| 权重卸载到内存 W13 | — | 全部（代码里无 offload 实现） | **未做** |
| lookup ngram | CUDA 内核 + fuzz 4000 全绿 + sidecar 加载器往返 | 真表 GPU gather / 前缀缓存(含 GDN 循环性) | **部分落地** |
| FreeToken 加载 FlashNext | P0 骨架 + 74,520 项绑定契约 + 转换器 | **真实 checkpoint**(TODO #552 等用户指路) | **架构就绪/权重缺失** |
| dflash2 接受率根因 | ①采样口径实锤 24→53.3% ②非 KV 精度 ③5 层容量差 | ④"树材料被单链浪费"未实施；口径闸门已定 | **部分找到** |




## 主控 (M) 追加 — 2026-09-10 09:5X

| ID | 状态 | 负责人 | 产物 | done 判据 | 验证者 / 命令 |
|---|---|---|---|---|---|
| S14 | DOING | A | `src/product/kv_bit_budget.h` 冷层 C++ 移植 + 主机 g++ 网格 parity 测试 | 与 Python DP 同网格逐行一致（首个分歧即判失败） | M: 读 `_collab/A_kvbit_cold_cpp.md` |
| S15 | DONE | B | `_collab/B_o1fix_verify.md`（O1 修复对抗复验：删除测试/dense 逐字节回归/工件泄漏/gdn_layers；合成等价路径替代复跑作者三命令） | **4/4 PASS**：手植假 config.h 重跑后被删且 .BLOCKED md5 不变；ctrl_2+MiniCPM5 config.h 与 sed 回退基线逐字节一致；grep Gdn 零命中、leaves/bindings.stub 不产出、manifest 保留原始 kinds；conv 模型 `gdn_layers()==0`+`#error` 与宣称格式同构 | M: 报告含全部原始命令；**观察 O2**: blocked→clean 重跑不回收旧 `config.h.BLOCKED`（无害, 建议清洁路径也 unlink） |
| S16 | 待 S15 复验 | M | `tools/archkit/adapt.py` O1 加固：未知层型→`Unknown`、`config.h.BLOCKED` 门禁、`#error` 兜底 | minicpm5-1b 出 `config.h` 且无 `#error`；lfm2/falcon 出 `config.h.BLOCKED` 且旧 `config.h` 被删 | B: S15 |
| S17 | DONE | C | `ninfer-fusion-repo/tools/archkit/dflash2_tree.py`+`test_dflash2_tree.py`（beam-L DDTree 建树器，引擎 selector 布局 [step][pred][cand]，默认关）+ `eval_ddtree.py --tree beam:L`（备份 `_eval_ddtree_pre_s17.py.bak`）。注：模块/测试系中断轮遗留（10:20），本轮修复测试 2 个假不变量（`paths[0]==单链` 对 beam 不成立——cum 最优可中途超越贪心链；"全路径同首秩"同假）+ 模块 docstring 同类错断言，并完成 eval 接线 | 合成测试 **26/26 PASS**（(a)预算 l∈{1,2,3,7}×{1,2,4,16}l 全守恒；(b)构造分支 EAL beam=4 vs chain=0 **实数断言**；(c) l=1 退化==单链 64 随机表）；OFF 路径 diff 备份 **0 删除行/60 新增** + py_compile OK（S4a 同法）；布线冒烟 `_collab/C_s17_smoke.py` PASS。**合成数字只证 builder 覆盖、非真实接受率**（离线无 pair 码本，行重复 per-row logp；真 selector 分数引擎运行时才有）——GPU 窗口必测 hit@1/hit@4：`wsl.exe -- bash -lc "cd /mnt/c/Users/User/Documents/ziqinzhang && python3 eval_ddtree.py --ckpt data/dflash2_ckpts/step_006000.pt --teacher --beam --tree beam:4"` | A/B: 跑上命令看 tree EAL delta + GATE 行；运行级 OFF diff 随 GPU 窗补（当前宿主 RAM 1.9GB、构建中，同 S4a 先例） |
| S18 | DONE | C | `_collab/C_s14_verify.md`（对 A 的 S14 冷层 parity 的对抗复验；harness `C_s14_grid.sh`/`C_s14_tie_probe.py`/`C_s14_tu.cpp`/`C_s14_rnd.cpp`，g++ 仅 /tmp，未动 src/**、未碰 GPU/构建树） | **发现真分歧 ×8**：二进制精确中点预算（4.125/5.625/6.625）× cold≥8 × L≥48 时 py≠cpp。最小复现 `--layers 48 --cold-pages 16 --bits 4.125`：py achieved 4.12/penalty 7.49 vs cpp 4.13/**7.47**（cpp 反而更优，py 容量不容）。根因=预算取整：`round(412.5)`=412（py 银行家）vs `(int)(x+0.5)`=413（cpp 半上），容量差 L 单位；**A 网格全 0.25 倍数 ⇒ .5 落点不可达**，声明在其网格内成立、边界外失效。浮点 tie 故事**证实**（1640 配置：噪声 tie 可达 4 对、0 次改 counts、cpp==py）；接口 source-compat **PASS**（2/3 参旧调用编译且==显式 cold=0，cold 活性 @budget 3.5）。REQUEST→A：`kv_bit_budget.h:137` 改 `(int)std::nearbyint(budget*100)`（+`<cmath>`；nearbyint 与 py round 在 412.5/413.5/562.5/412.9999… 全一致），改后重跑 `bash /mnt/c/Users/User/Documents/ziqinzhang/_collab/C_s14_grid.sh` 期望 `EXTENDED_GRID_ALL_MATCH`。3.37/4.13/5.62 与 L≤12（含全部中点）全 CLEAN——现行 0.25 档部署不受影响 | A: 修 header 后跑上命令复核；M: 归档 REQUEST |
| S19 | DONE | B | `_collab/B_s17_verify.md`（S17 对抗复验：测试复跑/diff 复核/flag 矩阵/stdlib 纯度 + beam≥chain 边界探针，不复跑作者命令） | **4/4 PASS + 2 纠偏**：实测 **27/27**（S17 行写 26，少计 1）exit 0；diff 复核 **0删/60增** 属实且行为全带门禁（OFF 结构级未动）；8 个坏 flag（含 beam:-1/beam: 4/无冒号）全 exit 2 零 traceback；两模块 import 后 sys.modules 无 torch/numpy | **边界实锤**（报告含最小复现）：链 cum 跌出 top-L 且 truth 骑链时 beam EAL 0 vs chain 4——builder 有意不含链（L=1 恒等已验 50/50）；建议链路径保底入树（≤l 节点换 EAL_tree≥EAL_chain 定理化） |
| S20 | DONE (patch 未应用) | A | `_collab/A_n1_patch.diff`（N1 `--kv-bit-budget <bits>`：5 文件/117 行 unified diff，**只评审不落树**——窗口 D 在编 src/）+ `_collab/A_n1_patch.md`。要点：parse 点 serve_options.cpp:266 **无模型知识**（实锤），延迟到 layouts_impl.h `make_sequence_planner_impl`（TextConfig::full_attention_layers 可见处）用 S14 parity DP 解析成表，喂既有 `--kv-layer-storage` 路径；与显式 layer-storage **互斥硬错**；冷耦合=N2（cold_cap=0），校准=N3；apps/cli+perplexity 两 parse 点未接（已列 NOT-doing） | `patch -p1 --dry-run` 对当前树 5/5 OK；**真 patched serve_options.cpp 独立 g++ -c 通过**；行为检查 7/7（含 L=16/4.5 spec==S14 golden `0-7:e8,8:int8,9-14:nvfp4,15:iso3`、互斥硬错、17 越界拒） | B/C: `wsl.exe -e bash -c "python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/A_n1_mkpatch.py"`（期望 PATCH_DRYRUN_OK）+ 按 `_collab/A_n1_patch.md` §3 重跑 n1_check 期望 ALL_CHECKS_PASS；源 `_collab/A_n1_compile_check.cpp` |
| S21 | DONE | B | `flashnext_bindings.py`(+`canonical_source_key` 单点规范化+audit 重建: 伴生/残余/影子/会计恒等) + `flashnext_convert.py`(SourceIndex 规范化+verify_plan 无静默丢弃+`--real-names` 真名单自检, 旧自指检查降级 [smoke]) + `_collab/B_s21_contract_fix.md`(含**下载后配方**) | 实测: raw **1/74520** 独立复现 → canonical **74174/74520, 伴生 221184, 残余 1081 全分桶(other=0), 74210+221184+1081=296475 恒等**; 缺源 346 逐族列明(96 norm+96 hc 误读+36×3 gdn+36 idx+6 ple+3 mtp+1 output_norm)且各配真实候选键; 21/21 抽样键落**正确**引擎(含 A_log→dt_bias 影子实锤 36 条); `--self-test --real-names` exit 0 | M: 跑配方步骤 1/2(`--audit` index.json 与 `--self-test --real-names`)核对四个钉死数字; **下一步 S22=346 行契约回填(候选键已列), S23=artifact writer+NVFP4 打包+128 分片 PLE 拼接** |
| S22 | DONE (待 C 复验) | A | `src/product/kv_bit_budget.h` 银行家舍入修复（S18 的 8 处 parity 破洞）: `+0.5` 半上 → `std::nearbyint`(ties-to-even, 同 Python round) + `<cmath>` + 失败模式注释（4.125*100=412.5 精确落 .5, capacity 差 `layers` 单位）; `_collab/A_s22_rounding_fix.md` + `A_s22_grid.sh`（复用 C 的 `C_s14_grid.sh` 原样 + phase5=C 全部 8 发散坐标） | 最小复现 48L/16c/4.125: before py 4.12/7.49 vs cpp 4.13/7.47 → **after 双侧 4.12/7.49 逐字同**; C 原网格 26 例/327 行 **DIVERGED=0** + phase5 8 例 0 差 + 舍入抽查 8/8 nearbyint==py round; 原始 S14 网格回归 27/27 + orig-vs-new 30/30 仍绿; 无新发散 | C: 重跑 `wsl.exe -e bash -c "bash /mnt/c/Users/User/Documents/ziqinzhang/_collab/A_s22_grid.sh"` 期望 `EXTENDED_GRID_ALL_MATCH`+`S22_ALL_MATCH`（本行非自证, 等 C 的独立复验） |
| S23 | DONE (patch 未应用) | C | `_collab/C_s23_tu_split/`（decode 巨 TU 按档拆分的**准备好的 patch 集**，未动 src/）: `mkpatch.py` 锚定生成器 + `patch/01..09`（重生成 `impl.cuh` **取自活体 decode.cu**——原 header 只被死文件 muse/g35 引用且已漂移 ft::observe 位置；DISPATCH 宏改 runtime 外部档调用 = e8 既有生产模式）+ 5 个新档 TU `{i8,nvfp4,fp8,iso3,bf16}.cu` + `tiers.h` + CMake hunk + **`apply.sh`**（幂等/先全量 dry-run 再写/逐文件 APPLIED 行/--dry-run 不写/不碰 cmake cache）+ `review/` 人读副本; 设计与证据 `_collab/C_s23_tu_split.md` | 影子树生命周期: DRYRUN_OK→9×APPLIED→APPLY OK exit 0→重跑 "already applied" exit 0→.orig/.rej=0; 生成器重跑 md5 全同（确定性）; 应用后 decode.cu 的 launch_tc_partial 引用=0（实例全移走）; 混合 EOL（impl/cmake=CRLF）逐文件保持（影子 dry-run 实测抓出，非假设）。**量**: 旧 TU ≈810 内核实例、25-35min 一格；拆后 decode≈48(reduce,<1min)/i8≈330(最重 ~8-12min)/nvfp4=36/bf16,fp8,iso3=144 各 ⇒ **-j2 峰值 < 旧单 TU 峰值**，-j4 待 memwatch 证实 | 窗口 H: `bash /mnt/c/Users/User/Documents/ziqinzhang/_collab/C_s23_tu_split/apply.sh --dry-run` 再去 --dry-run；与 ccache launcher 同窗落地（协调者定序）；构建后 `nm` 分区检查；GPU 判定 `_muse_verify.sh`/`_e8_postfix.sh`（见 md §验收） |
| S26 | DONE | B | `flashnext_bindings.py` 契约 rebase（−306 删除/真实不存在: 96 逐层范数+96 qsa.hc 误读+36 conv_bias+36 beta+36 旧 idx+2 ple+3 mtp+1 output_norm; +590 name-faithful: 384 hyper-connection+108 GDN a_log/in_proj_a/b+48 shared_expert_gate+36 融合 idx+7 ple+3 mixer+4 mtp; 改指: gate←in_proj_z、dt_bias 去 A_log 影子、ple→layers.1.ple.*、table←128 分片）+ `flashnext_convert.py` 随动(PENDING_SHAPE 460 维度待张量头/sidecar 消费全别名/引脚更新) + `_collab/B_s26_contract_rebase.md`(含更新版下载后配方) | **missing 346→0, engines 74804/74804, matched 74931+伴生 221184+残余 360=296475 恒等**; 残余=visual 333(text-first)+mtp 27(需引擎 MTP 层设计), `other`=0; A_log 影子 0; self-test exit 0 | M: `wsl.exe -e bash -c "cd /mnt/c/Users/User/Documents/ziqinzhang && python3 ninfer-fusion-repo/tools/archkit/flashnext_bindings.py --audit-gguf _collab/M_flashnext_names.txt"` 核 missing_count=0 与 residue 两桶; S27=writer(NVFP4 打包/PLE 拼接/artifact), 未开始 |
| S27 | DONE | B | `flashnext_convert.py` +`--measure-shapes`(实测头维度回填 PENDING_SHAPE, 容忍半截分片; 已随下载推进回填 18/460) + `specs/qwen4_exp_measured_shapes.json` + `_collab/B_s27_writer.md`(重排结论+最终配方) + round-trip 工件 `tools/archkit/out/flashnext_s27/` | **swizzle 答案(源码实证)**: 码字=逐字节原样(行主序, 偶元素低半字节, 无 nibble 重排); **尺度=必须重排**(自然 [N,K/16] 行主序 → 512B 128×4 tile, `nvfp4_gemv.cuh:87` 公式); 尾 F32=**1/weight_scale_2**(引擎做除法); input_scale 走元数据。**round-trip 12/12 字节级相等**(layer0 experts 0-3 × gate/up/down, codes/scale/div/几何/核公式抽样全 True, exit 0, 复用 Muse 投产的 `encode_nvfp4`) | exclude 后果: NVFP4 路径仅 73,728 专家张量(layer-*-experts-*); 其余全 BF16 直通(model-bf16-*); **PLE 表本体 FP8 存储**(model-plefp8-*, 10 分片未到位, 子布局待落地钉死不猜); M: 下载完成后跑配方步骤 3 期望 `460/460 backfilled`; S28=全量 artifact 写出器 |
| S24 | DONE (patch 未应用) | A | `_collab/A_s24_window_table.diff`（E3 滑窗表: 3 文件/174 行, 只评审不落树）+ `A_s24_mkpatch.py` + `A_s24_window_table.md`。路径镜像 layer_residual: TextConfig(is_swa_attention/sliding_window, `requires` 守卫——qwen3_6 家族变体无此访问器, Muse 39/52 层=2048) → DecoderStateSpec/PagedKVCacheLayout 新 `layer_sliding_windows` → plan_cache(短表拒绝, 同风格) → ctor → **两个 view 各补 `.sliding_window_tokens`**(现缺省 0=全量注意, 内核 5 处 `sliding_window > 0` 守卫实证 0=无窗); MTP cache 保持 `{}` | `patch -p1 --dry-run` 3/3 OK; 生成器断言锚点唯一+双 view 各恰一次插入(首版 double-insert 被自检抓出已修); 主机 g++ 语法检查**不可行**(core/paged_kv_cache.h 引 cuda_runtime_api.h, 已实测), TU 编译归应用窗口 | 窗口 H: 应用后 **Muse 8K/32K/57K 针尖 (`tools/archkit/longtest_57k.py`, §47 配置) 期望不降级, 降级=回滚不是解释**; qwen27 冒烟须输出逐字同(全零窗表) |
| S25 | DONE (patch 未应用) | A | `_collab/A_n1b_cold_pages.diff`（`--max-cold-pages N` 显式冷页上限: 6 文件/126 行, 与 N1 同管线 parse→ServeOptions→EngineOptions→inputs/impl→layouts_impl:151 推导; 0=保持今日推导）+ `A_n1b_mkpatch.py` + `A_n1b_stack_check.sh` + `A_n1b_cold_pages.md`。**precedence(代码实证)**: None 恒 0(decoder_state.cpp:180 无论策略都预留槽, None 下=死显存); **Host 即便显式 cap 也保持 0**(grep ColdPolicy::Host 零运行时消费者, program_impl.h:865/875 只认 Window/Disk——N2 落地 host 驱逐时再翻, 开放设计点已注明); Window/Disk cap 胜过推导 | `patch -p1 --dry-run` 6/6 OK; **N1→N1b 叠加实测全过**(offset 无 fuzz, 窗口 H 定序可用); 校验镜像 --cold-keep-tokens(拒非截断); cap 语义与 S12/S14 DP 对齐(冷层=0 热位+9536B 槽/页) | 窗口 H: N1 后应用本 patch 再编译; C: 复核 md §3 决策表与代码引用行号 |
| S28 | DONE (patch 未应用) | C | N3 轮1 行标定旁车: `tools/kv_rowscale_sidecar.py`（dump/write/identity/validate; 格式=NINFERKVRS1 64B 头[版本/层/kv头/头维/flags/模型哈希/标签]+BF16 payload, 与常量表逐字节兼容）+ `_collab/C_s28_rowscale/`（`review/gqa_isoquant_row_scale_loader.h` host-only 解析/校验、`..._loader.cu` **env 门控** `NINFER_KV_ROWSCALE` 未设=关/失败=硬抛错带字段名、`C_s28_mkpatch.py`→`patch/C_s28_loader.patch` 含 decoder_state.cpp::plan_decoder_state 单锚点 hunk——非 A 的 N1/S24/S25 文件集）+ `_collab/C_s28_rowscale_sidecar.md`。**身份闸门**: 非 identity 表 layers/kv_heads/head_dim 必须全等（16 层表×52 层 Muse=拒, 今日静默外模型表现在显式拒）; identity 表验全 1.0 且可欠覆盖; hash 双非零才强制 | dump 16384 词 range [0.5,2.0]; **往返证明**: dump→旁车→C++ parse==原表逐字节; 主机 g++ 测试 **12/12 PASS**（magic/version/crc/length 各自报字段名; 外模型拒且 upload 不可达; hash 不匹配拒; identity 52 层接受+全 1.0）; patch -p1 --dry-run OK | GPU 窗口: qwen 32K 针尖回归（同表旁车加载=预期逐位不变, 须实测）; Muse 32K/57K 标定-vs-identity 属轮2质量项; 引擎侧待窗口 H 批量应用补编（损坏旁车=可读硬错, 好旁车=`[kvrs] applied` 行）; 轮2: hash 接线（归 A 选项管线）+ --kv-calibrate 闭环（KvCalibrationCapture 死代码复活） |
| S29 | DONE | C | `_collab/C_s29_rounding_verify.md`（S22 舍入修复独立复验, 自有 harness; 脚本 `C_s29_sweep.sh`/`C_s29_rnd_dense.cpp`/`C_s29_dense_check.py`; 证据 `_collab/C_s29_rounding_verify.md`） | **4/4 PASS**: ① 原样重跑 `C_s14_grid.sh` = **26 例/327 行/0 分歧 EXTENDED_GRID_ALL_MATCH**（与 A 宣称计数一致）; ② 超出双方覆盖的扫描 **60 例 0 分歧**（L∈{5,9,20,36,40,56} 非{16,24,30,48,64}, cold 0..16, 预算=9 个 k/8 精确中点+6 个落非半点的 .005 邻域{4.115/4.135/5.615/5.635/6.615/6.635}+5 个三位小数+既有锚点; L=56/C=16 首轮 py 超时被跳过→850s 重跑 MATCH 已补）; ③ **稠密 12050 点**（%.17g 跨语言位同双精度）nearbyint==py round **0 不匹配** ⇒ 修复不会开出镜像洞, 旧匹配点未变; ④ A 确系原样复用（A_s22_grid.sh:13 直接 bash 我的脚本, phase5 覆盖我 8 行分歧表全部坐标） | A: S22 行可去"待 C 复验"; 残余=python 大 L×冷×大预算 >300s 属性能非正确性 |
| S30 | DONE (patch 未应用) | A | `_collab/A_s30_budget_cold.diff`（N2 耦合: **仅 layouts_impl.h**/147 行, 基座=N1+n1b 已应用态）。三 hunk: ① `effective_cold_pages()` 共享助手=S25 precedence 单一事实源（None 恒 0/**Host 恒 0——零运行时消费者, 不能放冷层, 代码里直说不绕**/Window/Disk 显式 cap 否则 keep_tokens 推导）② persistent_layout 推导改调助手（DP cold_cap 恒==实际预留池）③ N1 解析块改 4 参 DP 联解, 计划分流: cold 段→nvfp4 热窗+`cold_placed=R` 取证行（提示 `--cold-policy window|disk --max-cold-pages`）, 其余原样喂既有 override 路径——**不发明新存储 tier** | 叠加序 N1→n1b→s30 三步 dry-run 全 OK+全程 apply 实测; 主机检查 7/7（cold_cap=0 表==S14 golden 逐字/None 压显式 cap/Host=0/推导 128/64+16=18/两组 dump 样本核对含槽位 parse）; **S14 parity 复跑仍 0 分歧**（EXTENDED 26 例+PHASE5 8 例 DIVERGED=0）; 样本: L16/4.0/Disk8→cold 7-11, DP achieved 3.84 vs 热足迹 5.25 (nvfp4 窗代价, 人可核) | 窗口 H: N1→n1b→s30 顺序应用后编译+serve 冒烟看 `[kv-bit-budget]` 行; C/B: 复跑 `A_s30_check.cpp`（须 /tmp/s30patch/a 作 -I）+ `A_s22_grid.sh`; **nvfp4 热窗默认=单行可换点, 冷 A/B 后定** |
| S31 | DONE (patch 未应用) | D | `_collab/D_s31_503_p0.diff`（503 恢复设计 P0 **残差**, 3 文件/230 行: types.h EngineFailureState +category+last_request_error; engine_core.h 分配类 runtime_error→resource（cudaMalloc/cudaMallocHost 前缀, arena.cu:72/167/327 实证）+fail_all_locked 停摆日志/类别 + lane 遥测; http_server.cpp `/health` **脱敏**: category 上 wire、reason 只进日志, 免鉴权豁免 :305 不动）+ `_collab/D_s31_503_p0.md` + 生成器 `_collab/D_s31_mkpatch.py`。**关键核对: 设计文档滞后**——P0+P1 已在文档当晚落地为 W16（`51a7a8f`/`487562a`/`a7ca4dd`: §3.1 分类+§3.2 lane 失败+§3.5 health 503+§3.3 recover()+POST /recover+注入钩子全在树）; md §7 逐行漂移表（fail_all_locked 实为 :1959-1992 非 :1782-1804 等）; **门 1 钩子已在树**: `NINFER_FAULT_INJECT=request_once|invariant_once` + `MIN_ID` 跳 warmup（engine_core.h:1893 定义/:1389 调用, Setup 相位可归因）→ req-1 失败/req-2 完成可直接跑 | `wsl.exe -e python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/D_s31_mkpatch.py` → **实测 S31_PATCH_DRYRUN_OK**（逐文件+combined 4/4 OK, 锚点断言+自检计数全过, 三目标文件 pristine 未动, dry-run 零写入）; §4 四验收门**全部需 GPU 窗口**（精确命令在 md §5.2; 门 2 的 reason 改读 category+server log, 脱敏的必然结果） | A/M: 重跑上命令核 S31_PATCH_DRYRUN_OK + 抽查 md §7 漂移行号; 窗口 H: **涉公共头 types.h 会触发依赖 TU 重编, 建议并批一次编译**+四门实跑（门 4a CUDA abort 检查现在即过: device.cu:43） |
| S33 | DONE (patch 未应用) | B | `_collab/B_s33_ple_wiring.diff`（N6: 新增 `src/targets/qwen4_exp/impl/ple_runtime.h` 接线 seam——`PleRuntime` 持有唯一 PleTable、sidecar 根三级发现(显式>artifact 相邻 ple-root>关)、`gather_batch` 出 [16×160=2560, T] BF16; 默认关=attach 返回 nullptr, 配置了但 manifest 缺失=抛错带路径与关闭方法, 全链无零填充）+ `_collab/B_s33_ple_wiring.md`(符号级调用路径核查) | `patch -p1 --dry-run` **DRYRUN_OK**(仅新增文件, 与 window G 零冲突); 核查: op 侧全部存在(PleTable/PleLayout/from_manifest/gather, ple_table.h:28-56+ple_layout.h:73, CMake:29/:110 已编); **不存在(发现)**: ①qwen4_exp 运行时只有 2 文件 stub→gather 消费端无代码 ②registry 未注册(construct_target 无分派) ③config.h:11 `intermediate=None` 不可编译 ④EngineOptions/serve 无 --ple-sidecar ⑤eos 无载体 ⑥sidecar 契约 BF16 vs checkpoint 表 FP8(转换器须物化) | M: 跑 dry-run 命令; **等待**: FP8 子布局钉死(model-plefp8 落地)、引擎内数值核对、GPU 出文本测试, 及前置轮(运行时主体/注册/config 修复/serve 旗标) |
| S34 | DONE (patch 未应用) | C | N4 热/温/冷闭环轮1: `_collab/C_s34_closed_loop.md`（环路定义: 观测=ft::snapshot 累积和的**差分窗口均值**(不改 ft_stats.h)+EWMA α=.5; 决策=非深层且过 置信32轮/连续2窗低于25%分位切点/驻留3周期 三闸的最冷层填满 cold_cap(=S30 effective_cold_pages, NINFER_FT_COLD_PAGES); 行动=**dry-only**）+ `_collab/C_s34_cold_loop/`（`patch/C_s34_first_patch.diff`=新 `src/serve/kv_cold_policy.h` 纯host策略+kv_auto_relayout.cpp loop() 单锚点 tap; `C_s34_policy_test.cpp`）| **不变式**（可机检, `invariant_violations` 读决策时证据）: 降冷仅当 EWMA 连续 stable_cycles 窗 ≤切点+驻留≥3+非深层+观测≥32轮+池不溢; 未观测永不降。主机测试 **15/15 PASS**（含 12×200 周期随机流 **2400 决策 0 违例**, 池界/深层不冷恒真; 工作例: {2,5} 第4周期降冷、L5 离切点即出池、槽位由次冷层回填）；patch -p1 --dry-run OK。**重排成本界**: 32K ctx 每层 nvfp4 热窗 32MiB, cap=8 ⇒ ≤256MiB/次, PCIe 地板 ~16ms, 真失速=排干(≤120s)+graph 重捕 ⇒ 每周期至多一次 | GPU 窗口（补丁+S30 栈应用编译后）: `NINFER_FT_STATS=1 NINFER_FT_RELOAD_SECS=60 NINFER_FT_COLD_PAGES=8 ./apps/ninfer serve … 2>&1 \| tee dl/s34_dry.log` 看 `[ft][cold:dry]` 行; 真能量分布/切点合理性/质量矩阵(针尖 32K/57K 动态-vs-静态S30-vs-全热)全部待 GPU; live 接线=下轮(需 S30 池消费者, 盲接=静默no-op 已注明); TU 全编译需 CUDA 头(同 S24 先例), 新头已全host编译+测试 |
| S32 | DONE (patch 未应用) | A | W13 权重卸载 P0: `_collab/A_s32_w13_p0.diff`（新 `src/product/weight_residency.h` host-only 无 CUDA 依赖: WeightResidency{Resident/Host/Disk}+WeightSpan(layer,expert)+plan 载体挂 `binder.h MaterializationPlan.weight_residency`（空=今日行为, materializer 不动）+`classify_weight_residency`（专家优先→dense 最深层优先）+`WeightPageCache`=PleTable 模式有界 pinned LRU+epoch 免逐逐（分配器注入可全 host 测）; `--weight-host-bytes` **解析即硬拒**（P1 GEMM 钩子未落地前拒绝静默全驻留））+ `A_s32_mkpatch.py` + `A_s32_w13_p0.md`（热路径钩子=固定 device 页槽+原子位 hit 锁-free（PleTable 先例）/materializer.cpp:125 消费点/双预算决策, 全 file:line） | 干跑 **pristine + N1→n1b→s30 栈双 OK（U1 重锚后 0 fuzz）**; 主机检查 **10/10**（分类/LRU/epoch 免逐/字节镜像/**自抓异常安全 bug**: 原版先分配后逐出, 耗尽 throw 留脏态→改 make_room 先行） | GPU 窗口（P1 落地后）: 低显存强启不 OOM + 输出与全驻留**逐位同**（字节镜像, 针尖/定种子验证）; B/C: 复跑 `A_s32_check.cpp`（-I /tmp/s32patch/b/src）; P1=materializer/binder 消费+GEMM 钩子+cudaHostAlloc 注入一行 |
| S35 | DONE (patch 未应用) | D | 503 设计 **§3.4 源头治理(P2)**: `_collab/D_s35_frontload.md`（三执行体+worker 边界调用图闭包全量 throw 审计: (a)请求域可前移 10 处/结构缺口 1 处, (b)不变量保持, (c)arena/CUDA——**当前树执行路径已无请求几何直达 arena 尺寸的通路**, alloc-zero(:236 取证)/TT 守卫均已收口; 不可前移项逐个点名执行期输入）+ `_collab/D_s35_frontload.diff`（4 文件/161 行: 规划器 request_plan_impl.h **10 处 invalid_argument→RequestError 消息逐字不变**——sampling×3/prompt×3/context-length→ContextLengthExceeded/vision→InvalidMedia/rewrite+capture-frontier×3→新 kind InvalidPrompt(400 "invalid_prompt", 与 prepare 路径同 code); engine_core.h admission 两处 inspect_admission 折进 per-request try——**堵住"规划期异常无归因→整机停摆"**; types.h+generation_service.cpp 穷尽 switch 强制映射。规划期请求域错误外部身份 500→400（wait() 只 catch RequestError 实证 :424）, happy path 零变化, 全目标家族共用一份规划器）+ 生成器 `D_s35_mkpatch.py` | **静态门可跑**: `wsl.exe -e python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/D_s35_gate.py <root>` → shadow 树(S31+S35 已应用) **S35_GATE_PASS** / pristine 负对照 GATE-FAIL（门能抓未修态）; 生成器 → **S35_PATCH_DRYRUN_OK**（逐文件+combined 5/5）; **与 S31 叠加两向验证 offset 0 fuzz 0 reject**; 4 目标文件 pristine 未动 | A/M: 复跑生成器+门+读 md §1 审计表; 窗口 H: 与 S31 同涉 types.h **同批编译**+e2e（top_p=7→400 invalid_prompt 引擎存活; S31 四门一起回归; tok/s 零回退） |
| S38 | DONE (patch 未应用) | C | N3 轮2 捕获侧接线（与 M 共识后定稿）: `_collab/C_s38_calib_capture.md` + `patch/C_s38_capture.diff`（2 文件/160 行）+ `tools/kv_rowscale_sidecar.py` 增 `records`/`bake` + 合成帧 `C_s38_kvc_frames.py`。**接缝**=text_context_impl.h:955-993（kn=后rmsnorm+RoPE K, v, cache_positions, fidx, 流 s, Phase::Prefill 门; 同位 NINFER_KVDUMP_KV 块即证明）。**共识点**: ①部分覆盖编码=(a)bake 拒写部分表+(b)sidecar layers=观测集×S28 身份闸=超claim 结构性不可加载（O1 谎类写读双杀, 无 v2 头）; ②capture() 改**必传流** cudaMemcpyAsync+单次 Sync——裸 cudaMemcpy 对非 NULL 流是**静默错数据**非慢路径（代码内注释防"优化"回去）; ③**强制**校准界 NINFER_KV_CALIB_MAX_TOKENS=4096 累计/层（57K 跑飞不可能, 记录≈4100·T B, 默认协议 32 记录≈270MB）; ④批量序列 prefill **抛错**非警告（positions [width,batch] 会产出貌似合法的错位记录）; 校准跑 --no-cuda-graph（同 KVDUMP/FT 先例, 运行文档已注明）; 同步 ofstream 写保留=简单性+上限已强制 | patch -p1 --dry-run OK（双文件影子 apply 过, grep: Sync=1/上限=2/拒绝=1）; 工具 e2e 合成帧: full 16 层→bake→validate OK（scale [0.5,1.4609], **首版 RMS 平衡, 生产表是 Sinkhorn 约束, 质量待 GPU**）; 3/16→REFUSED 列缺失层 exit 1 无文件; 坏 magic→ISSUE | GPU 窗口: 按序 records→bake→validate→NINFER_KV_ROWSCALE 加载（md §6 全命令）; 待测: 真记录、首版 bake vs 常量表针尖对照、损坏/批量拒绝路径在引擎内 |
| S37 | DONE (patch 未应用) | B | `_collab/B_s37_stage_a.diff`（W7/P1 阶段 a, 6 文件/7 hunks: export 身份头+**形状派生几何交叉核验** `validate_stage_a_geometry`——spec 常量=期望, manifest 张量形状=artifact 侧唯一几何源（manifest 无语义几何, M 实读 reader.cpp:112-134）, hidden/vocab/q/kv/intermediate/layers 逐项断言, 缺失/不符抛错报双侧值; registry 识别分支把核验 PASS 判词拼进响亮拒绝; config.h `intermediate None→640`; CMake 注册; 旧 package.h `{{` 缺陷修）+ `_collab/B_s37_flashnext_p1.md`（接缝清单 9 项 file:line + 全树普查 + (a)→(f) 分阶段计划）+ 生成器 `_collab/B_s37_mkpatch.sh` | `patch -p1 --dry-run` **S37_DRYRUN_OK** 6/6; `g++ -fsyntax-only`: config static_assert 640/2560 **CONFIG_H_COMPILES** + export 头含核验体 **EXPORT_PKG_AND_CROSSCHECK_COMPILE**（host 检查实抓一真错: object_name 返 string_view, 已修）; 流程=骨架→M checkpoint→三项折叠共识（几何核验/context_cost 接缝 :210/:245/:55/:67/普查扩至全树 ~30 字段）, 见 md §0 | M: 复跑 dry-run+两 host 检查; 排序 (a)→(f) 已接受; **阶段 b 需 S28 writer 产核验所引张量名**; registry.cpp TU 编译归窗口 (CUDA 头); FP8 PLE 仅阻塞 (e)+端到端门 |
| S36 | DONE (patch 未应用) | A | Muse decode 腐败根因=**统一硬编码 256 家族**（覆盖 bf16+nvfp4 两类, 偶发=越界写跨平面竞态）: `_collab/A_s36_headdim.diff`（13 文件/**94 处经审计站点** `kGqaHeadDim`/`kGqaPrefillHeadDim`/nvfp4 Lead/Groups → `Geometry::HeadDim` 派生, 定义+smem 上界保留加警告注释; `prefill_i8` static_assert==4 结构性 256-only 列 (a)-blocked 不半修）+ `A_s36_sites.txt/sh`（105 站点分类表）+ `A_s36_h1_proof.cpp`（**主机数值证明**: Muse 128 双倍密度+head1 越行 8192B/页 入邻平面+src 整段 OOB; qwen 256 **110592+27648 偏移逐字节 old==new**）+ `A_s36_muse_decode.md`（H1-H4 裁决+bf16 透视+验收前置） | H4 行标定表 [0.5,2.0]+写读对称=算术否证; SWA 无 bf16 参数(0 hit)=否证; KVHeads==2 门仅 prefill=否证; (iv) `gqa_cache_index`(decode.cuh:37) 等全家族 256 vs 布局侧 head_dim=128 实证; `patch -p1 --dry-run` 13/13 OK（窗口 H 批中途落树后**重扫**站点, 漂移检查过）; 排除中途自身坐标错报(gap→越行)已自纠 | 下窗口: 应用+ccache 重建→ **Muse bf16 --max-new 4 零 NaN**(今 366) + KVDUMP 行界外字节未触(测竞态解释) + qwen 8/8 针尖不变; 未修二进制可测可证伪预测(失败不随绝对位置); M/C: 复跑 mkpatch+proof |
| S39 | DONE (实测, patch 未应用) | D | S34 冷策略对抗验证收官 `_collab/D_s39_cold_loop_verify.md`（原始命令+原始输出+PASS/FAIL+预测逐条判定）+ 独立检查器 `D_s39_checker.h`（判定只用原始观测+策略输出+cfg,绝不读 streak_at_decision/dwell_at_decision/rounds_this_window/invariant_violations; 含 6/6 自测: 控制组 0 违反 + 5 篡改全触发）+ `D_s39_adversarial.cpp`(11 场景+200x300 随机=60,000 决策) + `D_s39_cap01_probe.cpp`(预测 ii 扫描) + `D_s39_idle_probe.cpp`(真实 tap 差分均值) + `D_s39_net_vs_true.cpp`(计数口径) + `D_s39_dbg.cpp` + 修复提案 `D_s39_fix_proposal.diff`(未应用) + 门 `D_s39_gate.cpp` + 一键复跑 `D_s39_run.sh` + 完整日志 `D_s39_run_log.txt` | 实测(未修头 md5 7662e4, 295 行): **0 字面违反**; (b) 缺口 3 条(空窗冻结 streak / 低置信窗计入 streak / **真实 tap 空闲窗喂 (0.0,0) 把真能量恒定的层降级——提案只推迟不闭合**); 抖动 锚定族 318 与单槽族 263 flips/400c, 每次翻转五门全过(**不对称滞后**, 最坏 5724/4734 s/h > 3600 ⇒ 永久 reload 循环); "22 demotions"=正净增长口径, C 自己的流上少计 57%(net 681 vs true 1585); 无 (c) 重推导分歧(cut 差 5e-07, mirror==policy streak 0/12,837 分歧); 门 未修 FAIL(3)/提案 FAIL(1, 仅 G3 flips=106) | M: 复跑 `_collab/D_s39_run.sh` |
| S41 | DONE | B | 双树取证对账（build tree vs `ninfer-fusion-repo`）: `_collab/B_s41_tree_reconcile.md`（权威对账表）+ `B_s41_raw_diffs.txt`（原始 diff）+ `B_s41_probes/`（19 个可复跑探针）+ `B_s41_candidates/`（合并候选）。**方法**: 影子树重建（mirror+补丁序列 重放 → 与真 build 逐字节比）+ 补丁反向锚定；未写任一树、未按 mtime 定案 | **裁决 (三句)**: ①"build 权威"**成立**（21 受检文件 **19** 逐字节 = mirror+补丁）；②"agent 只出补丁"**证伪** —— **U1/U2/U3 三项 board 标 [DONE-A] 的修复只存在于 mirror**（`decoder_state.h` `kKvFp8QuantGroup` 256→16 / `gqa_attention.cpp` 白名单补 `DType::E8Kv` / `kv_options.h` U3 注释），build tree 完全缺失；③"只有 window 脚本应用"**双向证伪** —— `C_s23_tu_split/apply.sh:15` 默认 `NINFER_REPO=<mirror>` 且 window H `[4/6]` 未覆盖 env ⇒ **S23 拆分落进 mirror**（build 侧刻意排除 L3，日志原文 "9/9 patches apply cleanly against .../ninfer-fusion-repo"），另有 `_sync.sh`(mirror→build, `_changed.txt`) 与协调者手工编辑（nvfp4 门 / i8 `Groups`）两条**无补丁写路径** | 落地=**5 文件/3 步**: U2 `gqa_attention.cpp`+U3 `kv_options.h` 可整拷 mirror；U1 `decoder_state.h` 用候选 `904b7535…`（**保留 s24，不可整拷 mirror**）；`apps` `-rdynamic`/`src` `qpn_host` 两项**意图未定 ⇒ 暂缓**。A 类 26 文件 + s36 全 13 文件 + S23 六件（sha256 已列）**一律不动** | A/M: 复跑 `B_s41_probes/_b41_probe9.sh`（影子重建）+ `_b41_probe6.sh`（s36 反向锚定 13/13）；**J2 若报 `must use quant_group 16`=缺 U1、`invalid profile or interval`=缺 U2**；建议窗口脚本写死 `export NINFER_REPO=/home/user/ninfer-fusion` |
| S42 | DONE (A 独立复核; 未写 src/**、未用 GPU) | A | 两条 completion 的复核 + 修正补丁(未应用): `_collab/A_s42_completion_review.md` + `_collab/A_s42_i8_stride_fix.diff` + 锚定生成器 `A_s42_mkfix.py`（补丁=`i8.cuh` 新增 `gqa_i8_scale_row_clear<Groups>` / 4 处清零宽度 / 2 处 `cp_async<Groups*sizeof(__half)>` / 2 处 `cold_i8_decode_row<Groups>` + `cold_i8_kernels.cuh:72` 加 `template <int Groups=4>` 且外层 `g<4`→`g<Groups`） | **① `Groups=D/kGqaKvQuantGroup` 取值对**(tile 循环/Q 量化/q_scale 寄存器/`k_scale_s[Bc*Groups]`/`key_l*Groups+g` 寻址/V 组选择全自洽; `static_assert(QKKs==Groups*GroupKc)` 128=4=2·2 与 256=8=4·2 两侧成立)**但三处仍按 4 写死**: `decode_i8.cuh:450-451` `cp_async<8>` + `:438-439/:453-454` `make_int2(0,0)` ⇒ Muse `Bc=32,key_l=31` 写 half[62,66) ⊄ [0,64) = **smem 越界 4B + 覆盖邻键槽(同址不同值⇒cp.async 完成序未定义⇒间歇错尺度)**; `cold_i8_kernels.cuh:75-95` 外层 `g<4` 写 256 codes/4 scales 进调用者 `row_codes[128]/row_scales[2]` = **栈越界 128B+4B**(读侧值本身对); `kv_quant.cuh:47/:54/:103` 平面 LeadingExtent 仍 256 家族 vs 布局侧 `decoder_state.cpp:106-116` 的 head_dim 派生(128/2/64) ⇒ i8/E8 档整体 2×(越界写进池内相邻平面; 写读共用同一错 stride ⇒ 往返自洽, 单测看不出)——不在本次两文件内, 须与 `prefill_i8.cuh:367`(D 仍 256)/`:371` 几何化耦合, 故未放进小补丁 **② `if constexpr` 机制对**(丢弃分支不实例化是真规则; 128 下 `QKKs=2` vs `nvfp4.cuh:167 static_assert(QKKs==4)` 本会编译失败; 取用分支实参逐字未动=256 侧同)**但只堵了 decode**: prefill 侧 `prefill.cu:54-62` 的 `cudaFuncSetAttribute` 取址强制实例化(`:377/:431` Muse), × `prefill_nvfp4.cuh:1002 D=HeadDim`/`:1025 Mxf4QKKs=D/64=2`/`:1026` 断言 ⇒ **window J2 必在 `gqa_attention_prefill.cu` 编译失败**(`.o` 09:31:48 vs src 14:45:23 ⇒ 必重编; `_window_j2.sh:41` ⇒ MAKE_FAILED 不验证); 另 `decode_impl.cuh:489-503` 的**同宏副本无门**(被 `decode_muse.cu:15/:25` 引用; 今日 `src/CMakeLists.txt:77-78` 未列 muse/g35 ⇒ 不在本构建, 但 S40 拆 TU 正落此) **③ 验收会假 PASS**: `_muse_serve_accept.sh:52/:62-69` 只数 NAN 行、不看 text/HTTP ⇒ Muse+nvfp4 现在的"显式拒绝"、以及 13:44–13:48 四份 `cudaErrorIllegalAddress`(NAN=0 行)都会被判 PASS; `_e8_muse_check.sh:55-57` 三档 nvfp4 底 ⇒ 门落地后必 `MUSE_E8_VERDICT=FAIL`(**预期=档位不存在, 不是 256 修复失败**) | 下一个 Muse 缺陷排序(review §4, 每条带一条命令): **1 J2 prefill 断言(确定/现在) 2 上述验收语义(确定) 3 i8/E8 档 2×+两处宽度 bug(高; 换掉 nvfp4 底后立刻, 症状与今天同族) 4 SWA 对 Muse 默认 bf16 档完全无效(中; bf16/i8 decode+prefill 无 sliding_window 参数, >2048 静默语义差、无 NaN) 5 U7 page-fill E8 代码已在树(`prefill_i8.cuh:282-285`; 只剩改前/改后复跑, 被 2/3 阻塞) 6 行标定表=Muse 0-15 命中 qwen 表(`gqa_isoquant_row_scale.cuh:26-28` 的 16/4/256 界; 仅 nvfp4 消费 ⇒ 现被掩蔽) 7 `v_dtype=ISO3` 不对称(`decoder_state.cpp:316-322` vs `:374-380`; 写读一致 ⇒ 可比性混淆, 非坏)** | M 三条最便宜核对: `wsl.exe -e bash -c "python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/A_s42_mkfix.py"` ⇒ 末行 **S42_FIX_OK**(含 dry-run/apply/反向/shadow==new 与算术: Muse 数组 64 halves, 旧 8B 写 key_l=31→[62,66)=2 halves OOB, 新 4B→[62,64) OK; qwen 256 halves 两侧同宽); `wsl.exe -e bash -c "tail -40 /tmp/wj2_make.log"` ⇒ 期望 `prefill_nvfp4.cuh(1026): static assertion failed`; **A_s36_headdim.diff 已不可重放**(实测 13/13 文件、53 hunks 全 FAILED), 重放前先跑 `A_s36_mkpatch.py` 重生成 |
| S45 | DONE (patch 未应用; 未写 src/**、未用 GPU) | E3 | i8/E8 平面 LeadingExtents 的 256 家族收官: `_collab/E3_s45_i8_plane_stride.diff`（4 文件/11 锚点, `patch -p1 --dry-run` 通过）+ `_collab/E3_s45_i8_plane_stride.md`（18 站点消费者分类表 + 128/256 算术 + Muse 别名区域 + 门禁裁决）+ 生成器 `E3_s45_mkpatch.py` + 探针 `E3_s45_probe.py` | **① 分类**: 18 个平面索引消费者全部 **[G] 必须几何派生**（decode 融合 append 7 处 / 冷路径 cp_async 3 处 / fill+prompt 8 处, 含 3 处绕过 helper 的手写 `paged_kv_page_head_offset<kGqaKvQuant…>`）; 6 类 **[F] 合法固定量**（64 页格式 / 冷槽 256 容器 / 256-only tile 常量 / `PageIds=256` / 注释 / `decode.cu:611` reduce grid 惰性——已被 `d<Geometry::HeadDim` 兜底）; 补丁后族内 `paged_kv_page_head_offset<kGqaKvQuant…>` **0 残留** **② 修正+算术**: 4 项 LeadingExtent 全改 `Geometry::HeadDim` 派生（code 128/256、i4 64/128、scale 2/4、src 128/256）; **256 侧 798720/798720 偏移逐点相等 (0 回归)**, 128 侧 261888/262144 不等 ⇒ 真 2×; Muse 别名: `(kv_head h, page p)` 的写落在布局 **page h+2p 的整页**（双 kv_head + 全 64 page_offset）, `h+2p >= G` 的页 K 码整体写进 V 平面, V 码写包络 `49152G B` 超该层 4 平面集（`33792G B`）**15360G B**（G=256 ⇒ 3.9 MiB; 末层=越出池）; 往返自洽仅因读写共用同一错 stride **③ `prefill_i8`@128 裁决 = 响亮门禁, 不移植**: 256-only 在 tile 层（`SmemBytes==92672` / 4 d-consumers / `PVNtPerWarp==8` 在 128 下为 4）⇒ 须重推 smem 与 warp 划分, 无 GPU 不可验; 且 nvfp4/FP8 prefill 同为 256-only、`serve_muse.sh:25` 默认 bf16 ⇒ 不损失可用档; 门放 **两个入口**（append fill + attention）以免"先写后抛"; 句式仿 `gqa_attention_decode.cu:27-32 require_nvfp4_geometry_dim` | `wsl.exe -e bash -c "python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/E3_s45_mkpatch.py"` ⇒ `ANCHORS_OK 11/11` + `SHADOW_APPLY_OK 4/4`（影子树逐字节 == 目标文本）+ `LIVE_DRYRUN rc=0` + `S45_PATCH_OK`; `... E3_s45_probe.py` ⇒ 分隔符审计（3 对照文件 0 违例）+ 128/256 算术表 | ⚠ **阻塞发现**（`md §5.2`）: 15:07 落进 `src/ops/launcher/gqa_attention_prefill.cu` 的 4 处 launch 守卫**插在实参表内部**（`:142` 的 `}` 关闭 `:125` 开的块而 `()` 未闭合, `:147` 实参被孤立孤儿）⇒ **该 TU 无法编译**, 而 `src/CMakeLists.txt:79` 在编集内 ⇒ window J2 必在此失败; `PREFPILL_GUARD_OK` 只验了花括号平衡=真而不充分; 修复配方在 §5.2（mirror 26937B=守卫前文本, `diff -u` 仅这 8 处插入）; **我的补丁不含该修复**（属 M 的编辑, 混批会让 stride 补丁不可复核）——需要即出单独的守卫修复补丁 | GPU 窗口（只有它能证）: ① Muse i8/E8 修复后 prefill **响亮抛错**而非写坏 + decode 往返有限且正确 ② 未修二进制的别名预测=约半个页池后开始坏（不随绝对位置） ③ qwen 8/8 针尖**通过向量不变** ④ Muse bf16 `_muse_verify.sh`→`MUSE_VERIFY_PASS` 不变（S45 不碰 bf16 平面）|
| S45d | DONE (patch 未应用; 未写 src/**; **未开 nvcc/ptxas**) | E3 | **prefill 守卫 v2（逐臂）**, 取代 S45c: `_collab/E3_s45d_prefill_guard_v2.diff`（基座=pristine `b2da4c43…`/sha256 `4b8e0d68…`/453 行 → 结果 sha256 `ddf006de…`/484 行, 4458 B）+ 三件证明 `_collab/E3_s45d_prefill_guard_v2.md` + 生成器 `E3_s45d_mkpatch.py`（`--check` 可复跑）+ 可选伴生说明 `_collab/E3_s45d_s36_restore.md` | **① 形状（按你要求）**: 三个 dtype 分支条件**逐字保持**: `:110 } else if (cache.dtype == DType::NVFP4) {` / `:172 … ISO3) {` / `:200 … FP8_E4M3FN) {`, 每个臂体各自包 `if constexpr (256) { <原体> } else { throw nvfp4 prefill requires head_dim=256… }`（守卫 114/173/201, throw 167-171/195-199/223-227）; attr 组保持你认可的**无 else** 单守卫（`:58`, 4 条 odr-use 在 55/60/64/69, 理由: 该块在函数体顶层, bf16/i8 也会执行）`②` **diff 事实**: `diff pristine patched` ⇒ `^<`=**0**, `^>`=**31**（10 行 attr 组 + 3×7 行臂）；**无一行被修改**, 三臂体用 `sed` 抽取后逐行 diff = `NVFP4_BODY_IDENTICAL / ISO3_BODY_IDENTICAL / FP8_BODY_IDENTICAL`（52/21/21 行, 落在 patched 115-166/174-194/202-222）, 按你要求**未重排缩进** `③` **回归自查表**（md §c, 逐格带行号）: 256 → NVFP4 走原体（`<NVFP4,ISO3>` 133 或 `<NVFP4>` 150）/ ISO3 → 178 / FP8 → 206 / bf16 → 231 / i8 → 100 / E8 同 i8; 128 → NVFP4/ISO3/FP8 **各自 throw**（167-171/195-199/223-227）, bf16/i8/E8 **与 256 同**（S45d 不给 i8/E8 加门, 那是 `E3_s45_i8_plane_stride.diff` 的活; 两补丁行级不冲突, 可任意序）`④` **事实核对（不影响 v2 交付）**: S45c 的产物其实**没有**丢 ISO3/FP8 —— 它是链式的（产物 118 `if (… NVFP4) {` / 171 `} else if (… ISO3) {` / 193 `} else if (… FP8_E4M3FN) {` / 215 收链 / 216-220 if constexpr 的 else）, 所以"进入合并分支后不发射也不抛"需要那两个 `else if` 分隔行缺失才成立; 但只读 diff 确实看不出来（那两行不在任何 hunk 内）——这正是逐臂 v2 消除的歧义, 且逐臂后即使有人重排链也不可能静默穿透; 若你手上有已应用 S45c 的实测 trace 证明跳过, 发我, 我按 trace 查 | `patch -p1 --dry-run -i _collab/E3_s45d_prefill_guard_v2.diff`（对 `/home/user/ninfer-fusion`）⇒ `checking file src/ops/launcher/gqa_attention_prefill.cu` rc=0（原文已粘进 md §a）; 影子 apply == 目标字节; `E3_s45b_guard_audit.py`（不变式 A/B）在产物上 0 违例; 与伴生 `E3_s45b_s36_restore.diff` 链式 apply + 序列 dry-run 均 OK（`SEQ_DRYRUN_OK`） | 窗口 K（官方 make）: 该 TU 实编为最终判据（我不再自行编译; 我的编译树已按 PID 杀掉、未重启, `make ninfer -j1` 未受影响）; 之后 qwen 8/8 针尖通过向量不变 + Muse bf16 `_muse_verify.sh`→`MUSE_VERIFY_PASS` 不变 + Muse/任何 128 几何的 NVFP4/ISO3/FP8 prefill 抛可读错而非实例化失败 |
| S40 | DONE (patch 未应用) | C | `_collab/C_s40_tu_split_v2/`：**按当前构建树重推**的 L3 decode 巨 TU 拆分补丁集（9 patch，未写 src/）+ `_collab/C_s40_tu_split_v2.md`。`apply.sh` 改为**显式 repo-root**：`apply.sh [--dry-run] <repo-root>`，**无默认值 无 fallback**，bundle 内 `/mnt/`、`/home/`、`C:\`、`ziqinzhang` 命中 **0/0/0/0**；写入集先扫描（hunk 头非 9 个声明目标或含 `../` → REFUSED exit 2，**写前拦截**）。**核心发现：S23 落进 mirror 的那份拆分编译不过** —— mirror 全树 `require_nvfp4_geometry_dim` 命中 **0**（build 树 3），其 `nvfp4.cu` 由**非模板**函数（`:132/:147`）无条件实例化 `launch_nvfp4_for<GqaMuseGeometry>`，路径上仅 `if constexpr(writes_cache)` 与运行时 `if(iso3_v)`，两臂都实例化 ⇒ `HeadDim=128 ⇒ QKKs=D/64=2` vs `nvfp4.cuh:167 static_assert(QKKs==4)`，**该 TU 结构上无法编译**（主机 g++ 归约证明：带 Muse 实例 exit 1 / 去掉 exit 0）；S23 实为"协调者加守卫之前"712 行版的忠实重生成（737−712=**25 行 = 那道守卫**）⇒ 与 A/S42 所指 `decode_impl.cuh` 同宏副本无门**同一件事**，本次重生成把它补上（守卫在 `launch_nvfp4_tile` 内，Muse 零实例化 + 活体同措辞抛出）。9 文件：`decode.cu` 737→113、`impl.cuh` 618→591（重生成）、`tiers.h` + 5 新档 TU(i8/nvfp4/fp8/iso3/bf16)、CMake +5/−0 | `verify.py` **33/33 PASS (VERIFY_OK)**：把 `launch_tc_partial_*` 实例多重集**从活体宏与生成 TU 两侧独立展开**逐项比对（tier/TokenTile/Warps/Iso3V/MultiBatch/Masked **+实参串**）**144 = 144**（=768 档 launcher 实例，原先同处一次 nvcc）；另检守卫存在/仅一门内实例化/链路由/宽度门措辞逐字/CMake +5−0。`--dry-run /home/user/ninfer-fusion` → **9/9 clean exit 0**；影子树：`DRYRUN_WROTE_NOTHING`（全树摘要不变）→ 9×`APPLIED` → `diff -rq` **9/9 恰等于声明集 (CHANGED_SET_EXACT_MATCH)** → 重跑 `already applied` exit 0 → 应用字节==review 字节 → EOL 逐文件保持（impl/CMake CRLF，余 LF）；负控：坏根 / 无 src/ / 无参 / 篡改补丁指未声明文件 / `../` 遍历 **全部写前 REFUSED** | M: 复跑 `python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/C_s40_tu_split_v2/verify.py /home/user/ninfer-fusion`（期望 `VERIFY_OK`）+ `mkpatch.py /home/user/ninfer-fusion --check`（`CHECK_OK`）+ `evidence/path_audit.sh`（4 项 0 命中）。**本条不在 window H 的在编构建内**：原 737 行 TU 仍按 -j1 编（机器已两次因内存压力重启），拆分**走独立落地窗口**；board 里 window H 列表的"必需 s23"**对本拆分作废**（其余 11 补丁不受影响），且**不得**用 S23 的 apply.sh 落地（它仍会指向 mirror） |
| S40-d | DONE | C | S23 拆分语义审计（问题 3 的逐条答复，读 mirror 未写）：`evidence/mirror_delta.txt`。**结论：mirror 的 576 行 `impl.cuh` 不只"拆分"，它带着一个编译中断** Δ1=守卫整段消失（mirror 全树 0 命中，build 3）；Δ2=连带失去 128 档的运行时拒绝；Δ3=宽度门 `switch/default`→`if(width<1||width>6)`（同集合同措辞，行为等价）；Δ4=*cached* 档几何兜底由"静默走 35B"改为抛错（仅非法输入可达，有效输入零差异）。**非差异（=与活体一致，即 mirror 自称"从活体重生成"在这些点上成立）**：`ft::observe` 后置位置、i8 动态 smem 常量 `kGqaHeadDim`、reduce 段与 `I8\|NVFP4` 门、`single_row_batch_view` 字段、e8 交接 | 机械取证：guard 计数 0 vs 3；无条件实例化点与路径完备条件列表；几何常量 `GqaMuseGeometry=GqaGeometry<32,2,1,128>` + `QKKs=D/64` + `static_assert(QKKs==4)`；g++ 归约证明（exit 1 vs exit 0）；mirror↔S40 行数对照（impl 576/591、nvfp4 161/166、i8 128/144、decode 112/113、tiers 81/79） | A/M: 读 `_collab/C_s40_tu_split_v2.md §3`；**不要**把 mirror 的档 TU 覆盖到本 bundle（差的就是 Δ1 那道守卫）。附带记录（未改）：活体 `decode.cu` 的 i8 arena 仍用 `kGqaHeadDim` 而 S36 后的 `e8.cu` 同构处已用 `Geometry::HeadDim`（Muse 下 `DynamicArena=false` ⇒ 常量×0，惰性；留作 S36 后续站点）|
| S40-s | DONE | C | **范围声明（问题 5）**：本拆分**排除在在编构建之外**。正在跑的 window H 编的是**原始 737 行 `gqa_attention_decode.cu`（-j1）**；机器已两次因内存压力重启，不给该窗口增加任何新的 nvcc 调用。拆分作为**独立窗口**落地（落地时一并重配 `-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache`，拆分会把 TU 数乘起来）。落地无需 GPU 的三关：①`--dry-run` ②apply 后 `verify.py` ⇒ `VERIFY_OK` ③`nm` 分区检查（`decode.cu.o` 零档内 partial 符号 / 各档 `.o` 各含其己 / 每个 `gqa_attention_decode_*_launch` 恰好定义一次）+ 重链接 | `evidence/dryrun_build_tree.txt`（9/9 clean，exit 0）；`evidence/shadow_lifecycle.txt`（全生命周期 + 5 项负控）；`evidence/verify_output.txt`（33 检查）；`evidence/mkpatch_output.txt`+`mkpatch_check.txt`（锚定区间 + 每文件 sha256/行数/EOL + 确定性）；`evidence/path_audit.txt`（主机路径 0 命中 + 写出集 1:1） | M: 见 S40 行的复跑命令；GPU 窗口补 `_muse_verify.sh`→`MUSE_VERIFY_PASS` 与 qwen 32K 针尖不变（同核搬迁=行为一致性判据） |
| S44 | DONE (patch 未应用; 纯 CPU, 未写 src/, 未用 GPU) | E2 | `_collab/E2_s44_acceptance_counter.diff` + `_collab/E2_s44_acceptance_counter.md` + 生成器 `E2_s44_mkpatch.py` + 证据脚本 `E2_s44_syntax_check.py` / `E2_s44_fmt_check.py` | **事实更正**: 接受计数早就按请求累加且三后端同源（MTP `program_impl.h:12043-12049` / DFlash `:12232-12238` / DFlash2 `:12459-12465`；承载 `RequestControl::speculative_stats` @ `program.h:509`；三后端共用 `ops::speculative_accept_greedy_drafts`），**缺的只是带标签上报** —— 旧行 `4.55tok/round (50.9%)` 里的两个数就是接受长度与接受率。补丁=done 行加 `spec_drafted/spec_accepted/spec_accept_rate/spec_rounds/spec_fallback_steps`（+`spec_accept_len`）、JSON 加 `acceptance_rate`、并修 `apps/cli/main.cpp:221` 把 DFlash2 显示成 `mtp` 的错标；backend=None 仍只 `off`（零开销，不新增每轮工作）。定义: accepted=验证器保留的草稿前缀（每轮 +1 的纠正/bonus token 不计），部分接受轮只计其前缀 ⇒ `drafted>=accepted`、`rate∈[0,1]`、且 `gen-1 == accepted+rounds+fallback` | ①`patch -p1 --dry-run --fuzz=0` 三文件 clean exit 0 ②补丁后三个 TU 用 `build/compile_commands.json` 自身 flags `-fsyntax-only` **0 失败** ③补丁后格式化函数本机跑真数字 **0 失败**（dflash2 → `spec_accept_rate=0.5085`/`spec_accept_len=4.55tok/round`，与旧 `(50.9%)`/`4.55tok/round` 逐位一致；dflash → `0.1031`/`1.38`） ④`build/tests/ninfer_request_log_test` → `ok` | M: 复跑 `wsl.exe -e bash -c "cd /home/user/ninfer-fusion && patch -p1 --dry-run --fuzz=0 < /mnt/c/Users/User/Documents/ziqinzhang/_collab/E2_s44_acceptance_counter.diff"`；落地后每臂一条 serve（192-token greedy，**MTP 必须 `--spec mtp --draft-tokens 3`**，今天 `s4w_plain_mtp.log` 的 SERVE_FAILED 是 flag 校验而非引擎缺陷 ⇒ MTP 臂今天从未起来过）；判据见 md §5 (`drafted>=accepted`、`gen-1 == accepted+rounds+fallback`、`rounds+fallback == engine_timing.decode_rounds`) |
| S46 | DONE (patch 未应用) | E4 | F3b 修复: `_collab/E4_s46_cold_policy_fix.diff`（`patch -p1` 打到 `a/src/serve/kv_cold_policy.h`, 96 行/4 hunks: ①`rounds==0` 的空闲窗不再进 EWMA（tap 在 `count1<=count0` 时给 `(mean 0.0, rounds 0)`）②`rounds<floor` 的窗把 streak 清零（规范="EACH 窗"）③提升门改成"必须置信出现"——②的必要伴随, 否则层一空闲就被踢出冷池 ④注释）+ `_collab/E4_s46_f3b.cpp`（我自己的最小驱动, 空闲窗由被测头自己的 `window_from_cumulative` 产生）+ `_collab/E4_s46_regress.cpp` + `E4_s46_checker_fixed.h` + `E4_s46_mkchecker.py`（镜像与被测头同步的不变量回归, 逐处断言生成）+ `_collab/E4_s46_run.sh`（一键复跑）+ `_collab/E4_s46_run_log.txt`（314 行原始日志） | 建树 md5 **7662e4ad 未动**（`patch -p1 --dry-run` 对真建树 rc=0; 影子应用后 == `2b7d33ec`）。**F3b 我自建驱动**: before `FAIL(3)` / after `PASS(0)` —— 真能量恒 0.5、切点 0.4 的层原在 **t=4 被降级**（空闲窗 0.0 把 EWMA 拽到 0.28125, 并把切点从 0.4 拉到 0.3, 挤出真冷层 L3 给 L5 让位）, 现 **永不降级**; 与 D §4.3 数字逐条一致(t=4/cut 0.300/ewma 0.28125/`{0,1,2,5}`/持续族 flips 2→1)。**C 原测(未改)** 双侧 `TU_RESULT PASS`（demotions 22→23）。**D 对抗(未改)** 在同步镜像下: 0 违反流/0 重推导分歧/cut 差 5e-07, 随机 200x300 的 thin 类 spec-gap **归零**（普查 6110→1389 且 100% 为 F2 缺席窗）; 在过时镜像下 FAIL(4) 全为"镜像写死旧语义"（8b 4→5 与 D 提案头同值）。D 门 **G2 由 FAIL 转 PASS**（总 fails 3→2; 余 G1=F2 未修, G3=抖动未修）。**item 4 抖动不变**: F1 锚定 cap=4/5 仍 318 flips/400c(最坏 5724 s/h), F2 单槽 cap=1 仍 263(4734), before/after **diff 为空** ⇒ ">3600 s/h 永久 reload 循环" 结论原样保留, 仍需逐层翻转退避。**显式未修**: F2（缺席窗冻结 streak, `:129-138` 只遍历 observations）、A+（`rounds<floor` 不进 EWMA —— 会把长期 thin 层移出切点集合, 属独立语义变更, 留给 C/规范裁决）、8b 代价（低置信窗清 streak ⇒ 合法降级最多晚 1 窗 60 s） | M: 复跑 `bash /mnt/c/Users/User/Documents/ziqinzhang/_collab/E4_s46_run.sh`（重建+重跑+重写日志, 主机, 零 GPU） + 只读锚定 `cd /home/user/ninfer-fusion && patch -p1 --dry-run -i .../E4_s46_cold_policy_fix.diff`; 或直接读 `_collab/E4_s46_run_log.txt` / `_collab/E4_s46_cold_policy_fix.md` |
| S43 | DONE (只读调查；未改 src/，未用 GPU) — **根因已定，可只靠读码证明** | E | `_collab/E_s43_df2_verify.md`（+ 生成器 `_collab/E_s43_board_row.py`） | **根因: DFlash2 的 decode ingress 从不填写 `state_source_slots`/`state_destination_slots`** ⇒ 恒为 0（主机 ingress `*dflash2_host_ingress = {}` 零初始化 `program_impl.h:1068`，decode 填充循环 `:12396-12407` 只写 11 个字段），而 `state_image_store.h:401-417` 的 `selectors()` 返回的是 **device slot 号** ⇒ 每个 lane/每轮的 target verify 都把 GDN(48/64 层) 线性状态**从 state image device slot 0 读**、并把接续 hidden **往 slot 0 写**（消费侧 `dflash2_impl.h:371-372` → `speculative_target_impl.h:18,23,36`），而 KV 走正确的 `text_kv_table_rows` ⇒ **KV 与 recurrent state 互相不一致**；对照：ordinary `:11831-11832`、MTP `:11989-11990`、DFlash v1 `:12177-12178` 都填了（`grep -n 'dflash2_host_ingress->'` 全文无这两项） | 三症状同源闭环：①verify 列 0 logits 污染 ⇒ 停止符；且 **a=0 时发布列表里只有这一枚 target argmax**（`speculative_round.cuh:131-153`，oracle `tests/ops/test_speculative_round.cpp:104-119`）⇒ `finish=stop_token`、内容空、`gen=2`；②draft 的 context 特征来自**同一次**被污染 verify（`text_context_impl.h:872-878`）⇒ 提案系统性错 ⇒ 实测 `1.00tok/round (0.0%)`（0/7）；③prompt 依赖：重复型输入上被污染的 argmax 仍是数字 + GDN 对重复序列不敏感 ⇒ 那条 `4.55tok/round (50.9%)` gen=192 正常 | 日志算术（`gen=2, decode=31.4tok/s, decode_tokens=completion-1` `request_log.cpp:565`, `decode-host=31515us/round`）⇒ **只有 1 个 decode 轮** ⇒ 第 1 枚来自非 spec 的 Begin 行（`program_impl.h:1109-1119`），**停止符就是该 spec 轮列 0 的 argmax** ⇒ 基线 gen=129 vs dflash2 gen=2 是「第 2 枚 token 上 verify 分布 ≠ plain 分布」的硬证据；已排除：draft 上做 stop 检查（唯一判定点 `frontend.cpp:1078-1107`，只吃已发布 token）、max_new 计数（记账错只会 output_limit）、acceptance floor（需 drafted≥128 `spec_decision.h:45`，本请求仅 1 轮）、验尸器总接受（`target_argmax` 与 `drafts` 是互不重叠 region `round_state.cpp:178-213` + 实测 0%）、锚点错位（与 plain 同款 `ledger.back()`+`frontier`：`program_impl.h:12396` vs `:11826`）；**latent 契约违反**（非本次根因）：`dflash2_impl.h:186-193` 传 `attention_valid=width`（v1 传 `valid_columns=extent+1` `dflash_impl.h:223-228`）⇒ 无效尾列拿未来位置，违 `prepare_masked_block.h:20-27`+`gqa_attention.h:80-84`，但 `extent=7=k ⇒ 无 tail` 时不激活 | **Patch A（2 行，未应用）**: dflash2 ingress 填 `state_selectors(sequence).{source,destination}`（与 v1 `:12177-12178` 同构），append 路径 `:9733-9735` 同源遗漏建议一并补；**Patch B**（latent 1 行）: `attention_valid`→`valid_columns`；**探针**：`NINFER_DF2DBG` 一行（全文在 md §5.2，放 `program_impl.h:12437`、必须图外）可打出 `src/dst/tgt[]/drf[]/acc`；**下一步**：①落 Patch A 后同 prompt 复跑（期望 gen≫2、接受率回到 20-50% 量级）②若要事前证据先跑 md §5 实验 0（同 flag greedy 基线比 token1/2）+ 并发 2 请求放大 slot 串镜像；注：**`--spec dflash2` 宽度固定 7，`--draft-tokens` 只接受 0/7**（`src/product/speculative_options.h:39-44`）⇒ 窄窗兜底只能 `--spec mtp --draft-tokens 1` |
| S51 | DONE (patch 未应用; 纯 CPU; 未写 `src/**`; 未开 nvcc/ptxas/make; 无 GPU) | E8 | `_collab/E8_s51_nvfp4_silu.diff`（`patch -p1`, **1 hunk / +17 / −2**, 只动 `src/ops/common/math.cuh`；生成器 `_collab/s51_scratch/mkdiff.py`）+ `_collab/E8_s51_nvfp4_silu.md` + 探针 `_collab/E8_s51_silu_probe.py`/`.txt` + 独立交叉验证 `_collab/s51_scratch/cross_check.cpp`/`cross_check_out.txt` | **上游 PR #194 的机制在我们树里不存在**：全树 `__fdividef` 命中 **0**（含 third_party），`build/compile_commands.json` 的 153 个 `.cu` 无一带 `-use_fast_math` ⇒ 精确 `expf` + IEEE `/`；但**同缺陷换机制存在** —— `silu(x) = x / (1.0f + expf(-x))` 在 x ≤ **−88.72284** 时除数溢出为 `+inf`、`x / inf` 得 −0，而真值 **−2.6073e-37 = 22.18× bf16 最小正规数**（比上游的 −87.3365 晚 1.39，形态一样）。**全树只有 1 处 silu 定义**（`src/ops/common/math.cuh:13`）+ **66 个 device 调用点 / 18 文件**（kernel 19 · linear_swiglu 18 · linear 16 · sparse_moe 10 · gdn_input_proj 3）—— 上游的 file-local helper 在我们树里已被合并成共享定义 ⇒ 改 1 行修 66 处（与"硬编码 256 有 81 处"那族相反）。g++ **穷举全部 float32**：旧式静默错零 **1,998,749** 个值，新式反向错零 **0**，bf16 层 rescue **1,145,241** / regress **0** / 1-ulp 扰动 **2,432**（1.12e9 非零点的 2.2e-6），x∈[0,60) 逐位不变 **0 / 1,114,636,288**；两者都非零处 max rel 差 3.3e-7~5.0e-7（2.8–4.2 ulp）⇒ **"两者都有定义时完全相同" 是假的，必须写明**。错零窗口 = x ∈ (−88.7228, −97.4612]（下端恰是 bf16 舍入边界，上端是 22× bf16 最小正规数）。同族 `sigmoid`（`math.cuh:15`）同形态（边缘真值 2.94e-39 = **32× bf16 最小次正规**，窗口 (−88.72, −92.88]），**已作为同一 hunk 第二段给出、可删 7 行 `+` 丢弃**；`softplus`/`__expf` softmax 三处/`__frcp_rn` 十四处共 11 条判为**不需要**（§4.1 七条 + §4.2 四条，各带理由）。**未证实**：① sm_120a 上 `expf` 的确切溢出落点（没跑 nvcc/ptxas/GPU，只有主机两套 libm 互相印证到 88.7228394）；② 我们 kernel 是否真踩到（需数 `gate*alpha ≤ −88.72` 的实例，无 GPU；md §6.2 给了 `__any_sync` 计数器的最小实验）；③ 上游"SASS 完全相同"对我们形态**不成立**（我们多 ~1 FMUL + 1 SEL，无 SASS 证据）；④ **现有测试既证明不了也打不响这条修复**：`tests/ops/test_silu_mul.cpp:17` 的 gate 域是 ±12、`tests/ops/linear_swiglu/linear_swiglu_test_common.cpp:37` 的 A4 容差是 1.6e-1，都够不到 −88，落补丁后既不红也不绿 ⇒ **必须新增极端负值的回归门** | M: `wsl.exe -e bash -c "cd /home/user/ninfer-fusion && patch -p1 --dry-run --fuzz=0 < /mnt/c/Users/User/Documents/ziqinzhang/_collab/E8_s51_nvfp4_silu.diff"`（期望 exit 0 且 `md5sum src/ops/common/math.cuh` 仍为 `0ef644e3bd6b721b85b888ee67cac9af`；对 mirror 同命令亦 rc=0）；独立复算 `wsl.exe -e bash -c "python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/E8_s51_silu_probe.py"` 期望 §1 出现 `-88.72284 / -2.607329e-37`、§5 出现 `REGRESS ... : 0` |

### 事实更正 (M, 09:5X 核查)
- **窗口 B 续训是 CUDA OOM 崩的**, 不是被僵尸进程挤死: `dl/train-resume-b.out` 尾部是 backward 阶段
  `out of memory`, 09:15:32 exit -1。根因 = 与在跑的 nvcc/make 并行 (U8 的另一面)。
- 训练自 09:15:32 停止; 最新 checkpoint `data\dflash2_ckpts\step_000200.pt` (09:04:51)。
  **续训前置条件: `pgrep -a make` 与 `pgrep -a nvcc` 均为空。**
- GPU 占用表更新: 窗口 C 期间 GPU 空闲 (无 CUDA 进程), 当前唯一重负载是 WSL 侧 `make ninfer -j4`
  补编两个 decode TU; 该构建结束才会进入 Muse 验证 → e8 三档 → 续训。

## 主控 (M) 追加 2 — 2026-09-10 10:3X
### 事故与修复: 窗口脚本挂在会话上 ⇒ 被 SIGTERM 整树带走 (重要教训)
- 现象: 两个后台 exec 任务报 `exit code 15`; 复查 WSL ⇒ window C(13680)、make、两个 ptxas **全部消失**,
  `.o` 计数停在 5, `apps/ninfer` mtime 仍是 09:28:31 ⇒ **e8/decode 修复从未链接进二进制**,
  Muse 验证 / e8 三档 / 续训 **一步都没跑**。
- 根因: 窗口脚本是挂在会话上的后台任务; 会话被打断时 SIGTERM 连带整个进程树。
- 修法: `_window_d.sh` 用 **`setsid nohup`** 在 WSL 内脱离会话启动 (独立于 wsl.exe 客户端), 并加两道硬闸门:
  ① make 失败或无重链接 ⇒ 直接 `WINDD_FAIL` 退 3, **绝不用陈旧二进制做验证**;
  ② 因 ptxas 被杀在写中途, 先 `rm -f` 两个 decode `.o` 强制真重建 (半个 .o 的时间戳比源新, 会假链接)。
- 实况 (10:27 起): `make ninfer -j2` 已启动, 全程日志 `dl/window_d.log`。
### B/S15: O1 修复的独立复验 PASS 4/4 (零新泄漏)
- 删除测试: 手工种入带 `gdn_layers(){return 2;}` 的假 `config.h` → 复跑后 `CONFIG_H_GONE`;
- dense 回归: ctrl_2 与 MiniCPM5-1B 的 `config.h` **逐字节相同** (manifest 亦同);
- 泄漏检查: 两个 blocked 输出目录 `grep -rn Gdn` 零命中; leaves / bindings.stub 根本不生成;
- `gdn_layers()` 在 conv 注入下现在返回 **0**; `#error` 文本与声称格式逐字一致。
- B 的追加观察 O2 (已由 M 落地): blocked→clean 复跑会残留 `config.h.BLOCKED`; 现改为**互斥清理**
  (写 config.h 时删 .BLOCKED, 反之亦然)。
### FlashNext 契约审计 (N7 前置提前满足)
- 仓库自带 `model.safetensors.index.json` (34.3MB, 296,475 个张量名: model 296,443 / mtp 31 / lm_head 1) 已下载;
  `flashnext_bindings.py --audit-gguf` 正在对 74,520 项契约做全量匹配, 结果写 `_collab\M_flashnext_contract.md`。
### 脚本强化 (供窗口 D 使用)
- `_muse_verify.sh` / `_e8_postfix.sh` 已从"打印"改为"判定" (`MUSE_VERIFY_PASS|FAIL` / `E8_VERDICT=PASS|FAIL|REVIEW`),
  消除"模型缺失或 CLI 崩溃被读成 nan=0 通过"的假成功路径; 两脚本 `bash -n` 通过。

## 主控 (M) 追加 3 — 2026-09-10 10:5X (FlashNext 契约)
- **决定性测量** (`_collab\M_flashnext_prefix_probe.md`): 契约 74,520 条 vs 真实 296,475 键,
  原样命中 **1 条**; 加一层 `model.language_model.`→`model.` 规范化后命中 **74,174 条** / 消费 74,210 键。
  ⇒ 135GB 下完也导不进去的隐患, 在下完之前已定位并给出修法方向。
- 残余: 契约**量化尺度条目 = 0**, 而真实 NVFP4 四元组 73,728 组齐全 (48×512×3);
  `model.visual` 333 个张量 (= 视觉塔, 未建模, 多模态新范围); `mtp.*` 28 个待核对。
- 方法论盲区: `flashnext_convert.py` 的自检用契约自身生成的合成键 (自我一致性), 真实键零匹配也能"通过"。
- 已派 **S21 (agent D)**: 规范化入口 + NVFP4 伴生消费声明 + 自检改吃真实清单 + 20 个抽样键的"映射正确性"
  抽查 (不是"命中即可"), 证据 `_collab\D_flashnext_contract_fix.md`。
- 性能: 原 `audit()` 三重循环 (2,339 CPU-秒未完) 已由 `_flashnext_fast_audit.py` (set+bisect) 取代。

## 主控 (M) 追加 4 — 2026-09-10 11:1X

### S20 评审 (M 对 A 的 N1 补丁预案的核查)
- 结论: **可进下一个构建窗口**, 但要注意两点。
- 已核实 (实证, grep): `ninfer::product::kv_bit_budget_spec` 存在于 `src/product/kv_bit_budget.h:302`
  (第 3 参 `e8_limit` 有默认值 ⇒ 两参调用成立); `parse_kv_layer_storage` 声明于 `kv_options.h:43`,
  且 `serve_options.cpp:267` 已在用 `product::` 限定名调用 ⇒ `layouts_impl.h` 里同样写法可解析。
- **残余风险 1 (编译)**: 补丁只有 `serve_options.cpp` 做过 host 编译; `layouts_impl.h` 那半段
  (`TextConfig::full_attention_layers()` 可见性 + 命名空间) **未编译验证** ⇒ 安排为下个窗口的**第一条**,
  这样万一编译不过, 浪费最小 (先验 `serve_options.cpp` 那种便宜部分, 再验 header 段)。
- 残余风险 2 (范围): `apps/cli/options.cpp:144` 与 `apps/perplexity/main.cpp:115` 两个解析点未覆盖
  ⇒ `--kv-bit-budget` 目前只在 serve 路径可用 (记录在案, 不阻塞)。
- 设计决策 (接受): budget 与显式 `--kv-layer-storage` **互斥硬报错**; budget 解析时 `cold_cap=0`;
  `--cold-policy host` 当前派生 0 页且**没有 `--max-cold-pages` CLI 标志** ⇒ N2 需补 (已记 TODO)。
- A 的自查亮点: 首版 golden 断言用错, 被自己的机械校验抓住 ⇒ 矩阵式断言起效。

### S19 评审 (B 对 C 的 S17 的核查) — 有实质发现
- 4/4 PASS; 纠正计数: 测试实为 **27/27** (S17 行写的 26/26 少算一项)。
- **边界发现 (真发现)**: "beam-L ≥ chain" **不恒成立** —— 构建器有意不嵌套贪心链, 存在格
  (第 0 步后所有行 = -10) 使 chain EAL=4 而 beam EAL=0; L=1 时与链完全一致。
  ⇒ 建议硬化: **把贪心链作为种子放进树** (≤l 个额外节点), 使 `EAL_tree ≥ EAL_chain` 成为定理。
  已列入 C 的下一轮 (待其 S14 验证回合结束后派发)。
- 旗标校验 PASS: `beam:0/17/x/nonsense/4x/-1/"beam: 4"` 全部 exit 2 + 一行可读错误, 无静默回退。
- stdlib 纯净 PASS (import 后 `torch`/`numpy` 不在 `sys.modules`)。

## 主控 (M) 追加 6 — 2026-09-10 11:1X (事故 2: 内核 OOM killer)

- **根因 (实证, journalctl)**: 11:09:34 与 11:10:14 两次 `global_oom` 杀掉 **ptxas**
  (`total-vm:20362196kB, anon-rss:8191260kB`, `task_memcg=/init.scope`, `uid=1001`):
  两个并行 ptxas 峰值合计 ~23GB, 顶满本 WSL 的 24GB 上限 ⇒ 内核 OOM killer 出手, window D 同批死掉。
  这与事故 1 (会话 SIGTERM) 是**两个不同死因** —— 不要再把它们混为一谈。
- 处置: **window E** (`_window_e.sh`, 11:10:51 起, 依旧 `setsid` 脱离会话):
  ① `make ninfer -j1` (单个 ptxas ~14GB, 稳落在 23GB 内) + **最多 3 次自动重试**
  (make 天然从已落地的 .o 续跑, 不需要重编已完成 TU);
  ② 硬闸门保留: make 失败或无重链接 ⇒ `WINDE_FAIL` 退 3, **绝不用陈旧二进制做验证**;
  ③ 因 ptxas 被杀在写中途, 先 `rm -f` 两个 decode `.o` (半个 .o 的时间戳比源新, 会假链接);
  ④ 之后仍是: Muse 判定 → e8 三档判定 → 续训 W9。
- **教训 (U8 修正)**: 本机不只"构建与训练必须串行", **构建内部也必须串行** —— `-j2` 会顶满 24GB 被
  OOM killer 杀; `-j1` 是唯一稳的并发度。
- 代价: 两次死因合计烧掉约 1.5 小时构建时间; 已完成的 .o 全部保留 ⇒ window E 只需补 2 个 decode TU。

### 构建效率: M 的账与结构性修法 (2026-09-10 11:2X)
- 详见 `_collab/M_build_efficiency_plan.md`。要点: **make 不会从头开始** (已完成的 `.o` 全部保留);
  痛点是一格 **25–35 分钟**的巨 TU —— `gqa_attention_decode.cu` 在**一次 nvcc 调用**里实例化 7 个 KV 档
  × 每个 (width, MultiBatch, Masked) 组合, 且 `.o` 只在最后落盘 ⇒ 被杀在 30 分钟处 = 那一格全废。
- 三次死因是**同一根因**: WSL `.wslconfig` 天花板 24GB vs 单次编译峰值 (cicc + ptxas) 就顶满它。
- 已止血 (window G): 天花板 24→26GB、swap 16→32GB; `NVCC_PREPEND_FLAGS=--split-compile-extended=8`
  (device 编译分区 ⇒ 峰值随分区而非整 TU 增长, 且**不触发 CMake 全量重编**); `-j1` + 最多 4 次自动重试;
  后台峰值采样器 `_memwatch.sh` → `dl/memwatch.log`。
- 结构性修法留待下个构建窗口: **L2** ccache 作为 CUDA 编译器入口 (已装 4.11.3, 输入未变的重复编译 ~1 秒);
  **L3** 按 KV 档拆 TU (单格 25–35 分钟 → 5–8 分钟, 只重编被改的那一档; 峰值变小后构建与训练有望并行)。
### 窗口 H 批处理清单 (M 定稿, 2026-09-10 11:3X) —— 提高"每次重编"的性价比
今天为了 1 个 header 修复烧掉 4 次窗口 (C/D/E/F 全死) ⇒ 纪律改为**攒批**：一次编译、一次验证。

前置: window G 完成 (重建 + Muse/e8 判定 + 续训) 之后；顺序**前一步不通过就不做后一步**:

| # | 步骤 | 产物/命令 | 通过判据 |
|---|---|---|---|
| 1 | U7 修复前基线 (需 GPU, 与训练互斥故同窗口) | `bash _e8_muse_check.sh` | 产出 `MUSE_E8_VERDICT=…`; e8 低于 plain 属**预期**(§116c), SERVE_FAILED 也是有效结论 |
| 2 | L3 拆 TU (C/S23) | `_collab/C_s23_tu_split/` 补丁 + CMake 源列表 | `patch -p1 --dry-run` 全绿; 这是后续重编提速的前提 (单格 25–35 min → 5–8 min) |
| 3 | L2 ccache 入口 (可选) | `cmake -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache .` | 观察是否触发全量重编; 若触发则留到下次全量构建 |
| 4 | N1 `--kv-bit-budget` | `_collab/A_n1_patch.diff` (+ A/S22 的 `--max-cold-pages` 冷窗耦合) | 编译通过 + `[kv-bit-budget]` forensics 行 |
| 5 | U7 Muse page-fill 修复 | `_collab/M_muse_pagefill_patch.diff` (见 `M_u7_muse_pagefill.md`) | 步骤 1 的脚本复跑 → `MUSE_E8_VERDICT=PASS` |
| 6 | 一次编译 | `make ninfer -j2` (拆分后峰值随档变小, 可并行) | 硬闸门 `REBUILT OK` (无重链接 ⇒ 不验证) |
| 7 | 全判定 | `_muse_verify.sh` / `_e8_postfix.sh` / `_e8_muse_check.sh` / 57K 掉针 | `MUSE_VERIFY_PASS` + `E8_VERDICT=PASS` + U7 转 PASS + 掉针不退化 |
| 8 | 续训 | `_train_df2_resume.bat` | `dl/train-dflash2.log` 出现新 step 行 |

### 未落地项现状 (M 逐条核对, 2026-09-10 11:3X)
- **N1 `--kv-bit-budget`**: 补丁就绪 (A/S20), M 已核查符号 (`kv_bit_budget.h:302` / `kv_options.h:43`) 与
  `TextConfig::full_attention_layers()` 可见性 (`layouts_impl.h:138` 已在用) ⇒ 残余风险仅"未编译验证"。**待窗口 H**。
- **N2 冷窗**: Python/C++ 成本模型已就绪 (S12/S14); 但**引擎侧缺 `--max-cold-pages` CLI 标志**,
  且 `--cold-policy host` 当前派生 0 页 (A 实测) ⇒ 归入窗口 H 第 4 步一起做。
- **N3 运行时校准闭环 / N4 热温冷自动分配 / N5 W13 权重卸载 / N6 ngram 真表 GPU gather**: 未开工
  (N5/N6 是大件, 需单独立项; N3/N4 依赖 KV 侧先落地)。
- **N7 FlashNext**: 下载进行中; B/S21 正在修契约 (前缀规范化 + NVFP4 伴生 + 自检改吃真实清单)。
- **U7 Muse page-fill**: 补丁 + 协议就绪 (`M_u7_muse_pagefill.md`), 待窗口 H。
- **U6 大件**: W2③ fuse_draft/LABD 接线 / W7(等 ckpt) / W13 / Windows 移植 W-P1..P6 — 未开工。
- **§103 `sliding_window_tokens` 全局无赋值点**: 未开工 (Muse 39 个 SWA 层按全注意力读整条 cache ⇒
  语义超窗 + 显存/带宽白花); 修法明确 (从配置写进 `PagedKVLayerView.sliding_window_tokens` + 57K 回归),
  **建议排进窗口 H 之后的下一批** (它改的是注意力语义, 需要自己的掉针回归, 不宜与 U7 混批)。
- **其他解析点**: `apps/cli/options.cpp:144` / `apps/perplexity/main.cpp:115` 未覆盖 `--kv-bit-budget`
  (A 已标; 不阻塞 serve 路径)。
### 窗口 H 的定稿编排 + 一个关键时序发现 (M, 2026-09-10 11:2X)
- **时序发现 (实证)**: `build.make` 里 `.o` 规则显式依赖 `flags.make` 与 `includes_CUDA.rsp`
  （382 处引用）⇒ **开启 ccache 入口一定会触发全量 CUDA 重编**（冷缓存）。
  所以 L2 不能"顺手开"，必须与 **L3 拆 TU**（本身就要全量重编）**同批**做——一次把全量代价付清；
  之后每次改档只重编那一格（5–8 分钟），且输入未变的重复编译走 ccache 秒回。
- **编排已写成脚本** `_window_h.sh`（一趟做完，不再人肉分步）:
  `[1] U7 修复前基线(_e8_muse_check.sh)` → `[2] 全部补丁 dry-run（任一失败即整批中止, 绝不半途应用）`
  → `[3] 快照 src/`（`/tmp/pre_h_src.tgz`，一行命令可回滚）→ `[4] 应用 L3+L2+N1+U7(+S24)`
  → `[5] 一次编译（L3 落地则 -j2，否则 -j1；4 次重试）` → `[6] 三项判定脚本`
  → `[7] 续训`。支持 `SKIP_L3=1 / SKIP_L2=1` 分级降险。
- C/S23 已追加要求: 交付必须带 `_collab/C_s23_tu_split/apply.sh`（幂等、dry-run 感知、拒绝重复应用），
  因为窗口 H 是脱离会话的批量脚本，没有人在环里补救。
- 峰值实测（window G, 同一巨 TU）: cicc 峰值 ~7.8GB、整机 used 峰值 ~9.1GB，
  随后回落到 5.3GB（分区逐个处理、内存复用）；对比修复前 23GB+ 直接 OOM ⇒ **split-compile 有效**，
  且本 boot `journalctl -g global_oom` 计数 = 0。


### 结论修正 (M, 2026-09-10 11:3X): 详见 `_TODO.md` §123
- **LFM2 conv**: 我先前"全树没有独立短卷积叶子"的说法**作废** —— 那次 grep 被 `-First 12` 截断误导。
  实际 `include/ninfer/ops/causal_conv1d_silu.h` + `src/ops/wrapper/causal_conv1d_silu.cpp`(381 行)
  **已完整实现**且已被 GDN 路径使用; 真实权重头部实证 LFM2 形状 `conv.conv.weight=[2048,1,3]`
  (核宽 **3**)。差异只有 (a) 核宽 3 vs 算子 4 (可用最老抽头补 0 映射), (b) **SiLU 位置待实证**
  (HF `Lfm2ShortConv` 是 `x = conv(x) * B` 再 out_proj) ⇒ LFM2 属**小算子变体 + 层型接线**, 非大件。
- **N6 ngram**: `src/ops/ple/ple_table.{h,cu}` 的**真表 gather + 有界 pinned LRU 热行缓存**已实现,
  但**是死代码** (`src/ops/ple/` 之外零引用, 与 N3 的 `KvCalibrationCapture` 同病);
  引擎**已有**序列级前缀复用 (`generation_service.h:42 prefix_cache_hit_tokens`) ⇒
  N6 缺口 = **把 PleTable 接进 qwen4_exp 运行时 + 确认 GDN 循环状态下的前缀复用正确性**, 不是从零写内核。
- **方法论** (两处误判的共同根因): 能力判断必须 ① greps 不截断 ② 先 `-l` 列文件再逐文件确认
  ③ **同时 grep 消费者** (本仓库已出现两例"实现完整但从未实例化"的模块)。

### U7 补丁前提核实 (M, 2026-09-10 11:4X) —— 影响面确认只在 Muse
- 门控实测: `src/ops/launcher/gqa_attention_prefill.cu:230` 与 `gqa_attention_prefill_e8.cu:50` **两处一致**:
  `if (tokens >= 128 && Geometry::KVHeads == 2) {` ⇒ 命中 bulk page-fill 分支。
- qwen3.8-27b = 4 KV heads ⇒ 走另一条路 (与 §116d 的"qwen e8 走 decode/prefill"一致);
  Muse-Glimmer-30B = 2 KV heads ⇒ 只有它会进这个分支。
- ⇒ `_collab/M_muse_pagefill_patch.diff` 的影响面 = "Muse + e8 KV + append ≥ 128 token", 与 §116c 描述相符;
  验收用 `_e8_muse_check.sh` (已就绪, 判定式) 的修复前/后对照。

### A 三行交付 + 新发现: ColdPolicy::Host 是死路径 (M 汇总, 2026-09-10 11:5X)
#### S22 舍入修复 (已修, 待 C/S29 独立复验)
- `src/product/kv_bit_budget.h:137`: `(int)(x*100+0.5)` → **`std::nearbyint`** (半偶 vs 半上; `4.125*100=412.5` 处容量差 `layers` 单位)。
  最小复现前后: py `4.12/7.49` vs cpp `4.13/7.47` → 修后两侧均 `4.12/7.49`, 混合档完全一致。
- 扩展网格 (逐字复用 C 的 `C_s14_grid.sh`): 26 例/327 行 `EXTENDED_GRID_ALL_MATCH`;
  覆盖 C 的 8 行分歧表**每个坐标**的 phase-5 交叉: `DIVERGED=0`; 舍入抽查 8/8 `nearbyint==py round`;
  原 S14 网格回归仍 27/27。**未自证**, 已派 C/S29 用其自备 harness 独立复跑。
#### S24 逐层窗口表补丁就绪 (`_collab/A_s24_window_table.diff`, 3 文件 174 行)
- 通路与 `layer_residual` 完全同构: `TextConfig::is_swa_attention/sliding_window` →
  `DecoderStateSpec/PagedKVCacheLayout.layer_sliding_windows` → `plan_cache`(同款"短 span 直接拒绝"校验)
  → ctor → 两个 view 的 `.sliding_window_tokens`。
- **`sliding_window == 0` 的语义由代码回答 (不是我猜的)**: 5 处内核消费者全部 guard `sliding_window > 0`
  ⇒ 0 即"全注意力"; 且用 `if constexpr (requires ...)` 兜住无该访问器的目标 ⇒ qwen 路径**逐字节不变**。
- dry-run 3/3 可应用; A 的生成器自检还在交付前抓到它自己的一处**双重插入** bug。
- 验收已前置声明: Muse 8K/32K/57K 掉针**不得退化**, 否则回滚 (不解释)。
#### S25 `--max-cold-pages` 补丁就绪 (`_collab/A_n1b_cold_pages.diff`, 6 文件 126 行)
- 优先级**由代码决定**: `None` 恒强制 0 (否则冷槽是纯浪费的显存); `Window/Disk` 显式上限优先;
  校验风格与 `--cold-keep-tokens` 一致 (拒绝而非静默截断)。已验证可在 N1 之后叠加应用 (仅偏移)。
- **新发现 (今日第三例"实现完整但从未接线")**: `ColdPolicy::Host` **运行时零消费者** ——
  `program_impl.h` 只对 `Window/Disk` 放行 ⇒ 即使显式给上限, Host 也仍是 0 页。
  ⇒ 这正是 N2 剩下的设计点 (要么接线 Host, 要么删掉该策略), 已记入 N2 待办。
### 构建效率的量化证据 (window G, 与修复前同一巨 TU)
- 采样峰值 (`dl/memwatch.log`, 20s 粒度, 92 样本): **used 16.4GB / ptxas 14.6GB / cicc 7.8GB**;
  对比修复前"两个并行 ptxas 合计 23GB+ 触发 `global_oom`" ⇒ 峰值降到 ~70%, 且本 boot OOM 计数 **0**。


### LFM2 conv 结案 (源码实证, 见 `_TODO.md` §125)
- `causal_conv1d_fn(..., activation=None)` 的默认值就是 None, 且 LFM2 的调用**没传** ⇒ **无 SiLU**;
  `conv_bias=false` ⇒ 无 bias; `conv_L_cache=3` ⇒ 核宽 **3** (引擎算子宽 4)。
- 完整复合块: `out_proj( C * conv( B * x ) )` —— `in_proj` 出 B/C/x, **卷积前后各一道门**, 且 conv 输入是 `B*x`。
- ⇒ **不能直接复用** `causal_conv1d_silu`; 需 **no-SiLU/宽3 变体 + 两道逐元素门 + 新层型接线** (内核机制已具备)
  = **中等工作量**, 规格可施工。我先前"大件"(无算子)与随后"直接可复用"两种说法**都不准确**, 以本条为准。

### 大批交付 + 四条新主线 (M 汇总, 2026-09-10 12:2X)

#### 已交付 (每项都带机械证据)
- **B/S27 写出路径**: swizzle 问题用源码定案 —— **码字不需要 swizzle** (modelopt 的 U8 就是引擎的行主序
  [N,K/2]、偶数在低半字节), **尺度需要** (引擎 `blockscale-k16-m128x4-v1` 是 512B 的 128×4 分块,
  逐字取自 `nvfp4_gemv.cuh:87-95`), 而 `tools/artifact/layouts.py::encode_nvfp4` **已实现**该分块 ⇒ 无需新代码。
  尾 F32 = `1/weight_scale_2`, `input_scale` 走 plan 元数据。**真实 layer-0 专家张量往返 12/12 逐字节相等**。
  460 行形状回填机制已落 (`--measure-shapes`, 现回填 18/460, 其余受下载阻塞)。
  排除表后果: NVFP4 **只覆盖 73,728 个路由专家**; 其余 (attn/GDN/router/shared/hyper-conn/mixer/mtp/embed/lm_head) 原样 BF16;
  **新发现: PLE 表本身是 FP8 存储** (`model-plefp8-*`, 10 片未到, 落地后再定子布局, 不猜)。
- **A/S30 预算×冷容量耦合**: 147 行, 仅 `layouts_impl.h`; 应用顺序 N1→n1b→s30 **已用 dry-run 逐步证明**;
  `Host` 恒 0 写进代码 (不绕过); `cold_cap` 恒等于实际预留池; 冷区间→NVFP4 热窗 + forensics 提示。
  守卫: `cold_cap=0` 时热表与 S14 golden **逐字节一致**; S14 parity 复跑 26+8 例 **0 分歧**。
- **C/S23 拆 TU 补丁集 + apply.sh**: 5 个按档 TU (`i8/nvfp4/fp8/iso3/bf16`), 实例化数: decode ≈48 (→<1 分钟),
  i8 ≈330 (8–12 分钟), nvfp4=36, bf16/fp8/iso3=144 ⇒ **-j2 峰值 < 修复前单 TU 峰值**。
  apply.sh 幂等 + 全量 dry-run 前置 + 保留各文件 EOL。**副产发现**: 既有 `impl.cuh` 已漂移 (只被未编译的 muse/g35 TU 引用)。
- **C/S28 N3 第一轮**: 旁车格式 `NINFERKVRS1` + env 门控加载器 (`NINFER_KV_ROWSCALE`), 硬报错不回退;
  host 测试 **12/12 PASS**; **身份门拒绝对 52 层模型套用 16 层表** ⇒ 我发现的"Muse 套用 qwen 表"问题被做成显式拒绝。
- **C/S29 舍入复验**: 4/4 PASS —— 网格复跑一致 (26 例/327 行/0 分歧); 我方外扩 60 例 **0 分歧**;
  **12,050 点跨语言稠密比对 `nearbyint == py round` 0 不匹配** ⇒ A/S22 可去掉"待复验"。
- **D/S31 (W16)**: **关键发现: P0+P1 其实早已落地** (`51a7a8f`/`487562a`/`a7ca4dd`, 设计文档已过期);
  D 补上 4 处真实残留: ① `/health` **脱敏** (原先 `what()` 原文挂在**免鉴权**的线路上!) ② 载荷完整性
  ③ 分配类 `runtime_error` 由 invariant 改判 resource (证据 arena.cu:72/167-171/327) ④ lane 失败遥测加锁序。
  故障注入钩子**已在树里** (`NINFER_FAULT_INJECT=request_once|invariant_once`)。
#### 新派四条主线 (都 CPU-only 可开工)
- **S32 (A)**: W13 权重 host 卸载 P0 (复用 artifact mmap + KV 冷窗 pinned 通路 + PleTable 有界 LRU 模板)。
- **S33 (B)**: `PleTable` 接入 `qwen4_exp` 运行时 (N6 的实体; W2① 已证 gather 可用, 但它是死代码)。
- **S34 (C)**: 热/温/冷**闭环** (N4): 观测→周期决策→重布局, 带不变量与 dry-run 模式。
- **S35 (D)**: W16 §3.4 **校验前移**审计 (今天那次"1..4 token 尾巴 → 执行期抛错 → 永久 503"的根治)。
#### window H 批处理已扩到 **8 个补丁** (顺序已定)
`s23(拆TU) → n1 → n1b → s30 → u7 → s24 → s28 → s31` + ccache 入口; 必需 = `s23/n1/u7`, 其余可选。

### 独立复现 (M 亲自跑, 非 B 自证): FlashNext 契约审计 (2026-09-10 12:2X)
`wsl.exe -- python3 _flashnext_audit_recheck.py`（跑 archkit 的 `--audit-gguf` 打真实 296,475 名清单）:
```
contract_entries     74804
engines_covered      74804     <- 契约条目全部有源
missing_count        0
matched_sources      74931
quant_companions     221184    <- NVFP4 伴生 (推导消费, 未丢弃)
unmatched_count      360       <- 残余: visual 333 + mtp 27
accounting_ok        True      <- 74,931 + 221,184 + 360 = 296,475
missing_by_family    {}        <- 无任何族缺源
residue_by_bucket    {visual, mtp}
complete             False     <- 诚实: 残余未消费完 ⇒ 不是"全绿"
```
⇒ B/S26 的声明**独立成立**。两点需要显式记录（不能只留在桶名里）:
1. **visual 333 张量的处置是我们的策略决定**: FlashNext P1 目标是"纯文本最小栈"(W7), 视觉塔暂不导入。
   已记入本行, 并在 B 的下载后配方里作为已知未消费项列出 —— **不是**静默丢弃。
2. **mtp 27 是引擎侧待设计项** (MTP 头: 24 层 + mixer + fc_embedding); 引擎已有 MTP 算子
   (`mtp_pack.h`/`mtp_round.h`) 与 dflash2/dspark 的 MTP 先例 ⇒ 属"接线 + 设计", 待派。

### 接受率 eval 的路径/解释器坑 (M 冒烟发现, 2026-09-10 12:3X) —— 已在 GPU 窗口前拆掉
- 症状: 用 **WSL 的 python** 跑 `eval_ddtree.py` ⇒ `cache files: 0` +
  `FileNotFoundError: C:\...\data\Qwen3.8-27B/model.safetensors.index.json`。
- 根因两条, 都不是脚本 bug:
  1. `train_dflash2.py:47` 的 `CACHE_DIR` 走 `DF2_CACHE` 环境变量 (默认 `data\hs_cache`, 与实际的
     `hs_cache_topk2` 不同) ⇒ **必须导出 `DF2_CACHE`**, 否则采样到 0 个 cache 文件;
  2. `TARGET_DIR` 是**硬编码 Windows 路径** (`train_dflash2.py:48`), 且 `eval_ddtree.py` 用
     **Windows 侧 torch** 读 teacher 的 lm_head/embed ⇒ 该脚本必须在 **Windows python** 下运行
     (WSL python 解析 `C:\...` 必然失败)。`data/Qwen3.8-27B/` 本体存在 (index + 5GB safetensors) ✓。
- **我的 `_df2_accept_eval.sh` 有同一个坑** (里面调 `python3`), 已重写:
  `PY="/mnt/c/Program Files/Python312/python.exe"` + `export DF2_CACHE='C:\...\hs_cache_topk2'`
  + 日志落到工作区内 (两侧都能读) + 保留"训练存活则 ABORT"闸门。语法已通过。
- 教训 (与 §123 同类): **工具脚本的"能在哪跑"和"参数对不对"必须先用真实数据小规模试一次**,
  否则这种坑会等到 GPU 窗口才爆, 而 GPU 窗口是最贵的资源。

### S33 重大范围发现: `qwen4_exp` 运行时只是**桩** (B 列出缺失符号, 未编造) + eval 冒烟结果 (M, 12:3X)
#### B/S33 (FlashNext 运行时接线) — 交付与新发现
- 交付: `_collab/B_s33_ple_wiring.diff` (新文件 `src/targets/qwen4_exp/impl/ple_runtime.h`, 155 行),
  `PleRuntime` 持有唯一的 `PleTable`; sidecar 根目录三级发现 (显式覆盖 > artifact 邻接 `ple-root/` > OFF);
  **默认关闭 + 响亮失败** (配了但缺 manifest ⇒ 抛错并指明路径与关闭方法; 全链无零填充);
  `gather_batch` 输出 `[16×160=2560, T]` BF16 (= hidden 大小), 进入 layer 1 的 PLE 栈。`patch -p1 --dry-run` OK。
- **关键发现 (未编造, 逐条列为"不存在")**:
  1. **qwen4_exp 运行时只有 2 个文件的桩** ⇒ gather 的消费者**尚无代码**, 本补丁只是"接缝";
  2. **没有 qwen4_exp 的 registry dispatch** (`construct_target` 会抛 "no registered target");
  3. 桩里的 `config.h:11` 写着 **`intermediate = None`** = **非编译 C++**, 第一个 include 它的 TU 就会编译失败
     (PLE 的旋钮目前只是注释);
  4. `EngineOptions`/serve CLI **没有 PLE 字段/标志**;
  5. **没有 EOS id 载体** (无前端);
  6. sidecar 契约是 BF16, 而 checkpoint 的 PLE 表是 **FP8** ⇒ 转换器要物化 FP8→BF16, 或算子要加 FP8 行路径。
- ⇒ **结论 (修正 FlashNext 路线图)**: 权重导入(契约+写出器)已近就绪, 但**运行时是空的**; "FreeToken 加载 FlashNext"
  剩下的主体工作 = 实现 qwen4_exp 运行时 + registry + config 修复 + CLI/EOS/PLE 通路, **不是**接线一行的事。
  这条必须写进汇报, 不能沿用"架构就绪/权重缺失"的旧表述。
#### 接受率 eval 冒烟 (端到端, 真实 checkpoint)
- **通路验证成功**: `_df2_accept_eval.bat` (Windows 侧设 `DF2_CACHE` + Windows python) + `.sh` (编排+判定)
  跑完三段 (chain/beam/tree), 产出 `DECISION` / `GATE` / `tree EAL` 行; 日志 `data/df2_accept_eval.log`。
  关键证据: 日志出现 `cache files: 480 (sampling 8)` ⇒ **WSL 环境变量不跨界的问题已绕过** (改在 .bat 里设)。
- **早期结果 (诚实)**: 用 `step_000200` 的 draft ⇒ `GATE ... -> NO-TREE`, `teacher hit@1..16 = 0.0000`,
  `mean rank of teacher token = 11406`, 天花板诊断 `TOTAL 0.2575` (与采集期 25.2% 一致)。
  这与"只训了 200/6000 步"相符; 但**注意本次是 `--cpu` 模式**, 真数字必须用 6000 步 ckpt 在 GPU 上复测。
- 结论: GPU 窗口的接受率判定**已去风险** (命令、通路、判定逻辑全部实测过), 只欠"训练到 6000 步"。

### S34 交付 (热/温/冷闭环) + window H 扩到 9 补丁 + 内存实测 (M, 2026-09-10 12:3X)
#### C/S34: N4 闭环 —— 先找出 W5 骨架的两个缺口, 再设计
- 缺口 (实证): ① 既有 `kv_auto_relayout` 只重排 **dtype** (e8/iso3/nvfp4 三分位), **从不决定冷驻留**;
  ② `ft::snapshot()` 是**进程启动至今的累计均值** ⇒ 追不了相位变化。
- 设计: 用**快照差分**导出逐窗能量 (`(sum₁-sum₀)/(count₁-count₀)`, 无需改 `ft_stats.h`), EWMA(α=0.5) 平滑,
  用三道闸门把最冷的层填进冷池 (容量 = A/S30 的 `effective_cold_pages`, 经 `NINFER_FT_COLD_PAGES` 传入):
  ≥32 轮观测 / 连续 2 窗低于 25 分位 / 驻留 ≥3 周期; **深层保护层与未观测层永不降级**, 晋升故意快 (质量优先)。
- **不变量机器校验**: 12 条随机 200 周期流 = **2400 次决策、22 次降级、0 违反**; 池上限与"深层从不进冷"全程成立。
  (C 还纠正了自己一处错误断言 —— 忽略了"槽位回填"。)
- 补丁: `src/serve/kv_cold_policy.h` (纯宿主 std C++) + `kv_auto_relayout.cpp::loop()` 一处锚定挂点;
  **默认 dry-only** (只打 `[ft][cold:dry] would cold=[…]`), live 模式**刻意不接线** (需要 S30 的池消费者,
  盲接等于静默 no-op)。策略测试 15/15 PASS (host g++), dry-run OK。
- 代价上界: 32K 上下文下每层 nvfp4 热窗 32MiB, 最坏动作 (cold_cap=8) ≈256MiB ⇒ PCIe 地板 ~16ms,
  真实停顿时长取决于 drain (≤120s 上限) + 图重捕获 ⇒ **每周期最多一次动作**。
- 诚实标注: `dl/` 里**没有**真实 `[ft]` 数据, 样例是合成的; 真数字要等 GPU 窗口的质量矩阵。
#### window H 批处理: 9 个补丁
`必需 = s23(拆TU) / n1 / u7`; `可选 = n1b / s30 / s24 / s28 / s31 / s34`; 顺序已定, 语法通过 (13 处 PATCHES 登记)。
#### 内存实测 (构建期, 26GB 上限)
- `available 8071MB`, swap 仅 2MB, 无其它大消费者 (agent 侧工作不占内存);
- ptxas 出现 **20.2GB 瞬时峰值**后回落 15.4GB ⇒ 当前配置 (26GB 上限 + -j1 + split) **有足够余量**, 不需要为
  "20GB" 再动刀; 若后续 TUs 更重, 可把 split 从 8 提到 16/32 (代价只是编译更慢)。

### S32 交付 (W13 权重 host 卸载 P0) + window H 扩到 10 补丁 (M, 2026-09-10 12:4X)
- 新增 `src/product/weight_residency.h` (宿主可编译, 无 CUDA 依赖): `WeightResidency{Resident,Host,Disk}` +
  逐 (layer,expert) `WeightSpan` + 计划载体 `MaterializationPlan.weight_residency` (与 `layer_residual`/
  `layer_sliding_windows` 同构; **留空 = 今天的行为**, 物化器不动) + 加载期确定性分类 (先专家、再稠密由深到浅)
  + `WeightPageCache` (有界 pinned LRU + PleTable 的 epoch 豁免, 分配器**注入**以便 host 单测)。
- `--weight-host-bytes` **可解析但在 P1 的 GEMM 钩子落地前会响亮报错** —— A 明确写了理由: "被静默忽略的卸载开关
  正是 W13 要禁止的失败模式"。
- 热路径**只给设计不动手**: 固定设备页槽 (绑定的指针不变 ⇒ 无需逐 GEMM 追指针; 稳态成本 = 一次原子驻留位读,
  无锁, 有 PleTable 先例); 加载期消费点在 `materializer.cpp:125`; P1 待决一项 (epoch 关闭挂到 stream 事件)。
  预算与 KV 冷窗**分开记两本**, 并给出完整的耗尽语义表 (正常 LRU 驱逐; 超大 span / 仅在途 / 页竞技场耗尽 ⇒ 抛错)。
- **证据**: dry-run 在 pristine 与完整 N1→n1b→s30 栈上都通过 (U1 后重新锚定消除了首版一处 fuzz);
  host 检查 **10/10 PASS 且 -Wall 干净** —— 且该检查**抓到 A 自己第一版的真 bug**: `fault()` 先分配后驱逐,
  预算耗尽抛错后缓存超预算、span 仍驻留; 已修为**先 `make_room()` 再分配** + 单调使用序 LRU 保证受害者确定。
- 验收前置声明: GPU 窗口须显示"低显存 serve 能起而全驻留 OOM"且**输出与全驻留逐位一致** (byte mirror ⇒ 同 logits/token)。
#### window H 现为 10 个补丁
必需 `s23/n1/u7`; 可选 `n1b / s30 / s24 / s28 / s31 / s34 / s32`; 顺序固定, 语法已通过。

### window G 落地结果: 重建成功, 但两项判定暴露真问题 (M, 2026-09-10 12:5X)
#### 构建: **成功**
- `REBUILT OK (1789003711 -> 1789015094)`; `apps/ninfer` 12:38:14 重链接 (889,458,776 B)。
  两 TU 全程无 OOM, 峰值瞬时 20.2GB / available ≥8GB ⇒ 新配置 (-j1 + split + 26GB 上限) **经受住了考验**。
- **训练已恢复**: `step 201/6200 loss=2.1692` (PID 37808, 12:52:44), 从 step_000200 续跑; ETA ~4h (~16:50)。
  (注: window G 脚本自带的 resume 步骤**没能把训练留住**——`start /b` 起的子进程随脚本/工具调用结束被清掉;
  改用 `powershell Start-Process` 才真正独立。这条要写进工具纪律。)
#### 发现 1: e8 三档判定失败是**我的脚本 bug** (已修)
- 报错实证: `[error] ninfer-serve: unknown argument: kv-dtype nvfp4 --kv-layer-storage 0-7:e8`。
  根因: `_e8_postfix.sh` 里三行写成 `run e8_0_7 "kv-dtype nvfp4 --kv-layer-storage 0-7:e8"` =>
  整串被当**一个参数**且**漏了前导 `--`**; 而 `plain_nvfp4` 是分开传的 ⇒ 只有它正常 (8/8)。
  **这个 bug 从最初就在**: 也就是说"e8 三档对照"此前**从未真正跑过 e8**。
- 已修 (分开传参 + 前导 `--`), 语法通过; 下一个 GPU 窗口重跑即得真数据。
- 顺带: `--kv-layer-storage` 确实存在 (`serve_options.cpp:266`) 但**未列进用法文本** ⇒ 小文档缺口。
#### 发现 2: Muse + nvfp4 KV 产出 NaN (真问题, 已定界到 decode)
- 证据 (`/tmp/mv1.err`): headdbg 两遍 —— pass1 (prefill) `L00_attn -0.208 0.145 …` **正常**;
  pass2 (decode) `L00_attn -150528 -1.466e+06 1.97e+06 …` **爆到 1e6**, 随后 `L01..L51` 全 `nan` (共 118 行)。
  功能探针输出乱码 ⇒ `MUSE_VERIFY_FAIL`。
- **定性 (待实证)**: §116c 记载 Muse 默认 KV 表是 **BF16**、nvfp4 路径"从没被踩到"; 本次我们用
  `--kv-dtype nvfp4` 主动踩了它 ⇒ 更像**先前就存在的 Muse+nvfp4 路径缺陷**, 而非今天的回归。
- **决定性实验 (下个 GPU 窗口, 一条命令)**: 同 prompt 下
  `--kv-dtype bf16`(默认) vs `--kv-dtype nvfp4` 的 headdbg 对照 —— 若 bf16 正常则缺陷定界在
  "Muse + 量化 KV", 并直接关联 §116c 的 page-fill 缺陷与 N3 的"外来标定表"问题 (Muse 0-15 层在套用 qwen 的表)。
- 相关: `_e8_muse_check.sh` 已就绪 (含 `plain_bf16` 档), 与上述实验同窗执行。

### 协作协议 v2: 把"自己跑自己的"从机制上消掉 (M, 2026-09-10 13:3X) — 用户第二次点名后的结构性修改
- **诊断 (我的责任)**: v1 隐含的模式是 `满配简报 → 长自主回合(20–45min) → 交付 → 事后复核`;
  讨论只在回合两端发生, 中间无人参与。今日 7 个被抓到的缺陷里 **5 个作者自查、2 个事后复核、0 个被提前截住**。
- **v2 规则 (已写入 `_collab/PROTOCOL.md`)**: 一回合拆两段 ——
  **P1 思路段 (≤10min, 必须早返回)**: 只交 ①假设/计划(每条带 file:line 或可执行命令)
  ②"我可能在哪儿错"(≥2 条) ③一个能区分前两个假设的最便宜实验; 返回后**停下等 M 批评**。
  **M 的批评 (义务)**: 核实其 file:line 断言是否属实(自己 grep)、补缺失角度与我已知的事实、
  点出该类工作的常见错误模式、给出我认为最可能的答案让其反驳。**M 未回应 ⇒ 不得进入 P2**。
  **共识 → P2 执行**: M 写一句共识进 board 后才开工; P2 中若发现 P1 假设被推翻必须立刻返回。
- **本轮已按 v2 执行**: S36/S37/S38/S39 四条全部发了 CHECKPOINT 指令(早返回);
  其中 S36 我**先给出了自己的 4 条假设排序**(H1 反尺度读错平面 / H2 KVHeads==2 的 ISO3-V 路径 /
  H3 表越界 / H4 外来行标定表——我判 H4 最弱)并要求 A 逐条反驳 + 给出区分 H1/H2 的最便宜实验;
  S37 我预先提示"artifact 里是否有 per-layer geometry"(因为它 S27 的写出器就是将来要产出它的地方);
  S38 要求论证"捕获是否扰动热路径"与"部分覆盖如何在旁车里诚实表达"; S39 要求先给检查器设计
  并**预测会在哪儿破**(我押"dwell/streak 阈值在容量 0/1 时的相互作用"与"晋升过快导致跨阈值抖动")。
- 目标: 错误在**第 5 分钟**暴露, 而不是第 45 分钟。

### 待落地清单 + Muse 定界反转 + window I 启动 (M, 2026-09-10 13:4X)
#### ① 全量待落地清单已落文件: `_collab/M_pending_all.md`
A–G 七类共 **50 项**, 逐条标 已备/在跑/待做/待决/阻塞/已完成。用户要求"先全列出来", 已满足; 后续勾对以此表为准。
#### ② Muse NaN 的**定界反转** (实证, 推翻了 A/S36 的前提)
- `_muse_kv_compare.sh` 实测: **bf16 也 NaN (366 行)**, 而 nvfp4 239 行 ⇒ **与 KV 量化无关**
  ⇒ A 的 H1 (硬编码 256 vs Muse 128) 仍是**真 bug**(会毁 Muse+nvfp4), 但**不是**本次现象的原因。
- 按"遍"切分 (`_muse_nan_passes.py`, 一遍 = 一个 decode 步):
  | dtype | pass0(prefill) | pass1 | pass2 | pass3 | pass4 |
  |---|---|---|---|---|---|
  | bf16 | 正常 | 正常 | **L01_mlp 起 NaN** | 全 NaN | 全 NaN |
  | nvfp4 | 正常 | **L00 爆 1e6**, L02 起 NaN | 全 NaN | **又正常** | 又爆 2e8 |
- 三个新事实: ① 与量化无关; ② 只在 **decode 步**出问题 (prefill 永远正常); ③ **间歇性** (nvfp4 pass3 自愈)
  ⇒ 更像**读未初始化/陈旧内存**, 而非确定性布局错位。
- 已把反证 + 逐遍数据发给 A 并**改派方向**: 转攻 bf16 情形(无量化、更干净), 我的新首猜是
  "52 层/2 KV 头下 Muse 专属的 decode 路径"(SWA 索引算术 / `KVHeads==2` page 分支 / 位置-页记账),
  并给它两个最便宜的判别实验(`--max-new 1` vs 4 是否第一步就坏; prompt 长度 ±1 看破坏跟位置还是跟步序)。
- **重要事实**: 此前从未用"判定式"检查过 Muse (S1 的验收"重建后无 NaN"一直是 DOING) ⇒ 这可能是**长期存在的缺陷**,
  而非今天引入的回归; 这一点在汇报里必须说清。
#### ③ window I 已启动 (测量优先, **训练最后**), 训练已按纪律暂停
- 训练: 杀 pid 37808 (PowerShell `.Kill()`), 显存释放到 375 MiB; 最后 checkpoint `step_001200.pt` (日志到 step 1275)。
- `_window_i.sh`: [1] Muse bf16 vs nvfp4 对照 (已完成, 判定 REVIEW 正确) → [2] **当前 ckpt 的接受率机制判定**
  (step_001200, 300 anchors; 用于回答"接受率是否训练问题": 若 hit@1 随步数几乎不动即支持用户主张) →
  [3] 交棒 `_window_h.sh` (U7 基线 → 应用 **12 个补丁** → 一次编译(含 ccache) → 三项判定 → **最后恢复训练**)。
- window H 补丁数: **12** (必需 s23/n1/u7; 可选 n1b/s30/s24/s28/s31/s34/s32/s35/s38), 语法已通过。
#### ④ C/S38 按共识交付 (三个"交付前拦截"全部落地)
- `capture()` **强制要求 stream** (`cudaMemcpyAsync` + 一次 sync), 代码里写明"对非空 stream 用普通 cudaMemcpy
  是**静默错数据**而非慢路径"; **强制**逐层 token 上限 (`NINFER_KV_CALIB_MAX_TOKENS` 默认 4096) ⇒ 57K 失控结构上不可能;
  批模式 prefill **直接抛错**; 部分覆盖在写入与加载两端都被拒 (bake 拒写 + layers==观测集 + S28 身份门)。
- 证据: dry-run OK (两文件影子树)、grep 证实 1 sync/2 cap/1 硬拒、合成 `.kvc` 端到端
  (16 层 → bake → sidecar 32832B → validate OK; 部分 → REFUSED; 坏 magic → ISSUE)。
- 诚实标注: 烘表数学是**首版 RMS 平衡**, 不是生产用的 Sinkhorn 约束表 —— 工具自己会打印这条 caveat。

### 接受率机制数据 + 两个流程教训 (M, 2026-09-10 13:4X)

#### 接受率机制数据 (step_001200, teacher 口径, 300 blocks) — 用户的"非训练问题"主张有了实测支撑
| 指标 | 值 | 读法 |
|---|---|---|
| teacher hit@1 | **0.2719** | gate 要 ≥0.35; **比 step200 的 0 高** ⇒ 指标确实随训练上升 |
| hit@2/4/8/16 | 0.3276 / 0.4076 / 0.4943 / **0.5238** | hit@16 = 16 宽选择的**天花板**(教师 top-16 召回) |
| chain headroom (teacher) | 0.7281 | 1 − hit@1 |
| **ORACLE-GAP** (row_hit4 − path_hit4) | **+0.0686** | **单链验证浪费掉的树材料** |
| tree EAL (S17, beam:4, 28 节点/块) | chain 0.6200 → beam4 **0.7333**, **Δ=+0.1133 tok/块** | 树相对单链的**离线**期望接受长度收益 |
- 结论表述(两面都写): ① 指标随训练在动(step200 的 0.0000 是 8-anchor CPU 冒烟, 不可比); ② **但存在与训练无关的收益**:
  行式(oracle)比单链高 6.9 点 hit@4, 树比单链多接受 0.11 token/块 ⇒ §112④"树材料被单链浪费"**已量化**,
  这是"接受率低不全是训练问题"的直接证据。
- 注: tree 的数字是**离线 builder coverage**(行式无 pair codebook), 真正机接受率仍需引擎内复测。

#### 教训 1: 一个脚本 bug 要**grep 整个脚本族**
- window H 的 U7 基线打印出老式报错 ⇒ `_e8_muse_check.sh:51-52` 有与 `_e8_postfix.sh` **同款** bug(整串一参数+漏 `--`)。
  我早先只修了其中一个。已修并写入 in-code 说明。
- 家族排查结果: `_kv_matrix_57k.sh:56` 与 `_mtp_width_probe.sh:39` 是**合法**的(前者 `start_serve` 用不加引号的 `$1`
  且带 `shellcheck disable=SC2086`; 后者 `$KVARGS` 同样不加引号展开) ⇒ 只有两处真 bug, 都已修。
#### 教训 2: 不要为同一个 20GB 模型开两个实例
- 我在 window H 正跑 Muse 基线时启动了 Muse 判别实验 ⇒ **31,854 MiB 用 / 334 MiB 空闲**, 两个实例互相挤压。
  处置: 按 **PID 精确杀**判别实验的 CLI(先读 `/proc/<pid>/cmdline` 核对), 显存回到 10.6GB 空闲。
- 附: 我的一次 `pkill -f '_muse_discriminators.sh'` **把我自己的 shell 也杀了**(命令行里含该字符串) —— 铁律②的活教材;
  之后一律按 PID 杀。
- 判别实验**改到 window H 的编译期重跑**(那时 GPU 空闲, 且不与任何 Muse 实例冲突)。

### 重大根因确认: decode 索引 helper 硬编码 256 (A/S36 的 (iv), M 独立核实) — 2026-09-10 13:5X
#### 根因 (我自己读代码核实, 非转述)
- **几何本就支持 128**: `src/ops/kernel/gqa_attention_geometry.cuh:11-15` 的 `GqaGeometry` 带
  `static_assert(HeadDimValue == 128 || HeadDimValue == 256)`, 且 `:27 GqaMuseGeometry = GqaGeometry<32, 2, 1, 128>`
  (HeadDim=128) ⇒ 几何参数里**一直有** HeadDim。
- **但 decode 的索引 helper 用的是全局常量**: `gqa_attention_decode.cuh:22 inline constexpr int kGqaHeadDim = 256;`
  被用在 `gqa_cache_index`(:37)、`gqa_q_index`(:43)、`gqa_kv_new_index`(:51)、`gqa_partial_acc_index`(:60)
  以及 :176/:208 ⇒ **bf16/plain 的 decode 也用同一套 2× 拉伸**。
- **布局侧**按 `head_dim=128` 每头分配 ⇒ 写读都在错的 2× 偏移上:
  head-1 的行落在 `[16384, 32768)` ⇒ 越界写进**同层 V 码平面**区域, decode 读 V 时踩到。
- **prefill 为什么看着正常**: 它有自己的 `kGqaPrefillHeadDim = 256`, 且写读**在同一错偏移下自洽** ⇒
  本层注意力结果自洽, 垃圾只是落到别处 (这也意味着 prefill 的偏移**不可能**因此免责, 见下)。
- **间歇性来源**: 同一 kernel 里越界 K 写与 V 写**竞态** (在 seam 处重叠) ⇒ 哪边字节活下来取决于
  warp/block 执行顺序 ⇒ 逐遍不同、偶尔"自愈"(nvfp4 pass3)、随后又爆 (bf16 首个 NaN 在 L01_mlp = V 坏了,
  而不是 L00 数学错)。
#### 影响面
- **任何 head_dim ≠ 256 的模型在 decode 都被毁**; qwen3.8-27b (256) 恰好无恙 ⇒ 解释了"只有 Muse 坏"。
- Muse 的 bf16 与 nvfp4 **同时**中招 ⇒ 这才是今天 Muse NaN 的统一根因; A 先前找到的 nvfp4 三兄弟
  (`kGqaKvNvfp4HeadDim/CodeLead/ScaleLead`) 是**同族**, 需一并修。
- 同族规模: `kGqaHeadDim|kGqaPrefillHeadDim` 全树 **81 处** ⇒ 修法必须**逐处分类**(stride/shape 改几何派生;
  真硬件常量改名+注明; 无关项排除), 不允许只修"顺手找到的那几处"。
#### 处置
- 已给 A 共识: ① 81 处分类表; ② prefill 不得自动免责(要说明哪些消费者能观察到错位);
  ③ 验收双向 (host 偏移证明 + Muse bf16 无 NaN 且原污染区未被触碰 (KVDUMP 字节) + qwen 8/8 掉针不变)。
- **进入下一个窗口**: window H 的编译已开始, 不在本批; 但 window H 打开了 ccache ⇒ 下一次编译会便宜很多。
- 意义: 这是今天最重要的引擎修复候选 (很可能解释一个"已发布模型一直坏着"的事实), 且它是**确定性根因**,
  与"训练/精度"无关。

### 四路交付 + 一轮批量修复 + 一个时序陷阱 (M, 2026-09-10 15:1X)

#### 四路交付的关键结论 (都带机械证据)
- **S40 (C)**: `apply.sh` 已重写为 `apply.sh [--dry-run] <repo-root>`（零主机路径、包含性检查、幂等、EOL 保留），
  对**当前**构建树重新派生（live `decode.cu` 实为 **737** 行，我那次编辑 +25），验证 33/33 PASS、
  且**实例化多重集 144=144 完全相等**。**关键警告**: 镜像里那份重生成的 `impl.cuh` **丢失了我的 nvfp4 几何守卫**
  （`require_nvfp4_geometry_dim` 在镜像里 0 次）⇒ S23 的包是"守卫之前"的版本，**照抄会回退 Muse 修复** ⇒ 必须用 v2 从 live 树重派生。
- **S41 (B)**: 我的"子代理只出补丁"假设**被推翻**——**U1/U2/U3 三项已标记 DONE 的修复只存在于镜像**：
  `kKvFp8QuantGroup 256→16`、`gqa_attention.cpp` 的 `DType::E8Kv` 白名单、`kv_options.h` 的文档。
  最小修复清单 3 步（含 `decoder_state.h` 需**合并**而非覆盖，否则回退 s24）。
  并给出"缺 U1/U2 时测试会报什么错"的判据（`packed KV cache must use quant_group 16` / `invalid profile or interval`）。
- **S39 (D)**: 冷窗策略的独立对抗复核 —— 预测三条全中，且**发现比预测更强的真缺陷 F3b**:
  空闲窗（`mean 0.0 / rounds 0`）会经 `window_from_cumulative` 进入 EWMA 而**无置信度门**，导致
  **真实能量从未变化的层被降级**；提案修法**没关掉这个洞**（需"低于置信度 ⇒ 不更新 EWMA"）。
  另量化: 未修时最坏抖动 > 3600 s/h ⇒ **引擎永远追不上自己的重布局**（设计级问题）。
- **S42 (A)**: 我那次 `Groups = D / kGqaKvQuantGroup` 改动**值对但实现不全**（3 处仍按旧宽度 4 写）:
  `cp_async<8>`/`store_vec` 在 `Bc=32` 时 key 31 越界 4B 且覆盖下一 key 的槽（间歇错尺度，同源症状）;
  `cold_i8_decode_row` 仍 256 硬编码而调用方已声明 128 ⇒ **栈越界 128B+4B**;
  `gqa_attention_kv_quant.cuh` 的平面 LeadingExtents 仍是 256 家族 ⇒ **i8/E8 档在 Muse 上整体 2×**（读写同错自洽 ⇒ 回环测试看不出）。
  另外: **prefill 侧也在实例化那个 256-only 内核**（4×`cudaFuncSetAttribute` 的 odr-use + 4 次 launch）⇒ 我原先只挡 decode 不够。

#### 本轮已应用的修复 (一次性, 趁 prefill TU 尚未编译)
1. **U1** `kKvFp8QuantGroup = 16` ✓ (构建树原本缺)
2. **U2/U3** 从镜像拷回 `gqa_attention.cpp` (29409B) 与 `kv_options.h` (4019B) ✓
3. **A/S42 stride 补丁** 应用成功 (2 文件) ✓ —— 但见下"时序陷阱"
4. **prefill 侧守卫**: 8 处 (4 attr + 4 launch) 全部包上 `if constexpr (HeadDim == kGqaKvQuantHeadDim) else throw`
   ⇒ 8 guards / 花括号平衡 / `PREFPILL_GUARD_OK` ✓（首轮脚本误把 ISO3 attr 的续行当 launch，已按文本锚定修复）
5. **验收脚本语义修正**: `_muse_serve_accept.sh` 原版只看 NAN 行数 ⇒ **明确拒绝**与**cudaErrorIllegalAddress 崩溃**
   都会得 0 NAN 而误判 PASS; 现要求 启动成功 + HTTP 可解析 + 文本非乱码 + 0 NaN + 无 CUDA 错误,
   并把 REFUSED（几何不支持）单列为一态, 不算 PASS。

#### 时序陷阱 (必须记住)
- 在**编译进行中**改动了被编译 TU 依赖的头 (`decode_i8.cuh`) ⇒ `.o` 会在**之后**写成 ⇒ 比源新 ⇒
  make 视为最新 ⇒ **本轮二进制不含 stride 补丁**(静默)。计划: 本轮判定跑完后 `touch` 那两个头强制重编该格,
  并与 L3 拆分同批（反正都要重编）。
- 反面: 改**尚未编译**的 TU 的源/头是安全的 ⇒ 这就是为什么 U1/U2/U3/prefill 守卫能在本轮生效。

### dflash2 的 serve 级实测 (M, 2026-09-10 15:3X) —— 用户三问的答案

#### 三次调用才走对（两次错都在我）
1. `--spec dflash` ⇒ `object handle does not name a materialized tensor`：`dflash` 选的是 **DFlash(dspark 路)**，
   它去找 `dflash/*` 对象，而产物是 `dflash2/*` ⇒ 句柄物化失败。（错因是我的 flag，不是产物缺陷）
2. **不给 `--spec`** ⇒ 后端停在默认 **`SpeculativeBackend::None`**（`types.h:126`），
   `Package::resolved_auto_speculative` 从不触发 ⇒ `speculative=off`（两次跑的都是普通解码 ⇒ "速度相同"是平凡的）。
3. 正确: **`--spec auto`**（或显式 `--spec dflash2`）⇒ 自动按产物 `weights_id=nvfp4-dflash2` 选 DFlash2 ✓
- **文档缺口 (新)**: `parse_speculative_backend` 实际接受 `mtp|dflash|dflash2|auto`
  (`src/product/speculative_options.h:12-16`)，但 serve 的 usage 只列了 `mtp|dflash` ⇒ 使用者极易踩上面两个坑。

#### 数字（同一 prompt、同 192 token 生成、并发 1）
| 运行 | speculative | decode tok/s | wall |
|---|---|---|---|
| dflash2 (`--spec auto`) | **dflash2** | **141.6** | 1.52 s |
| dflash2 (`--spec dflash2`) | **dflash2** | **141.2** | 1.52 s |
| 基线（plain 产物，无 spec） | off | **53.8** | 3.41 s |
⇒ **dflash2 实测 ~2.6× 加速**。
- 短 prompt（2 token 那次）显示 29.7–31.4 tok/s：投机开销在极短生成上占主导，属正常。

#### ⚠ 同时暴露一个质量缺陷（比速度更重要）
- 同一短 prompt：**基线正常输出**（"杭州位于中国东南沿海、浙江省北部…"），
  而 **dflash2 路径 `gen=2 stop_token`、`content` 为空**（`reasoning_content` 仅"用户"2 字）。
- 两次 dflash2 运行（auto 与显式）都如此 ⇒ 不是 flag 问题，是**该路径本身**：
  最可能是 **verify 接受了错误的 token（恰为停止符）** 或停止条件被扰动。
- **这对"接受率"的含义**：引擎**不打印接受率计数**（`grep accept` 无产出），但从 2.6× 加速反推，
  真实接受率**不可能只有 0.2**（0.2 的话 7 宽度草稿只会在盈亏平衡附近徘徊）⇒ 用户记忆中的 ~0.2
  很可能来自**另一种口径**（例如语料基线 hit@1、或旧产物、或按位置统计的方式）。
- 需要加一个**接受率计数器**才能给出精确数字（accepted/drafted per round）——列入待办。

### S47 · dspark/DFlash 为何慢 + 10.3% 接受率归因（E5, 2026-09-10, CPU-only, 未写 src/）

| ID | 状态 | 负责人 | 产物 | done 判据 | 验证者 / 命令 |
|---|---|---|---|---|---|
| S47 | P2 建议待 M 定序 | E5 | `_collab/E5_s47_dspark_slow.md` + `_collab/E5_s47_dspark_ingress.md`（+ 只读探针 `E5_s47_dump.py`/`E5_s47_cmp.py`） | 三问均带 file:line 证据 + 一条可判别实验 | M: 读两 md；跑 slow §3 的 E-1（3 条命令，约 3 分钟 GPU） |
| S48 | DONE (patch 未应用; 纯 CPU; 未写 src/**, 未开 nvcc/ptxas) | E6 | `_collab/E6_s48_dspark_verify_pos.diff` (1 hunk, +13/-2, 只动 `dflash_impl.h`) + `_collab/E6_s48_dspark_verify_pos.md` + 生成器 `_collab/E6_s48_mkpatch.py` + 自检 `_collab/E6_s48_syntax_check.py` | **修 DSpark(DFlash v1) verify 位置表少一列**: 共享表 `frame.proposal_positions` 原先按 `prepare_masked_block(..., V=k)` 建 (`dflash_impl.h:223-231`), verify 自己的契约却是 `target_valid_columns=extent+1=width` (`program_impl.h:12170`) ⇒ 列 k 得 position F+k-1, 与列 k-1 同址。补丁 = 建表时 `V` 取 `width`(列 0..k-1 逐字节不变, 列 k 变 F+k; MTP `program_impl.h:11980` 与 DFlash2 `dflash2_impl.h:187-189` 都是 F+j), 之后把 `attention_valid` 复位为 `k` 只喂草稿注意力(5 层 V 不变 + `packed` 仍取列 0..k-1 ⇒ draft token 逐位不变)。KV: 列 k 与列 k-1 撞同一 cache slot (`small_t_bf16.cuh:150-168`, 存活者由同一 CTA 写序决定), a=k 轮 slot F+k 无人写; 但本回合 K/V 按 key 下标读 `input`(`:240-249`) ⇒ 列 0..k-1 不被污染。**更正 E5 §1.3**: `accepted by pos` 第 m 位 = drafts[m], 由列 m 判定 (`speculative_round.cuh:134,207`); 列 k 只在"该轮全接受(a=k)"时供 `t_star=row_targets[a]` ⇒ 这条**不是** 10.3% 的成因, 是修好 H1 后的正确性前置。**可证伪**: 整场无 a=k 轮 ⇒ token 流与 `accepted by pos` 必须逐字节不变(否则扰动到了前 k 列); 有 a=k 轮时只有 bonus token 与 slot 映射变化 | M: `wsl.exe -e bash -c "cd /home/user/ninfer-fusion && patch -p1 --dry-run < /mnt/c/Users/User/Documents/ziqinzhang/_collab/E6_s48_dspark_verify_pos.diff"` (期望 exit 0 且源文件 md5 不变 `60a51d1c69c73da6117ec55a858549a6`); 落批后按 md §5 用 `--spec dflash --draft-tokens 1`(最敏感: 任何被接受的草稿都会换掉 bonus) 与 `7` 各跑一条 greedy 对 `accepted by pos` |
| S50 | DONE (patch 未应用; 纯 CPU; 未写 src/**; 未开 nvcc/ptxas) | E7 | `_collab/E7_s50_kv_coverage.diff`（4 文件 26 hunk, +46/-38: `logical_kv_store.h` + `program.h` + `program_impl.h` + `test_context_store.cpp`）+ `_collab/E7_s50_kv_coverage.md` + `_collab/E7_s50_regression_sketch.diff`（store 级边界用例, 单 hunk +36 行） | **移植上游 `03177b9`（kv coverage 下界化）**: `materialize_to_tokens`→`ensure_mapped_to_tokens`; 删 `target < page_count` 那半谓词, 改 `target <= address.page_count` 早返回(不裁剪/不 unmap); 错误消息补 tokens/required/mapped/reserved/entitlement(`logical_kv_store.h:1494` 起, 并加 `#include <string>`)。调用点: `program_impl.h` 9 处 `materialize_sequence_kv`（8 调用 + 1 定义: `:8149 :8711 :9699 :10320 :11416 :11834 :11993 :12180 :12415`）+ 5 处直接调用（`:1168 :10328 :10330 :10852 :11308`）+ `program.h:1282` 声明 + 12 处测试, **参数逐个相同、只改标识符**（`diff -u -w -B` 归一化后 program.h/program_impl.h/test 三文件 rc=0）。**风险结论: 老 throw 不 load-bearing** —— 12 点里 11 点因 `create_active` 后 `page_count==0` 或轮首 `page_count==pages_for_tokens(已提交 frontier)`（上一轮 `:9985` 的 trim 保证）而结构性不可收缩。**唯一可达收缩点 = `:9917`→`:11416`** DFlash 终止结算的 context append: verify 在 `:12180` 已映射 `base_E+extent+1`（page=64），append 只要 `base_E+accepted`，跨页时老码必抛 `KV materialization exceeds active entitlement`（例 base_E=60/extent=8/accepted=2 ⇒ 1<2 页）⇒ 本补丁直接消掉该终止崩溃; 且该请求本身多余（此阶段只写 DFlash Full backend KV），上游 `03177b9` 已把它整条删除。**结算顺序不用改**: 我们 `resolve_pending_raw` 已是 recurrent `:9868` → hidden `:9897` → draft ctx `:9917` → `device.synchronize() :9925` → 发布 frontier `:9984` → 尾裁 `:9985`，与上游 doc 契约逐句一致，未找到违反行。**DFlash2（`:12415`）不受影响且无需重排**: main 请求恒为增长（轮首 `page_count==pages_for_tokens(frontier)`），backend 恒传 0（DFlash2 无 backend KV, `:12407`），且 `:9903`/`:11381` 把 context append 限死在 DFlash。另报 2 条未移植偏离（md §4）: `:11416` 的 Main KV 请求上游已删（新语义下已是 no-op）; `:8711` 的 backend 参数我们用 `speculative_backend==None?0:end` 而上游用 `backend_kv_cache()?end:0`，DFlash2 若走到该行会撞 `:10323` 的 `logic_error`（**可达性待证实**，与本补丁无关） | M: `wsl.exe -e bash -c "cd /home/user/ninfer-fusion && patch -p1 --dry-run --forward --fuzz=0 < /mnt/c/Users/User/Documents/ziqinzhang/_collab/E7_s50_kv_coverage.diff"`（期望 exit 0; 本会话在 Windows 侧, `/home/user/ninfer-fusion` 不可达 ⇒ 行号取自镜像 `ninfer-fusion-repo`, 在镜像上 --dry-run 与 --fuzz=0 均 rc=0） |
| S52 | DONE (未写 src/**; 纯 CPU, 只做文本读/git show/diff/patch --dry-run; 未开 nvcc/ptxas; 未用 GPU) | E9 | `_collab/E9_s52_dflash2_k_slice.md` + `_collab/E9_s52_dflash2_k_slice.diff` (6 文件, +44-32, 10309B, `patch -p1 --dry-run` rc=0) + 生成器 `_collab/E9_s52_mkpatch.py` | **scoping upstream `385b30c` 里能借的「dflash2 可变草稿宽度 K」最小切片**: 上游那一版是全量重构 (47 文件 +1114-982: proposal/verify extent 分离、partial terminal commit、Vision/Host 复用、DFlashConfig 与 dflash2 权重合一、schema v14)，但我们树里 dflash2 的帧布局/recipe/graph profile/envelope/extent 早已按运行期 K 泛化 ⇒ 只剩 **13 处硬点 / 6 文件**: product 门 `speculative_options.h:54` 与 target 门 `layouts_impl.h:869-870` (都是 `!=0 && !=7`)、运行期门 `dflash2_impl.h:355` (`k != DFlash2Config::block_drafts`)、selector scratch 形状实现侧 `dflash2_impl.h:336/338/341` **与 arena recipe 侧 `layouts_impl.h:739/742/745` 各写一遍必须同步**、selector 两道 op 域门 `ops/wrapper/dflash2_selector.cpp:34` + `ops/launcher/dflash2_selector.cu:18`、**最易漏的 conv 门 `ops/wrapper/dflash2_grouped_conv.cpp:22`** (它要求 `block_size == 8`，而 `dflash2_impl.h:214` 传的正是 `k+1`)。另有 11 处只记录宽度不拦路 (2 死配置: 35b/muse 的 `DFlash2Config::supported=false`; 4 注释/文档; 2 默认值 7 保留; 1 近义陷阱 `27b config.h:145 kMaximumDFlashDraftTokens=7` 是 DFlash v1 的帽, dflash2 分支从不读; 1 变死常量 `DFlash2Config::block_drafts`)，以及本来就 K 无关的帧/recipe/graph/envelope 全部不动。**关键更正**: 任务书的 d1 65.4 / d3 80.5 / d7 70.0 **不是 dflash2** —— `_collab/M_spec_sweep.md:9-11` 三行的 backend 列写的是 `dflash` (DSpark v1)，dflash2 的 d1/d3 是 SERVE_FAILED 且报错串就是本次要拆的门 (`:12-13`)，d7 是退化 `gen=2 / 31.7tok/s` (`:14`) ⇒ 「宽度是 dflash2 的真杠杆」**待证实** (上游自己写非因果 attention 使改变 W 后的 hidden/candidates/q 不是更宽 block 的前缀, `docs/maintainer/qwen3.8-27b-dflash2.md:386-390`)。切片后可达 K ∈ {1,3,7} (W 必须是 2 的幂: 内核取 block 内位置用 `t & (block_size-1)`, `ops/kernel/dflash2_grouped_conv.cuh:29`); K=15 还差 conv 的 T<=64 与 swa 的 T=1..16 优化域。**门槛偏差 (请裁决)**: 6 文件 > 任务书「<=3 文件」(行数 +44-32 远低于 60)，原因是 3 道 op 域门散在 3 个 op 文件，少改任何一道 K=3 都在运行期炸; 若不接受 6 文件，可删 diff、按 md Section 2.1 的表拆成 3 批 (product+target / targets 内部 / ops 层) 过窗 | M: `wsl.exe -e bash -c "cd /home/user/ninfer-fusion && patch -p1 --dry-run < /mnt/c/Users/User/Documents/ziqinzhang/_collab/E9_s52_dflash2_k_slice.diff"` (期望 rc=0 且 6 文件 md5 不变, 原文见 md Section 7.1); 落地+重编后按 md Section 5 用 `--spec dflash2 --draft-tokens 3` 与 `7` 各跑一条 (同产物 `qwen3_8_27b_nvfp4_dflash2.ninfer`, `--no-cuda-graph`, max_tokens 192), 判据: 两跑日志 `speculative=` **必须是 dflash2** (读到 mtp = 验收地板 `spec_decision.h:99-107` 把 extent 置 0 降级, 该跑作废) 且 `gen` 接近 192, 然后比 `decode=tok/s`: K=3 > K=7 成立则宽度杠杆对 dflash2 也成立, K=3 <= K=7 则证伪、瓶颈在草稿步固定开销/输入链 (与 E5 的 H1 同向) |
| S53 | DONE (纯 Python/CPU; 未写 src/**; 未开 nvcc/GPU; 未跑完整转换) | E10 | `tools/convert/common/source_map.py`(新, 507 行) + `tools/convert/check_source_map.py`(新, 702 行) + `tools/convert/qwen3_6_27b/recipe.py`(+314/-8) + `tools/convert/qwen3_6_27b/convert.py`(+64/-5) + `tools/convert/qwen3_6/common/conversion.py`(+14/-2) + `tools/convert/common/__init__.py`(+26) + `tools/archkit/adapt.py`(只动 tie/嵌入名, :101/:121/:245-262) + `_collab/S53_converter_tie.md`(含 198 行自检原始输出) + 中间件 `_collab/s53_scratch/`(before_* + 原始输出 + 报告样例) | **转换侧补齐 tied embedding 两件事**: (a) 嵌入别名按优先级解析(`model.embed_tokens.weight`/`model.embedding.weight`/`embed_tokens.weight`/`model.language_model.embed_tokens.weight`), 命中名+`matched_rank`+`present`+候选表全进报告 `source_binding` 段, 四个都不命中即 `SourceMapError`(列候选+样本键)不静默猜; (b) tie 物化: `tie_word_embeddings=true` **或** 索引里没有 `lm_head.weight` ⇒ `text/output_head` = 嵌入本身(仅当嵌入存成 `[hidden,vocab]` 才 `Transpose((1,0))`; `[hidden,vocab]`+draft head 组合显式报错), 报告给出 `provenance`(index / identity-from-embedding / transposed-from-embedding)+`source_bytes`+`object_bytes`+`reason`+`orientation_verified`; config 里两处硬写 `tie_word_embeddings: False`(`convert.py:40,63`)删除, 改为解析后上报; `binding=None` 路径与今天逐字节一致(绑定后 1118 条 recipe 与注册表逐元素相等, 离线报告用例 `source_preflight` 字典原样, `source_binding` 段不出现)。**真实 index 判定**: Spark-X2.5-4B(真索引 290 张量)命中 `model.embedding.weight`(rank 1/4)、`tie_word_embeddings=true`、需物化、朝向 identity(真 header `(131072,2560)` BF16, 分片1 `complete=True`)、物化体积 671,088,640 B; 对照 Qwen3.8-Flash-Next(真索引 296,475 张量, 命中 `model.language_model.embed_tokens.weight` + `lm_head.weight`)⇒ not-tied 不物化且绑定后 recipe 与注册表相等; Falcon-H1R-7B / MiniCPM5-1B 同判 not-tied; LFM2-2.6B-Exp(索引无 head 且无 tie 键)⇒ 物化。自检 `python tools/convert/check_source_map.py` = 61 检查 0 失败 PASS(有界读真分片 16 行 BF16 往返 81,920 B; `--full-materialize` 走真 `materialize_expression` 得到 `(131072,2560)` / 671,088,640 B)。adapt.py 只动 tie/嵌入名: `spec.embedding` 记命中键(复用转换器同一个解析器), `head:tied=true` 由 `new_op` 改判 `covered`(Spark 唯一 new_op ⇒ `out/spark-x2.5-4b/config.h.BLOCKED`→`config.h`), **这条改判请 M 裁决**(回滚 = 改回一行), manifest 的 need 列表/顺序不变 | M: `cd ninfer-fusion-repo && python tools/convert/check_source_map.py`(期望 `checks: 61, failures: 0` + `RESULT: PASS`); adapt 差异复核: `python -c "import json;b=json.load(open('_collab/s53_scratch/before_manifest.json',encoding='utf-8'));n=json.load(open('ninfer-fusion-repo/tools/archkit/out/spark-x2.5-4b/manifest.json',encoding='utf-8'));print([g['need'] for g in b['gaps']]==[g['need'] for g in n['gaps']],{g['need']:g['tier'] for g in n['gaps']}['head:tied=true'])"`(期望 `True covered`); 未验证项见 `_collab/S53_converter_tie.md` §6: 真实 27B checkpoint 不在本机(逐字节一致仅表达式级证明)、Spark 完整转换未跑(分片 2..5 仍在下载)、全量物化只到 head 张量(未编码/未加载)、引擎/GPU/nvcc 全程未用 |
| A5 | DONE (纯 CPU/磁盘取证; 未写 src/**, 未编译, 未用 GPU/引擎) | A5 | `_collab/A5_dspark_rootcause.md` + 补丁 `_collab/A5_round_probe.diff`(诊断探针, 2 hunk/+74, dry-run rc=0) 与 `_collab/A5_block_rows.diff`(候选 B 修法草案, 2 hunk/+5-12, dry-run rc=0) + 生成器 `_collab/A5_mkprobe.py`/`A5_mkblockrows.py`/`A5_inject.py` + 只读工具 `_collab/A5_weights_audit.py`(artifact⇄checkpoint 逐字节审计)、`_collab/A5_parse_round0.py`(探针读数器) | **H1(tap 层号/tap 点错) 读码级排除**: 引擎 tap 是 post-MLP residual(`text_context_impl.h:1226-1227`/`:1249-1250` → `variant.cpp:369-377` 的 `x←x+down·silu(gate_up·rmsnorm(x))`) = HF `hidden_states[layer+1]` (`candidate_generator.py:1651-1657` 消费的正是 `[i+1]`; `output_capturing.py:112-119` 证明 `hidden_states[0]`=embedding), 层号表 `[4,16,28,40,52]`=`tmp/dspark-config.json:19-25`↔`qwen3_6_27b/impl/config.h:113`; **新实测(磁盘/CPU)**: 服务那份 artifact 的 **55/55 个 `dflash/*` 张量与 checkpoint 逐字节相同**(`A5_weights_audit.py` 两次运行, 含 markov_w1/w2 未互换、qkv 行序、context_key←k_proj/context_value←v_proj) ⇒ 权重绑定错整族排除; **剩下两条 DSpark 专属候选**: A) 草稿 rope 约定(ckpt 声明 yarn/factor 32 `tmp/dspark-config.json:54-68`, 引擎草稿两条 rope 都走纯 rope_theta `dflash_impl.h:177/304`, target 侧只有 factor-4 `ops::rope_yarn4` 且需 `--yarn` `rope.h:44-53`), B) block 行→verify 列约定(引擎 bf16 pack rows 0..k-1 `dflash_impl.h:447-451` vs HF `[:, 1:]` 与引擎自有 W8 `source_column_offset=1`) ⇒ 整路错位一行; 两条都只伤 dspark 不伤 DFlash2(草稿用引擎自训纯 rope + 引擎自产 teacher, `train_dflash2.py:41/43/79-88`); **判别实验**: 打探针 → 一条 10 s GPU(同一 prompt 同时设 `NINFER_DSPARK_DUMP_DIR`/`NINFER_HS_DUMP_DIR`) → CPU 用本地 ckpt 按 (纯 rope|yarn-32)×(rows 0..k-1|1..k) 四组合复算, 能复现引擎 drafts/logits 的组合即真约定; **已排除 E-2 原路径的缺口**: Windows 侧 `data/Qwen3.8-27B` 只有 embed+lm_head(5.08 GB, index 仅 2 项) ⇒ 无完整 target 权重, HF 逐层 cos 跑不了, 探针复算不需要它; E5 引用的 `text_context_impl.h:1221/1244` 为旧行号(现 1227/1250, 差 46 行); E6 补丁已在此树落地(dflash_impl.h md5 `02cb3bc9…`, Windows 镜像仍 `60a51d1c…` 落后) | M: ① 复核 §3.1 的审计输出(两条命令); ② 跑 §2.3 判别实验(10 s GPU + CPU 复算); ③ 按结论选 `A5_block_rows.diff` 或改 rope 方案; `wsl.exe -e bash -c "cd /home/user/ninfer-fusion && patch -p1 --dry-run < /mnt/c/Users/User/Documents/ziqinzhang/_collab/A5_round_probe.diff"` 期望 rc=0 |
| A2 | DONE (纯 CPU/离线; 未跑 GPU/引擎; 未改 src/**; 未编译) | A2 | `_collab/A2_draft_ceiling.md` + 评估器 `_collab/A2_draft_eval.py`（两输入模式 tf=真 token 块 / mk=引擎 `[anchor,mask×7]`；行 `eng=ids16[a+s,0]`、`leg=ids16[a+1+s,0]`；另 `A2_ids16_conv.py`/`A2_teacher_entropy.py`/`A2_baselines.py`/`A2_mask_usage.py`/`A2_loss_curve.py`）+ 日志 `_collab/A2_eval_main.log` | **Q1 位置0上限**（真特征+真 ckpt，键=教师 top-1）：mk(引擎口径) **0.0309 (21/679)** @step1200、**0.0427 (29/679)** @step1800、0.0000 (0/679) @step400；tf(训练器输入口径) 0.1708/0.1222；即使最有利组合(tf+训练器自己的目标行)也只有 **0.2504** ⇒ 41.8% 不是当前草稿能到的高度。**Q2 形态**：两边同形——位置0 全链最难、位置1 反而更高（引擎 41.8%→47.8%；离线 tf 0.171→0.474、mk 0.043→0.621）；可比 prompt 上引擎 zh 4.51%（`M_patchA_effect.md` §2）≈ 离线 mk 3.1–4.3% ⇒ **无证据显示 verify 在丢草稿**；CLI 那档 41.8%/23,11,3,1（10.03% 总）来自更易 prompt 混合，离线文本样本复现不出。**Q3 对齐**：同一 ckpt 只换评估目标行，**shift=1（train_dflash2 当前默认）系统性更高**（tf slot0 0.2504 vs 0.1708 @1200；0.2047 vs 0.1222 @1800）⇒ checkpoint 学的是 shift=1，而引擎要的是 **shift=0**（证据链：`prepare_masked_block.h:20-27` ids[0]=anchor/ids[1:]=mask、`dflash2_impl.h:192-196,306-319` 草稿=列1..7、`speculative_round.cuh:18-37,131-141` 接受=drafts[s] vs verify 列 s 的 argmax、`program_impl.h:11822-11827` 普通解码定 frontier=最后 token 的 index）⇒ **目标行差一位 = 机制性根因之一**。**第二根因（新）**：引擎喂 mask 块，`train_dflash2.py:430-436` 喂真 token（语料从无 mask id 248070，`A2_mask_usage.py` 0/4468，max_id=248046）⇒ mask 口径命中率比 tf 低 **5.5×**（0.031 vs 0.171 @1200；@400 为 0/679）；仓库内两个参考训练器都是 mask 块 + shift 0（`dl/aeon-train_head.py:25-42,54-60`、`train_dspark.py:417-432`）。**外推**：教师 top-16 熵地板 **1.3654 nat**（6 文件 4468 位置），当前 loss ≈ 地板+0.48（近期斜率 −3.3e-4/step ⇒ 再 +1460 步到地板）；按 (loss, 位置0命中) 线性外推，到地板时 mk 口径仅 **≈0.10**、最有利口径 ≈0.66，要 0.9 需 CE≈0.95 < 地板 ⇒ **当前配方不可达**；且草稿低于平凡基线（prev_teacher 0.2946 / true_next 0.3646 / 我实测最好 0.2504）⇒ 不只是欠训练。**待证实**：引擎 41.8% 那批数据用的 artifact 是否为 step_001200（`dl/df2_w9_export.log:47,55` 显示 w9s1200 artifact 当时未生成）、ctx 2048(引擎) vs 128(训练/离线) 的影响（小样已测，§8b：mk 口径 slot0 随 ctx 128→512→2048 = 1/60→5/60→8/60，即主表对引擎是保守下界）、verify≠plain 对上限的定量压低 | M: 复核 `_collab/A2_draft_ceiling.md` §4 主表 + §1 命令；复跑 `py -3 -u _collab\A2_draft_eval.py --files 6 --anchors-per-file 200 --batch 16 --ctx 128 --dtype fp32 --lm-chunk 31040 --threads 12 --ckpts step_001200`（期望 tf eng/leg slot0 = 0.1708/0.2504；旧探针 `dl\shift3_probe.log` 476 对为 0.1618/0.2290，已互证：其代码里 `next` 臂才是引擎需求行，命名与自身 docstring 差一位）；决胜实验 = §10-1（加 `--mask-block` + `--target-shift 0` 小训 ~1k 步，看 mk/eng slot0 是否从 0.03 跳到 ≈0.25） |
- **更正上面那条**：接受率计数**早就有**（serve `src/serve/request_log.cpp:438-456` 打 `speculative=dflash 1.38tok/round (10.3%)`；CLI `apps/cli/main.cpp:238-245` 还打 `accepted by pos` 逐位置直方图）。今天扫不到 = 字段无标签 + `_spec_4way.sh:55` 把汇总行 `cut -c1-230` 截断（完整行在 `/home/user/s4w_*.log`）⇒ E2/S44 是加标签/修 CLI 错标，不是新增计数。
- **ingress 无 DFlash2 式漏填**：DFlash v1 decode 轮 11/11 字段全填（`program_impl.h:12164-12179`，两个 state slot 在 12177/12178，与 MTP `:11989` 同用 `state_selectors()` `:10028`）；`:11414` 属 `enqueue_dflash_context_append`（`11378-11449`），不进 decode 轮；verify/accept 与 MTP/DFlash2 同一实现（`speculative_target_impl.h:9-38`）。
- **已排除**：缺 markov 的 artifact（服务的是含 `dflash/markov_w1|w2` 的 v2 实体 —— `/home/user/models/*_dspark.ninfer` 24,316,119,552B，`E5_s47_cmp.py` 实证）、config 不匹配（`tmp/dspark-config.json` ↔ `config.h:74-106`）、40Q/8KV 桥、物理尾列污染、宽度（sweep：d1 65.4/d3 80.5/d7 70.0）。
- **新实证 off-by-one**：DFlash 的 verify 复用草稿块的 position 表（按 `V=k` 造，`dflash_impl.h:223-231`），比自身 `target_valid_columns=extent+1=k+1` 少一位 ⇒ 第 k 路验在重复位置、KV 同址双写（`dflash_impl.h:522,556-557`；MTP/DFlash2 都不缺位）。
- **首选假设 H1**：草稿输入链（目标特征/上下文→草稿 hidden）与 checkpoint 训练约定不一致 —— 上游 DSpark 按 HF `hidden_states` 训练，引擎 tap 是该层 post-MLP residual（`text_context_impl.h:1221/1244`）；DFlash2 草稿在引擎 NHS1 tap（`text_prefill_impl.h:85-160`）上训练 ⇒ 免疫。上游移植文档自己也停在这条未解（`DSPARK-ADAPTATION.md:151-158`）。
- **下一步**：① 修位置表少一位（独立小补丁）；② E-1（CLI 三臂 `accepted by pos` 剖面：`p_0` 低 ⇒ H1，`p_0` 正常而链断 ⇒ H2/H4）；③ E-2（`NINFER_HS_DUMP_DIR` dump ↔ HF `hidden_states[L+1]` 逐层 cos，>0.999 ⇒ H1 死）。
