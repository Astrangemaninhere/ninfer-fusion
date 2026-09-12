# B/S41 — 双树取证对账 (build tree `/home/user/ninfer-fusion` vs mirror `ninfer-fusion-repo`)

**方法 (只读，未写任一树)**: 逐文件 `diff -u`；**影子树重建**（拿 mirror 当基线，按 window H/J 的补丁顺序在 `/tmp` 重放，再与真 build tree 逐字节比）；`patch` 反向锚定（把补丁删掉的行拿去 mirror 里找存活）。
**证据**: `_collab/B_s41_raw_diffs.txt`（原始 diff）、`_collab/B_s41_probes/`（19 个可复跑探针）、`_collab/B_s41_candidates/`（合并候选）。
**判据声明**: 全程未用 mtime 定案；下面每条结论都带机械判据。

---

## 1. 对你的工作假设的裁决

> 假设: **"build tree 权威；agents 出补丁；只有 window 脚本应用补丁"**

**分三句裁决:**

| 子命题 | 裁决 | 判据 |
|---|---|---|
| build tree 权威 (即 mirror 陈旧) | **成立** | 影子树重建：21 个受检文件里 **19 个** build == mirror + 补丁序列，**逐字节相等**（`probe9`） |
| agents 只出补丁 | **证伪** | **U1/U2/U3 三项 board 标 `[DONE-A]` 的修复只存在于 mirror**（board:26-28、`_TODO.md` §117）；build tree 完全没有 |
| 只有 window 脚本应用补丁 | **证伪（两个方向都漏）** | ① window H 的 `s23` 被 `apply.sh` 默认 `REPO=<mirror>` 带进 **mirror**，不是 build tree；② `_sync.sh` 是 mirror→build 的搬运，独立于 window 脚本（输入 `_changed.txt`，mtime 09:14） |

### 实际写路径表（四路，而非一路）

| # | 方向 | 机制 | 证据 |
|---|---|---|---|
| W1 | → BUILD | window H/J/J2 `patch -p1 -d /home/user/ninfer-fusion` | `_window_h.sh:105`、`_window_j2.log` |
| W2 | → BUILD | **协调者手工编辑**（无补丁、无记录）: J2 自述"coordinator applied the two completion edits" | `_window_j2.sh` 注释；实测 `require_nvfp4_geometry_dim` 只在 build；`Groups = D/kGqaKvQuantGroup` 只在 build |
| W3 | → **MIRROR** | `C_s23_tu_split/apply.sh:15` `REPO="${NINFER_REPO:-.../ninfer-fusion-repo}"`；window H `[4/6]` 直接 `bash "$p"` 未覆盖 env | window_h.log: `dry-run: 9/9 patches apply cleanly against .../ninfer-fusion-repo` + `APPLIED src/...`。**H 打印 "apply to the live tree"，但 payload 目标是 mirror** |
| W4 | mirror → BUILD | `_sync.sh`（`_changed.txt` 驱动） | `_changed.txt` 存在，11 条，mtime 09:14；build 侧多个文件 mtime 09:12/09:28（部分搬运确实发生过） |
| W5 | → **MIRROR** | agent 直写：**所有** mkpatch 生成器都把 mirror 当读基线 | `A_n1_mkpatch.py:14`、`A_s24_mkpatch.py:16`、`A_s30_mkpatch.py:21`、`A_s32_mkpatch.py:20`、`A_s36_mkpatch.py:20`、`C_s23_tu_split/mkpatch.py:33` 全部 `REPO=.../ninfer-fusion-repo` |

**修正后的模型**: mirror 是**读基线**（也是部分轮次的**写面**），build tree 是**编译面**；两者由"补丁应用 + `_sync.sh` + 手工编辑"三条缝拼接，其中 `_sync.sh` 没有逆操作。⇒ **mirror 不能当"陈旧副本"整体丢弃**，它有 3 项已标 DONE 但从未落到编译面的修复。

---

## 2. 权威对账表

### A 类 — "补丁只落在 build tree"（mirror 陈旧，**无需动作**）: 26 个文件

全部经 `probe9` 影子重建**逐字节证明** `build == mirror + 补丁序列`。`+N/-M` 为 `diff -u MIRROR BUILD`。

| 文件 | 补丁 (轮次) | diff摘要 | 权威侧 |
|---|---|---|---|
| `include/ninfer/types.h` | n1 + n1b + s32 + s31 + s35（窗口H批 5 连击） | +39/-4 | **build** |
| `src/runtime/engine/engine_core.h` | s31 + s35 | +83/-8 | **build** |
| `src/serve/http_server.cpp` | s31（D 轮）| +26/-12：build 把 `/health` 的 `failure.reason` 原文换成 `category`，并把 `last_request_error` 结构化。**mirror 仍是"原文挂在免鉴权线路上"的泄漏版** | **build** |
| `src/serve/generation_service.cpp` | n1 + n1b + s32 + s35 | +10/-0 | **build** |
| `src/serve/serve_options.h` | n1 + n1b + s32 | +4/-0 | **build** |
| `src/serve/serve_options.cpp` | n1 + n1b + s32 | +35/-0 | **build** |
| `src/artifact/binder.h` | s32(W13) | +7/-0 | **build** |
| `src/targets/.../impl/runtime/request_plan_impl.h` | s35 | +22/-10 | **build** |
| `src/targets/.../impl/runtime/layouts.h` | n1b | +4/-0 | **build** |
| `src/targets/.../impl/runtime/layouts_impl.h` | n1 + n1b + s30 | +111/-6 | **build** |
| `src/targets/.../impl/runtime/kv_calibration.h` | s38 | +48/-5 | **build** |
| `src/targets/.../impl/runtime/text_context_impl.h` | s38 | +43/-0 | **build** |
| `src/targets/.../impl/state/decoder_state.cpp` | s24 + s28 | +37/-3 | **build** |
| `src/serve/kv_auto_relayout.cpp` | s34 | +52/-0 | **build** |
| `src/ops/kernel/gqa_attention_prefill_i8.cuh` | u7 (Muse page-fill) | +22/-10 | **build** |
| `src/ops/kernel/gqa_attention_decode.cuh` | s36 (A/S36 256-stride) | +9/-7 | **build** |
| `src/ops/kernel/gqa_attention_decode_{bf16,fp8,iso3,nvfp4}.cuh` | s36 | 各 5–9 处 | **build** |
| `src/ops/kernel/gqa_attention_kv_nvfp4.cuh` | s36 | | **build** |
| `src/ops/kernel/gqa_attention_prefill_{bf16,common,nvfp4}.cuh` | s36 | | **build** |
| `src/ops/launcher/gqa_attention_prefill.cu` | s36 | +4/-4 | **build** |
| `src/ops/launcher/gqa_attention_decode_e8.cu` | s36 | +1/-1 | **build** |
| `src/product/weight_residency.h` | s32 **新文件** | build-only | **build** |
| `src/serve/kv_cold_policy.h` | s34 **新文件** | build-only | **build** |
| `src/ops/kernel/gqa_isoquant_row_scale_loader.{cu,h}` | s28 **新文件** | build-only | **build** |

**s36 的机械锚定（最硬的一条）**: `A_s36_headdim.diff` 共删除 13 个文件里的若干行，**这些被删的行 100% 仍存活在 mirror**（`probe6`: 13/13 文件、逐行命中）。⇒ mirror 对 s36 是**确定性前态**，build 是后态。补漏/回退绝不可从 mirror 取。

### B 类 — "两侧都改了"（**需刻意合并**）: 5 个文件

| 文件 | build 侧来源 | mirror 侧来源 | diff -u 摘要 | 裁决 |
|---|---|---|---|---|
| `src/ops/launcher/gqa_attention_decode.cu` | 巨型 TU（737 行）+ s36 + **协调者 nvfp4 门**（`require_nvfp4_geometry_dim`） | **S23 拆分后的派发器**（112 行）+ `tiers.h` | `@@ -1,13 +1,479 @@`（+612/-7）: mirror 首行是 `#include "ops/launcher/gqa_attention_decode_impl.cuh"`；build 保留 `q_attention.h` + 6 个 kernel 头 | **build 权威**。镜像侧是 S23，**本轮刻意排除**（见 X 类） |
| `src/ops/launcher/gqa_attention_decode_impl.cuh` | s36 后的 `launch_for`（618 行，硬编码 0 处） | S23 的 `launch_for` 版（576 行，硬编码 2 处**仍存活**） | +99/-57 | **build 权威** |
| `src/ops/kernel/gqa_attention_decode_i8.cuh` | s36 + **协调者 `Groups` 参数化**（`D/kGqaKvQuantGroup`） | s36 前态（13 处硬编码 `kGqaHeadDim`） | +15/-15 | **build 权威**（`probe10` 逐行确认差异就是 `Groups`） |
| `src/targets/.../export/.../decoder_state.h` | s24（窗口表）+ **缺 U1** | **U1**（`kKvFp8QuantGroup` 16）+ 无 s24 | +7/-7 | **必须合并**: build 保留 s24、只补 U1 那一行（**不可整文件覆盖**） |
| `src/CMakeLists.txt` | 无 qpn、无 5 个 tier TU | S23 注册 5 个 tier TU + **`ops/linear/qpn/qpn_host.cu`** | +0/-6 | **混合**: 5 个 tier TU → 属于 X 类（不取）；`qpn_host.cu` → 见 M4 |

### M 类 — "**mirror 独有、build tree 缺失的真实工作**"（board 标 DONE 但从未落地）: 6 项

| # | 文件 | 是什么 | board/轮次 | diff 摘要 | 意图裁决 |
|---|---|---|---|---|---|
| **M1** | `src/targets/.../export/.../decoder_state.h` | **U1**: `kKvFp8QuantGroup 256→16`（对齐 kernel 的 16-组 scale 索引 + wrapper 的 "packed 必须 16" 校验） | board:26 `U1 [DONE-A]` | 1 行常量 | **明确** — 应从 mirror 取该行（与 s24 合并） |
| **M2** | `src/ops/wrapper/gqa_attention.cpp` | **U2**: `gqa_attention_workspace_capacity_bytes` 的 dtype 白名单补 `DType::E8Kv` | board:27 `U2 [DONE-A]` | +1/-6（**唯一**差异） | **明确** — 除该 hunk 外两侧无其它差异 ⇒ mirror 文件内容即目标 |
| **M3** | `src/product/kv_options.h` | **U3**: 文档化 `iso3/e8` 词法与 `all:bf16` 哨兵 caveat | board:28 `U3 [DOC-A]`；`_TODO` §117 | +3/-10（**纯注释**，无代码行） | **明确** — mirror 文件内容即目标 |
| **M4** | `src/CMakeLists.txt` | 注册 `ops/linear/qpn/qpn_host.cu`（文件两侧**逐字节相同**，只是 build 未注册） | **board 无 qpn 行**（off-board） | +1 行 | **不确定** — 该 TU 定义 `gemm_qpn_simt`，被 `qpn_probe*.cu`/`qpn_e2e_test.cu` 调用，但**那些 probe 两侧都没注册** ⇒ 注册它等于往紧凑构建里加一个无消费者的 TU。建议**暂缓** |
| **M5** | `apps/CMakeLists.txt` | `if(UNIX) target_link_options(ninfer-serve PRIVATE -rdynamic) endif()` | off-board | +3 行 | **不确定**（低风险，仅链接期符号导出） |
| **M6** | `apps/perplexity/main.cpp` | `prefill_chunk_tokens` 上报值 `1024`(build) vs `3072`(mirror) | Sep-9 08:5x | 1 行 | **不确定** — 位于输出 JSON 的**上报**块（非驱动参数）；两个值都可能是当时真实值 |

> M2/M3 的 mirror mtime (09:14–09:15) 与 `_changed.txt` (09:14) 同步列表同时刻 ⇒ **当时确实计划同步，但搬运没发生**（build 侧无对应 mtime）。

### X 类 — "mirror 独有但**绝对不要导入**": S23 拆分产物（6 文件）

| 文件 | 行数 | sha256 (mirror) |
|---|---|---|
| `src/ops/launcher/gqa_attention_decode_tiers.h` | 81 | `0a498d4b1a75d54c56abdbaac7544040745c1faf5b7d86f672775f370c318b2e` |
| `src/ops/launcher/gqa_attention_decode_i8.cu` | 128 | `a48fd99101ff2a5f0e5ef0eb3f30eeea0ede465358dfab59d219aeff1951b04a` |
| `src/ops/launcher/gqa_attention_decode_nvfp4.cu` | 161 | `d6123524ff9d42471ed66702d454d73832611852cad6ff857fd73fb323c93681` |
| `src/ops/launcher/gqa_attention_decode_fp8.cu` | 128 | `da8413ea766a87e0dced6fa5c2ee05f43b70a809b67374aa3cc7680e6f65f8b9` |
| `src/ops/launcher/gqa_attention_decode_iso3.cu` | 128 | `3e364a8b5b930ef6d78027b6b44e7caead2e70ee08902ffc20e58bda1a36e227` |
| `src/ops/launcher/gqa_attention_decode_bf16.cu` | 128 | `c0eb070a27bc4709bb191fe53f082b279de9c16265323b9033c12a75ae332c1f` |

**为什么不能导**: 它们是 `apply.sh` 走错树（W3）的产物，而 window J 明确写了 "L3 is deliberately EXCLUDED: its apply.sh targeted the wrong tree, and memory discipline says keep the original giant TU at -j1"。逐文件**按份拷贝会把三项已落地的修复一起回退**：
1. s36 的 256-stride 修复（13 文件，mirror 仍 100% 是前态）；
2. 协调者的 `require_nvfp4_geometry_dim` 门（只在 build 的 `decode.cu`）；
3. 协调者的 i8 `Groups` 参数化（只在 build 的 `i8.cuh`）。
⇒ 正确做法是 **C/S40 `tu_split_v2`**（已在 `_collab/C_s40_tu_split_v2/` 起手）在**已含上述三项**的源上重新拆分。

### E 类 — 仅行尾差异（**不是分歧**）

| 文件 | 事实 |
|---|---|
| `CMakeLists.txt`（根） | mirror 90/90 行 CRLF，build 0 行 CRLF；`diff -q --strip-trailing-cr` ⇒ **完全一致**。`diff -u` 显示 `@@ -1,90 +1,90 @@` 是假象 |

### 范围外（不构成引擎状态分歧）
`tools/`（archkit out/、specs/、qpn_port 生成物）、`tests/`、`__pycache__`、`.orig`/`.bak-*` 备份、`compat/`：多为**单向生成物/工作副本**，未纳入上表；如需另有清单请点名。

---

## 3. 最小有序落地清单（带 sha256）

**前置纪律**: ① window J2 正在 build（14:50 起 attempt 1），**任何写入都会作废其判定**，请等它走完 `[3/3]`；② 落地后必须重新配置+编译（`src/CMakeLists.txt` 变更尤其）；③ 落地前任一预期 base 不符即停。

### 基线校验（改之前先核）

| 文件 | build tree 当前 sha256 |
|---|---|
| `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/decoder_state.h` | `50f09a525dffab4a73fc1158ce89749950bbabb6fd0288eb0b966970ffdf008d` |
| `src/CMakeLists.txt` | `2a1fe64042e58d3e4f843872d08d0afa09e0b8ec957f3aeed5523da237532995` |
| `apps/CMakeLists.txt` | `d5a0e9a692d03da6254ac5fe82ce0e660cf8e8dc65424d1632e5b748cf2f274c` |
| `src/product/kv_options.h` | `30d1cf2ab793ea395439e7795bc5ddc57e36761759a45884e5a0c38aa02c6032` |
| `src/ops/wrapper/gqa_attention.cpp` | `64749c84c8b1aeac055edb1193f188dc5de312296039bdd44f1fc14ce52011b4` |

### 步骤（按序）

| 序 | 文件 | 目标 sha256 | 目标内容来源 | 手法 |
|---|---|---|---|---|
| 1 | `src/ops/wrapper/gqa_attention.cpp` | `9ad49220d57512decf8657a09bf0827a91e8d8575d078842fde9ffa0a82336ba` | mirror（= U2 唯一 hunk） | 整文件拷贝 |
| 2 | `src/product/kv_options.h` | `786c4e52eb81663806d227ad83a8e29271d47a094cfeee01c310053b1c2b4239` | mirror（= U3 纯注释） | 整文件拷贝 |
| 3 | `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/decoder_state.h` | `904b7535fa979436b0032444d364d8a97f1343bbaeb47bdaa7e60e51e9801314` | **候选**（= build + U1，保留 s24） | 用 `_collab/B_s41_candidates/decoder_state.h`（**不可用 mirror 版，会回退 s24**） |
| 4 | `apps/CMakeLists.txt` | `2f00e6df2b2de0e4b82d09b07503862b3faef8ebcfee689644599d1ee9848eab` | **候选**（= build + `-rdynamic`） | 可选/低风险；候选与 mirror 版**逐字节相同**（佐证该文件唯一差异就是这 3 行） |
| 5 | `src/CMakeLists.txt` | `0b3172b08f3276e8dffb1058573fb346285522d3591cde2a2eec18d42d20fceb` | **候选**（= build + `qpn_host.cu`，**不含** 5 个 tier TU） | **建议暂缓**（M4 意图不确定 + 会触发 reconfigure） |

候选均已落在 `_collab/B_s41_candidates/`（sha256 与上表一致，已复核）。
**不改动**（build 侧已正确，从 mirror 取会致回退）: 上表 A 类 26 个文件 + B 类前 3 个（`decode.cu`、`impl.cuh`、`i8.cuh`）+ X 类 6 个文件。

---

## 4. 无法判定 / 需你裁决

1. **M4 `qpn_host.cu` 注册** — off-board（board 全文无 `qpn`），无轮次、无记录；且 10 个 `qpn_*` probe 两侧都未注册。是"漏注册"还是"刻意不进构建"？我无法从证据定论。
2. **M5 `-rdynamic`** — off-board；仅链接期，风险低，但同样无记录。
3. **M6 `prefill_chunk_tokens` 3072 vs 1024** — 只影响 perplexity 输出里的**上报字段**；两侧 mtime 相差 12 分钟且镜像更新，无法从内容判谁是真。
4. **`gqa_attention.cpp` 的 build 侧 09:28:34 编辑** — 该文件在 build 里 mtime 09:28（晚于 mirror 的 09:14），但内容**恰好只差 U2 那一个 hunk**。我无法排除"有人刻意在 build 侧撤掉过 U2"；若你记得有过撤销，请以你为准，M2 降级为待确认。
5. **U1 的 i8 部分**（`_TODO` §117 "U1 的 i8 部分撤回"）已确认**不是**缺口，勿重复补。

## 5. 对当前 window J2 的即时影响（可操作）

- `_e8_postfix.sh` 走的是 `--kv-dtype nvfp4 --kv-layer-storage 0-7:e8`（**逐层** e8），**不经过** 全局 `--kv-dtype e8`/`fp8` ⇒ **U1/U2 不必然波及本窗判定**。
- 但若 J2 的 e8 档报 **`packed KV cache must use quant_group 16`** ⇒ 正是 M1/U1 缺失；报 **`invalid profile or interval`** ⇒ 正是 M2/U2 缺失。**先看这两条串，再决定是否把第 1–3 步插进下一个窗口**。
- 另注: 建议顺手给 `apply.sh` 加 `NINFER_REPO` 显式校验并写进窗口脚本（`export NINFER_REPO=/home/user/ninfer-fusion`），否则 S40 v2 会再咬一次。
