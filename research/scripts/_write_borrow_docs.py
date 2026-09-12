#!/usr/bin/env python3
"""Write the upstream-borrow review and the tree-hygiene note into _collab/."""
import datetime
import pathlib

C = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab")
stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

borrow = f"""# ninfer 上游更新与可借鉴清单（{stamp}）

## 0. 怎么拿到上游（已验证可用的路径）
- 上游是 **github.com/Neroued/ninfer**（我们 fork 的 base；我们的 fork = `Astrangemaninhere/ninfer-fusion`，
  镜像仓库 76 个 commit，**上游近期的修复一条都没有**）。
- WSL 侧连不上 Windows 的代理（代理只绑 Windows loopback），所以**从 Windows 侧 clone**：
  `powershell` 里设 `HTTP_PROXY/HTTPS_PROXY=http://127.0.0.1:10808` 然后
  `git clone --depth 60 https://github.com/Neroued/ninfer.git ninfer-upstream`（默认分支是 **master**，不是 main）。
  结果在 `C:\\Users\\User\\Documents\\ziqinzhang\\ninfer-upstream`，WSL 经 `/mnt/c/...` 直接读。
- PR 也能离线拿：`git fetch origin pull/<N>/head:pr<N>`（#226/#225/#194/#195 已拉到）。

## 1. 最值得抄的两条（都是我们当前 bug 的同一族）
### P0-A `03177b9` fix(runtime): preserve kv coverage during speculative terminal settlement
上游把 `materialize_to_tokens()` 改成 **`ensure_mapped_to_tokens()`**，语义从"精确匹配"变成
**"覆盖是下界"**：
```
- if (target < address.page_count || target > entitlement) throw "KV materialization exceeds active entitlement";
- if (target == address.page_count) return;
+ if (target > entitlement) throw "KV coverage exceeds active entitlement: tokens=… mapped_pages=… entitlement=…";
+ if (target <= address.page_count) return;      // 已有映射足够 → 直接返回，不截断
```
配套的 doc 明确写了三件事，正对我们实测的 dflash2 症状：
1) "投机映射可能已经超出本阶段所需；只有显式 truncate 才能释放，commit_frontier 才发布有效 token"；
2) "各阶段只保障自己会写入的 pool：target prefill/verify 负责 Main KV；draft context append 只保障
   DFlash Full backend KV；**DFlash2 的 draft context 全写固定 cyclic state，不需要 paged KV 物化**"；
3) "投机终止先按最终提交数量补齐 recurrent state、hidden 与 draft context，等 GPU 完成后再发布
   committed frontier，最后才裁掉未提交的尾页；**不能为了满足后续阶段更短的覆盖需求而提前裁 verify 的映射**"。
**我们树的状态（实测）**：`logical_kv_store.h:1494` 仍是 `materialize_to_tokens` + 那句旧的
"exceeds active entitlement" 抛错；`program_impl.h` 里有 **9 处** `materialize_sequence_kv` 调用。
**可行性**：上游该 commit 只改 3 个源文件（+20/-? 行级），但我们的这三个文件都已分叉（不是同一代文本），
所以要**语义适配**而不是 cherry-pick；另外它带了一个新测试 `tests/targets/qwen3_6/speculative_page_boundary.h`
（64 行）值得一并移植为回归门。

### P0-B PR #194 fix(nvfp4): keep the approximate SiLU off the zeroing range of `__fdividef`
13 行，单文件 `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh`：
```
- float swiglu_silu(float x) {{ return __fdividef(x, 1.0f + __expf(-x)); }}
+ float swiglu_silu(float x) {{
+     const float e = __expf(-fabsf(x));
+     const float r = __fdividef(1.0f, 1.0f + e);
+     return (x >= 0.0f ? x : x * e) * r;
+ }}
```
理由（上游实测）：`1 + __expf(-x)` 在 x < −87.34 时达到 2^126，`__fdividef` 直接返回 **0**，
而那里 SiLU 仍是正常 bf16（−9.6e-37）；写成上面这种"指数折到不会溢出的那一侧"后除数恒在 (1, 2]。
上游声称**编译出的 SASS 完全相同**（40 条、2 个 MUFU、无 CALL），所以是零成本正确性修复。
**我们树的状态**：文件存在但已分叉（需适配）；我们的 artifact 正是 nvfp4 W4A4，**每个 nvfp4 linear 的
SwiGLU 都在用这个近似**。

## 2. 其余可借鉴（按性价比）
| 来源 | 内容 | 为什么可能值得 | 可行性 |
|---|---|---|---|
| `385b30c` (09-06) | dflash2 **可配置草稿宽度 K=1..15**（五层草稿骨干 + top-16 条件选择器 + sparse rejection 走共享 Program；分离 proposal/verify extent；支持 eager/Graph、Text/Vision、部分终止提交） | 我们 dflash2 硬编码 7；我们自己的 sweep 就是 d3 80.5 > d7 70.0 tok/s；宽度的可调性也是接受率实验的杠杆 | **大**：dflash_impl +308 / layouts_impl +167 / program_impl +140 行，且两边都改过这些文件 → 只能选择性移植（先移植 `--draft-tokens` 对 dflash2 的放行 + extent 分离） |
| PR #226 | `mtp_pack` 契约硬化：eps 必须正有限、只接受已注册的 stem 宽度、契约语义写清 | 廉价健壮性，MTP 是我们最常用的后端 | 3 文件 +66 行，均已分叉 → 小适配 |
| PR #225 | 把 sigmoid gate 折进 causal-attention 的 reduce epilogue（算子 −8.3%，decode +0.31%） | 我们 bf16 路线可受益（FP8/NVFP4/K8V4 不折） | 中 |
| PR #195 | 无 (model, weights) 行匹配时**按同权重格式回退预设** + prefill 代价预测 | 与我们 `Package::resolve_weights` / `resolved_auto_speculative` 同族，能减少"选错后端"类事故 | 中 |
| `487f897`/`437e9f9`/`ce95491`/`7f14d96` (09-06/07) | MoE/S2 小 T decode 性能（一 CTA 一 token、按 route 选流水深度、shared-expert 预取、从 MoE down 尾部预热 L2） | FlashNext 是 512 专家 MoE，将来必用 | 中（等 FlashNext 权重到位） |
| `ee9d519` (09-03) | nvfp4 W4A4 TMA：CTA 按 token-fastest 光栅化 + 每激活 scale box 只取一次 | 我们 nvfp4 prefill 热点 | 中 |
| `b88c0f6` (09-08) | `fix(core): complete host uploads before returning` | host 上传完成语义（与我们踩过的"边改边编/半提交"同族） | 小 |
| `b158afe`/`641ef3e` | BPE 合并规则扁平哈希表；纯 ASCII 跳过 NFC | 前端微优化，收益小 | 小 |
| PR #224 | tool_choice 走生成提示 | 不是我们的路径 | 不抄 |

## 3. 建议的执行顺序
1. **P0-A / P0-B** 由子代理产出**针对我们树的适配 diff**（dry-run + shadow 证明），下一趟编译一起落。
2. 把 `speculative_page_boundary.h` 的边界用例思路移植成我们的测试（现有 tests 里没有 page boundary 覆盖）。
3. dflash2 可配置 K：先做"放行 `--draft-tokens` + proposal/verify extent 分离"的最小切片，跑我们的 sweep 复测 tok/s。
4. 其余按表排。
"""
(C / "M_upstream_borrow.md").write_text(borrow, encoding="utf-8")
print("wrote M_upstream_borrow.md (%d chars)" % len(borrow))

hygiene = f"""# 树/工作区卫生（善后记录，{stamp}）

## 1. 已做：patch 备份文件隔离
两棵树里的 `*.orig` / `*.bak*` / `*.rej`（13 个）已移到
`/home/user/ninfer-fusion/_orig_quarantine/`，`MANIFEST.txt` 逐条记了原路径与回放方法。
理由：它们是 `patch(1)` 的备份，对应补丁都已落地、且都作为 diff 归档在 `_collab/`；留在 `src/` 里只会污染
`grep`/`find`（今天至少有两次审计为此多绕了几步）。**没有删除**，只是搬家，随时 `mv` 回去。
清单（build tree 12 个）：`engine_core.h.orig`、`layouts_impl.h.orig`、`program_impl.h.bak_patchA`、
`decoder_state.cpp.orig`、`decoder_state.h.orig`、`generation_service.cpp.orig`、`serve_options.h.orig`、
`serve_options.cpp.orig`、`gqa_attention_decode_nvfp4.cuh.bak-prett6`、`gqa_attention_decode_impl.cuh.bak-prett6`、
`gqa_attention_decode_impl.cuh.orig`、`qwen_chat_template.jinja.bak4d34`；mirror 2 个。

## 2. 待你点头：根目录 730 个临时脚本
`C:\\Users\\User\\Documents\\ziqinzhang\\` 顶层现在有 **414 个 `_*.sh` + 316 个 `_*.py`**（今天一天就产出了大几十个），
命名靠时间与主题，活的和死的混在一起。**我没有动它们**，原因：活管线是按绝对路径互相调用的
（`_window_k3.sh` → `_post_build_measure.sh` / `_spec_4way.sh`），一刀切搬家会打断正在跑的链；而且有些是你
手工在用的工具（如 `_hf_chunked.py`、`_needle_check.sh`）。建议的整理方式（等你确认再执行）：
1. 先自动扫描"谁被谁引用"（在所有脚本里 grep 文件名），把**零引用且 24 小时内一次性**的脚本移到
   `_scratch/2026-09-10/`；
2. 在 `_scratch/README.md` 里按主题索引（构建/测量/导出/下载/代理），保留软链或说明；
3. 明确"长期工具"白名单（我建议：`_hf_chunked.py`、`_needle_check.sh`、`_train_df2_resume.bat`、
   `_spec_4way.sh`、`_post_build_measure.sh`、`_window_k*.sh`、以及 `tools/` 下的正式脚本）。

## 3. 其他遗留
- `_collab/*.SUPERSEDED`（5 个）：**保留**，是"哪一版被谁取代"的溯源记录（E3 的 S45b/S45c）。
- WSL `/tmp/e3s45b`、`/tmp/s45c`、`/tmp/pa_make_*.log`：子代理与构建的临时目录，重启即清，无需处理。
- `C:\\...\\ninfer-upstream`（新，60 commits 浅克隆）：**保留**，是上游借鉴的来源；
  另见 `_collab/M_upstream_borrow.md`。
- `dl/_chunk.tmp`：分块下载器的暂存文件，**在用**，勿删。
"""
(C / "M_tree_hygiene.md").write_text(hygiene, encoding="utf-8")
print("wrote M_tree_hygiene.md (%d chars)" % len(hygiene))
