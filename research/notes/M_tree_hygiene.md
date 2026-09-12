# 树/工作区卫生（善后记录，2026-09-10 16:22）

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
`C:\Users\User\Documents\ziqinzhang\` 顶层现在有 **414 个 `_*.sh` + 316 个 `_*.py`**（今天一天就产出了大几十个），
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
- `C:\...\ninfer-upstream`（新，60 commits 浅克隆）：**保留**，是上游借鉴的来源；
  另见 `_collab/M_upstream_borrow.md`。
- `dl/_chunk.tmp`：分块下载器的暂存文件，**在用**，勿删。
