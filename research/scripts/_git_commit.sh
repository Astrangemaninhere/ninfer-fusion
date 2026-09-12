#!/bin/bash
# 完成本地提交（可随时 push）：排掉备份杂物，纳入研究数据
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
OUT=$J/dl/_git_commit.txt
{
  echo "=== 本地提交 起 $(date '+%H:%M:%S') ==="
  cd "$R" || exit 3

  # 1) 补充 .gitignore：备份杂物与本地数据
  cat >> .gitignore <<'EOF'

# Local backup / quarantine clutter (not source)
_orig_quarantine/
*.orig
*.bak_*
*.diff
*.rej

# Local analysis dumps (kept out of git; regenerate via env-gated probes)
df2scores_*.bin
mtplg_*.bin
*.nsys-rep
*.ncu-rep
*.sqlite
EOF
  echo "  .gitignore 已补充"

  # 2) 把研究数据（报告/脚本/关键日志）纳入 research/（小体积）
  mkdir -p research/notes research/scripts research/logs
  cp -f "$J/_TODO.md" research/notes/TODO.md 2>/dev/null
  cp -f "$J"/_collab/*.md research/notes/ 2>/dev/null
  cp -f "$J"/_*.py research/scripts/ 2>/dev/null
  cp -f "$J"/_*.sh research/scripts/ 2>/dev/null
  # 关键日志（体积可控：只挑结论性日志）
  for f in speed_dflash2.log speed_mtp_k1.log mtp_eff.log headcost3.log colcost.log \
           _ceiling_verdict.txt _graphchk.txt _mtp_analysis.txt _gpuprobe.txt _repo_probe.txt; do
    [ -f "$J/dl/$f" ] && cp -f "$J/dl/$f" research/logs/ 2>/dev/null
  done
  echo "  research/ 内容: $(find research -type f | wc -l) 个文件, $(du -sh research 2>/dev/null | cut -f1)"

  # 3) 暂存并看统计
  git add -A 2>&1 | head -3
  echo "  暂存条目 = $(git diff --cached --name-only | wc -l)"
  echo "  暂存体积 = $(git diff --cached --numstat | awk '{a+=$1} END{printf "%.1f 文件行数(万)\n", a/10000}')"
  biggest=$(git diff --cached --name-only | while read -r f; do [ -f "$f" ] && stat -c '%s %n' "$f"; done | sort -rn | head -3)
  echo "  暂存里最大的 3 个文件:"; echo "$biggest" | awk '{printf "    %8.2f MB  %s\n", $1/1048576, $2}'

  # 4) 提交
  git -c user.name="$(git config user.name)" -c user.email="$(git config user.email)" \
    commit -q -m "ninfer-fusion: dflash2 诊断落盘 + 图捕获默认值修复 + MTP 效率实测

- 图捕获默认值 0 -> 16（4 处）：默认 MTP k=3 从 19.5 提升到 130 tok/s（6.7x），接受率逐位不变
- BF16 词汇表头权重档（WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head）+ bf16 linear dispatch/MMA 两形
- 诊断探针：NINFER_DF2SCORES（16x16 分数矩阵 + 候选 + frontier/anchor 对齐键）
- format_kv_cache 补齐 4 档（此前 4/7 档误报 unknown）；text_prefill_impl 注释修正
- 文档：docs/maintainer/speculative-dflash2-status.md（dflash2 低支持 + 等 PR）
- research/: 分析与实测报告、脚本、关键日志
" 2>&1 | head -5
  echo "  提交 rc=$?"
  git log --oneline -2 | sed 's/^/  /'
  echo "  工作区剩余改动 = $(git status --porcelain | wc -l)"
} > "$OUT" 2>&1
echo written
