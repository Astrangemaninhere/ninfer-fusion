#!/bin/bash
# 收集状态到 C: 侧可读文件（避免 wsl.exe 引号陷阱）
J=/mnt/c/Users/User/Documents/ziqinzhang/dl
OUT=$J/_status.txt
{
  echo "=== 发射器输出（tail）==="
  tail -12 /home/user/bg/mtplog.out 2>/dev/null || echo "(无)"
  echo "=== dump 产物 ==="
  echo "logits 份数 = $(ls "$J"/mtplg_logits_*.bin 2>/dev/null | wc -l)"
  echo "tokens 份数 = $(ls "$J"/mtplg_tokens_*.bin 2>/dev/null | wc -l)"
  if [ -f "$J/mtplg_logits_1.bin" ]; then
    echo "logits_1 字节 = $(stat -c%s "$J/mtplg_logits_1.bin")（期望 248320*2 = 496640）"
  fi
  echo "mtplog 打印行数 = $(grep -c mtplog "$J/mtplg_run.log" 2>/dev/null || echo 0)"
  echo "=== 运行摘要 ==="
  grep -E 'decode speed|acceptance length|acceptance rate|accepted by pos' "$J/mtplg_run.log" 2>/dev/null
  echo "=== 进程/宿主 ==="
  echo "ninfer 实例 = $(pgrep -x ninfer | wc -l)"
  uptime
} > "$OUT" 2>&1
echo "written $OUT"
