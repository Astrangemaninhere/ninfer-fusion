#!/bin/bash
# A3 三份 new_op 补丁 + A5 探针：dry-run 并标出「新增文件 / 改现有文件」。
# 新增文件会触发 cmake 重新配置（= 全量重编），所以必须分清。
R=/home/user/ninfer-fusion
C=/mnt/c/Users/User/Documents/ziqinzhang/_collab

for f in A3_spark_head_geometry.diff A3_spark_headwise_gate.diff A3_spark_gelu_mul.diff A5_round_probe.diff; do
  p="$C/$f"
  [ -f "$p" ] || { echo "[MISSING] $f"; continue; }
  echo "=== $f ==="
  # 补丁的换行可能两种，先原样再 strip
  if patch -p1 --dry-run -d "$R" < "$p" >/dev/null 2>&1; then
    applied=raw
  else
    tr -d '\r' < "$p" > "/tmp/a3_$f"
    if patch -p1 --dry-run -d "$R" < "/tmp/a3_$f" >/dev/null 2>&1; then applied=crstrip; else applied=NONE; fi
  fi
  echo "  可打性: $applied"
  [ "$applied" = NONE ] && { patch -p1 --dry-run -d "$R" < "$p" 2>&1 | grep -E 'FAILED|Hunk' | head -5; }
  echo "  触及文件:"
  grep -E '^\+\+\+ ' "$p" | sed 's|^+++ b/||' | sort -u | while read -r rel; do
    if [ -e "$R/$rel" ]; then echo "     [改] $rel"; else echo "     [新] $rel"; fi
  done
  n_new=$(grep -E '^\+\+\+ ' "$p" | sed 's|^+++ b/||' | sort -u | while read -r rel; do [ -e "$R/$rel" ] || echo x; done | wc -l)
  echo "  -> 新增文件数: $n_new"
  echo
done
