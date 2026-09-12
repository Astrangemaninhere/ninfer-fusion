#!/usr/bin/env python3
"""Append a 32-needle long-context stage to _spec_4way.sh (which K3 invokes later, so
the GPU lane covers the pending '8 needles was too noisy' item before training resumes).
Also record the S45d/S48 landings in _TODO.md."""
import datetime
import pathlib
import re

S = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_spec_4way.sh")
t = S.read_text(encoding="utf-8", errors="surrogateescape")

STAGE = '''
# --- long-context needle check: 32 needles (the 8-needle run was too noisy: the
# baseline itself scored 5/8 at 16K but 7/8 at 32K). MTP must MATCH the baseline;
# a speculative path that is fast and forgetful is a quality regression.
LT=$J_LONGTEST
lc() {  # label artifact spec-args...
  local label=$1 art=$2
  shift 2
  local log=/home/user/lc_$label.log
  pkill -f "$BIN" 2>/dev/null || true
  sleep 3
  : > "$log"
  ( cd /home/user/ninfer-fusion/build && setsid nohup "$BIN" "$art" --port "$PORT" \\
      --max-context 32768 --no-cuda-graph --no-thinking "$@" > "$log" 2>&1 < /dev/null & )
  local ok=0
  for _i in $(seq 1 180); do
    sleep 2
    curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && { ok=1; break; }
    pgrep -f "$BIN" >/dev/null || break
  done
  if [ $ok -eq 0 ]; then echo "[$label] SERVE_FAILED"; echo "$label|SERVE_FAILED|-|-" >> /tmp/s4w_lc; return; fi
  for ctx in 16384 32768; do
    local hits
    hits=$(timeout 1500 python3 "$LT" --port "$PORT" --context "$ctx" --needles 32 \\
             --model qwen3.8-27b 2>&1 | grep -oE 'needle hits [0-9]+/[0-9]+' | tail -1)
    echo "[$label ctx=$ctx] ${hits:-<no result>}"
    echo "$label|$ctx|${hits#needle hits }|$(grep -oE 'speculative=[a-z0-9_]+' "$log" | tail -1)" >> /tmp/s4w_lc
  done
}
J_LONGTEST=${J_LONGTEST:-/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit/longtest_57k.py}
: > /tmp/s4w_lc
if [ -f "$J_LONGTEST" ]; then
  echo "=== long-context 32-needle check $(date +%H:%M:%S) ==="
  lc lc_base    "$M/qwen3_8_27b_nvfp4.ninfer"
  lc lc_mtp3    "$M/qwen3_8_27b_nvfp4.ninfer"          --spec mtp --draft-tokens 3
  lc lc_dflash2 "$M/qwen3_8_27b_nvfp4_dflash2.ninfer"  --spec auto
  pkill -f "$BIN" 2>/dev/null || true
  {
    echo
    echo "## 长上下文 needle 32 发 ($(date +%F' '%H:%M))"
    echo
    echo "| 配置 | ctx | hits | backend |"
    echo "|---|---|---|---|"
    while IFS='|' read -r a b c d; do echo "| $a | $b | $c | $d |"; done < /tmp/s4w_lc
    echo
    echo "判读: 投机档命中数必须与基线相同; 低于基线即质量回归 (掉针=检索失败)。"
  } >> "$OUT"
else
  echo "  (longtest tool absent: $J_LONGTEST)"
fi
'''

anchor = "{\n  echo \"# 四档投机对比"
if anchor in t:
    t = t.replace(anchor, STAGE + "\n" + anchor, 1)
    S.write_text(t, encoding="utf-8", errors="surrogateescape")
    print("stage appended to _spec_4way.sh")
else:
    print("ANCHOR NOT FOUND - not modified")

T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
T.open("a", encoding="utf-8").write(f"""
### 8. 又落地两件（{stamp}）
- **S45d**（E3 的逐臂 prefill 守卫）：已应用，我独立复核 0 删除行 / +31 行 / 453→484、
  三条 dtype 条件逐字未变、每臂各自 `if constexpr + else-throw`、括号深度收 0。
- **S48**（E6 的 dspark verify 位置表 k→k+1）：已应用（`dflash_impl.h` md5 60a51d1c→02cb3bc9）。
  E6 同时纠正了 E5 的一处推断：`accepted by pos[m]` 由 verify 第 m 列决定，所以被污染的第 k 列
  只在 `a == extent == k`（全部草稿都被接受）时才有影响 ⇒ **这个 off-by-one 不是 10.3% 的成因**，
  而是高接受率恢复后的正确性前置条件。落地它不改变前 k 列（预测 draft/`p_0`/token 流逐字节不变），
  只影响全接受轮的 bonus token 与 KV 槽位映射。
- **本轮这一份编译将同时包含**：补丁 A、E2、E4、S45d、S48、`--spec` usage 文本 —— 一次 build 全覆盖。
- FlashNext/MiniCPM 的分块下载已重启（09:35 那次在 15:16 重新枚举后停住；规律仍是"整文件 GET 会卡、
  range GET 快"）。32-needle 长上下文检查已接进 `_spec_4way.sh` 的尾部（K3 的 GPU 窗口内跑）。
""")
print("todo updated")
