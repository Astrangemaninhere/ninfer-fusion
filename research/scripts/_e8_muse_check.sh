#!/bin/bash
# Muse + e8 KV check — the ONE e8 site that _e8_postfix.sh cannot reach.
# _e8_postfix.sh runs qwen3.8-27b (4 KV heads => decode/prefill path). Muse has
# 2 KV heads, so its bulk page-fill kernel (gqa_attention_prefill_i8.cuh:260-276)
# only executes here. Per _TODO.md 116c that kernel is KNOWN BROKEN (no Hadamard
# rotation + /127 divisor), so a LOW e8 score versus plain is the expected
# pre-fix symptom, not a surprise. After the fix this script must show parity.
set -u
PORT=8371
BIN=${NINFER_SERVE_BIN:-/home/user/ninfer-fusion/build/apps/ninfer-serve}
MODEL=${KV_MODEL:-/home/user/models/muse_glimmer_30b_nvfp4.ninfer}
LONGTEST=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit/longtest_57k.py
declare -A HITS=()

[ -x "$BIN" ] || { echo "FATAL: serve binary missing: $BIN"; echo MUSE_E8_VERDICT=FAIL; exit 2; }
[ -f "$MODEL" ] || { echo "FATAL: model missing: $MODEL"; echo MUSE_E8_VERDICT=FAIL; exit 2; }

run() {  # label extra-args...
  local label=$1
  shift
  pgrep -af ninfer-serve | sed 's/^/  kill? /'    # 铁律②: 杀之前先看 cmdline
  pkill -f "$BIN" 2>/dev/null || true
  sleep 3
  local log=/home/user/musee8_$label.log
  : > "$log"
  cd /home/user/ninfer-fusion/build || exit 1
  setsid nohup "$BIN" "$MODEL" --port "$PORT" --max-context 32768 --no-cuda-graph \
    "$@" > "$log" 2>&1 < /dev/null &
  local ok=0
  for _i in $(seq 1 150); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then ok=1; break; fi
    if ! pgrep -f "$BIN" >/dev/null; then break; fi
  done
  if [ $ok -eq 0 ]; then
    # Distinguish "flag rejected" from "crashed later": both are honest failures here.
    HITS[$label]="SERVE_FAILED"
    echo "[$label] SERVE_FAILED"
    grep -iE 'invalid|unknown|error|reject|not supported' "$log" | head -2
    tail -2 "$log"
    return
  fi
  local line
  line=$(python3 "$LONGTEST" --port "$PORT" --context 32768 --needles 8 --model muse-glimmer-30b 2>&1 \
    | grep -oE 'needle hits [0-9]+/[0-9]+' | tail -1)
  HITS[$label]="${line#needle hits }"
  echo "[$label] $line"
}

# NOTE (2026-09-10): the e8 lines used to pass ONE quoted string without a
# leading --, which the server rejects as `unknown argument: kv-dtype …`. Same
# family as the bug found in _e8_postfix.sh; grep the whole script family when
# one instance is found (the other two lookalikes are legitimate: start_serve
# takes $1 unquoted, _mtp_width_probe uses $KVARGS unquoted).
run plain_nvfp4 --kv-dtype nvfp4
run e8_0_7 --kv-dtype nvfp4 --kv-layer-storage 0-7:e8
run e8_8_15 --kv-dtype nvfp4 --kv-layer-storage 8-15:e8
run plain_bf16 --kv-dtype bf16

echo "=== verdict ==="
verdict=PASS
for k in plain_nvfp4 e8_0_7 e8_8_15 plain_bf16; do
  echo "  $k = ${HITS[$k]:-MISSING}"
  [[ "${HITS[$k]:-}" =~ ^[0-9]+/[0-9]+$ ]] || { verdict=FAIL; echo "  !! $k 未产出计数"; }
done
if [ "$verdict" = PASS ]; then
  p=${HITS[plain_nvfp4]%%/*}
  for k in e8_0_7 e8_8_15; do
    d=$(( ${HITS[$k]%%/*} - p ))
    echo "  $k - plain = $d"
    # §116c: 修复前这里预期为负 (Muse page-fill 写废值); 修复后应为 0 或正。
    [ "$d" -lt 0 ] && { verdict=REVIEW; echo "  !! $k 低于 plain ⇒ 符合 §116c 已知缺陷的特征"; }
  done
fi
echo MUSE_E8_VERDICT=$verdict
pgrep -af ninfer-serve | sed 's/^/  kill? /'
pkill -f "$BIN" 2>/dev/null || true
echo MUSE_E8_DONE
