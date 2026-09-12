#!/usr/bin/env bash
# _df2_ab.sh -- A/B acceptance harness for DFlash2 draft artifacts.
# Runs BOTH artifacts through the identical CLI command + fixture and prints a
# side-by-side table of the speculative summary + decode speed, then a verdict.
#
# Usage:  bash _df2_ab.sh [A=<artifact.ninfer>] [B=<artifact.ninfer>]
#         (key=value or bare positional; defaults: A = baseline
#          /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer,
#          B = $B_ARTIFACT env var -- required if not passed)
# Env:    NINFER_BIN   executable (default ~/ninfer/build/apps/ninfer)
#         FIXTURE      messages json (default /home/user/prefill-probe-msg.json)
#         MAX_NEW      (default 200)
# Greps are prefix-agnostic on purpose: the engine prints dflash2 metrics with
# an "mtp" label (apps/cli/main.cpp:222 ternary bug).
set -uo pipefail
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64

A="/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"
B="${B_ARTIFACT:-}"
NINFER_BIN="${NINFER_BIN:-$HOME/ninfer/build/apps/ninfer}"
FIXTURE="${FIXTURE:-/home/user/prefill-probe-msg.json}"
MAX_NEW="${MAX_NEW:-200}"
for arg in "$@"; do
  case "$arg" in
    A=*) A="${arg#A=}" ;;
    B=*) B="${arg#B=}" ;;
    *)   if [ -z "${A_SET:-}" ]; then A="$arg"; A_SET=1; else B="$arg"; fi ;;
  esac
done
[ -n "$B" ] || { echo "ERROR: B artifact not set (pass B=<path> or export B_ARTIFACT)"; exit 2; }
for f in "$A" "$B" "$FIXTURE" "$NINFER_BIN"; do
  [ -e "$f" ] || { echo "ERROR: not found: $f"; exit 2; }
done

# Refuse to run while the draft trainer is alive (Windows python via WSL interop).
ALIVE="$(powershell.exe -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' }) { 'ALIVE' }" 2>/dev/null | tr -d '\r')"
if [ "$ALIVE" = "ALIVE" ]; then
  echo "ABORT: train_dflash2 python is still running (GPU busy). Retry after training exits."
  exit 3
fi
if pgrep -af train_dflash2 >/dev/null 2>&1; then
  echo "ABORT: train_dflash2 process detected inside WSL. Retry after training exits."
  exit 3
fi

run_one() {  # $1 label, $2 artifact, $3 log
  "$NINFER_BIN" "$2" \
    --messages "$FIXTURE" --max-new "$MAX_NEW" --max-context 32768 --kv-capacity 32768 \
    --prefill-chunk 8192 --kv-dtype nvfp4 --temperature 1.0 --seed 7 \
    --spec dflash2 --draft-tokens 7 > /dev/null 2> "$3" || true
}
field() {  # $1 log, $2 metric name (grep pattern) -> last token of first match
  grep -m1 "$2" "$1" 2>/dev/null | awk '{print $NF}'
}

echo "== A: $A"
echo "== B: $B"
run_one A "$A" /home/user/df2-ab-A.log
run_one B "$B" /home/user/df2-ab-B.log

# Prefix-agnostic summary keys (engine may label the backend "mtp").
rows=("draft window:window" "rounds:rounds" "drafted tokens:drafted tokens" \
      "accepted tokens:accepted tokens" "acceptance rate:acceptance rate" \
      "acceptance length:acceptance length" "accepted by pos:accepted by pos" \
      "decode speed:decode speed")
printf "%-18s | %-28s | %s\n" "metric" "A (baseline)" "B (tuned)"
printf "%-18s-+-%-28s-+-%s\n" "------------------" "----------------------------" "----------------------------"
for r in "${rows[@]}"; do
  k="${r%%:*}"; pat="${r#*:}"
  va="$(field /home/user/df2-ab-A.log "$pat")"; vb="$(field /home/user/df2-ab-B.log "$pat")"
  printf "%-18s | %-28s | %s\n" "$k" "${va:-MISSING}" "${vb:-MISSING}"
done
ra="$(field /home/user/df2-ab-A.log 'acceptance rate')"; rb="$(field /home/user/df2-ab-B.log 'acceptance rate')"
na="${ra%\%}"; nb="${rb%\%}"
if [ -z "$na" ] || [ -z "$nb" ]; then echo "VERDICT: TIE (missing acceptance rate; inspect logs)"; exit 4; fi
verdict="$(awk -v a="$na" -v b="$nb" 'BEGIN{d=b-a; if(d>0.1) print "B_BETTER"; else if(d<-0.1) print "A_BETTER"; else print "TIE"}')"
echo "VERDICT: $verdict (acceptance A=$ra B=$rb, threshold 0.1%)"
