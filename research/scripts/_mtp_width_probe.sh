#!/bin/bash
# _mtp_width_probe.sh -- round-time probe for the DDTree verify-cost model.
# Runs the CLI twice on the same ~1.5K-char prompt with --spec mtp at draft
# width 1 (verify = 2 tokens) and width 5 (verify = 6 tokens), then prints
# rounds / drafted / accepted / acceptance / tok/s and computed ms/round for
# each plus the ratio. Decision context (L=4 DDTree = ~32 verify tokens):
#   ratio(w5/w1) <= 1.2  -> verify is still weight-bandwidth-bound; an L=4
#                           tree's verify is ~free and the W9 gate stands.
#   ratio > 1.5          -> compute-bound already at 6 tokens; re-derive the
#                           gate (expect L=2/4 only, tighter hit@4-hit@1).
# Env overrides (sane defaults = WSL box):
#   ARTIFACT=/home/user/models/qwen3_8_27b_nvfp4.ninfer
#   PROMPT_FILE=/home/user/spec_prompt_base.txt
#   MAXNEW=512   CHARS=1500   GRAPH_FLAG=--no-cuda-graph   KVARGS="--kv-dtype nvfp4"
# Run in the GPU window:  bash _mtp_width_probe.sh
set -u
ARTIFACT=${ARTIFACT:-/home/user/models/qwen3_8_27b_nvfp4.ninfer}
PROMPT_FILE=${PROMPT_FILE:-/home/user/spec_prompt_base.txt}
MAXNEW=${MAXNEW:-512}
CHARS=${CHARS:-1500}
GRAPH_FLAG=${GRAPH_FLAG:---no-cuda-graph}
KVARGS=${KVARGS:-"--kv-dtype nvfp4 --max-context 32768 --kv-capacity 32768"}

cd /home/user/ninfer-fusion
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64
[ -x ./build/apps/ninfer ] || { echo "FATAL: ./build/apps/ninfer not built"; exit 1; }
[ -f "$ARTIFACT" ] || { echo "FATAL: artifact $ARTIFACT missing (ARTIFACT=...)"; exit 1; }
[ -f "$PROMPT_FILE" ] || { echo "FATAL: prompt file $PROMPT_FILE missing (PROMPT_FILE=...)"; exit 1; }

PROMPT=$(python3 -c "
s = open('$PROMPT_FILE', encoding='utf-8', errors='ignore').read()
print(s[:$CHARS])
")

for W in 1 5; do
  LOG=/tmp/mtp_width_probe_w${W}.log
  echo "===== spec=mtp draft-tokens=$W  (verify = $((W+1)) tokens) ====="
  T0=$(date +%s%N)
  timeout 900 ./build/apps/ninfer "$ARTIFACT" --prompt "$PROMPT" $KVARGS \
    --spec mtp --draft-tokens $W --greedy --max-new $MAXNEW --no-thinking $GRAPH_FLAG \
    > "$LOG" 2>&1
  RC=$?
  T1=$(date +%s%N)
  [ $RC -ne 0 ] && { echo "RUN_FAIL w=$W rc=$RC (see $LOG)"; exit 2; }
  RAW=$(grep -aiE 'decode speed|acceptance rate|rounds|fallback|drafted tokens|accepted tokens' \
        "$LOG" | tr -d '\000' | head -8)
  echo "$RAW"
  ROUNDS=$(grep -ai 'rounds' "$LOG" | tr -d '\000' | grep -aoE '[0-9]+' | head -1)
  TOKS=$(grep -ai 'decode speed' "$LOG" | tr -d '\000' | grep -aoE '[0-9]+\.?[0-9]*' | head -1)
  if [ -z "$ROUNDS" ] || [ -z "$TOKS" ] || [ "$TOKS" = "0" ]; then
    echo "PARSE_FAIL w=$W (see $LOG)"; MS_R=""; continue
  fi
  # CAVEAT: ms/round below is a decode-throughput proxy (MAXNEW/tok_s/rounds),
  # not a verify-kernel timer. The 5-layer draft net is negligible next to the
  # 27B target, so round time ~= verify time -- but the wall cross-check below
  # says so instead of assuming it (prefill/startup inflate wall).
  MS_R=$(awk -v n="$MAXNEW" -v t="$TOKS" -v r="$ROUNDS" \
    'BEGIN{printf "%.2f", (n/t*1000.0)/r}')
  WALL_R=$(awk -v a="$T0" -v b="$T1" -v r="$ROUNDS" 'BEGIN{printf "%.2f", (b-a)/1e6/r}')
  DIV=$(awk -v c="$MS_R" -v w="$WALL_R" 'BEGIN{d=(w-c)/c; if(d<0)d=-d; printf "%.1f", d*100}')
  echo "w=$W: rounds=$ROUNDS tok/s=$TOKS ms/round=$MS_R wall_per_round_ms=$WALL_R (divergence ${DIV}%)"
  if awk -v d="$DIV" 'BEGIN{exit !(d>15)}'; then
    echo "  SEE-NVTX: wall vs decode proxy diverge >15% (prefill/startup or non-verify cost); run _df2_nsys.sh for a verify-only split"
  fi
  eval "MS_W${W}=\$MS_R"
done

if [ -n "${MS_W1:-}" ] && [ -n "${MS_W5:-}" ]; then
  RATIO=$(awk -v a="$MS_W5" -v b="$MS_W1" 'BEGIN{printf "%.3f", a/b}')
  VERDICT=$(awk -v r="$RATIO" 'BEGIN{
    if (r <= 1.2) print "BANDWIDTH-BOUND: L=4 tree verify ~free, W9 gate stands";
    else if (r <= 1.5) print "MIXED: marginal, prefer L=2/4 packing";
    else print "COMPUTE-BOUND: re-derive gate before building trees"}')
  echo "===== ratio ms/round(w5)/ms/round(w1) = $RATIO  ->  $VERDICT ====="
else
  echo "===== ratio unavailable (one run failed/parsed empty) ====="
  exit 2
fi
