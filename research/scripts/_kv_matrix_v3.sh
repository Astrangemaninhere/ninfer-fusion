#!/bin/bash
# temp: KV strategy matrix v3 — global tiers + layered mixes, with a correct readiness gate.
# Per config: KV payload bytes/token (request-log JSONL), prefill/decode tok/s, needle quality.
set -u
PORT=8350
BIN=${NINFER_SERVE_BIN:-/home/user/ninfer-fusion/build/apps/ninfer-serve}
MODEL=${KV_MODEL:-/home/user/models/qwen3_8_27b_nvfp4.ninfer}
MODEL_ID=${KV_MODEL_ID:-qwen3.8-27b}
CTX=${KV_CTX:-65536}
NEEDLE_CTX=${KV_NEEDLE_CTX:-32768}
NEEDLES=${KV_NEEDLES:-8}
LONGTEST=/mnt/c/Users/User/Documents/ziqinzhang/NI2A3F~1/tools/archkit/longtest_57k.py
OUT=${KV_OUT:-/home/user/kv_matrix_v3.csv}

# config = "dtype|table|label"
CONFIGS=(
  "bf16||all-bf16"
  "int8||all-int8"
  "fp8||all-fp8"
  "nvfp4||all-nvfp4"
  "iso3||all-iso3"
  "nvfp4|all:e8|all-e8"
  "nvfp4|0-7:e8|8e8+8nvfp4"
  "nvfp4|0-11:e8|12e8+4nvfp4"
  "nvfp4|0-3:e8,4-7:int8|4e8+4int8+8nvfp4"
)

echo "# device=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) model=$MODEL_ID ctx=$CTX needle_ctx=$NEEDLE_CTX needles=$NEEDLES" > "$OUT"
echo "label,dtype,table,payload_mib,bytes_per_token,prefill_tps,decode_tps,needle" >> "$OUT"

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r dtype table label <<< "$entry"
  pkill -x ninfer-serve 2>/dev/null
  sleep 4
  log=/home/user/kvm3_${label}.log
  jsonl=/tmp/kvm3_${label}.jsonl
  : > "$log"; rm -f "$jsonl"
  extra=()
  [ -n "$table" ] && extra=(--kv-layer-storage "$table")
  cd /home/user/ninfer-fusion/build || exit 1
  setsid nohup "$BIN" "$MODEL" --port "$PORT" --kv-dtype "$dtype" --max-context "$CTX" \
    --no-cuda-graph --request-log-jsonl "$jsonl" "${extra[@]}" > "$log" 2>&1 < /dev/null &
  ok=0
  for _i in $(seq 1 180); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then ok=1; break; fi
    if ! pgrep -x ninfer-serve >/dev/null; then break; fi
  done
  if [ $ok -eq 0 ]; then
    echo "$label,$dtype,$table,SERVE_FAILED,,,,," >> "$OUT"
    echo "[$label] SERVE_FAILED"; tail -2 "$log"
    continue
  fi
  sleep 2
  toks=$(grep -oE 'resolved=[0-9]+ tokens' "$log" | head -1 | grep -oE '[0-9]+')
  payload=$(python3 - "$jsonl" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            memory = record.get("memory") or {}
            if "kv_payload_bytes" in memory:
                print(memory["kv_payload_bytes"])
                break
except Exception:
    pass
PY
)
  mib=$(python3 -c "print(round(int('${payload:-0}')/1048576,1))" 2>/dev/null || echo "?")
  bpt=$(python3 -c "print(round(int('${payload:-0}')/max(1,int('${toks:-1}')),1))" 2>/dev/null || echo "?")
  ctx_text=$(python3 -c "print('这是一段用于测量预填充与解码速度的中文文本。' * 400, end='')")
  curl -s -m 600 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"$ctx_text\n\n一句话总结\"}],\"max_tokens\":32,\"temperature\":0}" \
    > /tmp/kvm3_speed.json
  speed=$(grep -oE 'prefill=[0-9.]+tok/s decode=[0-9.]+tok/s' "$log" | tail -1)
  pre=$(echo "$speed" | grep -oE 'prefill=[0-9.]+' | cut -d= -f2)
  dec=$(echo "$speed" | grep -oE 'decode=[0-9.]+' | cut -d= -f2)
  needle=$(python3 "$LONGTEST" --port "$PORT" --context "$NEEDLE_CTX" --needles "$NEEDLES" \
    --model "$MODEL_ID" 2>&1 | grep -oE 'needle hits [0-9]+/[0-9]+' | tail -1 | tr ' ' '_')
  echo "$label,$dtype,$table,$mib,$bpt,${pre:-?},${dec:-?},${needle:-?}" >> "$OUT"
  echo "[$label] payload=${mib}MiB bpt=${bpt} prefill=${pre} decode=${dec} ${needle:-no-needle-line}"
done
pkill -x ninfer-serve 2>/dev/null
echo "KV_MATRIX_V3_DONE out=$OUT"
