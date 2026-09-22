#!/bin/bash
# gqaattn measure bundle for ONE binary.  Usage: bash s50_measure.sh <tag> <binary>
# Produces, under /mnt/g/gqaattn/m/<tag>/ :
#   plain arms (no instrument): 256-token decode speed + the 256-token id stream sha16
#   nsys arms x2 @256 tokens  -> the per-kernel NOISE FLOOR (same binary, same window)
#   nsys arms x2 @32  tokens  -> the small-window FALSE-POSITIVE demo
#   help.txt                  -> the 0-line-diff check
# E1: --kv-row-scale off on every shot; kvrs table sha256 recorded and gated.
set -u
TAG="${1:?tag}"; BIN="${2:?binary}"; MC="${3:-4096}"
NSYS=/usr/local/cuda-13.3/bin/nsys; [ -x "$NSYS" ] || NSYS=/usr/local/cuda/bin/nsys
MODEL=/home/user/models/qwen3_8_27b_nvfp4.ninfer
PROMPT=/mnt/g/mtp15/p_4096.json
KVRS_EXPECT=fecbebc8096f33bf7a06f11c6dacc42e5711fbae74db3cbebc398c3bb5510713
O=/mnt/g/gqaattn/m/$TAG
mkdir -p "$O"
echo "=== measure tag=$TAG bin=$BIN maxctx=$MC $(date -Is) ==="
sha256sum "$BIN" | tr -s ' ' | cut -c1-16 | tee "$O/bin_sha16"
sha256sum "$MODEL.kvrowscale.bin" | cut -c1-64 | tee "$O/kvrs_sha256"
"$BIN" --help > "$O/help.txt" 2>&1; echo "help rc=$? lines=$(wc -l < "$O/help.txt")"

one () {  # $1=name  $2=maxnew  $3=extra
  local n="$1" mn="$2" ex="${3:-}"
  local d="$O/$n"; mkdir -p "$d"
  local T0=$(date +%s)
  ( cd /tmp && "$BIN" "$MODEL" --messages "$PROMPT" --max-context "$MC" \
      --kv-dtype nvfp4 --kv-row-scale off --max-new "$mn" $ex \
      --no-thinking --greedy --presence-penalty 0 --print-token-ids --raw-output \
      --spec none ) > "$d/stdout.txt" 2> "$d/stderr.txt"
  local rc=$?; local T1=$(date +%s)
  local dec=$(grep -oE 'decode speed[^0-9]*([0-9]+\.?[0-9]*)' "$d/stderr.txt" | grep -oE '[0-9]+\.?[0-9]*' | tail -1)
  local pre=$(grep -oE 'prefill speed[^0-9]*([0-9]+\.?[0-9]*)' "$d/stderr.txt" | grep -oE '[0-9]+\.?[0-9]*' | tail -1)
  local ids=$(grep -oE 'generated ids[ ]+.*' "$d/stdout.txt" | tr -s ' ' | sha256sum | cut -c1-16)
  local nt=$(grep -oE 'generated tokens[^0-9]*([0-9]+)' "$d/stderr.txt" | grep -oE '[0-9]+' | tail -1)
  echo "rc=$rc dec=${dec:-?} pre=${pre:-?} gen=${nt:-?} wall=$((T1-T0))s ids=$ids" | tee "$d/verdict.txt"
  sha256sum "$MODEL.kvrowscale.bin" | cut -c1-64 > "$d/kvrs_post"
}

nshot () { # $1=name  $2=maxnew
  local n="$1" mn="$2"; local d="$O/$n"; mkdir -p "$d"
  echo "--- nsys shot $n maxnew=$mn $(date -Is) ---"
  ( cd /tmp && "$NSYS" profile --force-overwrite=true -o "$d/prof" --trace=cuda,nvtx \
      "$BIN" "$MODEL" --messages "$PROMPT" --max-context "$MC" \
      --kv-dtype nvfp4 --kv-row-scale off --max-new "$mn" \
      --no-thinking --greedy --presence-penalty 0 --print-token-ids --raw-output \
      --spec none ) > "$d/stdout.txt" 2> "$d/stderr.txt"
  grep -E 'decode speed|prefill speed' "$d/stderr.txt" | sed 's/^/  /'
  grep -oE 'generated ids[ ]+.*' "$d/stdout.txt" | tr -s ' ' | sha256sum | cut -c1-16 > "$d/ids_sha16"
  timeout 900 "$NSYS" export --type sqlite --force-overwrite=true -o "$d/prof.sqlite" "$d/prof.nsys-rep" > "$d/export.txt" 2>&1
  echo "  export rc=$? sqlite=$(stat -c %s "$d/prof.sqlite" 2>/dev/null)"
  rm -f "$d/prof.nsys-rep" "$d/prof.qdrep" "$d/prof.sqlite-journal" 2>/dev/null
}

one dec256_a 256
one dec256_b 256
one dec32_a   32
nshot nsys256_a 256
nshot nsys256_b 256
nshot nsys32_a  32
nshot nsys32_b  32
echo "=== measure $TAG done $(date -Is) ==="
