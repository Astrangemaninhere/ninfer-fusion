#!/bin/bash
# Z1: resumable download from hf-mirror using curl -L -C -
REPO="$1"; DEST="$2"; shift 2
mkdir -p "$DEST"
for f in "$@"; do
  url="https://hf-mirror.com/${REPO}/resolve/main/${f}"
  out="${DEST}/${f}"
  mkdir -p "$(dirname "$out")"
  for try in 1 2 3 4 5 6 7 8; do
    timeout 1800 curl -L -C - --retry 3 --retry-delay 3 -s -o "$out" "$url"
    rc=$?
    if [ $rc -eq 0 ]; then
      echo "DONE  $f  $(stat -c %s "$out" 2>/dev/null) bytes"
      break
    fi
    echo "retry#$try rc=$rc  $f"
    sleep 4
  done
done
echo "=== download pass finished ==="
