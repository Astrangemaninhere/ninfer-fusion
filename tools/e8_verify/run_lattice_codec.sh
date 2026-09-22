#!/usr/bin/env bash
# run_lattice_codec.sh -- build tools/e8_verify/e8_lattice_codec_test.cpp against the REAL
# tree headers and run it. Host-only: no CUDA, no cmake, no GPU, no model.
set -u
T=/home/user/ninfer-fusion
OUT=${1:-/tmp/e8lc}
echo "codec sha16: $(sha256sum "$T/src/ops/kernel/e8_lattice_codec.cuh" | cut -c1-16)"
echo "lattice sha16: $(sha256sum "$T/src/ops/kernel/e8_lattice.cuh" | cut -c1-16)"
echo "test  sha16: $(sha256sum "$T/tools/e8_verify/e8_lattice_codec_test.cpp" | cut -c1-16)"
g++ -O2 -std=c++17 -I "$T/tools/e8_verify/host_shim" -I "$T/src" \
    "$T/tools/e8_verify/e8_lattice_codec_test.cpp" -o "$OUT" 2>/tmp/e8lc.err
if [ ! -x "$OUT" ]; then echo "BUILD FAILED:"; head -40 /tmp/e8lc.err; exit 1; fi
echo "build: OK"
"$OUT"
echo "rc=$?"
