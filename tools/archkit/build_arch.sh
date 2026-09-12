#!/usr/bin/env bash
# build_arch.sh — multi-GPU build matrix helper (CPU-only compilation; run on
# the target toolchain, no target GPU required).
#
# Usage:
#   ./build_arch.sh 86            # one arch -> build/ninfer-serve-sm86
#   ./build_arch.sh 75 86 89 120a # several archs, one dist dir each
#
# Each arch gets its own build dir so binaries can be shipped per GPU tier
# (see tools/archkit/_GPU_MATRIX.md). Defaults to Release + apps.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

ARCHS=("$@")
if [ "${#ARCHS[@]}" -eq 0 ]; then
  echo "usage: $0 <arch> [arch...]   e.g. $0 86 89 120a" >&2
  exit 2
fi

GEN="${NINFER_CMAKE_GENERATOR:-Ninja}"
for arch in "${ARCHS[@]}"; do
  bdir="build-${arch}"
  echo "== configuring ${bdir} (arch ${arch})"
  cmake -S . -B "${bdir}" -G "${GEN}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${arch}" \
    -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF
  echo "== building ${bdir}"
  cmake --build "${bdir}" -j"${NINFER_JOBS:-$(nproc)}"
  echo "== ${bdir} done:"
  ls -la "${bdir}/apps/ninfer-serve" 2>/dev/null || true
done
echo "matrix build complete"
