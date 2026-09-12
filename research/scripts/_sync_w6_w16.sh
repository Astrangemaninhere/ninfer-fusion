#!/bin/bash
# temp: md5-verified sync of the W6/W16 host-side changes (Windows -> WSL build tree).
# Run only when no build is in flight: engine_core.h/scheduler.h invalidate host TUs.
set -u
WIN=/mnt/c/Users/User/Documents/ziqinzhang/NI2A3F~1
WSL=/home/user/ninfer-fusion
FILES=(
  include/ninfer/engine.h
  src/runtime/engine/bandwidth_governor.h
  src/runtime/engine/engine.cpp
  src/runtime/engine/engine_core.h
  src/runtime/engine/scheduler.h
  src/serve/generation_service.cpp
  src/serve/generation_service.h
  src/serve/http_server.cpp
  src/serve/http_server.h
  src/targets/qwen3_6/impl/runtime/program.h
  src/targets/qwen3_6/impl/runtime/program_impl.h
  tests/test_bandwidth_governor.cpp
  tests/CMakeLists.txt
)
fail=0
for f in "${FILES[@]}"; do
  if ! cp -f "$WIN/$f" "$WSL/$f"; then echo "COPY_FAIL $f"; fail=1; continue; fi
  a=$(md5sum "$WIN/$f" | cut -d' ' -f1)
  b=$(md5sum "$WSL/$f" | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then echo "SYNC_OK $f"; else echo "SYNC_MISMATCH $f ($a != $b)"; fail=1; fi
done
if [ $fail -eq 0 ]; then echo "SYNC_ALL_OK"; else echo "SYNC_HAS_FAILURES"; fi
