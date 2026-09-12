#!/bin/bash
cd /home/user/ninfer-fusion || exit 1
echo "--- env 消费点 ---"
grep -rn "DF2_PAIR_SCALE\|pair_scale" --include=*.h --include=*.cuh --include=*.cpp src/ | head -40
echo "--- getenv 总览(dflash2 相关) ---"
grep -rn "getenv" --include=*.h --include=*.cuh --include=*.cpp src/targets/qwen3_6/impl/runtime/dflash2_impl.h src/ops/kernel/dflash2_selector.cuh | head -40
