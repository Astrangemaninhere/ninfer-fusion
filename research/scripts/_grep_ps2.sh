#!/bin/bash
cd /home/user/ninfer-fusion || exit 1
echo "--- dflash2_selector 调用点 ---"
grep -rn "dflash2_selector" --include=*.h --include=*.cuh --include=*.cpp --include=*.cu src/ | head -40
echo
echo "--- selector kernel 签名与 pair_scale 传递链 ---"
grep -rn "pair_scale\|selector_scale\|sel_scale" --include=*.h --include=*.cuh --include=*.cpp --include=*.cu src/ | head -40
