#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== text_context_impl.h 105-140 ==="
sed -n '105,140p' $R/src/targets/qwen3_6/impl/runtime/text_context_impl.h
echo
echo "=== kv_calibration.h 60-90（调用点/门控） ==="
sed -n '60,90p' $R/src/targets/qwen3_6/impl/runtime/kv_calibration.h
