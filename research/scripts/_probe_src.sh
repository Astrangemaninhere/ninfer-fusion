#!/bin/bash
R=/home/user/ninfer-fusion
echo "################ PROBE impl program_impl.h 12420-12520 ################"
sed -n '12420,12520p' $R/src/targets/qwen3_6/impl/runtime/program_impl.h
echo
echo "################ acceptance accounting (accepted_by_pos / accept) ################"
grep -rn "accepted_by_pos\|accept_count\|spec_accept_rate\|accepted_prefix" $R/src $R/apps $R/include 2>/dev/null \
  | sed "s|$R/||" | grep -v "\.orig:" | head -30
echo
echo "################ GPU / jobs ################"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
echo "--- running compute ---"
pgrep -af "ninfer|nvcc|python.*train" 2>/dev/null | head -10
echo "--- build tree binary ---"
ls -l --time-style=+%m-%d_%H:%M $R/build/apps/ninfer 2>/dev/null
