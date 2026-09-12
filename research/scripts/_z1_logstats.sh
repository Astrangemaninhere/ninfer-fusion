#!/bin/bash
echo "--- arg_utils disable_log_stats ---"
grep -n "disable_log_stats" /mnt/c/vllm/venv/Lib/engine 2>/dev/null
grep -n "disable_log_stats" /mnt/c/vllm/venv/Lib/site-packages/vllm/engine/arg_utils.py | head
echo "--- scheduler log_stats ---"
grep -n "log_stats" /mnt/c/vllm/venv/Lib/site-packages/vllm/v1/core/sched/scheduler.py | head
echo "--- where log_stats is passed ---"
grep -rn "log_stats=" /mnt/c/vllm/venv/Lib/site-packages/vllm/v1/ --include=*.py | head
echo "--- prom names ---"
grep -n "spec_decode" /mnt/c/vllm/venv/Lib/site-packages/vllm/v1/spec_decode/metrics.py | head -30
