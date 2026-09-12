#!/bin/bash
echo "=== 1) GPU 实况（占用/显存/进程） ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -3
echo "--- GPU 上的进程 ---"
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null | head -5
echo
echo "=== 2) 参考跑进程的内存/CPU ==="
ps -eo pid,etimes,pcpu,pmem,rss,args --sort=-rss | grep -E 'run_df2_ref|vllm.entrypoints' | grep -v grep | cut -c1-130 | head -4
echo
echo "=== 3) 系统内存 ==="
free -g | head -3
echo
echo "=== 4) 重试日志 ==="
cat /mnt/c/Users/User/Documents/ziqinzhang/dl/ref_retry.log 2>/dev/null | tail -8
echo "--- runner stdout ---"
tail -6 /mnt/c/Users/User/Documents/ziqinzhang/dl/vref_f4_stdout.log 2>/dev/null
