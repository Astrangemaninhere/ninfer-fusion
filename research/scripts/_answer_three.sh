#!/bin/bash
echo '=== ① 为什么 swap 只用了 5 GB ==='
echo "--- 当前内存/交换 ---"
free -m | sed -n '1,3p'
echo "--- 谁在占内存（前 8）---"
ps -eo rss,comm --sort=-rss | head -9 | awk 'NR==1{print "    RSS(MB)  进程"} NR>1{printf "    %8.0f  %s\n", $1/1024, $2}'
echo "--- ninfer 编译进度 ---"
echo "    nvcc=$(pgrep -c -x nvcc 2>/dev/null || echo 0)  进度=$(grep -oE '\[[ 0-9]+%\]' /mnt/c/Users/User/Documents/ziqinzhang/dl/par_build.log 2>/dev/null | tail -1)"
echo "    （SwapUsed 5GB 是"内核按需换出"的结果：只有真被压到的页才会下去，不是把所有空闲内存都搬下去）"
echo
echo '=== ② 显存现在有多少 ==='
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
echo "--- 谁的显存 ---"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null | head -5
echo
echo '=== ③ vLLM 装好了吗（后台任务已结束）==='
tail -14 /mnt/c/Users/User/Documents/ziqinzhang/dl/vllm_install.log 2>/dev/null | cut -c1-140
echo "--- 是否含 dflash2 ---"
if [ -x /mnt/c/vllm/venv-dflash2/Scripts/python.exe ]; then
  /mnt/c/vllm/venv-dflash2/Scripts/python.exe -c "import vllm,sys;print('VLLM',vllm.__version__);print(vllm.__file__)" 2>&1 | head -3
else
  echo "  （venv-dflash2 未建）"
fi
