#!/bin/bash
D=/mnt/c/Users/User/Documents/ziqinzhang/dl
echo "=== clean_vllm.log: 配置与尾部 ==="
grep -an 'method=\|speculative_config\|Serving\|Started server\|EngineCore' "$D/clean_vllm.log" 2>/dev/null | head -8
echo "--- tail 6 ---"
tail -6 "$D/clean_vllm.log" 2>/dev/null
echo
echo "=== 所有日志里出现过的 vLLM spec method ==="
grep -aoE '"method": *"[a-z]+"' "$D"/*.log 2>/dev/null | sort | uniq -c | sort -rn | head -8
echo
echo "=== data/draft_model 是否存在（那次 vLLM 用的草稿） ==="
ls -l /mnt/c/Users/User/Documents/ziqinzhang/data/draft_model 2>&1 | head -8
