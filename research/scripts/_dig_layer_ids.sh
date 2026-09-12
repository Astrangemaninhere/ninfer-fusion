#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo '=== dspark-pipeline.py 里 hs_layers_env 的用法（前后 12 行）==='
n=$(grep -n 'VLLM_DUMP_HS_LAYERS' "$J/dspark-pipeline.py" | head -1 | cut -d: -f1)
awk -v s=$((n-12)) -v e=$((n+6)) 'NR>=s && NR<=e {printf "%4d| %s\n", NR, $0}' "$J/dspark-pipeline.py" | cut -c1-140
echo
echo '=== TARGET_LAYER_IDS 的定义 ==='
grep -n 'TARGET_LAYER_IDS' "$J/dspark-pipeline.py" | head -5 | cut -c1-130
echo
echo '=== config 里的 hs_layers_env（原始）==='
grep -rn 'hs_layers_env' "$J"/*.json "$J"/data/*.json "$J"/collect-hs* 2>/dev/null | head -5 | cut -c1-140
echo
echo '=== 草稿 config 的 target_layer_ids 完整块 ==='
awk 'NR>=36 && NR<=44 {printf "%4d| %s\n", NR, $0}' "$J/data/draft_model/config.json" | cut -c1-90
echo
echo '=== 训练脚本里层号怎么用来取 hidden ==='
grep -n 'target_layer\|layer_ids\|hs_layer\|TARGET_LAYER' /mnt/c/Users/User/Documents/ziqinzhang/train_dspark.py 2>/dev/null | head -8 | cut -c1-140
