#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== 探针打印格式（270-300） ==="
sed -n '270,300p' $R/src/ops/kernel/dflash2_selector.cuh
