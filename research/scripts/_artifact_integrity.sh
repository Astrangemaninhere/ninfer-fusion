#!/bin/bash
# 查 artifact 加载是否有完整性校验 / 物化缓存（解释"改了不生效"的可能）
R=/home/user/ninfer-fusion
echo "=== 1) src/artifact 下与校验/缓存/身份相关的点 ==="
ls -1 $R/src/artifact 2>/dev/null | head -20
grep -rnE 'md5|sha|hash|checksum|digest|verify|identity|cache' $R/src/artifact/*.cpp $R/src/artifact/*.h 2>/dev/null | grep -v '\.orig' | head -25
echo
echo "=== 2) 物化（materialize）是否按 weights_id 缓存 ==="
grep -rnE 'weights_id|model_id|materializ' $R/src/artifact/*.cpp $R/src/artifact/*.h 2>/dev/null | grep -v '\.orig' | head -20
echo
echo "=== 3) 运行时是否打印过 artifact 校验信息（在日志里找） ==="
grep -rinE 'artifact ok|objects=|identity|weights_id' /mnt/c/Users/User/Documents/ziqinzhang/dl/fx_spec.log 2>/dev/null | head -6
echo
echo "=== 4) request_log 里的 proposal_head 字段（D 指出的预检） ==="
grep -n -B 3 -A 6 'proposal_head' $R/src/serve/request_log.cpp 2>/dev/null | head -24
