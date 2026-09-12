#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
echo "########## A) 训练器有没有"从指定 ckpt 初始化"的旗标（决定怎么续训） ##########"
sed -n '283,315p' $J/train_dflash2.py
echo "--- 装载段 355-375 ---"
sed -n '355,375p' $J/train_dflash2.py
echo
echo "########## B) 窗口类旋钮（热窗/滑窗）在 CLI 与变体默认里 ##########"
grep -nE '"--(sliding|window|kv-window|attn-window|hot)' $R/apps/cli/options.cpp | head -12
echo "--- sliding_window_tokens / implementation_window 的默认与来源 ---"
grep -rnE 'sliding_window_tokens|implementation_window|min_visible_keys|max_visible_keys' $R/src/targets/qwen3_6_27b/impl/variant.cpp $R/src/targets/qwen3_6/impl/runtime/layouts_impl.h 2>/dev/null | grep -v '\.orig' | head -15
echo
echo "########## C) kv 存储/预算相关旗标 ##########"
grep -nE '"--(kv-bit-budget|kv-layer-storage|kv-dtype|max-cold-pages|cold-policy)' $R/apps/cli/options.cpp | head -10
