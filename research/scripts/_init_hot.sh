#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
echo "########## 1) 训练器的草稿头初始化：随机还是承接？ ##########"
grep -nE 'init|load_state_dict|from_pretrained|artifact|torch.randn|normal_|resume|copy_|no_grad|requires_grad' $J/train_dflash2.py | head -30
echo
echo "########## 2) 热窗相关旋钮（源码 + CLI） ##########"
grep -rnE 'hot_window|hot-window|"hot"|hot_tokens|hot_pages|kv_hot|--hot' $R/apps/cli/options.cpp $R/src/serve/serve_options.cpp $R/include/ninfer/types.h 2>/dev/null | grep -v '\.orig' | head -20
echo "--- 全源码里 hot 的语义点（窗口相关） ---"
grep -rnE 'hot_window|hot_window_tokens|hot_pages' $R/src $R/include 2>/dev/null | grep -v '\.orig' | head -15
echo
echo "########## 3) 我这次跑测用的配置（日志里的 summary 全量） ##########"
grep -E '^summary' $J/dl/abz_cur.log 2>/dev/null | head -30
