#!/bin/bash
D=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream/docs/maintainer/qwen3.8-27b-dflash2.md
echo "################ lines mentioning feature / 25600 / concat / order ################"
grep -n "25600\|feature\|concat\|order\|H_f" $D | head -40
echo
echo "################ sections mentioning u_i / unary / codebook / scale ################"
grep -n "u_i\|unary\|codebook\|scale\|W_pred\|W_succ" $D | head -40
echo
echo "################ doc outline ################"
grep -n "^#" $D | head -40
