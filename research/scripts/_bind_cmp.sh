#!/bin/bash
R=/home/user/ninfer-fusion
echo "################ dflash2 + markov bindings ################"
grep -n "dflash2" $R/src/targets/qwen3_6_27b/impl/load/bindings.cpp | head -60
echo
echo "################ the selector / codebook binding block ################"
grep -n -B4 -A18 "predecessor_codebook\|successor_codebook" $R/src/targets/qwen3_6_27b/impl/load/bindings.cpp | head -80
echo
echo "################ markov bindings ################"
grep -n -B4 -A10 "markov_w1\|markov_w2" $R/src/targets/qwen3_6_27b/impl/load/bindings.cpp | head -60
