#!/bin/bash
R=/home/user/ninfer-fusion
U=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream
echo "################ apply_final_logit_policy definition(s) ################"
grep -rn "apply_final_logit_policy" $R/src $R/include 2>/dev/null | sed "s|$R/||" | grep -v "\.orig:" | head -20
echo
echo "################ its body ################"
F=$(grep -rl "void apply_final_logit_policy\|apply_final_logit_policy(" $R/src/targets 2>/dev/null | grep -v orig | head -3)
for f in $F; do echo "--- $f ---"; grep -n -A 25 "apply_final_logit_policy" "$f" | head -45; done
echo
echo "################ logit policy knobs (scale/softcap/temperature) ###############"
grep -rn "logit_scale\|final_logit\|softcap\|soft_cap\|logit_policy" $R/src/targets/qwen3_6_27b/impl/config.h $R/src/targets/qwen3_6/impl/*.h 2>/dev/null | sed "s|$R/||" | grep -v orig | head -25
echo
echo "################ UPSTREAM: does it apply the policy in the draft path? ###############"
grep -n "apply_final_logit_policy\|linear_topk" $U/src/targets/qwen3_6/impl/runtime/dflash_impl.h | head
echo
echo "################ OURS: kCfg in dflash2 path ################"
grep -n "kCfg\|using.*Cfg" $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h | head
