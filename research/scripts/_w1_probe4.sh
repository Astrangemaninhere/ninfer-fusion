#!/bin/bash
C=/mnt/c/Users/User/Documents/ziqinzhang/_collab
A=/home/user/ninfer-fusion
B=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
echo "###### M_patchA_effect.md (head)"; head -50 "$C/M_patchA_effect.md"
echo; echo "###### M_landed_set.md"; cat "$C/M_landed_set.md"
echo; echo "###### text_context_impl.h diffstat B->A:"
diff -u "$B/src/targets/qwen3_6/impl/runtime/text_context_impl.h" \
        "$A/src/targets/qwen3_6/impl/runtime/text_context_impl.h" | grep -cE '^[+-]'
echo "--- feature/dflash related +/- lines:"
diff -u "$B/src/targets/qwen3_6/impl/runtime/text_context_impl.h" \
        "$A/src/targets/qwen3_6/impl/runtime/text_context_impl.h" | grep -E '^[+-]' | grep -iE 'feature|catch|capture|dflash|draft|aux' | head -30
