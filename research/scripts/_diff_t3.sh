#!/bin/bash
R=/home/user/ninfer-fusion
D=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build/T3_src
echo "############ dflash2_impl.h: orig vs current ############"
diff -u <(tr -d '\r' < "$R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h.orig") \
        <(tr -d '\r' < "$R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h") | head -80
echo
echo "############ dflash_impl.h: orig vs current ############"
diff -u <(tr -d '\r' < "$R/src/targets/qwen3_6/impl/runtime/dflash_impl.h.orig") \
        <(tr -d '\r' < "$R/src/targets/qwen3_6/impl/runtime/dflash_impl.h") | head -120
echo
echo "############ dspark proposal region (dflash_impl.h 200-260) ############"
sed -n '200,260p' "$R/src/targets/qwen3_6/impl/runtime/dflash_impl.h" | tr -d '\r' | cat -n | sed 's/^/ /' | awk '{printf "%d\t%s\n", NR+199, substr($0, index($0,$2))}' | head -0
sed -n '200,260p' "$R/src/targets/qwen3_6/impl/runtime/dflash_impl.h" | tr -d '\r' > "$D/dspark_proposal_200_260.txt"
echo "written dspark_proposal_200_260.txt"
echo
echo "############ dspark proposal region (dflash_impl.h 430-500) ############"
sed -n '430,500p' "$R/src/targets/qwen3_6/impl/runtime/dflash_impl.h" | tr -d '\r' > "$D/dspark_proposal_430_500.txt"
echo "written dspark_proposal_430_500.txt"
