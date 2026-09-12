#!/bin/bash
Z=/mnt/c/Users/User/Documents/ziqinzhang
for f in df2_w9_export.log df2_serve.log df2_auto.log df2_exactness.log df2_final.log df2_quality.log df2_w9_v2.log; do
  echo "############ $f ############"
  tr -d '\r' < $Z/dl/$f 2>/dev/null | head -60
  echo
done
