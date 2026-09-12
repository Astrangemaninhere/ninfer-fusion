#!/bin/bash
R=/home/user/ninfer-fusion
for f in include/ninfer/ops/dflash2_selector.h \
         src/ops/launcher/dflash2_selector.h \
         src/ops/launcher/dflash2_selector.cu \
         src/ops/wrapper/dflash2_selector.cpp; do
  echo "################ $f ($(file -b "$R/$f")) ################"
  cat -n "$R/$f"
  echo
done
