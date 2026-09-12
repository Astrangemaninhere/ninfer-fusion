#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
for f in R2_vllm_same_collapse.md R3_1cat_vllm_verdict.md R1_dflash2_acceptance_findings.md; do
  echo "=================================================================="
  echo "########## $f"
  echo "=================================================================="
  cat "$J/_collab/build/$f" 2>/dev/null | head -60
  echo
done
