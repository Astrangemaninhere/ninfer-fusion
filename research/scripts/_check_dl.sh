#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang/dl
echo "--- 我的关键产物 ---"
for p in df2scores_scores_1.bin df2scores_cand_1.bin df2scores_unary_1.bin df2scores_front_1.bin df2scores_anch_1.bin df2scores_logits_1.bin df2scores_ph_1.bin df2scores_proj_1.bin; do
  if [ -f "$J/$p" ]; then echo "  在  $p"; else echo "  丢  $p"; fi
done
echo "--- 日志 ---"
for f in s3_scores.log s4_scores.log serial2.log serial_verdict.log ref_spec_metrics.txt fx_spec.log sel_probe.log; do
  if [ -f "$J/$f" ]; then echo "  在  $f ($(stat -c%s "$J/$f") 字节)"; else echo "  丢  $f"; fi
done
echo "--- 计数 ---"
echo "  dl 顶层文件数 = $(find "$J" -maxdepth 1 -type f | wc -l)"
echo "  df2scores scores 组数 = $(ls "$J"/df2scores_scores_*.bin 2>/dev/null | wc -l)"
echo "  参考 refcand npz 数 = $(ls "$J"/refcand_*.npz 2>/dev/null | wc -l)"
echo "  dl 下子目录数 = $(find "$J" -maxdepth 1 -type d | wc -l)"
