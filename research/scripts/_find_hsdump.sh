#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== hidden dump 开关（已知文件定点 grep） ==="
grep -n "NINFER_HS_DUMP_DIR\|NINFER_DSPARK_DUMP_DIR\|HS_DUMP\|hs_dump" \
  "$R/src/targets/qwen3_6/impl/runtime/program_impl.h" \
  "$R/src/targets/qwen3_6/impl/runtime/text_context_impl.h" \
  "$R/src/targets/qwen3_6/impl/runtime/text_prefill_impl.h" 2>/dev/null | head -20
echo
echo "=== GDN 状态动作 / replay 相关入口 ==="
grep -n "GdnStateAction\|RecordForReplay\|replay_records\|gdn_state" \
  "$R/src/targets/qwen3_6/impl/runtime/text_context.h" 2>/dev/null | head -20
echo
echo "=== gdn 相关 op 名（launcher 目录里的文件名） ==="
ls "$R/src/ops/launcher/" 2>/dev/null | grep -i gdn | head -10
