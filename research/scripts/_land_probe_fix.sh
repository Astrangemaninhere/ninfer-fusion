#!/bin/bash
# 落地：① 分析层修复（A5 附注 + 权威读法脚本）② 加 pending 探针 ③ 编译 ④ 跑 ⑤ 验证
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
LOG=$J/dl/land_probe_fix.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 落地分析层修复 + pending 探针 $(date '+%F %H:%M:%S') ==="

echo "--- 1) 分析层修复 ---"
"C:/Program Files/Python312/python.exe" "$J/_fix_analysis_layer.py" 2>/dev/null || python3 "$J/_fix_analysis_layer.py"

echo "--- 2) 加 pending 探针（C 的补丁） ---"
python3 - <<'PYEOF'
import pathlib, hashlib, shutil
F = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/dflash2_impl.h")
BAK = pathlib.Path("/home/user/df2feat_bak2"); BAK.mkdir(exist_ok=True)
shutil.copy2(F, BAK / F.name)
t = F.read_text()
anchor = """            dump_one("features", fv);
            dump_one("projected", context_roots.projected);
            dump_one("context", context_full);
"""
add = """            dump_one("features", fv);
            dump_one("projected", context_roots.projected);
            dump_one("context", context_full);
            // Sink destination: every tapped layer writes here, so this separates "the sink
            // never wrote slot k" from "prepare_ragged_prefix zero-filled the tail columns".
            dump_one("pending", dflash2_state(state).pending_features);
            std::fprintf(stderr, "[df2feat] pending ne={%d,%d,%d} cols=%d\\n",
                         dflash2_state(state).pending_features.ne[0],
                         dflash2_state(state).pending_features.ne[1],
                         dflash2_state(state).pending_features.ne[2], columns);
"""
assert t.count(anchor) == 1, "anchor %d" % t.count(anchor)
F.write_text(t.replace(anchor, add))
print("pending 探针已加, md5", hashlib.md5(F.read_bytes()).hexdigest()[:12])
PYEOF

echo "--- 3) 编译 ---"
touch $R/src/targets/qwen3_6_27b/impl/variant.cpp $R/src/targets/muse_glimmer_30b/impl/variant.cpp
cd "$R/build" || exit 4
make ninfer -j3 2>&1 | tail -3
rc=${PIPESTATUS[0]}; echo "make rc=$rc"
[ "$rc" -ne 0 ] && { echo BUILD_FAIL; exit 5; }

echo "--- 4) 跑（--no-cuda-graph） ---"
rm -f $J/dl/feat_*_*.bin
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
NINFER_DF2FEAT=1 timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt "$P" --max-new 16 --max-context 4096 --no-thinking --greedy --spec dflash2 \
  --no-cuda-graph > $J/dl/probe_fix_stdout.log 2>&1
echo "  rc=$?"
grep -E 'df2feat' $J/dl/probe_fix_stdout.log | head -12

echo "--- 5) 验证（权威读法） ---"
/home/user/vllm029/bin/python "$J/_feat_canonical_read.py" \
  $J/dl/feat_pending_5.bin $J/dl/feat_features_5.bin 2>&1 | head -14
echo LAND_PROBE_FIX_DONE
