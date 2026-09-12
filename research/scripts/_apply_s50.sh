#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
cd $R || exit 3
F="$J/_collab/E7_s50_kv_coverage.diff"
echo "=== apply S50 (KV coverage lower bound) ==="
patch -p1 < "$F" > /tmp/s50_ap.log 2>&1 && echo "  APPLIED" || { echo "  APPLY FAIL"; tail -6 /tmp/s50_ap.log; exit 1; }
grep -cE '^patching' /tmp/s50_ap.log | sed 's/^/  files patched: /'
echo
echo "=== post-apply state ==="
sed -n '1494,1506p' "$R/src/targets/qwen3_6/impl/runtime/logical_kv_store.h" | cat -n | sed 's/^/   /'
echo "  --- identifiers left (should be 0 for the old names) ---"
grep -c 'materialize_to_tokens' "$R/src/targets/qwen3_6/impl/runtime/logical_kv_store.h" || true
grep -c 'materialize_sequence_kv' "$R/src/targets/qwen3_6/impl/runtime/program_impl.h" || true
echo "  --- new names ---"
grep -c 'ensure_mapped_to_tokens' "$R/src/targets/qwen3_6/impl/runtime/logical_kv_store.h"
grep -c 'ensure_sequence_kv_mapped' "$R/src/targets/qwen3_6/impl/runtime/program_impl.h"
echo
echo "=== force the TUs that include these headers (no header deps in this build) ==="
for f in src/targets/qwen3_6_27b/impl/variant.cpp src/targets/muse_glimmer_30b/impl/variant.cpp \
         src/targets/qwen3_6_35b_a3b/impl/variant.cpp; do
  [ -f "$f" ] && touch "$f" && echo "  touched $f"
done
echo
echo "=== record + build state ==="
python3 - <<'PY'
import datetime, pathlib
T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
T.open("a", encoding="utf-8").write("""
### 9. 落地 S50：KV 覆盖改为下界语义（借用上游 03177b9，%s）
- 我方实测：`logical_kv_store.h:1498` 仍是 `if (target < page_count || target > entitlement) throw`
  —— 即"要求覆盖变小"会抛错。上游把它改成下界语义（已覆盖就早返回、不截断；错误信息带 tokens/pages/entitlement）。
- E7 的**可达性分析**（我已复核其引用的行）：唯一可证明可达的收缩点在 DFlash 终止结算——
  verify 已按 `frontier+extent+1` 映射（`program_impl.h:12180`），而 dspark 的 terminal
  `enqueue_dflash_context_append` 只要求 `max(text_kv_valid, end)`（`:11416`，end=base_E+accepted）。
  当"截断的接受 + 跨 64 token 页"同时发生（例：base_E=60/extent=8/accepted=2 ⇒ 1<2 页）**今天会抛错**，
  修补后变成无害早返回。⇒ 这是 dspark 侧一个真实可触发的失败模式（不解释 10.3% 接受率，但会真炸）。
- 已落地：4 文件 / 26 hunk / +46-38（含 14 处 `program_impl.h` 调用点改名，参数逐字节不变、由 E7 用
  `diff -u -w -B` 归一化标识符后证明，我也抽验了引用行）。同时已 touch 3 个 variant TU 强制重编。
- 未落地：E7 的 `E7_s50_regression_sketch.diff`（+36 行的 store 级页边界回归测试）留到能编 tests 的窗口。
""" % datetime.datetime.now().strftime("%H:%M"))
print("  _TODO.md updated")
PY
grep -oE '^\[[ 0-9]+%\][^"]*' /tmp/pa_make_1.log | tail -1
for p in $(pgrep -f bin/nvcc); do echo "  nvcc $(ps -o etime= -p $p | tr -d ' ')"; break; done
