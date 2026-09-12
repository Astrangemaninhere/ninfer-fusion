#!/bin/bash
# G-A 门：plain / dflash2 / mtp 三条路径的贪心 token id 是否逐位一致
# 同时看 --spec auto 到底选了哪个后端（这决定用户默认体验到的是 44 还是 104 tok/s）
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/dl/ga_gate.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
cd "$R/build" || { echo "ERR no build dir"; exit 3; }
echo "=== G-A 门（贪心 token id 一致性）起 $(date '+%m-%d %H:%M:%S') ==="
run() {
  local label="$1"; shift
  timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
    --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy --seed 7 \
    "$@" > $J/dl/ga_$label.log 2>&1
  echo "  --- $label (rc=$?) ---"
  grep -m 1 -E 'mtp acceptance rate|dflash2 acceptance rate|dflash acceptance rate' $J/dl/ga_$label.log | sed 's/^/    /'
  grep -m 1 'decode speed' $J/dl/ga_$label.log | sed 's/^/    /'
  # token id 行：形如 "... generated ids ..."（不同版本前缀可能不同）
  grep -m 1 -i 'generated ids' $J/dl/ga_$label.log | head -c 200 | sed 's/^/    IDS: /'
  echo
}
run plain
run dflash2 --spec dflash2
run mtp     --spec mtp --draft-tokens 3 --lm-head-draft
run auto    --spec auto
echo
echo "=== token id 序列逐位比对 ==="
python3 - <<'PY'
import pathlib, re
D = "/mnt/c/Users/User/Documents/ziqinzhang/dl"
def ids(name):
    p = pathlib.Path(f"{D}/ga_{name}.log")
    for line in p.read_text(errors="replace").splitlines():
        low = line.lower()
        if "generated ids" in low:
            tail = line.split("generated ids", 1)[1]
            return [int(x) for x in re.findall(r"-?\d+", tail)]
    return None
base = ids("plain")
print(f"plain   n={len(base) if base else 'None'}")
for k in ("dflash2", "mtp", "auto"):
    v = ids(k)
    if v is None:
        print(f"{k:<7} 未打印 id 行"); continue
    same = (v == base)
    first = next((i for i, (a, b) in enumerate(zip(v, base)) if a != b), None)
    print(f"{k:<7} n={len(v)} 与 plain 一致={same}" + ("" if same else f"  首个不同位={first}"))
PY
echo "=== 完成 $(date '+%m-%d %H:%M:%S') ==="
echo GA_GATE_DONE
