#!/usr/bin/env bash
# =============================================================================
# _land_s6.sh  --  land (or revert) the S6 TU split into ninfer-fusion
#
#   gqa_attention_decode_e8.cu    224 -> 54 lines + 4 arm TUs + 1 arms header
#   gqa_attention_prefill.cu      484 -> 66 lines + 1 batch TU + 1 arms header
#   gqa_attention_prefill_e8.cu   236 -> 95 lines + 2 arm TUs + 1 arms header
#   src/CMakeLists.txt: +7 sources inside add_library(ninfer_ops STATIC)
#
# This script NEVER runs make / nvcc / cmake.  It copies files, edits
# src/CMakeLists.txt with python, and greps.  The compile is the operator's step;
# the exact commands (incl. the G-A/G-B/G-C gates) are printed at the end.
#
# usage:
#   bash _land_s6.sh                 # preflight -> land -> structured self-check
#   bash _land_s6.sh --dry-run       # preflight only + planned actions, no writes
#   bash _land_s6.sh --revert        # restore the 3 .cu, delete the 10 new files,
#                                    # drop the 7 CMake lines, print md5s
#   bash _land_s6.sh --revert --dry-run
#
# exit codes: 0 ok | 2 preflight refused (nothing written) | 3 landed, self-check
#             found failures | 4 revert refused/failed
# test hook (NOT for the real tree): LAND_TEST_ROOT=<dir> LAND_SNAPSHOT_DIR=<dir>
# =============================================================================
set -u

J=/mnt/c/Users/User/Documents/ziqinzhang
R=${LAND_TEST_ROOT:-/home/user/ninfer-fusion}
COLLAB=$J/_collab/build
STAGED=$COLLAB/staged2                     # S6 staging dir
BACKUP=$COLLAB/backup
LANDDIR=${LAND_SNAPSHOT_DIR:-$COLLAB/land_backup_s6}
CMAKE=$R/src/CMakeLists.txt

MODE=land
DRY=0
for a in "$@"; do
  case "$a" in
    --revert)   MODE=revert ;;
    --dry-run)  DRY=1 ;;
    -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $a (see --help)"; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# expected artifacts: staged_name | dest_dir | dest_name | lines | md5 | first_line
#   .split = new content of an existing .cu ; .new = brand new file
# ---------------------------------------------------------------------------
G=$R/src/ops/launcher
ARTIFACTS=(
"gqa_attention_decode_e8.cu.split|$G|gqa_attention_decode_e8.cu|54|89841779f54a77cfeb92b4201ff72da3|// E8-tier decode launch: the packed 4-bit (E8-lattice K / i4 V) instantiations"
"gqa_attention_decode_e8_arms.cuh.new|$G|gqa_attention_decode_e8_arms.cuh|206|5e2a0e7c34a26628cb3fad350917e23d|#pragma once"
"gqa_attention_decode_e8_append_g27.cu.new|$G|gqa_attention_decode_e8_append_g27.cu|23|04a9008e47cdc42ed202e984f1bcde14|// ninfer::ops::detail - E8 decode arm: Gqa27Geometry x append input (GqaAppendInput)."
"gqa_attention_decode_e8_cached_g27.cu.new|$G|gqa_attention_decode_e8_cached_g27.cu|23|608fa72d1be15755f88aa31a09a9513d|// ninfer::ops::detail - E8 decode arm: Gqa27Geometry x cached input (GqaCachedInput)."
"gqa_attention_decode_e8_append_muse35.cu.new|$G|gqa_attention_decode_e8_append_muse35.cu|28|237061077c211cb678d5b64da536cc9e|// ninfer::ops::detail - E8 decode arm: GqaMuseGeometry + Gqa35Geometry x append input (GqaAppendInput)."
"gqa_attention_decode_e8_cached_muse35.cu.new|$G|gqa_attention_decode_e8_cached_muse35.cu|28|26feb2387a6bce8f1ed2e44820e85fc0|// ninfer::ops::detail - E8 decode arm: GqaMuseGeometry + Gqa35Geometry x cached input (GqaCachedInput)."
"gqa_attention_prefill.cu.split|$G|gqa_attention_prefill.cu|66|80af772afb47299dc46863d5d4a759b8|// ninfer::ops - gqa_attention prompt-scale launcher: fill k/v at device"
"gqa_attention_prefill_arms.cuh.new|$G|gqa_attention_prefill_arms.cuh|401|13d5ebf7b700feee048bdfeb82290592|#pragma once"
"gqa_attention_prefill_batch.cu.new|$G|gqa_attention_prefill_batch.cu|48|45c06db47f60ca8158c8ab13e3101a94|// ninfer::ops::detail - prompt-scale prefill arm: the fused masked/batch entry"
"gqa_attention_prefill_e8.cu.split|$G|gqa_attention_prefill_e8.cu|95|51afcdfbedfad17d454dd8127efc669a|// E8-tier prefill launches: the packed 4-bit (E8-lattice K / i4 V)"
"gqa_attention_prefill_e8_arms.cuh.new|$G|gqa_attention_prefill_e8_arms.cuh|88|0f6abf1d724002a5f388d759fbb425c9|#pragma once"
"gqa_attention_prefill_e8_kv_append.cu.new|$G|gqa_attention_prefill_e8_kv_append.cu|41|487ab541b0c470d6396735fdf3f15171|// ninfer::ops::detail - E8 prefill arm: the masked/batch KV-append entry"
"gqa_attention_prefill_e8_kv_append_single.cu.new|$G|gqa_attention_prefill_e8_kv_append_single.cu|48|59d1b9997ff3e1627ef45167fddbd878|// ninfer::ops::detail - E8 prefill arm: the single-sequence KV-append entry"
)

# pre-split files this landing overwrites:  dest_path | md5 | lines
ORIGS=(
"$G/gqa_attention_decode_e8.cu|ad99e1117be549e2c6e4a5977339774a|234"
"$G/gqa_attention_prefill.cu|50e265e6205d9fec6d5c979a223fcbaa|484"
"$G/gqa_attention_prefill_e8.cu|eb52dd6a9f2665027f157cb4c1602ba1|236"
)

# CMake plan: block_decl|anchor_line|new_line[,new_line...]
CMAKE_PLAN=(
"add_library(ninfer_ops STATIC|ops/launcher/gqa_attention_decode_e8.cu|ops/launcher/gqa_attention_decode_e8_append_g27.cu,ops/launcher/gqa_attention_decode_e8_cached_g27.cu,ops/launcher/gqa_attention_decode_e8_append_muse35.cu,ops/launcher/gqa_attention_decode_e8_cached_muse35.cu"
"add_library(ninfer_ops STATIC|ops/launcher/gqa_attention_prefill.cu|ops/launcher/gqa_attention_prefill_batch.cu"
"add_library(ninfer_ops STATIC|ops/launcher/gqa_attention_prefill_e8.cu|ops/launcher/gqa_attention_prefill_e8_kv_append.cu,ops/launcher/gqa_attention_prefill_e8_kv_append_single.cu"
)
CMAKE_LINES_EXPECT=7
CMAKE_PRISTINE_MD5=19c515e91a28bb0c4c0eeef613b04b0f

hr()  { echo "-------------------------------------------------------------------"; }
h1()  { echo; hr; echo "== $*"; hr; }
md5f(){ md5sum "$1" | cut -d' ' -f1; }
nl()  { wc -l < "$1" | tr -d ' '; }
die() { echo; echo "ABORT($1): ${*:2}"; exit "$1"; }

# ===========================================================================
# 1. PREFLIGHT
# ===========================================================================
preflight() {
  h1 "PREFLIGHT (read-only)"
  local fails=0

  echo "[P1] staged artifacts in $STAGED"
  local f sf dir name lines md5 first p am bm al af
  for f in "${ARTIFACTS[@]}"; do
    IFS='|' read -r sf dir name lines md5 first <<<"$f"
    p=$STAGED/$sf
    if [ ! -f "$p" ]; then echo "  FAIL missing staged:$sf"; fails=$((fails+1)); continue; fi
    am=$(md5f "$p"); bm=$(nl "$p"); af=$(head -1 "$p"); al=$(grep -c $'\r' "$p" || true)
    if [ "$am" != "$md5" ]; then echo "  FAIL $sf md5=$am want=$md5"; fails=$((fails+1)); fi
    if [ "$bm" != "$lines" ]; then echo "  FAIL $sf lines=$bm want=$lines"; fails=$((fails+1)); fi
    if [ "$af" != "$first" ]; then
      echo "  FAIL $sf first line:"; echo "        got : $af"; echo "        want: $first"; fails=$((fails+1))
    fi
    if [ "$al" != "0" ]; then echo "  FAIL $sf has $al CR line(s), must be LF-only"; fails=$((fails+1)); fi
    if [ -n "$(tail -c 1 "$p")" ]; then echo "  FAIL $sf does not end with a newline"; fails=$((fails+1)); fi
    [ "$am" = "$md5" ] && echo "  ok   $sf  lines=$bm md5=$am"
  done

  # the three .cu on the tree must still be the pre-split originals (or already S6)
  echo "[P2] tree .cu still pristine (or already landed)"
  for f in "${ORIGS[@]}"; do
    IFS='|' read -r p md5 lines <<<"$f"
    if [ ! -f "$p" ]; then echo "  FAIL missing $p"; fails=$((fails+1)); continue; fi
    am=$(md5f "$p")
    if [ "$am" = "$md5" ]; then
      echo "  ok   ${p#$R/}  ($(nl "$p") lines, md5=$am)"
    else
      local landed=0 g
      for g in "${ARTIFACTS[@]}"; do
        IFS='|' read -r _ gdir gname _ gmd5 _ <<<"$g"
        [ "$gdir/$gname" = "$p" ] && [ "$am" = "$gmd5" ] && landed=1
      done
      if [ "$landed" = "1" ]; then
        echo "  note ${p#$R/} already holds the S6 version (md5=$am) -> landing applied before"
      else
        echo "  FAIL ${p#$R/} md5=$am != pre-split md5=$md5"; fails=$((fails+1))
      fi
    fi
  done

  echo "[P3] revert sources"
  local bk=0 c
  for c in "$BACKUP/gqa_attention_decode_e8.cu.orig" "$BACKUP/gqa_attention_prefill.cu.orig" \
           "$BACKUP/gqa_attention_prefill_e8.cu.orig"; do
    [ -f "$c" ] && echo "  ok   $c  md5=$(md5f "$c")"
  done

  echo "[P4] destinations"
  local kind orig_md5 g
  for f in "${ARTIFACTS[@]}"; do
    IFS='|' read -r sf dir name lines md5 first <<<"$f"
    p=$dir/$name
    case "$sf" in *.split) kind=overwrite ;; *) kind=new ;; esac
    if [ ! -f "$p" ]; then
      if [ "$kind" = "new" ]; then echo "  ok   ${p#$R/} absent (will be created)"
      else echo "  FAIL ${p#$R/} missing but this landing must overwrite it"; fails=$((fails+1)); fi
      continue
    fi
    am=$(md5f "$p")
    if [ "$am" = "$md5" ]; then
      echo "  ok   ${p#$R/} already holds the staged content (copy is a no-op)"
    elif [ "$kind" = "overwrite" ]; then
      orig_md5=""
      for g in "${ORIGS[@]}"; do IFS='|' read -r gp gm _ <<<"$g"; [ "$gp" = "$p" ] && orig_md5=$gm; done
      if [ "$am" = "$orig_md5" ]; then echo "  ok   ${p#$R/} is the pristine pre-split file (will be overwritten)"
      else echo "  FAIL ${p#$R/} md5=$am is neither the pre-split ($orig_md5) nor the staged ($md5) version"
           fails=$((fails+1)); fi
    else
      echo "  FAIL ${p#$R/} md5=$am != staged=$md5 (edited by somebody else)"; fails=$((fails+1))
    fi
  done

  echo "[P5] src/CMakeLists.txt"
  echo "  lines=$(nl "$CMAKE") md5=$(md5f "$CMAKE")"
  [ "$(md5f "$CMAKE")" = "$CMAKE_PRISTINE_MD5" ] && echo "  ok   identical to the pre-split backup" \
    || echo "  note md5 differs from the pre-split backup (parallel edit); anchors are re-verified below"
  python3 - "$CMAKE" "${CMAKE_PLAN[@]}" <<'PY' || fails=$((fails+1))
import sys
want = 0
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
bad = 0
for p in sys.argv[2:]:
    blk, anch, news = p.split('|')
    starts = [i for i, l in enumerate(lines) if l.strip().startswith(blk)]
    if len(starts) != 1:
        print(f"  FAIL block '{blk}' found {len(starts)}x (want 1)"); bad += 1; continue
    i = starts[0]; j = i
    while j < len(lines) and ')' not in lines[j]:
        j += 1
    inside = [k for k in range(i, j + 1) if lines[k].strip() == anch]
    if len(inside) != 1:
        print(f"  FAIL anchor '{anch}' inside '{blk}' found {len(inside)}x (want 1)"); bad += 1
    else:
        for n in news.split(','):
            want += 1
            if n in [lines[k].strip() for k in range(i, j + 1)]:
                print(f"  note '{n}' already in '{blk}' -> will be skipped")
        print(f"  ok   '{blk}' lines {i+1}-{j+1}; anchor '{anch}' at line {inside[0]+1}")
print(f"  new lines planned: {want} (want $CMAKE_LINES_EXPECT)")
sys.exit(1 if bad else 0)
PY

  hr
  if [ "$fails" != "0" ]; then
    echo "PREFLIGHT: $fails failure(s) -> NOTHING WRITTEN."
    echo "  tree md5 mismatch   = somebody else changed that TU; diff by hand and decide."
    echo "  staged md5 mismatch = re-stage, $STAGED is the review baseline."
    exit 2
  fi
  echo "PREFLIGHT: OK (staged md5/lines/first-line/LF, tree md5, CMake anchors)"
}

# ===========================================================================
# 2. LAND
# ===========================================================================
snapshot() {
  local snap="$LANDDIR/pre_land_$(date +%Y%m%d_%H%M%S)" f p
  if [ "$DRY" = "1" ]; then echo "  [dry-run] would snapshot the pre-land state to $snap"; return 0; fi
  mkdir -p "$snap" || die 4 "cannot create $snap"
  for f in "${ORIGS[@]}"; do
    IFS='|' read -r p _ _ <<<"$f"; cp -p "$p" "$snap/$(basename "$p")"
  done
  cp -p "$CMAKE" "$snap/CMakeLists.txt"
  local b
  for b in gqa_attention_decode_e8.cu gqa_attention_prefill.cu gqa_attention_prefill_e8.cu; do
    [ -f "$BACKUP/$b.orig" ] || cp -p "$snap/$b" "$BACKUP/$b.orig"
  done
  { echo "# pre-land snapshot $(date '+%F %T')  root=$R"
    for p in "$snap"/*; do printf '%s  %s\n' "$(md5f "$p")" "$(basename "$p")"; done
  } > "$snap/MANIFEST.txt"
  printf '%s\n' "$snap" > "$LANDDIR/LAST.txt"
  echo "  snapshot: $snap"
  sed 's/^/    /' "$snap/MANIFEST.txt"
}

do_land() {
  h1 "LAND"
  local f sf dir name lines md5 first p
  for f in "${ARTIFACTS[@]}"; do
    IFS='|' read -r sf dir name lines md5 first <<<"$f"
    p=$dir/$name
    if cmp -s "$STAGED/$sf" "$p"; then
      echo "  = ${p#$R/} already byte-identical (no write needed)"
    elif [ "$DRY" = "1" ]; then
      echo "  + [dry-run] cp $(basename "$STAGED/$sf") -> ${p#$R/}"
    else
      local existed=0; [ -f "$p" ] && existed=1
      cp "$STAGED/$sf" "$p" || die 4 "cp failed: $p"
      [ "$existed" = "1" ] || chmod 644 "$p" 2>/dev/null || true
      echo "  + ${p#$R/}  lines=$(nl "$p") md5=$(md5f "$p") mode=$(stat -c %a "$p")"
    fi
  done
  echo
  echo "[CMAKE] $CMAKE"
  if [ "$DRY" = "1" ]; then
    python3 - "$CMAKE" plan "${CMAKE_PLAN[@]}" <<'PY'
import sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
strips = [l.strip() for l in lines]
for p in sys.argv[3:]:
    blk, anch, news = p.split('|')
    i = [k for k, l in enumerate(lines) if l.strip().startswith(blk)][0]
    j = i
    while ')' not in lines[j]:
        j += 1
    a = [k for k in range(i, j + 1) if lines[k].strip() == anch][0]
    off = 0
    for n in news.split(','):
        if n in strips:
            print(f"  = {n} present, skip")
            continue
        print(f"  + [dry-run] insert after line {a+1+off}: {n}")
        off += 1
PY
  else
    python3 - "$CMAKE" insert "${CMAKE_PLAN[@]}" <<'PY' || die 4 "CMake insert failed"
import hashlib, sys
path, mode = sys.argv[1], sys.argv[2]
raw = open(path, 'rb').read()
if b'\r' in raw:
    sys.exit("CMakeLists contains CR bytes; refusing to rewrite")
lines = raw.decode('utf-8').split('\n')
strips = [l.strip() for l in lines]
for p in sys.argv[3:]:
    blk, anch, news = p.split('|')
    starts = [k for k, l in enumerate(lines) if l.strip().startswith(blk)]
    if len(starts) != 1:
        sys.exit(f"block '{blk}' found {len(starts)}x")
    i = starts[0]
    j = i
    while ')' not in lines[j]:
        j += 1
    a = [k for k in range(i, j + 1) if lines[k].strip() == anch]
    if len(a) != 1:
        sys.exit(f"anchor '{anch}' in '{blk}' found {len(a)}x")
    a = a[0]
    want = news.split(',')
    ins = [n for n in want if n not in strips]
    if not ins:
        print(f"  = {blk}: all {len(want)} entries already present, skip")
        continue
    for off, n in enumerate(ins):
        lines.insert(a + 1 + off, '  ' + n)
        strips.insert(a + 1 + off, n)
    print(f"  + {blk}: inserted {len(ins)}/{len(want)} line(s) after line {a+1}")
open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
print(f"  CMakeLists now {len(lines)-1} lines, md5={hashlib.md5(open(path,'rb').read()).hexdigest()}")
PY
  fi
}

# ===========================================================================
# 3. STRUCTURED SELF-CHECK (python/grep only; no compiler)
# ===========================================================================
selfcheck() {
  h1 "SELF-CHECK"
  python3 - "$R" <<'PY'
import hashlib, os, re, sys
R     = sys.argv[1]
SRC   = os.path.join(R, 'src')
CMAKE = os.path.join(SRC, 'CMakeLists.txt')
fails = []
_c = {}
def read(p):
    if p not in _c:
        _c[p] = open(p, encoding='utf-8', errors='replace').read().split('\n')
    return _c[p]
def md5(p): return hashlib.md5(open(p, 'rb').read()).hexdigest()
def rel(p): return os.path.relpath(p, SRC).replace(os.sep, '/')

S6 = ['ops/launcher/gqa_attention_decode_e8.cu',
      'ops/launcher/gqa_attention_decode_e8_arms.cuh',
      'ops/launcher/gqa_attention_decode_e8_append_g27.cu',
      'ops/launcher/gqa_attention_decode_e8_cached_g27.cu',
      'ops/launcher/gqa_attention_decode_e8_append_muse35.cu',
      'ops/launcher/gqa_attention_decode_e8_cached_muse35.cu',
      'ops/launcher/gqa_attention_prefill.cu',
      'ops/launcher/gqa_attention_prefill_arms.cuh',
      'ops/launcher/gqa_attention_prefill_batch.cu',
      'ops/launcher/gqa_attention_prefill_e8.cu',
      'ops/launcher/gqa_attention_prefill_e8_arms.cuh',
      'ops/launcher/gqa_attention_prefill_e8_kv_append.cu',
      'ops/launcher/gqa_attention_prefill_e8_kv_append_single.cu']
NEWSRC = [s for s in S6 if s.endswith('.cu') and s not in
          ('ops/launcher/gqa_attention_decode_e8.cu', 'ops/launcher/gqa_attention_prefill.cu',
           'ops/launcher/gqa_attention_prefill_e8.cu')]

print("[SC1] hygiene: LF-only, newline-terminated, brace balance, ns envelope")
for r_ in S6:
    p = os.path.join(SRC, r_)
    if not os.path.isfile(p):
        print(f"       FAIL missing {r_}"); fails.append(r_); continue
    b = open(p, 'rb').read()
    t = b.decode('utf-8')
    ls = t.split('\n')[:-1]
    prob = []
    if b'\r' in b: prob.append('CR bytes')
    if not b.endswith(b'\n'): prob.append('no final newline')
    if t.count('{') != t.count('}'): prob.append(f"brace imbalance {t.count('{')}/{t.count('}')}")
    if sum(1 for l in ls if l.strip() == 'namespace ninfer::ops::detail {') != 1: prob.append('ns open')
    if sum(1 for l in ls if l.strip() == '} // namespace ninfer::ops::detail') != 1: prob.append('ns close')
    if prob:
        print(f"       FAIL {r_}: {', '.join(prob)}"); fails.append(r_)
    else:
        print(f"       ok   {r_:52s} {len(ls):4d} lines")

print("[SC2] include wiring (dispatchers/arms -> their arms header, exactly once)")
INC = [(S6[0], S6[1]), (S6[2], S6[1]), (S6[3], S6[1]), (S6[4], S6[1]), (S6[5], S6[1]),
       (S6[6], S6[7]), (S6[8], S6[7]), (S6[9], S6[10]), (S6[11], S6[10]), (S6[12], S6[10])]
bad = 0
for src, hdr in INC:
    if not os.path.isfile(os.path.join(SRC, src)):
        print(f"       FAIL {src}: missing (dry-run before landing?)"); bad += 1; continue
    n = '\n'.join(read(os.path.join(SRC, src))).count(f'#include "{hdr}"')
    if n != 1:
        print(f"       FAIL {src}: {n}x #{hdr} (want 1)"); fails.append(src); bad += 1
if not bad:
    print(f"       ok   {len(INC)} include edges (each arm/dispatcher includes its arms header once)")

print("[SC3] CMake: each new .cu exactly once inside add_library(ninfer_ops STATIC)")
raw = open(CMAKE, 'rb').read()
if b'\r' in raw:
    print("       FAIL src/CMakeLists.txt contains CR bytes"); fails.append('cmake CR')
cl = raw.decode('utf-8').split('\n')
strips = [l.strip() for l in cl]
ranges = {}
for i, l in enumerate(cl):
    m = re.match(r'\s*add_library\((\w+)', l)
    if m:
        j = i
        while ')' not in cl[j]:
            j += 1
        ranges[m.group(1)] = (i, j)
lo, hi = ranges['ninfer_ops']
for s in NEWSRC:
    n = strips.count(s)
    ok = (n == 1 and lo <= strips.index(s) <= hi)
    print(f"       {'ok  ' if ok else 'FAIL'} {s:52s} x{n}")
    if not ok: fails.append(s)
for h in (S6[1], S6[7], S6[10]):
    if h in strips:
        print(f"       FAIL header {h} listed as a source (headers need no entry)"); fails.append(h)
print(f"       ok   headers not listed; CMakeLists is {len(cl)-1} lines, md5={hashlib.md5(raw).hexdigest()}")

print("[SC4] hygiene: no staging leftovers under src/")
leak = []
for root, dirs, files in os.walk(SRC):
    dirs[:] = [d for d in dirs if d not in ('build', '.git')]
    for f in files:
        if f.endswith(('.new', '.split', '.orig', '.bak', '.rej')):
            leak.append(os.path.join(root, f))
if leak:
    print(f"       FAIL staging leftovers inside src/: {leak}"); fails.append('stray staging files')
else:
    print("       ok   no .new/.split/.orig/.bak/.rej under src/")

print()
if fails:
    print(f"SELF-CHECK: {len(fails)} FAILURE(S)")
    for f in fails:
        print("  - " + f)
    sys.exit(3)
print("SELF-CHECK: OK (0 failures)")
PY
  local rc=$?
  if [ "$rc" != "0" ]; then
    echo
    echo "!! self-check failed (rc=$rc): the tree holds a PARTIAL landing."
    echo "!! inspect the tree, then either fix by hand or run: bash $0 --revert"
    return 3
  fi
  h1 "NEXT STEPS (NOT executed by this script - run them in the compile window)"
  cat <<EOF
  1) reconfigure so the 7 new sources enter the build graph:
       cmake -S $R -B $R/build

  2) memory-aware parallel build (it sizes -j from MemAvailable itself):
       bash $J/_par_build.sh
     log: $J/dl/par_build.log

  3) new objects to expect (13 TUs replace 3):
       $R/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode_e8.cu.o            <- tiny (dispatcher)
       $R/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode_e8_{append,cached}_{g27,muse35}.cu.o
       $R/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_prefill.cu.o             <- direct pair
       $R/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_prefill_batch.cu.o
       $R/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_prefill_e8.cu.o
       $R/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_prefill_e8_kv_append{,_single}.cu.o
     pre-split sizes for comparison: decode_e8.o 180.7 MB / prefill.o 17.7 MB / prefill_e8.o 6.7 MB
     (the .o sizes must add up to ~the same total; the *per-arm* max is what matters)

  4) gates (all three must pass, else: on the same branch? always  bash $J/_land_s6.sh --revert)
       G-A  bash $J/_post_fix_verify2.sh          # greedy ids: mtp row must be IDENTICAL, empty output = no evidence
       G-B  3x fixed-prompt tok/s vs the pre-split binary (>= 0.98x), accept rate must not drop
       G-C  each backend starts and emits text: bf16 / nvfp4 / i8 / iso3 x (mtp/dflash/dflash2) x (27b/35b/Muse)

  rollback at any time:  bash $J/_land_s6.sh --revert
                         (restores the 3 .cu from $LANDDIR/LAST.txt or $BACKUP/*.orig,
                          deletes the 10 new files, drops the 7 CMake lines)
EOF
}

# ===========================================================================
# 4. REVERT
# ===========================================================================
do_revert() {
  h1 "REVERT"
  local fails=0 f sf dir name lines md5 first p am src snap="" c
  [ -f "$LANDDIR/LAST.txt" ] && snap=$(cat "$LANDDIR/LAST.txt")

  echo "[R1] restore the three pre-split .cu"
  for f in "${ORIGS[@]}"; do
    IFS='|' read -r p md5 lines <<<"$f"
    src=""
    for c in "$snap/$(basename "$p")" "$BACKUP/$(basename "$p").orig"; do
      [ -n "$c" ] && [ -f "$c" ] && [ "$(md5f "$c")" = "$md5" ] && { src=$c; break; }
    done
    if [ -z "$src" ]; then
      echo "  FAIL no verified backup for ${p#$R/} (want md5 $md5)"; fails=$((fails+1)); continue
    fi
    if [ "$DRY" = "1" ]; then
      echo "  [dry-run] cp $src -> ${p#$R/}   (restores md5 $md5)"
    else
      cp -p "$src" "$p"
      echo "  <- ${p#$R/} restored from $src  md5=$(md5f "$p")  lines=$(nl "$p")"
    fi
  done

  echo "[R2] delete the files this landing created"
  local isorig g
  for f in "${ARTIFACTS[@]}"; do
    IFS='|' read -r sf dir name lines md5 first <<<"$f"
    p=$dir/$name; isorig=0
    for g in "${ORIGS[@]}"; do IFS='|' read -r gp _ _ <<<"$g"; [ "$gp" = "$p" ] && isorig=1; done
    [ "$isorig" = "1" ] && continue
    if [ ! -f "$p" ]; then echo "  = ${p#$R/} already absent"; continue; fi
    am=$(md5f "$p")
    if [ "$am" != "$md5" ]; then
      echo "  WARN ${p#$R/} md5=$am != staged=$md5 -> NOT deleted (not created by this landing)"
      fails=$((fails+1)); continue
    fi
    if [ "$DRY" = "1" ]; then echo "  [dry-run] rm ${p#$R/}"
    else rm -f "$p"; echo "  - ${p#$R/} removed"; fi
  done

  echo "[R3] src/CMakeLists.txt"
  if [ "$DRY" = "1" ]; then
    python3 - "$CMAKE" "${CMAKE_PLAN[@]}" <<'PY'
import sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
for p in sys.argv[2:]:
    for n in p.split('|')[2].split(','):
        print(f"  [dry-run] remove '{n}' ({sum(1 for l in lines if l.strip() == n)}x)")
PY
  else
    python3 - "$CMAKE" remove "${CMAKE_PLAN[@]}" <<'PY' || { echo "  FAIL cmake remove"; fails=$((fails+1)); }
import hashlib, re, sys
path = sys.argv[1]
raw = open(path, 'rb').read()
if b'\r' in raw:
    sys.exit("CMakeLists contains CR bytes; refusing to rewrite")
lines = raw.decode('utf-8').split('\n')
ranges = {}
for i, l in enumerate(lines):
    m = re.match(r'\s*add_library\((\w+)', l)
    if m:
        j = i
        while ')' not in lines[j]:
            j += 1
        ranges[m.group(1)] = (i, j)
removed = 0
for p in sys.argv[3:]:
    blk, anch, news = p.split('|')
    name = re.match(r'add_library\((\w+)', blk).group(1)
    if name not in ranges:
        sys.exit(f"block '{blk}' not found")
    lo, hi = ranges[name]
    for n in news.split(','):
        hit, keep = 0, []
        for i, l in enumerate(lines):
            if l.strip() == n:
                hit += 1
                if not (lo <= i <= hi):
                    sys.exit(f"'{n}' at line {i+1} is outside '{blk}' - refusing to delete")
                continue
            keep.append(l)
        print(f"  - {blk}: removed {hit}x '{n}'" if hit else f"  = {blk}: '{n}' absent")
        removed += hit
        lines = keep
open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
print(f"  CMakeLists now {len(lines)-1} lines, md5={hashlib.md5(open(path,'rb').read()).hexdigest()} (removed {removed})")
PY
  fi

  echo
  echo "[R4] md5 cross-check (expected value in brackets)"
  for f in "${ORIGS[@]}"; do
    IFS='|' read -r p md5 lines <<<"$f"
    printf '  %-52s %s  [%s]\n' "${p#$R/}" "$(md5f "$p")" "$md5"
  done
  printf '  %-52s %s  [%s]\n' "src/CMakeLists.txt" "$(md5f "$CMAKE")" "$CMAKE_PRISTINE_MD5"
  for f in "${ARTIFACTS[@]}"; do
    IFS='|' read -r sf dir name _ _ _ <<<"$f"
    p=$dir/$name; isorig=0
    for g in "${ORIGS[@]}"; do IFS='|' read -r gp _ _ <<<"$g"; [ "$gp" = "$p" ] && isorig=1; done
    [ "$isorig" = "1" ] && continue
    if [ -f "$p" ]; then
      if [ "$DRY" = "1" ]; then echo "  [dry-run] still present (would be removed): ${p#$R/}"
      else echo "  STILL PRESENT: ${p#$R/}"; fails=$((fails+1)); fi
    fi
  done
  hr
  if [ "$fails" != "0" ]; then echo "REVERT: incomplete ($fails issue(s)) - see above"; return 4; fi
  echo "REVERT: OK - tree is back to the pre-S6 state (re-run cmake before building)"
}

# ===========================================================================
# main
# ===========================================================================
echo "=== _land_s6.sh  mode=$MODE dry-run=$DRY  $(date '+%F %T') ==="
echo "    tree=$R   staged=$STAGED"
preflight
if [ "$MODE" = "revert" ]; then
  do_revert || exit $?
  echo "=== done $(date '+%F %T') ==="
  exit 0
fi
snapshot
do_land
if [ "$DRY" = "1" ]; then
  echo
  echo "[dry-run] the self-check below runs against the CURRENT tree: FAILs are expected"
  echo "          until the landing is actually applied (nothing was written)."
  selfcheck
  echo "=== dry-run done $(date '+%F %T') ==="
  exit 0
fi
selfcheck || exit $?
echo "=== done $(date '+%F %T') ==="
