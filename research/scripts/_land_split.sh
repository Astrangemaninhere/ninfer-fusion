#!/usr/bin/env bash
# =============================================================================
# _land_split.sh  --  land (or revert) the two TU splits into ninfer-fusion
#
#   gqa:   src/ops/launcher/gqa_attention_decode.cu            737 -> 45 lines
#          + new src/ops/launcher/gqa_attention_decode_partial.cuh     (461)
#          + new src/ops/launcher/gqa_attention_decode_smallt.cu      (279)
#          + src/CMakeLists.txt: +1 source in add_library(ninfer_ops ...)
#   nvfp4: src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu              176 -> 77 lines
#          + new .../nvfp4_w4a4_tma_arms.cuh                    (156)
#          + new .../nvfp4_w4a4_tma_{attn,gdn,mlp,residual}.cu  (32/31/22/63)
#          + src/CMakeLists.txt: +4 sources in add_library(ninfer_nvfp4_tma ...)
#
# This script NEVER runs make / nvcc / cmake. It copies files, edits
# src/CMakeLists.txt with python, and greps. The compile is the operator's
# step; the exact commands are printed at the end (not executed).
#
# usage:
#   bash _land_split.sh                 # preflight -> land -> structured self-check
#   bash _land_split.sh --dry-run       # preflight only + planned actions, no writes
#   bash _land_split.sh --revert        # restore the 2 .cu from backup, delete the new
#                                       # files, drop the CMake lines, print md5s
#   bash _land_split.sh --revert --dry-run
#
# exit codes: 0 ok | 2 preflight refused (nothing written) | 3 landed, self-check
#             found failures | 4 revert refused/failed
# test hook (NOT for the real tree): LAND_TEST_ROOT=<dir> LAND_SNAPSHOT_DIR=<dir>
#   redirects the whole run to a copy of the tree, used by the dry self-test.
# =============================================================================
set -u

J=/mnt/c/Users/User/Documents/ziqinzhang
R=${LAND_TEST_ROOT:-/home/user/ninfer-fusion}
COLLAB=$J/_collab/build
STAGED=$COLLAB/staged
BACKUP=$COLLAB/backup
LANDDIR=${LAND_SNAPSHOT_DIR:-$COLLAB/land_backup}   # pre-land snapshots written here
CMAKE=$R/src/CMakeLists.txt

MODE=land
DRY=0
for a in "$@"; do
  case "$a" in
    --revert)   MODE=revert ;;
    --dry-run)  DRY=1 ;;
    -h|--help)  sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown arg: $a (see --help)"; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# expected artifacts
#   staged_name | dest_dir | dest_name | lines | md5 | first_line
#   .split = new content of an existing .cu ; .new = brand new file
# ---------------------------------------------------------------------------
G=$R/src/ops/launcher
N=$R/src/ops/linear/nvfp4
ARTIFACTS=(
"gqa_attention_decode.cu.split|$G|gqa_attention_decode.cu|45|ccca2fd8f972d965b4912e344215d7bc|// ninfer::ops - split-KV GQA small-T launcher and unified route dispatcher."
"gqa_attention_decode_partial.cuh.new|$G|gqa_attention_decode_partial.cuh|461|9c4a3ae3e71887ea2752f017135c6b7c|#pragma once"
"gqa_attention_decode_smallt.cu.new|$G|gqa_attention_decode_smallt.cu|279|f709636e0b4720b7ad05e73c8ed0e09c|// ninfer::ops::detail - small-T GQA launch path (kernel instantiations)."
"nvfp4_w4a4_tma.cu.split|$N|nvfp4_w4a4_tma.cu|77|f86c0ec8c5d670ed16dfe5d89d8dfa56|#include \"ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h\""
"nvfp4_w4a4_tma_arms.cuh.new|$N|nvfp4_w4a4_tma_arms.cuh|156|46a040c7beba0929df7f3c23473d3ead|#pragma once"
"nvfp4_w4a4_tma_attn.cu.new|$N|nvfp4_w4a4_tma_attn.cu|32|52ecc293e7bb34ceac08b6891f63e11e|#include \"ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh\""
"nvfp4_w4a4_tma_gdn.cu.new|$N|nvfp4_w4a4_tma_gdn.cu|31|e18dfaf93a354a6b00b07b630768b006|#include \"ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh\""
"nvfp4_w4a4_tma_mlp.cu.new|$N|nvfp4_w4a4_tma_mlp.cu|22|db863df54fef3a8b73258aeffa670309|#include \"ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh\""
"nvfp4_w4a4_tma_residual.cu.new|$N|nvfp4_w4a4_tma_residual.cu|63|371e735d148076186632062880c08a2f|#include \"ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh\""
)

# pre-split files this landing overwrites:  dest_path | md5 | lines
ORIGS=(
"$G/gqa_attention_decode.cu|ae3d1c729156d749a3f31631e5279467|777"
"$N/nvfp4_w4a4_tma.cu|12714635ae9ecb70b223fba5e619b2fa|176"
)

# CMake plan: block_decl|anchor_line|new_line[,new_line...]
CMAKE_PLAN=(
"add_library(ninfer_ops STATIC|ops/launcher/gqa_attention_decode.cu|ops/launcher/gqa_attention_decode_smallt.cu"
"add_library(ninfer_nvfp4_tma STATIC|ops/linear/nvfp4/nvfp4_w4a4_tma.cu|ops/linear/nvfp4/nvfp4_w4a4_tma_attn.cu,ops/linear/nvfp4/nvfp4_w4a4_tma_gdn.cu,ops/linear/nvfp4/nvfp4_w4a4_tma_mlp.cu,ops/linear/nvfp4/nvfp4_w4a4_tma_residual.cu"
)
CMAKE_LINES_EXPECT=5
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

  # -- 1a. staged artifacts: exist / md5 / lines / first line / LF-only -----
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

  # -- 1b. the two .cu on the tree must still be the pre-split originals ----
  #     Rationale: both files are fully overwritten. If an md5 differs from the
  #     recorded pre-split md5 then somebody else (another agent, or a half
  #     applied earlier landing) already edited that TU, and a blind copy would
  #     silently destroy their work -> refuse and let a human diff it.
  echo "[P2] tree .cu still pristine"
  for f in "${ORIGS[@]}"; do
    IFS='|' read -r p md5 lines <<<"$f"
    if [ ! -f "$p" ]; then echo "  FAIL missing $p"; fails=$((fails+1)); continue; fi
    am=$(md5f "$p")
    if [ "$am" = "$md5" ]; then
      echo "  ok   $p  ($(nl "$p") lines, md5=$am)"
    else
      local landed=0 g
      for g in "${ARTIFACTS[@]}"; do
        IFS='|' read -r _ gdir gname _ gmd5 _ <<<"$g"
        [ "$gdir/$gname" = "$p" ] && [ "$am" = "$gmd5" ] && landed=1
      done
      if [ "$landed" = "1" ]; then
        echo "  note $p already holds the SPLIT version (md5=$am) -> landing applied before"
      else
        echo "  FAIL $p md5=$am != pre-split md5=$md5"; fails=$((fails+1))
      fi
    fi
  done

  # -- 1c. verified backups, so --revert always has a source -----------------
  echo "[P3] revert sources"
  if [ -f "$BACKUP/nvfp4_w4a4_tma.cu.orig" ] &&
     [ "$(md5f "$BACKUP/nvfp4_w4a4_tma.cu.orig")" = "12714635ae9ecb70b223fba5e619b2fa" ]; then
    echo "  ok   $BACKUP/nvfp4_w4a4_tma.cu.orig"
  else
    echo "  WARN $BACKUP/nvfp4_w4a4_tma.cu.orig missing/mismatched (land snapshot will cover it)"
  fi
  local bk=0 c
  for c in "$BACKUP/gqa_attention_decode.cu.orig" /tmp/orig.cu "$COLLAB/tmp/orig_gqa_attention_decode.cu"; do
    if [ -f "$c" ] && [ "$(md5f "$c")" = "b0a130e21e02e42a0d1f9834dfac8aae" ]; then echo "  ok   $c"; bk=1; fi
  done
  [ "$bk" = "0" ] && echo "  WARN no gqa original found outside the tree (pre-land snapshot still created now)"

  # -- 1d. destinations: brand-new files absent or identical to staged;
  #        overwrite targets (*.split) pristine (1b) or already landed -------
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

  # -- 1e. CMake: anchors present once, new entries not yet listed ----------
  echo "[P5] src/CMakeLists.txt"
  echo "  lines=$(nl "$CMAKE") md5=$(md5f "$CMAKE")"
  [ "$(md5f "$CMAKE")" = "$CMAKE_PRISTINE_MD5" ] && echo "  ok   identical to the pre-split backup" \
    || echo "  note md5 differs from the pre-split backup (parallel edit); anchors are re-verified below"
  local n_new
  n_new=$(python3 - "$CMAKE" "${CMAKE_PLAN[@]}" <<'PY'
import sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
strips = [l.strip() for l in lines]
tgt = []
for p in sys.argv[2:]:
    tgt += p.split('|')[2].split(',')
print(sum(1 for t in tgt if t in strips))
PY
) || fails=$((fails+1))
  echo "  new source entries already present: $n_new / $CMAKE_LINES_EXPECT  (0 = clean first landing)"
  python3 - "$CMAKE" "${CMAKE_PLAN[@]}" <<'PY' || fails=$((fails+1))
import re, sys
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
            if n in [lines[k].strip() for k in range(i, j + 1)]:
                print(f"  note '{n}' already in '{blk}' -> will be skipped")
        print(f"  ok   '{blk}' lines {i+1}-{j+1}; anchor '{anch}' at line {inside[0]+1}")
sys.exit(1 if bad else 0)
PY

  hr
  if [ "$fails" != "0" ]; then
    echo "PREFLIGHT: $fails failure(s) -> NOTHING WRITTEN."
    echo "  tree md5 mismatch   = somebody else changed that TU; diff by hand and decide."
    echo "  staged md5 mismatch = re-stage, $STAGED is the review baseline."
    exit 2
  fi
  echo "PREFLIGHT: OK (staged md5/lines/first-line, tree md5, backups, CMake anchors)"
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
  [ -f "$BACKUP/gqa_attention_decode.cu.orig" ] ||
    cp -p "$snap/gqa_attention_decode.cu" "$BACKUP/gqa_attention_decode.cu.orig"
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
      # created files get 644 (same as the other split artifacts); files that
      # already existed keep whatever mode they had
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
# 3. STRUCTURED SELF-CHECK  (greps + python only; no compiler)
# ===========================================================================
selfcheck() {
  h1 "SELF-CHECK"
  python3 - "$R" <<'PY'
import hashlib, os, re, sys
R     = sys.argv[1]
SRC   = os.path.join(R, 'src')
CMAKE = os.path.join(SRC, 'CMakeLists.txt')
fails, infos = [], []
_c = {}
def read(p):
    if p not in _c:
        _c[p] = open(p, encoding='utf-8', errors='replace').read().split('\n')
    return _c[p]
def md5(p): return hashlib.md5(open(p, 'rb').read()).hexdigest()
def rel(p): return os.path.relpath(p, SRC).replace(os.sep, '/')

# --- the set of files the build actually sees: compiled .cu + include closure
lines = read(CMAKE)
blocks = []
for i, l in enumerate(lines):
    m = re.match(r'\s*add_library\((\w+)', l)
    if m:
        j = i
        while j < len(lines) and ')' not in lines[j]:
            j += 1
        srcs = []
        for k in range(i, min(j + 1, len(lines))):
            t = lines[k].strip().rstrip(')').strip()
            if re.search(r'\.(cu|cpp|c)$', t) and not t.startswith('${'):
                srcs.append(t)
        blocks.append((m.group(1), i + 1, j + 1, srcs))
compiled = sorted({s for _, _, _, ss in blocks for s in ss})
print(f"[SC1] add_library blocks {len(blocks)}, compiled sources {len(compiled)}")
for n, i, j, ss in blocks:
    if n in ('ninfer_ops', 'ninfer_nvfp4_tma'):
        print(f"       {n:18s} lines {i}-{j}  ({len(ss)} sources)")

INC = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
closure, unresolved, queue = set(), set(), [os.path.normpath(os.path.join(SRC, c)) for c in compiled]
while queue:
    p = queue.pop()
    if p in closure or not os.path.isfile(p):
        continue
    closure.add(p)
    for l in read(p):
        m = INC.match(l)
        if not m:
            continue
        for cand in (os.path.join(os.path.dirname(p), m.group(1)),
                     os.path.join(SRC, m.group(1)), os.path.join(R, 'include', m.group(1))):
            if os.path.isfile(cand):
                queue.append(os.path.normpath(cand)); break
        else:
            unresolved.add(m.group(1))
new_in_closure = sorted(rel(p) for p in closure
                        if re.search(r'(gqa_attention_decode_(partial|smallt)|nvfp4_w4a4_tma_(arms|attn|gdn|mlp|residual))', p))
print(f"[SC2] include closure {len(closure)} files, unresolved includes {len(unresolved)}")
print(f"       split files inside the closure: {len(new_in_closure)}"
      + ("" if len(new_in_closure) >= 5 else "  <-- fewer than expected: CMake entries missing?"))
for u in sorted(unresolved)[:5]:
    print(f"       note unresolved include: {u}")

# --- definition-site detection ---------------------------------------------
def anon_ranges(lines_):
    """line ranges of unnamed namespaces, via brace depth (col-agnostic, comment safe-ish)"""
    depth, stack, out = 0, [], []
    for k, l in enumerate(lines_):
        t = l.strip()
        if re.match(r'^namespace\s*\{', t) or re.match(r'^namespace\s+[\w:]+.*\{', t):
            kind = 0 if re.match(r'^namespace\s*\{', t) else 1
            stack.append((depth, kind, k)); depth += 1; continue
        nd = depth + l.count('{') - l.count('}')
        while stack and nd <= stack[-1][0]:
            d, kind, start = stack.pop()
            if kind == 0:
                out.append((start, k))
        depth = nd
    return out

def classify(lines_, i):
    """'def' if the head line at i opens a body, 'decl' if the signature ends with ;"""
    t = lines_[i]
    if '{' in t:
        return 'def'
    if t.strip().endswith(';'):
        return 'decl'
    for k in range(i + 1, min(i + 24, len(lines_))):
        s = lines_[k].strip()
        if not s or s.startswith('//'):
            continue
        if s.endswith(';'):
            return 'decl'
        if '{' in s:
            return 'def'
    return 'decl'

_cache_hits = {}
def hits(path, pat):
    key = (path, pat)
    if key in _cache_hits:
        return _cache_hits[key]
    ls, rx, out = read(path), re.compile(pat), []
    if not any(rx.match(l) for l in ls):
        _cache_hits[key] = out; return out
    anon = anon_ranges(ls)
    for i, l in enumerate(ls):
        if rx.match(l):
            out.append((i + 1, classify(ls, i), any(a <= i <= b for a, b in anon)))
    _cache_hits[key] = out
    return out

# --- name tables -----------------------------------------------------------
DISP_GQA  = 'ops/launcher/gqa_attention_decode.cu'
DISP_NVFP4 = 'ops/linear/nvfp4/nvfp4_w4a4_tma.cu'
SIBLINGS = {DISP_GQA, DISP_NVFP4, 'ops/launcher/gqa_attention_decode_smallt.cu'}
# external linkage: exactly one definition in the whole compiled closure
EXT = [
 ('launch_nvfp4_w4a4_tma_linear',         r'\s*void launch_nvfp4_w4a4_tma_linear\(',           'ops/linear/nvfp4/nvfp4_w4a4_tma.cu', 1),
 ('launch_nvfp4_w4a4_tma_attention',      r'\s*void launch_nvfp4_w4a4_tma_attention\(',        'ops/linear/nvfp4/nvfp4_w4a4_tma.cu', 1),
 ('launch_nvfp4_w4a4_tma_gdn',            r'\s*void launch_nvfp4_w4a4_tma_gdn\(',              'ops/linear/nvfp4/nvfp4_w4a4_tma.cu', 1),
 ('launch_nvfp4_w4a4_tma_linear_add',     r'\s*void launch_nvfp4_w4a4_tma_linear_add\(',       'ops/linear/nvfp4/nvfp4_w4a4_tma.cu', 1),
 ('launch_linear_add<>',                  r'\s*void launch_linear_add\(',                      'ops/linear/nvfp4/nvfp4_w4a4_tma_residual.cu', 1),
 ('arm attn_linear',                      r'\s*void launch_nvfp4_w4a4_tma_attn_linear\(',      'ops/linear/nvfp4/nvfp4_w4a4_tma_attn.cu', 1),
 ('arm attn_attention',                   r'\s*void launch_nvfp4_w4a4_tma_attn_attention\(',   'ops/linear/nvfp4/nvfp4_w4a4_tma_attn.cu', 1),
 ('arm gdn_linear',                       r'\s*void launch_nvfp4_w4a4_tma_gdn_linear\(',       'ops/linear/nvfp4/nvfp4_w4a4_tma_gdn.cu', 1),
 ('arm gdn_qkvz',                         r'\s*void launch_nvfp4_w4a4_tma_gdn_qkvz\(',         'ops/linear/nvfp4/nvfp4_w4a4_tma_gdn.cu', 1),
 ('arm mlp_linear',                       r'\s*void launch_nvfp4_w4a4_tma_mlp_linear\(',       'ops/linear/nvfp4/nvfp4_w4a4_tma_mlp.cu', 1),
 ('arm residual_linear',                  r'\s*void launch_nvfp4_w4a4_tma_residual_linear\(',  'ops/linear/nvfp4/nvfp4_w4a4_tma_residual.cu', 1),
 ('arm residual_linear_add',              r'\s*void launch_nvfp4_w4a4_tma_residual_linear_add\(', 'ops/linear/nvfp4/nvfp4_w4a4_tma_residual.cu', 1),
 ('gqa_attention_uses_small_t',           r'\s*bool gqa_attention_uses_small_t\(',             DISP_GQA, 1),
 ('gqa_attention_split_capacity',         r'\s*std::int32_t gqa_attention_split_capacity\(',   DISP_GQA, 1),
 ('gqa_attention_small_t_launch',         r'\s*void gqa_attention_small_t_launch\(',           'ops/launcher/gqa_attention_decode_smallt.cu', 1),
 ('gqa_attention_cached_small_t_launch',  r'\s*void gqa_attention_cached_small_t_launch\(',    'ops/launcher/gqa_attention_decode_smallt.cu', 1),
 ('gqa_attention_decode_e8_launch',       r'\s*void gqa_attention_decode_e8_launch\(',         'ops/launcher/gqa_attention_decode_e8.cu', 2),
]
# anonymous-namespace helpers: 1 def in the owner, 0 in the dispatcher/sibling TUs
INT = [
 ('gqa_small_t_split_upper_bound',    r'\s*(?:template <[^>]*>\s*)?std::int32_t gqa_small_t_split_upper_bound\(', 'ops/launcher/gqa_attention_decode_partial.cuh'),
 ('gqa_small_t_split_count',          r'\s*(?:template <[^>]*>\s*)?std::int32_t gqa_small_t_split_count\(',       'ops/launcher/gqa_attention_decode_partial.cuh'),
 ('gqa_small_t_launch_capacity',      r'\s*(?:template <[^>]*>\s*)?std::int32_t gqa_small_t_launch_capacity\(',   'ops/launcher/gqa_attention_decode_partial.cuh'),
 ('require_nvfp4_geometry_dim',       r'\s*\[\[noreturn\]\] inline void require_nvfp4_geometry_dim\(',            'ops/launcher/gqa_attention_decode_partial.cuh'),
 ('launch_tc_partial_bf16',           r'\s*void launch_tc_partial_bf16\(',  'ops/launcher/gqa_attention_decode_partial.cuh'),
 ('launch_tc_partial_fp8',            r'\s*void launch_tc_partial_fp8\(',   'ops/launcher/gqa_attention_decode_partial.cuh'),
 ('launch_tc_partial_iso3',           r'\s*void launch_tc_partial_iso3\(',  'ops/launcher/gqa_attention_decode_partial.cuh'),
 ('launch_tc_partial_i8',             r'\s*void launch_tc_partial_i8\(',    'ops/launcher/gqa_attention_decode_partial.cuh'),
 ('launch_tc_partial_nvfp4',          r'\s*void launch_tc_partial_nvfp4\(', 'ops/launcher/gqa_attention_decode_partial.cuh'),
 ('single_row_batch_view',            r'\s*PagedKVBatchLayerView single_row_batch_view\(', 'ops/launcher/gqa_attention_decode_smallt.cu'),
 ('gqa_attention_small_t_launch_for', r'\s*void gqa_attention_small_t_launch_for\(',      'ops/launcher/gqa_attention_decode_smallt.cu'),
 ('launch_tma<>',                     r'\s*void launch_tma\(',             'ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh'),
 ('launch_linear<>',                  r'\s*void launch_linear\(',          'ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh'),
 ('AttentionOutput',                  r'\s*struct AttentionOutput \{',     'ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh'),
]
# substrings that must be gone from the two dispatchers (uses of the shared
# helpers are still allowed - gqa_attention_decode.cu keeps calling
# gqa_small_t_launch_capacity<> from gqa_attention_split_capacity)
NOOP = [
 (DISP_GQA, ['launch_tc_partial_', 'single_row_batch_view', 'gqa_attention_small_t_launch_for',
             'NINFER_GQA_SMALL_T_DISPATCH', 'gqa_attention_decode_e8_launch']),
 (DISP_NVFP4, ['launch_tma<', 'launch_linear<', 'launch_linear_add<', 'nvfp4_w4a4_tma_kernel',
               'AttentionOutput', 'TmaM256N128', 'Nvfp4AttnInputGeometry', 'Nvfp4GdnInputGeometry',
               'Nvfp4MlpGateUpGeometry', 'Nvfp4Residual6144Geometry', 'Nvfp4Residual17408Geometry']),
]

# --- SC3 external-linkage uniqueness ---------------------------------------
print("[SC3] external-linkage symbols: exactly one definition in the compiled closure")
for label, pat, want_file, want_n in EXT:
    where = []
    for p in sorted(closure):
        d = [i for i, kind, an in hits(p, pat) if kind == 'def']
        if d:
            where.append((rel(p), d))
    if len(where) == 1 and where[0][0] == want_file and len(where[0][1]) == want_n:
        print(f"       ok   {label:26s} {want_file} x{want_n}")
    else:
        msg = (f"{label}: definitions " +
               (", ".join(f"{w} lines {d}" for w, d in where) or "NOWHERE") +
               f" (want {want_file} x{want_n})")
        print(f"       FAIL {msg}"); fails.append(msg)

# --- SC4 anon-ns helpers ---------------------------------------------------
print("[SC4] anonymous-namespace helpers: 1 def in owner, 0 in the dispatchers/sibling")
for label, pat, owner in INT:
    prob = []
    for p in sorted(closure):
        rp = rel(p)
        d = [i for i, kind, an in hits(p, pat) if kind == 'def']
        if not d:
            continue
        if rp == owner:
            if len(d) != 1:
                prob.append(f"{owner} has {len(d)} defs (want 1)")
        elif rp in SIBLINGS:
            prob.append(f"{rp} lines {d} (must not define it)")
        else:
            infos.append(f"same-name definition in the compiled set: {rp} lines {d} [{label}]"
                         f" internal-linkage={all(an for i, k, an in hits(p, pat) if k == 'def')}")
    if prob:
        print(f"       FAIL {label}: " + "; ".join(prob)); fails += prob
    else:
        print(f"       ok   {label:26s} {owner} x1, dispatchers x0")

# --- SC5 dispatchers must not mention the relocated machinery --------------
print("[SC5] dispatchers must not reference the relocated machinery")
for r_, pats in NOOP:
    txt = '\n'.join(read(os.path.join(SRC, r_)))
    bad = [x for x in pats if x in txt]
    if bad:
        print(f"       FAIL {r_} still contains: {', '.join(bad)}"); fails.append(r_ + ' ' + ','.join(bad))
    else:
        print(f"       ok   {r_} ({len(pats)} patterns absent, {txt.count(chr(10))+1} lines)")

# --- SC6 include wiring ----------------------------------------------------
print("[SC6] include wiring")
inc_bad = []
for r_, needle, want in [
    (DISP_GQA, 'ops/launcher/gqa_attention_decode_partial.cuh', 1),
    ('ops/launcher/gqa_attention_decode_smallt.cu', 'ops/launcher/gqa_attention_decode_partial.cuh', 1),
    (DISP_NVFP4, 'ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh', 1),
    ('ops/linear/nvfp4/nvfp4_w4a4_tma_attn.cu', 'ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh', 1),
    ('ops/linear/nvfp4/nvfp4_w4a4_tma_gdn.cu', 'ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh', 1),
    ('ops/linear/nvfp4/nvfp4_w4a4_tma_mlp.cu', 'ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh', 1),
    ('ops/linear/nvfp4/nvfp4_w4a4_tma_residual.cu', 'ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh', 1),
]:
    n = '\n'.join(read(os.path.join(SRC, r_))).count(f'#include "{needle}"')
    if n != want:
        inc_bad.append(f"{r_}: {n}x #{needle} (want {want})")
if inc_bad:
    print("       FAIL " + "; ".join(inc_bad)); fails += inc_bad
else:
    print("       ok   7 include edges (dispatchers -> partial.cuh / arms.cuh, arms -> arms.cuh)")

# --- SC7 CMake list --------------------------------------------------------
print("[SC7] CMake source lists")
raw = open(CMAKE, 'rb').read()
if b'\r' in raw:
    print("       FAIL src/CMakeLists.txt contains CR bytes"); fails.append('cmake CR')
cl = raw.decode('utf-8').split('\n')
strips = [l.strip() for l in cl]
blk_ranges = {}
for i, l in enumerate(cl):
    m = re.match(r'\s*add_library\((\w+)', l)
    if m:
        j = i
        while ')' not in cl[j]:
            j += 1
        blk_ranges[m.group(1)] = (i, j)
newtgt = ['ops/launcher/gqa_attention_decode_smallt.cu',
          'ops/linear/nvfp4/nvfp4_w4a4_tma_attn.cu', 'ops/linear/nvfp4/nvfp4_w4a4_tma_gdn.cu',
          'ops/linear/nvfp4/nvfp4_w4a4_tma_mlp.cu', 'ops/linear/nvfp4/nvfp4_w4a4_tma_residual.cu']
for t in newtgt:
    n = strips.count(t)
    ok = False
    if n == 1:
        idx = strips.index(t)
        ok = any(i <= idx <= j for nm, (i, j) in blk_ranges.items()
                 if nm in ('ninfer_ops', 'ninfer_nvfp4_tma'))
    if ok:
        print(f"       ok   {t} x1 inside its add_library block")
    else:
        print(f"       FAIL {t} appears {n}x / outside the block (want exactly 1 inside)")
        fails.append(t)
for h in ('ops/launcher/gqa_attention_decode_partial.cuh', 'ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh'):
    if h in strips:
        print(f"       FAIL header {h} listed as a source (headers need no entry)"); fails.append(h)
print(f"       ok   headers not listed; CMakeLists is {len(cl)-1} lines")

# --- SC8 hygiene -----------------------------------------------------------
print("[SC8] hygiene")
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
bad_h = []
landed = [DISP_GQA, 'ops/launcher/gqa_attention_decode_partial.cuh',
          'ops/launcher/gqa_attention_decode_smallt.cu', DISP_NVFP4,
          'ops/linear/nvfp4/nvfp4_w4a4_tma_arms.cuh'] + \
         [f'ops/linear/nvfp4/nvfp4_w4a4_tma_{a}.cu' for a in ('attn', 'gdn', 'mlp', 'residual')]
for r_ in landed:
    b = open(os.path.join(SRC, r_), 'rb').read()
    if b'\r' in b:
        bad_h.append(f"{r_} has CR")
    if not b.endswith(b'\n'):
        bad_h.append(f"{r_} lacks final newline")
if bad_h:
    print("       FAIL " + "; ".join(bad_h)); fails += bad_h
else:
    print("       ok   all 9 landed files are LF-only, newline-terminated")

# --- SC9 informational: duplicates outside the build -----------------------
print("[SC9] informational")
RELOC = [r'\s*void launch_tc_partial_', r'\s*std::int32_t gqa_small_t_',
         r'\s*void gqa_attention_small_t_launch_for\(', r'\s*void launch_linear_add\(',
         r'\s*void launch_tma\(', r'\s*void launch_linear\(',
         r'\s*void launch_nvfp4_w4a4_tma_attn_linear\(']
for root, dirs, files in os.walk(SRC):
    dirs[:] = [d for d in dirs if d not in ('build', '.git')]
    for f in sorted(files):
        if not f.endswith(('.cu', '.cuh')):
            continue
        p = os.path.join(root, f)
        if p in closure:
            continue
        ls = read(p)
        n = sum(1 for l in ls if any(re.match(rx, l) for rx in RELOC))
        if n:
            infos.append(f"INERT (not reachable from any compiled TU): {rel(p)} defines {n} relocated-name head(s)")
if not infos:
    print("       (none)")
for i in infos:
    print("       " + i)

print()
if fails:
    print(f"SELF-CHECK: {len(fails)} FAILURE(S)")
    for f in fails:
        print("  - " + f)
    sys.exit(3)
print(f"SELF-CHECK: OK  (0 failures, {len(infos)} informational note(s))")
PY
  local rc=$?
  if [ "$rc" != "0" ]; then
    echo
    echo "!! self-check failed (rc=$rc): the tree holds a PARTIAL landing."
    echo "!! inspect the tree, then either fix by hand or run: bash $0 --revert"
    return 3
  fi
}

# ===========================================================================
# 4. REVERT
# ===========================================================================
do_revert() {
  h1 "REVERT"
  local fails=0 f sf dir name lines md5 first p am src snap="" c
  [ -f "$LANDDIR/LAST.txt" ] && snap=$(cat "$LANDDIR/LAST.txt")

  echo "[R1] restore the two pre-split .cu"
  for f in "${ORIGS[@]}"; do
    IFS='|' read -r p md5 lines <<<"$f"
    src=""
    for c in "$snap/$(basename "$p")" "$BACKUP/$(basename "$p").orig"; do
      [ -n "$c" ] && [ -f "$c" ] && [ "$(md5f "$c")" = "$md5" ] && { src=$c; break; }
    done
    if [ -z "$src" ] && [ "$(basename "$p")" = "gqa_attention_decode.cu" ]; then
      for c in /tmp/orig.cu "$COLLAB/tmp/orig_gqa_attention_decode.cu"; do
        [ -f "$c" ] && [ "$(md5f "$c")" = "$md5" ] && { src=$c; break; }
      done
    fi
    if [ -z "$src" ]; then
      echo "  FAIL no verified backup for $p (want md5 $md5)"; fails=$((fails+1)); continue
    fi
    if [ "$DRY" = "1" ]; then
      echo "  [dry-run] cp $src -> ${p#$R/}   (restores md5 $md5)"
    else
      cp -p "$src" "$p"
      echo "  <- ${p#$R/} restored from $src  md5=$(md5f "$p")  lines=$(nl "$p")"
    fi
  done

  echo "[R2] delete files this landing created"
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
  printf '  %-58s %s  [%s]\n' "src/ops/launcher/gqa_attention_decode.cu" "$(md5f "$G/gqa_attention_decode.cu")" "b0a130e21e02e42a0d1f9834dfac8aae"
  printf '  %-58s %s  [%s]\n' "src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu"   "$(md5f "$N/nvfp4_w4a4_tma.cu")"   "12714635ae9ecb70b223fba5e619b2fa"
  printf '  %-58s %s  [%s]\n' "src/CMakeLists.txt"                        "$(md5f "$CMAKE")"                  "19c515e91a28bb0c4c0eeef613b04b0f"
  for p in "$G/gqa_attention_decode_partial.cuh" "$G/gqa_attention_decode_smallt.cu" \
           "$N/nvfp4_w4a4_tma_arms.cuh" "$N/nvfp4_w4a4_tma_attn.cu" "$N/nvfp4_w4a4_tma_gdn.cu" \
           "$N/nvfp4_w4a4_tma_mlp.cu" "$N/nvfp4_w4a4_tma_residual.cu"; do
    [ -f "$p" ] || continue
    if [ "$DRY" = "1" ]; then echo "  [dry-run] still present (would be removed): $p"
    else echo "  STILL PRESENT: $p"; fails=$((fails+1)); fi
  done
  hr
  if [ "$fails" != "0" ]; then echo "REVERT: incomplete ($fails issue(s)) - see above"; return 4; fi
  echo "REVERT: OK - tree is back to the pre-split state (re-run cmake before building)"
}

# ===========================================================================
# main
# ===========================================================================
echo "=== _land_split.sh  mode=$MODE dry-run=$DRY  $(date '+%F %T') ==="
echo "    tree=$R"
preflight
if [ "$MODE" = "revert" ]; then
  do_revert || exit $?
  echo "=== done $(date '+%F %T') ==="
  exit 0
fi
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

h1 "NEXT STEPS (NOT executed by this script - run them in the compile window)"
cat <<EOF
  1) reconfigure so the new sources enter the build graph:
       cmake -S $R -B $R/build

  2) memory-aware parallel build (it sizes -j from MemAvailable itself):
       bash $J/_par_build.sh
     narrower first pass, if you prefer:   bash $J/_par_build.sh ninfer-ops
     log: $J/dl/par_build.log

  3) new objects to expect:
       $R/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode_smallt.cu.o
       $R/build/src/CMakeFiles/ninfer_nvfp4_tma.dir/ops/linear/nvfp4/nvfp4_w4a4_tma_{attn,gdn,mlp,residual}.cu.o
     the two dispatcher .o files should be tiny (no kernel instantiations left).

  rollback at any time:  bash $J/_land_split.sh --revert
EOF
echo "=== done $(date '+%F %T') ==="
