#!/bin/bash
# A3 headwise gate fix: build the helper patch + verify it. CPU-only, scratch in /tmp.
# Never writes into the live tree /home/user/ninfer-fusion.
set -u
T=/home/user/ninfer-fusion
COL=/mnt/c/Users/User/Documents/ziqinzhang/_collab
S=$COL/a3_scratch/shadows/headwise_gate
W=/tmp/a3fix
REL=src/ops/wrapper/sigmoid_mul.cpp
OUT=$COL/build/A3_headwise_gate_fix.diff

rm -rf $W
mkdir -p $W/a/src/ops/wrapper $W/b/src/ops/wrapper $W/cur/src/ops/wrapper $W/pre/src/ops/wrapper
mkdir -p $W/inc/ops/launcher $W/o1/src/ops/wrapper $W/o2/src/ops/wrapper
cp $T/$REL $W/a/$REL
cp "$S/$REL" $W/pre/$REL
cp $S/src/ops/launcher/sigmoid_gate_mul.h $W/inc/ops/launcher/sigmoid_gate_mul.h

echo "=== STEP 1: build the patched (b) file with an anchored insertion ==="
python3 - <<'PY'
import pathlib
W = pathlib.Path("/tmp/a3fix")
rel = "src/ops/wrapper/sigmoid_mul.cpp"
src = (W / "a" / rel).read_text(encoding="utf-8")

anchor = "}\n\n} // namespace\n"
if src.count(anchor) != 1:
    raise SystemExit("ANCHOR FAIL: %d occurrences" % src.count(anchor))

helper = (
    "// Headwise scalar gate form: `x` is [D,H,T] and `gate` is [H,T] -- one sigmoid per (head,\n"
    "// token) broadcast over the head_dim axis. Selected by shape alone, and only for pairs the\n"
    "// per-element route below rejects: equal shapes satisfy it only for a single-element tensor,\n"
    "// where both routes compute the same value. Contract: ninfer/ops/sigmoid_mul.h.\n"
    "bool headwise_gate_shape(const Tensor& gate, const Tensor& x) {\n"
    "    return x.ne[3] == 1 && gate.ne[2] == 1 && gate.ne[3] == 1 && gate.ne[0] == x.ne[1] &&\n"
    "           gate.ne[1] == x.ne[2];\n"
    "}\n"
)
out = src.replace(anchor, "}\n\n" + helper + "\n} // namespace\n")

# structural assertions: helper must sit inside the anonymous namespace, before sigmoid_mul
lines = out.split("\n")
i_ns = next(i for i, l in enumerate(lines) if l.strip() == "namespace {")
i_close = next(i for i, l in enumerate(lines) if l.strip() == "} // namespace")
i_fn = next(i for i, l in enumerate(lines) if l.startswith("bool headwise_gate_shape("))
i_pub = next(i for i, l in enumerate(lines) if l.startswith("void sigmoid_mul("))
assert i_ns < i_fn < i_close < i_pub, (i_ns, i_fn, i_close, i_pub)
assert out.count("bool headwise_gate_shape(") == 1
longest = max(len(l) for l in helper.split("\n"))
assert longest <= 100, longest
(W / "b" / rel).write_text(out, encoding="utf-8")
print("OK: helper inserted at line %d, anonymous namespace closes at %d, sigmoid_mul at %d" %
      (i_fn + 1, i_close + 1, i_pub + 1))
print("OK: added lines = %d, longest added line = %d cols" %
      (len(helper.rstrip("\n").split("\n")), longest))
PY

echo
echo "=== STEP 2: emit the unified diff (byte-level, LF, repo convention) ==="
cd $W && diff -u --label a/$REL --label b/$REL a/$REL b/$REL > $W/fix.diff
cp $W/fix.diff $OUT
echo "--- patch content ---"
cat $W/fix.diff
echo "--- patch bytes: CR count, trailing newline ---"
grep -c $'\r' $OUT || true
tail -c 1 $OUT | od -c | head -2
wc -l $OUT

echo
echo "=== STEP 3: dry-run on a shadow of the CURRENT (headwise-unapplied) tree ==="
cp $T/$REL $W/cur/$REL
cd $W/cur && patch -p1 --dry-run --fuzz=0 -i $W/fix.diff; echo "RC=$?"
cd $W/cur && patch -p1 --dry-run -i $W/fix.diff; echo "RC=$?"

echo
echo "=== STEP 4: dry-run on the A3-applied shadow (broken state) ==="
cd $W/pre && patch -p1 --dry-run -i $W/fix.diff; echo "RC=$?"

echo
echo "=== STEP 5: both application orders converge byte-for-byte ==="
# order 1: A3 headwise first, then fix
cp $S/$REL $W/o1/$REL
cd $W/o1 && patch -p1 -i $W/fix.diff >/dev/null; echo "order1 fix-apply RC=$?"
# order 2: fix first, then A3 headwise (full 6-file diff on a shadow of the baseline)
rm -rf $W/o2 && mkdir -p $W/o2 && cp -r $S/* $W/o2/ 2>/dev/null || true
cd $W/o2 && patch -p1 -i $W/fix.diff; echo "order2 fix-apply RC=$?"
cd $W/o2 && patch -p1 -i $COL/A3_spark_headwise_gate.diff; echo "order2 A3-apply RC=$?"
echo "--- cmp order1 vs order2 wrapper ---"
cmp $W/o1/$REL $W/o2/$REL && echo "IDENTICAL"
echo "--- cmp order2 wrapper vs A3-shadow+unified helper (expect IDENTICAL) ---"
cmp $W/o2/$REL $W/b/$REL 2>/dev/null || true
echo "--- patched wrapper (numbered, order2) ---"
cat -n $W/o2/$REL
echo "--- diff vs A3 'b' staged file (the A3-only state) ---"
diff $W/b/$REL $S/$REL || true

echo
echo "=== STEP 6: host compile check (g++ -fsyntax-only, exact build flags) ==="
CXX=/usr/bin/c++
FLAGS="-std=gnu++20 -fsyntax-only -I$W/inc -I$T/include -I$T/src -I$T/third_party -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include/cccl"
echo "--- 6a NEGATIVE CONTROL: A3-applied wrapper without the helper (expect the reported error) ---"
$CXX $FLAGS $W/pre/$REL 2>&1 | head -12; echo "RC=${PIPESTATUS[0]}"
echo "--- 6b POSITIVE: A3-applied + fix (expect rc=0) ---"
$CXX $FLAGS $W/o1/$REL 2>&1 | head -12; echo "RC=${PIPESTATUS[0]}"
echo "--- 6c POSITIVE: current tree file + fix alone (expect rc=0) ---"
$CXX $FLAGS $W/o2/$REL 2>&1 | head -12; echo "RC=${PIPESTATUS[0]}"
echo "--- 6d POSITIVE (upstream headers, no A3 launcher decl): current file + fix ---"
cd $W && mkdir -p cur2 && cp $T/$REL cur2/$REL && patch -p1 -d cur2 -i $W/fix.diff >/dev/null && cp cur2/$REL cur3.cpp
$CXX -std=gnu++20 -fsyntax-only -I$T/include -I$T/src -I$T/third_party -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include/cccl cur3.cpp 2>&1 | head -8; echo "RC=${PIPESTATUS[0]}"

echo
echo "=== STEP 7: final artifacts ==="
ls -la $OUT
md5sum $OUT $W/b/$REL
echo done
