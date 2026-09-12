#!/bin/bash
# A3 headwise fix v4: final checks + assemble the evidence file.
set -u
T=/home/user/ninfer-fusion
COL=/mnt/c/Users/User/Documents/ziqinzhang/_collab
B=$COL/a3_scratch/b
L=$COL/a3_scratch/live
REL=src/ops/wrapper/sigmoid_mul.cpp
LH=src/ops/launcher/sigmoid_gate_mul.h
FIX=$COL/build/A3_headwise_gate_fix.diff
EV=$COL/build/A3_headwise_fix_evidence.txt
W=/tmp/a3fix4
rm -rf $W; mkdir -p $W/tree/src/ops/wrapper $W/inc/ops/launcher
cp $L/$REL $W/tree/$REL && cd $W/tree && patch -p1 -i $FIX >/dev/null && cd $W
cp $B/$LH $W/inc/ops/launcher/sigmoid_gate_mul.h

{
echo "A3 headwise gate fix — raw evidence"
echo "generated: $(date -Is)   host: $(uname -srm)   distro: $(lsb_release -ds 2>/dev/null || cat /etc/os-release | head -1)"
echo "compiler: $(/usr/bin/c++ --version | head -1)"
echo "patch:    $(patch --version | head -1)"
echo "scope:    CPU-only. No make / nvcc / cmake / GPU. Live tree untouched (scratch = /tmp)."
echo
echo "==========================================================================="
echo "[E1] the delivered patch (byte-level, generated from the live file)"
echo "==========================================================================="
cat -A $FIX | sed 's/\$$//' | head -0   # (patch is plain LF; see E2 for byte audit)
cat $FIX
echo "--- md5 ---"; md5sum $FIX
echo
echo "==========================================================================="
echo "[E2] byte properties (must be LF, no CR, final newline)"
echo "==========================================================================="
echo "CR count: $(grep -c $'\r' $FIX || true)"
echo "last byte: $(tail -c 1 $FIX | od -An -c | tr -d ' \n')"
wc -lc $FIX
echo
echo "==========================================================================="
echo "[E3] dry-run on the CURRENT live tree state (headwise not applied)"
echo "==========================================================================="
cd $W/tree && cp $L/$REL $W/tree/$REL
cd $W/tree && patch -p1 --dry-run --fuzz=0 -i $FIX; echo "rc=$?"
cd $W/tree && patch -p1 --dry-run          -i $FIX; echo "rc=$?"
echo
echo "==========================================================================="
echo "[E4] dry-run on the A3-applied state (helper missing -> the broken state)"
echo "==========================================================================="
mkdir -p $W/pre/src/ops/wrapper && cp $B/$REL $W/pre/$REL
cd $W/pre && patch -p1 --dry-run --fuzz=0 -i $FIX; echo "rc=$?"
echo
echo "==========================================================================="
echo "[E5] apply-order equivalence (byte-for-byte)"
echo "==========================================================================="
mkdir -p $W/oa/src/ops/wrapper $W/ob/src/ops/wrapper
cp $B/$REL $W/oa/$REL && cd $W/oa && patch -p1 -i $FIX >/dev/null && echo "order A (A3 -> fix): applied"
cp -r $L/* $W/ob/ 2>/dev/null
cd $W/ob && patch -p1 -i $FIX >/dev/null && echo "order B (fix -> A3): fix applied"
cd $W/ob && patch -p1 -i $COL/A3_spark_headwise_gate.diff >/dev/null && echo "order B: A3 headwise applied"
cmp $W/oa/$REL $W/ob/$REL && echo "cmp: IDENTICAL (both orders converge on the same bytes)"
echo "--- resulting wrapper, numbered ---"
cat -n $W/oa/$REL
echo
echo "==========================================================================="
echo "[E6] host compile checks, exact flags of the real build (g++ -fsyntax-only)"
echo "==========================================================================="
CXX=/usr/bin/c++
INC="-I$T/include -I$T/src -I$T/third_party -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include/cccl"
echo "--- 6a NEGATIVE CONTROL: A3-applied wrapper, helper missing ---"
$CXX -std=gnu++20 -fsyntax-only -I$W/inc $INC $B/$REL; echo "rc=$?"
echo "--- 6b POSITIVE: A3-applied + fix ---"
$CXX -std=gnu++20 -fsyntax-only -I$W/inc $INC $W/oa/$REL; echo "rc=$?"
echo "--- 6c POSITIVE: fix alone on the live file (upstream headers only) ---"
$CXX -std=gnu++20 -fsyntax-only $INC $W/tree/$REL; echo "rc=$?"
echo "--- 6d POSITIVE: the A3-applied TEST TU (host-compiled test) ---"
$CXX -std=gnu++20 -fno-fast-math -ffp-contract=off -fsyntax-only -I$B/include -I$T/tests -I$T/include -I$T/src -I$T/third_party $B/tests/ops/test_sigmoid_mul.cpp; echo "rc=$?"
echo
echo "==========================================================================="
echo "[E7] judgment matrix, mechanically evaluated on the shipped predicate"
echo "     (extracted from the patched file by sed; 'const Tensor&' -> 'const Shape&')"
echo "==========================================================================="
} > $EV 2>&1

sed -n '/^bool headwise_gate_shape/,/^}/p' $W/oa/$REL > $W/pred.txt
{
echo "--- shipped predicate (verbatim, md5 $(md5sum $W/pred.txt | cut -c1-32)) ---"
cat $W/pred.txt
} >> $EV

python3 - <<'PY' >> $EV 2>&1
import pathlib, subprocess
W = pathlib.Path("/tmp/a3fix4")
pred = (W / "pred.txt").read_text(encoding="utf-8").replace("const Tensor&", "const Shape&")
prog = r'''
#include <cstdint>
#include <cstdio>
struct Shape { std::int32_t ne[4]; };
''' + pred + r'''
struct Case { const char* label; std::int32_t g[4]; std::int32_t x[4]; const char* want; };
int main() {
    const Case cases[] = {
        {"headwise 16x256x1   gate[16,1]  x[256,16,1]",   {16,1,1,1},    {256,16,1,1},  "true"},
        {"headwise 16x256x6   gate[16,6]  x[256,16,6]",   {16,6,1,1},    {256,16,6,1},  "true"},
        {"headwise 16x256x12  gate[16,12] x[256,16,12]",  {16,12,1,1},   {256,16,12,1}, "true"},
        {"headwise 32x128x3   gate[32,3]  x[128,32,3]",   {32,3,1,1},    {128,32,3,1},  "true"},
        {"headwise 4x256x5    gate[4,5]   x[256,4,5]",    {4,5,1,1},     {256,4,5,1},   "true"},
        {"single element (documented overlap; both routes equal)", {1,1,1,1}, {1,1,1,1}, "true"},
        {"per-element [6144,1]   gate==x",                {6144,1,1,1},  {6144,1,1,1},  "false"},
        {"per-element [6144,48]  gate==x",                {6144,48,1,1}, {6144,48,1,1}, "false"},
        {"per-element [4096,17]  gate==x",                {4096,17,1,1}, {4096,17,1,1}, "false"},
        {"per-element [4096,128] gate==x",                {4096,128,1,1},{4096,128,1,1},"false"},
        {"per-element square [256,256] (A3 test, seed 501u)", {256,256,1,1}, {256,256,1,1}, "false"},
        {"per-element 1-D edge case [7]",                 {7,1,1,1},     {7,1,1,1},     "false"},
        {"per-element [256,256,4] equal 3-D shapes",      {256,256,4,1}, {256,256,4,1}, "false"},
        {"live 27b gate[256,24,T] x[256,24,T]",           {256,24,8,1},  {256,24,8,1},  "false"},
        {"live 35b gate[256,16,T] x[256,16,T]",           {256,16,8,1},  {256,16,8,1},  "false"},
        {"live muse gate[128,32,T] x[128,32,T]",          {128,32,8,1},  {128,32,8,1},  "false"},
        {"live mismatch gate[256,24,T] vs x[256,16,T]",   {256,24,8,1},  {256,16,8,1},  "false"},
        {"T in last axis: x[256,16,1,6] with gate[16,1]", {16,1,1,1},    {256,16,1,6},  "false"},
        {"gate not [H,T]: gate[16,6,2] x[256,16,6]",      {16,6,2,1},    {256,16,6,1},  "false"},
        {"H mismatch: gate[8,12] x[256,16,12]",           {8,12,1,1},    {256,16,12,1}, "false"},
        {"T mismatch: gate[16,7] x[256,16,12]",           {16,7,1,1},    {256,16,12,1}, "false"},
        {"degenerate empty x[0,16,12] (see report note)", {16,12,1,1},   {0,16,12,1},   "true"},
    };
    int bad = 0;
    for (const Case& c : cases) {
        const Shape g{c.g[0],c.g[1],c.g[2],c.g[3]}, x{c.x[0],c.x[1],c.x[2],c.x[3]};
        const bool got = headwise_gate_shape(g, x);
        const char* got_s = got ? "true" : "false";
        const bool ok = (std::string(got_s) == c.want);
        if (!ok) ++bad;
        std::printf("%-54s -> %-5s (expected %-5s) %s\n", c.label, got_s, c.want, ok ? "ok" : "MISMATCH");
    }
    std::printf("MISMATCHES=%d / %d cases\n", bad, (int)(sizeof(cases)/sizeof(cases[0])));
    return bad != 0;
}
'''.replace("#include <cstdint>", "#include <cstdint>\n#include <string>")
(W / "matrix.cpp").write_text(prog, encoding="utf-8")
r = subprocess.run(["/usr/bin/c++", "-std=gnu++20", "-O0", "-o", str(W / "matrix"), str(W / "matrix.cpp")],
                   capture_output=True, text=True)
print(r.stdout + r.stderr, end="")
if r.returncode != 0:
    raise SystemExit("matrix build failed")
r = subprocess.run([str(W / "matrix")], capture_output=True, text=True)
print(r.stdout + r.stderr, end="")
print("matrix rc=%d" % r.returncode)
PY

{
echo
echo "==========================================================================="
echo "[E8] state fingerprints"
echo "==========================================================================="
echo "live baseline (a3_scratch/live)  $REL md5: $(md5sum $L/$REL | cut -d' ' -f1)"
echo "live tree                        $REL md5: $(md5sum $T/$REL | cut -d' ' -f1)"
echo "A3-applied (a3_scratch/b)        $REL md5: $(md5sum $B/$REL | cut -d' ' -f1)"
echo "A3+fix result (this patch)       $REL md5: $(md5sum $W/oa/$REL | cut -d' ' -f1)"
echo "delivered patch md5: $(md5sum $FIX | cut -d' ' -f1)"
echo
echo "A3 patch file md5:   $(md5sum $COL/A3_spark_headwise_gate.diff | cut -d' ' -f1)  (deliverable 1 supplements this file; it is NOT modified)"
echo "A3 headwise touched files and their current live md5:"
cd $T && md5sum src/ops/wrapper/sigmoid_mul.cpp src/ops/launcher/sigmoid_gate_mul.h src/ops/launcher/sigmoid_gate_mul.cu src/ops/kernel/sigmoid_gate_mul.cuh include/ninfer/ops/sigmoid_mul.h tests/ops/test_sigmoid_mul.cpp
} >> $EV 2>&1

echo "=== evidence written: $EV ==="
wc -l $EV
echo "=== final artifact ==="
ls -la $FIX
