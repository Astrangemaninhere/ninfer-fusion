#!/bin/bash
# A3 headwise fix v6: final evidence with corrected E6 labels + call-site/-W audit.
set -u
T=/home/user/ninfer-fusion
COL=/mnt/c/Users/User/Documents/ziqinzhang/_collab
B=$COL/a3_scratch/b
L=$COL/a3_scratch/live
REL=src/ops/wrapper/sigmoid_mul.cpp
LH=src/ops/launcher/sigmoid_gate_mul.h
FIX=$COL/build/A3_headwise_gate_fix.diff
EV=$COL/build/A3_headwise_fix_evidence.txt
W=/tmp/a3fix6
rm -rf $W
mkdir -p $W/tree/src/ops/wrapper $W/fixed/src/ops/wrapper $W/inc/ops/launcher $W/pre/src/ops/wrapper
mkdir -p $W/oa/src/ops/wrapper $W/ob/src/ops/wrapper
cp $B/$LH $W/inc/ops/launcher/sigmoid_gate_mul.h
cp $L/$REL $W/tree/$REL
cp $L/$REL $W/fixed/$REL && (cd $W/fixed && patch -p1 -i $FIX >/dev/null)
cp $B/$REL $W/pre/$REL
cp $B/$REL $W/oa/$REL && (cd $W/oa && patch -p1 -i $FIX >/dev/null)
cp -r $L/* $W/ob/ 2>/dev/null && (cd $W/ob && patch -p1 -i $FIX >/dev/null && patch -p1 -i $COL/A3_spark_headwise_gate.diff >/dev/null)

CXX=/usr/bin/c++
INC="-I$T/include -I$T/src -I$T/third_party -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include/cccl"

{
echo "A3 headwise gate fix — raw evidence (v6)"
echo "generated: $(date -Is)   host: $(uname -srm)   distro: $(lsb_release -ds 2>/dev/null || head -1 /etc/os-release)"
echo "compiler: $(/usr/bin/c++ --version | head -1)"
echo "patch:    $(patch --version | head -1)"
echo "scope:    CPU-only host work. No make / nvcc / cmake / GPU. The live tree is not written;"
echo "          all scratch lives in /tmp and in _collab/a3_scratch/ (byte copies)."
echo
echo "==========================================================================="
echo "[E1] the delivered patch (byte-level; generated from the live file's bytes)"
echo "==========================================================================="
cat $FIX
echo "--- md5 ---"; md5sum $FIX
echo
echo "==========================================================================="
echo "[E2] byte properties (LF only, no CR, final newline)"
echo "==========================================================================="
echo "CR count: $(grep -c $'\r' $FIX || true)"
printf 'last byte: '; tail -c 1 $FIX | od -An -c | tr -d ' \n'; echo
wc -lc $FIX
echo
echo "==========================================================================="
echo "[E3] patch -p1 --dry-run on the CURRENT live tree state"
echo "==========================================================================="
cd $W/tree && patch -p1 --dry-run --fuzz=0 -i $FIX; echo "rc=$?"
cd $W/tree && patch -p1 --dry-run          -i $FIX; echo "rc=$?"
echo
echo "==========================================================================="
echo "[E4] patch -p1 --dry-run on the A3-applied state (the helper-missing state)"
echo "==========================================================================="
cd $W/pre && patch -p1 --dry-run --fuzz=0 -i $FIX; echo "rc=$?"
echo
echo "==========================================================================="
echo "[E5] apply-order equivalence: A3->fix vs fix->A3 converge byte-for-byte"
echo "==========================================================================="
cmp $W/oa/$REL $W/ob/$REL && echo "cmp: IDENTICAL"
echo
echo "==========================================================================="
echo "[E6] host compile checks with the exact real-build flags (g++ -fsyntax-only)"
echo "  wrapper TU flags (from build/compile_commands.json):"
echo "    /usr/bin/c++ -I<T>/include -I<T>/src -I<T>/third_party -isystem <cuda>/include"
echo "      -isystem <cuda>/include/cccl -O3 -DNDEBUG -std=gnu++20   [no -Wall/-Wextra/-Werror]"
echo "  test TU flags: -I<T>/tests -I<T>/include -I<T>/src -I<T>/third_party -isystem <cuda>/include"
echo "      -isystem <cuda>/include/cccl -O3 -DNDEBUG -std=gnu++20 -fno-fast-math -ffp-contract=off"
echo "==========================================================================="
echo "--- 6a NEGATIVE CONTROL: A3-applied wrapper, helper missing -> must fail exactly here ---"
$CXX -std=gnu++20 -fsyntax-only -I$W/inc $INC $B/$REL; echo "rc=$?"
echo "--- 6b POSITIVE: A3-applied wrapper + fix -> compiles ---"
$CXX -std=gnu++20 -fsyntax-only -I$W/inc $INC $W/oa/$REL; echo "rc=$?"
echo "--- 6c POSITIVE: live wrapper + fix alone (upstream headers) -> compiles ---"
$CXX -std=gnu++20 -fsyntax-only $INC $W/fixed/$REL; echo "rc=$?"
echo "--- 6d POSITIVE: A3-applied test TU (independent TU; uses only the public api + op_tester.h) ---"
$CXX -std=gnu++20 -fno-fast-math -ffp-contract=off -fsyntax-only -I$B/include -I$T/tests $INC $B/tests/ops/test_sigmoid_mul.cpp; echo "rc=$?"
echo
echo "==========================================================================="
echo "[E7] judgment matrix, evaluated mechanically on the shipped predicate"
echo "     (extracted by sed from the patched file; 'const Tensor&' -> 'const Shape&')"
echo "==========================================================================="
} > $EV 2>&1

sed -n '/^bool headwise_gate_shape/,/^}/p' $W/oa/$REL > $W/pred.txt
{
echo "--- shipped predicate verbatim (md5 $(md5sum $W/pred.txt | cut -c1-32)) ---"
cat $W/pred.txt
} >> $EV

python3 - >> $EV 2>&1 <<'PY'
import pathlib, subprocess
W = pathlib.Path("/tmp/a3fix6")
pred = (W / "pred.txt").read_text(encoding="utf-8").replace("const Tensor&", "const Shape&")
head = '#include <cstdint>\n#include <cstdio>\n#include <cstring>\nstruct Shape { std::int32_t ne[4]; };\n'
tail = r'''
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
        {"degenerate empty x[0,16,12] (report note R3)",  {16,12,1,1},   {0,16,12,1},   "true"},
    };
    int bad = 0;
    for (const Case& c : cases) {
        const Shape g{c.g[0],c.g[1],c.g[2],c.g[3]}, x{c.x[0],c.x[1],c.x[2],c.x[3]};
        const bool got = headwise_gate_shape(g, x);
        const char* got_s = got ? "true" : "false";
        const bool ok = (std::strcmp(got_s, c.want) == 0);
        if (!ok) ++bad;
        std::printf("%-54s -> %-5s (expected %-5s) %s\n", c.label, got_s, c.want, ok ? "ok" : "MISMATCH");
    }
    std::printf("MISMATCHES=%d / %d cases\n", bad, (int)(sizeof(cases)/sizeof(cases[0])));
    return bad != 0;
}
'''
(W / "matrix.cpp").write_text(head + pred + tail, encoding="utf-8")
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
echo "[E8] live-call-site audit + warning-flag audit (why nothing else can reach the branch)"
echo "==========================================================================="
echo "--- every ops::sigmoid_mul call site in the tree (src/apps/tools) ---"
grep -rn 'sigmoid_mul(' $T/src $T/apps $T/tools 2>/dev/null | grep -v 'sigmoid_mul.cpp:\|sigmoid_mul.h:' | grep -v '//'
echo "--- -Wall/-Wextra/-Werror/-Wunused anywhere in the build config ---"
grep -rn 'Werror\|Wextra\|Wunused\|add_compile_options\|CMAKE_CXX_FLAGS' $T/CMakeLists.txt $T/src/CMakeLists.txt $T/tests/CMakeLists.txt $T/apps/CMakeLists.txt 2>/dev/null || echo "(none: an unused helper cannot fail this build)"
echo "--- does the new-test-case name already appear in the live file? (must be no) ---"
grep -c 'run_headwise_case' $T/tests/ops/test_sigmoid_mul.cpp || true
echo
echo "==========================================================================="
echo "[E9] state fingerprints"
echo "==========================================================================="
echo "live baseline (a3_scratch/live)   $REL md5: $(md5sum $L/$REL | cut -d' ' -f1)"
echo "live tree                         $REL md5: $(md5sum $T/$REL | cut -d' ' -f1)"
echo "A3-applied (a3_scratch/b)         $REL md5: $(md5sum $B/$REL | cut -d' ' -f1)"
echo "A3+fix result (this patch)        $REL md5: $(md5sum $W/oa/$REL | cut -d' ' -f1)"
echo "fix-alone result (this patch)     $REL md5: $(md5sum $W/fixed/$REL | cut -d' ' -f1)"
echo "delivered patch                   md5: $(md5sum $FIX | cut -d' ' -f1)"
echo "A3_spark_headwise_gate.diff       md5: $(md5sum $COL/A3_spark_headwise_gate.diff | cut -d' ' -f1)  (NOT modified by this task)"
echo
echo "the 6 headwise files, current live md5 (all equal the A3 baseline => A3 headwise not applied):"
cd $T && md5sum src/ops/wrapper/sigmoid_mul.cpp src/ops/launcher/sigmoid_gate_mul.h src/ops/launcher/sigmoid_gate_mul.cu src/ops/kernel/sigmoid_gate_mul.cuh include/ninfer/ops/sigmoid_mul.h tests/ops/test_sigmoid_mul.cpp
echo
echo "A3 headwise diff dry-run against its own baseline shadow (from the A3 deliverable):"
cd $COL/a3_scratch/shadows/headwise_gate && patch -p1 --dry-run -i $COL/A3_spark_headwise_gate.diff; echo "rc=$?"
} >> $EV 2>&1

echo "=== evidence: $EV ($(wc -l < $EV) lines) ==="
