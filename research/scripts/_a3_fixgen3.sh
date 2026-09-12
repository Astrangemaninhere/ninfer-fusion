#!/bin/bash
# A3 headwise fix v3: judgment matrix on the SHIPPED predicate + patch-order fuzz test.
set -u
T=/home/user/ninfer-fusion
COL=/mnt/c/Users/User/Documents/ziqinzhang/_collab
L=$COL/a3_scratch/live
REL=src/ops/wrapper/sigmoid_mul.cpp
FIX=$COL/build/A3_headwise_gate_fix.diff
W=/tmp/a3fix3
rm -rf $W; mkdir -p $W/tree/src/ops/wrapper $W/tree2/src/ops/wrapper

echo "=== A: extract the shipped predicate verbatim from the patched file ==="
cp $L/$REL $W/tree/$REL && cd $W/tree && patch -p1 -i $FIX >/dev/null && cd $W
sed -n '/^bool headwise_gate_shape/,/^}/p' $W/tree/$REL | tee $W/pred.txt
echo "--- extracted function md5 ---"; md5sum $W/pred.txt

echo
echo "=== B: mechanical judgment matrix (Tensor renamed to Shape; expression byte-identical) ==="
python3 - <<'PY'
import pathlib, subprocess
W = pathlib.Path("/tmp/a3fix3")
pred = (W / "pred.txt").read_text(encoding="utf-8").replace("const Tensor&", "const Shape&")
prog = r'''
#include <cstdint>
#include <cstdio>
#include <vector>
struct Shape { std::int32_t ne[4]; };
''' + pred + r'''
struct Case { const char* label; std::int32_t g[4]; std::int32_t x[4]; bool want; int route; };
int main() {
    const Case cases[] = {
        // ---- headwise-accepted (one scalar per head/token, broadcast over head_dim) ----
        {"headwise 16x256x1  gate[16,1]      x[256,16,1]", {16,1,1,1},     {256,16,1,1},   true,  0},
        {"headwise 16x256x6  gate[16,6]      x[256,16,6]", {16,6,1,1},     {256,16,6,1},   true,  0},
        {"headwise 16x256x12 gate[16,12]     x[256,16,12]",{16,12,1,1},    {256,16,12,1},  true,  0},
        {"headwise 32x128x3  gate[32,3]      x[128,32,3]", {32,3,1,1},     {128,32,3,1},   true,  0},
        {"headwise 4x256x5   gate[4,5]       x[256,4,5]",  {4,5,1,1},      {256,4,5,1},    true,  0},
        {"headwise single    gate[1]         x[1,1,1]",    {1,1,1,1},      {1,1,1,1},      true,  0},
        // ---- rejected: shapes the per-element route owns (or plain mismatches) ----
        {"per-elem [6144,1]  gate==x",                     {6144,1,1,1},   {6144,1,1,1},   false, 1},
        {"per-elem [6144,48] gate==x",                     {6144,48,1,1},  {6144,48,1,1},  false, 1},
        {"per-elem [4096,17] gate==x",                     {4096,17,1,1},  {4096,17,1,1},  false, 1},
        {"per-elem [4096,128] gate==x",                    {4096,128,1,1}, {4096,128,1,1}, false, 1},
        {"per-elem square [256,256] (A3 test 501u)",       {256,256,1,1},  {256,256,1,1},  false, 1},
        {"per-elem 1-D edge case [7]",                     {7,1,1,1},      {7,1,1,1},      false, 1},
        {"per-elem [256,256,4] 3-D equal shapes",          {256,256,4,1},  {256,256,4,1},  false, 1},
        // ---- live-caller gate shapes ([head_dim, n_q, T]) must stay per-element ----
        {"live 27b gate[256,24,T] x[256,24,T]",            {256,24,8,1},   {256,24,8,1},   false, 1},
        {"live 35b gate[256,16,T] x[256,16,T]",            {256,16,8,1},   {256,16,8,1},   false, 1},
        {"live muse gate[128,32,T] x[128,32,T]",           {128,32,8,1},   {128,32,8,1},   false, 1},
        {"gate[256,24,T] vs x[256,16,T] (live mismatch)",  {256,24,8,1},   {256,16,8,1},   false, 1},
        // ---- near-miss headwise shapes ----
        {"T in last axis x[256,16,1,6] (not [D,H,T])",     {16,1,1,1},     {256,16,1,6},   false, 1},
        {"gate not [H,T]: gate[16,6,2] x[256,16,6]",       {16,6,2,1},     {256,16,6,1},   false, 1},
        {"gate[8,12] x[256,16,12] (H mismatch)",           {8,12,1,1},     {256,16,12,1},  false, 1},
        {"gate[16,7] x[256,16,12] (T mismatch)",           {16,7,1,1},     {256,16,12,1},  false, 1},
        {"empty x[0,16,12] gate[16,12]",                   {16,12,1,1},    {0,16,12,1},    false, 1},
    };
    int bad = 0;
    for (const Case& c : cases) {
        const Shape g{c.g[0],c.g[1],c.g[2],c.g[3]}, x{c.x[0],c.x[1],c.x[2],c.x[3]};
        const bool got = headwise_gate_shape(g, x);
        const bool ok = (got == c.want);
        if (!ok) ++bad;
        std::printf("%-52s gate[%d,%d,%d,%d] x[%d,%d,%d,%d] -> %-5s (want %-5s) %s\n",
                    c.label, c.g[0],c.g[1],c.g[2],c.g[3], c.x[0],c.x[1],c.x[2],c.x[3],
                    got ? "TRUE" : "false", c.want ? "TRUE" : "false", ok ? "ok" : "MISMATCH");
    }
    std::printf("MISMATCHES=%d / %zu cases\n", bad, sizeof(cases)/sizeof(cases[0]));
    return bad != 0;
}
'''
(W / "matrix.cpp").write_text(prog, encoding="utf-8")
r = subprocess.run(["/usr/bin/c++", "-std=gnu++20", "-O0", "-o", str(W / "matrix"), str(W / "matrix.cpp")],
                   capture_output=True, text=True)
print(r.stdout + r.stderr, end="")
if r.returncode != 0:
    raise SystemExit("BUILD FAILED")
r = subprocess.run([str(W / "matrix")], capture_output=True, text=True)
print(r.stdout + r.stderr, end="")
print("MATRIX RC=%d" % r.returncode)
PY

echo
echo "=== C: patch-order fuzz check (fix-first, then the original A3 diff at --fuzz=0) ==="
cp -r $L/* $W/tree2/ 2>/dev/null
cd $W/tree2 && patch -p1 -i $FIX >/dev/null
cd $W/tree2 && patch -p1 --dry-run --fuzz=0 -i $COL/A3_spark_headwise_gate.diff; echo "RC=$?"
echo "--- and the reverse order at --fuzz=0 (A3 first, then fix) ---"
cp -r $COL/a3_scratch/b/* $W/tree2b 2>/dev/null || { mkdir -p $W/tree2b; cp -r $COL/a3_scratch/b/* $W/tree2b/; }
cd $W/tree2b && patch -p1 --dry-run --fuzz=0 -i $FIX; echo "RC=$?"

echo
echo "=== D: host compile check of the A3-applied TEST TU (host-compiled test) ==="
python3 - <<'PY'
import json
db=json.load(open("/home/user/ninfer-fusion/build/compile_commands.json"))
hit=[e for e in db if e["file"].endswith("tests/ops/test_sigmoid_mul.cpp")]
print("test TU compile command:" if hit else "NO ENTRY for test_sigmoid_mul.cpp")
for e in hit: print("  ", e["command"][:900])
PY
echo
echo done
