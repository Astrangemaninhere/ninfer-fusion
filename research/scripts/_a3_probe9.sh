#!/bin/bash
T=/home/user/ninfer-fusion
C=/mnt/c/Users/User/Documents/ziqinzhang/_collab
cd $T
echo "############ core/tensor.h location"
find $T -name 'tensor.h' -not -path '*/build/*' 2>/dev/null
echo "############ compile command for wrapper from compile_commands.json"
python3 - <<'PY'
import json
p="/home/user/ninfer-fusion/build/compile_commands.json"
try:
    db=json.load(open(p))
except Exception as e:
    print("ERR",e); raise SystemExit
for e in db:
    f=e.get("file","").replace("\\","/")
    if f.endswith("ops/wrapper/sigmoid_mul.cpp") or f.endswith("ops/launcher/sigmoid_gate_mul.cu") or f.endswith("ops/kernel/sigmoid_gate_mul.cuh"):
        print("FILE:",f)
        print("  cmd:",e["command"][:1500])
        print()
PY
echo "############ warnings flags anywhere in repo cmake"
grep -rn 'Werror\|Wall\|Wunused\|add_compile_options\|CMAKE_CXX_FLAGS' $T/CMakeLists.txt $T/src/CMakeLists.txt $T/tests/CMakeLists.txt 2>/dev/null | head
ls $T/cmake 2>/dev/null
echo "############ verify_exact / verify_guards / GuardedDeviceBuffer decls"
grep -n 'verify_exact\|verify_guards\|class GuardedDeviceBuffer\|struct GuardedDeviceBuffer\|GuardedDeviceBuffer(' $T/tests/ops/op_tester.h | head -20
echo "############ gqa landed? registered_kv_heads / Gqa16x4Geometry"
grep -n 'registered_kv_heads\|Gqa16x4Geometry' $T/src/ops/wrapper/gqa_attention.cpp $T/src/ops/kernel/gqa_attention_geometry.cuh | head -10
echo "############ gelu_mul landed? (public header)"
head -12 $T/include/ninfer/ops/gelu_mul.h
echo "############ test_silu_mul.cpp diff vs .orig (summary)"
diff <(cat $T/tests/ops/test_silu_mul.cpp) <(cat $T/tests/ops/test_silu_mul.cpp.orig) | head -20
echo "############ test_gelu_mul.cpp vs A3 generator GELU_TEST? (first lines)"
head -20 $T/tests/ops/test_gelu_mul.cpp
echo "############ CRLF audit for the 6 headwise files"
for f in src/ops/wrapper/sigmoid_mul.cpp src/ops/kernel/sigmoid_gate_mul.cuh src/ops/launcher/sigmoid_gate_mul.h src/ops/launcher/sigmoid_gate_mul.cu include/ninfer/ops/sigmoid_mul.h tests/ops/test_sigmoid_mul.cpp; do
  echo "$f CR=$(grep -c $'\r' $f || true)"
done
echo "############ tail bytes of wrapper (no trailing newline?)"
tail -c 60 src/ops/wrapper/sigmoid_mul.cpp | od -c | tail -5
echo "############ A3 headwise diff bytes: check for CR"
grep -c $'\r' $C/A3_spark_headwise_gate.diff || true
file $C/A3_spark_headwise_gate.diff
