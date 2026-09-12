#!/bin/bash
cd /home/user/ninfer-fusion || exit 1
P=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build/T3_df2_position_probe.diff
echo "=== patch -p1 --dry-run ==="
patch -p1 --dry-run < "$P"; echo "rc=$?"
echo
echo "=== md5 unchanged? ==="
md5sum src/targets/qwen3_6/impl/runtime/program_impl.h
echo
echo "=== apply to a copy (not the tree) + bracket balance ==="
rm -rf /tmp/t3apply && mkdir -p /tmp/t3apply
cp src/targets/qwen3_6/impl/runtime/program_impl.h /tmp/t3apply/
cd /tmp/t3apply && patch -p1 < "$P" >/dev/null && echo "apply_rc=0"
python3 - <<'PY'
import io
p='/tmp/t3apply/src/targets/qwen3_6/impl/runtime/program_impl.h'
s=io.open(p,'rb').read().decode('utf-8')
for ch in '{}()[]':
    pass
pairs={'{':'}','(' :')','[':']'}
for a,b in pairs.items():
    print('%-3s %d / %d' % (a+b, s.count(a), s.count(b)))
print('probe lines:', s.count('[df2dbg]'))
print('crlf ok:', b'\r\n' in io.open(p,'rb').read())
PY
echo
echo "=== head of the diff (first 12 lines, cat -A style) ==="
head -12 "$P" | cat -A | head -14
