#!/bin/bash
echo "=== E6 patch (cat -A, 14 lines) ==="
head -14 /mnt/c/Users/User/Documents/ziqinzhang/_collab/E6_s48_dspark_verify_pos.diff | cat -A
echo
echo "=== line-ending census ==="
python3 - <<'PY'
import io
base = '/mnt/c/Users/User/Documents/ziqinzhang/'
for name in ['_collab/E6_s48_dspark_verify_pos.diff',
             '_collab/A5b_attention_valid_width.diff',
             '_collab/build/T3_df2_position_probe.diff']:
    b = io.open(base + name, 'rb').read()
    lines = b.split(b'\n')
    body = lines[:-1] if lines and lines[-1] == b'' else lines
    cr = sum(1 for l in body if l.endswith(b'\r'))
    print('%-46s lines=%-4d crlf=%-4d lf_only=%d' % (name, len(body), cr, len(body) - cr))
    # which lines are LF-only
    lf = [i for i, l in enumerate(body) if not l.endswith(b'\r')]
    print('   LF-only line indexes:', lf[:12])
PY
echo
echo "=== does the tree file have any LF-only lines in 12425..12440? ==="
sed -n '12425,12435p' /home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/program_impl.h | cat -A | head -12
