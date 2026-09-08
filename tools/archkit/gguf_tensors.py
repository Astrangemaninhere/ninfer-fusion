# -*- coding: utf-8 -*-
"""gguf_tensors.py - tensor scan (proven parse from gguf_debug)."""
import struct
import sys
from collections import Counter

GGML_TYPES = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
              8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K',
              14: 'Q6_K', 15: 'Q8_K', 30: 'BF16'}


def scan(path):
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF'
    version, n_tensors, n_kv = struct.unpack('<IQQ', f.read(20))

    def read_str():
        n = struct.unpack('<Q', f.read(8))[0]
        return f.read(n).decode('utf-8', 'replace')

    def skip_val(t):
        if t == 0:
            f.read(1)
        elif t == 1:
            f.read(1)
        elif t in (2, 3):
            f.read(2)
        elif t in (4, 5, 6):
            f.read(4)
        elif t == 7:
            f.read(1)
        elif t == 8:
            read_str()
        elif t == 9:
            (atype, count) = struct.unpack('<IQ', f.read(12))
            for _ in range(count):
                skip_val(atype)
        elif t in (10, 11, 12):
            f.read(8)
        else:
            raise SystemExit('unknown kv type %d' % t)

    for _ in range(n_kv):
        read_str()
        (t,) = struct.unpack('<I', f.read(4))
        skip_val(t)
    tensors = []
    for _ in range(n_tensors):
        name = read_str()
        n_dims = struct.unpack('<I', f.read(4))[0]
        # GGUF v3: dims 为 u64
        dims = struct.unpack('<' + 'Q' * n_dims, f.read(8 * n_dims))
        (ttype,) = struct.unpack('<I', f.read(4))
        f.read(8)
        tensors.append((name, GGML_TYPES.get(ttype, 'T%d' % ttype), dims))
    f.close()
    return tensors


def main():
    for p in sys.argv[1:]:
        t = scan(p)
        hist = Counter(x[1] for x in t)
        print('== %s tensors=%d' % (p.split('/')[-1], len(t)))
        print('types:', dict(hist))
        for name, ty, dims in t[:16]:
            print('  %-58s %-6s %s' % (name, ty, dims))


if __name__ == '__main__':
    main()
