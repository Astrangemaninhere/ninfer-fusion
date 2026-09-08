import glob
import struct

ROWS = 5 * 5120
HIDDEN = 5120
hits = tot = 0
for p in sorted(glob.glob('/tmp/vdump3/chunk_*.bin')):
    with open(p, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        (tokens,) = struct.unpack('<i', f.read(4))
        ids = list(struct.unpack('<%di' % tokens, f.read(4 * tokens)))
        f.read(ROWS * tokens * 2)
        f.read(HIDDEN * tokens * 2)
        rest = f.read()
    top1 = list(struct.unpack('<%di' % tokens, rest[:4 * tokens])) \
        if len(rest) >= 4 * tokens else None
    if top1 is None:
        print(p, 'NO TOP1')
        continue
    h = sum(1 for i in range(tokens - 1) if top1[i] == ids[i + 1])
    hits += h
    tot += tokens - 1
    print('%s tok=%d hit=%d (%.2f)' % (p.split('/')[-1], tokens, h,
                                       h / max(1, tokens - 1)))
print('TOTAL %d/%d = %.3f' % (hits, tot, hits / max(1, tot)))
