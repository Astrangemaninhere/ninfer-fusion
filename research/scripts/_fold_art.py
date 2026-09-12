"""Folding experiment artifact builder (clean).
For each fold pair (post_attention_norm -> mlp/gate_up):
  - decode gate_up rows, multiply by gain=1+w, requant (e4m3 scale + e2m1 codes)
  - zero the norm weight so rmsnorm(unit_offset=true) yields pure x*inv_r
Usage: python3 _fold_art.py SRC DST [--limit N]
"""
import json, struct, sys, time
import numpy as np

SRC = sys.argv[1]
DST = sys.argv[2]
LIMIT = None
if '--limit' in sys.argv:
    LIMIT = int(sys.argv[sys.argv.index('--limit') + 1])

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E4M3 = np.zeros(256)
for b in range(256):
    s = -1.0 if b & 0x80 else 1.0
    e = (b >> 3) & 0xF
    m = b & 0x7
    v = s * (m / 8.0) * 2.0 ** -6 if e == 0 else (float('nan') if e == 15 else s * (1 + m / 8.0) * 2.0 ** (e - 7))
    E4M3[b] = v
# e4m3 nearest code table over [0, 512)
TAB = np.zeros(8192, dtype=np.uint8)
for i in range(8192):
    v = min(i / 8192.0 * 512.0, 448.0)
    TAB[i] = int(np.nanargmin(np.abs(E4M3[:128] - v)))

def q_e4m3(vals):
    return TAB[np.clip((vals / 512.0 * 8192).astype(np.int64), 0, 8191)]

def q_e2m1(vals):
    s = np.where(vals < 0, 8, 0).astype(np.uint8)
    a = np.minimum(np.abs(vals), 6.0)
    idx = np.argmin(np.abs(E2M1[None, :] - a[:, None]), axis=1).astype(np.uint8)
    return (idx | s).astype(np.uint8)

def bf16_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

f = open(SRC, 'rb')
head = f.read(16)
assert head[:8] == b'NINFER\x00\x02'
jb = struct.unpack('<Q', head[8:16])[0]
directory = json.loads(f.read(jb).decode('utf-8'))
objs = directory['objects']
by_name = {o['name']: o for o in objs}
payload = (16 + jb + 4095) // 4096 * 4096

layers = sorted({o['name'].split('/mlp/gate_up')[0] for o in objs
                 if o['name'].endswith('/mlp/gate_up')})
pairs = [(L, L + '/post_attention_norm', L + '/mlp/gate_up') for L in layers
         if L + '/post_attention_norm' in by_name and L + '/mlp/gate_up' in by_name]
if LIMIT:
    pairs = pairs[:LIMIT]
print(f'pairs: {len(pairs)}')

def fold_layer(norm_obj, gu_obj):
    f.seek(payload + norm_obj['offset'])
    w16 = np.frombuffer(f.read(norm_obj['bytes']), dtype='<u2')
    gain = 1.0 + bf16_f32(w16).astype(np.float64)
    f.seek(payload + gu_obj['offset'])
    gb = f.read(gu_obj['bytes'])
    N, K = gu_obj['shape']
    K2 = K // 2
    code_bytes = N * K2
    scale_off = (code_bytes + 255) // 256 * 256
    nscales = N * K // 16
    div_off = scale_off + nscales
    divisor = struct.unpack('<f', gb[div_off:div_off + 4])[0]
    codes = np.frombuffer(gb, dtype=np.uint8, count=code_bytes).copy()
    scales = np.frombuffer(gb, dtype=np.uint8, count=nscales, offset=scale_off).copy()
    kt = K // 64
    gain64 = gain.astype(np.float64)
    lo_sign = np.where((np.arange(256) & 0x08) != 0, -1.0, 1.0)
    hi_sign = np.where((np.arange(256) & 0x80) != 0, -1.0, 1.0)
    for r0 in range(0, N, 128):
        r1 = min(r0 + 128, N)
        nrow = r1 - r0
        # scales per (row, group) with swizzle
        row_idx = np.arange(r0, r1)
        r32 = row_idx % 32
        rq = (row_idx % 128) >> 5
        ktile = np.arange(K // 16) // 4
        lane = np.arange(K // 16) % 4
        sc_pos = ((row_idx[:, None] // 128) * kt + ktile[None, :]) * 512 + r32[:, None] * 16 + rq[:, None] * 4 + lane[None, :]
        scv = E4M3[scales[sc_pos]] / divisor  # [nrow, groups]
        cb = codes[r0 * K2:r1 * K2].reshape(nrow, K2)
        lo = E2M1[cb & 0x07] * lo_sign[cb & 0x0F]
        hi = E2M1[(cb >> 4) & 0x07] * hi_sign[(cb >> 4) & 0x0F]
        sc_full = np.repeat(scv, 16, axis=1)  # [nrow, K]
        W = np.empty((nrow, K))
        W[:, 0::2] = lo * sc_full[:, 0::2]
        W[:, 1::2] = hi * sc_full[:, 1::2]
        Wf = W * gain64[None, :]
        gmax = np.abs(Wf).reshape(nrow, K // 16, 16).max(axis=2)  # [nrow, groups]
        scb = q_e4m3(gmax * divisor / 6.0)
        sc_dec = E4M3[scb]
        sc_dec_full = np.repeat(sc_dec, 16, axis=1)
        cn = q_e2m1(Wf * divisor / sc_dec_full)
        packed = ((cn[:, 1::2].astype(np.uint16) << 4) | cn[:, 0::2].astype(np.uint16)).astype(np.uint8)
        codes[r0 * K2:r1 * K2] = packed.reshape(-1)
        scales[sc_pos] = scb
    out = bytearray(gb)
    out[:code_bytes] = codes.tobytes()
    out[scale_off:scale_off + nscales] = scales.tobytes()
    return bytes(out)

t0 = time.time()
with open(DST, 'wb') as out:
    f.seek(0)
    out.write(f.read(payload))  # header
    objs_sorted = sorted(objs, key=lambda x: x['offset'])
    cursor = payload
    nfold = 0
    for o in objs_sorted:
        off, size, name = o['offset'], o['bytes'], o['name']
        if off > cursor:
            f.seek(cursor)
            out.write(f.read(off - cursor))
        cursor = off + size
        hit = None
        for L, norm, gu in pairs:
            if name == gu:
                hit = ('gu', L, norm, gu)
                break
        if name in {norm for _, norm, _ in pairs}:
            # zero the norm (gain=1 under unit_offset)
            f.seek(payload + off)
            f.read(size)  # skip
            out.write(bytes(size))
            continue
        if hit:
            no = by_name[hit[2]]
            guo = by_name[hit[3]]
            nb = fold_layer(no, guo)
            out.write(nb)
            nfold += 1
            print(f'  folded {hit[1].split("/")[-1]} ({time.time()-t0:.0f}s)', flush=True)
            continue
        f.seek(payload + off)
        out.write(f.read(size))
    f.seek(cursor)
    tail = f.read()
    out.write(tail)
print(f'done: {nfold} weights folded in {time.time()-t0:.0f}s -> {DST}')
