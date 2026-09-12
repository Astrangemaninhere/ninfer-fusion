"""Parallel folding artifact builder. Worker computes per-layer folded bytes
to temp files; main process streams the output in object order.
Usage: python3 _fold_art_par.py SRC DST [--limit N] [--jobs J]
"""
import json, struct, sys, time, os, tempfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np

SRC = sys.argv[1]
DST = sys.argv[2]
LIMIT = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
JOBS = int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else 8

# module-level tables for workers
E2M1 = None
E4M3 = None
TAB = None
E2M1_F = None
LO_VAL = None
HI_VAL = None
PAYLOAD = 0

def _init_tables():
    global E2M1, E4M3, TAB, E2M1_F, LO_VAL, HI_VAL
    E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    E4M3 = np.zeros(256)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        v = s * (m / 8.0) * 2.0 ** -6 if e == 0 else (float('nan') if e == 15 else s * (1 + m / 8.0) * 2.0 ** (e - 7))
        E4M3[b] = v
    TAB = np.zeros(8192, dtype=np.uint8)
    for i in range(8192):
        v = min(i / 8192.0 * 512.0, 448.0)
        TAB[i] = int(np.nanargmin(np.abs(E4M3[:128] - v)))
    E2M1_F = E2M1.astype(np.float32)
    lo_s = np.where((np.arange(256) & 0x08) != 0, -1.0, 1.0)
    hi_s = np.where((np.arange(256) & 0x80) != 0, -1.0, 1.0)
    LO_VAL = (E2M1[np.arange(256) & 0x07] * lo_s).astype(np.float32)
    HI_VAL = (E2M1[(np.arange(256) >> 4) & 0x07] * hi_s).astype(np.float32)

def q_e4m3(vals):
    return TAB[np.clip((vals / 512.0 * 8192.0).astype(np.int64), 0, 8191)]

def q_e2m1(vals):
    flat = vals.reshape(-1).astype(np.float32)
    s = np.where(flat < 0, 8, 0).astype(np.uint8)
    a = np.minimum(np.abs(flat), 6.0)
    idx = np.argmin(np.abs(E2M1_F[None, :] - a[:, None]), axis=1).astype(np.uint8)
    return (idx | s).astype(np.uint8).reshape(vals.shape)

def bf16_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

def fold_one(args):
    """args: (norm_name, gu_name, norm_offset, norm_bytes, gu_offset, gu_shape, tmp_out)"""
    norm_name, gu_name, noff, nbytes, goff, gshape, tmp_out = args
    _init_tables()
    N, K = gshape
    K2 = K // 2
    code_bytes = N * K2
    scale_off = (code_bytes + 255) // 256 * 256
    nscales = N * K // 16
    div_off = scale_off + nscales
    with open(SRC, 'rb') as f:
        f.seek(PAYLOAD + noff)
        w16 = np.frombuffer(f.read(nbytes), dtype='<u2')
        gain = (1.0 + bf16_f32(w16)).astype(np.float32)
        f.seek(PAYLOAD + goff)
        gb = f.read(code_bytes + nscales + 4)
    divisor = struct.unpack('<f', gb[div_off:div_off + 4])[0]
    codes = np.frombuffer(gb, dtype=np.uint8, count=code_bytes).copy()
    scales = np.frombuffer(gb, dtype=np.uint8, count=nscales, offset=scale_off).copy()
    kt = K // 64
    g = np.arange(K // 16)
    ktile = g // 4
    lane = g % 4
    CH = 4096
    for r0 in range(0, N, CH):
        r1 = min(r0 + CH, N)
        row_idx = np.arange(r0, r1)
        r32 = row_idx % 32
        rq = (row_idx % 128) >> 5
        sc_pos = ((row_idx[:, None] // 128) * kt + ktile[None, :]) * 512 + r32[:, None] * 16 + rq[:, None] * 4 + lane[None, :]
        scv = (E4M3[scales[sc_pos]] / divisor).astype(np.float32)
        cb = codes[r0 * K2:r1 * K2].reshape(-1, K2)
        lo = LO_VAL[cb]
        hi = HI_VAL[cb]
        sc_full = np.repeat(scv, 16, axis=1)
        W = np.empty((r1 - r0, K), dtype=np.float32)
        W[:, 0::2] = lo * sc_full[:, 0::2]
        W[:, 1::2] = hi * sc_full[:, 1::2]
        Wf = W * gain[None, :]
        gmax = np.abs(Wf).reshape(-1, K // 16, 16).max(axis=2)
        scb = q_e4m3(gmax * divisor / 6.0)
        sc_dec = E4M3[scb].astype(np.float32)
        sc_dec_full = np.repeat(sc_dec, 16, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(sc_dec_full > 0, Wf * divisor / sc_dec_full, 0.0)
        cn = q_e2m1(ratio)
        packed = ((cn[:, 1::2].astype(np.uint16) << 4) | cn[:, 0::2].astype(np.uint16)).astype(np.uint8)
        codes[r0 * K2:r1 * K2] = packed.reshape(-1)
        scales[sc_pos] = scb
    out = bytearray(gb)
    out[:code_bytes] = codes.tobytes()
    out[scale_off:scale_off + nscales] = scales.tobytes()
    with open(tmp_out, 'wb') as f:
        f.write(bytes(out))
    return len(out)

def main():
    global PAYLOAD
    f = open(SRC, 'rb')
    head = f.read(16)
    assert head[:8] == b'NINFER\x00\x02'
    jb = struct.unpack('<Q', head[8:16])[0]
    directory = json.loads(f.read(jb).decode('utf-8'))
    objs = directory['objects']
    by_name = {o['name']: o for o in objs}
    PAYLOAD = (16 + jb + 4095) // 4096 * 4096
    layers = sorted({o['name'].split('/mlp/gate_up')[0] for o in objs
                     if o['name'].endswith('/mlp/gate_up')})
    pairs = [(L, L + '/post_attention_norm', L + '/mlp/gate_up') for L in layers
             if L + '/post_attention_norm' in by_name and L + '/mlp/gate_up' in by_name]
    if LIMIT:
        pairs = pairs[:LIMIT]
    print(f'pairs: {len(pairs)}, jobs: {JOBS}', flush=True)

    tmpdir = tempfile.mkdtemp(prefix='fold_')
    args = []
    for L, norm, gu in pairs:
        no = by_name[norm]
        go = by_name[gu]
        tmp = os.path.join(tmpdir, L.replace('/', '_') + '.bin')
        args.append((norm, gu, no['offset'], no['bytes'], go['offset'], go['shape'], tmp))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=JOBS) as ex:
        sizes = list(ex.map(fold_one, args))
    print(f'compute done in {time.time()-t0:.0f}s', flush=True)

    tmp_by_name = {os.path.basename(a[6]).replace('.bin', ''): a[6] for a in args}
    # map layer path -> tmp
    tmp_map = {}
    for (L, norm, gu), a in zip(pairs, args):
        tmp_map[gu] = a[6]
        tmp_map[norm] = None  # zero

    t1 = time.time()
    with open(DST, 'wb') as out:
        f.seek(0)
        out.write(f.read(PAYLOAD))
        objs_sorted = sorted(objs, key=lambda x: x['offset'])
        cursor = PAYLOAD
        for o in objs_sorted:
            off, size, name = o['offset'], o['bytes'], o['name']
            if off > cursor:
                f.seek(cursor)
                out.write(f.read(off - cursor))
            cursor = off + size
            if name in tmp_map:
                if tmp_map[name] is None:
                    out.write(bytes(size))  # zeroed norm
                else:
                    with open(tmp_map[name], 'rb') as tf:
                        out.write(tf.read())
                continue
            f.seek(PAYLOAD + off)
            out.write(f.read(size))
        f.seek(cursor)
        out.write(f.read())
    print(f'write done in {time.time()-t1:.0f}s, total {time.time()-t0:.0f}s -> {DST}')

if __name__ == '__main__':
    main()
