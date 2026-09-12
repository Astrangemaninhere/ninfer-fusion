"""Debug: round-trip one NVFP4 row decode -> requant -> decode, expect small error."""
import json, struct, math, random

ART = '/home/user/models/qwen3_8_27b_nvfp4.ninfer'
f = open(ART, 'rb')
head = f.read(16)
jb = struct.unpack('<Q', head[8:16])[0]
d = json.loads(f.read(jb).decode('utf-8'))
objs = {o['name']: o for o in d['objects']}
payload = (16 + jb + 4095) // 4096 * 4096

g_bytes, go = read = (None, None)
o = objs['text/layers/0/mlp/gate_up']
f.seek(payload + o['offset'])
g_bytes = f.read(o['bytes'])
N, K = o['shape']
K2 = K // 2
code_bytes = N * K2
scale_off = (code_bytes + 255) // 256 * 256
div_off = scale_off + N * K // 16
divisor = struct.unpack('<f', g_bytes[div_off:div_off + 4])[0]
codes = g_bytes[:code_bytes]
scales = g_bytes[scale_off:scale_off + N * K // 16]
kt = K // 64

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
def e2m1_lo(b): return E2M1[b & 0x7] * (-1 if b & 0x8 else 1)
def e2m1_hi(b): return E2M1[(b >> 4) & 0x7] * (-1 if (b >> 4) & 0x8 else 1)

def e4m3_val(b):
    s = -1 if b & 0x80 else 1
    e = (b >> 3) & 0xF
    m = b & 0x7
    if e == 0:
        return s * (m / 8.0) * 2 ** -6
    if e == 15:
        return float('nan')
    return s * (1 + m / 8.0) * 2 ** (e - 7)

def quant_e4m3(v):
    if v == 0: return 0
    a = abs(v)
    best, bb = None, 0
    for b in range(128):
        x = e4m3_val(b)
        if x == x and (best is None or abs(x - a) < abs(best - a)):
            best, bb = x, b
    return bb

def quant_e2m1(v):
    s = 1 if v >= 0 else -1
    a = min(abs(v), 6.0)
    best, bb = None, 0
    for i, x in enumerate(E2M1):
        if best is None or abs(x - a) < abs(best - a):
            best, bb = x, i
    return bb | (8 if s < 0 else 0)

def decode_row(row):
    out = []
    for g in range(K // 16):
        ktile = g // 4
        lane = g % 4
        row_inner = row % 128
        row_mod32 = row_inner & 31
        row_q = row_inner >> 5
        so = (row // 128 * kt + ktile) * 512 + row_mod32 * 16 + row_q * 4 + lane
        sc = e4m3_val(scales[so]) / divisor
        base = row * K2
        for i in range(16):
            b = codes[base + g * 8 + i // 2]
            v = e2m1_lo(b) if i % 2 == 0 else e2m1_hi(b)
            out.append(v * sc)
    return out

def pack_row(vals):
    nc = bytearray(K2)
    ns = bytearray(K // 16)
    for g in range(K // 16):
        seg = vals[g * 16:(g + 1) * 16]
        mx = max(abs(v) for v in seg)
        if mx == 0:
            sc = 0
        else:
            sc = quant_e4m3(mx / 6.0)
        sv = e4m3_val(sc)
        ns[g] = sc
        for i in range(16):
            q = quant_e2m1(seg[i] / sv) if sv > 0 else 0
            idx = g * 8 + i // 2
            if i % 2 == 0:
                nc[idx] = (nc[idx] & 0xF0) | q
            else:
                nc[idx] = (nc[idx] & 0x0F) | (q << 4)
    # decode back with the same scale layout (write into a fake plane)
    out = []
    for g in range(K // 16):
        sv = e4m3_val(ns[g])
        for i in range(16):
            b = nc[g * 8 + i // 2]
            v = e2m1_lo(b) if i % 2 == 0 else e2m1_hi(b)
            out.append(v * sv)
    return out

# round trip on rows 0,1,127,128,129, 3000
for row in [0, 1, 63, 127, 128, 129, 3000, 17407]:
    W = decode_row(row)
    W2 = pack_row(W)
    # relative err
    num = sum((a - b) ** 2 for a, b in zip(W, W2))
    den = sum(a * a for a in W)
    print(f'row {row}: roundtrip rel err = {math.sqrt(num / den) * 100:.4f}%')
    # check nonzero scale distribution
    nz = sum(1 for v in W if v != 0)
    print(f'   nonzero: {nz}, max abs: {max(abs(v) for v in W):.4g}')
