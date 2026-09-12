"""Corrected: requant must match upstream scale semantics:
stored scale = e4m3(max_abs * divisor / 6); decoded scale value = e4m3/divisor.
code = e2m1(v / (scale_decoded)) = e2m1(v * divisor / e4m3_val(scale_byte))
"""
import json, struct, math, random, statistics

ART = '/home/user/models/qwen3_8_27b_nvfp4.ninfer'
f = open(ART, 'rb')
head = f.read(16)
jb = struct.unpack('<Q', head[8:16])[0]
d = json.loads(f.read(jb).decode('utf-8'))
objs = {o['name']: o for o in d['objects']}
payload = (16 + jb + 4095) // 4096 * 4096

def load(name):
    o = objs[name]
    f.seek(payload + o['offset'])
    return f.read(o['bytes']), o['shape']

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
def e2m1_lo(b): return E2M1[b & 0x7] * (-1 if b & 0x8 else 1)
def e2m1_hi(b): return E2M1[(b >> 4) & 0x7] * (-1 if (b >> 4) & 0x8 else 1)
def e4m3_val(b):
    s = -1 if b & 0x80 else 1
    e = (b >> 3) & 0xF
    m = b & 0x7
    if e == 0: return s * (m / 8.0) * 2 ** -6
    if e == 15: return float('nan')
    return s * (1 + m / 8.0) * 2 ** (e - 7)
def quant_e4m3(v):
    # e4m3 satfinite: clamp to 448
    a = min(abs(v), 448.0)
    best, bb = None, 0
    for b in range(128):
        x = e4m3_val(b)
        if x == x and (best is None or abs(x - a) < abs(best - a)):
            best, bb = x, b
    return bb
def quant_e2m1(v):
    # e2m1 satfinite clamp 6
    s = 1 if v >= 0 else -1
    a = min(abs(v), 6.0)
    best, bb = None, 0
    for i, x in enumerate(E2M1):
        if best is None or abs(x - a) < abs(best - a):
            best, bb = x, i
    return bb | (8 if s < 0 else 0)

g_bytes, shape = load('text/layers/0/mlp/gate_up')
N, K = shape
K2 = K // 2
code_bytes = N * K2
scale_off = (code_bytes + 255) // 256 * 256
div_off = scale_off + N * K // 16
divisor = struct.unpack('<f', g_bytes[div_off:div_off + 4])[0]
codes = g_bytes[:code_bytes]
scales = g_bytes[scale_off:scale_off + N * K // 16]
kt = K // 64
print(f'divisor = {divisor}')

def decode_row(row):
    out = []
    for g in range(K // 16):
        ktile = g // 4; lane = g % 4
        ri = row % 128; r32 = ri & 31; rq = ri >> 5
        so = (row // 128 * kt + ktile) * 512 + r32 * 16 + rq * 4 + lane
        sc = e4m3_val(scales[so]) / divisor
        base = row * K2
        for i in range(16):
            b = codes[base + g * 8 + i // 2]
            v = e2m1_lo(b) if i % 2 == 0 else e2m1_hi(b)
            out.append(v * sc)
    return out

def pack_row(vals, divisor):
    """Requantize decoded values (already /divisor applied). Store scale as
    e4m3(max_abs*divisor/6); codes e2m1(v*divisor/scale_e4m3)."""
    nc = bytearray(K2)
    ns = bytearray(K // 16)
    for g in range(K // 16):
        seg = vals[g * 16:(g + 1) * 16]
        mx = max(abs(v) for v in seg)
        sc = quant_e4m3(mx * divisor / 6.0) if mx > 0 else 0
        sv = e4m3_val(sc)  # e4m3 decoded
        ns[g] = sc
        for i in range(16):
            q = quant_e2m1(seg[i] * divisor / sv) if sv > 0 else 0
            idx = g * 8 + i // 2
            if i % 2 == 0: nc[idx] = (nc[idx] & 0xF0) | q
            else: nc[idx] = (nc[idx] & 0x0F) | (q << 4)
    out = []
    for g in range(K // 16):
        sv = e4m3_val(ns[g]) / divisor
        for i in range(16):
            b = nc[g * 8 + i // 2]
            v = e2m1_lo(b) if i % 2 == 0 else e2m1_hi(b)
            out.append(v * sv)
    return out

# round-trip check
for row in [0, 1, 63, 127, 128, 2000, 17407]:
    W = decode_row(row)
    W2 = pack_row(W, divisor)
    num = sum((a - b) ** 2 for a, b in zip(W, W2))
    den = sum(a * a for a in W)
    print(f'row {row}: roundtrip rel err = {math.sqrt(num / den) * 100:.4f}%')

# gain fold test with proper metric: weight-space relative RMS after fold requant
wb, _ = load('text/layers/0/post_attention_norm')
wv = struct.unpack('<%dH' % (len(wb) // 2), wb)
gain = [1 + struct.unpack('<f', struct.pack('<I', x << 16))[0] for x in wv]

errs = []
for row in range(0, 512):
    W = decode_row(row)
    Wg = [W[i] * gain[i] for i in range(K)]
    Wq = pack_row(Wg, divisor)
    num = sum((a - b) ** 2 for a, b in zip(Wq, Wg))
    den = sum(a * a for a in Wg)
    errs.append(math.sqrt(num / den))
print(f'fold requant weight-space rel RMS: mean={statistics.mean(errs)*100:.3f}% max={max(errs)*100:.3f}%')
