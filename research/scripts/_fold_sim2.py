"""Folding error: compare (a) original W decode, (b) requant(W*(1+w)) decode.
Output gemv error measured with random activations, baseline = ideal fp32 W*(1+w)*x.
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

def prepare(shape):
    N, K = shape
    g_bytes, _ = load('text/layers/0/mlp/gate_up')
    K2 = K // 2
    code_bytes = N * K2
    scale_off = (code_bytes + 255) // 256 * 256
    div_off = scale_off + N * K // 16
    divisor = struct.unpack('<f', g_bytes[div_off:div_off + 4])[0]
    return g_bytes, N, K, divisor

def decode_row(codes, scales, divisor, row, K, kt):
    out = []
    for g in range(K // 16):
        ktile = g // 4; lane = g % 4
        ri = row % 128; r32 = ri & 31; rq = ri >> 5
        so = (row // 128 * kt + ktile) * 512 + r32 * 16 + rq * 4 + lane
        sc = e4m3_val(scales[so]) / divisor
        base = row * (K // 2)
        for i in range(16):
            b = codes[base + g * 8 + i // 2]
            v = e2m1_lo(b) if i % 2 == 0 else e2m1_hi(b)
            out.append(v * sc)
    return out

def pack_row(vals, K):
    K2 = K // 2
    nc = bytearray(K2)
    ns = bytearray(K // 16)
    for g in range(K // 16):
        seg = vals[g * 16:(g + 1) * 16]
        mx = max(abs(v) for v in seg)
        sc = quant_e4m3(mx / 6.0) if mx > 0 else 0
        sv = e4m3_val(sc)
        ns[g] = sc
        for i in range(16):
            q = quant_e2m1(seg[i] / sv) if sv > 0 else 0
            idx = g * 8 + i // 2
            if i % 2 == 0: nc[idx] = (nc[idx] & 0xF0) | q
            else: nc[idx] = (nc[idx] & 0x0F) | (q << 4)
    out = []
    for g in range(K // 16):
        sv = e4m3_val(ns[g])
        for i in range(16):
            b = nc[g * 8 + i // 2]
            v = e2m1_lo(b) if i % 2 == 0 else e2m1_hi(b)
            out.append(v * sv)
    return out

# norm weight (bf16 -> float), gain = 1 + w
wb, _ = load('text/layers/0/post_attention_norm')
wv = struct.unpack('<%dH' % (len(wb) // 2), wb)
gain = [1 + struct.unpack('<e', struct.pack('<H', x))[0] for x in wv]

g_bytes, N, K, divisor = prepare((34816, 5120))
K2 = K // 2
scale_off = (N * K2 + 255) // 256 * 256
codes = g_bytes[:N * K2]
scales = g_bytes[scale_off:scale_off + N * K // 16]
kt = K // 64

random.seed(11)
x = [random.gauss(0, 1) for _ in range(K)]

rows = list(range(0, 256))  # 2 m_tiles
e_nofold = []   # original quant error vs ideal (W*g*x with fp32 W? no - vs decoded W * g * x)
e_fold = []     # fold error: requant(W*g) vs decoded-W*g  -> gemv diff / ideal magnitude
for row in rows:
    W = decode_row(codes, scales, divisor, row, K, kt)      # decoded original
    ideal = sum(W[i] * x[i] * gain[i] for i in range(K))     # reference (orig quant, ideal gain)
    # no-fold: nothing changes - reference IS no-fold path. Compare fold path:
    Wg = [W[i] * gain[i] for i in range(K)]
    Wq = pack_row(Wg, K)
    fold = sum(Wq[i] * x[i] for i in range(K))
    e_fold.append(abs(fold - ideal) / max(abs(ideal), 1e-9))
    # also: what is the error of requantizing W WITHOUT gain (pure requant cost)?
    Wq0 = pack_row(W, K)
    base = sum(Wq0[i] * x[i] * gain[i] for i in range(K))
    e_nofold.append(abs(base - ideal) / max(abs(ideal), 1e-9))

print(f'rows 0..255, gemv rel error vs reference (decoded W x gain):')
print(f'  requant-no-gain (pure requant): mean={statistics.mean(e_nofold)*100:.3f}%  p95={sorted(e_nofold)[int(len(e_nofold)*0.95)]*100:.3f}%')
print(f'  fold requant (W*(1+w)):         mean={statistics.mean(e_fold)*100:.3f}%  p95={sorted(e_fold)[int(len(e_fold)*0.95)]*100:.3f}%')
print(f'  gain range: min={min(gain):.4f} max={max(gain):.4f} mean={statistics.mean(gain):.4f}')
