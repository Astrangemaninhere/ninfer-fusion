"""Verify every folded layer in foldmlp artifact: decode sample rows and compare
against baseline decode x gain(1+w). Reports per-layer relative error."""
import json, struct
import numpy as np

BASE = '/home/user/models/qwen3_8_27b_nvfp4.ninfer'
FOLD = '/home/user/models/qwen3_8_27b_nvfp4_foldmlp.ninfer'

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E4M3 = np.zeros(256)
for b in range(256):
    s = -1.0 if b & 0x80 else 1.0
    e = (b >> 3) & 0xF
    m = b & 0x7
    v = s * (m / 8.0) * 2.0 ** -6 if e == 0 else (float('nan') if e == 15 else s * (1 + m / 8.0) * 2.0 ** (e - 7))
    E4M3[b] = v

def bf16_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

def open_art(path):
    f = open(path, 'rb')
    h = f.read(16)
    jb = struct.unpack('<Q', h[8:16])[0]
    d = json.loads(f.read(jb).decode('utf-8'))
    payload = (16 + jb + 4095) // 4096 * 4096
    return f, {o['name']: o for o in d['objects']}, payload

fB, obB, plB = open_art(BASE)
fF, obF, plF = open_art(FOLD)

def decode_rows(f, objs, payload, name, rows, K):
    o = objs[name]
    f.seek(payload + o['offset'])
    gb = f.read(o['bytes'])
    N, K = o['shape']
    K2 = K // 2
    code_bytes = N * K2
    scale_off = (code_bytes + 255) // 256 * 256
    nscales = N * K // 16
    div_off = scale_off + nscales
    divisor = struct.unpack('<f', gb[div_off:div_off + 4])[0]
    codes = np.frombuffer(gb, dtype=np.uint8, count=code_bytes)
    scales = np.frombuffer(gb, dtype=np.uint8, count=nscales, offset=scale_off)
    kt = K // 64
    out = {}
    for row in rows:
        vals = []
        for g in range(K // 16):
            ktile = g // 4; lane = g % 4
            ri = row % 128; r32 = ri & 31; rq = ri >> 5
            so = (row // 128 * kt + ktile) * 512 + r32 * 16 + rq * 4 + lane
            sc = E4M3[scales[so]] / divisor
            base = row * K2
            for i in range(16):
                b = codes[base + g * 8 + i // 2]
                if i % 2 == 0:
                    v = E2M1[b & 0x07] * (-1 if b & 0x08 else 1)
                else:
                    v = E2M1[(b >> 4) & 0x07] * (-1 if b & 0x80 else 1)
                vals.append(v * sc)
        out[row] = vals
    return out, divisor

def norm_gain(f, objs, payload, name):
    o = objs[name]
    f.seek(payload + o['offset'])
    w16 = np.frombuffer(f.read(o['bytes']), dtype='<u2')
    return (1.0 + bf16_f32(w16)).astype(np.float64)

layers = sorted({o['name'].split('/mlp/gate_up')[0] for o in obF
                 if o['name'].endswith('/mlp/gate_up')})
bad = []
for L in layers:
    gu = L + '/mlp/gate_up'
    nm = L + '/post_attention_norm'
    if nm not in obB or gu not in obB:
        continue
    gain = norm_gain(fB, obB, plB, nm)
    rows = [0, 63, 127, 1024]
    dB, _ = decode_rows(fB, obB, plB, gu, rows, 5120)
    dF, _ = decode_rows(fF, obF, plF, gu, rows, 5120)
    errs = []
    for row in rows:
        Wb = np.array(dB[row]) * gain
        Wf = np.array(dF[row])
        num = np.sqrt(np.sum((Wf - Wb) ** 2))
        den = np.sqrt(np.sum(Wb ** 2))
        errs.append(num / max(den, 1e-9))
    e = max(errs)
    flag = 'OK ' if e < 0.10 else 'BAD'
    if e >= 0.10:
        bad.append((L, e))
    print(f'{flag} {L.split("/")[-1]:>4s} max rel err {e*100:6.2f}%')
print('BAD layers:', len(bad))
for L, e in bad[:10]:
    print(' ', L, f'{e*100:.1f}%')
