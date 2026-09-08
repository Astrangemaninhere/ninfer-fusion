#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kv_iso_ref.py — ISO4/ISO3 KV 量化 Python 参考 (与 src/ops/kv/iso_codec.h
逐位对拍)。取整语义必须与 C++ lround 一致: 远离零的四舍五入 (不能用
python round() 的银行家舍入)。跑完与 iso_codec_test 输出逐字节比对。"""

import struct

V4 = [0.5, -0.5, 1.0, -1.0, 3.0, -3.0, 7.0, -7.0,
      6.5, 0.0, 2.0, -2.0, 4.5, -4.5, 1.5, -1.5]
V3 = [0.5, -0.5, 1.0, -1.0, 3.0, -3.0, 0.0, 2.5]


def lround(x: float) -> int:
    """C++ std::lround 语义: 远离零取整。"""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def f32_to_f16_bits(x: float) -> int:
    b = struct.unpack('<I', struct.pack('<f', x))[0]
    sign = (b >> 16) & 0x8000
    exp = ((b >> 23) & 0xff) - 127 + 15
    mant = (b >> 13) & 0x3ff
    if exp >= 31:
        return sign | 0x7c00
    if exp <= 0:
        if exp < -10:
            return sign
        m = mant | 0x400
        return sign | (m >> (14 - exp))
    return sign | (exp << 10) | mant


def iso4(v):
    mx = max(abs(x) for x in v)
    s = mx / 7.0
    ss = s if s > 0 else 1.0
    codes = []
    for i in range(0, len(v), 2):
        q = lambda x: (max(-7, min(7, lround(x / ss)))) & 0xf
        codes.append(q(v[i]) | (q(v[i + 1]) << 4))
    return codes, f32_to_f16_bits(s if s > 0 else 1.0)


def iso3(v):
    mx = max(abs(x) for x in v)
    s = mx / 3.0
    ss = s if s > 0 else 1.0
    packed = 0
    for i, x in enumerate(v):
        mag = max(0, min(3, lround(abs(x) / ss)))
        bits = ((1 << 2) if x < 0 else 0) | mag     # 符号(bit2) + 幅值(bit0-1)
        packed |= bits << (3 * i)                    # 元素 i 从第 3*i 位起
    return [packed & 0xff, (packed >> 8) & 0xff, (packed >> 16) & 0xff], \
        f32_to_f16_bits(s if s > 0 else 1.0)


if __name__ == '__main__':
    c4, s4 = iso4(V4)
    c3, s3 = iso3(V3)
    print('iso4 codes:', ' '.join('%02x' % x for x in c4), 'scale=%04x' % s4)
    print('iso3 codes:', ' '.join('%02x' % x for x in c3), 'scale=%04x' % s3)
