# -*- coding: utf-8 -*-
"""QPN prepack prototype: engine NVFP4 codes -> QPN fragment order (pure permutation).
Self-check: multiset of nibbles preserved + layout shape matches reference contract."""
import numpy as np

# ---- engine layout: NVFP4 packed [rows=n, cols=k/2] nibbles (e2m1), group k/16 ----
def qpn_prepack(codes: np.ndarray, scales: np.ndarray):
    """codes: uint8 [n, k/2]; scales: uint8 fp8-e4m3 [n, k/16]. -> qc [tiles,groups,32,8], qs."""
    n, k2 = codes.shape
    k = k2 * 2
    assert n % 32 == 0 and k % 64 == 0, (n, k)
    tiles, groups = n // 32, k // 16
    lane = np.arange(32)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0) * 4
    korder = np.array([0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15])
    nib = np.stack([codes & 0xF, codes >> 4], axis=-1).reshape(n, k)  # [n, k] nibbles
    qc = np.zeros((tiles, groups, 32, 8), dtype=np.uint8)
    qs = np.zeros((tiles, groups, 32), dtype=np.uint8)
    for t in range(tiles):
        ncol = t * 32 + col  # 32 rows picked per tile
        for g in range(groups):
            kidx = g * 16 + korder
            nb = nib[ncol][:, kidx]  # [32,16] nibbles
            qc[t, g] = nb[:, 0::2] | (nb[:, 1::2] << 4)
            qs[t, g] = scales[ncol, g]
    return qc, qs


def self_check(n=128, k=512, seed=7):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
    scales = rng.integers(0, 256, size=(n, k // 16), dtype=np.uint8)
    qc, qs = qpn_prepack(codes, scales)
    # multiset preservation
    nib = np.stack([codes & 0xF, codes >> 4], axis=-1).reshape(-1)
    qnib = np.stack([qc & 0xF, qc >> 4], axis=-1).reshape(-1)
    assert np.array_equal(np.sort(nib), np.sort(qnib)), 'nibble multiset mismatch'
    # scales coverage: each group/lane equals source row scale at col
    lane = np.arange(32)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0) * 4
    for t in (0, 3):
        for g in (0, 5):
            want = scales[t * 32 + col, g]
            assert np.array_equal(qs[t, g], want), 'scale tile mismatch'
    return qc.shape, qs.shape


if __name__ == '__main__':
    print('shapes ok:', self_check())
