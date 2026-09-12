# Muse serve-path acceptance (2026-09-10 18:08)

| dtype | state | nan_lines | text |
|---|---|---|---|
| bf16 | SERVE_FAILED | 118 | <no request: SERVE_FAILED> |
| nvfp4 | REFUSED | 0 | <no request: REFUSED> |

PASS requires state=CLEAN (started, HTTP body parsed, non-garbage text, 0 NaN, no CUDA error).
REFUSED = the engine explicitly refused this KV dtype for this geometry (expected for
nvfp4 on head_dim=128 until that kernel is ported) — it is NOT a pass.
verdict: **FAIL**
# Muse serve-path acceptance (2026-09-10 18:16)

| dtype | state | nan_lines | text |
|---|---|---|---|
| bf16 | SERVE_FAILED | 0 | <no request: SERVE_FAILED> |
| nvfp4 | REFUSED | 0 | <no request: REFUSED> |

PASS requires state=CLEAN (started, HTTP body parsed, non-garbage text, 0 NaN, no CUDA error).
REFUSED = the engine explicitly refused this KV dtype for this geometry (expected for
nvfp4 on head_dim=128 until that kernel is ported) — it is NOT a pass.
verdict: **FAIL**
