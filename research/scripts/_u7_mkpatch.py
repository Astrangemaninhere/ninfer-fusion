#!/usr/bin/env python3
"""Regenerate the U7 Muse page-fill patch as a VALID unified diff.

The previous patch was hand-written and `patch` rejected it as malformed
(hunk counts wrong). Anchored replacement + difflib is the reliable pattern the
subagents use; do the same here and prove it with a dry-run.
"""
import difflib
import pathlib
import subprocess

REPO = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\ninfer-fusion-repo")
REL = "src/ops/kernel/gqa_attention_prefill_i8.cuh"
SRC = REPO / REL
OUT = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\_collab\M_muse_pagefill_patch.diff")

orig = SRC.read_text(encoding="utf-8", errors="surrogateescape")
text = orig

# ---- replacement 1: the E8 branch (rotation missing, /127 should be /7,
#      lattice projection missing, two dead ifs) ------------------------------
old1 = """        const __half ksh_e = __float2half_rn(k_abs > 0.0f ? k_abs / 127.0f : 0.0f);
        const __half vsh_e = __float2half_rn(v_abs > 0.0f ? v_abs / 127.0f : 0.0f);
        const float ks_e   = __half2float(ksh_e);
        const float vs_e   = __half2float(vsh_e);
        const float kinv_e = ks_e > 0.0f ? 1.0f / ks_e : 0.0f;
        const float vinv_e = vs_e > 0.0f ? 1.0f / vs_e : 0.0f;
        const int c0  = max(-7, min(7, static_cast<int>(rintf(k0 * kinv_e))));
        const int c1  = max(-7, min(7, static_cast<int>(rintf(k1 * kinv_e))));
        if (token == 0 && kv_head == 0 && group == 0 && lane == 1) {
        }
        if (token == 0 && kv_head == 0 && group == 0 && lane == 0) {
        }
"""
new1 = """        // Same defect family as the already-fixed fill path above (see that block's
        // comment; _TODO.md 116/116c). Three things were wrong here at once: the group
        // max was taken BEFORE any rotation while the reader rotates Q; the E8 code
        // range is +-7 (not 127) so the scale was ~18x too small and every code
        // saturated the clamp; and the E8 lattice projection was missing entirely, so
        // this wrote plain rounded coordinates where the reader expects E8 ones.
        // Muse is the only geometry that reaches this kernel (KVHeads == 2).
        gqa_kv_hadamard64(k0, k1, FullMask);
        float k_abs_e = fmaxf(fabsf(k0), fabsf(k1));
        k_abs_e       = warp_max(k_abs_e, FullMask);
        const __half ksh_e = __float2half_rn(k_abs_e > 0.0f ? k_abs_e / 7.0f : 0.0f);
        const __half vsh_e = __float2half_rn(v_abs > 0.0f ? v_abs / 7.0f : 0.0f);
        const float ks_e   = __half2float(ksh_e);
        const float vs_e   = __half2float(vsh_e);
        const float kinv_e = ks_e > 0.0f ? 1.0f / ks_e : 0.0f;
        const float vinv_e = vs_e > 0.0f ? 1.0f / vs_e : 0.0f;
        float k0_scaled = k0 * kinv_e;
        float k1_scaled = k1 * kinv_e;
        e8_project_8d_warp(k0_scaled, k1_scaled, lane);
        const int c0  = max(-7, min(7, static_cast<int>(rintf(k0_scaled))));
        const int c1  = max(-7, min(7, static_cast<int>(rintf(k1_scaled))));
"""

# ---- replacement 2: the neighbour codes must be projected too ---------------
old2 = """        const float k0n_nb = __shfl_xor_sync(FullMask, k0, 1);
        const float k1n_nb = __shfl_xor_sync(FullMask, k1, 1);
        const float v0n_nb = __shfl_xor_sync(FullMask, v0, 1);
        const float v1n_nb = __shfl_xor_sync(FullMask, v1, 1);
        if ((lane & 1) == 0) {
            const int c0n  = max(-7, min(7, static_cast<int>(rintf(k0n_nb * kinv_e))));
            const int c1n  = max(-7, min(7, static_cast<int>(rintf(k1n_nb * kinv_e))));
"""
new2 = """        const float k0n_nb = __shfl_xor_sync(FullMask, k0, 1);
        const float k1n_nb = __shfl_xor_sync(FullMask, k1, 1);
        float k0n_s = k0n_nb * kinv_e;
        float k1n_s = k1n_nb * kinv_e;
        e8_project_8d_warp(k0n_s, k1n_s, lane);
        const float v0n_nb = __shfl_xor_sync(FullMask, v0, 1);
        const float v1n_nb = __shfl_xor_sync(FullMask, v1, 1);
        if ((lane & 1) == 0) {
            const int c0n  = max(-7, min(7, static_cast<int>(rintf(k0n_s))));
            const int c1n  = max(-7, min(7, static_cast<int>(rintf(k1n_s))));
"""

for i, (old, new) in enumerate(((old1, new1), (old2, new2)), 1):
    n = text.count(old)
    if n != 1:
        raise SystemExit("anchor %d matched %d times (must be exactly 1) — aborting" % (i, n))
    text = text.replace(old, new)

diff = list(difflib.unified_diff(
    orig.splitlines(keepends=True), text.splitlines(keepends=True),
    fromfile="a/" + REL, tofile="b/" + REL, n=3))
if not diff:
    raise SystemExit("no changes produced — anchors already applied?")

OUT.write_text("".join(diff), encoding="utf-8")
print("wrote %s (%d lines)" % (OUT, len(diff)))

# apply-check on a shadow copy (never touch the live tree)
import shutil
import tempfile
shadow = pathlib.Path(tempfile.mkdtemp(prefix="u7_")) / "a"
(shadow / pathlib.Path(REL).parent).mkdir(parents=True)
shutil.copy2(SRC, shadow / REL)
p = subprocess.run(["patch", "-p1", "--dry-run", "-i", str(OUT)], cwd=str(shadow),
                   capture_output=True, text=True)
print("dry-run rc=%d" % p.returncode)
print(p.stdout.strip() or p.stderr.strip())
if p.returncode != 0:
    raise SystemExit("PATCH WOULD NOT APPLY")
print("U7_PATCH_DRYRUN_OK")
