# Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
import argparse
import json
import os
import subprocess
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ORACLE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "..", "..", "..", "tools", "reference", "qwen3_8_flash_next", "oracle"))
if ORACLE_DIR not in sys.path:
    sys.path.insert(0, ORACLE_DIR)
import mtp_reference  # noqa: E402  (the vLLM-faithful reference; no second copy here)
from mtp_reference import (  # noqa: E402
    ALL_STAGES,
    HAVE_QWEN4_EXP,
    STEM_STAGES,
    dump_stages,
    gemma_rmsnorm,
    mtp_stem,
    stem_streams_are_distinct,
)


def test_stem_semantics_without_transformers() -> int:
    """Pins the three stem properties the old reconstruction got wrong, using pure
    torch so it runs on a box whose transformers has no qwen4_exp."""
    torch.manual_seed(0)
    hc, H, eps = 4, 8, 1e-6
    emb = torch.randn(2, 3, H)
    hid = torch.randn(2, 3, hc * H)
    w_en, w_hn = torch.randn(H) * 0.1, torch.randn(hc * H) * 0.1
    w_fe, w_fh = torch.randn(H, H), torch.randn(H, H)
    stages, init = mtp_stem(emb, hid, embedding_norm_weight=w_en, fc_embedding_weight=w_fe,
                            hidden_norm_weight=w_hn, fc_hidden_weight=w_fh, hc_count=hc, hidden_size=H, eps=eps)
    failures = 0

    def check(ok, msg):
        nonlocal failures
        if not ok:
            failures += 1
            print("FAIL:", msg)

    # 1. Four DISTINCT streams enter the layer (old stem: four identical copies).
    check(stem_streams_are_distinct(init, hc, H), "streams must differ after the stem")
    # 2. The hidden norm is ONE group over hc*H, not hc groups of H.
    x = hid[0, 0]
    single = gemma_rmsnorm(x, w_hn, eps)
    grouped = (x.view(hc, H) * torch.rsqrt(x.view(hc, H).pow(2).mean(-1, keepdim=True) + eps)).flatten() * (1 + w_hn)
    check(torch.allclose(stages["mtp_hidden_norm"][0, 0], single), "hidden norm must be single-group")
    check(not torch.allclose(single, grouped), "fixture must distinguish single-group from per-branch norm")
    # 3. Each stream is fc_hidden(norm(h))_i + fc_embedding(norm(e)) -- embedding broadcast, no mixer.
    s = init.unflatten(-1, (hc, H))
    ref_e = torch.nn.functional.linear(gemma_rmsnorm(emb, w_en, eps), w_fe)
    for i in range(hc):
        ref_i = torch.nn.functional.linear(stages["mtp_hidden_norm"].unflatten(-1, (hc, H))[..., i, :], w_fh) + ref_e
        check(torch.allclose(s[..., i, :], ref_i, atol=1e-5), f"stream {i} formula")
    # 4. Stage names are the corrected canonical set.
    check(tuple(stages.keys()) == STEM_STAGES, f"stem stage names {tuple(stages.keys())} != {STEM_STAGES}")
    print("stem semantics: " + ("OK" if failures == 0 else f"{failures} failure(s)"))
    return failures



def load_tensor(base_dir, rec):
    file_path = os.path.join(base_dir, rec['file'])
    dtype_str = rec.get('dtype', 'BF16')
    shape = rec.get('shape', [])
    with open(file_path, 'rb') as f:
        data = f.read()
    if dtype_str == 'BF16':
        u16 = np.frombuffer(data, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        arr = u32.view(np.float32)
    elif dtype_str == 'FP32':
        arr = np.frombuffer(data, dtype=np.float32)
    elif dtype_str == 'I32':
        arr = np.frombuffer(data, dtype=np.int32)
    elif dtype_str == 'I64':
        arr = np.frombuffer(data, dtype=np.int64)
    else:
        arr = np.frombuffer(data, dtype=np.float32)
    if shape:
        try:
            arr = arr.reshape(shape)
        except Exception:
            pass
    return arr.astype(np.float64)

def compare_tensors(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    if len(a_flat) != len(b_flat):
        raise ValueError(f"Tensor element count differs: {len(a_flat)} != {len(b_flat)}")
    diff = np.abs(a_flat - b_flat)
    max_diff = float(np.max(diff)) if len(diff) > 0 else 0.0
    norm_b = float(np.linalg.norm(b_flat))
    norm_diff = float(np.linalg.norm(diff))
    rel_l2 = (norm_diff / norm_b) if norm_b > 1e-12 else norm_diff
    norm_a = float(np.linalg.norm(a_flat))
    if norm_a > 1e-12 and norm_b > 1e-12:
        cosine = float(np.dot(a_flat, b_flat) / (norm_a * norm_b))
    else:
        cosine = 1.0 if norm_a == norm_b else 0.0
    return max_diff, rel_l2, cosine

def run_parity_test(ninfer_exe: str, dump_dir: str) -> int:
    if not HAVE_QWEN4_EXP:
        # Full-parity mode never degrades silently: no qwen4_exp means no verdict.
        print("FAIL: full-parity mode requires transformers with qwen4_exp and this interpreter has none. "
              "Use E:/NInfer/venv-qwen4exp (see tools/reference/qwen3_8_flash_next/oracle/README.md), "
              "or pass --stem-only for the dependency-free stem check.")
        return 2
    Qwen4ExpMTP = mtp_reference.Qwen4ExpMTPReference
    make_mtp_config = mtp_reference.make_mtp_config

    ninfer_dump = os.path.join(dump_dir, "ninfer")
    oracle_dump = os.path.join(dump_dir, "oracle")
    os.makedirs(ninfer_dump, exist_ok=True)
    os.makedirs(oracle_dump, exist_ok=True)

    print("=== STEP 1: Running NInfer C++ MTP Synthetic Test with StateDumper ===")
    res = subprocess.run([ninfer_exe, "--dump-states", ninfer_dump], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"NInfer test failed with code {res.returncode}: {res.stderr}")
        sys.exit(res.returncode)

    print("=== STEP 2: Running vLLM-faithful PyTorch Qwen4ExpMTP reference ===")
    with open(r"E:\NInfer\qwen3_8_flash_next\source\mixed\config.json", "r") as f:
        cfg = make_mtp_config(json.load(f)["text_config"])
    mtp = Qwen4ExpMTP(cfg)
    mtp.eval()
    dim = 2560
    with torch.no_grad():
        for prm in mtp.parameters():
            prm.zero_()
        mtp.fc_embedding.weight.copy_(torch.eye(dim))
        mtp.fc_hidden.weight.copy_(torch.eye(dim))
        for i in range(100):
            mtp.lm_head.weight[i, 0] = float(i + 1)
    # Per-stream distinct backbone: the C++ harness must feed the same values
    # (streams 1,2,3,4 = 1.0,2.0,3.0,4.0) or the stem comparison is meaningless.
    input_emb = torch.ones(1, 1, dim)
    backbone_h = torch.arange(1, 5, dtype=torch.float32).repeat_interleave(dim).view(1, 1, 4 * dim)
    with torch.no_grad():
        oracle_stages = {k: v[0, 0] for k, v in mtp(input_emb, backbone_h).items()}
    manifest = {"positions": []}
    dump_stages(oracle_dump, 0, 0, (0, 0, 0), oracle_stages, manifest)
    with open(os.path.join(oracle_dump, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("=== STEP 3: Stage-by-Stage Non-Vacuity & Parity Verification ===")
    with open(os.path.join(ninfer_dump, "manifest.json"), "r") as f:
        man_ninfer = json.load(f)
    ninfer_tensors = {t["name"]: t for t in man_ninfer["positions"][0]["tensors"]}
    oracle_tensors = {t["name"]: t for t in manifest["positions"][0]["tensors"]}

    passed = True
    canonical = set(ALL_STAGES)
    print(f"{'Stage Name':<28} | {'NInfer Norm':<12} | {'Kind':<6} | {'Status':<16}")
    print("-" * 72)
    for name, n_rec in ninfer_tensors.items():
        arr_n = load_tensor(ninfer_dump, n_rec)
        norm_n = float(np.linalg.norm(arr_n))
        finite = bool(np.all(np.isfinite(arr_n)))
        if name in canonical:
            # A canonical stage must be finite and non-zero: an all-zero hidden is a
            # wiring bug, not a value.
            ok, kind = finite and norm_n > 0.0, "stage"
            status = "OK" if ok else ("FAIL (NaN/Inf)" if not finite else "FAIL (VACUOUS)")
        else:
            # Everything else is a diagnostic (indexer selected counts, active blocks,
            # cache bookkeeping). Zero is a legitimate value there, e.g. for token 0,
            # so only NaN/Inf fails.
            ok, kind = finite, "diag"
            status = "OK" if ok else "FAIL (NaN/Inf)"
        passed &= ok
        print(f"{name:<28} | {norm_n:<12.4f} | {kind:<6} | {status:<16}")

    # Stale names from the old stem are a failure, not a skip: their presence means
    # the C++ side still mixes-and-repeats.
    for stale in ("mtp_hidden_mix", "mtp_trunk_input"):
        if stale in ninfer_tensors:
            print(f"FAIL: C++ still emits {stale}; the stem has not been corrected to the vLLM contract")
            passed = False

    # Presence: stem and head must both be there. Parity: STEM ONLY. The C++ harness
    # drives its decoder layer with its own (random) weights while this oracle zeroes
    # them, so layer and head values are fixture-dependent; they are checked for
    # presence and finiteness above, never for value.
    if "mtp_final_hidden" not in ninfer_tensors:
        print("FAIL: C++ dump is missing canonical stage mtp_final_hidden")
        passed = False
    for stage in STEM_STAGES:
        if stage not in ninfer_tensors:
            print(f"FAIL: C++ dump is missing canonical stage {stage}")
            passed = False
            continue
        arr_n = load_tensor(ninfer_dump, ninfer_tensors[stage])
        arr_o = load_tensor(oracle_dump, oracle_tensors[stage])
        max_d, rel_l2, cosine = compare_tensors(arr_n, arr_o)
        print(f"Stem Parity [{stage}]: max_diff={max_d:.6f}, rel_l2={rel_l2:.6f}, cosine={cosine:.6f}")
        if rel_l2 > 0.05:
            print(f"FAIL: stage {stage} diverged (rel_l2={rel_l2:.6f})")
            passed = False

    # The property that separates a correct stem from the old one, on the C++ dump itself.
    if "mtp_hyper_init" in ninfer_tensors:
        hi = torch.from_numpy(load_tensor(ninfer_dump, ninfer_tensors["mtp_hyper_init"])).reshape(-1)
        if not stem_streams_are_distinct(hi, 4, dim):
            print("FAIL: C++ mtp_hyper_init has four identical streams -- stem still mixes and repeats")
            passed = False

    print("\n=== " + ("ALL MTP STEM-PARITY AND NON-VACUITY CHECKS PASSED" if passed else "MTP PARITY CHECK FAILED") + " ===")
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ninfer-exe', default=r'E:\NInfer.gemini2\build\tests\ninfer_qwen3_8_flash_next_mtp_test.exe')
    parser.add_argument('--dump-dir', default=r'E:\NInfer.gemini2\build\dumps\mtp_parity')
    parser.add_argument('--stem-only', action='store_true', help='Run only the transformers-free stem semantics check')
    args = parser.parse_args()
    if args.stem_only:
        sys.exit(test_stem_semantics_without_transformers())
    sys.exit(run_parity_test(args.ninfer_exe, args.dump_dir))
