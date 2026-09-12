#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_patch.py -- integrity check for patch_dflash2.py output.

Compares src vs out artifact:
  * identity + object table identical (names, kinds, shapes, formats)
  * every payload byte-equal EXCEPT the dflash2/* tensors (replaced), which
    are compared against the checkpoint they were patched from.
"""
import argparse
import hashlib
import sys

import numpy as np
import torch

sys.path.insert(0, r"C:\Users\User\Documents\ziqinzhang\ninfer-fusion-repo")
from tools.artifact.container import Artifact  # noqa: E402


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None, help="checkpoint used for patch (optional)")
    args = ap.parse_args()

    sa, oa = Artifact.open(args.src), Artifact.open(args.out)
    if (sa.identity.model_id, sa.identity.weights_id) != (
            oa.identity.model_id, oa.identity.weights_id):
        print("FAIL: identity differs")
        return 2
    sn = [o.name for o in sa.objects]
    on = [o.name for o in oa.objects]
    if sn != on:
        print(f"FAIL: object tables differ ({len(sn)} vs {len(on)})")
        return 2
    for s, o in zip(sa.objects, oa.objects):
        if s.kind != o.kind:
            print(f"FAIL: object kind differs at {s.name}")
            return 2
        if s.kind == "tensor" and (
                tuple(s.shape), s.format, s.layout) != (tuple(o.shape), o.format, o.layout):
            print(f"FAIL: tensor spec differs at {s.name}")
            return 2
        if s.kind == "resource" and getattr(s, "encoding", None) != getattr(o, "encoding", None):
            print(f"FAIL: resource spec differs at {s.name}")
            return 2
    print(f"identity+table ok: {len(on)} objects")

    replaced = {n for n in on if n.startswith("dflash2/")}
    copied, checked = 0, 0
    for s, o in zip(sa.objects, oa.objects):
        if o.name in replaced:
            continue  # verified against ckpt below
        if o.kind != "tensor":
            continue
        copied += 1
        if digest(sa.payload(s)) != digest(oa.payload(o)):
            print(f"FAIL: copied tensor differs at {o.name}")
            return 2
        checked += 1
    print(f"copied tensors ok: {checked}/{copied} hash-equal")

    if args.ckpt:
        patch_dir = str(__import__("pathlib").Path(__file__).resolve().parent)
        if patch_dir not in sys.path:
            sys.path.insert(0, patch_dir)
        import patch_dflash2  # noqa: PLC0415

        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        for aname, (skey, shape) in patch_dflash2.MAPPING.items():
            key = patch_dflash2.lookup(sd, skey)
            o = oa.find(aname)
            t = sd[key].detach().to(torch.bfloat16).contiguous()
            if tuple(t.shape) != shape and t.numel() == np.prod(shape):
                t = t.view(shape)
            want = t.view(torch.uint16).cpu().numpy().tobytes()
            if oa.payload(o) != want:
                print(f"FAIL: replaced tensor differs from ckpt at {aname}")
                return 2
        print(f"replaced tensors ok: {len(patch_dflash2.MAPPING)} match checkpoint")
    print("verify ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
