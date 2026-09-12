#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_dflash2.py -- rebuild a qwen3.8-27B .ninfer artifact whose dflash2/
draft tensors come from a train_dflash2.py checkpoint (DFlash2Head state_dict).
All other objects (main weights, selector, vision...) are copied unchanged.

Mapping table: engine artifact name -> trainer state-dict key. Trainer layout
matches the engine dflash2 semantics (per-layer context projections, two-tap
grouped convs, BF16). Run with --dry-run to validate mapping/shapes only.
"""
import argparse
import io
import sys

import numpy as np
import torch

sys.path.insert(0, r"C:\Users\User\Documents\ziqinzhang\ninfer-fusion-repo")
from tools.artifact.container import Artifact, ArtifactWriter, ResourceSpec, TensorSpec  # noqa: E402

LAYERS = 5
# Checkpoints saved from a torch.compile-wrapped DraftModel carry an
# "_orig_mod." prefix on every key; unwrap when present.
CKPT_PREFIX = "_orig_mod."


def lookup(sd, base_key):
    if base_key in sd:
        return base_key
    prefixed = CKPT_PREFIX + base_key
    if prefixed in sd:
        return prefixed
    return None


# artifact name -> (state key, expected artifact shape)
# conv base kernels are trained grouped (2,2,320,16) but stored flat
# (2,2,5120) in the artifact; same element count, view is order-preserving.
MAPPING = {}
for i in range(LAYERS):
    MAPPING[f"dflash2/layers/{i}/input_norm"] = (
        f"layers.{i}.input_layernorm.weight", (5120,))
    MAPPING[f"dflash2/layers/{i}/post_attention_norm"] = (
        f"layers.{i}.post_attention_layernorm.weight", (5120,))
    MAPPING[f"dflash2/layers/{i}/attention/query_key_value"] = (
        f"layers.{i}.self_attn_qkv.weight", (6144, 5120))
    MAPPING[f"dflash2/layers/{i}/attention/context_key"] = (
        f"layers.{i}.context_key.weight", (1024, 5120))
    MAPPING[f"dflash2/layers/{i}/attention/context_value"] = (
        f"layers.{i}.context_value.weight", (1024, 5120))
    MAPPING[f"dflash2/layers/{i}/attention/query_norm"] = (
        f"layers.{i}.self_attn_q_norm.weight", (128,))
    MAPPING[f"dflash2/layers/{i}/attention/key_norm"] = (
        f"layers.{i}.self_attn_k_norm.weight", (128,))
    MAPPING[f"dflash2/layers/{i}/attention/output"] = (
        f"layers.{i}.self_attn_o.weight", (5120, 4096))
    MAPPING[f"dflash2/layers/{i}/attention_conv/base_kernel"] = (
        f"layers.{i}.attention_conv.base", (2, 2, 5120))
    MAPPING[f"dflash2/layers/{i}/attention_conv/kernel_projection"] = (
        f"layers.{i}.attention_conv.proj.weight", (1280, 5120))
    MAPPING[f"dflash2/layers/{i}/mlp/gate_up"] = (
        f"layers.{i}.mlp_gate_up.weight", (34816, 5120))
    MAPPING[f"dflash2/layers/{i}/mlp/down"] = (
        f"layers.{i}.mlp_down.weight", (5120, 17408))
    MAPPING[f"dflash2/layers/{i}/mlp_conv/base_kernel"] = (
        f"layers.{i}.mlp_conv.base", (2, 2, 5120))
    MAPPING[f"dflash2/layers/{i}/mlp_conv/kernel_projection"] = (
        f"layers.{i}.mlp_conv.proj.weight", (1280, 5120))
MAPPING["dflash2/feature_projection"] = ("fc.weight", (5120, 25600))
MAPPING["dflash2/context_norm"] = ("hidden_norm.weight", (5120,))
MAPPING["dflash2/final_norm"] = ("norm.weight", (5120,))


def tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().to(torch.bfloat16).contiguous()
    return t.view(torch.uint16).cpu().numpy().tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source .ninfer artifact")
    ap.add_argument("--ckpt", required=True, help="train step_XXXX.pt checkpoint")
    ap.add_argument("--out", default=None, help="output .ninfer (required unless --dry-run)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    art = Artifact.open(args.src)
    names = [o.name for o in art.objects]
    missing = [n for n in MAPPING if n not in names]
    if missing:
        print("MAPPING ERROR: artifact missing:", missing[:6])
        return 2
    print(f"artifact ok: {art.identity.model_id}/{art.identity.weights_id} "
          f"objects={len(names)}")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    miss_keys = [base for base, _ in MAPPING.values() if lookup(sd, base) is None]
    if miss_keys:
        print("CKPT ERROR: missing keys:", miss_keys[:8])
        return 2
    print(f"checkpoint ok: {len(sd)} keys")

    payloads = {}
    for aname, (skey, shape) in MAPPING.items():
        obj = art.find(aname)
        if tuple(obj.shape) != shape:
            print(f"SHAPE MISMATCH {aname}: artifact {tuple(obj.shape)} "
                  f"expected {shape}")
            return 2
        t = sd[lookup(sd, skey)]
        if tuple(t.shape) != shape:
            if t.numel() == np.prod(shape):
                print(f"note: {skey} reshaped {tuple(t.shape)} -> {shape}")
                t = t.detach().contiguous().view(shape)
            else:
                print(f"SHAPE MISMATCH {skey}: ckpt {tuple(t.shape)} expected {shape}")
                return 2
        payloads[aname] = tensor_bytes(t)
    print(f"mapping ok: {len(payloads)} tensors queued")

    if args.dry_run:
        print("dry-run ok (no file written)")
        return 0

    # Build the writer with specs reconstructed from the source objects,
    # then stream: replaced dflash2 tensors from the ckpt, all else copied.
    specs = []
    for o in art.objects:
        if o.kind == "tensor":
            specs.append(TensorSpec(name=o.name, shape=o.shape,
                                    format=o.format, layout=o.layout))
        else:
            specs.append(ResourceSpec(name=o.name, encoding=o.encoding, bytes=o.bytes))
    w = ArtifactWriter(args.out, art.identity, specs)
    for o in art.objects:
        if o.name in payloads:
            w.write(o.name, payloads[o.name])
        else:
            w.write(o.name, art.payload(o))
    w.finish()
    print("written:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
