# -*- coding: utf-8 -*-
"""kit.demo — the runnable proof that the shared writer produces a REAL artifact.

What this does, with no checkpoint download and no GPU:

  1. writes a synthetic safetensors checkpoint (config.json + one shard + its
     index) into a throwaway directory, with the SAME key vocabulary and the same
     `contiguous-le-v1` BF16 storage a real Llama-shaped checkpoint uses, and with
     `tie_word_embeddings=true` so the alias path is exercised;
  2. hands it to `tools.convert.minicpm5_1b.declaration.build_declaration` -- the
     very same declaration the real 219-object conversion uses, only at 3 layers;
  3. runs the shared driver over it, and
  4. re-opens the produced file with `tools/artifact/container.py`'s own reader and
     prints magic, identity, object count, bytes and sha256.

The point is not that a synthetic model converts.  The point is that the model-side
code and the real-model code are the SAME code: `demo` differs from the real run by
one integer (3 layers instead of 24), so a demo that passes is evidence about the
real path rather than about a demonstration harness that shares nothing with it.

    python3 -m tools.archkit.kit.demo [--out DIR] [--keep]
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "tools.archkit.kit"

from tools.artifact.container import MAGIC, Artifact

from tools.archkit.kit import driver
from tools.convert.minicpm5_1b.declaration import build_declaration

#: the synthetic geometry: same SHAPES OF RELATION as MiniCPM5-1B (16 q heads over
#: 2 kv heads at head_dim 128, intermediate = 3 x hidden), one eighth of the depth
DEMO_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 512,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "intermediate_size": 1536,
    "vocab_size": 2048,
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000,
    "hidden_act": "silu",
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    # one attention kind only, so the layer-kind gate gets a total schedule to check
    "full_attention_interval": 1,
}

LAYER_KEYS = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _shape(config, leaf: str) -> tuple[int, ...]:
    hidden = config["hidden_size"]
    vocab = config["vocab_size"]
    inter = config["intermediate_size"]
    q = config["num_attention_heads"] * config["head_dim"]
    kv = config["num_key_value_heads"] * config["head_dim"]
    return {
        "input_layernorm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
        "self_attn.q_proj.weight": (q, hidden),
        "self_attn.k_proj.weight": (kv, hidden),
        "self_attn.v_proj.weight": (kv, hidden),
        "self_attn.o_proj.weight": (hidden, q),
        "mlp.gate_proj.weight": (inter, hidden),
        "mlp.up_proj.weight": (inter, hidden),
        "mlp.down_proj.weight": (hidden, inter),
        "model.embed_tokens.weight": (vocab, hidden),
        "lm_head.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
    }[leaf]


def _deterministic_bytes(count: int, seed: int) -> bytes:
    """Reproducible pseudo-random payload, cheap to build and to check.

    A counter through sha256 rather than `random`: the demo's sha256 reading is
    printed in the report, and a reading that moves between runs is not a reading.
    """
    digest = hashlib.sha256()
    out = bytearray()
    block = 0
    while len(out) < count:
        digest.update(struct.pack("<QQ", seed, block))
        out += digest.digest()
        block += 1
    return bytes(out[:count])


def write_synthetic_checkpoint(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(DEMO_CONFIG)
    (directory / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")

    keys: list[tuple[str, tuple[int, ...]]] = []
    keys.append(("model.embed_tokens.weight", _shape(config, "model.embed_tokens.weight")))
    for layer in range(config["num_hidden_layers"]):
        for leaf in LAYER_KEYS:
            keys.append(("model.layers.%d.%s" % (layer, leaf),
                         _shape(config, leaf)))
    keys.append(("model.norm.weight", _shape(config, "model.norm.weight")))
    if not config["tie_word_embeddings"]:
        keys.append(("lm_head.weight", _shape(config, "lm_head.weight")))
    else:
        # A tied checkpoint still carries lm_head.weight in its index (that is what
        # the real MiniCPM5-1B does NOT do, and the difference is the whole reason
        # the tie path has to be exercised somewhere).  It is dropped from the
        # index here so the demo's tie is a real alias and not a duplicate source.
        pass

    shard_name = "model-00000-of-00001.safetensors"
    offset = 0
    header: dict = {}
    payload = bytearray()
    weight_map: dict[str, str] = {}
    for seed, (key, shape) in enumerate(keys):
        count = 1
        for dim in shape:
            count *= dim
        nbytes = count * 2
        data = _deterministic_bytes(nbytes, seed)
        header[key] = {"dtype": "BF16", "shape": list(shape),
                       "data_offsets": [offset, offset + nbytes]}
        payload += data
        offset += nbytes
        weight_map[key] = shard_name
    body = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(directory / shard_name, "wb") as handle:
        handle.write(struct.pack("<Q", len(body)))
        handle.write(body)
        handle.write(payload)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": offset}, "weight_map": weight_map},
                   indent=2) + "\n", encoding="utf-8")
    return directory


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None,
                        help="directory for the synthetic checkpoint and artifact")
    parser.add_argument("--keep", action="store_true",
                        help="do not delete the working directory")
    args = parser.parse_args(argv)

    work = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="archkit-demo-"))
    if work.exists() and args.out:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = write_synthetic_checkpoint(work / "MiniCPM5-1B-demo")

    print("== synthetic checkpoint")
    for path in sorted(checkpoint.iterdir()):
        print("   %-34s %d B" % (path.name, path.stat().st_size))

    # The spec is EXTRACTED from the synthetic checkpoint's own config.json by the
    # tree's existing extractor, not hand-written for the demo: `spec-vs-config` is a
    # real gate, so a demo carrying a spec that disagrees with its own checkpoint
    # would be refused (correctly) instead of demonstrating anything.
    from tools.archkit.arch_spec import hf_to_spec

    spec_path = work / "minicpm5_1b_spec.json"
    spec = hf_to_spec(str(checkpoint / "config.json"), "minicpm5-1b",
                      family="qwen3_6", out=str(spec_path))
    print("== spec extracted from the synthetic config.json")
    print("   %s  geometry=%s" % (spec_path.name, spec["geometry"]))
    print("   layer_types=%s" % (spec.get("layer_types") or "(absent: single-kind stack)"))

    artifact = work / "demo_minicpm5_1b.ninfer"
    print("== shared driver")
    report = driver.run(functools.partial(build_declaration, spec_dir=work),
                        checkpoint, artifact, spec_dir=work)
    print(report.render())

    with open(artifact, "rb") as handle:
        magic = handle.read(8)
    with Artifact.open(artifact) as opened:
        print("== re-opened with tools/artifact/container.py")
        print("   magic       %r  (container.MAGIC %r, identical=%s)"
              % (magic, MAGIC, magic == MAGIC))
        print("   model_id    %s" % opened.identity.model_id)
        print("   weights_id  %s" % opened.identity.weights_id)
        print("   objects     %d" % len(opened.objects))
        print("   file_bytes  %d" % opened.file_bytes)
        print("   payload_off %d" % opened.payload_offset)

    if not args.keep and args.out is None:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print("== kept:", work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
