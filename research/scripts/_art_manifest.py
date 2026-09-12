#!/usr/bin/env python3
"""Read the .ninfer container's JSON from the dflash2 artifact and compare the
declared draft/selector hyper-parameters (and tensor shapes) against the harness
constants we hardcoded in src/targets/qwen3_6_27b/impl/config.h."""
import json
import pathlib
import struct
import sys

ART = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"


def load_json(path):
    with open(path, "rb") as handle:
        magic = handle.read(8)
        if not magic.startswith(b"NINFER"):
            print("BAD magic:", magic)
            return None, None
        (length,) = struct.unpack("<Q", handle.read(8))
        payload_start = 16 + length
        payload_start = (payload_start + 4095) // 4096 * 4096
        raw = handle.read(length)
    return json.loads(raw.decode("utf-8")), payload_start


def main():
    doc, payload_start = load_json(ART)
    if doc is None:
        return 1
    print(f"magic OK, payload_start = {payload_start}")
    print("top-level keys:", sorted(doc.keys()))
    print()

    # --- any dflash2/dflash config block, printed whole ---
    for key in ("config", "target", "draft", "speculative", "dflash2", "dflash", "model"):
        if key in doc:
            print(f"===== doc[{key!r}] =====")
            print(json.dumps(doc[key], indent=2, ensure_ascii=False)[:6000])
            print()

    # --- tensor inventory for the draft/selector side ---
    def walk(node, prefix=""):
        out = []
        if isinstance(node, dict):
            for k, v in node.items():
                out += walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(node, list):
            if node and isinstance(node[0], dict) and "name" in node[0]:
                for item in node:
                    out.append((f"{prefix}.{item.get('name')}",
                                item.get("shape"), item.get("dtype")))
            else:
                for i, v in enumerate(node):
                    out.append((f"{prefix}[{i}]", v if not isinstance(v, (dict, list)) else type(v).__name__, None))
        else:
            out.append((prefix, node, None))
        return out

    entries = walk(doc)
    print("===== tensors matching selector / markov / conv / projection / norm =====")
    pats = ("selector", "markov", "conv", "projection", "context", "feature", "final_norm", "codebook")
    hits = [(n, s, d) for (n, s, d) in entries if any(p in str(n).lower() for p in pats)]
    for name, shape, dtype in hits[:120]:
        print(f"  {name:<70} {shape} {dtype}")
    print(f"  ({len(hits)} matches)")

    print()
    print("===== ALL tensors whose name mentions 'dflash' (first 80) =====")
    hits2 = [(n, s, d) for (n, s, d) in entries if "dflash" in str(n).lower()]
    for name, shape, dtype in hits2[:80]:
        print(f"  {name:<70} {shape} {dtype}")
    print(f"  ({len(hits2)} matches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
