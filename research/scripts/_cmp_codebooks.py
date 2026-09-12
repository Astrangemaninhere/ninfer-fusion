#!/usr/bin/env python3
"""Compare the .ninfer selector tables against the HF checkpoint, byte for byte.

If the artifact conversion altered the selector codebooks (or the hidden projection),
ninfer alone would mis-score edges — while any engine loading the HF checkpoint directly
(vLLM / 1Cat) would be fine. That is exactly the asymmetry we observe: the unary (from the
shared LM head) produces plausible tokens, while the edge term drives the walk to rare
ones. safetensors is parsed directly so no torch is needed.
"""

import json
import pathlib
import struct
import sys

ART = "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer"
HF_DIR = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-NVFP4-RTX5090")

WANT = [
    "dflash2/candidate_selector/predecessor_codebook",
    "dflash2/candidate_selector/successor_codebook",
    "dflash2/candidate_selector/hidden_projection",
    "dflash2/feature_projection",
    "dflash2/context_norm",
    "dflash2/final_norm",
    "dflash2/layers/0/attention/context_key",
]


def artifact_index():
    with open(ART, "rb") as handle:
        handle.read(8)
        (length,) = struct.unpack("<Q", handle.read(8))
        payload_start = 16 + length
        payload_start = (payload_start + 4095) // 4096 * 4096
        doc = json.loads(handle.read(length).decode("utf-8"))
    index = {}
    for item in doc["objects"]:
        if item.get("kind") == "tensor":
            index[item["name"]] = item
    return index, payload_start


def hf_index():
    index = {}
    for path in sorted(HF_DIR.glob("*.safetensors")):
        with open(path, "rb") as handle:
            (header_len,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(header_len).decode("utf-8"))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            index[name] = (path, 8 + header_len + meta["data_offsets"][0],
                           meta["data_offsets"][1] - meta["data_offsets"][0], meta["dtype"],
                           meta["shape"])
    return index


def read_artifact_tensor(item, payload_start, limit_mb=8):
    size = item.get("bytes", 0)
    take = min(size, limit_mb * 1024 * 1024)
    with open(ART, "rb") as handle:
        handle.seek(payload_start + item["offset"])
        return handle.read(take), size


def read_hf_tensor(path, offset, size, limit_mb=8):
    take = min(size, limit_mb * 1024 * 1024)
    with open(path, "rb") as handle:
        handle.seek(offset)
        return handle.read(take), size


def main() -> int:
    art, payload_start = artifact_index()
    hf = hf_index()
    print(f"artifact tensors: {len(art)}   hf tensors: {len(hf)}")
    print()
    if not hf:
        print("no HF safetensors found in", HF_DIR)
        return 1
    for name in WANT:
        a_item = art.get(name)
        h_name = name
        h_entry = hf.get(h_name)
        if h_entry is None:
            cands = [k for k in hf if name.split("/")[-1] in k]
            h_entry = hf.get(cands[0]) if cands else None
            h_name = cands[0] if cands else name
        if a_item is None or h_entry is None:
            print(f"  {name:<52} MISSING art={a_item is not None} hf={h_entry is not None}")
            continue
        a_bytes, a_size = read_artifact_tensor(a_item, payload_start)
        h_bytes, h_size = read_hf_tensor(*h_entry[:3])
        same = a_bytes == h_bytes
        note = "IDENTICAL" if same else "DIFFER"
        print(f"  {name:<52} art={a_size:>10}B hf={h_size:>10}B {h_entry[3]} "
              f"{h_entry[4]}  {note}")
        if not same and a_bytes and h_bytes:
            n = min(len(a_bytes), len(h_bytes))
            first = next((i for i in range(n) if a_bytes[i] != h_bytes[i]), n)
            print(f"      first byte difference at {first} "
                  f"(art={a_bytes[first:first+8].hex()} hf={h_bytes[first:first+8].hex()})")
            if first >= 2:
                a0 = struct.unpack_from("<e", a_bytes, 0)[0] if first >= 2 else None
                print(f"      art[0:4]={a_bytes[0:4].hex()}  hf[0:4]={h_bytes[0:4].hex()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
