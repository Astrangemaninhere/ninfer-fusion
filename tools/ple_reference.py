"""Independent reference for the Qwen4Exp PLE n-gram row derivation.

Implements the formula from llama.cpp qwen4exp.cpp (PR #27742) directly,
in Python with explicit uint64 masking so it cannot silently share a bug
with the C++ implementation. Also generates a synthetic sidecar fixture
(manifest + 4 data files) for the C++ selftest.

Usage:
  python ple_reference.py gen <outdir>          # write synthetic fixture
  python ple_reference.py derive <manifest>     # print rows for a sample
"""

import json
import os
import struct
import sys

M = [23703573157769, 20109073645365, 8052911324071]
HEADS_PER_NGRAM = 8
NGRAM = 3
N_HEADS = 16
ROW_DIM = 160
ROW_STRIDE = 320  # BF16 x 160
PARTS = 128
FILES = 4
ROWS_PER_HEAD = 2000  # synthetic: per-head vocab size
TOTAL_ROWS = ROWS_PER_HEAD * N_HEADS  # 32000
ROWS_PER_PART = TOTAL_ROWS // PARTS  # 250
MASK64 = (1 << 64) - 1


def derive_rows(tokens, prevs, eos, vocab_sizes, offsets, ngram=NGRAM, per_gram=HEADS_PER_NGRAM):
    n_prev = ngram - 1
    out = []
    for i, tok in enumerate(tokens):
        ctx = [tok, eos, eos]
        cut = False
        for s in range(1, ngram):
            t = prevs[i * n_prev + (s - 1)] if prevs else eos
            cut = cut or t < 0 or t == eos
            ctx[s] = eos if cut else t
        row = [0] * N_HEADS
        for n in range(2, ngram + 1):
            mixed = (ctx[0] * M[0]) & MASK64
            for j in range(1, n):
                mixed ^= (ctx[j] * M[j]) & MASK64
            base = (n - 2) * per_gram
            for g in range(per_gram):
                h = base + g
                row[h] = (mixed % vocab_sizes[h]) + offsets[h]
        out.append(row)
    return out


def gen_fixture(outdir):
    os.makedirs(outdir, exist_ok=True)
    vocab_sizes = [ROWS_PER_HEAD] * N_HEADS
    offsets = [h * ROWS_PER_HEAD for h in range(N_HEADS)]
    # 128 logical parts spread over 4 files, 250 rows each.
    parts, files = [], []
    for f in range(FILES):
        fbytes = 0
        for p in range(PARTS // FILES):
            part = f * (PARTS // FILES) + p
            parts.append({
                "logical_part": part,
                "physical_file_index": f,
                "global_row_start": ROWS_PER_PART * part,
                "file_offset": ROWS_PER_PART * p * ROW_STRIDE,
                "rows": ROWS_PER_PART,
                "payload_bytes": ROWS_PER_PART * ROW_STRIDE,
                "physical_file": f"ple/ple-bf16-{f + 1:05d}-of-{FILES:05d}.bin",
                "embedding_row_dimension": ROW_DIM,
                "row_stride_bytes": ROW_STRIDE,
            })
            fbytes += ROWS_PER_PART * ROW_STRIDE
        files.append({
            "index": f,
            "path": f"ple/ple-bf16-{f + 1:05d}-of-{FILES:05d}.bin",
            "file_bytes": fbytes,
            "payload_bytes": fbytes,
        })
    manifest = {
        "format_version": 1,
        "ngram_size": NGRAM,
        "heads_per_ngram": HEADS_PER_NGRAM,
        "number_of_ngram_heads": N_HEADS,
        "embedding_row_dimension": ROW_DIM,
        "row_stride_bytes": ROW_STRIDE,
        "padded_vocabulary_rows": TOTAL_ROWS,
        "usable_vocabulary_rows": TOTAL_ROWS,
        "total_parameter_count": TOTAL_ROWS * ROW_DIM,
        "alignment_bytes": 4096,
        "layer_multipliers": M,
        "per_head_offsets": offsets,
        "per_head_vocabulary_sizes": vocab_sizes,
        "logical_parts": parts,
        "physical_files": files,
        "storage_dtype": "BF16",
        "storage": "ssd_backed_bounded_page_cache",
    }
    os.makedirs(os.path.join(outdir, "ple"), exist_ok=True)
    with open(os.path.join(outdir, "ple-manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    # Data: element j of row i encodes i in the low 16 bits of each half.
    for f in range(FILES):
        rows = TOTAL_ROWS // FILES
        buf = bytearray()
        for i in range(rows):
            row_id = f * rows + i
            pattern = struct.pack("<H", row_id & 0xFFFF) * ROW_DIM
            buf += pattern
        with open(os.path.join(outdir, files[f]["path"]), "wb") as fh:
            fh.write(buf)
    print("fixture written to", outdir)


def main():
    cmd = sys.argv[1]
    if cmd == "gen":
        gen_fixture(sys.argv[2])
    elif cmd == "derive":
        with open(sys.argv[2]) as fh:
            m = json.load(fh)
        tokens = [5, 7, 7, 9, 9, 9, 100, 200, 300]
        eos = 151643  # use a value not colliding with the tokens above
        prevs = [eos] * (len(tokens) * (NGRAM - 1))
        rows = derive_rows(tokens, prevs, eos, m["per_head_vocabulary_sizes"], m["per_head_offsets"])
        for t, r in zip(tokens, rows):
            print(t, r)
    else:
        print("unknown command")


if __name__ == "__main__":
    main()
