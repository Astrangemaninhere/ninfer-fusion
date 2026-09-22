#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a real GGUF (`general.architecture = qwen35`) into a qwen3_5_9b artifact.

    python3 -m tools.convert.qwen3_5_9b.convert \
        --gguf  /path/Ornith-1.5-9B-Q4_K_M.gguf \
        --resources /path/to/six-resource-dir \
        --out out/qwen3_5_9b.ninfer

Why this exists
---------------
`tools/convert/gguf_extract.py` writes an intermediate bf16 safetensors and the
registry's converters read that.  Measured on the real file, that intermediate is
the reason the conversion never finished here: the bf16 model is ~18 GB and the
per-tensor path peaked at 12-14 GB of RAM on a 22 GB machine (n7land REPORT.md
section 5.3).  This converter therefore does **not** materialise it.  It reads the
GGUF directly and streams **row blocks**, so the largest live allocation is one
object payload (largest here: `text/token_embedding`, 795 MB as Q6) plus one row
chunk, instead of 14 GB.

Design notes that are load-bearing
----------------------------------
* GGUF `ne = (fastest, ..., outermost)` is the reverse of the HF
  `(out_features, in_features)` convention and the row-major payload is unchanged
  by the reversal, so one HF row is one contiguous GGUF run of `k` elements -- a
  byte range.  No transposition happens anywhere; `inventory.plan_tensors` states
  every object as row ranges of named GGUF tensors.
* A K-quant row is a whole number of 256-element super-blocks, so a row range is
  readable and dequantisable independently.  That is what makes row streaming
  exact rather than approximate.
* Row-split payloads are three planes (base, high, scale) whose row strides are in
  `row_split_geometry`, so a chunk of rows can be written straight into the final
  payload buffer at its plane offset.  `--self-test` proves that chunked encoding
  is byte-identical to the whole-matrix encoder on a control matrix; without that
  control the chunking would be an unverified reimplementation.
* MTP is DECLARED BY THE SOURCE, not required of it.  A GGUF whose
  `qwen35.nextn_predict_layers` is absent or 0 has no draft block, so its artifact carries no
  `mtp/*` object at all and its identity is `inventory.WEIGHTS_ID_NO_MTP` instead of
  `inventory.WEIGHTS_ID`.  Both products are complete for what they were written from; the
  identity is the statement the engine's own backend decision reads, so `--spec auto` cannot
  resolve to a backend the artifact has no objects for.
"""
from __future__ import annotations

import argparse
import traceback
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools import convert as _convert_pkg  # noqa: F401,E402  (package import side effects)
from tools.artifact import numeric as A_numeric                                    # noqa: E402
from tools.artifact import layouts as A_layouts                                    # noqa: E402
from tools.artifact.container import (                                             # noqa: E402
    ArtifactIdentity,
    ArtifactWriter,
    ResourceSpec as WriterResourceSpec,
    TensorSpec as WriterTensorSpec,
)
from tools.convert import gguf_kquant as K                                         # noqa: E402
from tools.convert import unit_offset_norms                                     # noqa: E402
from tools.convert.common.quantize import quantize_and_encode, quantize_matrix     # noqa: E402
from tools.convert.qwen3_5_9b import inventory as inv                              # noqa: E402

if not hasattr(A_layouts, "_pack_codes"):
    raise ImportError(
        "tools.artifact.layouts._pack_codes is gone: the row-streamed encoder packs code "
        "groups with it, and without it the streaming path would have to be reimplemented "
        "(which is exactly what the --self-test control exists to prevent)")


#: Rows read and quantised per step.  Bounds the live fp32/torch allocation; the
#: object payload being written is not affected by this value.
DEFAULT_CHUNK_ROWS = 2048


class ConversionRefused(RuntimeError):
    """Raised with the missing or inconsistent part named."""


class GgufReader:
    """Row-addressed reads out of a GGUF tensor table (one pread per row range)."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.kv, self.tensors, self.data_offset = K.read_tensor_table(self.path)
        self.by_name = {n: (tuple(d), int(t), int(o)) for n, d, t, o in self.tensors}
        self._fh = open(self.path, "rb")

    def close(self) -> None:
        self._fh.close()

    def n_rows(self, name: str) -> int:
        dims, _t, _o = self.by_name[name]
        return int(dims[-1]) if len(dims) > 1 else 1

    def row_bytes(self, name: str, k: int) -> int:
        _dims, ttype, _o = self.by_name[name]
        return int(K.tensor_nbytes(ttype, k))

    def read_rows(self, name: str, r0: int, r1: int, k: int) -> torch.Tensor:
        dims, ttype, off = self.by_name[name]
        if len(dims) < 2:
            raise ConversionRefused("tensor %s is not rank >= 2; cannot row-stream it" % name)
        if int(dims[0]) != k:
            raise ConversionRefused(
                "tensor %s has fastest dim %d but the object declares k=%d"
                % (name, int(dims[0]), k))
        try:
            rb = self.row_bytes(name, k)
        except (KeyError, ValueError) as exc:
            raise ConversionRefused(
                "tensor %s cannot be read in row units of %d elements: %s" % (name, k, exc))
        self._fh.seek(self.data_offset + off + r0 * rb)
        raw = self._fh.read((r1 - r0) * rb)
        if len(raw) != (r1 - r0) * rb:
            raise ConversionRefused("tensor %s is truncated at rows [%d,%d)" % (name, r0, r1))
        fp32 = K.to_fp32(ttype, raw, (r1 - r0) * k)
        return torch.from_numpy(np.ascontiguousarray(fp32)).reshape(r1 - r0, k)

    def read_flat(self, name: str, count: int) -> torch.Tensor:
        dims, ttype, off = self.by_name[name]
        total = 1
        for d in dims:
            total *= int(d)
        if total != count:
            raise ConversionRefused("tensor %s has %d elements, object declares %d"
                                    % (name, total, count))
        need = int(K.tensor_nbytes(ttype, total))
        self._fh.seek(self.data_offset + off)
        raw = self._fh.read(need)
        if len(raw) != need:
            raise ConversionRefused("tensor %s is truncated" % name)
        return torch.from_numpy(np.ascontiguousarray(K.to_fp32(ttype, raw, total)))


def _concat_rows(spec: inv.TensorSpec, reader: GgufReader, r0: int, r1: int, k: int):
    parts = []
    cursor = 0
    for name, base_row, count in spec.sources:
        s = max(r0 - cursor, 0)
        e = min(r1 - cursor, count)
        if s < e:
            parts.append(reader.read_rows(name, base_row + s, base_row + e, k))
        cursor += count
    if cursor != spec.shape[0]:
        raise ConversionRefused("object %s: sources cover %d rows, shape says %d"
                               % (spec.name, cursor, spec.shape[0]))
    if not parts:
        raise ConversionRefused("object %s: empty row range [%d,%d)" % (spec.name, r0, r1))
    return torch.cat(parts, dim=0)


def encode_quantized_chunks(spec: inv.TensorSpec, reader: GgufReader, chunk_rows: int):
    """Yield one quantized object's payload in file order, holding one row chunk at a time.

    The payload is three planes in a fixed order (base, then high, then the fp16 scale
    plane), each row-contiguous at a known offset.  Because the plane order is fixed,
    the payload can be produced plane-by-plane rather than assembled in memory: the
    first run of this converter held the whole payload (795 MB for
    ``text/token_embedding``) and died with ``OSError: [Errno 12] Cannot allocate
    memory`` on a machine that had 795 MB free.  Streaming it removes that allocation
    entirely, at the cost of re-reading and re-quantising the rows once per plane.
    """
    spec_fmt = A_numeric.get_format(spec.format)
    n, k = spec.shape
    geometry = A_layouts.row_split_geometry(spec_fmt, spec.shape)
    if geometry.groups_per_row * spec_fmt.group_size != geometry.k_pad:
        raise ConversionRefused("object %s: geometry groups do not cover k_pad" % spec.name)
    planes = (
        ("base", geometry.base_offset, geometry.base_bytes, geometry.base_row_bytes),
        ("high", geometry.high_offset, geometry.high_bytes, geometry.high_row_bytes),
        ("scale", geometry.scale_offset, geometry.scale_bytes, geometry.scale_row_bytes),
    )
    cursor = 0
    for plane, offset, nbytes, row_bytes in planes:
        if nbytes == 0:
            continue
        if row_bytes <= 0:
            raise ConversionRefused("object %s: plane %s has bytes but no row stride"
                                    % (spec.name, plane))
        if offset < cursor:
            raise ConversionRefused("object %s: plane %s overlaps the previous plane"
                                    % (spec.name, plane))
        if offset > cursor:
            yield bytes(offset - cursor)
        for r0 in range(0, n, chunk_rows):
            r1 = min(n, r0 + chunk_rows)
            logical = _concat_rows(spec, reader, r0, r1, k)
            quantized = quantize_matrix(logical, spec_fmt, device="cpu")
            rows = r1 - r0
            if plane == "scale":
                part = quantized.scales.contiguous().view(torch.uint8).reshape(rows, row_bytes)
            else:
                grouped = quantized.codes.reshape(-1, spec_fmt.group_size)
                base, high = A_layouts._pack_codes(grouped, spec_fmt)
                source = base if plane == "base" else high
                part = source.reshape(rows, row_bytes)
            yield part.numpy().tobytes()
            del logical, quantized, part
        cursor = offset + nbytes
    if cursor > geometry.payload_bytes:
        raise ConversionRefused("object %s: plane layout exceeds payload_bytes" % spec.name)
    if cursor < geometry.payload_bytes:
        yield bytes(geometry.payload_bytes - cursor)


def encode_quantized(spec: inv.TensorSpec, reader: GgufReader, chunk_rows: int) -> bytes:
    """The same payload as one buffer.  Only the self-test uses this form."""
    return b"".join(encode_quantized_chunks(spec, reader, chunk_rows))


def encode_direct(spec: inv.TensorSpec, reader: GgufReader) -> bytes:
    count = 1
    for dim in spec.shape:
        count *= dim
    if len(spec.sources) != 1:
        raise ConversionRefused("direct object %s must have exactly one source" % spec.name)
    flat = reader.read_flat(spec.sources[0][0], count).reshape(spec.shape)
    # The engine's rmsnorm epilogue adds one to this weight
    # (src/ops/kernel/rmsnorm.cuh:22) and the GDN control projection goes through that
    # epilogue with the flag set (src/ops/generic/rowsplit_generic.cu:519), so the
    # artifact must hold `gamma - 1`.  The GGUF holds the plain `gamma`.
    # Set + why: tools/convert/unit_offset_norms.py
    flat = unit_offset_norms.stored_values(flat, spec.name)
    if spec.format == inv.BF16:
        return A_layouts.encode_direct(flat.to(torch.bfloat16), "BF16")
    if spec.format == inv.FP32:
        return A_layouts.encode_direct(flat.to(torch.float32), "FP32")
    raise ConversionRefused("object %s has unsupported direct format %s"
                            % (spec.name, spec.format))


def writer_spec(spec: inv.TensorSpec):
    layout = (A_layouts.CONTIGUOUS_LE_V1.name if not spec.is_quantized
              else A_layouts.ROW_SPLIT_K128_V1.name)
    return WriterTensorSpec(name=spec.name, shape=tuple(spec.shape), format=spec.format,
                            layout=layout)


# ---------------------------------------------------------------------------
# Self-test: the chunked encoder must equal the library's whole-matrix encoder.
# ---------------------------------------------------------------------------
def self_test() -> int:
    """Control for the row streaming: chunked == whole, and a wrong chunk is caught."""
    torch.manual_seed(1234)
    failures = 0
    for n, k, fmt in ((9, 4096, inv.Q4), (9, 4096, inv.Q5), (9, 4096, inv.Q6),
                      (17, 12288, inv.Q4), (5, 4096, inv.Q5)):
        # A control matrix whose groups are far from uniform, so a row/chunk boundary
        # bug cannot hide behind a smooth tensor.
        weight = torch.randn(n, k, dtype=torch.float32) * 0.05
        weight[:, ::64] *= 7.0
        whole = quantize_and_encode(weight, fmt, device="cpu")

        class _Reader:
            def read_rows(self, name, r0, r1, kk):
                return weight[r0:r1].clone()

            def read_flat(self, name, count):
                return weight.reshape(-1)[:count].clone()

        spec = inv.TensorSpec("self-test", (n, k), fmt, (("control", 0, n),))
        chunks = list(encode_quantized_chunks(spec, _Reader(), 4))
        chunked = b"".join(chunks)
        want_bytes = A_layouts.encoded_size(A_layouts.ROW_SPLIT_K128_V1.name, fmt, (n, k))
        ok = chunked == whole
        lengths_ok = sum(len(c) for c in chunks) == want_bytes == len(whole)
        print("  %-14s n=%-3d k=%-6d chunked==whole : %s (%d B, %d chunks, lengths %s)"
              % (fmt, n, k, "IDENTICAL" if ok else "DIFFERENT", len(whole), len(chunks),
                 "OK" if lengths_ok else "MISMATCH"))
        failures += 0 if (ok and lengths_ok) else 1

        # Negative control: perturb one row and require the comparison to notice.
        class _Reader2(_Reader):
            def read_rows(self, name, r0, r1, kk):
                block = weight[r0:r1].clone()
                if r0 <= 4 < r1:
                    block[4 - r0, 0] = block[4 - r0, 0] + 1.0
                return block

        changed = encode_quantized(spec, _Reader2(), 4)
        detected = changed != whole
        print("  %-14s n=%-3d k=%-6d one-row perturbation detected : %s"
              % (fmt, n, k, "YES" if detected else "NO (control cannot fail!)"))
        failures += 0 if detected else 1
    print("self-test failures: %d" % failures)
    return failures


def resource_specs(directory: Path):
    specs, payloads = [], {}
    for name in inv.RESOURCE_NAMES:
        path = directory / Path(name).name
        if not path.is_file():
            raise ConversionRefused("frontend resource %s is absent from %s"
                                    % (Path(name).name, directory))
        data = path.read_bytes()
        if not data:
            raise ConversionRefused("frontend resource %s is empty" % path)
        specs.append(WriterResourceSpec(name=name, encoding="raw-bytes-v1", bytes=len(data)))
        payloads[name] = data
    return specs, payloads


def convert(gguf: str | Path, out: str | Path, resources: str | Path,
            *, chunk_rows: int = DEFAULT_CHUNK_ROWS,
            limit_layers: int | None = None) -> dict:
    started = time.perf_counter()
    reader = GgufReader(gguf)
    try:
        arch = str(reader.kv.get("general.architecture") or "")
        if arch != "qwen35":
            raise ConversionRefused(
                "this converter claims only GGUF `general.architecture = qwen35`; the file "
                "declares %r.  A different architecture is a different target, not a "
                "different invocation." % arch)
        # The draft block is DECLARED BY THE SOURCE, not required of it.  A GGUF with
        # `nextn_predict_layers` absent or 0 has no `blk.<LAYERS>.nextn.*` tensor and no draft
        # layer to convert, so its artifact carries no `mtp/*` object and its identity says so.
        # Two declarations are accepted, and every other pair is refused by naming both numbers:
        #   (LAYERS, 0)                             -> this source has no draft block
        #   (LAYERS + MTP_LAYERS, MTP_LAYERS)       -> this source declares one (today's path)
        block_count = int(reader.kv.get("qwen35.block_count") or 0)
        nextn = int(reader.kv.get("qwen35.nextn_predict_layers") or 0)
        with_mtp = nextn > 0
        accepted_geometry = ((inv.LAYERS, 0), (inv.LAYERS + inv.MTP_LAYERS, inv.MTP_LAYERS))
        if (block_count, nextn) not in accepted_geometry:
            raise ConversionRefused(
                "geometry mismatch: the file declares block_count=%d nextn=%d; this inventory "
                "is written for %d main layers with no draft block, or %d main + %d draft"
                % (block_count, nextn, inv.LAYERS, inv.LAYERS, inv.MTP_LAYERS))
        weights_id = inv.WEIGHTS_ID if with_mtp else inv.WEIGHTS_ID_NO_MTP
        for key, want in (("qwen35.embedding_length", inv.HIDDEN),
                          ("qwen35.feed_forward_length", inv.INTERMEDIATE),
                          ("qwen35.attention.head_count", inv.QUERY_HEADS),
                          ("qwen35.attention.head_count_kv", inv.KV_HEADS),
                          ("qwen35.attention.key_length", inv.HEAD_DIM),
                          ("qwen35.ssm.group_count", inv.GDN_KEY_HEADS),
                          ("qwen35.ssm.time_step_rank", inv.GDN_VALUE_HEADS),
                          ("qwen35.ssm.state_size", inv.GDN_KEY_HEAD_DIM)):
            got = int(reader.kv.get(key) or 0)
            if got != want:
                raise ConversionRefused("geometry mismatch %s: file %d, inventory %d"
                                        % (key, got, want))
        have = set(reader.by_name)
        if "token_embd.weight" not in have or "output.weight" not in have:
            raise ConversionRefused("token_embd.weight / output.weight absent: not a decoder GGUF")

        only = None if limit_layers is None else tuple(range(int(limit_layers)))
        tensor_specs = inv.plan_tensors(only, with_mtp=with_mtp)
        missing = sorted({s[0] for t in tensor_specs for s in t.sources} - have)
        if missing:
            raise ConversionRefused("GGUF is missing %d tensor(s) the inventory needs: %s"
                                    % (len(missing), ", ".join(missing[:6])))
        # The declaration and the tensor table must not disagree.  A file that says
        # `nextn_predict_layers` is absent while carrying `blk.<LAYERS>.nextn.*` would convert
        # into an artifact that silently drops a block the source has.
        stray_nextn = sorted(n for n in have if n.startswith("blk.%d.nextn." % inv.LAYERS))
        if not with_mtp and stray_nextn:
            raise ConversionRefused(
                "the file declares no draft block (nextn_predict_layers=%d) but carries %d "
                "nextn tensor(s) at blk.%d: %s"
                % (nextn, len(stray_nextn), inv.LAYERS, ", ".join(stray_nextn[:4])))

        rspecs, rpayloads = resource_specs(Path(resources))
        specs = list(rspecs) + [writer_spec(t) for t in tensor_specs]
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print("converting %s" % gguf)
        print("  arch=%s main_layers=%d draft_blocks=%d objects=%d (%d resources + %d tensors)"
              % (arch, inv.LAYERS, inv.MTP_LAYERS if with_mtp else 0, len(specs), len(rspecs),
                 len(tensor_specs)))
        print("  draft block: %s"
              % ("declared by the source (qwen35.nextn_predict_layers=%d, block_count=%d)"
                 " -> the 12 mtp/* objects are written" % (nextn, block_count) if with_mtp
                 else "ABSENT (qwen35.nextn_predict_layers=%d, block_count=%d) -> this artifact"
                 " carries no mtp/* object; identity %s" % (nextn, block_count, weights_id)))
        if limit_layers is not None:
            print("  LIMIT: only layers %s -> this artifact is deliberately incomplete and the "
                  "engine must refuse it by naming the first absent object" % (only,))

        written = []
        with ArtifactWriter(out_path, ArtifactIdentity(inv.MODEL_ID, weights_id), specs) as w:
            for spec in rspecs:
                w.write(spec.name, rpayloads[spec.name])
            for index, tspec in enumerate(tensor_specs, start=1):
                if tspec.is_quantized:
                    chunks = encode_quantized_chunks(tspec, reader, chunk_rows)
                else:
                    chunks = (encode_direct(tspec, reader),)
                w.write(tspec.name, chunks)
                if tspec.is_quantized:
                    nbytes = A_layouts.encoded_size(
                        A_layouts.ROW_SPLIT_K128_V1.name, tspec.format, tspec.shape)
                else:
                    nbytes = A_layouts.encoded_size(
                        A_layouts.CONTIGUOUS_LE_V1.name, tspec.format, tspec.shape)
                written.append({"name": tspec.name, "shape": list(tspec.shape),
                                "format": tspec.format, "bytes": nbytes})
                if index % 25 == 0 or index == len(tensor_specs):
                    print("  [%d/%d] %-58s %s (%d B)"
                          % (index, len(tensor_specs), tspec.name, tspec.format,
                             nbytes), flush=True)
        elapsed = time.perf_counter() - started
        report = {
            "recipe_id": inv.RECIPE_ID,
            "identity": {"model_id": inv.MODEL_ID, "weights_id": weights_id},
            "target_key": inv.TARGET_KEY,
            "source_gguf": str(gguf),
            "gguf_architecture": arch,
            "gguf_block_count": block_count,
            "main_layers": inv.LAYERS,
            "draft_blocks": inv.MTP_LAYERS if with_mtp else 0,
            # The artifact's own statement about the draft block.  The object table carries it
            # too -- the 12 names below are present or absent together -- and the identity
            # carries the flavour; this report is the third, human-readable copy of the fact,
            # which is why a reader can answer "does this product have MTP?" without opening
            # the multi-gigabyte container.
            "mtp": {
                "declared": with_mtp,
                "source_nextn_predict_layers": nextn,
                "source_block_count": block_count,
                "objects": list(inv.MTP_OBJECT_NAMES) if with_mtp else [],
                "weights_id": weights_id,
            },
            "limit_layers": None if only is None else list(only),
            "out_path": str(out_path),
            "out_bytes": out_path.stat().st_size,
            "elapsed_seconds": elapsed,
            "chunk_rows": chunk_rows,
            "frontend_resources": {
                n: hashlib.sha256(rpayloads[n]).hexdigest() for n in inv.RESOURCE_NAMES
            },
            "objects": written,
        }
        report_path = Path(str(out_path) + ".conversion.json")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print("complete: %d bytes in %.1fs -> %s" % (report["out_bytes"], elapsed, out_path))
        print("report: %s" % report_path)
        return report
    finally:
        reader.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=False)
    ap.add_argument("--out", required=False)
    ap.add_argument("--resources", required=False)
    ap.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    ap.add_argument("--limit-layers", type=int, default=None,
                    help="write only the first N main layers (deliberately incomplete artifact)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 1 if self_test() else 0
    if not (args.gguf and args.out and args.resources):
        print("--gguf, --out and --resources are all required (or pass --self-test)",
              file=sys.stderr)
        return 2
    try:
        convert(args.gguf, args.out, args.resources, chunk_rows=args.chunk_rows,
                limit_layers=args.limit_layers)
    except (ConversionRefused, ValueError, KeyError, OSError) as exc:
        # An OSError here is usually [Errno 12] ENOMEM from some allocation deep inside
        # the writer.  Printing str(exc) alone deletes the ONLY evidence of which
        # allocation failed, and it deletes it at exactly the moment it matters: the
        # frame chain is what tells a machine-OOM apart from a converter bug.
        print("REFUSED: %s" % exc, file=sys.stderr)
        traceback.print_exc()
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
