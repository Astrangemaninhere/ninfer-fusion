#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gguf_fold_back_bounded.py -- Route B, with a BOUNDED per-tensor working set.

WHY THIS FILE EXISTS, AND WHY IT IS NOT A SECOND IMPLEMENTATION OF THE GATE
--------------------------------------------------------------------------
`tools/convert/gguf_fold_back.py` streams tensor by tensor, but its working set for ONE tensor
is ~3.4x that tensor in fp32, and the ternary files' FIRST planned tensor is the 1,271,398,400-
element LM head / embedding pair.  MEASURED on this box (RTX 5090 D host, 21.5 GiB RAM):

    tools/convert/gguf_fold_back.py --src Ternary-Bonsai-2-27B-PTQ1_0.gguf --fold-route fold_back
      VmHWM  = 17,723,900 kB  (16.9 GiB)          <-- /proc/<pid>/status
      VmPeak = 21,948,052 kB  (20.9 GiB)
      MemAvailable fell to 81,340 kB ; 11 minutes of CPU, 0 payload bytes, `out` stationary

  The file's own docstring records the same thing measured by a peer line:
  "the first planned tensor (output.weight, 1,271,398,400 elements) does not complete at all in
   11.5 minutes of CPU (measured by the peer line: 0 bytes emitted, stationary thrash)."

  Where the 3.4x comes from, named: `fold_back_tensor` holds `rows` (a view of the decoded fp32,
  5.09 GB for this tensor), `primal` = `H.apply_inverse(rows)` (a second 5.09 GB), `back` =
  `H.apply_forward(primal)` (a third 5.09 GB), and `ternary_fraction` converts to **float64**
  twice (`np.asarray(w, dtype=np.float64)` then `np.abs(a)/d`), i.e. 2 x 10.2 GB of float64
  temporaries -- called on `rows` AND on `primal`.

WHAT THIS FILE CHANGES, AND WHAT IT DOES NOT
--------------------------------------------
CHANGED: the per-tensor traverse is split into ROW chunks.  Every one of the operations in
`fold_back_tensor` acts on the LAST axis (`axis=-1`) and on the width-sized row
(`ternary_fraction` reshapes `(-1, 128)`, the GDN reorder gathers within a row,
`_roundtrip_tol` is a per-tensor scale).  Rows are therefore independent, and chunking them
reproduces the arithmetic exactly -- provided a chunk boundary never splits a 128-block or a
1024-block, which is enforced here by construction and checked.

NOT CHANGED: the route decision (`gguf_fold_route.route_precheck` / `converter_admission`), the
ledger, the two guards inside `fold_back_tensor`, the `__metadata__` declaration, the config
builder and `OutputTransaction` are the TREE'S OWN, called as functions.  Nothing here re-derives
a verdict, and there is no flag that admits anything the tree's gate refuses.

ONE STRICTNESS DIFFERENCE, NAMED RATHER THAN HIDDEN
---------------------------------------------------
`fold_back_tensor` checks `max|A(A^-1 W) - W| <= atol + rtol * max|W|` with `max|W|` taken over
what it was handed.  Called per chunk, both the residual and the scale are CHUNK-LOCAL, so a
chunk is judged against a tolerance no LARGER than the whole-tensor one.  The check is therefore
STRICTER, never looser.  This file additionally reports the aggregate
`max_chunk_delta / (atol + rtol * global_max|W|)` so the whole-tensor statement is also on record.

A run of this file is accepted only if the tree's own post-condition, run inside the output
transaction on this run's ledger, admits it.  A partial traversal still refuses, by the same
predicate: the ledger must show every declared name unfolded AND verified.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert import gguf_extract as X          # noqa: E402
from tools.convert import gguf_fold_back as FB        # noqa: E402
from tools.convert import gguf_fold_route as FR       # noqa: E402
from tools.convert import gguf_hadamard as H          # noqa: E402
from tools.convert import gguf_kquant as K            # noqa: E402
from tools.convert import gguf_names as N             # noqa: E402
from tools.convert import unit_offset_norms        # noqa: E402  the +1 the epilogue adds

DEFAULT_CHUNK_BYTES = 192 << 20        # fp32 bytes of ONE chunk's working set


def block_bytes(ttype: int) -> int:
    """Bytes of one 128-value run of `ttype`, and the proof that the layout is linear in it."""
    n128 = K.tensor_nbytes(ttype, 128)
    n256 = K.tensor_nbytes(ttype, 256)
    if n256 != 2 * n128:
        raise FB.EmissionRefused(
            "type %s is not linear in 128-value runs (%d vs %d bytes); this bounded writer's "
            "chunking is only valid for linear layouts and it will not guess.  Nothing was "
            "written." % (K.type_name(ttype), n128, n256))
    return n128


def need_bytes_for(ttype: int, nelem: int) -> int:
    """`K.tensor_nbytes`, named so the flat single-chunk path reads clearly."""
    return K.tensor_nbytes(ttype, nelem)


def write_bounded(dest, src, data_offset, plan, route, declared, emitted, fwd, inv, by_width,
                  block, kv, ledger, *, chunk_bytes=DEFAULT_CHUNK_BYTES, log=None,
                  bounded_by=None):
    """`FB.write_safetensors_routeaware`'s output, produced with a bounded working set."""
    fold_names = set(fwd) | set(inv)
    # The file's own GDN geometry, when this run OWES the file's conventions to the emitted bytes.
    # Only `fold_back` on a `folded` source does: Route A copies the file as it stands and the
    # runtime owes every transform.
    gdn_geom = (FB.gdn_v_axis_geometry(kv)
                if (declared == FR.BASIS_FOLDED and route == FR.ROUTE_FOLD_BACK) else None)

    # ---- the header, byte for byte the same shape the stock writer emits
    header, offset = {}, 0
    for _name, key, shape, _ttype, _off, nelem, _dims in plan:
        nbytes = nelem * 2
        header[key] = {"dtype": "BF16", "shape": list(shape),
                       "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    header["__metadata__"] = {
        FR.BASIS_KEY: emitted,
        "prism.fold.route": route,
        "prism.fold.source.declared": declared,
        "prism.fold.gdn_head_order": FB.gdn_head_order_declaration(declared, emitted),
        "prism.fold.transform": str(kv.get(H.PREFIX + "transform", "")),
        "prism.fold.block_size": str(kv.get(H.PREFIX + "block_size", "")),
        "prism.fold.runtime_must_transform": "true" if emitted == FR.BASIS_FOLDED else "false",
        "prism.fold.bounded": "true" if bounded_by else "false",
        "prism.fold.bounded_by": str(bounded_by or ""),
        "prism.fold.tensors_written": str(len(plan)),
    }
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((-(8 + len(blob))) % 8)

    written = 0
    with open(src, "rb") as fh, open(dest, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        out.flush()
        for name, key, _shape, ttype, off, nelem, dims in plan:
            bb = block_bytes(ttype)                       # bytes per 128 values, linearity checked
            folded = (declared == FR.BASIS_FOLDED) and (name in fold_names)
            # The file's own v-axis head order, which the emitted bytes owe the consumer.  It is a
            # gather on the tensor's OUTER axis, so unlike the unfold it CANNOT be done chunk by
            # chunk here (a chunk is a contiguous run of rows and this gather is not), and such a
            # tensor is read in one piece with the bound asserted rather than hoped for.
            gperm = FB.gdn_v_axis_permutation(name, dims, gdn_geom)
            if folded:
                # ---- the FOLDED tensors: chunk by WHOLE ROWS of `width`.
                # The transform is per row over `axis=-1` and `ternary_fraction` reshapes
                # (-1, 128) over the WHOLE chunk, so a chunk must be a whole number of rows AND
                # its element count must be a multiple of 128.  Both hold automatically for the
                # folded set: the tree's `write_safetensors_routeaware` refuses any folded tensor
                # whose `dims[0]` is not a declared sign width, and every declared width
                # (5120 / 6144 / 17408) is a multiple of 1024.  Asserted below rather than assumed.
                width = int(dims[0])
                if width % 128 or width not in by_width:
                    raise FB.EmissionRefused(
                        "%s: folded tensor with ggml ne[0]=%d, which is not a declared sign width "
                        "that is a multiple of 128 (declared: %s).  This writer chunks folded "
                        "tensors by whole rows and will not guess a boundary inside a 128-block.  "
                        "Nothing was written." % (name, width, sorted(by_width)))
                if nelem % width:
                    raise FB.EmissionRefused(
                        "%s: %d elements is not a multiple of width %d.  Nothing was written."
                        % (name, nelem, width))
                rows = nelem // width
                per_row = bb * (width // 128)
                if per_row * 128 != bb * width:
                    raise FB.EmissionRefused("%s: non-linear width.  Nothing was written." % name)
                max_rows = max(1, int(chunk_bytes // (width * 4)))
                rows_per_chunk = max(1, (max_rows // 128) * 128)
                chunks = [(r0, min(rows, r0 + rows_per_chunk), r0 * per_row,
                           (min(rows, r0 + rows_per_chunk) - r0) * per_row,
                           (min(rows, r0 + rows_per_chunk) - r0) * width)
                          for r0 in range(0, rows, rows_per_chunk)]
            else:
                # ---- everything else is COPIED, so it has no row structure to respect at all;
                # chunk it flat, in whole 128-value units where the count allows, and in one piece
                # otherwise (those are the tiny ones: ssm_a is 48 values).  MEASURED: the previous
                # version computed a per-row stride for EVERY tensor and refused the whole file on
                # `blk.0.ssm_a` with "non-linear width", which is a defect of my planner and not a
                # property of the data.
                if nelem % 128 == 0:
                    # `bb` IS the number of bytes of one 128-value unit, so the byte offset of
                    # value v0 is (v0 // 128) * bb.  MEASURED DEFECT (found by THIS tool on the
                    # real file, at the LAST tensor): the first version wrote `per = bb * 128` and
                    # then multiplied by 128 AGAIN, so it asked for 128x the bytes.  For 850 of the
                    # 851 tensors that read silently ran PAST the tensor into its neighbours -- the
                    # values used were still right (an F32 decode takes the first `nval` elements),
                    # so every test that did not sit at the end of the file passed.  The 851st,
                    # `blk.63.post_attention_norm.weight`, is the last one on disk and it refused:
                    #     "range [0,5120) needs 2621440 bytes but the file ends after 20480"
                    # 5120 F32 values are 20,480 bytes; 2,621,440 is 128x that.
                    per = bb
                    per_chunk = max(128, (chunk_bytes // per) * 128)
                    chunks = [(v0, min(nelem, v0 + per_chunk), (v0 // 128) * per,
                               (min(nelem, v0 + per_chunk) - v0) // 128 * per,
                               min(nelem, v0 + per_chunk) - v0)
                              for v0 in range(0, nelem, per_chunk)]
                else:
                    chunks = [(0, nelem, 0, need_bytes_for(ttype, nelem), nelem)]

            if gperm is not None:
                if nelem > FB.GDN_PERMUTE_MAX_ELEMS:
                    raise FB.EmissionRefused(
                        "%s: %d elements must be re-indexed along its v axis in one piece and this "
                        "writer's bound for that is %d elements -- chunking it would reorder the "
                        "wrong rows, which is worse than not writing it.  Nothing was written."
                        % (name, nelem, FB.GDN_PERMUTE_MAX_ELEMS))
                chunks = [(0, nelem, 0, need_bytes_for(ttype, nelem), nelem)]

            delta_max = 0.0
            gmax = 0.0
            nch = 0
            for v0, v1, boff, bcnt, nval in chunks:
                fh.seek(data_offset + off + boff)
                raw = fh.read(bcnt)
                if len(raw) != bcnt:
                    raise FB.EmissionRefused(
                        "tensor %s range [%d,%d) needs %d bytes but the file ends after %d"
                        % (name, v0, v1, bcnt, len(raw)))
                fp32 = K.to_fp32(ttype, raw, nval)
                if not np.isfinite(fp32).all():
                    raise FB.EmissionRefused("tensor %s decoded to non-finite values" % name)
                if folded:
                    signs = by_width[width]
                    if route == FR.ROUTE_FOLD_BACK:
                        fp32, d = FB.fold_back_tensor(name, fp32, width, block, signs)
                        delta_max = max(delta_max, float(d))
                        arr = np.asarray(fp32, dtype=np.float32).reshape(-1, width)
                        gmax = max(gmax, float(np.abs(arr).max()))
                        nch += 1
                    else:
                        ledger.record(name, False, True, 0.0)
                if gperm is not None:
                    fp32 = FB.apply_gdn_v_axis_permutation(fp32, gperm, name)
                a_log = FB.gdn_a_log_from_stored(name, fp32, gdn_geom)
                if a_log is not None:
                    fp32 = a_log
                # The engine's rmsnorm epilogue adds one to this weight
                # (src/ops/kernel/rmsnorm.cuh:22) and the GDN control projection goes
                # through that epilogue with the flag set
                # (src/ops/generic/rowsplit_generic.cu:519), so the artifact must hold
                # `gamma - 1`.  A GGUF holds the plain `gamma`.  Set + why:
                # tools/convert/unit_offset_norms.py
                fp32 = unit_offset_norms.stored_values(fp32, key)
                out.write(K.fp32_to_bf16_bytes(fp32))
            if folded and route == FR.ROUTE_FOLD_BACK:
                ledger.record(name, True, True, delta_max)
                if log is not None:
                    log.write("  %-32s chunks=%-5d max_delta=%.3e global|W|max=%.6e\n"
                              % (name, nch, delta_max, gmax))
                    log.flush()
            written += 1
        out.flush()
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold-route", default=None, choices=list(FR.ROUTE_VALUES))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES,
                    help="fp32 bytes of one chunk's working set (default 192 MiB)")
    ap.add_argument("--only", default=None, metavar="REGEX",
                    help="DIAGNOSTIC, and the SAME semantics as the stock tool's flag: emit only "
                         "the matching tensors, narrow the ledger's expected set to that subset, "
                         "and declare prism.fold.bounded=true.  Used here for the A/B proof that "
                         "this writer reproduces the stock tool byte for byte on a real tensor.")
    ap.add_argument("--simulate-partial-fold", action="store_true",
                    help="TEST-ONLY RED CONTROL, identical in meaning to the stock tool's flag: "
                         "ask the post-condition about the WHOLE declared fold set while walking "
                         "a bounded subset, so the tree's own gate is shown to fire here too.  It "
                         "can only make the gate stricter.")
    args = ap.parse_args(argv)
    kv, tensors, data_offset = K.read_tensor_table(args.src)
    arch = str(kv.get("general.architecture") or "")
    wanted = [H.PREFIX + s for s in ("version", "block_size", "transform", "axis", "sign_mode",
                                     "sign_widths", "sign_values", "weight_names",
                                     "inverse_weight_names", "gdn_v_grouped")]
    wanted += ["qwen35.ssm.time_step_rank", "qwen35.ssm.group_count", "qwen35.ssm.inner_size",
               "qwen35.ssm.state_size", "general.architecture"]
    real = FB.read_metadata_values(args.src, wanted)
    kv = dict(kv)
    kv.update(real)
    if H.PREFIX + "version" in real:
        for key in wanted[:10]:
            if key not in real:
                raise FB.EmissionRefused(
                    "the file declares %sversion but its metadata has no %r.  Nothing was written."
                    % (H.PREFIX, key))

    declared = FR.declared_basis_of(kv)
    route = args.fold_route if args.fold_route is not None else FR.route_for_declared_basis(declared)

    rep = N.coverage(kv, tensors, arch)
    N.require_full_coverage(rep)

    plan, declared, emitted, fwd, inv, by_width, block, contract = FB.plan_tensors_routeaware(
        kv, tensors, arch, route, declared)

    bounded_by = None
    if args.only:
        try:
            rx = re.compile(args.only)
        except re.error as exc:
            raise FB.EmissionRefused("--only %r is not a regular expression: %s.  Nothing was "
                                     "written." % (args.only, exc)) from None
        before = len(plan)
        plan = [p for p in plan if rx.search(p[0])]
        if not plan:
            raise FB.EmissionRefused(
                "--only %r matched none of the %d planned tensors; a bounded run that bounds "
                "itself to nothing is a mistake, not a result.  Nothing was written."
                % (args.only, before))
        bounded_by = args.only
        print("*** BOUNDED DIAGNOSTIC RUN: --only %r selects %d of %d tensors. ***"
              % (args.only, len(plan), before))
        print("*** The emitted checkpoint declares prism.fold.bounded=true. ***")

    fold_set = sorted(set(fwd) | set(inv))
    if bounded_by is not None:
        fold_set = [n for n in fold_set if any(p[0] == n for p in plan)]
    if args.simulate_partial_fold:
        fold_set = sorted(set(fwd) | set(inv))
        print("*** RED CONTROL: --simulate-partial-fold asks the gate about all %d declared "
              "tensors while walking %d.  The post-condition MUST refuse. ***"
              % (len(fold_set), len(plan)))
    ledger = FR.FoldLedger(expected_names=fold_set)

    FR.route_precheck(route, declared, emitted)          # the tree's early gate, unchanged

    print("source  : %s" % args.src)
    print("arch    : %s  layers %d main + %d draft"
          % (arch, rep["main_layers"], rep["nextn_layers"]))
    print("declared: %s   route: %s   emitted declaration: %s%s"
          % (declared, route, emitted,
             "" if args.fold_route is None else
             "  (--fold-route=%s was supplied on this command line)" % route))
    print("fold set: %d forward + %d inverse = %d" % (len(fwd), len(inv), len(fwd) + len(inv)))
    print("chunk   : %d MiB fp32 per chunk (bounded writer)" % (args.chunk_bytes >> 20))
    print("precheck: admitted (the request and the declarations agree); the post-condition runs "
          "inside the output transaction and reads the ledger")

    out_dir = Path(args.out)
    created_dir = not out_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)
    have_head = any(n == "output.weight" for n, _d, _t, _o in tensors)
    ledger_log = open(str(out_dir.parent / (out_dir.name + ".ledger.txt")), "w")
    with X.OutputTransaction(out_dir, created_dir) as txn:
        written = write_bounded(txn.staging("model.safetensors"), args.src, data_offset, plan,
                               route, declared, emitted, fwd, inv, by_width, block, kv, ledger,
                               chunk_bytes=args.chunk_bytes, log=ledger_log,
                               bounded_by=bounded_by)
        txn.staging("config.json").write_text(
            json.dumps(X.config_from_metadata(
                kv, arch, have_head,
                fold_declaration=FB.basis_declaration(route, declared, emitted)),
                ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            FR.converter_admission(route, declared, ledger, emitted)
        except FR.FoldRouteRefused as exc:
            raise FB.EmissionRefused("%s  Nothing was written.  %s" % (exc, ledger.render())) from None
    ledger_log.close()
    if args.report:
        print(ledger.render())
    print("extracted %d/%d tensors -> %s" % (written, len(plan), out_dir))
    print("basis   : %s=%s  runtime_must_transform=%s"
          % (FR.BASIS_KEY, emitted, emitted == FR.BASIS_FOLDED))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FB.EmissionRefused, X.ExtractionRefused, N.UnmappedTensors, FR.FoldRouteRefused,
            H.UnreadableMetadata, H.UnprovenRotation, ValueError) as exc:
        text = str(exc)
        if "Nothing was written." not in text:
            text = text + "  Nothing was written."
        print("REFUSED: %s" % text, file=sys.stderr)
        raise SystemExit(3)
