#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gguf_extract.py — GGUF -> bf16 safetensors for the qwen3.5/3.6/3.8 text family.

Reads the container, name-maps the tensor table, dequantises each tensor to fp32 and
writes bf16 safetensors for ``convert.py`` to consume.

What it accepts (the ggml types actually present in real K-quant releases)
-------------------------------------------------------------------------
``F32``, ``F16``, ``BF16`` (pass-through) plus ``Q4_K`` and ``Q6_K``, which is what
``Ornith-1.5-9B-Q4_K_M.gguf`` is published as (measured: 223 Q4_K + 35 Q6_K + 184
F32 tensors).  A type that is known but not decoded (Q5_K, Q2_K, Q8_0, ...) is
refused by name, not by "unrecognised file".

Two defects this replaces
-------------------------
* The header was read as ``"<IQI"`` (20 bytes total) instead of
  ``"<IQQ"`` / 24 bytes (``tools/gui/model_import.py:318`` had the same bug), which
  left the reader 4 bytes short and turned the first metadata string length into
  85,899,345,920 bytes on the real file -- surfacing as a bare ``MemoryError`` whose
  ``str()`` is empty, i.e. a blank error message.
* Tensors that matched no name and tensors of a type with no decoder were **dropped
  silently** while the tool exited 0, producing a checkpoint that was missing whole
  layers.  Now every tensor is accounted for or the tool refuses.

Correct-by-construction notes
-----------------------------
* The ggml type table and the block layouts come from
  :mod:`tools.convert.gguf_kquant`, which cross-checks its table against
  ``tools/archkit/gguf_tensors.py`` (the one copy in this tree that was already
  right) so they cannot drift.  The old table here was shifted by one, so ``Q4_0``
  was copied out as ``BF16`` -- wrong numbers, no error.
* GGUF stores ``ne = (fastest-varying, ..., outermost)``, the reverse of the HF
  ``(out_features, in_features)`` convention; :mod:`tools.convert.gguf_names` does
  the reversal.  The old code copied the GGUF dims into the safetensors header, so
  every 2-D tensor declared a transposed shape while the payload was unchanged.
* ``Ornith`` has 33 blocks and ``qwen35.nextn_predict_layers = 1``, so the main
  stack is 32 layers and block 32 is the draft block.  ``gguf_names.layer_split``
  reports both, and the draft block's tensors go under ``mtp.*``.
* The output is streamed one tensor at a time: a 9B model is ~18 GB in bf16 and the
  old code held the whole ``payload`` dict in RAM.
* The ROUTE IS READ FROM THE SOURCE, and ``--fold-route`` only overrides it.  A source
  that declares its weights are in the Hadamard basis takes ``fold_back``, which this
  file does not implement and hands to ``tools/convert/gguf_fold_back.py`` by name; a
  source that declares no fold takes ``pass_through``, which is what this file has always
  done.  ``refuse`` is a value an operator types and never a default: this file was
  measured refusing every source while it defaulted there, including the unflagged call
  at ``tools/convert/convert_runner.py:320``.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert import gguf_kquant as K          # noqa: E402
from tools.convert import gguf_names as N           # noqa: E402
from tools.convert import gguf_hadamard as H        # noqa: E402
from tools.convert import gguf_fold_route as FR     # noqa: E402
from tools.convert import unit_offset_norms          # noqa: E402  the +1 the epilogue adds

#: llama.cpp's ``general.architecture`` -> the HF ``model_type`` this tree routes.
#: ``qwen35`` is llama.cpp's spelling for the qwen3.5/3.6/3.8 text family; see
#: ``tools/archkit/gguf_spec.py:63-68``, which records the same mapping.
#:
#: The value is the *route*, not a label: two independent consumers read it and both
#: require the qwen3.5 spelling.
#:   * ``tools/convert/import_model.py:72`` DECODER_FAMILIES (and the GUI's twin at
#:     ``tools/gui/model_import.py:241``) is ``("qwen3_5", "qwen3_5_moe", "muse_glimmer")``,
#:     so a ``model_type`` outside it makes the front door report
#:     "没有任何已注册 target 接受 model_type=..." -- which is what the real
#:     Ornith-1.5-9B-Q4_K_M GGUF got while ``src/targets/qwen3_5_9b`` was already
#:     registered (measured 2026-09-17).
#:   * the registered tier the generic chain hands its config to states the contract it
#:     enforces: ``tools/convert/qwen3_6_27b/convert.py:10-19``
#:     ``_ROOT_CONFIG = {"architectures": ["Qwen3_5ForConditionalGeneration"],
#:     "model_type": "qwen3_5", ...}``, checked member-by-member by its
#:     ``validate_config`` (``:118``).  ``("qwen3", "Qwen3ForCausalLM")`` satisfied
#:     neither member.
#: ``qwen3`` (the literal arch string) is deliberately left alone: no llama.cpp build in
#: this record emits it for this family, and ``tools/gui/test_model_import.py:199`` pins
#: the old value.  Changing it is a separate call, not a silent part of this one.
ARCH_TO_MODEL_TYPE = {
    "qwen35": ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
    "qwen3": ("qwen3", "Qwen3ForCausalLM"),
    "qwen2": ("qwen2", "Qwen2ForCausalLM"),
}


class ExtractionRefused(RuntimeError):
    """Raised with the missing part named; never a bare exit code."""


def declared_basis(kv) -> str:
    """What the SOURCE declares its weights are: ``folded`` or ``primal``.

    The head-order proof is asked FIRST and by name, because it is a different question from the
    declaration and it is the one this reader has always refused: a file that declares a rotation
    and does not say which head order ``blk.*.ssm_out.weight`` was kept in raises
    ``gguf_hadamard.UnprovenRotation``, and an absent key is the unproven case rather than the
    permissive one.  The basis itself is then read by
    :func:`gguf_fold_route.declared_basis_of` -- the same function ``tools/convert/gguf_fold_back.py``
    calls -- so the two tools cannot come to different answers about which files are folded.
    """
    try:
        # The value is not used here: the REFUSAL is the point, and it has to happen before a
        # route is derived from the declaration or a plan is built.  Whether the head order is
        # grouped is the fold-back tool's question, asked there.
        H.gdn_ssm_out_is_grouped(kv)
    except H.UnprovenRotation as exc:
        raise ExtractionRefused("%s  Nothing was written." % exc) from None
    return FR.declared_basis_of(kv)


def plan_tensors(kv, tensors, route=None):
    """-> [(gguf_name, hf_key, hf_shape, type_id, offset, n_elements)]

    ``route`` may be omitted, in which case it is derived from the source's own declaration
    (:func:`gguf_fold_route.route_for_declared_basis`).  It is never defaulted to ``refuse``:
    that value means "emit nothing", it is a value an operator types, and defaulting it made this
    gate refuse every source -- measured, and it is why ``tools/convert/convert_runner.py:320``'s
    unflagged command is exercised in this record's proofs.
    """
    arch = str(kv.get("general.architecture") or "")
    if arch not in N.RULES:
        raise ExtractionRefused(
            "no GGUF tensor-name rules for architecture %r; this tree knows %s. "
            "Adding one means adding a rule set to tools/convert/gguf_names.py and "
            "a target under src/targets/, not guessing the names."
            % (arch, ", ".join(sorted(N.RULES))))
    plan, seen = [], {}

    # A Hadamard-folded file stores blk.*.ssm_out.weight in the fold's GROUPED head
    # order and expects the RUNTIME to permute the activation before the signs and the
    # rotation; the same tensor in an unrotated file is converted to TILED order.  The
    # fold is not something a checkpoint can carry, so an extraction of a rotated file
    # is not a usable checkpoint for a runtime that does not implement the transform,
    # and the ssm_out order difference would corrupt a converter that reorders GDN heads
    # uniformly.  Name which of the two the file is instead of leaving it to the reader.
    #
    # The head order is *declared*, and the one case this must not guess is a file that
    # declares the rotation and omits the field: an absent key is unproven, and neither
    # answer is safe for it.  gdn_ssm_out_is_grouped refuses that case by raising instead
    # of answering False, and the refusal is carried out here, before any tensor is mapped
    # or any byte is written.
    # ---- ROUTE-AWARE FOLD GUARD -------------------------------------------------
    # Before: `if grouped: refuse`, unconditionally.  A folded file was refused no matter what
    # the operator asked for, which is SAFE but says nothing about which route was wanted.
    #
    # After: the predicate is the route decision in `gguf_fold_route`.  What did NOT change is
    # the direction of the answer -- this tool has NO CHANNEL to declare a folded checkpoint
    # (it writes a safetensors header with no `__metadata__`), so `pass_through` is refused HERE
    # for exactly the reason it is refused everywhere: an undeclared folded checkpoint is read
    # as an unrotated one and every folded matmul is silently wrong.  `refuse` refuses, and it
    # is nobody's DEFAULT any more: it is a value an operator types (see `main`).  No value of
    # `--fold-route` makes this guard permissive -- which is the
    # property the guard is for, and it is demonstrated below rather than asserted.
    #
    # The route that DOES admit the ternary is `fold_back`, and it is not implemented in this
    # file: it lives in `tools/convert/gguf_fold_back.py`, which reuses this module's name rules,
    # codec table, output transaction and config builder and adds the unfold plus its proof.
    # `main` DERIVES that route from the source's own declaration and hands the whole job to
    # it by name, before a directory is created; this gate is then reached only by the routes
    # that stay here -- pass_through and refuse on a folded source, either route on a primal
    # one.
    declared = declared_basis(kv)
    if route is None:
        route = FR.route_for_declared_basis(declared)
    # pass_through copies the weights, so the output's basis IS the source's; fold_back
    # would make it primal.  Nothing below emits a declaration (``emits_declaration`` is
    # False), so this value only ever reaches the guard's refusal MESSAGE -- but it is
    # written correctly, so that the day this writer grows a declaration channel the
    # condition changes in one place and not two.
    emitted = declared
    emits_declaration = False        # this writer emits no __metadata__; see above
    try:
        FR.route_precheck(route, declared,
                          emitted if emits_declaration else None)
    except FR.FoldRouteRefused as exc:
        # WHICH ROUTE ADMITS THIS FILE IS A FACT ABOUT THIS FILE, and it is stated from
        # `declared` rather than assumed.  The previous wording answered "fold_back" for
        # every source, so the refusal of a PRIMAL source (which owes nothing, and is
        # admitted by `pass_through`) sent its reader to the fold-back tool for a transform
        # there is nothing to apply.  The refusal itself is unchanged: same exception, same
        # message from the guard, same "Nothing was written".
        if declared == FR.BASIS_FOLDED:
            admitting = ("the route that admits this file is fold_back, in "
                         "tools/convert/gguf_fold_back.py")
        elif declared == FR.BASIS_PRIMAL:
            admitting = ("the route that admits this file is pass_through -- a primal source "
                         "owes no transform; fold_back has nothing to fold here")
        else:
            admitting = ("no route admits a source that declares basis=%r: the fold state has "
                         "to be declared before any route can be derived" % (declared,))
        raise ExtractionRefused(
            "%s  [route-aware fold guard; --fold-route=%s; declared basis=%s; the fold set is "
            "%d weight(s) plus %d inverse-lookup table(s); the transform is %s, block %s.  %s.]  "
            "Nothing was written."
            % (exc, route, declared,
               H.declared_count(kv.get(H.PREFIX + "weight_names")),
               H.declared_count(kv.get(H.PREFIX + "inverse_weight_names")),
               kv.get(H.PREFIX + "transform"), kv.get(H.PREFIX + "block_size"),
               admitting)) from None
    if declared == FR.BASIS_FOLDED and route == FR.ROUTE_PASS_THROUGH:
        # unreachable while `emits_declaration` is False; kept so that the day this writer grows a
        # declaration channel the condition above flips in ONE place and not two.
        raise ExtractionRefused("pass_through admitted with no declaration channel.  Nothing was written.")

    for name, dims, ttype, off in tensors:
        # A type with no decoder is refused HERE, before write_safetensors opens the
        # output: this module's own docstring promises "A type that is known but not
        # decoded (Q5_K, Q2_K, Q8_0, ...) is refused by name", and the only refusal that
        # existed (K.to_fp32) fired from inside the write loop -- after the safetensors
        # header and part of the payload were already on disk.  The caller then saw
        # rc=1 and a truncated model.safetensors.  The vocabulary below is K's own, so
        # it cannot drift from what the dequantiser would have said.
        # G1/MEASURED: /home/user/models/bonsai/Bonsai-27B-Q1_0.gguf declares
        # {F32: 353, type41: 498}; K.tensor_nbytes raised KeyError for type 41 and
        # 113,384 B had already been written.
        if ttype not in K.DEQUANTIZERS:
            raise ExtractionRefused(
                "tensor %r is stored as ggml type %d (%s) and this reader decodes %s; "
                "a type whose layout is known here but which is not decoded yet: %s.  "
                "Nothing was written."
                % (name, ttype, K.type_name(ttype), K.SUPPORTED_NAMES,
                   ", ".join("%s(%d)" % (K.type_name(t), t) for t in K.KNOWN_UNSUPPORTED)
                   or "none"))
        got = N.translate(arch, name, dims, kv)
        if got is None:
            raise ExtractionRefused(
                "tensor %r has no name rule for architecture %r; writing the rest "
                "would produce a checkpoint missing this tensor" % (name, arch))
        key, shape = got
        if key in seen:
            raise ExtractionRefused(
                "two GGUF tensors map to the same HF key %r: %s and %s"
                % (key, seen[key], name))
        seen[key] = name
        nelem = 1
        for d in dims:
            nelem *= int(d)
        if int(np.prod(shape)) != nelem:
            raise ExtractionRefused(
                "shape reversal changed the element count for %r: gguf %s -> hf %s"
                % (name, tuple(dims), shape))
        plan.append((name, key, tuple(shape), ttype, off, nelem))
    return plan


def config_from_metadata(kv, arch: str, have_output_head: bool, fold_declaration=None) -> dict:
    """A config.json derived from the GGUF metadata, not a hardcoded family.

    The previous version wrote ``{'architectures': ['Qwen3ForCausalLM'],
    'model_type': 'qwen3'}`` for *every* input, so anything downstream that asked
    "what family is this?" was reading this tool's own guess back.

    ``fold_declaration`` is the basis the WEIGHTS are in, plus how they got there, or ``None``
    for a writer that declares nothing.  It goes in under the key spellings the safetensors
    ``__metadata__`` map uses (``prism.fold.*``, with ``gguf_fold_route.BASIS_KEY`` for the
    basis itself), because a checkpoint whose bytes are in the rotation's basis and whose config
    does not say so is read as an unrotated one -- and the two differ in one tensor's inner head
    order under the same tensor name.  The default is ``None`` and not a guess: a writer that
    does not know the basis must not assert one.

    This is the declaration ``gguf_fold_route``'s guard asks the fold-back route to carry
    ("--fold-route=fold_back makes the weights primal, so the output must declare 'primal'"); the
    tool that performs the unfold, ``tools/convert/gguf_fold_back.py``, passes the block below.
    MEASURED 2026-09-20: with the route derived from the source, this file hands a rotated source
    to that tool, which admits it and writes ``prism.fold.basis = primal`` here.
    """
    p = lambda k: kv.get("%s.%s" % (arch, k))
    model_type, architecture = ARCH_TO_MODEL_TYPE[arch]
    main_layers, nextn = N.layer_split(kv, arch)
    config = {
        "architectures": [architecture],
        "model_type": model_type,
        "hidden_size": int(p("embedding_length") or 0),
        "num_hidden_layers": main_layers,
        "num_attention_heads": int(p("attention.head_count") or 0),
        "num_key_value_heads": int(p("attention.head_count_kv") or 0),
        "head_dim": int(p("attention.key_length") or 0),
        "intermediate_size": int(p("feed_forward_length") or 0),
        "vocab_size": len(kv.get("tokenizer.ggml.tokens") or []),
        "max_position_embeddings": int(p("context_length") or 0),
        "rms_norm_eps": float(p("attention.layer_norm_rms_epsilon") or 1e-6),
        "rope_theta": float(p("rope.freq_base") or 0.0),
        # An absent output.weight means the head is tied to the embedding.  The old
        # file asserted a Qwen3 identity and said nothing about tying at all.
        "tie_word_embeddings": not have_output_head,
        # Provenance, so nothing downstream has to guess where these numbers came from.
        "_ninfer_source": {
            "reader": "tools/convert/gguf_extract.py",
            "gguf_architecture": arch,
            "geometry_from": "gguf_metadata",
            "block_count_including_draft": int(p("block_count") or 0),
            "nextn_predict_layers": nextn,
        },
    }
    if fold_declaration is not None:
        config.update(fold_declaration)
    return config


#: Suffix of a file that is being built.  Nothing a consumer looks for ever ends in
#: this: the name only exists between ``open`` and the ``os.replace`` that publishes
#: the real one.
INCOMPLETE_SUFFIX = ".partial"

#: The name a consumer checks for to decide "is this checkpoint here?".  It is published
#: LAST, so it is absent for as long as ``config.json`` is absent.
CHECKPOINT_MARKER = "model.safetensors"


class OutputTransaction:
    """Output files that appear together or not at all.

    ``os.replace`` is atomic inside one filesystem, so a reader either sees no
    ``model.safetensors`` or a complete one -- never the header of a whole model in
    front of half a payload.  Both of this tool's in-loop refusals (short read,
    non-finite decode) and any crash in between used to leave exactly that: a
    safetensors file whose header declared every tensor in the plan with its full
    size, in a directory with no ``config.json``.  G1/MEASURED before this change:
    five named refusals, rc=3, residue 47832 / 49272 / 49216 / 440 / 440 B.

    The invariant: a reader that finds ``model.safetensors`` finds the whole file and
    finds ``config.json`` beside it.  A reader that finds neither has lost nothing.

    A refusal removes both temporaries, and removes the output directory itself if
    this call created it and left nothing else in it.  ``SIGKILL`` is outside any
    program's reach: it can leave a ``.partial`` behind, which is why the temporaries
    are named so that nothing looks for them.
    """

    def __init__(self, out_dir: Path, created_dir: bool):
        self.out_dir = out_dir
        self.created_dir = created_dir
        self._published: list[tuple[Path, Path]] = []

    def staging(self, name: str) -> Path:
        """The temporary path to write ``name`` to; ``name`` appears only on commit."""
        staged = self.out_dir / (name + INCOMPLETE_SUFFIX)
        self._published.append((staged, self.out_dir / name))
        return staged

    def __enter__(self) -> "OutputTransaction":
        return self

    def _discard(self) -> None:
        for staged, _final in self._published:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                pass                       # already gone, or not ours to remove
        if self.created_dir:
            try:
                self.out_dir.rmdir()       # succeeds only if we left it empty
            except OSError:
                pass

    def __exit__(self, exc_type, _exc, _tb) -> bool:
        if exc_type is not None:
            self._discard()
            return False
        # config.json first, the checkpoint marker last: a consumer gates on the marker,
        # so it must not be able to see it before the config that describes it.
        order = sorted(self._published,
                       key=lambda pair: pair[1].name == CHECKPOINT_MARKER)
        try:
            for staged, final in order:
                os.replace(staged, final)
        except BaseException:
            # A rename can still fail (EXDEV / ENOSPC / permissions).  Drop whatever is
            # still staged.  The invariant survives: the marker is renamed last, so a
            # failure anywhere in this loop means it was never published.
            self._discard()
            raise
        return False


def write_safetensors(dest: Path, src: str, data_offset: int, plan) -> int:
    """Stream every tensor to ``dest``; RAM stays at one tensor.

    ``dest`` is a temporary name inside the output directory, not a name a consumer
    reads: publishing it is :meth:`OutputTransaction.__exit__`'s job.  The payload is
    streamed because a 9B model is ~18 GB in bf16, and that is also why the rename has
    to happen after the last tensor rather than after the header.
    """
    header, offset = {}, 0
    for _name, key, shape, _ttype, _off, nelem in plan:
        nbytes = nelem * 2                       # bf16
        header[key] = {"dtype": "BF16", "shape": list(shape),
                       "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((-(8 + len(blob))) % 8)      # safetensors header is 8-aligned
    written = 0
    with open(src, "rb") as fh, open(dest, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        # Flushed before the first payload byte, so that a run that is killed still leaves the
        # thing it has already declared.  MEASURED 2026-09-20 (dl/foldroute/): this box's Python
        # has io.DEFAULT_BUFFER_SIZE = 131072 and the full 851-tensor header is 113,488 B, so
        # WITHOUT this call a cut run's .partial is 0 BYTES -- "the staging file appears" is the
        # admission evidence this tree reads, and without the flush it is the whole of it, with
        # nothing on disk to say what the bytes were going to be.
        out.flush()
        for name, key, _shape, ttype, off, nelem in plan:
            need = K.tensor_nbytes(ttype, nelem)
            fh.seek(data_offset + off)
            raw = fh.read(need)
            if len(raw) != need:
                raise ExtractionRefused(
                    "tensor %s needs %d bytes at offset %d but the file ends after "
                    "%d; the download is incomplete" % (name, need, off, len(raw)))
            fp32 = K.to_fp32(ttype, raw, nelem)
            if not np.isfinite(fp32).all():
                raise ExtractionRefused(
                    "tensor %s (%s) decoded to non-finite values; refusing to write it"
                    % (name, K.type_name(ttype)))
            # The engine's rmsnorm epilogue adds one to this weight
            # (src/ops/kernel/rmsnorm.cuh:22) and the GDN control projection goes
            # through that epilogue with the flag set
            # (src/ops/generic/rowsplit_generic.cu:519), so the artifact must hold
            # `gamma - 1`.  A GGUF holds the plain `gamma` and this is the one place
            # its value exists.  Named set + why: tools/convert/unit_offset_norms.py
            fp32 = unit_offset_norms.stored_values(fp32, key)
            out.write(K.fp32_to_bf16_bytes(fp32))
            written += 1
        out.flush()
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="input .gguf")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--coverage", action="store_true",
                    help="print mapped/total per tensor role and exit")
    # --------------------------------------------------------------------------------------------
    # DEVIATIONS FROM THE DELIVERED PATCH, named here rather than in a record file so that the next
    # reader of this source sees them.  Both are about WHERE THE ROUTE COMES FROM.
    #
    # (1) The patch as delivered defaulted this flag to FR.ROUTE_REFUSE.  Measured consequence, one
    #     call deep: route_precheck('refuse', 'primal', None) RAISES, so the patched tool refused a
    #     source that declares NO fold at all -- every source -- and the operator's remedy was a
    #     flag they were never told about.  The victims are in this tree and are not hypothetical:
    #         tools/convert/convert_runner.py:320   builds `gguf_extract.py --src .. --out ..`
    #         tools/convert/import_model.py:733     prints that same command as the advice
    #     Neither passes --fold-route, so the generic GGUF -> bf16 chain went from rc=0 to rc=3 on
    #     every source it had always extracted.
    #
    # (2) A default is still a guess, whichever value it is.  THIS FLAG NOW DEFAULTS TO NOTHING
    #     (`None`), and when it is absent the route is DERIVED FROM THE SOURCE, which is where the
    #     answer actually lives:
    #         the source declares a fold (`prism.hadamard.*` / basename 'folded') -> fold_back
    #         the source declares none                                             -> pass_through
    #     `refuse` is not an answer that derivation can give; it is a value an operator types when
    #     they want nothing emitted.  MEASURED on the two files this family ships, through the
    #     unflagged command this record exercises (dl/foldroute/):
    #         Bonsai-27B-Q1_0.gguf              -> pass_through, ADMITTED, bytes written
    #         Ternary-Bonsai-2-27B-PTQ1_0.gguf  -> fold_back, ADMITTED, handed to
    #                                              tools/convert/gguf_fold_back.py, which unfolds
    #
    # Why neither deviation weakens the guard the patch exists for.  MEASURED, every route value
    # against the real folded ternary, with the declaration channel this writer actually has
    # (none):
    #     pass_through + declared=folded + emitted=None -> REFUSED by this gate
    #     fold_back    + declared=folded                -> this gate refuses (no channel here), and
    #                                                      `main` hands the route to the tool whose
    #                                                      output DOES declare `primal` -- in the
    #                                                      safetensors __metadata__ and in
    #                                                      config.json -- so the guard is satisfied
    #                                                      rather than bypassed, and its post-
    #                                                      condition re-runs there on the ledger
    #     refuse       + declared=folded                -> REFUSED
    # A folded source still cannot be PASSED THROUGH by this writer under any value, because a
    # checkpoint in the rotation's basis that does not SAY so is read as an unrotated one and every
    # folded matmul is silently wrong.  That property is untouched.
    # --------------------------------------------------------------------------------------------
    ap.add_argument("--fold-route", default=None, choices=list(FR.ROUTE_VALUES),
                    help="which of the two routes to take.  DEFAULT: derived from the source "
                         "itself -- a source that declares a fold gets 'fold_back', one that "
                         "declares none gets 'pass_through'.  Give the flag to override; "
                         "'refuse' emits nothing and is never chosen for you.")
    ap.add_argument("--allow-partial", action="store_true",
                    help="report unmapped tensors instead of refusing (diagnostics)")
    args = ap.parse_args(argv)

    # A source this reader cannot open or cannot parse is THIS tool's refusal, not a
    # traceback: the entry point below already turns ExtractionRefused into
    # "REFUSED: ..." + exit 3, and until now that class was reachable only for the fold
    # set.  G1/MEASURED, dl/unreachable/logs/mx_extract_*.err (rc=1 + traceback).
    try:
        kv, tensors, data_offset = K.read_tensor_table(args.src)
    except (ExtractionRefused, N.UnmappedTensors):
        raise
    except Exception as exc:                        # noqa: BLE001 - refusal, not crash
        raise ExtractionRefused(
            "%s cannot be read as a GGUF tensor table: %s: %s  Nothing was written."
            % (args.src, type(exc).__name__, exc)) from None
    arch = str(kv.get("general.architecture") or "")
    # Same rule as above, for a source that parses but whose architecture this tree has
    # no name rules for.  plan_tensors has refused that case by name all along, but the
    # coverage report is built first and N.coverage raises a bare KeyError, so that
    # refusal was unreachable and the caller got rc=1 (a crash) for a considered
    # refusal.  G1/MEASURED: /home/user/scratch/svca/fix/spec_qwen3_f16.gguf (arch=qwen3,
    # N.RULES == ['qwen35']).
    try:
        rep = N.coverage(kv, tensors, arch)
    except (ExtractionRefused, N.UnmappedTensors):
        raise
    except Exception as exc:                        # noqa: BLE001 - refusal, not crash
        raise ExtractionRefused(
            "%s: the tensor table was read but not name-mapped: %s: %s  "
            "Nothing was written." % (args.src, type(exc).__name__, exc)) from None
    if args.coverage:
        print(N.render(rep))
        return 0 if not rep["unmapped"] else 1
    if not args.allow_partial:
        N.require_full_coverage(rep)

    # ---- THE ROUTE IS A FACT ABOUT THE SOURCE, AND IT IS READ FROM THE SOURCE ----------
    #
    # A checkpoint that declares its basis has already said who applies the transform: a `folded`
    # file still owes it, a `primal` file owes nothing.  So the unflagged command has an answer
    # available that is a property of the FILE, and this is where it is read.  What this replaces
    # is not merely a stricter default, it is a wrong one: `refuse` means "emit nothing", so
    # defaulting there made the unflagged command at tools/convert/convert_runner.py:320 -- the
    # one this tool is called by, and the one import_model.py:733 prints as advice -- refuse EVERY
    # source, including the unrotated ones it had always extracted.  `refuse` remains available;
    # it is only ever a value an operator types.
    #
    # An explicit --fold-route still decides.  The derivation runs only when the operator gave
    # none, which is the case both callers in this tree are in.
    declared = declared_basis(kv)
    if args.fold_route is not None:
        route = args.fold_route
        print("route   : %s (--fold-route was given, so it decides)" % route)
    else:
        route = FR.route_for_declared_basis(declared)
        print("route   : %s (derived from the source's own declaration; no --fold-route given)"
              % route)
    print("basis   : the source declares %s" % declared)

    if declared == FR.BASIS_FOLDED and route == FR.ROUTE_FOLD_BACK:
        # ---- THE HAND-OFF ---------------------------------------------------------------
        # fold_back is the one route that can emit a usable checkpoint from a rotated source, and
        # THIS FILE CANNOT PERFORM IT: it decodes and re-encodes, it applies no transform, and its
        # output channel carries no declaration (see plan_tensors below, whose gate refuses a
        # folded source under every value it is asked about).  The arithmetic (A^-1 plus the GDN
        # head unpermutation), the ledger of what was really unfolded, and the declaration the
        # guard asks for -- basis `primal`, in the safetensors __metadata__ AND in config.json --
        # live in tools/convert/gguf_fold_back.py.
        #
        # Ask the guard HERE, with the declaration that output really carries, and before a
        # directory is created.  It is asked AGAIN inside that tool's output transaction, on the
        # ledger of what the traverse actually did, and THAT is the binding one
        # (gguf_fold_route.converter_admission).  Satisfying the guard is the point of the
        # hand-off: `fold_back` is admissible exactly because its output declares `primal`, and
        # the bytes it declares that over are the unfolded ones.
        try:
            FR.route_precheck(route, declared, FR.BASIS_PRIMAL)
        except FR.FoldRouteRefused as exc:
            raise ExtractionRefused("%s  Nothing was written." % exc) from None
        if args.allow_partial:
            print("note    : --allow-partial is this file's diagnostic mode and is NOT carried "
                  "over; the fold-back path requires full coverage, by its own gate.")
        print("hand-off: the unfold and the declaration are in tools/convert/gguf_fold_back.py; "
              "this file transforms no tensor.")
        from tools.convert import gguf_fold_back as FB        # lazy: FB imports this module
        # The routing flag travels as the OPERATOR stated it, which is not the same
        # string as the route this file derived.  Passing the derived route as
        # `--fold-route` made tools/convert/gguf_fold_back.py's own status line report
        # "(--fold-route=fold_back was supplied on this command line)" for a flag nobody
        # typed (measured: `gguf_extract.py <folded>.gguf --out <dir>` with no
        # --fold-route, and the hand-off's argv was the only place the flag existed).
        # Withholding it is NOT a weaker request: gguf_fold_back derives the route from
        # the same declaration with the same function
        # (gguf_fold_route.route_for_declared_basis, on `declared_basis_of(kv)`), and its
        # route_precheck -- the binding one, run inside its output transaction on the
        # ledger the traverse actually filled -- is reached either way.
        hand_off = ["--src", args.src, "--out", args.out]
        if args.fold_route is not None:
            hand_off += ["--fold-route", args.fold_route]
        try:
            return FB.main(hand_off)
        except (FB.EmissionRefused, FR.FoldRouteRefused, H.UnreadableMetadata,
                H.UnprovenRotation, N.UnmappedTensors, ValueError) as exc:
            # The classes the fold-back tool's own entry point turns into "REFUSED: ..." + rc=3.
            # Called as a function its handler does not run, so they are re-raised as this
            # module's refusal: the caller gets one named refusal and an exit code, never a
            # traceback, and nothing is published.
            raise ExtractionRefused(str(exc)) from None

    print("source  : %s" % args.src)
    print("arch    : %s" % arch)
    print("layers  : %d main + %d draft block(s)"
          % (rep["main_layers"], rep["nextn_layers"]))
    print("coverage: %d/%d tensors mapped" % (rep["mapped"], rep["total"]))
    types: dict[str, int] = {}
    for _n, _d, ttype, _o in tensors:
        types[K.type_name(ttype)] = types.get(K.type_name(ttype), 0) + 1
    print("types   : %s" % dict(sorted(types.items())))

    plan = plan_tensors(kv, tensors, route)
    out_dir = Path(args.out)
    created_dir = not out_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)
    have_head = any(n == "output.weight" for n, _d, _t, _o in tensors)
    # Both files are staged under INCOMPLETE_SUFFIX and published only once every byte
    # of both is on disk, so this call either produces a complete checkpoint directory
    # or produces nothing.
    with OutputTransaction(out_dir, created_dir) as txn:
        written = write_safetensors(txn.staging("model.safetensors"),
                                    args.src, data_offset, plan)
        txn.staging("config.json").write_text(
            json.dumps(config_from_metadata(kv, arch, have_head),
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("extracted %d/%d tensors -> %s" % (written, len(plan), out_dir))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ExtractionRefused, N.UnmappedTensors) as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        raise SystemExit(3)
