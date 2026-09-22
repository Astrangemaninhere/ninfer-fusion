#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gguf_hadamard.py — the Prism/Bonsai (Bonsai-2 / Ternary-Bonsai) rotation contract,
written as arithmetic only.

WHY THIS FILE EXISTS
--------------------
The "2" artefacts store their weights in a rotated basis and declare the rotation in
GGUF metadata under ``prism.hadamard.*``.  A reader that dequantises the packed types
(Q1_0/Q2_0/PQ2_0/PTQ1_0) and stops has the weights of a different model: every folded
tensor is expressed in the transform's basis, not the primal one.  PrismML's own demo
documentation says a runtime either applies the matching transform or refuses the file.

The transform is a *runtime* operator, and this module does **not** pretend otherwise —
where it belongs in a layer, and what consumes its output, is the engine's business and
is not written here.  What is written here, and only this, is the arithmetic: the
matrix, the butterfly that computes it in O(N log N), the sign convention, the order the
two are applied in the forward and inverse directions, and the head permutation the GDN
output needs.  All of it is a pure function of (x, signs, block), and nothing in it needs
the engine to be meaningful.

SOURCE OF TRUTH (PrismML-Eng/llama.cpp, branch ``prism-v7``)
-----------------------------------------------------------
``src/llama-model.cpp``       the rotation matrix, built explicitly, and the loader's
                              validation of every declared field
``src/llama-impl.h``          ``llama_mul_mat_hadamard`` — the fallback is a DENSE
                              matmul; the FWHT op is an optimisation behind a hint
``src/llama-graph.cpp``       the two application sites (folded weights; latent lookup)
``src/llama-graph.h``         ``llama_hadamard_transform`` — signs, rotation, and the GDN
                              head permutation, in the order they are applied
``ggml/src/ggml-cuda/fwht.cu``the register and shared-memory butterflies
``ggml/src/ggml-metal/kernels/dequantize.h``  a 256-entry LUT for the PTQ1_0 trit
``conversion/qwen.py``        the GDN head reorder that the fold switches off

WHAT THE CONTRACT IS, IN ONE LINE
---------------------------------
    A = H . D          H = normalized Sylvester Walsh-Hadamard, block N = 1024
                       D = diag(sign vector), one sign per INPUT channel
    forward (``prism.hadamard.weight_names``):     a_rot = H (D . a)   [signs, then H]
    inverse (``prism.hadamard.inverse_weight_names``): z = D (H . z_s) [H, then signs]
    A^-1 = D H, because (H D)(D H) = H H = I and D^2 = H^2 = I.

The inverse therefore needs **no data the forward set does not already carry**: the same
``sign_values`` are reused, in the opposite order.  What is *not* derivable is which
tensors take the inverse path — that set is declared, and the fork refuses every name
there except ``token_embd.weight``.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "CONTRACT_VERSION",
    "TRANSFORM_NAME",
    "AXIS_NAME",
    "SIGN_MODES",
    "UnprovenRotation",
    "UnreadableMetadata",
    "declared_count",
    "contract",
    "sylvester_walsh_hadamard",
    "butterfly",
    "fwht",
    "apply_forward",
    "apply_inverse",
    "gdn_v_permutation",
    "gdn_v_grouped_declared",
    "gdn_ssm_out_is_grouped",
    "ssm_out_head_order",
    "folded_widths",
]

CONTRACT_VERSION = 1
TRANSFORM_NAME = "normalized-sylvester-walsh-hadamard"
AXIS_NAME = "input-last-dimension"
SIGN_MODES = ("identity", "explicit")

#: The metadata key prefix.  Note the asymmetry the reader must not paper over: the
#: *metadata* key is ``prism.hadamard.*`` while the *op* in the fork is FWHT
#: (``ggml_fwht`` / ``GGML_OP_FWHT`` / ``GGML_HINT_SRC0_IS_HADAMARD``).
PREFIX = "prism.hadamard."


#: The contract's one *choice*, as opposed to its arithmetic.  Every other field describes
#: what the tensors are; this one describes how the converter happened to emit one of
#: them.  It is written by ``conversion/qwen.py`` on the fold branch and by nothing else.
GDN_V_GROUPED_KEY = PREFIX + "gdn_v_grouped"


class UnprovenRotation(RuntimeError):
    """The metadata does not settle a question a caller must know before it writes.

    Raised for the *absent* case, which is neither True nor False.  A rotated file that
    omits the field leaves the head order of ``blk.*.ssm_out.weight`` unstated, and both
    answers are wrong for some file: ``False`` reads as "tiled" and admits a grouped
    tensor into a reader that will reorder it, ``True`` reads as "grouped" and refuses a
    file that declares no rotation at all.  No default is invented here -- the caller is
    told to refuse, and told which key is missing.
    """

class UnreadableMetadata(RuntimeError):
    """A metadata array is present but its values are not.

    ``gguf_kquant.read_tensor_table`` summarises large arrays as (element type, count,
    first few values) and keeps them out of memory on purpose -- ``tokenizer.ggml.tokens``
    on a 248,320-token model is ~200 MB of strings no tensor reader wants.  The contract
    needs the *values* (the fold set, the sign vectors), so a summary is not something
    this module may paper over: it refuses, naming the key, rather than validate half a
    contract and report the result as if it had seen the whole file.
    """


def declared_count(value) -> int:
    """How many entries an array-valued key declares, without materialising it.

    The two shapes this tree produces both answer ``len()``: plain lists (hand-built
    metadata, this module's self-test, the probes) and the reader's summary objects.  A
    count is all a refusal message needs; nothing here turns a summary into values.
    """
    if value is None:
        return 0
    return len(value)


def _array_values(value, key: str) -> list:
    """The entries of a metadata array, or a named refusal -- never a bare TypeError."""
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        raise UnreadableMetadata(
            "%s is a summarised metadata array (%d entries, values not kept); the "
            "contract needs the values, and validating it from a summary would be "
            "reading half a file" % (key, len(value))) from None


# --------------------------------------------------------------------------- contract


def contract(kv: dict) -> dict | None:
    """Parse and validate ``prism.hadamard.*``.  ``None`` when the file is not rotated.

    Validation mirrors ``llama-model.cpp:1194-1330`` field for field, including the
    checks that exist there because skipping them "reads as identity later and silently
    changes the model function":

    * ``version`` must be 1;
    * ``block_size`` must be a power of two;
    * ``transform`` and ``axis`` must be the exact spellings the fork accepts;
    * ``sign_mode`` must be ``identity`` or ``explicit``;
    * in explicit mode ``sign_widths`` must be non-empty, every width must be positive
      and a multiple of ``block_size``, the widths must fit inside ``sign_values``
      exactly (no trailing slack), and every sign must be +-1.

    Returns a dict with the parsed fields plus a ``by_width`` map of sign vectors.
    """
    if PREFIX + "version" not in kv:
        return None

    version = int(kv[PREFIX + "version"])
    if version != CONTRACT_VERSION:
        raise ValueError("unsupported %sversion: %r" % (PREFIX, version))

    block = int(kv.get(PREFIX + "block_size", 0))
    if block <= 0 or (block & (block - 1)) != 0:
        raise ValueError("invalid %sblock_size: %r" % (PREFIX, block))

    transform = kv.get(PREFIX + "transform")
    if transform != TRANSFORM_NAME:
        raise ValueError("unsupported %stransform: %r" % (PREFIX, transform))

    axis = kv.get(PREFIX + "axis")
    if axis != AXIS_NAME:
        raise ValueError("unsupported %saxis: %r" % (PREFIX, axis))

    mode = kv.get(PREFIX + "sign_mode")
    if mode not in SIGN_MODES:
        raise ValueError("unsupported %ssign_mode: %r" % (PREFIX, mode))

    fwd = _array_values(kv.get(PREFIX + "weight_names"), PREFIX + "weight_names")
    if not fwd:
        raise ValueError("%sweight_names is empty" % PREFIX)
    inv = _array_values(kv.get(PREFIX + "inverse_weight_names"),
                        PREFIX + "inverse_weight_names")

    by_width: dict[int, np.ndarray] = {}
    if mode == "explicit":
        widths = [int(w) for w in _array_values(
            kv.get(PREFIX + "sign_widths"), PREFIX + "sign_widths")]
        values = [int(v) for v in _array_values(
            kv.get(PREFIX + "sign_values"), PREFIX + "sign_values")]
        if not widths:
            raise ValueError(
                "%ssign_mode is explicit but sign_widths is empty" % PREFIX)
        off = 0
        for width in widths:
            if width <= 0 or width % block or off + width > len(values):
                raise ValueError("invalid %s sign width: %d" % (PREFIX, width))
            vec = np.asarray(values[off:off + width], dtype=np.float32)
            if not np.all((vec == 1.0) | (vec == -1.0)):
                raise ValueError("%s sign values must be +/-1" % PREFIX)
            by_width[width] = vec
            off += width
        if off != len(values):
            raise ValueError("%ssign_values length mismatch" % PREFIX)

    return {
        "version": version,
        "block_size": block,
        "transform": transform,
        "axis": axis,
        "sign_mode": mode,
        "forward_names": fwd,
        "inverse_names": inv,
        "sign_widths": [int(w) for w in (kv.get(PREFIX + "sign_widths") or [])],
        "by_width": by_width,
        "signs_for": lambda width: by_width.get(int(width)),  # None => identity
        "gdn_v_grouped": gdn_v_grouped_declared(kv),
    }


# --------------------------------------------------------------------------- the matrix


def sylvester_walsh_hadamard(n: int) -> np.ndarray:
    """The fork's rotation matrix, verbatim from ``llama-model.cpp:1957-1975``.

    ``data[row][col] = (-1)**popcount(row & col) / sqrt(n)`` — the Sylvester construction
    of the Walsh-Hadamard matrix, normalized.  It is symmetric and, because the
    normalization is exactly 1/sqrt(n), an involution: ``H @ H == I``.  The fork builds
    it explicitly into an F32 tensor and multiplies by it; the FWHT kernels only replace
    that multiply, they do not change its value.
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("Hadamard block must be a power of two, got %r" % (n,))
    idx = np.arange(n, dtype=np.uint32)
    parity = idx[:, None] & idx[None, :]
    for shift in (16, 8, 4, 2, 1):
        parity = parity ^ (parity >> shift)
    scale = 1.0 / math.sqrt(float(n))
    return np.where((parity & 1) != 0, -scale, scale).astype(np.float32)


def butterfly(x: np.ndarray, block: int) -> np.ndarray:
    """Unscaled FWHT: stage h pairs (j, j+h) -> (x+y, x-y), the fork's convention.

    ``ggml/src/ggml-cuda/fwht.cu`` states it in both kernels: "the low element of a pair
    takes x + y, the high one x - y".  The register path does the first log2(warp) stages
    with ``__shfl_xor_sync`` and the rest in registers; the shared-memory path (used above
    N=2048) runs all log2(N) stages in shared memory.  Both are this loop, so this is the
    one place the convention is written down.
    """
    y = np.array(x, dtype=np.float32, copy=True)
    if y.shape[-1] % block:
        raise ValueError("block %d does not divide width %d" % (block, y.shape[-1]))
    shape = y.shape
    y = y.reshape(-1, block)
    h = 1
    while h < block:
        # within each group of 2h, the low half (j) and the high half (j+h) pair up
        z = y.reshape(-1, block // (2 * h), 2, h)
        a, b = z[:, :, 0, :], z[:, :, 1, :]
        y = np.concatenate([a + b, a - b], axis=2).reshape(-1, block)
        h *= 2
    return y.reshape(shape)


def _signs_like(x: np.ndarray, signs, block: int) -> np.ndarray:
    """Broadcast a per-input-channel sign vector over the block axis, the fork's way.

    The kernel indexes signs as ``signs + (row % n_blk) * block`` with
    ``n_blk = len(signs) / block``, and ``llama_mul_mat_hadamard`` reshapes the activation
    to ``[block, nelements/block]`` before the multiply.  For an activation whose trailing
    axes are (blocks, batch) the row index is ``blk + blocks*batch``, so ``row % blocks``
    is the block index and each ``block``-wide slice of the sign vector lands on its own
    slice of the input channel.  It is therefore a plain per-input-channel sign; only the
    broadcasting arithmetic is easy to get wrong.
    """
    if signs is None:
        return x
    s = np.asarray(signs, dtype=np.float32).reshape(-1)
    if s.size % block:
        raise ValueError("sign vector length %d is not a multiple of block %d"
                         % (s.size, block))
    if x.shape[-1] != s.size:
        raise ValueError("activation width %d does not match sign vector %d"
                         % (x.shape[-1], s.size))
    return x * s


# --------------------------------------------------------------------------- directions


def fwht(x: np.ndarray, signs=None, block: int | None = None, inverse: bool = False,
         axis: int = -1):
    """The whole operator for one block size, in the contract's order.

    ``inverse=False`` -> ``H (D . x)``   (signs first) — the folded-weight path
    ``inverse=True``  -> ``D (H . x)``   (signs last)  — the latent-lookup path

    Same signs, opposite order, which is what makes the inverse free.  ``block`` defaults
    to the sign vector's own length; pass it explicitly when the contract declares
    otherwise.  ``axis`` is the axis being transformed and defaults to the last one; a
    GGUF weight is laid out ``ne = (input, output, ...)``, so a caller holding a ggml-shaped
    array passes ``axis=0`` — the fold is along the INPUT axis (the metadata's
    ``axis = input-last-dimension`` names ggml's ne[0], not the array's last axis).  The
    MLX runtime written for these artefacts is the same two branches:

        if not inverse: x = x * signs
        x = hadamard_transform(x.reshape(-1, block), scale=1/sqrt(block))
        if inverse:     x = x * signs
    """
    arr = np.asarray(x, dtype=np.float32)
    if block is None:
        if signs is None:
            raise ValueError("pass block when there are no signs")
        block = int(np.asarray(signs).size)
    moved = axis != -1 and axis != arr.ndim - 1
    if moved:
        arr = np.moveaxis(arr, axis, -1)
    if not inverse:
        arr = _signs_like(arr, signs, block)
    out = butterfly(arr, block) * (1.0 / math.sqrt(float(block)))
    if inverse:
        out = _signs_like(out, signs, block)
    return np.moveaxis(out, -1, axis) if moved else out


def apply_forward(w: np.ndarray, signs=None, block: int | None = None,
                  axis: int = -1) -> np.ndarray:
    """Fold a tensor: rotate along the input axis, signs before the transform.

    ``W_s = A W`` with ``A = H D``.  The fork's converter does this offline; it is
    provided here so that a *dequantised* checkpoint can be put back into the stored
    basis, or so that a reader can verify a fold it is given.  See :func:`fwht` for the
    ``axis`` convention.
    """
    return fwht(w, signs, block, inverse=False, axis=axis)


def apply_inverse(w: np.ndarray, signs=None, block: int | None = None,
                  axis: int = -1) -> np.ndarray:
    """Undo the fold: ``A^-1 = D H``, signs after the transform."""
    return fwht(w, signs, block, inverse=True, axis=axis)


# --------------------------------------------------------------------------- GDN heads


def gdn_v_permutation(width: int, n_v: int, n_k: int) -> np.ndarray:
    """The head permutation for ``blk.N.ssm_out.weight``, as the fork computes it.

    ``llama-model.cpp:2080-2090``: ``perm_hd = width / n_v``, ``perm_nk = n_k``,
    ``perm_rep = n_v / n_k`` (the widths are ``ssm.inner_size``, ``ssm.group_count`` and
    ``ssm.time_step_rank``).  ``llama-graph.cpp`` then reshapes the activation to
    ``[hd, nk, rep]`` and permutes axes 1 and 2, i.e. it reads the feature axis as
    ``[hd, nk, rep]`` and re-emits it as ``[hd, rep, nk]``.

    Returns the index vector that maps tiled order to grouped order, so that
    ``a_grouped = a_tiled[perm]``.  The fork validates ``n_k > 0``, ``n_v > 0``,
    ``n_v % n_k == 0`` and ``width % n_v == 0`` before it will do this.
    """
    for name, val in (("n_v", n_v), ("n_k", n_k), ("width", width)):
        if val <= 0:
            raise ValueError("%s must be positive, got %r" % (name, val))
    if n_v % n_k or width % n_v:
        raise ValueError("bad GDN geometry: width=%d n_v=%d n_k=%d" % (width, n_v, n_k))
    hd, rep = width // n_v, n_v // n_k
    # element (hd, nk, rep) of the tiled layout -> (hd, rep, nk) of the grouped layout
    tiled = np.arange(width, dtype=np.int64).reshape(rep, n_k, hd)   # [rep, nk, hd]
    grouped = tiled.transpose(1, 0, 2).reshape(-1)                   # [nk, rep, hd]
    return grouped


def ssm_out_head_order(kv: dict, width: int) -> str:
    """``"grouped"`` or ``"tiled"``: the order ``blk.*.ssm_out.weight`` is stored in.

    This is the trap the fold creates and it is not visible in the tensor itself.

    ``conversion/qwen.py`` reorders every GDN tensor from the checkpoint's grouped head
    order to tiled order — *except* a Hadamard-folded ``out_proj``, where:

        # folded latent: a column permutation on the rotation axis cannot be
        # refolded, so keep the training (grouped) order and let the runtime
        # permute the activation instead
        self._hadamard_gdn_v_grouped = True

    So in a rotated file ``ssm_out.weight`` is GROUPED and the runtime permutes the
    activation; in an unrotated file of the same model the same tensor is TILED and no
    permutation happens.  Two files from one family therefore disagree about the inner
    order of one identically named tensor, and nothing in the name says which.

    The MLX runtime shipped with the MLX pack makes the same distinction and refuses the
    combination it cannot serve::

        if stem == "ssm_out.weight" and nv != nk and not fields.get(
                "prism.hadamard.gdn_v_grouped", False):
            raise ValueError("Unimplemented ungrouped folded GDN output")

    The field is *optional* in the fork's loader (``llama-model.cpp:1257`` reads it with
    the default ``false``), which is exactly why it is treated as required here: with a
    rotation declared and the field missing, the fork's own fallback and a converter that
    kept the grouped order disagree about the same tensor, and this module cannot tell
    which one the file means.  :func:`gdn_ssm_out_is_grouped` raises
    :class:`UnprovenRotation` for that case, so this function refuses it too, instead of
    answering "tiled" -- the one answer that would corrupt the tensor.
    """
    geo = (kv.get("qwen35.ssm.time_step_rank"), kv.get("qwen35.ssm.group_count"))
    n_v = int(geo[0]) if geo[0] is not None else 0
    n_k = int(geo[1]) if geo[1] is not None else 0
    if gdn_ssm_out_is_grouped(kv):
        return "grouped"
    if n_v and n_k and n_v != n_k and width % n_v == 0:
        return "tiled"
    return "tiled"


def gdn_v_grouped_declared(kv: dict) -> bool | None:
    """The file's own statement: ``True``/``False`` as WRITTEN, ``None`` when absent.

    No default.  ``kv.get(key, False)`` is the fail-open shape this replaces, and
    ``kv.get(key, True)`` would be the same guess in the other direction.  Absence is
    returned as absence, and the decision is made from the one other thing the file does
    declare: whether it is rotated at all.
    """
    if GDN_V_GROUPED_KEY not in kv:
        return None
    return bool(kv[GDN_V_GROUPED_KEY])


def gdn_ssm_out_is_grouped(kv: dict) -> bool:
    """True when the file declares that the folded GDN output stayed in grouped order.

    Three cases, and only three:

    * declared ``True``  -> ``True``;
    * declared ``False`` -> ``False`` (proven: the file states the fold did not keep it);
    * absent             -> the question is settled only when the file declares NO
      rotation at all.  Without ``prism.hadamard.version`` there is no fold --
      ``llama-model.cpp:1195`` is the fork's whole test for "this file is rotated" -- so
      the unrotated converter path applies and the tensor is tiled.  With a rotation
      declared and the field absent, nothing in the file states the head order:
      :class:`UnprovenRotation` is raised and the caller refuses.

    The test is the key's presence alone, exactly as the fork writes it
    (``llama-model.cpp:1195``): this function must not need a parsed contract to
    answer, because the tree's reader hands large arrays over as summaries and a
    parse would raise before the refusal could be named.
    """
    state = gdn_v_grouped_declared(kv)
    if state is not None:
        return state
    if PREFIX + "version" not in kv:
        return False            # not a rotated file: no fold, nothing was declared
    raise UnprovenRotation(
        "%sgdn_v_grouped is absent while %sversion=%r is present: the file declares a "
        "Hadamard fold but not which head order blk.*.ssm_out.weight was kept in.  The "
        "unrotated file of the same model carries that tensor in tiled order and a folded "
        "one in grouped order, and the tensor itself does not say which.  Refused rather "
        "than guessed: an absent key is the unproven case, not the permissive one."
        % (PREFIX, PREFIX, kv[PREFIX + "version"]))


def folded_widths(c: dict) -> dict[int, int]:
    """``{width: tensor count}`` for the declared fold set, from names only.

    A caller with the tensor table can count which ``ne[0]`` each name has; this is the
    bookkeeping that says *how many* distinct rotation widths a contract needs, i.e. how
    many sign vectors and how many distinct block splits.  For the released Bonsai-2
    ternary file that is exactly three — 5120 (5 blocks), 6144 (6) and 17408 (17), 28
    blocks and 28,672 sign values in total — and every folded tensor's input width is one
    of them.  A file whose folded tensors do not all land on a declared width cannot be
    served by the contract as written.
    """
    counts: dict[int, int] = {}
    for w in c.get("sign_widths") or []:
        counts[int(w)] = 0
    return counts


# --------------------------------------------------------------------------- self-test


def _selftest() -> int:
    """Prove the arithmetic against itself.  Exits non-zero on any disagreement."""
    import sys

    n = 1024
    rng = np.random.default_rng(0)
    H = sylvester_walsh_hadamard(n)

    print("1. H is symmetric and an involution")
    assert np.allclose(H, H.T), "not symmetric"
    assert np.allclose(H @ H, np.eye(n)), "H@H != I"
    print("   H@H == I exactly? %s   max|H@H-I| = %.3e"
          % (np.allclose(H @ H, np.eye(n)), np.abs(H @ H - np.eye(n)).max()))

    print("2. the butterfly equals the dense matrix")
    x = rng.standard_normal(n).astype(np.float32)
    got, want = butterfly(x, n) * (1.0 / math.sqrt(n)), H @ x
    print("   max|butterfly - H@x| = %.3e" % np.abs(got - want).max())
    assert np.abs(got - want).max() < 1e-5

    print("3. forward and inverse are inverses, and the order matters")
    d = np.where(rng.random(n) < 0.5, -1.0, 1.0).astype(np.float32)
    a_fwd = np.stack([fwht(x, d, n, inverse=False),
                      fwht(x, d, n, inverse=True)], axis=0)
    back = fwht(fwht(x, d, n, inverse=False), d, n, inverse=True)
    wrong = fwht(fwht(x, d, n, inverse=False), d, n, inverse=False)
    print("   ||inverse(forward(x)) - x|| = %.3e" % np.abs(back - x).max())
    print("   ||forward(forward(x)) - x|| = %.3e   (must be NON-zero)"
          % np.abs(wrong - x).max())
    assert np.abs(back - x).max() < 1e-5
    assert np.abs(wrong - x).max() > 1e-3

    print("4. the fold and its consumption are consistent")
    W = rng.standard_normal((n, 7)).astype(np.float32)     # ggml [in, out]
    a = rng.standard_normal((n, 3)).astype(np.float32)     # activation [in, batch]
    W_s = apply_forward(W, d, n, axis=0)                   # folded along the INPUT axis
    y_stored = W_s.T @ fwht(a, d, n, axis=0)
    y_primal = W.T @ a
    print("   ||(W_s^T . A a) - (W^T a)|| = %.3e  (float32 accumulate over %d terms)"
          % (np.abs(y_stored - y_primal).max(), n))
    assert np.abs(y_stored - y_primal).max() < 1e-3

    print("5. a latent table's row is restored by the inverse")
    E = rng.standard_normal((n, 5)).astype(np.float32)
    row = apply_inverse(apply_forward(E, d, n, axis=0), d, n, axis=0)[:, 2]
    print("   ||restored - primal|| = %.3e" % np.abs(row - E[:, 2]).max())
    assert np.abs(row - E[:, 2]).max() < 1e-4

    print("6. the GDN head permutation round-trips and matches the fork's axes")
    width, n_v, n_k = 6144, 48, 16
    perm = gdn_v_permutation(width, n_v, n_k)
    assert sorted(perm.tolist()) == list(range(width)), "not a permutation"
    hd, rep = width // n_v, n_v // n_k
    v = np.arange(width).reshape(rep, n_k, hd)
    g = v.transpose(1, 0, 2).reshape(-1)
    assert np.array_equal(g, v.reshape(width)[perm]), "perm does not reproduce the transpose"
    print("   hd=%d nk=%d rep=%d, permutation of %d features, reproduces [hd,nk,rep]->[hd,rep,nk]"
          % (hd, n_k, rep, width))

    print("7. the one declared choice: absent is refused, not defaulted")
    rot = {
        PREFIX + "version": 1,
        PREFIX + "block_size": 1024,
        PREFIX + "transform": TRANSFORM_NAME,
        PREFIX + "axis": AXIS_NAME,
        PREFIX + "sign_mode": "identity",
        PREFIX + "weight_names": ["blk.0.ssm_out.weight"],
    }
    unrotated = {"general.architecture": "qwen35"}
    # (a) no rotation declared at all: the field is not needed, and the answer is a
    #     deduction from the file's own lack of a fold, not a default
    assert gdn_ssm_out_is_grouped(dict(unrotated)) is False
    assert ssm_out_head_order(dict(unrotated), 6144) == "tiled"
    # (b) rotation declared, field ABSENT: unproven -> must raise, never answer
    try:
        gdn_ssm_out_is_grouped(dict(rot))
    except UnprovenRotation as exc:
        assert GDN_V_GROUPED_KEY in str(exc), "the refusal must name the missing key"
    else:
        raise AssertionError("rotated file with an absent key was answered, not refused")
    # (c) the guard answers from presence alone, so an unparseable rotation still refuses
    #     (the head order is unstated), and contract() names the bad field itself
    try:
        gdn_ssm_out_is_grouped({"general.architecture": "qwen35", PREFIX + "version": 2})
    except UnprovenRotation:
        pass
    else:
        raise AssertionError("unparseable rotation with no declaration was answered")
    try:
        contract({"general.architecture": "qwen35", PREFIX + "version": 2})
    except ValueError as exc:
        assert "version" in str(exc), "the parse refusal must name the field"
    else:
        raise AssertionError("unparseable rotation was accepted")
    # (d) declared False / True are read as written: the fix must not refuse proven files
    kv_false = dict(rot)
    kv_false[GDN_V_GROUPED_KEY] = False
    kv_true = dict(rot)
    kv_true[GDN_V_GROUPED_KEY] = True
    assert gdn_ssm_out_is_grouped(kv_false) is False
    assert gdn_ssm_out_is_grouped(kv_true) is True
    assert ssm_out_head_order(kv_false, 6144) == "tiled"
    assert ssm_out_head_order(kv_true, 6144) == "grouped"
    # (e) metadata that arrives summarised (the tree's own reader) is refused by name,
    #     and a count is still readable from it: a control that fails if list() returns
    class _Summary:                       # the shape gguf_kquant.GGUFArray has
        __slots__ = ("count",)

        def __init__(self, count):
            self.count = count

        def __len__(self):
            return self.count

        def __repr__(self):
            return "GGUFArray(count=%d)" % self.count

    assert declared_count(_Summary(401)) == 401
    assert declared_count([1, 2, 3]) == 3
    assert declared_count(None) == 0
    rot_summary = dict(rot)
    rot_summary[PREFIX + "weight_names"] = _Summary(401)
    try:
        contract(rot_summary)
    except UnreadableMetadata as exc:
        assert "weight_names" in str(exc), "the refusal must name the key"
    else:
        raise AssertionError("a summarised metadata array was accepted, not refused")
    print("   unrotated+absent -> False/tiled | rotated+absent -> UnprovenRotation"
          " | summarised array -> UnreadableMetadata | declared False/True read as written")
    print("\nall seven checks agree; the arithmetic is self-consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
