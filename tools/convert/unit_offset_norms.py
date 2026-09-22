#!/usr/bin/env python3
"""The unit-offset norm gain, declared once, for every GGUF-fed writer in this tree.

WHY THIS FILE EXISTS
--------------------
The engine does not store a plain RMSNorm gain.  For every norm in this family except the
gated one, the epilogue adds one to the stored weight before it multiplies:

    src/ops/kernel/rmsnorm.cuh:22
        if constexpr (Epilogue == RmsEpilogue::Offset) { weight += 1.0f; }

and the GDN control projection reaches that same epilogue with the flag set:

    src/ops/generic/rowsplit_generic.cu:519
        rmsnorm(x, norm_weight, eps, /*unit_offset=*/true, h, stream);

so the artifact has to store ``gamma - 1`` and not ``gamma``.  The official (ModelOpt /
HF) checkpoint route stores ``gamma - 1``.  A GGUF stores the plain ``gamma``
(``blk.N.attn_norm.weight`` and its siblings), so a writer that copies a GGUF into HF
names WITHOUT subtracting one hands the engine ``gamma``, the engine computes
``1 + gamma``, and every layer's input is scaled by roughly two.

MEASURED, on disk, on two artifacts that declare the same ``model_id = qwen3.8-27b``:

  * ``qwen3_8_27b_nvfp4.ninfer``     (weights_id ``nvfp4``, registered, speaks)
      ``text/layers/0/input_norm``          mean -0.033365
      ``text/layers/0/post_attention_norm`` mean -0.217285
  * ``Ternary-Bonsai-...gdnfix.bonsairun.ninfer`` (weights_id ``groupwise-int``, gibbers)
      ``text/layers/0/input_norm``          mean +0.968102
      ``text/layers/0/post_attention_norm`` mean +0.780193

Same objects, one higher by 0.9980..1.0010 in all 64 layers and in every unit-offset norm,
while ``text/layers/0/gdn/norm`` -- the gated one, which the engine does NOT offset --
differs by only -0.0065.  Readings: ``dl/normfix/logs/12_greensum.txt``, ``15b.txt``,
``13_tails.txt``.

WHAT THIS MODULE IS
-------------------
The set, stated once, in both of the name spaces it has to be stated in, plus the
transform, plus a self-test that can fail.  ``hf_key_needs_deoffset`` keys on the HF names
``tools/convert/gguf_names.py`` emits; ``object_needs_deoffset`` keys on the artifact object
names the inventory modules emit.  The two spaces are disjoint in shape, so one entry point
(``stored_values``) can ask both.

SCOPE, AND WHY IT IS NOT IN THE SHARED ARTIFACT ENCODER
-------------------------------------------------------
Only writers whose INPUT IS A GGUF go through this module.  The official checkpoint route
reads an HF/ModelOpt shard directory that already holds ``gamma - 1``, so subtracting one
there would take it to ``gamma - 2``.  That is why the transform lives here and at the GGUF
readers, and NOT in ``tools/convert/qwen3_6/common/conversion.py``'s
``encode_tensor_payload``, which the repack route and the official route SHARE.

Three exclusions are named rather than inferred, because each of them would otherwise be
swallowed by a plausible-looking rule and each of them would be a silent wrong:

  * ``.../gdn/norm`` -- the gated norm, whose leaf is ``ops::gated_rmsnorm`` and takes no
    offset.  Its HF key ``...linear_attn.norm.weight`` ends with ``.norm.weight``, so a
    suffix rule on ``.norm.weight`` would flag it.  ``--self-test`` demonstrates that trap
    on purpose.
  * ``dflash2/`` and ``dflash/`` -- the draft-head layers.  Their leaves call
    ``ops::rmsnorm(..., false, ...)`` (``src/targets/qwen3_6/impl/runtime/dflash2_impl.h:254``
    and ``dflash_impl.h:248``), so their stored value IS the plain gain and subtracting one
    would be the defect rather than the fix.
  * everything outside ``text/layers/``, ``text/final_norm`` and ``mtp/`` -- the object-name
    predicate is stated against those namespaces rather than against any name ending in a
    norm-ish suffix.
"""
from __future__ import annotations

import sys

# --------------------------------------------------------------------------- name spaces

#: The artifact object namespaces this module claims.  Named, not inferred.
OBJ_NAMESPACES = (
    "text/layers/",
    "text/final_norm",
    "mtp/",
)

#: Artifact object names, as written by the inventory modules, that carry a norm the engine
#: offsets.  Suffix form.
OBJ_SUFFIXES = (
    "/input_norm",
    "/post_attention_norm",
    "/query_norm",
    "/key_norm",
    "/final_norm",
    "/embedding_norm",
    "/hidden_norm",
)

#: The gated norm, by object name.  Named, not inferred.
OBJ_NOT_OFFSET_SUFFIXES = (
    "/gdn/norm",
)

#: HF source keys, as emitted by ``tools/convert/gguf_names.py`` for the qwen35 family, that
#: carry a norm the engine offsets.  Suffix form.
HF_SUFFIX_KEYS = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
)

#: HF source keys that are tested exactly, because a suffix rule on them would be ambiguous.
HF_EXACT_KEYS = (
    "model.language_model.norm.weight",     # -> text/final_norm
    "mtp.pre_fc_norm_embedding.weight",     # -> mtp/embedding_norm
    "mtp.pre_fc_norm_hidden.weight",        # -> mtp/hidden_norm
    "mtp.norm.weight",                      # -> mtp/final_norm
)

#: HF source keys the engine does NOT offset.  ``...linear_attn.norm.weight`` ->
#: ``text/layers/N/gdn/norm``, the gated norm.
HF_NOT_OFFSET_KEYS = (
    "linear_attn.norm.weight",
)


def hf_key_needs_deoffset(key: str) -> bool:
    """True when a GGUF-sourced HF tensor key holds a norm the engine consumes as 1 + w."""
    if key.endswith(HF_NOT_OFFSET_KEYS) or key in HF_NOT_OFFSET_KEYS:
        return False
    if key in HF_EXACT_KEYS:
        return True
    return key.endswith(HF_SUFFIX_KEYS)


def object_needs_deoffset(name: str) -> bool:
    """True when an artifact OBJECT name holds a norm the engine consumes as 1 + w."""
    if not any(name.startswith(ns) or name == ns for ns in OBJ_NAMESPACES):
        return False
    if name.endswith(OBJ_NOT_OFFSET_SUFFIXES):
        return False
    return name.endswith(OBJ_SUFFIXES)


def needs_deoffset(name: str) -> bool:
    return object_needs_deoffset(name) or hf_key_needs_deoffset(name)


def stored_values(values, name: str):
    """``values`` minus one when ``name`` is a unit-offset norm, unchanged otherwise.

    ``values`` is an fp32 array (numpy or torch) or anything that supports ``- 1.0``.
    Subtracting one invents no precision: if ``v`` is representable in bf16 then so is
    ``v - 1`` for the magnitudes at issue, so a bf16 round trip afterwards is exact.
    """
    if not needs_deoffset(name):
        return values
    return values - 1.0


# --------------------------------------------------------------------------- control

#: A deliberately WRONG predicate, kept so the self-test can show that the explicit
#: exclusion above is load-bearing rather than decorative.
def _naive_suffix_predicate(key: str) -> bool:
    return key.endswith(".norm.weight")


def _self_test() -> int:
    import numpy as np

    failures = [0]

    def check(label, got, want):
        ok = got == want
        failures[0] += 0 if ok else 1
        print("  %-62s %-6s %s"
              % (label, str(got), "OK" if ok else "*** FAIL (want %s)" % want))

    print("=== the gated norm must NOT be flagged, and the trap is real ===")
    check("object text/layers/0/gdn/norm",
          object_needs_deoffset("text/layers/0/gdn/norm"), False)
    check("hf key ...layers.0.linear_attn.norm.weight",
          hf_key_needs_deoffset("model.language_model.layers.0.linear_attn.norm.weight"), False)
    check("a naive '.norm.weight' suffix rule WOULD flag it",
          _naive_suffix_predicate(
              "model.language_model.layers.0.linear_attn.norm.weight"), True)

    print("=== the dflash draft layers must NOT be flagged (their leaf uses false) ===")
    check("object dflash2/layers/0/input_norm",
          object_needs_deoffset("dflash2/layers/0/input_norm"), False)
    check("object dflash/layers/0/input_norm",
          object_needs_deoffset("dflash/layers/0/input_norm"), False)

    print("=== the flagged set ===")
    for k in ("model.language_model.layers.7.input_layernorm.weight",
              "model.language_model.layers.7.post_attention_layernorm.weight",
              "model.language_model.layers.7.self_attn.q_norm.weight",
              "model.language_model.layers.7.self_attn.k_norm.weight",
              "model.language_model.norm.weight",
              "mtp.pre_fc_norm_embedding.weight",
              "mtp.pre_fc_norm_hidden.weight",
              "mtp.norm.weight"):
        check("hf key " + k, hf_key_needs_deoffset(k), True)
    for n in ("text/layers/0/input_norm", "text/layers/0/post_attention_norm",
              "text/layers/3/attention/query_norm", "text/layers/3/attention/key_norm",
              "text/final_norm", "mtp/layer/input_norm", "mtp/layer/post_attention_norm",
              "mtp/layer/attention/query_norm", "mtp/layer/attention/key_norm",
              "mtp/embedding_norm", "mtp/hidden_norm", "mtp/final_norm"):
        check("object " + n, object_needs_deoffset(n), True)

    print("=== the untouched set ===")
    for n in ("text/layers/0/gdn/a_log", "text/layers/0/gdn/convolution",
              "text/layers/0/gdn/norm", "text/layers/0/mlp/gate_up",
              "text/layers/0/attention/output", "text/token_embedding",
              "text/output_head", "mtp/input_projection", "vision/layers/0/norm1/weight"):
        check("object " + n, object_needs_deoffset(n), False)
    for k in ("model.language_model.embed_tokens.weight",
              "model.language_model.layers.0.linear_attn.conv1d.weight",
              "lm_head.weight", "model.language_model.layers.0.mlp.gate_proj.weight"):
        check("hf key " + k, hf_key_needs_deoffset(k), False)

    print("=== the arithmetic ===")
    v = np.array([1.0, 1.5, 0.5, -0.25], dtype=np.float32)
    check("gamma - 1 on a known vector",
          [round(float(x), 6) for x in stored_values(v, "text/layers/0/input_norm")],
          [0.0, 0.5, -0.5, -1.25])
    check("gdn/norm is untouched",
          [float(x) for x in stored_values(v, "text/layers/0/gdn/norm")],
          [1.0, 1.5, 0.5, -0.25])
    check("a bf16 value stays exact through -1",
          [float(np.float32(np.float16(1.5) - 1.0))], [0.5])

    print("self-test failures: %d" % failures[0])
    return failures[0]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--self-test":
        return 1 if _self_test() else 0
    if argv and argv[0] == "--names":
        for n in argv[1:]:
            print("%-58s %s" % (n, "DE-OFFSET" if needs_deoffset(n) else "leave alone"))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
