#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gguf_names.py — GGUF tensor names -> HuggingFace source keys, per architecture.

Why this exists
---------------
``tools/convert/gguf_extract.py`` carried a 12-entry table of qwen-shaped names
and dropped everything that did not match *silently*: the output checkpoint was
missing whole blocks while the tool exited 0.  Measured on the real
``Ornith-1.5-9B-Q4_K_M.gguf``: 442 tensors, 171 matched that table, 271 matched
nothing (see :func:`coverage`).

How this works
--------------
A *rule set* per architecture (``RULES``), not a table of 442 rows: 26 regex
patterns that between them cover every role of a qwen3.5/GDN-hybrid decoder, with
``{i}`` substituted per block.  Adding an architecture means adding a rule set,
not enumerating its tensors.

Two things the rules encode that the old table did not
------------------------------------------------------
1. **Shape.**  GGUF stores ``ne = (fastest-varying, ..., outermost)``, i.e. the
   reverse of the HF ``(out_features, in_features)`` convention.  The old code
   copied the GGUF dims straight into the safetensors header, so every emitted
   2-D tensor declared a transposed shape.  The row-major bytes are the same, so
   nothing failed -- a reader following the declared shape got a transposed
   matrix.  ``hf_shape()`` is the reversal (with ``conv1d`` keeping its unit
   middle axis; see ``CONV1D``).
2. **The MTP block.**  ``Ornith`` has 33 blocks and
   ``qwen35.nextn_predict_layers = 1``: block 32 is the nextn/draft block, and the
   main stack is 32 layers.  Calling ``block_count`` "layers" and treating block
   32 as an ordinary full-attention layer is what ``tools/archkit/gguf_spec.py``
   used to do; it now imports :func:`layer_split` instead, and
   ``tools/gui/model_import.py`` reaches the same rule through
   ``_gguf_main_layers`` -- so the spec generator, the wizard and this module all
   report 32 for that file.  :func:`layer_split` reports both numbers.

Provenance of the source keys
-----------------------------
The keys produced here are the ones the registered converters declare in
``tools/convert/qwen3_6_27b/recipe.py`` and
``tools/convert/qwen3_6_35b_a3b/inventory.py`` (``model.language_model.*``,
``linear_attn.*``, ``mtp.*``).  The role identifications were checked against
those modules' own declared shapes, e.g.

  * ``linear_attn.in_proj_z.weight`` is (value_dim, hidden) = (48, 5120) there;
    Ornith's ``blk.N.attn_gate.weight`` is ``[hidden=4096, 4096]`` -> (4096, 4096)
    and ``value_dim = 32 * 128 = 4096``.  Match.
  * ``linear_attn.in_proj_a/b.weight`` are (num_v_heads, hidden) = (48, 5120);
    Ornith's ``ssm_alpha/ssm_beta`` are ``[4096, 32]`` -> (32, 4096) with
    ``num_v_heads = ssm.time_step_rank = 32``.  Match.
  * ``mtp.fc.weight`` is (2*hidden, hidden) = (5120, 10240); Ornith's
    ``blk.32.nextn.eh_proj.weight`` is ``[4096, 8192]`` -> (8192, 4096) and
    ``2*hidden = 8192``.  Match.

Nothing here is inferred from a file name.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from tools.convert.gguf_kquant import read_tensor_table, type_name
except ImportError:  # run-by-path: python3 tools/convert/gguf_names.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.convert.gguf_kquant import read_tensor_table, type_name


class UnmappedTensors(RuntimeError):
    """Raised instead of dropping tensors: a partial checkpoint is a wrong model."""


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern
    target: str          # template; {i} = block index
    axis_order: str = "reversed"   # reversed | as-is | conv1d


#: Per-architecture rule sets.  Key = GGUF ``general.architecture``.
RULES: dict[str, tuple[Rule, ...]] = {
    # qwen3.5 / qwen3.6 / qwen3.8 text family, and Ornith-1.5 (which llama.cpp
    # writes as `qwen35`).  Common blocks apply to every layer; the attention
    # blocks differ by layer kind and are resolved by which name is present.
    "qwen35": (
        Rule(re.compile(r"^token_embd\.weight$"),
             "model.language_model.embed_tokens.weight"),
        Rule(re.compile(r"^output_norm\.weight$"), "model.language_model.norm.weight"),
        Rule(re.compile(r"^output\.weight$"), "lm_head.weight"),
        # --- nextn / MTP block-level tensors
        Rule(re.compile(r"^blk\.\d+\.nextn\.eh_proj\.weight$"), "mtp.fc.weight"),
        Rule(re.compile(r"^blk\.\d+\.nextn\.enorm\.weight$"),
             "mtp.pre_fc_norm_embedding.weight"),
        Rule(re.compile(r"^blk\.\d+\.nextn\.hnorm\.weight$"),
             "mtp.pre_fc_norm_hidden.weight"),
        Rule(re.compile(r"^blk\.\d+\.nextn\.shared_head_norm\.weight$"), "mtp.norm.weight"),
        # --- every layer
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.attn_norm\.weight$"),
             "{p}input_layernorm.weight"),
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.post_attention_norm\.weight$"),
             "{p}post_attention_layernorm.weight"),
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.ffn_gate\.weight$"), "{p}mlp.gate_proj.weight"),
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.ffn_up\.weight$"), "{p}mlp.up_proj.weight"),
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.ffn_down\.weight$"), "{p}mlp.down_proj.weight"),
        # --- full-attention layers
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.attn_q\.weight$"),
             "{p}self_attn.q_proj.weight"),
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.attn_k\.weight$"),
             "{p}self_attn.k_proj.weight"),
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.attn_v\.weight$"),
             "{p}self_attn.v_proj.weight"),
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.attn_output\.weight$"),
             "{p}self_attn.o_proj.weight"),
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.attn_q_norm\.weight$"),
             "{p}self_attn.q_norm.weight"),
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.attn_k_norm\.weight$"),
             "{p}self_attn.k_norm.weight"),
        # --- linear_attention (gated delta net) layers
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.attn_qkv\.weight$"),
             "{p}linear_attn.in_proj_qkv.weight"),
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.attn_gate\.weight$"),
             "{p}linear_attn.in_proj_z.weight"),
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.ssm_alpha\.weight$"),
             "{p}linear_attn.in_proj_a.weight"),
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.ssm_beta\.weight$"),
             "{p}linear_attn.in_proj_b.weight"),
        # gguf ssm_conv1d is (kernel, channels); HF conv1d is (channels, 1, kernel)
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.ssm_conv1d\.weight$"),
             "{p}linear_attn.conv1d.weight", "conv1d"),
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.ssm_dt\.bias$"), "{p}linear_attn.dt_bias"),
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.ssm_a$"), "{p}linear_attn.A_log"),
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.ssm_norm\.weight$"),
             "{p}linear_attn.norm.weight"),
        Rule(re.compile(r"^blk\.(?P<i>\d+)\.ssm_out\.weight$"),
             "{p}linear_attn.out_proj.weight"),
    ),
}


def _nextn_key(arch: str) -> str:
    return "%s.nextn_predict_layers" % arch


def _block_key(arch: str) -> str:
    return "%s.block_count" % arch


def layer_split(kv: dict, arch: str) -> tuple[int, int]:
    """(main_layers, nextn_layers) from the metadata, not from the block count.

    ``block_count`` counts the nextn/draft blocks too: Ornith reports 33, of which
    32 are the main stack and 1 is the draft block whose tensors are named
    ``blk.32.nextn.*``.  Treating 33 as ``layers`` and block 32 as ordinary gives a
    model that is one layer too deep and loads the draft weights as a normal layer.
    """
    blocks = int(kv.get(_block_key(arch)) or 0)
    nextn = int(kv.get(_nextn_key(arch)) or 0)
    if nextn > blocks:
        raise ValueError("%s.nextn_predict_layers=%d exceeds block_count=%d"
                         % (arch, nextn, blocks))
    return blocks - nextn, nextn


def hf_shape(gguf_dims: tuple, axis_order: str) -> tuple:
    """GGUF ``ne`` -> HF shape.  The row-major payload is unchanged by this."""
    dims = tuple(int(d) for d in gguf_dims)
    if axis_order == "as-is":
        return dims
    if axis_order == "conv1d":
        if len(dims) != 2:
            raise ValueError("conv1d expects 2 gguf dims (kernel, channels), got %r" % (dims,))
        return (dims[1], 1, dims[0])
    return tuple(reversed(dims))


def translate(arch: str, name: str, dims: tuple, kv: dict) -> tuple[str, tuple] | None:
    """One GGUF tensor name -> (hf_key, hf_shape), or None when unruled."""
    rules = RULES.get(arch)
    if rules is None:
        raise KeyError("no name rules for GGUF architecture %r (have: %s)"
                       % (arch, ", ".join(sorted(RULES))))
    main_layers, nextn = layer_split(kv, arch)
    for rule in rules:
        m = rule.pattern.match(name)
        if not m:
            continue
        target = rule.target
        if "{p}" in target:
            i = int(m.group("i"))
            if i >= main_layers:
                # the draft block's own attention/MLP lives under mtp.layers.0
                if i >= main_layers + nextn:
                    return None
                prefix = "mtp.layers.0."
            else:
                prefix = "model.language_model.layers.%d." % i
            target = target.replace("{p}", prefix)
        return target, hf_shape(dims, rule.axis_order)
    return None


def coverage(kv: dict, tensors, arch: str | None = None) -> dict:
    """mapped/total per tensor role, plus the unmapped remainder spelled out.

    ``tensors`` is the list from :func:`tools.convert.gguf_kquant.read_tensor_table`.
    """
    arch = arch or str(kv.get("general.architecture") or "")
    main_layers, nextn = layer_split(kv, arch)
    mapped: dict[str, tuple[str, tuple]] = {}
    unmapped: list[str] = []
    per_role: dict[str, int] = {}
    per_role_mapped: dict[str, int] = {}
    collisions: list[str] = []
    for name, dims, _ttype, _off in tensors:
        role = re.sub(r"^blk\.\d+\.", "blk.N.", name)
        per_role[role] = per_role.get(role, 0) + 1
        got = translate(arch, name, dims, kv)
        if got is None:
            unmapped.append(name)
            continue
        per_role_mapped[role] = per_role_mapped.get(role, 0) + 1
        key, shape = got
        if key in mapped:
            collisions.append("%s <- %s and %s" % (key, mapped[key][0], name))
            continue
        mapped[key] = (name, shape)
    return {
        "arch": arch, "main_layers": main_layers, "nextn_layers": nextn,
        "total": len(tensors), "mapped": len(mapped),
        "unmapped": unmapped, "collisions": collisions,
        "roles": {r: "%d/%d" % (per_role_mapped.get(r, 0), n) for r, n in sorted(per_role.items())},
    }


def require_full_coverage(report: dict) -> None:
    """Refuse loudly.  A name with no rule is a layer the artifact would not have."""
    problems = []
    if report["unmapped"]:
        problems.append("%d tensor(s) have no name rule for architecture %r: %s"
                        % (len(report["unmapped"]), report["arch"],
                           ", ".join(report["unmapped"][:8])
                           + (" ..." if len(report["unmapped"]) > 8 else "")))
    if report["collisions"]:
        problems.append("%d Hf key collision(s): %s" % (len(report["collisions"]),
                                                        "; ".join(report["collisions"][:4])))
    if problems:
        raise UnmappedTensors(
            "refusing to write a partial checkpoint -- " + " | ".join(problems))


def render(report: dict) -> str:
    lines = [
        "architecture   : %s" % report["arch"],
        "main layers    : %d   (+%d nextn/draft block(s))"
        % (report["main_layers"], report["nextn_layers"]),
        "coverage       : %d / %d tensors mapped" % (report["mapped"], report["total"]),
        "roles          :",
    ]
    for role, frac in report["roles"].items():
        lines.append("  %-36s %s" % (role, frac))
    if report["unmapped"]:
        lines.append("UNMAPPED (%d):" % len(report["unmapped"]))
        for n in report["unmapped"]:
            lines.append("  " + n)
    else:
        lines.append("UNMAPPED: none")
    return "\n".join(lines)


#: Exit status of every refusal this tool makes.  Not 1: a traceback also exits 1, and a
#: caller that only reads the status must be able to tell "REFUSED" from "the tool broke".
REFUSED_EXIT = 3


def _refuse(message: str) -> int:
    """One named refusal on stderr, plus the status that means "refused"."""
    print("REFUSED: %s" % message, file=sys.stderr)
    return REFUSED_EXIT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="report instead of refusing when tensors are unmapped")
    args = ap.parse_args(argv)
    # A source this reader cannot open, or cannot parse, is a refusal of THIS tool, not a
    # traceback: the exception class and its text are carried into the message, so turning
    # it into a refusal loses no information about the cause.
    try:
        kv, tensors, data_offset = read_tensor_table(args.gguf)
    except Exception as exc:                                 # noqa: BLE001 - refusal, not crash
        return _refuse("%s cannot be read as a GGUF tensor table: %s: %s"
                       % (args.gguf, type(exc).__name__, exc))
    # An architecture with no rule set, or metadata that contradicts itself, is the same
    # kind of answer: this tool has nothing to say about that file, and says so by name.
    try:
        rep = coverage(kv, tensors)
    except (KeyError, ValueError) as exc:
        detail = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        return _refuse("%s: %s (no report was produced for %s)"
                       % (type(exc).__name__, detail, args.gguf))
    rep["data_offset"] = data_offset
    # The report goes to stdout, the refusal to stderr: with --json, stdout stays one
    # JSON document instead of JSON followed by prose.
    print(json.dumps(rep, ensure_ascii=False, indent=2) if args.json else render(rep))
    if not args.allow_partial:
        try:
            require_full_coverage(rep)
        except UnmappedTensors as exc:
            return _refuse(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
