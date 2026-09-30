#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FlashNext (qwen4_exp) weight-bindings contract: runnable checks.

Covers the two mechanical defects that made the previously recorded audit
useless, plus a live gate against the real checkpoint index when it is present:

  1. prefix normalization  — the checkpoint nests the language tower under
     `model.language_model.`; without the rewrite exactly 1 of 74,520 contract
     entries matches (measured, 296,475-key index).
  2. NVFP4 companion accounting — the contract names only `.weight`; the
     checkpoint publishes quads (weight + weight_scale + weight_scale_2 +
     input_scale) for the 512-expert MoE tensors. They must be consumed by
     derivation or reported, never silently dropped.
  3. the audit itself must terminate: the shipped matcher was
     O(entries x sources) ~= 2.2e10 comparisons.

Run directly (`python3 test_flashnext_contract.py`) or under pytest.
The live gate is skipped, with a printed reason, when the index is not on this
box — there is no qwen4_exp/.ninfer artifact here, so nothing end-to-end runs.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARCHKIT = REPO / "tools" / "archkit"
if str(ARCHKIT) not in sys.path:
    sys.path.insert(0, str(ARCHKIT))

import flashnext_bindings as C  # noqa: E402

LIVE_INDEX = Path(os.environ.get(
    "NINFER_FLASHNEXT_INDEX",
    "/mnt/c/Users/User/Documents/ziqinzhang/models/"
    "Qwen3.8-Flash-Next-ABLITERATED-NVFP4/model.safetensors.index.json",
))

# Measured on the real index (296,475 keys) on 2026-09-17, re-pinned 2026-09-22
# (FN-SRC) for the idx_norm alias: matched +12, missing -12, unexpected -12.
# Pinned so that a
# contract edit which changes coverage shows up as a diff instead of drifting.
LIVE_SOURCE_KEYS = 296_475
LIVE_MATCHED = 74_294
LIVE_MISSING = 334
LIVE_COMPANIONS = 221_184        # 512 experts x 48 layers x 3 tensors x 3 cohorts
LIVE_QUANTISED = 73_728          # 512 experts x 48 layers x 3 tensors
LIVE_UNEXPECTED = 633            # all named in the residue report
                                 # (681 - 36 in_proj_b keys - 12 indexer k_layernorm keys)
LIVE_AMBIGUOUS = 0               # a_log and dt_bias are separate entries

_FAILURES: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        _FAILURES.append(label)


# ---------------------------------------------------------------------------
def _first_alias_key(entry: dict) -> str | None:
    """The concrete source key the first alias resolves to for this entry."""
    keys = C._expected_source_keys(entry)
    return keys[0] if keys else None


def _synthetic_full() -> list[str]:
    """Every engine satisfied: each entry's first alias, plus expert quads."""
    keys: list[str] = []
    for entry in C.E:
        k = _first_alias_key(entry)
        if k is None:
            continue
        keys.append(k)
        if k.endswith(".weight") and ".mlp.experts." in k:
            stem = k[: -len(".weight")]
            keys += [f"{stem}.{c}" for c in C._NVFP4_COMPANIONS]
    return keys


# ---------------------------------------------------------------------------
def test_normalize_strips_language_model() -> None:
    print("test_normalize_strips_language_model")
    check(C.normalize_source_key(
        "model.language_model.layers.5.linear_attn.in_proj_qkv.weight")
        == "model.layers.5.linear_attn.in_proj_qkv.weight",
        "strips model.language_model.")
    check(C.normalize_source_key("lm_head.weight") == "lm_head.weight",
          "leaves already-flat keys alone")
    check(C.normalize_source_key(C.normalize_source_key(
        "model.language_model.layers.0.mlp.gate.weight"))
        == "model.layers.0.mlp.gate.weight", "idempotent")


def test_layer_isolation() -> None:
    print("test_layer_isolation")
    # Layer 4 is a GDN layer (QSA sits at 3, 7, 11, ... 47).
    entry = next(e for e in C.E if e["engine"] == "layer.4.gdn.in_qkv")
    keys = C._expected_source_keys(entry)
    check(all(".4." in k for k in keys), "entry for layer 4 only yields layer-4 keys",
          repr(keys))
    check("model.layers.4.linear_attn.in_proj_qkv.weight" in keys,
          "the real checkpoint's name for this tensor is among them")
    # A key set containing only layer 4 must not satisfy any other layer.
    rep = C.audit(keys)
    check(rep["engines_covered"] == 1, "exactly one engine is covered",
          str(rep["engines_covered"]))
    check("layer.3.gdn.in_qkv" not in rep["missing_engines"]
          and rep["missing_count"] == len(C.E) - 1,
          "every other layer stays missing", str(rep["missing_count"]))


def test_audit_terminates_and_is_complete_on_a_full_synthetic_model() -> None:
    print("test_audit_terminates_and_is_complete_on_a_full_synthetic_model")
    keys = _synthetic_full()
    check(len(C.E) == 74_628,
          "contract is still 74,628 entries "
          "(74,520 + 36 a_log + 36 in_proj_a + 36 in_proj_b)",
          str(len(C.E)))
    rep = C.audit(keys)
    check(rep["missing_count"] == 0, "no engine left without a source",
          str(rep["missing_engines"][:3]))
    check(rep["unexpected_count"] == 0, "no unconsumed source key",
          str(rep["unexpected_sources"][:3]))
    check(rep["incomplete_quad_count"] == 0, "no incomplete NVFP4 quad",
          str(rep["incomplete_nvfp4_quads"][:3]))
    check(rep["complete"] is True, "audit reports complete")


def test_missing_engine_is_named() -> None:
    print("test_missing_engine_is_named")
    keys = _synthetic_full()
    victim = "model.layers.3.mlp.experts.17.gate_proj.weight"
    keys.remove(victim)
    rep = C.audit(keys)
    check(rep["missing_count"] == 1, "exactly one engine loses its source",
          str(rep["missing_count"]))
    check(rep["missing_engines"][0] == "layer.3.moe.e17.gate",
          "and it is named", str(rep["missing_engines"][:1]))


def test_nvfp4_companions_are_consumed_not_dropped() -> None:
    print("test_nvfp4_companions_are_consumed_not_dropped")
    keys = _synthetic_full()
    rep = C.audit(keys)
    check(rep["unexpected_count"] == 0,
          "quantiser keys are classified as companions, not unexpected")
    check(rep["companion_keys_consumed"] > 0, "companions were counted",
          str(rep["companion_keys_consumed"]))
    check(rep["quantised_tensors"] > 0, "quantised tensors were counted")


def test_incomplete_quad_is_flagged() -> None:
    print("test_incomplete_quad_is_flagged")
    keys = _synthetic_full()
    victim = "model.layers.3.mlp.experts.17.gate_proj.weight_scale_2"
    keys.remove(victim)
    rep = C.audit(keys)
    check(rep["incomplete_quad_count"] == 1, "one half-quantised tensor is flagged",
          str(rep["incomplete_quad_count"]))
    check(any("gate_proj" in s for s in rep["incomplete_nvfp4_quads"]),
          "and it is named", str(rep["incomplete_nvfp4_quads"][:2]))
    check(rep["complete"] is False, "so the audit is not complete")


def test_unbound_regions_are_named() -> None:
    print("test_unbound_regions_are_named")
    keys = _synthetic_full() + [
        "model.visual.blocks.0.attn.qkv.weight",
        "mtp.fc_hidden.weight",
    ]
    rep = C.audit(keys)
    check(rep["unexpected_count"] == 0,
          "vision and MTP keys are classified, not 'unexpected'",
          str(rep["unexpected_sources"][:3]))
    check(rep["unbound_regions"].get("model.visual.") == 1
          and rep["unbound_regions"].get("mtp.") == 1,
          "both regions reported with a count", str(rep["unbound_regions"]))


def test_a_log_and_dt_bias_are_separate_entries() -> None:
    """A_log is its own operator, not a second alias of dt_bias.

    This test USED to assert the fold as the expected state:
        check("layer.0.gdn.dt_bias" in rep["ambiguous_aliases"][0], ...)
    The fold is gone, so the same measurement now has to come out the other way; this is
    the fixed-side reading of the identical check.
    """
    print("test_a_log_and_dt_bias_are_separate_entries")
    rep = C.audit(_synthetic_full())
    check(rep["ambiguous_aliases"] == [],
          "no contract entry is satisfied by two source keys",
          str(rep["ambiguous_aliases"][:1]))
    check("layer.0.gdn.a_log" not in rep["missing_engines"],
          "the a_log engine has its own source")
    check("layer.0.gdn.dt_bias" not in rep["missing_engines"],
          "the dt_bias engine keeps its own source")
    a_log = C._expected_source_keys(
        next(e for e in C.E if e["engine"] == "layer.0.gdn.a_log"))
    check(a_log[0] == "model.layers.0.linear_attn.A_log",
          "the a_log entry binds the checkpoint's A_log FIRST", repr(a_log))
    dt_bias = C._expected_source_keys(
        next(e for e in C.E if e["engine"] == "layer.0.gdn.dt_bias"))
    check("model.layers.0.linear_attn.A_log" not in dt_bias,
          "while dt_bias no longer accepts A_log", repr(dt_bias))
    in_proj_a = C._expected_source_keys(
        next(e for e in C.E if e["engine"] == "layer.0.gdn.in_proj_a"))
    check("model.layers.0.linear_attn.in_proj_a.weight" in in_proj_a,
          "and in_proj_a has a contract home", repr(in_proj_a))
    in_proj_b = C._expected_source_keys(
        next(e for e in C.E if e["engine"] == "layer.0.gdn.in_proj_b"))
    check("model.layers.0.linear_attn.in_proj_b.weight" in in_proj_b,
          "and in_proj_b has a contract home", repr(in_proj_b))
    check(all("A_log" not in k and "dt_bias" not in k for k in in_proj_b),
          "in_proj_b is not folded onto any bias entry", repr(in_proj_b))
    # The real checkpoint carries BOTH keys per layer; the live gate measures that.
    rep = C.audit(_synthetic_full() + ["model.layers.0.linear_attn.A_log",
                                       "model.layers.0.linear_attn.dt_bias"])
    check(rep["ambiguous_aliases"] == [],
          "both real keys present, still no ambiguity",
          str(rep["ambiguous_aliases"][:1]))


def test_unknown_prefix_is_still_rejected() -> None:
    print("test_unknown_prefix_is_still_rejected")
    keys = _synthetic_full() + ["model.mystery_tower.0.weight"]
    rep = C.audit(keys)
    check(rep["unexpected_count"] == 1, "an unnamed region is reported as unexpected",
          str(rep["unexpected_count"]))
    check(rep["unexpected_sources"][0] == "model.mystery_tower.0.weight",
          "and listed verbatim", str(rep["unexpected_sources"][:1]))


# ---------------------------------------------------------------------------
def test_live_index_gate() -> None:
    print("test_live_index_gate")
    if not LIVE_INDEX.exists():
        print(f"  SKIP  real checkpoint index not on this box: {LIVE_INDEX}")
        print("        (no qwen4_exp/.ninfer artifact exists here either, so no")
        print("         end-to-end FlashNext run is possible on this machine)")
        return
    idx = json.loads(LIVE_INDEX.read_text(encoding="utf-8"))
    keys = list(idx["weight_map"].keys())
    rep = C.audit(keys)
    check(len(keys) == LIVE_SOURCE_KEYS, "source key count",
          f"{len(keys)} != {LIVE_SOURCE_KEYS}")
    check(rep["matched_sources"] == LIVE_MATCHED, "matched sources",
          f"{rep['matched_sources']} != {LIVE_MATCHED}")
    check(rep["missing_count"] == LIVE_MISSING, "missing engines",
          f"{rep['missing_count']} != {LIVE_MISSING}")
    check(rep["companion_keys_consumed"] == LIVE_COMPANIONS, "companions consumed",
          f"{rep['companion_keys_consumed']} != {LIVE_COMPANIONS}")
    check(rep["quantised_tensors"] == LIVE_QUANTISED, "quantised tensors",
          f"{rep['quantised_tensors']} != {LIVE_QUANTISED}")
    check(rep["incomplete_quad_count"] == 0, "every started NVFP4 quad is complete",
          str(rep["incomplete_nvfp4_quads"][:3]))
    check(rep["unexpected_count"] == LIVE_UNEXPECTED, "unexpected residue",
          f"{rep['unexpected_count']} != {LIVE_UNEXPECTED}")
    norm = {C.normalize_source_key(x) for x in keys}
    ambiguous = [e for e in C.E
                 if len([k for k in C._expected_source_keys(e) if k in norm]) > 1]
    check(len(ambiguous) == LIVE_AMBIGUOUS, "ambiguous entries",
          f"{len(ambiguous)} != {LIVE_AMBIGUOUS}")
    # The residue must stay *named*: every unexpected key is real model weight
    # that the contract does not yet cover, and the report says which.
    print("        residue is fully enumerated in the audit report "
          f"({rep['unexpected_count']} keys, {rep['missing_count']} missing engines)")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main() -> int:
    print(f"flashnext contract tests (repo={REPO})")
    print(f"live index: {LIVE_INDEX}")
    print()
    for fn in TESTS:
        fn()
    print()
    if _FAILURES:
        print(f"FAILED {len(_FAILURES)}/{len(TESTS)}: {_FAILURES}")
        return 1
    print(f"OK  {len(TESTS)}/{len(TESTS)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
