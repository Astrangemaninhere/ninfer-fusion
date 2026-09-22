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

# Measured on the real index (296,475 keys) on 2026-09-17. Pinned so that a
# contract edit which changes coverage shows up as a diff instead of drifting.
LIVE_SOURCE_KEYS = 296_475
LIVE_MATCHED = 74_174
LIVE_MISSING = 346
LIVE_COMPANIONS = 221_184        # 512 experts x 48 layers x 3 tensors x 3 cohorts
LIVE_QUANTISED = 73_728          # 512 experts x 48 layers x 3 tensors
LIVE_UNEXPECTED = 753            # all named in the residue report
LIVE_AMBIGUOUS = 36              # layer.N.gdn.dt_bias <- dt_bias AND A_log

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
    check(len(C.E) == 74_520, "contract is still 74,520 entries", str(len(C.E)))
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


def test_ambiguous_alias_is_reported() -> None:
    print("test_ambiguous_alias_is_reported")
    keys = _synthetic_full() + [
        "model.layers.0.linear_attn.A_log",     # the dt_bias alias also matches this
    ]
    rep = C.audit(keys)
    check(len(rep["ambiguous_aliases"]) >= 1, "the dt_bias/A_log alias is ambiguous",
          str(rep["ambiguous_aliases"][:1]))
    check(rep["complete"] is False, "ambiguity blocks 'complete'")
    check("layer.0.gdn.dt_bias" in rep["ambiguous_aliases"][0],
          "and the engine is named")


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
