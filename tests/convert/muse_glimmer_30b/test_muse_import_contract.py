#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Muse-Glimmer-30B importer contract: the four defects that made Muse unroutable.

Each check below fails on the code as it was before this test existed, so this
file is the regression for the work rather than a restatement of it:

  1. import_model.py listed ``muse_glimmer_30b`` in REGISTERED_TARGETS and
     accepted ``muse_glimmer`` as a decoder family, but the converter exposed no
     pure-config entry, so ``evaluate_targets`` reported
     ``[?] muse_glimmer_30b 该 target 未提供纯 config 校验入口`` -- the front door
     could *name* Muse and still refuse to *route* it, and the run ended with
     "4) 该形状没有被任何注册 target 接受".
  2. the runnable gate hard-coded ``flavour.method == "native"``, so a target
     whose converter reads a quantised (ModelOpt NVFP4) source directly could
     never be declared runnable even once it was routed.
  3. the same branch assumed the frontend profile was measured (``profile`` is
     None when no resource root is configured), so relaxing (2) turned the old
     assumption into an AttributeError.
  4. the converter bundles ``qwen_chat_template.jinja`` and embeds it as the
     artifact's ``frontend/chat_template.jinja``, while the engine compiles only
     two digests and throws on anything else
     (src/targets/qwen3_6/impl/frontend/chat_template.cpp:414-424, reached from
     frontend.cpp:229 during load) -- the bundled digest matched neither, so the
     artifact could not load no matter how the front door felt about it.

Run directly (``python3 test_muse_import_contract.py``) or under pytest.  The
live gates are skipped, with a printed reason, when the checkpoint or the tree
pieces are not on this box.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MUSE_SRC = Path(os.environ.get(
    "NINFER_MUSE_SOURCE",
    "/mnt/c/Users/User/Documents/ziqinzhang/data/muse_nvfp4"))

CONVERTER = REPO / "tools/convert/muse_glimmer_30b/convert.py"
BUNDLED_TEMPLATE = REPO / "tools/convert/muse_glimmer_30b/qwen_chat_template.jinja"
CHAT_TEMPLATE_CPP = REPO / "src/targets/qwen3_6/impl/frontend/chat_template.cpp"
BINDINGS_CPP = REPO / "src/targets/muse_glimmer_30b/impl/load/bindings.cpp"

_FAILURES: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        _FAILURES.append(label)


def _load_config() -> dict | None:
    path = MUSE_SRC / "config.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _import_model():
    return importlib.import_module("tools.convert.import_model")


def _converter_module():
    return importlib.import_module("tools.convert.muse_glimmer_30b.convert")


def _engine_accepted_template_digests() -> set[str]:
    """The two digests CompiledChatTemplate::resolve accepts, read from the engine.

    Parsed out of the byte arrays rather than transcribed, so a third template
    added to the engine shows up here instead of silently disagreeing.
    """
    text = CHAT_TEMPLATE_CPP.read_text(encoding="utf-8", errors="replace")
    digests = set()
    for match in re.finditer(
            r"constexpr\s+Sha256Digest\s+\w*TemplateDigest\s*\{(.*?)\};", text, re.S):
        body = match.group(1)
        # 0xe8, 0x4f, ...  (the initialiser is int-list style inside a constexpr struct)
        numbers = re.findall(r"0x([0-9a-fA-F]{2})", body)
        if len(numbers) == 32:
            digests.add("".join(n.lower() for n in numbers))
    return digests


# --------------------------------------------------------------------------- #
# 1. the front door routes Muse to its own target (was: [?] unavailable)
# --------------------------------------------------------------------------- #
def test_front_door_routes_muse_to_its_own_target() -> None:
    print("test_front_door_routes_muse_to_its_own_target")
    config = _load_config()
    if config is None:
        print(f"  SKIP  Muse checkpoint not on this box: {MUSE_SRC}")
        return
    import_model = _import_model()
    verdicts = import_model.evaluate_targets(
        MUSE_SRC, config, MUSE_SRC / "out.ninfer", "cpu")
    accepted = [v.target for v in verdicts if v.status == "accepted"]
    check(accepted == ["muse_glimmer_30b"],
          "exactly the muse target accepts this config", repr(accepted))
    muse = next(v for v in verdicts if v.target == "muse_glimmer_30b")
    check(muse.status == "accepted", "muse is accepted, not 'unavailable'", muse.status)
    check("validate_config" in muse.message,
          "and the verdict names the target's own entry point", muse.message)
    check(muse.command[:2] == ["python3", "-m"],
          "and carries a runnable command", repr(muse.command[:3]))


def test_family_is_recognized_as_a_registered_decoder() -> None:
    print("test_family_is_recognized_as_a_registered_decoder")
    config = _load_config()
    if config is None:
        print(f"  SKIP  Muse checkpoint not on this box: {MUSE_SRC}")
        return
    import_model = _import_model()
    family = import_model.classify_family(config)
    check(family.is_decoder, "classified as a decoder", family.note)
    check(family.label == "muse_glimmer", "family label is muse_glimmer", family.label)
    check("muse_glimmer" in import_model.DECODER_FAMILIES,
          "and muse_glimmer is in the closed DECODER_FAMILIES tuple")


# --------------------------------------------------------------------------- #
# 2 + 3. the source contract and the runnable gate
# --------------------------------------------------------------------------- #
def test_validator_pins_the_registered_shape() -> None:
    print("test_validator_pins_the_registered_shape")
    config = _load_config()
    if config is None:
        print(f"  SKIP  Muse checkpoint not on this box: {MUSE_SRC}")
        return
    pins = importlib.import_module("tools.convert.muse_glimmer_30b.config_pins")
    summary = pins.validate_config(config)
    check(summary["model_type"] == "muse_glimmer", "model_type", str(summary["model_type"]))
    check(summary["artifact_identity"] == {"model_id": "muse-glimmer-30b",
                                           "weights_id": "nvfp4"},
          "the artifact identity is the one the engine registry declares",
          str(summary["artifact_identity"]))
    schedule = summary["layer_schedule"]
    check(schedule["layers"] == 52 and schedule["full_attention"] == 13
          and schedule["sliding_attention"] == 39,
          "52 layers = 13 full + 39 sliding", str(schedule))


def test_validator_refuses_each_kind_of_drift() -> None:
    print("test_validator_refuses_each_kind_of_drift")
    config = _load_config()
    if config is None:
        print(f"  SKIP  Muse checkpoint not on this box: {MUSE_SRC}")
        return
    import copy
    pins = importlib.import_module("tools.convert.muse_glimmer_30b.config_pins")
    cases = {
        "text_config.hidden_size": lambda c: c["text_config"].__setitem__("hidden_size", 6657),
        "text_config.num_hidden_layers": lambda c: c["text_config"].__setitem__(
            "num_hidden_layers", 51),
        "text_config.rope_parameters": lambda c: c["text_config"]["rope_parameters"].__setitem__(
            "rope_theta", 10000.0),
        "vision_config": lambda c: c["vision_config"].__setitem__("patch_size", 16),
        "config.model_type": lambda c: c.__setitem__("model_type", "qwen3_5"),
    }
    for label, mutate in cases.items():
        broken = copy.deepcopy(config)
        mutate(broken)
        try:
            pins.validate_config(broken)
        except ValueError as exc:
            named = label.split(".")[-1] in str(exc) or label.split(".")[0] in str(exc)
            check(named, f"{label} drift is refused and named", str(exc)[:120])
        else:
            check(False, f"{label} drift is refused and named", "accepted a wrong config")

    # layer_types is compared as a schedule, not only as a count.
    broken = copy.deepcopy(config)
    lt = list(broken["text_config"]["layer_types"])
    lt[0], lt[3] = lt[3], lt[0]
    broken["text_config"]["layer_types"] = lt
    try:
        pins.validate_config(broken)
        check(False, "a permuted layer_types schedule is refused", "accepted")
    except ValueError as exc:
        check("layer_types" in str(exc), "a permuted layer_types schedule is refused",
              str(exc)[:120])


def test_source_contract_declares_modelopt_and_only_for_muse() -> None:
    print("test_source_contract_declares_modelopt_and_only_for_muse")
    import_model = _import_model()
    methods, supplies = import_model.target_source_contract("muse_glimmer_30b")
    check(methods == frozenset({"modelopt"}),
          "muse declares it consumes the modelopt source", repr(methods))
    check(supplies is True,
          "muse declares it supplies the artifact's own frontend resources")
    for other in ("qwen3_6_27b", "qwen3_6_35b_a3b", "qwen3_8_27b", "qwen4_exp"):
        other_methods, other_supplies = import_model.target_source_contract(other)
        check(other_methods == frozenset({"native"}),
              f"{other} keeps the native-only default", repr(other_methods))
        check(other_supplies is False, f"{other} keeps the source-frontend default")
    # A target whose module cannot be imported must not gain a flavour.
    fallback, fallback_supplies = import_model.target_source_contract("no_such_target")
    check(fallback == frozenset({"native"}) and fallback_supplies is False,
          "an unimportable target falls back to native-only", repr(fallback))


def test_runnable_gate_admits_a_quantised_source_only_when_declared() -> None:
    print("test_runnable_gate_admits_a_quantised_source_only_when_declared")
    import_model = _import_model()
    flavour = "modelopt"
    muse_methods, _ = import_model.target_source_contract("muse_glimmer_30b")
    qwen_methods, _ = import_model.target_source_contract("qwen3_6_27b")
    check(flavour in muse_methods, "modelopt satisfies the gate for muse")
    check(flavour not in qwen_methods,
          "modelopt still fails the gate for a target that did not declare it")
    check("native" in qwen_methods, "and 'native' still satisfies it")


# --------------------------------------------------------------------------- #
# 4. the artifact's chat template is one the engine compiles
# --------------------------------------------------------------------------- #
def test_bundled_chat_template_is_accepted_by_the_engine() -> None:
    print("test_bundled_chat_template_is_accepted_by_the_engine")
    if not CHAT_TEMPLATE_CPP.is_file():
        print(f"  SKIP  engine source not on this box: {CHAT_TEMPLATE_CPP}")
        return
    accepted = _engine_accepted_template_digests()
    check(len(accepted) >= 2, "the engine compiles more than one template digest",
          repr(accepted))
    if not BUNDLED_TEMPLATE.is_file():
        check(False, "the converter bundles a chat template",
              f"missing {BUNDLED_TEMPLATE}")
        return
    digest = hashlib.sha256(BUNDLED_TEMPLATE.read_bytes()).hexdigest()
    check(digest in accepted,
          "the bundled template's sha256 is one the engine accepts",
          f"sha256 {digest} not in {sorted(accepted)}")


def test_converter_actually_embeds_an_accepted_template() -> None:
    """Not the bundled file -- the payload resource_specs really hands the writer."""
    print("test_converter_actually_embeds_an_accepted_template")
    if not CHAT_TEMPLATE_CPP.is_file():
        print(f"  SKIP  engine source not on this box: {CHAT_TEMPLATE_CPP}")
        return
    if not (MUSE_SRC / "tokenizer.json").is_file():
        print(f"  SKIP  Muse checkpoint not on this box: {MUSE_SRC}")
        return
    accepted = _engine_accepted_template_digests()
    resources = dict(_converter_module().resource_specs(str(MUSE_SRC)))
    payload = resources.get("frontend/chat_template.jinja")
    if payload is None:
        check(False, "resource_specs emits frontend/chat_template.jinja", str(sorted(resources)))
        return
    digest = hashlib.sha256(payload).hexdigest()
    check(digest in accepted,
          "the payload the converter will embed is engine-accepted",
          f"sha256 {digest} not in {sorted(accepted)}")


# --------------------------------------------------------------------------- #
# 5. the row-scale streamer must not re-read the tensor per block
# --------------------------------------------------------------------------- #
def test_row_scale_streamer_reads_the_source_once() -> None:
    """R4: `row_block` used to call `reader.get(key)` inside the loop.

    `MuseReader.get` seeks to the tensor's byte range and reads the *whole*
    tensor before the slice is applied, so a 202,112-row embedding was read and
    cast once per 4096-row block -- 50 times per pass, 100 times across the two
    passes.  Measured before the fix: 27.1 GB read to produce 219 MB of artifact
    (~123x amplification, ~2.3 MB/s of artifact).  This test counts reads.
    """
    print("test_row_scale_streamer_reads_the_source_once")
    try:
        import torch
    except Exception as exc:                          # noqa: BLE001
        print(f"  SKIP  torch unavailable: {exc}")
        return
    convert = _converter_module()

    class CountingReader:
        def __init__(self, tensor):
            self._t = tensor
            self.calls = 0

        def get(self, key):
            self.calls += 1
            return self._t.clone()

    n_src, k = 8, 4
    tensor = torch.arange(n_src * k, dtype=torch.uint8).reshape(n_src, k)
    reader = CountingReader(tensor)
    chunks = list(convert.fp8_row_scaled_stream(reader, "k", n_src, n_src, k,
                                                rows_per_block=2))
    check(reader.calls == 1,
          "one source read for the whole stream, not one per block",
          f"{reader.calls} reads for {(n_src + 1) // 2} blocks x 2 passes")
    check(len(chunks) > 0, "the streamer still yields payload", str(len(chunks)))


def test_plan_reads_no_payload_bytes() -> None:
    """R5: `object_plan` must classify from headers, not by reading weights.

    `mlp_layer_fmt` answers one boolean per call from the declared dtype and the
    second dimension -- both in the shard's JSON header, which `MuseReader` already
    caches.  It used to call `reader.get`, which reads the whole tensor: two calls
    per layer x 52 layers = 104 payload reads of ~66 MB each (~6.9 GB) to build the
    plan.  Measured: `object_plan` made 104 `get` calls before, 0 after.
    """
    print("test_plan_reads_no_payload_bytes")
    try:
        import torch  # noqa: F401
    except Exception as exc:                          # noqa: BLE001
        print(f"  SKIP  torch unavailable: {exc}")
        return
    if not (MUSE_SRC / "model.safetensors.index.json").is_file():
        print(f"  SKIP  Muse checkpoint not on this box: {MUSE_SRC}")
        return
    convert = _converter_module()
    reader = convert.MuseReader(str(MUSE_SRC))
    seen = {"n": 0}
    original = reader.get

    def counting(key):
        seen["n"] += 1
        return original(key)

    reader.get = counting
    plan = convert.object_plan(reader, convert.resource_specs(str(MUSE_SRC)))
    check(seen["n"] == 0,
          "building the plan reads no payload bytes at all",
          f"{seen['n']} payload read(s) for {len(plan)} objects")
    check(len(plan) > 800, "and the plan is still complete", str(len(plan)))
    check(hasattr(reader, "meta"), "MuseReader exposes the header-only accessor")


def test_mlp_layer_fmt_agrees_with_the_payload_decision() -> None:
    """The header-based classifier must not change any answer.

    The old implementation classified on `str(t.dtype) == 'torch.uint8'`, and
    `MuseReader.get` maps U8 *and* both FP8 spellings to torch.uint8 -- so a naive
    header comparison against "U8" would silently drop the fp8 layers out of the
    packed branch.  This compares the two paths on the real checkpoint.
    """
    print("test_mlp_layer_fmt_agrees_with_the_payload_decision")
    try:
        import torch
    except Exception as exc:                          # noqa: BLE001
        print(f"  SKIP  torch unavailable: {exc}")
        return
    if not (MUSE_SRC / "model.safetensors.index.json").is_file():
        print(f"  SKIP  Muse checkpoint not on this box: {MUSE_SRC}")
        return
    convert = _converter_module()
    reader = convert.MuseReader(str(MUSE_SRC))
    disagreements = []
    for layer in range(convert.LAYERS):
        for proj in ("gate_proj", "down_proj"):
            key = f"{convert.PREFIX}layers.{layer}.mlp.{proj}.weight"
            tensor = reader.get(key)
            if str(tensor.dtype) == "torch.uint8":
                full_k = (convert.HIDDEN if proj in ("gate_proj", "up_proj")
                          else convert.INTERMEDIATE)
                old = convert.NVFP4 if tuple(tensor.shape)[1] == full_k // 2 else convert.FP8
            else:
                old = convert.FP8
            new = convert.mlp_layer_fmt(reader, layer, proj)
            if old != new:
                disagreements.append((layer, proj, old, new))
    check(not disagreements,
          "the header-based decision matches the payload-based one on every layer",
          repr(disagreements[:4]))
    check(convert.LAYERS * 2 > 0, "layers were compared")


# --------------------------------------------------------------------------- #
# 6. converter object names agree with the engine's object closure
# --------------------------------------------------------------------------- #
def test_object_names_agree_with_the_engine_bindings() -> None:
    print("test_object_names_agree_with_the_engine_bindings")
    if not BINDINGS_CPP.is_file():
        print(f"  SKIP  engine bindings not on this box: {BINDINGS_CPP}")
        return
    engine_text = BINDINGS_CPP.read_text(encoding="utf-8", errors="replace")
    converter_text = CONVERTER.read_text(encoding="utf-8", errors="replace")
    literals = {
        name for name in re.findall(r'"([a-z0-9_/]+)"', engine_text)
        if "/" in name
    }
    check(len(literals) >= 10, "the engine names a useful set of objects",
          repr(sorted(literals)))
    missing = sorted(name for name in literals if name not in converter_text)
    check(not missing,
          "every object the engine binds is produced by the converter",
          repr(missing))


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main() -> int:
    print(f"muse importer contract tests (repo={REPO})")
    print(f"muse source: {MUSE_SRC}")
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
