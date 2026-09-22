# -*- coding: utf-8 -*-
"""archkit_target.py - the new-target assembly contract: check + scaffold.

Why this file exists
--------------------
"Add a new architecture" is today a *tribal* checklist: the required file set
lives in prose (_AUTOADAPT.md 2b, _ARCHKIT.md 3) and the per-emitter field
contract lives inside the emitters.  A spec can therefore look "adapted"
(adapt_all.py prints a green 5/5) while nothing in the tree can register or
load the model - and the one file that *is* auto-generated (impl/config.h) can
silently carry a wrong value, because the emitter's assumptions are not
reported anywhere a user can see them.

This tool makes that contract executable:

  check    <spec.json> [--tree ROOT]
      [A] spec contract        - every field an emitter consumes, and what
                                 happens when it is absent.
      [B] emitter verdict      - calls the real emitter and reports its verdict
                                 verbatim (no re-implementation of its rules).
      [C] engine prerequisites - attention head geometry registration (delegated
                                 to tools/archkit/check_geometry.py).
      [D] assembly delta       - the 8 target files + 3 registration edits,
                                 per-file: present / missing / who can produce it.
  scaffold <spec.json> --out DIR
      Writes the one file of the contract that is 100% derivable and carries no
      model constants: the per-target CMakeLists.txt (source list + include dirs,
      shape taken from src/targets/muse_glimmer_30b/CMakeLists.txt:1-9).

Everything else in the contract is tagged HUMAN or SCAFFOLD on purpose: those
files couple to the artifact container API and to per-variant weight layouts,
which is exactly the boundary _AUTOADAPT.md:100-103 draws ("bindings/package
deeply couples to the artifact container API + per-variant layout").

Inventories this tool encodes (authoritative tree, 2026-09-13):
  * registered + loadable reference: src/targets/muse_glimmer_30b - 9 files /
    2284 lines (CMakeLists.txt, export/ninfer/targets/muse_glimmer_30b/package.h,
    impl/{config.h,variant.h,variant.cpp,package.cpp},
    impl/load/{bindings.h,bindings.cpp})
  * registration sites: src/targets/registry.cpp:3 (include),
    registry.cpp:247-256 (one row), registry.cpp:354-406 (Loaded/Instance ctors),
    src/targets/registry.h:5-7 (include), registry.h:18-20 (family alias),
    registry.h:86-116 (Loaded/Instance pair), registry.h:118-120 (ActiveTarget
    variant), src/CMakeLists.txt:367-374 (add_subdirectory).

Usage:
  python3 archkit_target.py check specs/foo_spec.json [--tree /path/to/repo]
  python3 archkit_target.py scaffold specs/foo_spec.json --out /tmp/foo_scaffold
Exit code: 0 = nothing blocking, 1 = at least one blocking gap.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arch_spec  # noqa: E402

DEFAULT_TREE = HERE.parent.parent


# ---------------------------------------------------------------------------
# A. spec contract: field -> consumers, and the failure mode when absent.
#    Consumers are cited as file:line in the authoritative tree; the point of
#    the table is that a spec field with no consumer is a *drop*, not a feature.
# ---------------------------------------------------------------------------
# (field, required_by_emitter, consumer citations)
SPEC_FIELDS = (
    ("model_id", True, ["arch_spec.py:39 (REQUIRED)", "registry identity: registry.cpp:243"]),
    ("geometry.hidden", True, ["gen_full_target.py:52", "adapt.py:128"]),
    ("geometry.layers", True, ["gen_full_target.py:37 (== len(layer_types))"]),
    ("geometry.query_heads", True, ["gen_full_target.py:69", "check_geometry.py:81"]),
    ("geometry.kv_heads", True, ["gen_full_target.py:70", "check_geometry.py:82"]),
    ("geometry.head_dim", True, ["gen_full_target.py:71", "check_geometry.py:83"]),
    ("geometry.intermediate", True, ["gen_full_target.py:60-66 (hard raise when absent)",
                                     "gen_target.py:66"]),
    ("geometry.vocab", True, ["gen_full_target.py:67 -> output_rows/token_domain"]),
    ("geometry.max_ctx", True, ["gen_full_target.py:68 -> kNativeContext"]),
    ("geometry.rms_eps", False, ["gen_full_target.py:72 (default 1e-6)"]),
    ("layer_types|layer_kind_order", True,
     ["gen_full_target.py:35-37 (per-layer kind table)", "adapt.py:53"]),
    ("knobs.rope_theta|rope.rope_theta", False, ["gen_full_target.py:74-75 (default 1e7)"]),
    ("knobs.sliding_window", False, ["gen_full_target.py:77", "adapt.py:72"]),
    ("knobs.qk_scale_factor", False, ["gen_full_target.py:78"]),
    ("knobs.output_multiplier", False, ["gen_full_target.py:79"]),
    ("knobs.final_logit_softcapping", False, ["gen_full_target.py:80"]),
    ("knobs.tie_word_embeddings", False, ["gen_full_target.py:81"]),
    ("knobs.layer_rope_theta", False,
     ["gen_full_target.py:42-44 (len == layers) and :46-50 (full layer <=> theta 0)"]),
)

# Fields that exist in shipped specs but have no consumer in this tree.  Each
# one is a silent drop: the spec author believed it was honoured.
DROPPED_FIELDS = (
    ("attention.partial_rotary_factor",
     "specs/qwen4_exp_spec.json:72, specs/gemma4-31b_spec.json:86; no emitter reads it, so "
     "rotary_dim stays == head_dim (gen_full_target.py:76). qwen4-exp declares 0.25 of 256."),
    ("attention.rope_parameters.<kind>.partial_rotary_factor",
     "specs/gemma4-31b_spec.json:84-93 (full 0.25 / sliding 1.0); per-kind by design, and "
     "TextConfig has a single rotary_dim slot (src/targets/muse_glimmer_30b/impl/config.h:34)."),
    ("attention.global_head_dim / num_global_key_value_heads",
     "heterogeneous attention geometry; the engine GQA registry is exact-shape "
     "(src/ops/kernel/gqa_attention_geometry.cuh:25-32), so this is a new_op."),
    ("weights", "declared in _ARCHKIT.md:34; only gen_stubs_v2 emits a TODO skeleton."),
)

# ---------------------------------------------------------------------------
# B. the emitter under test.  We call it, we do not re-implement it.
# ---------------------------------------------------------------------------
EMITTER = "gen_full_target.py"


# ---------------------------------------------------------------------------
# D. the assembly contract.
# ---------------------------------------------------------------------------
# (path template relative to src/targets/<ns>/, producer, role)
TARGET_FILES = (
    ("CMakeLists.txt", "GENERATED", "target_sources(3 TU) + 2 include dirs"),
    ("impl/config.h", "GENERATED",
     "gen_full_target.py --emit-to <dir>/impl (TextConfig + layer kind/theta tables)"),
    ("impl/variant.h", "HUMAN", "Variant full surface (~15 leaves + 4 graph profiles + aliases)"),
    ("impl/variant.cpp", "HUMAN", "leaf bodies (shared ops) + instantiate macro"),
    ("impl/load/bindings.h", "HUMAN", "artifact object plan + runtime payload structs"),
    ("impl/load/bindings.cpp", "HUMAN", "key -> plan expansion (per-variant layouts)"),
    ("impl/package.cpp", "HUMAN", "LoadPlan/LoadedModel + the 9 Package statics"),
    ("export/ninfer/targets/{ns}/package.h", "SCAFFOLD",
     "identity header: model_id/target_key/WeightsProfile + family runtime aliases"),
)

# (file, what is needed, how it is checked)
EDIT_SITES = (
    ("src/targets/registry.h", "include + family alias + Loaded/Instance pair + ActiveTarget row",
     "grep for the target namespace"),
    ("src/targets/registry.cpp", "include + one kTargetRegistrations row + ctor definitions",
     "grep for the target namespace"),
    ("src/CMakeLists.txt", "add_subdirectory(targets/<ns>)", "grep for the subdirectory"),
)

PACKAGE_STATICS = (
    "sampling_defaults", "resolve_weights", "resolved_auto_speculative", "plan_load",
    "construct_loaded_model", "make_frontend", "make_sequence_planner", "create_program",
    "export_head_weights",
)


def ns_of(model_id: str) -> str:
    return model_id.replace("-", "_")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def spec_path(path: Path) -> Path:
    """Accept a spec json or a model dir containing one convention of specs/."""
    if path.is_dir():
        cands = sorted(path.glob("*_spec.json"))
        if not cands:
            raise SystemExit("no *_spec.json in %s" % path)
        return cands[0]
    return path


# ---------------------------------------------------------------------------
# A
# ---------------------------------------------------------------------------
def section_spec(spec: dict) -> tuple[int, int]:
    geo = spec.get("geometry") or {}
    print("[A] spec contract")
    blocked = 0
    seen = 0
    for field, required, consumers in SPEC_FIELDS:
        head = field.split(".")[0]
        if field == "layer_types|layer_kind_order":
            present = bool(spec.get("layer_types") or spec.get("layer_kind_order"))
        elif field.startswith("knobs."):
            present = field.split(".", 1)[1] in (spec.get("knobs") or {})
        elif field == "knobs.rope_theta|rope.rope_theta":
            present = ("rope_theta" in (spec.get("knobs") or {})
                       or "rope_theta" in (spec.get("rope") or {}))
        elif head == "geometry":
            present = field.split(".", 1)[1] in geo
        else:
            present = field in spec
        if present:
            seen += 1
        tag = "ok " if present else ("MISS" if required else "opt ")
        if not present and required:
            blocked += 1
        print("    %-4s %-46s %s" % (tag, field, "; ".join(consumers)))
    print("    layer_types n=%s layers=%s %s"
          % (len(spec.get("layer_types") or spec.get("layer_kind_order") or []),
             geo.get("layers"),
             "" if (not geo.get("layers") or
                    len(spec.get("layer_types") or spec.get("layer_kind_order") or [])
                    == geo.get("layers")) else "<- LENGTH MISMATCH (emitter asserts)"))
    print("    declared-but-unconsumed:")
    for field, note in DROPPED_FIELDS:
        print("      ~ %-46s %s" % (field, note))
    print("    (%d/%d contract fields present)" % (seen, len(SPEC_FIELDS)))
    return blocked, seen


# ---------------------------------------------------------------------------
# B
# ---------------------------------------------------------------------------
def section_emitter(spec: dict) -> int:
    """Call the real emitter with (spec, knobs) and report its verdict verbatim."""
    print("[B] emitter verdict (%s)" % EMITTER)
    try:
        import gen_full_target as gft
    except Exception as error:                                   # pragma: no cover
        print("    cannot import %s: %s" % (EMITTER, error))
        return 1
    knobs = spec.get("knobs") or {}
    try:
        text = gft.gen_text_config(spec, knobs)
    except Exception as error:
        print("    REFUSED: %s: %s" % (type(error).__name__, error))
        print("    -> impl/config.h cannot be auto-generated for this spec as it stands.")
        return 1
    print("    OK: emits %d bytes of TextConfig" % len(text))
    print("    -> impl/config.h is auto-generatable (still needs --emit-to <target>/impl)")
    return 0


# ---------------------------------------------------------------------------
# C
# ---------------------------------------------------------------------------
def section_engine(spec_path_: Path, tree: Path) -> int:
    print("[C] engine prerequisites (attention head geometry)")
    gate = HERE / "check_geometry.py"
    if not gate.exists():
        print("    check_geometry.py missing; skipped")
        return 0
    # flush first: the delegate writes straight to the inherited stdout, and an
    # unflushed buffer would let its report land above this section header.
    sys.stdout.flush()
    rc = subprocess.call([sys.executable, str(gate), str(spec_path_), "--repo", str(tree)])
    if rc != 0:
        print("    -> BLOCKING: the (q, kv, head_dim) triple is not served by the GQA kernels.")
        print("       A new geometry needs: alias in src/ops/kernel/gqa_attention_geometry.cuh,")
        print("       guards in src/ops/launcher/gqa_attention_decode.cu, a per-geometry TU")
        print("       (see gqa_attention_decode_muse.cu / _g35.cu) and a src/CMakeLists.txt entry.")
    return rc


# ---------------------------------------------------------------------------
# D
# ---------------------------------------------------------------------------
def section_assembly(spec: dict, tree: Path) -> int:
    mid = spec.get("model_id", "?")
    ns = ns_of(mid)
    tdir = tree / "src" / "targets" / ns
    print("[D] assembly delta  (src/targets/%s/  + 3 registration sites)" % ns)
    missing = []
    for rel, producer, role in TARGET_FILES:
        p = tdir / rel.format(ns=ns)
        ok = p.exists()
        if not ok:
            missing.append((producer, rel.format(ns=ns)))
        print("    %-5s %-12s %-46s %s"
              % ("ok" if ok else "MISS", producer, rel.format(ns=ns), role))
    for f, what, _how in EDIT_SITES:
        p = tree / f
        hit = False
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="replace")
            hit = ns in txt
        print("    %-5s %-12s %-46s %s"
              % ("ok" if hit else "MISS", "EDIT", f, what if not hit else "already mentions %s" % ns))
        if not hit:
            missing.append(("EDIT", f))
    human = [m for m in missing if m[0] == "HUMAN"]
    auto = [m for m in missing if m[0] == "GENERATED"]
    print("    framework surface a target must implement (Package statics): %s"
          % ", ".join(PACKAGE_STATICS))
    print("    delta: %d missing (%d HUMAN, %d GENERATED, %d SCAFFOLD, %d EDIT)"
          % (len(missing), len(human), len(auto),
             len([m for m in missing if m[0] == "SCAFFOLD"]),
             len([m for m in missing if m[0] == "EDIT"])))
    return len(missing)


def cmd_check(args) -> int:
    sp = spec_path(Path(args.spec))
    spec = load(sp)
    tree = Path(args.tree or DEFAULT_TREE)
    print("== archkit_target check: %s (model_id=%s family=%s)"
          % (sp, spec.get("model_id"), spec.get("family")))
    print("   tree: %s" % tree)
    print()
    b_spec, _ = section_spec(spec)
    print()
    b_emit = section_emitter(spec)
    print()
    b_geo = section_engine(sp, tree)
    print()
    n_missing = section_assembly(spec, tree)
    print()
    blocking = b_spec + b_emit + b_geo
    print("[E] verdict")
    if blocking:
        print("    NOT registrable-and-loadable yet: %d spec blocker(s), "
              "emitter refused=%s, geometry gap=%s, %d assembly item(s) open."
              % (b_spec, bool(b_emit), bool(b_geo), n_missing))
    else:
        print("    spec/emitter/geometry clear; %d assembly item(s) still open." % n_missing)
    return 1 if (blocking or n_missing) else 0


# ---------------------------------------------------------------------------
# scaffold
# ---------------------------------------------------------------------------
CMAKE_TEMPLATE = """# Generated by tools/archkit/archkit_target.py scaffold (spec: {mid}).
# Shape taken from src/targets/muse_glimmer_30b/CMakeLists.txt:1-9 (the smallest
# registered + loadable target).  The three translation units are the contract's
# HUMAN files; stage them in only once they exist, otherwise the engine build
# fails at the missing source rather than at the missing symbol.
target_sources(ninfer_engine PRIVATE
  ${{CMAKE_CURRENT_SOURCE_DIR}}/impl/package.cpp
  ${{CMAKE_CURRENT_SOURCE_DIR}}/impl/load/bindings.cpp
  ${{CMAKE_CURRENT_SOURCE_DIR}}/impl/variant.cpp)

target_include_directories(ninfer_engine PRIVATE
  ${{CMAKE_CURRENT_SOURCE_DIR}}/impl
  ${{CMAKE_CURRENT_SOURCE_DIR}}/export)
"""


def cmd_scaffold(args) -> int:
    sp = spec_path(Path(args.spec))
    spec = load(sp)
    mid = spec["model_id"]
    ns = ns_of(mid)
    out = Path(args.out)
    (out / "src" / "targets" / ns).mkdir(parents=True, exist_ok=True)
    written = []
    f = out / "src" / "targets" / ns / "CMakeLists.txt"
    f.write_text(CMAKE_TEMPLATE.format(mid=mid), encoding="utf-8")
    written.append(f)
    # Everything else in the contract is emitted as a named placeholder, so the
    # target directory shape is complete and the remaining work is countable.
    for rel, producer, role in TARGET_FILES:
        if producer == "GENERATED" and rel.endswith("config.h"):
            continue  # produced by gen_full_target.py --emit-to, not by us
        if producer == "GENERATED":
            continue
        p = out / "src" / "targets" / ns / rel.format(ns=ns)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("%s: TODO(assembler) %s (%s)\n" % (producer, rel.format(ns=ns), role),
                     encoding="utf-8")
        written.append(p)
    print("== archkit_target scaffold: %s (model_id=%s, ns=%s)" % (sp, mid, ns))
    print("   NOTE: only CMakeLists.txt is generatable text; the rest are placeholders")
    print("         naming what a human/scaffold step must supply.")
    for p in written:
        print("   %s" % p)
    print("   next: python3 %s check %s --tree <repo>" % (Path(__file__).name, sp))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("spec")
    c.add_argument("--tree", default=None)
    c.set_defaults(func=cmd_check)
    s = sub.add_parser("scaffold")
    s.add_argument("spec")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_scaffold)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
