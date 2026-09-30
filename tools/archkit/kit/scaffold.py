# -*- coding: utf-8 -*-
"""kit.scaffold — "support model X" as filling in a form, with the form machine-checked.

    python3 -m tools.archkit.kit.scaffold \
      --spec tools/archkit/specs/minicpm5_1b_spec.json \
      --id   minicpm5_1b \
      --out  out/archkit/minicpm5_1b

What comes out, and what each piece is FOR:

  ``impl/spec_contract.h``          MECHANICAL, generated, compiles green on its own.
                                    Constants from the spec + the invariants that
                                    bind them, as `static_assert`s.  This is the
                                    engine-side half of the interlock: the day a
                                    hand-written `config.h` disagrees with the spec,
                                    the build stops instead of the runtime quietly
                                    executing fewer layers than the model has.
  ``impl/spec_contract_check.cpp``  the translation unit the compile gate compiles.
  ``impl/<file>.skeleton``          every one of the four file groups, with the
                                    mechanical part filled and the rest marked
                                    `TODO(hand): <file>:<line> — <why it cannot be
                                    generated>`.  A TODO without a why is a bug.
  ``tests/test_load_plan.cpp.skeleton``  the structural gate heritage named as the
                                    pipeline's thin spot.
  ``registry_insert.tsv``           the tree-level registration points, computed from
                                    the LIVE tree (file, line, anchor, exact text).
  ``adapt_report.md``               the gap report: which family-side hooks this
                                    model must change, measured with kit/refcount.py.
  ``check.json``                    the ruler's verdict (see kit/check.py).

The prohibition is the load-bearing part.  `adapt.py` generated a `config.h` for
spark whose own review listed four reasons it was unusable, the sharpest being
`full_attention_layers()=9 + gdn_layers()=0` summing to 9 on a 36-layer stack, so
27 layers would never execute.  That is why `config.h` is emitted as a SKELETON with
its hand half named, and why `spec_contract.h` asserts the sum instead of writing
the accessors: the generator is allowed to state the invariant, and is not allowed
to state the answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "tools.archkit.kit"

from .refcount import count_references, declared_in_config, probed_by_runtime

#: the four file groups a target owns, in the order heritage's table lists them
TARGET_FILE_GROUPS = (
    ("impl/config.h", "MECHANICAL-PARTIAL",
     "the TextConfig the family runtime instantiates"),
    ("impl/variant.h", "SKELETON", "the leaf's declared surface"),
    ("impl/variant.cpp", "SKELETON", "the leaf's definitions"),
    ("impl/load/bindings.h", "MECHANICAL-PARTIAL",
     "the object plan the artifact is bound against"),
    ("impl/load/bindings.cpp", "MECHANICAL-PARTIAL",
     "the binding table, one row per artifact object"),
    ("impl/package.cpp", "SKELETON",
     "identity / weights / defaults / plan_load"),
    ("export/ninfer/targets/<id>/package.h", "SKELETON", "the exported package header"),
    ("CMakeLists.txt", "MECHANICAL", "the target's own build file"),
)

#: what a generator is FORBIDDEN to write, and why.  Printed into every skeleton and
#: into the guide, so the prohibition travels with the artefact.
FORBIDDEN = (
    ("impl/config.h (the TextConfig body)",
     "adapt.py measured that a generated TextConfig is unusable: its "
     "full_attention_layers()=9 + gdn_layers()=0 sum to 9 on a 36-layer stack, so 27 "
     "layers would never execute and NOTHING WOULD FAIL. The real file is 548 hand "
     "lines + about 20 static_asserts. The generator states the invariant "
     "(spec_contract.h) and not the accessors."),
    ("per-layer tables (layer_kind / layer_rope_theta / layer_rotary_dim / kFullLayers)",
     "they are the model's semantics, not its geometry; a wrong table is silent."),
    ("the family hook accessors (rope_theta_at, rotary_dim_at, qk_norm_enabled, ...)",
     "a hook whose declaration the shared runtime never reads is dead weight; which "
     "hooks are live is a MEASUREMENT (kit/refcount.py), and making a dead hook live "
     "is a family-side change with its own review."),
    ("src/targets/registry.h, src/targets/registry.cpp, src/CMakeLists.txt",
     "tree-level registration is an overlay on files other lines own. The generator "
     "prints file:line insertion points instead of editing them."),
    ("anything under tests/ or the front door's REGISTERED_TARGETS",
     "same reason: they are shared statements, and a second writer is a conflict."),
)


def cpp_ident(name: str) -> str:
    """`minicpm5-1b` is not a C++ identifier; fold everything else to `_`.

    This is one line and it is here because adapt.py had to add it after the fact
    (`adapt.py:49-63`): the spec's model_id is an HF-style name and the tree's
    identifiers are snake_case.
    """
    folded = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if folded[:1].isdigit():
        folded = "_" + folded
    return folded


def camel_alias(name: str) -> str:
    """`spark_x2_5_4b` -> `SparkX2_5_4B`, the spelling `registry.h` already uses.

    Not `"".join(p.capitalize())`: that yields `SparkX2_5_4b`, and the tree spells
    the last token `4B`.  The rule is in registry.h's own text (`using SparkX2_5_4B
    = spark_x2_5_4b::Package;`): upper-case the first letter of each token, and when
    a token starts with a digit, upper-case the first LETTER it has.
    """
    parts: list[str] = []
    for token in name.split("_"):
        if not token:
            continue
        if token[0].isdigit():
            for index, char in enumerate(token):
                if char.isalpha():
                    token = token[:index] + char.upper() + token[index + 1:]
                    break
            parts.append(token)
        else:
            parts.append(token[0].upper() + token[1:])
    return "_".join(parts) if len(parts) == 2 else "".join(parts)


def _layer_kind_counts(spec: Mapping[str, Any]) -> dict[str, int]:
    kinds = spec.get("layer_types") or []
    counts: dict[str, int] = {}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def assert_block(condition: str, message_lines: list[str]) -> str:
    """A `static_assert` whose message is one adjacent literal per line."""
    lines = ["static_assert(%s," % condition]
    last = len(message_lines) - 1
    for index, text in enumerate(message_lines):
        # adjacent literals concatenate with nothing between them, so every line but
        # the last carries its own trailing space; without it the assert text reads
        # "does not coverevery layer", which is what the first run printed.
        body = text.replace('"', '\\"') + (" " if index != last else "")
        lines.append('    "%s"%s' % (body, ");" if index == last else ""))
    return "\n".join(lines)


def spec_contract_header(spec: Mapping[str, Any], target_id: str, marker: str) -> str:
    g = spec["geometry"]
    layers = int(g["layers"])
    counts = _layer_kind_counts(spec)
    qsize = int(g["query_heads"]) * int(g["head_dim"])
    kvsize = int(g["kv_heads"]) * int(g["head_dim"])
    full = counts.get("full_attention", 0) + counts.get("full", 0)
    sliding = counts.get("sliding_attention", 0) + counts.get("sliding", 0)
    gdn = counts.get("gdn", 0) + counts.get("linear_attention", 0)
    lines: list[str] = []
    add = lines.append
    add("// GENERATED by tools/archkit/kit/scaffold.py -- DO NOT EDIT BY HAND.")
    add("//")
    add("// Source: tools/archkit/specs/%s_spec.json" % target_id)
    add("// Marker: %s" % marker)
    add("//")
    add("// This file is the MECHANICAL half only.  It states the geometry the spec")
    add("// declares and the invariants that bind it, so that the hand-written")
    add("// `config.h` next door cannot disagree with the spec in silence.  It does")
    add("// NOT declare the family accessors, the per-layer tables, or the knobs --")
    add("// see the FORBIDDEN list in kit/scaffold.py for why, and the TODO(hand)")
    add("// block at the bottom of this file for what is left to a human.")
    add("")
    add("#ifndef NINFER_TARGETS_%s_IMPL_SPEC_CONTRACT_H" % cpp_ident(target_id).upper())
    add("#define NINFER_TARGETS_%s_IMPL_SPEC_CONTRACT_H" % cpp_ident(target_id).upper())
    add("")
    add("#include <cstdint>")
    add("")
    add("namespace ninfer::targets::%s::detail::spec_contract {" % cpp_ident(target_id))
    add("")
    add("// A string literal, not a macro: it survives preprocessing and reaches the")
    add("// object file, so the compile gate can show that the TU it compiled included")
    add("// THIS copy of the header and not another one on the include path (R52).")
    add('inline constexpr const char* kScaffoldMarker = "%s";' % marker)
    add('inline constexpr const char* kSpecModelId = "%s";' % str(spec["model_id"]))
    add('inline constexpr const char* kHfArchitecture = "%s";'
        % (",".join((spec.get("hf") or {}).get("architectures") or [])))
    add("")
    add("// ---- geometry, copied from the spec -------------------------------")
    for key in ("hidden", "layers", "query_heads", "kv_heads", "head_dim", "vocab",
                "max_ctx", "intermediate"):
        add("inline constexpr int %-12s = %d;" % (key, int(g[key])))
    if "rms_eps" in g:
        add("inline constexpr double rms_eps     = %r;" % float(g["rms_eps"]))
    else:
        add("// TODO(hand): rms_eps is NOT in the spec, so it is not generated.  See")
        add("// the TODO(hand) block below; a default here would be a silent guess.")
    # rope_theta is emitted ONLY when the spec carries exactly one value.  Spark's
    # spec carries two (full 5000000 / sliding 10000) and muse's carries none in
    # `geometry` while its real config.h declares 500000.0F -- the first version of
    # this generator emitted 0.0F for both, and the REVERSE diff caught it by name
    # (`MECHANICAL MISMATCH rope_theta: real '10000.0F', generated '0.0F'`).  A
    # generated default for a value the spec does not state is the exact failure mode
    # the prohibition list exists to prevent, so the generator now says nothing.
    if "rope_theta" in g:
        add("inline constexpr float rope_theta  = %rF;" % float(g["rope_theta"]))
    elif (spec.get("rope") or {}).get("rope_theta") is not None:
        add("inline constexpr float rope_theta  = %rF;"
            % float(spec["rope"]["rope_theta"]))
    else:
        by_kind = spec.get("rope_by_kind") or {}
        add("// TODO(hand): rope_theta is NOT generated -- the spec carries no single")
        add("// value%s.  Per-kind theta is a HAND decision: no family hook reads a"
            % (" (%s)" % json.dumps(by_kind, sort_keys=True) if by_kind else ""))
        add("// per-layer theta in the shared runtime today (see kit/refcount.py), so")
        add("// declaring a table here would change nothing and be read by no one.")
    add("")
    add("// ---- derived, and asserted so the derivation is not a hope --------")
    add("inline constexpr int query_size = query_heads * head_dim;")
    add("inline constexpr int kv_size    = kv_heads * head_dim;")
    add("")
    add("// ---- the layer schedule, as three counts and one sum --------------")
    add("inline constexpr int full_layers    = %d;" % full)
    add("inline constexpr int sliding_layers = %d;" % sliding)
    add("inline constexpr int gdn_layers     = %d;" % gdn)
    add("inline constexpr int schedule_total = full_layers + sliding_layers + gdn_layers;")
    add("")
    # Every assert message is emitted as ADJACENT string literals, one per line.
    # A single literal broken across lines is a C++ syntax error, and the first
    # version of this file shipped exactly that: the POS arm of the compile gate went
    # red on its own generator's output, which is the gate doing its job.
    add(assert_block("schedule_total == layers", [
        "THE ADAPT.PY DEFECT, AS A COMPILE ERROR: the layer schedule does not cover",
        "every layer.  A generated config.h for spark reported 9 full + 0 gdn on a",
        "36-layer stack; nine plus zero is nine, so 27 layers would never execute",
        "and nothing would have failed.  Fix the spec, not this assert.",
    ]))
    add(assert_block("head_dim == 128 || head_dim == 256", [
        "the shared GQA kernel domain is head_dim in {128, 256} (src/ops/kernel/",
        "gqa_attention_geometry.cuh:16 is the static_assert this mirrors); a model",
        "outside it is a different kernel, not a table row.",
    ]))
    add(assert_block("query_size % 128 == 0 && kv_size % 128 == 0", [
        "query/kv width must be a whole number of 128-row blocks.",
    ]))
    add(assert_block("vocab > 0 && intermediate > 0 && hidden > 0", [
        "geometry must be positive; a zero here means the spec extraction found no",
        "keys and copied defaults.",
    ]))
    add("// ---- TODO(hand): what this file must NOT decide --------------------")
    add("//")
    add("// 1. `TextConfig`: the family's struct, its slots and its accessors.  548 hand")
    add("//    lines for spark, with about 20 static_asserts of the OBSTACLE (see")
    add("//    src/targets/gemma4_31b/impl/config.h:197-268 for the shape of that).")
    add("// 2. the per-layer tables: layer_kind, layer_rope_theta, layer_rotary_dim,")
    add("//    kFullLayers.  Their contents are the model's semantics.")
    add("// 3. every family hook that kit/refcount.py reports as ZERO-REFERENCE: the")
    add("//    runtime does not read it today, so declaring it changes nothing and")
    add("//    relying on it changes nothing either.")
    add("// 4. output_rows / token_domain, which are the CHECKPOINT's tokenizer domain")
    add("//    and the artifact's padded row count -- a property of the head layout, not")
    add("//    of the spec's geometry.")
    add("")
    add("}  // namespace ninfer::targets::%s::detail::spec_contract"
        % cpp_ident(target_id))
    add("")
    add("#endif")
    add("")
    return "\n".join(lines)


def spec_contract_tu(target_id: str, marker: str) -> str:
    ident = cpp_ident(target_id)
    return "\n".join([
        "// GENERATED by tools/archkit/kit/scaffold.py -- the translation unit the",
        "// compile gate compiles (kit/compilegate.sh).  It exists so that the",
        "// mechanical contract is COMPILED and not merely written: a static_assert",
        "// that nobody instantiates is a comment.",
        "//",
        "// Marker: %s" % marker,
        "",
        "#include \"impl/spec_contract.h\"",
        "",
        "extern \"C\" const char* ninfer_%s_scaffold_marker() {" % ident,
        "    return ninfer::targets::%s::detail::spec_contract::kScaffoldMarker;"
        % ident,
        "}",
        "",
        "extern \"C\" int ninfer_%s_scaffold_schedule_total() {" % ident,
        "    return ninfer::targets::%s::detail::spec_contract::schedule_total;"
        % ident,
        "}",
        "",
    ])


def insertion_points(repo_root: Path, target_id: str) -> list[dict[str, str]]:
    """Where the tree-level registration rows go, computed from the LIVE tree.

    Nothing is edited.  Every entry is (file, line of the anchor, the anchor's own
    text, the text to insert) so a second party can apply it and diff, and so a line
    number that has moved since this was generated is visible rather than trusted.
    """
    ident = cpp_ident(target_id)
    camel = camel_alias(ident)
    out: list[dict[str, str]] = []

    def first_match(path: Path, pattern: str) -> tuple[int, str]:
        if not path.is_file():
            return 0, ""
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        regex = re.compile(pattern)
        for number, line in enumerate(text, start=1):
            if regex.search(line):
                return number, line
        return 0, ""

    cmake = repo_root / "src" / "CMakeLists.txt"
    line, anchor = first_match(cmake, r"add_subdirectory\(targets/[a-z0-9_]+\)")
    out.append({
        "file": "src/CMakeLists.txt",
        "anchor_line": str(line),
        "anchor": anchor.strip(),
        "insert": "add_subdirectory(targets/%s)" % ident,
        "kind": "NEW-ROW",
    })
    registry_h = repo_root / "src" / "targets" / "registry.h"
    for label, pattern, text in (
        ("include", r'#include\s*[<"]ninfer/targets/[a-z0-9_]+/', None),
        ("using", r"^\s*using\s+[A-Za-z0-9_]+\s*=\s*[a-z0-9_]+::Package;", None),
    ):
        line, anchor = first_match(registry_h, pattern)
        if text is None:
            if label == "include":
                text = '#include "ninfer/targets/%s/package.h"' % ident
            else:
                text = "using %s = %s::Package;" % (camel, ident)
        out.append({
            "file": "src/targets/registry.h",
            "anchor_line": str(line),
            "anchor": anchor.strip(),
            "insert": text,
            "kind": "NEW-ROW (%s block)" % label,
        })
    registry_cpp = repo_root / "src" / "targets" / "registry.cpp"
    line, anchor = first_match(registry_cpp, r"kTargetRegistrations")
    out.append({
        "file": "src/targets/registry.cpp",
        "anchor_line": str(line),
        "anchor": anchor.strip(),
        "insert": '{"%s", ...},   // plus the Loaded<>/<T>Instance pair in registry.h'
                  % ident,
        "kind": "NEW-ROW",
    })
    return out


def skeleton(path: str, target_id: str, spec: Mapping[str, Any], body: list[str]) -> str:
    ident = cpp_ident(target_id)
    return "\n".join([
        "// SKELETON -- generated by tools/archkit/kit/scaffold.py.",
        "// Target: %s   (spec hf.architectures=%s)"
        % (path, (spec.get("hf") or {}).get("architectures")),
        "//",
        "// The mechanical part of this file is filled below.  Everything else is a",
        "// `TODO(hand)` WITH A REASON, because a TODO without a why is a bug report",
        "// against whoever finds it next.",
        "",
    ] + body + [
        "",
        "// ---- TODO(hand) ----",
        "// %s is NOT generated: %s" % (path, dict((p, r) for p, r in FORBIDDEN
                                                  ).get(path, "see kit/scaffold.py "
                                                              "FORBIDDEN.")),
        "// The compile gate for this file is `tools/archkit/kit/compilegate.sh` over",
        "// impl/spec_contract_check.cpp; this skeleton has no gate of its own until a",
        "// human turns it into the real file.",
        "",
    ])


def build_report(spec: Mapping[str, Any], target_id: str, repo_root: Path,
                 runtime_rel: str) -> tuple[str, dict]:
    g = spec["geometry"]
    counts = _layer_kind_counts(spec)
    hooks: dict[str, int] = {}
    zero: list[str] = []
    runtime = repo_root / runtime_rel
    for candidate in sorted((repo_root / "src" / "targets").glob("*/impl/config.h")):
        declared = declared_in_config(candidate)
        if not declared:
            continue
        names = sorted(set(declared) | probed_by_runtime(runtime))
        counts_refs = count_references(runtime, names)
        for name in names:
            hooks.setdefault(name, counts_refs[name]["total"])
    zero = sorted(n for n, c in hooks.items() if c == 0)
    lines = [
        "# adapt_report — %s" % target_id,
        "",
        "Generated by `tools/archkit/kit/scaffold.py` from "
        "`tools/archkit/specs/%s_spec.json`." % target_id,
        "",
        "## What the spec settles (mechanical: filled, and compiled)",
        "",
        "| quantity | value | source |",
        "|---|---|---|",
        "| hidden | %d | spec geometry |" % int(g["hidden"]),
        "| layers | %d | spec geometry |" % int(g["layers"]),
        "| query/kv heads | %d / %d | spec geometry |"
        % (int(g["query_heads"]), int(g["kv_heads"])),
        "| head_dim | %d | spec geometry |" % int(g["head_dim"]),
        "| intermediate | %d | spec geometry |" % int(g["intermediate"]),
        "| vocab | %d | spec geometry |" % int(g["vocab"]),
        "| layer schedule | %s | spec layer_types |"
        % ", ".join("%s=%d" % kv for kv in sorted(counts.items())),
        "| schedule sum vs layers | %d vs %d | asserted in impl/spec_contract.h |"
        % (sum(counts.values()), int(g["layers"])),
        "",
        "## What the family side still has to change, measured",
        "",
        "`kit/refcount.py` counts how many times each accessor a target DECLARES, or "
        "the family runtime PROBES, is referenced in `%s`." % runtime_rel,
        "",
        "**Zero-reference accessors today: %d of %d measured.**" % (len(zero), len(hooks)),
        "",
        "```",
    ]
    for name in zero:
        lines.append("%s" % name)
    lines += [
        "```",
        "",
        "A new model does NOT get these for free: the declaration compiles and is "
        "never read. If this model needs one of them to take effect, that is a "
        "family-side land with its own review, and it belongs in the target's "
        "`WORK_ITEMS` by name rather than in a hope.",
        "",
        "## What is forbidden to generate, and why",
        "",
    ]
    for path, reason in FORBIDDEN:
        lines.append("* **%s** — %s" % (path, reason))
    lines.append("")
    return "\n".join(lines), {"zero_reference": zero, "accessors": hooks}


def generate(spec_path: Path, target_id: str, out_dir: Path, repo_root: Path,
             runtime_rel: str) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    ident = cpp_ident(target_id)
    spec_digest = hashlib.sha256(spec_path.read_bytes()).hexdigest()[:16]
    marker = "ARCHKIT_SCAFFOLD_%s_POST_%s" % (ident.upper(), spec_digest)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "impl" / "load").mkdir(parents=True, exist_ok=True)
    (out_dir / "tests").mkdir(parents=True, exist_ok=True)

    written: list[tuple[str, int]] = []

    def emit(relative: str, text: str) -> None:
        path = out_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append((relative, len(text.encode("utf-8"))))

    emit("impl/spec_contract.h", spec_contract_header(spec, target_id, marker))
    emit("impl/spec_contract_check.cpp", spec_contract_tu(target_id, marker))
    for relative, kind, what in TARGET_FILE_GROUPS:
        real = relative.replace("<id>", ident)
        emit(real + ".skeleton",
             skeleton(real, target_id, spec,
                      ["// kind: %s — %s" % (kind, what),
                       "// mechanical part: see impl/spec_contract.h for the numbers.",
                       "namespace ninfer::targets::%s {" % ident,
                       "// TODO(hand): the %s for %s." % (what, ident),
                       "}  // namespace ninfer::targets::%s" % ident]))
    emit("tests/test_load_plan.cpp.skeleton",
         skeleton("tests/test_load_plan.cpp", target_id, spec,
                  ["// The structural gate heritage named as this pipeline's thin spot:",
                   "// tests/targets/ holds four directories today and spark is not one",
                   "// of them, so a target can be registered and never have its plan",
                   "// read back by anything.",
                   "// TODO(hand): build the artifact plan, call plan_load, and assert",
                   "// that every object name in impl/load/bindings.cpp is present."]))
    points = insertion_points(repo_root, target_id)
    emit("registry_insert.tsv",
         "file\tanchor_line\tanchor\tinsert\tkind\n" + "\n".join(
             "\t".join([p["file"], p["anchor_line"], p["anchor"], p["insert"], p["kind"]])
             for p in points) + "\n")
    report, measured = build_report(spec, target_id, repo_root, runtime_rel)
    emit("adapt_report.md", report)
    verdict = {
        "target_id": ident,
        "spec": str(spec_path),
        "spec_sha256_16": spec_digest,
        "marker": marker,
        "generated": [{"path": p, "bytes": b} for p, b in written],
        "insertion_points": points,
        "zero_reference_accessors": measured["zero_reference"],
        "accessors_measured": len(measured["accessors"]),
    }
    emit("check.json", json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    return verdict


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[3])
    parser.add_argument("--runtime",
                        default="src/targets/qwen3_6/impl/runtime")
    args = parser.parse_args(argv)
    if not args.spec.is_file():
        print("REFUSED: missing spec file %s" % args.spec, file=sys.stderr)
        return 2
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if not spec.get("geometry"):
        print("REFUSED: spec %s has an empty 'geometry' object; an extraction that "
              "found no geometry keys must not pass as a spec" % args.spec,
              file=sys.stderr)
        return 2
    verdict = generate(args.spec, args.id, args.out, args.repo_root, args.runtime)
    print("== scaffold for %s -> %s" % (args.id, args.out))
    for entry in verdict["generated"]:
        print("   %-44s %6d B" % (entry["path"], entry["bytes"]))
    print("== insertion points (computed from the live tree; nothing edited)")
    for point in verdict["insertion_points"]:
        print("   %s:%s  [%s]" % (point["file"], point["anchor_line"], point["kind"]))
        print("       after: %s" % (point["anchor"][:90] or "(anchor not found)"))
        print("       insert: %s" % point["insert"])
    print("== zero-reference accessors measured: %d/%d"
          % (len(verdict["zero_reference_accessors"]), verdict["accessors_measured"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
