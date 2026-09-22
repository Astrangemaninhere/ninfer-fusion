#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-check the C++ binder against the Python inventory, independently.

Two statements of one contract (which objects the artifact must contain, with which
shape and numeric format) live in two languages: `src/targets/qwen3_5_9b/impl/load/
bindings.cpp` binds them, `tools/convert/qwen3_5_9b/inventory.py` writes them.  The
engine's binder would notice a disagreement only at load time, on a multi-gigabyte
artifact; this compares the two texts directly.

Independence, and where it is not perfect
----------------------------------------
The names, formats, shapes and the full/GDN branch structure are read out of the C++
text, never from the inventory.  The one rule shared with the inventory is the layer
schedule (`(l+1) % full_attention_interval == 0`), which belongs to the family's own
topology header and is read here from that header rather than restated.
`--self-test` perturbs the C++ text in memory and requires the comparison to go red, so
a parser that silently matched nothing cannot report success.

Two arms, because the draft block is optional
---------------------------------------------
`bindings.cpp` binds the 12 `mtp/*` objects only when the artifact declares them, and an
artifact converted from a source with no `nextn` block declares none.  So this compares TWO
plans: the inventory WITH the draft block against the binder as written (the comparison that
has always existed), and the inventory WITHOUT it against the binder minus its `mtp/*` region.
For the second arm to mean anything, the first thing it checks is that the binder's `mtp/*`
region is exactly the inventory's twelve names -- a region the parser stopped recognising fails
loudly instead of agreeing with an empty set -- and it also requires `bind_mtp` to read the
artifact's own object table (`find_tensor`), because that read is the whole of the
optionalization: names, formats and shapes alone would match a binder that still demands them.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("/home/user/ninfer-fusion")
PKG = ROOT / "src/targets/qwen3_5_9b"
BINDINGS = PKG / "impl/load/bindings.cpp"
MTP_PREFIX = "mtp/"
CONFIG = PKG / "impl/config.h"
TOPOLOGY = ROOT / "src/targets/qwen3_6/export/ninfer/targets/qwen3_6/hybrid_topology.h"
FRONTEND = ROOT / "src/targets/qwen3_6/export/ninfer/targets/qwen3_6/frontend.h"

sys.path.insert(0, str(ROOT))

CALL = re.compile(
    r"\b(?:artifact::)?(bind_weight|bind_device_tensor|bind_mtp)\("
    r"\s*(?:binder,\s*)?"
    r"(?P<name>\"[^\"]*\"|[A-Za-z_]\w*\s*\+\s*\"[^\"]*\")"
    r"\s*,\s*(?P<fmt>[A-Za-z_:0-9]+)\s*,\s*\{(?P<shape>[^{}]*)\}\s*\)",
    re.S)
ALIAS = re.compile(r"constexpr NumericFormat (\w+)\s*=\s*NumericFormat::(\w+);")
CONST = re.compile(r"static constexpr int (\w+)\s*=\s*([^;]+);")


def brace_body(text: str, header: str) -> str:
    start = text.index(header)
    open_brace = text.index("{", start)
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:i]
    raise SystemExit("unbalanced braces after %r" % header)


def resolve_constants() -> dict[str, int]:
    """Every `static constexpr int` in config.h, resolved in file order.

    Cross-struct references (`DFlashConfig::feature_rows = feature_layers *
    TextConfig::hidden`) are followed by dropping the struct qualifier once each side
    is resolved.  A name that is defined in two structs with two *different* values is
    refused rather than silently taking one of them.
    """
    pairs = [(m.group(1), m.group(2).strip()) for m in CONST.finditer(CONFIG.read_text("utf-8"))]
    interval = int(re.search(r"kHybridAttentionInterval\s*=\s*(\d+)",
                             TOPOLOGY.read_text("utf-8")).group(1))
    domain = int(re.search(r"kTokenDomain\s*=\s*(\d+)",
                           FRONTEND.read_text("utf-8")).group(1))
    values = {"qwen3_6::kHybridAttentionInterval": interval, "qwen3_6::kTokenDomain": domain}
    raw = pairs + [("full_attention_interval", "qwen3_6::kHybridAttentionInterval"),
                   ("token_domain", "static_cast<int>(qwen3_6::kTokenDomain)")]
    shadowed: set[str] = set()
    for _ in range(40):
        progressed = False
        for name, expr in raw:
            work = re.sub(r"static_cast<[^>]+>\(", "(", expr)
            for qualifier in ("TextConfig::", "DFlashConfig::", "DFlash2Config::",
                              "VisionConfig::"):
                work = work.replace(qualifier, "")
            for known, value in sorted(values.items(), key=lambda kv: -len(kv[0])):
                work = work.replace(known, str(value))
            if re.search(r"[A-Za-z_]", work):
                continue
            try:
                resolved = int(eval(work, {"__builtins__": {}}, {}))  # noqa: S307
            except Exception:
                continue
            if name in values:
                # A name reused across the config structs (TextConfig::layers = 32 and
                # DFlashConfig::layers = 1) is normal.  TextConfig is parsed first and
                # is the only struct the shapes read, so its value stands; the reuse is
                # reported instead of silently ignored.
                if values[name] != resolved:
                    shadowed.add("%s=%d (kept %d)" % (name, resolved, values[name]))
                continue
            values[name] = resolved
            progressed = True
        if not progressed:
            break
    missing = sorted({n for n, _e in raw if n not in values})
    if missing:
        raise SystemExit("unresolved config constants: %s" % ", ".join(missing))
    if shadowed:
        print("note: names reused across config structs (later value not used): %s"
              % ", ".join(sorted(shadowed)))
    return values


def eval_int(expr: str, cfg: dict[str, int]) -> int:
    work = re.sub(r"static_cast<[^>]+>\(", "(", expr.strip())
    work = work.replace("TextConfig::", "")
    for name, value in sorted(cfg.items(), key=lambda kv: -len(kv[0])):
        work = work.replace(name, str(value))
    if re.search(r"[A-Za-z_]", work):
        raise SystemExit("unresolved expression %r" % expr)
    return int(eval(work, {"__builtins__": {}}, {}))  # noqa: S307


def objects_in(text: str, cfg: dict[str, int], layers: list[int],
               aliases: dict[str, str]) -> dict:
    """(name -> (format, shape)) for every binder call in `text`, for these layers.

    `aliases` maps the file-scope `constexpr NumericFormat kX = NumericFormat::Y;`
    spellings, which live outside the parsed regions and must be supplied by the caller.
    """
    found: dict[str, tuple] = {}
    calls = 0
    for m in CALL.finditer(text):
        calls += 1
        fmt = aliases.get(m.group("fmt"), m.group("fmt").replace("NumericFormat::", ""))
        if fmt.startswith("k") and fmt not in aliases.values():
            raise SystemExit("unresolved format symbol %r" % fmt)
        shape = tuple(eval_int(p, cfg) for p in m.group("shape").split(",") if p.strip())
        name_expr = " ".join(m.group("name").split())
        if name_expr.startswith('"'):
            names = [name_expr.strip('"')]
        else:
            suffix = re.match(r'[A-Za-z_]\w*\s*\+\s*"(.*)"', name_expr).group(1)
            names = ["text/layers/%d/%s" % (l, suffix) for l in layers]
        for name in names:
            if name in found:
                raise SystemExit("duplicate bound object %r" % name)
            found[name] = (fmt, shape)
    if calls == 0 and text.strip():
        raise SystemExit("no binder call parsed out of a non-empty region: the parser is blind")
    return found


def branch_span(text: str, start: int) -> tuple[int, int]:
    """End index of the brace block that begins at/after `start`."""
    open_brace = text.index("{", start)
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise SystemExit("unbalanced braces")


def cpp_plan(text: str | None = None) -> tuple[dict[str, int], dict]:
    cfg = resolve_constants()
    if text is None:
        text = BINDINGS.read_text("utf-8")
    aliases = dict(ALIAS.findall(text))
    interval = cfg["qwen3_6::kHybridAttentionInterval"]
    layers = list(range(cfg["layers"]))
    full = [l for l in layers if (l + 1) % interval == 0]
    gdn = [l for l in layers if l not in full]

    loop = brace_body(text, "void bind_text_layers(artifact::Binder& binder, BindingPlan& out)")
    branch_at = loop.index("if (target.is_full_attention)")
    full_block = brace_body(loop[branch_at:], "if (target.is_full_attention)")
    else_at = loop.index("else {", branch_at)
    gdn_block = brace_body(loop[else_at:], "else")
    # The four objects bound after the if/else apply to EVERY layer; the block before it
    # (input_norm) does too.  Splicing the branch out is what separates "common" from
    # "full-only" / "gdn-only".
    common = loop[:branch_at] + loop[branch_span(loop, else_at):]

    out = {}
    out.update(objects_in(common, cfg, layers, aliases))
    out.update(objects_in(full_block, cfg, full, aliases))
    out.update(objects_in(gdn_block, cfg, gdn, aliases))
    tail = text[text.index("ArtifactLoadPlan bind_artifact"):]
    out.update(objects_in(tail, cfg, [], aliases))
    return cfg, out


def split_mtp(cpp: dict) -> tuple[dict, dict]:
    """(the binder's `mtp/*` region, every other object it binds).

    A partition, not a filter: `arms()` requires the mtp half to be exactly the inventory's
    twelve names, so a `bind_mtp` spelling this parser stopped recognising shows up as a
    disagreement rather than as an empty region that trivially agrees.
    """
    mtp = {k: v for k, v in cpp.items() if k.startswith(MTP_PREFIX)}
    rest = {k: v for k, v in cpp.items() if not k.startswith(MTP_PREFIX)}
    return mtp, rest


def mtp_declaration_read(text: str) -> bool:
    """True when the binder's `bind_mtp` lambda tests the artifact's own object table.

    That test is the whole of "MTP is optional": anything else (an unconditional
    `bind_tensor`, a placement change) still demands the 12 objects of every artifact,
    including one converted from a source that has no draft block.  It is asserted from the
    text because no name, format or shape changes when it is reverted.
    """
    try:
        body = brace_body(text, "const auto bind_mtp = [&](std::string_view name, NumericFormat")
    except (ValueError, SystemExit):
        return False
    return "find_tensor" in body


def arms(cpp: dict, mtp_names, specs_mtp, specs_no_mtp,
         text: str) -> tuple[list[str], list[str]]:
    """The two verdicts: (binder vs inventory-with-MTP, binder-minus-MTP vs inventory-without).

    `text` is passed so the declaration read can be asserted on the same text this dict was
    parsed from, mutations included.
    """
    mtp, rest = split_mtp(cpp)
    problems_mtp: list[str] = []
    if not mtp_declaration_read(text):
        problems_mtp.append(
            "the mtp region is not declaration-driven: bind_mtp does not read the artifact's "
            "own object table (Binder::find_tensor), so an artifact written from a source with "
            "no draft block is refused again")
    if set(mtp) != set(mtp_names):
        problems_mtp.append(
            "the binder's mtp region is not the inventory's %d objects: missing %s / extra %s"
            % (len(mtp_names), sorted(set(mtp_names) - set(mtp)), sorted(set(mtp) - set(mtp_names))))
    problems_mtp += ["mtp-arm: " + p for p in compare(cpp, specs_mtp)]
    problems_no = ["no-mtp-arm: " + p for p in compare(rest, specs_no_mtp)]
    return problems_mtp, problems_no


def compare(cpp: dict, specs) -> list[str]:
    py = {s.name: (s.format, tuple(s.shape)) for s in specs}
    problems = []
    for name in sorted(set(cpp) | set(py)):
        if name not in cpp:
            problems.append("inventory-only (not bound by the C++): %s" % name)
        elif name not in py:
            problems.append("cpp-only (absent from the inventory): %s" % name)
        elif cpp[name] != py[name]:
            problems.append("MISMATCH %s: cpp=%s inventory=%s" % (name, cpp[name], py[name]))
    return problems


def self_test(cpp: dict, cpp_rest: dict, saved: str, mtp_names, specs_mtp,
              specs_no_mtp) -> int:
    """Controls.  Each perturbation must turn the arm it belongs to red, and no perturbation
    is written to disk: the mutation is in memory, so a crash here cannot leave a mutated
    binder behind (the previous version wrote the file and restored it in a finally block).
    """
    failures = 0
    checks = 0

    cases = (
        # (label, old, new, must the mtp arm go RED?, must the no-mtp arm go RED?)
        ("a common-layer shape (both arms)",
         "{TextConfig::query_size + TextConfig::kv_size,\n"
         "                                          TextConfig::hidden}",
         "{TextConfig::query_size + TextConfig::kv_size + 1,\n"
         "                                          TextConfig::hidden}",
         True, True),
        ("the mtp input_projection shape (mtp arm only)",
         'bind_mtp("mtp/input_projection", kProjection4,\n'
         "                                        {TextConfig::hidden, TextConfig::mtp_input_rows})",
         'bind_mtp("mtp/input_projection", kProjection4,\n'
         "                                        {TextConfig::hidden + 1, TextConfig::mtp_input_rows})",
         True, False),
        ("the declaration read (mtp lambda stops reading the artifact)",
         "if (binder.find_tensor(name) == nullptr) {\n"
         "            if (mtp_absent == 0) { mtp_first_absent = name; }\n"
         "            ++mtp_absent;\n"
         "            return artifact::ObjectHandle{};   // unused: `declared` stays false\n"
         "        }",
         "if (name.empty()) {\n"
         "            return artifact::ObjectHandle{};   // unused: `declared` stays false\n"
         "        }",
         True, False),
    )
    for label, old, new, want_mtp, want_no in cases:
        checks += 1
        if old not in saved:
            print("   CONTROL BROKEN: perturbation anchor absent (%s)" % label)
            failures += 1
            continue
        mutated = saved.replace(old, new, 1)
        _cfg, parsed = cpp_plan(mutated)
        red_mtp, red_no = arms(parsed, mtp_names, specs_mtp, specs_no_mtp, mutated)
        ok = (bool(red_mtp) == want_mtp) and (bool(red_no) == want_no)
        print("   perturb %-52s mtp arm %-5s (want %-5s) no-mtp arm %-5s (want %-5s) : %s"
              % (label, "RED" if red_mtp else "green", "RED" if want_mtp else "green",
                 "RED" if red_no else "green", "RED" if want_no else "green",
                 "OK" if ok else "WRONG"))
        failures += 0 if ok else 1

    # Control 4, the mode inversion: the no-mtp arm's comparison must be able to say no.
    # Judged against the WITH-MTP inventory it must disagree, or "no-mtp agrees" is a
    # statement about a comparison that cannot fail.
    checks += 1
    inverted = compare(cpp_rest, specs_mtp)
    print("   no-mtp arm judged against the with-MTP inventory -> %d disagreement(s) "
          "(want > 0) : %s" % (len(inverted), "OK" if inverted else "WRONG"))
    failures += 0 if inverted else 1

    checks += 1
    same = cpp_rest == cpp
    print("   the two arms parsed different object sets -> %s : %s"
          % (not same, "OK" if not same else "WRONG"))
    failures += 0 if not same else 1

    checks += 1
    untouched = BINDINGS.read_text("utf-8") == saved
    print("   the binder text on disk is unchanged by the controls -> %s : %s"
          % (untouched, "OK" if untouched else "WRONG"))
    failures += 0 if untouched else 1

    print("   controls passing: %d/%d" % (checks - failures, checks))
    return failures


def main(argv) -> int:
    from tools.convert.qwen3_5_9b import inventory as inv

    specs_mtp = inv.plan_tensors()
    specs_no_mtp = inv.plan_tensors(with_mtp=False)
    saved = BINDINGS.read_text("utf-8")
    _cfg, cpp = cpp_plan(saved)
    mtp, rest = split_mtp(cpp)
    problems_mtp, problems_no = arms(cpp, inv.MTP_OBJECT_NAMES, specs_mtp, specs_no_mtp, saved)
    problems = problems_mtp + problems_no

    print("C++ objects parsed : %d (%d mtp/* + %d other)" % (len(cpp), len(mtp), len(rest)))
    print("inventory objects  : %d with the draft block, %d without"
          % (len(specs_mtp), len(specs_no_mtp)))
    print("disagreements      : %d (%d mtp arm, %d no-mtp arm)"
          % (len(problems), len(problems_mtp), len(problems_no)))
    for p in problems[:40]:
        print("   " + p)
    rc = 1 if problems else 0
    if "--self-test" in argv:
        print("--- controls ---")
        rc |= 1 if self_test(cpp, rest, saved, inv.MTP_OBJECT_NAMES, specs_mtp, specs_no_mtp) else 0
    print("VERDICT:", "AGREE" if rc == 0 else "DISAGREE / CONTROL FAILED")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
