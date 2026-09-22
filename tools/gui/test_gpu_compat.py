# -*- coding: utf-8 -*-
"""test_gpu_compat 自测: "这块卡能不能跑" 不许由常量或 sm 号作答。

THE DEFECT THIS PINS (measured 2026-09-18, SIMFIX Route A):
    ninfer-gui.py built the environment-card row with a HARDCODED `'ok': True`:
        env['items'].insert(0, {'name': '显卡世代', 'ok': True, 'detail': ...})
    gui_page.html renders `ok` as the pass glyph in a list where every OTHER `ok` means
    "this self-check requirement is satisfied" (GPU present / python present / engine found):
        d.className='env-item '+(it.ok?'ok':'bad');
        d.innerHTML='<span class="st">'+(it.ok?'✔':'✖')+' '+it.name+'</span>...'
    So a green tick was rendered for a row whose `features` is gpu_compat.TIERS' UNVERIFIED
    route plan. On an sm_70 card the tick said "仅 fp16 张量核" about a card this build cannot
    run at all -- and gpu_compat's own docstring says no TIERS row "构成'这个模型能跑'的说法".
    Worse, the module's honest fields (`engine_ready=None`, `capability_source="unknown"`)
    were read by NOTHING (`grep -n compat gui_page.html` had no match), so the refusal to
    infer was decoration and the constant was the verdict.

THE CONTRACT:
  * `ok` is True only when the engine's OWN capability probe reported the answer
    (`capability_source == "engine-probe"` AND `engine_ready is True`) -- both fields, so a
    boolean set without an instrument is still not a pass;
  * `ok` is False only for a MEASURED refusal;
  * `ok` is None (NOT MEASURED) otherwise, and the renderer must show neither ✔ nor ✖.

RUN:  python3 tools/gui/test_gpu_compat.py          (assertions only, no GPU needed)
      python3 tools/gui/test_gpu_compat.py --mutate (shows the assertions FAIL on the old shape)
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_compat as gc  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = 0
FAILS = 0


def check(cond, label, detail=""):
    global CHECKS, FAILS
    CHECKS += 1
    if cond:
        print("  ok   %s" % label)
    else:
        FAILS += 1
        print("  FAIL %s%s" % (label, (" | " + detail) if detail else ""))


def read(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read()


def row(gen="RTX 50 系", feat="fp4 张量核路线", advice="见引擎探针", **kw):
    base = {"present": True, "generation": gen, "features": feat, "advice": advice,
            "engine_ready": None, "capability_source": gc.CAPABILITY_SOURCE_UNKNOWN}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------------------
# 1. The three-state decision, and it is fail-closed on TWO fields.
# ---------------------------------------------------------------------------------------
def t_decision():
    print("1. capability_state / env_item: the state is computed, and from two fields")
    check(gc.capability_state(row()) == gc.CAPABILITY_UNMEASURED,
          "an unprobed card is UNMEASURED", gc.capability_state(row()))
    check(gc.env_item(row())["ok"] is None,
          "UNMEASURED renders as ok=None (neither pass nor fail)",
          repr(gc.env_item(row())["ok"]))
    check(gc.capability_state(row(engine_ready=True)) == gc.CAPABILITY_UNMEASURED,
          "engine_ready=True with NO instrument behind it is still UNMEASURED "
          "(a boolean is not a source)", gc.capability_state(row(engine_ready=True)))
    check(gc.env_item(row(engine_ready=True))["ok"] is None,
          "…and it still does not render as a pass", repr(gc.env_item(row(engine_ready=True))["ok"]))
    check(gc.capability_state(row(engine_ready=True,
                                  capability_source=gc.CAPABILITY_SOURCE_ENGINE_PROBE))
          == gc.CAPABILITY_MEASURED, "probe-sourced + engine_ready=True is MEASURED")
    check(gc.env_item(row(engine_ready=True,
                          capability_source=gc.CAPABILITY_SOURCE_ENGINE_PROBE))["ok"] is True,
          "…and only that renders as a pass")
    check(gc.capability_state(row(engine_ready=False,
                                  capability_source=gc.CAPABILITY_SOURCE_ENGINE_PROBE))
          == gc.CAPABILITY_REFUSED, "probe-sourced + engine_ready=False is REFUSED")
    check(gc.env_item(row(engine_ready=False,
                          capability_source=gc.CAPABILITY_SOURCE_ENGINE_PROBE))["ok"] is False,
          "…and that renders as a measured fail")
    check(gc.CAPABILITY_SOURCE_ENGINE_PROBE != gc.CAPABILITY_SOURCE_UNKNOWN,
          "the two sources are distinct constants")


# ---------------------------------------------------------------------------------------
# 2. No reachable state of the REAL module can produce a pass today.
# ---------------------------------------------------------------------------------------
def mint_sites(rel):
    """Every CODE site in `rel` that can produce the literal "engine-probe".

    Prose is not a mint site: a docstring or a comment that names the value is documentation,
    while a literal used as an expression value is a way to write the provenance without the
    instrument. Parsed, not grepped, so the assertion cannot be satisfied by rewording a
    comment -- the same failure the file's own docstring complains about.
    """
    import ast
    tree = ast.parse(read(rel))
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    sites = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and node.value == "engine-probe"):
            continue
        p = parent.get(node)
        if isinstance(p, ast.Expr):
            continue                       # a bare string expression: prose/docstring
        tgt = ""
        if isinstance(p, ast.Assign):
            tgt = ",".join(getattr(t, "id", "?") for t in p.targets)
        sites.append(tgt or type(p).__name__)
    return sites


def t_real_query():
    print("2. the real query(): nothing measured it, so nothing may pass")
    q = gc.query().as_dict()
    st = gc.capability_state(q)
    check(st == gc.CAPABILITY_UNMEASURED,
          "gpu_compat.query() reports UNMEASURED on this box (present=%s sm=%s)"
          % (q.get("present"), q.get("sm")), st)
    it = gc.env_item(q)
    check(it["ok"] is not True,
          "the environment row is not a pass (ok=%r)" % (it["ok"],))
    check(it.get("capability_state") == gc.CAPABILITY_UNMEASURED,
          "the row carries its state for a renderer that wants it")
    sites = mint_sites("tools/gui/gpu_compat.py")
    check(sites == ["CAPABILITY_SOURCE_ENGINE_PROBE"],
          "the only CODE site that can write 'engine-probe' is the constant definition",
          "sites=%r" % (sites,))


# ---------------------------------------------------------------------------------------
# 3. The two consumers: neither may turn "unknown" into a verdict.
# ---------------------------------------------------------------------------------------
def t_consumers():
    print("3. the consumers (the constants that were the defect)")
    g = read("ninfer-gui.py")
    check("gpu_compat.env_item(" in g,
          "ninfer-gui.py builds the row with the computed function")
    check(re.search(r"'name':\s*'显卡世代',\s*'ok':\s*True", g) is None,
          "ninfer-gui.py no longer stamps a constant 'ok': True on the GPU-generation row")
    check(re.search(r"'name':\s*'显卡世代'[^}]*'ok':\s*True", g) is None,
          "…nor anywhere else in a 显卡世代 row")
    h = read("gui_page.html")
    check("it.ok===null" in h or "it.ok===true" in h,
          "gui_page.html distinguishes the third state explicitly")
    check(re.search(r"\(it\.ok\?", h) is None,
          "gui_page.html no longer uses truthiness on `ok` (truthiness maps None -> the ✖ glyph)")
    check("'?'" in h and "unknown" in h,
          "the third state gets a neutral class and a neutral glyph")
    # the renderer's own mapping, read out of the file, must agree with the three states
    m = re.search(r"const glyph=it\.ok===true\?'([^']*)':\(it\.ok===false\?'([^']*)':'([^']*)'\)", h)
    check(m is not None, "the glyph mapping is present and parseable")
    if m:
        tick, cross, neutral = m.group(1), m.group(2), m.group(3)
        check(tick != neutral and cross != neutral,
              "the unmeasured glyph is distinct from both the tick and the cross (%r/%r/%r)"
              % (tick, cross, neutral))


# ---------------------------------------------------------------------------------------
# 4. SELF-PROOF. The old shape must FAIL these assertions, or the test is not evidence.
# ---------------------------------------------------------------------------------------
def guess_constant_row():
    """The exact literal ninfer-gui.py used: `ok` as a constant True."""
    return {"name": "显卡世代", "ok": True, "detail": "x", "fix": None}


def the_invariant(it):
    """The single assertion the whole test exists for."""
    return it.get("ok") is not True or it.get("capability_state") == gc.CAPABILITY_MEASURED


def mutate():
    print("MUTATION CHECK: the old constant shape must be CAUGHT (no env, no GPU)")
    old = guess_constant_row()
    check(the_invariant(old) is False,
          "the old `ok: True` constant shape VIOLATES the invariant (this assertion failing "
          "would mean the test cannot see the defect)")
    fixed = gc.env_item(row())
    check(the_invariant(fixed) is True,
          "the landed shape SATISFIES the invariant")
    print()
    if FAILS == 0:
        print("== MUTATION: the old shape is CAUGHT and the landed shape is accepted"
              " (%d checks) ==" % CHECKS)
        return 0
    print("== MUTATION: %d unexpected failure(s) ==" % FAILS)
    return 1


def main():
    print("== test_gpu_compat: a capability row must be measured, not stamped ==")
    t_decision()
    t_real_query()
    t_consumers()
    print()
    print("== %s ==" % ("ALL PASS" if FAILS == 0 else "HAS FAILURES"))
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        raise SystemExit(mutate())
    raise SystemExit(main())
