#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_serve_sim_route.py -- Route D, guarded by a test that FAILS when the route reopens.

WHY THIS FILE EXISTS. `tools/gui/probe_serve_sim_env.py` (SIMFIX2) measures the same two hazards
and exits 77 -- the tree's SKIP convention. A probe that can only SKIP cannot redden anything, and
"a guard that exists but is never run" is this project's most-repeated defect, one level up. This
test is the registered half: it exits NON-ZERO when either hazard is back.

THE ROUTE IT CLOSES. A GUI whose spawned engine inherits NINFER_SIM_ARCH from the shell that
launched the GUI shows a simulated run under a pane that still carries the REAL card's
nvidia-smi badge. That is a simulated result presented as a real-hardware one, which this project
forbids: a build configuration is not a support claim, and only a probe on real hardware is
evidence of support. The banner matters for the same reason -- sim_banner() prints ONCE per
process at startup, so a trailing log window loses the only marker that says so.

WHAT IS MEASURED, AND HOW. Not by re-implementing the logic, and NOT by grepping for a string --
the first draft of T3 did that and matched its own explanatory comment, which is the same defect
SIMFIX2 recorded one layer down ("the mint-site check matched prose"). T3 is an AST property.

  T1  drives the GUI's REAL start path. `serve_gui.subprocess` is replaced with a shim whose
      Popen records its arguments instead of spawning, the real HTTP handler is served on an
      ephemeral loopback port, and POST /api/start is performed against it with the simulator
      pair EXPORTED in os.environ. The env actually handed to the engine is then inspected.
  T2  drives the real `log_append`/`_shown_log` pair with ~60 kB of ordinary engine output after
      a banner line, and asserts the banner is still in what /api/state would serve. It also
      asserts a plain tail window WOULD have lost it, so the test cannot pass vacuously.
  T3  is an AST property over serve_gui.py: every call that passes an `env=` keyword must pass a
      name bound from `engine_env()`. Prose and comments cannot satisfy it, and it fails on the
      pre-fix source -- `--subject` exists so that is demonstrated rather than asserted.

HOST ONLY. No CUDA, no device, no model, no nvidia-smi, no engine process is spawned (Popen is
replaced), and no lock is taken. Safe to register in a suite anyone runs.

Exit 0 = both hazards closed. Exit 1 = a hazard is back, or a check could not be evaluated. There
is deliberately NO 77 path: a skip with nothing behind it is the defect this file exists to avoid.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import socket
import sys
import threading
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import serve_gui  # noqa: E402  (after the sys.path insert, deliberately)

FAILURES: list[str] = []
SUBJECT = os.path.join(HERE, "serve_gui.py")


def check(cond: bool, label: str, detail: str = "") -> bool:
    print("  %s %s%s" % ("PASS" if cond else "FAIL", label, ("  -- " + detail) if detail else ""))
    if not cond:
        FAILURES.append(label + ((" -- " + detail) if detail else ""))
    return cond


class _FakeProc:
    """Stands in for the engine process: no engine is ever spawned by this test."""

    def __init__(self):
        self.stdout = iter(())
        self.returncode = None

    def poll(self):
        return None

    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


class _Recorder:
    """Records the arguments the GUI would have spawned the engine with."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append({'cmd': list(cmd), 'env': dict(kw.get('env') or {}), 'kw': kw})
        return _FakeProc()


class _SubprocessShim:
    """serve_gui's view of `subprocess`, so the global module is untouched."""

    PIPE = -1
    STDOUT = -2
    DEVNULL = -3

    def __init__(self, recorder):
        self.Popen = recorder

    def run(self, *a, **k):  # pragma: no cover - /api/start must not shell out
        raise AssertionError("serve_gui called subprocess.run during /api/start: %r" % (a,))


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def t1_engine_env_is_scrubbed() -> None:
    print("T1  the engine the GUI spawns does not inherit the test-only simulator")
    saved_env = dict(os.environ)
    saved_popen_mod = serve_gui.subprocess
    saved_proc = serve_gui._proc
    rec = _Recorder()
    srv = None
    try:
        os.environ["NINFER_SIM_ARCH"] = "70"
        os.environ["NINFER_SIM_ARCH_ACK"] = "I-UNDERSTAND-THIS-IS-NOT-A-V100"
        serve_gui.subprocess = _SubprocessShim(rec)
        serve_gui._proc = None
        serve_gui._session = {}
        serve_gui._log = ""
        serve_gui._sim_pin = ""

        port = _free_port()
        srv = serve_gui.ThreadingHTTPServer(("127.0.0.1", port), serve_gui.H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()

        body = json.dumps(serve_gui.DEFAULT_REQ).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:%d/api/start" % port, data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=20) as r:
            resp = json.loads(r.read().decode())

        check(bool(rec.calls) and "env" in rec.calls[0],
              "POST /api/start reached the engine-spawn site",
              "recorded %d spawn call(s)" % len(rec.calls))
        if not rec.calls:
            return
        env = rec.calls[0]["env"]
        check("NINFER_SIM_ARCH" not in env,
              "NINFER_SIM_ARCH is NOT in the spawned engine's environment",
              "exported in our own env, absent in the child's")
        check("NINFER_SIM_ARCH_ACK" not in env,
              "NINFER_SIM_ARCH_ACK is NOT in the spawned engine's environment")
        check(env.get("LD_LIBRARY_PATH") == "/usr/local/cuda-13.3/lib64",
              "the env is still the one the GUI intends (LD_LIBRARY_PATH forced)",
              repr(env.get("LD_LIBRARY_PATH")))
        check(bool(rec.calls[0]["cmd"]) and rec.calls[0]["cmd"][0] == serve_gui.SERVE_BIN,
              "the command is unchanged (the GUI still starts its own engine)",
              os.path.basename(rec.calls[0]["cmd"][0]))

        # THE DROP MUST BE VISIBLE, not silent: /api/state carries it and the log says it.
        sim = (serve_gui._session or {}).get("sim")
        check(isinstance(sim, dict) and sim.get("requested") is False,
              "/api/state reports the sim state explicitly", "sim=%r" % (sim,))
        check(isinstance(sim, dict) and sorted(sim.get("dropped") or []) ==
              ["NINFER_SIM_ARCH", "NINFER_SIM_ARCH_ACK"],
              "the drop NAMES the keys it removed", "dropped=%r" % ((sim or {}).get("dropped"),))
        check("NINFER_SIM_ARCH" in (serve_gui._log or ""),
              "the log tells the operator the keys were dropped",
              repr((serve_gui._log or "").splitlines()[-1][:150] if serve_gui._log else ""))
        check(isinstance(resp, dict) and resp.get("running") is True,
              "the start response is still well formed", "running=%r" % resp.get("running"))
    finally:
        serve_gui.subprocess = saved_popen_mod
        serve_gui._proc = saved_proc
        if srv is not None:
            try:
                srv.shutdown()
                srv.server_close()
            except Exception:
                pass
        os.environ.clear()
        os.environ.update(saved_env)


def t2_banner_survives_volume() -> None:
    print("T2  the SIMULATED banner is not evicted by log volume")
    saved_log, saved_pin = serve_gui._log, serve_gui._sim_pin
    try:
        serve_gui._log = ""
        serve_gui._sim_pin = ""
        banner = ("[route][SIMULATED ARCHITECTURE] reporting capability 70 while executing the "
                  "kernels this binary was built with\n")
        serve_gui.log_append(banner)
        check(serve_gui.SIM_MARKER in serve_gui._shown_log(),
              "banner is visible immediately after it is printed")

        # Far past both windows (the reader's 12000/8000 and the served 6000). _log is a WINDOW,
        # so the volume to count is what was fed in, not len(_log).
        fed = len(banner)
        for i in range(1200):
            line = ("step %04d  tok=%d  t=%.3f ms  ordinary line here\n"
                    % (i, i * 7 % 512, i * 0.137))
            fed += len(line)
            serve_gui.log_append(line)

        shown = serve_gui._shown_log()
        tail_alone = serve_gui._log[-(serve_gui.LOG_TAIL_CHARS):]
        check(fed > 60000, "the volume really was large enough to evict",
              "%d chars fed through log_append" % fed)
        check(len(serve_gui._log) <= 12000, "the reader window actually engaged",
              "len(_log)=%d (bounded by the 12000 threshold)" % len(serve_gui._log))
        check(serve_gui.SIM_MARKER not in tail_alone,
              "a plain tail window WOULD have lost it (so the pin is load-bearing)",
              "tail window = last %d chars" % serve_gui.LOG_TAIL_CHARS)
        check(serve_gui.SIM_MARKER in shown,
              "the banner is STILL in what /api/state would serve",
              "shown log = %d chars" % len(shown))
        check(bool(serve_gui._sim_pin) and serve_gui.SIM_MARKER in serve_gui._sim_pin,
              "the pin holds the marker itself, not a copy of the whole log",
              "pin = %d chars" % len(serve_gui._sim_pin))
    finally:
        serve_gui._log, serve_gui._sim_pin = saved_log, saved_pin


def _env_binding_sites(tree: ast.AST):
    """Names bound from `engine_env()`, and every `env=` keyword that is NOT one of them.

    The hazard expressed as a property of the AST rather than of the text: a call that passes an
    environment to a child must pass a name this module obtained from engine_env(). A comment
    mentioning `dict(os.environ, ...)` cannot satisfy it, and the pre-fix form -- which bound
    `env` from `dict(os.environ, ...)` directly -- cannot either.
    """
    def callee_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    safe: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and callee_name(node.value.func) == "engine_env":
            for tgt in node.targets:
                for e in (tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]):
                    if isinstance(e, ast.Name):
                        safe.add(e.id)

    unsafe = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords or []:
            if kw.arg != "env":
                continue
            v = kw.value
            if not (isinstance(v, ast.Name) and v.id in safe):
                unsafe.append("line %d: env=%s" % (getattr(node, "lineno", 0),
                                                   ast.unparse(v)[:70]))
    return safe, unsafe


def t3_the_shape_is_still_guarded() -> None:
    print("T3  AST property: every child env comes from engine_env()")
    with open(SUBJECT, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=SUBJECT)
    print("  subject for T3: %s" % SUBJECT)
    safe, unsafe = _env_binding_sites(tree)
    check(bool(safe), "engine_env() result is bound to a name",
          "bound: %s" % ", ".join(sorted(safe)))
    check(not unsafe, "no call passes a child env that did not come from engine_env()",
          "; ".join(unsafe) if unsafe else "0 unsafe env= sites")
    check("'log': _shown_log()" in src, "/api/state serves the pinned log (not a bare tail)")
    check("'sim': _sim_state()" in src, "/api/state reports the sim state")
    check(bool(ast.parse(src).body), "the module parses as Python")


def main() -> int:
    global SUBJECT
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=SUBJECT,
                    help="serve_gui.py to check (default: the one beside this test). Used to "
                         "demonstrate that T3 FAILS on the pre-fix source.")
    ap.add_argument("--skip-t1", action="store_true",
                    help="T1/2 import serve_gui and always test the installed module, so a "
                         "regression run against another --subject checks T3 only.")
    a = ap.parse_args()
    SUBJECT = a.subject

    print("== test_serve_sim_route: Route D (the simulator reaching the GUI-spawned engine) ==")
    print("   installed module: %s" % os.path.join(HERE, "serve_gui.py"))
    print("   T3 subject      : %s" % SUBJECT)
    print("   os.environ at start: NINFER_SIM_ARCH=%r\n" % os.environ.get("NINFER_SIM_ARCH"))
    if not a.skip_t1:
        t1_engine_env_is_scrubbed()
        print()
        t2_banner_survives_volume()
        print()
    t3_the_shape_is_still_guarded()
    print()
    if FAILURES:
        print("VERDICT: FAIL -- %d check(s) failed:" % len(FAILURES))
        for f in FAILURES:
            print("   - %s" % f)
        return 1
    print("VERDICT: PASS -- the GUI-spawned engine cannot inherit the simulator, and the banner "
          "survives any log volume.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
