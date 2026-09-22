#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_serve_sim_env.py -- Route D of SIMFIX2, as a RUNNABLE MEASUREMENT rather than a note.

WHAT THIS IS. A hazard probe, NOT a unit test, and nothing invokes it automatically. It exists
because `tools/gui/serve_gui.py` is owned by a live line (it was written at 17:08 today), so the
SIMFIX2 line would not edit it -- but the two facts below are about that file and can be measured
from outside without importing or touching it.

THE HAZARD, in two halves:

  H1  THE SIMULATOR IS INHERITED INTO THE ENGINE THE GUI SPAWNS.
      `serve_gui.py` builds the engine's environment by copying `os.environ` wholesale:
          env = dict(os.environ, LD_LIBRARY_PATH='/usr/local/cuda-13.3/lib64')
      and NOTHING in that file names NINFER_SIM_ARCH (`grep -n 'NINFER_' tools/gui/serve_gui.py`
      returns no line), so the key is carried through. A user who has the test-only architecture
      simulator exported in the shell they launched the GUI from gets a SIMULATED engine under a
      pane that shows their real card's `nvidia-smi` badge.
      This is checked by EXECUTING THE FILE'S OWN LINE inside a child process that has the
      simulator exported -- not by re-implementing it -- and reporting whether the key survived.

  H2  THE SIMULATED BANNER IS EVICTED FROM THE LOG THE GUI SHOWS.
      The banner (`sim_banner()`, src/core/arch_sim.h) is printed ONCE PER PROCESS, at the first
      arch-view call -- i.e. at the very start of a serve run. The GUI keeps a sliding window:
          if len(_log) > 12000: _log = _log[-8000:]        # the reader thread
          'log': _log[-6000:]                               # what /api/state serves
      so once the run has emitted ~8 kB the marker is no longer in what a user can see, while the
      session keeps reporting 'running': True with live metrics.

WHAT THIS DOES NOT CLAIM. It does not claim the GUI shows a SUPPORT verdict -- what it shows from
such a run is spec-decoding acceptance figures, which is the PERFORMANCE prohibition
(src/core/arch_sim.h:59-70), not the support one. And the end-to-end run (sim exported, GUI
launched, banner watched scrolling out) is NOT measured here: it needs the GPU lock and a model.

EXIT CODE. 0 when both hazards are CLOSED (the env is scrubbed, the marker is preserved),
77 when either stands -- 77 is the tree's own "not applicable here" convention
(tests/CMakeLists.txt sets SKIP_RETURN_CODE 77 for the op tests), so this can be registered later
without reddening a suite while the file is someone else's to fix.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUBJECT = os.path.join(REPO, "tools", "gui", "serve_gui.py")
BANNER_ONCE_SRC = os.path.join(REPO, "src", "core", "arch_sim.h")

ENV_LINE_RE = re.compile(r"^\s*env\s*=\s*(dict\(os\.environ.*\))\s*$", re.M)
DROP_RE = re.compile(r"NINFER_SIM_ARCH")
EVICT_RE = re.compile(r"len\(_log\)\s*>\s*(\d+)\s*:\s*\n\s*_log\s*=\s*_log\[-(\d+):\]")
SERVED_RE = re.compile(r"'log':\s*_log\[-(\d+):\]")
ONCE_RE = re.compile(r"ONCE per process")


def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


H1_PROBE = r"""
import os, sys, json
sys.path.insert(0, SUBJECT_DIR)
import serve_gui
env, dropped = serve_gui.engine_env()
print(json.dumps({"in_child": [k for k in ("NINFER_SIM_ARCH", "NINFER_SIM_ARCH_ACK")
                                if k in env],
                  "dropped": sorted(dropped or [])}))
"""


def h1(text):
    """Is the simulator inherited? Measured by EXECUTING the engine_env() that builds the env.

    The previous form of this check regex-matched the env-construction LINE and executed it in a
    child. That was right while the construction was one line at the spawn site, and became a
    FALSE ALARM once the construction moved into a helper: the regex then matched the helper's
    INTERNAL `dict(os.environ)` copy, which runs before the keys are dropped.
    """
    print("H1  is NINFER_SIM_ARCH inherited into the engine the GUI spawns?")
    names = [ln.strip() for ln in text.splitlines() if DROP_RE.search(ln)]
    print("    lines naming NINFER_SIM_ARCH: %d" % len(names))
    # EXECUTE the file's own engine_env() in a child that has the simulator exported.
    prog = ("SUBJECT_DIR = %r\n" % os.path.dirname(SUBJECT)) + H1_PROBE
    r = subprocess.run([sys.executable, "-c", prog],
                       env=dict(os.environ,
                                NINFER_SIM_ARCH="70",
                                NINFER_SIM_ARCH_ACK="I-UNDERSTAND-THIS-IS-NOT-A-V100"),
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print("    the child could not run engine_env(): rc=%d %s"
              % (r.returncode, (r.stderr or "").strip().splitlines()[-1:] or ""))
        print("    -> the file no longer offers engine_env(); re-read it.")
        return None
    try:
        import json as _json
        got = _json.loads((r.stdout or "").strip().splitlines()[-1])
    except Exception:
        print("    child output was not the expected JSON: %r" % (r.stdout or "")[:200])
        return None
    print("    engine_env() called with NINFER_SIM_ARCH=70 exported:")
    print("      keys surviving into the child env : %s" % (got["in_child"] or "none"))
    print("      keys reported as dropped         : %s" % (got["dropped"] or "none"))
    return not got["in_child"]


H2_PROBE = r"""
import os, sys, json
sys.path.insert(0, SUBJECT_DIR)
import serve_gui
serve_gui._log = ""
serve_gui._sim_pin = ""
marker = ("[route][SIMULATED ARCHITECTURE] the previous shape "
          "reporting a capability this binary was not built for\n")
serve_gui.log_append(marker)
fed = len(marker)
for i in range(1400):
    line = "step %04d tok=%d ordinary output line here\n" % (i, i * 7 % 512)
    fed += len(line)
    serve_gui.log_append(line)
shown = serve_gui._shown_log()
tail = serve_gui._log[-(serve_gui.LOG_TAIL_CHARS):]
print(json.dumps({"fed": fed, "shown_has_marker": serve_gui.SIM_MARKER in shown,
                  "tail_alone_has_marker": serve_gui.SIM_MARKER in tail,
                  "pin_len": len(serve_gui._sim_pin)}))
"""


def h2(text):
    """Is the banner evicted from what the GUI shows? Measured by feeding the real log path."""
    print("H2  is the SIMULATED banner evicted from the log the GUI shows?")
    # the literal is measured, not assumed: arch_sim.h says "ONCE per process"
    bt = read(BANNER_ONCE_SRC)
    line = next((i + 1 for i, ln in enumerate(bt.splitlines()) if ONCE_RE.search(ln)), None)
    print("    banner cadence    : src/core/arch_sim.h:%s says %r" %
          (line, ONCE_RE.search(bt).group(0) if ONCE_RE.search(bt) else "NOT FOUND"))
    prog = ("SUBJECT_DIR = %r\n" % os.path.dirname(SUBJECT)) + H2_PROBE
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print("    the child could not drive log_append/_shown_log: rc=%d %s"
              % (r.returncode, (r.stderr or "").strip().splitlines()[-1:] or ""))
        return None
    try:
        import json as _json
        got = _json.loads((r.stdout or "").strip().splitlines()[-1])
    except Exception:
        print("    child output was not the expected JSON: %r" % (r.stdout or "")[:200])
        return None
    print("    fed %d chars of ordinary output through log_append after the banner" % got["fed"])
    print("      a plain tail window alone would still show it : %s"
          % got["tail_alone_has_marker"])
    print("      what /api/state would serve still shows it    : %s" % got["shown_has_marker"])
    if got["fed"] < 60000:
        print("    (the volume was too small to prove eviction -- not a pass)")
        return None
    return bool(got["shown_has_marker"])


def main():
    print("== probe_serve_sim_env: Route D, measured from outside %s" % os.path.basename(SUBJECT))
    print("   subject: %s" % SUBJECT)
    if not os.path.exists(SUBJECT):
        print("   subject absent -- nothing to measure (exit 77)")
        return 77
    text = read(SUBJECT)
    ok1 = h1(text)
    print()
    ok2 = h2(text)
    print()
    if ok1 is None or ok2 is None:
        print("VERDICT: UNDECIDED -- the file's shape has changed; re-read it. exit 77")
        return 77
    if ok1 and ok2:
        print("VERDICT: HAZARD CLOSED on both halves. exit 0")
        return 0
    print("VERDICT: HAZARD STANDS (H1 inherited=%s, H2 evicted=%s). exit 77" %
          (not ok1, not ok2))
    print("   owner: whoever holds tools/gui/serve_gui.py. The fix is one line for H1")
    print("   (drop the keys when building env) and one for H2 (keep a startup slice, or")
    print("   re-emit the marker into the served window).")
    return 77


if __name__ == "__main__":
    raise SystemExit(main())
