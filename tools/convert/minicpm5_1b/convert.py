# -*- coding: utf-8 -*-
"""Convert the MiniCPM5-1B checkpoint into one `.ninfer` artifact.

    python -m tools.convert.minicpm5_1b.convert \
      --model /path/to/MiniCPM5-1B \
      --out /home/user/models/minicpm5_1b_bf16.ninfer \
      --spec-dir tools/archkit/specs \
      --plan            # optional: gates only, nothing written

Compare this file with `tools/convert/muse_glimmer_30b/convert.py` (555 lines, of
which the body of `convert()` is a 40-arm if/elif chain over object names) and with
`tools/convert/spark_x2_5_4b/convert.py` (1,194 lines).  What is left here is a
DECLARATION: a spec, a role table, a recipe table, and the tie rule.  Everything
else -- config/index reading and their named refusals, spec validation, object
planning and offset allocation, the gate battery, the streaming write, the
re-open with the container's own reader, the byte count and the sha256 -- is
`tools/archkit/kit/driver.py`, one copy, shared by every model that uses it.

The file is short on purpose and that is the deliverable: a model-side module that
grows past a declaration has re-implemented the driver, and `kit/check.py` says so.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):                                  # run-by-path support
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "tools.convert.minicpm5_1b"

from tools.archkit.kit import driver

from .declaration import build_declaration


def main(argv=None) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    spec_dir = repo_root / "tools" / "archkit" / "specs"
    if argv is None:
        argv = sys.argv[1:]
    # --spec-dir defaults to the tree's spec directory; an explicit flag still wins.
    if "--spec-dir" not in argv:
        argv = list(argv) + ["--spec-dir", str(spec_dir)]
    if "--out" not in argv and "--plan" in argv:
        # `--plan` never writes, but the shared driver's CLI insists on a target
        # name.  A plan run is allowed to name a path it will not create.
        argv = list(argv) + ["--out", str(Path("out") / "minicpm5_1b.ninfer")]
    return driver.main(build_declaration, argv)


if __name__ == "__main__":
    raise SystemExit(main())
