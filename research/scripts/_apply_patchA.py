#!/usr/bin/env python3
"""Patch A applicator (idempotent). argv[1] = target file; writes the diff only for the
build tree. Keeps the C: mirror text in sync so agent diffs stay against the same base."""
import difflib
import hashlib
import pathlib
import sys

BUILD = "/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/program_impl.h"
MIRROR = ("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/"
          "src/targets/qwen3_6/impl/runtime/program_impl.h")
DIFF = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_patchA_dflash2_state_slots.diff")

ANCHOR = """            dflash2_host_ingress->active_lanes[row] = static_cast<std::int32_t>(sequence.lane);
            dflash2_host_ingress->sampling[row]     = request.sampling_host;
"""
INSERT = """            dflash2_host_ingress->active_lanes[row] = static_cast<std::int32_t>(sequence.lane);
            dflash2_host_ingress->sampling[row]     = request.sampling_host;
            // The verify reads the recurrent (GDN) state from these device slots and
            // scatters the continuation hidden back to them; leaving them at their
            // zero-initialised value made KV and state disagree on every round.
            const StateImageSelectors selectors                 = state_selectors(sequence);
            dflash2_host_ingress->state_source_slots[row]      = selectors.source;
            dflash2_host_ingress->state_destination_slots[row] = selectors.destination;
"""

target = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else BUILD)
src = target.read_text()
if "dflash2_host_ingress->state_source_slots[row]" in src:
    print("%s: ALREADY_PATCHED" % target)
    sys.exit(0)
n = src.count(ANCHOR)
if n != 1:
    print("%s: ANCHOR_COUNT=%d (expected 1) - refusing" % (target, n))
    sys.exit(2)
new = src.replace(ANCHOR, INSERT)
backup = target.with_suffix(target.suffix + ".bak_patchA")
if not backup.exists():
    backup.write_text(src)
target.write_text(new)
print("%s: PATCHED md5 %s -> %s" % (target.name,
                                    hashlib.md5(src.encode()).hexdigest()[:8],
                                    hashlib.md5(new.encode()).hexdigest()[:8]))
if str(target) == BUILD:
    d = difflib.unified_diff(src.splitlines(True), new.splitlines(True),
                             fromfile="a/src/targets/qwen3_6/impl/runtime/program_impl.h",
                             tofile="b/src/targets/qwen3_6/impl/runtime/program_impl.h", n=4)
    DIFF.write_text("".join(d))
    print("diff -> %s" % DIFF)
