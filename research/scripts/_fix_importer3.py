#!/usr/bin/env python3
"""Importer follow-ups for the Spark-X2.5 finding:
  (a) the enriched spec was never written back to specs/<id>_spec.json (hf_to_spec had
      already written the older dict), so the extra fields only lived in memory;
  (b) `head:tied=true` blocks the import, but the concrete unblock is a *conversion*
      step, not an engine op: materialise lm_head = embed^T at convert time. Say so, with
      the cost, instead of leaving the reader to guess."""
import pathlib
import sys

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit/adapt.py")
src = P.read_text(encoding="utf-8")

A = """    spec['raw_text_config'] = tc
    return spec"""
B = """    spec['raw_text_config'] = tc
    # hf_to_spec writes its own (older) dict to specs/<id>_spec.json, so persist the
    # enriched one too - otherwise the rope/gate/act fields below only exist in memory
    # and the gate/audit scripts that re-read the file see a different spec.
    try:
        with open(os.path.join(REPO, 'tools', 'archkit', 'specs', model_id + '_spec.json'),
                  'w', encoding='utf-8') as f:
            json.dump(spec, f, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception as e:  # never let a persistence hiccup block the gap report
        spec['spec_write_error'] = str(e)
    return spec"""
if B.splitlines()[2] not in src:
    if src.count(A) != 1:
        print("ANCHOR1 count=%d - refusing" % src.count(A)); sys.exit(2)
    src = src.replace(A, B)
    print("patched: extract_spec persists the enriched spec")
else:
    print("spec persistence already patched")

C = """        out.append(('head:tied=true', 'new_op', 'tied head (embed^T reuse, engine flavor)'))"""
D = """        g0 = spec.get('geometry') or {}
        tie_mb = (g0.get('vocab', 0) * g0.get('hidden', 0) * 2) / 1e6
        out.append(('head:tied=true', 'new_op',
                    'tied head: 转换期物化 lm_head = embed^T (约 %.0f MB) 或引擎 flavor 复用;'
                    ' 引擎现按独立 lm_head 对象加载, converter 目前硬写 tie=False' % tie_mb))"""
if D.strip() not in src:
    if src.count(C) != 1:
        print("ANCHOR2 count=%d - refusing" % src.count(C)); sys.exit(2)
    src = src.replace(C, D)
    print("patched: tied-head action now names the conversion path and its cost")
else:
    print("tie message already patched")

P.write_text(src, encoding="utf-8")
print("now %d lines" % (src.count("\n") + 1))
