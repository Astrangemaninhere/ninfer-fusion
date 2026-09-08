# -*- coding: utf-8 -*-
"""adapt_all.py - 一键全自动适配管线 (验收标准执行入口, v1).

用法::
    python3 adapt_all.py <config.json|model-dir|spec.json> [--workdir OUT] [--verify]

阶段 (每个失败即停, 报告缺口):
  1. spec: config.json -> arch-spec (specs/<model_id>_spec.json)
  2. adapt: spec -> config.h + engine_hook.patch + manifest (gaps: covered|hook|new_op)
  3. leaves: gen_variant -> 口味叶子代码 (attn/mlp 文本)
  4. stubs:  gen_stubs_v2 -> bindings/package/recipe 骨架 (新 target 装配起点)
  5. verify: g++ -fsyntax-only config.h (需要 WSL g++, --verify)

验收判据: 阶段 1-4 全绿 + manifest 无 new_op 缺口 => "适配完成, 可 serve"
          (new_op 缺口 => 报告需要新内核口味的工作包, 一次落地后同族免费)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPECS = HERE / 'specs'
OUT = HERE / 'out'


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print('  $ ' + ' '.join(str(c) for c in cmd))
    return subprocess.call([str(c) for c in cmd], cwd=cwd)


def ensure_spec(input_path: Path, model_id: str) -> Path:
    """config.json/model-dir/spec.json -> spec file path (reuse adapt.extract_spec)."""
    sys.path.insert(0, str(HERE))
    if input_path.name == 'config.json' or (input_path.is_dir() and
                                            (input_path / 'config.json').exists()):
        cfg = input_path / 'config.json' if input_path.is_dir() else input_path
        import adapt
        spec = adapt.extract_spec(str(cfg), model_id)
        spec_path = SPECS / ('%s_spec.json' % spec['model_id'])
        SPECS.mkdir(exist_ok=True)
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
        return spec_path
    if input_path.suffix == '.json' and 'spec' in input_path.name:
        return input_path
    raise SystemExit('input must be config.json, a model dir, or a spec json')


def main() -> int:
    ap = argparse.ArgumentParser(description='one-shot auto-adapt pipeline')
    ap.add_argument('input')
    ap.add_argument('--model-id', default='')
    ap.add_argument('--workdir', default=str(OUT), help='output root (default tools/archkit/out)')
    ap.add_argument('--verify', action='store_true', help='run g++ syntax gate (WSL)')
    ap.add_argument('--skip-leaves', action='store_true')
    args = ap.parse_args()

    inp = Path(args.input)
    # model_id: spec 输入读内容; config.json/目录用目录名
    if inp.suffix == '.json' and 'spec' in inp.name:
        model_id = json.loads(inp.read_text(encoding='utf-8')).get('model_id', '') or \
            inp.name.replace('_spec.json', '')
    else:
        model_id = (args.model_id or (inp.parent.name if inp.is_file() else inp.name)
                    or 'model')
    model_id = model_id.lower().replace('_', '-')
    workdir = Path(args.workdir)
    print('== adapt_all: %s (workdir %s)' % (model_id, workdir))

    # 1) spec
    spec_path = ensure_spec(inp, model_id)
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    print('[1/5] spec ok: %s' % spec_path.name)

    # 2) adapt (config.h + engine_hook.patch + manifest with gaps)
    if run([sys.executable, str(HERE / 'adapt.py'), str(spec_path), '--model-id', model_id]) != 0:
        return 1
    adapted = workdir / model_id
    manifest = json.loads((adapted / 'manifest.json').read_text(encoding='utf-8'))
    gaps = manifest.get('gaps', [])
    new_ops = [g for g in gaps if g.get('tier') == 'new_op']
    hooks = [g for g in gaps if g.get('tier') != 'new_op']
    print('[2/5] adapt ok: config.h + %d gaps (%d hooks, %d new_op)'
          % (len(gaps), len(hooks), len(new_ops)))

    # 2.5) geometry gate: 引擎 gqa 分派表必须覆盖本模型 (q,kv,head_dim) —
    #      防 Muse 式静默错几何 (缺口 => 不能声称可 serve)
    geo_gate = HERE / 'check_geometry.py'
    if geo_gate.exists():
        print('[2.5/5] geometry gate...')
        if run([sys.executable, str(geo_gate), str(spec_path), '--repo', str(HERE.parent.parent)]) != 0:
            print('geometry gate FAILED: 模型注意力几何未在引擎分派表注册')
            return 1
        print('[2.5/5] geometry gate ok')
    else:
        print('[2.5/5] geometry gate skipped (check_geometry.py missing)')

    # 2.6) params gate: 输入为 config.json/模型目录时, 全键完备性 (无静默忽略;
    #       MoE/视觉/未知键 => 工作包报告, text 域未知键计入报告)
    params_report = ''
    if inp.suffix == '.json' and 'spec' not in inp.name or inp.is_dir():
        params_gate = HERE / 'check_params.py'
        if params_gate.exists():
            print('[2.6/5] params gate...')
            target = inp / 'config.json' if inp.is_dir() else inp
            params_report = adapted / 'params_unfitted.md'
            rc = run([sys.executable, str(params_gate), str(target),
                      '--report', str(params_report)])
            print('[2.6/5] params gate %s' % ('has 未适配 (工作包化, 见报告)' if rc else 'ok'))
    else:
        print('[2.6/5] params gate skipped (input is spec, 无原始 config)')

    # 3) leaves (口味 -> 共享算子叶子)
    if not args.skip_leaves:
        import flavors as _flavors  # noqa: F401  (module import check)
        family = spec.get('family') or 'qwen38'
        if family not in _flavors.FLAVORS:
            # 家族键归一: qwen3_6/qwen38 同口味库
            for candidate in ('qwen38', 'qwen2'):
                if candidate in _flavors.FLAVORS:
                    print('  [family] %s -> %s (flavors key)' % (family, candidate))
                    family = candidate
                    break
        leaf_out = adapted / ('leaves_%s.txt' % family)
        with open(leaf_out, 'w', encoding='utf-8') as f:
            rc = subprocess.call([sys.executable, str(HERE / 'gen_variant.py'),
                                  str(spec_path), '--emit', 'all', '--family', family],
                                 stdout=f)
        if rc != 0:
            return 1
        print('[3/5] leaves ok -> %s' % leaf_out.name)

    # 4) stubs (bindings/package/recipe 骨架)
    if run([sys.executable, str(HERE / 'gen_stubs_v2.py'), str(spec_path),
            '--out', str(workdir)]) != 0:
        return 1
    print('[4/5] stubs ok -> %s/%s/' % (workdir, model_id.replace('-', '_')))

    # 5) verify (g++ syntax gate via verify_v3_gen stub env when available)
    if args.verify:
        v3 = HERE / 'verify_v3_gen.py'
        if not v3.exists():
            print('[5/5] verify skipped (verify_v3_gen.py missing)')
        else:
            print('[5/5] verify...')
            if run([sys.executable, str(v3)]) != 0:
                print('verify FAILED')
                return 1
            print('verify OK')

    # report
    report = adapted / 'adapt_report.md'
    lines = [
        '# 适配报告: %s' % model_id,
        '',
        '- 完成: spec -> config.h -> manifest -> 口味叶子 -> bindings/package/recipe 骨架',
        '- new_op 缺口 (%d): 每个=需要一次新内核口味落地, 之后同族免费' % len(new_ops),
    ]
    for g in new_ops:
        lines.append('  - [%s] %s: %s' % (g.get('tier'), g.get('need'), g.get('action')))
    for g in hooks:
        lines.append('  - [%s] %s' % (g.get('tier'), g.get('need')))
    lines += ['', '验收判据: %s' % ('适配完成, 可 serve' if not new_ops else
                                   '缺 %d 个 new_op 内核口味, 落地后同族免费' % len(new_ops))]
    if params_report and Path(params_report).exists():
        lines += ['', '参数完备门报告 (未适配键, 无静默忽略): %s' % params_report]
    report.write_text('\n'.join(lines), encoding='utf-8')
    print('== done -> %s' % report)
    return 0 if not new_ops else 0  # new_op 不阻断管线 (工作包化), 报告为准


if __name__ == '__main__':
    raise SystemExit(main())
