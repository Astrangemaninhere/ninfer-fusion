# -*- coding: utf-8 -*-
"""adapt_all.py - 一键全自动适配管线 (验收标准执行入口, v1).

用法::
    python3 adapt_all.py <config.json|model-dir|spec.json> [--workdir OUT] [--verify]
                         [--require-servable] [--allow-unfinished] [--selftest]

阶段 (每个失败即停, 报告缺口):
  1. spec: config.json -> arch-spec (specs/<model_id>_spec.json)
  2. adapt: spec -> config.h + engine_hook.patch + manifest (gaps: covered|hook|new_op)
  3. leaves: gen_variant -> 口味叶子代码 (attn/mlp 文本)
  4. stubs:  gen_stubs_v2 -> bindings/package/recipe 骨架 (新 target 装配起点)
  5. verify: gen_variant 语法门 (对 stub 编译生成的叶子, --verify)

验收判据 (这里改过)。从前的判据是 "阶段 1-4 全绿 + manifest 无 new_op 缺口 =>
适配完成, 可 serve", 而阶段 1-4 全绿只说明"文件生成成功" —— 它不检查任何引擎侧
事实, 所以那个结论是凭空断言的 (假完整)。现在结论只由真跑过的检查决定, 共四种:

  servable    清单可读且带 gaps **列表**、无 new_op 缺口; 几何门真跑过并通过
              (它解析引擎源码里的 gqa 分派表, 是唯一的引擎侧检查); 且 --verify
              的语法门真跑过并通过。
  unverified  上面任一条没做 (最常见: 没给 --verify, 或几何门被跳过)。
  needs_work  清单里有 new_op 缺口 (需要一次新内核口味落地, 之后同族免费)。
  unmeasured  清单里没有 gaps **列表** —— 字段缺失 / 显式 null / 不是列表 /
              写者自己声明 `"gaps_measured": false`, 四者同判。**"未测量"不等于
              "没有缺口"**: gen_target.py 写的清单是前两种, adapt.py 在层型读不出来
              时写第三种。这一条曾经是假绿灯的入口: 旧判据是 `"gaps" not in data`,
              而 `"gaps": null` **是**在 data 里的, 于是显式 null 溜过去被渲染成
              "没有缺口、可以跑"。

只有 servable 才写"可 serve"; 其余都写明缺的是哪一步。而且即便 servable, 它证明的
也只是"几何与引擎分派表一致 + 生成代码语法通过 + 缺口已测量且无 new_op", **不是**
"引擎已经加载/推理过这个模型"——那需要一次真实加载, 本管线不做。

退出码 (默认就是结论, 不是"跑完了"):
  0  servable
  3  needs_work / unverified / unmeasured (三种都非 0)
  1  管线本身失败 (没有清单 / 清单读不出来 / 某个门返回非 0 / adapt.py 以 2 以外的
     非 0 码退出)
--require-servable = 非 servable 返回 3 —— 与默认行为相同, 保留这个名字只为兼容
                     最初的接口约定 (那时默认 0, 要靠这个旗标才变 3)。
--allow-unfinished = 非 servable 也返回 0 (旧的"工作包化"行为)。**这时退出码不再是
                     结论**, 只是"管线跑完了"; 用之前请确认调用方真的会去读报告。

⚠️ 阶段 2 的 adapt.py 有一个自己的"拒写头文件"退出码 2 (层型序列读不出来, 或几何没有
dense FFN 宽度)。那**不是**"管线本身失败": 它仍然写了一份完整的 manifest.json, 缺口
未测量这一事实就在里面, 而这个文件的判据读到它之后给出的结论是 `unmeasured` / rc=3 ——
与 `tools/gui/model_import.py --gaps <id>` 对同一份清单报的码一致。从前这里把任何非 0
都吞成 rc 1, 于是同一个产物被两个入口分别叫"管线失败"和"无法判定"。现在 rc 2 被读出来,
并且它让 `servable` 不可达 (产物集缺一个下游都假设存在的文件), 见
`adaptation_verdict(header_written=...)`。
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

#: The two names a manifest may carry. The gap-report reader
#: (tools/gui/model_import.find_manifest) accepts both, manifest.json first, and this
#: pipeline has to look in exactly the same places -- looking for only one name, under
#: the wrong directory, is how it used to lose track of the manifest it had just written.
MANIFEST_NAMES = ('manifest.json', 'arch_manifest.json')


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print('  $ ' + ' '.join(str(c) for c in cmd))
    return subprocess.call([str(c) for c in cmd], cwd=cwd)


def find_manifest(adapted: Path):
    """adapt 产物目录里的清单 -> (path, data); 两个名字都认, 找不到返回 (None, None)。

    "找不到清单"和"清单读不出来"必须分开报: 前者是没有产出, 后者是产出坏了。
    """
    for name in MANIFEST_NAMES:
        path = adapted / name
        if not path.is_file():
            continue
        try:
            return path, json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:  # 清单坏了 != 没有清单, 但要停下来说清楚
            print('  manifest 读取失败: %s (%s)' % (path, exc))
            return path, None
    return None, None


def gap_state(manifest: dict) -> tuple[bool, list, str]:
    """(measured, gaps, shape) —— **与缺口报告读者同一条规则**。

    规则的唯一权威实现在 tools/gui/model_import.manifest_gap_state; 这里逐字复刻,
    因为两侧各写一份"算不算测过"的判断, 正是当初写者/读者打架的原因。两边是否真的
    一致由 --selftest 用同一批 fixture 实测 (不是口头声明)。

      'missing'      没有 gaps 字段
      'null'         gaps 存在但为 null
      'not_a_list:X' 存在但不是列表 (或清单本身不是对象)
      'flag_false:K' 写者自己声明 gaps_measured: false (K 是 gaps 字段的实际内容,
                     因为"带了列表但说 false"对一份 gaps 为 null 的清单是错的描述)
      'alias_key:measured'  清单带了一个没人读的旗标键 `measured` (唯一有效的拼法是
                     gaps_measured)。不猜: 一个没被读的旗标看起来像有一条检查在跑,
                     而 {"gaps": [], "measured": false} (写者说它没测) 曾经被判成"测过了"
      'bad_element:T@i'     gaps 是列表, 但第 i 个元素不是对象。这里以前会
                     AttributeError (对字符串调 .get), 在读者那边表现为"栈 + rc 1",
                     与"没有清单"同一个退出码
      ''             带列表且写者没有否认 => 真的测过 (空列表也算一次测量)
    """
    if not isinstance(manifest, dict):
        return False, [], 'not_a_list:%s' % type(manifest).__name__
    if 'measured' in manifest:
        return False, [], 'alias_key:measured'
    if manifest.get('gaps_measured') is False:
        return False, [], 'flag_false:%s' % _gaps_field_kind(manifest)
    if 'gaps' not in manifest:
        return False, [], 'missing'
    raw = manifest.get('gaps')
    if raw is None:
        return False, [], 'null'
    if not isinstance(raw, list):
        return False, [], 'not_a_list:%s' % type(raw).__name__
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            return False, [], 'bad_element:%s@%d' % (type(item).__name__, i)
    return True, raw, ''


def _gaps_field_kind(manifest: dict) -> str:
    """gaps 字段实际是什么 (missing / null / list / 其它 JSON 类型名)。"""
    if 'gaps' not in manifest:
        return 'missing'
    raw = manifest.get('gaps')
    if raw is None:
        return 'null'
    if isinstance(raw, list):
        return 'list'
    return type(raw).__name__


SHAPE_TEXT = {
    'missing': '清单里根本没有 gaps 字段',
    'null': 'gaps 字段存在但值是 null (显式空值也是"没测过")',
    'flag_false:missing': '清单写着 "gaps_measured": false (gaps 字段缺失)',
    'flag_false:null': '清单写着 "gaps_measured": false (gaps 字段是 null)',
    'flag_false:list': '清单带了一个 gaps 列表, 但同时写着 "gaps_measured": false',
    'alias_key:measured': '清单带着没人读的旗标键 `measured` (唯一有效的拼法是 '
                          '"gaps_measured")',
}


def shape_text(shape: str) -> str:
    if shape in SHAPE_TEXT:
        return SHAPE_TEXT[shape]
    if shape.startswith('not_a_list:'):
        return 'gaps 字段存在但不是列表 (是 %s)' % shape.split(':', 1)[1]
    if shape.startswith('flag_false:'):
        return ('清单写着 "gaps_measured": false (gaps 字段实际是 %s)'
                % shape.split(':', 1)[1])
    if shape.startswith('bad_element:'):
        spec = shape.split(':', 1)[1]
        kind, _, index = spec.partition('@')
        return 'gaps 是列表, 但第 %s 个元素不是对象 (是 %s)' % (index, kind)
    return shape


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


def adaptation_verdict(gaps_measured: bool, new_ops: list, geometry_gate, syntax_gate,
                       header_written: bool = True) -> str:
    """The verdict, derived from what actually ran -- nothing else.

    Extracted so it can be exercised for every input instead of being an expression
    buried in the middle of main(): a status that no one can test is how `return 0`
    used to pass for "stages 1-4 were green".

      unmeasured  the manifest carries no gaps LIST: the gaps were never measured, so
                  nothing can be said about them (NOT the same as "no gaps")
      needs_work  a new_op gap is open (one kernel flavour lands, the family is free)
      unverified  no new_op gap, but a check that would be needed to say "servable"
                  did not run or did not pass (the artifact set is incomplete, a
                  geometry gate / syntax gate did not pass)
      servable    measured gaps with no new_op, geometry gate passed, syntax gate passed

    `header_written` is part of the "did the checks that 'servable' means actually
    pass" group, not a separate state: stage 2 (adapt.py) exits 2 exactly when it
    REFUSED to write config.h (unreadable layer-kind sequence, or a geometry with no
    dense-FFN width). That is an artifact set that is missing a file every downstream
    step of this pipeline assumes -- so it can never be `servable`, whatever the gap
    list says. Without this input, a spec whose layer kinds could not be read but
    whose `intermediate` was fine (adapt.py rc 2, gaps MEASURED) would be reported
    servable with no config.h on disk, which is the same "file names exist therefore
    the model runs" claim the docstring above was rewritten to kill. The default True
    keeps every caller that never saw a refusal unchanged.
    """
    if not gaps_measured:
        return 'unmeasured'
    if new_ops:
        return 'needs_work'
    if not header_written:
        return 'unverified'
    if geometry_gate is not True or syntax_gate is not True:
        return 'unverified'
    return 'servable'


def _selftest() -> int:
    """同一批 fixture 喂给本文件的判据和缺口报告读者的判据, 必须逐条一致。

    没有这个对照, "两边口径一致"就只是这句话本身。断言的是"两个实现给出同一个
    结论", 不是"实现是对的" —— 后者由 tools/gui/test_model_import.py 钉住。
    """
    sys.path.insert(0, str(HERE.parent / 'gui'))
    import model_import as reader  # noqa: E402
    cases = {
        'missing':      {'model_id': 'c', 'geometry': {'layers': 33}},
        'null':         {'model_id': 'c', 'gaps': None, 'gaps_measured': False},
        'not_a_list':   {'model_id': 'c', 'gaps': 'none'},
        'flag_false':   {'model_id': 'c', 'gaps': [], 'gaps_measured': False},
        'empty_list':   {'model_id': 'c', 'gaps': []},
        'one_new_op':   {'model_id': 'c',
                         'gaps': [{'need': 'x', 'tier': 'new_op', 'action': 'y'}]},
        'one_hook':     {'model_id': 'c',
                         'gaps': [{'need': 'x', 'tier': 'hook', 'action': 'y'}]},
        # Two shapes the replica did not have. Without them here, "two copies of one
        # rule" is checked only on the inputs both copies already agreed about:
        #   alias_key    a manifest spelling the flag `measured` (nothing reads it)
        #   bad_element  a gaps list whose element is not an object (used to be an
        #                AttributeError on the reader side, i.e. a traceback + rc 1)
        'alias_key':    {'model_id': 'c', 'gaps': [], 'measured': False},
        'bad_element':  {'model_id': 'c', 'gaps': ['attention:linear(gdn)']},
        'bad_element2': {'model_id': 'c', 'gaps': [{'need': 'x', 'tier': 'hook'}, 7]},
    }
    bad = 0
    for name, data in cases.items():
        mine_m, mine_g, mine_shape = gap_state(data)
        theirs_m, theirs_g, their_shape = reader.manifest_gap_state(data)
        same = (mine_m == theirs_m and len(mine_g) == len(theirs_g)
                and mine_shape == their_shape)
        if not same:
            bad += 1
        print('  %s %-11s adapt_all(measured=%-5s n=%d shape=%-18s) reader(measured=%-5s '
              'n=%d shape=%-18s)'
              % ('OK  ' if same else 'DIFF', name, mine_m, len(mine_g), mine_shape,
                 theirs_m, len(theirs_g), their_shape))
    print('  -- verdicts (all inputs of adaptation_verdict) --')
    for measured, newops, geo, syn, header in (
            (True, [], True, True, True), (True, [], True, False, True),
            (True, [], False, True, True), (True, [], False, False, True),
            (True, ['g'], True, True, True), (True, ['g'], False, False, True),
            (False, [], True, True, True), (False, ['g'], False, False, True),
            # header_written=False is adapt.py rc=2: the file every later stage reads
            # was refused, so no combination of the other three may print `servable`.
            # Kept in the same table so the "a status nobody can test" defect cannot
            # come back through the new input.
            (True, [], True, True, False), (True, [], False, True, False)):
        print('  measured=%-5s new_ops=%d geometry=%-5s syntax=%-5s header=%-5s -> %s'
              % (measured, len(newops), geo, syn, header,
                 adaptation_verdict(measured, newops, geo, syn, header_written=header)))
    print('== selftest %s' % ('PASS' if not bad else 'HAS %d DIFF' % bad))
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description='one-shot auto-adapt pipeline')
    ap.add_argument('input', nargs='?', default='',
                    help='model folder / config.json / archkit spec.json '
                         '(not needed with --selftest)')
    ap.add_argument('--model-id', default='')
    ap.add_argument('--workdir', default=str(OUT), help='output root (default tools/archkit/out)')
    ap.add_argument('--verify', action='store_true', help='run the gen_variant syntax gate (WSL)')
    ap.add_argument('--require-servable', action='store_true',
                    help='exit 3 unless the verdict is servable (same as the default now)')
    ap.add_argument('--allow-unfinished', action='store_true',
                    help='exit 0 even when the verdict is not servable (the old '
                         'behaviour; the exit code then stops being a verdict)')
    ap.add_argument('--skip-leaves', action='store_true')
    ap.add_argument('--selftest', action='store_true',
                    help="check this file's gap predicate against the gap-report reader")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if not args.input:
        ap.error('input is required (unless --selftest)')

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
    #
    # adapt.py's exit code is READ, not just tested for zero. It returns 2 when it
    # refused to write config.h (layer-kind sequence unreadable, or a geometry with no
    # dense-FFN width) -- and in that case it still wrote a complete manifest.json,
    # with `"gaps": null` + `"gaps_measured": false` when the reason was the layer
    # kinds. Collapsing that into `return 1` made this pipeline report "管线本身失败"
    # for an outcome it had already understood: the manifest IS readable, `gap_state`
    # below reads it, and the verdict for it is `unmeasured` -- which this file's own
    # docstring maps to rc 3, the same code tools/gui/model_import.py --gaps reports
    # for that same manifest. Two entry points calling one artifact "pipeline failed"
    # and "unknown/unmeasured" is the disagreement this pipeline exists to prevent.
    # The refusal is carried forward as `header_refused` instead, and it makes
    # `servable` unreachable below.
    adapt_rc = run([sys.executable, str(HERE / 'adapt.py'), str(spec_path), '--model-id',
                    model_id])
    if adapt_rc not in (0, 2):
        return 1
    header_refused = adapt_rc == 2
    if header_refused:
        print('  adapt.py rc=2: it REFUSED to write config.h (reason above). Its manifest '
              'is still read below -- the verdict becomes unmeasured / unverified, never '
              'servable, because the artifact set is incomplete by adapt.py\'s own report.')
    # adapt.py has no --out: it always writes into HERE/out/<model_id>. So that is where
    # its manifest must be looked for -- looking under --workdir was a second way for
    # this pipeline to lose track of its own output.
    adapted = OUT / model_id
    if workdir.resolve() != OUT.resolve():
        print('  note: adapt.py always writes to %s, whatever --workdir says (%s)'
              % (adapted, workdir))
    manifest_path, manifest = find_manifest(adapted)
    if manifest_path is None:
        print('[2/5] FAIL: no manifest in %s (looked for %s) -- cannot report gaps, so '
              'this run cannot claim an adaptation either'
              % (adapted, ' / '.join(MANIFEST_NAMES)))
        return 1
    if manifest is None:
        return 1
    # Measured vs empty. A missing gaps list is NOT "no gaps" -- see the docstring.
    gaps_measured, gaps, gap_shape = gap_state(manifest)
    new_ops = [g for g in gaps if g.get('tier') == 'new_op']
    hooks = [g for g in gaps if g.get('tier') != 'new_op']
    # "adapt ok: config.h + ..." claimed a file that stage 2 may have refused; the
    # header is now named only when it was actually written.
    header_state = 'config.h' if not header_refused else 'NO config.h (refused)'
    if gaps_measured:
        print('[2/5] adapt ok: %s + %d gaps (%d hooks, %d new_op)'
              % (header_state, len(gaps), len(hooks), len(new_ops)))
    else:
        print('[2/5] adapt ok: %s, but %s carries no gaps list (%s) -> gaps NOT '
              'MEASURED (which is not the same as no gaps)'
              % (header_state, manifest_path.name, shape_text(gap_shape)))

    # 2.5) geometry gate: the engine's gqa dispatch table must cover this model
    #      (q, kv, head_dim) -- this is the ONE engine-side check in this pipeline.
    geometry_gate = None
    geo_gate = HERE / 'check_geometry.py'
    if geo_gate.exists():
        print('[2.5/5] geometry gate...')
        if run([sys.executable, str(geo_gate), str(spec_path), '--repo',
                str(HERE.parent.parent)]) != 0:
            print('geometry gate FAILED: 模型注意力几何未在引擎分派表注册')
            return 1
        geometry_gate = True
        print('[2.5/5] geometry gate ok')
    else:
        geometry_gate = False
        print('[2.5/5] geometry gate SKIPPED (check_geometry.py missing) -- nothing here '
              'has checked the engine, so this run cannot claim servable')

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

    # 5) verify: the gen_variant syntax gate (compiles generated leaves against a stub
    #    header -- NOT this model's config.h against the engine).
    syntax_gate = None
    if args.verify:
        v3 = HERE / 'verify_v3_gen.py'
        if not v3.exists():
            syntax_gate = False
            print('[5/5] verify SKIPPED (verify_v3_gen.py missing)')
        else:
            print('[5/5] verify...')
            if run([sys.executable, str(v3)]) != 0:
                print('verify FAILED')
                return 1
            syntax_gate = True
            print('verify OK')
    else:
        syntax_gate = False
        print('[5/5] verify not requested (--verify) -> generated code was never compiled')

    # ---- verdict: derived from the checks that actually ran --------------------
    verdict = adaptation_verdict(gaps_measured, new_ops, geometry_gate, syntax_gate,
                                header_written=not header_refused)

    missing = []
    if not gaps_measured:
        missing.append('清单没有 gaps 列表 (%s) —— 缺口未测量' % shape_text(gap_shape))
    if header_refused:
        missing.append('adapt.py 拒写 config.h (rc=2, 原因是层型序列读不出来或几何没有 '
                       'dense FFN 宽度) —— 产物集不完整, 所以不可能是 servable')
    if geometry_gate is not True:
        missing.append('引擎侧几何门没跑 (或没通过)')
    if syntax_gate is not True:
        missing.append('生成的代码从未编译过 (要加 --verify)')

    # report
    report = adapted / 'adapt_report.md'
    lines = [
        '# 适配报告: %s' % model_id,
        '',
        '- 结论 (verdict): **%s**' % verdict,
        '- 清单: %s (gaps: %s)' % (manifest_path.name,
                                   'measured' if gaps_measured else 'NOT MEASURED'),
        '- 头文件: %s' % (header_state if not header_refused
                          else '未生成 (adapt.py rc=2 拒写; 原因见上)'),
        '- 完成: spec -> %s -> manifest -> 口味叶子 -> bindings/package/recipe 骨架'
        % ('config.h' if not header_refused else '(config.h 被拒)'),
        '- new_op 缺口 (%d): 每个=需要一次新内核口味落地, 之后同族免费' % len(new_ops),
    ]
    for g in new_ops:
        lines.append('  - [%s] %s: %s' % (g.get('tier'), g.get('need'), g.get('action')))
    for g in hooks:
        lines.append('  - [%s] %s' % (g.get('tier'), g.get('need')))
    lines.append('')
    if verdict == 'servable':
        lines.append('验收判据: %s' % '适配完成, 可 serve')
        lines.append('  依据: 清单测过缺口且无 new_op; 几何门通过 (引擎分派表覆盖本模型'
                     '几何); 生成代码语法门通过。')
        lines.append('  未经: 真实加载/推理 —— "引擎能吃"的最高证据是跑一次, 本管线没有跑。')
    else:
        lines.append('验收判据: **不构成可 serve** —— 缺: %s'
                     % ('; '.join(missing) if missing
                        else '%d 个 new_op 缺口 (落地后同族免费)' % len(new_ops)))
        lines.append('  (判据由实跑过的检查决定: 只有清单测过缺口、几何门真过、生成代码'
                     '真编译过, 才写"可 serve"。文件都生成了 != 引擎能吃。)')
    report.write_text('\n'.join(lines), encoding='utf-8')
    print('== verdict: %s' % verdict)
    print('== done -> %s' % report)
    if args.allow_unfinished:
        print('== --allow-unfinished: verdict %s -> exit 0 anyway (退出码不再是结论)'
              % verdict)
        return 0
    if verdict != 'servable':
        print('== verdict is %s (not servable) -> exit 3' % verdict)
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
