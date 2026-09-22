#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""serve_gui_selftest.py - headless acceptance test for the NInfer serve console.

Runs with plain Python 3 and local HTTP only.  It never starts the engine, the
GPU or nvcc, and it never calls /api/start: the smoke test only reads pages and
switches the language.

What is proved, with the raw evidence printed for every check:

  1. placeholders   every @@...@@ token in the page template is substituted
                    before the bytes reach the browser: the rendered page has
                    none left (zh and en), and every @@t.<key>@@ resolves in
                    both languages;
  2. cmdline parity a request built from the untouched page yields a command
                    line byte-identical to the pre-refactor baseline, which is
                    read back out of git (HEAD:tools/gui/serve_gui.py);
  3. advanced params a filled field is appended in registry order, and the
                    order does not depend on the request's key order;
  4. flag typing    bool flags carry no value, int/float flags carry one,
                    empty values emit nothing at all;
  5. HTTP smoke     a real server on a free port: GET /, /?lang=en, /?lang=zh,
                    checked for leftover placeholders, the registry count on
                    screen, and zh vs en really being different text;
  6. lang switch    POST /api/lang goes through gui_i18n.lang_from_request,
                    answers with the language, persists it (gui_lang.txt) and
                    changes what the next plain GET / returns.

Usage:  python serve_gui_selftest.py
Exit code 0 = everything passed; 1 = at least one check failed.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import random
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
LANG_FILE = HERE / 'gui_lang.txt'
BASELINE_REF = 'HEAD:tools/gui/serve_gui.py'

# Snapshot the persisted language before anything imports gui_i18n (the import
# itself rewrites the file), so the test can hand the developer's setting back.
_ORIG_LANG_BYTES = LANG_FILE.read_bytes() if LANG_FILE.exists() else None

sys.path.insert(0, str(HERE))
import gui_i18n          # noqa: E402
import serve_gui as sg   # noqa: E402

try:
    sys.stdout.reconfigure(errors='backslashreplace')
except Exception:        # older/odd stdout - plain printing still works
    pass

# Frozen by hand from the baseline (git HEAD:tools/gui/serve_gui.py) right
# before the refactor: an independent witness next to the git-derived one.
FROZEN_DEFAULT = [
    '/home/user/ninfer-fusion/build/apps/ninfer-serve',
    '/home/user/models/qwen3_8_27b_nvfp4.ninfer',
    '--port', '8000', '--kv-dtype', 'nvfp4', '--max-context', '32768',
]

# Slider ranges, mirrored from the page markup.
SLIDER_DOMAIN = {
    'kv': list(range(5)), 'ctx': list(range(3, 9)), 'spec': list(range(4)),
    'draft': list(range(1, 6)), 'conc': list(range(1, 5)), 'vis': [0, 1],
    'cold': list(range(4)), 'hotwin': list(range(9)), 'graph': [0, 1],
}
KVLAYER_SAMPLES = ['', '0-15:rk4v4,32-63:iso4e', '   ', ' hot=bf16,tail=fp16 ']
MODEL_SAMPLES = [
    '/home/user/models/qwen3_8_27b_nvfp4.ninfer',
    '/models/x.ninfer',
    'weird model name.ninfer',
]

SAMPLE_COUNT = 20000

# The language switcher names every language in its own script (the button that
# switches to Chinese reads "Chinese" on the English page).  That single label is
# the only CJK allowed to survive on the English page.
LANG_BUTTON_SELF_NAME = '\u4e2d\u6587'


# ---------------------------------------------------------------------------
# tiny check harness
# ---------------------------------------------------------------------------
class Report:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []

    def check(self, ok: bool, name: str, evidence: str = '') -> bool:
        (self.passed if ok else self.failed).append(name)
        print('%-4s %s' % ('PASS' if ok else 'FAIL', name))
        for line in str(evidence).rstrip('\n').splitlines():
            print('       %s' % line)
        print()
        return ok

    def done(self) -> int:
        print('=' * 74)
        print('%d passed, %d failed' % (len(self.passed), len(self.failed)))
        for name in self.failed:
            print('   FAILED: %s' % name)
        print('VERDICT:', 'PASS' if not self.failed else 'FAIL')
        return 0 if not self.failed else 1


R = Report()


def snippet(text: str, needle: str, pad: int = 70) -> str:
    """A +/-pad window around needle, for pasting a rendered fragment."""
    i = text.find(needle)
    if i < 0:
        return '(not found: %r)' % needle
    lo, hi = max(0, i - pad), min(len(text), i + len(needle) + pad)
    return '%s...%s' % ('<<' if lo else '[', '%s...>>' % text[lo:hi] if hi < len(text)
                        else '%s]' % text[lo:hi])


def strip_style_and_comments(html: str) -> str:
    body = re.sub(r'<style>.*?</style>', '', html, flags=re.S)
    body = re.sub(r'<!--.*?-->', '', body, flags=re.S)
    return body


def cjk_contexts(text: str, pad: int = 30) -> list[str]:
    return re.findall(r'.{0,%d}[\u4e00-\u9fff]+.{0,%d}' % (pad, pad), text)


def count_badge(html: str) -> str:
    """The <span class="count"> text of the all-parameters summary, as served."""
    m = re.search(r'<span class="count">(.*?)</span>', html, re.S)
    return m.group(1).strip() if m else ''


# ---------------------------------------------------------------------------
# 0. baseline from git
# ---------------------------------------------------------------------------
def load_baseline():
    """(module, note) for the pre-refactor serve_gui.py, or (None, reason)."""
    try:
        out = subprocess.run(['git', '-C', str(REPO), 'show', BASELINE_REF],
                             capture_output=True, text=True, check=True)
    except Exception as e:
        return None, 'git could not read %s: %s' % (BASELINE_REF, e)
    d = tempfile.mkdtemp(prefix='serve_gui_baseline_')
    p = pathlib.Path(d) / 'baseline_serve_gui.py'
    p.write_text(out.stdout, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('baseline_serve_gui', str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, '%s -> %s (%d bytes)' % (BASELINE_REF, p.name, len(out.stdout))


# ---------------------------------------------------------------------------
# 1. placeholders
# ---------------------------------------------------------------------------
def check_placeholders() -> None:
    print('--- 1. placeholder substitution ---')
    used = sorted(set(sg.T_PLACEHOLDER.findall(sg.PAGE)))
    raw = sorted(set(m.group(1) for m in sg.PLACEHOLDER.finditer(sg.PAGE)
                     if not m.group(1).startswith('t.')))
    R.check(raw == sorted(sg.RAW_PLACEHOLDERS),
            'template placeholders are exactly the i18n keys plus the 4 raw ones',
            'text keys (%d): %s\nraw keys      : %s\nraw in PAGE   : %s\nRAW_PLACEHOLDERS: %s'
            % (len(used), ', '.join(used), raw, raw, list(sg.RAW_PLACEHOLDERS)))

    holes = [k for k in used for lg in ('zh', 'en') if not gui_i18n.has('serve.' + k, lg)]
    R.check(not holes, 'every @@t.<key>@@ has a zh and an en entry (no key reaches the page)',
            'checked %d keys x2 languages; missing: %s' % (len(used), holes or 'none'))

    frags = {}
    for lang in ('zh', 'en'):
        gui_i18n.set_lang(lang)
        page = sg.render_page('/?lang=%s' % lang)
        frags[lang] = page
        left = sorted(set(m.group(0) for m in sg.PLACEHOLDER.finditer(page)))
        R.check(not left, 'rendered page [%s] has no @@ left' % lang,
                'page bytes      : %d\nleftover @@     : %s\n@@ occurrences  : %d'
                % (len(page), left or 'none', page.count('@@')))

    z, e = frags['zh'], frags['en']
    R.check(z != e and '<html lang="zh">' in z and '<html lang="en">' in e,
            'the two languages render different pages with the right lang attribute',
            'zh html lang    : %s' % re.search(r'<html lang="[^"]*"', z).group(0) +
            '\nen html lang    : %s' % re.search(r'<html lang="[^"]*"', e).group(0) +
            '\nbytes           : zh %d / en %d' % (len(z), len(e)))

    print('    side-by-side fragments (same label, two languages):')
    print('      %-18s | %-34s | %s' % ('key', 'zh', 'en'))
    table = gui_i18n.STRINGS
    for key in ('title', 'badge_idle', 'lbl_model', 'lbl_kv', 'btn_start',
                'adv_summary', 'log_ready'):
        row = table['serve.' + key]
        print('      %-18s | %-34s | %s' % ('serve.' + key, row['zh'], row['en']))
    print('      page markup served for the same region:')
    for key, needle in (('title', '<title>'), ('lbl_kv', 'id="kv"'),
                        ('btn_start', 'id="start"'), ('adv_summary', '<summary>')):
        print('        %s zh: %s' % (key.ljust(11), snippet(z, needle, 40)))
        print('        %s en: %s' % (''.ljust(11), snippet(e, needle, 40)))
    print()


# ---------------------------------------------------------------------------
# 2. command-line parity with the pre-refactor baseline
# ---------------------------------------------------------------------------
def check_parity() -> None:
    print('--- 2. default command line vs the pre-refactor baseline ---')
    base_mod, note = load_baseline()
    print('    baseline: %s' % note)

    page_req = dict(sg.DEFAULT_REQ)
    cmd = sg.build_cmd(page_req)
    R.check(cmd == FROZEN_DEFAULT,
            'untouched page -> the frozen pre-refactor command line, element by element',
            'got     : %s\nfrozen  : %s\nidentical: %s (len %d/%d)'
            % (json.dumps(cmd), json.dumps(FROZEN_DEFAULT), cmd == FROZEN_DEFAULT,
               len(cmd), len(FROZEN_DEFAULT)))

    if base_mod is None:
        R.check(False, 'git baseline comparison', note)
        return

    named = {
        'page defaults': dict(sg.DEFAULT_REQ),
        'page defaults + adv={}': dict(sg.DEFAULT_REQ, adv={}),
        'all sliders up': dict(sg.DEFAULT_REQ, kv=0, ctx=8, spec=3, draft=5, conc=4,
                               vis=1, cold=3, hotwin=8, kvlayer='0-15:rk4v4', graph=0),
        'kv=4 / cold=window / hot=0': dict(sg.DEFAULT_REQ, kv=4, cold=1, hotwin=0),
        'ctx=min / graph off': dict(sg.DEFAULT_REQ, ctx=3, graph=0),
        'spec=mtp / draft=1': dict(sg.DEFAULT_REQ, spec=1, draft=1),
    }
    bad = [(n, json.dumps(base_mod.build_cmd(r)), json.dumps(sg.build_cmd(r)))
           for n, r in named.items() if base_mod.build_cmd(r) != sg.build_cmd(r)]
    R.check(not bad, '%d named requests match the baseline byte for byte' % len(named),
            '\n'.join('%s\n   baseline: %s\n   current : %s' % b for b in bad) or
            '\n'.join('%-28s %s' % (n, json.dumps(sg.build_cmd(r)))
                      for n, r in named.items()))

    rng = random.Random(20260910)
    mismatches = []
    for i in range(SAMPLE_COUNT):
        req = {'model': rng.choice(MODEL_SAMPLES), 'port': str(rng.randint(1024, 65535))}
        for k, dom in SLIDER_DOMAIN.items():
            req[k] = rng.choice(dom)
        req['kvlayer'] = rng.choice(KVLAYER_SAMPLES)
        req['adv'] = {}
        want, got = base_mod.build_cmd(req), sg.build_cmd(req)
        if want != got:
            mismatches.append((req, want, got))
    R.check(not mismatches,
            '%d randomized slider requests (adv={}) match the baseline byte for byte'
            % SAMPLE_COUNT,
            'mismatches: %d' % len(mismatches) if not mismatches else
            json.dumps(mismatches[:3], ensure_ascii=False))

    # Empty values never reach the command line -> the untouched page contract.
    empty_bad = []
    for flag in sg.ADV_ORDER:
        for val in ('', '   ', None, False):
            req = dict(sg.DEFAULT_REQ, adv={flag: val})
            if sg.build_cmd(req) != base_mod.build_cmd(dict(sg.DEFAULT_REQ)):
                empty_bad.append((flag, val, sg.build_cmd(req)))
    R.check(not empty_bad,
            'every advanced flag with an empty value leaves the baseline command untouched',
            'flags x 4 empty forms = %d cases; violations: %s'
            % (len(sg.ADV_ORDER) * 4, empty_bad[:3] or 'none'))
    print()


# ---------------------------------------------------------------------------
# 3. advanced parameters land in a stable order
# ---------------------------------------------------------------------------
def check_advanced_order() -> None:
    print('--- 3. advanced parameters: registry order, no key-order dependence ---')
    base = dict(sg.DEFAULT_REQ)
    filled = {'--temperature': '0.7', '--seed': '42', '--no-prefix-reuse': True}
    cmd = sg.build_cmd(dict(base, adv=dict(filled)))
    rev = sg.build_cmd(dict(base, adv=dict(reversed(list(filled.items())))))
    idx = {f: cmd.index(f) for f in filled}
    order_ok = idx['--seed'] < idx['--temperature'] < idx['--no-prefix-reuse']
    tail = cmd[cmd.index('--max-context') + 2:]

    R.check(all(f in cmd for f in filled) and order_ok,
            'filled flags appear in registry order (--seed, --temperature, --no-prefix-reuse)',
            'registry order : %s' % [f for f in sg.ADV_ORDER if f in filled] +
            '\npositions      : %s' % idx +
            '\ncommand tail   : %s' % json.dumps(tail, ensure_ascii=False) +
            '\nvalues         : --seed 42, --temperature 0.7, --no-prefix-reuse (bool, bare)')
    R.check(cmd == rev,
            'the same three flags sent in reverse key order give the same command line',
            'forward : %s\nreversed: %s\nidentical: %s'
            % (json.dumps(tail, ensure_ascii=False), json.dumps(rev[rev.index('--max-context') + 2:],
                                                               ensure_ascii=False), cmd == rev))

    every = sg.build_cmd(dict(base, adv={f: 'x' for f in sg.ADV_ORDER
                                         if f not in sg.BASIC_FLAGS and f not in sg.GUI_SKIP}))
    got = [f for f in every if f in sg.ADV_ORDER and f not in sg.BASIC_FLAGS]
    want = [f for f in sg.ADV_ORDER if f not in sg.BASIC_FLAGS and f not in sg.GUI_SKIP]
    R.check(got == want,
            'all %d unlocked registry flags, all filled at once, come out in registry order'
            % len(want),
            'emitted order == ADV_ORDER (filtered): %s\nfirst 8: %s' % (got == want, got[:8]))

    unknown = sg.build_cmd(dict(base, adv={'--zzz-unknown': 'v', '--temperature': '0.7'}))
    R.check(unknown[-2:] == ['--zzz-unknown', 'v'] and '--temperature' in unknown,
            'a flag outside the registry is still passed through, after the known ones',
            'tail: %s' % json.dumps(unknown[-6:], ensure_ascii=False))
    print()


# ---------------------------------------------------------------------------
# 4. typing: bool bare, int/float with value, empty never emitted
# ---------------------------------------------------------------------------
def check_typing() -> None:
    print('--- 4. flag typing (bool bare / int+float valued / empty silent) ---')

    def adv_args(**adv):
        return sg._adv_args({'adv': adv})

    cases = [
        ('bool True  -> bare flag', adv_args(**{'--no-prefix-reuse': True}),
         ['--no-prefix-reuse']),
        ('bool False -> nothing', adv_args(**{'--no-prefix-reuse': False}), []),
        ('int 42     -> flag + value', adv_args(**{'--seed': 42}), ['--seed', '42']),
        ('int str    -> flag + value', adv_args(**{'--seed': '42'}), ['--seed', '42']),
        ('int 0      -> flag + value', adv_args(**{'--seed': 0}), ['--seed', '0']),
        ('float str  -> flag + value', adv_args(**{'--temperature': '0.7'}),
         ['--temperature', '0.7']),
        ('float num  -> flag + value', adv_args(**{'--temperature': 0.7}),
         ['--temperature', '0.7']),
        ('empty str  -> nothing', adv_args(**{'--seed': ''}), []),
        ('spaces     -> nothing', adv_args(**{'--seed': '   '}), []),
        ('None       -> nothing', adv_args(**{'--seed': None}), []),
        ('locked flag-> nothing', adv_args(**{'--no-cuda-graph': True}), []),
        ('--help     -> nothing', adv_args(**{'--help': True}), []),
    ]
    bad = [c for c in cases if c[1] != c[2]]
    R.check(not bad, 'bool/int/float/empty behaviour on %d cases' % len(cases),
            '\n'.join('%-28s -> %s' % (c[0], json.dumps(c[1], ensure_ascii=False))
                      for c in cases) +
            ('\nMISMATCH: %s' % bad if bad else ''))

    # Walk the whole command line: nothing may follow a bool flag.
    bool_flags = {it['flag'] for g in sg.REGISTRY['group_order']
                  for it in sg.group_items(sg.REGISTRY, g)
                  if it.get('type') == 'bool' and it.get('flag')}
    req = dict(sg.DEFAULT_REQ, adv={f: True for f in bool_flags
                                    if f not in sg.BASIC_FLAGS and f not in sg.GUI_SKIP},
               kv=3)
    cmd = sg.build_cmd(req)
    offenders = [cmd[i] for i in range(len(cmd) - 1)
                 if cmd[i] in bool_flags and not cmd[i + 1].startswith('--')]
    emitted = [f for f in cmd if f in bool_flags]
    R.check(emitted and not offenders,
            'all %d bool flags emitted bare (%d of %d renderable)'
            % (len(emitted), len(emitted), len(bool_flags)),
            'bool flags in the command: %s\nflags followed by a value: %s'
            % (json.dumps(emitted, ensure_ascii=False), offenders or 'none'))

    int_flags = {it['flag']: it.get('type') for g in sg.REGISTRY['group_order']
                 for it in sg.group_items(sg.REGISTRY, g)
                 if it.get('type') in ('int', 'float')}
    probe = {'--seed': '7', '--temperature': '0.25', '--top-k': '3',
             '--top-p': '0.9', '--prefill-chunk': '512'}
    built = sg.build_cmd(dict(sg.DEFAULT_REQ, adv=probe))
    int_ok = all(built[built.index(f) + 1] == v for f, v in probe.items())
    R.check(int_ok, 'int/float flags carry their value (types from the registry)',
            'registry types  : %s' % {f: int_flags.get(f) for f in probe} +
            '\ncommand tail    : %s' % json.dumps(built[built.index('--max-context') + 2:],
                                                 ensure_ascii=False))
    print()


# ---------------------------------------------------------------------------
# 5./6. HTTP smoke + language switch
# ---------------------------------------------------------------------------
def free_port() -> int:
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def http(url: str, body=None, timeout: float = 10.0):
    data = json.dumps(body).encode('utf-8') if body is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={'Content-Type': 'application/json'} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, dict(r.headers), r.read().decode('utf-8')


def check_http() -> None:
    print('--- 5./6. headless HTTP smoke + language switch ---')
    port = free_port()
    url = 'http://127.0.0.1:%d' % port
    print('    free port picked: %d (never 8077/8078)\n' % port)
    # render_page() above has been switching the language, which persists; pin it
    # so the first plain GET is deterministic.  The original value is restored
    # in main()'s finally block.
    gui_i18n.set_lang('zh')
    print('    gui_lang.txt pinned to %r for the smoke run\n' % gui_i18n.current())
    proc = subprocess.Popen([sys.executable, '-u', str(HERE / 'serve_gui.py'), str(port)],
                            cwd=str(HERE), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        up = False
        for _ in range(100):
            if proc.poll() is not None:
                break
            try:
                http(url + '/api/state', timeout=1.0)
                up = True
                break
            except Exception:
                time.sleep(0.1)
        if not R.check(up, 'serve_gui.py answers on %s (child pid %s)' % (url, proc.pid),
                       'server not reachable' if not up else
                       'GET /api/state -> 200 (readiness probe); no engine started'):
            return

        st, hdr, zh0 = http(url + '/')
        R.check(st == 200 and 'text/html' in hdr.get('Content-Type', '')
                and '<html lang="zh">' in zh0,
                'GET / (no query) -> 200 text/html in the persisted language',
                'status: %s\ncontent-type: %s\nbytes: %d\npage: %s'
                % (st, hdr.get('Content-Type'), len(zh0),
                   re.search(r'<html lang="[^"]*"', zh0).group(0)))

        st, _, en = http(url + '/?lang=en')
        st2, _, zh = http(url + '/?lang=zh')
        R.check(zh0.count('@@') == en.count('@@') == zh.count('@@') == 0,
                'no @@ reaches the browser on / , /?lang=en , /?lang=zh',
                'leftover @@: / -> %d , /?lang=en -> %d , /?lang=zh -> %d'
                % (zh0.count('@@'), en.count('@@'), zh.count('@@')))

        R.check('<html lang="en">' in en and '<html lang="zh">' in zh and en != zh,
                'the served page follows the ?lang= query',
                'en page: %s\nzh page: %s\nbytes  : en %d / zh %d / differ %s'
                % (re.search(r'<html lang="[^"]*"', en).group(0),
                   re.search(r'<html lang="[^"]*"', zh).group(0),
                   len(en), len(zh), en != zh))

        st, _, plain = http(url + '/')
        R.check(plain == zh, 'a plain GET right after ?lang=zh serves the same bytes '
                             '(the page is a pure function of the language)',
                'plain GET / bytes: %d\n?lang=zh bytes   : %d\nidentical: %s'
                % (len(plain), len(zh), plain == zh))

        pairs = [('title', 'serve.title'), ('badge_idle', 'serve.badge_idle'),
                 ('lbl_kv', 'serve.lbl_kv'), ('btn_start', 'serve.btn_start'),
                 ('adv_summary', 'serve.adv_summary'), ('adv_hint', 'serve.adv_hint'),
                 ('language', 'serve.language'), ('lbl_hotwin', 'serve.lbl_hotwin'),
                 ('ph_kvlayer', 'serve.ph_kvlayer'), ('log_ready', 'serve.log_ready')]
        table = gui_i18n.STRINGS
        diff_pairs = [(n, table[k]['zh'], table[k]['en']) for n, k in pairs
                      if table[k]['zh'] != table[k]['en']]
        bad = [(n, z, e) for n, z, e in diff_pairs
               if not (z in zh and e in en and z not in en)]
        R.check(len(diff_pairs) == len(pairs) and not bad,
                'the same label really is different text in zh and en (%d pairs)' % len(pairs),
                '\n'.join('%-14s zh %-24s en %s' % (n, z, e) for n, z, e in diff_pairs) +
                ('\nMISMATCH: %s' % bad if bad else ''))

        body_en = strip_style_and_comments(en)
        runs = re.findall(r'[\u4e00-\u9fff]+', body_en)
        # The one deliberate exception: a language switcher shows each language
        # under its own name, so the button that switches TO Chinese stays
        # "Chinese" in English.  Anything else is a leak.
        exceptions = {LANG_BUTTON_SELF_NAME}
        leaked = sorted(set(runs) - exceptions)
        R.check(not leaked,
                'no Chinese user-visible text on the English page (style and HTML '
                'comments stripped; only the "%s" language button is expected)'
                % LANG_BUTTON_SELF_NAME,
                'CJK runs found : %s\nnot expected   : %s\ncontexts       :\n%s'
                % (runs or 'none', leaked or 'none',
                   '\n'.join('   | %s' % h for h in cjk_contexts(body_en)) or '   (none)'))
        bar_en = snippet(body_en, 'class="langbar"', 70)
        bar_zh = snippet(strip_style_and_comments(zh), 'class="langbar"', 70)
        R.check(LANG_BUTTON_SELF_NAME in body_en and 'Language' in body_en
                and 'lang_zh' in en and 'lang_en' in en,
                'the language bar itself is translated and still offers both languages',
                'en: %s\nzh: %s' % (bar_en, bar_zh))

        total = sg.REGISTRY['total']
        controls = en.count('data-flag="')
        span_en, span_zh = count_badge(en), count_badge(zh)
        R.check(str(total) in en and str(total) in zh and controls == total
                and span_en and str(total) in span_en and 'registry' in span_en
                and span_zh and str(total) in span_zh,
                'the registry count is on screen and every entry got a control (%d)' % total,
                'en badge         : %s\nzh badge         : %s\n'
                'data-flag inputs : %d\nregistry total   : %s'
                % (span_en, span_zh, controls, total))

        st, _, post = http(url + '/api/lang', {'lang': 'en'})
        saved = LANG_FILE.read_text(encoding='utf-8')
        st2, _, after = http(url + '/')
        R.check(json.loads(post).get('lang') == 'en' and saved == 'en'
                and '<html lang="en">' in after,
                'POST /api/lang {en} -> answers en, persists to gui_lang.txt, next GET / is en',
                'response body      : %s\nbody lang field    : %s\ngui_lang.txt       : %r\n'
                'GET / (no ?lang)   : %s'
                % (post, json.loads(post).get('lang'), saved,
                   re.search(r'<html lang="[^"]*"', after).group(0)))

        st, _, post_zh = http(url + '/api/lang', {'lang': 'zh'})
        st2, _, back = http(url + '/')
        R.check(json.loads(post_zh).get('lang') == 'zh'
                and LANG_FILE.read_text(encoding='utf-8') == 'zh'
                and '<html lang="zh">' in back,
                'POST /api/lang {zh} switches back (round trip)',
                'response body: %s\ngui_lang.txt : %r\nGET /        : %s'
                % (post_zh, LANG_FILE.read_text(encoding='utf-8'),
                   re.search(r'<html lang="[^"]*"', back).group(0)))

        st, _, junk = http(url + '/?lang=xx')
        R.check(st == 200 and '<html lang="zh">' in junk,
                'an unknown language falls back to zh instead of breaking the page',
                'GET /?lang=xx -> %s, %s' % (st, re.search(r'<html lang="[^"]*"', junk).group(0)))

        src = (HERE / 'serve_gui.py').read_text(encoding='utf-8')
        post_route = src.split("elif route == '/api/lang':", 1)
        R.check(len(post_route) == 2 and 'lang_from_request' in post_route[1][:200],
                "/api/lang is wired to gui_i18n.lang_from_request in the source",
                'handler body: %s' % (post_route[1][:120].strip().replace('\n', ' ')
                                      if len(post_route) == 2 else 'route not found'))
        print('    raw fragments served over HTTP:')
        print('      en <title>: %s' % snippet(en, '<title>', 30))
        print('      zh <title>: %s' % snippet(zh, '<title>', 30))
        print('      en summary: %s' % snippet(en, '<summary>', 40))
        print('      zh summary: %s' % snippet(zh, '<summary>', 40))
        print('      en POST /api/lang body: %s' % post)
        print()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        out = proc.stdout.read() if proc.stdout else ''
        if out.strip():
            print('    child stdout/stderr: %s' % out.strip()[:400])
        print('    child process stopped (no engine was ever launched)\n')


# ---------------------------------------------------------------------------
def main() -> int:
    print('serve_gui self-test  |  gui dir: %s' % HERE)
    print('python %s  |  registry: %s' % (sys.version.split()[0], sg.REGISTRY_PATH))
    print('registry entries: %s (source: %s)' % (sg.REGISTRY.get('total'),
                                                sg.REGISTRY.get('source') or '?'))
    print('persisted language at start: %r\n' % (LANG_FILE.read_text(encoding='utf-8')
                                                 if LANG_FILE.exists() else None))
    try:
        check_placeholders()
        check_parity()
        check_advanced_order()
        check_typing()
        check_http()
    finally:
        if _ORIG_LANG_BYTES is not None:
            LANG_FILE.write_bytes(_ORIG_LANG_BYTES)
            restored = LANG_FILE.read_text(encoding='utf-8')
        else:
            LANG_FILE.unlink(missing_ok=True)
            restored = '(file removed)'
        print('gui_lang.txt restored to: %r' % restored)
    return R.done()


if __name__ == '__main__':
    raise SystemExit(main())
