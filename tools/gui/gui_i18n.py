# -*- coding: utf-8 -*-
"""gui_i18n.py — 界面翻译核心（zh/en 双语，全功能覆盖）。

设计约束（写给后续维护者也是写给检查器）：
  * 唯一入口 `t(key, **kw)`；缺 key 时**不静默返回 key**，而是返回中文 fallback 并记录，
    由 `gui_i18n_check.py` 在 CI/自检里把它变成失败（否则"翻译完成"无从证明）。
  * 语言表分散在 `i18n_*.py` 里，每个模块导出 `STRINGS = {key: {'zh': ..., 'en': ...}}`，
    本文件自动合并 —— 这样多个作者可以互不冲突地各写一份表。
  * 语言选择持久化：环境变量 NINFER_GUI_LANG > 上次选择（源码树 = 本文件旁的
    `gui_lang.txt`；打包版 = `%LOCALAPPDATA%\\NInfer\\gui_lang.txt`，因为打包版解包目录
    `_MEIPASS` 每次运行都重建，写在那里等于没写）> 随包带来的那份 > 默认 zh。
  * 语言切换要能影响**已渲染**页面：前端约定 `?lang=` 与 `localStorage['ninferLang']`，
    `lang_from_request(path)` 供各 GUI 的 HTTP 处理器使用。

用法：
    from gui_i18n import t, set_lang, current, lang_from_request, missing_keys
    t('serve.title')                      -> '服务控制台' / 'Serve console'
    t('serve.port_hint', port=8077)       -> 带占位符的句子
"""
from __future__ import annotations

import importlib
import os
import pathlib
import sys
from typing import Any

_HERE = pathlib.Path(__file__).resolve().parent


def _lang_file() -> pathlib.Path:
    """Where the chosen language is remembered.

    In a source tree that is the file next to this module (unchanged).  A frozen
    build imports this module from the PYZ, so ``__file__`` resolves inside
    ``sys._MEIPASS`` -- a directory the onefile bootloader creates for each run and
    deletes on exit.  Writing the preference there meant it was gone by the next
    start: measured 2026-09-14, the packaged NInfer.exe wrote
    ``%TEMP%\\_MEI420762\\gui_lang.txt`` during a run, and the next run came up in the
    default language again, i.e. the user's choice never survived a restart.  Frozen
    builds therefore keep it with the rest of the per-user state
    (``%LOCALAPPDATA%\\NInfer``), or beside the exe when LOCALAPPDATA is unset.
    ``NINFER_GUI_LANG_FILE`` names the file explicitly (test hook, wins over both).
    """
    override = os.environ.get('NINFER_GUI_LANG_FILE')
    if override:
        return pathlib.Path(override)
    if getattr(sys, 'frozen', False):
        base = os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_DATA_HOME')
        if base:
            return pathlib.Path(base) / 'NInfer' / 'gui_lang.txt'
        return pathlib.Path(os.path.abspath(sys.executable)).parent / 'gui_lang.txt'
    return _HERE / 'gui_lang.txt'


_LANG_FILE = _lang_file()
#: The table the packager SHIPPED, at the path the spec's datas put it (a frozen bundle
#: unpacks the repository's tools/gui tree there).  Used only to pick the initial
#: language when the user has not chosen one yet, so that a packaged build starts in
#: the same language as the source tree it was built from (measured before this: the
#: source tree came up 'en' from tools/gui/gui_lang.txt while the exe came up 'zh').
_SHIPPED_LANG_FILE = _HERE / 'tools' / 'gui' / 'gui_lang.txt'

_LANGS = ('zh', 'en')
_lang = 'zh'
_missing: set[str] = set()


def _load_strings() -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for p in sorted(_HERE.glob('i18n_*.py')):
        if p.name == 'gui_i18n.py':
            continue
        mod_name = p.stem
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:  # 单表坏掉不应让整个界面崩
            print('[gui_i18n] WARN cannot import %s: %s' % (mod_name, e))
            continue
        table = getattr(mod, 'STRINGS', None)
        if not isinstance(table, dict):
            continue
        for k, v in table.items():
            if not isinstance(v, dict):
                continue
            merged.setdefault(k, {}).update({lg: v[lg] for lg in _LANGS if lg in v})
    return merged


STRINGS = _load_strings()


def available() -> tuple[str, ...]:
    return tuple(lg for lg in _LANGS if any(lg in v for v in STRINGS.values()))


def set_lang(lang: str | None) -> str:
    """设置当前语言；未知值退回 zh。写盘失败不影响运行。"""
    global _lang
    lang = (lang or '').strip().lower()[:2]
    _lang = lang if lang in _LANGS else 'zh'
    try:
        _LANG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LANG_FILE.write_text(_lang, encoding='utf-8')
    except OSError:
        pass
    return _lang


def _remembered_lang() -> str:
    """用户上次选的语言；没选过就用打包/仓库里带来的那份，最后才默认 zh。

    两个候选都必须真的读出 zh/en 才算数：坏值不能让界面卡在一个不存在的语言上。
    """
    for p in (_LANG_FILE, _SHIPPED_LANG_FILE):
        try:
            v = p.read_text(encoding='utf-8').strip().lower()[:2]
        except OSError:
            continue
        if v in _LANGS:
            return v
    return 'zh'


# 模块导入时确定初始语言（env > 持久化的选择 > 随包带来的那份 > zh）
set_lang(os.environ.get('NINFER_GUI_LANG') or _remembered_lang())


def current() -> str:
    return _lang


def t(key: str, **kw: Any) -> str:
    """翻译 key。缺英文时回退中文；两者都缺则记录并返回 key 本身（检查器会报错）。"""
    row = STRINGS.get(key)
    if not row:
        _missing.add(key)
        return key
    text = row.get(_lang) or row.get('zh')
    if text is None:
        _missing.add(key)
        return key
    if kw:
        try:
            return text.format(**kw)
        except (KeyError, IndexError, ValueError):
            return text
    return text


def has(key: str, lang: str | None = None) -> bool:
    row = STRINGS.get(key)
    if not row:
        return False
    return (lang or _lang) in row


def missing_keys() -> set[str]:
    """运行期遇到过的缺口（t() 调用过但表里没有）。"""
    return set(_missing)


def coverage() -> dict[str, int]:
    """每种语言的条目数 + 只有中文没有英文的条目（应为 0）。"""
    out = {lg: sum(1 for v in STRINGS.values() if lg in v) for lg in _LANGS}
    out['zh_only'] = sum(1 for v in STRINGS.values() if 'zh' in v and 'en' not in v)
    out['keys'] = len(STRINGS)
    return out


def lang_from_request(path: str, body: dict | None = None) -> str:
    """HTTP 层用：?lang=en / POST body {'lang': 'en'} / 都不给则用当前设置。"""
    if body and isinstance(body.get('lang'), str):
        return set_lang(body['lang'])
    q = path.split('?', 1)[1] if '?' in path else ''
    for part in q.split('&'):
        if part.startswith('lang='):
            return set_lang(part[5:])
    return _lang


if __name__ == '__main__':
    cov = coverage()
    print('gui_i18n: %d keys, %s' % (cov['keys'], {k: v for k, v in cov.items() if k != 'keys'}))
