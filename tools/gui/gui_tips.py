#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gui_tips.py — plain-language hover hints for every knob (zh/en).

Novice mode renders these as mouse-over tooltips. Every entry follows the
   who reads it -> what it does -> what happens if you raise/lower it
shape, avoids AI jargon, and prefers a household metaphor over a definition.

Keys are the HTML control ids (see ninfer-gui.py). The GUI reads this module at
startup, so a new parameter needs one more entry here and one more row in the
i18n table -- nothing hardcoded in a page.

The texts themselves live in i18n_misc.py (`tips.*` / `tipg.*` / `tipgl.*`).
This file pins the ids and resolves them through gui_i18n.t() at ACCESS time
(`_LiveTable`), so a zh/en switch also updates tooltips that were built earlier,
and no user-visible text is stored here.

Self-check: python gui_tips.py  -> entry counts + any id missing zh/en.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gui_i18n import t  # noqa: E402


class _LiveTable(dict):
    """id -> text mapping that re-resolves through t() on every access.

    Keeps the historical call shapes (`TIPS['steps']`, `dict(TIPS.items())`,
    `TIPS.get(id)`) working while the current language may change between two
    reads.
    """

    def __init__(self, build):
        dict.__init__(self)
        self._build = build

    def _fresh(self) -> dict:
        return self._build()

    def __getitem__(self, key):
        return self._fresh()[key]

    def __contains__(self, key):
        return key in self._fresh()

    def __iter__(self):
        return iter(self._fresh())

    def __len__(self):
        return len(self._fresh())

    def get(self, key, default=None):
        return self._fresh().get(key, default)

    def keys(self):
        return self._fresh().keys()

    def values(self):
        return self._fresh().values()

    def items(self):
        return self._fresh().items()


# ---------------------------------------------------------------------------
# 一、训练参数 (微调 tab)
# ---------------------------------------------------------------------------
def _tips() -> dict:
    return {
        # ---- 训练 ----
        'steps': t('tips.steps'),
        'batch': t('tips.batch'),
        'anchors': t('tips.anchors'),
        'ctx': t('tips.ctx'),
        'lr': t('tips.lr'),
        'resume': t('tips.resume'),
        'ddtree': t('tips.ddtree'),

        # ---- 数据采集 ----
        'c_rag': t('tips.c_rag'),
        'c_qa': t('tips.c_qa'),
        'c_code': t('tips.c_code'),
        'c_out': t('tips.c_out'),

        # ---- 服务(serve)----
        'smodel': t('tips.smodel'),
        'sctx': t('tips.sctx'),
        'sspec': t('tips.sspec'),
        'sdt': t('tips.sdt'),
        'slabd': t('tips.slabd'),
        'slabd2': t('tips.slabd2'),
        'scold': t('tips.scold'),
        'scold2': t('tips.scold2'),
        'scoldpath': t('tips.scoldpath'),
        'scoldgb': t('tips.scoldgb'),
        'scoldgb2': t('tips.scoldgb2'),
        'sktiers': t('tips.sktiers'),
        'snvfp4mode': t('tips.snvfp4mode'),

        # ---- 导入向导 (wizard) ----
        'import_path': t('tips.import_path'),
        'import_run_ctx': t('tips.import_run_ctx'),
        'wizard_note': t('tips.wizard_note'),
    }


TIPS = _LiveTable(_tips)

# Ordered ids, for pages that want a stable list instead of dict ordering.
TIP_IDS = ('steps', 'batch', 'anchors', 'ctx', 'lr', 'resume', 'ddtree',
           'c_rag', 'c_qa', 'c_code', 'c_out',
           'smodel', 'sctx', 'sspec', 'sdt', 'slabd', 'slabd2', 'scold',
           'scold2', 'scoldpath', 'scoldgb', 'scoldgb2', 'sktiers', 'snvfp4mode',
           'import_path', 'import_run_ctx', 'wizard_note')


def tip(control_id: str, default: str = '') -> str:
    """Tooltip text for one HTML control id, in the current language."""
    return t('tips.%s' % control_id) if control_id in TIP_IDS else default


# ---------------------------------------------------------------------------
# 二、参数分组 (GUI 按组渲染"这是什么"的小标题)
# ---------------------------------------------------------------------------
def _groups() -> dict:
    return {
        'train': t('tipg.train'),
        'collect': t('tipg.collect'),
        'serve': t('tipg.serve'),
        'import': t('tipg.import'),
    }


GROUPS = _LiveTable(_groups)

GROUP_IDS = ('train', 'collect', 'serve', 'import')


def group(group_id: str, default: str = '') -> str:
    """Caption for a parameter group, in the current language."""
    return GROUPS.get(group_id, default)


# ---------------------------------------------------------------------------
# 三、概念名词解释 (环境自检结果 / 结论卡片里的术语都会用到)
# ---------------------------------------------------------------------------
def _glossary() -> dict:
    """term -> explanation, in the current language."""
    out = {}
    for gid in GLOSSARY_IDS:
        term, text = glossary_entry(gid)
        out[term] = text
    return out


GLOSSARY_IDS = ('vram', 'token', 'quant', 'nvfp4', 'context', 'spec_decode',
                'gguf', 'safetensors')


def glossary_entry(term_id: str) -> tuple:
    """(term, explanation) for one glossary id, in the current language."""
    return t('tipgl.%s.term' % term_id), t('tipgl.%s.text' % term_id)


GLOSSARY = _LiveTable(_glossary)


if __name__ == '__main__':
    import gui_i18n as i18n

    rows = ([('tips.%s' % i) for i in TIP_IDS]
            + [('tipg.%s' % i) for i in GROUP_IDS]
            + ['tipgl.%s.%s' % (i, s) for i in GLOSSARY_IDS for s in ('term', 'text')])
    missing = ['%s/%s' % (k, lg) for k in rows for lg in ('zh', 'en')
               if not i18n.has(k, lg)]
    print('gui_tips: %d tips / %d groups / %d glossary keys (%d i18n rows)'
          % (len(TIP_IDS), len(GROUP_IDS), 2 * len(GLOSSARY_IDS), len(rows)))
    print('missing zh/en entries: %d %s' % (len(missing), missing or '[]'))
