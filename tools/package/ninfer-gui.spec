# -*- mode: python ; coding: utf-8 -*-
# ninfer-gui.spec — PyInstaller 单文件封装 (NInfer 统一入口).
# 用法: pyinstaller tools/package/ninfer-gui.spec
# 产物: dist/NInfer.exe —— 纯标准库 GUI (服务/导入向导/训练面板), 自检环境。

import os

# This spec lives at <repo>\tools\package\ninfer-gui.spec, so the repository is two
# directories up and now also holds the GUI entry point (ninfer-gui.py) and its
# page (gui_page.html) at its root, next to tools/.  Both used to be absolute
# paths naming a sibling `ninfer-fusion-repo` checkout - a stale mirror, so the
# exe was packaged from whatever that copy held - while the entry lived one
# level above the repository, so a clean checkout could not be packaged at all.
# PyInstaller defines SPECPATH as the DIRECTORY holding the spec, while running the
# spec through plain python leaves __file__ as the spec FILE.  Taking dirname() of
# SPECPATH as well therefore resolves one directory too high: REPO became
# <workspace>, the entry resolved to <workspace>\ninfer-gui.py, and PyInstaller
# failed with "script ... not found" so no exe was produced at all.  Anchor on the
# spec FILE, which both routes agree on.
_SPEC = os.path.abspath(globals().get('SPEC')
                        or globals().get('__file__')
                        or os.path.join(globals().get('SPECPATH', '.'), 'ninfer-gui.spec'))
REPO = os.path.abspath(os.path.join(os.path.dirname(_SPEC), '..', '..'))
GUI_ROOT = REPO   # the entry + page now live in the repository itself
# The four wizard modules live in tools/gui, NOT next to the entry point.  A
# hiddenimports entry is resolved against pathex/sys.path: without this
# directory on pathex PyInstaller only warns "hidden import ... not found"
# and still bundles nothing -- the same defect one layer down.
GUI_PY = os.path.join(REPO, 'tools', 'gui')

a = Analysis(
    [os.path.join(GUI_ROOT, 'ninfer-gui.py')],
    pathex=[GUI_ROOT, GUI_PY],
    datas=[
        (os.path.join(GUI_ROOT, 'gui_page.html'), '.'),
        # 新手识别/讲解/世代表 (纯 python 模块, 打包为源码目录供运行时 import)
        # gui_i18n.py:_load_strings() finds its tables by GLOBBING i18n_*.py next to
        # its own __file__.  PyInstaller stores a bundled module at the bundle ROOT
        # (gui_i18n.__file__ = <_MEIPASS>\gui_i18n.py, measured), while the directory
        # datas land in <_MEIPASS>\tools\gui -- so the glob returns nothing, STRINGS
        # stays empty and every t() silently returns its own key (t('imp.env.gpu')
        # == 'imp.env.gpu').  The exe still answers 200 on 8077, i.e. a green smoke
        # test over a UI labelled with raw keys.  Shipping the tables at the bundle
        # root too makes the glob yield the names again; the modules themselves come
        # from the PYZ via hiddenimports, never from these source copies.
        (os.path.join(REPO, 'tools', 'gui', 'i18n_misc.py'), '.'),
        (os.path.join(REPO, 'tools', 'gui', 'i18n_serve.py'), '.'),
        (os.path.join(REPO, 'tools', 'gui'), 'tools/gui'),
        (os.path.join(REPO, 'tools', 'archkit'), 'tools/archkit'),
    ],
    # ninfer-gui.py imports these four at TOP LEVEL (ninfer-gui.py:39-42).  Shipping
    # them only as datas is not enough: the frozen app then compiles the .py from
    # _MEIPASS at run time, and `from __future__ import annotations`
    # (tools/gui/model_import.py:52) needs the __future__ module, which PyInstaller
    # does not ship => ModuleNotFoundError: No module named '__future__' at import
    # time, before ThreadingHTTPServer(...).serve_forever() is ever reached, so 8077
    # is never served.  Compiled into the PYZ they are imported from the archive and
    # the __future__ directive is consumed at freeze time instead.
    hiddenimports=[
        'model_import',      # tools/gui/model_import.py
        'gui_tips',          # -> gui_i18n
        'gpu_compat',        # tools/gui/gpu_compat.py
        'convert_runner',    # tools/gui/convert_runner.py
        # gui_i18n._load_strings() globs i18n_*.py and importlib.import_module()s them
        # by a name static analysis cannot see, and it swallows the failure.  Without
        # these the exe still answers 200 on 8077 but renders raw i18n keys, i.e. a
        # green smoke test over a textless UI -- so they are bundled too.
        'gui_i18n', 'i18n_misc', 'i18n_serve',
    ],
    # numpy is NOT excluded.  model_import.py asks tools/convert/gguf_kquant.py (which the
    # PYZ already ships, pulled in by that very import) which ggml quant types this repo
    # can decode, and gguf_names.py for its per-architecture tensor-name rules.  With
    # numpy excluded both raised ModuleNotFoundError inside the frozen import, every one
    # of those callers fell back to its "the reader is missing" literal, and the wizard
    # then blamed the FILE: measured 2026-09-14 on Ornith-1.5-9B-Q4_K_M.gguf (5.78 GB),
    # the exe answered unsupported_quant "this GGUF uses a quantisation this repo cannot
    # decode (it decodes F32/F16/BF16)" plus "271 tensors have no GGUF->HF name mapping"
    # and "layer accounting disagrees ... 0 nextn block(s) ... (blk.32)" while reading
    # 33 layers, where the source tree on the same machine read 32 layers and answered
    # unsupported_arch with no issues at all.  The bundle carries the reader that answers.
    excludes=['torch', 'safetensors', 'matplotlib', 'PIL', 'tkinter'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='NInfer',
    debug=False,
    strip=False,
    upx=False,
    console=False,          # 无控制台窗口 (浏览器接管界面)
    icon=None,
)
