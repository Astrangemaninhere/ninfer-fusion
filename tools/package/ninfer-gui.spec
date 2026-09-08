# -*- mode: python ; coding: utf-8 -*-
# ninfer-gui.spec — PyInstaller 单文件封装 (NInfer 统一入口).
# 用法: pyinstaller tools/package/ninfer-gui.spec
# 产物: dist/NInfer.exe —— 纯标准库 GUI (服务/导入向导/训练面板), 自检环境。

import os

ROOT = os.path.abspath(os.path.join(os.getcwd()))
GUI_ROOT = r'C:\Users\User\Documents\ziqinzhang'

a = Analysis(
    [os.path.join(GUI_ROOT, 'ninfer-gui.py')],
    pathex=[GUI_ROOT],
    datas=[
        (os.path.join(GUI_ROOT, 'gui_page.html'), '.'),
        # 新手识别/讲解/世代表 (纯 python 模块, 打包为源码目录供运行时 import)
        (r'C:\Users\User\Documents\ziqinzhang\ninfer-fusion-repo\tools\gui', 'tools/gui'),
        (r'C:\Users\User\Documents\ziqinzhang\ninfer-fusion-repo\tools\archkit', 'tools/archkit'),
    ],
    hiddenimports=[],
    excludes=['torch', 'numpy', 'safetensors', 'matplotlib', 'PIL', 'tkinter'],
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
