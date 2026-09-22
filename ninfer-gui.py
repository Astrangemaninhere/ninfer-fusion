#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ninfer-gui.py — NInfer 统一入口(单窗口, 新手优先)。

把历史上的四个独立小 GUI 合成一个, 面向两种人:
  * 完全不懂 AI 的新手:  首页三步走 —— 贴模型路径 -> 自动识别 -> 一键运行。
  * 开发者/老手:         服务/微调/语料/转换 四个工具 tab(旧页面逻辑内嵌)。

架构:
  * 单文件 HTTP 服务 (ThreadingHTTPServer, 默认 8077)
  * 新手能力由 tools/gui/model_import.py (识别/判定/显存) +
    tools/gui/gui_tips.py (悬浮讲解字典) 提供 —— 那两个模块可单测。
  * rag/convert 两个老 GUI 首次打开时自动以子进程拉起 (8787/8788), 页面内嵌。
  * serve/训练/采集 逻辑从老 ninfer-train-gui.py 原样并入 (同一文件不再外挂)。
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.dirname(os.path.abspath(__file__))
# PyInstaller 单文件打包: 数据/模块在 _MEIPASS 解包目录 (tools/gui, tools/archkit...)
_FROZEN = getattr(sys, 'frozen', False)
if _FROZEN:
    BASE = getattr(sys, '_MEIPASS', BASE)
    REPO = BASE
else:
    # This entry now lives at the root of the repository it drives, so the
    # repository is its own directory whenever tools/gui is there.  A working
    # copy may still run it from outside a checkout (the old layout: the entry
    # in a work directory with the checkout under ninfer-fusion-repo/).
    REPO = BASE if os.path.isdir(os.path.join(BASE, 'tools', 'gui')) \
        else os.path.join(BASE, 'ninfer-fusion-repo')
# Where archkit writes its spec/out.  A frozen build unpacks to _MEIPASS, which is
# deleted when the GUI exits, so REPO (== _MEIPASS there) must not hold the result:
# the endpoint would report ok with paths that are already gone.  Persistent next to
# the exe when frozen; byte-for-byte the old location in a source tree.
ARCHKIT_HOME = (os.path.join(os.path.dirname(os.path.abspath(sys.executable)), 'archkit-work')
                if _FROZEN else os.path.join(REPO, 'tools', 'archkit'))
# --------------------------------------------------------------------------
# WSL-side locations
# --------------------------------------------------------------------------
# The engine only ever runs on the WSL side (every launch below goes through wsl.exe), so this file
# has to name paths that exist over there.  Those were literals: /home/user/models, /home/user/bench
# and /home/user/ninfer-fusion.  A checkout handed to anyone whose WSL user is not `user`, or whose
# tree is not at /home/user/ninfer-fusion, then failed SILENTLY -- two of the five sites redirect
# stderr to /dev/null, and a wrong /home/user path is indistinguishable from an empty directory.
#
# The defaults below ARE those literals, so an existing installation behaves byte-for-byte as
# before; the four names exist so a different installation can say where it put things:
#   NINFER_WSL_HOME    default /home/user
#   NINFER_WSL_REPO    default $NINFER_WSL_HOME/ninfer-fusion   (the engine's source tree)
#   NINFER_WSL_MODELS  default $NINFER_WSL_HOME/models          (the .ninfer directory)
#   NINFER_WSL_BENCH   default $NINFER_WSL_HOME/bench           (the collector's directory)
#
# WHY ENVIRONMENT VARIABLES AND NOT DETECTION: this process runs on the Windows side and the paths it
# needs are on the other side of wsl.exe, so there is nothing to look up.  A tree's WSL-side path is
# not derivable from any Windows-side path, and the two are not even required to be the same tree
# (the packaged .exe unpacks itself to a temp directory).
#
# NOTE ON THE JOINS: os.path.join is deliberately NOT used here, although this file uses it
# everywhere else.  This process is a WINDOWS process, so os.path.join is ntpath.join and would
# produce "/home/user\models"; these strings are consumed by bash on the other side of wsl.exe,
# where the separator is always "/".  Hence the explicit + "/".
WSL_HOME = os.environ.get('NINFER_WSL_HOME', '/home/user').rstrip('/')
WSL_REPO = os.environ.get('NINFER_WSL_REPO', WSL_HOME + '/ninfer-fusion').rstrip('/')
WSL_MODELS = os.environ.get('NINFER_WSL_MODELS', WSL_HOME + '/models').rstrip('/')
WSL_BENCH = os.environ.get('NINFER_WSL_BENCH', WSL_HOME + '/bench').rstrip('/')

# --------------------------------------------------------------------------
# Child-process interpreter
# --------------------------------------------------------------------------
# Every child launch in this file used to be `[sys.executable, script]`.  In a frozen
# build sys.executable IS this exe, so the child ignores argv, tries to bind 8077 and
# hangs.  Measured on 2026-09-14 against the packaged build (/api/archkit_spec): the
# server dropped the connection after 244 s -- two back-to-back 120 s
# subprocess.TimeoutExpired on ['NInfer.exe', 'arch_spec.py'] -- and left two NInfer
# processes behind, while the same two scripts ran fine under a real python.  So resolve
# a real interpreter, and when there is none say so: never fall back to this exe, and
# never let the caller hang with no answer.
_PY = None
_PY_TRIED = False


def _candidate_pythons():
    """Candidate interpreters, best first.  NINFER_PYTHON overrides everything."""
    override = os.environ.get('NINFER_PYTHON')
    if override:
        # An explicit choice is authoritative: quietly falling back to something else
        # would hide a typo and could spawn an interpreter the operator did not ask for.
        return [os.path.abspath(override)]
    seen, out = set(), []

    def add(p):
        if not p:
            return
        try:
            p = os.path.abspath(p)
        except Exception:
            return
        if p not in seen:
            seen.add(p)
            out.append(p)

    add(getattr(sys, '_base_executable', None))   # PyInstaller may point at the real one
    for name in ('python', 'python3', 'python3.12'):
        add(shutil.which(name))
    for pat in ('C:/Program Files/Python3*/python.exe',
                os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Programs',
                             'Python', 'Python3*', 'python.exe')):
        for p in sorted(glob.glob(pat), reverse=True):
            add(p)
    return out


def real_python():
    """Absolute path of a real Python 3 interpreter, or None.

    A candidate counts only if it starts and reports a 3.x version under the very
    environment the child would inherit; that also filters the Windows Store stub
    (...\\WindowsApps\\python3.exe), which would otherwise just open the Store.
    """
    global _PY, _PY_TRIED
    if _PY_TRIED:
        return _PY
    _PY_TRIED = True
    if not _FROZEN:
        _PY = sys.executable          # a source tree's sys.executable IS one
        return _PY
    me = os.path.abspath(sys.executable)
    for c in _candidate_pythons():
        if os.path.abspath(c) == me or not os.path.isfile(c):
            continue
        try:
            r = subprocess.run([c, '-c', 'import sys; sys.stdout.write("%d.%d" % sys.version_info[:2])'],
                               capture_output=True, text=True, timeout=20)
        except Exception:
            continue
        if r.returncode == 0 and (r.stdout or '').startswith('3.'):
            _PY = c
            return _PY
    _PY = None
    return None


NO_PYTHON_ERR = ('找不到可用的 Python 解释器, '
                 '无法启动子进程: 本程序是打包版, '
                 'sys.executable 指向 NInfer.exe 本身。'
                 '请安装 Python 3 (并把 python 加进 PATH), '
                 '或用环境变量 NINFER_PYTHON 指定解释器。')


def real_python_or_err():
    """(path, '') or (None, reason) -- for handlers that must answer, not hang."""
    py = real_python()
    return (py, '') if py else (None, NO_PYTHON_ERR)


sys.path.insert(0, os.path.join(REPO, 'tools', 'gui'))
sys.path.insert(0, BASE)          # train_dspark/train_dflash2/pipeline_cfg 同目录
import model_import as mi          # noqa: E402  识别/判定/显存/环境
import gui_tips as tips            # noqa: E402  悬浮讲解
import gpu_compat                  # noqa: E402  显卡世代/档位判定
import convert_runner              # noqa: E402  导入后自动转换编排

LOG = os.path.join(BASE, 'dl', 'train-dspark.log')
TOOLS = {                         # 老 GUI: name -> (脚本, 端口)
    'rag':     ('ninfer-rag-gui.py', 8787),
    'convert': ('ninfer-convert-gui.py', 8788),
}
_tool_proc = {}


def tool_url(name):
    return 'http://127.0.0.1:%d' % TOOLS[name][1]


def ensure_tool(name):
    """老工具 GUI 懒启动: 端口有响应就直接用, 否则拉起子进程。"""
    import urllib.request
    port = TOOLS[name][1]
    try:
        urllib.request.urlopen('http://127.0.0.1:%d/' % port, timeout=1)
        return True, ''
    except Exception:
        pass
    script = os.path.join(BASE, TOOLS[name][0])
    if not os.path.exists(script):
        return False, 'missing script %s' % script
    py, pyerr = real_python_or_err()
    if py is None:
        return False, pyerr
    logpath = os.path.join(BASE, 'dl', 'tool-%s.log' % name)
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    proc = subprocess.Popen([py, script], stdout=open(logpath, 'ab'),
                            stderr=subprocess.STDOUT, cwd=BASE)
    _tool_proc[name] = proc
    for _ in range(30):           # 最多等 3 秒端口起来
        try:
            urllib.request.urlopen('http://127.0.0.1:%d/' % port, timeout=0.5)
            return True, ''
        except Exception:
            time.sleep(0.1)
    return True, 'started, still warming up'


# --------------------------------------------------------------------------
# 训练/采集/服务 状态 (自 ninfer-train-gui.py 并入)
# --------------------------------------------------------------------------
proc = None
cproc = None
cstate = 'idle'
cprog = '-'
sproc = None
sstate = 'idle'
sres = '-'
cpacks = '-'
csize = '-'
_clast = 0.0
state = 'idle'
last_tail = ''
gpu_cache = ('-', '-')


def _active_log():
    global LOG
    try:
        out = subprocess.run(
            ['powershell.exe', '-NoProfile', '-Command',
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10)
        cmds = out.stdout or ''
        if 'train_dflash2.py' in cmds:
            return os.path.join(BASE, 'dl', 'train-dflash2.log')
        if 'train_dspark.py' in cmds:
            return os.path.join(BASE, 'dl', 'train-dspark.log')
    except Exception:
        pass
    return LOG


def read_log_tail(path, n=4000):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode('utf-8', 'replace')
    except OSError:
        return ''


def parse_progress(tail):
    m = re.findall(r'step (\d+)/(\d+) loss=([\d.eE+-]+) lr=([\d.eE+-]+) steps/s=([\d.]+)', tail)
    if m:
        st, tot, loss, lr, sps = m[-1]
        return {'step': '%s/%s' % (st, tot), 'loss': loss, 'lr': lr, 'sps': sps}
    return {'step': '-', 'loss': '-', 'lr': '-', 'sps': '-'}


def gpu_info():
    global gpu_cache
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5)
        parts = (out.stdout or '').strip().split(',')
        gpu_cache = ('%s%%' % parts[0].strip(),
                     '%s/%s MiB' % (parts[1].strip(), parts[2].strip()))
    except Exception:
        pass
    return gpu_cache


def launch_script(cmd, logpath):
    global proc
    if proc and proc.poll() is None:
        return False, 'already running'
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    with open(logpath, 'ab') as f:
        f.write(b'\n=== launch %s ===\n' % time.strftime('%Y-%m-%d %H:%M:%S').encode())
    proc = subprocess.Popen(cmd, stdout=open(logpath, 'ab'), stderr=subprocess.STDOUT,
                            cwd=BASE)
    return True, ''


PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'gui_page.html'), encoding='utf-8').read()


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj, ctype='application/json'):
        body = json.dumps(obj, ensure_ascii=False).encode() if not isinstance(obj, str) \
            else obj.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype +
                         ('; charset=utf-8' if ctype == 'text/html' else ''))
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        global state, proc, cproc, LOG, last_tail, cstate, cprog, cpacks, csize, _clast
        global sproc, sstate, sres
        p = self.path
        if p in ('/', '/index.html'):
            self._send(200, PAGE, 'text/html')
            return
        if p.startswith('/api/archkit_spec'):
            # 一键生成接入规格: 对任意 HF config 目录跑 ARCHKIT extract+gen
            import urllib.parse as up
            import re as _re
            q = up.parse_qs(up.urlparse(p).query)
            cfg_path = q.get('config', [''])[0].strip()
            model_id = q.get('model_id', [''])[0].strip()
            if not cfg_path or not os.path.isfile(cfg_path):
                self._send(200, {'ok': False,
                                 'err': '需要 config.json 文件路径'})
                return
            if not model_id:
                model_id = _re.sub(r'[^a-zA-Z0-9_-]+', '-',
                                   os.path.basename(os.path.dirname(cfg_path))).strip('-')
                if not model_id:
                    model_id = 'model'
            ar = os.path.join(REPO, 'tools', 'archkit')
            specp = os.path.join(ARCHKIT_HOME, 'specs', model_id + '_spec.json')
            outp = os.path.join(ARCHKIT_HOME, 'out', model_id)
            ch = os.path.join(outp, 'config.h')
            gone = [f for f in ('arch_spec.py', 'gen_target.py')
                    if not os.path.isfile(os.path.join(ar, f))]
            if gone:
                self._send(200, {'ok': False,
                                 'err': '%s 里缺少 %s' % (ar, ', '.join(gone))})
                return
            py, pyerr = real_python_or_err()
            if py is None:
                self._send(200, {'ok': False, 'err': pyerr})
                return
            try:
                os.makedirs(os.path.dirname(specp), exist_ok=True)
                os.makedirs(outp, exist_ok=True)
                r1 = subprocess.run([py, os.path.join(ar, 'arch_spec.py'), 'extract',
                                     cfg_path, model_id, specp],
                                    capture_output=True, text=True, timeout=120,
                                    encoding='utf-8', errors='replace')
                r2 = subprocess.run([py, os.path.join(ar, 'gen_target.py'), specp,
                                     '--out', outp],
                                    capture_output=True, text=True, timeout=120,
                                    encoding='utf-8', errors='replace')
            except subprocess.TimeoutExpired as e:
                self._send(200, {'ok': False, 'interp': py,
                                 'err': '生成超时 (>120s): %s' % (e.cmd,)})
                return
            except OSError as e:
                self._send(200, {'ok': False, 'interp': py, 'err': str(e)[:200]})
                return
            ok = os.path.isfile(specp) and os.path.isfile(ch)
            self._send(200, {'ok': ok, 'interp': py,
                             'spec': specp if ok else '',
                             'config_h': ch if ok else '',
                             'err': (r1.stderr or r2.stderr or '')[-300:] if not ok else ''})
            return
        if p.startswith('/api/import_convert'):
            # 自动转换: scan -> 编排 -> 后台任务
            import urllib.parse as up
            q = up.parse_qs(up.urlparse(p).query)
            path = q.get('path', [''])[0].strip()
            if not path:
                self._send(200, {'ok': False, 'err': 'empty path'})
                return
            scan = mi.scan_path(path).as_dict()
            job = convert_runner.runner.start(scan)
            self._send(200, {'ok': True, 'job': job.job_id,
                             'state': job.state, 'error': job.error})
            return
        if p.startswith('/api/import_job'):
            # 任务状态; 完成后自动发布到 WSL 模型目录 (供 serve)
            import urllib.parse as up
            q = up.parse_qs(up.urlparse(p).query)
            jid = q.get('id', [''])[0]
            st = convert_runner.runner.status(jid)
            if st is None:
                self._send(200, {'ok': False, 'err': 'unknown job'})
                return
            if st['state'] == 'done' and st.get('out_artifact') and \
                    not st.get('published'):
                art = os.path.join(convert_runner.MODELS_DIR, st['out_artifact'])
                if os.path.exists(art):
                    r = subprocess.run(['wsl.exe', '-d', 'Ubuntu', 'bash', '-lc',
                                        "cp '%s' '%s/' 2>/dev/null"
                                        % (art.replace('\\', '/').replace('C:', '/mnt/c'), WSL_MODELS)],
                                       capture_output=True, timeout=120)
                    st['published'] = r.returncode == 0
                    st['log'] = (st.get('log') or []) + ['[publish] wsl cp rc=%d' %
                                                         r.returncode]
            self._send(200, {'ok': True, 'job': st})
            return
        if p == '/api/env':
            env = mi.env_report()
            gpu = mi.gpu_report()
            compat = gpu_compat.query().as_dict()
            # 环境卡补一行"显卡世代/档位"说明 (多型号 GPU 支持的一等公民信息)。`ok` 由
            # gpu_compat.env_item() 计算, 不再是常量 True: 这一行讲的是路线**规划**, 而
            # 卡片自身的能力只有引擎的能力探针能回答, 所以未测量时状态是 None ——
            # gui_page.html 把它渲染成中性的 '?', 既不打勾也不打叉 (打叉同样是未测量的判定)。
            if compat.get('present'):
                env['items'].insert(0, gpu_compat.env_item(compat))
            self._send(200, {'ok': True, 'env': env['items'], 'gpu': gpu.as_dict(),
                             'compat': compat})
            return
        if p == '/api/tips':
            self._send(200, {'ok': True, 'tips': tips.TIPS, 'glossary': tips.GLOSSARY,
                             'groups': tips.GROUPS})
            return
        if p.startswith('/api/browse'):
            import urllib.parse as up
            q = up.parse_qs(up.urlparse(p).query)
            base = q.get('path', [''])[0].strip() or BASE
            # 目录浏览给"不会打路径"的新手用: 永远返回绝对路径,
            # 只允许存在于本机的绝对路径, 拒绝任何穿越。
            try:
                base = os.path.abspath(base)
                if not os.path.isdir(base):
                    base = os.path.dirname(base)
                entries = []
                for name in sorted(os.listdir(base)):
                    if name.startswith('.'):
                        continue
                    fp = os.path.join(base, name)
                    if os.path.isdir(fp):
                        entries.append({'name': name, 'dir': True})
                    elif name.endswith(('.ninfer', '.gguf')) or \
                            (name == 'config.json'):
                        entries.append({'name': name, 'dir': False})
                    if len(entries) >= 200:
                        break
            except Exception as e:
                self._send(200, {'ok': False, 'err': str(e)})
                return
            self._send(200, {'ok': True, 'path': base,
                             'parent': os.path.dirname(base) if base != os.path.dirname(base)
                             else base, 'entries': entries})
            return
        if p.startswith('/api/import'):
            import urllib.parse as up
            q = up.parse_qs(up.urlparse(p).query)
            path = q.get('path', [''])[0].strip()
            ctx = int(q.get('ctx', ['8192'])[0] or 8192)
            if not path:
                self._send(200, {'ok': False, 'err': 'empty path'})
                return
            scan = mi.scan_path(path)
            v = mi.verdict(scan, mi.gpu_report(), ctx)
            out = {'ok': True, 'scan': scan.as_dict(),
                   'verdict': {'action': v.action, 'level': v.level,
                               'title': v.title, 'detail': v.detail, 'tips': v.tips}}
            if scan.kind == 'ninfer' and v.action == 'run':
                # 顺带告诉新手: 这个模型在运行列表里叫什么(在 WSL/本地模型目录)
                out['models'] = list_models()
            self._send(200, out)
            return
        if p.startswith('/api/tool'):
            import urllib.parse as up
            q = up.parse_qs(up.urlparse(p).query)
            name = q.get('name', [''])[0]
            if name not in TOOLS:
                self._send(200, {'ok': False, 'err': 'unknown tool'})
                return
            ok, err = ensure_tool(name)
            self._send(200, {'ok': ok, 'err': err, 'url': tool_url(name)})
            return
        # ---------------- 状态 / 训练 / 采集 (并入自 train-gui) ----------------
        if p == '/api/status':
            tail = read_log_tail(_active_log())
            last_tail = tail
            prog = parse_progress(tail)
            if proc is not None and proc.poll() is not None:
                state = 'done'
            u, v = gpu_info()
            cst = cstate
            if cproc is not None and cproc.poll() is not None:
                cst = 'done'
            self._send(200, {'state': state, 'log_tail': tail[-3500:],
                             'step': prog['step'], 'loss': prog['loss'],
                             'lr': prog['lr'], 'sps': prog['sps'], 'gpu': u, 'vram': v,
                             'cstate': cst, 'cpacks': cpacks, 'csize': csize,
                             'cprog': cprog})
            return
        if p.startswith('/api/start'):
            import urllib.parse as up
            q = up.parse_qs(up.urlparse(p).query)
            cfg = json.loads(q.get('cfg', ['{}'])[0])
            script = os.path.join(BASE, cfg.get('script', 'train_dspark.py'))
            if not os.path.exists(script):
                self._send(200, {'ok': False, 'err': 'script missing'})
                return
            py, pyerr = real_python_or_err()
            if py is None:
                self._send(200, {'ok': False, 'err': pyerr})
                return
            cmd = [py, script, '--steps', str(cfg.get('steps', 6000)),
                   '--batch-seqs', str(cfg.get('batch', 4)),
                   '--anchors-per-seq', str(cfg.get('anchors', 1)),
                   '--max-ctx', str(cfg.get('ctx', 2048)), '--lr', str(cfg.get('lr', '1e-4'))]
            if cfg.get('resume'):
                cmd += ['--resume']
            if cfg.get('ddtree'):
                cmd += ['--ddtree-topk', '16']
            logpath = os.path.join(BASE, 'dl',
                                   os.path.splitext(os.path.basename(script))[0] + '.log')
            ok, err = launch_script(cmd, logpath)
            if ok:
                LOG = logpath
                state = 'running'
            self._send(200, {'ok': ok, 'err': err})
            return
        if p == '/api/stop':
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
            state = 'idle'
            self._send(200, {'ok': True})
            return
        if p == '/api/collect':
            import urllib.parse as up
            q = up.parse_qs(up.urlparse(p).query)
            cfg = json.loads(q.get('cfg', ['{}'])[0])
            out_dir = str(cfg.get('out', 'hs_cache2'))
            if not re.fullmatch(r'[A-Za-z0-9_]+', out_dir):
                self._send(200, {'ok': False, 'err': 'bad output name'})
                return
            if cproc and cproc.poll() is None:
                self._send(200, {'ok': False, 'err': 'collector running'})
                return
            cmd = ("cd '%s' && COLLECT_RAG_N=%d COLLECT_QA_N=%d "
                   'COLLECT_CODE_N=%d COLLECT_OUT=%s python3 collect_full.py'
                   % (WSL_BENCH, int(cfg.get('rag', 800)), int(cfg.get('qa', 400)),
                      int(cfg.get('code', 200)), out_dir))
            clog = os.path.join(BASE, 'dl', 'collect-full-gui.log')
            cproc = subprocess.Popen(['wsl.exe', '-d', 'Ubuntu', 'bash', '-lc', cmd],
                                     stdout=open(clog, 'ab'), stderr=subprocess.STDOUT)
            state = 'collecting'
            cstate = 'collecting'
            cprog = 'starting...'
            self._send(200, {'ok': True})
            return
        if p == '/api/cstop':
            if cproc and cproc.poll() is None:
                cproc.terminate()
                try:
                    cproc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    cproc.kill()
            subprocess.run(['wsl.exe', '-d', 'Ubuntu', 'bash', '-lc',
                            "pkill -9 -f 'collect_full.p[y]'"],
                           capture_output=True)
            cproc = None
            state = 'idle'
            cstate = 'idle'
            self._send(200, {'ok': True})
            return
        # ---------------- serve 服务管理 ----------------
        if p == '/api/models':
            self._send(200, {'ok': True, 'models': list_models()})
            return
        if p == '/api/serve_params':
            # 全参数注册表 (引擎源码抽取; GUI 分组面板渲染)
            reg_path = os.path.join(REPO, 'tools', 'gui', 'serve_params.json')
            try:
                reg = json.load(open(reg_path, encoding='utf-8'))
                self._send(200, {'ok': True, 'registry': reg})
            except Exception as e:
                self._send(200, {'ok': False, 'err': str(e)[:120], 'registry': None})
            return
        if p.startswith('/api/serve'):
            import urllib.parse as up
            q = up.parse_qs(up.urlparse(p).query)
            cfg = json.loads(q.get('cfg', ['{}'])[0])
            model = str(cfg.get('model', ''))
            if not model.endswith('.ninfer'):
                self._send(200, {'ok': False, 'err': 'pick a model first'})
                return
            mpath = WSL_MODELS + '/' + model
            if sproc and sproc.poll() is None:
                self._send(200, {'ok': False, 'err': 'serve running'})
                return
            ctx = str(cfg.get('ctx', '8192'))
            spec = str(cfg.get('spec', ''))
            dt = str(cfg.get('dt', '3'))
            kv_layer = str(cfg.get('kv_layer', '')).strip()
            nvfp4_mode = str(cfg.get('nvfp4_mode', 'fusion'))
            graph = str(cfg.get('graph', 'on'))
            # KV 布局优先级: 逐层自定义 > pure/fusion 模式 > 引擎默认
            if not kv_layer:
                kv_layer = 'all:nvfp4' if nvfp4_mode == 'pure' else ''
            cmd = ("cd '%s' && ./build/apps/ninfer-serve '%s' "
                   '--port 8000 --kv-dtype nvfp4%s '
                   '--max-context %s --kv-capacity %s'
                   % (WSL_REPO, mpath, (" --kv-layer-storage '%s'" % kv_layer) if kv_layer else '',
                      ctx, ctx))
            if graph == 'off':
                cmd += ' --no-cuda-graph'
            if spec:
                cmd += ' --spec %s --draft-tokens %s' % (spec, dt)
            # 冷存储 (SSD 外挂): ngram/PLE 表与 KV 冷页将来都挂这条链上
            cold = str(cfg.get('cold', ''))
            if cold in ('disk', 'window', 'host'):
                cmd += ' --cold-policy %s' % cold
                try:
                    keep = int(float(cfg.get('cold_keep', 0) or 0))
                except (TypeError, ValueError):
                    keep = 0
                if keep > 0:
                    cmd += ' --cold-keep-tokens %d' % keep
                try:
                    cold_gb = int(float(cfg.get('cold_gb', 0) or 0))
                except (TypeError, ValueError):
                    cold_gb = 0
                if cold == 'disk':
                    cold_path = str(cfg.get('cold_path', '')).strip()
                    if cold_path:
                        cmd += " --cold-disk-path '%s'" % cold_path
                    if cold_gb > 0:
                        cmd += ' --cold-disk-bytes %d' % (cold_gb << 30)
                elif cold == 'host' and cold_gb > 0:
                    cmd += ' --cold-host-bytes %d' % (cold_gb << 30)
            # ---- 全参数面板透传 (注册表驱动; 空值=引擎默认) ----
            extras = cfg.get('extras') or {}
            handled = {'model', 'ctx', 'spec', 'dt', 'cold', 'cold_keep', 'cold_path',
                       'cold_gb', 'kv_tiers', 'nvfp4_mode', 'graph', 'kv_layer', 'extras'}
            for key, val in extras.items():
                if key in handled:
                    continue
                sval = str(val).strip()
                if not sval:
                    continue
                flag = '--' + str(key).replace('_', '-')
                # 布尔型 (值 True/False) 只拼旗标; 其余拼值 (路径类加引号)
                if sval.lower() in ('true', '1', 'on'):
                    cmd += ' ' + flag
                elif sval.lower() in ('false', '0', 'off'):
                    continue
                elif key.endswith(('_path', '_file')) or '/' in sval or '\\' in sval:
                    cmd += " %s '%s'" % (flag, sval)
                else:
                    cmd += ' %s %s' % (flag, sval)
            slog = os.path.join(BASE, 'dl', 'serve-gui.log')
            sproc = subprocess.Popen(['wsl.exe', '-d', 'Ubuntu', 'bash', '-lc', cmd],
                                     stdout=open(slog, 'ab'), stderr=subprocess.STDOUT)
            sstate = 'starting'
            sres = 'serve starting: ' + model
            self._send(200, {'ok': True})
            return
        if p == '/api/serve_stop':
            if sproc and sproc.poll() is None:
                sproc.terminate()
                try:
                    sproc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    sproc.kill()
            subprocess.run(['wsl.exe', '-d', 'Ubuntu', 'bash', '-lc',
                            "pkill -9 -f 'ninfer-serv[e]'"], capture_output=True)
            sproc = None
            sstate = 'idle'
            self._send(200, {'ok': True})
            return
        if p == '/api/serve_status':
            if sproc is not None and sproc.poll() is not None:
                sstate = 'stopped'
                sproc = None
            if sstate == 'starting':
                try:
                    import urllib.request
                    urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=2)
                    sstate = 'running'
                    sres = 'serve up on :8000'
                except Exception:
                    pass
            self._send(200, {'ok': True, 'serve': sstate or 'idle',
                             'models': list_models(), 'res': sres or '-'})
            return
        if p.startswith('/api/smoke'):
            if sstate != 'running':
                self._send(200, {'ok': False, 'err': 'serve not running'})
                return
            try:
                import urllib.request
                body = json.dumps({'model': 'qwen3.8-27b',
                                   'messages': [{'role': 'user',
                                                 'content': 'Say hello in one sentence.'}],
                                   'max_tokens': 32}).encode()
                req = urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',
                                             data=body,
                                             headers={'Content-Type': 'application/json'})
                resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
                sres = str(resp['choices'][0]['message']['content'])[:300]
                self._send(200, {'ok': True})
            except Exception as e:
                sres = 'smoke err: ' + str(e)[:120]
                self._send(200, {'ok': False, 'err': sres})
            return
        self._send(404, {'err': 'not found'})

    def log_message(self, *a):
        pass


def list_models():
    """本机 + WSL 模型目录里的 .ninfer 文件名 (服务页下拉用)。"""
    models = []
    try:
        r = subprocess.run(['wsl.exe', '-d', 'Ubuntu', 'bash', '-lc',
                            "ls '%s'/*.ninfer 2>/dev/null" % WSL_MODELS],
                           capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=15)
        models += sorted(os.path.basename(x) for x in (r.stdout or '').split())
    except Exception:
        pass
    for d in (os.path.join(BASE, 'models'),):
        if os.path.isdir(d):
            models += [f for f in sorted(os.listdir(d)) if f.endswith('.ninfer')]
    return sorted(set(models))


if __name__ == '__main__':
    port = int(os.environ.get('NINFER_GUI_PORT', '8077'))
    print('NInfer 统一入口: http://127.0.0.1:%d' % port, flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
