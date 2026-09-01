#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NInfer serve 控制台 — slider-minimal single-page app.
Launch/stop ninfer-serve with slider controls (KV tier, context, MTP drafts,
vision) and watch the live log. Backend runs serve as a subprocess.
"""
import json
import os
import signal
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>NInfer Serve</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1a2332;--mut:#6b7686;--acc:#3b82f6;
      --line:#e5e8ee;--ok:#16a34a;--bad:#dc2626;--run:#f59e0b}
*{box-sizing:border-box;margin:0;padding:0}
body{font:14px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);
     color:var(--ink);display:flex;justify-content:center;padding:28px 16px}
.wrap{width:780px;max-width:100%}
h1{font-size:20px;font-weight:650;display:flex;align-items:center;gap:10px}
.badge{font-size:11px;padding:2px 10px;border-radius:999px;font-weight:600}
.badge.idle{background:#eef2f8;color:var(--mut)}
.badge.run{background:#fef3c7;color:#b45309}
.sub{color:var(--mut);font-size:12px;margin:2px 0 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
      padding:20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(20,30,50,.05)}
.row{display:flex;align-items:center;gap:14px;margin-bottom:13px}
.row label{width:120px;color:var(--mut);font-size:13px;flex:none}
input[type=range]{flex:1;accent-color:var(--acc)}
.val{width:86px;text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
input[type=text]{flex:1;border:1px solid var(--line);border-radius:10px;
      padding:8px 12px;font:inherit;outline:none}
.btnrow{display:flex;gap:10px;margin-top:4px}
button{border:0;border-radius:10px;padding:10px 22px;font:inherit;font-weight:600;cursor:pointer}
#start{background:var(--acc);color:#fff}
#stop{background:var(--bad);color:#fff;display:none}
#log{background:#0f172a;color:#cbd5e1;border-radius:12px;padding:14px;
     font:12px/1.5 ui-monospace,Consolas,monospace;height:300px;overflow:auto;
     white-space:pre-wrap}
#gpu{font:12px ui-monospace,monospace;color:var(--mut);margin-top:10px}
</style></head><body><div class="wrap">
<h1>NInfer Serve <span class="badge idle" id="badge">未启动</span></h1>
<div class="sub">RTX 5090 · ninfer-serve 控制台 — 滑块简约版</div>
<div class="card">
  <div class="row"><label>模型</label>
    <input id="model" type="text" value="/home/user/models/qwen3_8_27b_nvfp4.ninfer"></div>
  <div class="row"><label>KV 精度档</label>
    <input id="kv" type="range" min="0" max="4" value="3" step="1">
    <div class="val" id="kv_v">NVFP4</div></div>
  <div class="row"><label>上下文</label>
    <input id="ctx" type="range" min="3" max="8" value="5" step="1">
    <div class="val" id="ctx_v">32k</div></div>
  <div class="row"><label>投机档位</label>
    <input id="spec" type="range" min="0" max="3" value="0" step="1">
    <div class="val" id="spec_v">无</div></div>
  <div class="row"><label>草稿数</label>
    <input id="draft" type="range" min="1" max="5" value="3" step="1">
    <div class="val" id="draft_v">3</div></div>
  <div class="row"><label>并发</label>
    <input id="conc" type="range" min="1" max="4" value="1" step="1">
    <div class="val" id="conc_v">1</div></div>
  <div class="row"><label>多模态</label>
    <input id="vis" type="range" min="0" max="1" value="0" step="1">
    <div class="val" id="vis_v">关</div></div>
  <div class="row"><label>端口</label>
    <input id="port" type="text" value="8000"></div>
  <div class="btnrow">
    <button id="start" onclick="ctrl('start')">启动</button>
    <button id="stop" onclick="ctrl('stop')">停止</button>
  </div>
</div>
<div class="card"><div id="log">就绪。</div><div id="gpu"></div></div>
<script>
const $=id=>document.getElementById(id);
const KV=['BF16','INT8','FP8','NVFP4','E8 混合'];
const CTX=[8,16,32,64,128,256];
const SPEC=['无','MTP','DFlash','DFlash2'];
$('kv').oninput=e=>$('kv_v').textContent=KV[+e.target.value];
$('ctx').oninput=e=>$('ctx_v').textContent=CTX[+e.target.value-3]+'k';
$('spec').oninput=e=>$('spec_v').textContent=SPEC[+e.target.value];
$('draft').oninput=e=>$('draft_v').textContent=e.target.value;
$('conc').oninput=e=>$('conc_v').textContent=e.target.value;
$('vis').oninput=e=>$('vis_v').textContent=+e.target.value?'开':'关';
async function ctrl(a){
  const r=await fetch('/api/'+a,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model:$('model').value.trim(),kv:+$('kv').value,
      ctx:+$('ctx').value,spec:+$('spec').value,draft:+$('draft').value,
      conc:+$('conc').value,vis:+$('vis').value,port:$('port').value.trim()})});
  const d=await r.json();
  $('badge').textContent=d.running?'运行中':'未启动';
  $('badge').className='badge '+(d.running?'run':'idle');
  $('start').style.display=d.running?'none':'';
  $('stop').style.display=d.running?'':'none';
}
async function poll(){
  try{
    const r=await fetch('/api/state');const d=await r.json();
    $('badge').textContent=d.running?'运行中':'未启动';
    $('badge').className='badge '+(d.running?'run':'idle');
    $('start').style.display=d.running?'none':'';
    $('stop').style.display=d.running?'':'none';
    $('log').textContent=d.log||'(无日志)';
    $('log').scrollTop=$('log').scrollHeight;
    if(d.gpu)$('gpu').textContent='GPU: '+d.gpu;
  }catch(e){}
}
setInterval(poll,1500);poll();
</script></body></html>"""

KV_ARGS = {
    0: '--kv-dtype', 'bf16',
    1: '--kv-dtype', 'int8',
    2: '--kv-dtype', 'fp8',
    3: '--kv-dtype', 'nvfp4',
    4: '--kv-layer-storage', '0-9:e8,10-15:nvfp4',
}
CTX_KB = [8192, 16384, 32768, 65536, 131072, 262144]
SPEC_ARGS = ['', '--spec mtp', '--spec dflash', '--spec dflash2']
SERVE_BIN = '/home/user/ninfer-fusion/build/apps/ninfer-serve'
_proc = None
_log = '就绪。'


def build_cmd(req):
    cmd = [SERVE_BIN, req['model'], '--port', req['port']]
    cmd += KV_ARGS.get(int(req['kv']), ['--kv-dtype', 'nvfp4']).copy()
    cmd += ['--max-context', str(CTX_KB[int(req['ctx']) - 3])]
    if int(req['spec']):
        cmd += SPEC_ARGS[int(req['spec'])].split()
        cmd += ['--draft-tokens', str(int(req['draft'])), '--lm-head-draft']
    if int(req['conc']) > 1:
        cmd += ['--max-concurrency', str(int(req['conc']))]
    if int(req['vis']):
        cmd += ['--vision']
    cmd += ['--no-cuda-graph']
    return cmd


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            body = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/api/state':
            gpu = ''
            try:
                out = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                     '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
                gpu = out.stdout.strip()
            except Exception:
                pass
            body = json.dumps({'running': _proc is not None and _proc.poll() is None,
                               'log': _log[-6000:], 'gpu': gpu}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global _proc, _log
        n = int(self.headers.get('Content-Length', 0))
        req = json.loads(self.rfile.read(n))
        if self.path == '/api/start':
            if _proc is not None and _proc.poll() is None:
                self._send({'running': True, 'log': _log})
                return
            cmd = build_cmd(req)
            _log = '启动: ' + ' '.join(cmd) + '\n'
            env = dict(os.environ, LD_LIBRARY_PATH='/usr/local/cuda-13.3/lib64')
            _proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, env=env)

            def reader():
                global _log
                for line in _proc.stdout:
                    _log += line
                    if len(_log) > 12000:
                        _log = _log[-8000:]
            import threading
            threading.Thread(target=reader, daemon=True).start()
            self._send({'running': True, 'log': _log})
        elif self.path == '/api/stop':
            if _proc is not None:
                _proc.terminate()
                try:
                    _proc.wait(timeout=15)
                except Exception:
                    _proc.kill()
                _log += '\n[已停止]\n'
                _proc = None
            self._send({'running': False, 'log': _log})
        else:
            self.send_response(404)
            self.end_headers()

    def _send(self, d):
        body = json.dumps(d).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8789
    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    print('NInfer Serve GUI: http://127.0.0.1:%d  (serve: %s)' % (port, SERVE_BIN))
    srv.serve_forever()


if __name__ == '__main__':
    main()
