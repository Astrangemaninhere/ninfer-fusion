#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NInfer 模型转换 GUI — slider-minimal single-page app.
Wraps tools/convert/qwen3_8_27b: BF16 convert and NVFP4 quantized convert.
Sliders: quant mode, KV cache tier, context ceiling, batch width.
"""
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>NInfer 转换</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1a2332;--mut:#6b7686;--acc:#3b82f6;
      --line:#e5e8ee;--ok:#16a34a;--warn:#d97706}
*{box-sizing:border-box;margin:0;padding:0}
body{font:14px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);
     color:var(--ink);display:flex;justify-content:center;padding:32px 16px}
.wrap{width:760px;max-width:100%}
h1{font-size:20px;font-weight:650}
.sub{color:var(--mut);font-size:12px;margin:2px 0 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
      padding:20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(20,30,50,.05)}
.row{display:flex;align-items:center;gap:14px;margin-bottom:14px}
.row label{width:120px;color:var(--mut);font-size:13px;flex:none}
input[type=range]{flex:1;accent-color:var(--acc)}
.val{width:76px;text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
input[type=text]{flex:1;border:1px solid var(--line);border-radius:10px;
      padding:9px 12px;font:inherit;outline:none}
input[type=text]:focus{border-color:var(--acc)}
button{background:var(--acc);color:#fff;border:0;border-radius:10px;
       padding:10px 22px;font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.55;cursor:default}
#log{background:#0f172a;color:#cbd5e1;border-radius:12px;padding:14px;
     font:12px/1.5 ui-monospace,Consolas,monospace;min-height:140px;
     max-height:320px;overflow:auto;white-space:pre-wrap}
#status{font-size:12px;margin-left:auto;color:var(--mut)}
.mode{display:inline-block;padding:2px 10px;border-radius:8px;font-size:12px;
      font-weight:600;margin-left:8px}
</style></head><body><div class="wrap">
<h1>NInfer 模型转换</h1>
<div class="sub">safetensors（HF / vLLM）→ .ninfer — 滑块简约版</div>
<div class="card">
  <div class="row"><label>模型目录</label>
    <input id="model" type="text" placeholder="/path/to/Qwen3.8-27B"></div>
  <div class="row"><label>量化源目录</label>
    <input id="qmodel" type="text" placeholder="（NVFP4 转换时填写）"></div>
  <div class="row"><label>输出路径</label>
    <input id="out" type="text" placeholder="out/qwen3_8_27b.ninfer"></div>
  <div class="row"><label>量化档位</label>
    <input id="quant" type="range" min="0" max="3" value="0" step="1">
    <div class="val" id="quant_v">BF16</div></div>
  <div class="row"><label>KV 精度档</label>
    <input id="kv" type="range" min="0" max="4" value="3" step="1">
    <div class="val" id="kv_v">NVFP4</div></div>
  <div class="row"><label>上下文上限</label>
    <input id="ctx" type="range" min="3" max="8" value="6" step="1">
    <div class="val" id="ctx_v">64k</div></div>
  <div class="row"><label>批量宽度</label>
    <input id="batch" type="range" min="1" max="8" value="1" step="1">
    <div class="val" id="batch_v">1</div></div>
  <div class="row"><button id="go" onclick="run()">开始转换</button>
    <span id="status"></span></div>
</div>
<div class="card"><div id="log">就绪。</div></div>
<script>
const $=id=>document.getElementById(id);
const QUANT=['BF16','FP8','NVFP4','Groupwise'];
const KV=['BF16','INT8','FP8','NVFP4','E8 混合'];
const CTX=[8,16,32,64,128,256];
$('quant').oninput=e=>$('quant_v').textContent=QUANT[+e.target.value];
$('kv').oninput=e=>$('kv_v').textContent=KV[+e.target.value];
$('ctx').oninput=e=>$('ctx_v').textContent=CTX[+e.target.value-3]+'k';
$('batch').oninput=e=>$('batch_v').textContent=e.target.value;
async function run(){
  const model=$('model').value.trim(), out=$('out').value.trim();
  if(!model||!out){$('status').textContent='填模型和输出路径';return}
  $('go').disabled=true;$('status').textContent='转换中…';
  $('log').textContent='';
  try{
    const r=await fetch('/api/convert',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model,out,quant:+$('quant').value,qmodel:$('qmodel').value.trim(),
                           kv:+$('kv').value,ctx:+$('ctx').value,batch:+$('batch').value})});
    const d=await r.json();
    $('log').textContent=d.log||'';
    $('status').textContent=d.ok?'完成 ✓':('失败: '+(d.error||''));
  }catch(e){$('status').textContent='请求失败'}
  $('go').disabled=false;
}
</script></body></html>"""

QUANT_NAMES = ['bf16', 'fp8', 'nvfp4', 'groupwise']
KV_ARGS = {
    0: ['--kv-dtype', 'bf16'],
    1: ['--kv-dtype', 'int8'],
    2: ['--kv-dtype', 'fp8'],
    3: ['--kv-dtype', 'nvfp4'],
    4: ['--kv-layer-storage', '0-9:e8,10-15:nvfp4'],
}
CONVERT_DIR = None  # set from env


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
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != '/api/convert':
            self.send_response(404)
            self.end_headers()
            return
        n = int(self.headers.get('Content-Length', 0))
        req = json.loads(self.rfile.read(n))
        model = req['model']
        out = req['out']
        quant = QUANT_NAMES[int(req['quant'])]
        qmodel = req.get('qmodel') or ''
        log = []
        ok = False
        error = None
        try:
            script = 'convert_nvfp4.py' if quant == 'nvfp4' else 'convert.py'
            if quant == 'groupwise':
                # groupwise q4/q5 weights come pre-quantized; bf16 convert path
                script = 'convert.py'
            cmd = [sys.executable, os.path.join(CONVERT_DIR, 'qwen3_8_27b', script),
                   '--model', model, '--out', out]
            if quant == 'nvfp4':
                if not qmodel:
                    raise ValueError('NVFP4 需要量化源目录')
                cmd += ['--quantized-model', qmodel]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            log = p.stdout[-4000:]
            if p.stderr:
                log += '\n[stderr]\n' + p.stderr[-2000:]
            ok = p.returncode == 0 and os.path.exists(out)
            if not ok:
                error = 'exit %d' % p.returncode
        except Exception as e:
            error = str(e)
        body = json.dumps({'ok': ok, 'log': log, 'error': error}).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    global CONVERT_DIR
    CONVERT_DIR = os.environ.get('NINFER_CONVERT_DIR',
                                 os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              '..', '..', 'tools', 'convert'))
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8788
    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    print('NInfer Convert GUI: http://127.0.0.1:%d  (convert dir: %s)' % (port, CONVERT_DIR))
    srv.serve_forever()


if __name__ == '__main__':
    main()
