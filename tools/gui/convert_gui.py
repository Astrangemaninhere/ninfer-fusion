#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NInfer 模型转换 GUI — slider-minimal single-page app.
Wraps tools/convert/qwen3_8_27b: BF16 convert and NVFP4 quantized convert.
Sliders: quant mode, KV cache tier, context ceiling, batch width.

Bilingual: every user-visible string goes through gui_i18n, under conv. keys.
HTML text is written as @@t.<key>@@ placeholders that render_page() substitutes
right before the bytes go out (same contract as serve_gui.py); the JS side reads
a rendered string table and never hardcodes UI text. ?lang=en flips the page,
POST /api/lang persists it.
"""
import json
import os
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gui_i18n import t, current, lang_from_request  # noqa: E402

PAGE = """<!doctype html><html lang="@@lang@@"><head><meta charset="utf-8">
<title>@@t.tab_title@@</title>
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
.langbar{display:flex;align-items:center;gap:8px;justify-content:flex-end;margin:-8px 0 10px}
.langbar span{color:var(--mut);font-size:12px}
.langbtn{background:#eef2f8;color:var(--mut);padding:4px 12px;font-size:12px;border-radius:999px}
.langbtn.on{background:var(--acc);color:#fff}
</style></head><body><div class="wrap">
<h1>@@t.title@@</h1>
<div class="sub">@@t.sub@@</div>
<div class="langbar"><span>@@t.misc_lang_label@@</span>
  <button class="langbtn" id="lang_zh" onclick="setLang('zh')">@@t.misc_lang_zh@@</button>
  <button class="langbtn" id="lang_en" onclick="setLang('en')">@@t.misc_lang_en@@</button></div>
<div class="card">
  <div class="row"><label>@@t.lbl_model@@</label>
    <input id="model" type="text" placeholder="@@t.ph_model@@"></div>
  <div class="row"><label>@@t.lbl_qmodel@@</label>
    <input id="qmodel" type="text" placeholder="@@t.ph_qmodel@@"></div>
  <div class="row"><label>@@t.lbl_out@@</label>
    <input id="out" type="text" placeholder="@@t.ph_out@@"></div>
  <div class="row"><label>@@t.lbl_quant@@</label>
    <input id="quant" type="range" min="0" max="3" value="0" step="1">
    <div class="val" id="quant_v">@@t.quant_0@@</div></div>
  <div class="row"><label>@@t.lbl_kv@@</label>
    <input id="kv" type="range" min="0" max="4" value="3" step="1">
    <div class="val" id="kv_v">@@t.kv_3@@</div></div>
  <div class="row"><label>@@t.lbl_ctx@@</label>
    <input id="ctx" type="range" min="3" max="8" value="6" step="1">
    <div class="val" id="ctx_v">64k</div></div>
  <div class="row"><label>@@t.lbl_batch@@</label>
    <input id="batch" type="range" min="1" max="8" value="1" step="1">
    <div class="val" id="batch_v">1</div></div>
  <div class="row"><button id="go" onclick="run()">@@t.btn_go@@</button>
    <span id="status"></span></div>
</div>
<div class="card"><div id="log">@@t.log_ready@@</div></div>
<script>
const I18N=@@i18n@@;
const PAGE_LANG='@@lang@@';
const $=id=>document.getElementById(id);
const QUANT=[I18N.quant_0,I18N.quant_1,I18N.quant_2,I18N.quant_3];
const KV=[I18N.kv_0,I18N.kv_1,I18N.kv_2,I18N.kv_3,I18N.kv_4];
const CTX=[8,16,32,64,128,256];
function fill(s,kw){
  let out=s;
  for(const k in kw)out=out.split('{'+k+'}').join(kw[k]);
  return out;
}
$('quant').oninput=e=>$('quant_v').textContent=QUANT[+e.target.value];
$('kv').oninput=e=>$('kv_v').textContent=KV[+e.target.value];
$('ctx').oninput=e=>$('ctx_v').textContent=CTX[+e.target.value-3]+'k';
$('batch').oninput=e=>$('batch_v').textContent=e.target.value;
async function run(){
  const model=$('model').value.trim(), out=$('out').value.trim();
  if(!model||!out){$('status').textContent=I18N.status_need_paths;return}
  $('go').disabled=true;$('status').textContent=I18N.status_running;
  $('log').textContent='';
  try{
    const r=await fetch('/api/convert',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model,out,quant:+$('quant').value,qmodel:$('qmodel').value.trim(),
                           kv:+$('kv').value,ctx:+$('ctx').value,batch:+$('batch').value,
                           lang:PAGE_LANG})});
    const d=await r.json();
    $('log').textContent=d.log||'';
    $('status').textContent=d.ok?I18N.status_done
                               :fill(I18N.status_failed,{err:d.error||''});
  }catch(e){$('status').textContent=I18N.status_req_failed}
  $('go').disabled=false;
}
async function setLang(l){
  try{
    const r=await fetch('/api/lang',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({lang:l})});
    const d=await r.json();
    localStorage.setItem('ninferLang',d.lang||l);
  }catch(e){localStorage.setItem('ninferLang',l);}
  location.reload();
}
(function markLang(){
  const b=$('lang_'+PAGE_LANG);
  if(b)b.classList.add('on');
  const saved=localStorage.getItem('ninferLang');
  if(saved&&saved!==PAGE_LANG&&!sessionStorage.getItem('langSync')){
    sessionStorage.setItem('langSync','1');
    fetch('/api/lang',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({lang:saved})}).then(()=>location.reload()).catch(()=>{});
  }else if(saved===PAGE_LANG){
    sessionStorage.removeItem('langSync');
  }
})();
</script></body></html>"""

QUANT_NAMES = ['bf16', 'fp8', 'nvfp4', 'groupwise']
KV_ARGS = {
    0: ['--kv-dtype', 'bf16'],
    1: ['--kv-dtype', 'int8'],
    2: ['--kv-dtype', 'fp8'],
    3: ['--kv-dtype', 'nvfp4'],
    4: ['--kv-layer-storage', '0-9:rk4v4,10-15:nvfp4'],
}
# Keys the JS side reads out of the rendered table (status words + slider labels).
JS_KEYS = ('quant_0', 'quant_1', 'quant_2', 'quant_3',
           'kv_0', 'kv_1', 'kv_2', 'kv_3', 'kv_4',
           'status_need_paths', 'status_running', 'status_done', 'status_failed',
           'status_req_failed')
# An @@t.<name>@@ placeholder in PAGE resolves to the conv.<name> i18n key --
# except these few, which live under their own prefixes (shared with serve).
PLACEHOLDER_NS = {
    'misc_lang_label': 'misc.lang_label',
    'misc_lang_zh': 'misc.lang_zh',
    'misc_lang_en': 'misc.lang_en',
}
CONVERT_DIR = None  # set from env


def placeholder_key(name: str) -> str:
    """@@t.<name>@@ -> the i18n key it stands for."""
    return PLACEHOLDER_NS.get(name, 'conv.' + name)


def js_table() -> str:
    return json.dumps({k: t('conv.' + k) for k in JS_KEYS}, ensure_ascii=False)


def render_page(path: str = '/') -> str:
    lang_from_request(path)
    page = PAGE
    for key in sorted(set(re.findall(r'@@t\.([a-z0-9_]+)@@', PAGE))):
        page = page.replace('@@t.%s@@' % key, t(placeholder_key(key)))
    page = page.replace('@@i18n@@', js_table())
    page = page.replace('@@lang@@', current())
    return page


def build_cmd(req):
    """Slider state -> converter argv. Unchanged from the pre-i18n version."""
    quant = QUANT_NAMES[int(req['quant'])]
    script = 'convert_nvfp4.py' if quant == 'nvfp4' else 'convert.py'
    # groupwise q4/q5 weights come pre-quantized; bf16 convert path
    cmd = [sys.executable, os.path.join(CONVERT_DIR, 'qwen3_8_27b', script),
           '--model', req['model'], '--out', req['out']]
    if quant == 'nvfp4':
        if not req.get('qmodel'):
            raise ValueError(t('conv.err_need_qmodel'))
        cmd += ['--quantized-model', req['qmodel']]
    return cmd


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        route = self.path.split('?', 1)[0]
        if route in ('/', '/index.html'):
            try:
                lang_from_request(self.path)
                body = render_page(self.path).encode('utf-8')
            except Exception as e:  # 渲染炸了要看得见, 不能白屏
                body = ('NInfer Convert: render error: %s' % e).encode('utf-8')
            self._raw(body, 'text/html; charset=utf-8')
        elif route == '/api/i18n':
            lang_from_request(self.path)
            self._json({'lang': current(),
                        'strings': {k: t('conv.' + k) for k in JS_KEYS}})
        else:
            self._json({'error': t('conv.status_req_failed')}, 404)

    def do_POST(self):
        route = self.path.split('?', 1)[0]
        if route == '/api/lang':
            n = int(self.headers.get('Content-Length', 0) or 0)
            try:
                req = json.loads(self.rfile.read(n) or b'{}')
            except Exception:
                req = {}
            self._json({'lang': lang_from_request(self.path, req)})
            return
        n = int(self.headers.get('Content-Length', 0) or 0)
        try:
            req = json.loads(self.rfile.read(n) or b'{}')
        except Exception:
            self._json({'ok': False, 'log': [], 'error': t('conv.status_req_failed')})
            return
        # The page tells us which language it is showing so every answer below
        # (including the 404) comes back in that language.
        lang_from_request(self.path, req)
        if route != '/api/convert':
            self._json({'error': t('conv.status_req_failed')}, 404)
            return
        log = ''
        ok = False
        error = None
        try:
            cmd = build_cmd(req)
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            log = p.stdout[-4000:]
            if p.stderr:
                log += '\n[stderr]\n' + p.stderr[-2000:]
            ok = p.returncode == 0 and os.path.exists(req['out'])
            if not ok:
                error = t('conv.err_exit', code=p.returncode)
        except Exception as e:
            error = str(e)
        self._json({'ok': ok, 'log': log, 'error': error})

    def _raw(self, body, ctype):
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, d, code=200):
        body = json.dumps(d, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
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
    print(t('conv.console_ready', port=port, dir=CONVERT_DIR))
    srv.serve_forever()


if __name__ == '__main__':
    main()
