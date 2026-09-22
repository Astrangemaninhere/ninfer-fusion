#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NInfer RAG GUI — slider-minimal single-page app.
Query a local Chroma collection (bge-small-zh embeddings) with slider
controls: top-k, similarity floor, snippet window. Zero deps beyond the
existing RAG stack (sentence-transformers + chromadb).

Bilingual: every user-visible string goes through gui_i18n, under rag. keys.
HTML text is written as @@t.<key>@@ placeholders that render_page() substitutes
right before the bytes go out (same contract as serve_gui.py); the JS side reads
a rendered string table and never hardcodes UI text. ?lang=en flips the page,
POST /api/lang persists it.
"""
import html
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

sys.path.insert(0, '/home/user')

from gui_i18n import t, current, lang_from_request  # noqa: E402

EMBED_PATH = '/home/user/models/bge-small-zh'
DB_PATH = '/home/user/.rag_proxy/chromadb'
COLLECTION = 'wenxin_test'

PAGE = """<!doctype html><html lang="@@lang@@"><head><meta charset="utf-8">
<title>@@t.tab_title@@</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1a2332;--mut:#6b7686;--acc:#3b82f6;
      --line:#e5e8ee;--ok:#16a34a}
*{box-sizing:border-box;margin:0;padding:0}
body{font:14px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);
     color:var(--ink);display:flex;justify-content:center;padding:32px 16px}
.wrap{width:720px;max-width:100%}
h1{font-size:20px;font-weight:650;margin-bottom:2px}
.sub{color:var(--mut);font-size:12px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
      padding:20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(20,30,50,.05)}
.row{display:flex;align-items:center;gap:14px;margin-bottom:14px}
.row:last-child{margin-bottom:0}
.row label{width:110px;color:var(--mut);font-size:13px;flex:none}
input[type=range]{flex:1;accent-color:var(--acc);height:20px}
.val{width:64px;text-align:right;font-variant-numeric:tabular-nums;
     color:var(--ink);font-weight:600}
input[type=text],textarea{width:100%;border:1px solid var(--line);border-radius:10px;
      padding:10px 12px;font:inherit;color:var(--ink);outline:none}
textarea{min-height:88px;resize:vertical}
input[type=text]:focus,textarea:focus{border-color:var(--acc)}
button{background:var(--acc);color:#fff;border:0;border-radius:10px;
       padding:10px 22px;font:inherit;font-weight:600;cursor:pointer}
button:hover{filter:brightness(1.06)}
button:disabled{opacity:.55;cursor:default}
#status{color:var(--mut);font-size:12px;margin-left:auto}
#results{margin-top:4px}
.hit{background:var(--card);border:1px solid var(--line);border-radius:12px;
     padding:14px 16px;margin-bottom:10px}
.hit .meta{display:flex;gap:12px;color:var(--mut);font-size:12px;margin-bottom:6px}
.hit .dist{color:var(--ok);font-weight:600}
.hit .pages{color:var(--acc)}
.hit .text{font-size:13px;white-space:pre-wrap}
.tag{display:inline-block;background:#eef2f8;color:var(--mut);border-radius:6px;
     padding:1px 8px;font-size:11px;margin-left:6px}
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
  <div class="row"><label>@@t.lbl_topk@@</label>
    <input id="k" type="range" min="1" max="20" value="5">
    <div class="val" id="k_v">5</div></div>
  <div class="row"><label>@@t.lbl_floor@@</label>
    <input id="th" type="range" min="0" max="95" value="30" step="1">
    <div class="val" id="th_v">0.30</div></div>
  <div class="row"><label>@@t.lbl_snippet@@</label>
    <input id="ctx" type="range" min="0" max="3" value="1">
    <div class="val" id="ctx_v">±1@@t.seg_suffix@@</div></div>
  <textarea id="q" placeholder="@@t.ph_query@@"></textarea>
  <div class="row" style="margin-top:14px">
    <button id="go" onclick="run()">@@t.btn_search@@</button><span id="status"></span>
  </div>
</div>
<div id="results"></div>
<script>
const I18N=@@i18n@@;
const PAGE_LANG='@@lang@@';
const $=id=>document.getElementById(id);
function fill(s,kw){
  let out=s;
  for(const k in kw)out=out.split('{'+k+'}').join(kw[k]);
  return out;
}
for(const [id,fmt] of [['k',v=>v],['th',v=>(v/100).toFixed(2)],
                       ['ctx',v=>'±'+v+I18N.seg_suffix]]){
  $(id).oninput=e=>$(id+'_v').textContent=fmt(+e.target.value);
}
async function run(){
  const q=$('q').value.trim(); if(!q)return;
  $('go').disabled=true; $('status').textContent=I18N.status_searching;
  $('results').innerHTML='';
  try{
    const r=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({q,k:+$('k').value,th:+$('th').value/100,
                           ctx:+$('ctx').value,lang:PAGE_LANG})});
    const d=await r.json();
    $('status').textContent=d.error?fill(I18N.status_error,{err:d.error})
                                   :fill(I18N.status_results,{n:d.total,ms:d.ms});
    $('results').innerHTML=d.hits.map(h=>
      '<div class="hit"><div class="meta"><span class="dist">'+
      fill(I18N.score,{score:h.score.toFixed(3)})+'</span>'+
      '<span class="pages">'+h.pages.map(p=>fill(I18N.page,{p:p})).join(' · ')+
      '</span></div><div class="text">'+h.text+'</div></div>').join('');
  }catch(e){$('status').textContent=I18N.status_req_failed;}
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

# Keys the JS side reads out of the rendered table (dynamic status/snippet words).
JS_KEYS = ('seg_suffix', 'score', 'page', 'status_searching', 'status_error',
           'status_results', 'status_req_failed')
# An @@t.<name>@@ placeholder in PAGE resolves to the rag.<name> i18n key --
# except these few, which live under their own prefixes (shared with serve).
PLACEHOLDER_NS = {
    'misc_lang_label': 'misc.lang_label',
    'misc_lang_zh': 'misc.lang_zh',
    'misc_lang_en': 'misc.lang_en',
}

_embed = None
_col = None


def placeholder_key(name: str) -> str:
    """@@t.<name>@@ -> the i18n key it stands for."""
    return PLACEHOLDER_NS.get(name, 'rag.' + name)


def js_table() -> str:
    return json.dumps({k: t('rag.' + k) for k in JS_KEYS}, ensure_ascii=False)


def render_page(path: str = '/') -> str:
    lang_from_request(path)
    page = PAGE
    for key in sorted(set(re.findall(r'@@t\.([a-z0-9_]+)@@', PAGE))):
        page = page.replace('@@t.%s@@' % key, t(placeholder_key(key)))
    page = page.replace('@@i18n@@', js_table())
    page = page.replace('@@lang@@', current())
    return page


def load():
    global _embed, _col
    from sentence_transformers import SentenceTransformer
    import chromadb
    _embed = SentenceTransformer(EMBED_PATH)
    _col = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION)


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
                body = ('NInfer RAG: render error: %s' % e).encode('utf-8')
            self._raw(body, 'text/html; charset=utf-8')
        elif route == '/api/i18n':
            lang_from_request(self.path)
            self._json({'lang': current(),
                        'strings': {k: t('rag.' + k) for k in JS_KEYS}})
        else:
            self._json({'error': t('rag.status_req_failed')}, 404)

    def do_POST(self):
        route = self.path.split('?', 1)[0]
        n = int(self.headers.get('Content-Length', 0) or 0)
        try:
            req = json.loads(self.rfile.read(n) or b'{}')
        except Exception:
            req = {}
        # The page tells us which language it is showing so every answer below
        # (including the 404) comes back in that language.
        lang_from_request(self.path, req)
        if route == '/api/lang':
            self._json({'lang': current()})
            return
        if route != '/api/query':
            self._json({'error': t('rag.status_req_failed')}, 404)
            return
        import time
        t0 = time.time()
        q = req.get('q', '')
        k = int(req.get('k', 5))
        th = float(req.get('th', 0.3))
        ctx = int(req.get('ctx', 1))
        try:
            qe = _embed.encode([q], normalize_embeddings=True).tolist()
            r = _col.query(query_embeddings=qe, n_results=max(k, 20))
            hits = []
            docs = r['documents'][0]
            dists = r['distances'][0]
            for doc, dist in zip(docs, dists):
                # chroma default space is L2; map distance monotonically to [0,1]
                score = 1.0 / (1.0 + max(0.0, dist))
                if score < th:
                    continue
                pages = sorted(set(int(p) for p in re.findall(r'\[page (\d+)\]', doc)))
                text = re.sub(r'\[page \d+\]', '', doc)
                if ctx:
                    text = text[: len(text)]
                hits.append({'score': score, 'pages': pages, 'text': html.escape(text[:600])})
                if len(hits) >= k:
                    break
            self._json({'hits': hits, 'total': len(hits),
                        'ms': int((time.time() - t0) * 1000), 'error': None})
        except Exception as e:  # 模型没加载好也要给出人话, 不能 500 白屏
            self._json({'hits': [], 'total': 0, 'ms': int((time.time() - t0) * 1000),
                        'error': t('rag.status_error', err=e)})

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
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    threading.Thread(target=load, daemon=True).start()
    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    print(t('rag.console_ready', port=port))
    srv.serve_forever()


if __name__ == '__main__':
    main()
