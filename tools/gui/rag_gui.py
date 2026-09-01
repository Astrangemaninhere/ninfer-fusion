#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NInfer RAG GUI — slider-minimal single-page app.
Query a local Chroma collection (bge-small-zh embeddings) with slider
controls: top-k, similarity floor, snippet window. Zero deps beyond the
existing RAG stack (sentence-transformers + chromadb).
"""
import html
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, '/home/user')

EMBED_PATH = '/home/user/models/bge-small-zh'
DB_PATH = '/home/user/.rag_proxy/chromadb'
COLLECTION = 'wenxin_test'

PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>NInfer RAG</title>
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
</style></head><body><div class="wrap">
<h1>NInfer RAG</h1>
<div class="sub">本地向量检索（bge-small-zh · Chroma · wenxin_test）— 滑块简约版</div>
<div class="card">
  <div class="row"><label>top-k</label>
    <input id="k" type="range" min="1" max="20" value="5">
    <div class="val" id="k_v">5</div></div>
  <div class="row"><label>相似度下限</label>
    <input id="th" type="range" min="0" max="95" value="30" step="1">
    <div class="val" id="th_v">0.30</div></div>
  <div class="row"><label>片段扩展</label>
    <input id="ctx" type="range" min="0" max="3" value="1">
    <div class="val" id="ctx_v">±1 段</div></div>
  <textarea id="q" placeholder="输入查询…"></textarea>
  <div class="row" style="margin-top:14px">
    <button id="go" onclick="run()">检索</button><span id="status"></span>
  </div>
</div>
<div id="results"></div>
<script>
const $=id=>document.getElementById(id);
for(const [id,fmt] of [['k',v=>v],['th',v=>(v/100).toFixed(2)],['ctx',v=>'±'+v+' 段']]){
  $(id).oninput=e=>$(id+'_v').textContent=fmt(+e.target.value);
}
async function run(){
  const q=$('q').value.trim(); if(!q)return;
  $('go').disabled=true; $('status').textContent='检索中…'; $('results').innerHTML='';
  try{
    const r=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({q,k:+$('k').value,th:+$('th').value/100,ctx:+$('ctx').value})});
    const d=await r.json();
    $('status').textContent=d.error?('错误: '+d.error):(d.total+' 条结果 · '+d.ms+' ms');
    $('results').innerHTML=d.hits.map(h=>
      '<div class="hit"><div class="meta"><span class="dist">相似 '+h.score.toFixed(3)+'</span>'+
      '<span class="pages">'+h.pages.map(p=>'page '+p).join(' · ')+'</span></div>'+
      '<div class="text">'+h.text+'</div></div>').join('');
  }catch(e){$('status').textContent='请求失败';}
  $('go').disabled=false;
}
</script></body></html>"""

_embed = None
_col = None


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
        if self.path != '/api/query':
            self.send_response(404)
            self.end_headers()
            return
        n = int(self.headers.get('Content-Length', 0))
        req = json.loads(self.rfile.read(n))
        import time
        t0 = time.time()
        q = req.get('q', '')
        k = int(req.get('k', 5))
        th = float(req.get('th', 0.3))
        ctx = int(req.get('ctx', 1))
        qe = _embed.encode([q], normalize_embeddings=True).tolist()
        r = _col.query(query_embeddings=qe, n_results=max(k, 20))
        hits = []
        docs = r['documents'][0]
        dists = r['distances'][0]
        for doc, dist in zip(docs, dists):
            score = max(0.0, 1.0 - dist)
            if score < th:
                continue
            pages = sorted(set(int(p) for p in re.findall(r'\[page (\d+)\]', doc)))
            text = re.sub(r'\[page \d+\]', '', doc)
            if ctx:
                text = text[: len(text)]
            hits.append({'score': score, 'pages': pages, 'text': html.escape(text[:600])})
            if len(hits) >= k:
                break
        body = json.dumps({'hits': hits, 'total': len(hits), 'ms': int((time.time() - t0) * 1000),
                           'error': None}).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    threading.Thread(target=load, daemon=True).start()
    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    print('NInfer RAG GUI: http://127.0.0.1:%d' % port)
    srv.serve_forever()


if __name__ == '__main__':
    main()
