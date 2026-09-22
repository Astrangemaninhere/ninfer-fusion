#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NInfer serve console - slider-minimal single page plus an engine-driven
"all parameters" panel.

Two layers, on purpose:
  * the slider panel (newbie mode) is untouched: same knobs, same order, and
    the same command line as before.
  * a collapsible "all parameters" section is rendered *from the registry*
    (serve_params.json, produced by extract_serve_params.py). Engine flags
    therefore show up here automatically once the extractor is re-run.
    Nothing in that section reaches the command line unless the user filled
    it in; an untouched panel yields a byte-identical command line.

Bilingual: every user-visible string is an i18n entry under the serve.* prefix,
resolved through gui_i18n; the JS side reads a rendered string table and never
hardcodes UI text.
"""
import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gui_i18n import t, current, lang_from_request  # noqa: E402

try:
    import gui_tips  # noqa: E402
except Exception:  # 没有讲解字典也要能跑
    gui_tips = None

REGISTRY_PATH = _HERE / 'serve_params.json'
GROUP_ORDER_FALLBACK = ('core', 'kv', 'speculative', 'sampling', 'thinking', 'media',
                        'perf', 'network', 'diagnostics', 'advanced', 'ungrouped')

PAGE = """<!doctype html><html lang="@@lang@@"><head><meta charset="utf-8">
<title>@@t.title@@</title>
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
input[type=text],input[type=number]{flex:1;border:1px solid var(--line);border-radius:10px;
      padding:8px 12px;font:inherit;outline:none;min-width:0}
.btnrow{display:flex;gap:10px;margin-top:4px}
button{border:0;border-radius:10px;padding:10px 22px;font:inherit;font-weight:600;cursor:pointer}
#start{background:var(--acc);color:#fff}
#stop{background:var(--bad);color:#fff;display:none}
#log{background:#0f172a;color:#cbd5e1;border-radius:12px;padding:14px;
     font:12px/1.5 ui-monospace,Consolas,monospace;height:300px;overflow:auto;
     white-space:pre-wrap}
#gpu{font:12px ui-monospace,monospace;color:var(--mut);margin-top:10px}
.langbar{display:flex;align-items:center;gap:8px;justify-content:flex-end;margin:-8px 0 10px}
.langbar span{color:var(--mut);font-size:12px}
.langbtn{background:#eef2f8;color:var(--mut);padding:4px 12px;font-size:12px;border-radius:999px}
.langbtn.on{background:var(--acc);color:#fff}
details.card>summary{cursor:pointer;font-weight:650;font-size:15px;list-style:none;
      display:flex;align-items:center;gap:10px;flex-wrap:wrap}
details.card>summary::-webkit-details-marker{display:none}
details.card>summary::before{content:'\\25B8';color:var(--mut);transition:transform .15s}
details.card[open]>summary::before{transform:rotate(90deg)}
.count{font-size:11px;font-weight:600;color:#b45309;background:#fef3c7;
       padding:2px 10px;border-radius:999px}
.advhint{color:var(--mut);font-size:12px;margin:10px 0 4px}
.grp{border-top:1px solid var(--line);margin-top:12px;padding-top:10px}
.grp h3{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);
        margin-bottom:8px}
.advrow{margin-bottom:9px}
.advrow label.aname{width:230px;font-family:ui-monospace,Consolas,monospace;font-size:12px;
        color:var(--ink);flex:none;overflow-wrap:anywhere}
.advrow input{font-size:12px;padding:6px 10px}
.advrow input[type=checkbox]{flex:none;width:16px;height:16px;accent-color:var(--acc)}
.advrow input:disabled{background:#f4f5f7;color:var(--mut)}
.alock{color:var(--mut);font-size:11px;font-style:italic;flex:none}
.akey{margin-left:auto;color:var(--mut);font-size:11px;font-family:ui-monospace,monospace}
.badge.meas{background:#dcfce7;color:#15803d}
.badge.conf{background:#e0e7ff;color:#4338ca}
.spechd{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.spechd h2{font-size:15px;font-weight:650}
.specgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px 18px;margin:12px 0 4px}
.specgrid>div{display:flex;justify-content:space-between;gap:8px;border-bottom:1px dashed var(--line);
      padding-bottom:4px}
.specgrid .k{color:var(--mut);font-size:12px}
.specgrid b{font-variant-numeric:tabular-nums}
.specrow{display:flex;align-items:center;gap:12px;margin-top:12px}
.specrow .k{color:var(--mut);font-size:12px;flex:none;width:96px}
#spec_pos_line{font:12px ui-monospace,Consolas,monospace;font-variant-numeric:tabular-nums}
.posbars{display:flex;flex-wrap:wrap;gap:5px;align-items:flex-end;flex:1;min-height:30px}
.posbars .pb{display:flex;flex-direction:column;align-items:center;gap:2px;cursor:default}
.posbars .pb i{display:block;width:18px;background:var(--acc);border-radius:3px 3px 0 0}
.posbars .pb s{text-decoration:none;font-size:10px;color:var(--mut);
      font-variant-numeric:tabular-nums}
.posbars .ph{color:var(--mut);font-size:12px}
</style></head><body><div class="wrap">
<h1>NInfer Serve <span class="badge idle" id="badge">@@t.badge_idle@@</span></h1>
<div class="sub">@@t.subtitle@@</div>
<div class="langbar"><span>@@t.language@@</span>
  <button class="langbtn" id="lang_zh" onclick="setLang('zh')">@@t.lang_zh@@</button>
  <button class="langbtn" id="lang_en" onclick="setLang('en')">@@t.lang_en@@</button></div>
<div class="card">
  <div class="row"><label>@@t.lbl_model@@</label>
    <input id="model" type="text" value="/home/user/models/qwen3_8_27b_nvfp4.ninfer"></div>
  <div class="row"><label>@@t.lbl_kv@@</label>
    <input id="kv" type="range" min="0" max="4" value="3" step="1">
    <div class="val" id="kv_v">@@t.kv_3@@</div></div>
  <div class="row"><label>@@t.lbl_ctx@@</label>
    <input id="ctx" type="range" min="3" max="8" value="5" step="1">
    <div class="val" id="ctx_v">32k</div></div>
  <div class="row"><label>@@t.lbl_spec@@</label>
    <input id="spec" type="range" min="0" max="3" value="0" step="1">
    <div class="val" id="spec_v">@@t.spec_0@@</div></div>
  <div class="row"><label>@@t.lbl_draft@@</label>
    <input id="draft" type="range" min="1" max="5" value="3" step="1">
    <div class="val" id="draft_v">3</div></div>
  <div class="row"><label>@@t.lbl_conc@@</label>
    <input id="conc" type="range" min="1" max="4" value="1" step="1">
    <div class="val" id="conc_v">1</div></div>
  <div class="row"><label>@@t.lbl_vis@@</label>
    <input id="vis" type="range" min="0" max="1" value="0" step="1">
    <div class="val" id="vis_v">@@t.val_off@@</div></div>
  <div class="row"><label>@@t.lbl_cold@@</label>
    <input id="cold" type="range" min="0" max="3" value="0" step="1">
    <div class="val" id="cold_v">@@t.cold_0@@</div></div>
  <div class="row"><label>@@t.lbl_hotwin@@</label>
    <input id="hotwin" type="range" min="0" max="8" value="4" step="1">
    <div class="val" id="hotwin_v">4k</div>
    <span style="color:var(--mut);font-size:11px">@@t.hotwin_note@@</span></div>
  <div class="row"><label>@@t.lbl_kvlayer@@</label>
    <input id="kvlayer" type="text" value="" placeholder="@@t.ph_kvlayer@@">
  </div>
  <div class="row"><label>@@t.lbl_graph@@</label>
    <input id="graph" type="range" min="0" max="1" value="1" step="1">
    <div class="val" id="graph_v">@@t.val_on@@</div></div>
  <div class="row"><label>@@t.lbl_port@@</label>
    <input id="port" type="text" value="8000"></div>
  <div class="btnrow">
    <button id="start" onclick="ctrl('start')">@@t.btn_start@@</button>
    <button id="stop" onclick="ctrl('stop')">@@t.btn_stop@@</button>
  </div>
</div>
<div class="card" id="speccard">
  <div class="spechd"><h2>@@t.spec_title@@</h2>
    <span class="badge idle" id="spec_badge">@@t.spec_none@@</span>
    <span class="akey" id="spec_src"></span></div>
  <div class="specgrid">
    <div><span class="k">@@t.spec_tier@@</span><b id="spec_tier">-</b></div>
    <div><span class="k">@@t.spec_window@@</span><b id="spec_window">-</b></div>
    <div><span class="k">@@t.spec_rounds@@</span><b id="spec_rounds">-</b></div>
    <div><span class="k">@@t.spec_rate@@</span><b id="spec_rate">-</b></div>
    <div><span class="k">@@t.spec_len@@</span><b id="spec_len">-</b></div>
    <div><span class="k">@@t.spec_fallback@@</span><b id="spec_fallback">-</b></div>
    <div><span class="k">@@t.spec_drafted@@</span><b id="spec_drafted">-</b></div>
    <div><span class="k">@@t.spec_accepted@@</span><b id="spec_accepted">-</b></div>
  </div>
  <div class="specrow"><span class="k">@@t.spec_pos@@</span>
    <b id="spec_pos_line">-</b></div>
  <div class="posbars" id="spec_pos"><span class="ph">@@t.spec_pos_none@@</span></div>
  <div class="advhint">@@t.spec_hint@@</div>
</div>
<details class="card" id="advbox">
  <summary>@@t.adv_summary@@ <span class="count">@@registry@@</span></summary>
  <div class="advhint">@@t.adv_hint@@</div>
  @@adv@@
</details>
<div class="card"><div id="log">@@t.log_ready@@</div><div id="gpu"></div></div>
<script>
const I18N=@@i18n@@;
const PAGE_LANG='@@lang@@';
const $=id=>document.getElementById(id);
const KV=[I18N.kv_0,I18N.kv_1,I18N.kv_2,I18N.kv_3,I18N.kv_4];
const CTX=[8,16,32,64,128,256];
const SPEC=[I18N.spec_0,I18N.spec_1,I18N.spec_2,I18N.spec_3];
const DASH=I18N.spec_na;
function num(v){return (v===null||v===undefined)?DASH:v;}
function pct(v){return (v===null||v===undefined)?DASH:(100*v).toFixed(2)+'%';}
function paintSpec(s){
  s=s||{};
  const b=$('spec_badge');
  const meas=!!s.backend;
  b.textContent=meas?I18N.spec_measured:(s.configured?I18N.spec_configured:I18N.spec_none);
  b.className='badge '+(meas?'meas':(s.configured?'conf':'idle'));
  $('spec_src').textContent=(s.sources&&s.sources.length)
      ?I18N.spec_src+': '+s.sources.join(' + '):'';
  const tier=s.backend||s.configured||'';
  $('spec_tier').textContent=tier?(tier+(s.draft_tokens?' d'+s.draft_tokens:'')):DASH;
  $('spec_window').textContent=num(s.draft_window);
  $('spec_rounds').textContent=num(s.rounds);
  $('spec_rate').textContent=pct(s.accept_rate);
  $('spec_len').textContent=(s.accept_len===null||s.accept_len===undefined)
      ?DASH:(+s.accept_len).toFixed(2)+' '+I18N.spec_len_unit;
  $('spec_fallback').textContent=num(s.fallback_steps);
  $('spec_drafted').textContent=num(s.drafted);
  $('spec_accepted').textContent=num(s.accepted);
  const p=(s.by_position&&s.by_position.length)?s.by_position:[];
  $('spec_pos_line').textContent=p.length
      ?p.join(',')+'  /  '+I18N.spec_rounds_word+'='+num(s.rounds):DASH;
  const host=$('spec_pos');
  if(!p.length){host.innerHTML='';host.appendChild(posNone());return;}
  const mx=Math.max.apply(null,p.concat([1]));
  host.textContent='';
  p.forEach(function(v,i){
    const rate=(s.rounds&&s.rounds>0)?(v/s.rounds):null;
    const cell=document.createElement('span');
    cell.className='pb';
    cell.title='p'+i+': '+v+(rate===null?'':' / '+rate.toFixed(3)+I18N.spec_per_round);
    const bar=document.createElement('i');
    bar.style.height=Math.round(3+27*v/mx)+'px';
    const lab=document.createElement('s');
    lab.textContent=String(i);
    cell.appendChild(bar);cell.appendChild(lab);host.appendChild(cell);
  });
}
function posNone(){
  const el=document.createElement('span');
  el.className='ph';
  el.textContent=I18N.spec_pos_none;
  return el;
}
const COLD=[I18N.cold_0,I18N.cold_1,I18N.cold_2,I18N.cold_3];
const HOT=[0,128,512,1024,2048,4096,8192,16384,32768];
$('kv').oninput=e=>$('kv_v').textContent=KV[+e.target.value];
$('ctx').oninput=e=>$('ctx_v').textContent=CTX[+e.target.value-3]+'k';
$('spec').oninput=e=>$('spec_v').textContent=SPEC[+e.target.value];
$('draft').oninput=e=>$('draft_v').textContent=e.target.value;
$('conc').oninput=e=>$('conc_v').textContent=e.target.value;
$('vis').oninput=e=>$('vis_v').textContent=+e.target.value?I18N.val_on:I18N.val_off;
$('cold').oninput=e=>$('cold_v').textContent=COLD[+e.target.value];
$('hotwin').oninput=e=>{const h=HOT[+e.target.value];
  $('hotwin_v').textContent=h?(h>=1024?(h/1024)+'k':h):I18N.hot_default;};
$('graph').oninput=e=>$('graph_v').textContent=+e.target.value?I18N.val_on:I18N.val_off;
function collectAdv(){
  const o={};
  document.querySelectorAll('[data-flag]').forEach(function(e){
    if(e.disabled)return;
    if(e.dataset.type==='bool'){ if(e.checked)o[e.dataset.flag]=true; return; }
    const v=(e.value||'').trim();
    if(v)o[e.dataset.flag]=v;
  });
  return o;
}
function paint(d){
  $('badge').textContent=d.running?I18N.badge_run:I18N.badge_idle;
  $('badge').className='badge '+(d.running?'run':'idle');
  $('start').style.display=d.running?'none':'';
  $('stop').style.display=d.running?'':'none';
}
async function ctrl(a){
  const r=await fetch('/api/'+a,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model:$('model').value.trim(),kv:+$('kv').value,
      ctx:+$('ctx').value,spec:+$('spec').value,draft:+$('draft').value,
      conc:+$('conc').value,vis:+$('vis').value,cold:+$('cold').value,
      hotwin:+$('hotwin').value,kvlayer:$('kvlayer').value.trim(),
      graph:+$('graph').value,port:$('port').value.trim(),adv:collectAdv()})});
  const d=await r.json();
  paint(d);paintSpec(d.spec);
  if(d.log){$('log').textContent=d.log;$('log').scrollTop=$('log').scrollHeight;}
  if(d.error)$('log').textContent=d.error;
}
async function poll(){
  try{
    const r=await fetch('/api/state');const d=await r.json();
    paint(d);paintSpec(d.spec);
    $('log').textContent=d.log||I18N.log_none;
    $('log').scrollTop=$('log').scrollHeight;
    if(d.gpu)$('gpu').textContent='GPU: '+d.gpu;
  }catch(e){}
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
setInterval(poll,1500);poll();
</script></body></html>"""

# 滑块面板"占用"的旗标: 全参数面板里仍然列出(可见性), 但置灰, 避免同一旗标出现两次.
BASIC_FLAGS = frozenset([
    '--port', '--kv-dtype', '--kv-layer-storage', '--max-context', '--spec',
    '--draft-tokens', '--lm-head-draft', '--max-concurrency', '--vision',
    '--cold-policy', '--cold-keep-tokens', '--no-cuda-graph',
])
# 纯帮助开关, 不是服务参数, 不在面板里渲染.
GUI_SKIP = frozenset(['--help'])
# gui_tips.TIPS 是按 ninfer-gui.py 的控件 id 写的, 这里把旗标名映射过去复用.
TIP_ALIAS = {
    'model': 'smodel', 'max-context': 'sctx', 'spec': 'sspec', 'draft-tokens': 'sdt',
    'cold-policy': 'scold', 'cold-disk-path': 'scoldpath', 'cold-disk-bytes': 'scoldgb',
    'kv-layer-storage': 'sktiers', 'kv-dtype': 'snvfp4mode',
}
# JS 侧要用的词条: t() 的结果渲染进页面, JS 只读这张表.
JS_KEYS = ('badge_idle', 'badge_run', 'val_on', 'val_off', 'hot_default', 'log_none',
           'kv_0', 'kv_1', 'kv_2', 'kv_3', 'kv_4',
           'spec_0', 'spec_1', 'spec_2', 'spec_3',
           'cold_0', 'cold_1', 'cold_2', 'cold_3',
           # 投机卡片: JS 侧只读这张表, 没有任何硬编码文案
           'spec_none', 'spec_configured', 'spec_measured', 'spec_na', 'spec_len_unit',
           'spec_rounds_word', 'spec_pos_none', 'spec_src', 'spec_per_round')

KV_ARGS = {
    0: ['--kv-dtype', 'bf16'],
    1: ['--kv-dtype', 'int8'],
    2: ['--kv-dtype', 'fp8'],
    3: ['--kv-dtype', 'nvfp4'],
    4: ['--kv-layer-storage', '0-9:rk4v4,10-15:nvfp4'],
}
COLD_POLICY = ['', 'window', 'host', 'disk']
HOT_TOKENS = [0, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768]
CTX_KB = [8192, 16384, 32768, 65536, 131072, 262144]
SPEC_ARGS = ['', '--spec mtp', '--spec dflash', '--spec dflash2']
SERVE_BIN = '/home/user/ninfer-fusion/build/apps/ninfer-serve'
DEFAULT_MODEL = '/home/user/models/qwen3_8_27b_nvfp4.ninfer'
# 页面刚打开时滑块/输入框的取值(自检用它复现"什么都没动"的请求).
DEFAULT_REQ = {
    'model': DEFAULT_MODEL, 'kv': 3, 'ctx': 5, 'spec': 0, 'draft': 3, 'conc': 1,
    'vis': 0, 'cold': 0, 'hotwin': 4, 'kvlayer': '', 'graph': 1, 'port': '8000',
}
_proc = None
_log = ''
_session = {}
_sim_pin = ''

# ---------------------------------------------------------------------------
# ROUTE D -- THE TEST-ONLY ARCHITECTURE SIMULATOR MUST NOT REACH THE ENGINE THIS GUI SPAWNS.
#
# src/core/arch_sim.h requires THREE keys before it will report a lower compute capability: the
# build-time NINFER_ARCH_SIM_ENABLED (root CMakeLists.txt option NINFER_ENABLE_ARCH_SIM, OFF by
# default), and the environment PAIR NINFER_SIM_ARCH + NINFER_SIM_ARCH_ACK. The engine refuses to
# simulate without the ACK, and says why -- that refusal is the point of the pair.
#
# The hazard this block closes: a GUI that builds its child's environment with
# `dict(os.environ, ...)` carries that pair straight through from whatever shell launched the
# GUI. Nobody declares it, nothing displays it, and the pane beside the log still shows the REAL
# card's nvidia-smi badge -- so a simulated run is indistinguishable from a real one to the
# operator. That is the one thing this project forbids: a build configuration is not a support
# claim, and only a probe on real hardware is evidence of support.
#
# So the pair is dropped BY NAME, and the drop is RECORDED rather than silent: it appears in
# /api/state under 'sim' and as a line in the log. A silent drop would be its own defect in the
# other direction -- an operator who exported NINFER_SIM_ARCH expecting a simulated run would get
# a real one and never be told. There is deliberately no ambient path back in: this GUI has no
# request field that sets either key, so a simulated run cannot be started from it at all.
# ---------------------------------------------------------------------------
SIM_ENV_KEYS = ('NINFER_SIM_ARCH', 'NINFER_SIM_ARCH_ACK')

# The marker sim_banner() prints (src/core/arch_sim.h). It is printed ONCE PER PROCESS, at the
# first arch-view call, i.e. at the very start of a serve run -- while the log pane keeps only a
# trailing window. Without the pin below, a few kB of ordinary output evicts the only indication
# that the numbers on screen came from a simulated architecture.
SIM_MARKER = 'SIMULATED ARCHITECTURE'


def engine_env():
    """The environment for the spawned engine: ours, minus the test-only simulator pair.

    Returns (env, dropped), where `dropped` names the keys that WERE set in this process -- i.e.
    the operator's shell was simulating, and this function is the reason the engine is not.
    """
    env = dict(os.environ)
    dropped = sorted(k for k in SIM_ENV_KEYS if k in env)
    for k in SIM_ENV_KEYS:
        env.pop(k, None)
    env['LD_LIBRARY_PATH'] = '/usr/local/cuda-13.3/lib64'
    return env, dropped


def log_append(chunk):
    """Append engine output to the shown log, pinning any SIMULATED marker against eviction.

    The reader keeps a trailing window, which is right for ordinary output and wrong for the one
    line that says the run is not what it looks like. A marker seen once is hoisted into _sim_pin
    and re-prepended on every eviction, so it survives any volume of output rather than the first
    few kB of it.
    """
    global _log, _sim_pin
    if SIM_MARKER in chunk:
        _sim_pin = _sim_pin + chunk if _sim_pin else chunk
    _log += chunk
    if len(_log) > 12000:
        _log = _sim_pin + _log[-8000:]


def _shown_log():
    """What the log pane is served: the tail, always preceded by any pinned SIMULATED marker.

    A plain `_log[-LOG_TAIL_CHARS:]` is right for ordinary output and wrong for that one line, so
    the pin is honoured here as well as on eviction. BOTH windows have to respect it: the reader's
    12000/8000 and this one.
    """
    tail = _log[-LOG_TAIL_CHARS:]
    return _sim_pin + tail if _sim_pin else tail


def _sim_state():
    """Whether the engine this GUI spawned is simulating: a fact to read, not to infer."""
    return dict((_session or {}).get('sim') or {'requested': False, 'dropped': []})


# --------------------------------------------------------------------------
# 引擎注册表
# --------------------------------------------------------------------------
def load_registry(path=None):
    """读 extract_serve_params.py 产出的注册表; 读不到就降级成空表(界面照常能用)."""
    p = Path(path or REGISTRY_PATH)
    empty = {'total': 0, 'groups': {}, 'ungrouped': [], 'source': '',
             'group_order': list(GROUP_ORDER_FALLBACK)}
    try:
        reg = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return empty
    if not isinstance(reg, dict) or not isinstance(reg.get('groups'), dict):
        return empty
    reg.setdefault('ungrouped', [])
    if not isinstance(reg.get('group_order'), list) or not reg['group_order']:
        reg['group_order'] = list(GROUP_ORDER_FALLBACK)
    if not reg.get('total'):
        reg['total'] = len(adv_order(reg))
    return reg


def adv_order(reg):
    """全参数面板/命令行的稳定顺序: 先按注册表的分组顺序, 组内保持注册表自身的次序."""
    out = []
    for g in list(reg.get('group_order') or []) + ['ungrouped']:
        items = reg.get('ungrouped', []) if g == 'ungrouped' else reg.get('groups', {}).get(g, [])
        for it in items or []:
            flag = it.get('flag') if isinstance(it, dict) else None
            if flag and flag not in out:
                out.append(flag)
    return [f for f in out if f not in GUI_SKIP]


REGISTRY = load_registry()
ADV_ORDER = adv_order(REGISTRY)


def group_items(reg, gname):
    if gname == 'ungrouped':
        return reg.get('ungrouped') or []
    return (reg.get('groups') or {}).get(gname) or []


def tip_for(flag, item):
    """Hover explanation: the plain-language tip first, else the engine usage.

    gui_tips entries resolve through the i18n table at access time, so they are
    already in the page's language; the registry hint is the engine's own text.
    Chinese prefers the plain-language tip, English prefers the engine wording,
    and neither language ever falls back to a tip written for the other one.
    """
    tip = ''
    name = flag[2:]
    for cand in (flag, name, TIP_ALIAS.get(name)):
        if gui_tips is not None and cand and cand in getattr(gui_tips, 'TIPS', {}):
            tip = gui_tips.TIPS[cand]
            break
    hint = ''
    if isinstance(item, dict):
        hint = (item.get('hint') or item.get('arg') or '').strip()
    if current() == 'zh':
        return tip or hint
    return hint or tip


def placeholder_for(item, ftype):
    arg = (item.get('arg') or '').strip()
    hint = (item.get('hint') or '').strip()
    if ftype in ('int', 'float'):
        return arg or hint or t('serve.adv_placeholder')
    return (hint or arg)[:90] or t('serve.adv_placeholder')


def _control_html(item):
    flag = item.get('flag', '')
    ftype = item.get('type') or ('string' if item.get('takes_value') else 'bool')
    locked = flag in BASIC_FLAGS
    dis = ' disabled' if locked else ''
    title = html.escape(tip_for(flag, item), quote=True)
    lock = ('<span class="alock">%s</span>' % t('serve.adv_controlled')) if locked else ''
    key = html.escape(flag)
    if ftype == 'bool':
        ctl = '<input type="checkbox" data-flag="%s" data-type="bool"%s>' % (key, dis)
    else:
        if ftype == 'int':
            itype, step = 'number', '1'
        elif ftype == 'float':
            itype, step = 'number', 'any'
        else:
            itype, step = 'text', ''
        step_attr = ' step="%s"' % step if itype == 'number' else ''
        ctl = ('<input type="%s"%s data-flag="%s" data-type="%s" placeholder="%s"%s>'
               % (itype, step_attr, key, ftype,
                  html.escape(placeholder_for(item, ftype), quote=True), dis))
    return ('<div class="row advrow"><label class="aname" title="%s">%s</label>%s%s</div>'
            % (title, key, ctl, lock))


def adv_html(reg):
    if not reg.get('total'):
        return '<div class="advhint">%s</div>' % html.escape(t('serve.adv_no_registry'))
    order = list(reg.get('group_order') or GROUP_ORDER_FALLBACK)
    if 'ungrouped' not in order:
        order.append('ungrouped')
    parts = []
    for g in order:
        items = [it for it in group_items(reg, g)
                 if isinstance(it, dict) and it.get('flag') not in GUI_SKIP]
        if not items:
            continue
        parts.append('<div class="grp"><h3>%s</h3>' % html.escape(t('serve.grp_' + g)))
        parts.extend(_control_html(it) for it in items)
        parts.append('</div>')
    return ''.join(parts)


def registry_badge(reg):
    if not reg.get('total'):
        return t('serve.adv_no_registry')
    return t('serve.adv_registry', n=reg['total'])


def js_table():
    """String table handed to the page JS, already in the page's language."""
    return json.dumps({k: t('serve.' + k) for k in JS_KEYS}, ensure_ascii=False)


# 页面模板里的占位符: @@t.<词条>@@ 走翻译表, 其余四个由 render_page 直接填.
PLACEHOLDER = re.compile(r'@@([^@]*)@@')
T_PLACEHOLDER = re.compile(r'@@t\.([a-z0-9_]+)@@')
RAW_PLACEHOLDERS = ('i18n', 'lang', 'registry', 'adv')


def render_page(path='/'):
    lang_from_request(path)
    page = PAGE
    for key in sorted(set(T_PLACEHOLDER.findall(PAGE))):
        page = page.replace('@@t.%s@@' % key, t('serve.' + key))
    page = page.replace('@@i18n@@', js_table())
    page = page.replace('@@lang@@', current())
    page = page.replace('@@registry@@', registry_badge(REGISTRY))
    page = page.replace('@@adv@@', adv_html(REGISTRY))
    left = sorted(set(m.group(0) for m in PLACEHOLDER.finditer(page)))
    if left:
        # Every placeholder must be gone before the bytes leave for the
        # browser: a leftover would show up verbatim in the UI.  Fail loudly
        # (do_GET turns this into a visible error page) instead of shipping a
        # half-rendered page.
        raise ValueError('unsubstituted placeholder(s) in rendered page: %s' % left)
    return page


# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------
def _adv_args(req):
    """只收用户真填了的全参数; 顺序 = 注册表顺序(稳定), 未登记的旗标按名字排最后."""
    adv = req.get('adv') or {}
    if not isinstance(adv, dict):
        return []
    known = [f for f in ADV_ORDER if f in adv]
    extra = sorted(k for k in adv if isinstance(k, str) and k.startswith('--')
                   and k not in ADV_ORDER)
    out = []
    for flag in known + extra:
        if flag in BASIC_FLAGS or flag in GUI_SKIP:
            continue
        val = adv.get(flag)
        if val is True:
            out.append(flag)
        elif val is False or val is None:
            continue
        elif isinstance(val, (int, float)):
            out.extend([flag, str(val)])
        else:
            text = str(val).strip()
            if text:
                out.extend([flag, text])
    return out


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
    policy = COLD_POLICY[int(req['cold'])] if int(req.get('cold', 0)) else ''
    if policy:
        cmd += ['--cold-policy', policy]
        keep = HOT_TOKENS[int(req['hotwin'])] if int(req.get('hotwin', 0)) else 0
        if keep > 0:
            cmd += ['--cold-keep-tokens', str(keep)]
    kvlayer = str(req.get('kvlayer', '') or '').strip()
    if kvlayer:
        cmd += ['--kv-layer-storage', kvlayer]
    if not int(req.get('graph', 1)):
        cmd += ['--no-cuda-graph']
    cmd += _adv_args(req)
    return cmd


# --------------------------------------------------------------------------
# 投机档位 / 接受率 / 位置剖面
# --------------------------------------------------------------------------
# 数据源优先级（高到低）: 引擎指标端点 > --request-log-jsonl 的 JSONL > serve stdout 尾部.
# 引擎**当前没有**投机指标端点 —— src/serve/http_server.cpp:364-405 只注册了 /health 与 /v1/*,
# 所以实际上落到日志解析；一旦引擎补上端点（下面这张候选表里任一名字），这里自动改用端点,
# 不需要再改界面。三个来源字段同名同义，解析出来的是同一份参考系。
ENGINE_METRIC_PATHS = ('/api/spec_stats', '/metrics', '/v1/metrics')
PROBE_TTL_SECONDS = 10.0
JSONL_TAIL_BYTES = 65536
LOG_TAIL_CHARS = 6000
_PROBE = {'at': 0.0}

# serve 的 request-done 行 (src/serve/request_log.cpp:464-479 打印).
FIELD_RE = re.compile(r'spec_([a-z_]+)=([0-9.]+)')
BACKEND_RE = re.compile(r'speculative=([A-Za-z0-9_]+)')
# CLI 的汇总行 (apps/cli/main.cpp:226-248): <backend> acceptance rate / length / accepted by pos.
CLI_RATE_RE = re.compile(r'(\S+)\s+acceptance rate\s+([0-9.]+)\s*%')
CLI_LEN_RE = re.compile(r'(\S+)\s+acceptance length\s+([0-9.]+)\s*tok/round')
CLI_POS_RE = re.compile(r'(\S+)\s+accepted by pos\s+([0-9,\s]+)')


def _as_int(v):
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v):
    try:
        return None if v in (None, '') else float(v)
    except (TypeError, ValueError):
        return None


def parse_spec_text(text):
    """serve stdout 的 request-done 行: speculative=<backend> spec_drafted=... spec_accepted=..."""
    hit = None
    for line in (text or '').splitlines():
        m = BACKEND_RE.search(line)
        if not m:
            continue
        if m.group(1) == 'off':
            hit = {'backend': '', 'enabled': False}
            continue
        f = dict(FIELD_RE.findall(line))
        hit = {'backend': m.group(1), 'enabled': True,
               'drafted': _as_int(f.get('drafted')), 'accepted': _as_int(f.get('accepted')),
               'accept_rate': _as_float(f.get('accept_rate')),
               'rounds': _as_int(f.get('rounds')),
               'fallback_steps': _as_int(f.get('fallback_steps')),
               'accept_len': _as_float(f.get('accept_len'))}
    return hit


def parse_spec_summary(text):
    """CLI 汇总行。位置直方图只在 CLI 与 JSONL 里有, serve 的文本行不打印。"""
    out = {}
    for line in (text or '').splitlines():
        m = CLI_RATE_RE.search(line)
        if m:
            out['backend'] = m.group(1)
            out['accept_rate'] = float(m.group(2)) / 100.0
        m = CLI_LEN_RE.search(line)
        if m:
            out['backend'] = m.group(1)
            out['accept_len'] = float(m.group(2))
        m = CLI_POS_RE.search(line)
        if m:
            out['backend'] = m.group(1)
            out['by_position'] = [int(x) for x in m.group(2).replace(' ', '').strip(',').split(',')
                                  if x.isdigit()]
    return out or None


def parse_spec_jsonl(text):
    """JSONL request_done 记录 —— 唯一带 accepted_per_position 的 serve 侧来源."""
    hit = None
    for line in (text or '').splitlines():
        line = line.strip()
        if not line.startswith('{') or 'request_done' not in line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        sp = rec.get('speculative')
        if not isinstance(sp, dict):
            continue
        backend = sp.get('backend')
        hit = {'backend': '' if backend in (None, 'none', 'None', 'off') else str(backend),
               'enabled': True,
               'draft_window': _as_int(sp.get('draft_window')),
               'drafted': _as_int(sp.get('drafted_tokens')),
               'accepted': _as_int(sp.get('accepted_tokens')),
               'accept_rate': _as_float(sp.get('acceptance_rate')),
               'rounds': _as_int(sp.get('rounds')),
               'fallback_steps': _as_int(sp.get('fallback_steps')),
               'by_position': [int(x) for x in (sp.get('accepted_per_position') or [])
                               if _as_int(x) is not None]}
    return hit


def _request_log_jsonl_path():
    cmd = (_session or {}).get('cmd') or []
    for i, a in enumerate(cmd):
        if a == '--request-log-jsonl' and i + 1 < len(cmd):
            return cmd[i + 1]
    return ''


def _jsonl_tail():
    path = _request_log_jsonl_path()
    if not path:
        return ''
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - JSONL_TAIL_BYTES))
            return f.read().decode('utf-8', 'replace')
    except OSError:
        return ''


def _spec_from_engine_json(data):
    """引擎指标端点的返回 -> 同一份字段; 结构与 JSONL 记录对齐即可."""
    if not isinstance(data, dict):
        return None
    sp = data.get('speculative') if isinstance(data.get('speculative'), dict) else data
    keys = ('accepted_per_position', 'acceptance_rate', 'accepted_tokens', 'drafted_tokens',
            'rounds', 'backend', 'tier')
    if not any(k in sp for k in keys):
        return None
    backend = sp.get('backend', sp.get('tier'))
    return {'backend': '' if backend in (None, 'none', 'None', 'off') else str(backend),
            'enabled': True,
            'draft_window': _as_int(sp.get('draft_window')),
            'drafted': _as_int(sp.get('drafted_tokens', sp.get('drafted'))),
            'accepted': _as_int(sp.get('accepted_tokens', sp.get('accepted'))),
            'accept_rate': _as_float(sp.get('acceptance_rate', sp.get('accept_rate'))),
            'rounds': _as_int(sp.get('rounds')),
            'fallback_steps': _as_int(sp.get('fallback_steps')),
            'by_position': [int(x) for x in (sp.get('accepted_per_position') or [])
                            if _as_int(x) is not None]}


def probe_engine_metrics():
    """试引擎的指标端点: 没有就返回 None (连接被拒立即返回, 不拖慢轮询)."""
    # 10 秒 TTL 缓存: /api/state 每 1.5 秒轮询一次, 不能每次都去打同一个 404.
    now = time.time()
    if now - _PROBE['at'] < PROBE_TTL_SECONDS:
        return None
    _PROBE['at'] = now
    port = str((_session or {}).get('port') or '').strip()
    if not port or _proc is None or _proc.poll() is not None:
        return None
    for path in ENGINE_METRIC_PATHS:
        try:
            with urllib.request.urlopen('http://127.0.0.1:%s%s' % (port, path),
                                        timeout=0.4) as r:
                data = json.loads(r.read().decode('utf-8', 'replace'))
        except Exception:
            continue
        got = _spec_from_engine_json(data)
        if got:
            got['source_path'] = path
            return got
    return None


def spec_stats():
    """当前会话的投机统计: 档位 / 接受率 / 位置剖面（多源合并, 高优先级覆盖低优先级）."""
    running = _proc is not None and _proc.poll() is None
    st = {'running': running, 'configured': '', 'draft_tokens': 0, 'backend': '',
          'enabled': None, 'draft_window': None, 'drafted': None, 'accepted': None,
          'accept_rate': None, 'rounds': None, 'fallback_steps': None, 'accept_len': None,
          'by_position': [], 'sources': []}
    req = (_session or {}).get('req') or {}
    try:
        idx = int(req.get('spec', 0) or 0)
    except (TypeError, ValueError):
        idx = 0
    if 0 < idx < len(SPEC_ARGS):
        st['configured'] = SPEC_ARGS[idx].split()[-1]
        st['draft_tokens'] = _as_int(req.get('draft')) or 0
    for tag, got in (('log-tail', parse_spec_text(_log[-LOG_TAIL_CHARS:])),
                     ('cli-summary', parse_spec_summary(_log[-LOG_TAIL_CHARS:])),
                     ('jsonl', parse_spec_jsonl(_jsonl_tail())),
                     ('engine-http', probe_engine_metrics())):
        if not got:
            continue
        st['sources'].append(tag)
        for k, v in got.items():
            if k == 'source_path':
                continue
            if v is None or v == [] or v == '':
                continue
            st[k] = v
    # 只有分子分母没有比值时补一个（CLI 汇总里两者可能不同时出现）。
    if st['accept_rate'] is None and st['drafted']:
        st['accept_rate'] = float(st['accepted'] or 0) / float(st['drafted'])
    return st


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.split('?', 1)[0] in ('/', '/index.html'):
            try:
                lang_from_request(self.path)
                body = render_page(self.path).encode('utf-8')
            except Exception as e:  # 渲染炸了要看得见, 不能白屏
                body = ('NInfer Serve: render error: %s' % e).encode('utf-8')
            self._raw(body, 'text/html; charset=utf-8')
        elif self.path.split('?', 1)[0] == '/api/state':
            gpu = ''
            try:
                out = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                     '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
                gpu = out.stdout.strip()
            except Exception:
                pass
            self._json({'running': _proc is not None and _proc.poll() is None,
                        'log': _shown_log() or t('serve.log_ready'), 'gpu': gpu,
                        'sim': _sim_state(),
                        'spec': spec_stats()})
        elif self.path.split('?', 1)[0] == '/api/spec_stats':
            self._json(spec_stats())
        elif self.path.split('?', 1)[0] == '/api/i18n':
            lang_from_request(self.path)
            self._json({'lang': current(), 'strings': {k: t('serve.' + k) for k in JS_KEYS}})
        else:
            self._json({'error': t('serve.err_not_found')}, 404)

    def do_POST(self):
        global _proc, _log, _session
        n = int(self.headers.get('Content-Length', 0) or 0)
        try:
            req = json.loads(self.rfile.read(n) or b'{}')
        except Exception:
            self._json({'running': _proc is not None and _proc.poll() is None,
                        'log': _log, 'error': t('serve.err_bad_request')}, 400)
            return
        route = self.path.split('?', 1)[0]
        if route == '/api/start':
            if _proc is not None and _proc.poll() is None:
                self._json({'running': True, 'log': _log})
                return
            cmd = build_cmd(req)
            # 记住本次会话: 投机卡片要用请求里的 spec/draft 显示"已配置"档位, 并据此找
            # --request-log-jsonl 的文件名（位置直方图只在该 JSONL 记录里）.
            env, sim_dropped = engine_env()
            _session = {'req': req, 'cmd': list(cmd), 'port': str(req.get('port', '')),
                        'sim': {'requested': False, 'dropped': sim_dropped}}
            _sim_pin = ''
            _log = t('serve.log_started') + ' '.join(cmd) + '\n'
            if sim_dropped:
                _log += t('serve.log_sim_scrubbed', keys=', '.join(sim_dropped)) + '\n'
            _proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, env=env)

            def reader():
                for line in _proc.stdout:
                    log_append(line)
            import threading
            threading.Thread(target=reader, daemon=True).start()
            self._json({'running': True, 'log': _log})
        elif route == '/api/stop':
            if _proc is not None:
                try:
                    _proc.terminate()
                    try:
                        _proc.wait(timeout=15)
                    except Exception:
                        _proc.kill()
                    _log += '\n' + t('serve.log_stopped') + '\n'
                except Exception as e:
                    _log += '\n' + t('serve.err_stop_failed', err=e) + '\n'
                _proc = None
            self._json({'running': False, 'log': _log})
        elif route == '/api/lang':
            self._json({'lang': lang_from_request(self.path, req)})
        else:
            self._json({'error': t('serve.err_not_found')}, 404)

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
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8789
    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    print(t('serve.console_ready', port=port, bin=SERVE_BIN))
    srv.serve_forever()


if __name__ == '__main__':
    main()
