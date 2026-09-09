#!/usr/bin/env python3
"""longtest_57k.py — 57K long-context needle acceptance for layered KV configs.

Protocol (§38/§61): big planted-needle context + long generation + retrieval
questions, run against a live ninfer-serve. Pass = all needles retrieved
verbatim at the target context. This is the acceptance gate for layered
KV storage (e8/nvfp4 mixes) after a reload or cold start.

Usage:
  python longtest_57k.py --port 8321 --context 57344 [--needles 8] [--model muse-glimmer-30b]
  python longtest_57k.py --port 8321 --context 57344 --reload "0-11:e8,12-15:nvfp4"
                         [--reloads "all:bf16" "0-11:e8,12-15:nvfp4"]   # sequential configs
Exit 0 iff every tested config passes.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request

FILLER_THEMES = [
    "城市轨道交通的班次安排与换乘提示", "海边小镇的渔市清晨", "老式图书馆的借阅规则",
    "高山气象站的冬季值守", "社区菜园的轮作计划", "长途列车的卧铺铺位分配",
    "云层观测与降水记录", "面包房凌晨备料的流程", "灯塔守护人的日志格式",
    "小学运动会的项目编排", "猫咪寄养的注意事项", "屋顶花园的排水设计",
    "废旧电池的回收路径", "古镇石桥的修缮记录", "深夜电台的听众来信栏目",
    "河流汛期的巡查安排", "手作皮具的工作台布局", "地铁隧道通风井的巡检",
    "校园广播站的值班表", "渔船出海前的安全检查",
]

NEEDLE_TEMPLATE = "【档案编号 {code}】{city}站的月度密钥是 {secret}，请妥善保存。"
CITIES = ["苍梧", "临溪", "白泷", "云栖", "鹿鸣", "栖霞", "平潮", "泗水",
          "青崖", "望津", "流杯", "横塘", "石门", "芦渡", "南屏", "北屿"]


def make_context(target_tokens: int, needle_count: int, rng: random.Random):
    """Build filler paragraphs with needles planted at even depth intervals.

    Returns (context_text, needles) where needles = [(code, city, secret, question, answer)].
    Token estimate: Chinese ≈ 1 token/char for this engine family; keep ~5% slack
    by oversizing filler and truncating on the last paragraph boundary.
    """
    needles = []
    for i in range(needle_count):
        code = f"N{i:02d}-{rng.randint(100, 999)}"
        city = CITIES[i % len(CITIES)]
        secret = "".join(rng.choice("ABCDEFGHJKMNPQRSTUVWXYZ23456789") for _ in range(6))
        needles.append((code, city, secret,
                        f"{city}站的月度密钥是什么？只回答密钥本身。",
                        secret))
    paras: list[str] = []
    est = 0
    idx = 0
    needle_every = max(1, target_tokens // (needle_count + 2))
    next_needle_at = needle_every // 2
    while est < target_tokens:
        theme = FILLER_THEMES[idx % len(FILLER_THEMES)]
        body = (f"{theme}：本段记录例行事项。值岗人员按表执行巡检，"
                f"异常情况上报值班长，并注意与相邻班次交接时的口头确认。"
                f"设备读数在标准区间内浮动属于正常现象，无需额外记录。")
        est += len(body) + 2
        if est >= next_needle_at and needles:
            code, city, secret, _q, _a = needles.pop(0)
            body += " " + NEEDLE_TEMPLATE.format(code=code, city=city, secret=secret)
            next_needle_at += needle_every
        paras.append(body)
        idx += 1
    return "\n\n".join(paras), needles


def chat(port: int, model: str, messages: list[dict], max_tokens: int, timeout: int = 600) -> str:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"model": model, "messages": messages,
                         "max_tokens": max_tokens, "temperature": 0}).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"] or ""


def reload_kv(port: int, spec: str) -> None:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/reload_kv",
        data=json.dumps({"kv_layer_storage": spec}).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    print(f"  reload {spec}: {out.get('status')}", flush=True)


def run_config(port: int, model: str, ctx_text: str, needles, label: str) -> bool:
    print(f"[{label}] context chars={len(ctx_text)} needles={len(needles)}", flush=True)
    t0 = time.time()
    answer = chat(port, model, [
        {"role": "system", "content": "从档案中精确检索。只回答被问到的密钥本身，不要解释。"},
        {"role": "user", "content": ctx_text + "\n\n" +
         " ".join(q for _c, _s, q, _a in [(n[0], n[1], n[3], n[4]) for n in needles])},
    ], max_tokens=120)
    dt = time.time() - t0
    ok_all = True
    for _code, _city, secret, _q, _a in needles:
        if secret not in answer:
            ok_all = False
    hits = sum(1 for _c, _s, _q, a in needles if a in answer)
    print(f"[{label}] prefill+decode {dt:.1f}s, needle hits {hits}/{len(needles)} -> "
          f"{'PASS' if ok_all else 'FAIL'}", flush=True)
    if not ok_all:
        print(f"[{label}] answer snippet: {answer[:400]}", flush=True)
    return ok_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--model", default="muse-glimmer-30b")
    ap.add_argument("--context", type=int, default=57344)
    ap.add_argument("--needles", type=int, default=8)
    ap.add_argument("--seed", type=int, default=57000)
    ap.add_argument("--reload", default=None, help="single spec to reload before testing")
    ap.add_argument("--reloads", nargs="*", default=None,
                    help="sequential configs, each reloaded then tested")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ctx_text, needles = make_context(args.context, args.needles, rng)

    configs = args.reloads if args.reloads else ([args.reload] if args.reload else [None])
    all_ok = True
    for spec in configs:
        if spec:
            reload_kv(args.port, spec)
        label = spec or "cold-config"
        all_ok &= run_config(args.port, args.model, ctx_text, needles, label)
    print("LONGTEST_PASS" if all_ok else "LONGTEST_FAIL", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
