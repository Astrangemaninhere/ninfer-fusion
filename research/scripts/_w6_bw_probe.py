#!/usr/bin/env python3
"""W6 probe: decode inter-token latency of an in-flight request while a 64K prefill runs.

The engine owns a single prefill lane, so a *new* short request cannot prefill while the big
prompt prefills. The observable effect of the governor is therefore on a request that is already
decoding when the big prefill starts:

  phase 1 (solo)      : stream 48 tokens with no background load -> inter-token baseline
  phase 2 (contended) : start the same stream, then start a 64K prompt; inter-token latency of
                        the stream after the big prefill begins is the contended number

Output: SOLO_MEDIAN_MS / UNDER_MEDIAN_MS / RATIO. The caller applies the pass threshold.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

sys.path.insert(0, "/mnt/c/Users/User/Documents/ziqinzhang/NI2A3F~1/tools/archkit")
import longtest_57k  # noqa: E402


def stream_chat(port: int, model: str, messages, max_tokens: int, on_token=None,
                timeout: int = 900) -> tuple[str, list[float]]:
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0, "stream": True}
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"content-type": "application/json"})
    stamps: list[float] = []
    text: list[str] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            # Thinking models stream into reasoning_content; plain answers use content.
            piece = delta.get("content") or delta.get("reasoning_content") or ""
            if piece:
                now = time.time()
                stamps.append(now)
                text.append(piece)
                if on_token is not None:
                    on_token(now)
    return "".join(text), stamps


def median_gap(stamps: list[float], lo: float, hi: float) -> float:
    gaps = [1000.0 * (stamps[i + 1] - stamps[i])
            for i in range(len(stamps) - 1)
            if lo <= stamps[i + 1] <= hi]
    return statistics.median(gaps) if gaps else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8326)
    ap.add_argument("--model", default="muse-glimmer-30b")
    ap.add_argument("--context", type=int, default=65536)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--label", default="case")
    args = ap.parse_args()
    print(f"== probe[{args.label}] context={args.context} tokens={args.tokens}", flush=True)

    short = [{"role": "user", "content": "从1数到40，只用阿拉伯数字和逗号。"}]

    # Phase 1: solo inter-token latency.
    solo_stamps: list[float] = []
    for i in range(2):
        _text, stamps = stream_chat(args.port, args.model, short, args.tokens)
        if len(stamps) < 8:
            print(f"  solo[{i}] too few tokens ({len(stamps)})", flush=True)
            continue
        solo_stamps = stamps
        gaps = [1000.0 * (stamps[j + 1] - stamps[j]) for j in range(len(stamps) - 1)]
        print(f"  solo[{i}] tokens={len(stamps)} median_gap={statistics.median(gaps):.0f}ms",
              flush=True)

    # Phase 2: stream while a 64K prefill runs.
    import random
    ctx_text, _needles = longtest_57k.make_context(args.context, 2, random.Random(57000))
    print(f"  background context chars={len(ctx_text)}", flush=True)
    background: dict[str, float] = {}
    background_error: list[str] = []

    def prefill_load() -> None:
        background["start"] = time.time()
        try:
            stream_chat(args.port, args.model,
                        [{"role": "user", "content": ctx_text + "\n\n只回答：好的"}], 4)
        except Exception as exc:  # noqa: BLE001
            background_error.append(str(exc))
        background["end"] = time.time()

    result: dict[str, list[float]] = {}

    def decode_stream() -> None:
        try:
            _text, stamps = stream_chat(args.port, args.model, short, args.tokens * 6)
            result["stamps"] = stamps
        except Exception as exc:  # noqa: BLE001
            background_error.append(f"stream: {exc}")

    decode_thread = threading.Thread(target=decode_stream, daemon=True)
    decode_thread.start()
    time.sleep(1.5)  # let the stream reach steady decode before the big prefill lands
    load_thread = threading.Thread(target=prefill_load, daemon=True)
    load_thread.start()
    decode_thread.join(timeout=900)
    load_thread.join(timeout=900)

    stamps = result.get("stamps", [])
    if len(stamps) < 8:
        print(f"  contended stream produced too few tokens ({len(stamps)})", flush=True)
        print("SOLO_MEDIAN_MS=nan\nUNDER_MEDIAN_MS=nan\nRATIO=nan", flush=True)
        return 0
    lo = background.get("start", stamps[0])
    hi = background.get("end", stamps[-1])
    solo_med = median_gap(solo_stamps, 0.0, float("inf")) if len(solo_stamps) > 1 else float("nan")
    under_med = median_gap(stamps, lo, hi)
    ratio = under_med / solo_med if solo_med and solo_med > 0 else float("nan")
    print(f"  contended tokens={len(stamps)} background_ms={1000.0 * (hi - lo):.0f}", flush=True)
    if background_error:
        print(f"  background_error={background_error[:1]}", flush=True)
    print(f"SOLO_MEDIAN_MS={solo_med:.1f}", flush=True)
    print(f"UNDER_MEDIAN_MS={under_med:.1f}", flush=True)
    print(f"RATIO={ratio:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
