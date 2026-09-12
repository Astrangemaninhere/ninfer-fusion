#!/usr/bin/env python3
"""W16 probe: request-domain fault must fail exactly one request, later ones must succeed.

Sends up to --requests short requests and checks:
  * at least one response carries the injected fault reason (HTTP 4xx/5xx)
  * at least one request after the failing one succeeded (engine kept serving)
  * /health is ok
Exit 0 = PASS.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def post(port: int, path: str, payload: dict, timeout: int = 120):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return 0, f"transport error: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8327)
    ap.add_argument("--requests", type=int, default=4)
    args = ap.parse_args()

    payload = {"model": "muse-glimmer-30b",
               "messages": [{"role": "user", "content": "3+4? digits only"}],
               "max_tokens": 8}
    outcomes: list[tuple[int, str]] = []
    for i in range(args.requests):
        status, body = post(args.port, "/v1/chat/completions", payload)
        outcomes.append((status, body))
        print(f"  req[{i}] HTTP {status} {body[:120]!r}", flush=True)

    fault_index = next((i for i, (_s, b) in enumerate(outcomes)
                        if "injected request-domain fault" in b), -1)
    success_after = any(status == 200 and i > fault_index
                        for i, (status, _b) in enumerate(outcomes))
    health_status, health_body = post(args.port, "/health", {}) if False else (0, "")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=5) as r:
            health_status, health_body = r.status, r.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        health_body = f"health error: {exc}"

    ok = fault_index >= 0 and success_after and health_status == 200
    print(f"fault_index={fault_index} success_after={success_after} "
          f"health={health_status} {health_body[:120]}", flush=True)
    print("W16_REQUEST_FAULT_" + ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
