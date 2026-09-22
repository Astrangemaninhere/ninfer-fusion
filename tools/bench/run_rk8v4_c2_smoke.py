# PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
# Source repo : sergiuszm/ninfer-4090
# Branch      : rtx4090-port
# Commit      : 1bd56c9a1bdf457c6188391a9385d44d86e953aa
# Source path : tools/bench/run_rk8v4_c2_smoke.py
# sha256(src) : b8e5eee3e9bcaeed6b4004769315f43d2de7005ff09fa780a20df4ade8fb0a73
# Landed by   : /home/user/scratch/PATCHSET/MERGE (fork-survey merge, 2026-09-18)
# Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
# Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
# Note        : the codecs in this family label a rotation axis -- rk<N>v<M> = rotated K at N bits against V at M bits (rk4v4-e8 = rotated K 4b, V 4b, rk4v4 in the mix), which is a DIFFERENT axis from src/kvcfg/kv_formats.h's tier vocabulary (hot/tail/cold, Rk4v4).
import json
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(r"G:\python\custom-kernel-3090\infer-qwen38-replayssm-sm86")
SERVER = ROOT / "build-sm86-replayssm" / "apps" / "Release" / "ninfer-serve.exe"
MODEL = Path(r"G:\python\custom-kernel-3090\qwen3_8_27b.ninfer")
OUTPUT_DIR = ROOT / "benchmark_results" / "20260815_rk8v4_quality" / "02_c2_api_smoke"
PORT = 18086


def post(index: int) -> dict:
    payload = {
        "model": "qwen3.8-27b",
        "messages": [
            {
                "role": "user",
                "content": f"Request {index}: explain one invariant required by rollback DSU.",
            }
        ],
        "max_tokens": 128,
        "temperature": 0,
        "stream": False,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    return {"request": index, "elapsed_seconds": time.perf_counter() - started, "response": body}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        str(SERVER), str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
        "--max-context", "4096", "--kv-capacity", "8192", "--max-concurrency", "2",
        "--max-pending-requests", "4", "--prefill-chunk", "1024", "--kv-dtype", "rk8v4",
        "--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft",
    ]
    with (OUTPUT_DIR / "server.stdout.log").open("w", encoding="utf-8") as stdout_file, (
        OUTPUT_DIR / "server.stderr.log"
    ).open("w", encoding="utf-8") as stderr_file:
        server = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, text=True)
        try:
            for _ in range(120):
                if server.poll() is not None:
                    raise RuntimeError(f"server exited with {server.returncode}")
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1).read()
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.25)
            else:
                raise TimeoutError("server did not become healthy")
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(post, (1, 2)))
            (OUTPUT_DIR / "responses.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps({"returncodes": [0, 0], "elapsed": [r["elapsed_seconds"] for r in results]}, indent=2))
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)


if __name__ == "__main__":
    main()
