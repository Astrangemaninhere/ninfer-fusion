#!/usr/bin/env python3
"""前台加载目标模型（不启服务），拿完整 traceback。"""
import os, traceback, sys
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
T = r"C:\Users\User\Documents\ziqinzhang\models\Qwen3.8-27B-NVFP4-RTX5090"
print("loading:", T, flush=True)
try:
    from vllm import LLM
    llm = LLM(model=T, gpu_memory_utilization=0.80, max_model_len=2048,
              enforce_eager=True, disable_log_stats=False)
    print("LOADED OK", flush=True)
    from vllm import SamplingParams
    out = llm.generate(["hello"], SamplingParams(max_tokens=16, temperature=0, ignore_eos=True))
    print("GEN OK:", out[0].outputs[0].text[:60], flush=True)
except Exception:
    print("=== FULL TRACEBACK ===", flush=True)
    traceback.print_exc()
    tb = traceback.format_exc()
    for line in tb.splitlines():
        if "safetensors" in line.lower() or "not fully" in line or "incomplete" in line:
            print(">>> " + line, flush=True)
