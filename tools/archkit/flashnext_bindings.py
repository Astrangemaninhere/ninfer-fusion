#!/usr/bin/env python3
"""flashnext_bindings.py — Qwen3.8-Flash-Next (qwen4_exp) weight bindings contract.

P0 step 2 of _flashnext_plan: the authoritative mapping from checkpoint tensor
names to ninfer engine tensors for the qwen4_exp target. The upstream HF source
is not vendored, so this contract DEFINES our canonical names and ACCEPTED
source alias patterns (HF Qwen-MoE style + llama.cpp qwen4exp style). When a
real checkpoint arrives, `audit()` proves the key set covers every engine
tensor and every source key is consumed — nothing silently dropped.

Sources of truth:
  tools/archkit/specs/qwen4_exp_spec.json   (geometry, 48 layers, MoE 512x10,
                                             PLE ngram, indexer/hc, MTP, GDN)
  RESEARCH-FLASHNEXT.md                     (architecture, QSA at [3,7,...,47],
                                             PLE row derivation, MTP head)
  ple-manifest.json                         (PLE table: 20M x 160 BF16, 16 heads)

Usage:
  python flashnext_bindings.py --emit bindings.json
  python flashnext_bindings.py --audit index.json     # safetensors index
  python flashnext_bindings.py --audit-gguf names.txt # one GGUF tensor name/line
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from pathlib import Path

SPEC = Path(__file__).parent / "specs" / "qwen4_exp_spec.json"

N_LAYERS = 48
GDN_PATTERN = [0, 1, 2] * 12        # 3x linear_attention then 1x full_attention
QSA_LAYERS = list(range(3, 48, 4))  # [3, 7, ..., 47]
GDN_LAYERS = [i for i in range(N_LAYERS) if i not in QSA_LAYERS]
N_GDN, N_QSA = len(GDN_LAYERS), len(QSA_LAYERS)


def _spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Contract entries.
#   engine : canonical ninfer tensor name (converter target)
#   shape  : [in, out] logical expectation (None = data-dependent)
#   alias  : ordered glob patterns matched against checkpoint keys; first hit
#            wins; "{i}" is the layer index placeholder
#   role   : why it exists (converter docs)
E: list[dict] = []


def e(engine: str, shape: str, aliases: list[str], role: str, layer_scope: str = "all") -> None:
    E.append({"engine": engine, "shape": shape, "alias": aliases,
              "role": role, "layers": layer_scope})


# ---- global ----
e("token_embd", "[vocab,2560]",
  ["model.embed_tokens.weight", "token_embd.weight"], "input embedding")
e("output_head", "[2560,vocab]",
  ["lm_head.weight", "output.weight"], "logit head (untied for qwen4_exp)")

# ---- per-layer shared norms ----
for i in range(N_LAYERS):
    e(f"layer.{i}.attn_norm", "[2560]",
      [f"model.layers.{i}.input_layernorm.weight", f"blk.{i}.attn_norm.weight"],
      "pre-attention norm", layer_scope=str(i))
    e(f"layer.{i}.ffn_norm", "[2560]",
      [f"model.layers.{i}.post_attention_layernorm.weight", f"blk.{i}.ffn_norm.weight"],
      "pre-MoE norm", layer_scope=str(i))

# ---- GDN (Gated DeltaNet) layers: 36x ----
for i in GDN_LAYERS:
    p = f"model.layers.{i}.linear_attn"          # HF-style guess
    g = f"blk.{i}"                                # llama.cpp-style guess
    e(f"layer.{i}.gdn.in_qkv", "[2560, K+V]",
      [f"{p}.in_proj_qkv.weight", f"{p}.qkv_proj.weight", f"{g}.gdn_in_proj.weight"],
      "fused Q/K/V input proj (16 k-heads x128, 48 v-heads x128)", layer_scope=str(i))
    e(f"layer.{i}.gdn.conv", "[conv,channels]",
      [f"{p}.conv1d.weight", f"{p}.conv.weight", f"{g}.gdn_conv.weight"],
      "depthwise causal conv, kernel=4", layer_scope=str(i))
    e(f"layer.{i}.gdn.conv_bias", "[channels]",
      [f"{p}.conv1d.bias", f"{g}.gdn_conv.bias"],
      "conv bias (optional)", layer_scope=str(i))
    e(f"layer.{i}.gdn.beta", "[48]",
      [f"{p}.beta.weight", f"{p}.b_proj.weight", f"{g}.gdn_beta.weight"],
      "delta gate beta (per v-head)", layer_scope=str(i))
    e(f"layer.{i}.gdn.dt_bias", "[48]",
      [f"{p}.dt_bias", f"{p}.A_log", f"{g}.gdn_dt_bias"],
      "decay bias/log-A per v-head", layer_scope=str(i))
    e(f"layer.{i}.gdn.norm", "[128]",
      [f"{p}.norm.weight", f"{g}.gdn_norm.weight"],
      "group RMSNorm over v heads (sigmoid output gate type)", layer_scope=str(i))
    e(f"layer.{i}.gdn.gate", "[2560,6144]",
      [f"{p}.gate_proj.weight", f"{g}.gdn_gate.weight"],
      "sigmoid output gate (2560 -> 48x128)", layer_scope=str(i))
    e(f"layer.{i}.gdn.out", "[6144,2560]",
      [f"{p}.out_proj.weight", f"{g}.gdn_out.weight"],
      "output proj (48x128 -> 2560)", layer_scope=str(i))

# ---- QSA (sparse full attention) layers: 12x, with indexer + hc ----
for i in QSA_LAYERS:
    p = f"model.layers.{i}.self_attn"
    g = f"blk.{i}"
    e(f"layer.{i}.qsa.q", "[2560,6144]",
      [f"{p}.q_proj.weight", f"{g}.attn_q.weight"], "24 q-heads x256", layer_scope=str(i))
    e(f"layer.{i}.qsa.k", "[2560,512]",
      [f"{p}.k_proj.weight", f"{g}.attn_k.weight"], "2 kv-heads x256", layer_scope=str(i))
    e(f"layer.{i}.qsa.v", "[2560,512]",
      [f"{p}.v_proj.weight", f"{g}.attn_v.weight"], "2 kv-heads x256", layer_scope=str(i))
    e(f"layer.{i}.qsa.o", "[6144,2560]",
      [f"{p}.o_proj.weight", f"{g}.attn_o.weight"], "output", layer_scope=str(i))
    e(f"layer.{i}.qsa.q_norm", "[256]",
      [f"{p}.q_norm.weight", f"{g}.attn_q_norm.weight"], "per-head q RMSNorm", layer_scope=str(i))
    e(f"layer.{i}.qsa.k_norm", "[256]",
      [f"{p}.k_norm.weight", f"{g}.attn_k_norm.weight"], "per-head k RMSNorm", layer_scope=str(i))
    # indexer: 4 heads x128 over 1 kv-head, compressed by ratio 4, budget 2048
    e(f"layer.{i}.qsa.idx_wq", "[2560,512]",
      [f"{p}.indexer.wq_b.weight", f"{g}.idx_wq.weight"],
      "indexer query proj (4x128)", layer_scope=str(i))
    e(f"layer.{i}.qsa.idx_wk", "[2560,128]",
      [f"{p}.indexer.wk.weight", f"{g}.idx_wk.weight"],
      "indexer key proj (1x128)", layer_scope=str(i))
    e(f"layer.{i}.qsa.idx_norm", "[128]",
      [f"{p}.indexer.k_norm.weight", f"{g}.idx_norm.weight"],
      "indexer key norm", layer_scope=str(i))
    # hc (hierarchical compression): 4 groups, lowrank 320
    for hc in range(4):
        e(f"layer.{i}.qsa.hc.{hc}.down", "[2560,320]",
          [f"{p}.hc.{hc}.down_proj.weight", f"{g}.hc{hc}_down.weight"],
          f"hc group {hc} compress", layer_scope=str(i))
        e(f"layer.{i}.qsa.hc.{hc}.up", "[320,2560]",
          [f"{p}.hc.{hc}.up_proj.weight", f"{g}.hc{hc}_up.weight"],
          f"hc group {hc} expand", layer_scope=str(i))

# ---- MoE (all 48 layers): 512 routed experts top-10 + 1 shared ----
for i in range(N_LAYERS):
    p = f"model.layers.{i}.mlp"
    g = f"blk.{i}"
    e(f"layer.{i}.moe.router", "[2560,512]",
      [f"{p}.gate.weight", f"{g}.ffn_gate_inp.weight"], "router logits", layer_scope=str(i))
    e(f"layer.{i}.moe.sh_gate", "[2560,640]",
      [f"{p}.shared_expert.gate_proj.weight", f"{g}.ffn_gate_shexp.weight"],
      "shared expert gate", layer_scope=str(i))
    e(f"layer.{i}.moe.sh_up", "[2560,640]",
      [f"{p}.shared_expert.up_proj.weight", f"{g}.ffn_up_shexp.weight"],
      "shared expert up", layer_scope=str(i))
    e(f"layer.{i}.moe.sh_down", "[640,2560]",
      [f"{p}.shared_expert.down_proj.weight", f"{g}.ffn_down_shexp.weight"],
      "shared expert down", layer_scope=str(i))
    for x in range(512):
        b = f"{p}.experts.{x}"
        bg = f"{g}.ffn_exp_{x}"
        e(f"layer.{i}.moe.e{x}.gate", "[2560,640]",
          [f"{b}.gate_proj.weight", f"{bg}.gate.weight"], "routed expert gate",
          layer_scope=str(i))
        e(f"layer.{i}.moe.e{x}.up", "[2560,640]",
          [f"{b}.up_proj.weight", f"{bg}.up.weight"], "routed expert up",
          layer_scope=str(i))
        e(f"layer.{i}.moe.e{x}.down", "[640,2560]",
          [f"{b}.down_proj.weight", f"{bg}.down.weight"], "routed expert down",
          layer_scope=str(i))

# ---- PLE n-gram residual stack (sits at layer id 2 per spec; 1-based docs -> 2) ----
e("ple.table", "[20019200,160]",
  ["ple.ngram_embed.weight", "ngram_emb.weight"],
  "20M-row x160 BF16 lookup; NEVER GPU-resident (SSD runtime contract)")
e("ple.key", "[2560,2560]",
  ["ple.key_proj.weight", "ple_key.weight"], "PLE key linear (residual stack)")
e("ple.value", "[2560,2560]",
  ["ple.value_proj.weight", "ple_value.weight"], "PLE value linear")
e("ple.norm", "[2560]",
  ["ple.norm.weight", "ple_norm.weight"], "grouped RMSNorm after k/v")
e("ple.gate_query", "[2560,2560]",
  ["ple.gate_query.weight", "ple_gate_q.weight"], "query side of sigmoid gate")
e("ple.conv", "[conv,2560]",
  ["ple.conv.weight", "ple_conv.weight"], "depthwise causal conv kernel=4 (ngram_size conv)")

# ---- MTP head (1 hidden layer, no dedicated embedding) ----
e("mtp.fc", "[5120,2560]",
  ["mtp.fc.weight", "mtp_fc.weight"], "concat[h, draft_h] -> h projection")
e("mtp.norm", "[2560]",
  ["mtp.norm.weight", "mtp_norm.weight"], "MTP pre-norm")
e("mtp.head_norm", "[2560]",
  ["mtp.head_norm.weight", "mtp_head_norm.weight"], "MTP output norm (pre-logit)")

# ---- final norm ----
e("output_norm", "[2560]",
  ["model.norm.weight", "output_norm.weight"], "final RMSNorm")


# ---------------------------------------------------------------------------
def audit(sources: list[str]) -> dict:
    """Match checkpoint keys against alias patterns; report coverage both ways."""
    matched: dict[str, str] = {}     # source key -> engine name
    engine_hit: dict[str, str] = {}  # engine name -> source key
    unmatched_src: list[str] = []
    for key in sources:
        hit = None
        for entry in E:
            for pat in entry["alias"]:
                if fnmatch.fnmatch(key, pat):
                    hit = entry["engine"]
                    break
            if hit:
                break
        if hit:
            matched[key] = hit
            engine_hit.setdefault(hit, key)
        else:
            unmatched_src.append(key)
    missing = [entry["engine"] for entry in E if entry["engine"] not in engine_hit]
    return {
        "contract_entries": len(E),
        "matched_sources": len(matched),
        "engines_covered": len(engine_hit),
        "missing_engines": missing[:40] + ([f"... (+{len(missing)-40})"] if len(missing) > 40 else []),
        "missing_count": len(missing),
        "unmatched_sources": unmatched_src[:40],
        "unmatched_count": len(unmatched_src),
        "complete": len(missing) == 0 and len(unmatched_src) == 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", metavar="JSON", help="dump the contract as JSON")
    ap.add_argument("--audit", metavar="INDEX_JSON", help="audit a safetensors index")
    ap.add_argument("--audit-gguf", metavar="NAMES_TXT", help="audit GGUF tensor names (one per line)")
    args = ap.parse_args()

    if args.emit:
        Path(args.emit).write_text(json.dumps(
            {"spec": str(SPEC), "n_layers": N_LAYERS, "gdn_layers": GDN_LAYERS,
             "qsa_layers": QSA_LAYERS, "entries": E}, indent=1, ensure_ascii=False),
            encoding="utf-8")
        print(f"wrote {args.emit}: {len(E)} contract entries")
        return 0

    sources: list[str] | None = None
    if args.audit:
        idx = json.loads(Path(args.audit).read_text(encoding="utf-8"))
        sources = list(idx.get("weight_map", idx).keys())
    elif args.audit_gguf:
        sources = [ln.strip().split(":")[0].strip() or ln.strip()
                   for ln in Path(args.audit_gguf).read_text(encoding="utf-8").splitlines()
                   if ln.strip()]
    if sources is None:
        ap.print_help()
        return 2

    report = audit(sources)
    json.dump(report, sys.stdout, indent=1, ensure_ascii=False)
    print()
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
