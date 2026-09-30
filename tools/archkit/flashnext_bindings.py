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
    e(f"layer.{i}.gdn.in_proj_a", "[2560,48]",
      [f"{p}.in_proj_a.weight", f"{g}.gdn_in_proj_a.weight"],
      "delta-gate alpha input proj (per v-head)", layer_scope=str(i))
    # in_proj_b is the SECOND HALF of one engine tensor, not a second `beta` and not a
    # separate operator.  The geometry, measured on both sides:
    #   src/targets/qwen3_8_flash_next/impl/model_view.h:64   Weight a_b_projection;
    #   src/targets/qwen3_8_flash_next/impl/gdn.cpp:105,171   exact_bf16_weight(..., 96, 2'560)
    #   .../impl/load/materialized.cpp:149                    bf16_weight(..., 96, 2'560)
    #   .../impl/gdn_kernels.cu:143                           reads a_b_projection.qdata as ONE buffer
    #   tools/convert/qwen3_8_flash_next/inventory.py:85      a_b_projection (96, 2_560) BF16
    #   tools/convert/qwen3_8_flash_next/recipe.py:114-122    concat-rows(in_proj_a (48,2560), in_proj_b (48,2560))
    # The engine wants the FUSED parent: A occupies rows [0,48) and B rows [48,96) of a
    # BF16 [96,2560] tensor, exactly as include/ninfer/ops/gdn_gating_proj.h:51-52 splits
    # its registered parents (`[96,5120]` for the 27B geometry, `[64,2048]` for 35B-A3B;
    # Flash-Next's hidden is 2560, which is why its parent is [96,2560] and not [96,5120]).
    # The checkpoint carries both halves per GDN layer -- `...linear_attn.in_proj_a.weight`
    # and `...linear_attn.in_proj_b.weight`, BF16 [48,2560], 36 each.
    # They stay TWO contract entries because `audit()` admits exactly one source key per
    # entry: `flashnext_bindings.py:162-163` records `len(hits) > 1` as an ambiguous alias
    # and binds only hits[0], so a single entry carrying both keys would be ambiguous AND
    # would still leave one key unconsumed.  The fusion lives where it already lives, in
    # the recipe's `concat-rows` transform.
    e(f"layer.{i}.gdn.in_proj_b", "[2560,48]",
      [f"{p}.in_proj_b.weight", f"{g}.gdn_in_proj_b.weight"],
      "delta-gate beta input proj, second half of the fused a_b_projection",
      layer_scope=str(i))
    e(f"layer.{i}.gdn.dt_bias", "[48]",
      [f"{p}.dt_bias", f"{g}.gdn_dt_bias"],
      "decay bias per v-head (softplus pre-activation)", layer_scope=str(i))
    # A_log is a DISTINCT operator, not a second alias of dt_bias. The engine declares
    # them as two tensors and feeds them as two independent arguments:
    #   src/targets/qwen3_5_9b/impl/load/bindings.h:230-231  Tensor a_log; Tensor dt_bias;
    #   include/ninfer/ops/gdn_gating_proj.h:44-46           gdn_gating_proj(x, a_weight,
    #                                                        b_weight, A_log, dt_bias, ...)
    #   include/ninfer/ops/gdn_gating_proj.h:32              g[h,t] = -exp(A_log[h]) *
    #                                                        softplus(a[h,t] + dt_bias[h])
    # With A_log in dt_bias's alias list the fold bound whichever name resolved first and
    # left the other one unconsumed, so the engine's a_log had no source at all: 36 real
    # checkpoint keys (`...linear_attn.A_log`) fell into the unexpected residue and
    # `audit()` reported 36 ambiguous aliases. Two operators, two entries.
    # `-exp` deliberately stays on the ENGINE side: measured on this checkpoint A_log is
    # signed and O(1) (36/36 layers MIXED, min=-2.445 max=+2.316, 409/1728 negative), i.e.
    # the raw log form the engine expects. See dl/flashnextaudit/REPORT.md section 3.
    e(f"layer.{i}.gdn.a_log", "[48]",
      [f"{p}.A_log", f"{g}.gdn_a_log"],
      "log-A decay scale per v-head (engine applies -exp to it)", layer_scope=str(i))
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
    # MEASURED 2026-09-22 (FN-SRC): the third alias is the name THIS checkpoint ships.
    # `{p}.indexer.k_norm.weight` exists nowhere -- 0 of 296,475 keys, and 0 at the
    # safetensors-header level -- while
    #   model.language_model.layers.<L>.self_attn.indexer.k_layernorm.weight  BF16 [128]
    # exists for all 12 QSA layers and was falling into the audit's unexpected residue
    # while this engine entry was counted missing.  Two independent readings already
    # bind the operator under that name:
    #   recipe.py:224  ap + "indexer.k_layernorm.weight"  -> object
    #     `attention/indexer/key_norm`, which is this entry's role (and :343 for MTP);
    #   load/bindings.cpp:172  .indexer_key_norm = bind("indexer/key_norm",
    #     NumericFormat::BF16, kBf16Layout, {128}), with materialized.cpp:160
    #     bf16_tensor(backing, plan.indexer_key_norm, {128}).
    # Appended THIRD, so alias[0] -- which flashnext_convert's synthetic-name
    # self-tests read -- is unchanged.
    # NOT folded in: the engine also declares `indexer_query_norm`
    # (load/bindings.cpp:173) and the disk ships `{p}.indexer.q_layernorm.weight`
    # BF16 [128] as well.  That second operator has no contract entry, and the two
    # names have IDENTICAL shapes, so a set-based audit cannot distinguish a correct
    # binding from a swapped one.  Adding it is a modelling addition (it needs a
    # `_rule` and an `artifact_shape` branch), not an alias edit.
    e(f"layer.{i}.qsa.idx_norm", "[128]",
      [f"{p}.indexer.k_norm.weight", f"{g}.idx_norm.weight",
       f"{p}.indexer.k_layernorm.weight"],
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
e("ple.table", "[320001536,160]",
  ["ple.ngram_embed.weight", "ngram_emb.weight"],
  "320001536-row x160 BF16 lookup; NEVER GPU-resident (SSD runtime contract)")
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
# Source-key normalization.
#
# The real Qwen3.8-Flash-Next checkpoint nests the language tower one segment
# deeper than the HF Qwen-MoE convention these aliases are written in:
#     model.language_model.layers.5.linear_attn.in_proj_qkv.weight
# where the contract (and llama.cpp's qwen4exp reader) says
#     model.layers.5.linear_attn.in_proj_qkv.weight
# Without this rewrite exactly 1 of the 74,520 entries matches. Re-derived from
# the shipped index (296,475 keys, 'model.language_model' x296,110).
_PREFIX_REWRITES: tuple[tuple[str, str], ...] = (
    ("model.language_model.", "model."),
)

# Regions the checkpoint carries that this text contract deliberately does not
# bind. Each is named with why it is not a silent drop.
UNBOUND_REGIONS: tuple[tuple[str, str], ...] = (
    ("model.visual.", "vision tower encoder; consumed by the vision target, not by this text contract"),
    ("mtp.", "MTP draft head; bound by the MTP binder, not by this text contract"),
)


def normalize_source_key(key: str) -> str:
    """Map a checkpoint key onto the contract's naming convention."""
    for old, new in _PREFIX_REWRITES:
        if key.startswith(old):
            return new + key[len(old):]
    return key


def _layer_of(engine: str) -> "str | None":
    m = re.match(r"layer\.(\d+)\.", engine)
    return m.group(1) if m else None


def _expected_source_keys(entry: dict) -> list:
    """Concrete source keys this entry accepts, in alias order.

    Each entry already carries its layer index (it is in the engine name), so
    matching needs no scan of the checkpoint: build the candidate keys and test
    set membership. The previous implementation tested every entry against
    every source key (74,520 x 296,475 ~= 2.2e10 comparisons), which is why the
    audit recorded in M_flashnext_contract.md never returned a result.
    """
    layer = _layer_of(entry["engine"])
    out = []
    for pat in entry["alias"]:
        if "{i}" in pat:
            if layer is None:
                continue
            out.append(pat.replace("{i}", layer))
        else:
            out.append(pat)
    return out


# NVFP4 publishes each logical tensor as a quad; the contract names only the
# `.weight` member, so the converter has to derive the rest or drop them.
_NVFP4_COMPANIONS = ("weight_scale", "weight_scale_2", "input_scale")
_LEAF_RE = re.compile(r"^(?P<stem>.+)\.(?P<leaf>[A-Za-z0-9_]+)$")


def audit(sources) -> dict:
    """Two-way coverage audit of the contract against a real key set.

    Forward:  every contract entry must find a source key, else the engine
              tensor has no weight source.
    Reverse:  every source key must be consumed, be a derived NVFP4 companion,
              or fall in a named UNBOUND_REGIONS prefix, else it is silently
              dropped.
    """
    srcset = set(sources)
    norm = {normalize_source_key(k) for k in srcset}

    matched = {}      # normalized source key -> engine
    engine_hit = {}   # engine -> normalized source key
    ambiguous = []
    for entry in E:
        if entry["engine"] in engine_hit:
            continue
        hits = [k for k in _expected_source_keys(entry) if k in norm]
        if not hits:
            continue
        if len(hits) > 1:
            ambiguous.append("%s: %r" % (entry["engine"], hits))
        engine_hit[entry["engine"]] = hits[0]
        matched[hits[0]] = entry["engine"]

    missing = [e["engine"] for e in E if e["engine"] not in engine_hit]

    consumed = set(matched)
    companions = set()
    for k in consumed:
        m = _LEAF_RE.match(k)
        if m and m.group("leaf") == "weight":
            for c in _NVFP4_COMPANIONS:
                ck = "%s.%s" % (m.group("stem"), c)
                if ck in norm:
                    companions.add(ck)

    unbound = {}
    unexpected = []
    for k in norm:
        if k in consumed or k in companions:
            continue
        for prefix, _reason in UNBOUND_REGIONS:
            if k.startswith(prefix):
                unbound[prefix] = unbound.get(prefix, 0) + 1
                break
        else:
            unexpected.append(k)

    # A tensor is a quad only if it starts one: flag a stem when at least one
    # companion is present but the set is not complete. Tensors that carry a
    # bare `.weight` and no companion at all are simply not quantised in this
    # checkpoint (measured: only the 512-expert MoE tensors are).
    quantised = set()
    incomplete = []
    for k in consumed:
        m = _LEAF_RE.match(k)
        if not (m and m.group("leaf") == "weight"):
            continue
        stem = m.group("stem")
        present = [c for c in _NVFP4_COMPANIONS if "%s.%s" % (stem, c) in norm]
        if not present:
            continue
        quantised.add(stem)
        if len(present) != len(_NVFP4_COMPANIONS):
            incomplete.append("%s (has %s)" % (stem, ",".join(present)))

    return {
        "contract_entries": len(E),
        "source_keys": len(srcset),
        "matched_sources": len(matched),
        "engines_covered": len(engine_hit),
        "ambiguous_aliases": ambiguous[:10],
        "missing_engines": missing[:40] + (["... (+%d)" % (len(missing) - 40)] if len(missing) > 40 else []),
        "missing_count": len(missing),
        "companion_keys_consumed": len(companions),
        "quantised_tensors": len(quantised),
        "unbound_regions": unbound,
        "unexpected_sources": sorted(unexpected)[:40],
        "unexpected_count": len(unexpected),
        "incomplete_nvfp4_quads": sorted(incomplete)[:20],
        "incomplete_quad_count": len(incomplete),
        "complete": (len(missing) == 0 and len(unexpected) == 0
                     and len(incomplete) == 0 and len(ambiguous) == 0),
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
