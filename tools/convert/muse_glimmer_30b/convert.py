# -*- coding: utf-8 -*-
"""Build the Muse-Glimmer-30B NVFP4 artifact from the NVIDIA NVFP4-QAT repo.

Layout (locked by the muse_glimmer_30b bindings):
  - linear layers (mlp gate_up/down, lm_head): NVFP4 direct pass-through
    (engine NVFP4 layout == source: packed e2m1 (n, k/2) + per-16 fp8 scales
    (n, k/16) + f32 divisor == source weight_scale_2 scalar)
  - attention projections (fp8 e4m3 source): fp8 row-scale re-encode
  - norms: bf16 with 1+w baking; qk norms: all-ones
  - embed: fp8 row-scale
"""
from __future__ import annotations

import argparse
import io
import json
import os
import struct
import time
from pathlib import Path

import torch

from tools.artifact.container import (ArtifactIdentity, ArtifactWriter, ResourceSpec, TensorSpec)
from tools.artifact.container import RAW_BYTES_V1
from tools.artifact.layouts import encode_direct, encode_nvfp4
from tools.convert.qwen3_6.common.inventory import CONTIGUOUS_LAYOUT

HIDDEN = 6656
LAYERS = 52
INTERMEDIATE = 19968
QHEADS, KVHEADS, HD = 32, 2, 128
VOCAB = 202048      # token domain (真 vocab)
ENGINE_VOCAB = 202112  # artifact 行 (N%128==0)
MODEL_ID = "muse-glimmer-30b"
WEIGHTS_ID = "nvfp4"
Q = QHEADS * HD          # 4096
KV = KVHEADS * HD        # 256
FUSED = 2 * Q + 2 * KV   # 8704  (q | k | gate | v)
GATE_UP = 2 * INTERMEDIATE  # 39936 (gate | up)

BF16 = "BF16"
FP32 = "FP32"
FP8 = "FP8_E4M3FN_ROW_BF16S"
NVFP4 = "NVFP4"
BLOCK_SCALE_LAYOUT = "blockscale-k16-m128x4-v1"
ROW_SCALE_LAYOUT = "row-scale-v1"

PREFIX = "model.language_model."


class MuseReader:
    """Manual safetensors reader (per-shard header parse + seek)."""

    def __init__(self, directory: str):
        self.directory = directory
        index = json.load(io.open(os.path.join(directory, "model.safetensors.index.json"),
                                  encoding="utf-8"))
        self._map = index["weight_map"]
        self._headers = {}

    def _header(self, shard: str) -> tuple[dict, int]:
        if shard not in self._headers:
            path = os.path.join(self.directory, shard)
            with open(path, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                self._headers[shard] = (json.loads(f.read(hlen).decode("utf-8")),
                                        8 + hlen)
        return self._headers[shard]

    def contains(self, key: str) -> bool:
        return key in self._map

    def get(self, key: str) -> torch.Tensor:
        shard = self._map[key]
        header, base = self._header(shard)
        meta = header[key]
        begin, end = meta["data_offsets"]
        dtype = meta["dtype"]
        shape = meta["shape"]
        path = os.path.join(self.directory, shard)
        with open(path, "rb") as f:
            f.seek(base + begin)
            raw = f.read(end - begin)
        if dtype == "U8":
            return torch.frombuffer(raw, dtype=torch.uint8).reshape(shape).clone().to(_DEVICE)
        if dtype == "BF16":
            return torch.frombuffer(raw, dtype=torch.bfloat16).reshape(shape).clone().to(_DEVICE)
        if dtype == "F32":
            return torch.frombuffer(raw, dtype=torch.float32).reshape(shape).clone().to(_DEVICE)
        if dtype in ("F8_E4M3", "F8_E4M3FN"):
            return torch.frombuffer(raw, dtype=torch.uint8).reshape(shape).clone().to(_DEVICE)
        raise ValueError(f"{key}: unsupported dtype {dtype}")


_DEVICE = torch.device('cpu')


_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32)


def e4m3_to_float(bits: torch.Tensor) -> torch.Tensor:
    sign = (bits >> 7) & 1
    exp = (bits >> 3) & 0x0F
    man = bits & 0x07
    value = torch.where(
        exp == 0,
        man.float() * (2.0 ** -6.0),
        (1.0 + man.float() / 8.0) * (2.0 ** (exp.float() - 7.0)),
    )
    return torch.where(sign == 1, -value, value)


def _scalar(reader: MuseReader, key: str, field: str) -> float:
    t = reader.get(key + "." + field)
    if t.numel() != 1:
        raise ValueError(f"{key}.{field}: expected scalar, got {tuple(t.shape)}")
    return float(t.reshape(()).float())


def fp8_row_scaled_stream(reader: MuseReader, key: str, n_src: int, n_total: int,
                             k: int, rows_per_block: int = 4096):
    """流式 fp8 row-scale payload (分块 yield; 源行 < 目标行时补零行)."""
    from tools.artifact.layouts import row_scale_geometry

    geometry = row_scale_geometry(FP8, (n_total, k))

    def row_block(b0, b1):
        b1s = min(b1, n_src)
        tb = reader.get(key)[b0:b1s].to(torch.bfloat16).float()
        if b1s < b1:
            pad = torch.zeros(b1 - b1s, k, dtype=torch.float32, device=_DEVICE)
            tb = torch.cat([tb, pad], dim=0)
        return tb

    # 遍 1: codes 段
    for b0 in range(0, n_total, rows_per_block):
        b1 = min(b0 + rows_per_block, n_total)
        tb = row_block(b0, b1)
        maxabs = tb.abs().amax(dim=1).clamp_min(1e-12)
        scale = (maxabs / 448.0).to(torch.bfloat16)
        codes = (tb / scale.unsqueeze(1).float()).clamp(-448.0, 448.0)
        codes = codes.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
        yield codes.cpu().numpy().tobytes()
        del tb, codes, scale
    # 遍 2: scales 段
    for b0 in range(0, n_total, rows_per_block):
        b1 = min(b0 + rows_per_block, n_total)
        tb = row_block(b0, b1)
        maxabs = tb.abs().amax(dim=1).clamp_min(1e-12)
        scale = (maxabs / 448.0).to(torch.bfloat16)
        yield encode_direct(scale.cpu(), "BF16")
        del tb, scale


def fp8_row_scaled_blocks(t: torch.Tensor, rows_per_block: int | None = 8192) -> bytes:
    from tools.artifact.layouts import row_scale_geometry

    n, k = t.shape
    geometry = row_scale_geometry(FP8, (n, k))
    payload = bytearray(geometry.payload_bytes)
    block = rows_per_block or n
    for b0 in range(0, n, block):
        b1 = min(b0 + block, n)
        tb = t[b0:b1].float()
        maxabs = tb.abs().amax(dim=1).clamp_min(1e-12)
        scale = (maxabs / 448.0).to(torch.bfloat16)
        codes = (tb / scale.unsqueeze(1).float()).clamp(-448.0, 448.0)
        codes = codes.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
        cbytes = codes.cpu().numpy().tobytes()
        code_off = b0 * k
        payload[code_off:code_off + len(cbytes)] = cbytes
        sbytes = encode_direct(scale.cpu(), "BF16")
        s_off = geometry.scale_plane_offset + b0 * 2
        payload[s_off:s_off + len(sbytes)] = sbytes
    return bytes(payload)


def bake_norm(reader: MuseReader, key: str) -> bytes:
    w = reader.get(key + ".weight")
    if w.dtype != torch.bfloat16:
        w = w.to(torch.bfloat16)
    baked = (1.0 + w.float()).to(torch.bfloat16).contiguous()
    return encode_direct(baked, "BF16")


def ones_norm(n: int) -> bytes:
    return encode_direct(torch.ones(n, dtype=torch.bfloat16), "BF16")


def obj(name: str, fmt: str, shape: tuple) -> TensorSpec:
    if fmt == BF16 or fmt == FP32:
        layout = CONTIGUOUS_LAYOUT
    elif fmt == NVFP4:
        layout = BLOCK_SCALE_LAYOUT
    else:
        layout = ROW_SCALE_LAYOUT
    return TensorSpec(name, shape, fmt, layout)


def mlp_layer_fmt(reader: MuseReader, layer: int, proj: str) -> str:
    """源 dtype -> 对象格式: packed nvfp4 -> NVFP4; fp8 全宽/bf16 -> FP8."""
    key = f"{PREFIX}layers.{layer}.mlp.{proj}.weight"
    t = reader.get(key)
    if str(t.dtype) == 'torch.uint8':
        k = tuple(t.shape)[1]
        full_k = HIDDEN if proj in ('gate_proj', 'up_proj') else INTERMEDIATE
        return NVFP4 if k == full_k // 2 else FP8
    return FP8  # bf16 直通也 fp8 编码


def resource_specs(model_dir: str):
    """frontend 资源对象 (按目录文件; tokenizer_config 合成 qwen3.6 前端语义;
    chat_template 兜底内嵌)."""
    specs = []
    for name in ("frontend/tokenizer.json", "frontend/generation_config.json"):
        path = os.path.join(model_dir, name.removeprefix("frontend/"))
        with open(path, "rb") as f:
            data = f.read()
        specs.append((name, data))
    # tokenizer_config.json: patch qwen3.6 前端所需字段 (真分词在 tokenizer.json,
    # 此文件仅供引擎前端特殊 token/模板控制).
    tc = json.load(open(os.path.join(model_dir, "tokenizer_config.json"),
                        encoding="utf-8"))
    tc["add_bos_token"] = False
    tc["add_prefix_space"] = False
    tc["pad_token"] = "<|endoftext|>"
    # chat_template 字段必须存在 (前端校验): 兜底拼接模板 (官方未发布).
    tmpl = tc.get("chat_template")
    if isinstance(tmpl, dict):
        tmpl = tmpl.get("default", "")
    if not tmpl:
        tmpl = ("{% for message in messages %}{{ message['content'] }}"
                "{% if not loop.last %}\n{% endif %}{% endfor %}")
    # 引擎前端对 chat_template 做官方白名单 sha256 校验 -> 用 qwen 官方模板内容
    # (生成走 qwen 对话格式; Muse 官方模板发布后可替换).
    tmpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "qwen_chat_template.jinja")
    with open(tmpl_path, "rb") as f:
        qwen_tmpl = f.read()
    tc["chat_template"] = qwen_tmpl.decode("utf-8")
    # added_tokens_decoder 引擎前端强制要求存在 (Muse 官方未发布); 空对象即可.
    tc.setdefault("added_tokens_decoder", {})
    specs.append(("frontend/tokenizer_config.json",
                  json.dumps(tc, ensure_ascii=False).encode("utf-8")))
    specs.append(("frontend/chat_template.jinja", qwen_tmpl))
    # 处理器配置: 引擎前端按 qwen 视觉几何编译 (patch 16/temporal 2/merge 2) 且无条件校验
    # -> 直接嵌入 qwen 官方同款 preprocessor 资源 (与 pinned hash 一致, 文本场景语义等同 qwen).
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "qwen_preprocessor_config.json"), "rb") as f:
        specs.append(("frontend/preprocessor_config.json", f.read()))
    with open(os.path.join(here, "qwen_video_preprocessor_config.json"), "rb") as f:
        specs.append(("frontend/video_preprocessor_config.json", f.read()))
    return specs


def object_plan(reader: MuseReader, resources) -> list[TensorSpec]:
    plan = []
    for name, data in resources:
        plan.append(ResourceSpec(name, RAW_BYTES_V1, len(data)))
    plan.append(obj("text/token_embedding", FP8, (ENGINE_VOCAB, HIDDEN)))
    for layer in range(LAYERS):
        p = f"text/layers/{layer}/"
        gu_fmt = mlp_layer_fmt(reader, layer, 'gate_proj')
        dn_fmt = mlp_layer_fmt(reader, layer, 'down_proj')
        entries = [
            obj(p + "input_norm", BF16, (HIDDEN,)),
            obj(p + "attention/query", FP8, (Q, HIDDEN)),
            obj(p + "attention/key", FP8, (KV, HIDDEN)),
            obj(p + "attention/gate", FP8, (Q, HIDDEN)),
            obj(p + "attention/value", FP8, (KV, HIDDEN)),
            obj(p + "attention/query_norm", BF16, (HD,)),
            obj(p + "attention/key_norm", BF16, (HD,)),
            obj(p + "attention/output", FP8, (HIDDEN, Q)),
            obj(p + "post_attention_layernorm", BF16, (HIDDEN,)),
            obj(p + "pre_feedforward_layernorm", BF16, (HIDDEN,)),
            obj(p + "mlp/gate", gu_fmt, (INTERMEDIATE, HIDDEN)),
        ]
        if gu_fmt == NVFP4:
            entries.append(obj(p + "mlp/gate/input_scale_divisor", FP32, ()))
        entries += [
            obj(p + "mlp/up", gu_fmt, (INTERMEDIATE, HIDDEN)),
        ]
        if gu_fmt == NVFP4:
            entries.append(obj(p + "mlp/up/input_scale_divisor", FP32, ()))
        entries.append(obj(p + "mlp/down", dn_fmt, (HIDDEN, INTERMEDIATE)))
        if dn_fmt == NVFP4:
            entries.append(obj(p + "mlp/down/input_scale_divisor", FP32, ()))
        entries.append(obj(p + "post_feedforward_layernorm", BF16, (HIDDEN,)))
        plan += entries
    plan += [
        obj("text/final_norm", BF16, (HIDDEN,)),
        obj("text/output_head", NVFP4, (ENGINE_VOCAB, HIDDEN)),
        obj("text/output_head/input_scale_divisor", FP32, ()),
    ]
    return plan


def encode_nvfp4_direct(reader: MuseReader, key: str, n: int, k: int,
                        divisor: float | None = None, pad_n: int | None = None) -> bytes:
    """NVFP4 源直通: packed (n,k/2) + per-16 fp8 s1 + f32 divisor (=s2).
    pad_n: 目标 artifact 行数 (源行不足时补零行; 零 scale 0x00 合法)."""
    packed = reader.get(key + ".weight")
    if packed.dtype != torch.uint8 or tuple(packed.shape) != (n, k // 2):
        raise ValueError(f"{key}: expected packed {(n, k // 2)}, got {tuple(packed.shape)}")
    s1 = reader.get(key + ".weight_scale")
    if tuple(s1.shape) != (n, k // 16):
        raise ValueError(f"{key}: scale matrix {tuple(s1.shape)} != {(n, k // 16)}")
    if divisor is None:
        divisor = _scalar(reader, key, "weight_scale_2") if reader.contains(
            key + ".weight_scale_2") else 1.0
    if pad_n is not None and pad_n > n:
        packed = torch.cat([packed, torch.zeros(pad_n - n, k // 2, dtype=torch.uint8,
                                        device=_DEVICE)], dim=0)
        s1 = torch.cat([s1, torch.zeros(pad_n - n, k // 16, dtype=torch.uint8,
                             device=_DEVICE)], dim=0)
    return encode_nvfp4(packed, s1.view(torch.uint8),
                        struct.pack("<f", divisor), (pad_n or n, k))


def fused_attn_payload(reader: MuseReader, layer: int, out_only: bool = False) -> bytes:
    base = f"{PREFIX}layers.{layer}.self_attn."
    if out_only:
        return fp8_row_scaled_blocks(
            dequant_any(reader, base + "o_proj", HIDDEN, Q), 4096)
    parts = []
    for proj, rows in (("q_proj", Q), ("k_proj", KV), ("gate_proj", Q), ("v_proj", KV)):
        parts.append(dequant_any(reader, base + proj, rows, HIDDEN))
    return fp8_row_scaled_blocks(torch.cat(parts, dim=0), rows_per_block=4096)


def dequant_any(reader: MuseReader, key: str, n: int, k: int) -> torch.Tensor:
    """统一三态 dequant -> float:
      bf16 直通; fp8 全宽 uint8 (n,k) x 标量; packed uint8 (n,k/2) x per-16
      fp8 矩阵 x s2 标量(可选)."""
    full = reader.get(key + ".weight")
    if full.dtype == torch.bfloat16:
        if tuple(full.shape) != (n, k):
            raise ValueError(f"{key}: bf16 shape {tuple(full.shape)} != {(n, k)}")
        return full.float()
    if full.dtype != torch.uint8:
        raise ValueError(f"{key}: unsupported dtype {full.dtype}")
    shp = tuple(full.shape)
    if shp == (n, k):
        s1 = _scalar(reader, key, "weight_scale")
        return e4m3_to_float(full) * s1
    if shp == (n, k // 2):
        s1 = reader.get(key + ".weight_scale")
        if tuple(s1.shape) != (n, k // 16):
            raise ValueError(f"{key}: scale matrix {tuple(s1.shape)} != {(n, k // 16)}")
        s2 = _scalar(reader, key, "weight_scale_2") if reader.contains(
            key + ".weight_scale_2") else 1.0
        low = (full & 0x0F).long()
        hi = ((full >> 4) & 0x0F).long()
        vals = torch.stack([_E2M1_TABLE[low], _E2M1_TABLE[hi]], dim=-1).reshape(n, k)
        s1f = e4m3_to_float(s1.view(torch.uint8)).reshape(n, k // 16, 1)
        out = vals.reshape(n, k // 16, 16) * s1f
        return out.reshape(n, k) * s2
    raise ValueError(f"{key}: unexpected shape {shp}")


def fused_gateup_payload(reader: MuseReader, layer: int, fmt: str) -> bytes:
    base = f"{PREFIX}layers.{layer}.mlp."
    if fmt == FP8:
        g = dequant_any(reader, base + "gate_proj", INTERMEDIATE, HIDDEN)
        u = dequant_any(reader, base + "up_proj", INTERMEDIATE, HIDDEN)
        return fp8_row_scaled_blocks(torch.cat([g, u], dim=0), 4096)
    g = reader.get(base + "gate_proj.weight")
    u = reader.get(base + "up_proj.weight")
    packed = torch.cat([g, u], dim=0)
    s1 = torch.cat([reader.get(base + "gate_proj.weight_scale"),
                    reader.get(base + "up_proj.weight_scale")], dim=0)
    divisor = _scalar(reader, base + "gate_proj", "weight_scale_2") if reader.contains(
        base + "gate_proj.weight_scale_2") else 1.0
    return encode_nvfp4(packed, s1.view(torch.uint8),
                        struct.pack("<f", divisor), (GATE_UP, HIDDEN))


def convert(args) -> None:
    torch.manual_seed(0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    reader = MuseReader(args.model)
    resources = resource_specs(args.model)
    plan = object_plan(reader, resources)
    print(f"plan: {len(plan)} objects ({len(resources)} resources)", flush=True)
    with ArtifactWriter(out, ArtifactIdentity(MODEL_ID, WEIGHTS_ID), plan) as writer:
        res_map = {n: d for n, d in resources}
        for index, spec in enumerate(plan, start=1):
            t0 = time.time()
            name = spec.name
            if name in res_map:
                payload = res_map[name]
                writer.write(name, payload)
                print(f"[{index}/{len(plan)}] {name} ({time.time() - t0:.1f}s)",
                      flush=True)
                continue
            if name == "text/token_embedding":
                payload = fp8_row_scaled_stream(reader, PREFIX + "embed_tokens.weight",
                                                VOCAB, ENGINE_VOCAB, HIDDEN)
            elif name == "text/final_norm":
                payload = bake_norm(reader, PREFIX + "norm")
            elif name.endswith("/input_norm"):
                layer = name.split("/")[2]
                payload = bake_norm(reader, f"{PREFIX}layers.{layer}.input_layernorm")
            elif name.endswith("/post_attention_layernorm"):
                layer = name.split("/")[2]
                payload = bake_norm(reader, f"{PREFIX}layers.{layer}.post_attention_layernorm")
            elif name.endswith("/pre_feedforward_layernorm"):
                layer = name.split("/")[2]
                payload = bake_norm(reader, f"{PREFIX}layers.{layer}.pre_feedforward_layernorm")
            elif name.endswith("/post_feedforward_layernorm"):
                layer = name.split("/")[2]
                payload = bake_norm(reader, f"{PREFIX}layers.{layer}.post_feedforward_layernorm")
            elif name.endswith("/attention/query_norm"):
                payload = ones_norm(HD)
            elif name.endswith("/attention/key_norm"):
                payload = ones_norm(HD)
            elif name.endswith("/attention/query"):
                layer = int(name.split("/")[2])
                payload = fp8_row_scaled_blocks(
                    dequant_any(reader, f"{PREFIX}layers.{layer}.self_attn.q_proj", Q, HIDDEN),
                    4096)
            elif name.endswith("/attention/key"):
                layer = int(name.split("/")[2])
                payload = fp8_row_scaled_blocks(
                    dequant_any(reader, f"{PREFIX}layers.{layer}.self_attn.k_proj", KV, HIDDEN),
                    4096)
            elif name.endswith("/attention/gate"):
                layer = int(name.split("/")[2])
                payload = fp8_row_scaled_blocks(
                    dequant_any(reader, f"{PREFIX}layers.{layer}.self_attn.gate_proj", Q, HIDDEN),
                    4096)
            elif name.endswith("/attention/value"):
                layer = int(name.split("/")[2])
                payload = fp8_row_scaled_blocks(
                    dequant_any(reader, f"{PREFIX}layers.{layer}.self_attn.v_proj", KV, HIDDEN),
                    4096)
            elif name.endswith("/attention/output"):
                layer = int(name.split("/")[2])
                payload = fp8_row_scaled_blocks(
                    dequant_any(reader, f"{PREFIX}layers.{layer}.self_attn.o_proj", HIDDEN, Q),
                    4096)
            elif name.endswith("/mlp/gate"):
                layer = int(name.split("/")[2])
                fmt = mlp_layer_fmt(reader, layer, 'gate_proj')
                if fmt == FP8:
                    payload = fp8_row_scaled_blocks(
                        dequant_any(reader, f"{PREFIX}layers.{layer}.mlp.gate_proj",
                                    INTERMEDIATE, HIDDEN), 4096)
                else:
                    payload = encode_nvfp4_direct(
                        reader, f"{PREFIX}layers.{layer}.mlp.gate_proj", INTERMEDIATE, HIDDEN)
            elif name.endswith("/mlp/up"):
                layer = int(name.split("/")[2])
                fmt = mlp_layer_fmt(reader, layer, 'up_proj')
                if fmt == FP8:
                    payload = fp8_row_scaled_blocks(
                        dequant_any(reader, f"{PREFIX}layers.{layer}.mlp.up_proj",
                                    INTERMEDIATE, HIDDEN), 4096)
                else:
                    payload = encode_nvfp4_direct(
                        reader, f"{PREFIX}layers.{layer}.mlp.up_proj", INTERMEDIATE, HIDDEN)
            elif name.endswith("/mlp/down"):
                layer = int(name.split("/")[2])
                if mlp_layer_fmt(reader, layer, 'down_proj') == FP8:
                    payload = fp8_row_scaled_blocks(
                        dequant_any(reader, f"{PREFIX}layers.{layer}.mlp.down_proj",
                                    HIDDEN, INTERMEDIATE), 4096)
                else:
                    payload = encode_nvfp4_direct(
                        reader, f"{PREFIX}layers.{layer}.mlp.down_proj", HIDDEN,
                        INTERMEDIATE)
            elif name.endswith("/input_scale_divisor"):
                payload = encode_direct(torch.ones(1, dtype=torch.float32), "FP32")
            elif name == "text/output_head":
                payload = encode_nvfp4_direct(reader, "lm_head", VOCAB, HIDDEN,
                                            pad_n=ENGINE_VOCAB)
            else:
                raise RuntimeError("unhandled object: " + name)
            writer.write(name, payload)
            del payload
            print(f"[{index}/{len(plan)}] {name} ({time.time() - t0:.1f}s)", flush=True)
    print("conversion complete:", out, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    global _DEVICE, _E2M1_TABLE
    if getattr(args, 'device', 'cpu') == 'cuda' and torch.cuda.is_available():
        _DEVICE = torch.device('cuda')
    _E2M1_TABLE = _E2M1_TABLE.to(_DEVICE)
    convert(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
