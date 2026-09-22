#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model_import.py — recognition/decision engine behind the model import wizard.

Aimed at users who know nothing about AI. Handed a model folder or file, this
module answers three questions in plain language:

  1. What format is this?   (.ninfer / GGUF / HuggingFace-safetensors / other)
  2. Will this machine run it?  (architecture supported by ninfer, VRAM enough)
  3. What should happen next?   (run as-is / convert automatically / politely
                                 explain why it is not supported)

Design rules:
  * Standard library only -- usable from the GUI, from packaging and offline,
    never imports torch.
  * Any failing step degrades to "say what is known", never raises at the user.
  * Every user-visible string goes through gui_i18n.t() (keys in i18n_misc.py);
    sizes are quoted in GB, never in bytes.

Conversion matrix (verdict() hands out the route, the CLIs do the work):
  .ninfer                              -> run directly
  HF safetensors, whitelisted arch, bf16     -> convert.py        (groupwise-int, CPU ok)
  HF safetensors, whitelisted arch, NVFP4    -> convert_nvfp4.py  (needs an unsloth source)
  GGUF F16/BF16/F32 (whitelisted arch)       -> gguf_extract.py   (bf16 first, then convert.py)
  GGUF k-quant / other arch / unrecognised   -> polite refusal + advice

It also surfaces the archkit gap report. Two writers put a manifest under
tools/archkit/out/<model-id>/, and find_manifest() accepts both names:

  * tools/archkit/adapt.py      -> manifest.json, carrying gaps [{need, tier,
    action}] plus config.h next to it when a header could be emitted. This is the
    only writer that has an operator catalogue. It does NOT withhold the header
    under a `config.h.BLOCKED` name: a `new_op` gap is reported in the manifest
    while config.h is still written (measured on out/ornith-1.5-9b-q4-k-m/), and
    the refusal path (unreadable layer kinds, geometry with no dense-FFN width)
    writes no header at all and records the reason in the manifest. Which header
    file the report names is therefore read off the disk -- see
    header_file_state().
  * tools/archkit/gen_target.py -> manifest.json (arch_manifest.json before S55),
    carrying NO gap data at all: it has no catalogue, so it writes
    `"gaps": null, "gaps_measured": false` to say "not measured".

A blocked import is reported as a verdict, never as a silent success -- and so is
an unmeasured one. Four distinct outcomes, four distinct exit codes, and the exit
code is 1:1 with the verdict (never with "the report printed OK"):

  CLEAR      a gaps LIST was read and it has no new_op entry        (--gaps rc 0)
  BLOCKED    a gaps LIST was read and it has a new_op entry         (--gaps rc 3)
             rc used to be 0 here, which made this the one place in the tree where
             the verdict said "not fit to convert" and the exit code said "fine".
             No caller in this tree consumed it, but `python ... --gaps X && ...`
             would have become a false gate the moment one was written. The three
             "not servable" outcomes now all carry a non-zero code, and the code
             says WHICH one (this is also what tools/archkit/adapt_all.py means by
             its own rc 3).
  UNKNOWN    no gaps list was readable: the field is absent, or it is null, or it
             is not a list, or an element is not an object, or an element's `tier`
             is not a tier this tree writes (hook / new_op / covered / post -- the
             verdict is decided by `tier == "new_op"`, so an unreadable tier is what
             "no new_op gaps" cannot be said on), or the manifest
             carries a `measured` key (see below)                  (--gaps rc 2)
             "Not measured" is NOT "no gaps" -- a manifest that says `"gaps": null`
             must never render as "the engine can take it as-is" (that false
             all-clear is exactly what `gaps: null` used to produce here, because
             the guard tested `"gaps" not in data` and an explicit null IS in data).
  (--gaps rc 1 = no manifest, or the file does not parse.)
             This is the ONLY remaining rc-1 outcome: "the manifest could not be
             read". It used to be shared with "the gaps list has an element that is
             not an object", because that path raised AttributeError out of
             import_gap_report() -- the GUI's caller then got a Python traceback AND
             rc 1, indistinguishable from a missing file.

  `--require-servable` collapses the verdict to a gate: rc 0 only for CLEAR,
  everything that was read but is not clear -> rc 3 (the reason still goes to
  stdout, plus one line on stderr). Use it at a call site that means "stop unless
  this model is servable", so the intent is in the command line instead of being
  inferred from a number.

The ONE honoured flag key is `gaps_measured`. A manifest that spells it `measured`
is not quietly accepted: nothing in this tree reads `measured`, so a writer using
that spelling would have its own denial ignored (`{"gaps": [], "measured": false}`
used to read as CLEAR -- a measured, gap-free model) and its claim accepted by
luck. Such a manifest is reported as UNKNOWN and the shape names the key, because
an unhonoured flag is worse than a missing one: it looks like a check is running.

See import_gap_report() / render_gap_report_text() at the bottom of this file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gui_i18n import has as i18n_has, set_lang, t  # noqa: E402

# ---------------------------------------------------------------- constants

ARTIFACT_MAGIC = b"NINFER\x00\x02"   # ninfer's own container format header
GGUF_MAGIC = b"GGUF"                 # llama.cpp ecosystem format header
SUPPORTED_MODEL_IDS = ("qwen3.6-27b", "qwen3.8-27b")  # engine registry whitelist
# GGUF tensor type -> bytes per element (size estimate).  These are the *real* ggml
# ids from the gguf spec (`tools/archkit/gguf_tensors.py` GGML_TYPES); the table here
# used to be shifted by one from id 9 up and to call id 2 "BF16", which is Q4_0 --
# measured on Ornith-1.5-9B-Q4_K_M.gguf, whose table is {0: 184, 12: 223, 14: 35} =
# F32 x184, Q4_K x223, Q6_K x35 (the same histogram tools/archkit/gguf_tensors.py
# prints, and the same one specs/ornith-1.5-9b-q4-k-m_spec.json records).  With the old
# table the same file printed "Q5_K/Q8_K", and a Q4_0 (id 2) file printed "BF16 half"
# and was offered a conversion.
GGUF_TYPE_NAMES = {
    0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 4: 'Q4_2', 5: 'Q4_3', 6: 'Q5_0',
    7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K',
    14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S',
    20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S', 23: 'IQ4_XS', 24: 'I8', 25: 'I16',
    26: 'I32', 27: 'I64', 28: 'F64', 29: 'IQ1_M', 30: 'BF16',
}
#: Tensor types this tree's GGUF reader can turn into bf16.  Asked of the reader
#: (tools/convert/gguf_kquant.DEQUANTIZERS) instead of kept as a second literal: as a
#: literal {F32,F16,BF16} the wizard went on telling the user "there is no Q4_K/Q6_K
#: decoder" long after one existed, and refused a file the converter on the same
#: machine accepted.  The literal survives only as the fallback for a checkout where
#: the reader cannot be imported.
GGUF_COPYABLE_TYPE_IDS_FALLBACK = {0, 1, 30}


def _reader_copyable_type_ids() -> set:
    """The ggml ids tools/convert/gguf_kquant.py has a dequantiser for."""
    try:
        import sys as _sys
        from pathlib import Path as _Path
        root = str(_Path(__file__).resolve().parents[2])
        if root not in _sys.path:
            _sys.path.insert(0, root)
        from tools.convert import gguf_kquant as _kq
        return set(_kq.DEQUANTIZERS)
    except Exception:                       # noqa: BLE001 - fall back, never crash
        return set(GGUF_COPYABLE_TYPE_IDS_FALLBACK)


GGUF_COPYABLE_TYPE_IDS = _reader_copyable_type_ids()


class _RulePatterns:
    """Cached compiled patterns of tools/convert/gguf_names.RULES per architecture.

    The wizard used to know only its own qwen-shaped literal, so for the real Ornith
    file it reported "271 tensors have no GGUF->HF name mapping" while the converter
    it hands the file to mapped 442/442.  Two copies of one mapping is how that
    happens; this asks the mapping's owner.
    """

    _cache: dict = {}

    @classmethod
    def for_arch(cls, arch: str) -> list:
        if arch in cls._cache:
            return cls._cache[arch]
        patterns = []
        try:
            import sys as _sys
            from pathlib import Path as _Path
            root = str(_Path(__file__).resolve().parents[2])
            if root not in _sys.path:
                _sys.path.insert(0, root)
            from tools.convert import gguf_names as _gn
            patterns = [rule.pattern for rule in _gn.RULES.get(arch or "", ())]
        except Exception:                   # noqa: BLE001 - fall back, never crash
            patterns = []
        cls._cache[arch] = patterns
        return patterns


def gguf_reader_support_names() -> str:
    """Human spelling of what the reader decodes, for the refusal text."""
    try:
        import sys as _sys
        from pathlib import Path as _Path
        root = str(_Path(__file__).resolve().parents[2])
        if root not in _sys.path:
            _sys.path.insert(0, root)
        from tools.convert import gguf_kquant as _kq
        return "/".join(sorted(_kq.type_name(t) for t in _kq.DEQUANTIZERS))
    except Exception:                       # noqa: BLE001
        return "F32/F16/BF16"
GGUF_ELEMENT_BYTES = {0: 4, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 1, 9: 1,
                      10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 30: 2}
# GGUF tensor type id -> i18n key (names shown to the user, both languages).
#: K-series (llama.cpp's block-compressed family): unsupported today.
KQUANT_TYPE_IDS = {10, 11, 12, 13, 14, 15}
GGUF_QUANT_KEYS = {
    0: 'imp.qn.f32', 1: 'imp.qn.f16', 2: 'imp.qn.q4_0', 3: 'imp.qn.q4_1',
    4: 'imp.qn.q4_2', 5: 'imp.qn.q4_3', 6: 'imp.qn.q5_0', 7: 'imp.qn.q5_1',
    8: 'imp.qn.q8_0', 9: 'imp.qn.q8_1', 10: 'imp.qn.q2_k', 11: 'imp.qn.q3_k',
    12: 'imp.qn.q4_k', 13: 'imp.qn.q5_k', 14: 'imp.qn.q6_k', 15: 'imp.qn.q8_k',
    16: 'imp.qn.iq', 17: 'imp.qn.iq', 18: 'imp.qn.iq', 19: 'imp.qn.iq',
    20: 'imp.qn.iq', 21: 'imp.qn.iq', 22: 'imp.qn.iq', 23: 'imp.qn.iq',
    24: 'imp.qn.i8', 25: 'imp.qn.i16', 26: 'imp.qn.i32', 27: 'imp.qn.i64',
    28: 'imp.qn.f64', 29: 'imp.qn.iq', 30: 'imp.qn.bf16',
}

# GGUF file header, straight from the gguf spec:
#   magic "GGUF"(4) + version(u32) + tensor_count(u64) + metadata_kv_count(u64) = 24 B
# It used to be read as "<IQI" (4+8+4 = 20 B), which leaves 4 bytes of the file
# unconsumed: the metadata block then starts 4 bytes early and the very first "string
# length" comes out as garbage.  Measured on a real Ornith-1.5-9B-Q4_K_M.gguf
# (5.4 GiB): the bogus length is 85_899_345_920 bytes (80 GiB), so the scan died in a
# bare MemoryError -- which the caller renders as "scan error: " with nothing after it.
# tools/archkit/gguf_tensors.py has always used the correct "<IQQ"; this now matches it.
GGUF_HEADER = "<IQQ"
GGUF_HEADER_BYTES = 4 + struct.calcsize(GGUF_HEADER)
#: Sanity ceilings.  A GGUF metadata string is a short key/value blurb, never
#: megabytes, and the tensor table of the largest model this engine runs is a few
#: thousand entries.  These bounds turn "the parser is misaligned" into a named error
#: instead of a multi-gigabyte read request.
GGUF_MAX_STRING = 1 << 24        # 16 MiB
GGUF_MAX_ENTRIES = 1 << 20
#: Metadata keys that name the model family.  The GGUF carries its own architecture;
#: the file name is only the fallback, because a renamed download must still work.
GGUF_ARCH_KEYS = ("general.architecture", "general.basename", "general.name")
#: Metadata keys that carry the layer accounting.  `<arch>.block_count` counts the
#: nextn/draft blocks TOO, so on its own it is not the decoder depth -- see
#: _gguf_main_layers().  Matched by suffix because the prefix is the *GGUF*
#: architecture's (`qwen35.block_count`), not the family this wizard files it under.
GGUF_LAYER_KV_SUFFIXES = (".block_count", ".nextn_predict_layers")
#: ggml metadata integer types -> struct format.  Both of the keys above are ktype 4
#: (uint32) in the real file (measured on Ornith-1.5-9B-Q4_K_M.gguf, 42 KV entries),
#: but the two other spellings llama.cpp can emit are read too rather than skipped.
GGUF_INT_KV_FORMATS = {4: "<I", 5: "<i", 10: "<Q", 11: "<q"}
#: Decoder families the engine actually has a target for.  The registered targets'
#: validate_config() accept exactly these (tools/convert/qwen3_6/convert.py,
#: tools/convert/qwen3_6_35b_a3b/convert.py, muse_glimmer_30b), and
#: tools/convert/import_model.py:72 lists the same set.  Quoted in refusals so the
#: user is told which piece is missing instead of only "not supported".
ENGINE_DECODER_FAMILIES = ("qwen3_5", "qwen3_5_moe", "muse_glimmer",
                           "qwen4_exp", "qwen4_exp_text")
#: Tensor names tools/convert/gguf_extract.py's NAME_MAP can translate.  Anything else
#: in a GGUF has no mapping at all, which is why an unknown family cannot be converted
#: even when its tensors are plain F16.
GGUF_MAPPED_NAMES = ("token_embd.weight", "output_norm.weight", "output.weight",
                     "attn_norm.weight", "attn_norm_2.weight", "attn_q.weight",
                     "attn_k.weight", "attn_v.weight", "attn_output.weight",
                     "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")

# Official download links (environment self-check / error advice for newcomers)
LINKS = {
    "nvidia_driver": "https://www.nvidia.cn/drivers/",       # NVIDIA driver
    "python": "https://www.python.org/downloads/",            # Python
    "huggingface": "https://huggingface.co/",                 # model hub
}

# Engine TextConfig geometry (qwen3_6_27b target): only this Qwen3 spec can run
ENGINE_HIDDEN = 5120
ENGINE_LAYERS = 64
#: The convert targets this checkout registers (tools/convert/import_model.py:54-59
#: REGISTERED_TARGETS).  A refusal quotes them so "not supported" becomes "no target
#: for this family, the registered ones are ...".
REGISTERED_TARGET_NAMES = ("qwen4_exp", "qwen3_8_27b", "qwen3_6_27b", "qwen3_6_35b_a3b",
                           "muse_glimmer_30b")


# ---------------------------------------------------------------- data

@dataclass
class GpuInfo:
    """nvidia-smi probe result; present=False when no probe is possible."""
    present: bool = False
    name: str = ""
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    driver: str = ""

    def as_dict(self) -> dict:
        return {"present": self.present, "name": self.name,
                "vram_total_gb": self.vram_total_gb, "vram_free_gb": self.vram_free_gb,
                "driver": self.driver}


@dataclass
class ModelScan:
    """Read-only scan verdict for one path the user handed in."""
    kind: str = "unknown"                 # ninfer | gguf | hf | lora | unknown
    path: str = ""
    friendly: str = ""                    # one-line format description
    model_id: str = ""                    # whitelist id (ninfer: artifact identity; hf: geometry)
    weights_id: str = ""                  # ninfer quantisation tier (nvfp4-dflash2 etc.)
    arch_note: str = ""                   # architecture description (for humans)
    params_billions: float = 0.0          # parameters (estimated)
    weight_bytes: int = 0                 # stored weight bytes (~ lower bound for memory)
    quant_name: str = ""                  # quantisation description
    gguf_types: dict = field(default_factory=dict)   # gguf: type_id -> tensor count
    hidden_size: int = 0                  # hf text_config.hidden_size (geometry check)
    n_layers: int = 0                     # hf text_config.num_hidden_layers
    issues: list = field(default_factory=list)       # non-fatal problems found while scanning
    unreadable: bool = False              # the file exists but cannot be read at all
    config_error: str = ""                # config.json could not be parsed (why)
    unmapped: list = field(default_factory=list)     # gguf tensors no converter can name
    arch_names: list = field(default_factory=list)   # model_type / architectures as declared

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Verdict:
    """Human-readable disposition. action is one of:
    run | convert_groupwise | convert_nvfp4 | need_nvfp4_source |
    vram_short | unsupported_arch | unsupported_quant | need_base_fuse | unknown"""
    action: str = "unknown"
    level: str = "info"                   # ok | warn | bad
    title: str = ""                       # headline, e.g. "this model can run!"
    detail: str = ""                      # long-form, plain language
    tips: list = field(default_factory=list)      # bullet advice


# ---------------------------------------------------------------- probing

def probe_file_magic(path: Path) -> bytes:
    """Read the first 8 bytes of a file; b'' on failure."""
    try:
        with open(path, "rb") as f:
            return f.read(8)
    except OSError:
        return b""


def scan_path(path_str: str) -> ModelScan:
    """Entry point: user-supplied file/folder path -> full scan verdict. Never raises."""
    p = Path(path_str)
    scan = ModelScan(path=str(p))
    try:
        if not p.exists():
            scan.issues.append(t('imp.err_not_found'))
            return scan
        if p.is_file():
            magic = probe_file_magic(p)
            if magic.startswith(ARTIFACT_MAGIC) or p.suffix.lower() == ".ninfer":
                return _scan_ninfer(p)
            if magic.startswith(GGUF_MAGIC):
                return _scan_gguf(p)
            scan.kind = "unknown"
            scan.friendly = t('imp.f_unknown_file')
            scan.issues.append(t('imp.err_unknown_file'))
            return scan
        # Folder: safetensors model first (config.json + .safetensors)
        cfg = p / "config.json"
        sts = list(p.glob("*.safetensors"))
        if cfg.is_file() and sts:
            return _scan_hf(p)
        if cfg.is_file() and not sts:
            # config.json is there but no weights: "this folder holds no model file"
            # is a contradiction the user cannot act on. Say which half is missing.
            scan.kind = "hf"
            scan.friendly = t('imp.f_hf_noconfig_weights')
            scan.unreadable = True
            scan.issues.append(t('imp.err_cfg_no_weights', path=str(p)))
            return scan
        # LoRA adapter folder (adapter_config.json, peft_type=LORA)
        acfg = p / "adapter_config.json"
        if acfg.is_file():
            return _scan_lora(p)
        # A folder may just contain loose .gguf / .ninfer files
        ggs = list(p.glob("*.gguf"))
        nfs = list(p.glob("*.ninfer"))
        if ggs:
            return _scan_gguf(ggs[0])
        if nfs:
            return _scan_ninfer(nfs[0])
        scan.kind = "unknown"
        scan.friendly = t('imp.f_empty_dir')
        scan.issues.append(t('imp.err_no_model_in_dir'))
        return scan
    except Exception as e:  # any surprise degrades instead of crashing
        scan.issues.append(t('imp.err_scan_failed', err=e))
        return scan


def _read_hf_config(p: Path) -> dict:
    try:
        with open(p / "config.json", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_hf_config_ex(p: Path) -> tuple[dict, str]:
    """config.json plus why it could not be read.

    A half-downloaded or hand-edited config.json used to fall through to an empty
    dict, after which the config looks like "no architecture" and the user was told
    their *architecture* is unsupported.  The real cause (unparseable file) has to
    survive into the verdict.
    """
    cfg_path = p / "config.json"
    if not cfg_path.is_file():
        return {}, t('imp.cfg_missing')
    try:
        with open(cfg_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {}, t('imp.cfg_unparseable', err=e)
    if not isinstance(data, dict):
        return {}, t('imp.cfg_not_object', kind=type(data).__name__)
    return data, ""


def _quant_label(type_id: int) -> str:
    key = GGUF_QUANT_KEYS.get(type_id)
    return t(key) if key else t('imp.qn.unknown', id=type_id)


def _scan_ninfer(p: Path) -> ModelScan:
    """ninfer's own container: try to read the identity, never fatal if unreadable."""
    s = ModelScan(kind="ninfer", path=str(p), friendly=t('imp.f_ninfer'))
    try:
        # Stream only the first 1MB (the container directory JSON sits up front).
        # read_bytes() is forbidden here: a ten-plus GB artifact would blow up RAM.
        with open(p, "rb") as f:
            head = f.read(1 << 20)
        m = re.search(rb'"model_id"\s*:\s*"([^"]+)"', head)
        w = re.search(rb'"weights_id"\s*:\s*"([^"]+)"', head)
        if m:
            s.model_id = m.group(1).decode("utf-8", "replace")
        if w:
            s.weights_id = w.group(1).decode("utf-8", "replace")
        s.weight_bytes = p.stat().st_size
        s.params_billions = round(s.weight_bytes / (1 << 30) * 1.0, 2)  # rough (KV tables included)
        s.quant_name = t('imp.q_ninfer')
    except OSError:
        s.issues.append(t('imp.err_read_failed'))
    return s


def _scan_lora(p: Path) -> ModelScan:
    """LoRA/PEFT adapter folder: cannot run on its own, needs a base plus an offline merge."""
    s = ModelScan(kind="lora", path=str(p), friendly=t('imp.f_lora'))
    try:
        with open(p / "adapter_config.json", encoding="utf-8") as f:
            ac = json.load(f)
        s.quant_name = t('imp.lora_rank', r=ac.get("r", "?"))
        base = ac.get("base_model_name_or_path") or ""
        peft = ac.get("peft_type", "?")
        s.arch_note = (t('imp.lora_base', peft=peft, base=base) if base
                       else t('imp.lora_base_missing', peft=peft))
        total = sum(f.stat().st_size for f in p.glob("*.safetensors"))
        s.weight_bytes = total
        s.issues.append(t('imp.issue_lora_needs_base'))
    except OSError as e:
        s.issues.append(t('imp.err_lora_cfg', err=e))
    return s


def _scan_hf(p: Path) -> ModelScan:
    """HuggingFace layout folder (vLLM eats the same shape): config.json + safetensors."""
    s = ModelScan(kind="hf", path=str(p), friendly=t('imp.f_hf'))
    cfg, cfg_err = _read_hf_config_ex(p)
    s.config_error = cfg_err
    if cfg_err:
        s.issues.append(cfg_err)
    archs = cfg.get("architectures") or []
    mt = cfg.get("model_type", "")
    s.arch_names = [str(mt)] + [str(a) for a in archs]
    # Multimodal models keep the text geometry under text_config
    tc = cfg.get("text_config") or {}
    s.hidden_size = int(tc.get("hidden_size") or cfg.get("hidden_size") or 0)
    s.n_layers = int(tc.get("num_hidden_layers") or cfg.get("num_hidden_layers") or 0)
    s.arch_note = (t('imp.arch_known', archs=", ".join(archs) or "?", mt=mt) if mt
                   else t('imp.arch_unknown'))
    s.model_id = _family_from_arch(archs, mt, s.hidden_size, s.n_layers)
    if s.model_id == "qwen3-candidate":
        s.arch_note += t('imp.arch_qwen_candidate')
    elif not s.model_id and not cfg_err:
        s.arch_note += t('imp.arch_not_supported')
    if not cfg_err:
        s.arch_note += _shape_facts(cfg)
    # Parameter count / weight size straight from the file sizes (bf16 ~ 2 bytes/param)
    total = 0
    try:
        for f in p.glob("*.safetensors"):
            total += f.stat().st_size
    except OSError:
        pass
    s.weight_bytes = total
    s.params_billions = round(total / 2 / 1e9, 2) if total else 0.0
    s.quant_name = _hf_quant_hint(cfg)
    return s


def _shape_facts(cfg: dict) -> str:
    """The concrete architecture facts a refusal must name to be actionable.

    "not on the supported list" tells a user nothing about *what* is missing; the
    layer mix, the extra towers and the per-layer scalars do.  Everything here is read
    from the user's own config.json -- nothing is inferred about the engine.
    """
    tc = cfg.get("text_config") or cfg
    facts = []
    layers = tc.get("layer_types")
    if isinstance(layers, list) and layers:
        hist = {}
        for item in layers:
            hist[str(item)] = hist.get(str(item), 0) + 1
        facts.append(t('imp.fact_layer_types',
                       mix=", ".join("%s x%d" % (k, v) for k, v in sorted(hist.items()))))
    rope = tc.get("rope_parameters")
    if isinstance(rope, dict):
        parts = []
        for name, row in rope.items():
            if isinstance(row, dict):
                parts.append("%s:rope_type=%s,partial_rotary_factor=%s" % (
                    name, row.get("rope_type"), row.get("partial_rotary_factor")))
        if parts:
            facts.append(t('imp.fact_rope', rope="; ".join(parts)))
    towers = [key for key in ("vision_config", "audio_config") if cfg.get(key)]
    if towers:
        facts.append(t('imp.fact_towers', towers=", ".join(towers)))
    quant = cfg.get("quantization_config")
    if isinstance(quant, dict):
        facts.append(t('imp.fact_quant',
                       method=quant.get("quant_method") or quant.get("quant_algo") or "?"))
    return "".join(facts)


def find_spec_files(model_type: str, archs=()) -> list:
    """archkit specs that already describe this model type (bounded directory scan).

    The repository already holds e.g. specs/gemma4-31b_spec.json; naming it turns
    "not supported" into "the access spec exists, run the adapt pipeline on it".
    """
    want = {str(model_type).lower()} | {str(a).lower() for a in archs}
    want.discard("")
    if not want:
        return []
    hits = []
    try:
        for path in sorted(ARCHKIT_OUT.parent.glob("specs/*.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    spec = json.load(f)
            except Exception:
                continue
            hf = spec.get("hf") if isinstance(spec.get("hf"), dict) else {}
            names = {str(hf.get("model_type", "")).lower(),
                     str(spec.get("model_id", "")).lower()}
            names |= {str(a).lower() for a in (hf.get("architectures") or [])}
            names.discard("")
            if names & want or any(w and w in n for n in names for w in want):
                hits.append(path)
    except OSError:
        return []
    return hits


def _family_from_arch(archs: list, model_type: str, hidden: int, layers: int) -> str:
    """HF architecture -> support verdict.

    Returns:
      'qwen3.8-27b'      full match with the supported spec (Qwen3 + 5120 hidden + 64 layers)
      'qwen3-candidate'  Qwen3 family, but the geometry is not the supported spec
      ''                 not in the family at all
    """
    joined = " ".join([*archs, model_type]).lower().replace("_", "")
    if "qwen3" not in joined and "qwen2.5" not in joined:
        return ""
    if (hidden == ENGINE_HIDDEN and layers == ENGINE_LAYERS) or hidden == 0:
        # Geometry matches, or the config carries none (the name looks right) -> offer it
        # as a candidate; the converter does the final check.
        return "qwen3.8-27b" if "qwen3" in joined else "qwen3-candidate"
    return "qwen3-candidate"


def _hf_quant_hint(cfg: dict) -> str:
    """Guess the quantisation tier from config traces; only used for the wording."""
    qc = str(cfg.get("quantization_config") or {})
    if "nvfp4" in qc.lower() or "fp4" in qc.lower():
        return t('imp.q_nvfp4_pre')
    if "bitsandbytes" in qc.lower():
        return t('imp.q_bitsandbytes')
    return t('imp.q_bf16')


def _scan_gguf(p: Path) -> ModelScan:
    """llama.cpp GGUF: parse the header plus the tensor table, judge quant tier and family."""
    s = ModelScan(kind="gguf", path=str(p), friendly=t('imp.f_gguf'))
    try:
        with open(p, "rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                s.issues.append(t('imp.err_gguf_header'))
                s.unreadable = True
                return s
            raw = f.read(struct.calcsize(GGUF_HEADER))
            if len(raw) != struct.calcsize(GGUF_HEADER):
                raise struct.error(t('imp.err_gguf_short_header',
                                     got=4 + len(raw), need=GGUF_HEADER_BYTES))
            version, n_tensors, n_kv = struct.unpack(GGUF_HEADER, raw)
            if n_tensors > GGUF_MAX_ENTRIES or n_kv > GGUF_MAX_ENTRIES:
                raise struct.error(t('imp.err_gguf_implausible',
                                     tensors=n_tensors, kv=n_kv))
            s.friendly = t('imp.f_gguf_versioned', ver=version, n=n_tensors)
            # Skip the metadata KV block: each entry = key(str) + type(u32) + value.
            # The declared architecture is kept -- the file name is only a fallback,
            # so a renamed download is still recognised (and a wrongly named one is
            # not silently accepted).
            declared = ""
            layer_kv = {}                       # <arch>.block_count / .nextn_predict_layers
            for _ in range(n_kv):
                key = _read_gguf_str(f)
                ktype = struct.unpack("<I", f.read(4))[0]
                if ktype == 8 and key in GGUF_ARCH_KEYS and not declared:
                    declared = _read_gguf_str(f)
                elif key.endswith(GGUF_LAYER_KV_SUFFIXES) and ktype in GGUF_INT_KV_FORMATS:
                    # Kept, not skipped: the tensor table alone cannot tell a draft block
                    # from a decoder layer (blk.32 of the real Ornith carries BOTH), so
                    # these two numbers are the only evidence for the decoder depth.
                    fmt = GGUF_INT_KV_FORMATS[ktype]
                    layer_kv[key] = struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
                else:
                    _skip_gguf_value(f, ktype)
            # Tensor table: name(str) + shape(u32 count + i64 per dim) + type(u32) + offset(i64)
            unmapped = []
            embed_shape = ()
            max_layer = -1
            nextn_blocks = set()                # blocks that carry `.nextn.` tensors
            for _ in range(n_tensors):
                name = _read_gguf_str(f)           # tensor name
                ndims = struct.unpack("<I", f.read(4))[0]
                if ndims > 8:
                    raise struct.error(t('imp.err_gguf_bad_ndims', name=name, n=ndims))
                shape = struct.unpack("<%dq" % ndims, f.read(8 * ndims))
                ttype = struct.unpack("<I", f.read(4))[0]
                f.read(8)                          # offset
                s.gguf_types[ttype] = s.gguf_types.get(ttype, 0) + 1
                if not _gguf_name_is_convertible(name, declared):
                    unmapped.append(name)
                if name == "token_embd.weight" and len(shape) == 2:
                    embed_shape = shape
                if name.startswith("blk.") and "." in name[4:]:
                    head, rest = name[4:].split(".", 1)
                    if head.isdigit():
                        max_layer = max(max_layer, int(head))
                        if rest.startswith("nextn."):
                            nextn_blocks.add(int(head))
            # Rough parameter count: the main weights dominate the file, so file size is safer
            s.weight_bytes = p.stat().st_size
            s.params_billions = round(s.weight_bytes / 1e9, 2)  # rough (quantisation compressed)
            names = [_quant_label(tid) for tid in sorted(s.gguf_types)]
            s.quant_name = "/".join(names) if names else _quant_label(-1)
            # Only a type the reader cannot decode is a problem.  Testing "is it
            # K-quant" was equivalent while no K-quant had a decoder; now that Q4_K
            # and Q6_K do, that test reports a problem where there is none.
            if any(tid not in GGUF_COPYABLE_TYPE_IDS for tid in s.gguf_types):
                if any(tid in KQUANT_TYPE_IDS for tid in s.gguf_types):
                    s.issues.append(t('imp.issue_kquant'))
                else:
                    s.issues.append(t('imp.issue_quant_other'))
            # Architecture: prefer the GGUF's own general.architecture, fall back to the
            # file name -- and require the *geometry* to match before promising the
            # supported spec (a GGUF carries no geometry KV, so the tensor table is the
            # only source of truth: token_embd gives the width, blk.N gives the depth).
            s.hidden_size = int(embed_shape[0]) if embed_shape else 0
            s.n_layers, layer_issue = _gguf_main_layers(layer_kv, declared, max_layer,
                                                        nextn_blocks)
            if layer_issue:
                s.issues.append(layer_issue)
            s.model_id = _family_from_gguf(declared, p.name, s.hidden_size, s.n_layers)
            s.arch_note = (declared or t('imp.arch_unknown_from_name'))
            s.arch_names = [declared, p.stem]
            if s.hidden_size:
                s.arch_note += t('imp.arch_gguf_geometry',
                                 hidden=s.hidden_size, layers=s.n_layers)
            if unmapped:
                # Not gated on the family: a tensor with no name mapping is exactly the
                # evidence of the missing piece, whoever the model belongs to.
                s.unmapped = unmapped[:6]
                s.issues.append(t('imp.issue_gguf_unmapped', n=len(unmapped),
                                  names=", ".join(unmapped[:3])))
    except (OSError, struct.error) as e:
        s.issues.append(t('imp.err_gguf_parse', err=e))
        s.unreadable = True
    return s


def _gguf_main_layers(layer_kv: dict, arch: str, max_layer: int,
                      nextn_blocks: set) -> tuple:
    """(decoder depth, an issue string when the two pieces of evidence disagree).

    `max(blk.N)+1` is the number of BLOCKS, and a GGUF with a draft/MTP block has one
    block more than it has decoder layers.  Measured on the real
    ``ornith-1.5-9b/Ornith-1.5-9B-Q4_K_M.gguf``: 442 tensors, 33 blocks
    (``blk.0`` .. ``blk.32``), 33 = ``qwen35.block_count``, and
    ``qwen35.nextn_predict_layers = 1`` (both ktype 4 = uint32).  The decoder stack is
    therefore **32** layers, of which 8 are full attention, and the wizard reporting 33
    disagreed with the spec the same wizard tells the user to feed to
    ``tools/archkit/adapt_all.py`` (``geometry.layers = 32`` + ``mtp = 1``, built by
    ``tools/archkit/gguf_spec.py`` from the same file).

    `blk.32` does NOT give itself away by holding only ``nextn`` tensors: it carries a
    full attn/ffn set (``attn_q/k/v/output``, ``qk_norm``, ``ffn_*``, ``attn_norm``)
    *next to* ``blk.32.nextn.{eh_proj,enorm,hnorm,shared_head_norm}`` -- so "count the
    blocks, drop the ones that only hold nextn weights" is wrong here too, and only the
    metadata names the draft block.

    The accounting rule is not respelled here: ``tools/convert/gguf_names.layer_split``
    owns it, and ``tools/archkit/gguf_spec._split_nextn`` imports the same function.
    Two spellings of "how many layers does this file have" is exactly the writer/reader
    split this tree has paid for before, and a wizard number that contradicts the
    spec-generator number for the same file is that split in its user-visible form.

    The block count stays the fallback (a GGUF with no such metadata), and the two
    pieces of evidence are COMPARED instead of averaged: a file that carries
    ``.nextn.`` tensors its metadata does not declare, or declares a draft block whose
    tensors are not where they must be, gets a named issue -- silently picking either
    number is how the depth goes wrong by one with nothing said.
    """
    blocks = max_layer + 1 if max_layer >= 0 else 0
    main, declared_nextn = 0, 0
    try:
        import sys as _sys
        from pathlib import Path as _Path
        root = str(_Path(__file__).resolve().parents[2])
        if root not in _sys.path:
            _sys.path.insert(0, root)
        from tools.convert.gguf_names import layer_split
    except Exception:                       # noqa: BLE001 - fall back, never crash
        layer_split = None
    if layer_split is not None:
        try:
            main, declared_nextn = layer_split(layer_kv, arch or "")
        except (ValueError, KeyError):
            main, declared_nextn = 0, 0
    observed = len(nextn_blocks)
    issue = ""
    if declared_nextn or observed:
        if declared_nextn != observed:
            issue = t('imp.issue_gguf_layers',
                      declared=declared_nextn, observed=observed, blocks=blocks,
                      index=", ".join("blk.%d" % i for i in sorted(nextn_blocks)) or "-")
    return (main or blocks), issue


def _gguf_name_is_convertible(name: str, arch: str = "") -> bool:
    """Can any converter in this tree turn this gguf tensor name into a HF name?

    The rule-based map (tools/convert/gguf_names.RULES) is the authority when the
    file declares an architecture it knows; the qwen-shaped literal is only the
    fallback for an architecture the rule map has no entry for.  Knowing only the
    literal is what made the wizard report "271 tensors have no name mapping" for a
    file the converter maps 442/442.
    """
    for pattern in _RulePatterns.for_arch(arch):
        if pattern.fullmatch(name):
            return True
    if name in GGUF_MAPPED_NAMES:
        return True
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    return bool(m and m.group(2) in GGUF_MAPPED_NAMES)


def _family_from_gguf_name(name: str) -> str:
    low = name.lower().replace("_", "")
    if "qwen3" in low or "qwen2.5" in low:
        # No geometry inside a GGUF, so only a candidate is possible; the converter
        # pre-check has the final say.
        return "qwen3.8-27b" if "qwen3" in low else "qwen3-candidate"
    return ""


def _family_from_gguf(declared: str, fname: str, hidden: int, layers: int) -> str:
    """GGUF family verdict from the declared architecture *and* the real geometry.

    The name alone is not evidence: a renamed download must still be recognised, and a
    correctly named file with the wrong shape must not be promised a conversion.  The
    tensor table gives hidden/layers, so the same geometry check the HF path uses
    (ENGINE_HIDDEN/ENGINE_LAYERS) applies here too.
    """
    if not _family_from_gguf_name(" ".join([declared or "", fname])):
        return ""
    if hidden == ENGINE_HIDDEN and layers == ENGINE_LAYERS:
        return "qwen3.8-27b"
    return "qwen3-candidate"


def _unsupported_arch_tips(scan: ModelScan) -> list:
    """What is actually missing, and the one command that starts closing the gap."""
    tips = []
    if scan.unmapped:
        tips.append(t('imp.tip_gguf_unmapped', names=", ".join(scan.unmapped[:3])))
    specs = find_spec_files("", scan.arch_names)
    if specs:
        tips.append(t('imp.tip_spec_exists',
                      specs=", ".join(p.name for p in specs)))
    tips.append(t('imp.tip_adapt_all', path=scan.path,
                  tools="tools/archkit/adapt_all.py"))
    tips.append(t('imp.tip_hf_search', url=LINKS["huggingface"]))
    return tips


def _read_gguf_str(f) -> str:
    """GGUF string: u64 byte length + bytes, decoded. Bounded (see GGUF_MAX_STRING).

    The bound is the difference between "this file is misaligned/corrupt, here is the
    number that gave it away" and an empty MemoryError.
    """
    n = struct.unpack("<Q", f.read(8))[0]
    if n > GGUF_MAX_STRING:
        raise struct.error(t('imp.err_gguf_string_too_long', n=n, cap=GGUF_MAX_STRING))
    return f.read(n).decode("utf-8", "replace")


def _skip_str(f):
    """GGUF string: u64 length + bytes."""
    _read_gguf_str(f)


def _skip_gguf_value(f, ktype: int, depth: int = 0):
    """Skip one GGUF metadata value. Type table from the gguf spec, 0..13.
    Depth and count are capped so a hostile or corrupt file cannot hang the scan."""
    if depth > 4:
        raise struct.error("metadata nesting too deep")
    if ktype == 0:          # uint8
        f.read(1)
    elif ktype == 1:        # int8
        f.read(1)
    elif ktype == 2:        # uint16
        f.read(2)
    elif ktype == 3:        # int16
        f.read(2)
    elif ktype == 4:        # uint32
        f.read(4)
    elif ktype == 5:        # int32
        f.read(4)
    elif ktype == 6:        # float32
        f.read(4)
    elif ktype == 7:        # bool
        f.read(1)
    elif ktype == 8:        # string
        _skip_str(f)
    elif ktype == 9:        # array
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        if count > (1 << 20):
            raise struct.error("metadata array too large")
        for _ in range(count):
            _skip_gguf_value(f, elem_type, depth + 1)
    elif ktype == 10:       # uint64
        f.read(8)
    elif ktype == 11:       # int64
        f.read(8)
    elif ktype == 12:       # float64
        f.read(8)
    else:
        raise struct.error("unknown metadata type %d" % ktype)


# ---------------------------------------------------------------- judgement

def gpu_report() -> GpuInfo:
    """Probe the GPU through nvidia-smi. present=False without an NVIDIA card/driver."""
    info = GpuInfo()
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8)
        if out.returncode != 0 or not out.stdout.strip():
            return info
        parts = [x.strip() for x in out.stdout.splitlines()[0].split(",")]
        info.present = True
        info.name = parts[0]
        info.vram_total_gb = round(float(parts[1]) / 1024, 1)
        info.vram_free_gb = round(float(parts[2]) / 1024, 1)
        info.driver = parts[3] if len(parts) > 3 else ""
    except Exception:
        pass
    return info


def vram_need_bytes(scan: ModelScan, ctx_tokens: int = 8192) -> int:
    """Rough lower bound for the VRAM this model needs.

    Components: resident weights (~0.55 bytes/param after conversion) + KV cache +
    scratch. The engine arena decides the exact number; this is the "will it fit"
    test for a newcomer.
    """
    if scan.kind == "ninfer":
        # ninfer weights are resident as they are; KV and scratch roughly 0.4MB/token
        return int(scan.weight_bytes + ctx_tokens * 400_000)
    # bf16 source weights convert to ~0.55x (NVFP4 is about 4.2 bits/param + scales)
    est_weights = int(scan.weight_bytes * 0.55)
    return int(est_weights + ctx_tokens * 400_000)


def verdict(scan: ModelScan, gpu: Optional[GpuInfo] = None,
            ctx_tokens: int = 8192) -> Verdict:
    """Turn a scan into advice a newcomer can act on."""
    gpu = gpu or gpu_report()
    v = Verdict()
    if scan.kind == "ninfer":
        v.action, v.level = "run", "ok"
        v.title = t('imp.v_ninfer_run_title')
        v.detail = t('imp.v_ninfer_run_detail', quant=scan.quant_name or t('imp.q_ninfer'))
        if scan.model_id and scan.model_id not in SUPPORTED_MODEL_IDS:
            v.action, v.level = "unsupported_arch", "bad"
            v.title = t('imp.v_ninfer_bad_family_title')
            v.detail = t('imp.v_ninfer_bad_family_detail', family=scan.model_id)
        if gpu.present:
            need = vram_need_bytes(scan, ctx_tokens)
            if need > gpu.vram_total_gb * (1 << 30):
                v.action, v.level = "vram_short", "bad"
                v.title = t('imp.v_vram_title')
                v.detail = t('imp.v_vram_detail', gb=scan.weight_bytes / 1e9, gpu=gpu.name,
                             vram=gpu.vram_total_gb, ctx=ctx_tokens)
                v.tips = [t('imp.tip_ctx_smaller'), t('imp.tip_smaller_model'),
                          t('imp.tip_bigger_gpu')]
        else:
            v.level = "warn"
            v.tips.append(t('imp.tip_no_nvidia'))
            v.tips.append(t('imp.tip_install_driver', url=LINKS["nvidia_driver"]))
        return v
    if scan.kind == "hf":
        if scan.unreadable and not scan.config_error:
            # config.json present, weights absent -- name the missing half.
            v.action, v.level = "unreadable", "bad"
            v.title = t('imp.v_hf_incomplete_title')
            v.detail = t('imp.v_hf_incomplete_detail', path=scan.path,
                         err="; ".join(scan.issues) or "?")
            v.tips = [t('imp.tip_redownload'), t('imp.tip_hf_search', url=LINKS["huggingface"])]
            return v
        if scan.config_error:
            # The file is unreadable, so nothing downstream can be judged.  Saying
            # "your architecture is not supported" here is a misdiagnosis.
            v.action, v.level = "unreadable", "bad"
            v.title = t('imp.v_cfg_broken_title')
            v.detail = t('imp.v_cfg_broken_detail', path=scan.path, err=scan.config_error)
            v.tips = [t('imp.tip_redownload_config'), t('imp.tip_redownload')]
            return v
        if scan.model_id == "qwen3-candidate":
            v.action, v.level = "unsupported_arch", "bad"
            v.title = t('imp.v_hf_candidate_title')
            v.detail = t('imp.v_hf_candidate_detail', layers=scan.n_layers,
                         hidden=scan.hidden_size)
            v.tips.append(t('imp.tip_hf_download_27b', url=LINKS["huggingface"]))
            return v
        if scan.model_id not in SUPPORTED_MODEL_IDS:
            v.action, v.level = "unsupported_arch", "bad"
            v.title = t('imp.v_hf_unsupported_title')
            v.detail = t('imp.v_hf_unsupported_detail2',
                         facts=scan.arch_note, families=", ".join(ENGINE_DECODER_FAMILIES),
                         targets=", ".join(REGISTERED_TARGET_NAMES))
            v.tips.extend(_unsupported_arch_tips(scan))
            return v
        v.action = "convert_groupwise"
        v.level = "ok"
        v.title = t('imp.v_hf_convert_title')
        v.detail = t('imp.v_hf_convert_detail', family=scan.model_id)
        need = vram_need_bytes(scan, ctx_tokens)
        if gpu.present and need > gpu.vram_total_gb * (1 << 30) * 0.92:
            v.action, v.level = "vram_short", "bad"
            v.title = t('imp.v_convert_vram_title')
            v.detail = t('imp.v_convert_vram_detail', quant=scan.quant_name,
                         params=scan.params_billions * 10, need=need / 1e9,
                         vram=gpu.vram_total_gb)
            v.tips = [t('imp.tip_vram_scales'), t('imp.tip_qwen_small'),
                      t('imp.tip_driver_update', url=LINKS["nvidia_driver"])]
            return v
        if scan.quant_name != t('imp.q_nvfp4_pre'):
            v.tips.append(t('imp.tip_prefer_nvfp4'))
        return v
    if scan.kind == "gguf":
        if scan.unreadable:
            # Nothing could be read, so no statement about the family is possible.
            # Saying "this family is not supported" for a truncated download sends the
            # user looking for another model instead of finishing the download.
            v.action, v.level = "unreadable", "bad"
            v.title = t('imp.v_gguf_broken_title')
            v.detail = t('imp.v_gguf_broken_detail', path=scan.path,
                         err="; ".join(scan.issues) or "?")
            v.tips = [t('imp.tip_redownload'), t('imp.tip_gguf_f16_only')]
            return v
        if any(tid not in GGUF_COPYABLE_TYPE_IDS for tid in scan.gguf_types):
            bad = [tid for tid in sorted(scan.gguf_types) if tid not in GGUF_COPYABLE_TYPE_IDS]
            v.action, v.level = "unsupported_quant", "bad"
            v.title = t('imp.v_gguf_kquant_title')
            named = ", ".join(_quant_label(tid) for tid in bad)
            v.detail = t('imp.v_gguf_kquant_detail2', quant=named,
                         support=gguf_reader_support_names())
            v.tips.extend(_unsupported_arch_tips(scan))
            return v
        if scan.model_id == "qwen3-candidate" or scan.model_id not in SUPPORTED_MODEL_IDS:
            v.action, v.level = "unsupported_arch", "bad"
            v.title = t('imp.v_gguf_unsupported_title')
            v.detail = t('imp.v_gguf_unsupported_detail2',
                         arch=scan.arch_note or t('imp.arch_unknown'),
                         families=", ".join(ENGINE_DECODER_FAMILIES),
                         targets=", ".join(REGISTERED_TARGET_NAMES))
            v.tips.extend(_unsupported_arch_tips(scan))
            return v
        v.action = "convert_groupwise"
        v.level = "ok"
        v.title = t('imp.v_gguf_convert_title')
        v.detail = t('imp.v_gguf_convert_detail')
        v.tips.append(t('imp.tip_gguf_f16_only'))
        return v
    if scan.kind == "lora":
        v.action, v.level = "need_base_fuse", "warn"
        v.title = t('imp.v_lora_title')
        v.detail = t('imp.v_lora_detail', rank=scan.quant_name or "?")
        v.tips.append(t('imp.tip_lora_tools'))
        return v
    v.action = "unknown"
    v.level = "bad"
    v.title = t('imp.v_unknown_title')
    v.detail = t('imp.v_unknown_detail')
    return v


def env_report() -> dict:
    """Environment self-check: GPU/driver/Python/engine artifact. For newcomers and
    for the packaged build's first-run check."""
    gpu = gpu_report()
    items = []
    if gpu.present:
        items.append({"name": t('imp.env.gpu'), "ok": True,
                      "detail": t('imp.env.gpu_ok', name=gpu.name,
                                  vram=gpu.vram_total_gb, driver=gpu.driver)})
    else:
        items.append({"name": t('imp.env.gpu'), "ok": False,
                      "detail": t('imp.env.gpu_missing'),
                      "fix": t('imp.env.gpu_fix', url=LINKS["nvidia_driver"])})
    items.append({"name": t('imp.env.python'), "ok": True,
                  "detail": sys.version.split()[0]})
    # Engine artifact: prefer the Windows exe (packaged build), else WSL ninfer-serve (dev)
    engine = None
    for cand in (Path(sys.executable).parent / "ninfer-serve.exe",
                 Path("ninfer-serve.exe")):
        if cand.exists():
            engine = str(cand)
            break
    if engine is None:
        try:
            r = subprocess.run(["wsl.exe", "-d", "Ubuntu", "bash", "-lc",
                                "ls /home/user/ninfer-fusion/build/apps/ninfer-serve 2>/dev/null"],
                               capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=20)
            if r.returncode == 0 and r.stdout.strip():
                engine = "wsl:" + r.stdout.strip()
        except Exception:
            pass
    items.append({"name": t('imp.env.engine'), "ok": engine is not None,
                  "detail": engine or t('imp.env.engine_missing'),
                  "fix": None if engine else t('imp.env.engine_fix')})
    return {"items": items}


# ------------------------------------------- import gap report (archkit manifest)
#
# tools/archkit/adapt.py (v4 auto-adaptation) writes, per model:
#     tools/archkit/out/<model-id>/manifest.json   gaps: [{need, tier, action}]
#     tools/archkit/out/<model-id>/config.h        unless the header was refused
# A `new_op` gap means the header, where one is written, would claim coverage the
# engine does not have. adapt.py reports that gap in the manifest; it does NOT
# withhold the file, and no writer in this tree emits the name `config.h.BLOCKED`
# (the refusal path leaves no header behind at all and records the reason in the
# manifest instead). Which header file the report names is read off the directory --
# see header_file_state() -- because the wizard used to print that phantom name for a
# model whose config.h was sitting right there. The wizard shows a verdict, the
# blocking gaps, and the concrete way out.

ARCHKIT_OUT = Path(__file__).resolve().parents[2] / "tools" / "archkit" / "out"

#: Manifest file names to look for, in order.  See find_manifest().
#: adapt.py and (since S55) gen_target.py both write `manifest.json`; the second
#: name is kept because three pre-S55 gen_target artifacts in this tree use it.
MANIFEST_NAMES = ("manifest.json", "arch_manifest.json")

TIER_ORDER = ("hook", "new_op", "covered", "post")
TIER_KEYS = {"hook": "imp.tier.hook", "new_op": "imp.tier.new_op",
             "covered": "imp.tier.covered", "post": "imp.tier.post"}

# Action rendering, keyed by the leading `need` prefix adapt.py emits (its action
# strings are written in Chinese). Item: (regex over `need`, i18n key, group
# names). A key may also be a {tier: key, 'default': key} dict when the same need
# shape is rendered differently per tier.
_GAP_ACTION_ROWS = (
    (re.compile(r"^attention:gqa_full$"), "imp.gap.attention_gqa_full", ()),
    (re.compile(r"^attention:linear\(gdn\)$"), "imp.gap.attention_gdn", ()),
    (re.compile(r"^attention:sliding_window\((\d+)\)$"),
     "imp.gap.attention_swa", ("window",)),
    (re.compile(r"^attn:qk_norm$"), "imp.gap.attn_qk_norm", ()),
    (re.compile(r"^attn:headwise_output_gate\(([^)]*)\)$"),
     "imp.gap.attn_headwise_gate", ("mode",)),
    (re.compile(r"^attn:hybrid_global_hd=(\d+)\(kv=([^)]*)\)\s*vs\s*local hd=(\S+)$"),
     "imp.gap.attn_hybrid_hd", ("ghd", "gkv", "lhd")),
    (re.compile(r"^attn_scale:qk_scale_factor=(.+)$"), "imp.gap.attn_scale", ("scale",)),
    (re.compile(r"^mlp:act=(\w+)$"),
     {"default": "imp.gap.mlp_act", "new_op": "imp.gap.mlp_act_new_op"}, ("act",)),
    # adapt.py reclassified the tied head from new_op to covered once the converter
    # grew materialisation (S53): the engine still loads an independent lm_head object,
    # so "no new operator" is true, but the conversion-time cost must stay visible.
    (re.compile(r"^head:tied=true$"),
     {"default": "imp.gap.head_tied_true", "covered": "imp.gap.head_tied_true_covered"},
     ("mb",)),
    (re.compile(r"^head:tied=false$"), "imp.gap.head_tied_false", ()),
    (re.compile(r"^head:logit_softcap=(.+)$"), "imp.gap.head_logit_softcap", ("cap",)),
    (re.compile(r"^token_domain:vocab=(\d+)!=family (\d+)$"),
     "imp.gap.token_domain", ("vocab", "family")),
    (re.compile(r"^layers:(\d+)>family cap (\d+)$"),
     "imp.gap.layers_over_cap", ("layers", "cap")),
    (re.compile(r"^layers:(\d+)>(\d+)$"), "imp.gap.layers_audit", ("layers", "cap")),
    (re.compile(r"^layer_scale:output_multiplier=(.+)$"), "imp.gap.layer_scale", ("mult",)),
    (re.compile(r"^rope:per_layer_theta\(n=(\d+), zero=(\d+)\)$"),
     "imp.gap.rope_per_layer_theta", ("n", "zero")),
    (re.compile(r"^rope:per_type_theta\{(.+)\}$"),
     "imp.gap.rope_per_type_theta", ("table",)),
    (re.compile(r"^rope:partial_rotary\{(.+)\}$"), "imp.gap.rope_partial_kind", ("table",)),
    (re.compile(r"^rope:partial_rotary=(.+)$"), "imp.gap.rope_partial_scalar", ("factor",)),
    (re.compile(r"^new_op:layer_kinds\((.+)\)$"), "imp.gap.layer_kinds", ("kinds",)),
    (re.compile(r"^new_op:state_space\((.+)\)$"), "imp.gap.state_space", ("keys",)),
    (re.compile(r"^moe:experts=([^,]+),top=(.+)$"), "imp.gap.moe", ("experts", "top")),
    (re.compile(r"^vision$"), "imp.gap.vision", ()),
    (re.compile(r"^quant_geometry$"), "imp.gap.quant_geometry", ()),
)

# adapt.py quotes the tied-head cost inline: "... lm_head = embed^T (约 671 MB) ...".
_TIE_MB_RE = re.compile(r"约\s*([\d.]+)\s*MB")


def find_manifest(model_id: str) -> Optional[Path]:
    """tools/archkit/out/<model-id>/manifest.json, tolerant of -/_ spellings.

    Two producers write into that directory with different file names: adapt.py
    (tools/archkit/adapt.py:210) writes `manifest.json` and carries the `gaps` list
    this module renders, while gen_target.py (tools/archkit/gen_target.py:163) writes
    `arch_manifest.json` with no gap data.  Only looking for the first name made the
    wizard silently report "no gaps" (or nothing at all) for every model produced by
    the gen_target path -- including gemma4-31b, whose config.h and arch_manifest.json
    are both present in the tree today.
    """
    if not model_id:
        return None
    candidates, seen = [], set()
    for cand in (model_id, model_id.replace("-", "_"), model_id.replace("_", "-")):
        if cand.lower() in seen:
            continue
        seen.add(cand.lower())
        candidates.append(cand)
    for cand in candidates:
        for name in MANIFEST_NAMES:
            p = ARCHKIT_OUT / cand / name
            if p.is_file():
                return p
    return None


def manifest_gap_state(manifest: dict) -> tuple[bool, list, str]:
    """(measured, gaps, shape) for one parsed manifest.

    "Measured" means exactly: the manifest carries a `gaps` LIST of OBJECTS, and no
    `gaps_measured` flag denies it. An empty list is a measurement ("nothing is
    missing"); everything else is the ABSENCE of one, and `shape` names what was
    actually found so the report never says "there is no gaps field" about a file
    where the field is present and null.

    shape:
      'missing'       no `gaps` key at all        (gen_target.py before S55)
      'null'          `"gaps": null`             (gen_target.py since S55)
      'not_a_list:X'  a scalar/object/dict of type X
      'flag_false:K'  `"gaps_measured": false` -- the writer denies its own field, so
                      the gaps list (if any) is not a measurement; K names what the
                      `gaps` field actually held ('missing'/'null'/a JSON type name),
                      because "it carries a gaps list but says false" is wrong about a
                      manifest whose `gaps` is null, and a label that is wrong about
                      the file is the same defect as a verdict that is.
      'alias_key:measured'
                      the manifest carries a top-level `measured` key. Nothing in
                      this tree reads that spelling; the flag is `gaps_measured`.
                      A file that carries an unhonoured flag gets no benefit of the
                      doubt: `{"gaps": [], "measured": false}` -- a writer saying it
                      did NOT measure -- used to reach CLEAR, which is the false
                      all-clear this whole function exists to prevent. The shape
                      names the key so the message can say which spelling to use.
                      It is checked BEFORE `gaps_measured`, so a manifest carrying
                      both is not silently waved through on the strength of the
                      spelling that happens to be honoured.
      'bad_element:T@i'
                      `gaps` IS a list, but element i is a T, not an object. This
                      used to be an AttributeError raised from the row loop
                      (`g.get("need")` on a string), i.e. a Python traceback out of
                      main() and rc 1 -- the same code that means "no manifest".
                      A list whose elements cannot be read is not a measurement of
                      anything, so it lands here with the element and its index.
      'bad_tier:T@i'
                      `gaps[i]` IS an object, but its `tier` is not one of the four
                      tiers this tree's writer emits (`adapt.py` emits only
                      'hook' / 'new_op' / 'covered' / 'post'; measured over every
                      manifest on disk -- see test_model_import.py). T names the
                      tier as found, or the literal `missing` when the key is absent.
                      WHY THIS IS A SHAPE AND NOT A ROW: the verdict is decided by
                      `tier == "new_op"` alone (`blocking = [... if r["tier"] ==
                      "new_op"]`), so an element whose tier is absent or spelled
                      differently does not block -- and a list that cannot be read
                      element-wise is the same "the guard tests one shape" defect as
                      `bad_element`. Measured before this branch existed: a
                      hand-built `{"gaps": [{"need": "new_op:attention:linear(gdn)",
                      "action": "..."}]}` printed
                      `VERDICT: CLEAR -- config.h generated, no new_op gaps`,
                      a `gaps: 1 (hook 0 / new_op 0 / post 0)` summary and a row
                      naming `new_op:attention:linear(gdn)` all in one report, and
                      exited 0 -- with `--require-servable` too. A report that says
                      "no new_op gaps" while listing one is the false all-clear this
                      function exists to prevent, and `tier` is exactly the field
                      whose absence re-opens it.

    This is a function, not an inline expression, because the whole defect class
    here is "the guard tests one shape of the absence and the other shapes fall
    through to success". Every shape has to reach the same verdict, and the only
    way to show that is to feed all of them in -- see test_model_import.py.
    """
    if not isinstance(manifest, dict):
        return False, [], "not_a_list:%s" % type(manifest).__name__
    if "measured" in manifest:
        return False, [], "alias_key:measured"
    if manifest.get("gaps_measured") is False:
        return False, [], "flag_false:%s" % _gaps_field_kind(manifest)
    if "gaps" not in manifest:
        return False, [], "missing"
    raw = manifest.get("gaps")
    if raw is None:
        return False, [], "null"
    if not isinstance(raw, list):
        return False, [], "not_a_list:%s" % type(raw).__name__
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            return False, [], "bad_element:%s@%d" % (type(item).__name__, i)
        # A row only decides anything through its `tier` (see the docstring): a tier
        # this tree's writer cannot emit means this list is not a measurement we can
        # act on, so it joins the other shapes instead of rendering as CLEAR.
        tier = item.get("tier")
        if tier not in TIER_ORDER:
            return False, [], "bad_tier:%s@%d" % (
                "missing" if "tier" not in item else str(tier), i)
    return True, raw, ""


def _gaps_field_kind(manifest: dict) -> str:
    """What the `gaps` field actually holds, naming the absence rather than hiding it.

    Needed because `"gaps_measured": false` can accompany any of the shapes, and the
    report has to describe the file it actually read -- a `gaps: null` manifest that
    also denies the flag used to be described as "carries a gaps list but says false",
    which is false about the file (there is no list).
    """
    if "gaps" not in manifest:
        return "missing"
    raw = manifest.get("gaps")
    if raw is None:
        return "null"
    if isinstance(raw, list):
        return "list"
    return type(raw).__name__


def gap_shape_text(shape: str) -> str:
    """Translated description of what manifest_gap_state() found instead of gaps."""
    if shape == "missing":
        return t('imp.gaps.shape_missing')
    if shape == "null":
        return t('imp.gaps.shape_null')
    if shape == "alias_key:measured":
        # Says which key is not honoured AND which one is, because "the flag is
        # wrong" is only actionable together with the spelling that is read.
        return t('imp.gaps.shape_alias_key')
    if shape.startswith("flag_false"):
        kind = shape.split(":", 1)[1] if ":" in shape else "?"
        return t('imp.gaps.shape_flag_false', kind=kind)
    if shape.startswith("not_a_list:"):
        return t('imp.gaps.shape_not_a_list', kind=shape.split(":", 1)[1])
    if shape.startswith("bad_element:"):
        spec = shape.split(":", 1)[1]
        kind, _, index = spec.rpartition("@")
        return t('imp.gaps.shape_bad_element', kind=kind, index=index)
    if shape.startswith("bad_tier:"):
        # rpartition, not partition: the tier as found is arbitrary text from the
        # file, and `partition` would read its first '@' as the index separator.
        spec = shape.split(":", 1)[1]
        tier, _, index = spec.rpartition("@")
        return t('imp.gaps.shape_bad_tier', tier=tier, index=index)
    return shape


def gui_hint_claims(hint: dict) -> list[str]:
    """The claims a gen_target.py manifest makes about itself, as readable strings.

    gen_target.py used to write `gui_hint.supported: true` and NOTHING in the tree
    read it -- a claim about the engine with no check behind it and no consumer, so
    it could not be wrong out loud. It is read here (and asserted in
    test_model_import.py), and gen_target.py now writes `engine_supported` instead
    of an unsupported `supported`, with the reason spelled out.
    """
    if not isinstance(hint, dict):
        return []
    claims = []
    for key in ("spec_extracted", "engine_supported", "supported"):
        if key in hint:
            v = hint[key]
            claims.append("%s=%s" % (key, "true" if v is True else
                                     "false" if v is False else str(v)))
    return claims


def tied_head_mb(action: str) -> Optional[str]:
    """MB cost adapt.py quotes for materialising a tied lm_head, or None."""
    m = _TIE_MB_RE.search(action or "")
    return m.group(1) if m else None


def tier_label(tier: str) -> str:
    """Translated tier word (hook / new_op / post + anything a later adapt.py adds)."""
    return t(TIER_KEYS.get(tier, "imp.tier.other"))


def gap_action_text(need: str, tier: str, action: str) -> str:
    """Engine-side action of one gap, in the current language.

    Known `need` shapes render from the i18n table (en is keyed by the `need`
    prefix, because adapt.py writes the action in Chinese). An unknown shape
    falls back to a message that names the translated tier and keeps the
    importer's raw action -- new gap kinds stay readable in both languages.
    """
    for pattern, key, groups in _GAP_ACTION_ROWS:
        m = pattern.match(need or "")
        if not m:
            continue
        if isinstance(key, dict):
            key = key.get(tier) or key["default"]
        # Group names are matched positionally against the regex captures, except
        # `mb`: the tied-head cost is not in `need` at all, it is quoted inline by
        # adapt.py in `action` ("... 约 671 MB ..."), so it has to be pulled out
        # separately. A row may therefore name a group its pattern does not capture.
        kw, missing_cost = {}, False
        for i, name in enumerate(groups):
            if name == "mb":
                mb = tied_head_mb(action)
                if mb is None:
                    missing_cost = True  # cost not quoted -> keep the raw importer text
                    break
                kw["mb"] = mb
            elif i < len(m.groups()):
                kw[name] = (m.groups()[i] or "").strip()
        if missing_cost:
            # The shape *is* known, only the size adapt.py quotes is missing. Say the
            # same thing without a number rather than claiming an untranslated kind.
            if i18n_has(key + ".nosize"):
                return t(key + ".nosize")
            break
        return t(key, **kw)
    return t('imp.gaps.fallback', tier=tier_label(tier), action=action)


def header_file_state(manifest_path) -> str:
    """Which TextConfig header, if any, actually sits next to this manifest.

    A fact about the directory, not a claim derived from the verdict. It used to be
    the expression `"config.h.BLOCKED" if blocking else "config.h"`, which was wrong
    twice over:

      * nothing in this tree writes `config.h.BLOCKED`. The refusal path in adapt.py
        writes NO header at all (and records the reason in the manifest); a `new_op`
        gap does not withhold the header either. So the report named a file that does
        not exist -- a sentence about the disk that never looked at the disk.
      * measured counter-example, on the model the wizard is pointed at today:
        `tools/archkit/out/ornith-1.5-9b-q4-k-m/` carries one `new_op` gap
        (`attention:linear(gdn)`) AND a 929-byte `config.h`, with no
        `config.h.BLOCKED` anywhere. The old expression reported the opposite for
        that exact directory.

    Presence is all a directory listing can establish: a `config.h` left over from an
    earlier run is indistinguishable from a fresh one this way (adapt.py prints that
    warning on the refusal path, and the manifest carries the spec any header would
    have been built from -- comparing the two is what would settle it, and that is not
    something a scan of the folder can do). Returning the empty string means "no
    header next to this manifest", which the renderer says out loud instead of naming
    a file. `"unknown"` stays reserved for "the gaps were never measured", where the
    header's usability is unmeasured too.
    """
    if not manifest_path:
        return ""
    d = Path(manifest_path).parent
    for name in ("config.h", "config.h.BLOCKED"):
        if (d / name).is_file():
            return name
    return ""


def import_gap_report(model_id: str, manifest_path: str = "",
                      lang: str = "") -> dict:
    """Structured, translated gap report for one adapted model.

    Returns (all strings rendered in the current language; pass lang= to switch
    first):
      model_id, manifest, ok, error, blocked, blocking[], header_file,
      gaps_measured, verdict ('blocked'|'clear'|'unmeasured'), gap_shape,
      gap_count, tier_counts{}, tiers [{tier, label, rows[{need, action,
      raw_action, translated}]}], rows[], next_steps[], notes[]

    `gaps_measured` and `verdict` are the load-bearing fields. `ok` only means the
    manifest was found and parsed -- it says nothing about whether the gaps were
    measured, so a caller must not treat `ok` as a green light. An empty gap list
    is a measurement; a missing / null / non-list `gaps`, a list whose elements are
    not objects (or whose `tier` is not a tier this tree writes), and a manifest
    carrying an unhonoured `measured` key are not.

    `header_file` is a fact about the directory -- the name of the header file next to
    the manifest, `""` when there is none, `"unknown"` when the gaps were unmeasured
    (see header_file_state()). It is NOT the verdict in file-name form, and in
    particular a `new_op` gap does not withhold the header.

    The four outcomes and their exit codes (main() maps them 1:1 through
    gap_report_rc(), so a shell caller that only looks at rc cannot be fooled
    either):
      rc 0  ok=True,  gaps_measured=True   verdict clear
      rc 3  ok=True,  gaps_measured=True   verdict blocked (non-zero on purpose:
                                           BLOCKED used to exit 0, which made the
                                           verdict "not fit to convert" and the
                                           exit code "fine" disagree)
      rc 1  ok=False                       no manifest, or it does not parse
      rc 2  ok=True,  gaps_measured=False  verdict 'unmeasured' -- read, but the
                                           gaps were never measured, so the answer
                                           is unknown and NOT a green light
    """
    if lang:
        set_lang(lang)
    path = Path(manifest_path) if manifest_path else find_manifest(model_id)
    base = {"model_id": model_id, "manifest": str(path) if path else "", "ok": False,
            "error": "", "blocked": False, "blocking": [], "header_file": "",
            "gaps_measured": False, "verdict": "unmeasured", "gap_shape": "",
            "gui_hint_claims": [],
            "gap_count": 0, "tier_counts": {}, "tiers": [], "rows": [],
            "next_steps": [], "notes": []}
    if not path or not path.is_file():
        base["error"] = t('imp.gaps.no_manifest', id=model_id)
        return base
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        base["error"] = t('imp.gaps.bad_manifest', err=e)
        return base

    gaps_measured, gaps, gap_shape = manifest_gap_state(data)
    if not gaps_measured:
        # No gaps LIST was readable. Whether the field is absent, null, or not a
        # list does not matter: all of them mean "the gaps were never measured",
        # and none of them may render as "no gaps, the engine can eat it".
        # `"gaps": null` used to slip through -- the guard tested
        # `"gaps" not in data`, and an explicit null IS in data -- so a manifest
        # written by gen_target.py produced VERDICT: CLEAR / rc 0. Measured on
        # such a manifest; test_model_import.py carries both cases as fixtures.
        hint = data.get("gui_hint") if isinstance(data.get("gui_hint"), dict) else {}
        claims = gui_hint_claims(hint)
        base["gaps_measured"] = False
        base["verdict"] = "unmeasured"
        # An unmeasured manifest cannot vouch for the header either way.
        base["header_file"] = "unknown"
        base["gap_shape"] = gap_shape
        base["gui_hint_claims"] = claims
        base["error"] = t('imp.gaps.gaps_unmeasured', path=str(path),
                          shape=gap_shape_text(gap_shape),
                          hint=hint.get("import_note") or "",
                          claims=', '.join(claims))
        # `ok` means "found and parsed", nothing more -- that is what its docstring
        # says, and it has to mean it, or the three outcomes below collapse into two:
        # with ok=False here, main() could never reach its rc=2 branch and
        # render_gap_report_text() printed the raw error plus the usage line instead of
        # the UNKNOWN verdict it had been written to print. An unmeasured manifest IS a
        # readable manifest; it is just not a decisive one.
        base["ok"] = True
        return base
    rows, by_tier = [], {}
    for g in gaps:
        need = str(g.get("need") or "")
        tier = str(g.get("tier") or "")
        raw = str(g.get("action") or "")
        action = gap_action_text(need, tier, raw)
        row = {"need": need, "tier": tier, "action": action, "raw_action": raw,
               "translated": action != raw}
        rows.append(row)
        by_tier.setdefault(tier, []).append(row)

    blocking = [r["need"] for r in rows if r["tier"] == "new_op"]
    order = [x for x in TIER_ORDER if x in by_tier]
    order += sorted(k for k in by_tier if k not in TIER_ORDER)
    report = dict(base)
    report.update({
        "model_id": data.get("model_id") or model_id,
        "ok": True, "blocked": bool(blocking), "blocking": blocking,
        # gaps came from an actual list, so this report can be trusted to be
        # about something. `clear` here means "a measured list with no new_op".
        "gaps_measured": True, "verdict": "blocked" if blocking else "clear",
        # Read off the directory, not derived from the gap tiers: adapt.py does not
        # withhold the header when a new_op gap is open, and no writer in this tree
        # produces `config.h.BLOCKED` at all.
        "header_file": header_file_state(path),
        "gap_count": len(rows),
        "tier_counts": {k: len(v) for k, v in by_tier.items()},
        "tiers": [{"tier": k, "label": tier_label(k), "rows": by_tier[k]} for k in order],
        "rows": rows,
    })
    # The way out of a block: a conversion step where the converter can do the job
    # (tied head) and real engine work where it cannot.
    for need in blocking:
        if need.startswith("head:tied=true"):
            raw = next((r["raw_action"] for r in rows if r["need"] == need), "")
            report["next_steps"].append(
                t('imp.gaps.tie_step', mb=tied_head_mb(raw) or "?"))
            note = t('imp.gaps.tie_note')
            if note not in report["notes"]:
                report["notes"].append(note)
        else:
            report["next_steps"].append(t('imp.gaps.blocking_engine', need=need))
    return report


def render_gap_report_text(report: dict) -> str:
    """Plain-text rendering of import_gap_report(), current language."""
    out = [t('imp.gaps.title'), ""]
    if not report.get("ok"):
        out.append(report.get("error") or t('imp.gaps.bad_manifest', err="?"))
        out.append("")
        out.append(t('imp.gaps.usage'))
        return "\n".join(out)
    counts = report.get("tier_counts") or {}
    out.append(t('imp.gaps.line_model', model=report["model_id"]))
    out.append(t('imp.gaps.line_manifest', path=report["manifest"]))
    if not report.get("gaps_measured"):
        # UNMEASURED. The summary line above is deliberately absent: printing
        # "gaps: 0 (hook 0 / new_op 0 / post 0)" for a manifest that was never
        # measured is the false all-clear in numeric form. The verdict comes
        # first, and it is not CLEAR.
        out.append(t('imp.gaps.verdict_unmeasured'))
        out.append(t('imp.gaps.header_unmeasured'))
        # Which shape of the absence was found is load-bearing: "no gaps field at all"
        # and "gaps is explicitly null" are different files and are told apart here.
        out.append(report.get("error") or "")
        out.append(t('imp.gaps.gaps_unmeasured_how'))
        return "\n".join(out)
    out.append(t('imp.gaps.line_summary', total=report["gap_count"],
                 hook=counts.get("hook", 0), new_op=counts.get("new_op", 0),
                 post=counts.get("post", 0)))
    # The verdict comes before the detail: a blocked import must never read as
    # success, and neither may an unmeasured one (handled above).
    out.append(t('imp.gaps.verdict_blocked') if report["blocked"]
               else t('imp.gaps.verdict_ok'))
    # The header line is decided by the FILE, not by the verdict: a new_op gap leaves
    # config.h in place, so "blocked" is not a reason to claim a withheld header. An
    # empty header_file means the directory holds none -- say that, and say what it
    # costs (a stale config.h from an earlier run is the trap in that case).
    header = report.get("header_file") or ""
    if header:
        out.append(t('imp.gaps.header_ok', name=header))
    else:
        out.append(t('imp.gaps.header_missing'))
    if report["blocked"]:
        out.append(t('imp.gaps.blocking', items="; ".join(report["blocking"])))
    if report["next_steps"]:
        out.append("")
        out.append(t('imp.gaps.next_step_head'))
        for i, step in enumerate(report["next_steps"], 1):
            out.append("  %d. %s" % (i, step))
    for note in report["notes"]:
        out.append("  ! %s" % note)
    if not report["gap_count"]:
        out.append("")
        out.append(t('imp.gaps.no_gaps'))
        return "\n".join(out)
    for group in report["tiers"]:
        out.append("")
        out.append(t('imp.gaps.tier_head', tier=group["label"], n=len(group["rows"])))
        for row in group["rows"]:
            out.append(t('imp.gaps.row', need=row["need"]))
            out.append(t('imp.gaps.row_action', action=row["action"]))
    return "\n".join(out)


#: Exit codes of `--gaps`, one per readable outcome -- see the module docstring.
#: 3 is deliberately the same number tools/archkit/adapt_all.py uses for "read and
#: judged, not servable", so the two entry points speak one vocabulary.
GAP_RC_UNREADABLE = 1     # no manifest, or it does not parse
GAP_RC_UNKNOWN = 2        # read, but the gaps were never measured
GAP_RC_BLOCKED = 3        # read, measured, and there is a new_op gap


def gap_report_rc(report: dict, require_servable: bool = False) -> int:
    """The exit code of one gap report. Extracted from main() so it can be exercised
    for every outcome instead of being an expression buried in the argument parser.

    0 CLEAR / 1 unreadable / 2 UNKNOWN / 3 BLOCKED. The code follows the verdict, not
    "the report printed" -- BLOCKED is non-zero, because `python model_import.py
    --gaps X && continue` is a gate whether or not anyone has written it yet.

    require_servable=True collapses the two readable-but-not-servable outcomes into
    rc 3: at that call site the caller has said out loud that it wants a gate, so a
    single "not servable" code is what it asked for. Without the flag the number
    still distinguishes "unknown" from "blocked" for a caller that can act on the
    difference.
    """
    if not report.get("ok"):
        return GAP_RC_UNREADABLE
    if not report.get("gaps_measured"):
        return GAP_RC_BLOCKED if require_servable else GAP_RC_UNKNOWN
    if report.get("blocked"):
        return GAP_RC_BLOCKED
    return 0


def _prescan_lang(argv) -> str:
    """Pick up --lang before argparse builds the parser.

    argparse bakes the translated help/description in at construction time and
    `--help` exits from inside parse_args(), so the language has to be settled
    first for `--lang en --help` to actually read as English.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    for i, a in enumerate(args):
        if a == '--lang' and i + 1 < len(args):
            return args[i + 1]
        if a.startswith('--lang='):
            return a.split('=', 1)[1]
    return ''


def main(argv=None) -> int:
    """CLI self-test: python model_import.py <path>... | --gaps <model-id>
    [--lang zh|en] [--require-servable]"""
    preset = _prescan_lang(argv)
    if preset:
        set_lang(preset)
    ap = argparse.ArgumentParser(description=t('imp.cli.desc'), add_help=True)
    ap.add_argument("paths", nargs="*", help=t('imp.cli.paths_help'))
    ap.add_argument("--gaps", metavar="MODEL_ID", default="",
                    help=t('imp.cli.gaps_help'))
    ap.add_argument("--require-servable", action="store_true",
                    help=t('imp.cli.require_servable_help'))
    ap.add_argument("--lang", choices=("zh", "en"), default="", help=t('misc.lang_label'))
    a = ap.parse_args(argv)
    if a.lang:
        set_lang(a.lang)
    if a.gaps:
        rep = import_gap_report(a.gaps)
        print(render_gap_report_text(rep))
        rc = gap_report_rc(rep, require_servable=a.require_servable)
        # The gate says WHY on stderr, so a shell caller that only logs the exit code
        # of `--gaps X --require-servable` still gets a sentence next to the number.
        # Only when the flag actually decided something (read, but not servable): for
        # rc 1 the report above already says the manifest could not be read, and
        # naming a verdict for a file that was never parsed would be a claim about it.
        if a.require_servable and rep.get("ok") and rc != 0:
            print(t('imp.cli.gaps_gate_refused',
                    verdict=rep.get("verdict", "?"), rc=rc), file=sys.stderr)
        return rc
    if not a.paths:
        print(t('imp.gaps.usage'))
        return 2
    for path in a.paths:
        sc = scan_path(path)
        v = verdict(sc)
        print(t('imp.cli.path', path=path))
        print(t('imp.cli.format', kind=t('imp.kind.%s' % sc.kind), friendly=sc.friendly))
        if sc.model_id:
            print(t('imp.cli.family', family=sc.model_id))
        if sc.arch_note:
            print(t('imp.cli.arch', note=sc.arch_note))
        if sc.unmapped:
            print(t('imp.cli.unmapped', n=len(sc.unmapped),
                    names=", ".join(sc.unmapped)))
        if sc.weight_bytes:
            print(t('imp.cli.size', gb="%.2f" % (sc.weight_bytes / 1e9),
                    quant=sc.quant_name or "-"))
        if sc.issues:
            print(t('imp.cli.issues', issues="; ".join(sc.issues)))
        print(t('imp.cli.verdict', level=v.level, title=v.title))
        print("  %s" % v.detail)
        for tip in v.tips:
            print(t('imp.cli.bullet', text=tip))
        # A manifest from an earlier adapt.py run is the authoritative gap list for
        # this model: surface it here instead of letting the import look clean.
        if sc.model_id and find_manifest(sc.model_id):
            print()
            print(render_gap_report_text(import_gap_report(sc.model_id)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
