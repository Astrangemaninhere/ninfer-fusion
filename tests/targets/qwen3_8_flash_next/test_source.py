# Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
from __future__ import annotations

import json
from pathlib import Path
import struct

import pytest

from tools.convert.qwen3_8_flash_next.source import SourceError, read_safetensor_directory


def _write_safetensor(path: Path, *, truncate: bool = False) -> None:
    directory = {
        "codes": {"dtype": "U8", "shape": [2, 2], "data_offsets": [0, 4]},
        "scale": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
    }
    header = json.dumps(directory, separators=(",", ":")).encode("utf-8")
    payload = b"\x01\x02\x03\x04" + struct.pack("<f", 1.0)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload[:-1 if truncate else None])


def test_raw_safetensor_directory_exposes_exact_payload_slices(tmp_path):
    path = tmp_path / "fixture.safetensors"
    _write_safetensor(path)
    directory = read_safetensor_directory(path)
    assert directory.header_bytes > 0
    assert directory.tensors["codes"].dtype == "U8"
    assert directory.tensors["codes"].shape == (2, 2)
    assert directory.tensors["codes"].bytes == 4
    assert directory.tensors["scale"].absolute_offset == 8 + directory.header_bytes + 4


def test_raw_safetensor_directory_rejects_truncated_payload(tmp_path):
    path = tmp_path / "truncated.safetensors"
    _write_safetensor(path, truncate=True)
    with pytest.raises(SourceError, match="payload size mismatch"):
        read_safetensor_directory(path)


def test_validate_config_accepts_pinned_bos_eos_and_rejects_drift(tmp_path):
    from tools.convert.qwen3_8_flash_next.source import _validate_config

    valid_config = {
        "model_type": "qwen4_exp",
        "text_config": {
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "hc_count": 4,
            "ngram_vocab_size_base": 20_000_000,
            "split_ngram_parts": 128,
            "ple_embed_dim": 2560,
            "vocab_size": 248_320,
            "bos_token_id": 248_044,
            "eos_token_id": 248_044,
        },
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(valid_config), encoding="utf-8")
    _validate_config(cfg_path)

    # Drift on bos_token_id
    drift_bos = json.loads(json.dumps(valid_config))
    drift_bos["text_config"]["bos_token_id"] = 151643
    cfg_path.write_text(json.dumps(drift_bos), encoding="utf-8")
    with pytest.raises(SourceError, match="bos_token_id mismatch"):
        _validate_config(cfg_path)

    # Drift on eos_token_id
    drift_eos = json.loads(json.dumps(valid_config))
    drift_eos["text_config"]["eos_token_id"] = 248046
    cfg_path.write_text(json.dumps(drift_eos), encoding="utf-8")
    with pytest.raises(SourceError, match="eos_token_id mismatch"):
        _validate_config(cfg_path)
