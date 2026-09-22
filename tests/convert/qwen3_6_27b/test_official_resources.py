from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.convert.qwen3_6.common.official_resources import (
    OFFICIAL_RESOURCE_SHA256,
    validate_official_resource_hashes,
)
from tools.convert.qwen3_6_27b import convert as convert_27b
from tools.convert.qwen3_6_35b_a3b import convert as convert_35b


def _official_dir(variable: str) -> Path | None:
    """Opt-in official HF directory.

    The tree bakes in no machine path, so an unset variable (or a missing
    directory) makes the checks below skip instead of failing.
    """

    value = os.environ.get(variable, "")
    return Path(value) if value else None


MODEL_27B = _official_dir("NINFER_OFFICIAL_27B_DIR")
MODEL_35B = _official_dir("NINFER_OFFICIAL_35B_DIR")
UNSLOTH_TOKENIZER_SHA256 = (
    "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"
)


@pytest.mark.parametrize(
    ("loader", "model_dir", "variable"),
    (
        (convert_27b.load_resources, MODEL_27B, "NINFER_OFFICIAL_27B_DIR"),
        (convert_35b.load_resources, MODEL_35B, "NINFER_OFFICIAL_35B_DIR"),
    ),
)
def test_both_official_sources_pass_the_shared_preflight(loader, model_dir, variable):
    if model_dir is None or not model_dir.is_dir():
        pytest.skip(f"set {variable} to the official HF directory to run this check")
    resources = loader(model_dir)

    assert tuple(resource.name for resource in resources) == tuple(
        OFFICIAL_RESOURCE_SHA256
    )


def test_unsloth_tokenizer_hash_is_rejected():
    hashes = dict(OFFICIAL_RESOURCE_SHA256)
    hashes["frontend/tokenizer.json"] = UNSLOTH_TOKENIZER_SHA256

    with pytest.raises(
        ValueError,
        match=(
            "tokenizer.json.*expected "
            + OFFICIAL_RESOURCE_SHA256["frontend/tokenizer.json"]
            + ".*got "
            + UNSLOTH_TOKENIZER_SHA256
        ),
    ):
        validate_official_resource_hashes(hashes)
