"""Pinned official Qwen3.6 frontend resources used by artifact conversion.

This module owns only the checkpoint-invariant resource profile.  Exact-target
converters continue to own config, tensor inventory, recipes, and execution
geometry.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

from .conversion import ResourcePayload, load_resources
from .inventory import ResourceSpec


OFFICIAL_RESOURCE_SHA256 = {
    "frontend/tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "frontend/tokenizer_config.json": (
        "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b"
    ),
    "frontend/chat_template.jinja": (
        "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
    ),
    "frontend/generation_config.json": (
        "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
    ),
    "frontend/preprocessor_config.json": (
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    ),
    "frontend/video_preprocessor_config.json": (
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
    ),
}

#: Revisions of the same six resources that a *registered variant* legitimately ships,
#: keyed by resource name and then by sha256, with the provenance of each hash.
#:
#: This is not a relaxation of the pin: each entry is the recorded digest of one
#: reviewed file that belongs to a registered upstream release, it only ever answers
#: for the resource name it is filed under, and any digest outside both tables stays
#: ``unproven`` and stays fatal without ``--allow-frontend-drift``.  It exists because
#: the single pin describes one revision of one release; a variant of the same base
#: model that ships its own tokenizer_config / template / generation_config would
#: otherwise be refused for being a different *revision* rather than a different
#: *model*, and the only way past that today is the blanket opt-out.
REGISTERED_RESOURCE_SHA256 = {
    "frontend/tokenizer_config.json": {
        "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27": (
            "Qwen3.8-27B 的注册钉死值：tools/convert/qwen3_8_27b/convert.py 的 "
            "OFFICIAL_RESOURCE_SHA256 就是这一份，该 target 的转换器直接接受它"
        ),
    },
    "frontend/chat_template.jinja": {
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041": (
            "Qwen3.8-27B 的注册钉死值：tools/convert/qwen3_8_27b/convert.py 的 "
            "OFFICIAL_RESOURCE_SHA256 就是这一份，该 target 的转换器直接接受它"
        ),
    },
    "frontend/generation_config.json": {
        "b8eb74d15e0a56623d00ccd14950a4bb87fabbf84b5cc030dcc904b899fb1eb5": (
            "Qwen3.8-27B-ET-Uncensored-NVFP4/W4A4+W8A8 随件（214 B）；内容只声明 "
            "bos_token_id=248044 / eos_token_id=[248046, 248044] / pad_token_id=248044 "
            "（与 config.json 逐项一致）与 do_sample/temperature=1.0/top_k=20/top_p=0.95 "
            "（与 Qwen3.8 的采样默认值逐项一致），不含任何分词规则或词表"
        ),
    },
}


def validate_official_resource_hashes(
    actual_hashes: Mapping[str, str],
) -> None:
    """Require the complete official six-resource profile."""

    expected_names = tuple(OFFICIAL_RESOURCE_SHA256)
    actual_names = tuple(actual_hashes)
    if actual_names != expected_names:
        raise ValueError(
            "Qwen3.6 frontend resource set mismatch: "
            f"expected {expected_names!r}, got {actual_names!r}"
        )
    for name, expected in OFFICIAL_RESOURCE_SHA256.items():
        actual = actual_hashes[name]
        if actual != expected:
            filename = name.removeprefix("frontend/")
            raise ValueError(
                f"official Qwen3.6 resource hash mismatch for {filename}: "
                f"expected {expected}, got {actual}"
            )


def validate_official_resources(resources: Sequence[ResourcePayload]) -> None:
    """Hash and validate already loaded resource payloads."""

    hashes = {
        resource.name: hashlib.sha256(resource.data).hexdigest()
        for resource in resources
    }
    if len(hashes) != len(resources):
        raise ValueError("Qwen3.6 frontend resource set contains duplicate names")
    validate_official_resource_hashes(hashes)


def load_official_resources(
    model_dir: str | Path,
    resource_specs: Sequence[ResourceSpec],
) -> tuple[ResourcePayload, ...]:
    """Load exactly the pinned official resource set from a source checkpoint."""

    spec_names = tuple(spec.name for spec in resource_specs)
    expected_names = tuple(OFFICIAL_RESOURCE_SHA256)
    if spec_names != expected_names:
        raise ValueError(
            "converter resource inventory does not match the official Qwen3.6 profile: "
            f"expected {expected_names!r}, got {spec_names!r}"
        )
    resources = load_resources(model_dir, resource_specs)
    validate_official_resources(resources)
    return resources


__all__ = [
    "OFFICIAL_RESOURCE_SHA256",
    "REGISTERED_RESOURCE_SHA256",
    "load_official_resources",
    "validate_official_resource_hashes",
    "validate_official_resources",
]
