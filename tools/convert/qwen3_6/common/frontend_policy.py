"""Evidence-based frontend resource resolution for imported checkpoints.

The registered profile pins one sha256 per frontend resource.  That pin is a
*revision* pin: it names the exact file from the official repository, and until
now any deviation was fatal.  It was fatal for a good reason - token id drift
silently corrupts generation - but the pin is strictly stronger than the
requirement, and refusing a checkpoint whose tokenizer is provably the same
tokenizer is the wrong answer when the goal is to ingest any local model.

This module separates the two concerns.  Each resource is resolved to exactly
one of four states, with the evidence that produced it:

``pinned``      the file's sha256 equals the registered pin;
``consistent``  a different revision, but its semantic content agrees with what
                this checkpoint declares (for a tokenizer: the declared special
                token ids resolve to the right tokens and every id fits inside
                ``vocab_size``);
``unproven``    a different revision with no available proof of equivalence;
``missing``     no usable file.

``missing`` is always fatal: an artifact cannot be built without a tokenizer.
``pinned`` and ``consistent`` are accepted.  ``unproven`` is accepted only when
the caller explicitly asks for it, and every non-pinned resource is reported so
the deviation travels with the conversion report instead of being invisible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


#: The resource names the registered profile owns, in canonical order.
FRONTEND_NAMES: tuple[str, ...] = (
    "frontend/tokenizer.json",
    "frontend/tokenizer_config.json",
    "frontend/chat_template.jinja",
    "frontend/generation_config.json",
    "frontend/preprocessor_config.json",
    "frontend/video_preprocessor_config.json",
)

#: Config keys whose value must exist in the tokenizer's id space.
SPECIAL_TOKEN_KEYS: tuple[str, ...] = (
    "vision_start_token_id",
    "vision_end_token_id",
    "image_token_id",
    "video_token_id",
    "bos_token_id",
    "eos_token_id",
)


@dataclass(frozen=True, slots=True)
class FrontendResolution:
    name: str
    status: str
    provider: Path | None
    sha256: str | None
    evidence: str

    @property
    def pinned(self) -> bool:
        return self.status == "pinned"


@dataclass(frozen=True, slots=True)
class FrontendProfile:
    source: Path
    resolutions: tuple[FrontendResolution, ...]

    @property
    def by_name(self) -> dict[str, FrontendResolution]:
        return {item.name: item for item in self.resolutions}

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.resolutions if item.status == "missing")

    @property
    def unproven(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.resolutions if item.status == "unproven")

    @property
    def deviations(self) -> tuple[FrontendResolution, ...]:
        return tuple(item for item in self.resolutions if item.status != "pinned")

    def report(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "resources": {
                item.name: {
                    "status": item.status,
                    "provider": str(item.provider) if item.provider else None,
                    "sha256": item.sha256,
                    "evidence": item.evidence,
                }
                for item in self.resolutions
            },
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _token_id_map(path: Path) -> dict[str, int] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    ids: dict[str, int] = {}
    model = payload.get("model")
    if isinstance(model, Mapping) and isinstance(model.get("vocab"), Mapping):
        for token, index in model["vocab"].items():
            if isinstance(index, int):
                ids[str(token)] = index
    added = payload.get("added_tokens")
    if isinstance(added, list):
        for entry in added:
            if isinstance(entry, Mapping) and "content" in entry and isinstance(entry.get("id"), int):
                ids[str(entry["content"])] = int(entry["id"])
    return ids or None


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, Mapping) else config


def tokenizer_consistency(tokenizer_path: Path, config: Mapping[str, Any]) -> tuple[bool, str]:
    """Prove the tokenizer agrees with what this checkpoint declares.

    Two checks, both necessary for correct generation: every declared special
    token id must resolve to a token that exists, and the largest id in the
    tokenizer must fit inside the declared vocabulary (the embedding table is
    allowed to be padded; it is not allowed to be smaller).
    """

    ids = _token_id_map(tokenizer_path)
    if ids is None:
        return False, f"{tokenizer_path} 无法解析为 tokenizer.json"
    by_id: dict[int, str] = {}
    for token, index in ids.items():
        by_id.setdefault(index, token)

    problems: list[str] = []
    checked: list[str] = []
    for key in SPECIAL_TOKEN_KEYS:
        wanted = config.get(key)
        if wanted is None:
            wanted = _text_config(config).get(key)
        if not isinstance(wanted, int):
            continue
        token = by_id.get(wanted)
        if token is None:
            problems.append(f"{key}={wanted} 在 tokenizer 里不存在")
        else:
            checked.append(f"{key}={wanted}->{token!r}")

    vocab_size = _text_config(config).get("vocab_size")
    max_id = max(ids.values())
    if isinstance(vocab_size, int) and max_id >= vocab_size:
        problems.append(f"tokenizer 最大 id {max_id} 超出 vocab_size {vocab_size}")

    if problems:
        return False, "；".join(problems)
    return True, (
        f"特殊 token id 全部对得上（{len(checked)} 项：{'; '.join(checked[:4])}）；"
        f"最大 id {max_id} < vocab_size {vocab_size}"
    )


def _hunt(name: str, roots: Sequence[Path], exclude: Path) -> list[Path]:
    bare = name.removeprefix("frontend/")
    hits: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        direct = root / bare
        if direct.is_file() and direct.resolve() != exclude.resolve():
            hits.append(direct)
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.resolve() != exclude.resolve():
                candidate = child / bare
                if candidate.is_file():
                    hits.append(candidate)
    return hits


def _provider_matches_family(candidate: Path, config: Mapping[str, Any], tokenizer: bool) -> tuple[bool, str]:
    """Refuse a candidate that belongs to a different model.

    A tokenizer from an unrelated checkpoint has a different id space; a chat
    template from an unrelated checkpoint formats prompts differently.  Both are
    detectable from the candidate's own config.json, so neither is ever offered.
    """

    config_path = candidate.parent / "config.json"
    if not config_path.is_file():
        return False, "候选目录没有 config.json，无法判定归属"
    try:
        with config_path.open(encoding="utf-8") as handle:
            other = json.load(handle)
    except (OSError, ValueError) as exc:
        return False, f"候选 config.json 不可读（{exc}）"
    if not isinstance(other, Mapping):
        return False, "候选 config.json 不是对象"

    if tokenizer:
        mine = _text_config(config).get("vocab_size")
        theirs = _text_config(other).get("vocab_size")
        if mine is None or theirs is None:
            return False, f"无法比较词表（本 {mine} / 候选 {theirs}）"
        if mine != theirs:
            return False, f"词表不符（本 {mine} / 候选 {theirs}）"
        return True, f"词表一致（{mine}）"
    mine_type = config.get("model_type")
    theirs_type = other.get("model_type")
    if mine_type is None or theirs_type is None:
        return False, f"无法比较族（本 {mine_type} / 候选 {theirs_type}）"
    if mine_type != theirs_type:
        return False, f"族不符（本 {mine_type} / 候选 {theirs_type}）"
    return True, f"族一致（{mine_type}）"


def resolve_frontend_profile(
    model_dir: str | Path,
    config: Mapping[str, Any],
    *,
    pins: Mapping[str, str] | None = None,
    roots: Sequence[str | Path] = (),
) -> FrontendProfile:
    """Resolve every pinned resource with an explicit, testable evidence order."""

    if pins is None:
        from .official_resources import OFFICIAL_RESOURCE_SHA256

        pins = OFFICIAL_RESOURCE_SHA256

    source = Path(model_dir)
    search_roots = [Path(root) for root in roots]
    tokenizer_seeds = [source / "tokenizer.json"] + _hunt("frontend/tokenizer.json", search_roots, source)

    resolutions: list[FrontendResolution] = []
    for name in FRONTEND_NAMES:
        if name not in pins:
            continue
        bare = name.removeprefix("frontend/")
        local = source / bare
        pin = pins[name]

        if local.is_file():
            local_hash = sha256_file(local)
            if local_hash == pin:
                resolutions.append(FrontendResolution(
                    name, "pinned", local, local_hash, "sha256 与钉死值一致（源件自带）"))
                continue
            provider, origin = local, "源件自带"
        else:
            local_hash = None
            provider, origin = None, ""

        # A copy that matches the pin is strictly better than a drifted local file.
        if provider is None or local_hash != pin:
            for candidate in _hunt(name, search_roots, source):
                if sha256_file(candidate) == pin:
                    provider, origin = candidate, "本地钉死副本"
                    break

        if provider is None:
            bare_is_tokenizer = bare.startswith("tokenizer")
            accepted: list[tuple[Path, str]] = []
            rejected: list[str] = []
            for candidate in _hunt(name, search_roots, source):
                ok, why = _provider_matches_family(candidate, config, bare_is_tokenizer)
                if ok:
                    accepted.append((candidate, why))
                else:
                    rejected.append(f"{candidate}（{why}）")
            if accepted:
                provider, origin = accepted[0][0], f"本地同族副本（{accepted[0][1]}）"
            else:
                resolutions.append(FrontendResolution(
                    name, "missing", None, None,
                    "源件没有该文件；本地同名候选均被排除：" + ("；".join(rejected[:3]) if rejected else "无候选"),
                ))
                continue

        digest = sha256_file(provider)
        if digest == pin:
            resolutions.append(FrontendResolution(
                name, "pinned", provider, digest, f"{origin}；sha256 与钉死值一致",
            ))
            continue
        if name == "frontend/tokenizer.json":
            ok, note = tokenizer_consistency(provider, config)
            status = "consistent" if ok else "unproven"
            if ok and any(seed.is_file() and _token_id_map(seed) == _token_id_map(provider)
                          for seed in tokenizer_seeds if seed.resolve() != provider.resolve()):
                note += "；且与第二份本地副本的 token->id 映射逐项相同"
            resolutions.append(FrontendResolution(name, status, provider, digest, f"{origin}；{note}"))
        elif name == "frontend/tokenizer_config.json":
            resolutions.append(FrontendResolution(
                name, "unproven", provider, digest,
                f"{origin}；added_tokens 影响 draft 短名单 id 映射，无法证明与钉死版本等价",
            ))
        else:
            resolutions.append(FrontendResolution(
                name, "unproven", provider, digest,
                f"{origin}；无法证明与钉死版本等价（sha256 {digest[:16]}）",
            ))

    return FrontendProfile(source, tuple(resolutions))


def acceptability_error(profile: FrontendProfile, *, allow_unproven: bool) -> str | None:
    """Return the reason the profile cannot be used, or None when it can."""

    if profile.missing:
        return (
            "前端资源缺件：" + ", ".join(profile.missing)
            + "；补救：从官方仓库按钉死 revision 取回这些文件，或把 --resource-root 指向含它们的目录"
        )
    if profile.unproven and not allow_unproven:
        names = ", ".join(profile.unproven)
        return (
            f"前端资源 {names} 与钉死版本不同且无法证明等价；"
            "确认后可用 --allow-frontend-drift 放行（偏离会记入转换报告）"
        )
    return None


__all__ = [
    "FRONTEND_NAMES",
    "FrontendProfile",
    "FrontendResolution",
    "SPECIAL_TOKEN_KEYS",
    "acceptability_error",
    "resolve_frontend_profile",
    "sha256_file",
    "tokenizer_consistency",
]
