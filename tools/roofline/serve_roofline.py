#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""serve_roofline.py — per-card (artifact x card) memory-bandwidth roofline for the SERVE path.

WHAT THIS ANSWERS
-----------------
"On this card, what fraction of the theoretical bandwidth limit is the artifact actually
running at, at this context length?"  That is:

    tok/s_measured / tok/s_theory,   with
    tok/s_theory(card, artifact, ctx) = bw_measured(card) / (weight_bytes + kv_bytes(ctx))

The definition of the theoretical limit is NOT invented here. It is the one already in this
tree, verbatim, from tools/archkit/qpn_port/placement_planner.py:31-38:

    decode  (batch=1, bandwidth-bound): t = (weight_bytes + kv_bytes(ctx)) / bw
    prefill (compute-bound):            t = flops / compute + kv_bytes(ctx) / bw

This program implements the FIRST line only, because that is the line that is a genuine
roofline at batch 1. See --notes for where this model is valid and where it is not.

WHERE EACH NUMBER COMES FROM (no number is invented; each is sourced)
--------------------------------------------------------------------
numerator   weight bytes      artifact's own object directory (tools/artifact/container.py),
                              and, when a serve record exists, the engine's own
                              artifact.host_to_device_bytes (resident weight bytes).
numerator   kv bytes/token    engine-reported: MemorySummary.kv_payload_bytes divided by the
                              resolved KV capacity in tokens (engine.kv_capacity). Both are
                              emitted by ninfer-serve itself (request-log "server_start"), or
                              read off serve stdout ("KV capacity ... resolved=N tokens").
                              This is NOT derivable from the artifact: docs/maintainer/
                              artifact-container.md §3.1 says the directory carries "no model
                              graph", so geometry lives in the compiled contract.
denominator bw_measured       the measured pure-read ceiling 1674.5 GB/s from
                              tools/hbm_bandwidth_probe.cu, cited at bench/README.md:232.
                              Override with --bandwidth-gbs and the provenance is reprinted.
measured    tok/s             the serve binary's own accounting, from the request-log
                              "request_done" record: (completion_tokens - 1) / decode_seconds.
                              The -1 is the engine's own convention
                              (src/serve/request_log.cpp: "Prefill emits the first token;
                              the remaining (gen - 1) come from decode").
per-card    calibrated?       engine.context_cost.{hardware_class, model_id, weights_id,
                              transfer_source, prefill_source} from the same "server_start"
                              record. prefill_source/transfer_source == "generic-default"
                              means the compiled table had NO row for this (card, model,
                              weights) triple and the generic model priced it. That is the
                              difference between a measurement and a fallback, and it is
                              reported, not smoothed over. Cross-checked against the literal
                              rows compiled into src/runtime/engine/context_cost_defaults.cpp.

WHAT IT WILL NOT DO
-------------------
It will not produce a number for a card it did not run on. Rows it cannot measure are printed
as "unmeasurable here", not filled from the generic model. And it never sources a speed figure
from arch-sim: src/core/arch_sim.h:59-70 is explicit that "No performance claim of any kind may
be sourced from a simulated run", so a V100-simulated speed cannot exist in this instrument's
output by construction.

MODES
-----
  report   default. Everything it can do with what is on disk.
  --run    additionally launch the serve binary, drive it over HTTP, and measure.
           Needs the GPU; take the GPU lock first (see --print-lock-hint).
  --matrix print the per-card matrix skeleton and which rows are calibratable here.
  --notes  print the "what the ratio means / does not mean" note.

USAGE
-----
  python3 tools/roofline/serve_roofline.py report --artifact out/model.ninfer --ctx 8192
  python3 tools/roofline/serve_roofline.py report --artifact out/model.ninfer --ctx 8192 \\
          --from-log /path/to/requests.jsonl --serve-stdout /path/to/serve_stdout.txt
  python3 tools/roofline/serve_roofline.py report --artifact out/model.ninfer --ctx 8192 --run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------------------------
# Fixed provenance. These are facts about the repository, not choices this program makes.
# --------------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The measured pure-read ceiling and the two files that carry it. Change neither silently:
#: if --bandwidth-gbs is passed, the override and the original are both printed.
DEFAULT_BANDWIDTH_GBS = 1674.5
BANDWIDTH_PROVENANCE = (
    "measured pure-read ceiling 1674.5 GB/s from tools/hbm_bandwidth_probe.cu, "
    "cited at bench/README.md:232"
)
#: bench/README.md:229 also records a fixed RTX 5090 DRAM reference of 1792 GB/s. It is a
#: datasheet-style reference, not this box's measurement, so it is reported but never defaulted to.
DATASHEET_5090_GBS = 1792.0

#: The roofline definition, quoted so nobody has to take this program's word for it.
ROOFLINE_DEFINITION_SOURCE = "tools/archkit/qpn_port/placement_planner.py:31-38"

#: The single compiled cost-model row. Literal, from
#: src/runtime/engine/context_cost_defaults.cpp:51. Used only to report what the table covers.
COMPILED_HARDWARE_CLASS_ROW = "nvidia-geforce-rtx-5090-sm120"

NAN = float("nan")


def sha256_16(path: Path) -> str:
    """First 16 hex characters of a file's sha256 — the project's binary identity convention."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def human_bytes(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


# --------------------------------------------------------------------------------------------
# Card identity. This mirrors runtime::context_cost_hardware_class (context_cost.cpp:424-438)
# so the instrument can say which hardware_class a card WOULD need, before any run.
# --------------------------------------------------------------------------------------------


def hardware_class_for(gpu_name: str, major: int, minor: int) -> str:
    """Slug a GPU name the way the engine does, then append -sm<major><minor>.

    Deliberately a transcription of context_cost.cpp:424-438 (alnum lowercased, every other
    run of characters collapsed to a single '-', 'nvidia-' prefix forced, sm suffix appended).
    It is only used to predict/complain, never to substitute for what the engine reports.
    """
    slug: list[str] = []
    for character in gpu_name:
        if character.isalnum():
            slug.append(character.lower())
        elif slug and slug[-1] != "-":
            slug.append("-")
    text = "".join(slug).rstrip("-")
    if not text:
        text = "gpu"
    if text != "nvidia" and not text.startswith("nvidia-"):
        text = "nvidia-" + text
    return f"{text}-sm{major}{minor}"


@dataclass
class Card:
    """One (card, driver) identity, from whichever source was available."""

    name: str
    major: int = 0
    minor: int = 0
    total_memory_bytes: int = 0
    provenance: str = "unknown"

    @property
    def hardware_class(self) -> str:
        return hardware_class_for(self.name, self.major, self.minor)

    @property
    def sm(self) -> str:
        return f"sm_{self.major}{self.minor}"

    @property
    def covered_by_compiled_row(self) -> bool:
        return self.hardware_class == COMPILED_HARDWARE_CLASS_ROW


# --------------------------------------------------------------------------------------------
# Artifact byte accounting. The reader is the repository's own; this program does not
# re-implement the container format, because a second definition of a format is a liability.
# --------------------------------------------------------------------------------------------


@dataclass
class ArtifactBytes:
    path: Path
    file_bytes: int
    payload_offset: int
    payload_bytes: int          # sum of object bytes == what the loader can actually read
    tensor_bytes: int
    resource_bytes: int
    tensor_count: int
    resource_count: int
    model_id: str
    weights_id: str
    formats: dict[str, int] = field(default_factory=dict)
    reader: str = "tools/artifact/container.py"


def read_artifact_bytes(path: Path) -> ArtifactBytes:
    if not path.is_file():
        raise SystemExit(f"artifact not found: {path}")
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from tools.artifact.container import Artifact, ResourceObject, TensorObject
    except Exception as error:  # pragma: no cover - environment failure, reported not guessed
        raise SystemExit(
            f"cannot import the repository's artifact reader (tools/artifact/container.py) "
            f"from {REPO_ROOT}: {error!r}. This program refuses to parse .ninfer itself: a "
            f"second definition of the container format is exactly the drift this tree avoids."
        )
    with Artifact.open(path) as artifact:
        tensors = [obj for obj in artifact.objects if isinstance(obj, TensorObject)]
        resources = [obj for obj in artifact.objects if isinstance(obj, ResourceObject)]
        tensor_bytes = sum(obj.bytes for obj in tensors)
        resource_bytes = sum(obj.bytes for obj in resources)
        formats: dict[str, int] = {}
        for obj in tensors:
            formats[obj.format] = formats.get(obj.format, 0) + obj.bytes
        return ArtifactBytes(
            path=Path(path),
            file_bytes=artifact.file_bytes,
            payload_offset=artifact.payload_offset,
            payload_bytes=tensor_bytes + resource_bytes,
            tensor_bytes=tensor_bytes,
            resource_bytes=resource_bytes,
            tensor_count=len(tensors),
            resource_count=len(resources),
            model_id=artifact.identity.model_id,
            weights_id=artifact.identity.weights_id,
            formats=dict(sorted(formats.items(), key=lambda item: -item[1])),
        )


# --------------------------------------------------------------------------------------------
# Serve record parsing: stdout capacity line + request-log JSONL.
# --------------------------------------------------------------------------------------------

KV_CAPACITY_RE = re.compile(r"KV capacity\s+(\S+)\s+resolved=(\d+)\s+tokens")
COST_PROFILE_RE = re.compile(
    r"context-cost-transfer=(\S+)\s+context-cost-prefill=(\S+)\s+cost-profile=(\S+)"
)


@dataclass
class ServeRun:
    """One serve PROCESS's worth of records. A JSONL file may hold several.

    This distinction is not cosmetic. --request-log-jsonl appends, so a directory reused across
    runs holds more than one server_start. Charging one run's bytes against another run's
    timings produces a ratio that is real in both halves and false as a whole. Measured on this
    box: two runs of the SAME binary, SAME artifact, SAME argv, two minutes apart, decoded
    44.7 and 22.0 tok/s. Mixing them would have averaged a 2x-apart pair into one headline.
    """

    instance_id: str
    index: int
    config: dict[str, Any] = field(default_factory=dict)
    requests: list[dict[str, Any]] = field(default_factory=list)
    cost_profile_line: bool = False

    @property
    def kv_bytes_per_token(self) -> float:
        capacity = self.config.get("kv_capacity", 0)
        payload = self.config.get("kv_payload_bytes", 0)
        if not capacity or not payload:
            return NAN
        return payload / float(capacity)


@dataclass
class ServeRecord:
    """The engine-side facts of one serve log, however they were obtained."""

    source: str                                   # "request-log" | "serve-stdout" | "live"
    provenance: str = ""
    argv: list[str] = field(default_factory=list)
    card: Card | None = None
    model_id: str = ""
    weights_id: str = ""
    target: str = ""
    hardware_class: str = ""
    transfer_source: str = ""
    prefill_source: str = ""
    kv_capacity: int = 0
    kv_capacity_mode: str = ""
    kv_payload_bytes: int = 0
    host_to_device_bytes: int = 0
    artifact_bytes_read: int = 0
    artifact_size_bytes: int = 0
    tensor_count: int = 0
    load_seconds: float = NAN
    max_context: int = 0
    kv_cache: str = ""
    requests: list[dict[str, Any]] = field(default_factory=list)
    runs: list[ServeRun] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def kv_bytes_per_token(self) -> float:
        if self.kv_capacity <= 0 or self.kv_payload_bytes <= 0:
            return NAN
        return self.kv_payload_bytes / float(self.kv_capacity)


def _coerce_card(environment: dict[str, Any] | None, provenance: str) -> Card | None:
    if not environment:
        return None
    return Card(
        name=str(environment.get("gpu_name", "?")),
        major=int(environment.get("compute_capability_major", 0)),
        minor=int(environment.get("compute_capability_minor", 0)),
        total_memory_bytes=int(environment.get("total_device_memory_bytes", 0)),
        provenance=provenance,
    )


def parse_request_log(path: Path) -> ServeRecord:
    """Read a ninfer-serve --request-log-jsonl file. No GPU, no engine, no guessing.

    Keeps every server_start apart, so a file holding several runs can never be read as one.
    """
    record = ServeRecord(source="request-log", provenance=str(path))
    current: ServeRun | None = None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                record.notes.append("skipped one unparseable JSONL line")
                continue
            kind = event.get("event")
            if kind == "server_start":
                current = ServeRun(
                    instance_id=str(event.get("server_instance_id", "")),
                    index=len(record.runs),
                )
                record.runs.append(current)
                if not record.argv:
                    record.argv = [str(item) for item in event.get("argv", [])]
                if record.card is None:
                    record.card = _coerce_card(event.get("environment"), "request-log server_start")
                artifact = event.get("artifact") or {}
                engine = event.get("engine") or {}
                memory = event.get("memory") or {}
                cost = engine.get("context_cost") or {}
                current.config = {
                    "target": str(artifact.get("target", "")),
                    "weights_id": str(artifact.get("weights_id", "")),
                    "host_to_device_bytes": int(artifact.get("host_to_device_bytes", 0)),
                    "artifact_bytes_read": int(artifact.get("bytes_read", 0)),
                    "artifact_size_bytes": int(artifact.get("size_bytes", 0)),
                    "tensor_count": int(artifact.get("tensor_count", 0)),
                    "load_seconds": float(artifact.get("load_seconds", NAN)),
                    "kv_capacity": int(engine.get("kv_capacity", 0)),
                    "kv_capacity_mode": str(engine.get("kv_capacity_mode", "")),
                    "kv_cache": str(engine.get("kv_cache", "")),
                    "max_context": int(engine.get("max_context", 0)),
                    "kv_payload_bytes": int(memory.get("kv_payload_bytes", 0)),
                    "hardware_class": str(cost.get("hardware_class", "")),
                    "transfer_source": str(cost.get("transfer_source", "")),
                    "prefill_source": str(cost.get("prefill_source", "")),
                    "model_id": str(cost.get("model_id", "")),
                    "cost_weights_id": str(cost.get("weights_id", "")),
                }
            elif kind == "request_done":
                if current is None:
                    record.notes.append("a request_done appeared before any server_start")
                    continue
                result = event.get("result") or {}
                timing = event.get("timings_seconds") or {}
                engine_timing = event.get("engine_timing") or {}
                decode_block = engine_timing.get("decode") or {}
                current.requests.append(
                    {
                        "prompt_tokens": int(result.get("prompt_tokens", 0)),
                        "completion_tokens": int(result.get("completion_tokens", 0)),
                        "prefill_seconds": float(timing.get("prefill", NAN)),
                        "decode_seconds": float(timing.get("decode", NAN)),
                        "ttft_seconds": float(timing.get("ttft", NAN)),
                        "total_seconds": float(timing.get("total", NAN)),
                        "decode_rounds": int(decode_block.get("rounds", 0)),
                        "finish_reason": str(result.get("finish_reason", "")),
                        "prefix_cache_hit_tokens": int(
                            result.get("prefix_cache_hit_tokens", 0)
                        ),
                    }
                )
    if not record.runs:
        raise SystemExit(f"no server_start record in {path}: not a ninfer-serve request log")
    if len(record.runs) > 1:
        record.notes.append(
            f"this file holds {len(record.runs)} separate serve runs (--request-log-jsonl "
            f"appends). Only one run's bytes may be charged against its own timings; the "
            f"selection is reported below."
        )
    return record


def admit_run(record: ServeRecord, run: ServeRun, min_decode_tokens: int) -> ServeRecord:
    """Make `run` the run this record reports: its config, its requests, nothing else's."""
    record.requests = run.requests
    config = run.config
    record.target = config.get("target", "")
    record.weights_id = config.get("weights_id", "")
    record.host_to_device_bytes = config.get("host_to_device_bytes", 0)
    record.artifact_bytes_read = config.get("artifact_bytes_read", 0)
    record.artifact_size_bytes = config.get("artifact_size_bytes", 0)
    record.tensor_count = config.get("tensor_count", 0)
    record.load_seconds = config.get("load_seconds", NAN)
    record.kv_capacity = config.get("kv_capacity", 0)
    record.kv_capacity_mode = config.get("kv_capacity_mode", "")
    record.kv_cache = config.get("kv_cache", "")
    record.max_context = config.get("max_context", 0)
    record.kv_payload_bytes = config.get("kv_payload_bytes", 0)
    record.hardware_class = config.get("hardware_class", "")
    record.transfer_source = config.get("transfer_source", "")
    record.prefill_source = config.get("prefill_source", "")
    record.model_id = config.get("model_id", "")
    if config.get("cost_weights_id"):
        record.weights_id = config["cost_weights_id"]
    return record


def usable_measurements(run: ServeRun, min_decode_tokens: int) -> list[tuple[int, float, dict[str, Any]]]:
    out: list[tuple[int, float, dict[str, Any]]] = []
    for request in run.requests:
        rate = measured_decode_tps(request)
        if rate != rate:
            continue
        if request.get("completion_tokens", 0) - 1 < min_decode_tokens:
            continue
        out.append((request["prompt_tokens"], rate, request))
    return out


def select_run(record: ServeRecord, min_decode_tokens: int) -> ServeRun:
    """Pick the run to report: the one with the most usable measurements, latest breaking ties.

    Not the first, and not the union. A run with no usable measurement is still a candidate
    only if no run has one, in which case the latest run is described and its short-generation
    rates are printed as excluded.
    """
    ranked = sorted(
        record.runs,
        key=lambda run: (len(usable_measurements(run, min_decode_tokens)), run.index),
        reverse=True,
    )
    return ranked[0]


def parse_serve_stdout(path: Path, record: ServeRecord) -> ServeRecord:
    """Fill in what serve stdout carries and the JSONL does not: the resolved KV capacity."""
    text = path.read_text(encoding="utf-8", errors="replace")
    capacity = KV_CAPACITY_RE.search(text)
    if capacity:
        record.kv_capacity_mode = record.kv_capacity_mode or capacity.group(1)
        if not record.kv_capacity:
            record.kv_capacity = int(capacity.group(2))
    profile = COST_PROFILE_RE.search(text)
    if profile:
        transfer, prefill, triple = profile.groups()
        record.transfer_source = record.transfer_source or transfer
        record.prefill_source = record.prefill_source or prefill
        parts = triple.split("/")
        if len(parts) == 3:
            record.hardware_class = record.hardware_class or parts[0]
            record.model_id = record.model_id or parts[1]
            record.weights_id = record.weights_id or parts[2]
    if not record.kv_capacity:
        record.notes.append(
            f"no 'KV capacity ... resolved=N tokens' line in {path}; "
            "kv bytes/token cannot be formed from this source alone"
        )
    return record


# --------------------------------------------------------------------------------------------
# The roofline itself.
# --------------------------------------------------------------------------------------------


@dataclass
class Ratio:
    card: Card | None
    ctx: int
    weight_bytes: float
    weight_source: str
    kv_bytes_per_token: float
    kv_source: str
    kv_bytes_at_ctx: float
    bandwidth_gbs: float
    bandwidth_source: str

    @property
    def bytes_per_token(self) -> float:
        return self.weight_bytes + self.kv_bytes_at_ctx

    @property
    def theory_seconds(self) -> float:
        if self.bandwidth_gbs <= 0 or self.bytes_per_token <= 0:
            return NAN
        return self.bytes_per_token / (self.bandwidth_gbs * 1e9)

    @property
    def theory_seconds_per_token_from_weights_only(self) -> float:
        if self.bandwidth_gbs <= 0:
            return NAN
        return self.weight_bytes / (self.bandwidth_gbs * 1e9)

    @property
    def theory_tokens_per_second(self) -> float:
        seconds = self.theory_seconds
        return NAN if not seconds or seconds != seconds or seconds <= 0 else 1.0 / seconds

    def utilization(self, measured_tokens_per_second: float) -> float:
        theory = self.theory_tokens_per_second
        if theory != theory or theory <= 0:
            return NAN
        return measured_tokens_per_second / theory

    def with_ctx(self, ctx: float) -> "Ratio":
        """Same card, same weights, same bandwidth -- a different length of KV read.

        The theory MUST be formed at the context that was actually read, not at the context
        that was requested. --ctx is a target for a run and a filter for a log; the KV term is
        a function of the real attended length, and using the requested number instead is the
        kind of quiet substitution this instrument exists to remove.
        """
        return Ratio(
            card=self.card,
            ctx=ctx,
            weight_bytes=self.weight_bytes,
            weight_source=self.weight_source,
            kv_bytes_per_token=self.kv_bytes_per_token,
            kv_source=self.kv_source,
            kv_bytes_at_ctx=(
                NAN if self.kv_bytes_per_token != self.kv_bytes_per_token
                else self.kv_bytes_per_token * ctx
            ),
            bandwidth_gbs=self.bandwidth_gbs,
            bandwidth_source=self.bandwidth_source,
        )


def attended_length(request: dict[str, Any]) -> float:
    """The mean KV length a decode token actually attended, for this request.

    Decoding N tokens after a prompt of P attends P, P+1, ... P+N-1, whose mean is
    P + (N-1)/2. Using P alone would understate the read by half the generation, and using
    the capacity would overstate it wildly; this is the honest middle and it is stated.
    """
    prompt = float(request.get("prompt_tokens", 0))
    decode_tokens = max(0, int(request.get("completion_tokens", 0)) - 1)
    return prompt + max(0, decode_tokens - 1) / 2.0


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    """Ordinary least squares y = intercept + slope*x, plus R^2. None if underdetermined.

    The ATTN-COST-DECOMPOSITION report is the standard this imitates: a fit with its R^2 and
    the range over which it holds, not a single point. Same shape here: ms/round against the
    context actually read.
    """
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r_squared = NAN if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return intercept, slope, r_squared


def measured_decode_tps(request: dict[str, Any]) -> float:
    """The serve binary's own decode rate: (gen - 1) tokens over its own decode seconds."""
    tokens = request.get("completion_tokens", 0)
    seconds = request.get("decode_seconds", NAN)
    if tokens is None or tokens <= 1 or seconds != seconds or seconds <= 0:
        return NAN
    return (tokens - 1) / seconds


# --------------------------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------------------------


def print_card_block(card: Card, record: ServeRecord, artifact: ArtifactBytes) -> None:
    print("card")
    print(f"  gpu_name                : {card.name}")
    print(f"  compute capability      : {card.sm}   (major={card.major} minor={card.minor})")
    print(f"  total device memory     : {human_bytes(card.total_memory_bytes)}")
    print(f"  identity source         : {card.provenance}")
    print(f"  hardware_class          : {card.hardware_class}")
    row_state = "PRESENT" if card.covered_by_compiled_row else "ABSENT"
    print(
        f"  compiled cost-model row : {row_state}"
        f"   (compiled table lists exactly one: {COMPILED_HARDWARE_CLASS_ROW},"
        f" src/runtime/engine/context_cost_defaults.cpp:51)"
    )
    if record.transfer_source or record.prefill_source:
        calibrated = (
            record.transfer_source not in ("", "generic-default")
            or record.prefill_source not in ("", "generic-default")
        )
        verdict = "calibrated row used" if calibrated else "GENERIC FALLBACK"
        print(
            f"  engine-reported pricing : transfer_source={record.transfer_source or '?'} "
            f"prefill_source={record.prefill_source or '?'}  -> {verdict}"
        )
        print(
            "                            engine model_id/weights_id: "
            f"{record.model_id or '?'} / {record.weights_id or '?'}"
        )
    else:
        print(
            "  engine-reported pricing : unavailable from this source"
            " (no cost-profile line and no context_cost block)"
        )
    print(f"  artifact identity       : {artifact.model_id} / {artifact.weights_id}")


def print_bytes_block(artifact: ArtifactBytes, record: ServeRecord | None, ratio: Ratio) -> None:
    print("bytes")
    print(
        f"  artifact file           : {human_bytes(artifact.file_bytes)}"
        f"   ({artifact.file_bytes} B)  [{artifact.path.name}]"
    )
    print(
        f"  artifact objects        : {human_bytes(artifact.payload_bytes)}"
        f"   = {artifact.tensor_count} tensors ({human_bytes(artifact.tensor_bytes)})"
        f" + {artifact.resource_count} resources ({human_bytes(artifact.resource_bytes)})"
    )
    print(f"  artifact reader         : {artifact.reader}")
    if artifact.formats:
        top = ", ".join(
            f"{name}={human_bytes(size)}" for name, size in list(artifact.formats.items())[:4]
        )
        print(f"  tensor bytes by format  : {top}")
    if record is not None:
        if record.host_to_device_bytes:
            print(
                f"  engine weight H2D       : {human_bytes(record.host_to_device_bytes)}"
                f"   ({record.host_to_device_bytes} B)"
                f"  load={record.load_seconds:.2f} s"
                f"  tensors={record.tensor_count}"
            )
            delta = record.host_to_device_bytes - artifact.tensor_bytes
            if artifact.tensor_bytes and abs(delta) / artifact.tensor_bytes > 0.01:
                print(
                    f"    note: H2D differs from artifact tensor bytes by {delta} B "
                    f"({100.0 * delta / artifact.tensor_bytes:+.2f}%); "
                    "H2D is what the card actually holds, artifact bytes are what the file stores"
                )
        if record.artifact_bytes_read:
            print(
                f"  engine file bytes read  : {human_bytes(record.artifact_bytes_read)}"
                f"   ({record.artifact_bytes_read} B)"
            )
        if record.kv_capacity:
            print(
                f"  KV capacity             : {record.kv_capacity} tokens"
                f"  mode={record.kv_capacity_mode or '?'}  dtype={record.kv_cache or '?'}"
            )
        if record.kv_payload_bytes:
            print(
                f"  KV payload @ capacity   : {human_bytes(record.kv_payload_bytes)}"
                f"   ({record.kv_payload_bytes} B)"
            )
    print(f"  weight_bytes used       : {ratio.weight_bytes:.0f} B   source: {ratio.weight_source}")
    if ratio.kv_bytes_per_token == ratio.kv_bytes_per_token:
        print(
            f"  kv bytes/token          : {ratio.kv_bytes_per_token:.4f} B   "
            f"source: {ratio.kv_source}"
        )
        print(
            "  kv bytes at a ctx       : computed per request at that request's own attended "
            "length (see the roofline table)"
        )
    else:
        print(f"  kv bytes/token          : NOT MEASURED   blocker: {ratio.kv_source}")


def print_roofline_block(ratio: Ratio, measured: list[tuple[int, float, dict[str, Any]]]) -> None:
    print("roofline  (definition: %s)" % ROOFLINE_DEFINITION_SOURCE)
    print(
        f"  bandwidth               : {ratio.bandwidth_gbs:.1f} GB/s   source: {ratio.bandwidth_source}"
    )
    print(
        f"  weight_bytes            : {ratio.weight_bytes:.0f} B   source: {ratio.weight_source}"
    )
    if ratio.kv_bytes_per_token == ratio.kv_bytes_per_token:
        print(
            f"  kv bytes/token          : {ratio.kv_bytes_per_token:.4f} B   "
            f"source: {ratio.kv_source}"
        )
    else:
        print(f"  kv bytes/token          : NOT MEASURED   blocker: {ratio.kv_source}")
    print(
        "  theory is formed PER REQUEST at the context that request actually read "
        "(prompt + (decode-1)/2),"
    )
    print("  not at the value passed to --ctx; --ctx is a target for --run and a filter for --from-log.")
    print()
    if not measured:
        print("  measured                : NOT MEASURED   (see blockers below)")
        print("  ratio  measured/theory  : NOT MEASURED")
        return
    print(
        "  measured (serve request_done: (completion_tokens-1)/decode_seconds)"
    )
    print(
        f"    {'ctx_read':>9} {'prompt':>7} {'gen':>5} {'dn_tok':>7} {'rounds':>7} "
        f"{'decode_s':>9} {'tok/s':>8} {'theory':>8} {'ratio':>7}"
    )
    rows: list[tuple[float, float, float, dict[str, Any]]] = []
    ratios: list[float] = []
    for _, tps, request in measured:
        ctx_read = attended_length(request)
        per_request = ratio.with_ctx(ctx_read)
        util = per_request.utilization(tps)
        rows.append((ctx_read, tps, per_request.theory_tokens_per_second, request))
        if util == util:
            ratios.append(util)
        print(
            f"    {ctx_read:>9.1f} {request.get('prompt_tokens', 0):>7} "
            f"{request.get('completion_tokens', 0):>5} "
            f"{max(0, request.get('completion_tokens', 0) - 1):>7} "
            f"{request.get('decode_rounds', 0):>7} {request.get('decode_seconds', NAN):>9.4f} "
            f"{tps:>8.2f} {per_request.theory_tokens_per_second:>8.2f} "
            + (f"{util:>7.3f}" if util == util else f"{'n/a':>7}")
        )
    if ratios:
        print()
        print(
            f"  ratio spread            : min={min(ratios):.3f}  max={max(ratios):.3f}  "
            f"mean={sum(ratios) / len(ratios):.3f}   (measured/theory, per request)"
        )
        print(
            f"  decode tokens per rep   : {sorted({max(0, r.get('completion_tokens', 0) - 1) for _, _, r in measured})}"
            "   (the guard's currency: a ratio without its decode-token count is not a ratio)"
        )
        print(
            f"  ctx range covered       : {min(r[0] for r in rows):.0f} .. {max(r[0] for r in rows):.0f} tokens"
        )

    # The fit. Only when the sweep actually varied the context; a fit over one context is a
    # point pretending to be a line, and printing an R^2 for it would be theatre.
    xs = [row[0] for row in rows]
    per_round_ms: list[float] = []
    kept_x: list[float] = []
    for ctx_read, _, _, request in rows:
        rounds = request.get("decode_rounds", 0)
        seconds = request.get("decode_seconds", NAN)
        if rounds and seconds == seconds and seconds > 0:
            per_round_ms.append(seconds / rounds * 1e3)
            kept_x.append(ctx_read)
    if len({round(x) for x in kept_x}) >= 3:
        fit = linear_fit(kept_x, per_round_ms)
        if fit is not None:
            intercept, slope, r_squared = fit
            residuals = [
                y - (intercept + slope * x) for x, y in zip(kept_x, per_round_ms)
            ]
            print()
            print("  per-round decomposition (the ATTN-COST-DECOMPOSITION shape)")
            print(
                f"    ms/round = {intercept:.4f} + {slope:.6e} x ctx_read"
                f"   R^2 = {r_squared:.4f}   N = {len(kept_x)}"
            )
            print(
                f"    residuals (ms): min={min(residuals):+.4f} max={max(residuals):+.4f} "
                f"rms={(sum(r * r for r in residuals) / len(residuals)) ** 0.5:.4f}"
            )
            if slope > 0 and intercept > 0:
                crossing = intercept / slope
                print(
                    f"    the byte term overtakes the fixed term at ctx_read ~ {crossing:,.0f} tokens;"
                )
                print(
                    f"    below that the fixed term dominates and the ratio is NOT a bandwidth"
                    f" statement"
                )
            print(
                f"    holds over ctx_read {min(kept_x):.0f}..{max(kept_x):.0f} only; "
                f"extrapolation outside the swept range is not claimed"
            )
    elif len(kept_x) >= 1:
        print()
        print(
            f"  per-round decomposition : NOT FITTED — needs >=3 distinct contexts, have "
            f"{len({round(x) for x in kept_x})}. A fit over one context is a point pretending"
        )
        print("                            to be a line. Sweep --ctx with commas inside one --run.")


def print_validity_note() -> None:
    print(
        """
note — what the ratio means, and what it does not
------------------------------------------------
MEANS. For batch-1 decode of a dense-ish transformer, one token costs one pass over the
resident weights plus one pass over the KV the token attends to. Nothing else scales with the
token count, so the cost is bytes, and the ceiling is bytes/second. The ratio is therefore
"fraction of achievable HBM read bandwidth actually spent on the bytes this token needs", and
1.0 is the point where the model is bandwidth-saturated. It is an upper bound on any further
speedup from kernel work, not a promise of one.

DOES NOT MEAN. Four places it is invalid, all of them real here:
  * PREFILL. Prefill is compute-bound and the roofline's own second line applies
    (t = flops/compute + kv/bw). A prefill tok/s divided by this theory number is meaningless;
    prefill rates are reported but never ratioed.
  * BATCH > 1. At batch N the weights are read once for N tokens, so the per-token byte cost
    falls roughly as 1/N while compute does not. Measured at max_concurrency=1 only.
  * SPECULATION. With MTP/DFlash the token count is inflated by accepted draft tokens and the
    per-round byte cost amortizes differently. This instrument forces the non-speculative
    reading (default backend) and says so; ratios with a draft head enabled are a different
    quantity and are not comparable with these.
  * VIRTUAL/QUANTIZED WEIGHTS THAT ARE NOT RESIDENT. If a weight-offload budget is set, part of
    weight_bytes is not read from HBM per token. The default budget is 0 (a no-op,
    src/targets/registry.cpp), which is the case this reports.

RESIDUALS, NOT A SINGLE POINT. A ratio at one context length is not a model. The fit this
imitates — ATTN-COST-DECOMPOSITION, ms/round = 17.22 + 1.116e-5 * visible_keys (R^2 = 0.942),
85.8-99.4 % key-count-independent — is a model with a residual and a range. To get that here,
run this at >= 3 context lengths and check that theory tracks measured with roughly constant
ratio; a ratio that drifts with ctx means the fixed per-round term (host exposure, launch,
sampling) dominates the byte term, and the roofline is describing less of the runtime than the
number suggests.

WHAT THE RECORDED RUNS ON THIS BOX ALREADY SHOW (this is why the guard below exists). Across 20
recorded serve request logs for the SAME artifact and the SAME kv bytes/token (18048 B), the
engines' own decode rate ranges from 11.8 to 70.7 tok/s. It is not noise: it tracks the
generation length and the run configuration. Every value above ~55 tok/s comes from a request
with gen=2, i.e. ONE decode round, so it is a single round's latency with none of the fixed
per-round cost amortized — and at ctx~130 that happens to read as ratio 0.75-0.91, which would
be a spectacular number and a false one. Requests with fewer than --min-decode-tokens decode
rounds are therefore excluded and LISTED, never averaged in. Per-round decode exposure across
these runs is 21-46 ms/round against a ~12.9 ms/token byte bound, so at short context most of
the round is not the byte term, and the honest statement there is "dominated by the fixed term",
not "utilization".

RANGE. Trustworthy where the byte term dominates: long context, batch 1, no speculation,
weights resident, and enough decode rounds that the fixed per-round cost is amortized. Outside
that, print the number with its caveat or print "not measured".
"""
    )


MATRIX_COLUMNS = (
    "card (hardware_class)",
    "present here",
    "compiled row",
    "artifact+ctx calibratable here",
)


def print_matrix(record: ServeRecord | None) -> None:
    """Per-card matrix skeleton: the rows it needs, and which are calibratable on this box."""
    here = record.card if record is not None else None
    print("per-card matrix skeleton")
    print(
        "  A row is 'calibratable here' only if the card is physically present, because both\n"
        "  halves of the ratio need measured quantities: bw_measured on that card and a real\n"
        "  serve run on that card. A row that cannot be measured is printed as unmeasurable,\n"
        "  never filled from the generic cost model.\n"
    )
    rows = [
        ("nvidia-geforce-rtx-5090-sm120", "the only compiled cost-model row"),
        ("nvidia-geforce-rtx-5090-d-sm120", "this box's card: 'RTX 5090 D' slugs with a '-d'"),
        ("nvidia-geforce-rtx-5090-laptop-sm120", "unmeasured card"),
        ("nvidia-b200-sm100", "unmeasured card (also the sm_100 arch_caps row this tree paid for)"),
        ("nvidia-h100-sm90", "unmeasured card"),
        ("nvidia-a100-sm80", "unmeasured card"),
        ("nvidia-v100-sm70", "unmeasured card; also the arch-sim target"),
    ]
    width = max(len(name) for name, _ in rows)
    print(f"  {'card (hardware_class)':<{width}}  {'present':>8}  {'compiled':>9}  {'calibratable':>13}")
    for name, note in rows:
        present = here is not None and here.hardware_class == name
        compiled = name == COMPILED_HARDWARE_CLASS_ROW
        print(
            f"  {name:<{width}}  {'YES' if present else 'no':>8}  "
            f"{'YES' if compiled else 'no':>9}  {'YES' if present else 'no':>13}"
            f"   # {note}"
        )
    print()
    print("  observations")
    if here is not None:
        print(
            f"    * The card present is {here.name}, whose hardware_class is\n"
            f"      '{here.hardware_class}'. The compiled table's single row is\n"
            f"      '{COMPILED_HARDWARE_CLASS_ROW}'. Row matches present card: "
            f"{'YES' if here.covered_by_compiled_row else 'NO'}"
            + ("" if here.covered_by_compiled_row else
               ".\n      So every engine cost number on this box is a generic-model number.")
        )
    else:
        print(
            "    * no card identity was available for this invocation, so 'present' is all no.\n"
            "      That is a fact about this invocation, not about the box."
        )
    print(
        "    * Exactly one of these rows is calibratable on this box. The rest need the card.\n"
        "      arch-sim cannot fill them: src/core/arch_sim.h:59-70 forbids sourcing any\n"
        "      performance number from a simulated run, so a 'V100 simulated' speed may not be\n"
        "      produced by this instrument at all."
    )
    print(
        "    * Columns the matrix needs per cell, none of which are speed numbers and all of\n"
        "      which are already obtainable: artifact identity (model_id/weights_id), KV config\n"
        "      (dtype tier table + resolved capacity), route selection and outcome\n"
        "      (src/core/kernel_route.h), and >=1 generated token (the verify_artifact shape)."
    )


def print_lock_hint(artifact: Path, ctx: int) -> None:
    print()
    print("to take a real measurement, in order:")
    print("  1. take the GPU lock fairly (the protocol is: source the lock script, then run).")
    print("     sh/_lock.sh is NOT present in this work tree; it was not found by this "
          "instrument's search either.")
    print("     /home/user/scratch/BACKUP-LOCK8/_lock.sh.v8-fifo-pre-probe documents the v8 "
          "interface:")
    print("       source <lock.sh> gpu [idle_wait_s] [lock_wait_s]   # fd 9 is then the lock")
    print("       exec 9>&-                                          # release")
    print("  2. then:")
    print(
        f"     python3 tools/roofline/serve_roofline.py report "
        f"--artifact {artifact} --ctx {ctx} --run"
    )
    print(
        "  3. an arm is only a speed number if the GPU was acquired cleanly. Record the\n"
        "     memory at acquisition time next to the ratio."
    )


# --------------------------------------------------------------------------------------------
# Live measurement over the serve binary (needs the GPU).
# --------------------------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"content-type": "application/json"}, method="GET" if payload is None else "POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


FILLER = (
    "In the following passage the same observation is repeated so that the context occupies "
    "the requested number of tokens. The measurement below depends only on the size of the "
    "context and the bytes the decoder must read per token. "
)


def run_selftest() -> int:
    """Known-answer tests for everything that does not need the GPU.

    The point is to be able to say which parts of this program are PROVEN and which are not.
    The ratio arithmetic, the unit handling, the hardware_class mapping and the decode-round
    guard are all checkable against hand-computed answers; the HTTP driving in --run is not
    checkable without the card and is therefore reported as unexercised rather than implied.
    """
    failures: list[str] = []

    def check(label: str, got: Any, want: Any, tol: float = 1e-9) -> None:
        ok = (
            abs(got - want) <= tol
            if isinstance(got, float) and isinstance(want, float)
            else got == want
        )
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got {got!r} want {want!r}")
        if not ok:
            failures.append(label)

    print("selftest — known-answer checks (no GPU, no engine, no artifact required)")
    print("  card identity slug (must mirror context_cost.cpp:424-438)")
    check(
        "hardware_class_for('NVIDIA GeForce RTX 5090 D', 12, 0)",
        hardware_class_for("NVIDIA GeForce RTX 5090 D", 12, 0),
        "nvidia-geforce-rtx-5090-d-sm120",
    )
    check(
        "hardware_class_for('NVIDIA GeForce RTX 5090', 12, 0) == the compiled row",
        hardware_class_for("NVIDIA GeForce RTX 5090", 12, 0),
        COMPILED_HARDWARE_CLASS_ROW,
    )
    check(
        "hardware_class_for('NVIDIA H100 80GB HBM3', 9, 0)",
        hardware_class_for("NVIDIA H100 80GB HBM3", 9, 0),
        "nvidia-h100-80gb-hbm3-sm90",
    )

    print("  ratio arithmetic against hand-computed values")
    # weights 21,468,047,724 B; kv 18,048 B/token at ctx 8192; bw 1674.5 GB/s
    card = Card(name="NVIDIA GeForce RTX 5090 D", major=12, minor=0)
    ratio = Ratio(
        card=card,
        ctx=8192,
        weight_bytes=21468047724.0,
        weight_source="selftest",
        kv_bytes_per_token=18048.0,
        kv_source="selftest",
        kv_bytes_at_ctx=18048.0 * 8192,
        bandwidth_gbs=1674.5,
        bandwidth_source="selftest",
    )
    check("bytes per token", ratio.bytes_per_token, 21615896940.0, tol=1.0)
    check("theory ms/token", ratio.theory_seconds * 1e3, 12.9089, tol=5e-4)
    check("theory tok/s", ratio.theory_tokens_per_second, 77.4661, tol=5e-3)
    check("utilization(44.694821813404964)", ratio.utilization(44.694821813404964), 0.576959, tol=1e-5)
    check("utilization == 1.0 at the limit", ratio.utilization(ratio.theory_tokens_per_second), 1.0)

    print("  decode-rate convention (serve's own: (gen-1)/decode_seconds)")
    check("gen=16 decode=0.75 -> 20 tok/s", measured_decode_tps({"completion_tokens": 16, "decode_seconds": 0.75}), 20.0)
    check("gen=1 -> no decode rate", measured_decode_tps({"completion_tokens": 1, "decode_seconds": 0.5}) != measured_decode_tps({"completion_tokens": 1, "decode_seconds": 0.5}), True)
    check("gen=0 -> no decode rate", measured_decode_tps({"completion_tokens": 0, "decode_seconds": 0.5}) != measured_decode_tps({"completion_tokens": 0, "decode_seconds": 0.5}), True)

    print("  the two numbers the theory must never quietly borrow from each other")
    check(
        "datasheet 1792 GB/s is NOT the default",
        DEFAULT_BANDWIDTH_GBS == 1674.5,
        True,
    )
    check("the roofline's provenance is a real path", ROOFLINE_DEFINITION_SOURCE.startswith("tools/"), True)

    print("  run separation (one JSONL may hold several serve runs, because it appends)")
    run_a = ServeRun(
        instance_id="A",
        index=0,
        requests=[{"completion_tokens": 24, "decode_seconds": 0.5, "prompt_tokens": 85,
                   "decode_rounds": 23}],
    )
    run_b = ServeRun(
        instance_id="B",
        index=1,
        requests=[{"completion_tokens": 2, "decode_seconds": 0.1, "prompt_tokens": 85,
                   "decode_rounds": 1}],
    )
    container = ServeRecord(source="selftest", runs=[run_a, run_b])
    check("a 1-round run is not usable", len(usable_measurements(run_b, 16)), 0)
    check("a 23-round run is usable", len(usable_measurements(run_a, 16)), 1)
    check("selection prefers the usable run", select_run(container, 16).instance_id, "A")
    run_b.requests = [
        {"completion_tokens": 24, "decode_seconds": 0.5, "prompt_tokens": 85, "decode_rounds": 23},
        {"completion_tokens": 24, "decode_seconds": 0.5, "prompt_tokens": 85, "decode_rounds": 23},
    ]
    check("latest wins when usable counts tie at the top", select_run(container, 16).instance_id, "B")
    check(
        "runs are never pooled",
        len(select_run(container, 16).requests),
        2,
    )

    print("  per-request context, and the fit (a single point is not a model)")
    check(
        "attended_length(prompt=100, gen=2) == 100 (one decode token)",
        attended_length({"prompt_tokens": 100, "completion_tokens": 2}),
        100.0,
    )
    check(
        "attended_length(prompt=100, gen=4) == 101 (3 decode tokens, mean +1)",
        attended_length({"prompt_tokens": 100, "completion_tokens": 4}),
        101.0,
    )
    repriced = ratio.with_ctx(100)
    check("with_ctx re-prices the KV term", repriced.kv_bytes_at_ctx, 18048.0 * 100, tol=1.0)
    check("with_ctx keeps the weights", repriced.weight_bytes, 21468047724.0)
    check("with_ctx keeps the bandwidth", repriced.bandwidth_gbs, 1674.5)
    exact = linear_fit([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
    check("linear_fit intercept on an exact line", exact[0], 1.0, tol=1e-12)
    check("linear_fit slope on an exact line", exact[1], 2.0, tol=1e-12)
    check("linear_fit R^2 on an exact line", exact[2], 1.0, tol=1e-12)
    check("linear_fit refuses 2 points", linear_fit([0.0, 1.0], [1.0, 2.0]), None)

    print("  what is NOT covered by this selftest")
    print("    * the --run path (launching ninfer-serve, driving it over HTTP, reading its own")
    print("      request log) has NOT been executed on hardware by this program's author. It is")
    print("      exercised only up to the point of process launch. Treat --run as unproven until")
    print("      someone with the GPU lock runs it once and reads the output.")
    print(f"\n  {len(failures)} failure(s)" + (f": {failures}" if failures else ""))
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="report",
        choices=("report", "matrix", "notes"),
        help="report (default) | matrix | notes",
    )
    parser.add_argument("--artifact", type=Path, required=False, help="the .ninfer artifact")
    parser.add_argument(
        "--ctx",
        default=None,
        help="context target in tokens. For --run: one value or a comma-separated sweep, e.g. "
        "1024,8192,32768 (3+ distinct values additionally yield the per-round fit and its R^2). "
        "For --from-log: a selection hint only; the theory always uses each request's own "
        "attended length.",
    )
    parser.add_argument("--binary", type=Path, default=Path("build/apps/ninfer-serve"),
                        help="the binary the user would actually serve with")
    parser.add_argument("--bandwidth-gbs", type=float, default=DEFAULT_BANDWIDTH_GBS)
    parser.add_argument("--from-log", type=Path, help="an existing serve request-log JSONL")
    parser.add_argument("--serve-stdout", type=Path, help="serve stdout (carries KV capacity)")
    parser.add_argument("--run", action="store_true", help="launch the serve binary and measure")
    parser.add_argument("--max-new", type=int, default=128, help="tokens to generate per rep")
    parser.add_argument(
        "--min-decode-tokens",
        type=int,
        default=16,
        help="a request with fewer decode rounds than this is not accepted as a decode "
        "measurement (a 1-round rate is not a decode rate)",
    )
    parser.add_argument("--max-measurements", type=int, default=8)
    parser.add_argument("--reps", type=int, default=2, help="how many requests to issue")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--matrix", action="store_true", help="print the per-card matrix skeleton")
    parser.add_argument("--notes", action="store_true", help="print the validity note")
    parser.add_argument("--selftest", action="store_true", help="known-answer checks (no GPU)")
    parser.add_argument("--json", action="store_true", help="also emit a machine-readable summary")
    args = parser.parse_args(argv)

    # --ctx is a sweep target for --run and a selection hint for --from-log. It is NOT the
    # context used to form the theory: that is each request's own attended length.
    ctx_targets: list[int] = []
    if args.ctx:
        if isinstance(args.ctx, int):
            ctx_targets = [args.ctx]
        else:
            for piece in str(args.ctx).replace(" ", "").split(","):
                if piece:
                    ctx_targets.append(int(piece))
    for value in ctx_targets:
        if value < 8:
            raise SystemExit(f"--ctx value {value} is too small to measure a decode")
    if args.run and not ctx_targets:
        ctx_targets = [8192]
    args.ctx_targets = ctx_targets

    if args.selftest:
        return run_selftest()

    if args.mode == "notes":
        args.notes = True
    elif args.mode == "matrix":
        args.matrix = True

    if args.notes:
        print_validity_note()
        if not args.artifact and not args.matrix:
            return 0

    artifact_bytes: ArtifactBytes | None = None
    if args.artifact:
        artifact_bytes = read_artifact_bytes(args.artifact)

    if args.matrix:
        record = None
        if args.from_log:
            record = parse_request_log(args.from_log)
        else:
            # The matrix must still be able to say which row is present, so ask nvidia-smi
            # directly. That is a query, not an allocation, and it is the only way the matrix
            # can be honest about a card the instrument is not allowed to run on.
            probe = live_card_probe()
            record = None if probe is None else ServeRecord(source="nvidia-smi")
            if probe is not None:
                record.card = probe
        print_matrix(record)
        if not args.artifact:
            return 0

    if artifact_bytes is None:
        print("nothing to report: pass --artifact (and see --matrix / --notes)", file=sys.stderr)
        return 2

    print("=" * 90)
    print("ROOFLINE — artifact x card, memory-bandwidth limit, serve path")
    print("=" * 90)
    print(f"artifact : {args.artifact}")
    print(f"ctx      : {args.ctx} tokens")
    print(f"binary   : {args.binary}   [the serve path; ninfer and ninfer-serve are separate]")
    print(f"repo     : {REPO_ROOT}")
    print()

    record: ServeRecord | None = None
    run_result: dict[str, Any] = {}

    if args.from_log:
        record = parse_request_log(args.from_log)
        chosen = select_run(record, args.min_decode_tokens)
        if len(record.runs) > 1:
            print(f"this log holds {len(record.runs)} serve runs; reporting run #"
                  f"{chosen.index} (id {chosen.instance_id}), which has "
                  f"{len(usable_measurements(chosen, args.min_decode_tokens))} usable "
                  f"measurement(s):")
            for run in record.runs:
                usable = usable_measurements(run, args.min_decode_tokens)
                rates = ", ".join(f"{rate:.2f}" for _, rate, _ in usable) or "none"
                print(
                    f"   run #{run.index} id={run.instance_id} "
                    f"requests={len(run.requests)} usable={len(usable)} tok/s=[{rates}]"
                    + ("  <== reported" if run.index == chosen.index else "")
                )
            print(
                "   The runs are NOT pooled: a JSONL appends, and one run's bytes must not be\n"
                "   charged against another run's timings."
            )
            print()
        record = admit_run(record, chosen, args.min_decode_tokens)
        if args.serve_stdout:
            record = parse_serve_stdout(args.serve_stdout, record)
        print(f"measurement source : RECORDED serve run, not reproduced by this invocation")
        print(f"                     request log: {args.from_log}")
        print(f"                     sha256(16) : {sha256_16(args.from_log)}")
        print(
            "                     the log's own argv identifies the binary, but the log does not\n"
            "                     record a binary hash; a recorded ratio is therefore a ratio for\n"
            "                     THAT build, which may not be the build on disk now."
        )
        print()
    elif args.run:
        record, run_result = run_measurement(args)
    else:
        if args.artifact:
            print("no --from-log and no --run: no measured half is available.")
            print()

    card = record.card if record is not None else live_card_probe()
    bandwidth_source = BANDWIDTH_PROVENANCE
    if args.bandwidth_gbs != DEFAULT_BANDWIDTH_GBS:
        bandwidth_source = (
            f"OVERRIDE by --bandwidth-gbs; the tree's measured figure is "
            f"{DEFAULT_BANDWIDTH_GBS} GB/s ({BANDWIDTH_PROVENANCE})"
        )

    if card is not None:
        print_card_block(card, record or ServeRecord(source="none"), artifact_bytes)
        print()

    if record is not None:
        weight_bytes = float(record.host_to_device_bytes or artifact_bytes.tensor_bytes)
        weight_source = (
            "engine artifact.host_to_device_bytes (resident weight bytes)"
            if record.host_to_device_bytes
            else "artifact tensor bytes (engine H2D unavailable in this source)"
        )
        kv_per_token = record.kv_bytes_per_token
        if kv_per_token == kv_per_token:
            kv_source = (
                f"engine kv_payload_bytes {record.kv_payload_bytes} B / kv_capacity "
                f"{record.kv_capacity} tokens"
            )
        else:
            kv_per_token = NAN
            kv_source = (
                "kv_payload_bytes and/or kv_capacity missing from this source "
                "(need a request log with memory.kv_payload_bytes and engine.kv_capacity, "
                "or a serve stdout with the 'KV capacity ... resolved=' line)"
            )
    else:
        weight_bytes = float(artifact_bytes.tensor_bytes)
        weight_source = "artifact tensor bytes (no serve record available)"
        kv_per_token = NAN
        kv_source = "no serve record available: geometry lives in the compiled contract, not in the artifact"

    # This Ratio is the parameter carrier: card, weights, bandwidth, kv/token. The KV term is
    # re-priced per request by with_ctx() at that request's own attended length, because the
    # value passed to --ctx is a target, not a measurement. The value below is only a default
    # so the object is well-formed.
    kv_at_ctx = (
        NAN
        if kv_per_token != kv_per_token
        else kv_per_token * (max(args.ctx_targets) if args.ctx_targets else 0)
    )
    ratio = Ratio(
        card=card,
        ctx=args.ctx,
        weight_bytes=weight_bytes,
        weight_source=weight_source,
        kv_bytes_per_token=kv_per_token,
        kv_source=kv_source,
        kv_bytes_at_ctx=0.0 if kv_at_ctx != kv_at_ctx else kv_at_ctx,
        bandwidth_gbs=args.bandwidth_gbs,
        bandwidth_source=bandwidth_source,
    )

    print_bytes_block(artifact_bytes, record, ratio)
    print()

    measured: list[tuple[int, float, dict[str, Any]]] = []
    rejected: list[tuple[int, float, dict[str, Any]]] = []
    if record is not None and record.requests:
        ordered = sorted(
            record.requests,
            key=lambda item: (
                # --ctx, when given for a log, is a selection hint: nearest first.
                0.0
                if not args.ctx_targets
                else min(abs(item["prompt_tokens"] - target) for target in args.ctx_targets)
            ),
        )
        for request in ordered:
            tps = measured_decode_tps(request)
            if tps != tps:
                continue
            # A decode rate computed from a handful of rounds is not a decode measurement: with
            # gen=2 there is ONE decode round, so the figure is a single round's latency with
            # none of the per-round fixed cost amortized, and it reads several times the honest
            # rate. That is a real effect measured on this box (recorded runs range 11.8 to
            # 70.7 tok/s for the SAME artifact and the same kv bytes/token, purely on generation
            # length and run configuration). Those points are excluded and listed, not averaged
            # into the answer.
            entry = (request["prompt_tokens"], tps, request)
            if request.get("completion_tokens", 0) - 1 < args.min_decode_tokens:
                rejected.append(entry)
            elif len(measured) < args.max_measurements:
                measured.append(entry)
    print_roofline_block(ratio, measured)
    if rejected:
        print()
        print(
            f"  excluded as NOT a decode measurement "
            f"(fewer than {args.min_decode_tokens} decode rounds):"
        )
        for prompt_tokens, tps, request in rejected[:6]:
            print(
                f"    prompt={prompt_tokens} gen={request.get('completion_tokens', 0)} "
                f"rounds={request.get('decode_rounds', 0)} -> {tps:.2f} tok/s "
                f"(one round, per-round fixed cost unamortized; would read as ratio "
                f"{ratio.utilization(tps):.3f} and it would be wrong)"
            )
        if len(rejected) > 6:
            print(f"    ... and {len(rejected) - 6} more")

    print()
    print("blockers / caveats")
    if record is None:
        print(
            "  * this invocation did not run the serve binary and was given no serve record, so\n"
            "    there is no measured half. Run with --from-log FILE or (holding the GPU) --run."
        )
    elif not args.run:
        print(
            "  * the measured half is from a RECORDED run. It is a real serve measurement on a\n"
            "    real card, but not one this line acquired the GPU for, and not necessarily the\n"
            "    binary on disk now. Do not quote it as this line's measurement."
        )
    if record is not None and record.notes:
        for note in record.notes:
            print(f"  * {note}")
    if card is not None and not card.covered_by_compiled_row:
        print(
            f"  * this card's hardware_class ({card.hardware_class}) has NO compiled cost-model\n"
            f"    row; the engine priced it with the generic model. Any engine cost number on\n"
            f"    this box is a fallback number, and must not be read as a measurement."
        )

    if args.json:
        # The per-request table is the answer; the summary below must not smuggle in the
        # template's --ctx. Each row is priced at its own attended length, and the headline is
        # the best row's own ratio, not a ratio formed against a context nobody read.
        rows_for_json: list[tuple[float, float, float, float]] = []
        for _, tps, request in measured:
            ctx_read = attended_length(request)
            per_request = ratio.with_ctx(ctx_read)
            util = per_request.utilization(tps)
            if util == util:
                rows_for_json.append((ctx_read, tps, per_request.theory_tokens_per_second, util))
        best_row = max(rows_for_json, key=lambda row: row[3]) if rows_for_json else None
        print()
        print("json")
        print(
            json.dumps(
                {
                    "artifact": str(args.artifact),
                    "artifact_sha16": sha256_16(args.artifact),
                    "ctx": str(args.ctx),
                    "card": None
                    if card is None
                    else {
                        "gpu_name": card.name,
                        "sm": card.sm,
                        "hardware_class": card.hardware_class,
                        "compiled_row": card.covered_by_compiled_row,
                    },
                    "engine_reported_pricing": {
                        "transfer_source": (record.transfer_source if record else ""),
                        "prefill_source": (record.prefill_source if record else ""),
                        "generic_fallback": bool(
                            record
                            and (
                                record.transfer_source in ("", "generic-default")
                                or record.prefill_source in ("", "generic-default")
                            )
                        ),
                    },
                    "weight_bytes": ratio.weight_bytes,
                    "kv_bytes_per_token": None
                    if ratio.kv_bytes_per_token != ratio.kv_bytes_per_token
                    else ratio.kv_bytes_per_token,
                    "bandwidth_gbs": ratio.bandwidth_gbs,
                    "bandwidth_source": ratio.bandwidth_source,
                    "rows": [
                        {
                            "ctx_read": ctx_read,
                            "measured_tokens_per_second": tps,
                            "theory_tokens_per_second": theory,
                            "ratio": util,
                        }
                        for ctx_read, tps, theory, util in rows_for_json
                    ],
                    "best_row": None
                    if best_row is None
                    else {
                        "ctx_read": best_row[0],
                        "measured_tokens_per_second": best_row[1],
                        "theory_tokens_per_second": best_row[2],
                        "ratio": best_row[3],
                    },
                    "ratio_band": None
                    if not rows_for_json
                    else [
                        min(row[3] for row in rows_for_json),
                        max(row[3] for row in rows_for_json),
                    ],
                    "measurement_source": "recorded-log"
                    if args.from_log
                    else ("live-run" if args.run else "none"),
                    "run": run_result,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    return 0


def live_card_probe() -> Card | None:
    """Read the card identity without touching the GPU's compute: nvidia-smi query only.

    This is a query, not an allocation. It is what makes the matrix able to say which row is
    present even when the instrument is not allowed to run anything.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    line = result.stdout.strip().splitlines()[0]
    parts = [piece.strip() for piece in line.split(",")]
    if len(parts) < 3:
        return None
    capability = parts[1]
    major, _, minor = capability.partition(".")
    memory = re.sub(r"[^0-9]", "", parts[2])
    return Card(
        name=parts[0],
        major=int(major or 0),
        minor=int(minor or 0),
        total_memory_bytes=int(memory or 0) * 1024 * 1024,
        provenance="nvidia-smi query (does not allocate the GPU)",
    )


def server_model_id(path: Path) -> str:
    """The model id the server expects in a request body.

    The server 404s a request whose "model" is not its public id, so a client that invents a
    label (or omits the field) gets Not Found on every call. Ask the running server what it
    calls itself instead of guessing: server_start carries server.public_model_id.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if event.get("event") == "server_start":
                    server = event.get("server") or {}
                    if server.get("public_model_id"):
                        return str(server["public_model_id"])
                    engine = event.get("engine") or {}
                    cost = engine.get("context_cost") or {}
                    return str(cost.get("model_id", ""))
    except Exception:
        pass
    return ""


def run_measurement(args: argparse.Namespace) -> tuple[ServeRecord, dict[str, Any]]:
    """Launch the serve binary, drive it over HTTP, and read the engine's own accounting."""
    binary: Path = args.binary
    if not binary.is_file():
        raise SystemExit(f"serve binary not found: {binary}")
    binary_sha = sha256_16(binary)
    print(f"binary sha16       : {binary_sha}   ({binary}, mtime "
          f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(binary.stat().st_mtime))})")
    engine = REPO_ROOT / "build/apps/ninfer"
    if engine.is_file():
        engine_sha = sha256_16(engine)
        print(
            f"     (for contrast, {engine} sha16 = {engine_sha}, mtime "
            f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(engine.stat().st_mtime))})"
        )
        # Only warn when the binary actually used IS the churning build-dir artifact. If the
        # caller pinned a copy (the project's own discipline when the build dir is relinking),
        # the pinned bytes are what was measured and an age comparison against a moving target
        # would be a false warning.
        if "build/apps" in str(binary):
            if engine.stat().st_mtime > binary.stat().st_mtime + 60:
                print(
                    "     WARNING: ninfer-serve is OLDER than ninfer. Any serve-path speed\n"
                    "     measured here describes the stale serve binary, not the current engine."
                )
        else:
            print(
                "     this run measured a PINNED COPY, so a relink of the build dir during the\n"
                "     arm cannot change what was measured; the copy's sha16 above is the identity."
            )
    ctxs = list(args.ctx_targets) or [8192]
    max_ctx = max(ctxs)
    port = args.port or free_port()
    log_dir = Path("/tmp/roofline")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    jsonl = log_dir / f"serve_{stamp}.jsonl"
    stdout_path = log_dir / f"serve_{stamp}.stdout.txt"
    argv = [
        str(binary),
        str(args.artifact),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-context",
        # Headroom for the generation, not just the prompt. With --kv-capacity auto the engine
        # resolves capacity to exactly max_context (observed: max_context 32768 -> capacity
        # 32768; 24576 -> 24576), so a prompt sized to max_context leaves no room for the
        # requested output tokens and the server answers 400. Two arms hit exactly that. The
        # sweep's top target must therefore sit inside the capacity, by the generation length.
        str(max(max_ctx + args.max_new + 64, 2048)),
        "--kv-capacity",
        "auto",
        "--max-concurrency",
        "1",
        "--request-log-jsonl",
        str(jsonl),
        "--greedy",
        "--default-max-tokens",
        str(args.max_new),
    ]
    print(f"binary sha16       : {binary_sha}")
    print(f"ctx sweep          : {ctxs}  (max-context={max(max_ctx, 2048)}, kv-capacity=auto)")
    print(f"argv               : {' '.join(argv)}")
    print("acquiring the GPU: this invocation does NOT take the lock itself; the caller must")
    print("hold it. Memory at acquisition must be reported with the ratio to be uncontaminated.")
    with open(stdout_path, "w", encoding="utf-8") as out:
        process = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.time() + 900
            while time.time() < deadline:
                if jsonl.is_file() and jsonl.stat().st_size > 0:
                    break
                if process.poll() is not None:
                    raise SystemExit(
                        f"serve exited rc={process.returncode} before starting; see {stdout_path}"
                    )
                time.sleep(2)
            else:
                raise SystemExit(f"serve did not start within 900 s; see {stdout_path}")
            base = f"http://127.0.0.1:{port}"
            for attempt in range(60):
                try:
                    http_json(base + "/health", None, 5.0)
                    break
                except Exception:
                    time.sleep(1)
            # The server names itself; a request whose "model" is not that id is answered 404.
            model_id = server_model_id(jsonl)
            if not model_id:
                raise SystemExit(
                    "could not read server.public_model_id from the serve log; refusing to "
                    "guess a model id (a wrong one makes every request 404)"
                )
            print(f"server model id    : {model_id}")
            count_token_failures = 0
            for target_ctx in ctxs:
                prompt_words, counted = _fit_prompt(base, target_ctx, model_id)
                if not counted:
                    count_token_failures += 1
                    if count_token_failures == 1:
                        print(
                            "  WARNING: /v1/messages/count_tokens did not answer; the prompt\n"
                            "  length is whatever the text happens to be. The ACTUAL context is\n"
                            "  still read back from request_done, so the ratio stays honest, but\n"
                            "  the sweep targets are then only approximate."
                        )
                for rep in range(args.reps):
                    body = {
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt_words}],
                        "max_tokens": args.max_new,
                        "stream": False,
                    }
                    started = time.time()
                    try:
                        http_json(base + "/v1/chat/completions", body, 1800.0)
                    except Exception as error:
                        # A failed rep is not a failed arm: the JSONL already holds every
                        # completed request, and --from-log can read whatever landed. Say so
                        # rather than dying and stranding the window.
                        print(f"  ctx~{target_ctx} rep {rep + 1}/{args.reps} FAILED: {error!r}")
                        print(f"  (completed reps are already durable in {jsonl})")
                        continue
                    print(
                        f"  ctx~{target_ctx} rep {rep + 1}/{args.reps} "
                        f"done in {time.time() - started:.2f} s wall"
                    )
                    time.sleep(1.0)
        finally:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
    record = parse_request_log(jsonl)
    record.source = "live"
    record = parse_serve_stdout(stdout_path, record)
    # The same run-admission the --from-log path uses. Without it the record's config fields
    # (kv_payload_bytes, host_to_device_bytes, cost sources) stay at their dataclass defaults
    # and the report claims they are "missing" when they are in the log all along. That is a
    # reporting bug that reads as a data gap, which is worse than either.
    chosen = select_run(record, args.min_decode_tokens)
    record = admit_run(record, chosen, args.min_decode_tokens)
    if len(record.runs) > 1:
        record.notes.append(
            f"this live log holds {len(record.runs)} runs; reported run #{chosen.index}"
        )
    run_result = {
        "binary": str(binary),
        "binary_sha16": binary_sha,
        "argv": argv,
        "ctx_sweep": ctxs,
        "request_log": str(jsonl),
        "serve_stdout": str(stdout_path),
        "runs_in_log": len(record.runs),
        "reported_run": chosen.index,
    }
    print()
    return record, run_result


def _fit_prompt(base: str, target_ctx: int, model_id: str) -> tuple[str, bool]:
    """Build a prompt whose token count is close to target_ctx, using serve's own counter.

    Returns (prompt, whether the server counted it). The count is taken from the server, not
    estimated: token estimation by character count is the kind of invented denominator this
    instrument exists to remove. If the endpoint is unavailable the achieved count is instead
    read back from request_done, and reported -- never silently assumed.
    """
    if target_ctx <= 8:
        return FILLER, True
    low, high = 1, max(2, target_ctx)
    best = FILLER
    counted_any = False
    for _ in range(14):
        repeats = (low + high) // 2
        candidate = FILLER * repeats
        try:
            counted = http_json(
                base + "/v1/messages/count_tokens",
                {"model": model_id,
                 "messages": [{"role": "user", "content": candidate}]},
                120.0,
            )
            total = int(counted.get("input_tokens") or counted.get("input_tokens_count") or 0)
        except Exception:
            return (best if counted_any else candidate), counted_any
        if total == 0:
            return (best if counted_any else candidate), counted_any
        counted_any = True
        best = candidate
        if total < target_ctx:
            low = repeats + 1
        else:
            high = repeats - 1
        if abs(total - target_ctx) <= max(8, target_ctx // 200):
            return candidate, True
        if low > high:
            break
    return best, counted_any


if __name__ == "__main__":
    raise SystemExit(main())
