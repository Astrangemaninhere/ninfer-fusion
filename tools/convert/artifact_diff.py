"""Object-by-object comparison of two `.ninfer` artifacts.

This is the acceptance gate for an imported artifact: it answers "does this build
reproduce the reference, object by object, and where exactly does it not" without
knowing anything about the model.  Every comparison goes through the registered
layout decoders, so the verdict is about the bytes the engine will actually read,
not about a re-derivation of them.

Per object the verdict is one of:

``identical``   the stored payloads are byte-for-byte equal;
``logical``     the payloads differ but both decode to the same logical values;
``divergent``   the logical values differ; the report carries the magnitude
                (max absolute difference and relative RMS), which is what a
                caller needs to judge a re-quantisation;
``structural``  format, layout, or shape differ, or one side is missing.

Exit status is 0 only when nothing is ``divergent`` or ``structural``, so the
tool doubles as a gate in a conversion pipeline.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from tools.artifact import layouts as L
from tools.artifact.container import Artifact, ResourceObject, TensorObject
from tools.artifact.numeric import decode_e2m1_word


DIRECT = frozenset(("BF16", "FP32", "I32"))
ROW_SPLIT = frozenset(("Q4G64_F16S", "Q5G64_F16S", "Q6G64_F16S", "W8G32_F16S"))
E2M1_TABLE = torch.tensor([decode_e2m1_word(value) for value in range(16)], dtype=torch.float32)


@dataclass(slots=True)
class ObjectVerdict:
    name: str
    status: str
    format: str = ""
    detail: str = ""
    max_abs: float | None = None
    relative_rms: float | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "status": self.status, "format": self.format}
        if self.detail:
            out["detail"] = self.detail
        if self.max_abs is not None:
            out["max_abs"] = self.max_abs
        if self.relative_rms is not None:
            out["relative_rms"] = self.relative_rms
        return out


def _payload(artifact: Artifact, obj: TensorObject | ResourceObject) -> bytes:
    with open(artifact.path, "rb") as handle:
        handle.seek(artifact.payload_offset + obj.offset)
        return handle.read(obj.bytes)


def _decode_nvfp4_logical(payload: bytes, shape: Sequence[int]) -> torch.Tensor:
    """Logical values of an NVFP4 object: E2M1 codes x E4M3 scales / divisor."""

    codes, scales, divisor = L.decode_nvfp4_words(payload, shape)
    nibbles = torch.empty((codes.shape[0], codes.shape[1] * 2), dtype=torch.uint8)
    nibbles[:, 0::2] = codes & 0x0F
    nibbles[:, 1::2] = (codes >> 4) & 0x0F
    values = E2M1_TABLE[nibbles.long()]
    scale_values = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return values * scale_values / float(divisor)


def _logical(payload: bytes, fmt: str, shape: Sequence[int]) -> torch.Tensor | None:
    """Decode a stored payload to comparable logical values, or None."""

    if fmt in DIRECT:
        return L.decode_direct(payload, fmt, shape, device="cpu").float()
    if fmt == "FP8_E4M3FN_ROW_BF16S":
        return L.dequantize_fp8_row_scaled(payload, shape, dtype=torch.float32)
    if fmt in ROW_SPLIT:
        return L.dequantize_row_split(payload, fmt, shape, device="cpu", dtype=torch.float32)
    if fmt == "NVFP4":
        return _decode_nvfp4_logical(payload, shape)
    return None


def compare_artifacts(
    left_path: Path, right_path: Path, *, names: Sequence[str] | None = None
) -> tuple[list[ObjectVerdict], dict[str, Any]]:
    verdicts: list[ObjectVerdict] = []
    with Artifact.open(left_path) as left, Artifact.open(right_path) as right:
        left_map = {obj.name: obj for obj in left.objects}
        right_map = {obj.name: obj for obj in right.objects}
        selected = list(names) if names else sorted(set(left_map) | set(right_map))

        for name in selected:
            a = left_map.get(name)
            b = right_map.get(name)
            if a is None or b is None:
                verdicts.append(ObjectVerdict(
                    name, "structural", detail=f"missing on {'left' if a is None else 'right'}"))
                continue
            if isinstance(a, ResourceObject) or isinstance(b, ResourceObject):
                pa, pb = _payload(left, a), _payload(right, b)
                if pa == pb:
                    verdicts.append(ObjectVerdict(name, "identical", format="resource"))
                else:
                    verdicts.append(ObjectVerdict(name, "divergent", format="resource",
                                                  detail=f"{len(pa)} B vs {len(pb)} B"))
                continue
            assert isinstance(a, TensorObject) and isinstance(b, TensorObject)
            if (a.format, a.layout, tuple(a.shape)) != (b.format, b.layout, tuple(b.shape)):
                verdicts.append(ObjectVerdict(
                    name, "structural", format=a.format,
                    detail=f"{a.format}/{a.layout} {tuple(a.shape)} vs "
                           f"{b.format}/{b.layout} {tuple(b.shape)}"))
                continue
            pa, pb = _payload(left, a), _payload(right, b)
            if pa == pb:
                verdicts.append(ObjectVerdict(name, "identical", format=a.format))
                continue
            left_values = _logical(pa, a.format, a.shape)
            right_values = _logical(pb, b.format, b.shape)
            if left_values is None or right_values is None or left_values.shape != right_values.shape:
                verdicts.append(ObjectVerdict(name, "divergent", format=a.format,
                                              detail="payloads differ; undecodable"))
                continue
            difference = (left_values - right_values).abs()
            max_abs = float(difference.max())
            reference = float(left_values.pow(2).mean().sqrt()) or 1.0
            relative = float(difference.pow(2).mean().sqrt()) / reference
            verdicts.append(ObjectVerdict(
                name, "logical" if max_abs == 0.0 else "divergent", format=a.format,
                max_abs=max_abs, relative_rms=relative,
            ))

    prefixes = sorted({v.name.split("/")[0] for v in verdicts})
    summary = {
        "left": str(left_path),
        "right": str(right_path),
        "objects": len(verdicts),
        "status": dict(Counter(v.status for v in verdicts)),
        "by_prefix": {
            prefix: dict(Counter(v.status for v in verdicts if v.name.split("/")[0] == prefix))
            for prefix in prefixes
        },
    }
    return verdicts, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--name", action="append", default=None,
                        help="限定的对象名（可重复）；缺省比较全部")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--list-divergent", action="store_true", help="逐个列出偏离/结构不同的对象")
    args = parser.parse_args(argv)

    verdicts, summary = compare_artifacts(args.left, args.right, names=args.name)

    if args.json:
        print(json.dumps({"summary": summary, "objects": [v.as_dict() for v in verdicts]},
                         ensure_ascii=False, indent=2))
    else:
        print(f"  left : {summary['left']}")
        print(f"  right: {summary['right']}")
        print(f"  objects: {summary['objects']}")
        print("  status :")
        for status, count in sorted(summary["status"].items()):
            print(f"    {status:<11} {count}")
        print("  by prefix (identical/logical/divergent/structural):")
        for prefix, counts in summary["by_prefix"].items():
            print("    %-10s id=%-5d log=%-5d div=%-5d str=%d"
                  % (prefix, counts.get("identical", 0), counts.get("logical", 0),
                     counts.get("divergent", 0), counts.get("structural", 0)))
        if args.list_divergent:
            print("  divergent / structural:")
            for verdict in verdicts:
                if verdict.status in ("divergent", "structural"):
                    extra = ""
                    if verdict.max_abs is not None:
                        extra = f" max_abs={verdict.max_abs:.4g} rel_rms={verdict.relative_rms:.4g}"
                    print(f"    [{verdict.status}] {verdict.name} ({verdict.format}) "
                          f"{verdict.detail}{extra}")

    blocked = summary["status"].get("divergent") or summary["status"].get("structural")
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
