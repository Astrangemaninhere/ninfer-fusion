"""Grouped symmetric quantization used by NInfer artifact converters.

The persistent numeric format fixes the code range, group size, and binary16
scale.  Model-specific recipes decide which tensors use those formats; this
module only performs the registered numeric transform.

Two registered formats cannot be produced by :func:`quantize_matrix`, because
their persistent form carries a per-*row* multiplier rather than a per-group
one.  ``FP8_E4M3FN_ROW_BF16S`` is covered here: its rule was recovered by
inverting a released artifact (``row_scale == bf16(fp32(amax_row / 448))`` held
for every one of 14336 rows, with no power-of-two snapping), so it is the same
kind of closed numeric transform as the grouped formats, not a heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tools.artifact.layouts import encode_fp8_row_scaled, encode_row_split, row_split_geometry
from tools.artifact.numeric import QuantFormat, get_format


_FP16_MIN_SUBNORMAL = 2.0**-24

#: Largest finite magnitude of E4M3FN, and therefore the row-scale denominator.
FP8_E4M3FN_MAX = 448.0


@dataclass(frozen=True, slots=True)
class QuantizedMatrix:
    """Physical code groups and binary16 scales for one logical matrix."""

    codes: torch.Tensor
    scales: torch.Tensor


@dataclass(frozen=True, slots=True)
class QuantizedFp8RowMatrix:
    """Physical E4M3FN code words and one binary16 multiplier per logical row."""

    codes: torch.Tensor
    row_scales: torch.Tensor


def _canonical_scale_words(
    max_abs: torch.Tensor,
    qmax: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical binary16 scales and binary32 reciprocals on the host.

    CUDA division is not correctly rounded at every binary16 scale boundary.
    The host oracle performs the specified division in binary64, explicitly
    rounds through binary32 and binary16, then computes the reciprocal in the
    same way.  A binary32 input divided by these small integer denominators has
    enough binary64 precision for the final binary32 rounding to be exact.
    """

    host_max = max_abs.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(host_max).all():
        raise ValueError("grouped quantization source contains NaN or infinity")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw_scale = (host_max.astype(np.float64) / float(qmax)).astype(np.float32)
        scale = raw_scale.astype(np.float16)
    underflow = (scale == 0) & (host_max > 0)
    if underflow.any():
        scale = scale.copy()
        scale[underflow] = np.array(_FP16_MIN_SUBNORMAL, dtype=np.float16)
    if np.any((host_max > 0) & (~np.isfinite(scale) | (scale <= 0))):
        raise ValueError("grouped quantization scale is not finite and positive")

    reciprocal = np.zeros(host_max.shape, dtype=np.float32)
    positive = scale > 0
    reciprocal[positive] = (
        1.0 / scale[positive].astype(np.float64)
    ).astype(np.float32)
    return torch.from_numpy(scale), torch.from_numpy(reciprocal)


def pick_device(preferred: str | torch.device = "cuda") -> torch.device:
    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def quantize_matrix(
    weight: torch.Tensor,
    format: str | QuantFormat,
    *,
    device: str | torch.device | None = None,
) -> QuantizedMatrix:
    """Quantize logical ``[N,K]`` values, including registered K padding.

    Scales are rounded to binary16 before codes are selected because those are
    the exact scales consumed after loading.  Padding values are zero and do
    not affect a partially populated final group.
    """

    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, QuantFormat):
        raise ValueError("grouped quantization requires a quantized numeric format")
    if weight.dim() != 2:
        raise ValueError(f"grouped quantization requires rank 2, got {tuple(weight.shape)}")
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")

    geometry = row_split_geometry(spec, weight.shape)
    target = pick_device() if device is None else pick_device(device)
    logical = weight.detach().to(device=target, dtype=torch.float32)
    if geometry.k_pad != geometry.k:
        physical = torch.zeros(
            (geometry.n, geometry.k_pad), dtype=torch.float32, device=target
        )
        physical[:, : geometry.k].copy_(logical)
        logical = physical

    grouped = logical.reshape(
        geometry.n, geometry.groups_per_row, spec.group_size
    )
    max_abs = grouped.abs().amax(dim=2)
    host_scales, host_reciprocal = _canonical_scale_words(max_abs, spec.qmax)
    scales = host_scales.to(target)
    reciprocal = host_reciprocal.to(target)
    codes = torch.clamp(
        torch.round(grouped * reciprocal.unsqueeze(-1)), spec.qmin, spec.qmax
    ).to(torch.int8)
    return QuantizedMatrix(codes=codes, scales=scales)


def quantize_and_encode(
    weight: torch.Tensor,
    format: str | QuantFormat,
    *,
    device: str | torch.device | None = None,
) -> bytes:
    """Quantize a logical matrix and encode ``row-split-k128-v1`` bytes."""

    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, QuantFormat):
        raise ValueError("grouped quantization requires a quantized numeric format")
    quantized = quantize_matrix(weight, spec, device=device)
    return encode_row_split(quantized.codes, quantized.scales, spec, weight.shape)


def _canonical_bf16_scales(max_abs: torch.Tensor) -> torch.Tensor:
    """Return the canonical BF16 row multipliers for ``FP8_E4M3FN_ROW_BF16S``.

    The released artifacts fix the rounding as binary64 division, then an
    explicit binary32 step, then binary16-width rounding.  A binary32 input
    divided by 448 has enough binary64 precision for the final rounding to be
    exact, so the host is the oracle.
    """

    host_max = max_abs.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(host_max).all():
        raise ValueError("row-scaled quantization source contains NaN or infinity")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw = (host_max.astype(np.float64) / FP8_E4M3FN_MAX).astype(np.float32)
    return torch.from_numpy(raw).to(torch.bfloat16)


def quantize_fp8_row_matrix(
    weight: torch.Tensor,
    *,
    device: str | torch.device | None = None,
) -> QuantizedFp8RowMatrix:
    """Quantize logical ``[N,K]`` values to E4M3FN codes with per-row BF16 scales.

    A row whose magnitude is entirely zero keeps a zero multiplier; the layout
    validator admits that only together with signed-zero codes, which is exactly
    what a zero numerator produces here.
    """

    if weight.dim() != 2:
        raise ValueError(f"row-scaled quantization requires rank 2, got {tuple(weight.shape)}")
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")

    target = pick_device() if device is None else pick_device(device)
    logical = weight.detach().to(device=target, dtype=torch.float32)
    max_abs = logical.abs().amax(dim=1)
    row_scales = _canonical_bf16_scales(max_abs).to(target)

    positive = row_scales > 0
    reciprocal = torch.zeros_like(row_scales, dtype=torch.float32)
    reciprocal[positive] = 1.0 / row_scales[positive].to(torch.float32)
    ratios = logical * reciprocal.unsqueeze(1)
    codes = (
        torch.clamp(ratios, -FP8_E4M3FN_MAX, FP8_E4M3FN_MAX)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    return QuantizedFp8RowMatrix(codes=codes, row_scales=row_scales)


def quantize_and_encode_fp8_row(
    weight: torch.Tensor,
    *,
    device: str | torch.device | None = None,
) -> bytes:
    """Quantize a logical matrix and encode ``row-scale-v1`` bytes."""

    quantized = quantize_fp8_row_matrix(weight, device=device)
    return encode_fp8_row_scaled(quantized.codes, quantized.row_scales, weight.shape)


__all__ = [
    "FP8_E4M3FN_MAX",
    "QuantizedFp8RowMatrix",
    "QuantizedMatrix",
    "pick_device",
    "quantize_and_encode",
    "quantize_and_encode_fp8_row",
    "quantize_fp8_row_matrix",
    "quantize_matrix",
]
