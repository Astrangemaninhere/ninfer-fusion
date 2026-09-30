"""Closed registry of persistent NInfer tensor numeric formats."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from types import MappingProxyType
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class DirectFormat:
    """One fixed-width word per logical tensor element."""

    name: str
    word_bytes: int


@dataclass(frozen=True, slots=True)
class QuantFormat:
    """Signed grouped codes with one binary16 multiplier per group."""

    name: str
    bits: int
    group_size: int
    qmin: int
    qmax: int


@dataclass(frozen=True, slots=True)
class Nvfp4Format:
    """E2M1 weights with one E4M3FN scale word per K-axis group."""

    name: str
    group_size: int


@dataclass(frozen=True, slots=True)
class Fp8RowFormat:
    """E4M3FN weights with one BF16 multiplier per logical row."""

    name: str


@dataclass(frozen=True, slots=True)
class PackedU4Format:
    """Unsigned four-bit codes with one binary16 multiplier per K-axis group.

    Mirrors the engine's `artifact::NumericFormat::U4Z8G16_F16S`
    (src/artifact/reader.h), the PLE n-gram shard encoding `packed-u4-g16-v1`:
    packed nibbles plus one FP16 scale word per 16 K-axis columns.  It is NOT an
    `Nvfp4Format` -- NVFP4 pairs E2M1 codes with E4M3FN scale words, this pairs U4
    codes with FP16 scale words.
    """

    name: str
    group_size: int


NumericFormat: TypeAlias = (
    DirectFormat | QuantFormat | Nvfp4Format | Fp8RowFormat | PackedU4Format
)


BF16 = DirectFormat("BF16", 2)
FP32 = DirectFormat("FP32", 4)
I32 = DirectFormat("I32", 4)

Q4G64_F16S = QuantFormat("Q4G64_F16S", 4, 64, -8, 7)
Q5G64_F16S = QuantFormat("Q5G64_F16S", 5, 64, -16, 15)
Q6G64_F16S = QuantFormat("Q6G64_F16S", 6, 64, -32, 31)
W8G32_F16S = QuantFormat("W8G32_F16S", 8, 32, -127, 127)


#: 1-bit grouped codes, group 64 -- the bonsai-family rung.  Mirrors
#: ``artifact::NumericFormat::Q1G64_F16S`` (src/artifact/reader.h) and the
#: ``quant_geometry()`` case ``{64, 8, 0}`` (src/artifact/storage_layouts.cpp), i.e.
#: 8 bytes of base plane per 64 values == 1.00 bit, high plane empty.
Q1G64_F16S = QuantFormat("Q1G64_F16S", 1, 64, -1, 1)

#: 2-bit grouped codes, group 64 -- Prism PQ2_0 / upstream Q2_0.  ``{64, 16, 0}`` == 2.00 bit.
Q2G64_F16S = QuantFormat("Q2G64_F16S", 2, 64, -2, 1)

#: 3-bit grouped codes, group 64 -- ``{64, 16, 8}`` == 2 + 1 bit.  NOT GSQ: see the
#: ``arch_caps.h`` note; GSQ's pack-int32-le-v1 (pack_factor 10) is a different packing.
Q3G64_F16S = QuantFormat("Q3G64_F16S", 3, 64, -4, 3)

NVFP4 = Nvfp4Format("NVFP4", 16)
FP8_E4M3FN_ROW_BF16S = Fp8RowFormat("FP8_E4M3FN_ROW_BF16S")
FP8_E4M3FN_ROW_F32S = Fp8RowFormat("FP8_E4M3FN_ROW_F32S")
U4Z8G16_F16S = PackedU4Format("U4Z8G16_F16S", 16)


DIRECT_FORMATS = MappingProxyType(
    {item.name: item for item in (BF16, FP32, I32)}
)
QUANT_FORMATS = MappingProxyType(
    {
        item.name: item
        for item in (Q1G64_F16S, Q2G64_F16S, Q3G64_F16S,
                 Q4G64_F16S, Q5G64_F16S, Q6G64_F16S, W8G32_F16S)
    }
)
NVFP4_FORMATS = MappingProxyType({NVFP4.name: NVFP4})
FP8_ROW_FORMATS = MappingProxyType(
    {
        FP8_E4M3FN_ROW_BF16S.name: FP8_E4M3FN_ROW_BF16S,
        FP8_E4M3FN_ROW_F32S.name: FP8_E4M3FN_ROW_F32S,
    }
)
PACKED_U4_FORMATS = MappingProxyType({U4Z8G16_F16S.name: U4Z8G16_F16S})
NUMERIC_FORMATS = MappingProxyType(
    {**DIRECT_FORMATS, **QUANT_FORMATS, **NVFP4_FORMATS, **FP8_ROW_FORMATS,
     **PACKED_U4_FORMATS}
)


_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def decode_e2m1_word(word: int) -> float:
    """Decode one exact four-bit E2M1 word, including signed zero."""

    if type(word) is not int or not 0 <= word <= 0xF:
        raise ValueError("E2M1 word must be an integer in [0, 15]")
    magnitude = _E2M1_MAGNITUDES[word & 0x7]
    return math.copysign(magnitude, -1.0 if word & 0x8 else 1.0)


def decode_e4m3fn_word(word: int) -> float:
    """Decode one exact eight-bit E4M3FN word."""

    if type(word) is not int or not 0 <= word <= 0xFF:
        raise ValueError("E4M3FN word must be an integer in [0, 255]")
    sign = -1.0 if word & 0x80 else 1.0
    exponent = (word >> 3) & 0xF
    fraction = word & 0x7
    if exponent == 0:
        if fraction == 0:
            return math.copysign(0.0, sign)
        return sign * fraction * (2.0**-9)
    if exponent == 0xF and fraction == 0x7:
        return math.copysign(math.nan, sign)
    return sign * (1.0 + fraction / 8.0) * (2.0 ** (exponent - 7))


def valid_nvfp4_scale_word(word: int) -> bool:
    """Return whether *word* is an admitted nonnegative finite E4M3FN scale."""

    return (
        type(word) is int
        and 0 <= word <= 0xFF
        and word & 0x80 == 0
        and word != 0x7F
    )


def valid_fp8_weight_word(word: int) -> bool:
    """Return whether *word* is a finite E4M3FN weight code."""

    return type(word) is int and 0 <= word <= 0xFF and (word & 0x7F) != 0x7F


def valid_fp8_row_scale_word(word: int) -> bool:
    """Return whether *word* is a nonnegative finite BF16 multiplier."""

    if type(word) is not int or not 0 <= word <= 0xFFFF or word & 0x8000:
        return False
    value = struct.unpack("<f", struct.pack("<I", word << 16))[0]
    return math.isfinite(value)


def valid_fp32_row_scale_word(word: int) -> bool:
    """Return whether *word* is a nonnegative finite FP32 multiplier."""

    if type(word) is not int or not 0 <= word <= 0xFFFFFFFF or word & 0x80000000:
        return False
    value = struct.unpack("<f", struct.pack("<I", word))[0]
    return math.isfinite(value)


def valid_positive_fp32_word(word: int) -> bool:
    """Return whether an IEEE binary32 word represents a finite positive value."""

    if type(word) is not int or not 0 <= word <= 0xFFFFFFFF:
        return False
    value = struct.unpack("<f", struct.pack("<I", word))[0]
    return math.isfinite(value) and value > 0.0


def get_format(name: str) -> NumericFormat:
    """Return the registered format named *name*."""

    try:
        return NUMERIC_FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown numeric format: {name!r}") from None


__all__ = [
    "BF16",
    "DIRECT_FORMATS",
    "DirectFormat",
    "FP8_E4M3FN_ROW_BF16S",
    "FP8_E4M3FN_ROW_F32S",
    "FP8_ROW_FORMATS",
    "FP32",
    "Fp8RowFormat",
    "I32",
    "NUMERIC_FORMATS",
    "NVFP4",
    "NVFP4_FORMATS",
    "Nvfp4Format",
    "NumericFormat",
    "PACKED_U4_FORMATS",
    "PackedU4Format",
    "Q1G64_F16S",
    "Q2G64_F16S",
    "Q3G64_F16S",
    "Q4G64_F16S",
    "Q5G64_F16S",
    "Q6G64_F16S",
    "QUANT_FORMATS",
    "QuantFormat",
    "U4Z8G16_F16S",
    "W8G32_F16S",
    "decode_e2m1_word",
    "decode_e4m3fn_word",
    "get_format",
    "valid_fp32_row_scale_word",
    "valid_fp8_row_scale_word",
    "valid_fp8_weight_word",
    "valid_nvfp4_scale_word",
    "valid_positive_fp32_word",
]
