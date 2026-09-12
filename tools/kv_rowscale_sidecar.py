"""KV row-scale sidecar: format spec, dump, write, identity, validate.

The engine reads a row-scale table through ``NINFER_KV_ROWSCALE=<path>`` once at
decoder-state planning (``src/ops/kernel/gqa_isoquant_row_scale_loader.cu``); an
unset variable means the feature is off and nothing happens.  The row scales are
the Sinkhorn-constrained per-channel multipliers of the rotated NVFP4 K domain
(``gqa_isoquant_row_scale.cuh``): for every full-attention ``(layer, kv_head)`` a
per-channel ``s_d`` in ``[0.5, 2.0]`` balances the rotated K row RMS before
E4M3/E2M1 quantization, K is multiplied by ``s_d`` on cache write and Q by
``1/s_d`` before QK, so ``QK^T`` is preserved.

Format (little endian, 64-byte header followed by the payload)::

     0  16  magic "NINFERKVRS1" NUL-padded (bytes 11..15 must be zero)
    16   4  u32 version = 1
    20   4  u32 layers     (table extent)
    24   4  u32 kv_heads
    28   4  u32 head_dim
    32   4  u32 flags      bit0 = identity table
    36   4  u32 crc32      zlib-style, over the payload
    40   8  u64 model_hash sha256(model artifact)[:8], 0 = unspecified
    48  16  tag[16] NUL-padded
    64  ..  payload: u16[layers * kv_heads * head_dim], BF16 bit patterns,
            row-major [layer][kv_head][d]

Identity rules, mirrored from the engine's host-side parser so a file that
validates here validates there: a non-identity table must match the model
geometry exactly (the 16-layer-table-under-52-layer-model case is refused), its
payload must fill the pool capacity exactly, and every word must be BF16 1.0
(``0x3F80``); an identity table may cover fewer layers than the model.  Every
failure names the offending field.

Subcommands::

    dump      --out payload.bin [--source <gqa_isoquant_row_scale.cu>]
    write     --words payload.bin --out table.kvrs --tag NAME --layers N \\
              --kv-heads N --head-dim N [--model-hash H]
    identity  --out table.kvrs --tag NAME --layers N --kv-heads N --head-dim N
    validate  table.kvrs [--expect-layers N] [--expect-kv-heads N]
              [--expect-head-dim N] [--expect-model-hash H]
    selfcheck

``selfcheck`` is the acceptance the design note asks for: it dumps the baked
pool, writes a sidecar from it, parses that back, and requires the payload bytes
to equal the baked words - so loading the sidecar cannot change numerics for the
model the table was baked from, by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


MAGIC = b"NINFERKVRS1"
VERSION = 1
HEADER_BYTES = 64
FLAG_IDENTITY = 0x1
POOL_CAPACITY = 16 * 4 * 256          # kKvRowScalePoolCapacity / kKvRowScalePoolWords
BF16_ONE = 0x3F80

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL_SOURCE = REPO_ROOT / "src" / "ops" / "kernel" / "gqa_isoquant_row_scale.cu"


class SidecarError(ValueError):
    """A validation failure, carrying the engine's field-named wording."""


@dataclass(frozen=True, slots=True)
class Sidecar:
    version: int
    layers: int
    kv_heads: int
    head_dim: int
    flags: int
    crc: int
    model_hash: int
    tag: str
    words: tuple[int, ...]

    @property
    def identity(self) -> bool:
        return bool(self.flags & FLAG_IDENTITY)

    def describe(self) -> str:
        lo = min(self.words)
        hi = max(self.words)
        mean = sum(self.words) / len(self.words)
        return (f"layers={self.layers} kv_heads={self.kv_heads} head_dim={self.head_dim} "
                f"words={len(self.words)} identity={int(self.identity)} "
                f"crc={self.crc:08x} tag={self.tag!r} words_lo_hi={lo:#06x}..{hi:#06x} "
                f"raw_mean={mean:.1f}")


def bf16_to_float(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (word & 0xFFFF) << 16))[0]


def float_to_bf16(value: float) -> int:
    """Round a binary32 value to BF16 with round-to-nearest-even."""

    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def parse(blob: bytes) -> Sidecar:
    """Parse and validate a sidecar, mirroring the engine's own checks."""

    if len(blob) < HEADER_BYTES:
        raise SidecarError(f"size: {len(blob)} bytes < 64 header")
    if blob[:11] != MAGIC or any(blob[11:16]):
        raise SidecarError("magic: sidecar is not a NINFERKVRS1 file")
    (version, layers, kv_heads, head_dim, flags, crc) = struct.unpack_from("<6I", blob, 16)
    if version != VERSION:
        raise SidecarError(f"version: {version} != {VERSION}")
    (model_hash,) = struct.unpack_from("<Q", blob, 40)
    tag = blob[48:64].split(b"\0", 1)[0].decode("ascii", "replace")

    expected_words = layers * kv_heads * head_dim
    payload = blob[HEADER_BYTES:]
    if expected_words != len(payload) // 2 or len(payload) % 2 != 0:
        raise SidecarError(
            f"length: payload {len(payload)} bytes != {expected_words * 2} for "
            f"{layers}x{kv_heads}x{head_dim}")
    if zlib.crc32(payload) != crc:
        raise SidecarError("crc32: payload crc mismatch")
    words = tuple(struct.unpack_from("<H", payload, 2 * i)[0] for i in range(expected_words))
    if flags & FLAG_IDENTITY:
        for word in words:
            if word != BF16_ONE:
                raise SidecarError("identity.payload: non-1.0 word in identity-flagged table")
    return Sidecar(version, layers, kv_heads, head_dim, flags, crc, model_hash, tag, words)


def check(sidecar: Sidecar, *, layers: int, kv_heads: int, head_dim: int,
          model_hash: int = 0) -> None:
    """Apply the engine's identity gate against a model geometry."""

    if not sidecar.identity and sidecar.layers != layers:
        raise SidecarError(
            f"identity.layers: table {sidecar.layers} != model {layers} "
            "(foreign-model table refused)")
    if sidecar.identity and sidecar.layers > layers:
        raise SidecarError(f"identity.layers: identity table {sidecar.layers} > model {layers}")
    if sidecar.kv_heads != kv_heads:
        raise SidecarError(f"identity.kv_heads: table {sidecar.kv_heads} != model {kv_heads}")
    if sidecar.head_dim != head_dim:
        raise SidecarError(f"identity.head_dim: table {sidecar.head_dim} != model {head_dim}")
    if len(sidecar.words) > POOL_CAPACITY:
        raise SidecarError(
            f"symbol_extent: table has {len(sidecar.words)} words > "
            f"kGqaKvRowScalePool capacity {POOL_CAPACITY}")
    if model_hash != 0 and sidecar.model_hash != 0 and sidecar.model_hash != model_hash:
        raise SidecarError(
            f"model_hash: sidecar {sidecar.model_hash} != model {model_hash}")


def build(words: Sequence[int], *, layers: int, kv_heads: int, head_dim: int,
          tag: str, model_hash: int = 0, identity: bool = False) -> bytes:
    """Assemble a sidecar from BF16 words."""

    if len(words) != layers * kv_heads * head_dim:
        raise SidecarError(
            f"length: {len(words)} words != {layers}x{kv_heads}x{head_dim}")
    if len(words) > POOL_CAPACITY:
        raise SidecarError(f"symbol_extent: {len(words)} words > {POOL_CAPACITY}")
    payload = struct.pack(f"<{len(words)}H", *words)
    flags = FLAG_IDENTITY if identity else 0
    if identity and any(word != BF16_ONE for word in words):
        raise SidecarError("identity.payload: non-1.0 word in identity-flagged table")
    header = bytearray(HEADER_BYTES)
    header[0:11] = MAGIC
    struct.pack_into("<6I", header, 16, VERSION, layers, kv_heads, head_dim, flags,
                     zlib.crc32(payload))
    struct.pack_into("<Q", header, 40, model_hash & 0xFFFFFFFFFFFFFFFF)
    encoded = tag.encode("ascii", "replace")[:16]
    header[48:48 + len(encoded)] = encoded
    return bytes(header) + payload


_ARRAY = re.compile(
    r"kGqaKvRowScalePool\s*\[[^\]]*\]\s*=\s*\{(?P<body>.*?)\}\s*;", re.DOTALL)


def dump_baked_pool(source: Path = DEFAULT_POOL_SOURCE) -> tuple[int, ...]:
    """Extract the baked BF16 pool from the kernel's initializer.

    Parsing the initializer (rather than reading device memory) is what makes the
    round-trip proof meaningful: the bytes compared are the ones the compiler
    bakes, not the ones a particular build happened to upload.
    """

    text = source.read_text(encoding="utf-8", errors="replace")
    match = _ARRAY.search(text)
    if match is None:
        raise SidecarError(f"baked pool initializer not found in {source}")
    # The initializer is decimal and carries a per-layer separator comment
    # ("// ---- layer 0 : kv_head 0 .. 3 ----") whose numbers would otherwise be
    # counted as words (three per layer over sixteen layers is exactly the 48-word
    # surplus a comment-blind tokenizer reports).
    body = re.sub(r"//[^\n]*", "", match.group("body"))
    tokens = re.findall(r"0[xX][0-9a-fA-F]+|\d+", body)
    words = []
    for token in tokens:
        value = int(token, 16) if token.lower().startswith("0x") else int(token)
        if not 0 <= value <= 0xFFFF:
            raise SidecarError(f"baked pool word out of range: {token}")
        words.append(value)
    if len(words) != POOL_CAPACITY:
        raise SidecarError(
            f"symbol_extent: baked pool has {len(words)} words, expected {POOL_CAPACITY}")
    return tuple(words)


def documented_payload_crc(source: Path = DEFAULT_POOL_SOURCE) -> int | None:
    """The crc32 the source records in its own comment, if it records one.

    The kernel's header comment states the payload crc so the table stays
    auditable across refactors; using it here cross-checks the tool against an
    independent record instead of against itself.
    """

    text = source.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Payload crc32[^=]*=\s*([0-9a-fA-F]{8})", text)
    return int(match.group(1), 16) if match else None


def model_hash_of(path: Path) -> int:
    """First eight bytes of the model artifact's sha256, as the header carries it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return int.from_bytes(digest.digest()[:8], "little")


def _report(sidecar: Sidecar, extra: str = "") -> None:
    print(f"  {sidecar.describe()}{(' ' + extra) if extra else ''}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dump", help="extract the baked pool words")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", type=Path, default=DEFAULT_POOL_SOURCE)

    def geometry(sub_parser: argparse.ArgumentParser) -> None:
        sub_parser.add_argument("--layers", type=int, required=True)
        sub_parser.add_argument("--kv-heads", type=int, required=True)
        sub_parser.add_argument("--head-dim", type=int, required=True)
        sub_parser.add_argument("--tag", required=True)
        sub_parser.add_argument("--model-hash", type=lambda v: int(v, 0), default=0)

    p = sub.add_parser("write", help="write a sidecar from dumped words")
    p.add_argument("--words", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    geometry(p)

    p = sub.add_parser("identity", help="write an all-BF16-1.0 sidecar")
    p.add_argument("--out", type=Path, required=True)
    geometry(p)

    p = sub.add_parser("validate", help="parse and optionally gate a sidecar")
    p.add_argument("table", type=Path)
    p.add_argument("--expect-layers", type=int)
    p.add_argument("--expect-kv-heads", type=int)
    p.add_argument("--expect-head-dim", type=int)
    p.add_argument("--expect-model-hash", type=lambda v: int(v, 0), default=0)

    sub.add_parser("selfcheck", help="round-trip the baked pool through the format")

    args = parser.parse_args(argv)

    if args.command == "dump":
        words = dump_baked_pool(args.source)
        args.out.write_bytes(struct.pack(f"<{len(words)}H", *words))
        print(f"  dump: {len(words)} words -> {args.out}")
        print(f"        value range [{bf16_to_float(min(words)):.4f}, "
              f"{bf16_to_float(max(words)):.4f}]")
        return 0

    if args.command in ("write", "identity"):
        if args.command == "write":
            raw = args.words.read_bytes()
            if len(raw) % 2:
                raise SidecarError(f"length: {args.words} has an odd byte count")
            words = struct.unpack(f"<{len(raw) // 2}H", raw)
            identity = False
        else:
            words = (BF16_ONE,) * (args.layers * args.kv_heads * args.head_dim)
            identity = True
        blob = build(words, layers=args.layers, kv_heads=args.kv_heads,
                     head_dim=args.head_dim, tag=args.tag, model_hash=args.model_hash,
                     identity=identity)
        args.out.write_bytes(blob)
        _report(parse(blob), f"-> {args.out} ({len(blob)} bytes)")
        return 0

    if args.command == "validate":
        sidecar = parse(args.table.read_bytes())
        _report(sidecar)
        if args.expect_layers is not None:
            check(sidecar, layers=args.expect_layers,
                  kv_heads=args.expect_kv_heads if args.expect_kv_heads is not None else sidecar.kv_heads,
                  head_dim=args.expect_head_dim if args.expect_head_dim is not None else sidecar.head_dim,
                  model_hash=args.expect_model_hash)
            print("  gate: OK")
        return 0

    # selfcheck
    words = dump_baked_pool()
    blob = build(words, layers=16, kv_heads=4, head_dim=256, tag="qwen3_8_27b")
    sidecar = parse(blob)
    if sidecar.words != words:
        raise SidecarError("roundtrip: parsed words differ from the baked pool")
    print("  roundtrip: payload bytes == baked pool words (bit compare) OK")
    check(sidecar, layers=16, kv_heads=4, head_dim=256)
    print("  gate 16/4/256: OK")
    recorded = documented_payload_crc()
    if recorded is not None:
        if recorded != sidecar.crc:
            raise SidecarError(
                f"crc32: produced {sidecar.crc:08x} != the value the source records "
                f"({recorded:08x})")
        print(f"  crc32 {sidecar.crc:08x} == the value recorded in the source OK")
    try:
        check(sidecar, layers=52, kv_heads=4, head_dim=256)
    except SidecarError as exc:
        print(f"  gate 52/4/256: refused ({exc}) OK")
    else:
        raise SidecarError("gate 52/4/256 was accepted but must be refused")
    identity = parse(build((BF16_ONE,) * (16 * 4 * 256), layers=16, kv_heads=4, head_dim=256,
                           tag="qwen3_8_27b", identity=True))
    check(identity, layers=52, kv_heads=4, head_dim=256)
    print("  identity under a 52-layer model: accepted OK")
    for field, mutated in (("magic", b"X" + blob[1:]),
                           ("version", blob[:16] + struct.pack("<I", 9) + blob[20:]),
                           ("crc", blob[:36] + struct.pack("<I", 0xDEADBEEF) + blob[40:]),
                           ("truncated", blob[:-2])):
        try:
            parse(mutated)
        except SidecarError as exc:
            print(f"  corrupt {field}: refused ({exc}) OK")
        else:
            raise SidecarError(f"corrupt {field} was accepted but must be refused")
    sizes = len(blob)
    print(f"  sidecar size {sizes} bytes = 64 + 2 * {len(words)} OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SidecarError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
