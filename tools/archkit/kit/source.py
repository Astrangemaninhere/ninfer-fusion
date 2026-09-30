# -*- coding: utf-8 -*-
"""kit.source — the shared checkpoint reader: index, headers, byte ranges.

Two files are REQUIRED of any source directory that reaches the driver, and each
one that is absent produces a `KitRefusal` that names it:

  * ``config.json``                     — the geometry the spec is checked against
  * ``model.safetensors.index.json``    — the complete, authoritative key set

Both are refused BY NAME, before any header is parsed and before any payload
function is called.  The index is not a convenience: it is the denominator of the
R29 reading, because a plan that cannot be compared against the checkpoint's own
key list is a claim rather than a measurement.

Payloads are produced as byte RANGES of the shard files (`raw_range`), which is
why a BF16 pass-through conversion needs no torch and no GPU: nothing is
dequantized, cast or multiplied.  `meta()` reads the shard header only, so
classifying 219 tensors does not read 2.16 GB.
"""

from __future__ import annotations

import io
import json
import os
import struct
from pathlib import Path
from typing import Iterator, Mapping

from .contract import KitRefusal

CONFIG_NAME = "config.json"
INDEX_NAME = "model.safetensors.index.json"
_CHUNK = 1 << 22  # 4 MiB: bounded resident memory for a streaming payload


class SafetensorsSource:
    """A checkpoint directory read through its index, headers and byte ranges."""

    def __init__(self, directory: str | os.PathLike[str]):
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise KitRefusal("source directory %s does not exist" % self.directory)

        config_path = self.directory / CONFIG_NAME
        if not config_path.is_file():
            raise KitRefusal("missing %s under %s" % (CONFIG_NAME, self.directory))
        with io.open(config_path, encoding="utf-8") as handle:
            self.config: dict = json.load(handle)

        index_path = self.directory / INDEX_NAME
        if not index_path.is_file():
            raise KitRefusal("missing %s under %s" % (INDEX_NAME, self.directory))
        with io.open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise KitRefusal(
                "%s carries no usable 'weight_map' object" % INDEX_NAME)
        self.index_path = index_path
        self.weight_map: Mapping[str, str] = weight_map
        self.index_metadata: Mapping[str, object] = index.get("metadata") or {}
        self._headers: dict[str, tuple[dict, int]] = {}

    # ---- the authoritative key set -------------------------------------

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.weight_map))

    def contains(self, key: str) -> bool:
        return key in self.weight_map

    def shards(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.weight_map.values())))

    # ---- headers: metadata without reading the payload ------------------

    def _header(self, shard: str) -> tuple[dict, int]:
        cached = self._headers.get(shard)
        if cached is not None:
            return cached
        path = self.directory / shard
        if not path.is_file():
            raise KitRefusal("shard %s named by %s is not under %s"
                             % (shard, INDEX_NAME, self.directory))
        header_bytes = path.stat().st_size
        with open(path, "rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise KitRefusal("shard %s is shorter than its own 8-byte header "
                                 "length word" % shard)
            length = struct.unpack("<Q", raw)[0]
            if length == 0 or length + 8 > header_bytes:
                raise KitRefusal("shard %s declares a %d-byte header, which does not "
                                 "fit in its %d bytes" % (shard, length, header_bytes))
            body = handle.read(length)
        try:
            header = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise KitRefusal("shard %s has an unreadable safetensors header: %s"
                             % (shard, exc)) from exc
        entry = (header, 8 + length)
        self._headers[shard] = entry
        return entry

    def meta(self, key: str) -> tuple[str, tuple[int, ...], int]:
        """(dtype, shape, payload bytes) of one source tensor, header only."""
        shard = self.weight_map.get(key)
        if shard is None:
            raise KitRefusal("source key %s is not in %s" % (key, INDEX_NAME))
        header, _base = self._header(shard)
        entry = header.get(key)
        if entry is None:
            raise KitRefusal("source key %s is named by %s but absent from the header "
                             "of shard %s" % (key, INDEX_NAME, shard))
        begin, end = entry["data_offsets"]
        return str(entry["dtype"]), tuple(int(d) for d in entry["shape"]), int(end - begin)

    def element_bytes(self, key: str) -> int:
        _dtype, shape, payload = self.meta(key)
        count = 1
        for dim in shape:
            count *= dim
        if count == 0:
            raise KitRefusal("source key %s declares an empty shape" % key)
        if payload % count:
            raise KitRefusal("source key %s declares %d payload bytes over %d elements, "
                             "which is not a whole number of bytes per element"
                             % (key, payload, count))
        return payload // count

    # ---- payloads: byte ranges, streamed --------------------------------

    def _range(self, key: str) -> tuple[Path, int, int]:
        shard = self.weight_map.get(key)
        if shard is None:
            raise KitRefusal("source key %s is not in %s" % (key, INDEX_NAME))
        header, base = self._header(shard)
        entry = header.get(key)
        if entry is None:
            raise KitRefusal("source key %s is named by %s but absent from shard %s"
                             % (key, INDEX_NAME, shard))
        begin, end = entry["data_offsets"]
        path = self.directory / shard
        size = path.stat().st_size
        if base + end > size:
            raise KitRefusal("source key %s runs to byte %d of shard %s, which is only "
                             "%d bytes" % (key, base + end, shard, size))
        return path, base + begin, int(end - begin)

    def range_length(self, key: str) -> int:
        return self._range(key)[2]

    def raw_range(self, key: str) -> Iterator[memoryview]:
        """Stream one source tensor's stored bytes, chunk by chunk."""
        path, offset, length = self._range(key)
        remaining = length
        with open(path, "rb") as handle:
            handle.seek(offset)
            while remaining > 0:
                chunk = handle.read(min(_CHUNK, remaining))
                if not chunk:
                    raise KitRefusal(
                        "shard %s ended %d bytes early while reading %s"
                        % (path.name, remaining, key))
                remaining -= len(chunk)
                yield memoryview(chunk)

    def raw_bytes(self, key: str) -> bytes:
        path, offset, length = self._range(key)
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
        if len(data) != length:
            raise KitRefusal("shard %s ended %d bytes early while reading %s"
                             % (path.name, length - len(data), key))
        return data

    def rows(self, key: str, first: int, count: int) -> Iterator[memoryview]:
        """Stream `count` consecutive ROWS of a 2-D source tensor.

        This is the whole of a tied-embedding or a row-window payload: a tie is a
        byte window, so it never needs the tensor to be materialised.
        """
        _dtype, shape, _payload = self.meta(key)
        if len(shape) != 2:
            raise KitRefusal("source key %s is %d-D; a row window needs 2-D"
                             % (key, len(shape)))
        width = self.element_bytes(key) * shape[1]
        path, offset, length = self._range(key)
        if first < 0 or count < 0 or (first + count) > shape[0]:
            raise KitRefusal("row window [%d, %d) of %s is outside its %d rows"
                             % (first, first + count, key, shape[0]))
        start = offset + first * width
        remaining = count * width
        if start + remaining > offset + length:
            raise KitRefusal("row window of %s overruns the stored tensor" % key)
        with open(path, "rb") as handle:
            handle.seek(start)
            while remaining > 0:
                chunk = handle.read(min(_CHUNK, remaining))
                if not chunk:
                    raise KitRefusal("shard %s ended early in a row window of %s"
                                     % (path.name, key))
                remaining -= len(chunk)
                yield memoryview(chunk)
