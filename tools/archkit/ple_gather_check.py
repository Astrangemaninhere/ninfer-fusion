#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ple_gather_check.py — PLE n-gram 真表 gather 校验器 (CPU 侧权威通道)。

背景 (§W2 / §9d): PLE 表的 sidecar 构建器与引擎往返已闭环, 唯一缺口是「真表 gather」
未验。本工具做两件事, 全部可离线复算:

  1. **manifest 语义审计** — 用 HF 官方实现 (transformers.models.qwen4_exp.
     modeling_qwen4_exp) 的算法独立复算 manifest 的三组关键参数:
       · layer_multipliers = _build_layer_multipliers(vocab, ngram, ple_layer_index, seed)
         (splitmix64 + 黄金比例常数, seed 默认 1234)
       · per_head_vocabulary_sizes = 第 h 个 >= base 的素数 (h=0..15)
       · per_head_offsets = 逐头累计
     复算不一致 = manifest 或公式有问题, 直接硬失败。

  2. **真表 gather 校验** — 按公式算 16 行号, 经 logical_parts 映射到物理文件 + 字节
     偏移, mmap 读 160 个 BF16, 输出 golden JSON (供 tools/ple_gather_test.cu 的
     GPU 通道逐字节对拍)。

公式 (HF 源码实证, 与 tools/ple_reference.py 一致):
    mixed = tok0*M0  XOR  tok1*M1  [XOR  tok2*M2]        # bigram / trigram 各自算
    row[h] = mixed % per_head_vocabulary_sizes[h] + per_head_offsets[h]
    h = (n-2)*heads_per_ngram + g                          # 头 0-7 bigram, 8-15 trigram
上下文: ctx[0]=当前 token, ctx[s]=前 s 个 token; 遇 EOS 截断 (后续位填 EOS)。

用法:
  python tools/archkit/ple_gather_check.py --manifest <ple-manifest.json> [--data-dir DIR]
      [--tokens "1,2,3,..."] [--emit-golden out.json] [--sample-bytes 0]
  python tools/archkit/ple_gather_check.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
PRIME_1 = 10007
DEFAULT_SEED = 1234


# ---------------------------------------------------------------------------
# HF 官方算法 (逐行对照 modeling_qwen4_exp.py, 不共享实现)
def splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def build_layer_multipliers(unigram_vocab_size: int, ngram_size: int,
                            ple_layer_index: int, seed: int) -> list[int]:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + PRIME_1 * ple_layer_index
    out = []
    for index in range(ngram_size):
        value = (base_seed + SPLITMIX_GAMMA * (index + 1)) & MASK64
        out.append(2 * (splitmix64(value) % half_bound) + 1)
    return out


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % d for d in range(3, math.isqrt(value) + 1, 2))


def nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not is_prime(prime):
            prime += 1
    return prime


def derive_rows(tokens: list[int], prevs: list[int] | None, eos: int,
                vocab_sizes: list[int], offsets: list[int],
                multipliers: list[int], ngram: int = 3,
                heads_per_ngram: int = 8) -> list[list[int]]:
    n_prev = ngram - 1
    out = []
    for i, tok in enumerate(tokens):
        ctx = [tok] + [eos] * n_prev
        cut = False
        for s in range(1, ngram):
            prev = prevs[i * n_prev + (s - 1)] if prevs else eos
            cut = cut or prev < 0 or prev == eos
            ctx[s] = eos if cut else prev
        row = [0] * len(vocab_sizes)
        for n in range(2, ngram + 1):
            mixed = (ctx[0] * multipliers[0]) & MASK64
            for j in range(1, n):
                mixed ^= (ctx[j] * multipliers[j]) & MASK64
            base = (n - 2) * heads_per_ngram
            for g in range(heads_per_ngram):
                h = base + g
                row[h] = (mixed % vocab_sizes[h]) + offsets[h]
        out.append(row)
    return out


# ---------------------------------------------------------------------------
def audit_manifest(manifest: dict, seed: int = DEFAULT_SEED,
                   ngram_base: int | None = None,
                   divisor: int | None = None) -> dict:
    """用 HF 算法独立复算 manifest 的关键参数; 不一致即失败。"""
    heads = int(manifest["number_of_ngram_heads"])
    per_ngram = int(manifest["heads_per_ngram"])
    ngram = int(manifest["ngram_size"])
    base = int(ngram_base if ngram_base is not None
               else manifest["ngram_vocab_size_base"])
    divisor = int(divisor if divisor is not None
                  else manifest.get("make_ngram_vocab_size_divisible_by", 128))
    vocab = int(manifest.get("unigram_vocab_size") or manifest.get("vocab_size") or 0)

    report: dict = {"checks": [], "ok": True}

    def check(name: str, expected, actual) -> None:
        good = expected == actual
        report["checks"].append({"name": name, "ok": good,
                                 "expected": expected, "actual": actual})
        report["ok"] = report["ok"] and good

    # 1) 每头词表 = 第 h 个 >= base 的素数
    want_sizes = [nth_prime_after(base - 1, h + 1) for h in range(heads)]
    check("per_head_vocabulary_sizes", want_sizes, manifest["per_head_vocabulary_sizes"])

    # 2) 偏移 = 逐头累计
    want_offsets, acc = [], 0
    for size in want_sizes:
        want_offsets.append(acc)
        acc += size
    check("per_head_offsets", want_offsets, manifest["per_head_offsets"])

    # 3) 对齐后总行数
    want_padded = math.ceil(acc / divisor) * divisor
    check("padded_vocabulary_rows", want_padded, manifest["padded_vocabulary_rows"])

    # 4) layer_multipliers (需 vocab_size + seed; 缺则跳过并如实标注)
    if vocab > 0:
        want_m = build_layer_multipliers(vocab, ngram, 0, seed)
        check(f"layer_multipliers(seed={seed},vocab={vocab})",
              want_m, manifest["layer_multipliers"])
    else:
        report["checks"].append({"name": "layer_multipliers", "ok": None,
                                 "note": "manifest 未带 vocab_size; 传 --vocab-size 可复算"})

    # 5) 行宽一致性
    check("row_stride_bytes", int(manifest["embedding_row_dimension"]) * 2,
          int(manifest["row_stride_bytes"]))
    check("heads_per_ngram*ngram_heads",
          heads, (ngram - 1) * per_ngram)
    return report


# ---------------------------------------------------------------------------
class TableReader:
    """按 logical_parts 把全局行号映射到物理文件 + 偏移, 只读 mmap。"""

    def __init__(self, manifest: dict, data_dir: Path) -> None:
        self.dir = data_dir
        self.row_stride = int(manifest["row_stride_bytes"])
        self.row_dim = int(manifest["embedding_row_dimension"])
        self.parts = sorted(manifest["logical_parts"], key=lambda p: p["global_row_start"])
        self.total_rows = int(manifest["padded_vocabulary_rows"])
        # 物理文件名以 manifest 为准 (不要自己拼格式)
        self.file_paths = [item["path"] for item in
                           sorted(manifest["physical_files"], key=lambda f: f["index"])]
        self._maps: dict[int, object] = {}
        # 覆盖性检查: parts 必须连续覆盖 [0, total_rows)
        cursor = 0
        for part in self.parts:
            if int(part["global_row_start"]) != cursor:
                raise ValueError(f"logical_parts 不连续 @ {cursor} vs {part['global_row_start']}")
            cursor += int(part["rows"])
        if cursor != self.total_rows:
            raise ValueError(f"logical_parts 覆盖 {cursor} 行, manifest 声明 {self.total_rows}")

    def locate(self, row: int) -> tuple[int, int]:
        for part in self.parts:
            start = int(part["global_row_start"])
            if start <= row < start + int(part["rows"]):
                offset = int(part["file_offset"]) + (row - start) * self.row_stride
                return int(part["physical_file_index"]), offset
        raise IndexError(f"row {row} out of range [0,{self.total_rows})")

    def _mmap(self, index: int):
        import mmap
        if index not in self._maps:
            path = self.dir / self.file_paths[index]
            handle = open(path, "rb")
            self._maps[index] = (handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ))
        return self._maps[index][1]

    def gather(self, row: int) -> bytes:
        index, offset = self.locate(row)
        mm = self._mmap(index)
        return bytes(mm[offset:offset + self.row_stride])

    def close(self) -> None:
        for handle, mm in self._maps.values():
            mm.close()
            handle.close()
        self._maps.clear()


def sample_tokens(count: int, vocab: int = 248320, eos: int = 248044) -> list[int]:
    """确定性样本: 混合小 id / 大 id / EOS 截断场景 (不依赖随机数实现)。"""
    tokens = []
    value = 7
    for i in range(count):
        if i and i % 7 == 0:
            tokens.append(eos)          # 触发 EOS 截断
        else:
            value = (value * 1103515245 + 12345) & 0x7FFFFFFF
            tokens.append(value % vocab)
    return tokens


def run_check(manifest_path: Path, data_dir: Path | None, tokens: list[int] | None,
              emit_golden: Path | None, vocab_size: int | None, seed: int,
              ngram_base: int | None, divisor: int | None) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if vocab_size:
        manifest["unigram_vocab_size"] = vocab_size
    print("== manifest 语义审计 ==", flush=True)
    report = audit_manifest(manifest, seed, ngram_base, divisor)
    for item in report["checks"]:
        if item["ok"] is None:
            print(f"  SKIP  {item['name']}: {item.get('note')}")
        else:
            mark = "OK  " if item["ok"] else "FAIL"
            print(f"  {mark}  {item['name']}")
            if not item["ok"]:
                print(f"        expected={item['expected']}")
                print(f"        actual  ={item['actual']}")
    if not report["ok"]:
        print("manifest 审计 FAIL — 公式或 manifest 有问题, 停止", flush=True)
        return 1

    if tokens is None:
        tokens = sample_tokens(24)
    eos = int(manifest.get("eos_token_id") or 248044)
    prevs = [eos] * (len(tokens) * (int(manifest["ngram_size"]) - 1))
    rows = derive_rows(tokens, prevs, eos,
                       manifest["per_head_vocabulary_sizes"],
                       manifest["per_head_offsets"],
                       manifest["layer_multipliers"],
                       int(manifest["ngram_size"]), int(manifest["heads_per_ngram"]))
    usable = int(manifest["usable_vocabulary_rows"])
    flat = [r for row in rows for r in row]
    print(f"\n== 行号范围检查 (usable={usable}) ==")
    bad = [r for r in flat if not (0 <= r < usable)]
    print(f"  rows={len(flat)} 越界={len(bad)}")
    if bad:
        print("  越界样本:", bad[:5])
        return 1

    if data_dir is not None:
        print(f"\n== 真表 gather ({data_dir}) ==")
        reader = TableReader(manifest, data_dir)
        try:
            digest = hashlib.sha256()
            gathered = []
            for row_index, row in enumerate(rows):
                vecs = [reader.gather(r) for r in row]
                gathered.append([vec.hex()[:32] for vec in vecs])
                for vec in vecs:
                    digest.update(vec)
            print(f"  gather 行数={len(rows)}x{len(rows[0])} 字节={len(rows) * len(rows[0]) * reader.row_stride}")
            print(f"  载荷 sha256={digest.hexdigest()}")
            # 抽样打印一行, 便于人工核对
            print(f"  样本 token={tokens[0]} rows={rows[0][:4]}...")
            if emit_golden:
                golden = {
                    "manifest": str(manifest_path),
                    "eos": eos,
                    "tokens": tokens,
                    "prevs": prevs,
                    "rows": rows,
                    "row_stride_bytes": reader.row_stride,
                    "payload_sha256": digest.hexdigest(),
                    "sample_hex": gathered[0],
                }
                emit_golden.write_text(json.dumps(golden, indent=1), encoding="utf-8")
                print(f"  golden 写入 {emit_golden}")
        finally:
            reader.close()
    else:
        print("\n(未给 --data-dir: 只做 manifest 审计与行号推导)")
    return 0


def self_test() -> int:
    """用仓库合成 fixture (ple_reference.py gen) 验证: 审计 + 推导 + gather 往返。"""
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import ple_reference  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        ple_reference.gen_fixture(tmp)
        manifest_path = Path(tmp) / "ple-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # 合成 fixture 的 per-head 词表不是素数 → 审计应失败 (反向门)
        manifest["ngram_vocab_size_base"] = manifest["per_head_vocabulary_sizes"][0] - 1
        report = audit_manifest(manifest)
        assert not report["ok"], "合成 fixture 不是素数词表, 审计应当失败"
        # gather 往返: fixture 每行低 16 位编码 row_id
        reader = TableReader(manifest, Path(tmp))
        try:
            for row in (0, 1, 249, 250, 31999):
                vec = reader.gather(row)
                assert len(vec) == 320, row
                lo = int.from_bytes(vec[:2], "little")
                assert lo == (row & 0xFFFF), (row, lo)
        finally:
            reader.close()
        # 推导: 与 ple_reference.derive_rows 交叉对拍
        tokens = [5, 7, 7, 9, 9, 9, 100, 200, 300]
        eos = 151643
        prevs = [eos] * (len(tokens) * 2)
        ours = derive_rows(tokens, prevs, eos, manifest["per_head_vocabulary_sizes"],
                           manifest["per_head_offsets"], manifest["layer_multipliers"])
        theirs = ple_reference.derive_rows(tokens, prevs, eos,
                                           manifest["per_head_vocabulary_sizes"],
                                           manifest["per_head_offsets"])
        assert ours == theirs, "derive_rows 与仓库参考实现不一致"
        print("[self-test] PASS (审计反向门 + gather 往返 + derive_rows 交叉对拍)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="ple-manifest.json 路径")
    ap.add_argument("--data-dir", help="物理分片所在目录 (manifest 同级)")
    ap.add_argument("--tokens", help="逗号分隔 token 序列 (默认确定性样本)")
    ap.add_argument("--emit-golden", help="写出 golden JSON (供 GPU 通道对拍)")
    ap.add_argument("--vocab-size", type=int, help="unigram vocab (复算 multipliers 用)")
    ap.add_argument("--ngram-base", type=int,
                    help="ngram_vocab_size_base (manifest 缺失时传入, 默认 20000000)")
    ap.add_argument("--divisor", type=int, help="make_ngram_vocab_size_divisible_by (默认 128)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="HF seed (默认 1234)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.manifest:
        ap.print_help()
        return 2
    tokens = None
    if args.tokens:
        tokens = [int(tok) for tok in args.tokens.replace(" ", "").split(",") if tok]
    data_dir = Path(args.data_dir) if args.data_dir else None
    golden = Path(args.emit_golden) if args.emit_golden else None
    return run_check(Path(args.manifest), data_dir, tokens, golden, args.vocab_size,
                     args.seed, args.ngram_base, args.divisor)


if __name__ == "__main__":
    sys.exit(main())
