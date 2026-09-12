#!/usr/bin/env python3
"""两项小修（八项里最便宜的两条）：
 (i)  apps/cli/main.cpp: format_kv_cache 补齐 4 个缺失分支（此前 4/7 档误报 "unknown"）
 (ii) text_prefill_impl.h: 修正 dump 记录里 last 的语义注释（实测是 final-norm **之前**）
行尾自动探测；锚点断言唯一；备份到 /home/user/eight_bak/。
"""
import pathlib
import sys

R = pathlib.Path("/home/user/ninfer-fusion")
BAKDIR = pathlib.Path("/home/user/eight_bak")


def patch(rel, anchor_lines, repl_lines, label):
    p = R / rel
    raw = p.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    bak = BAKDIR / rel
    bak.parent.mkdir(parents=True, exist_ok=True)
    if not bak.exists():
        bak.write_bytes(raw.encode("utf-8"))
        print(f"  备份 -> {bak}")
    anchor = nl.join(anchor_lines)
    if raw.count(anchor) != 1:
        print(f"FAIL {label}: 锚点命中 {raw.count(anchor)} 次")
        sys.exit(3)
    p.write_bytes(raw.replace(anchor, nl.join(repl_lines), 1).encode("utf-8"))
    print(f"OK  {label}")


patch(
    "apps/cli/main.cpp",
    [
        "    case ninfer::KvCacheStorage::Fp8E4M3Row256:",
        '        return "fp8-e4m3-row256";',
        "    }",
    ],
    [
        "    case ninfer::KvCacheStorage::Fp8E4M3Row256:",
        '        return "fp8-e4m3-row256";',
        "    case ninfer::KvCacheStorage::Nvfp4Group16:",
        '        return "nvfp4-group16";',
        "    case ninfer::KvCacheStorage::Fp8Group16:",
        '        return "fp8-group16";',
        "    case ninfer::KvCacheStorage::Iso3Group16:",
        '        return "iso3-group16";',
        "    case ninfer::KvCacheStorage::E8Group64:",
        '        return "e8-group64";',
        "    }",
    ],
    "(i) format_kv_cache 补齐 4 档（此前误报 unknown）",
)

patch(
    "src/targets/qwen3_6/impl/runtime/text_prefill_impl.h",
    [
        "// u16 bf16 feat[feature_rows x tokens] (five target layers concatenated in",
        "// capture order), u16 bf16 last[hidden x tokens] (post-final-norm hidden).",
    ],
    [
        "// u16 bf16 feat[feature_rows x tokens] (five target layers concatenated in",
        "// capture order), u16 bf16 last[hidden x tokens] (final-norm **input**, i.e.",
        "// the hidden *before* text/final_norm is applied; consumers must apply it",
        "// themselves. Verified empirically: RMS(last)=1.87 > max|final_norm|=1.71,",
        "// while RMS(rmsnorm(last))=0.97 ~ RMS(final_norm)=0.95.)",
    ],
    "(ii) 修正 last 的语义注释（post → pre final-norm）",
)

m = (R / "apps/cli/main.cpp").read_bytes().decode()
h = (R / "src/targets/qwen3_6/impl/runtime/text_prefill_impl.h").read_bytes().decode()
assert m.count('return "nvfp4-group16";') == 1 and m.count('return "e8-group64";') == 1
assert h.count("final-norm **input**") == 1
print("回读校验通过")
