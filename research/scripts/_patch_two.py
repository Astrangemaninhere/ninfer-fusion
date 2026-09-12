#!/usr/bin/env python3
"""两处补丁（共用一次重编）：
 A. dflash2_impl.h：NINFER_DF2SCORES 块里追加 frontier/anchor 两个 aux dump —— 给 hit@b 提供唯一对齐键。
 B. bf16_gemm_mma.cu：补 Bf16GemvGeometry<248320,5120> 与 <131072,5120>，解锁 BF16 词汇表头。
行尾自动探测；每处锚点断言恰好 1 次。
"""
import pathlib
import sys

R = pathlib.Path("/home/user/ninfer-fusion")


def patch(rel, anchor_lines, repl_lines, label, expect=1):
    p = R / rel
    raw = p.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    BAK = pathlib.Path("/home/user/bf16head_bak") / rel
    BAK.parent.mkdir(parents=True, exist_ok=True)
    if not BAK.exists():
        BAK.write_bytes(raw.encode("utf-8"))
        print(f"  备份 -> {BAK}")
    anchor = nl.join(anchor_lines)
    n = raw.count(anchor)
    if n != expect:
        print(f"FAIL {label}: 锚点命中 {n} 次（期望 {expect}）")
        sys.exit(3)
    out = raw.replace(anchor, nl.join(repl_lines), expect)
    p.write_bytes(out.encode("utf-8"))
    print(f"OK  {label}")


# ---- A. dump 轮次标记 ----
patch(
    "src/targets/qwen3_6/impl/runtime/dflash2_impl.h",
    [
        '            dump_one("scores", scores);',
        '            dump_one("cand", candidates);',
        '            dump_one("unary", unary);',
    ],
    [
        '            dump_one("scores", scores);',
        '            dump_one("cand", candidates);',
        '            dump_one("unary", unary);',
        '            // 轮次标记：frontier 是该轮起始执行前沿、anchor 是上一轮被接受的 token；',
        '            // 二者合起来给离线 hit@b 提供唯一对齐键（原先只用 draft[0] 会在常用标点上撞车）。',
        '            dump_one("front", frontiers);',
        '            dump_one("anch", anchors);',
    ],
    "A dump 追加 frontier/anchor 标记",
)

# ---- B. bf16 MMA 全词表头 ----
patch(
    "src/ops/linear/bf16/bf16_gemm_mma.cu",
    [
        "    if (weight.n == 256 && weight.k == 5120) {",
        "        launch_geometry<Bf16GemvGeometry<256, 5120>>(x, weight, out, stream);",
        "        return;",
        "    }",
        '    throw std::invalid_argument("bf16 linear MMA: unsupported exact problem");',
    ],
    [
        "    if (weight.n == 256 && weight.k == 5120) {",
        "        launch_geometry<Bf16GemvGeometry<256, 5120>>(x, weight, out, stream);",
        "        return;",
        "    }",
        "    // 词汇表头/短名单头：BF16 档 artifact 需要（head 量化误差的对照面）。",
        "    if (weight.n == 248320 && weight.k == 5120) {",
        "        launch_geometry<Bf16GemvGeometry<248320, 5120>>(x, weight, out, stream);",
        "        return;",
        "    }",
        "    if (weight.n == 131072 && weight.k == 5120) {",
        "        launch_geometry<Bf16GemvGeometry<131072, 5120>>(x, weight, out, stream);",
        "        return;",
        "    }",
        '    throw std::invalid_argument("bf16 linear MMA: unsupported exact problem");',
    ],
    "B bf16 MMA 补全词表两形",
)

h = (R / "src/targets/qwen3_6/impl/runtime/dflash2_impl.h").read_bytes().decode()
m = (R / "src/ops/linear/bf16/bf16_gemm_mma.cu").read_bytes().decode()
assert h.count('dump_one("front", frontiers)') == 1 and h.count('dump_one("anch", anchors)') == 1
assert m.count("Bf16GemvGeometry<248320, 5120>") == 1 and m.count("Bf16GemvGeometry<131072, 5120>") == 1
print("回读校验通过")
