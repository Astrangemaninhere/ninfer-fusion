#!/usr/bin/env python3
"""E3/S24 窗口接线的生产者缺失：`layouts_impl.h` 用
    if constexpr (requires { TextConfig::sliding_window; TextConfig::is_swa_attention(0); })
判断"该变体是否声明了窗口成员"，但这里的 TextConfig 是**非依赖**的具体类型 ⇒ 名字查找在解析期
发生，requires 为假也挡不住硬错。修法不是改 guards，而是把生产者补上：qz 家族本来就是全注意力，
声明 `sliding_window = 0` + `is_swa_attention() -> false` ⇒ requires 为真、循环把窗口全填 0，
与"全零表 = 全注意力"完全等价（Muse 已有这两个成员，不动）。"""
import pathlib

ANCHOR_MUSE_STYLE = "    [[nodiscard]] static constexpr bool qk_norm_enabled()"

BLOCK = """    // E3/S24: the shared window wiring in layouts_impl.h binds
    // TextConfig::sliding_window + TextConfig::is_swa_attention when a variant
    // declares them. This family is full attention on every layer, so declare the
    // no-op values explicitly (all-zero window table = full attention, identical to
    // the previous behaviour) instead of letting the guard hit a name lookup error.
    static constexpr int sliding_window = 0;
    [[nodiscard]] static constexpr bool is_swa_attention(int /*layer*/) { return false; }

"""

for rel in ("src/targets/qwen3_6_27b/impl/config.h", "src/targets/qwen3_6_35b_a3b/impl/config.h"):
    P = pathlib.Path("/home/user/ninfer-fusion") / rel
    src = P.read_text(encoding="utf-8")
    if "is_swa_attention" in src:
        print("%s: 已有成员，跳过" % rel)
        continue
    idx = src.find(ANCHOR_MUSE_STYLE)
    if idx == -1:
        # 退而求其次：插到 sliding_window/qk_norm 相关区段的开头
        idx = src.find("struct TextConfig {")
        idx = src.find("\n", idx) + 1 if idx != -1 else -1
        assert idx != -1, "%s: 找不到插入点" % rel
    src = src[:idx] + BLOCK + src[idx:]
    P.write_text(src, encoding="utf-8")
    print("%s: 已补 sliding_window=0 + is_swa_attention()" % rel)
