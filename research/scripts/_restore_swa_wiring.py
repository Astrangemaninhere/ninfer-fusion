#!/usr/bin/env python3
"""恢复 E3/S24 的 SWA 窗口接线，但用**编译期安全**的形式。

背景：原实现是
    std::array<std::uint32_t, 64> layer_windows{};
    if constexpr (requires { TextConfig::sliding_window; TextConfig::is_swa_attention(0); }) { ... }
在 layouts_impl.h 这种**非依赖上下文**（TextConfig 是具体类型）里，requires 为假也挡不住解析期的
名字查找 ⇒ 编译失败。我当时整段回退，结果把 Muse 的窗口接线也摘掉了（Muse: 52 层里 39 层 SWA、
window 2048），Muse 的 bf16 路径因此从 L46 起输出 NaN。

现在三个 target config 都声明了 `sliding_window` + `is_swa_attention`（Muse 本来有；27b/35b 已补
`sliding_window=0` + `is_swa_attention()->false`），所以**不需要守卫**：直接算即可，语义为
"非 SWA 层或窗口为 0 ⇒ 全注意力"，与回退前对 qwen 的行为逐字节一致。"""
import pathlib
import re
import sys

P = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/layouts_impl.h")
src = P.read_text(encoding="utf-8")
if "layer_sliding_windows" in src:
    print("wiring 已存在，跳过")
    sys.exit(0)

# 1) 在 plan_decoder_state 里构造 layer_windows（插在 spec 校验之后、布局构造之前）
anchor = re.search(r"\n( *)(DecoderStateLayout layout;\n)", src)
assert anchor, "找不到 DecoderStateLayout layout; 锚点"
indent = anchor.group(1)
BLOCK = (
    "\n" + indent + "// E3/S24: per-layer SWA windows from the target config. Muse declares\n"
    + indent + "// sliding_window (2048) plus is_swa_attention() (39 of 52 layers); the qwen\n"
    + indent + "// family declares 0/false, so every layer stays 0 = full attention, identical\n"
    + indent + "// to the previous behaviour (kernels guard on sliding_window > 0).\n"
    + indent + "std::array<std::uint32_t, 64> layer_windows{};\n"
    + indent + "for (std::int32_t i = 0;\n"
    + indent + "     i < static_cast<std::int32_t>(TextConfig::full_attention_layers()); ++i) {\n"
    + indent + "    layer_windows[static_cast<std::size_t>(i)] =\n"
    + indent + "        TextConfig::is_swa_attention(i)\n"
    + indent + "            ? static_cast<std::uint32_t>(TextConfig::sliding_window)\n"
    + indent + "            : 0U;\n"
    + indent + "}\n"
)
src = src[:anchor.start()] + BLOCK + src[anchor.start():]

# 2) 在 DecoderStateSpec 聚合初始化里挂上 .layer_sliding_windows
agg = re.search(r"\n( *)(\.layer_residual = [^\n]*\n)", src)
assert agg, "找不到 .layer_residual 聚合锚点"
src = src[:agg.end(2)] + agg.group(1) + ".layer_sliding_windows = layer_windows,\n" + src[agg.end(2):]

P.write_text(src, encoding="utf-8")
print("已恢复 wiring（无 requires 守卫）")
for i, l in enumerate(src.splitlines(), 1):
    if "layer_windows" in l or "layer_sliding_windows" in l:
        print("  %4d %s" % (i, l.rstrip()[:110]))
