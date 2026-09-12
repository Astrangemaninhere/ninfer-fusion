#!/usr/bin/env python3
"""恢复 SWA 窗口接线（按真实锚点）：
  * 计算块插在 `out.decoder = qwen3_6::plan_decoder_state(` 之前；
  * 聚合里 `.layer_sliding_windows = layer_windows,` 插在 `.layer_residual` 那行之后；
  * 不加 `requires` 守卫（三个 target config 现在都有 `sliding_window` + `is_swa_attention`）。"""
import pathlib
import sys

P = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/layouts_impl.h")
lines = P.read_text(encoding="utf-8").splitlines(keepends=True)
if any("layer_sliding_windows" in l for l in lines):
    print("已存在 wiring，跳过")
    sys.exit(0)

agg_idx = next((i for i, l in enumerate(lines) if ".layer_residual" in l), None)
assert agg_idx is not None, "找不到 .layer_residual"
# 计算块的插入点：包含 "DecoderStateSpec{" 的那行之前（聚合开始处）
spec_idx = next((i for i, l in enumerate(lines) if "DecoderStateSpec{" in l), None)
assert spec_idx is not None, "找不到 DecoderStateSpec{"
# 往上找该语句的起始行（`out.decoder = plan_decoder_state(`）
stmt_idx = spec_idx
while stmt_idx > 0 and "plan_decoder_state(" not in lines[stmt_idx]:
    stmt_idx -= 1
assert "plan_decoder_state(" in lines[stmt_idx], "找不到 plan_decoder_state( 调用"
base_indent = lines[stmt_idx][:len(lines[stmt_idx]) - len(lines[stmt_idx].lstrip())] + "    "

BLOCK = [
    base_indent + "// E3/S24: per-layer SWA windows from the target config. Muse declares\n",
    base_indent + "// sliding_window (2048) + is_swa_attention() (39 of its 52 layers); the qwen\n",
    base_indent + "// family declares 0 + false, i.e. every layer keeps window 0 = full attention,\n",
    base_indent + "// byte-identical to the behaviour before this wiring (kernels guard on > 0).\n",
    base_indent + "std::array<std::uint32_t, 64> layer_windows{};\n",
    base_indent + "for (std::int32_t i = 0;\n",
    base_indent + "     i < static_cast<std::int32_t>(TextConfig::full_attention_layers()); ++i) {\n",
    base_indent + "    layer_windows[static_cast<std::size_t>(i)] =\n",
    base_indent + "        TextConfig::is_swa_attention(i)\n",
    base_indent + "            ? static_cast<std::uint32_t>(TextConfig::sliding_window)\n",
    base_indent + "            : 0U;\n",
    base_indent + "}\n",
]
lines[stmt_idx:stmt_idx] = BLOCK
# 聚合项：在 .layer_residual 之后插入（aggregate 里缩进更浅）
agg_idx = next(i for i, l in enumerate(lines) if ".layer_residual" in l)
agg_indent = lines[agg_idx][:len(lines[agg_idx]) - len(lines[agg_idx].lstrip())]
lines.insert(agg_idx + 1, agg_indent + ".layer_sliding_windows    = layer_windows,\n")

P.write_text("".join(lines), encoding="utf-8")
print("已恢复 wiring；关键行：")
for i, l in enumerate("".join(lines).splitlines(), 1):
    if "layer_windows" in l:
        print("  %4d %s" % (i, l.rstrip()[:110]))
