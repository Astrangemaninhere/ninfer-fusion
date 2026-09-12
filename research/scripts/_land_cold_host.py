#!/usr/bin/env python3
"""N2 遗留落地：ColdPolicy::Host 静默 0 页 -> 响亮。

证据（本次实测）：
  - CLI/serve 都接受 --cold-policy host（apps/cli/options.cpp:169, serve/serve_options.cpp:303）
  - cold_host_bytes(4GiB) 管线一直在（types.h:216 -> layouts.h -> program.h -> program_impl.h:773
    -> layouts_impl.h:921,1108），但没有 evict 消费者
  - layouts_impl.h:118-129 effective_cold_pages 是 S30 唯一真源，Host -> 0 页
守 G-C：不删策略、不缩小能力面；只把静默改成可听可见。
可回滚：备份 + md5 + 打印 patch -R 指令。
"""
import hashlib, pathlib, shutil, sys

R = pathlib.Path("/home/user/ninfer-fusion")
BAK = pathlib.Path("/home/user/cold_host_bak"); BAK.mkdir(exist_ok=True)
def md5(p): return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()

A = R / "src/targets/qwen3_6/impl/runtime/layouts_impl.h"
B = R / "src/targets/qwen3_6/impl/runtime/program_impl.h"
for f in (A, B):
    shutil.copy2(f, BAK / f.name)
    print(f"backup {f.name} md5={md5(f)} -> {BAK}")

# ---- 编辑 1：contradiction 报错（host + 显式冷页数） ----
anchor1 = ("                                                        std::uint32_t explicit_pages) {\n"
           "    switch (policy) {")
add1 = ("                                                        std::uint32_t explicit_pages) {\n"
        "    // Host has no eviction consumer, so it can never supply cold pages. Asking for a\n"
        "    // non-zero cap together with it is a contradiction that used to be swallowed into a\n"
        "    // silently empty cold pool; make the contradiction loud instead.\n"
        "    if (policy == ColdPolicy::Host && explicit_pages != 0) {\n"
        "        throw std::invalid_argument(\n"
        "            \"cold policy 'host' has no eviction consumer yet and resolves to 0 cold pages, \"\n"
        "            \"so --max-cold-pages cannot be honoured; drop the cap or use --cold-policy \"\n"
        "            \"window|disk\");\n"
        "    }\n"
        "    switch (policy) {")
ta = A.read_text()
assert ta.count(anchor1) == 1, f"anchor1 命中 {ta.count(anchor1)} 次"
A.write_text(ta.replace(anchor1, add1))
print("edit1 ok: layouts_impl.h effective_cold_pages 前置矛盾检查")

# ---- 编辑 2：单纯 host 的一次性告警 ----
anchor2 = ("    if (cold_policy == ColdPolicy::Window || cold_policy == ColdPolicy::Disk) {\n"
           "        const std::int32_t requant_heads = decoder->text_kv.batch_layer_view(0).num_kv_heads;")
add2 = ("    if (cold_policy == ColdPolicy::Host) {\n"
        "        // Declared on the CLI but not wired to a host-eviction consumer: the cold pool\n"
        "        // stays at 0 pages (see effective_cold_pages), so the flag buys nothing. Say so\n"
        "        // once at construction rather than letting the layout quietly disagree.\n"
        "        std::fprintf(stderr,\n"
        "                     \"[cold] --cold-policy host is not implemented: cold pool stays at 0 \"\n"
        "                     \"pages; use --cold-policy window|disk\\n\");\n"
        "    }\n"
        "    if (cold_policy == ColdPolicy::Window || cold_policy == ColdPolicy::Disk) {\n"
        "        const std::int32_t requant_heads = decoder->text_kv.batch_layer_view(0).num_kv_heads;")
tb = B.read_text()
assert tb.count(anchor2) == 1, f"anchor2 命中 {tb.count(anchor2)} 次"
B.write_text(tb.replace(anchor2, add2))
print("edit2 ok: program_impl.h 构造期一次性告警")

for f in (A, B):
    shutil.copy2(f, BAK / (f.name + ".new"))
    print(f"new    {f.name} md5={md5(f)}")
print("\n回滚: cp -f %s/*.h /home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/ （用不带 .new 的备份）" % BAK)
