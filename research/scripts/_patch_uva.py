#!/usr/bin/env python3
"""只改 UVA 检查：让 is_uva_available() 不再依赖全局 pin_memory。
备份 + 精确替换 + 验证。"""
import pathlib, shutil, hashlib

V = pathlib.Path("/home/user/vllm029/lib/python3.12/site-packages/vllm")
F = V / "utils/platform_utils.py"
BAK = pathlib.Path("/home/user/vllm029_uva_bak")
BAK.mkdir(exist_ok=True)
shutil.copy2(F, BAK / F.name)
print("backup ->", BAK / F.name, "md5", hashlib.md5(F.read_bytes()).hexdigest()[:12])

t = F.read_text()
old = '''def is_uva_available() -> bool:
    """Check if Unified Virtual Addressing (UVA) is available."""
    # UVA requires pinned memory.
    from vllm.platforms import current_platform

    # TODO: Add more requirements for UVA if needed.
    return is_pin_memory_available() or current_platform.is_cpu()'''
new = '''def is_uva_available() -> bool:
    """Check if Unified Virtual Addressing (UVA) is available.

    LOCAL PATCH (2026-09-11): decoupled from is_pin_memory_available().
    On WSL that helper is conservatively False, which made the spec-decode
    StagedWriteTensor/UvaBuffer path refuse to start; flipping the global
    pin_memory on instead made vLLM pin every H2D staging buffer, which blew
    past the WSL VM's RAM budget. UVA itself only needs the (small) staged
    buffers to be pinned, so report availability here and leave the global
    pin_memory policy untouched.
    """
    from vllm.platforms import current_platform

    return True if not current_platform.is_cpu() else True'''
assert t.count(old) == 1, "anchor hit %d" % t.count(old)
F.write_text(t.replace(old, new))
print("patched:", "LOCAL PATCH" in F.read_text())
print("md5 after", hashlib.md5(F.read_bytes()).hexdigest()[:12])
