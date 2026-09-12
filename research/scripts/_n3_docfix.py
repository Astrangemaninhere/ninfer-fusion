#!/usr/bin/env python3
"""N3 收尾：两处陈旧注释引用了已被删除的 EngineOptions.kv_calibration_dir。
实测：该字段在 types.h/serve_options.h/layouts.h 全无，CLI/serve 也无旗标
⇒ 真正的门是 NINFER_KV_CALIB_DIR（kvcalib_enabled()）。只改注释，不动行为。"""
import hashlib, pathlib, shutil

R = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime")
BAK = pathlib.Path("/home/user/n3doc_bak"); BAK.mkdir(exist_ok=True)
def md5(p): return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()

E = [
 (R / "kv_calibration.h",
  "// Offline KV calibration capture. When EngineOptions.kv_calibration_dir is set,\n",
  "// Offline KV calibration capture. Gated by kvcalib_enabled() (env\n"
  "// NINFER_KV_CALIB_DIR; the EngineOptions.kv_calibration_dir field this comment\n"
  "// used to name no longer exists),\n"),
 (R / "text_context_impl.h",
  "// KvCalibrationCapture was dead code (EngineOptions.kv_calibration_dir\n"
  "// has no consumer). Env-gated like the S28 loader: NINFER_KV_CALIB_DIR\n",
  "// KvCalibrationCapture was dead code: the EngineOptions field it documented\n"
  "// has since been removed, so the env var IS the gate. Env-gated like the S28\n"
  "// loader: NINFER_KV_CALIB_DIR\n"),
]
for p, old, new in E:
    shutil.copy2(p, BAK / p.name)
    t = p.read_text()
    assert t.count(old) == 1, f"{p.name}: 锚点命中 {t.count(old)} 次"
    p.write_text(t.replace(old, new))
    print(f"ok {p.name}  md5 {md5(BAK / p.name)} -> {md5(p)}")
print("回滚: cp -f /home/user/n3doc_bak/*.h src/targets/qwen3_6/impl/runtime/")
