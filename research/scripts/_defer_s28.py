#!/usr/bin/env python3
"""S28 半落地：decoder_state.cpp 调用了 `kv_rowscale_sidecar_apply_from_env`，但
  ① 头文件里没有它的声明（只有 parse/check）；
  ② 定义所在的 `gqa_isoquant_row_scale_loader.cu` **没有登记进 src/CMakeLists.txt**；
⇒ 该符号从未被编译/链接过，整棵树的这一路径一直编不过。
本补丁：把调用点摘掉并留下精确 TODO（该特性由 env 门控、未设时本来就不做事 ⇒ 行为中立），
把"补完 S28"列入下一趟编译队列（三件事：头声明 + CMake 注册 + 那个 .cu 的首次编译验证）。"""
import pathlib

P = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6/impl/state/decoder_state.cpp")
src = P.read_text(encoding="utf-8")
CALL = """    // S28: apply the sidecar before any KV write path can observe the table.
    (void)::ninfer::ops::kv_rowscale_sidecar_apply_from_env(
        static_cast<std::uint32_t>(spec.full_attention_layers),
        static_cast<std::uint32_t>(spec.kv_heads),
        static_cast<std::uint32_t>(spec.attention_head_dim),
        0);  // model artifact hash: wired in round 2 (needs options plumbing)
"""
TODO = """    // S28 (N3) row-scale sidecar is NOT active yet: the call was removed from this
    // build because the patch landed in three-quarters — `kv_rowscale_sidecar_apply_from_env`
    // has no declaration in gqa_isoquant_row_scale_loader.h, and its definition in
    // gqa_isoquant_row_scale_loader.cu is not registered in src/CMakeLists.txt, so the
    // symbol was never compiled or linked (the whole path failed to build). The feature
    // is gated on NINFER_KV_ROWSCALE and does nothing when unset, so removing the call is
    // behaviour-neutral for every current run. Next build window: add the declaration,
    // register the .cu, and compile it for the first time.
"""
if "S28 (N3) row-scale sidecar is NOT active yet" in src:
    print("已是摘除状态")
else:
    assert src.count(CALL) == 1, "调用点锚点不唯一: %d" % src.count(CALL)
    src = src.replace(CALL, TODO)
    P.write_text(src, encoding="utf-8")
    print("已摘除调用点并留 TODO（行为中立：env 未设时该特性不做事）")
