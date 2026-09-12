#pragma once

// FlashNext (Qwen4Exp) package 占位 — P0 骨架。
// 完整 package (frontend/bindings/weights profile) 依赖:
//   1) MoE swiglu 专家内核 (new_op 工作包)
//   2) PLE n-gram gather 表 (sidecar 构建器已闭环, 引擎真表 gather 待验证)
//   3) 多层 MTP (现有单层 MTP 扩展)
// 本占位仅注册 identity, 供 registry 编译与 GUI 引擎列表可见。

#include "impl/config.h"

namespace ninfer::targets::qwen4_exp {{

inline constexpr const char* kModelId = "qwen4-exp";

}} // namespace ninfer::targets::qwen4_exp
