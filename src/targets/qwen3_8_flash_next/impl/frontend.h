// Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
#pragma once

#include "targets/qwen3_8_flash_next/impl/load/loader.h"
#include <ninfer/targets/qwen3_6/frontend.h>

namespace ninfer::targets::qwen3_8_flash_next::detail {

[[nodiscard]] qwen3_6::Frontend make_frontend(const StandaloneLoadedModel& model,
                                              const qwen3_6::FrontendOptions& options);

} // namespace ninfer::targets::qwen3_8_flash_next::detail