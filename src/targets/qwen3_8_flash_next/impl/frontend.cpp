// Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
#include "targets/qwen3_8_flash_next/impl/frontend.h"

namespace ninfer::targets::qwen3_8_flash_next::detail {

qwen3_6::Frontend make_frontend(const StandaloneLoadedModel& model,
                                const qwen3_6::FrontendOptions& options) {
    return qwen3_6::make_frontend(model.frontend_resources(), options);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail