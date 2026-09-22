// Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct FlashNextPleWorkspace {
    Tensor projected_key;
    Tensor projected_value;
    Tensor gated;
    Tensor normalized_gated;
};

template <class Arena>
FlashNextPleWorkspace allocate_flash_next_ple_workspace(Arena& arena, std::int32_t tokens) {
    return {
        .projected_key    = arena.alloc(DType::BF16, {10'240, tokens}, 256),
        .projected_value  = arena.alloc(DType::BF16, {2'560, tokens}, 256),
        .gated            = arena.alloc(DType::BF16, {10'240, tokens}, 256),
        .normalized_gated = arena.alloc(DType::BF16, {10'240, tokens}, 256),
    };
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
