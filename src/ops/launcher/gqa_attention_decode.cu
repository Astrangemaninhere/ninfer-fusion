// ninfer::ops - split-KV GQA small-T launcher and unified route dispatcher.
#include "ops/launcher/gqa_attention.h"
#include "ops/launcher/gqa_attention_decode_partial.cuh"

#include "ops/common/ft_stats.h"
#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_decode_bf16.cuh"
#include "ops/kernel/gqa_attention_decode_fp8.cuh"
#include "ops/kernel/gqa_attention_decode_iso3.cuh"
#include "ops/kernel/gqa_attention_decode_i8.cuh"
#include "ops/kernel/gqa_attention_decode_nvfp4.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/gqa_attention.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

bool gqa_attention_uses_small_t(std::int32_t tokens) { return tokens >= 1 && tokens <= 6; }

std::int32_t gqa_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                          DType cache_dtype, GqaExecutionEnvelope envelope) {
    if (tokens < 1 || tokens > 6 ||
        (cache_dtype != DType::BF16 && cache_dtype != DType::I8 &&
         cache_dtype != DType::NVFP4 && cache_dtype != DType::FP8_E4M3FN &&
         cache_dtype != DType::ISO3 && cache_dtype != DType::E8Kv) ||
        envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys) {
        throw std::invalid_argument("gqa_attention split capacity: invalid profile");
    }
    if (q_heads == Gqa27Geometry::QHeads) {
        return gqa_small_t_launch_capacity<Gqa27Geometry>(envelope, tokens, cache_dtype);
    }
    if (q_heads == Gqa35Geometry::QHeads) {
        return gqa_small_t_launch_capacity<Gqa35Geometry>(envelope, tokens, cache_dtype);
    }
    if (q_heads == GqaMuseGeometry::QHeads) {
        return gqa_small_t_launch_capacity<GqaMuseGeometry>(envelope, tokens, cache_dtype);
    }
    throw std::invalid_argument("gqa_attention split capacity: unsupported head geometry");
}

} // namespace ninfer::ops::detail
