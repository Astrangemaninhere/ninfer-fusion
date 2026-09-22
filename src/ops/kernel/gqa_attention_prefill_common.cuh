#pragma once

// Shared Qwen3.6 GQA dimensions and leaf PTX helpers used by the independently tuned
// BF16 and INT8 prompt kernels. This file deliberately owns no staging policy,
// shared-memory arena, warp schedule, or kernel body.

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

// S36: 256-reference constant. Geometry-generic kernels MUST use Geometry::HeadDim;
// this value is only an upper bound / legacy reference (see _collab/A_s36_muse_decode.md).
inline constexpr int kGqaPrefillHeadDim = 256;

inline constexpr int kGqaPrefillBr        = 64;
inline constexpr int kGqaPrefillBc        = 64;
inline constexpr int kGqaPrefillThreads   = 128;
inline constexpr int kGqaPrefillSmemBytes = (kGqaPrefillBr + 2 * kGqaPrefillBc) *
                                            kGqaPrefillHeadDim *
                                            static_cast<int>(sizeof(__nv_bfloat16));

// NVFP4 prefill runs a warp-specialized producer/consumer pair. Four producer
// warps dequantize packed K/V into two ping-pong BF16 tiles per tensor while
// four consumer warps run the BF16 tensor-core attention body. Bc=32 keeps the
// four BF16 tiles + Q tile + sync flags inside the sm_120 opt-in smem ceiling.
inline constexpr int kNvfp4PrefillBr      = 64;
inline constexpr int kNvfp4PrefillBc      = 32;
inline constexpr int kNvfp4PrefillThreads = 256;
inline constexpr int kNvfp4PrefillSmemBytes =
    kNvfp4PrefillBr * kGqaPrefillHeadDim * static_cast<int>(sizeof(__nv_bfloat16)) +
    4 * kNvfp4PrefillBc * kGqaPrefillHeadDim * static_cast<int>(sizeof(__nv_bfloat16)) + 64;

// M1 (MTP tree verify): the per-column ancestor masks of one verify round, one uint64 word per
// column, laid out [width, batch] exactly as the runtime hands them over. Bit i of word j means
// "column j may attend column i"; bit 0 is the committed anchor and is always set, so a CHAIN
// round's word j is the prefix (1 << (min(j, extent) + 1)) - 1 and admits nothing the
// position-causal cut does not already admit. `nullptr` is every chain round and every plain
// prefill.
//
// It is a RUNTIME pointer and not a template parameter on purpose: these two metadata types are
// also the metadata of plain prefill and of the chain verify, and both of those paths must keep
// compiling to the code they compiled to before this field existed. A masked launch is refused by
// name in gqa_attention_prompt_launch whenever the route cannot honour it, so no launch can carry
// a non-null mask and then drop it.
struct GqaPrefillDirectMetadata {
    const std::int32_t* table;
    // Always null here: this is the single-row prompt metadata of gqa_attention_cached and of the
    // unbatched gqa_attention_prompt_attention_launch, and neither entry has a parameter that
    // could carry a mask. The member exists so the bf16 prompt body can be a template over both
    // metadata types and read one name.
    const std::uint64_t* column_masks = nullptr;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const { return width; }

    __device__ __forceinline__ const std::int32_t* block_table() const { return table; }
};

template <bool Masked>
struct GqaPrefillBatchMetadata {
    const std::int32_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    std::int32_t table_stride;
    // M1: see the note above. Non-null only for the BF16 prompt verify of a tree round, and only
    // at batch 1: the accessors below reduce this metadata to the SINGLE table row table_rows[0],
    // so a masked launch at batch > 1 is refused in gqa_attention_prompt_launch rather than made
    // to index a row the caller never filled.
    const std::uint64_t* column_masks = nullptr;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const {
        if constexpr (Masked) {
            const std::int32_t valid = valid_columns[0];
            return valid <= 0 ? 0 : (valid < width ? valid : width);
        }
        return width;
    }

    __device__ __forceinline__ const std::int32_t* block_table() const {
        return tables + static_cast<std::int64_t>(table_rows[0]) * table_stride;
    }
};

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_prefill_q_index(int q_head, int d, int token) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(Geometry::HeadDim) *
                                              (static_cast<std::int64_t>(q_head) +
                                               static_cast<std::int64_t>(Geometry::QHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ void gqa_prefill_zero_output_rows(__nv_bfloat16* out, int q_head,
                                                             int row_begin, int row_end, int tid,
                                                             int threads) {
    if (row_begin >= row_end) { return; }
    const int elements = (row_end - row_begin) * Geometry::HeadDim;
    for (int element = tid; element < elements; element += threads) {
        const int row = row_begin + element / Geometry::HeadDim;
        const int d   = element - (row - row_begin) * Geometry::HeadDim;
        out[gqa_prefill_q_index<Geometry>(q_head, d, row)] = __float2bfloat16(0.0f);
    }
}

// XOR-swizzled b16 element address. INT8 operands use the same layout by packing
// two consecutive signed bytes into each b16 lane before ldmatrix.
__device__ __forceinline__ int gqa_prefill_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ unsigned gqa_prefill_swz_addr(unsigned lane_base, unsigned ck,
                                                         unsigned as, unsigned r) {
    return lane_base + ((ck | as) ^ r);
}

} // namespace ninfer::ops
