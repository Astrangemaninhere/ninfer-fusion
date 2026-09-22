#pragma once

#include "core/cyclic_kv_cache.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Host launch-resource promise for device-selected prefix append. It bounds every device count
 * during capture/replay but neither selects nor publishes a committed frontier.
 */
struct KVCacheAppendPrefixExecutionEnvelope {
    std::uint32_t min_count = 0;
    std::uint32_t max_count = 0;
};

/**
 * The e8 lattice family's table handle, as the OP LAYER can see it: two DEVICE pointers,
 * supplied by the caller.
 *
 * WHY THIS IS A HANDLE AND NOT AN OWNER
 * -------------------------------------
 * The 3-bit and 2-bit e8 K plate is written by an E8 lattice codec that needs 6 428 B of
 * stage-1/stage-2 tables (4 368 + 2 060; `ops/kernel/e8_lattice_kv_plane.cuh:74-75`). The
 * host-side contents have a codec-of-record accessor (`ops/kv/e8_lattice_plane_codec.cuh`'s
 * `e8_lattice_stage1_table()` / `e8_lattice_stage2_table()`), but the DEVICE arm reads the
 * tables from device memory, and `kv_cache_append` owns no allocation by contract ("The Op
 * owns no persistent allocation" -- this header, above). So where the 6 428 B live is the
 * caller's decision and this op only borrows them. A null pair is refused by name rather
 * than launched, because a null handle would fault on device instead of refusing on host.
 *
 * WHY `const void*` AND NOT THE CODEC'S OWN TYPE
 * ----------------------------------------------
 * The codec's `E8KvLatticeTables` is declared in a CUDA-only header that carries
 * `__device__` members, so it cannot appear in this header or in the host-compiled TU that
 * implements the admission. These two members are typed as the raw device addresses and are
 * cast to the codec's types exactly once, inside the CUDA TU that launches the arm.
 *
 * SWAPPING THE TWO IS A WRONG-PLANE READ, so they are named rather than positional.
 */
struct E8KvAppendTables {
    const void* stage1 = nullptr;   // device E8LatticeStage1, 4 368 B
    const void* stage2 = nullptr;   // device E8LatticeStage2, 2 060 B
};

/**
 * Append every K/V row to single-sequence paged growing-cache storage.
 *
 * k/v are contiguous BF16 [256,4|2,T] and positions is contiguous sequential device I32 [T].
 * BF16 cache rows are copied bit-for-bit. INT8-G64 cache rows use one scale for each contiguous
 * 64-value group. For codec input values x, the persistent INT8 group encoding is
 *
 *   a          = max_i abs(FP32(x[i]))
 *   scale_bits = FP16_RNE(a / 127)
 *   s          = FP32(scale_bits)
 *   inv        = s == 0 ? 0 : FP32(1 / s)
 *   code[i]    = s == 0 ? 0 : I8(clamp(RNE_even(FP32(x[i]) * inv), -127, 127))
 *   decode[i]  = FP32(code[i]) * s.
 *
 * FP8_E4M3FN cache rows use one FP16 scale for the complete D256 row:
 *
 *   a          = max_i abs(FP32(x[i]))
 *   a == 0: scale_bits=FP16(+0), code[i]=E4M3FN(+0)
 *   a != 0: raw_scale  = a / 448
 *           scale_bits = FP16_RNE(clamp(raw_scale, 0x1p-24, 65504))
 *           s          = FP32(scale_bits)
 *           inv        = FP32(1 / s)
 *           code[i]    = E4M3FN_RNE_SATFINITE(FP32(x[i]) * inv)
 *   decode[i] = FP32(E4M3FN(code[i])) * s.
 *
 * V uses represented BF16 source values directly as x. For both quantized profiles, K is a paired
 * physical representation for causal Attention: its implementation-owned fixed orthogonal
 * preparation selects x, and the causal consumer applies the matching private Q preparation. The
 * transform and raw K code/scale bytes are not standalone mathematical outputs. Standalone and
 * fused append produce the same consumable K representation. Every addressed code/value and scale
 * is overwritten, and no unrelated cache row is read or written. Inputs and every cache
 * plane/table are pairwise non-overlapping. The Op owns no persistent allocation, frontier,
 * request identity, or commit authority.
 *
 * THE e8 FAMILY IS REFUSED HERE BY NAME. This overload writes ONE code per element ({fp8, i8,
 * bf16} arm, no e8 arm), so it cannot fill a packed K plate: over a 128/96/64-byte e8 row
 * stride its BF16 fallback would write 256 bf16 elements per row. An e8 pool therefore
 * appends through the tables-carrying overload below, or through the gqa e8 prefill launch.
 */
void kv_cache_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                     PagedKVLayerView cache, cudaStream_t stream);

/**
 * Append every K/V row to a paged growing cache laid out at its OWN e8 tier extent.
 *
 * THIS IS THE ADMISSION FOR THE PACKED e8 FAMILY, AND IT IS A SEPARATE OVERLOAD ON PURPOSE.
 * `cache.dtype` selects both the K row extent and the arm: E8K3Kv is a 96-byte K plate and
 * E8K2Kv a 64-byte one, beside a V plate that this family does NOT narrow (128 B/row at every
 * width -- product/kv_e8_width.h:131-137). The extents are not restated here; they are read
 * from `d256_kv_cache_profile(cache.dtype)`, the same table the unpacked tiers are checked
 * against, so this overload cannot admit a width the geometry of record does not publish.
 *
 * The K and V scales use the family's g64 profile: one FP16 per 64-channel group, 4 groups per
 * row, on BOTH planes.
 *
 * What a caller must guarantee, in addition to the contract above:
 *   * `tables.stage1` / `tables.stage2` are DEVICE addresses of the codec's stage-1 and
 *     stage-2 tables (4 368 + 2 060 B), valid for the stream. A null pair is refused.
 *   * the pool is laid out at `d256_kv_cache_profile(cache.dtype)`: k_pages at
 *     `code_leading_extent`, v_pages at `v_code_leading_extent`, both scale planes at
 *     `scale_leading_extent`. A pool at any other extent, including the unpacked 256 or the
 *     4-bit tier's 128 K plate, is refused rather than written.
 *
 * E8Kv (the shipped 4-bit tier) is NOT served here: its K plate is the packed i4 codec at
 * 128 B/row and writing a 96/64-byte lattice plate into it is the corruption the extent check
 * refuses. E8Kv appends through the gqa e8 prefill launch. A non-e8 dtype keeps the exact
 * behaviour of the overload above (it is forwarded to it).
 */
void kv_cache_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                     PagedKVLayerView cache, E8KvAppendTables tables, cudaStream_t stream);

/**
 * Append device-selected exact BF16 prefixes to batched paged growing-cache storage.
 *
 * k/v are contiguous BF16 [128,8,T,B], positions is contiguous device I32 [T,B], and counts and
 * table_rows are contiguous device I32 [B]. For row b and i in [0,counts[b]), k/v[:, :, i, b]
 * are copied bit-for-bit to logical position positions[i,b] through table row table_rows[b]. The
 * paged planes use head-major order [128,64,Nphysical,8]. No byte belonging only to the rejected
 * physical tail [counts[b],T) is written. Inputs are unchanged, and the Op neither decides nor
 * publishes a committed frontier.
 *
 * The caller guarantees T>0, B=1..8, envelope.min_count <= counts[b] <= envelope.max_count <= T,
 * sequential nonnegative live positions, materialized table entries for every position allowed by
 * the envelope, and pairwise non-aliasing of inputs and cache storage.
 */
void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& table_rows,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            PagedKVBatchLayerView cache, cudaStream_t stream);

/**
 * Append device-selected exact BF16 prefixes to lane-owned cyclic storage.
 *
 * k/v, positions, counts, and their exact-copy and mutation contracts match the paged overload;
 * lanes[b] selects the destination lane. The registered geometry is D=128, Hkv=8, capacity equal
 * to the window in {2048, 4096}, and absolute position p maps to slot p mod window. The caller
 * guarantees that each row's existing live interval ends immediately before positions[0,b],
 * advancing it by counts[b] makes every overwritten old slot dead, and one row commits at most the
 * ring capacity. Consequently, no two live writes race for one physical slot. The Op does not own
 * or publish the lane frontier.
 */
void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& lanes,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            CyclicKVCacheLayerView cache, std::uint32_t window,
                            cudaStream_t stream);

} // namespace ninfer::ops
