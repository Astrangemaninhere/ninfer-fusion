#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// The YaRN extension multiplies the native key budget by 4 (include/ninfer/ops/softmax_attention.h:18
// spells the same bound as kCausalAttentionMaximumVisibleKeysYarn = 4 * 262144), and every decode
// kernel already sizes its page-id table for exactly it (src/ops/kernel/paged_kv_address.cuh:47
// static_asserts paged_kv_page_ids(4 * 262144) == 256). The bound below is that same number, so a
// --yarn context of the full 4x native capacity is admitted instead of being refused here while
// the kernels can already index it. Raising an upper-bound test only admits the interval
// (1'010'000, 1'048'576]; every value at or below the old bound decides as it did before.
inline constexpr std::uint32_t kGqaAttentionMaximumVisibleKeys = 1'048'576;

struct GqaExecutionEnvelope {
    std::uint32_t min_visible_keys = 0;
    std::uint32_t max_visible_keys = 0;
    // Split-geometry reference for the split-KV small-T decode path: the key count the
    // split partition is derived from. It has to be a constant of the produced graph
    // (the sequence key capacity), because the batch-1 decode of a row and the wide MTP
    // verify of the same row must partition that row's keys identically -- a partition
    // that follows the live window reassociates the fp32 split reduction and flips
    // ULP-level ties between arms (k=3 vs k=9, plain vs speculative). 0 keeps the legacy
    // window-driven partition.
    //
    // SCOPE -- this field alone does not make the sentence above true, and before the route
    // clause below it did not hold at all for a wide verify. Only the split-KV small-T routes
    // consume it: gqa_small_t_split_reference -> gqa_small_t_split_units
    // (src/ops/launcher/gqa_attention_decode_split.h:73-79,
    // src/ops/kernel/gqa_attention_decode.cuh:114-122). The prompt route never sees it, because
    // gqa_attention_prompt_launch takes no GqaExecutionEnvelope at all
    // (src/ops/launcher/gqa_attention.h:69-72); it partitions keys into kGqaPrefillBc blocks with
    // its own online softmax (src/ops/kernel/gqa_attention_prefill_bf16.cuh:140,251,263,394-395)
    // and a row's reduction stops being a function of the row's own key prefix. A caller that
    // pins this field therefore also requires that its (q_heads, width, batch) triple does not
    // resolve to GqaAttentionRoute::Prompt; gqa_attention_route_for
    // (src/ops/launcher/gqa_attention_route_contract.h) enforces that by sending a pinned
    // multi-column verify to the chunked small-T route. Gqa16x4Geometry (16 q-heads / 4 kv-heads)
    // is the one named exception, and the reason is in that header.
    std::uint32_t split_reference_keys = 0;
};

/**
 * Shared numerical contract for A1/A2/A3.
 *
 * Public q/k/v inputs and BF16 cache values are interpreted after their BF16 storage boundary.
 * INT8-G64 cache rows use one FP16 scale for each contiguous 64-element group. For BF16 source
 * values x, their exact observable encoding is:
 *
 *   a          = max_i abs(FP32(x[i]))
 *   scale_bits = FP16_RNE(a / 127)
 *   s          = FP32(scale_bits)
 *   inv        = s == 0 ? 0 : FP32(1 / s)
 *   code[i]    = s == 0 ? 0 : I8(clamp(RNE_even(FP32(x[i]) * inv), -127, 127))
 *   decode[i]  = FP32(code[i]) * s
 *
 * A1 and A2 produce identical code and scale bits. The common ideal attention oracle uses BF16 Q
 * and logical cache values (BF16 values for a BF16 cache, FP32 decode above for INT8-G64), then
 * evaluates score dot products, stable softmax, and value reduction in FP64. The BF16 Op output is
 * promoted to FP64 for comparison with that result.
 *
 * The registered INT8 implementation defines Q8-G64, paired with INT8-G64 K, as its native query
 * compute profile. Its profile-defined query quantization and any narrower staging do not replace
 * BF16 Q in the ideal oracle. BF16-cache and INT8-cache compute profiles therefore have separate
 * named numerical criteria owned by the GQA conformance test. Those envelopes apply to the
 * registered geometries, tested token extents, conformance matrix, and target-representative
 * activation range; they are not a universal error bound for arbitrary adversarial BF16 tensors.
 * A1 and A3 are each qualified directly against the ideal oracle. A1-versus-A3 parity is only an
 * additional consistency check.
 */

/**
 * Returns the transient arena capacity required for every W in the inclusive interval at one
 * exact logical batch size. Head geometry, cache dtype, and execution envelope are the fixed
 * implementation profile. Invalid profiles or intervals throw; a legal B=1 prompt route may
 * return zero.
 */
[[nodiscard]] std::size_t
gqa_attention_workspace_capacity_bytes(std::int32_t head_dim, std::int32_t q_heads,
                                          DType cache_dtype,
                                       GqaExecutionEnvelope envelope, std::int32_t batch_size,
                                       std::int32_t min_width, std::int32_t max_width);

/**
 * A1: append K/V for B independent sequences and compute causal grouped-query attention. Let
 * Vb=W when valid_columns is empty and Vb=valid_columns[b] otherwise. For row b, query head h,
 * kvh=floor(h/group), 0<=j<Vb, p=positions[j,b], and that row's populated cache history [0,p]:
 *
 *   score[x]      = scale * dot(q[:,h,j,b], K_cache[b][:,x,kvh]), 0 <= x <= p
 *   probability   = softmax_x(score)
 *   ideal[:,h,j,b] = sum_x probability[x] * V_cache[b][:,x,kvh].
 *
 * The registered geometries are `[256,24|4,W,B]` group 6, `[256,16|2,W,B]` group 8, and
 * `[256,16|4,W,B]` group 4 (Spark-X2.5; bf16/fp8/iso4e KV tiers).
 * q/k/v/out are contiguous BF16 in request-major order, positions is contiguous I32 [W,B], and
 * kv_table_rows is contiguous I32 [B]. valid_columns is either contiguous I32 [B], or an empty
 * Tensor meaning every row has exactly W valid columns. This dense/masked choice is part of the
 * call topology; it is not inferred by copying device metadata to the host. B=1 accepts every
 * positive W in the current prefill/decode domain; B=2..8 accepts W=1..16. Cache storage is BF16
 * or INT8-G64 under the shared numerical contract above. PagedKVBatchLayerView supplies shared
 * planes and the complete block-table matrix; kv_table_rows[b] selects one row for sequence b.
 *
 * In masked form, every row's valid columns are the prefix [0,valid_columns[b]); positions in that
 * prefix are sequential and address populated causal histories. Each nonempty row repeats its
 * final valid position through the invalid tail; an empty row uses zero positions. Other
 * invalid-tail inputs contain safe dummy values. A1 does not modify cache for invalid columns and
 * writes exact BF16 zero to their output. The caller guarantees that the maximum final valid
 * position plus one over nonempty rows lies in the declared execution envelope. The envelope is a
 * host launch-resource promise over that batch maximum; it does not alter any row's causal mask.
 *
 * column_masks is either an empty Tensor (the dense spelling, and the only spelling this Op's
 * unchanged callers use) or I64 [W,B]: bit i of column_masks[b][j] selects whether verify column j
 * may attend column i of the SAME round block. Bit 0 is the anchor column and must be set. Keys
 * whose cache position lies below the block's first position are the shared causal history and stay
 * unconditionally visible, so a mask equal to the causal prefix (1 << (j+1)) - 1 -- exactly what a
 * chain's column j means -- removes no key and is identical to passing no mask. An ancestor mask
 * that is NOT such a prefix (a real tree: a sibling is never an ancestor) hides keys the
 * position-causal cut would have exposed, which is the whole point of the argument.
 *
 * It is consumed by the BF16 decode (small-T) routes and -- at B = 1, which is the only batch size
 * the prompt route is reachable at -- by the BF16 prompt body, which applies the same cut at the
 * same place. Every other combination is REFUSED BY NAME rather than silently exempted: this Op
 * refuses a non-BF16 cache and a round wider than the verify width, and
 * gqa_attention_prompt_launch refuses a batch above 1, a width above one query block, or a
 * non-BF16 tier reaching it by another path. Silently dropping a mask would verify a chain and
 * call it a tree.
 *
 * q/k/v/positions/valid_columns/kv_table_rows/out, every cache plane/table, and live workspace
 * suballocations are pairwise non-overlapping. The Op overwrites every addressed cache row but
 * owns no persistent frontier, allocation, request identity, or commit authority.
 */
void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   const Tensor& valid_columns, const Tensor& column_masks,
                   const Tensor& kv_table_rows, float scale, PagedKVBatchLayerView cache,
                   GqaExecutionEnvelope envelope, WorkspaceArena& workspace, Tensor& out,
                   cudaStream_t stream);

/**
 * A2: perform only the cache-write part of A1. k/v are contiguous BF16 `[256,4|2,T]`, positions is
 * contiguous sequential I32 [T], and every addressed code and INT8 scale is overwritten. It reads
 * no unrelated cache row, receives no execution envelope, and owns no persistent frontier.
 */
void gqa_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                   PagedKVLayerView cache, cudaStream_t stream);

/**
 * A3: compute causal attention from an already populated cache without accepting new K/V or
 * mutating any cache plane. q/out are contiguous BF16 `[256,24|16,T]`, positions is contiguous
 * sequential I32 [T], and the mathematical formula and execution-envelope contract are identical
 * to A1. Caller workspace is reported by gqa_attention_workspace_capacity_bytes().
 */
void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale,
                          const PagedKVLayerView& cache, GqaExecutionEnvelope envelope,
                          WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
