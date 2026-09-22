#pragma once

// Implements: include/ninfer/ops/mtp_round.h
// Match: request-major fixed K=1..5 autoregressive MTP round transition.

#include "ninfer/ops/mtp_round.h"
#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

__global__ void mtp_prepare_next_round_kernel(
    const std::int32_t* verify_ids, const std::int32_t* next_anchors, const std::int32_t* accepted,
    const std::int32_t* updated_frontiers, const std::int32_t* remaining_budgets,
    const std::int32_t* licensed_counts, const std::int32_t* rope_deltas,
    std::int32_t* alignment_ids, std::int32_t* next_extents, std::int32_t* ar_positions,
    std::int32_t* ar_rope_positions, std::int32_t* ar_valid_columns, std::int32_t k,
    std::int32_t ar_step_stride, std::int32_t max_context,
    const std::int32_t* __restrict__ svip_cuts) {
    const int row = static_cast<int>(blockIdx.y);
    const int T   = k + 1;
    int a         = accepted[row];
    a             = a < 0 ? 0 : (a > k ? k : a);
    for (int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x; j < T;
         j += blockDim.x * gridDim.x) {
        alignment_ids[row * T + j] = j < a ? verify_ids[row * T + j + 1] : next_anchors[row];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int licensed       = licensed_counts[row];
        const int remaining      = remaining_budgets[row] - licensed;
        const int budget_extent  = remaining > 1 ? remaining - 1 : 0;
        const int context_extent = max_context - updated_frontiers[row] - 1;
        int next                 = budget_extent < context_extent ? budget_extent : context_extent;
        next                     = next < 0 ? 0 : (next > k ? k : next);
        if (svip_cuts != nullptr) {
            const int cut = svip_cuts[row];
            next = cut < next ? (cut < 0 ? 0 : cut) : next;
        }
        next_extents[row]        = next;
        const int steps          = k > 1 ? k - 1 : 1;
        for (int s = 0; s < steps; ++s) {
            const int offset          = s * ar_step_stride + row;
            const int position        = updated_frontiers[row] + s;
            ar_positions[offset]      = position;
            ar_rope_positions[offset] = position + rope_deltas[row];
            // Step s is live iff it is inside the next round's draft count. `s + 1 < next`
            // dropped the last draft whenever the budget was exactly enough (next == steps);
            // the two forms are identical for next > steps.
            ar_valid_columns[offset]  = s < next ? 1 : 0;
        }
    }
}


// Adaptive draft window (auto mode): grow the window by one when the entire live window
// was accepted, collapse to just past the last accepted draft on any rejection. Stateless
// on purpose: it reads only this round's accepted count and the extent that was actually
// drafted, so no persistent per-request state is needed. Cost model: round(k) = a + b*k
// with b/a ~ 0.06, so draft column i pays off only while the per-column hit rate stays
// above that ratio, and the round in flight is the cheapest available estimate of it.
__global__ void mtp_adaptive_extents_kernel(const std::int32_t* accepted,
                                            const std::int32_t* current_extents,
                                            std::int32_t* cuts, std::int32_t k_max,
                                            std::int32_t batch) {
    const int row = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
    if (row >= batch) { return; }
    const int a = accepted[row] < 0 ? 0 : accepted[row];
    const int e = current_extents[row] < 0 ? 0 : current_extents[row];
    int next = (e > 0 && a >= e) ? e + 1 : a + 1;
    if (next > k_max) { next = k_max; }
    if (next < 1) { next = 1; }
    cuts[row] = next;
}

// SVIP: entropy-based draft length cap. One block per row scans the verify
// columns col = accepted+1 .. k (the positions that predict next round's
// drafts) and caps the draft count before the first column whose softmax
// entropy (nats) exceeds the threshold. A fully accepted row keeps the
// caller-extent behavior (cap = k).
__global__ void mtp_svip_entropy_extents_kernel(
    const __nv_bfloat16* __restrict__ logits, const std::int32_t* __restrict__ accepted,
    std::int32_t* __restrict__ cuts, int vocab, int cols, int batch, float threshold) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= batch) { return; }
    const int tid       = static_cast<int>(threadIdx.x);
    const int a         = accepted[row];
    const std::int64_t v_stride = static_cast<std::int64_t>(cols) * batch;
    const __nv_bfloat16* row_base = logits + row;  // layout [vocab][cols][batch]
    const int k = cols - 1;
    cuts[row] = k;

    for (int col = a + 1; col < cols; ++col) {
        const __nv_bfloat16* col_base = row_base + static_cast<std::int64_t>(col) * batch;
        float m = -CUDART_INF_F;
        for (int v = tid; v < vocab; v += blockDim.x) {
            m = fmaxf(m, __bfloat162float(col_base[static_cast<std::int64_t>(v) * v_stride]));
        }
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        }
        __shared__ float s_max;
        if (tid == 0) { s_max = m == -CUDART_INF_F ? 0.0F : m; }
        __syncthreads();
        m = s_max;

        float sum = 0.0F;
        float sln = 0.0F;
        for (int v = tid; v < vocab; v += blockDim.x) {
            const float x = __bfloat162float(col_base[static_cast<std::int64_t>(v) * v_stride]);
            const float e = __expf(x - m);
            sum += e;
            sln += e * (x - m);
        }
        __shared__ float s_sum[8];
        __shared__ float s_sln[8];
        const int warp = tid >> 5;
        const int lane = tid & 31;
        for (int off = 16; off > 0; off >>= 1) {
            sum = sum + __shfl_xor_sync(0xffffffffu, sum, off);
            sln = sln + __shfl_xor_sync(0xffffffffu, sln, off);
        }
        if (lane == 0) { s_sum[warp] = sum; s_sln[warp] = sln; }
        __syncthreads();
        if (warp == 0) {
            float tsum = lane < (blockDim.x >> 5) ? s_sum[lane] : 0.0F;
            float tsln = lane < (blockDim.x >> 5) ? s_sln[lane] : 0.0F;
            for (int off = 16; off > 0; off >>= 1) {
                tsum = tsum + __shfl_xor_sync(0xffffffffu, tsum, off);
                tsln = tsln + __shfl_xor_sync(0xffffffffu, tsln, off);
            }
            if (lane == 0) {
                const float entropy = logf(tsum) - tsln / tsum;
                if (entropy > threshold) {
                    cuts[row] = col - a - 1;
                }
            }
        }
        __syncthreads();
        if (cuts[row] != k) { break; }  // cut found by this or an earlier column
    }
}

// ---------------------------------------------------------------------------------------------
// mtp_tree_commit_history (see include/ninfer/ops/mtp_round.h for the contract).
//
// One block per batch row. Thread 0 turns the accepted column's ancestor bit set into the chain
// column list -- the map every consumer of the accepted node needs -- and the whole block then
// moves each chain node's K/V row from its own column position to the position its DEPTH gives it.
// One block per ROW (not one per chain element) is what makes the in-place move safe: the copies
// run in ascending chain order with a barrier between them, and chain_sources[i] > i + 1 for every
// moved element, so the write step i performs can only land on a source an earlier step already
// read and can never land on a source a later step still has to read. (chain_sources[i] == i + 1 is
// the chain round's spelling: it is skipped, so a chain round moves zero bytes.)
template <int HeadDim, int KVHeads>
__global__ void mtp_tree_commit_history_kernel(
    const std::uint64_t* __restrict__ column_masks, const std::int32_t* __restrict__ column_depths,
    const std::int32_t* __restrict__ accepted_columns,
    const std::int32_t* __restrict__ base_frontiers, const std::int32_t* __restrict__ table_rows,
    const std::int32_t* __restrict__ block_tables, std::int32_t table_stride,
    std::int32_t* __restrict__ chain_sources, std::int32_t* __restrict__ commit_flags,
    __nv_bfloat16* __restrict__ cache_k, __nv_bfloat16* __restrict__ cache_v, std::int32_t width,
    std::int32_t batch, std::int32_t logical_capacity) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= batch) { return; }
    const int tid = static_cast<int>(threadIdx.x);

    __shared__ std::int32_t chain[kMtpTreeMaximumWidth];
    __shared__ std::int32_t chain_len;
    __shared__ std::int32_t frontier;
    __shared__ std::int32_t flags;

    std::int32_t* row_sources = chain_sources + static_cast<std::int64_t>(row) * width;
    if (tid == 0) {
        std::int32_t row_flags = 0;
        for (int j = 0; j < width; ++j) { row_sources[j] = -1; }
        for (int i = 0; i < kMtpTreeMaximumWidth; ++i) { chain[i] = -1; }
        int len                  = 0;
        const std::int32_t taken = accepted_columns[row];
        if (taken < 0 || taken >= width) {
            row_flags |= kMtpTreeFlagColumnOutOfRange;
        } else if (taken > 0) {
            const std::uint64_t mask =
                column_masks[static_cast<std::int64_t>(row) * width + taken];
            if (((mask >> static_cast<unsigned>(taken)) & 1ULL) == 0) {
                row_flags |= kMtpTreeFlagMaskMissingSelf;
            }
            // Bit 0 is the anchor column and never moves; bits 1..taken are the chain's nodes, and
            // ascending bit index is depth order.
            for (int j = 1; j <= taken && len < kMtpTreeMaximumWidth; ++j) {
                if (((mask >> static_cast<unsigned>(j)) & 1ULL) == 0) { continue; }
                chain[len++] = j;
            }
            if (len == 0 || chain[len - 1] != taken) { row_flags |= kMtpTreeFlagChainOrder; }
            if (column_depths[static_cast<std::int64_t>(row) * width + taken] != len) {
                row_flags |= kMtpTreeFlagDepthMismatch;
            }
            const std::int32_t base = base_frontiers[row];
            if (base < 0 || base + len >= logical_capacity) {
                row_flags |= kMtpTreeFlagPositionOutOfRange;
            }
            for (int i = 0; i < len; ++i) { row_sources[i] = chain[i]; }
        }
        chain_len         = len;
        frontier          = base_frontiers[row];
        flags             = row_flags;
        commit_flags[row] = row_flags;
    }
    __syncthreads();
    // A non-zero flag already says (loudly, via the caller) why nothing may be committed.
    if (flags != 0 || chain_len <= 0) { return; }

    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_rows[row]) * table_stride;
    for (int i = 0; i < chain_len; ++i) {
        const std::int32_t dst_pos = frontier + (i + 1);
        const std::int32_t src_pos = frontier + chain[i];
        if (src_pos == dst_pos) { continue; }
        for (int idx = tid; idx < KVHeads * (HeadDim / 8); idx += blockDim.x) {
            const int kv_head = idx / (HeadDim / 8);
            const int d       = (idx - kv_head * (HeadDim / 8)) * 8;
            const std::int64_t src_off =
                paged_kv_element_offset<HeadDim, KVHeads>(block_table, kv_head, src_pos, d);
            const std::int64_t dst_off =
                paged_kv_element_offset<HeadDim, KVHeads>(block_table, kv_head, dst_pos, d);
            store_vec(&cache_k[dst_off], load_vec<int4>(&cache_k[src_off]));
            store_vec(&cache_v[dst_off], load_vec<int4>(&cache_v[src_off]));
        }
        __syncthreads();
    }
}

// 件 6: re-index the verify hidden from verify-COLUMN order to accepted-chain-DEPTH order.
//
// `hidden` and `out` are BF16 [D,W,B] (logical), addressed here as 16-bit words so the copy is
// bit-for-bit by construction and needs no bfloat16 operator overload. chain_sources is I32 [W,B]
// (row-major, W fastest), valid_counts I32 [B]. Column j of row b is a contiguous D-element run.
__global__ void mtp_draft_align_hidden_kernel(const std::uint16_t* __restrict__ hidden,
                                              const std::int32_t* __restrict__ chain_sources,
                                              const std::int32_t* __restrict__ valid_counts,
                                              std::uint16_t* __restrict__ out, std::int32_t width,
                                              std::int32_t batch, std::int32_t head_dim) {
    const std::int32_t row   = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t j     = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t valid = valid_counts[row];
    std::int32_t src         = j;
    if (j != 0 && j < valid) { src = chain_sources[(j - 1) * batch + row]; }
    // Never read outside the frame: an inconsistent table (see kMtpTreeFlag*) must degrade to a
    // wrong gather, not to an out-of-range read. Its own check is mtp_tree_commit_history's flags,
    // which the runtime turns into a hard error.
    if (src < 0 || src >= width) { src = 0; }
    const std::uint16_t* s = hidden + (static_cast<std::int64_t>(src) * batch + row) * head_dim;
    std::uint16_t* d       = out + (static_cast<std::int64_t>(j) * batch + row) * head_dim;
    for (std::int32_t i = threadIdx.x; i < head_dim; i += blockDim.x) { d[i] = s[i]; }
}

} // namespace ninfer::ops
