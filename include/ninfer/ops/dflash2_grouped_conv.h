#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: DFlash2 grouped convolution pass
 *
 * `hidden` is contiguous BF16 [H,T]: the token-major block produced by the
 * draft projections. `delta` is contiguous BF16 [N,T] with N=2*taps*G and is
 * the exact token-major output of the kernel projection (N fastest). `side`
 * selects the pass's tap bank. `base` is a BF16 [taps,H] Weight (one side of
 * the stored [2,taps,H] base kernel). `out` is contiguous BF16 [H,T] and is
 * completely overwritten.
 *
 * For token t with in-block position p = t & (block_size-1) and channel h in
 * group g = h / group_size,
 *
 *   out[h,t] = hidden[h,t] * (base[0,h] + delta[G*(side*taps)+g,t])
 *            + sum_{tap>=1, p>=tap} hidden[h,t-tap]
 *              * (base[tap,h] + delta[G*(side*taps+tap)+g,t]).
 *
 * Cross-block reads introduced by t-tap are multiplied by the zero in-block
 * position mask, so rows never leak into a neighbour block. The registered
 * domain is H=5120, T=1..64, taps=2, G=320, group_size=16, block_size=8,
 * side in {0,1}. Inputs and base are unchanged and the Op owns no workspace or
 * persistent state. Intermediate arithmetic is FP32 and out is rounded to BF16.
 */
void dflash2_grouped_conv(const Tensor& hidden, const Tensor& delta, const Weight& base,
                          std::int32_t block_size, std::int32_t group_size, std::int32_t taps,
                          std::int32_t side, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
