#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: static order-0 rANS codec for NVFP4 E2M1 code nibbles.
 *
 * Symbols are 4-bit code nibbles (alphabet size 16). `codes` is packed U8
 * with the low nibble at even symbol index and the high nibble at odd symbol
 * index. The codec is static rANS with SCALE_BITS=12, SCALE=4096, MASK=4095,
 * RANS_BYTE_L=1<<23 and a 32-bit state. Frequencies sum to 4096 and every
 * frequency is non-zero. `out` capacity must be at least
 * ceil(1.1 * packed code bytes) + 64 bytes per stream. On overflow encode sets
 * out_size = 0 and the caller falls back to raw packed codes.
 *
 * The encoder processes symbols in REVERSE order. While the state is
 * >= ((RANS_BYTE_L >> SCALE_BITS) << 8) * freq[s] it emits one renorm byte
 * (x & 0xff) and shifts x right by 8, repeating until it is below that bound;
 * it then applies
 *
 *   x = ((x / freq[s]) << SCALE_BITS) + (x % freq[s]) + start[s].
 *
 * The stream is the renorm bytes in emission order followed by the final
 * state as 4 little-endian bytes at the end. The decoder initializes the state
 * from the last 4 little-endian bytes and, after each inverse state update,
 * consumes renorm bytes LIFO from the end of the stream with
 * x = (x << 8) | previous_byte while x < RANS_BYTE_L.
 *
 * Shapes:
 * - Single stream: codes U8 [bytes], symbol_count I32 [1],
 *   frequencies U16 [16], out U8 [capacity], out_size I32 [1].
 * - Batched (B streams): codes U8 [bytes_per_stream, B],
 *   symbol_count I32 [B], frequencies U16 [16, B],
 *   out U8 [capacity_per_stream, B], out_size I32 [B].
 *
 * Decode mirrors the same shapes with data U8 [capacity_per_stream, B],
 * data_size I32 [B], frequencies U16 [16, B], symbol_count I32 [B], and
 * codes_out U8 [bytes_per_stream, B]. For batched decode, data_size[b] bytes
 * are consumed starting at data + b * capacity_per_stream.
 */
void entropy_nvfp4_encode(const Tensor& codes, const Tensor& symbol_count,
                          Tensor& frequencies, Tensor& out, Tensor& out_size,
                          cudaStream_t stream);

void entropy_nvfp4_decode(const Tensor& data, const Tensor& data_size,
                          const Tensor& frequencies, const Tensor& symbol_count,
                          Tensor& codes_out, cudaStream_t stream);

} // namespace ninfer::ops
