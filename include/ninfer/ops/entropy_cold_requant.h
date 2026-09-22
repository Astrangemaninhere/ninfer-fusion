#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Cold-page requantization for the entropy slot codec (revision 2).
// Requantizes one or more stored KV page planes (E2M1 g16 NVFP4 or int8 g64)
// into fresh NVFP4 planes whose codes carry one E4M3FN scale per 64 channels
// (replicated into the four g16 scale slots). The output layout matches the
// page-major nvfp4 planes entropy_nvfp4_slot_encode_raw consumes, so the slot
// codec, decode path, scale scatter, and attention producers stay unchanged;
// the g64 requant is what makes the rANS streams compressible: on real Qwen3.8
// pages the requantized codes measure 3.31-3.86 bits/code (K 3.8455 mean, worst
// half 3.8613; V 3.3133) against 3.64-3.95 bits/code for the g16 storage codes
// (K 3.9533, V 3.6415). INSTRUMENT: dl/ransceil/rans_probe.cu (see its REPORT.md),
// which includes this header and runs the real entropy_nvfp4_slot_encode_kernel on
// engine-dumped .kvc planes. The '2.0-2.6 bits/code' this line carried until
// 2026-09-18 was never a measurement of these frames and is retracted here.
enum class EntropyColdRequantMode : int {
    Nvfp4G16   = 0,
    Int8G64    = 1,
    Iso4eVG16  = 2,
    // BF16-COLD-LAND A3. The bf16 resident plane is raw bf16 (head_dim x page_tokens
    // elements, 32768 B/head-page) with NO code plane and NO scale plane, so this arm
    // reads src_codes as a bf16 array and ignores src_scales (pass nullptr). The OUTPUT
    // is byte-identical to the Int8G64 arm: packed E2M1 nibbles plus one E4M3FN scale
    // per 64 channels, replicated into the four group-16 scale slots -- which is why the
    // raw slot, its pack kernel and its readers stay shared with the int8 tier.
    Bf16G64    = 3,
    // BF16-COLD-LAND E3. The E8 tier's planes are ALREADY packed 4-bit codes (one nibble
    // per element, 128 B per 64-token row) plus one fp16 scale per 64 channels, so this
    // arm copies the code plane VERBATIM and requantizes only the scale into the raw
    // slot's E4M3 g16 tail (replicated over the four g16 slots of the g64 group). Code
    // fidelity is therefore exact on this arm -- the only loss is the scale's precision.
    Rk4v4KvG64 = 4,
};

// src planes use page-major strides: codes kv_heads*8192 (nvfp4) or
// kv_heads*16384 (int8) bytes per page, scales kv_heads*1024 (nvfp4) or
// kv_heads*512 (int8). dst planes are nvfp4 page-major: codes
// [128, 64, kv_heads, page_count], scales [16, 64, kv_heads, page_count].
void entropy_cold_requant_raw(const std::uint8_t* src_codes, const std::uint8_t* src_scales,
                              EntropyColdRequantMode mode, int kv_heads, int page_count,
                              std::uint8_t* dst_codes, std::uint8_t* dst_scales,
                              cudaStream_t stream);

} // namespace ninfer::ops
