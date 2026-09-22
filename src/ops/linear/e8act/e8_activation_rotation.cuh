#pragma once
// e8_activation_rotation.cuh -- THE ACTIVATION-SIDE HADAMARD, the one component a
// weight-side E8 needs that the tree does not have anywhere under src/ops/linear*.
//
// WHAT IS MISSING, MEASURED
// ------------------------
// `hadamard` names 35 files under src/ and ZERO of them is under src/ops/linear,
// src/ops/linear_add, src/ops/linear_pair or src/ops/linear_swiglu. Every existing
// operator is a KV / attention operator: it rotates a key or a query row along the
// head_dim axis. None of them is an operator on the CONTRACTION AXIS of a matmul
// operand, which is the axis a quantised weight's rotation is defined against.
//
// WHY THE RECIPE NEEDS THIS AND CANNOT GET IT FROM THE CODEC
// ----------------------------------------------------------
// The tree's weight-side rotation, if it ever exists, is the MIRROR of the KV-side one:
//
//   KV side (shipped, src/ops/kernel/e8_lattice_kv_plane.cuh:284-291):
//     the writer hands the codec RAW K, the CODEC applies H64 internally, and the reader
//     emits the rotated domain and stops. The codec owns the rotation.
//
//   WEIGHT side (this file):
//     the rotation is folded into the stored weight OFFLINE, W' = W . H (H acts on the
//     contraction axis k). The stored bytes are therefore already rotated, so a codec that
//     rotated again would rotate TWICE -- H64 is its own inverse, so the second rotation
//     silently undoes the first and the lattice quantises the natural-domain coordinates.
//     That is the exact failure `e8_lattice_kv_plane.cuh:28-44` prices at +1.202 dB at w3
//     and +0.209 dB at w2, and it is SILENT in both directions.
//
//   ==> ON THIS SIDE THE ACTIVATION CARRIES THE ROTATION, ONLINE, AND THE CODEC MUST NOT.
//
// The two conventions are a matched pair. `kE8ActRotationIsFoldedIntoWeights` below names
// this side's choice in one place so a later reader cannot pick the other one by accident.
//
// THE ARITHMETIC IS NOT RE-SPELLED, AND THAT IS THE POINT OF THE FILE
// ------------------------------------------------------------------
// The 64-point Sylvester butterfly already exists, once, at
// src/ops/kernel/e8_lattice_codec.cuh:112-136 as `e8_lattice_hadamard64_t<R>`. This file
// CALLS it. There is no second spelling of the butterfly here, on purpose:
// `e8_lattice_codec.cuh:107-111` states the doctrine ("Two hand-written copies would be
// free to drift, which is the exact defect class this file warns about elsewhere"), and a
// second copy would also destroy the only property that matters for a weight-side E8 --
// that the activation is rotated by the SAME orthogonal map the weights were rotated by.
// What this file adds is the ADDRESS MAP (which 64 elements, at what stride), which is the
// part the codec does not have and cannot have: the codec does not know what a matmul
// operand is.
//
// WHY THE ADDRESS MAP IS TRIVIAL HERE AND IS NOT IN THE KV PATH
// -----------------------------------------------------------
// `linear` requires x to be [K, T] and CONTIGUOUS (src/ops/linear/linear.cpp:58-70), which
// makes ne[0] = K the innermost axis on this tree's Tensor. The K elements of one token are
// therefore a CONTIGUOUS run of K floats, and the contraction-axis 64-block of a token is a
// CONTIGUOUS run of 64. So the map is `x + t*K + block*64` for t in [0,T) -- no transpose,
// no gather, no shuffle. This is exactly the shape the KV writer already rotates
// (`e8_lattice_kv_plane.cuh:288-291` rotates a contiguous `float y[64]`).
//
// THE KNOBS AND THE TARGET FLOOR
// ------------------------------
// The butterfly is adds, subtracts and one multiply by 1/8 (a power of two, hence exact in
// fp32). No shuffle, no shared memory, no DP4A, no tensor core, no fp64-class instruction
// (the `float` instantiation of the called function is the one `e8_lattice_codec.cuh:414-415`
// measures as zero fp64-class at SASS). 64 live floats per thread is the tree's own register
// budget for this shape -- the KV writer holds the same array.
//
// HOST-CALLABLE, deliberately, for the same reason the codec is
// ----------------------------------------------------------
// The properties below are claims about ARITHMETIC, and the only instrument that can check
// them without a card is a host build of this same header, with the three CUDA keywords
// defined away, exactly as `tools/e8_verify/e8_lattice_codec_test.cpp` does it. A claim
// that has no host instrument is a claim nobody has checked.

#include <cstdint>

#include <type_traits>
#include "ops/kernel/e8_lattice_codec.cuh"   // brings in ops/kernel/e8_lattice.cuh

namespace ninfer::ops::e8act {

// ---------------------------------------------------------------- the domain contract
// The weight side's convention, named once. See the block above: this is the MIRROR of
// `kE8KvLatticeInputIsNatural` (src/ops/kernel/e8_lattice_kv_plane.cuh:216-217) and the two
// must never be mixed inside one matmul.
inline constexpr bool kE8ActRotationIsFoldedIntoWeights = true;  // W stores W.H
inline constexpr bool kE8ActCodecMustNotRotate           = true;  // and x arrives rotated

// ---------------------------------------------------------------- geometry
// The block is the codec's own group. It is NOT re-declared as a literal: if the codec's
// group ever moves, this file moves with it and the static_assert below fails loudly rather
// than producing a rotation that is orthogonal over the wrong length.
inline constexpr int kE8ActBlock = kE8LatticeGroup;
static_assert(kE8ActBlock == 64, "the activation rotation block is the codec's 64-group");
static_assert(kE8ActBlock % kE8LatticeDim == 0,
              "the rotation block must be a whole number of the lattice's 8-dim blocks");

// The only shape precondition. `linear` registers a fixed (n, k, t) set
// (src/ops/linear/q4/q4_dispatch.cpp:7-120), so a k not divisible by 64 is a shape this
// operator refuses BY NAME rather than a shape it pads: padding would change the dot
// product, and a silent pad is the failure this tree spends whole headers forbidding.
// `__host__ __device__ constexpr`, not `constexpr __device__`, and that is a
// measured constraint rather than a style choice. A `constexpr __device__` function cannot be called from a
// `__host__` function: nvcc reports "calling a constexpr __device__ function from a
// __host__ function is not allowed" and offers `--expt-relaxed-constexpr`, which this
// tree's build line does not carry and which its own doctrine refuses -- see
// src/ops/kernel/e8_lattice_kv_plane.cuh:179-185, where the SAME constraint is recorded in
// the other direction and the flag is explicitly rejected in favour of a spelling that
// does not need it. This file hit that wall in the mirror direction on its first compile;
// the first `-c` was rc=2 with four errors, all four this one.
[[nodiscard]] __host__ __device__ constexpr bool e8act_shape_supported(std::int64_t k) noexcept {
    return k > 0 && (k % static_cast<std::int64_t>(kE8ActBlock)) == 0;
}
static_assert(e8act_shape_supported(5120) && e8act_shape_supported(64),
              "the tree's registered k values are 64-block aligned");
static_assert(!e8act_shape_supported(6145), "a ragged k is refused, not padded");

// ---------------------------------------------------------------- the rotation
// ONE spelling: the codec's butterfly, called. `R` is instantiated at `float` only on the
// device path, because that is the instantiation measured free of fp64-class instructions.
__device__ __forceinline__ void e8act_rotate_block_t(float (&v)[kE8ActBlock]) {
    e8_lattice_hadamard64_t<float>(v);
}

// One contiguous 64-block, in place. This is the operator; everything below is addressing.
__device__ __forceinline__ void e8act_rotate_block_inplace(float* block) {
    if (block == nullptr) { return; }
    float v[kE8ActBlock];
#pragma unroll
    for (int i = 0; i < kE8ActBlock; ++i) { v[i] = block[i]; }
    e8act_rotate_block_t(v);
#pragma unroll
    for (int i = 0; i < kE8ActBlock; ++i) { block[i] = v[i]; }
}

// One token's contraction-axis run of `k` elements (k a multiple of 64), in place.
// This is the activation of a `linear` for ONE token: [K, T] row-major with K innermost.
__device__ __forceinline__ void e8act_rotate_row_inplace(float* row, int k) {
    if (row == nullptr || !e8act_shape_supported(k)) { return; }
    const int groups = k / kE8ActBlock;
    for (int g = 0; g < groups; ++g) {
        e8act_rotate_block_inplace(row + static_cast<std::ptrdiff_t>(g) * kE8ActBlock);
    }
}

// The same, out of place. `dst` and `src` may alias: the block is staged in registers before
// the store, so an in-place call is `e8act_rotate_row_out(dst, dst, k)`.
__device__ __forceinline__ void e8act_rotate_row_out(float* dst, const float* src, int k) {
    if (dst == nullptr || !e8act_shape_supported(k)) { return; }
    const int groups = k / kE8ActBlock;
    for (int g = 0; g < groups; ++g) {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(g) * kE8ActBlock;
        float v[kE8ActBlock];
#pragma unroll
        for (int i = 0; i < kE8ActBlock; ++i) { v[i] = src[off + i]; }
        e8act_rotate_block_t(v);
#pragma unroll
        for (int i = 0; i < kE8ActBlock; ++i) { dst[off + i] = v[i]; }
    }
}

// ---------------------------------------------------------------- the block count
// One thread per 64-block; the grid is (blocks_per_token, tokens). Kept here beside the
// operator so the launch site cannot partition the plane differently from the operator's
// own map -- the two would then disagree about which 64 elements form a block, silently.
[[nodiscard]] __host__ __device__ constexpr std::int64_t e8act_blocks(std::int64_t k,
                                                                 std::int64_t t) {
    return e8act_shape_supported(k) ? (k / kE8ActBlock) * t : 0;
}

}   // namespace ninfer::ops::e8act
