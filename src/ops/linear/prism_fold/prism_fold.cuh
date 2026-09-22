#pragma once
// prism_fold.cuh -- THE ARITHMETIC of Route A: bring the ACTIVATION into the weights' basis.
//
// WHAT ROUTE A IS, IN ONE LINE
// ----------------------------
// A rotated checkpoint stores `W_s = A W` with `A = H . D` (normalized Sylvester
// Walsh-Hadamard, block 1024, `D` = diag(signs)).  The matmul the runtime wants is
// `W^T a`.  Since `A` is orthogonal,
//
//     W_s^T (A a) = (A W)^T (A a) = W^T A^T A a = W^T a
//
// so the runtime may keep the stored bytes untouched and rotate the ACTIVATION instead:
//
//     a_rot = A a = H (D a)          <-- signs FIRST, then the transform
//
// plus, for `blk.*.ssm_out.weight` only, a permutation of the activation's feature axis,
// because a folded file keeps that one tensor in the training (grouped) head order while the
// unrotated file of the same model carries it tiled (`gguf_hadamard.ssm_out_head_order`).
// The runtime order is: PERMUTE, then SIGN, then ROTATE -- the order the fold's own refusal
// message has always stated ("permute the ACTIVATION to match, before the signs and the
// rotation").  Permuting after the rotation would be a different, wrong operator: a
// permutation and a dense rotation do not commute.
//
// WHY THIS HEADER IS `__host__ __device__` AND NOT DEVICE-ONLY
// -----------------------------------------------------------
// Every claim below is a claim about ARITHMETIC, and the only instrument that can check
// arithmetic on a line with no GPU window is a host build of the same source.  The tree has
// that precedent and states its doctrine in `src/ops/linear/e8act/e8_activation_rotation.cuh`
// ("HOST-CALLABLE, deliberately ... A claim that has no host instrument is a claim nobody has
// checked") and in `tools/e8_verify/e8_lattice_codec_test.cpp`.  This header follows both: the
// REAL source below is what the host probe compiles and executes, not a paraphrase of it.
//
// The `PRISM_FOLD_HD` spelling rather than bare `__host__ __device__` is deliberate and is the
// one deviation from the e8act file's style.  That file relies on the probe defining the CUDA
// keywords away before the first include; a macro that expands to nothing under a non-nvcc
// compiler makes the same header compile unmodified, which removes a step where the probe's
// stub and the compiler's meaning can disagree.
//
// NO FWHT HINT, NO NEW KERNEL CLASS
// --------------------------------
// `prism.hadamard.*` names a DENSE multiply by an explicit matrix.  The fork's `ggml_fwht` /
// `GGML_HINT_SRC0_IS_HADAMARD` is an OPTIMISATION behind a hint, not a correctness
// requirement: measured, `H @ H == I` to 0.000e+00 and both CUDA butterflies equal `H . x` to
// 2.4e-15.  What is written below is the O(N log N) stage loop because it is the cheaper exact
// spelling, NOT because the engine needs the fork's op.  `linear` needs no new kernel class for
// the H half.
//
// SHAPE POLICY: REFUSE, NEVER PAD
// -------------------------------
// Padding a ragged width would change the dot product silently.  Every entry point below
// returns `false` for a shape it cannot serve and the caller refuses by name.

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define PRISM_FOLD_HD __host__ __device__
#define PRISM_FOLD_INLINE __forceinline__
#else
#define PRISM_FOLD_HD
#define PRISM_FOLD_INLINE inline
#endif

namespace ninfer::ops::prism_fold {

//: The one block size the released Bonsai-2 ternary contract declares
//: (`prism.hadamard.block_size` = 1024).  Declared as a constant so a file that says otherwise
//: is refused by name rather than silently rotated over the wrong length.
inline constexpr int kPrismFoldBlock = 1024;

//: The three widths the released contract declares as `sign_widths` = [5120, 6144, 17408].
//: They are not enforced here -- a caller passes the width it has -- but they are the only
//: widths that file's sign table covers, so naming them makes a wrong width a visible thing.
inline constexpr int kPrismFoldSignWidths[] = {5120, 6144, 17408};

PRISM_FOLD_HD constexpr bool prism_fold_block_ok(int block) noexcept {
    return block > 0 && (block & (block - 1)) == 0;
}

//: A block index.  `block` must be a power of two dividing `width`; anything else is refused
//: rather than padded, because a pad changes the dot product and does so silently.
PRISM_FOLD_HD constexpr bool prism_fold_width_ok(int width, int block) noexcept {
    return width > 0 && block > 0 && prism_fold_block_ok(block) && (width % block) == 0;
}

//: The GDN geometry the permutation needs, as the fork validates it before it will reorder.
PRISM_FOLD_HD constexpr bool prism_fold_gdn_geometry_ok(int width, int n_v, int n_k) noexcept {
    return width > 0 && n_v > 0 && n_k > 0 && (n_v % n_k) == 0 && (width % n_v) == 0;
}

// --------------------------------------------------------------------------- the butterfly
// ONE stage of the unscaled Sylvester butterfly, spelled once and called by both the serial
// loop below and the parallel kernel.  Convention, from `ggml/src/ggml-cuda/fwht.cu`: "the low
// element of a pair takes x + y, the high one x - y".  Pairs are `(base + j, base + h + j)` for
// every `base` a multiple of `2h` and `j` in `[0, h)`.
//
// `v` holds `n` elements, `h` is the stage's half-width, `h` must be a power of two and
// `h <= n/2`, and `n` must be a multiple of `2h`.
//
// The stage is written as a flat loop over PAIRS `p` in `[0, n/2)` and then split into a range
// function, so that the serial loop below and the parallel kernel share ONE spelling of the
// arithmetic and differ only in which pairs each worker takes.  A kernel that re-spelled the
// index decomposition would be free to disagree with the serial path about which two elements
// pair up, and that disagreement is silent.
//
//   pair p  <=>  base = (p / h) * 2h,  j = p % h  <=>  elements (base + j, base + h + j)
//
// which is the same (base, j) traversal the nested loops produce: `p` in `[0, 2h)` gives
// `base = 0` and every `j`, `p` in `[2h, 4h)` gives `base = 2h`, and so on.
PRISM_FOLD_HD PRISM_FOLD_INLINE bool prism_fold_butterfly_stage_range(float* v, int n, int h,
                                                                    int p_begin,
                                                                    int p_end) noexcept {
    if (v == nullptr || n <= 0 || h <= 0 || (h & (h - 1)) != 0 || n % (2 * h) != 0) {
        return false;
    }
    const int pairs = n / 2;
    if (p_begin < 0 || p_end > pairs || p_begin > p_end) {
        return false;
    }
    for (int p = p_begin; p < p_end; ++p) {
        const int base = (p / h) * 2 * h;
        const int j = p % h;
        const float a = v[base + j];
        const float b = v[base + h + j];
        v[base + j] = a + b;
        v[base + h + j] = a - b;
    }
    return true;
}

PRISM_FOLD_HD PRISM_FOLD_INLINE bool prism_fold_butterfly_stage(float* v, int n, int h) noexcept {
    if (n <= 0 || (n % 2) != 0) {
        return false;
    }
    return prism_fold_butterfly_stage_range(v, n, h, 0, n / 2);
}

//: The whole unscaled transform over `n` elements, `n` a power of two.  `log2(n)` stages,
//: `h = 1, 2, 4, ... n/2`, in that order -- the same order `gguf_hadamard.butterfly` uses, which
//: is the fork's.
PRISM_FOLD_HD PRISM_FOLD_INLINE bool prism_fold_butterfly(float* v, int n) noexcept {
    if (v == nullptr || !prism_fold_block_ok(n)) {
        return false;
    }
    for (int h = 1; h < n; h *= 2) {
        if (!prism_fold_butterfly_stage(v, n, h)) {
            return false;
        }
    }
    return true;
}

//: The normalization that makes the transform the fork's `H` rather than the raw Walsh matrix:
//: multiply by `1/sqrt(n)`.  For `n` a power of two this is exact in fp32 (n = 1024 gives
//: exactly 1/32), which is why the round-trip measured on real bytes is exact rather than
//: merely close.
PRISM_FOLD_HD PRISM_FOLD_INLINE bool prism_fold_normalize(float* v, int n) noexcept {
    if (v == nullptr || !prism_fold_block_ok(n)) {
        return false;
    }
    const float scale = 1.0f / sqrtf(static_cast<float>(n));
    for (int i = 0; i < n; ++i) {
        v[i] *= scale;
    }
    return true;
}

// --------------------------------------------------------------------------- A a = H (D a)
//: The forward operator on ONE contiguous run of `width` elements: signs first, then the
//: blockwise transform.  This is `A a` with `A = H D`, the direction Route A's runtime needs.
//:
//: `dst` and `src` may alias (Route A rotates in place in mind); the signs are read from
//: `signs[0..width)`, which is the contract's own sign vector for this width, NOT broadcast.
PRISM_FOLD_HD PRISM_FOLD_INLINE bool prism_fold_forward_out(float* dst, const float* src,
                                                           const float* signs, int width,
                                                           int block) noexcept {
    if (dst == nullptr || src == nullptr || signs == nullptr) {
        return false;
    }
    if (!prism_fold_width_ok(width, block)) {
        return false;
    }
    for (int off = 0; off < width; off += block) {
        // D first: the sign vector is indexed by input channel, one entry per element of the
        // block, and it is shared across every block (the sign vector spans the whole width).
        float v[kPrismFoldBlock];
        if (block > kPrismFoldBlock) {
            return false;
        }
        for (int i = 0; i < block; ++i) {
            v[i] = src[off + i] * signs[off + i];
        }
        if (!prism_fold_butterfly(v, block)) {
            return false;
        }
        if (!prism_fold_normalize(v, block)) {
            return false;
        }
        for (int i = 0; i < block; ++i) {
            dst[off + i] = v[i];
        }
    }
    return true;
}

//: `A^-1 z = D (H z)`: the transform first, the signs LAST.  The runtime does not need this
//: direction (a runtime that could invert the fold would be a runtime that folded twice), but
//: the converter does and the probe needs both directions to check that the same butterfly
//: serves them.  Same signs, opposite order -- the inverse is free, and no second sign table is
//: needed, which is a carried FACT about this contract, not an economy invented here.
PRISM_FOLD_HD PRISM_FOLD_INLINE bool prism_fold_inverse_out(float* dst, const float* src,
                                                           const float* signs, int width,
                                                           int block) noexcept {
    if (dst == nullptr || src == nullptr || signs == nullptr) {
        return false;
    }
    if (!prism_fold_width_ok(width, block)) {
        return false;
    }
    for (int off = 0; off < width; off += block) {
        float v[kPrismFoldBlock];
        if (block > kPrismFoldBlock) {
            return false;
        }
        for (int i = 0; i < block; ++i) {
            v[i] = src[off + i];
        }
        if (!prism_fold_butterfly(v, block)) {
            return false;
        }
        if (!prism_fold_normalize(v, block)) {
            return false;
        }
        for (int i = 0; i < block; ++i) {
            dst[off + i] = v[i] * signs[off + i];   // signs LAST
        }
    }
    return true;
}

// --------------------------------------------------------------------------- the GDN heads
//: The source index of grouped position `j`, i.e. `a_grouped[j] = a_tiled[source_index(j)]`.
//:
//: Derived from `gguf_hadamard.gdn_v_permutation` rather than tabulated, so the runtime needs no
//: 6144-entry device table per layer.  That function builds
//: `tiled = arange(width).reshape(rep, n_k, hd)` and returns `tiled.transpose(1, 0, 2)`, i.e.
//: the grouped layout is `[n_k, rep, hd]` -- the FORK's `[hd, nk, rep] -> [hd, rep, nk]` read
//: in the other axis order, which is the same reorder because `hd` is the innermost axis in
//: both.  So, with `hd = width / n_v` and `rep = n_v / n_k`:
//:
//:     grouped position j  ->  k = (j / hd) / rep,  r = (j / hd) % rep,  h = j % hd
//:     tiled   position    ->  ((r * n_k) + k) * hd + h
PRISM_FOLD_HD PRISM_FOLD_INLINE int prism_fold_gdn_source_index(int j, int n_k, int rep,
                                                               int hd) noexcept {
    const int head = j / hd;          // which of the n_v grouped slots
    const int h = j % hd;             // position inside the head
    const int k = head / rep;         // the group this slot belongs to
    const int r = head % rep;         // the replica inside the group
    return ((r * n_k) + k) * hd + h;
}

//: The gather itself: `dst[j] = src[prism_fold_gdn_source_index(j)]` over `width` features.
//: This is the operator the fold's refusal message has always described the runtime as owing,
//: and it belongs on the ACTIVATION for the same reason the rotation does: a column permutation
//: on the rotation axis cannot be refolded, so the fork keeps the training order in the file
//: and pays for it here.
PRISM_FOLD_HD PRISM_FOLD_INLINE bool prism_fold_gdn_permute_out(float* dst, const float* src,
                                                              int width, int n_v,
                                                              int n_k) noexcept {
    if (dst == nullptr || src == nullptr) {
        return false;
    }
    if (!prism_fold_gdn_geometry_ok(width, n_v, n_k)) {
        return false;
    }
    const int hd = width / n_v;
    const int rep = n_v / n_k;
    for (int j = 0; j < width; ++j) {
        dst[j] = src[prism_fold_gdn_source_index(j, n_k, rep, hd)];
    }
    return true;
}

//: The two-step Route A activation contract for `blk.*.ssm_out.weight`, in the order that is
//: load-bearing: PERMUTE, then SIGN, then ROTATE.  `grouped` selects the permutation; the four
//: other tensors' widths take the rotate alone.  Exposed as one function so that a caller
//: cannot pick the two halves up in the wrong order without noticing: there is no entry point
//: here that rotates before it permutes.
PRISM_FOLD_HD PRISM_FOLD_INLINE bool prism_fold_gdn_activation_out(float* dst, const float* src,
                                                                 const float* signs, int width,
                                                                 int block, int n_v,
                                                                 int n_k) noexcept {
    if (dst == nullptr || src == nullptr || signs == nullptr) {
        return false;
    }
    if (!prism_fold_gdn_permute_out(dst, src, width, n_v, n_k)) {
        return false;
    }
    // In place over the just-permuted buffer: `prism_fold_forward_out` stages each block before
    // the store, so an aliased call is the in-place call.
    return prism_fold_forward_out(dst, dst, signs, width, block);
}

}   // namespace ninfer::ops::prism_fold
