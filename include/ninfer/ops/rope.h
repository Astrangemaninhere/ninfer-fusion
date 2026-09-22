#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cmath>
#include <stdexcept>

namespace ninfer::ops {

// ---------------------------------------------------------------------------
// LANDED FROM THE FORK SURVEY (Apache-2.0). Source repo cometkim/ninfer, branch
// feat/1m-context, commit 2cc56b5db39f951c1e91b2caec8b39274d795682, path
// include/ninfer/ops/rope.h. This is the half the 2026-09-18 additive merge did NOT land:
// src/models/qwen3_5/program/rope_scaling.h and tests/models/qwen3_5/test_rope_scaling.cpp
// were landed consuming `ops::RopeFrequencies`, `ops::rope_linear_frequencies` and
// ops::rope_vision_frequencies, none of which existed anywhere in this tree -- so the borrow
// could not compile. Measured RED before this block: 15 g++ errors, all rooted in
// "'RopeFrequencies' in namespace 'ninfer::ops' does not name a type".
//
// DEVIATION, deliberate and the only one: the fork DECLARES the two builders here and DEFINES
// them in src/ops/wrapper/rope.cpp. They are defined inline here instead, because (a) both are
// pure host math over `double` with no Tensor, CUDA or device dependency, and (b) it keeps the
// mechanism callable from a host-only test target that must build and run while the engine's
// device link does not -- the same property tests/CMakeLists.txt already requires of its
// contract targets. The signatures are byte-identical to the fork's.
// ---------------------------------------------------------------------------

/** Maximum supported rotary pairs; RopeFrequencies storage is sized by this bound. */
inline constexpr int kRopeMaxPairs = 128;

/**
 * Pair-frequency table driving every rope mode. `inv_frequency[i]` is the cycle frequency of
 * rotary pair i; only [0, rotary_dim/2) entries are read. `attention_factor` is a q-side
 * temperature: the rotated dimensions of every q row scale by its square while k rows rotate
 * unscaled, so cached K is factor-free and the scores of the rotated dimensions scale by
 * attention_factor^2; 1.0F leaves the rotation untouched. Unrotated dimensions are never
 * affected. The two host builders cover the linear checkpoint tables; YaRN-shaped tables are
 * constructed by the owning target.
 */
struct RopeFrequencies {
    double inv_frequency[kRopeMaxPairs] = {};
    float attention_factor              = 1.0F;
};

/** Linear table theta^(-2i/rotary_dim), rotary_dim <= 2 * kRopeMaxPairs. */
inline RopeFrequencies rope_linear_frequencies(float theta, int rotary_dim) {
    if (!(theta > 0.0f) || !std::isfinite(theta)) {
        throw std::invalid_argument("rope: theta must be positive and finite");
    }
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > 2 * kRopeMaxPairs) {
        throw std::invalid_argument("rope: rotary_dim must be a positive even value <= 256");
    }
    RopeFrequencies frequencies;
    const double base = static_cast<double>(theta);
    for (int pair = 0; pair < rotary_dim / 2; ++pair) {
        frequencies.inv_frequency[pair] = std::pow(base, -2.0 * pair / rotary_dim);
    }
    return frequencies;
}

/** Vision 2-D table: 36 pairs wrapping an 18-entry theta^(-2j/36) ladder. */
inline RopeFrequencies rope_vision_frequencies(float theta) {
    if (!(theta > 0.0f) || !std::isfinite(theta)) {
        throw std::invalid_argument("rope: theta must be positive and finite");
    }
    RopeFrequencies frequencies;
    const double base = static_cast<double>(theta);
    for (int pair = 0; pair < 36; ++pair) {
        frequencies.inv_frequency[pair] = std::pow(base, -2.0 * (pair % 18) / 36.0);
    }
    return frequencies;
}

/**
 * Applies split-half NeoX RoPE in place. For pair i in [0,rotary_dim/2), angle phi(i,t), and
 * each head:
 *
 *   ideal[i]              = x[i] * cos(phi) - x[i+R/2] * sin(phi)
 *   ideal[i+rotary_dim/2] = x[i+R/2] * cos(phi) + x[i] * sin(phi).
 *
 * Dimensions [rotary_dim,head_dim) are unchanged. Supported modes are:
 *
 * - Text 1-D: positions I32 [T], either head_dim=256 with even 0<rotary_dim<=256, or the
 *   DFlash full-head domain head_dim=rotary_dim=128; phi=positions[t]*theta^(-2*i/rotary_dim).
 * - Text MRoPE: positions I32 [T,3], head_dim=256, rotary_dim=64; pair i uses axis i%3 with
 *   the same frequency as Text 1-D.
 * - Vision 2-D: positions I32 [T,2], head_dim=rotary_dim=72; pairs 0..17 use axis 0 and pairs
 *   18..35 use axis 1, each with local frequency theta^(-2*(i%18)/36).
 *
 * positions is contiguous and theta is positive and finite. Q/K tensors are BF16
 * [head_dim,heads,T] with positive head counts, contiguous head features and heads, and an optional
 * padded token stride. The registered optimized domains are D256/R64 Text Q/K head geometries
 * 24/4 and 16/2, D128/R128 1-D Text geometry 32/8, plus Vision geometry 16/16. q and k must not
 * overlap one another or positions. The Op mutates only dimensions [0,rotary_dim) of the supplied
 * Q/K tensor storage. The oracle evaluates the rotated dimensions naively in FP64 from the
 * represented inputs. The updated BF16 values are promoted and compared directly with that result;
 * output storage rounding belongs to the Op's numerical criterion, not the oracle. Unrotated
 * dimensions remain bit-exact. Private kernel arithmetic is implementation-defined. The Op uses no
 * workspace or persistent state.
 */
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          cudaStream_t stream);

// Single-tensor form with the same formula and storage contract. The head count comes directly
// from x; Q versus K role does not change the transformation.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x, cudaStream_t stream);

/**
 * Static YaRN factor-4 rope: the Text 1-D / DFlash 1-D transformations with the yarn4 frequency
 * tables (theta must be 1e7) and the yarn4 attention scaling 1.1386 folded into the sincos
 * coefficients. Registered domains are Text D256/R64 (heads 24/4, 16/2) and DFlash D128/R128
 * (32/8). Same storage contract as rope().
 */
void rope_yarn4(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                cudaStream_t stream);

// Single-tensor form of rope_yarn4.
void rope_yarn4(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                cudaStream_t stream);

} // namespace ninfer::ops
