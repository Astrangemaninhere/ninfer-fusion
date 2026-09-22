// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : cometkim/ninfer
// Branch      : feat/1m-context
// Commit      : 2cc56b5db39f951c1e91b2caec8b39274d795682
// Source path : src/models/qwen3_5/program/rope_scaling.h
// sha256(src) : 5b65f8f09cf9b92f871291b142a275d5a4d0d6bfcfc13293e0c4c2ced81f2d2f
// Landed by   : /home/user/scratch/PATCHSET/MERGE (fork-survey merge, 2026-09-18)
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : the codecs in this family label a rotation axis -- rk<N>v<M> = rotated K at N bits against V at M bits (rk4v4-e8 = rotated K 4b, V 4b, rk4v4 in the mix), which is a DIFFERENT axis from src/kvcfg/kv_formats.h's tier vocabulary (hot/tail/cold, Rk4v4).
#pragma once
// Qwen3.6 family text RoPE scaling: the YaRN frequency table builder used by the sequence
// plan. Pure host math over the checkpoint's rope constants - no Variant instantiation.

#include "ninfer/ops/rope.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {

/**
 * YaRN frequency table (HF `_compute_yarn_parameters` semantics): pairs below `low` extrapolate
 * the linear frequency, pairs in [low, high] blend extrapolation with interpolation by
 * e = 1 - (i - low)/(high - low), and pairs above `high` interpolate at inv/factor. The ramp
 * bounds are floor/ceil of the correction dimension
 * rotary_dim * ln(original_positions / (beta * 2pi)) / (2 ln theta), clamped to leave a
 * non-empty extrapolation and interpolation segment (HF clamps `high` to rotary_dim - 1
 * rather than rotary_dim/2 - 1, so extreme beta_slow values leave HF blending the top pairs
 * while this builder fully interpolates them; irrelevant at the checkpoint defaults).
 * `temperature` scales the attention factor
 * (temperature * ln(factor) + 1, HF default 0.1), which travels in RopeFrequencies as the
 * q-side temperature squared on q and leaves cached K factor-free.
 */
inline ops::RopeFrequencies rope_yarn_frequencies(float theta, int rotary_dim,
                                                  std::uint32_t original_positions, float factor,
                                                  float temperature = 0.1F,
                                                  float beta_fast   = 32.0F,
                                                  float beta_slow   = 1.0F) {
    if (!(factor > 1.0F) || !std::isfinite(factor)) {
        throw std::invalid_argument("rope scaling: YaRN factor must exceed 1");
    }
    if (!(temperature > 0.0F) || !std::isfinite(temperature)) {
        throw std::invalid_argument("rope scaling: YaRN temperature must be positive");
    }
    if (!(beta_fast > beta_slow && beta_slow > 0.0F) || !std::isfinite(beta_fast) ||
        !std::isfinite(beta_slow)) {
        throw std::invalid_argument("rope scaling: YaRN requires beta_fast > beta_slow > 0");
    }
    constexpr double kTwoPi = 6.28318530717958648;
    const double base       = static_cast<double>(theta);
    const double original   = static_cast<double>(original_positions);
    const auto correction   = [&](double beta) {
        return static_cast<double>(rotary_dim) * std::log(original / (beta * kTwoPi)) /
               (2.0 * std::log(base));
    };
    const int half = rotary_dim / 2;
    const int low =
        std::clamp(static_cast<int>(std::floor(correction(beta_fast))), 0, half - 2);
    const int high =
        std::clamp(static_cast<int>(std::ceil(correction(beta_slow))), low + 1, half - 1);

    ops::RopeFrequencies frequencies;
    frequencies.attention_factor =
        static_cast<float>(static_cast<double>(temperature) * std::log(static_cast<double>(factor)) +
                           1.0);
    for (int i = 0; i < half; ++i) {
        const double linear = std::pow(base, -2.0 * i / rotary_dim);
        if (i < low) {
            frequencies.inv_frequency[i] = linear;
        } else if (i > high) {
            frequencies.inv_frequency[i] = linear / static_cast<double>(factor);
        } else {
            const double extrap = static_cast<double>(i - low) / static_cast<double>(high - low);
            frequencies.inv_frequency[i] =
                linear * ((1.0 - extrap) + extrap / static_cast<double>(factor));
        }
    }
    return frequencies;
}

} // namespace ninfer::models::qwen3_5
