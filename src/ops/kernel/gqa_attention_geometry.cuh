#pragma once

// Exact grouped-query head geometries served by the Qwen3.6 GQA kernels. Head
// dimension, cache format, and tile policy are shared; head mapping remains a
// compile-time property so each registered shape gets an independent kernel.

namespace ninfer::ops {

template <int QHeadsValue, int KVHeadsValue, int DecodeSplitScaleValue,
          int HeadDimValue = 256>
struct GqaGeometry {
    static_assert(QHeadsValue > 0 && KVHeadsValue > 0);
    static_assert(QHeadsValue % KVHeadsValue == 0);
    static_assert(DecodeSplitScaleValue > 0);
    static_assert(HeadDimValue == 128 || HeadDimValue == 256);

    static constexpr int QHeads           = QHeadsValue;
    static constexpr int KVHeads          = KVHeadsValue;
    static constexpr int GroupSize        = QHeads / KVHeads;
    static constexpr int DecodeSplitScale = DecodeSplitScaleValue;
    static constexpr int DecodeSplits     = 85 * DecodeSplitScale;
    static constexpr int HeadDim          = HeadDimValue;
};

// ---------------------------------------------------------------------------------------
// NINFER_GQA_SINGLE_WAVE -- the one-170-SM-wave split policy, OPT-IN. The switch itself is
// gqa_single_wave_env() in ops/kernel/gqa_attention_decode.cuh.
//
// A decode launch of the split-KV small-T family is KVHeads * splits CTAs and those CTAs
// are 1 per SM (measured, not assumed: 512 threads at 100 registers and 58.7 KiB of shared
// memory per block put BOTH `Block Limit Registers` and `Block Limit Shared Mem` at 1 -- ncu,
// grid (4,50,1) on the 27B/24-4 geometry). A split count above 170 / KVHeads therefore buys
// a SECOND wave whose CTAs each still pay the full per-CTA latency while the rest of the
// grid idles: at 200 CTAs the measured sm__cycles_active / sm__cycles_elapsed was 56.6 %,
// which is 200/(170*2) = 58.8 % to within the sampling error.
//
// This is what NINFER_GQA_SINGLE_WAVE=1 substitutes for Geometry::DecodeSplits
// (= 85 * DecodeSplitScale: 340 CTAs = two waves at KVHeads 4, scale 1).
//
// COVERAGE -- measured vs assumed. Do NOT read this as a no-op outside the 27B shape:
//   Gqa27Geometry    <24,4,1,256>  DecodeSplits=85  -> 170/4 = 42  CHANGES (measured: kernel -33 %)
//   Gqa16x4Geometry  <16,4,1,256>  DecodeSplits=85  -> 170/4 = 42  CHANGES (Spark-X2.5, NOT measured)
//   Gqa35Geometry    <16,2,2>      DecodeSplits=170 -> 170/2 = 85  CHANGES (NOT measured: scale is 2,
//                                                                  so "KVHeads==2 gives the old value"
//                                                                  does not hold for it)
//   GqaMuseGeometry  <32,2,1,128>  DecodeSplits=85  -> 170/2 = 85  identical (the only real no-op)
// DecodeSplitScale is part of DecodeSplits, which is why the scale-2 geometry moves too.
// ---------------------------------------------------------------------------------------
inline constexpr int kGqaDecodeWaveSms = 170;
template <typename Geometry>
constexpr int kGqaDecodeWaveSplitsPerHead = kGqaDecodeWaveSms / Geometry::KVHeads;

using Gqa27Geometry = GqaGeometry<24, 4, 1>;
using Gqa35Geometry = GqaGeometry<16, 2, 2>;
using GqaMuseGeometry = GqaGeometry<32, 2, 1, 128>;

// Spark-X2.5: 16 q-heads / 4 kv-heads at head_dim 256 (group 4).
// DecodeSplitScale stays 1: a decode launch is KVHeads * DecodeSplits CTAs, and four
// KV heads is the same count as Gqa27Geometry, so 85 splits keeps the same CTA count.
using Gqa16x4Geometry = GqaGeometry<16, 4, 1>;

} // namespace ninfer::ops
