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

using Gqa27Geometry = GqaGeometry<24, 4, 1>;
using Gqa35Geometry = GqaGeometry<16, 2, 2>;
using GqaMuseGeometry = GqaGeometry<32, 2, 1, 128>;

// Spark-X2.5: 16 q-heads / 4 kv-heads at head_dim 256 (group 4).
// DecodeSplitScale stays 1: a decode launch is KVHeads * DecodeSplits CTAs, and four
// KV heads is the same count as Gqa27Geometry, so 85 splits keeps the same CTA count.
using Gqa16x4Geometry = GqaGeometry<16, 4, 1>;

} // namespace ninfer::ops
