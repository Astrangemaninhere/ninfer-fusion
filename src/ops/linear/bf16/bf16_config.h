#pragma once

#include <cstdint>

namespace ninfer::ops::detail {

enum class Bf16ActivationAccess : std::uint8_t {
    Direct,
    Shared,
};

enum class Bf16WeightCache : std::uint8_t {
    Default,
    Streaming,
};

enum class Bf16PhaseOrder : std::uint8_t {
    Sequential,
    RowSwizzled,
};

enum class Bf16SmallTActivationAccess : std::uint8_t {
    DirectStream,
    WarpPacked,
};

template <std::int32_t OutputRows, std::int32_t InputRows>
struct Bf16GemvGeometry {
    static_assert(OutputRows > 0 && InputRows > 0);

    static constexpr std::int32_t kOutputRows = OutputRows;
    static constexpr std::int32_t kInputRows  = InputRows;
};

template <int WarpsPerCta, int WarpsPerRow, int RowsPerWarp, int ValuesPerLane,
          int AccumulatorChains, Bf16ActivationAccess ActivationAccess, Bf16WeightCache WeightCache,
          Bf16PhaseOrder PhaseOrder, int PhaseStride, int PrefetchDepth, int PhaseUnroll,
          int MinBlocksPerSm>
struct Bf16GemvSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(WarpsPerRow > 0 && WarpsPerRow <= WarpsPerCta);
    static_assert((WarpsPerCta % WarpsPerRow) == 0);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 4 || ValuesPerLane == 8 || ValuesPerLane == 16);
    static_assert(AccumulatorChains > 0 && AccumulatorChains <= ValuesPerLane);
    static_assert((AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(PrefetchDepth == 1 || PrefetchDepth == 2);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4 || PhaseUnroll == 8);
    static_assert(PhaseStride > 0);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kWarpsPerRow       = WarpsPerRow;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kActivationAccess = ActivationAccess;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kPhaseOrder       = PhaseOrder;
    static constexpr int kPhaseStride       = PhaseStride;
    static constexpr int kPrefetchDepth     = PrefetchDepth;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowGroupsPerCta   = WarpsPerCta / WarpsPerRow;
    static constexpr int kRowsPerCta        = kRowGroupsPerCta * RowsPerWarp;
};

template <int WarpsPerCta, int WarpsPerRow, int RowsPerWarp, int ValuesPerLane,
          int AccumulatorChains, int TokenBatch, Bf16SmallTActivationAccess ActivationAccess,
          Bf16WeightCache WeightCache, Bf16PhaseOrder PhaseOrder, int PhaseStride, int PhaseUnroll,
          int PrefetchDepth, int MinBlocksPerSm>
struct Bf16SmallTInnerSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(WarpsPerRow > 0 && WarpsPerRow <= WarpsPerCta);
    static_assert((WarpsPerCta % WarpsPerRow) == 0);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 4 || ValuesPerLane == 8 || ValuesPerLane == 16);
    static_assert(AccumulatorChains > 0 && AccumulatorChains <= ValuesPerLane);
    static_assert((AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(TokenBatch == 1 || TokenBatch == 2 || TokenBatch == 4 || TokenBatch == 8);
    static_assert(PhaseStride > 0);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4 || PhaseUnroll == 8);
    static_assert(PrefetchDepth == 1 || PrefetchDepth == 2);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kWarpsPerRow       = WarpsPerRow;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr int kTokenBatch        = TokenBatch;
    static constexpr auto kActivationAccess = ActivationAccess;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kPhaseOrder       = PhaseOrder;
    static constexpr int kPhaseStride       = PhaseStride;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr int kPrefetchDepth     = PrefetchDepth;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowGroupsPerCta   = WarpsPerCta / WarpsPerRow;
    static constexpr int kRowsPerCta        = kRowGroupsPerCta * RowsPerWarp;
};

// Measured winner: four CTA warps, one warp per row group, eight rows per warp, eight BF16 values
// per lane, four accumulator chains, direct activation loads, default weight caching, and
// row-swizzled K phases. Geometry remains a template argument so each exact problem can retain or
// replace the schedule independently after measurement.
template <class Geometry>
struct Bf16LinearDecodeScheduleSelector {
    using Type =
        Bf16GemvSchedule<4, 1, 8, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                         Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
};

template <>
struct Bf16LinearDecodeScheduleSelector<Bf16GemvGeometry<5120, 6144>> {
    using Type =
        Bf16GemvSchedule<8, 2, 2, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                         Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 1>;
};

template <class Geometry>
using Bf16LinearDecodeSchedule = typename Bf16LinearDecodeScheduleSelector<Geometry>::Type;

// ---------------------------------------------------------------------------
// ⚠️ DIAGNOSTIC ONLY -- PATCH A1 (scratch/PATCHSET/A1_schedule_pin.diff)
// ---------------------------------------------------------------------------
// PIN THE SCHEDULE KEY TO ONE CONSTANT, so that ONE build can run two arms
// (width 9 vs width 10) under an IDENTICAL small-T schedule. This is the
// decisive experiment the predecessor named and did NOT run.
//
// WHY IT IS DECISIVE. MEASURED, not quoted: the per-T schedule this header actually
// produces for the 5120x6144 output projection was dumped by host-compiling this very
// header (scratch/PATCHSET/REPORT.md section 2) and is
//   T=2      RPW4 VPL8  WP Seq unroll2      T=3      RPW4 VPL16 WP Seq unroll2
//   T=4      RPW2 VPL8  WP Seq unroll2      T=5,7    RPW4 VPL8  WP Seq unroll2
//   T=6      RPW2 VPL8  WP Seq unroll2      T=8      RPW4 VPL16 WP Seq unroll2
//   T=9      RPW2 VPL8  DS Seq unroll1      T=10..16 RPW2 VPL8  DS RowSwizzled unroll2
// (T=1 is NOT an instance of this schedule at all: bf16_launch.h:13 sets
//  kBf16SmallTMinTokens == 2 and the static_assert below refuses T < 2.)
// So the sweep is not "3 classes": T<=8 alone is FOUR distinct tuples.
// ⚠️ THE SHARP FORM OF THE CONTRADICTION: T=10..16 is ONE template parameter tuple,
// byte-for-byte, and yet the measurement splits it -- w10 -> ae102e65a14de985 while
// w11..16 -> 1a13d088fd735fb2, with w11 == w16 byte-identically. ONE schedule cannot
// produce TWO answers, so the schedule CANNOT be the whole story.
// ⚠️ AND THE NEUTRALITY PROOF CUTS THE OTHER WAY: `--spec none` is bit-identical to
// `--draft-tokens 7`, i.e. the output does NOT move across four distinct T<=8 tuples
// (VPL16 at T=3/T=8, RPW2 at T=4/T=6) -- so a schedule change of this size is
// numerically NEUTRAL there, and "the schedule" is not a free variable that can be
// judged by inspection either way. Pinning settles it in ONE build instead of a
// 16-arm sweep: under kSchedulePin = 8, T=9 and T=10 run the SAME tuple as T=8, so
//   * both arms == 01e4e0d4fa414b71 (== `--spec none`)  => the schedule IS the cause;
//   * both arms equal but != 01e4e0d4fa414b71          => a width-dependent path, not tiling;
//   * the arms still differ from each other            => the sweep is not even the whole
//                                                         width dependence.
//
// `phase0` (bf16_small_t.cuh:101-103: Sequential => 0, else a row-rotated
// phase) IS the fp32 K-accumulation order, because AccumulatorChains is 1 in
// the instantiation below; so this pin moves real arithmetic order, not just
// occupancy, which is exactly what makes it a usable probe.
//
// ONE-LINE REVERT: set kSchedulePin below back to 0 (0 = off, sweep restored).
template <class Geometry, int ActiveTokensRequested>
struct Bf16LinearSmallTProductionSchedule {
    static constexpr int kSchedulePin = 0; // ⚠️ DIAGNOSTIC: 0 = off (per-T sweep)
    static constexpr int ActiveTokens =
        kSchedulePin != 0 ? kSchedulePin : ActiveTokensRequested;
    static_assert(ActiveTokens >= 2 && ActiveTokens <= 32);
    static constexpr bool kOutputProjectionGeometry =
        Geometry::kOutputRows == 5120 && Geometry::kInputRows == 6144;
    // Per-T winners from the complete production sweep. In particular, two rows per warp wins
    // beyond T=8 despite its higher register count because it removes enough repeated issue work.
    static constexpr int kRowsPerWarp =
        kOutputProjectionGeometry
            ? ((ActiveTokens == 4 || ActiveTokens == 6) ? 2 : (ActiveTokens <= 8 ? 4 : 2))
            : (ActiveTokens <= 4 ? 8 : (ActiveTokens <= 8 ? 4 : 2));
    static constexpr int kValuesPerLane =
        kOutputProjectionGeometry && (ActiveTokens == 3 || ActiveTokens == 8) ? 16 : 8;
    static constexpr Bf16SmallTActivationAccess kActivationAccess =
        ActiveTokens <= 8 ? Bf16SmallTActivationAccess::WarpPacked
                          : Bf16SmallTActivationAccess::DirectStream;
    static constexpr bool kSequential = ActiveTokens <= 9 || ActiveTokens >= 17;
    static constexpr bool kUnroll2 =
        kOutputProjectionGeometry
            ? ((ActiveTokens >= 2 && ActiveTokens <= 8) ||
               (ActiveTokens >= 10 && ActiveTokens <= 19) || ActiveTokens >= 27)
            : (ActiveTokens == 4 || ActiveTokens == 5 || ActiveTokens == 8 || ActiveTokens >= 10);
    static constexpr bool kStreaming = !kOutputProjectionGeometry && ActiveTokens == 7;
    static constexpr Bf16WeightCache kWeightCache =
        kStreaming ? Bf16WeightCache::Streaming : Bf16WeightCache::Default;
    static constexpr Bf16PhaseOrder kPhaseOrder =
        kSequential ? Bf16PhaseOrder::Sequential : Bf16PhaseOrder::RowSwizzled;
    using Type =
        Bf16SmallTInnerSchedule<4, 1, kRowsPerWarp, kValuesPerLane, 1, 4, kActivationAccess,
                                kWeightCache, kPhaseOrder, 1, kUnroll2 ? 2 : 1, 1, 2>;
};

} // namespace ninfer::ops::detail
