#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

enum class Nvfp4ScaleAccess : std::uint8_t {
    StagedRaw,
    Direct,
};

enum class Nvfp4CodeCache : std::uint8_t {
    Default,
    Streaming,
};

enum class Nvfp4SmallTActivationAccess : std::uint8_t {
    PairStream,
    TokenPacked,
    SharedPhase,
};

enum class Nvfp4SmallTBlockOrder : std::uint8_t {
    RowsContiguous,
    TokenTilesContiguous,
};

template <std::int32_t OutputRows, std::int32_t InputRows>
struct Nvfp4GemvGeometry {
    static_assert(OutputRows > 0 && InputRows > 0);
    static_assert((OutputRows % 128) == 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kOutputRows       = OutputRows;
    static constexpr std::int32_t kInputRows        = InputRows;
    static constexpr std::int32_t kGroupsPerRow     = InputRows / 16;
    static constexpr std::int32_t kScaleTilesPerRow = InputRows / 64;
    static constexpr std::int32_t kCodeBytesPerRow  = InputRows / 2;
};

template <std::int32_t InputRows>
struct Nvfp4ActivationGeometry {
    static_assert(InputRows > 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kInputRows       = InputRows;
    static constexpr std::int32_t kGroupsPerRow    = InputRows / 16;
    static constexpr std::int32_t kCodeBytesPerRow = InputRows / 2;
};

template <int WarpsPerCta, int RowsPerWarp, int ValuesPerLane, int AccumulatorChains,
          Nvfp4ScaleAccess ScaleAccess, Nvfp4CodeCache CodeCache, int MinBlocksPerSm,
          int PhaseUnroll = 0>
struct Nvfp4GemvSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16 || ValuesPerLane == 32);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane / 2);
    static_assert(MinBlocksPerSm > 0);
    static_assert(PhaseUnroll == 0 || PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    // 0 keeps `#pragma unroll` (full) exactly as before this patch; 1/2/4 select a partial
    // unroll of the K-phase loop. The fp8 sibling of this kernel has had that knob all along
    // (Fp8GemvSchedule::kPhaseUnroll, fp8_config.h:41, measured winner 2).
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowsPerCta        = WarpsPerCta * RowsPerWarp;
    static constexpr int kPairsPerLane      = ValuesPerLane / 2;
};

template <int WarpsPerCta, int WarpsPerRow, int RowsPerWarp, int ValuesPerLane, int TokenTile,
          int AccumulatorChains, Nvfp4SmallTActivationAccess ActivationAccess,
          Nvfp4ScaleAccess ScaleAccess, Nvfp4CodeCache CodeCache, int PhaseUnroll,
          Nvfp4SmallTBlockOrder BlockOrder, int MinBlocksPerSm>
struct Nvfp4SmallTSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(WarpsPerRow > 0 && WarpsPerRow <= WarpsPerCta);
    static_assert((WarpsPerCta % WarpsPerRow) == 0);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16 || ValuesPerLane == 32);
    static_assert(TokenTile > 0);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane / 2);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kWarpsPerRow       = WarpsPerRow;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kTokenTile         = TokenTile;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kActivationAccess = ActivationAccess;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr auto kBlockOrder       = BlockOrder;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowGroupsPerCta   = WarpsPerCta / WarpsPerRow;
    static constexpr int kRowsPerCta        = kRowGroupsPerCta * RowsPerWarp;
    static constexpr int kPairsPerLane      = ValuesPerLane / 2;
};

using Nvfp4AttnInputGeometry     = Nvfp4GemvGeometry<14336, 5120>;
using Nvfp4GdnInputGeometry      = Nvfp4GemvGeometry<16384, 5120>;
using Nvfp4MlpGateUpGeometry     = Nvfp4GemvGeometry<34816, 5120>;
using Nvfp4Residual6144Geometry  = Nvfp4GemvGeometry<5120, 6144>;
using Nvfp4Residual17408Geometry = Nvfp4GemvGeometry<5120, 17408>;

// Muse-Glimmer-30B geometries (hidden 6656 MLP + NVFP4 output head).
using Nvfp4MuseMlpGateUpGeometry = Nvfp4GemvGeometry<19968, 6656>;
using Nvfp4MuseMlpDownGeometry   = Nvfp4GemvGeometry<6656, 19968>;
using Nvfp4MuseVocabularyGeometry = Nvfp4GemvGeometry<202112, 6656>;

using Nvfp4Activation5120Geometry  = Nvfp4ActivationGeometry<5120>;
using Nvfp4Activation6144Geometry  = Nvfp4ActivationGeometry<6144>;
using Nvfp4Activation17408Geometry = Nvfp4ActivationGeometry<17408>;

// Row split of the two fused projections. This is the ONLY place the QKV/Z and Q/K/Gate/V
// row counts are written down: the launchers that stride those outputs, the workspace shapes
// and the row-routing epilogues all reference these, so a geometry change fails to compile
// instead of silently mis-striding one producer.
inline constexpr std::int32_t kNvfp4GdnQkvRows    = 10240;
inline constexpr std::int32_t kNvfp4GdnZRows      = 6144;
inline constexpr std::int32_t kNvfp4AttnQueryRows = 6144;
inline constexpr std::int32_t kNvfp4AttnKeyRows   = 1024;
inline constexpr std::int32_t kNvfp4AttnGateRows  = 6144;
inline constexpr std::int32_t kNvfp4AttnKeyBegin  = kNvfp4AttnQueryRows;
inline constexpr std::int32_t kNvfp4AttnGateBegin = kNvfp4AttnKeyBegin + kNvfp4AttnKeyRows;
inline constexpr std::int32_t kNvfp4AttnValueBegin = kNvfp4AttnGateBegin + kNvfp4AttnGateRows;

static_assert(kNvfp4GdnQkvRows + kNvfp4GdnZRows == Nvfp4GdnInputGeometry::kOutputRows,
              "GDN fused projection row split must tile the geometry's output rows");
static_assert(kNvfp4AttnValueBegin + kNvfp4AttnKeyRows == Nvfp4AttnInputGeometry::kOutputRows,
              "attention fused projection row split must tile the geometry's output rows");
static_assert((kNvfp4GdnQkvRows % 128) == 0 && (kNvfp4GdnZRows % 128) == 0,
              "GDN rows must tile the 128-row block");
static_assert((kNvfp4AttnQueryRows % 128) == 0 && (kNvfp4AttnKeyRows % 128) == 0 &&
                  (kNvfp4AttnGateRows % 128) == 0,
              "attention rows must tile the 128-row block");

enum class Nvfp4Problem : std::uint8_t {
    AttnInput,
    GdnInput,
    MlpGateUp,
    Residual6144,
    Residual17408,
    MuseMlpGateUp,
    MuseMlpDown,
    MuseVocabulary,
};

inline constexpr bool is_nvfp4_linear_problem(std::int32_t output_rows, std::int32_t input_rows) {
    return (output_rows == Nvfp4AttnInputGeometry::kOutputRows &&
            input_rows == Nvfp4AttnInputGeometry::kInputRows) ||
           (output_rows == Nvfp4GdnInputGeometry::kOutputRows &&
            input_rows == Nvfp4GdnInputGeometry::kInputRows) ||
           (output_rows == Nvfp4MlpGateUpGeometry::kOutputRows &&
            input_rows == Nvfp4MlpGateUpGeometry::kInputRows) ||
           (output_rows == Nvfp4Residual6144Geometry::kOutputRows &&
            input_rows == Nvfp4Residual6144Geometry::kInputRows) ||
           (output_rows == Nvfp4Residual17408Geometry::kOutputRows &&
            input_rows == Nvfp4Residual17408Geometry::kInputRows) ||
           (output_rows == Nvfp4MuseMlpGateUpGeometry::kOutputRows &&
            input_rows == Nvfp4MuseMlpGateUpGeometry::kInputRows) ||
           (output_rows == Nvfp4MuseMlpDownGeometry::kOutputRows &&
            input_rows == Nvfp4MuseMlpDownGeometry::kInputRows) ||
           (output_rows == Nvfp4MuseVocabularyGeometry::kOutputRows &&
            input_rows == Nvfp4MuseVocabularyGeometry::kInputRows);
}

inline Nvfp4Problem resolve_nvfp4_problem(std::int32_t output_rows, std::int32_t input_rows) {
    if (output_rows == Nvfp4AttnInputGeometry::kOutputRows &&
        input_rows == Nvfp4AttnInputGeometry::kInputRows) {
        return Nvfp4Problem::AttnInput;
    }
    if (output_rows == Nvfp4GdnInputGeometry::kOutputRows &&
        input_rows == Nvfp4GdnInputGeometry::kInputRows) {
        return Nvfp4Problem::GdnInput;
    }
    if (output_rows == Nvfp4MlpGateUpGeometry::kOutputRows &&
        input_rows == Nvfp4MlpGateUpGeometry::kInputRows) {
        return Nvfp4Problem::MlpGateUp;
    }
    if (output_rows == Nvfp4Residual6144Geometry::kOutputRows &&
        input_rows == Nvfp4Residual6144Geometry::kInputRows) {
        return Nvfp4Problem::Residual6144;
    }
    if (output_rows == Nvfp4Residual17408Geometry::kOutputRows &&
        input_rows == Nvfp4Residual17408Geometry::kInputRows) {
        return Nvfp4Problem::Residual17408;
    }
    if (output_rows == Nvfp4MuseMlpGateUpGeometry::kOutputRows &&
        input_rows == Nvfp4MuseMlpGateUpGeometry::kInputRows) {
        return Nvfp4Problem::MuseMlpGateUp;
    }
    if (output_rows == Nvfp4MuseMlpDownGeometry::kOutputRows &&
        input_rows == Nvfp4MuseMlpDownGeometry::kInputRows) {
        return Nvfp4Problem::MuseMlpDown;
    }
    if (output_rows == Nvfp4MuseVocabularyGeometry::kOutputRows &&
        input_rows == Nvfp4MuseVocabularyGeometry::kInputRows) {
        return Nvfp4Problem::MuseVocabulary;
    }
    throw std::invalid_argument("unsupported NVFP4 problem");
}

// RTX 5090 cold-cache winner among the measured decode schedules.
template <class Geometry>
struct Nvfp4LinearDecodeProductionSchedule {
    using Type =
        Nvfp4GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2, 2>;
};

inline constexpr std::int32_t kNvfp4FirstSmallT = 2;
inline constexpr std::int32_t kNvfp4LastSmallT  = 32;

// Length of a whole-family small-T launcher table. Written once: every table below sizes
// itself with this and every lookup is range-checked against the ceiling that registered it.
inline constexpr std::int32_t kNvfp4SmallTCount = kNvfp4LastSmallT - kNvfp4FirstSmallT + 1;

// The A16 (W16A16) tier is registered separately from the A4 small-T family above, so each
// family owns the ceiling its registrar enforces and its launcher table is exactly as long
// as that ceiling implies.
inline constexpr std::int32_t kNvfp4LinearSwiGluA16Ceiling = 16;
inline constexpr std::int32_t kNvfp4GdnConvA16Ceiling      = 16;

// Range-checked index into a per-T launcher table. Tables are indexed by
// (tokens - kNvfp4FirstSmallT); an out-of-range T used to walk off the end of the table
// silently, so the contract is enforced here once for every caller.
[[nodiscard]] inline std::size_t nvfp4_small_t_index(std::int32_t tokens, std::int32_t ceiling,
                                                     const char* what) {
    if (tokens < kNvfp4FirstSmallT || tokens > ceiling) {
        throw std::invalid_argument(std::string(what) +
                                    ": T outside the registered small-T range");
    }
    return static_cast<std::size_t>(tokens - kNvfp4FirstSmallT);
}

// Compile-time-ceiling spelling of the same lookup. The call sites that already know the
// ceiling their registrar enforces (nvfp4_linear_add_small_t) name it in the template
// argument instead of repeating it as a runtime literal; the range check is identical
// because it is delegated to the 3-argument form above.
template <std::int32_t Ceiling>
[[nodiscard]] inline std::size_t nvfp4_small_t_index(std::int32_t tokens, const char* what) {
    return nvfp4_small_t_index(tokens, Ceiling, what);
}

// Same contract, plus the table length is tied to the ceiling its registrar enforces.
template <std::int32_t Ceiling, class Launcher, std::size_t Count>
[[nodiscard]] inline Launcher nvfp4_small_t_launcher(const std::array<Launcher, Count>& table,
                                                     std::int32_t tokens, const char* what) {
    static_assert(Count > 0, "empty small-T launcher table");
    static_assert(static_cast<std::int32_t>(Count) == Ceiling - kNvfp4FirstSmallT + 1,
                  "small-T launcher table length must match its registered ceiling");
    return table[nvfp4_small_t_index(tokens, Ceiling, what)];
}

// RTX 5090 cold-cache winners for contiguous Linear output. T=2..4 amortizes activation loads
// through shared staging; T=5..32 keeps one packed activation tile per warp. The warp-count changes
// are measured occupancy/register crossovers, not semantic frontiers.
template <class Geometry, int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta   = ActiveTokens >= 17 ? 4 : (ActiveTokens >= 13 ? 16 : 8);
    static constexpr int kValuesPerLane = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Nvfp4SmallTActivationAccess::SharedPhase
                                                  : Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// G1's wider N benefits from keeping four warps per CTA throughout the A16 policy boundary. Only
// T=2 amortizes activation traffic enough for shared staging to win.
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4GdnInputGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta       = 4;
    static constexpr int kValuesPerLane     = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens == 2
                                                  ? Nvfp4SmallTActivationAccess::SharedPhase
                                                  : Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// At N=5120, R1 needs the larger CTA only for the last three A16 token counts. The unoptimized
// A16-only tail keeps the established generic schedule.
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual6144Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta   = ActiveTokens <= 16 ? (ActiveTokens >= 14 ? 16 : 4) : 4;
    static constexpr int kValuesPerLane = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// R2's longer K moves the stable four-to-sixteen-warp crossover to T=8.
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual17408Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta       = ActiveTokens <= 16 ? (ActiveTokens >= 8 ? 16 : 4) : 4;
    static constexpr int kValuesPerLane     = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

} // namespace ninfer::ops::detail
