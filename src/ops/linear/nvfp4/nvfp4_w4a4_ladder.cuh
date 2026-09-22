#pragma once

// The single MMA schedule ladder for the W4A4 GEMM family.
//
// Four launchers (linear, linear_add, gdn_input, attn_input) each carried their own copy of
// the same "which schedule runs at this T" decision chain, plus their own copy of the five
// schedule aliases. Those copies are equivalent once the geometry is accounted for: residual
// problems and the GDN projection take different crossovers at T in [65,128] and [385,512],
// and everything else is identical. The ladder is therefore stated once, here, driven by two
// geometry attributes, and every launcher visits it.
//
// INVARIANT: this is the only place the T crossovers and the schedule identities are written.
// Retuning a schedule or a crossover means editing this header and nothing else.
//
// The visit helper turns the runtime shape into a compile-time schedule and calls `visit`
// with an instance of it, so each launcher keeps its own output epilogue while sharing the
// decision.

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

// Geometry attributes the ladder keys off, derived from the geometry itself (never re-typed
// at a call site).
template <class Geometry>
inline constexpr bool kNvfp4IsResidualGeometry =
    std::is_same_v<Geometry, Nvfp4Residual6144Geometry> ||
    std::is_same_v<Geometry, Nvfp4Residual17408Geometry>;

template <class Geometry>
inline constexpr bool kNvfp4IsGdnInputGeometry =
    (Geometry::kOutputRows == Nvfp4GdnInputGeometry::kOutputRows);

enum class Nvfp4W4a4MmaShape : std::uint8_t {
    M32N64,
    M32N128,
    M64N128,
    M128N128Pipelined,
    M128N128Resident,
};

// The schedule identities. These five lines used to be five `using` aliases repeated in every
// launcher translation unit.
template <Nvfp4W4a4MmaShape Shape>
struct Nvfp4W4a4MmaScheduleOf;

template <>
struct Nvfp4W4a4MmaScheduleOf<Nvfp4W4a4MmaShape::M32N64> {
    using Type = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
};
template <>
struct Nvfp4W4a4MmaScheduleOf<Nvfp4W4a4MmaShape::M32N128> {
    using Type = Nvfp4W4a4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
};
template <>
struct Nvfp4W4a4MmaScheduleOf<Nvfp4W4a4MmaShape::M64N128> {
    using Type = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
};
template <>
struct Nvfp4W4a4MmaScheduleOf<Nvfp4W4a4MmaShape::M128N128Pipelined> {
    using Type = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
};
template <>
struct Nvfp4W4a4MmaScheduleOf<Nvfp4W4a4MmaShape::M128N128Resident> {
    using Type = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;
};

[[nodiscard]] inline constexpr Nvfp4W4a4MmaShape
nvfp4_w4a4_mma_shape(std::int32_t tokens, bool residual_geometry, bool gdn_geometry) {
    if (tokens <= 64) { return Nvfp4W4a4MmaShape::M32N64; }
    if (tokens <= 96) { return Nvfp4W4a4MmaShape::M32N128; }
    if (tokens <= 128) {
        return residual_geometry ? Nvfp4W4a4MmaShape::M32N128
                                 : Nvfp4W4a4MmaShape::M128N128Pipelined;
    }
    if (tokens <= 192) { return Nvfp4W4a4MmaShape::M64N128; }
    if (tokens <= 384) { return Nvfp4W4a4MmaShape::M128N128Resident; }
    if (tokens <= 512) {
        return gdn_geometry ? Nvfp4W4a4MmaShape::M128N128Resident
                            : Nvfp4W4a4MmaShape::M128N128Pipelined;
    }
    return Nvfp4W4a4MmaShape::M128N128Resident;
}

template <class Visit>
inline void nvfp4_w4a4_visit_mma_shape(Nvfp4W4a4MmaShape shape, Visit&& visit) {
    switch (shape) {
    case Nvfp4W4a4MmaShape::M32N64:
        visit(typename Nvfp4W4a4MmaScheduleOf<Nvfp4W4a4MmaShape::M32N64>::Type{});
        return;
    case Nvfp4W4a4MmaShape::M32N128:
        visit(typename Nvfp4W4a4MmaScheduleOf<Nvfp4W4a4MmaShape::M32N128>::Type{});
        return;
    case Nvfp4W4a4MmaShape::M64N128:
        visit(typename Nvfp4W4a4MmaScheduleOf<Nvfp4W4a4MmaShape::M64N128>::Type{});
        return;
    case Nvfp4W4a4MmaShape::M128N128Pipelined:
        visit(typename Nvfp4W4a4MmaScheduleOf<Nvfp4W4a4MmaShape::M128N128Pipelined>::Type{});
        return;
    case Nvfp4W4a4MmaShape::M128N128Resident:
        visit(typename Nvfp4W4a4MmaScheduleOf<Nvfp4W4a4MmaShape::M128N128Resident>::Type{});
        return;
    }
}

} // namespace ninfer::ops::detail
