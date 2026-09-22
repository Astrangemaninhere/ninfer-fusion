#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/silu_mul.h"
#include "core/layout.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Q4LinearSwiGluScheduleId schedule;
};

constexpr std::array<RouteSpec, 10> kRoutes{{
    {{1, 1}, Q4LinearSwiGluScheduleId::GemvPair},
    {{2, 32}, Q4LinearSwiGluScheduleId::SmallTExact},
    {{33, 40}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40},
    {{41, 48}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48},
    {{49, 128}, Q4LinearSwiGluScheduleId::Materialized},
    {{129, 256}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
    {{257, 384}, Q4LinearSwiGluScheduleId::Materialized},
    {{385, 512}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
    {{513, 640}, Q4LinearSwiGluScheduleId::Materialized},
    {{641, kAnyCols}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
}};

// The 4096-wide text stack's geometry: gate_up_rows = 2 x intermediate 12288, output_rows =
// intermediate, k = hidden 4096. It takes this op's generic Materialized route at every column
// count -- a plain q4 linear into a scratch followed by silu_mul, both ops with their own
// registered catalogues -- so a second geometry costs a registration row, not a kernel family.
constexpr std::array<RouteSpec, 1> kK4096Routes{{
    {{1, kAnyCols}, Q4LinearSwiGluScheduleId::Materialized},
}};

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes) noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return routes.back().cols.last == kAnyCols &&
           expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(kRoutes) && catalog_is_closed(kK4096Routes),
              "Q4 LinearSwiGLU routes must be exact, contiguous, and closed");

struct SwiGluGeometry {
    std::int32_t gate_up_rows;
    std::int32_t output_rows;
    std::int32_t k;
    std::int32_t padded_k;
    const RouteSpec* routes;
    std::size_t route_count;
};

constexpr SwiGluGeometry kGeometries[]{
    {34816, 17408, 5120, 5120, kRoutes.data(), kRoutes.size()},
    {24576, 12288, 4096, 4096, kK4096Routes.data(), kK4096Routes.size()},
};

const SwiGluGeometry* find_geometry(const Q4LinearSwiGluProblem& problem) noexcept {
    for (const SwiGluGeometry& geometry : kGeometries) {
        if (problem.gate_up_rows == geometry.gate_up_rows &&
            problem.output_rows == geometry.output_rows && problem.k == geometry.k &&
            problem.padded_k == geometry.padded_k) {
            return &geometry;
        }
    }
    return nullptr;
}

template <class Allocator>
Tensor allocate_materialized_workspace(Allocator& allocator, std::int32_t rows, std::int32_t cols) {
    return allocator.alloc(DType::BF16, {rows, cols});
}

std::size_t materialized_workspace_bytes(std::int32_t rows, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_materialized_workspace(layout, rows, cols);
    return layout.peak_bytes(1);
}

} // namespace

const char* q4_linear_swiglu_schedule_name(Q4LinearSwiGluScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4LinearSwiGluScheduleId::GemvPair:
        return "linear_swiglu.q4.gemv.paired_rows";
    case Q4LinearSwiGluScheduleId::SmallTExact:
        return "linear_swiglu.q4.mma.small_t.exact";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c40";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c48";
    case Q4LinearSwiGluScheduleId::Materialized:
        return "linear_swiglu.q4.materialized";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c128";
    }
    return "linear_swiglu.q4.unknown";
}

bool q4_linear_swiglu_admits(const Q4LinearSwiGluProblem& problem) noexcept {
    return find_geometry(problem) != nullptr && problem.cols >= 1;
}

Q4LinearSwiGluPlan q4_linear_swiglu_resolve_plan(const Q4LinearSwiGluProblem& problem) {
    if (!q4_linear_swiglu_admits(problem)) {
        throw std::invalid_argument(
            "q4 linear_swiglu: exact problem or column count is not admitted");
    }

    const SwiGluGeometry* geometry = find_geometry(problem);
    if (geometry == nullptr) {
        throw std::logic_error("q4 linear_swiglu: admitted problem has no geometry row");
    }
    for (std::size_t i = 0; i < geometry->route_count; ++i) {
        const RouteSpec& route = geometry->routes[i];
        if (!route.cols.contains(problem.cols)) { continue; }
        Q4LinearSwiGluPlan plan{
            route.schedule,
            0,
        };
        switch (route.schedule) {
        case Q4LinearSwiGluScheduleId::GemvPair:
        case Q4LinearSwiGluScheduleId::SmallTExact:
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40:
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48:
            return plan;
        case Q4LinearSwiGluScheduleId::Materialized:
            plan.workspace_bytes = materialized_workspace_bytes(problem.gate_up_rows, problem.cols);
            return plan;
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
            return plan;
        }
    }
    throw std::logic_error("q4 linear_swiglu: admitted problem has no covering route");
}

std::size_t q4_linear_swiglu_capacity_workspace_bytes(std::int32_t gate_up_rows,
                                                      std::int32_t output_rows, std::int32_t k,
                                                      std::int32_t padded_k, std::int32_t min_cols,
                                                      std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("q4 linear_swiglu: invalid column interval");
    }
    (void)q4_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, min_cols});
    (void)q4_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, max_cols});

    const SwiGluGeometry* geometry =
        find_geometry({gate_up_rows, output_rows, k, padded_k, min_cols});
    std::size_t maximum = 0;
    if (geometry != nullptr) {
        for (std::size_t i = 0; i < geometry->route_count; ++i) {
            const RouteSpec& route = geometry->routes[i];
            if (route.cols.last < min_cols || route.cols.first > max_cols) { continue; }
            const std::int32_t endpoint = std::min(route.cols.last, max_cols);
            maximum                     = std::max(maximum, q4_linear_swiglu_resolve_plan(
                                            {gate_up_rows, output_rows, k, padded_k, endpoint})
                                                                .workspace_bytes);
        }
    }
    return maximum;
}

void q4_linear_swiglu_execute_plan(const Q4LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, WorkspaceArena& ws, cudaStream_t stream) {
    const Q4LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q4LinearSwiGluPlan resolved = q4_linear_swiglu_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("q4 linear_swiglu: plan does not match the exact problem");
    }

    switch (plan.schedule) {
    case Q4LinearSwiGluScheduleId::GemvPair:
        q4_linear_swiglu_gemv_pair_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::SmallTExact:
        q4_linear_swiglu_small_t_exact_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40:
        q4_linear_swiglu_mma_split_half_pair_r32_c40_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48:
        q4_linear_swiglu_mma_split_half_pair_r32_c48_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::Materialized: {
        auto scratch_scope = ws.scope();
        Tensor gate_up = allocate_materialized_workspace(ws, problem.gate_up_rows, problem.cols);
        linear(x, w, gate_up, stream);
        silu_mul(gate_up.slice(0, 0, problem.output_rows),
                 gate_up.slice(0, problem.output_rows, problem.output_rows), out, stream);
        return;
    }
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(x, w, out, stream);
        return;
    }
    throw std::logic_error("q4 linear_swiglu: unknown schedule");
}

void q4_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream) {
    const Q4LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q4LinearSwiGluPlan plan = q4_linear_swiglu_resolve_plan(problem);
    q4_linear_swiglu_execute_plan(plan, x, w, out, ws, stream);
}

} // namespace ninfer::ops::detail
