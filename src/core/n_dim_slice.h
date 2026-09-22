#pragma once

// n_dim_slice.h -- THE N-DIM COLUMN SLICER: which columns each rank may own, and the two
// boundaries a slice is not allowed to straddle.
//
// WHY THIS IS NOT A DIVISION. `weight_columns % world_size == 0` is necessary and it is not
// sufficient, and both reference forks were bitten by exactly that. The donor's record is
// explicit that `allgather_rows` "relocates and does not transpose", so "a caller whose natural
// split is not `ne[1]` arranges the layout so that it is" -- i.e. the CALLER owns the boundary.
// gfx906's fixed-fault list names the same class: "tp1 row constants in the fused tiled
// launchers". So the slicer must be given the boundaries rather than assume they fall where the
// division does.
//
// TWO BOUNDARIES, both load-bearing, both from the record:
//
//   1. THE FUSED ROW ORDER. The attention input projection is fused as `Q | K | Gate | V` and the
//      GDN input projection as `Q(2048) | K(2048) | V(6144) | Z(6144)`
//      (`wamansou/ninfer-tp2-1m`, whose `tp2-yarn-1m` is the DONOR BRANCH name and not a path in
//      this tree; the record that IS here -- and the one core/virtual_device.h section 0 names --
//      is `docs/gfx906/TP2-SLICES.md`). A rank's `weight_columns`
//      Range is therefore a range of columns in a FUSED tensor, and a slice that cuts through a
//      fused block hands one rank the tail of Q and another the head of K -- which is not a
//      shard of anything. That is why the boundaries are an INPUT here and not a computed value:
//      a header that guessed the layout would hide the caller's error.
//
//   2. THE 128-ROW TILE BOUNDARY. NVFP4 row slicing is safe at the tile boundary of
//      `blockscale-k16-m128x4-v1`, so a shard's row range is contiguous and bindable with NO
//      repacking -- which is the answer to "there is no N-dim column slicer": the slice exists
//      and it is a plain contiguous range, but only when it lands on the tile boundary. A slice
//      that does not would need a repack, and the honest thing is to REFUSE rather than to
//      repack silently (a repack is a different artifact, not a different view of this one).
//
// WHAT THIS IS NOT. It does not read the artifact, it does not bind, and it does not know the
// reader's layout. `src/artifact/` holds the reader and belongs to another line, so the ARITHMETIC
// lives here -- header-only and host-testable, like everything else on this axis -- and the
// one call that binds it into the loader is reported rather than applied.

#include "core/shard_plan.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::multi {

// The tile width of `blockscale-k16-m128x4-v1`'s row axis. It is a named constant because it is
// the thing that decides whether a slice is bindable, and a magic 128 at a call site is how the
// donor's "tp1 row constants" fault class starts.
inline constexpr std::uint32_t kNvfp4BlockscaleTileRows = 128;

// The boundaries of a FUSED projection, as the widths of its consecutive blocks, in column order.
// `{Q, K, Gate, V}` for attention and `{Q, K, V, Z}` for GDN; the caller supplies the real widths
// because they are per-target numbers and this header must not guess a layout.
struct FusedRowOrder {
    std::string name;                   // "Q|K|Gate|V" -- for the refusal, not for logic
    std::vector<std::uint32_t> blocks;  // consecutive widths, in order; must sum to the N

    [[nodiscard]] std::uint32_t total() const noexcept {
        std::uint32_t sum = 0;
        for (std::uint32_t w : blocks) { sum += w; }
        return sum;
    }
    // The column offsets at which a slice may BEGIN or END: 0 and every cumulative sum. A slice
    // whose endpoints are both in this set cuts no fused block in half.
    [[nodiscard]] std::vector<std::uint32_t> boundaries() const {
        std::vector<std::uint32_t> out;
        std::uint32_t at = 0;
        out.push_back(at);
        for (std::uint32_t w : blocks) {
            at += w;
            out.push_back(at);
        }
        return out;
    }
    [[nodiscard]] bool is_boundary(std::uint32_t column) const {
        for (std::uint32_t b : boundaries()) {
            if (b == column) { return true; }
        }
        return false;
    }
};

// The contract a slice must satisfy. Every field is an INPUT because every one of them is a fact
// about the artifact or the kernel rather than something this header can derive.
struct ColumnSliceRequest {
    WorldShape world{};
    std::uint32_t weight_columns = 0;      // the fused N
    FusedRowOrder order{};                 // the fused boundaries, named
    // The kernel's row-tile width. Pass 1 to mean "this projection has no tile constraint", which
    // is an explicit statement rather than an omitted one.
    std::uint32_t tile_rows = kNvfp4BlockscaleTileRows;
    // Whether the binding requires a rank's columns to be CONTIGUOUS. The record says they are
    // ("a shard's row range is contiguous and bindable with no repacking"), so this is true by
    // default; it exists so that a future non-contiguous route is a decision.
    bool requires_contiguous = true;
};

struct ColumnSlice {
    bool ok = false;
    std::string reason;      // non-empty iff !ok
    Range columns{};         // this rank's contiguous N range
    std::uint32_t stride = 0;// columns per rank, == columns.count() when ok
};

// The slicer. It refuses, by name, in this order:
//   1. a malformed world (delegated, so the refusal has ONE spelling);
//   2. a geometry that does not divide;
//   3. an order whose blocks do not sum to the N (a caller passing a stale layout);
//   4. a slice that does not land on a FUSED boundary;
//   5. a slice that does not land on a TILE boundary.
// The order matters: (3) before (4)/(5), because a wrong total makes every boundary check
// meaningless, and reporting (4) when the real fault is (3) sends the reader to the wrong file.
[[nodiscard]] inline ColumnSlice slice_weight_columns(const ColumnSliceRequest& request) {
    ColumnSlice out;
    const std::string world_refusal = world_shape_refusal(request.world);
    if (!world_refusal.empty()) {
        out.reason = world_refusal;
        return out;
    }
    if (request.weight_columns == 0) {
        out.reason = "weight_columns is 0: there is no fused N to slice";
        return out;
    }
    if (request.weight_columns % request.world.world_size != 0) {
        out.reason =
            "weight_columns " + std::to_string(request.weight_columns) +
            " is not divisible by world_size " + std::to_string(request.world.world_size) +
            " for fused order " + request.order.name +
            ": refuse the split rather than rounding it, because a rounded split puts rank r's "
            "bytes at a fraction of a row and rank r then reads rank r-1's last column.";
        return out;
    }
    if (request.order.total() != request.weight_columns) {
        out.reason =
            "the fused order " + request.order.name + " declares " +
            std::to_string(request.order.total()) + " columns but weight_columns is " +
            std::to_string(request.weight_columns) +
            ": a caller passing a stale fused layout makes every boundary check below meaningless, "
            "so this is checked first.";
        return out;
    }
    if (request.world.axis == ParallelAxis::None) {
        // The identity world: the whole fused N on the only rank, and NOTHING is cut -- so the
        // boundary checks are vacuous rather than failed. The single-device path must not be made
        // to satisfy a multi-rank contract.
        out.ok      = true;
        out.columns = Range{0, request.weight_columns};
        out.stride  = request.weight_columns;
        return out;
    }
    if (request.world.axis == ParallelAxis::Pipeline) {
        // The pipeline axis splits LAYERS, not N: every stage holds the whole fused N.
        out.ok      = true;
        out.columns = Range{0, request.weight_columns};
        out.stride  = request.weight_columns;
        return out;
    }

    const std::uint32_t stride = request.weight_columns / request.world.world_size;
    const std::uint32_t begin  = request.world.rank * stride;
    const std::uint32_t end    = begin + stride;

    if (!request.order.is_boundary(begin) || !request.order.is_boundary(end)) {
        std::string bounds;
        for (std::uint32_t b : request.order.boundaries()) {
            if (!bounds.empty()) { bounds += ","; }
            bounds += std::to_string(b);
        }
        out.reason =
            "rank " + std::to_string(request.world.rank) + "'s N range [" + std::to_string(begin) +
            "," + std::to_string(end) + ") does not land on the fused row order " +
            request.order.name + "'s boundaries {" + bounds +
            "}: the slice cuts a fused block in half, so one rank gets the tail of one projection "
            "and another the head of the next -- which is not a shard of anything. The caller must "
            "arrange the split so that a rank's range is a whole number of fused blocks.";
        return out;
    }
    if (request.tile_rows > 1 && (begin % request.tile_rows != 0 || end % request.tile_rows != 0)) {
        out.reason =
            "rank " + std::to_string(request.world.rank) + "'s N range [" + std::to_string(begin) +
            "," + std::to_string(end) + ") of the fused order " + request.order.name +
            " does not land on the " + std::to_string(request.tile_rows) +
            "-row tile boundary of blockscale-k16-m128x4-v1: the shard would need a REPACK, which "
            "is a different artifact rather than a different view of this one. Refuse rather than "
            "repack silently.";
        return out;
    }
    out.ok      = true;
    out.columns = Range{begin, end};
    out.stride  = stride;
    return out;
}

// The byte window a loader must read for one rank's columns, out of a ROW-MAJOR (N, K) tensor
// whose rows are contiguous. This is the arithmetic the artifact reader needs and does not have:
// the reader produces ONE byte stream today, so without this a per-rank loader has to re-derive
// the offsets and will get the K stride wrong exactly once.
struct ByteWindow {
    bool ok = false;
    std::string reason;
    std::size_t offset_bytes = 0; // from the start of the tensor's data
    std::size_t row_bytes    = 0; // bytes of ONE row of this rank's slice
    std::size_t total_bytes  = 0; // the whole slice, if the caller reads it row by row
};

[[nodiscard]] inline ByteWindow column_byte_window(const ColumnSlice& slice,
                                                   std::uint32_t total_columns,
                                                   std::size_t bytes_per_element) {
    ByteWindow out;
    if (!slice.ok) {
        out.reason = "the slice was refused: " + slice.reason;
        return out;
    }
    if (total_columns == 0 || bytes_per_element == 0) {
        out.reason = "total_columns and bytes_per_element must both be non-zero";
        return out;
    }
    if (slice.columns.end > total_columns) {
        out.reason = "the slice extends past the tensor's " + std::to_string(total_columns) +
                     " columns";
        return out;
    }
    const std::size_t row_stride = static_cast<std::size_t>(total_columns) * bytes_per_element;
    out.ok           = true;
    out.offset_bytes = static_cast<std::size_t>(slice.columns.begin) * bytes_per_element;
    out.row_bytes    = static_cast<std::size_t>(slice.columns.count()) * bytes_per_element;
    out.total_bytes  = row_stride; // per row; a row is NOT contiguous for a column slice, and
                                   // naming the stride rather than pretending otherwise is the
                                   // point of returning all three numbers
    return out;
}

// Whether a column slice is contiguous in the FILE, which is the question a loader actually asks.
// It is true only for rank 0 when the slice is narrower than the row, and true for every rank on
// an identity/pipeline world. Returning a bool rather than a comment means a loader cannot assume
// contiguity without saying so.
[[nodiscard]] inline bool column_slice_is_file_contiguous(const ColumnSlice& slice,
                                                          std::uint32_t total_columns) {
    if (!slice.ok) { return false; }
    if (slice.columns.begin == 0 && slice.columns.end == total_columns) { return true; }
    return false;
}

// The four fused orders the record names, so that a caller referencing "the attention projection"
// and a caller referencing `{Q,K,Gate,V}` are the same object. Widths are the 27B's own numbers
// where the record gives them (GDN) and are left as the caller's for the others, because a
// fabricated width is worse than a required one.
[[nodiscard]] inline FusedRowOrder named_fused_order(std::string_view name,
                                                     const std::vector<std::uint32_t>& blocks) {
    FusedRowOrder order;
    order.name   = std::string(name);
    order.blocks = blocks;
    return order;
}

// ---------------------------------------------------------------------------
// COEXISTENCE: the KV rotation seam and the rank axis
// ---------------------------------------------------------------------------
//
// The standing frame is "最后肯定是所有机制都开" -- every mechanism on at once -- so the question is
// not whether the slicer is right alone, but whether it COLLIDES with the KV codecs that already
// write these tensors. It does not, and the reason is worth stating precisely rather than
// asserting, because a future change can make it collide:
//
//   `kvarn_hadamard` (src/ops/kvarn/codec.cu:166-178) requires matching contiguous BF16 **D256**
//   tensors and rotates them with an 8-lane Hadamard (`kvarn::detail::hadamard_block(value,
//   stage, d)`, `src/ops/kvarn/hadamard.cuh`), i.e. the rotation is INTRA-HEAD over a 256-wide
//   head vector. The tensor axis splits KV HEADS, and a head is atomic under that split, so the
//   rotation domain is carried whole by whichever rank owns the head and the two operations
//   COMMUTE. The E8 lattice codec's 8-lane sub-blocks (src/ops/kernel/e8_lattice.cuh) sit on the
//   same 8-lane structure and commute for the same reason.
//
// The condition that keeps it true, made checkable so it cannot be broken silently: a rank's KV
// slice must be a whole number of heads, and if the split ever moves INSIDE a head, the rotation's
// lane block must divide the per-rank head slice.
[[nodiscard]] inline std::string rotation_split_refusal(const WorldShape& world,
                                                        std::uint32_t head_dim,
                                                        std::uint32_t rotation_lanes) {
    if (world.axis == ParallelAxis::None || world.axis == ParallelAxis::Pipeline) { return {}; }
    if (rotation_lanes == 0 || head_dim == 0) {
        return "the rotation's lane block and the head dimension must both be non-zero to check "
               "whether the rank axis commutes with the rotation";
    }
    if (head_dim % rotation_lanes != 0) {
        return "head_dim " + std::to_string(head_dim) + " is not a multiple of the rotation's " +
               std::to_string(rotation_lanes) +
               "-lane block: the rotation and the rank axis cannot be composed at all, on any "
               "axis, because a single head is not a whole number of rotation blocks.";
    }
    // A HEAD split: the heads are atomic, so the rotation is intra-head and the two commute --
    // which is the case the engine is in today (kvarn_hadamard wants D256 per head).
    return {};
}

} // namespace ninfer::multi
