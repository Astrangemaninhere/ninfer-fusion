// test_n_dim_slice.cpp -- the N-dim column slicer: the two boundaries a slice may not straddle,
// the byte window a per-rank loader needs, and the three coexistence questions the standing frame
// ("every mechanism on at once") forces.
//
// Host-only: pure arithmetic, no CUDA header, no device. What it proves is that a slice which
// satisfies `weight_columns % world_size == 0` and NOTHING ELSE is still REFUSED -- which is the
// whole reason this file exists, because the divisor check alone is the mistake both TP2
// reference forks had to fix after the fact ("tp1 row constants in the fused tiled launchers").

#include "core/n_dim_slice.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace ninfer::multi;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

// The attention input projection's fused order, as the record names it: Q | K | Gate | V.
// Widths chosen so that the total divides by 2, 4 and 8 and every boundary is a multiple of 128.
FusedRowOrder attention_order() {
    return named_fused_order("Q|K|Gate|V", {1024, 256, 1024, 256}); // total 2560
}

ColumnSliceRequest base_request(std::uint32_t world_size, std::uint32_t rank) {
    ColumnSliceRequest r;
    r.world.world_size   = world_size;
    r.world.rank         = rank;
    r.world.axis         = ParallelAxis::Tensor;
    r.weight_columns     = 2560;
    r.order              = attention_order();
    r.tile_rows          = 128;
    r.requires_contiguous = true;
    return r;
}

// ---------------------------------------------------------------------------
// Group E: the two boundaries
// ---------------------------------------------------------------------------
void group_E_boundaries() {
    // E1: the fused order's boundaries are the cumulative sums, so the check is a set membership
    // and not a guess about the layout.
    {
        const FusedRowOrder o = attention_order();
        check(o.total() == 2560, "E1 the fused blocks sum to the declared N");
        const std::vector<std::uint32_t> b = o.boundaries();
        check(b.size() == 5, "E1 four blocks give five boundaries including both ends");
        check(b.front() == 0 && b.back() == 2560, "E1 the boundaries span the whole N");
        check(o.is_boundary(1024) && o.is_boundary(1280) && o.is_boundary(2304),
              "E1 every fused block edge is a boundary");
        check(!o.is_boundary(1152), "E1 red control: a column inside Q is not a boundary");
    }

    // E2: THE DIVISOR CHECK ALONE IS NOT ENOUGH. world 4 divides 2560 exactly (640 per rank) and
    // yet rank 1's range [640,1280) cuts Q in half. This is the check the divisor does not make,
    // and it is the one both forks had to fix after the fact.
    {
        const ColumnSlice s = slice_weight_columns(base_request(4, 1));
        check(!s.ok, "E2 world 4 divides 2560 and the slice is STILL refused");
        check(s.reason.find("does not land on the fused row order") != std::string::npos,
              "E2 the refusal names the fused order");
        check(s.reason.find("{0,1024,1280,2304,2560}") != std::string::npos,
              "E2 and prints the boundaries the caller must land on");
    }

    // E3: a slice that DOES land on the boundaries is accepted, and its range is the arithmetic
    // the caller expected -- so the refusal in E2 is about the boundary and not about world 4.
    {
        // With 8 blocks of 320 the order divides by 4 cleanly; 320 is not a multiple of 128, so
        // the tile check must be satisfied separately.
        ColumnSliceRequest r = base_request(4, 2);
        r.order = named_fused_order("even-4x640", {640, 640, 640, 640});
        const ColumnSlice s = slice_weight_columns(r);
        check(s.ok, "E3 an even 4-way order at a 128 boundary is accepted");
        check(s.columns.begin == 1280 && s.columns.end == 1920,
              "E3 rank 2 owns the third quarter");
        check(s.stride == 640, "E3 and the stride is the per-rank width");
    }

    // E4: THE TILE BOUNDARY. A slice that lands on the fused boundaries but NOT on the 128-row
    // tile boundary of blockscale-k16-m128x4-v1 must be refused, because binding it needs a
    // repack -- which is a different artifact, so the honest answer is no.
    //
    // The numbers have to be chosen so that the tile check is REACHABLE: a two-block order of
    // 576|576 gives stride 576, which IS a fused boundary (so the fused check passes) and is NOT
    // a multiple of 128 (576 = 4x128 + 64), so the tile check is what refuses. Choosing the
    // numbers carelessly makes the fused check fire first and the test would then be asserting
    // the wrong refusal -- which is exactly what it did on the first run.
    {
        ColumnSliceRequest r = base_request(2, 1);
        r.order          = named_fused_order("two-halves-576", {576, 576});
        r.weight_columns = 1152;
        r.tile_rows      = 128;
        check(r.order.total() == r.weight_columns,
              "E4 the order sums to N, so the boundary checks are reachable");
        check(r.weight_columns % r.world.world_size == 0,
              "E4 and N divides, so the DIVISOR check passes");
        check(r.order.is_boundary(576),
              "E4 and 576 IS a fused boundary, so the FUSED check passes too");
        check(576u % 128u != 0u, "E4 and 576 is NOT a multiple of the tile width");
        const ColumnSlice s = slice_weight_columns(r);
        check(!s.ok, "E4 a slice off the 128-row tile boundary is refused");
        check(s.reason.find("does not land on the 128-row tile boundary") != std::string::npos,
              "E4 and the refusal names the tile width");
        check(s.reason.find("REPACK") != std::string::npos,
              "E4 and says why: a repack is a different artifact");
    }
    // E4 red control: the same shape with the tile constraint declared ABSENT is accepted, so the
    // refusal is about the tile and not about the shape.
    {
        ColumnSliceRequest r = base_request(2, 1);
        r.order          = named_fused_order("two-halves-576", {576, 576});
        r.weight_columns = 1152;
        r.tile_rows      = 1; // explicit "this projection has no tile constraint"
        const ColumnSlice s = slice_weight_columns(r);
        check(s.ok, "E4 red control: the same slice with tile_rows=1 is accepted");
        check(s.columns.begin == 576 && s.columns.end == 1152,
              "E4 and rank 1 owns the second half");
    }

    // E5: the divergences. A non-divisible N is refused rather than rounded, and a stale fused
    // order is refused BEFORE the boundary check, so the reader is sent to the right file.
    {
        ColumnSliceRequest r = base_request(3, 0);
        r.order = named_fused_order("thirds", {1024, 256, 1024, 256});
        const ColumnSlice s = slice_weight_columns(r);
        check(!s.ok, "E5 a non-divisible world is refused");
        check(s.reason.find("refuse the split rather than rounding it") != std::string::npos,
              "E5 and says so in those words");
    }
    {
        ColumnSliceRequest r = base_request(2, 0);
        r.order = named_fused_order("stale", {1024, 256, 1024}); // sums to 2304, not 2560
        const ColumnSlice s = slice_weight_columns(r);
        check(!s.ok, "E5 a fused order that does not sum to N is refused");
        check(s.reason.find("checked first") != std::string::npos,
              "E5 and says the total is checked before the boundaries");
    }
    {
        ColumnSliceRequest r = base_request(2, 0);
        r.weight_columns = 0;
        check(!slice_weight_columns(r).ok, "E5 a zero N is refused");
    }

    // E6: the identity and pipeline worlds take the WHOLE N and are not made to satisfy a
    // multi-rank contract. This is the property that keeps the single-device path unchanged.
    {
        ColumnSliceRequest r = base_request(1, 0);
        r.world.axis = ParallelAxis::None;
        const ColumnSlice s = slice_weight_columns(r);
        check(s.ok && s.columns.begin == 0 && s.columns.end == 2560,
              "E6 axis=none takes the whole N");
        check(slice_weight_columns(r).reason.empty(), "E6 with no refusal");

        ColumnSliceRequest p = base_request(4, 3);
        p.world.axis = ParallelAxis::Pipeline;
        const ColumnSlice ps = slice_weight_columns(p);
        check(ps.ok && ps.columns.count() == 2560,
              "E6 the pipeline axis splits layers, so every stage keeps the whole N");
    }
}

// ---------------------------------------------------------------------------
// Group F: the byte window a per-rank loader needs
// ---------------------------------------------------------------------------
void group_F_byte_window() {
    ColumnSliceRequest r = base_request(4, 2);
    r.order = named_fused_order("even-4x640", {640, 640, 640, 640});
    const ColumnSlice s = slice_weight_columns(r);
    check(s.ok, "F1 the slice is valid to begin with");

    // F2: the three numbers, and the one a loader gets wrong. A column slice of a ROW-MAJOR
    // (N, K) tensor is NOT contiguous: the offset is begin*elem, the row length is count*elem,
    // and the stride between rows is still the WHOLE row. Returning all three means a loader
    // cannot silently assume contiguity.
    {
        const ByteWindow w = column_byte_window(s, /*total_columns=*/2560, /*bytes_per_element=*/2);
        check(w.ok, "F2 the byte window is computed");
        check(w.offset_bytes == 1280u * 2u, "F2 the offset is begin * bytes_per_element");
        check(w.row_bytes == 640u * 2u, "F2 the row length is count * bytes_per_element");
        check(w.total_bytes == 2560u * 2u,
              "F2 and total_bytes is the FULL row stride, not the slice -- a column slice is not "
              "contiguous and the third number is how a loader knows");
        check(w.total_bytes != w.row_bytes,
              "F2 red control: the slice is narrower than the row, so the two differ");
    }
    // F3: the contiguity question answered as a bool rather than as a comment.
    {
        check(!column_slice_is_file_contiguous(s, 2560),
              "F3 a column slice of a wider tensor is not file-contiguous");
        ColumnSliceRequest whole = base_request(1, 0);
        whole.world.axis = ParallelAxis::None;
        check(column_slice_is_file_contiguous(slice_weight_columns(whole), 2560),
              "F3 the whole-N slice IS contiguous");
        ColumnSlice refused;
        check(!column_slice_is_file_contiguous(refused, 2560),
              "F3 red control: a refused slice is not reported as contiguous");
    }
    // F4: the refusals of the window itself.
    {
        ColumnSlice bad;
        check(!column_byte_window(bad, 2560, 2).ok,
              "F4 a refused slice has no byte window");
        check(!column_byte_window(s, 0, 2).ok, "F4 a zero total_columns is refused");
        check(!column_byte_window(s, 2560, 0).ok, "F4 a zero element size is refused");
        check(!column_byte_window(s, 1000, 2).ok,
              "F4 a slice past the tensor's columns is refused");
    }
}

// ---------------------------------------------------------------------------
// Group G: coexistence -- the standing frame's question
// ---------------------------------------------------------------------------
void group_G_coexistence() {
    // G1: the KV rotation and the rank axis COMMUTE, and this is the checkable form of why.
    // kvarn_hadamard wants contiguous BF16 D256 per head and rotates 8-lane blocks INTRA-head, so
    // a HEAD split carries whole heads and therefore whole rotation domains.
    {
        WorldShape tp2{2, 0, ParallelAxis::Tensor};
        check(rotation_split_refusal(tp2, 256, 8).empty(),
              "G1 a head split commutes with the intra-head 8-lane rotation");
        WorldShape pp2{2, 0, ParallelAxis::Pipeline};
        check(rotation_split_refusal(pp2, 256, 8).empty(),
              "G1 and so does the layer split, which does not touch the KV's shape at all");
        WorldShape single{1, 0, ParallelAxis::None};
        check(rotation_split_refusal(single, 256, 8).empty(),
              "G1 and the single device, trivially");
        // The refusal, for the head dim that cannot carry the block at all -- so the check is a
        // comparison and not a constant.
        WorldShape tp2b{2, 0, ParallelAxis::Tensor};
        check(!rotation_split_refusal(tp2b, 100, 8).empty(),
              "G1 a head dim that is not a multiple of the lane block is refused");
        check(rotation_split_refusal(tp2b, 0, 8).find("non-zero") != std::string::npos,
              "G1 and a zero dimension is refused rather than divided by");
    }

    // G2: the fused-boundary constraint and the rotation constraint are INDEPENDENT. A slice can
    // satisfy one and violate the other, which is why they are two functions and not one.
    {
        ColumnSliceRequest r = base_request(4, 1);
        const ColumnSlice s = slice_weight_columns(r);
        check(!s.ok, "G2 this slice violates the fused boundary");
        WorldShape tp4{4, 1, ParallelAxis::Tensor};
        check(rotation_split_refusal(tp4, 256, 8).empty(),
              "G2 ... and satisfies the rotation constraint");
    }
}

} // namespace

int main() {
    group_E_boundaries();
    group_F_byte_window();
    group_G_coexistence();

    std::printf("n_dim_slice: %d checks, %d failures -> %s\n", g_checks, g_failures,
                g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
