#include "ninfer/ops/kv_cache_append.h"

#include "ops/kv_cache/append/launch.h"
#include "ops/kv_cache/d256_profile.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim       = 128;
constexpr std::int32_t kKVHeads       = 8;
constexpr std::uint32_t kWindow       = 4096;
constexpr std::int32_t kFullHeadDim   = 256;
constexpr const char* kAppendOp       = "kv_cache_append";
constexpr const char* kPrefixAppendOp = "kv_cache_append_prefix";

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must be contiguous and non-null");
    }
}

// ⚠ A PACKED e8 CODE PLANE IS BYTE-ADDRESSED, AND `is_contiguous()` CANNOT BE USED ON IT.
//
// `Tensor::is_contiguous()` calls `dtype_size()` on its FIRST line, and `dtype_size` refuses
// the packed e8 codes BY DESIGN -- `core/dtype.cpp` names nvfp4/iso3/e8kv and says a packed
// plane "is sized by the per-layer KV storage table, not by rows*columns*element_size". So
// handing an e8 plane to `is_contiguous()` does not return false and does not return true: it
// THROWS, with a message about element sizes. Measured: `DType code=10`, backtrace through
// `Tensor::is_contiguous()`, i.e. a correctly laid out pool refused for the WRONG REASON --
// which is exactly the defect class `d256_profile.h` opens by naming.
//
// This checks the property the arm's addressing actually assumes, in the plane's own unit:
// one BYTE per code, and each axis the contiguous product of the one before it. That is the
// layout `paged_kv_element_offset<RowCodeBytes, KVHeads>` walks, and it is what
// `set_contiguous_strides` would have written had `dtype_size` returned 1 for a code plane.
// It is NOT a relaxation: a strided, overlapping or zero-stride plane is refused here.
//
// The same message as `require_contiguous_nonnull` on purpose -- one wall, one spelling, so
// that the two routes cannot be told apart by their wording, only by which planes they accept.
void require_packed_plane(const Tensor& tensor, const char* name) {
    std::int64_t want = 1;
    bool ok           = (tensor.data != nullptr);
    for (int i = 0; ok && i < 4; ++i) {
        if (tensor.ne[i] <= 0 || tensor.nb[i] != want) { ok = false; break; }
        want *= tensor.ne[i];
    }
    if (!ok) {
        throw std::invalid_argument(std::string(kAppendOp) + ": " + name +
                                    " must be contiguous and non-null");
    }
}

// The e8 lattice family's caller-supplied table handle. The Op owns no allocation, so it
// borrows the codec's 6 428 B rather than building them; a null pair would fault on device,
// so it is refused here, on the host, naming the requirement.
void require_e8_tables(E8KvAppendTables tables) {
    if (tables.stage1 == nullptr || tables.stage2 == nullptr) {
        throw std::invalid_argument(
            "kv_cache_append: the e8 family needs the codec's 6 428 B of stage-1/stage-2 "
            "tables and this call supplied a null handle; the handle is a device address and "
            "a null one would fault on device rather than refuse here");
    }
}

// THE INPUT SHAPE, ONE SPELLING for both entry points below. The messages and their order
// are the ones this op has always raised.
struct AppendInputShape {
    std::int32_t kv_heads;
    std::int32_t tokens;
};

[[nodiscard]] AppendInputShape validate_append_inputs(const Tensor& k, const Tensor& v,
                                                      const Tensor& positions) {
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("kv_cache_append: k/v must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("kv_cache_append: positions must be I32");
    }
    const std::int32_t kv_heads = k.ne[1];
    if (kv_heads != 4 && kv_heads != 2) {
        throw std::invalid_argument("kv_cache_append: unsupported KV head geometry");
    }
    const std::int32_t tokens = k.ne[2];
    if (tokens <= 0) { throw std::invalid_argument("kv_cache_append: T must be positive"); }
    require_shape(k, kFullHeadDim, kv_heads, tokens, 1, kAppendOp, "k");
    require_shape(v, kFullHeadDim, kv_heads, tokens, 1, kAppendOp, "v");
    require_shape(positions, tokens, 1, 1, 1, kAppendOp, "positions");
    require_contiguous_nonnull(k, kAppendOp, "k");
    require_contiguous_nonnull(v, kAppendOp, "v");
    require_contiguous_nonnull(positions, kAppendOp, "positions");
    return {kv_heads, tokens};
}

void require_within_capacity(std::int32_t tokens, std::uint32_t capacity) {
    if (static_cast<std::uint32_t>(tokens) > capacity) {
        throw std::invalid_argument("kv_cache_append: T exceeds cache capacity");
    }
}

// `e8_route` IS THE WHOLE DIFFERENCE BETWEEN THE TWO ENTRY POINTS, AND IT IS NOT A WIDENING.
//
// The op writes ONE code per element on the {fp8, i8, bf16} arm, which has no e8 arm at all
// (`launch.cu`'s `launch_full`: fp8, then i8, else bf16). Handing that arm an e8 pool would
// write 256 bf16 elements over a 128/96/64-byte row stride. So the e8 family is refused by
// name unless the caller arrived through the overload that ALSO carries the tables the
// narrow arm needs -- the only route on which an e8 pool is written at its own extent
// instead of the unpacked tiers' 256.
//
// WHAT IS STILL REFUSED, IN BOTH ROUTES:
//   * an e8 dtype on the tables-less route: the same named refusal as before this change;
//   * a dtype the profile table does not know: still "unsupported D256 KV-cache dtype";
//   * an e8 pool whose K plate is not ITS OWN width's extent (a W3 pool at 128 or 64, a W2
//     pool at 96 or 128, any e8 pool at the unpacked 256);
//   * an e8 pool whose V plate is not the family's W4 (128) extent -- this family narrows K
//     only, so a W3 pool with a 96-byte V plate describes a plane that does not exist;
//   * an e8 pool whose scale planes are not the g64 profile's 4-per-row;
//   * every geometry, dtype, contiguity, capacity and block-table check below, unchanged.
std::uint32_t validate_full_cache(const PagedKVLayerView& cache, std::int32_t kv_heads,
                                  bool e8_route) {
    D256KVCacheProfile profile{};
    try {
        profile = d256_kv_cache_profile(cache.dtype);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("kv_cache_append: invalid cache geometry or dtype");
    }
    if (cache.num_kv_heads != kv_heads || cache.head_dim != kFullHeadDim ||
        cache.quant_group != profile.quant_group) {
        throw std::invalid_argument("kv_cache_append: invalid cache geometry or dtype");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_table.ne[0];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("kv_cache_append: invalid cache capacity");
    }

    if (cache.k_pages.dtype != profile.code_dtype || cache.v_pages.dtype != profile.code_dtype) {
        throw std::invalid_argument("kv_cache_append: invalid cache code dtype");
    }
    // ⚠ THE e8 FAMILY IS REFUSED BY NAME UNLESS THIS ROUTE CARRIES THE TABLES.
    // The reason is unchanged and is the arm's: the wired arm of `kv_cache_append_launch` is
    // {fp8, i8, bf16} and has no e8 arm, so it cannot fill a packed K plane at all. Handing
    // it an e8 pool would not "admit the tier" -- it would write 256 bf16 elements over a
    // 128/96/64-byte row stride. What was wrong was only the REASON given to a caller: a
    // pool laid out at its own correct e8 extent was rejected as "invalid shape for cache k
    // pages", which sends the next reader to fix geometry that is already right. The refusal
    // now fires ONLY where it is true (this tables-less route) and names where to go instead.
    if (!e8_route && d256_kv_cache_is_e8_family(cache.dtype)) {
        throw std::invalid_argument(
            "kv_cache_append: the e8 family (E8Kv/E8K3Kv/E8K2Kv) has no append arm in this "
            "op -- it stores one code per element and cannot fill a packed K plane; the e8 "
            "tiers append through the gqa e8 append launch");
    }
    // ⚠ THE EXTENTS COME FROM THE PROFILE, NOT FROM `kFullHeadDim`. For BF16/I8/FP8 the
    // profile's extent IS 256, so this comparison is the same one it always was and nothing
    // moves for them. For the e8 family it is 128/96/64 -- and the V plate's extent is a
    // SEPARATE field because this family narrows K only, so a W3 pool is 96 B/row on K
    // beside 128 B/row on V. Checking both planes at one extent would reject every correct
    // narrow pool; that is the defect this field exists to prevent.
    require_shape(cache.k_pages, profile.code_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, kAppendOp, "cache k pages");
    require_shape(cache.v_pages, profile.v_code_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, kAppendOp, "cache v pages");
    if (e8_route) {
        // `is_contiguous()` would throw on these two (see `require_packed_plane`), so the
        // e8 code planes are checked on their BYTE strides instead. The scale planes below
        // are FP16 and keep the ordinary check.
        require_packed_plane(cache.k_pages, "cache k pages");
        require_packed_plane(cache.v_pages, "cache v pages");
    } else {
        require_contiguous_nonnull(cache.k_pages, kAppendOp, "cache k pages");
        require_contiguous_nonnull(cache.v_pages, kAppendOp, "cache v pages");
    }
    if (cache.block_table.dtype != DType::I32) {
        throw std::invalid_argument("kv_cache_append: block table must be I32");
    }
    require_shape(cache.block_table, logical_pages, 1, 1, 1, kAppendOp, "block table");
    require_contiguous_nonnull(cache.block_table, kAppendOp, "block table");

    if (cache.dtype == DType::BF16) {
        if (cache.k_scale_pages.data != nullptr || cache.v_scale_pages.data != nullptr) {
            throw std::invalid_argument("kv_cache_append: BF16 cache must not have scales");
        }
        return static_cast<std::uint32_t>(capacity);
    }

    if (cache.k_scale_pages.dtype != DType::FP16 || cache.v_scale_pages.dtype != DType::FP16) {
        throw std::invalid_argument("kv_cache_append: invalid cache scale dtype");
    }
    require_shape(cache.k_scale_pages, profile.scale_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, kAppendOp, "cache k scale pages");
    require_shape(cache.v_scale_pages, profile.scale_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, kAppendOp, "cache v scale pages");
    require_contiguous_nonnull(cache.k_scale_pages, kAppendOp, "cache k scale pages");
    require_contiguous_nonnull(cache.v_scale_pages, kAppendOp, "cache v scale pages");
    return static_cast<std::uint32_t>(capacity);
}

void require_vector_aligned(const Tensor& tensor, const char* name) {
    require_contiguous_nonnull(tensor, kPrefixAppendOp, name);
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & 15u) != 0u) {
        throw std::invalid_argument("kv_cache_append_prefix: " + std::string(name) +
                                    " must be 16-byte aligned");
    }
}

detail::KVCacheAppendPrefixPlan validate_inputs(const Tensor& k, const Tensor& v,
                                                const Tensor& positions, const Tensor& counts,
                                                const Tensor& selectors,
                                                KVCacheAppendPrefixExecutionEnvelope envelope) {
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("kv_cache_append_prefix: k/v must be BF16");
    }
    if (positions.dtype != DType::I32 || counts.dtype != DType::I32 ||
        selectors.dtype != DType::I32) {
        throw std::invalid_argument(
            "kv_cache_append_prefix: positions/counts/selectors must be I32");
    }
    const std::int32_t tokens = k.ne[2];
    const std::int32_t batch  = k.ne[3];
    if (tokens < 1) { throw std::invalid_argument("kv_cache_append_prefix: T must be positive"); }
    if (batch < 1 || batch > 8) {
        throw std::invalid_argument("kv_cache_append_prefix: B must be 1..8");
    }
    require_shape(k, kHeadDim, kKVHeads, tokens, batch, kPrefixAppendOp, "k");
    require_shape(v, kHeadDim, kKVHeads, tokens, batch, kPrefixAppendOp, "v");
    require_shape(positions, tokens, batch, 1, 1, kPrefixAppendOp, "positions");
    require_shape(counts, batch, 1, 1, 1, kPrefixAppendOp, "counts");
    require_shape(selectors, batch, 1, 1, 1, kPrefixAppendOp, "selectors");
    require_vector_aligned(k, "k");
    require_vector_aligned(v, "v");
    require_contiguous_nonnull(positions, kPrefixAppendOp, "positions");
    require_contiguous_nonnull(counts, kPrefixAppendOp, "counts");
    require_contiguous_nonnull(selectors, kPrefixAppendOp, "selectors");
    return detail::kv_cache_append_prefix_resolve_plan(tokens, envelope);
}

void validate_paged_cache(const PagedKVBatchLayerView& cache,
                          KVCacheAppendPrefixExecutionEnvelope envelope) {
    if (cache.dtype != DType::BF16 || cache.quant_group != 0 || cache.num_kv_heads != kKVHeads ||
        cache.head_dim != kHeadDim || cache.k_pages.ne[2] <= 0 ||
        cache.k_pages.ne[2] != cache.v_pages.ne[2] || cache.block_tables.ne[0] <= 0 ||
        envelope.max_count >
            static_cast<std::uint32_t>(cache.block_tables.ne[0]) * kPagedKVPageSize) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid paged cache");
    }
    const std::int32_t physical_pages = cache.k_pages.ne[2];
    if (cache.k_pages.dtype != DType::BF16 || cache.v_pages.dtype != DType::BF16 ||
        cache.k_pages.ne[0] != kHeadDim || cache.k_pages.ne[1] != kPagedKVPageSize ||
        cache.k_pages.ne[3] != kKVHeads || cache.v_pages.ne[0] != kHeadDim ||
        cache.v_pages.ne[1] != kPagedKVPageSize || cache.v_pages.ne[2] != physical_pages ||
        cache.v_pages.ne[3] != kKVHeads || cache.k_scale_pages.data != nullptr ||
        cache.v_scale_pages.data != nullptr || cache.block_tables.dtype != DType::I32 ||
        cache.block_tables.ne[1] <= 0 || cache.block_tables.ne[2] != 1 ||
        cache.block_tables.ne[3] != 1) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid paged cache tensors");
    }
    require_vector_aligned(cache.k_pages, "cache k pages");
    require_vector_aligned(cache.v_pages, "cache v pages");
    require_contiguous_nonnull(cache.block_tables, kPrefixAppendOp, "cache block tables");
}

void validate_cyclic_cache(const CyclicKVCacheLayerView& cache,
                           KVCacheAppendPrefixExecutionEnvelope envelope,
                           std::uint32_t window) {
    if (cache.num_kv_heads != kKVHeads || cache.head_dim != kHeadDim ||
        cache.capacity != window ||
        (window != 2048 && window != 4096) ||
        cache.padded_capacity < cache.capacity ||
        cache.padded_capacity >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        envelope.max_count > cache.capacity) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid cyclic cache");
    }
    const auto padded = static_cast<std::int32_t>(cache.padded_capacity);
    if (cache.k.dtype != DType::BF16 || cache.v.dtype != DType::BF16 || cache.k.ne[0] != kHeadDim ||
        cache.k.ne[1] != padded || cache.k.ne[2] != kKVHeads || cache.v.ne[0] != kHeadDim ||
        cache.v.ne[1] != padded || cache.v.ne[2] != kKVHeads ||
        cache.v.ne[3] != cache.lane_capacity || cache.lane_capacity <= 0 ||
        cache.k.ne[3] != cache.lane_capacity) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid cyclic cache tensors");
    }
    require_vector_aligned(cache.k, "cache k");
    require_vector_aligned(cache.v, "cache v");
}

} // namespace

void kv_cache_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                     PagedKVLayerView cache, cudaStream_t stream) {
    const auto shape = validate_append_inputs(k, v, positions);
    const std::uint32_t capacity = validate_full_cache(cache, shape.kv_heads, /*e8_route=*/false);
    require_within_capacity(shape.tokens, capacity);
    detail::kv_cache_append_launch(k, v, positions, cache, stream);
}

void kv_cache_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                     PagedKVLayerView cache, E8KvAppendTables tables, cudaStream_t stream) {
    // A non-e8 dtype is NOT this overload's business: it keeps the exact path and the exact
    // refusals it had, so no admission is widened by arriving here. This is a forward, not a
    // second spelling of the append.
    if (!d256_kv_cache_is_e8_family(cache.dtype)) {
        kv_cache_append(k, v, positions, cache, stream);
        return;
    }
    const auto shape = validate_append_inputs(k, v, positions);
    require_e8_tables(tables);
    const std::uint32_t capacity = validate_full_cache(cache, shape.kv_heads, /*e8_route=*/true);
    require_within_capacity(shape.tokens, capacity);
    detail::kv_cache_append_e8_lattice_tables_launch(k, v, positions, cache, tables, stream);
}

void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& table_rows,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            PagedKVBatchLayerView cache, cudaStream_t stream) {
    const auto plan = validate_inputs(k, v, positions, counts, table_rows, envelope);
    validate_paged_cache(cache, envelope);
    detail::kv_cache_append_prefix_launch(k, v, positions, counts, table_rows, cache, plan, stream);
}

void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& lanes,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            CyclicKVCacheLayerView cache, std::uint32_t window,
                            cudaStream_t stream) {
    const auto plan = validate_inputs(k, v, positions, counts, lanes, envelope);
    validate_cyclic_cache(cache, envelope, window);
    detail::kv_cache_append_prefix_launch(k, v, positions, counts, lanes, cache, plan, window,
                                          stream);
}

} // namespace ninfer::ops
