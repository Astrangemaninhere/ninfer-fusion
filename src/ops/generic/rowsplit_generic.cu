#include "ops/generic/rowsplit_generic.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ninfer/ops/rmsnorm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kGroupK      = 64;
constexpr int kLowBytes    = 32;
constexpr int kScaleBytes  = 2;
constexpr int kThreads     = 256;
constexpr int kColumns     = 4;
constexpr int kMaxWarps    = kThreads / 32;

int high_bytes_per_group(QType qtype) noexcept {
    switch (qtype) {
    case QType::Q4G64_F16S:
        return 0;
    case QType::Q5G64_F16S:
        return 8;
    case QType::Q6G64_F16S:
        return 16;
    default:
        return -1;
    }
}

// The destination's token-block stride, in elements.  A dim-0 row view of a wider parent keeps
// the parent's nb[1] (Tensor::slice does not rewrite strides), which is the same leading
// dimension the registered SIMT launcher reads off its own destination
// (q4_rowsplit_gemm_simt.cu: `out.nb[1] / sizeof(__nv_bfloat16)`).
int destination_leading_rows(const Tensor& out) noexcept {
    return static_cast<int>(out.nb[1] / static_cast<std::int64_t>(sizeof(bf16)));
}

// The destination is usable when dimension zero -- the row dimension, which is the innermost
// one in this engine -- is contiguous and the token stride is a whole number of elements.
// Full contiguity is NOT required and must not be: gdn_input_proj hands this decoder a row
// view of the packed qkv block.
bool destination_layout_ok(const Tensor& out) noexcept {
    return out.nb[0] == static_cast<std::int64_t>(sizeof(bf16)) && out.nb[1] > 0 &&
           (out.nb[1] % static_cast<std::int64_t>(sizeof(bf16))) == 0;
}

bool rowsplit_layout_ok(const Weight& w) noexcept {
    const int high_bytes = high_bytes_per_group(w.qtype);
    if (high_bytes < 0) { return false; }
    if (w.layout != QuantLayout::RowSplit || w.scale_dtype != DType::FP16) { return false; }
    if (w.group_size != kGroupK || w.group != kGroupK) { return false; }
    if (w.ndim != 2 || w.n <= 0 || w.k <= 0) { return false; }
    if (w.shape[0] != w.n || w.shape[1] != w.k) { return false; }
    if (w.padded_shape[0] != w.n || w.padded_shape[1] != w.k) { return false; }
    if ((w.k % kGroupK) != 0) { return false; }
    if (w.qdata == nullptr || w.scales == nullptr) { return false; }
    if (high_bytes > 0 && w.qhigh == nullptr) { return false; }
    if (high_bytes == 0 && w.qhigh != nullptr) { return false; }
    return true;
}

// ---------------------------------------------------------------------------
// Weight decode.  Element j (0..63) of group g of a row:
//   low nibble  : codes[g*32 + (j >> 1)] nibble (j & 1)
//   Q4          : q = (nibble ^ 8) - 8
//   Q5          : q = ((nibble | high_bit(g, j) << 4) ^ 0x10) - 0x10
//   Q6          : q = ((nibble | high_pair(g, j) << 4) ^ 0x20) - 0x20
//   scale       : fp16 at scales[g*2]
// ---------------------------------------------------------------------------
template <int Bits>
__device__ __forceinline__ float decode_weight(const std::uint8_t* __restrict__ codes,
                                               const std::uint8_t* __restrict__ high,
                                               const std::uint8_t* __restrict__ scales,
                                               int group, int j) {
    const std::uint8_t packed = codes[group * kLowBytes + (j >> 1)];
    const int low             = (j & 1) ? (packed >> 4) : (packed & 0x0fu);
    int q                     = low;
    if constexpr (Bits == 5) {
        const int bit = (high[group * 8 + (j >> 3)] >> (j & 7)) & 1;
        q             = low | (bit << 4);
        q             = (q ^ 0x10) - 0x10;
    } else if constexpr (Bits == 6) {
        const int bits = (high[group * 16 + (j >> 2)] >> ((j & 3) * 2)) & 3;
        q              = low | (bits << 4);
        q              = (q ^ 0x20) - 0x20;
    } else {
        q = (low ^ 0x08) - 0x08;
    }
    const float scale = __half2float(__ushort_as_half(
        *reinterpret_cast<const std::uint16_t*>(scales + group * kScaleBytes)));
    return static_cast<float>(q) * scale;
}

// One block per (output row, group of `kColumns` columns).  Threads stride the reduction dim; each
// thread keeps one accumulator per column and the block tree-reduces them at the end.
template <int Bits, bool Residual>
__global__ void __launch_bounds__(kThreads) generic_rowsplit_gemm_kernel(
    const bf16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    bf16* __restrict__ out, int k, int t, int out_l2) {
    const int row   = static_cast<int>(blockIdx.x);
    const int col0  = static_cast<int>(blockIdx.y) * kColumns;
    const int ncols = min(kColumns, t - col0);
    if (ncols <= 0) { return; }

    const std::int64_t groups = static_cast<std::int64_t>(k) / kGroupK;
    const std::uint8_t* row_codes = codes + static_cast<std::int64_t>(row) * groups * kLowBytes;
    const std::uint8_t* row_high =
        high == nullptr ? nullptr
                        : high + static_cast<std::int64_t>(row) * groups * (Bits == 5 ? 8 : 16);
    const std::uint8_t* row_scales =
        scales + static_cast<std::int64_t>(row) * groups * kScaleBytes;

    float acc[kColumns];
    #pragma unroll
    for (int c = 0; c < kColumns; ++c) { acc[c] = 0.0f; }

    for (int j = static_cast<int>(threadIdx.x); j < k; j += kThreads) {
        const float w = decode_weight<Bits>(row_codes, row_high, row_scales, j >> 6, j & 63);
        // x is {K, T} in THIS engine's convention -- dimension zero innermost -- so the k
        // index is the fast one and the token index strides by k.  The registered SIMT kernel
        // reads exactly this (q4_rowsplit_gemm_simt.cuh: `x + (col0 + col) * k + xk`); the
        // previous `x[j * t + col0]` was the transpose and silently evaluated a different
        // product.
        const bf16* xrow = x + static_cast<std::int64_t>(col0) * k + j;
        #pragma unroll
        for (int c = 0; c < kColumns; ++c) {
            if (c < ncols) { acc[c] = fmaf(w, __bfloat162float(xrow[c * k]), acc[c]); }
        }
    }

    __shared__ float partial[kColumns][kMaxWarps];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    #pragma unroll
    for (int c = 0; c < kColumns; ++c) {
        float value = acc[c];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) { partial[c][warp] = value; }
    }
    __syncthreads();

    if (warp != 0) { return; }
    const int warps = kThreads / 32;
    #pragma unroll
    for (int c = 0; c < kColumns; ++c) {
        if (c >= ncols) { continue; }
        float value = (lane < warps) ? partial[c][lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane != 0) { continue; }
        // out is {N, T} with dimension zero innermost: the token index strides by the
        // destination's leading row count, which for a view of a wider parent is the parent's.
        const std::int64_t index = static_cast<std::int64_t>(col0 + c) * out_l2 + row;
        if (Residual) { value += __bfloat162float(out[index]); }
        out[index] = __float2bfloat16_rn(value);
    }
}

__global__ void __launch_bounds__(kThreads) generic_bf16_gemm_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ w, bf16* __restrict__ out, int k, int t,
    int out_l2) {
    const int row   = static_cast<int>(blockIdx.x);
    const int col0  = static_cast<int>(blockIdx.y) * kColumns;
    const int ncols = min(kColumns, t - col0);
    if (ncols <= 0) { return; }

    const bf16* wrow = w + static_cast<std::int64_t>(row) * k;
    float acc[kColumns];
    #pragma unroll
    for (int c = 0; c < kColumns; ++c) { acc[c] = 0.0f; }

    for (int j = static_cast<int>(threadIdx.x); j < k; j += kThreads) {
        const float wv   = __bfloat162float(wrow[j]);
        // Same convention as generic_rowsplit_gemm_kernel above: x is {K, T}, k innermost.
        const bf16* xrow = x + static_cast<std::int64_t>(col0) * k + j;
        #pragma unroll
        for (int c = 0; c < kColumns; ++c) {
            if (c < ncols) { acc[c] = fmaf(wv, __bfloat162float(xrow[c * k]), acc[c]); }
        }
    }

    __shared__ float partial[kColumns][kMaxWarps];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    #pragma unroll
    for (int c = 0; c < kColumns; ++c) {
        float value = acc[c];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) { partial[c][warp] = value; }
    }
    __syncthreads();

    if (warp != 0) { return; }
    const int warps = kThreads / 32;
    #pragma unroll
    for (int c = 0; c < kColumns; ++c) {
        if (c >= ncols) { continue; }
        float value = (lane < warps) ? partial[c][lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane != 0) { continue; }
        out[static_cast<std::int64_t>(col0 + c) * out_l2 + row] = __float2bfloat16_rn(value);
    }
}

// The GDN gating projection: logical rows [0, heads) are the `a` projection and [heads, 2*heads)
// are `b`.  Row `r` of `a` goes to g[r,t], row `r` of `b` to beta[r,t]; both planes are stored
// [t, heads] (dimension zero fastest).
__global__ void __launch_bounds__(kThreads) generic_gating_kernel(
    const bf16* __restrict__ h, const bf16* __restrict__ a_weight,
    const bf16* __restrict__ b_weight, const float* __restrict__ A_log,
    const float* __restrict__ dt_bias, float* __restrict__ g, float* __restrict__ beta, int heads,
    int k, int t) {
    const int logical = static_cast<int>(blockIdx.x);
    const int col0    = static_cast<int>(blockIdx.y) * kColumns;
    const int ncols   = min(kColumns, t - col0);
    if (ncols <= 0) { return; }

    const bool is_b  = logical >= heads;
    const int row    = is_b ? logical - heads : logical;
    const bf16* wrow = (is_b ? b_weight : a_weight) + static_cast<std::int64_t>(row) * k;

    float acc[kColumns];
    #pragma unroll
    for (int c = 0; c < kColumns; ++c) { acc[c] = 0.0f; }

    for (int j = static_cast<int>(threadIdx.x); j < k; j += kThreads) {
        const float wv   = __bfloat162float(wrow[j]);
        // h is the rmsnorm output: {K, T} in this engine's convention, so the k index is the
        // fast one -- exactly as the store below writes (col0 + c) * heads + row.  The
        // previous `h[j * t + col0]` read a different element than the one it stored, so the
        // two halves of this kernel disagreed with each other.
        const bf16* hrow = h + static_cast<std::int64_t>(col0) * k + j;
        #pragma unroll
        for (int c = 0; c < kColumns; ++c) {
            if (c < ncols) { acc[c] = fmaf(wv, __bfloat162float(hrow[c * k]), acc[c]); }
        }
    }

    __shared__ float partial[kColumns][kMaxWarps];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    #pragma unroll
    for (int c = 0; c < kColumns; ++c) {
        float value = acc[c];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) { partial[c][warp] = value; }
    }
    __syncthreads();

    if (warp != 0) { return; }
    const int warps = kThreads / 32;
    #pragma unroll
    for (int c = 0; c < kColumns; ++c) {
        if (c >= ncols) { continue; }
        float value = (lane < warps) ? partial[c][lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane != 0) { continue; }
        const std::int64_t index = static_cast<std::int64_t>(col0 + c) * heads + row;
        if (is_b) {
            beta[index] = sigmoid(value);
        } else {
            g[index] = -expf(A_log[row]) * softplus(value + dt_bias[row]);
        }
    }
}

// out[i,t] = silu(gate_up[i,t]) * gate_up[i + rows, t]
__global__ void generic_swiglu_split_kernel(const bf16* __restrict__ gate_up,
                                            bf16* __restrict__ out, std::int64_t total_rows,
                                            std::int64_t columns, std::int64_t rows) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total_rows * columns) { return; }
    const std::int64_t column = index % columns;
    const std::int64_t row    = index / columns;
    // Both planes are {Rows, T} with dimension zero innermost, so token `column` owns a block
    // of `2 * rows` (gate_up) or `rows` (out) consecutive elements starting at that column.
    const std::int64_t gate_up_l2 = 2 * rows;
    const float gate = __bfloat162float(gate_up[column * gate_up_l2 + row]);
    const float up   = __bfloat162float(gate_up[column * gate_up_l2 + row + rows]);
    out[column * rows + row] = __float2bfloat16_rn(silu(gate) * up);
}

dim3 gemm_grid(std::int32_t rows, std::int32_t t) {
    return dim3(static_cast<unsigned>(rows), static_cast<unsigned>((t + kColumns - 1) / kColumns));
}

void require_common(const Tensor& x, const Weight& w, const Tensor& out) {
    if (!generic_rowsplit_weight_ok(w)) {
        throw std::invalid_argument("generic linear: weight is not a decodable row-split payload");
    }
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("generic linear: x/out must be BF16");
    }
    if (x.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("generic linear: x/out data must be non-null");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || x.ne[1] != out.ne[1]) {
        throw std::invalid_argument("generic linear: expected [K,T] x [N,K] -> [N,T]");
    }
    if (x.ne[1] <= 0 || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("generic linear: unsupported rank or token extent");
    }
    if (!x.is_contiguous()) {
        throw std::invalid_argument("generic linear: x must be contiguous");
    }
    if (!destination_layout_ok(out)) {
        throw std::invalid_argument("generic linear: out must have a contiguous row dimension");
    }
}

template <int Bits, bool Residual>
void launch_rowsplit(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const int k = w.k;
    const int t = x.ne[1];
    generic_rowsplit_gemm_kernel<Bits, Residual><<<gemm_grid(w.n, t), kThreads, 0, stream>>>(
        static_cast<const bf16*>(x.data),
        static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh),
        static_cast<const std::uint8_t*>(w.scales),
        static_cast<bf16*>(out.data), k, t, destination_leading_rows(out));
    CUDA_CHECK(cudaGetLastError());
}

template <bool Residual>
void launch_rowsplit_by_format(const Tensor& x, const Weight& w, Tensor& out,
                               cudaStream_t stream) {
    switch (w.qtype) {
    case QType::Q4G64_F16S:
        launch_rowsplit<4, Residual>(x, w, out, stream);
        return;
    case QType::Q5G64_F16S:
        launch_rowsplit<5, Residual>(x, w, out, stream);
        return;
    case QType::Q6G64_F16S:
        launch_rowsplit<6, Residual>(x, w, out, stream);
        return;
    default:
        break;
    }
    throw std::invalid_argument("generic linear: unsupported row-split format");
}

} // namespace

bool generic_fallback_enabled() noexcept {
    // Read once: the value is a process-level kill switch, and a test that flips it mid-run would
    // be testing the cache, not the fallback.
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_GENERIC_ROWDEC");
        if (value == nullptr) { return true; }
        return !(value[0] == '0' && value[1] == '\0');
    }();
    return enabled;
}

bool generic_rowsplit_shape_capable(std::int32_t output_rows, std::int32_t input_rows) noexcept {
    return generic_fallback_enabled() && output_rows > 0 && input_rows > 0 &&
           (input_rows % kGroupK) == 0;
}

bool generic_rowsplit_weight_ok(const Weight& w) noexcept { return rowsplit_layout_ok(w); }

bool generic_rowsplit_problem_ok(const Tensor& x, const Weight& w, const Tensor& out) noexcept {
    if (!generic_fallback_enabled()) { return false; }
    if (!rowsplit_layout_ok(w)) { return false; }
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) { return false; }
    if (x.data == nullptr || out.data == nullptr) { return false; }
    if (x.ne[0] != w.k || out.ne[0] != w.n || x.ne[1] != out.ne[1]) { return false; }
    if (x.ne[1] <= 0 || x.ne[2] != 1 || x.ne[3] != 1) { return false; }
    if (out.ne[2] != 1 || out.ne[3] != 1) { return false; }
    // Mirrors require_common exactly: the two must agree on what dispatch accepts, or a caller
    // guarded by this would refuse a problem the decoder can serve.
    return x.is_contiguous() && destination_layout_ok(out);
}

void generic_rowsplit_linear_dispatch(const Tensor& x, const Weight& w, Tensor& out,
                                      cudaStream_t stream) {
    require_common(x, w, out);
    launch_rowsplit_by_format<false>(x, w, out, stream);
}

void generic_rowsplit_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual,
                                          cudaStream_t stream) {
    require_common(x, w, residual);
    launch_rowsplit_by_format<true>(x, w, residual, stream);
}

Weight generic_rowsplit_row_view(const Weight& w, std::int32_t row_begin, std::int32_t rows) {
    if (!rowsplit_layout_ok(w) || row_begin < 0 || rows <= 0 || row_begin + rows > w.n) {
        throw std::invalid_argument("generic linear: invalid row view");
    }
    const std::int64_t groups    = static_cast<std::int64_t>(w.padded_shape[1]) / kGroupK;
    const int high_bytes         = high_bytes_per_group(w.qtype);
    const std::int64_t low_row   = groups * kLowBytes;
    const std::int64_t high_row  = groups * high_bytes;
    const std::int64_t scale_row = groups * kScaleBytes;

    Weight view          = w;
    view.qdata           = static_cast<const std::uint8_t*>(w.qdata) +
                 static_cast<std::int64_t>(row_begin) * low_row;
    view.qhigh = high_bytes == 0
                     ? nullptr
                     : static_cast<const std::uint8_t*>(w.qhigh) +
                           static_cast<std::int64_t>(row_begin) * high_row;
    view.scales = static_cast<const std::uint8_t*>(w.scales) +
                  static_cast<std::int64_t>(row_begin) * scale_row;
    view.n               = rows;
    view.shape[0]        = rows;
    view.padded_shape[0] = rows;
    return view;
}

bool generic_bf16_weight_ok(const Weight& w) noexcept {
    if (w.qtype != QType::BF16_CTRL || w.layout != QuantLayout::Contiguous) { return false; }
    if (w.n <= 0 || w.k <= 0 || w.ndim != 2) { return false; }
    if (w.shape[0] != w.n || w.shape[1] != w.k) { return false; }
    if (w.padded_shape[0] != w.n || w.padded_shape[1] != w.k) { return false; }
    if (w.qhigh != nullptr || w.scales != nullptr) { return false; }
    if (w.qdata == nullptr) { return false; }
    const std::uint64_t needed = static_cast<std::uint64_t>(w.n) *
                                 static_cast<std::uint64_t>(w.k) * sizeof(std::uint16_t);
    return w.payload_bytes >= needed;
}

void generic_bf16_linear_dispatch(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream) {
    if (!generic_bf16_weight_ok(w)) {
        throw std::invalid_argument("generic linear: weight is not a contiguous BF16 payload");
    }
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 || x.data == nullptr ||
        out.data == nullptr) {
        throw std::invalid_argument("generic linear: x/out must be non-null BF16");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || x.ne[1] != out.ne[1] || x.ne[1] <= 0 ||
        x.ne[2] != 1 || x.ne[3] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("generic linear: expected [K,T] x [N,K] -> [N,T]");
    }
    if (!x.is_contiguous()) {
        throw std::invalid_argument("generic linear: x must be contiguous");
    }
    if (!destination_layout_ok(out)) {
        throw std::invalid_argument("generic linear: out must have a contiguous row dimension");
    }
    const int k = w.k;
    const int t = x.ne[1];
    generic_bf16_gemm_kernel<<<gemm_grid(w.n, t), kThreads, 0, stream>>>(
        static_cast<const bf16*>(x.data), static_cast<const bf16*>(w.qdata),
        static_cast<bf16*>(out.data), k, t, destination_leading_rows(out));
    CUDA_CHECK(cudaGetLastError());
}

bool generic_norm_gating_problem_ok(const Tensor& x, const Tensor& norm_weight,
                                    const Weight& a_weight, const Weight& b_weight,
                                    const Tensor& A_log, const Tensor& dt_bias, const Tensor& h,
                                    const Tensor& g, const Tensor& beta) noexcept {
    if (!generic_fallback_enabled()) { return false; }
    if (!generic_bf16_weight_ok(a_weight) || !generic_bf16_weight_ok(b_weight)) { return false; }
    const std::int32_t heads = a_weight.n;
    if (b_weight.n != heads || b_weight.k != a_weight.k) { return false; }
    const std::int32_t k = a_weight.k;
    if (x.dtype != DType::BF16 || h.dtype != DType::BF16) { return false; }
    if (x.data == nullptr || h.data == nullptr || g.data == nullptr || beta.data == nullptr) {
        return false;
    }
    if (x.ne[0] != k || h.ne[0] != k) { return false; }
    if (x.ne[1] <= 0 || x.ne[1] != h.ne[1] || x.ne[2] != 1 || x.ne[3] != 1) { return false; }
    if (norm_weight.dtype != DType::BF16 || norm_weight.ne[0] != k || norm_weight.ne[1] != 1) {
        return false;
    }
    if (A_log.dtype != DType::FP32 || A_log.ne[0] != heads || A_log.ne[1] != 1) { return false; }
    if (dt_bias.dtype != DType::FP32 || dt_bias.ne[0] != heads || dt_bias.ne[1] != 1) {
        return false;
    }
    if (g.dtype != DType::FP32 || beta.dtype != DType::FP32) { return false; }
    if (g.ne[0] != heads || beta.ne[0] != heads) { return false; }
    if (g.ne[1] != x.ne[1] || beta.ne[1] != x.ne[1]) { return false; }
    return x.is_contiguous() && h.is_contiguous() && g.is_contiguous() && beta.is_contiguous();
}

void generic_norm_gating_proj_dispatch(const Tensor& x, const Tensor& norm_weight, float eps,
                                       const Weight& a_weight, const Weight& b_weight,
                                       const Tensor& A_log, const Tensor& dt_bias, Tensor& h,
                                       Tensor& g, Tensor& beta, cudaStream_t stream) {
    if (!generic_norm_gating_problem_ok(x, norm_weight, a_weight, b_weight, A_log, dt_bias, h, g,
                                        beta)) {
        throw std::invalid_argument("generic gdn gating: unsupported problem");
    }
    // The registered kernel stores the normalized activation with the unit-offset gain
    // (x * inv * (1 + norm_weight)) and reduces the projections against that same normalized
    // activation, so one rmsnorm call reproduces both halves exactly.
    rmsnorm(x, norm_weight, eps, /*unit_offset=*/true, h, stream);

    const int heads = a_weight.n;
    const int k     = a_weight.k;
    const int t     = x.ne[1];
    generic_gating_kernel<<<gemm_grid(2 * heads, t), kThreads, 0, stream>>>(
        static_cast<const bf16*>(h.data), static_cast<const bf16*>(a_weight.qdata),
        static_cast<const bf16*>(b_weight.qdata), static_cast<const float*>(A_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data), heads, k, t);
    CUDA_CHECK(cudaGetLastError());
}

void generic_swiglu_split_dispatch(const Tensor& gate_up, Tensor& out, cudaStream_t stream) {
    if (gate_up.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("generic linear_swiglu: gate_up/out must be BF16");
    }
    const std::int64_t rows    = out.ne[0];
    const std::int64_t columns = out.ne[1];
    if (gate_up.ne[0] != 2 * rows || gate_up.ne[1] != columns) {
        throw std::invalid_argument("generic linear_swiglu: gate_up must be [2I,T]");
    }
    const std::int64_t total   = rows * columns;
    if (total == 0) { return; }
    const std::int64_t threads = 256;
    const std::int64_t blocks  = (total + threads - 1) / threads;
    // The kernel's guard is `index >= total_rows * columns`, so `total_rows` is the number of
    // ROWS and must be `rows`.  Passing `total` here made the guard admit rows*columns*columns
    // indices: dead at every engine MLP (there `rows` is a multiple of 256, so
    // blocks*threads == total exactly and no such index exists) and out of bounds on a
    // 30-element destination at 6x5.
    generic_swiglu_split_kernel<<<static_cast<unsigned>(blocks), static_cast<unsigned>(threads), 0,
                                  stream>>>(static_cast<const bf16*>(gate_up.data),
                                            static_cast<bf16*>(out.data), rows, columns, rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
