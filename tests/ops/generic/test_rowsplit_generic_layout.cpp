// tests/ops/generic/test_rowsplit_generic_layout.cpp
//
// Regression test for two measured defects in src/ops/generic/rowsplit_generic.cu.
//
// DEFECT 1 -- the generic row-split decoder indexed BOTH operands for a token-inner layout this
// engine does not have:
//     xrow = x + j * t + col0;                 (pre-fix)
//     out[ row * t + col0 + c ]                (pre-fix)
// whereas the REGISTERED q4 kernels are the authority for the convention:
//     q4_rowsplit_gemm_simt.cuh : "out[Rows, Cols] = W[Rows, K] * x[K, Cols]"
//                              : x   + (col0 + col) * k + xk
//                              : out[(col0 + col) * out_ld + row]
//     q4_rowsplit_gemm_simt.cu  : out_ld = out.nb[1] / sizeof(__nv_bfloat16)
// The engine's tensors are ne[0]-innermost, so for x{K,T} element (j,col) is at col*K + j and for
// out{N,T} element (row,col) is at col*out_ld + row.
//
// DEFECT 2 -- generic_swiglu_split_dispatch passed `total` (= rows*columns) into the kernel's
// `total_rows` parameter, whose guard is `index >= total_rows * columns`. The guard therefore
// admitted rows*columns*columns indices. At every engine MLP geometry `rows` is a multiple of 256
// (qwen3_5_9b intermediate = 12288 = 48 * 256) so `blocks * threads == total` exactly and the
// guard was merely DEAD; at 6x5 it admits 30..149 on a 30-element destination, i.e. it reads and
// writes out of bounds. The parameter is named `total_rows` and the guard multiplies it by
// `columns`, so the quantity it meant is the destination ROW COUNT -- which the caller already had
// in scope as `rows`. The fix passes `rows`.
//
// The reference below is an INDEPENDENT host dequantizer written from the storage spec
// (ops/linear/q4/q4_rowsplit_storage.cuh), not from the kernel, and the test carries a POSITIVE
// CONTROL: the same problem through the REGISTERED dispatcher. A FAIL therefore means something.
// It also asserts agreement between the registered and generic routes on a shape both serve, so
// the tracked sibling -- not only the host reference -- pins the convention.

#include "core/tensor.h"
#include "ops/generic/rowsplit_generic.h"
#include "ops/linear/q4/q4_dispatch.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using ninfer::DType;
using ninfer::QType;
using ninfer::QuantLayout;
using ninfer::Tensor;
using ninfer::Weight;
using ninfer::ops::LinearPolicy;

namespace {

constexpr int kGroupK   = 64;
constexpr int kLowBytes = 32;

// bf16 bit patterns used as sentinels. 0x3F80 is +1.0f; 0x7F80 is +inf.
constexpr std::uint16_t kSentinel     = 0x3F80u;
constexpr std::uint16_t kGateUpTrap   = 0x7F80u;

double bf16_to_double(std::uint16_t bits) {
    __nv_bfloat16 value;
    std::memcpy(&value, &bits, sizeof(value));
    return static_cast<double>(__bfloat162float(value));
}

std::uint16_t float_to_bf16(float value) {
    const __nv_bfloat16 converted = __float2bfloat16_rn(value);
    std::uint16_t bits            = 0;
    std::memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

float half_to_float(std::uint16_t bits) {
    __half value;
    std::memcpy(&value, &bits, sizeof(value));
    return __half2float(value);
}

// THIS engine's convention: ne[0] is innermost, so element (i, t) of a tensor sized {I, T} sits at
// t * I + i. The registered SIMT launcher spells the same thing as col * out_ld + row.
std::size_t engine_at(int i, int t, int inner_extent) {
    return static_cast<std::size_t>(t) * static_cast<std::size_t>(inner_extent) +
           static_cast<std::size_t>(i);
}

// ---------------------------------------------------------------------------------------------
// A row-split Q4G64_F16S payload (k128-v1) plus the host-side ideal product it stands for.
// ---------------------------------------------------------------------------------------------
struct Payload {
    int rows = 0;
    int k    = 0;
    int cols = 0;
    std::vector<std::uint8_t> codes;
    std::vector<std::uint16_t> scales;
    std::vector<std::uint16_t> x_bits;      // {k, cols}: element (j,col) at col*k + j
    std::vector<double> reference;          // element (row,col) at row*cols + col
};

Payload make_payload(int rows, int k, int cols, std::uint32_t seed) {
    Payload p;
    p.rows         = rows;
    p.k            = k;
    p.cols         = cols;
    const int groups = k / kGroupK;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> code_dist(0, 255);
    std::uniform_int_distribution<int> scale_dist(0x2000, 0x3BFF);  // finite positive fp16
    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);

    p.codes.assign(static_cast<std::size_t>(rows) * groups * kLowBytes, 0);
    for (std::uint8_t& byte : p.codes) { byte = static_cast<std::uint8_t>(code_dist(rng)); }
    p.scales.assign(static_cast<std::size_t>(rows) * groups, 0);
    for (std::uint16_t& bits : p.scales) { bits = static_cast<std::uint16_t>(scale_dist(rng)); }

    std::vector<double> weight(static_cast<std::size_t>(rows) * k, 0.0);
    for (int row = 0; row < rows; ++row) {
        for (int j = 0; j < k; ++j) {
            const int group  = j / kGroupK;
            const int within = j % kGroupK;
            const std::uint8_t packed =
                p.codes[(static_cast<std::size_t>(row) * groups + group) * kLowBytes + (within >> 1)];
            int q = (within & 1) ? (packed >> 4) : (packed & 0x0Fu);
            q     = (q ^ 0x08) - 0x08;
            const float scale =
                half_to_float(p.scales[static_cast<std::size_t>(row) * groups + group]);
            weight[static_cast<std::size_t>(row) * k + j] = static_cast<double>(q) * scale;
        }
    }

    p.x_bits.assign(static_cast<std::size_t>(cols) * k, 0);
    for (std::uint16_t& bits : p.x_bits) { bits = float_to_bf16(x_dist(rng)); }

    p.reference.assign(static_cast<std::size_t>(rows) * cols, 0.0);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            double acc = 0.0;
            for (int j = 0; j < k; ++j) {
                acc += weight[static_cast<std::size_t>(row) * k + j] *
                       bf16_to_double(p.x_bits[static_cast<std::size_t>(col) * k + j]);
            }
            p.reference[static_cast<std::size_t>(row) * cols + col] = acc;
        }
    }
    return p;
}

Weight make_weight(const Payload& p, const void* d_codes, const void* d_scales) {
    Weight w{};
    w.payload         = d_codes;
    w.payload_bytes   = p.codes.size();
    w.qtype           = QType::Q4G64_F16S;
    w.group_size      = kGroupK;
    w.shape[0]        = p.rows;
    w.shape[1]        = p.k;
    w.padded_shape[0] = p.rows;
    w.padded_shape[1] = p.k;
    w.ndim            = 2;
    w.qdata           = d_codes;
    w.qhigh           = nullptr;
    w.scales          = d_scales;
    w.n               = p.rows;
    w.k               = p.k;
    w.group           = kGroupK;
    w.layout          = QuantLayout::RowSplit;
    w.scale_dtype     = DType::FP16;
    return w;
}

int failures = 0;

void report(const std::string& label, bool ok, const std::string& detail) {
    std::printf("  %-62s %s  %s\n", label.c_str(), ok ? "OK  " : "FAIL", detail.c_str());
    if (!ok) { ++failures; }
}

// The buffer the kernel wrote, read back THROUGH THE VIEW the caller handed it. `device_rows` is
// the underlying buffer's row extent (the parent's, for a dim-0 row view) and `origin` the first
// row the view covers.
struct Comparison {
    double worst = 0.0;
    double scale = 0.0;
    int bad      = 0;
    int cells    = 0;
};

bool close_enough(double have, double want) {
    return std::fabs(have - want) <= 0.01 * std::max(1.0, std::fabs(want));
}

Comparison compare_engine(const Payload& p, const std::vector<std::uint16_t>& got, int origin,
                          int device_rows) {
    Comparison c;
    for (int row = 0; row < p.rows; ++row) {
        for (int col = 0; col < p.cols; ++col) {
            const double want = p.reference[static_cast<std::size_t>(row) * p.cols + col];
            const double have = bf16_to_double(got[engine_at(origin + row, col, device_rows)]);
            c.worst           = std::max(c.worst, std::fabs(have - want));
            c.scale           = std::max(c.scale, std::fabs(want));
            ++c.cells;
            if (!close_enough(have, want)) { ++c.bad; }
        }
    }
    return c;
}

// The PRE-FIX decoder's reading: out[row * t + col]. At one token the two coincide, so the
// discriminator only exists for cols > 1.
// The PRE-FIX decoder's reading of the same buffer: it stored at out[row * t + col], so read it
// back that way and require the reading to DISAGREE. Only meaningful for a contiguous destination
// (origin == 0), which is the only place this is used.
Comparison compare_token_major(const Payload& p, const std::vector<std::uint16_t>& got) {
    Comparison c;
    for (int row = 0; row < p.rows; ++row) {
        for (int col = 0; col < p.cols; ++col) {
            const double want = p.reference[static_cast<std::size_t>(row) * p.cols + col];
            const std::size_t at = static_cast<std::size_t>(row) * p.cols + col;
            if (at >= got.size()) { continue; }
            const double have = bf16_to_double(got[at]);
            c.worst           = std::max(c.worst, std::fabs(have - want));
            c.scale           = std::max(c.scale, std::fabs(want));
            ++c.cells;
            if (!close_enough(have, want)) { ++c.bad; }
        }
    }
    return c;
}

std::string line_of(const char* read, const Comparison& c) {
    char buffer[192];
    std::snprintf(buffer, sizeof(buffer), "%-10s worst=%-12.6g (ref max %-10.6g) bad=%d/%d", read,
                  c.worst, c.scale, c.bad, c.cells);
    return std::string(buffer);
}

// Agreement between the REGISTERED output and the GENERIC output, element by element, on the shape
// both routes serve. This is the assertion that makes a git-tracked sibling the authority.
int disagreement_engine(const Payload& p, const std::vector<std::uint16_t>& a,
                        const std::vector<std::uint16_t>& b, int device_rows) {
    int bad = 0;
    for (int row = 0; row < p.rows; ++row) {
        for (int col = 0; col < p.cols; ++col) {
            const std::size_t at = engine_at(row, col, device_rows);
            if (!close_enough(bf16_to_double(a[at]), bf16_to_double(b[at]))) { ++bad; }
        }
    }
    return bad;
}

bool cuda_ready(std::string& why) {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        why = std::string("cudaGetDeviceCount: ") + cudaGetErrorString(err);
        return false;
    }
    if (count == 0) {
        why = "no CUDA device";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    std::string why;
    if (!cuda_ready(why)) {
        std::printf("SKIP (return 77): %s\n", why.c_str());
        return 77;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("rowsplit_generic layout regression test -- device %s (sm_%d%d)\n", prop.name,
                prop.major, prop.minor);

    // =========================================================================================
    // 1/2/3. Shared shape (1024, 5120) at T=1 and T=8: the registry carries it, so the REGISTERED
    // dispatcher is the POSITIVE CONTROL and the reference point for "the two routes agree".
    // =========================================================================================
    std::printf("\n== CONTROL + AGREEMENT: registered q4 vs generic decoder vs reference ==\n");
    for (int tokens : {1, 8}) {
        const Payload p = make_payload(1024, 5120, tokens, 4242u);
        void* d_codes   = nullptr;
        void* d_scales  = nullptr;
        void* d_x       = nullptr;
        void* d_out     = nullptr;
        const std::size_t out_bytes = static_cast<std::size_t>(p.rows) * p.cols * 2;
        cudaMalloc(&d_codes, p.codes.size());
        cudaMalloc(&d_scales, p.scales.size() * 2);
        cudaMalloc(&d_x, p.x_bits.size() * 2);
        cudaMalloc(&d_out, out_bytes);
        cudaMemcpy(d_codes, p.codes.data(), p.codes.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scales, p.scales.data(), p.scales.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, p.x_bits.data(), p.x_bits.size() * 2, cudaMemcpyHostToDevice);

        const Weight w = make_weight(p, d_codes, d_scales);
        Tensor x(d_x, DType::BF16, {p.k, p.cols});
        Tensor out(d_out, DType::BF16, {p.rows, p.cols});

        const std::string tag = "T=" + std::to_string(tokens);
        const bool registered = ninfer::ops::detail::q4_a16_shape_registered(p.rows, p.k, p.cols);
        report("registry carries (1024,5120) " + tag, registered, "positive control");

        std::vector<std::uint16_t> from_registered(out_bytes / 2, 0);
        if (registered) {
            cudaMemset(d_out, 0, out_bytes);
            ninfer::ops::detail::q4_dispatch(x, w, out, LinearPolicy::A16Only, nullptr);
            const cudaError_t e = cudaDeviceSynchronize();
            report("registered dispatch ran " + tag, e == cudaSuccess, cudaGetErrorString(e));
            cudaMemcpy(from_registered.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);
            const Comparison c = compare_engine(p, from_registered, 0, p.rows);
            report("REGISTERED == independent reference (engine layout) " + tag, c.bad == 0,
                   line_of("engine", c));
        }

        cudaMemset(d_out, 0, out_bytes);
        report("generic admits the same problem " + tag,
               ninfer::ops::detail::generic_rowsplit_problem_ok(x, w, out), "");
        std::string note;
        bool threw = false;
        try {
            ninfer::ops::detail::generic_rowsplit_linear_dispatch(x, w, out, nullptr);
        } catch (const std::exception& error) {
            threw = true;
            note  = error.what();
        }
        if (threw) {
            report("GENERIC == independent reference (engine layout) " + tag, false,
                   "refused: " + note);
        } else {
            const cudaError_t e2 = cudaDeviceSynchronize();
            if (e2 != cudaSuccess) {
                report("GENERIC ran " + tag, false, cudaGetErrorString(e2));
            } else {
                std::vector<std::uint16_t> got(out_bytes / 2, 0);
                cudaMemcpy(got.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);
                const Comparison g = compare_engine(p, got, 0, p.rows);
                report("GENERIC == independent reference (engine layout) " + tag, g.bad == 0,
                       line_of("engine", g));
                if (registered) {
                    const int diff =
                        disagreement_engine(p, from_registered, got, p.rows);
                    report("GENERIC == REGISTERED element-wise (tracked sibling) " + tag,
                           diff == 0, "disagreeing cells=" + std::to_string(diff));
                }
                if (tokens == 1) {
                    std::printf("  %-62s SKIP  both indexings coincide at one token\n",
                                "token-major reading does NOT match");
                } else {
                    const Comparison tm = compare_token_major(p, got);
                    report("token-major reading does NOT match (the PRE-FIX index)", tm.bad > 0,
                           line_of("tokenmajor", tm));
                }
            }
        }
        cudaFree(d_codes);
        cudaFree(d_scales);
        cudaFree(d_x);
        cudaFree(d_out);
    }

    // =========================================================================================
    // 4. An unregistered shape: the registry refuses, the generic decoder must be right.
    // =========================================================================================
    std::printf("\n== SHAPE: unregistered (8,256) T=5, contiguous destination ==\n");
    {
        const Payload p = make_payload(8, 256, 5, 20260918u);
        void* d_codes   = nullptr;
        void* d_scales  = nullptr;
        void* d_x       = nullptr;
        void* d_out     = nullptr;
        const std::size_t out_bytes = static_cast<std::size_t>(p.rows) * p.cols * 2;
        cudaMalloc(&d_codes, p.codes.size());
        cudaMalloc(&d_scales, p.scales.size() * 2);
        cudaMalloc(&d_x, p.x_bits.size() * 2);
        cudaMalloc(&d_out, out_bytes);
        cudaMemcpy(d_codes, p.codes.data(), p.codes.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scales, p.scales.data(), p.scales.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, p.x_bits.data(), p.x_bits.size() * 2, cudaMemcpyHostToDevice);
        cudaMemset(d_out, 0, out_bytes);
        const Weight w = make_weight(p, d_codes, d_scales);
        Tensor x(d_x, DType::BF16, {p.k, p.cols});
        Tensor out(d_out, DType::BF16, {p.rows, p.cols});
        report("registry refuses (8,256)x5",
               !ninfer::ops::detail::q4_a16_shape_registered(p.rows, p.k, p.cols), "");
        report("generic admits it",
               ninfer::ops::detail::generic_rowsplit_problem_ok(x, w, out), "");
        ninfer::ops::detail::generic_rowsplit_linear_dispatch(x, w, out, nullptr);
        const cudaError_t e = cudaDeviceSynchronize();
        report("generic ran", e == cudaSuccess, cudaGetErrorString(e));
        std::vector<std::uint16_t> got(out_bytes / 2, 0);
        cudaMemcpy(got.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);
        const Comparison c = compare_engine(p, got, 0, p.rows);
        report("GENERIC == independent reference (engine layout)", c.bad == 0, line_of("engine", c));
        const Comparison tm = compare_token_major(p, got);
        report("token-major reading does NOT match (the PRE-FIX index)", tm.bad > 0,
               line_of("tokenmajor", tm));
        cudaFree(d_codes);
        cudaFree(d_scales);
        cudaFree(d_x);
        cudaFree(d_out);
    }

    // =========================================================================================
    // 5. A dim-0 row view as the destination -- the shape gdn_input_proj hands this decoder
    //    (qkv.slice(0, 0, qk_weight.n)). Tensor::slice keeps the parent's nb, so the view is
    //    legitimately non-contiguous; it must be served, and only its own rows written.
    // =========================================================================================
    std::printf("\n== ROWVIEW: dim-0 destination row views ==\n");
    {
        const Payload p       = make_payload(1024, 5120, 4, 77u);
        const int parent_rows = 2048;
        void* d_codes         = nullptr;
        void* d_scales        = nullptr;
        void* d_x             = nullptr;
        void* d_parent        = nullptr;
        const std::size_t parent_bytes =
            static_cast<std::size_t>(parent_rows) * p.cols * 2;
        cudaMalloc(&d_codes, p.codes.size());
        cudaMalloc(&d_scales, p.scales.size() * 2);
        cudaMalloc(&d_x, p.x_bits.size() * 2);
        cudaMalloc(&d_parent, parent_bytes);
        cudaMemcpy(d_codes, p.codes.data(), p.codes.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scales, p.scales.data(), p.scales.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, p.x_bits.data(), p.x_bits.size() * 2, cudaMemcpyHostToDevice);

        std::vector<std::uint16_t> sentinel(static_cast<std::size_t>(parent_rows) * p.cols,
                                            kSentinel);
        const Weight w = make_weight(p, d_codes, d_scales);
        Tensor x(d_x, DType::BF16, {p.k, p.cols});
        for (int start : {0, 1024}) {
            cudaMemcpy(d_parent, sentinel.data(), parent_bytes, cudaMemcpyHostToDevice);
            Tensor parent(d_parent, DType::BF16, {parent_rows, p.cols});
            Tensor dest = parent.slice(0, start, p.rows);
            const std::string tag = "start=" + std::to_string(start) +
                                    " nb[1]=" + std::to_string(dest.nb[1]);
            report("generic admits the row view [" + tag + "]",
                   ninfer::ops::detail::generic_rowsplit_problem_ok(x, w, dest), "admission agrees");
            std::string note;
            bool threw = false;
            try {
                ninfer::ops::detail::generic_rowsplit_linear_dispatch(x, w, dest, nullptr);
            } catch (const std::exception& error) {
                threw = true;
                note  = error.what();
            }
            if (threw) {
                report("row view dispatched [" + tag + "]", false, "REFUSED: " + note);
                continue;
            }
            const cudaError_t e = cudaDeviceSynchronize();
            report("row view dispatched [" + tag + "]", e == cudaSuccess, cudaGetErrorString(e));
            std::vector<std::uint16_t> got(sentinel.size(), 0);
            cudaMemcpy(got.data(), d_parent, parent_bytes, cudaMemcpyDeviceToHost);
            int touched = 0;
            for (int row = 0; row < parent_rows; ++row) {
                if (row >= start && row < start + p.rows) { continue; }
                for (int col = 0; col < p.cols; ++col) {
                    if (got[engine_at(row, col, parent_rows)] != kSentinel) { ++touched; }
                }
            }
            report("sentinel rows untouched [" + tag + "]", touched == 0,
                   std::to_string(touched) + " cells changed outside the view");
            const Comparison c = compare_engine(p, got, start, parent_rows);
            report("view == independent reference (engine layout) [" + tag + "]", c.bad == 0,
                   line_of("engine", c));
        }
        cudaFree(d_codes);
        cudaFree(d_scales);
        cudaFree(d_x);
        cudaFree(d_parent);
    }

    // =========================================================================================
    // 6. SWIGLU at 6x5 -- the DEAD-and-OUT-OF-BOUNDS guard.  rows*columns = 30, the launch is
    //    ceil(30/256) = 1 block of 256 threads, so indices 30..255 exist; the PRE-FIX guard
    //    admitted index < total_rows * columns = 30 * 5 = 150, i.e. 120 indices past the end of a
    //    30-element destination. Both buffers are allocated with a sentinel region beyond their
    //    declared extent, so the overrun is measured rather than inferred. The gate_up padding is
    //    +inf so an out-of-range READ also shows up as a non-finite result.
    // =========================================================================================
    std::printf("\n== SWIGLU 6x5: the guard must be LIVE and the shape in-bounds ==\n");
    {
        const int rows         = 6;
        const int cols         = 5;
        const int gate_up_rows = 2 * rows;
        const std::size_t gate_elems = static_cast<std::size_t>(gate_up_rows) * cols;  // 60
        const std::size_t out_elems  = static_cast<std::size_t>(rows) * cols;          // 30
        const std::size_t gate_alloc = 512;   // 452 sentinel cells past the declared extent
        const std::size_t out_alloc  = 512;   // 482 sentinel cells past the declared extent

        std::mt19937 rng(31337u);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        std::vector<std::uint16_t> gate_host(gate_alloc, kGateUpTrap);
        for (std::size_t i = 0; i < gate_elems; ++i) { gate_host[i] = float_to_bf16(dist(rng)); }

        void* d_gate = nullptr;
        void* d_out  = nullptr;
        cudaMalloc(&d_gate, gate_alloc * 2);
        cudaMalloc(&d_out, out_alloc * 2);
        cudaMemcpy(d_gate, gate_host.data(), gate_alloc * 2, cudaMemcpyHostToDevice);
        std::vector<std::uint16_t> out_seed(out_alloc, kSentinel);
        cudaMemcpy(d_out, out_seed.data(), out_alloc * 2, cudaMemcpyHostToDevice);

        Tensor gate_up(d_gate, DType::BF16, {gate_up_rows, cols});
        Tensor out(d_out, DType::BF16, {rows, cols});
        ninfer::ops::detail::generic_swiglu_split_dispatch(gate_up, out, nullptr);
        const cudaError_t e = cudaDeviceSynchronize();
        report("generic swiglu ran at 6x5", e == cudaSuccess, cudaGetErrorString(e));

        std::vector<std::uint16_t> got_out(out_alloc, 0);
        std::vector<std::uint16_t> got_gate(gate_alloc, 0);
        cudaMemcpy(got_out.data(), d_out, out_alloc * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(got_gate.data(), d_gate, gate_alloc * 2, cudaMemcpyDeviceToHost);

        int bad = 0, printed = 0;
        double worst = 0.0;
        for (int col = 0; col < cols; ++col) {
            for (int i = 0; i < rows; ++i) {
                const double gate = bf16_to_double(gate_host[engine_at(i, col, gate_up_rows)]);
                const double up   = bf16_to_double(gate_host[engine_at(i + rows, col, gate_up_rows)]);
                const double want = gate / (1.0 + std::exp(-gate)) * up;
                const double have = bf16_to_double(got_out[engine_at(i, col, rows)]);
                worst = std::max(worst, std::fabs(have - want));
                if (!close_enough(have, want)) {
                    ++bad;
                    if (printed < 6) {
                        ++printed;
                        std::printf("      i=%d t=%d gate=%+.6f up=%+.6f want=%+.6f got=%+.6f\n", i,
                                    col, gate, up, want, have);
                    }
                }
            }
        }
        char line[192];
        std::snprintf(line, sizeof(line), "worst abs err=%.6g bad=%d/%d", worst, bad,
                      static_cast<int>(out_elems));
        report("GENERIC swiglu == independent reference (engine layout)", bad == 0, line);

        int out_clobbered = 0, gate_changed = 0;
        for (std::size_t i = out_elems; i < out_alloc; ++i) {
            if (got_out[i] != kSentinel) { ++out_clobbered; }
        }
        for (std::size_t i = 0; i < gate_alloc; ++i) {
            if (got_gate[i] != gate_host[i]) { ++gate_changed; }
        }
        char gline[224];
        std::snprintf(gline, sizeof(gline),
                      "%d of %d sentinel cells past out[0..%d] changed (pre-fix: >0)", out_clobbered,
                      static_cast<int>(out_alloc - out_elems), static_cast<int>(out_elems) - 1);
        report("destination written only inside its declared extent", out_clobbered == 0, gline);
        report("gate_up is read-only (source plane unchanged)", gate_changed == 0,
               std::to_string(gate_changed) + " cells changed");
        cudaFree(d_gate);
        cudaFree(d_out);
    }

    // =========================================================================================
    // 7. SWIGLU at an engine MLP geometry: intermediate 12288 (qwen3_5_9b
    //    TextConfig::intermediate = 48 * 256), T=3.  Here rows*columns IS a multiple of the
    //    256-thread launch, which is exactly why the pre-fix guard was merely dead rather than
    //    harmful -- this arm exists so the shipped geometry is covered by the same assertions.
    // =========================================================================================
    std::printf("\n== SWIGLU 12288x3: a shipped MLP geometry (no-regression arm) ==\n");
    {
        const int rows         = 12288;
        const int cols         = 3;
        const int gate_up_rows = 2 * rows;
        const std::size_t gate_elems = static_cast<std::size_t>(gate_up_rows) * cols;
        const std::size_t out_elems  = static_cast<std::size_t>(rows) * cols;
        const std::size_t pad        = 512;
        const std::size_t gate_alloc = gate_elems + pad;
        const std::size_t out_alloc  = out_elems + pad;

        std::mt19937 rng(991u);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        std::vector<std::uint16_t> gate_host(gate_alloc, kGateUpTrap);
        for (std::size_t i = 0; i < gate_elems; ++i) { gate_host[i] = float_to_bf16(dist(rng)); }

        void* d_gate = nullptr;
        void* d_out  = nullptr;
        cudaMalloc(&d_gate, gate_alloc * 2);
        cudaMalloc(&d_out, out_alloc * 2);
        cudaMemcpy(d_gate, gate_host.data(), gate_alloc * 2, cudaMemcpyHostToDevice);
        std::vector<std::uint16_t> out_seed(out_alloc, kSentinel);
        cudaMemcpy(d_out, out_seed.data(), out_alloc * 2, cudaMemcpyHostToDevice);

        Tensor gate_up(d_gate, DType::BF16, {gate_up_rows, cols});
        Tensor out(d_out, DType::BF16, {rows, cols});
        ninfer::ops::detail::generic_swiglu_split_dispatch(gate_up, out, nullptr);
        const cudaError_t e = cudaDeviceSynchronize();
        report("generic swiglu ran at 12288x3", e == cudaSuccess, cudaGetErrorString(e));

        std::vector<std::uint16_t> got_out(out_alloc, 0);
        cudaMemcpy(got_out.data(), d_out, out_alloc * 2, cudaMemcpyDeviceToHost);
        int bad = 0;
        double worst = 0.0;
        for (int col = 0; col < cols; ++col) {
            for (int i = 0; i < rows; ++i) {
                const double gate = bf16_to_double(gate_host[engine_at(i, col, gate_up_rows)]);
                const double up   = bf16_to_double(gate_host[engine_at(i + rows, col, gate_up_rows)]);
                const double want = gate / (1.0 + std::exp(-gate)) * up;
                const double have = bf16_to_double(got_out[engine_at(i, col, rows)]);
                worst = std::max(worst, std::fabs(have - want));
                if (!close_enough(have, want)) { ++bad; }
            }
        }
        char line[192];
        std::snprintf(line, sizeof(line), "worst abs err=%.6g bad=%d/%d", worst, bad,
                      static_cast<int>(out_elems));
        report("GENERIC swiglu == independent reference (engine layout)", bad == 0, line);
        int clobbered = 0;
        for (std::size_t i = out_elems; i < out_alloc; ++i) {
            if (got_out[i] != kSentinel) { ++clobbered; }
        }
        report("destination written only inside its declared extent", clobbered == 0,
               std::to_string(clobbered) + " sentinel cells changed");
        cudaFree(d_gate);
        cudaFree(d_out);
    }

    // =========================================================================================
    // 8. GDN gating projection -- the SAME defect at a third site, which FIRSTSERVE-3's patch does
    //    not cover. generic_gating_kernel read its INPUT (the rmsnorm output h) as
    //    h[j*t + col0 + c] while its own store four lines later used (col0+c)*heads + row: the two
    //    halves of one kernel disagreeing about the convention, which cannot both be right. The
    //    reference reads back the h the kernel actually produced, so this isolates the projection's
    //    layout -- a FAIL means the index, not the rounding.
    // =========================================================================================
    std::printf("\n== GATING: generic norm-gating projection, engine layout ==\n");
    {
        const int k     = 64;
        const int t     = 5;
        const int heads = 3;
        const float eps = 1e-6f;

        std::mt19937 rng(2026u);
        std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> w_dist(-0.3f, 0.3f);

        std::vector<std::uint16_t> x_bits(static_cast<std::size_t>(k) * t);
        for (std::uint16_t& v : x_bits) { v = float_to_bf16(x_dist(rng)); }
        std::vector<std::uint16_t> norm_bits(k, float_to_bf16(1.0f));
        std::vector<std::uint16_t> a_bits(static_cast<std::size_t>(heads) * k);
        std::vector<std::uint16_t> b_bits(static_cast<std::size_t>(heads) * k);
        for (std::uint16_t& v : a_bits) { v = float_to_bf16(w_dist(rng)); }
        for (std::uint16_t& v : b_bits) { v = float_to_bf16(w_dist(rng)); }
        std::vector<float> a_log(heads), dt_bias(heads);
        for (int r = 0; r < heads; ++r) {
            a_log[r]   = 0.2f + 0.1f * static_cast<float>(r);
            dt_bias[r] = 0.05f * static_cast<float>(r);
        }

        void* d_x    = nullptr;
        void* d_norm = nullptr;
        void* d_a    = nullptr;
        void* d_b    = nullptr;
        void* d_alog = nullptr;
        void* d_dt   = nullptr;
        void* d_h    = nullptr;
        void* d_g    = nullptr;
        void* d_beta = nullptr;
        cudaMalloc(&d_x, x_bits.size() * 2);
        cudaMalloc(&d_norm, norm_bits.size() * 2);
        cudaMalloc(&d_a, a_bits.size() * 2);
        cudaMalloc(&d_b, b_bits.size() * 2);
        cudaMalloc(&d_alog, static_cast<std::size_t>(heads) * 4);
        cudaMalloc(&d_dt, static_cast<std::size_t>(heads) * 4);
        cudaMalloc(&d_h, x_bits.size() * 2);
        cudaMalloc(&d_g, static_cast<std::size_t>(heads) * t * 4);
        cudaMalloc(&d_beta, static_cast<std::size_t>(heads) * t * 4);
        cudaMemcpy(d_x, x_bits.data(), x_bits.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_norm, norm_bits.data(), norm_bits.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_a, a_bits.data(), a_bits.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b_bits.data(), b_bits.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_alog, a_log.data(), static_cast<std::size_t>(heads) * 4,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_dt, dt_bias.data(), static_cast<std::size_t>(heads) * 4,
                   cudaMemcpyHostToDevice);

        auto bf16_weight = [](std::int32_t n, std::int32_t kk, const void* data) {
            Weight w{};
            w.payload         = data;
            w.payload_bytes   = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(kk) * 2;
            w.qtype           = QType::BF16_CTRL;
            w.shape[0]        = n;
            w.shape[1]        = kk;
            w.padded_shape[0] = n;
            w.padded_shape[1] = kk;
            w.ndim            = 2;
            w.qdata           = data;
            w.n               = n;
            w.k               = kk;
            w.layout          = QuantLayout::Contiguous;
            return w;
        };

        Tensor x(d_x, DType::BF16, {k, t});
        Tensor norm_weight(d_norm, DType::BF16, {k, 1});
        Tensor a_log_t(d_alog, DType::FP32, {heads, 1});
        Tensor dt_t(d_dt, DType::FP32, {heads, 1});
        Tensor h(d_h, DType::BF16, {k, t});
        Tensor g(d_g, DType::FP32, {heads, t});
        Tensor beta(d_beta, DType::FP32, {heads, t});
        const Weight a_w = bf16_weight(heads, k, d_a);
        const Weight b_w = bf16_weight(heads, k, d_b);

        report("generic admits the gating problem",
               ninfer::ops::detail::generic_norm_gating_problem_ok(x, norm_weight, a_w, b_w,
                                                                   a_log_t, dt_t, h, g, beta),
               "admission agrees");
        ninfer::ops::detail::generic_norm_gating_proj_dispatch(x, norm_weight, eps, a_w, b_w,
                                                               a_log_t, dt_t, h, g, beta, nullptr);
        const cudaError_t e = cudaDeviceSynchronize();
        report("gating dispatch ran", e == cudaSuccess, cudaGetErrorString(e));

        std::vector<std::uint16_t> h_bits(x_bits.size(), 0);
        std::vector<float> got_g(static_cast<std::size_t>(heads) * t, 0.0f);
        std::vector<float> got_beta(static_cast<std::size_t>(heads) * t, 0.0f);
        cudaMemcpy(h_bits.data(), d_h, h_bits.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(got_g.data(), d_g, got_g.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(got_beta.data(), d_beta, got_beta.size() * 4, cudaMemcpyDeviceToHost);

        int bad_g = 0, bad_beta = 0;
        double worst = 0.0;
        for (int col = 0; col < t; ++col) {
            for (int row = 0; row < heads; ++row) {
                double acc_a = 0.0, acc_b = 0.0;
                for (int j = 0; j < k; ++j) {
                    const double hv = bf16_to_double(h_bits[engine_at(j, col, k)]);
                    acc_a += bf16_to_double(a_bits[static_cast<std::size_t>(row) * k + j]) * hv;
                    acc_b += bf16_to_double(b_bits[static_cast<std::size_t>(row) * k + j]) * hv;
                }
                const double arg = acc_a + static_cast<double>(dt_bias[row]);
                const double sp  = (arg > 20.0) ? arg : std::log1p(std::exp(arg));
                const double want_g = -std::exp(static_cast<double>(a_log[row])) * sp;
                const double eb     = std::exp(-std::fabs(acc_b));
                const double want_beta = (acc_b >= 0.0 ? 1.0 : eb) / (1.0 + eb);
                const std::size_t at = engine_at(row, col, heads);
                worst = std::max(worst, std::fabs(static_cast<double>(got_g[at]) - want_g));
                if (!close_enough(static_cast<double>(got_g[at]), want_g)) { ++bad_g; }
                if (!close_enough(static_cast<double>(got_beta[at]), want_beta)) { ++bad_beta; }
            }
        }
        char gline[192];
        std::snprintf(gline, sizeof(gline), "worst abs err=%.6g bad=%d/%d", worst, bad_g, heads * t);
        report("gating g == independent reference (engine layout)", bad_g == 0, gline);
        report("gating beta == independent reference (engine layout)", bad_beta == 0,
               "bad=" + std::to_string(bad_beta) + "/" + std::to_string(heads * t));
        cudaFree(d_x);
        cudaFree(d_norm);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_alog);
        cudaFree(d_dt);
        cudaFree(d_h);
        cudaFree(d_g);
        cudaFree(d_beta);
    }

    std::printf("\n%s  (failures=%d)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
