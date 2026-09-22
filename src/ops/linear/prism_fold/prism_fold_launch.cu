// prism_fold_launch.cu -- the ONE __global__ that forces Route A's arithmetic to instantiate,
// plus its launcher.  See prism_fold_launch.h for the contract, the order, and the declared
// BF16 gap.
//
// THE SPLIT OF LABOUR, AND WHY THE KERNEL DOES NOT RE-SPELL THE BUTTERFLY
// ---------------------------------------------------------------------
// Every element of the arithmetic lives in `prism_fold.cuh` as a `__host__ __device__` inline
// function: `prism_fold_butterfly_stage_range` owns which two elements pair up at stage `h`,
// `prism_fold_gdn_source_index` owns which activation channel a grouped position reads, and
// `prism_fold_forward_out` owns the serial composition.  This file owns ONLY the parallel
// partition -- how many workers there are and which slice of the pair range each takes -- and
// the shared-memory staging.  That is the whole reason the `.cuh` is host-callable: the
// partition can be simulated on the host and compared against the serial path, which is the
// only evidence available for this kernel on a line with no GPU window.  A kernel that
// re-derived the pair index would be free to disagree with the serial path *silently*, and
// that disagreement is precisely the defect class this operator exists to avoid.

#include "ops/linear/prism_fold/prism_fold_launch.h"

// The build-list admission test.  `prism_fold.h` reads this macro to decide
// `kPrismFoldRuntimeSupportsTransform`; defining it here is a per-TU statement that THIS
// translation unit carries the device half, and it is deliberately NOT the statement that the
// ENGINE carries it.  The engine's TUs are compiled without this macro, so the engine refuses
// a folded checkpoint until `prism_fold_launch.cu` is admitted to `ninfer_ops` -- which is a
// CMakeLists edit this line is forbidden to make.  See the header.
#define PRISM_FOLD_ACTIVATION_TRANSFORM_LINKED 1
#include "ops/linear/prism_fold/prism_fold.cuh"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ninfer::ops::prism_fold {
namespace {

//: One CUDA block per (token, rotation block).  Threads cooperate on the shared-memory stage.
//: 256 threads for a 1024-wide block is 4 elements per thread of the gather and the store, and
//: `1024/2 = 512` pairs at the first stage, so every stage has at least two pairs per thread --
//: no stage is a serialisation point with idle lanes.
constexpr int kPrismFoldThreads = 256;

//: Gather + sign into shared memory, apply the `log2(block)` stages with a barrier between
//: them, then scale and store.  The one barriered loop is why the blockwise transform is the
//: shared-memory realisation rather than the fork's register butterfly: a whole 1024-element
//: block cannot live in one thread's registers (1024 fp32 = 4 KiB, against a 255-register
//: budget), so the tree's own "64 live floats per thread" budget does not reach this width.
//: This is a CHOICE OF REALISATION, not a numerical approximation: the fork's own
//: `ggml/src/ggml-cuda/fwht.cu` uses the shared-memory path above N = 2048 for the same reason
//: and the two paths are bit-identical in exact arithmetic.
__global__ void prism_fold_activation_kernel(const float* __restrict__ src,
                                            float* __restrict__ dst,
                                            const float* __restrict__ signs, int width,
                                            int block, int n_v, int n_k) {
    extern __shared__ float smem[];
    const int off = blockIdx.x * block;          // this block's slice of the contraction axis
    const std::size_t token_base = static_cast<std::size_t>(blockIdx.y) * width;

    const bool permute = (n_v > 0 && n_k > 0);
    const int hd = permute ? width / n_v : 0;
    const int rep = permute ? n_v / n_k : 0;

    for (int i = threadIdx.x; i < block; i += kPrismFoldThreads) {
        int channel = off + i;
        if (permute) {
            channel = prism_fold_gdn_source_index(off + i, n_k, rep, hd);
        }
        // D first, and the sign vector is indexed by the OUTPUT channel of the rotation, not by
        // the gathered one: the signs live in the rotated basis's own coordinate order, which
        // is the order the weight was stored in and therefore the order `A a` produces.
        smem[i] = src[token_base + channel] * signs[off + i];
    }
    __syncthreads();

    const int pairs = block / 2;
    for (int h = 1; h < block; h *= 2) {
        // The SAME range primitive the serial path calls, one pair per call.  `h` is a power of
        // two and divides `block/2` for every stage, so the range check inside cannot fail.
        for (int p = threadIdx.x; p < pairs; p += kPrismFoldThreads) {
            (void)prism_fold_butterfly_stage_range(smem, block, h, p, p + 1);
        }
        __syncthreads();
    }

    const float scale = 1.0f / sqrtf(static_cast<float>(block));
    for (int i = threadIdx.x; i < block; i += kPrismFoldThreads) {
        dst[token_base + off + i] = smem[i] * scale;
    }
}

}   // namespace

bool prism_fold_activation_shape_ok(const Tensor& x, const Tensor& out, const Tensor& signs,
                                    int block, int n_v, int n_k) noexcept {
    if (x.dtype != DType::FP32 || out.dtype != DType::FP32 || signs.dtype != DType::FP32) {
        return false;
    }
    if (x.ne[2] != 1 || x.ne[3] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        return false;
    }
    if (x.ne[0] != out.ne[0] || x.ne[1] != out.ne[1]) {
        return false;
    }
    if (x.data == nullptr || out.data == nullptr || signs.data == nullptr) {
        return false;
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        return false;
    }
    const int k = x.ne[0];
    if (!prism_fold_width_ok(k, block) || block > kPrismFoldBlock) {
        return false;
    }
    if (signs.ne[0] != k) {
        return false;
    }
    if ((n_v == 0) != (n_k == 0)) {
        return false;   // both zero (no permutation) or both set; half a geometry is a typo
    }
    if (n_v != 0 && !prism_fold_gdn_geometry_ok(k, n_v, n_k)) {
        return false;
    }
    return true;
}

void prism_fold_activation_launch(const Tensor& x, Tensor& out, const Tensor& signs, int block,
                                  int n_v, int n_k, cudaStream_t stream) {
    if (!prism_fold_activation_shape_ok(x, out, signs, block, n_v, n_k)) {
        throw std::invalid_argument(
            "prism_fold: refusing the activation transform -- x/out/signs must be F32 [K,T] "
            "contiguous with K a positive multiple of a power-of-two block no larger than " +
            std::to_string(kPrismFoldBlock) + ", signs must be exactly [K], and (n_v,n_k) must "
            "be both zero or a legal GDN geometry pair.  A pad would change the dot product and "
            "would do it silently, so this refuses instead of padding.");
    }
    if (x.ne[0] == 0 || x.ne[1] == 0) {
        return;   // an empty problem is a no-op, not a launch with a zero grid dimension
    }
    const int width = x.ne[0];
    if (out.data == x.data) {
        throw std::invalid_argument(
            "prism_fold: in-place is not served by the parallel path -- the kernel stages only "
            "its own rotation block, and a caller that wants in-place semantics must use the "
            "host-callable prism_fold_forward_out, which stages each block before the store.  "
            "Refused by name rather than made silently conditional.");
    }
    const dim3 grid(static_cast<unsigned>(width / block), static_cast<unsigned>(x.ne[1]));
    prism_fold_activation_kernel<<<grid, kPrismFoldThreads, block * sizeof(float), stream>>>(
        static_cast<const float*>(x.data), static_cast<float*>(out.data),
        static_cast<const float*>(signs.data), width, block, n_v, n_k);
}

}   // namespace ninfer::ops::prism_fold
