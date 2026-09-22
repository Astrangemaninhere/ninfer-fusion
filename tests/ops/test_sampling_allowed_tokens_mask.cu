// The `allowed_tokens` mask contract of the sampling Op.
//
// WHAT THIS TEST IS FOR
// ---------------------
// `SamplingConfig::allowed_tokens` is a read-only device bitset: bit v set means vocabulary token v
// is permitted, bit v clear means its adjusted logit is negative infinity BEFORE ranking and
// filtering. The mask is the correctness prerequisite for any structured/constrained output that
// reaches this Op.
//
// The mask was originally honoured only on the routes that already consulted
// sampling_adjusted_logit(), i.e. the ones taken when a presence/frequency penalty is active. The
// greedy no-penalty fast path in sample_row_kernel() argmaxes the RAW logits and never calls
// sampling_adjusted_logit() at all, so with `temperature <= 0`, `presence_penalty == 0` and
// `frequency_penalty == 0` a mask was silently ignored and a masked-out token could be committed.
// That is the defect this file pins:
//
//   CASE 1  greedy + zero penalties + mask allowing token 3 only, while the raw argmax is token 7.
//           A correct Op returns 3.  The defective Op returns 7.
//   CASE 2  the negative control: the same logits with NO mask must still return the raw argmax 7,
//           so a pass cannot come from the mask path having been taken unconditionally.
//   CASE 3  the contrast that localises the defect: the same mask on the POSITIVE-temperature route
//           returns the allowed token in both states, because that route already consulted the mask.
//
// WHAT THIS TEST DELIBERATELY DOES NOT DO
// ---------------------------------------
// It launches sample_row_kernel() -- the single-block route -- directly, with one row and
// token_domain well inside kSamplerTileItems. It does not go through the public ops::sample()
// dispatch (that lives in a .cu and would drag the engine's device link in) and it does not cover
// the multi-block partial/group route, which reaches the same sampling_adjusted_logit() through
// sampling_build_truncated_*_fast(). The point of this target is that it links NO ninfer library:
// it is header-only plus CUDA::cudart, so it keeps building and running independently of the
// engine's device link.
//
// No GPU is required to *build* it; a GPU is required to *run* it.

#include "ops/kernel/sampling.cuh" // sample_row_kernel, kSamplerBlock, kSamplerTileItems

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// The Op surface lives in ninfer::ops (sampling.cuh opens that namespace).
using namespace ninfer::ops; // NOLINT(google-build-using-namespace) -- a test TU

namespace {

int g_failures = 0;
int g_checks   = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    ++g_checks;
    if (ok) {
        std::fprintf(stdout, "ok   %s%s\n", what.c_str(),
                     detail.empty() ? "" : ("  [" + detail + "]").c_str());
        return;
    }
    ++g_failures;
    std::fprintf(stdout, "FAIL %s%s\n", what.c_str(),
                 detail.empty() ? "" : ("  [" + detail + "]").c_str());
}

void check_cuda(cudaError_t err, const char* what) {
    check(err == cudaSuccess, std::string("cuda ") + what,
          err == cudaSuccess ? "" : cudaGetErrorString(err));
}

// Runs one row through the single-block sampling route and returns the committed token id.
std::int32_t run_one_row(float temperature, std::int32_t raw_argmax_token,
                         const std::vector<std::int32_t>& mask_words, bool with_mask,
                         std::int32_t* out_ok) {
    constexpr std::int32_t kTokenDomain = 64;
    constexpr std::int32_t kRows        = 1;

    std::vector<__nv_bfloat16> host_logits(kTokenDomain);
    for (std::int32_t v = 0; v < kTokenDomain; ++v) {
        host_logits[static_cast<std::size_t>(v)] = __float2bfloat16(0.0f);
    }
    host_logits[static_cast<std::size_t>(raw_argmax_token)] = __float2bfloat16(10.0f);
    host_logits[static_cast<std::size_t>(3)]                = __float2bfloat16(1.0f);

    __nv_bfloat16* dev_logits    = nullptr;
    std::int32_t* dev_out        = nullptr;
    SamplingConfig* dev_configs  = nullptr;
    std::int32_t* dev_positions  = nullptr;
    std::int32_t* dev_mask       = nullptr;
    check_cuda(cudaMalloc(&dev_logits, host_logits.size() * sizeof(__nv_bfloat16)), "malloc logits");
    check_cuda(cudaMalloc(&dev_out, sizeof(std::int32_t)), "malloc out");
    check_cuda(cudaMalloc(&dev_configs, sizeof(SamplingConfig)), "malloc configs");
    check_cuda(cudaMalloc(&dev_positions, kRows * sizeof(std::int32_t)), "malloc positions");
    if (with_mask) {
        check_cuda(cudaMalloc(&dev_mask, mask_words.size() * sizeof(std::int32_t)), "malloc mask");
    }
    if (g_failures != 0) { return -1; }

    SamplingConfig cfg;
    cfg.temperature          = temperature;
    cfg.top_k                = 20;
    cfg.top_p                = 1.0f;
    cfg.min_p                = 0.0f;
    cfg.presence_penalty     = 0.0f;
    cfg.frequency_penalty    = 0.0f;
    cfg.seed                 = 12345ull;
    cfg.token_counts         = nullptr;
    cfg.allowed_tokens       = nullptr; // set below only for the masked arms

    const std::int32_t position = 0;
    check_cuda(cudaMemcpy(dev_logits, host_logits.data(),
                          host_logits.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
               "copy logits");
    check_cuda(cudaMemcpy(dev_positions, &position, sizeof(position), cudaMemcpyHostToDevice),
               "copy position");
    if (with_mask) {
        check_cuda(cudaMemcpy(dev_mask, mask_words.data(),
                              mask_words.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                   "copy mask");
        cfg.allowed_tokens = dev_mask;
    }
    check_cuda(cudaMemcpy(dev_configs, &cfg, sizeof(cfg), cudaMemcpyHostToDevice), "copy configs");

    sample_row_kernel<<<kRows, kSamplerBlock>>>(dev_logits, dev_out, dev_configs, dev_positions,
                                               static_cast<std::int32_t>(kSamplePurposeDecode),
                                               kTokenDomain, kTokenDomain);
    check_cuda(cudaGetLastError(), "launch sample_row_kernel");
    check_cuda(cudaDeviceSynchronize(), "sync");

    std::int32_t picked = -1;
    check_cuda(cudaMemcpy(&picked, dev_out, sizeof(picked), cudaMemcpyDeviceToHost), "copy out");

    cudaFree(dev_logits);
    cudaFree(dev_out);
    cudaFree(dev_configs);
    cudaFree(dev_positions);
    if (dev_mask != nullptr) { cudaFree(dev_mask); }
    if (out_ok != nullptr) { *out_ok = g_failures == 0 ? 1 : 0; }
    return picked;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        // The contract is about device behaviour; with no device there is nothing to assert and
        // the run must not be reported as a pass.
        std::fprintf(stdout, "SKIP no CUDA device available\n");
        return 77;
    }

    // A mask with token 3 (and only token 3) permitted: word 0 has bit 3 set, word 1 is zero
    // (token_domain = 64 => ceil(64/32) = 2 words).
    const std::vector<std::int32_t> mask_only_token3 = {static_cast<std::int32_t>(1u << 3), 0};

    std::int32_t ok = 0;
    const std::int32_t case1 = run_one_row(0.0f, 7, mask_only_token3, /*with_mask=*/true, &ok);
    if (ok) {
        check(case1 == 3,
              "CASE 1 greedy + zero penalties + mask must NOT return a masked-out token",
              "raw argmax is 7, the only allowed token is 3, returned " +
                  std::to_string(case1));
    }

    ok = 0;
    const std::int32_t case2 = run_one_row(0.0f, 7, mask_only_token3, /*with_mask=*/false, &ok);
    if (ok) {
        check(case2 == 7,
              "CASE 2 negative control: with no mask the raw-logit argmax is still returned",
              "returned " + std::to_string(case2));
    }

    ok = 0;
    const std::int32_t case3 = run_one_row(1.0f, 7, mask_only_token3, /*with_mask=*/true, &ok);
    if (ok) {
        check(case3 == 3,
              "CASE 3 contrast: the positive-temperature route honours the same mask",
              "returned " + std::to_string(case3));
    }

    std::fprintf(stdout, "\n%d checks, %d failures\n", g_checks, g_failures);
    if (g_checks == 0) {
        std::fprintf(stdout, "NO CHECKS RAN\n");
        return 1;
    }
    return g_failures == 0 ? 0 : 1;
}
