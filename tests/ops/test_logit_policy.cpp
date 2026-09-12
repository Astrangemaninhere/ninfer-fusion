// ninfer::ops - logit_policy numerical qualification. Oracle evaluates the
// documented policy in FP64 from the represented BF16 inputs; output storage
// rounding belongs to the Op's numerical criterion, not the oracle.
#include "ninfer/ops/logit_policy.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion logit_policy_bf16_criterion() {
    return {/*absolute*/ 2.0e-5, /*relative*/ 4.5e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<double> logit_policy_oracle(const std::vector<float>& x, float multiplier,
                                        float cap) {
    std::vector<double> expected(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        double v = static_cast<double>(x[i]) * static_cast<double>(multiplier);
        if (cap > 0.0f) { v = static_cast<double>(cap) * std::tanh(v / static_cast<double>(cap)); }
        expected[i] = v;
    }
    return expected;
}

int run_case(const char* label, std::int32_t rows, std::int32_t columns, std::uint32_t seed,
             float multiplier, float cap, float lo, float hi) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> x(count);
    fill_uniform(x, seed, lo, hi);
    round_to_bf16(x);

    const auto expected  = logit_policy_oracle(x, multiplier, cap);
    const auto x_bits    = encode_bf16(x);
    GuardedDeviceBuffer device_x(count * sizeof(std::uint16_t));
    device_x.copy_from_host(x_bits.data(), device_x.bytes());

    Tensor x_tensor(device_x.data(), DType::BF16, {rows, columns});
    ops::logit_policy(x_tensor, multiplier, cap, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_x.data(), count), expected,
                                    logit_policy_bf16_criterion());
    failures += device_x.verify_guards("logit_policy x");
    return failures;
}

int run_identity_case() {
    // Default architecture policy (multiplier = 1, cap = 0) must be a
    // bit-exact identity: qwen-family variants compile this call away, but a
    // manual invocation still has to preserve every value.
    const std::size_t count = 8192;
    std::vector<float> x(count);
    fill_uniform(x, 7u, -12.0f, 12.0f);
    round_to_bf16(x);

    const auto x_bits = encode_bf16(x);
    GuardedDeviceBuffer device_x(count * sizeof(std::uint16_t));
    device_x.copy_from_host(x_bits.data(), device_x.bytes());

    Tensor x_tensor(device_x.data(), DType::BF16, {static_cast<std::int32_t>(count)});
    ops::logit_policy(x_tensor, 1.0f, 0.0f, nullptr);
    cuda_synchronize();

    int failures = verify_exact("logit_policy identity",
                                from_device<std::uint16_t>(device_x.data(), count), x_bits);
    failures += device_x.verify_guards("logit_policy identity x");
    return failures;
}

int run_edge_case() {
    // Saturation: cap*tanh saturates toward +/-cap; include exact +/-cap
    // regime values and a negative-cap (disabled tanh) sanity case.
    std::vector<float> x{-40.0f, -15.0f, -3.0f, 0.0f, 0.5f, 3.0f, 15.0f, 40.0f};
    round_to_bf16(x);

    {
        const float mult = 0.196116f, cap = 20.0f;  // Muse-Glimmer policy
        const auto expected  = logit_policy_oracle(x, mult, cap);
        const auto x_bits    = encode_bf16(x);
        GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
        device_x.copy_from_host(x_bits.data(), device_x.bytes());

        Tensor x_tensor(device_x.data(), DType::BF16, {static_cast<int>(x.size())});
        ops::logit_policy(x_tensor, mult, cap, nullptr);
        cuda_synchronize();

        int failures = verify_pointwise("logit_policy edge softcap",
                                        from_device_bf16(device_x.data(), x.size()), expected,
                                        logit_policy_bf16_criterion());
        failures += device_x.verify_guards("logit_policy edge x");
        return failures;
    }
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    // Muse-Glimmer policy (mult = 0.196116, cap = 20) over realistic logits.
    failures += run_case("logit_policy muse [151936,1]", 151936, 1, 101u, 0.196116f, 20.0f,
                         -30.0f, 30.0f);
    failures += run_case("logit_policy muse [151936,8]", 151936, 8, 102u, 0.196116f, 20.0f,
                         -30.0f, 30.0f);
    // Odd tails and vectorization boundaries.
    failures += run_case("logit_policy [4096,17]", 4096, 17, 201u, 0.7f, 2.5f, -10.0f, 10.0f);
    failures += run_case("logit_policy [4096,128]", 4096, 128, 301u, 1.25f, 4.0f, -6.0f, 6.0f);
    // Multiplier-only (cap disabled) route.
    failures += run_case("logit_policy mult-only [1000,3]", 1000, 3, 401u, 0.196116f, -1.0f,
                         -8.0f, 8.0f);
    failures += run_identity_case();
    failures += run_edge_case();
    std::cout << (failures ? "FAIL" : "OK") << " logit_policy\n";
    return failures ? 1 : 0;
}
