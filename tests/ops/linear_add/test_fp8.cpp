#include "ninfer/ops/linear_add.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr double kBf16UnitRoundoff = 1.0 / 256.0;
constexpr ReductionCriterion kA16Tolerance{
    kBf16UnitRoundoff,
    kBf16UnitRoundoff,
    2.0 * kBf16UnitRoundoff,
};
constexpr ReductionCriterion kA8Tolerance{0.04, kBf16UnitRoundoff, 0.06};

// ⚠️ ACCEPT2 — THE CONTRACT THIS SUITE ASSERTS, named rather than transcribed. The verify tier must
// be the batch-1 decode's tier, i.e. A16, for EVERY width a verify round can present. `kVerifyWidthCeiling`
// is the widest a chain-verify round reaches today (kMtpDecodeMaximumDrafts + 1 = 16);
// `kWideVerifyWidthCeiling` is the widened domain the ngram-draft line introduces (it widens verify to
// 1..63). Above that domain only prefill and the draft model reach, where A8 is legal and measured.
constexpr std::int32_t kVerifyWidthCeiling     = 16;
constexpr std::int32_t kWideVerifyWidthCeiling = 64;

struct Invocation {
    std::int32_t tokens;
    ops::LinearPolicy policy;
};

std::vector<std::int32_t> sampled_indices(std::int32_t extent) {
    std::vector<std::int32_t> result;
    constexpr std::int32_t kSamples = 32;
    for (std::int32_t sample = 0; sample < kSamples; ++sample) {
        const std::int32_t index = static_cast<std::int32_t>(
            (static_cast<std::int64_t>(extent - 1) * sample) / (kSamples - 1));
        if (index >= 0 && index < extent &&
            std::find(result.begin(), result.end(), index) == result.end()) {
            result.push_back(index);
        }
    }
    return result;
}

std::vector<std::uint16_t> make_activation(std::int32_t rows, std::int32_t tokens,
                                           std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < rows; ++row) {
            std::uint32_t value = seed ^ (static_cast<std::uint32_t>(row) * 0x9e3779b9U) ^
                                  (static_cast<std::uint32_t>(token) * 0x85ebca6bU);
            value ^= value >> 16;
            value *= 0x7feb352dU;
            value ^= value >> 15;
            const float represented =
                static_cast<float>(static_cast<int>(value & 0xffU) - 128) * (1.0F / 256.0F);
            result[static_cast<std::size_t>(token) * rows + row] = f32_to_bf16(represented);
        }
    }
    return result;
}

std::vector<std::uint16_t> make_residual(std::int32_t rows, std::int32_t tokens,
                                         std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::uint32_t coordinate = static_cast<std::uint32_t>(row) * 23U +
                                             static_cast<std::uint32_t>(token) * 41U + seed * 7U;
            const float represented =
                static_cast<float>(static_cast<int>(coordinate & 0xffU) - 128) * (1.0F / 128.0F);
            result[static_cast<std::size_t>(token) * rows + row] = f32_to_bf16(represented);
        }
    }
    return result;
}

int verify_preserved(const GuardedDeviceBuffer& device, std::span<const std::uint8_t> expected,
                     std::string_view label) {
    std::vector<std::uint8_t> actual(expected.size());
    device.copy_to_host(actual.data(), actual.size());
    if (std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) { return 0; }
    std::cerr << label << ": payload was modified\n";
    return 1;
}

int run_shape(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    // The sampled widths are the CONTRACT's boundaries, not the predicate's old crossover:
    //   1/2  = the batch-1 decode;  16 = the current verify ceiling;  64 = the widened
    //   domain's last A16 width;  65 = the first width where A8 is legal again;  1024 = a
    //   prefill-scale width.
    const std::array invocations{
        Invocation{1, ops::LinearPolicy::A16Only},
        Invocation{2, ops::LinearPolicy::A16Only},
        Invocation{26, ops::LinearPolicy::A16Only},
        Invocation{kVerifyWidthCeiling, ops::LinearPolicy::AllowA8},
        Invocation{kWideVerifyWidthCeiling, ops::LinearPolicy::AllowA8},
        Invocation{kWideVerifyWidthCeiling + 1, ops::LinearPolicy::AllowA8},
        Invocation{1024, ops::LinearPolicy::AllowA8},
    };
    constexpr std::int32_t kMaximumTokens = 1024;
    quantized_weight::PackedWeight host_weight =
        quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16S, n, k, seed);
    const std::vector<std::int32_t> rows = sampled_indices(n);
    const std::vector<float> materialized_weight =
        quantized_weight::materialize_rows_fp32(host_weight, rows);
    const std::vector<std::uint16_t> activation = make_activation(k, kMaximumTokens, seed + 1U);
    const std::vector<std::uint16_t> initial_residual = make_residual(n, kMaximumTokens, seed + 2U);

    GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation.data(), device_activation.bytes());
    GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());

    int failures = 0;
    for (const Invocation invocation : invocations) {
        const std::size_t output_words = static_cast<std::size_t>(n) * invocation.tokens;
        GuardedDeviceBuffer output(output_words * sizeof(std::uint16_t));
        output.copy_from_host(initial_residual.data(), output.bytes());
        Tensor x(device_activation.data(), DType::BF16, {k, invocation.tokens});
        Tensor residual(output.data(), DType::BF16, {n, invocation.tokens});
        const std::size_t capacity = ops::linear_add_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, n, k, invocation.policy, invocation.tokens,
            invocation.tokens);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::linear_add(x, weight, residual, invocation.policy, workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "synchronize FP8 linear_add");

        // The tier is read off the CONTRACT (the named domain), never re-derived from the
        // production predicate: a suite that recomputes the predicate ratifies the predicate.
        const bool a8 = invocation.policy == ops::LinearPolicy::AllowA8 &&
                        invocation.tokens > kWideVerifyWidthCeiling;
        const std::string label = "FP8 linear_add [" + std::to_string(n) + "," + std::to_string(k) +
                                  "] " + (a8 ? "A8" : "A16") +
                                  " T=" + std::to_string(invocation.tokens);
        if (workspace.used() != 0 || workspace.peak_used() != capacity) {
            std::cerr << label << ": workspace query/execution high-water mismatch\n";
            ++failures;
        }
        failures += output.verify_guards(label);

        std::vector<std::uint16_t> actual_bits(output_words);
        output.copy_to_host(actual_bits.data(), output.bytes());
        const std::vector<std::int32_t> tokens = sampled_indices(invocation.tokens);
        std::vector<double> actual;
        std::vector<double> expected;
        actual.reserve(rows.size() * tokens.size());
        expected.reserve(rows.size() * tokens.size());
        for (std::size_t sampled_row = 0; sampled_row < rows.size(); ++sampled_row) {
            const std::int32_t row = rows[sampled_row];
            const float* weight_row =
                materialized_weight.data() + sampled_row * static_cast<std::size_t>(k);
            for (const std::int32_t token : tokens) {
                double sum = 0.0;
                const std::uint16_t* activation_row =
                    activation.data() + static_cast<std::size_t>(token) * k;
                for (std::int32_t column = 0; column < k; ++column) {
                    sum += static_cast<double>(weight_row[column]) *
                           static_cast<double>(bf16_to_f32(activation_row[column]));
                }
                const std::size_t index = static_cast<std::size_t>(token) * n + row;
                actual.push_back(static_cast<double>(bf16_to_f32(actual_bits[index])));
                expected.push_back(sum + static_cast<double>(bf16_to_f32(initial_residual[index])));
            }
        }
        failures += verify_reduction(label, actual, expected, a8 ? kA8Tolerance : kA16Tolerance);
    }

    failures += device_activation.verify_guards("FP8 linear_add activation");
    failures += device_weight.verify_guards("FP8 linear_add weight");
    failures += verify_preserved(
        device_activation,
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(activation.data()),
                                      activation.size() * sizeof(std::uint16_t)),
        "FP8 linear_add activation");
    failures += verify_preserved(device_weight, host_weight.payload, "FP8 linear_add weight");

    const std::size_t a16_interval = ops::linear_add_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, n, k, ops::LinearPolicy::A16Only, 1, 2048);
    // THE VERIFY DOMAIN, asserted as a DOMAIN. A16 needs no storage and A8 does, so
    // "capacity == 0 for every width the verify can present" IS "the verify takes the
    // decode's own tier" -- the same instrument FIX-C2's run_fp8_verify_domain_contract uses
    // for the sibling projection. Every width in [1, kWideVerifyWidthCeiling] must report 0
    // under the permissive policy, and the first width ABOVE the domain must not.
    std::size_t widest_domain_nonzero = 0;
    for (std::int32_t tokens = 1; tokens <= kWideVerifyWidthCeiling; ++tokens) {
        widest_domain_nonzero |=
            ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S, n, k,
                                                     ops::LinearPolicy::AllowA8, tokens,
                                                     tokens);
    }
    const std::size_t just_above = ops::linear_add_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, n, k, ops::LinearPolicy::AllowA8,
        kWideVerifyWidthCeiling + 1, kWideVerifyWidthCeiling + 1);
    const std::size_t through_1024 = ops::linear_add_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, n, k, ops::LinearPolicy::AllowA8, 1, 1024);
    const std::size_t exact_1024 = ops::linear_add_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, n, k, ops::LinearPolicy::AllowA8, 1024, 1024);
    if (a16_interval != 0 || widest_domain_nonzero != 0 || just_above == 0 ||
        through_1024 != exact_1024) {
        std::cerr << "FP8 linear_add [" << n << ',' << k
                  << "]: workspace interval contract mismatch: the A8 scratch must be zero "
                     "for every width in [1," << kWideVerifyWidthCeiling
                  << "] and non-zero at "
                  << (kWideVerifyWidthCeiling + 1) << " (a16=" << a16_interval
                  << " domain=" << widest_domain_nonzero << " just_above=" << just_above
                  << ")\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    // The crossover constants are gone from the call: after the fix the first A8 width is
    // kWideVerifyWidthCeiling + 1 for BOTH registered geometries, which is what the interval
    // assertion above measures directly.
    failures += run_shape(5120, 6144, 861U);
    failures += run_shape(5120, 17408, 863U);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " FP8 linear_add\n";
    return failures == 0 ? 0 : 1;
}
