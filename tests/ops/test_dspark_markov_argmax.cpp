#include "ninfer/ops/dspark_markov_argmax.h"
#include "ops/op_tester.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kV = 248320;
constexpr int kR = 256;
constexpr int kK = 7;

std::uint16_t pattern_bf16(std::uint32_t i) {
    const std::uint32_t mixed = i * 1664525u + 1013904223u;
    const float value = -0.25f + static_cast<float>(mixed % 1024u) * (0.5f / 1024.0f);
    return f32_to_bf16(value);
}

std::uint16_t small_pattern_bf16(std::uint32_t i) {
    const std::uint32_t mixed = i * 1103515245u + 12345u;
    const float value = -0.025f + static_cast<float>(mixed % 2048u) * (0.05f / 2048.0f);
    return f32_to_bf16(value);
}

int run_case(std::int32_t batch) {
    const std::size_t logits_count = static_cast<std::size_t>(kV) * kK * batch;
    std::vector<std::uint16_t> logits(logits_count);
    for (std::size_t i = 0; i < logits_count; ++i) { logits[i] = pattern_bf16(i); }
    // One unambiguous winner per column: the Markov bias is bounded by
    // 256 * 0.025^2 = 0.16, so a +8.0 base logit keeps the argmax unique.
    std::vector<std::int32_t> winners(static_cast<std::size_t>(kK) * batch);
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t j = 0; j < kK; ++j) {
            const std::int32_t winner = (12345 + b * 7919 + j * 104729) % kV;
            winners[static_cast<std::size_t>(b) * kK + j] = winner;
            logits[(static_cast<std::size_t>(b) * kK + j) * kV + winner] =
                f32_to_bf16(8.0f + static_cast<float>(j));
        }
    }

    std::vector<std::uint16_t> w1(static_cast<std::size_t>(kV) * kR);
    std::vector<std::uint16_t> w2(static_cast<std::size_t>(kV) * kR);
    for (std::size_t i = 0; i < w1.size(); ++i) { w1[i] = small_pattern_bf16(i + 7); }
    for (std::size_t i = 0; i < w2.size(); ++i) { w2[i] = small_pattern_bf16(i + 101); }

    std::vector<std::int32_t> anchors(batch);
    for (std::int32_t b = 0; b < batch; ++b) { anchors[b] = (1000 + b * 7001) % kV; }

    // Oracle: sequential argmax over base logit + Markov bias.
    std::vector<std::int32_t> expected(static_cast<std::size_t>(kK) * batch);
    for (std::int32_t b = 0; b < batch; ++b) {
        std::int32_t previous = anchors[b];
        for (std::int32_t j = 0; j < kK; ++j) {
            const std::size_t base = (static_cast<std::size_t>(b) * kK + j) * kV;
            std::int32_t best       = 0;
            float best_value        = -1e30f;
            for (std::int32_t v = 0; v < kV; ++v) {
                float value = bf16_to_f32(logits[base + v]);
                const std::size_t p = static_cast<std::size_t>(previous) * kR;
                const std::size_t q = static_cast<std::size_t>(v) * kR;
                for (std::int32_t r = 0; r < kR; ++r) {
                    value += bf16_to_f32(w1[p + r]) * bf16_to_f32(w2[q + r]);
                }
                if (value > best_value) {
                    best       = v;
                    best_value = value;
                }
            }
            expected[static_cast<std::size_t>(b) * kK + j] = best;
            previous                                       = best;
        }
    }

    GuardedDeviceBuffer d_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_w1(w1.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_w2(w2.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_anchors(anchors.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_drafts(expected.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_best_value(anchors.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_best_index(anchors.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_extents(anchors.size() * sizeof(std::int32_t));
    std::vector<std::int32_t> initial_extents(batch, kK);
    d_extents.copy_from_host(initial_extents.data(), initial_extents.size() * sizeof(std::int32_t));
    d_logits.copy_from_host(logits.data(), logits.size() * sizeof(std::uint16_t));
    d_w1.copy_from_host(w1.data(), w1.size() * sizeof(std::uint16_t));
    d_w2.copy_from_host(w2.data(), w2.size() * sizeof(std::uint16_t));
    d_anchors.copy_from_host(anchors.data(), anchors.size() * sizeof(std::int32_t));
    d_drafts.fill(0xcd);
    d_best_value.fill(0xcd);
    d_best_index.fill(0xcd);

    Tensor logits_t(d_logits.data(), DType::BF16, {kV, kK * batch});
    Tensor anchors_t(d_anchors.data(), DType::I32, {batch});
    Tensor drafts_t(d_drafts.data(), DType::I32, {kK * batch});
    Tensor best_value_t(d_best_value.data(), DType::I32, {batch});
    Tensor best_index_t(d_best_index.data(), DType::I32, {batch});
    Tensor extents_t(d_extents.data(), DType::I32, {batch});
    Weight w1_t{};
    w1_t.qdata = d_w1.data();
    w1_t.qtype = QType::BF16_CTRL;
    w1_t.n     = kV;
    w1_t.k     = kR;
    Weight w2_t = w1_t;
    w2_t.qdata  = d_w2.data();

    ops::dspark_markov_argmax(logits_t, w1_t, w2_t, anchors_t, drafts_t, best_value_t,
                              best_index_t, extents_t, 10.0f, nullptr);
    cuda_synchronize();

    const auto actual =
        from_device<std::int32_t>(d_drafts.data(), static_cast<std::size_t>(kK) * batch);
    const auto actual_extents = from_device<std::int32_t>(d_extents.data(), batch);
    if (actual != expected) {
        for (std::size_t i = 0; i < actual.size() && i < 8; ++i) {
            std::cerr << "markov mismatch i=" << i << " actual=" << actual[i]
                      << " expected=" << expected[i] << '\n';
        }
    }
    const std::string label = "dspark_markov_argmax B=" + std::to_string(batch);
    int failures            = verify_exact(label.c_str(), actual, expected);
    failures += verify_exact((label + " extents").c_str(), actual_extents, initial_extents);
    failures += d_drafts.verify_guards((label + " drafts guards").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_case(1);
    failures += run_case(2);
    std::cout << (failures ? "FAIL" : "OK") << " dspark_markov_argmax\n";
    return failures ? 1 : 0;
}
