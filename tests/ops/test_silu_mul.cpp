#include "ninfer/ops/silu_mul.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion silu_mul_bf16_criterion() {
    return {/*absolute*/ 2.0e-5, /*relative*/ 4.05e-3};
}

// The deep-negative tail of SiLU is where the fp32 form decides the result:
// expf(-x) overflows to +inf for x below -88.7228, `x / inf` then returns -0, and
// the true SiLU there is still a normal bf16 (-2.6e-37 at that edge, 22 x the
// bf16 min normal). The gate is a bf16 tensor, so the defect is reachable on the
// bf16 grid: the first grid point below the edge is -89 and the rescued window
// runs to -97.46, while -88.7 quantizes to -88.5 and therefore pins the
// unaffected side of the edge.
//
// The criterion is relative and has NO absolute term on purpose: with the
// suite's usual 2.0e-5 absolute slack a hard zero passes wherever the true value
// is ~1e-37, so it could not see this defect at all. Here the limit is
// 2^-6 * |expected| (two bf16 quanta), which absorbs the kernel's fp32 rounding
// and, because the limit is then exactly zero, demands an exact zero from every
// row whose correctly rounded bf16 result is a zero.
constexpr PointwiseCriterion silu_mul_extreme_negative_criterion() {
    return {/*absolute*/ 0.0, /*relative*/ 1.5625e-2}; // 2^-6, two bf16 quanta
}

// The Op stores bf16, so the oracle is the correctly rounded bf16 result of the
// fp64 ideal evaluated from the represented inputs (the same storage rounding
// the Op applies to its fp32 result; without it a genuine zero would be compared
// against an unrepresentable ideal).
std::vector<double> silu_mul_bf16_oracle(const std::vector<float>& gate,
                                         const std::vector<float>& up) {
    std::vector<double> expected(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const double g     = gate[i];
        const double ideal = (g / (1.0 + std::exp(-g))) * static_cast<double>(up[i]);
        expected[i] = static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(ideal))));
    }
    return expected;
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<double> silu_mul_oracle(const std::vector<float>& gate, const std::vector<float>& up) {
    std::vector<double> expected(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const double g = gate[i];
        expected[i]    = (g / (1.0 + std::exp(-g))) * static_cast<double>(up[i]);
    }
    return expected;
}

int run_contiguous_case(const char* label, std::int32_t rows, std::int32_t columns,
                        std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), up(count);
    fill_uniform(gate, seed, -12.0f, 12.0f);
    fill_uniform(up, seed + 1, -8.0f, 8.0f);
    round_to_bf16(gate);
    round_to_bf16(up);

    const auto expected  = silu_mul_oracle(gate, up);
    const auto gate_bits = encode_bf16(gate);
    const auto up_bits   = encode_bf16(up);
    GuardedDeviceBuffer device_gate(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_up(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_up.copy_from_host(up_bits.data(), device_up.bytes());

    Tensor gate_tensor(device_gate.data(), DType::BF16, {rows, columns});
    Tensor up_tensor(device_up.data(), DType::BF16, {rows, columns});
    Tensor out_tensor(device_out.data(), DType::BF16, {rows, columns});
    ops::silu_mul(gate_tensor, up_tensor, out_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), count), expected,
                                    silu_mul_bf16_criterion());
    failures += verify_exact("silu_mul gate unchanged",
                             from_device<std::uint16_t>(device_gate.data(), count), gate_bits);
    failures += verify_exact("silu_mul up unchanged",
                             from_device<std::uint16_t>(device_up.data(), count), up_bits);
    failures += device_gate.verify_guards("silu_mul gate");
    failures += device_up.verify_guards("silu_mul up");
    failures += device_out.verify_guards("silu_mul out");
    return failures;
}

int run_strided_gate_up_case() {
    constexpr std::int32_t rows    = 17408;
    constexpr std::int32_t columns = 17;
    const std::size_t count        = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), up(count);
    fill_uniform(gate, 301u, -12.0f, 12.0f);
    fill_uniform(up, 302u, -8.0f, 8.0f);
    round_to_bf16(gate);
    round_to_bf16(up);

    std::vector<float> gate_up(2 * count);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t source = static_cast<std::size_t>(column) * rows;
        const std::size_t target = static_cast<std::size_t>(column) * 2 * rows;
        std::copy_n(gate.data() + source, rows, gate_up.data() + target);
        std::copy_n(up.data() + source, rows, gate_up.data() + target + rows);
    }

    const auto expected     = silu_mul_oracle(gate, up);
    const auto gate_up_bits = encode_bf16(gate_up);
    GuardedDeviceBuffer device_gate_up(gate_up_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    device_gate_up.copy_from_host(gate_up_bits.data(), device_gate_up.bytes());

    Tensor gate_up_tensor(device_gate_up.data(), DType::BF16, {2 * rows, columns});
    Tensor gate_tensor = gate_up_tensor.slice(0, 0, rows);
    Tensor up_tensor   = gate_up_tensor.slice(0, rows, rows);
    Tensor out_tensor(device_out.data(), DType::BF16, {rows, columns});
    ops::silu_mul(gate_tensor, up_tensor, out_tensor, nullptr);
    cuda_synchronize();

    int failures =
        verify_pointwise("silu_mul strided gate/up", from_device_bf16(device_out.data(), count),
                         expected, silu_mul_bf16_criterion());
    failures += verify_exact("silu_mul strided inputs unchanged",
                             from_device<std::uint16_t>(device_gate_up.data(), gate_up_bits.size()),
                             gate_up_bits);
    failures += device_gate_up.verify_guards("silu_mul strided inputs");
    failures += device_out.verify_guards("silu_mul strided out");
    return failures;
}

int run_edge_case() {
    std::vector<float> gate{-30.0f, -8.0f, -1.0f, 0.0f, 1.0f, 8.0f, 30.0f};
    std::vector<float> up{2.0f, -3.0f, 0.5f, -4.0f, 5.0f, -6.0f, 7.0f};
    round_to_bf16(gate);
    round_to_bf16(up);

    const auto expected  = silu_mul_oracle(gate, up);
    const auto gate_bits = encode_bf16(gate);
    const auto up_bits   = encode_bf16(up);
    GuardedDeviceBuffer device_gate(gate_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_up(up_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(expected.size() * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_up.copy_from_host(up_bits.data(), device_up.bytes());

    Tensor gate_tensor(device_gate.data(), DType::BF16, {static_cast<int>(gate.size())});
    Tensor up_tensor(device_up.data(), DType::BF16, {static_cast<int>(up.size())});
    Tensor out_tensor(device_out.data(), DType::BF16, {static_cast<int>(expected.size())});
    ops::silu_mul(gate_tensor, up_tensor, out_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise("silu_mul edge values",
                                    from_device_bf16(device_out.data(), expected.size()), expected,
                                    silu_mul_bf16_criterion());
    failures += device_gate.verify_guards("silu_mul edge gate");
    failures += device_up.verify_guards("silu_mul edge up");
    failures += device_out.verify_guards("silu_mul edge out");
    return failures;
}

// Regression case for the S51 SiLU fix (src/ops/common/math.cuh folded the
// exponential onto the side that cannot overflow). It probes the whole bf16
// negative tail of the gate: the pre-fix form returned +-0 for every bf16 input
// at or below -89, while the true SiLU is still a representable bf16 down to -97.
int run_extreme_negative_case() {
    std::vector<float> gate, up;
    const auto add_row = [&](float g, float u) {
        gate.push_back(g);
        up.push_back(u);
    };

    add_row(-88.7f, 1.0f);   // mandated; bf16 -88.5, above the edge: must not change
    add_row(-89.0f, 1.0f);   // first bf16 grid point below the fp32 edge (-88.7228)
    add_row(-90.0f, 2.0f);   // rescued window, non-unit up
    add_row(-92.0f, -0.5f);  // rescued window, negative up (sign flip)
    add_row(-95.0f, 1.0f);   // rescued window, deep
    add_row(-97.0f, 1.0f);   // 0.46 above the window bottom (-97.4611588)
    add_row(-100.0f, 1.0f);  // mandated; bf16 ideal is zero (3.7e-42 < half a subnormal)
    add_row(-1000.0f, 1.0f); // mandated; expf underflows, bf16 ideal is zero
    add_row(3.0f, -2.0f);    // control: unaffected side
    add_row(0.0f, 0.5f);     // control: exact zero
    // Ladder over the bf16 grid: -(88.5 + 0.5k) for k in [0, 64) is exactly
    // representable in bf16 (ulp is 0.5 below 128), so it covers -88.5 .. -120.0:
    // first the rescued window, then the range where the true result underflows
    // bf16 (those rows must come back as an exact zero, not as a fabricated one).
    for (int k = 0; k < 64; ++k) { add_row(-(88.5f + 0.5f * static_cast<float>(k)), 1.0f); }

    round_to_bf16(gate);
    round_to_bf16(up);
    const auto expected  = silu_mul_bf16_oracle(gate, up);
    const auto gate_bits = encode_bf16(gate);
    const auto up_bits   = encode_bf16(up);
    GuardedDeviceBuffer device_gate(gate_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_up(up_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(expected.size() * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_up.copy_from_host(up_bits.data(), device_up.bytes());

    Tensor gate_tensor(device_gate.data(), DType::BF16, {static_cast<int>(gate.size())});
    Tensor up_tensor(device_up.data(), DType::BF16, {static_cast<int>(up.size())});
    Tensor out_tensor(device_out.data(), DType::BF16, {static_cast<int>(expected.size())});
    ops::silu_mul(gate_tensor, up_tensor, out_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise("silu_mul extreme negatives",
                                    from_device_bf16(device_out.data(), expected.size()), expected,
                                    silu_mul_extreme_negative_criterion());
    failures += device_gate.verify_guards("silu_mul extreme gate");
    failures += device_up.verify_guards("silu_mul extreme up");
    failures += device_out.verify_guards("silu_mul extreme out");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_contiguous_case("silu_mul [17408,1]", 17408, 1, 101u);
    failures += run_contiguous_case("silu_mul [17408,48]", 17408, 48, 102u);
    failures += run_strided_gate_up_case();
    failures += run_edge_case();
    failures += run_extreme_negative_case();
    std::cout << (failures ? "FAIL" : "OK") << " silu_mul\n";
    return failures ? 1 : 0;
}
