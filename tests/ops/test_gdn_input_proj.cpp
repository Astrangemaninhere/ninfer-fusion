#include "ninfer/ops/gdn_input_proj.h"

#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

// This criterion belongs to the complete A16 GDN-input-projection Op.
constexpr ReductionCriterion kGdnInputProjA16Tolerance{3.0e-3, 4.0e-3, 3.5e-3};
constexpr ReductionCriterion kFp8GdnInputProjA16Tolerance{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
constexpr ReductionCriterion kFp8GdnInputProjA8Tolerance{0.04, 1.0 / 256.0, 0.06};
constexpr ReductionCriterion kGdnInputProjA4Tolerance{0.16, 4.0e-3, 0.16};
constexpr std::int32_t kA8SampleRows = 31;

// THE CONTRACT, written as a contract and not as a second copy of the production
// predicate.  The batch-1 chain-verify width of an MTP round is
// `--draft-tokens + 1`, the CLI caps `--draft-tokens` at 15, so the verify domain is
// exactly [1, kVerifyWidthCeiling].  `--greedy --spec mtp --draft-tokens k` is
// required to reproduce `--spec none` byte for byte, and the batch-1 decode reaches
// this projection at T = 1, where the route is the BF16-activation (A16) tier.
// Therefore EVERY width in the verify domain must take A16 and must come out
// bit-for-bit identical to the T = 1 decode.  The FP8 A8 tier quantises the
// ACTIVATIONS, so it is not a re-ordering of the same arithmetic but a different
// one -- it moved the GDN input by whole percent on probed channels of the very
// first row -- and it may only be selected OUTSIDE this domain, where only prefill
// runs.
//
// The previous revision of this file transcribed the production crossover
// (`policy == AllowA8 && tokens >= 8`) and then graded the A8 arm against the loose
// A8 tolerance it had itself chosen for that boundary, so the suite RATIFIED the
// width-dependent-precision defect instead of catching it.
constexpr std::int32_t kVerifyWidthCeiling = 16;

int verify_output_range(std::string_view label, const GuardedBf16Tensor& output,
                        std::int32_t full_rows, std::int32_t output_row_offset,
                        std::int32_t output_rows, const quantized_weight::PackedWeight& weight,
                        std::int32_t weight_row_offset, const std::vector<float>& activation,
                        std::int32_t hidden, std::int32_t tokens) {
    const std::vector<double> actual =
        gather_rows(output.values(), full_rows, output_row_offset, output_rows, tokens);
    const std::vector<double> expected =
        projection_oracle(weight, weight_row_offset, output_rows, activation, hidden, tokens);
    return compare(label, actual, expected, kGdnInputProjA16Tolerance);
}

int run_q4_q5_case(DevicePackedWeight& query_key, DevicePackedWeight& value_z_weight,
                   std::int32_t tokens, std::int32_t hidden, std::int32_t qk_rows,
                   std::int32_t value_rows, std::int32_t z_rows) {
    const std::int32_t kHidden          = hidden;
    const std::int32_t kQkRows          = qk_rows;
    const std::int32_t kValueRows       = value_rows;
    const std::int32_t kZRows           = z_rows;
    const std::int32_t kRows            = kQkRows + kValueRows;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 401U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor output   = qkv.tensor();
    Tensor z_output = z.tensor();
    ops::gdn_input_proj(x, query_key.view(), value_z_weight.view(), output, z_output, nullptr);
    cuda_synchronize();

    const std::string suffix = " Q4/Q5 A16 T=" + std::to_string(tokens);
    int failures             = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    failures += verify_output_range("gdn qk" + suffix, qkv, kRows, 0, kQkRows, query_key.host, 0,
                                    activation, kHidden, tokens);
    failures += verify_output_range("gdn value" + suffix, qkv, kRows, kQkRows, kValueRows,
                                    value_z_weight.host, 0, activation, kHidden, tokens);
    failures += verify_output_range("gdn z" + suffix, z, kZRows, 0, kZRows, value_z_weight.host,
                                    kValueRows, activation, kHidden, tokens);
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += query_key.verify_preserved("gdn query/key weight" + suffix);
    failures += value_z_weight.verify_preserved("gdn value/z weight" + suffix);
    return failures;
}

// {hidden, qk_rows, value_rows, z_rows}. query_key is [qk_rows, hidden], value_z is
// [value_rows + z_rows, hidden] and qkv is [qk_rows + value_rows, tokens]. The second row is the
// 4096-wide GDN-hybrid text stack, whose value/z split differs from the first row's -- which is
// exactly the transcription this case function used to carry.
int run_q4_q5() {
    constexpr std::int32_t kGeometries[][4]{
        {5120, 4096, 6144, 6144},
        {4096, 4096, 4096, 4096},
    };
    int failures = 0;
    for (const auto& geometry : kGeometries) {
        DevicePackedWeight query_key(quantized_weight::make_patterned_weight(
            QType::Q4G64_F16S, geometry[1], geometry[0], 409U));
        DevicePackedWeight value_z_weight(quantized_weight::make_patterned_weight(
            QType::Q5G64_F16S, geometry[2] + geometry[3], geometry[0], 419U));
        for (const std::int32_t tokens : {1, 2, 16, 17}) {
            failures += run_q4_q5_case(query_key, value_z_weight, tokens, geometry[0], geometry[1],
                                       geometry[2], geometry[3]);
        }
    }
    return failures;
}

int run_w8_case(DevicePackedWeight& parent, std::int32_t tokens) {
    constexpr std::int32_t kHidden      = 2048;
    constexpr std::int32_t kQkvRows     = 8192;
    constexpr std::int32_t kZRows       = 4096;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 501U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kQkvRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor qkv_output = qkv.tensor();
    Tensor z_output   = z.tensor();
    ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, nullptr);
    cuda_synchronize();

    const std::string suffix = " W8 A16 T=" + std::to_string(tokens);
    int failures             = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    failures += verify_output_range("gdn qkv" + suffix, qkv, kQkvRows, 0, kQkvRows, parent.host, 0,
                                    activation, kHidden, tokens);
    failures += verify_output_range("gdn z" + suffix, z, kZRows, 0, kZRows, parent.host, kQkvRows,
                                    activation, kHidden, tokens);
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("gdn parent weight" + suffix);
    return failures;
}

int run_w8() {
    constexpr std::int32_t kHidden = 2048;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::W8G32_F16S, 12288, kHidden, 503U));
    int failures = 0;
    for (const std::int32_t tokens : {1, 2, 97}) { failures += run_w8_case(parent, tokens); }
    return failures;
}

int verify_output_range_sampled(std::string_view label, const GuardedBf16Tensor& output,
                                std::int32_t full_rows, std::int32_t output_row_offset,
                                std::int32_t output_rows,
                                const quantized_weight::PackedWeight& weight,
                                std::int32_t weight_row_offset,
                                const std::vector<float>& activation, std::int32_t hidden,
                                std::int32_t tokens, const ReductionCriterion& criterion,
                                std::int32_t sample_count = 7) {
    const std::vector<double> values     = output.values();
    const std::vector<std::int32_t> rows = sampled_rows(output_rows, sample_count);
    std::vector<std::int32_t> selected_tokens;
    for (const std::int32_t token :
         {0, 1, tokens / 4, tokens / 2, (3 * tokens) / 4, tokens - 2, tokens - 1}) {
        if (token >= 0 && token < tokens &&
            std::find(selected_tokens.begin(), selected_tokens.end(), token) ==
                selected_tokens.end()) {
            selected_tokens.push_back(token);
        }
    }
    std::vector<double> actual;
    std::vector<double> expected;
    actual.reserve(rows.size() * selected_tokens.size());
    expected.reserve(rows.size() * selected_tokens.size());
    for (const std::int32_t local_row : rows) {
        const std::int32_t output_row = output_row_offset + local_row;
        const std::int32_t weight_row = weight_row_offset + local_row;
        for (const std::int32_t token : selected_tokens) {
            actual.push_back(values[static_cast<std::size_t>(token) * full_rows + output_row]);
            expected.push_back(quantized_weight::dot_fp64(
                weight, weight_row, activation.data() + static_cast<std::size_t>(token) * hidden,
                hidden));
        }
    }
    return compare(label, actual, expected, criterion);
}

int run_nvfp4_case(DevicePackedWeight& parent, std::int32_t tokens, ops::LinearPolicy policy) {
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kQkvRows     = 10240;
    constexpr std::int32_t kZRows       = 6144;
    constexpr std::int32_t kRows        = kQkvRows + kZRows;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 601U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kQkvRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x                   = Tensor(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor qkv_output          = qkv.tensor();
    Tensor z_output            = z.tensor();
    const std::size_t capacity = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::NVFP4, kRows, kHidden, policy, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, policy, workspace, nullptr);
    cuda_synchronize();

    const bool a4                       = policy == ops::LinearPolicy::AllowA4;
    const ReductionCriterion& criterion = a4 ? kGdnInputProjA4Tolerance : kGdnInputProjA16Tolerance;
    const std::string suffix =
        std::string(" NVFP4 ") + (a4 ? "A4" : "A16") + " T=" + std::to_string(tokens);
    int failures = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    failures += verify_output_range_sampled("gdn query" + suffix, qkv, kQkvRows, 0, 2048,
                                            parent.host, 0, activation, kHidden, tokens, criterion);
    failures +=
        verify_output_range_sampled("gdn key" + suffix, qkv, kQkvRows, 2048, 2048, parent.host,
                                    2048, activation, kHidden, tokens, criterion);
    failures +=
        verify_output_range_sampled("gdn value" + suffix, qkv, kQkvRows, 4096, 6144, parent.host,
                                    4096, activation, kHidden, tokens, criterion);
    failures += verify_output_range_sampled("gdn z" + suffix, z, kZRows, 0, kZRows, parent.host,
                                            kQkvRows, activation, kHidden, tokens, criterion);
    if (workspace.peak_used() != capacity) {
        std::cerr << "gdn workspace" << suffix << ": query/execution high-water mismatch\n";
        ++failures;
    }
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("gdn parent weight" + suffix);
    return failures;
}

int run_nvfp4() {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kRows   = 16384;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kRows, kHidden, 607U, options));
    int failures = 0;
    failures += run_nvfp4_case(parent, 1, ops::LinearPolicy::A16Only);
    failures += run_nvfp4_case(parent, 4, ops::LinearPolicy::A16Only);
    failures += run_nvfp4_case(parent, 1, ops::LinearPolicy::AllowA4);
    failures += run_nvfp4_case(parent, 2, ops::LinearPolicy::AllowA4);
    failures += run_nvfp4_case(parent, 17, ops::LinearPolicy::AllowA4);
    failures += run_nvfp4_case(parent, 1024, ops::LinearPolicy::AllowA4);
    return failures;
}

int run_fp8_case(DevicePackedWeight& parent, std::int32_t tokens, ops::LinearPolicy policy,
                 bool convenience = false) {
    constexpr std::int32_t kHidden  = 5120;
    constexpr std::int32_t kQkvRows = 10240;
    constexpr std::int32_t kZRows   = 6144;
    constexpr std::int32_t kRows    = kQkvRows + kZRows;
    const std::vector<float> activation =
        make_bf16_activation(kHidden, tokens, 617U + static_cast<std::uint32_t>(tokens));
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kQkvRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x                   = Tensor(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor qkv_output          = qkv.tensor();
    Tensor z_output            = z.tensor();
    const std::size_t capacity = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, policy, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    if (convenience) {
        ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, nullptr);
    } else {
        ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, policy, workspace, nullptr);
    }
    cuda_synchronize();

    // The tier is read off the CONTRACT (the verify domain ends at
    // kVerifyWidthCeiling), never off a second copy of the production predicate: a
    // width inside the verify domain reduces like the batch-1 decode and is therefore
    // graded against the tight A16 criterion, and only a width beyond the domain may
    // use the loose A8 one.
    const bool a8 = policy == ops::LinearPolicy::AllowA8 && tokens > kVerifyWidthCeiling;
    const ReductionCriterion& criterion =
        a8 ? kFp8GdnInputProjA8Tolerance : kFp8GdnInputProjA16Tolerance;
    const std::int32_t sample_count = a8 ? kA8SampleRows : 7;
    const std::string suffix =
        std::string(" FP8 ") + (a8 ? "A8" : "A16") + " T=" + std::to_string(tokens);
    int failures = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    failures +=
        verify_output_range_sampled("gdn query" + suffix, qkv, kQkvRows, 0, 2048, parent.host, 0,
                                    activation, kHidden, tokens, criterion, sample_count);
    failures +=
        verify_output_range_sampled("gdn key" + suffix, qkv, kQkvRows, 2048, 2048, parent.host,
                                    2048, activation, kHidden, tokens, criterion, sample_count);
    failures +=
        verify_output_range_sampled("gdn value" + suffix, qkv, kQkvRows, 4096, 6144, parent.host,
                                    4096, activation, kHidden, tokens, criterion, sample_count);
    failures +=
        verify_output_range_sampled("gdn z" + suffix, z, kZRows, 0, kZRows, parent.host, kQkvRows,
                                    activation, kHidden, tokens, criterion, sample_count);
    if (workspace.peak_used() != capacity) {
        std::cerr << "gdn workspace" << suffix << ": query/execution high-water mismatch\n";
        ++failures;
    }
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("gdn parent weight" + suffix);
    return failures;
}

// Runs one full FP8 GDN input projection and returns the RAW BF16 PAYLOAD of both
// outputs, so comparing two runs is a bitwise comparison and not a tolerance verdict.
struct Fp8RawOutput {
    std::vector<std::uint16_t> qkv;
    std::vector<std::uint16_t> z;
};

Fp8RawOutput run_fp8_raw(DevicePackedWeight& parent, std::int32_t tokens,
                         ops::LinearPolicy policy) {
    constexpr std::int32_t kHidden  = 5120;
    constexpr std::int32_t kQkvRows = 10240;
    constexpr std::int32_t kZRows   = 6144;
    constexpr std::int32_t kRows    = kQkvRows + kZRows;
    const std::vector<float> activation =
        make_bf16_activation(kHidden, tokens, 617U + static_cast<std::uint32_t>(tokens));
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kQkvRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x          = Tensor(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor qkv_output = qkv.tensor();
    Tensor z_output   = z.tensor();
    const std::size_t capacity = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, policy, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, policy, workspace, nullptr);
    cuda_synchronize();
    return {qkv.bits(), z.bits()};
}

int compare_bitwise(std::string_view label, const std::vector<std::uint16_t>& actual,
                    const std::vector<std::uint16_t>& expected) {
    if (actual.size() != expected.size()) {
        std::cerr << label << ": payload size mismatch (" << actual.size() << " vs "
                  << expected.size() << ")\n";
        return 1;
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] == expected[index]) { continue; }
        const double lhs = static_cast<double>(bf16_to_f32(actual[index]));
        const double rhs = static_cast<double>(bf16_to_f32(expected[index]));
        const double den = std::max(std::abs(rhs), std::numeric_limits<double>::min());
        std::cerr << label << ": output element " << index
                  << " is not the batch-1 decode's value: allow-a8=0x" << std::hex << actual[index]
                  << std::dec << " (" << lhs << ") vs a16=0x" << std::hex << expected[index]
                  << std::dec << " (" << rhs << "), relative " << std::abs(lhs - rhs) / den << "\n";
        return 1;
    }
    return 0;
}

// THE CONTRACT TEST.  For every width in the verify domain the A8-permissive policy must
// produce the SAME BITS as the A16-only policy, because the batch-1 decode of the same row
// takes A16 and a verify round has to reproduce that decode exactly.  This asserts the
// contract instead of transcribing the crossover, so it fails on any revision that lets a
// verify width take a different activation precision -- which is what `tokens >= 8 ? A8 :
// A16` did over T = 8..16.
int run_fp8_verify_domain_contract(DevicePackedWeight& parent) {
    int failures = 0;
    for (std::int32_t tokens = 1; tokens <= kVerifyWidthCeiling; ++tokens) {
        const Fp8RawOutput verify = run_fp8_raw(parent, tokens, ops::LinearPolicy::AllowA8);
        const Fp8RawOutput decode = run_fp8_raw(parent, tokens, ops::LinearPolicy::A16Only);
        const std::string suffix = " FP8 verify-domain T=" + std::to_string(tokens);
        failures += compare_bitwise("gdn qkv (allow-a8 vs a16)" + suffix, verify.qkv, decode.qkv);
        failures += compare_bitwise("gdn z (allow-a8 vs a16)" + suffix, verify.z, decode.z);
    }
    return failures;
}

int run_fp8() {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kRows   = 16384;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, 613U));

    int failures          = 0;
    const std::size_t one = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::AllowA8, 1, 1);
    const std::size_t seven = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::AllowA8, 7, 7);
    const std::size_t eight = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::AllowA8, 8, 8);
    const std::size_t forty_eight = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::AllowA8, 48, 48);
    const std::size_t hot_interval = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::AllowA8, 1, 48);
    const std::size_t exact_1024 = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::AllowA8, 1024, 1024);
    const std::size_t a16 = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::A16Only, 1, 2048);
    // The workspace interval contract follows from the SAME verify-domain contract: no width
    // in [1, kVerifyWidthCeiling] may need the A8 scratch (every one of them takes A16), and
    // the A8 tier is legal from kVerifyWidthCeiling + 1 up, so the ceiling shows up here as a
    // 0 / non-0 edge rather than as a hard-coded token count.
    bool verify_domain_needs_a8_scratch = false;
    for (std::int32_t tokens = 1; tokens <= kVerifyWidthCeiling; ++tokens) {
        verify_domain_needs_a8_scratch |= ops::gdn_input_proj_workspace_capacity_bytes(
                                              QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden,
                                              ops::LinearPolicy::AllowA8, tokens, tokens) != 0;
    }
    const std::size_t just_above = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::AllowA8,
        kVerifyWidthCeiling + 1, kVerifyWidthCeiling + 1);
    if (one != 0 || seven != 0 || eight != 0 || verify_domain_needs_a8_scratch ||
        forty_eight == 0 || just_above == 0 || hot_interval != forty_eight ||
        exact_1024 <= forty_eight || a16 != 0) {
        std::cerr << "FP8 gdn input workspace interval contract mismatch: the A8 scratch must be "
                     "zero for every width in the verify domain [1,"
                  << kVerifyWidthCeiling << "] and non-zero from " << (kVerifyWidthCeiling + 1)
                  << " up (one=" << one << " seven=" << seven << " eight=" << eight
                  << " just_above=" << just_above << " forty_eight=" << forty_eight
                  << " hot_interval=" << hot_interval << " exact_1024=" << exact_1024
                  << " a16=" << a16 << ")\n";
        ++failures;
    }

    failures += run_fp8_case(parent, 1, ops::LinearPolicy::A16Only, true);
    failures += run_fp8_case(parent, 2, ops::LinearPolicy::A16Only);
    // A8 is graded only where the CONTRACT allows it to be selected: at and beyond the first
    // width outside the verify domain.  48/65/1024 are prefill widths and keep the looser
    // criterion that belongs to the A8 tier.  17 is the first such width and is the edge of
    // the ceiling, so a regression that moves the ceiling is visible here as well.
    for (const std::int32_t tokens : {1, 2, 7, 8, 17, 48, 65, 1024}) {
        failures += run_fp8_case(parent, tokens, ops::LinearPolicy::AllowA8);
    }
    failures += run_fp8_verify_domain_contract(parent);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_q4_q5();
    failures += run_w8();
    failures += run_nvfp4();
    failures += run_fp8();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_input_proj\n";
    return failures == 0 ? 0 : 1;
}
