#include "ninfer/ops/gdn_input_proj.h"

#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

// Restart equivalence for the fused projection/conv/snapshot Op.
//
// The Op publishes, for every column it consumes, the convolution state that the next column
// starts from. Restarting the same Op from that published slot must therefore reproduce the
// column that followed it, exactly: the published slot is the only channel through which one
// column reaches the next, and the documented window (docs/maintainer/replayssm-gdn.md, section
// 5) is a sequence of BF16-represented columns.
//
// Arm A runs one call over T columns from slot SA.
// Arm B runs the same call, same width, same route, starting from the slot arm A published for
// column j - 1, with column j moved to position 0.
//
// Column 0 of arm B must equal column j of arm A bit for bit.
//
// The z output is the bare projection: it never passes through the convolution and therefore is
// not affected by the carry. Comparing z first establishes that the projection of a given column
// does not depend on the position that column occupies in the call, so any query/key/value
// mismatch that remains is attributable to the convolution state alone.

constexpr std::int32_t kQueryRows = 2048;
constexpr std::int32_t kKeyRows   = 2048;
constexpr std::int32_t kValueRows = 4096;
constexpr std::int32_t kZRows     = 4096;
constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
constexpr std::int32_t kHidden    = 2048;

std::vector<float> make_conv_weight(std::int32_t channels, std::uint32_t seed) {
    std::vector<float> weight(static_cast<std::size_t>(channels) * 4);
    fill_uniform(weight, seed, -0.02F, 0.02F);
    round_to_bf16(weight);
    return weight;
}

std::vector<std::uint16_t> make_state(std::int32_t channels, std::int32_t slots,
                                      std::int32_t initial_slot, std::uint32_t seed) {
    const std::size_t slot_stride = static_cast<std::size_t>(channels) * 3;
    std::vector<std::uint16_t> state(slot_stride * slots, 0U);
    std::vector<float> initial(slot_stride);
    fill_uniform(initial, seed, -0.05F, 0.05F);
    round_to_bf16(initial);
    const std::size_t initial_base = static_cast<std::size_t>(initial_slot) * slot_stride;
    for (std::size_t index = 0; index < initial.size(); ++index) {
        state[initial_base + index] = f32_to_bf16(initial[index]);
    }
    return state;
}

struct Outputs {
    std::vector<std::uint16_t> query;
    std::vector<std::uint16_t> key;
    std::vector<std::uint16_t> value;
    std::vector<std::uint16_t> z;
    std::vector<std::uint16_t> state;
};

// One call of the Op over `tokens` columns of `activation_bits`, starting from `initial_slot`
// and publishing at `snapshot_base`.
Outputs run_call(DevicePackedWeight& parent, const std::vector<std::uint16_t>& activation_bits,
                 const std::vector<std::uint16_t>& conv_weight_bits,
                 const std::vector<std::uint16_t>& state_before, std::int32_t slots,
                 std::int32_t tokens, std::int32_t initial_slot, std::int32_t snapshot_base_slot) {
    const std::vector<std::int32_t> initial_value{initial_slot};
    const std::vector<std::int32_t> snapshot_base_value{snapshot_base_slot};

    DeviceBuffer device_activation    = to_device(activation_bits);
    DeviceBuffer device_conv_weight   = to_device(conv_weight_bits);
    DeviceBuffer device_initial       = to_device(initial_value);
    DeviceBuffer device_snapshot_base = to_device(snapshot_base_value);
    GuardedBf16Tensor state(kChannels * 3, slots);
    state.copy_from_bits(state_before);
    GuardedBf16Tensor query(kQueryRows, tokens);
    GuardedBf16Tensor key(kKeyRows, tokens);
    GuardedBf16Tensor value(kValueRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);

    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor conv(device_conv_weight.p, DType::BF16, {kChannels, 4});
    Tensor conv_state(state.data(), DType::BF16, {kChannels, 3, slots});
    Tensor initial(device_initial.p, DType::I32, {1});
    Tensor snapshot_base(device_snapshot_base.p, DType::I32, {1});
    Tensor q        = query.tensor();
    Tensor k        = key.tensor();
    Tensor v        = value.tensor();
    Tensor z_output = z.tensor();

    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, 1, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(1, workspace_bytes));

    ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, conv_state, Tensor{}, initial,
                                      snapshot_base, q, k, v, z_output, workspace, nullptr);
    cuda_synchronize();

    return Outputs{query.bits(), key.bits(), value.bits(), z.bits(), state.bits()};
}

// Bit-exact comparison of one column of two outputs.
int compare_column(const std::string& label, const std::vector<std::uint16_t>& left,
                   std::int32_t left_column, const std::vector<std::uint16_t>& right,
                   std::int32_t right_column, std::int32_t rows) {
    const std::size_t left_base  = static_cast<std::size_t>(left_column) * rows;
    const std::size_t right_base = static_cast<std::size_t>(right_column) * rows;
    std::int32_t mismatches      = 0;
    std::int32_t first_row       = -1;
    for (std::int32_t row = 0; row < rows; ++row) {
        if (left[left_base + static_cast<std::size_t>(row)] !=
            right[right_base + static_cast<std::size_t>(row)]) {
            if (first_row < 0) { first_row = row; }
            ++mismatches;
        }
    }
    if (mismatches == 0) { return 0; }
    std::cerr << label << ": " << mismatches << " of " << rows << " rows differ, first at row "
              << first_row << " ("
              << bf16_to_f32(left[left_base + static_cast<std::size_t>(first_row)]) << " vs "
              << bf16_to_f32(right[right_base + static_cast<std::size_t>(first_row)]) << ")\n";
    return 1;
}

int compare_slot(const std::string& label, const std::vector<std::uint16_t>& left,
                 std::int32_t left_slot, const std::vector<std::uint16_t>& right,
                 std::int32_t right_slot) {
    const std::size_t stride     = static_cast<std::size_t>(kChannels) * 3;
    const std::size_t left_base  = static_cast<std::size_t>(left_slot) * stride;
    const std::size_t right_base = static_cast<std::size_t>(right_slot) * stride;
    if (std::equal(left.begin() + static_cast<std::ptrdiff_t>(left_base),
                   left.begin() + static_cast<std::ptrdiff_t>(left_base + stride),
                   right.begin() + static_cast<std::ptrdiff_t>(right_base))) {
        return 0;
    }
    std::cerr << label << ": published state slot differs\n";
    return 1;
}

int run_restart_case(DevicePackedWeight& parent, std::int32_t tokens) {
    const std::int32_t base_a = 1;
    const std::int32_t base_b = tokens + 2;
    const std::int32_t slots  = 2 * tokens + 4;
    const std::int32_t slot_a = 2 * tokens + 3;

    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 701U + tokens);
    const std::vector<std::uint16_t> activation_bits  = bf16_bits(activation);
    const std::vector<float> conv_weight              = make_conv_weight(kChannels, 709U);
    const std::vector<std::uint16_t> conv_weight_bits = bf16_bits(conv_weight);
    const std::vector<std::uint16_t> state_before =
        make_state(kChannels, slots, slot_a, 719U + tokens);

    const Outputs arm_a = run_call(parent, activation_bits, conv_weight_bits, state_before, slots,
                                   tokens, slot_a, base_a);

    int failures = 0;
    for (std::int32_t column = 1; column < tokens; ++column) {
        // Column j moved to position 0; the rest of the call is padding that cannot reach it.
        std::vector<std::uint16_t> shifted(activation_bits.size());
        for (std::int32_t local = 0; local < tokens; ++local) {
            const std::int32_t source = std::min(column + local, tokens - 1);
            std::copy_n(activation_bits.begin() + static_cast<std::ptrdiff_t>(source) * kHidden,
                        kHidden, shifted.begin() + static_cast<std::ptrdiff_t>(local) * kHidden);
        }
        // Arm A published the state entering column j into slot base_a + j - 1.
        const Outputs arm_b = run_call(parent, shifted, conv_weight_bits, arm_a.state, slots,
                                       tokens, base_a + column - 1, base_b);

        const std::string suffix =
            " W8 T=" + std::to_string(tokens) + " column=" + std::to_string(column);

        // Control: the projection itself must not depend on the column's position.
        const int projection_failures =
            compare_column("restart z (control)" + suffix, arm_b.z, 0, arm_a.z, column, kZRows);
        failures += projection_failures;
        if (projection_failures != 0) {
            std::cerr << "restart" << suffix
                      << ": projection is position dependent, carry comparison is not conclusive\n";
        }

        failures += compare_column("restart query" + suffix, arm_b.query, 0, arm_a.query, column,
                                   kQueryRows);
        failures +=
            compare_column("restart key" + suffix, arm_b.key, 0, arm_a.key, column, kKeyRows);
        failures += compare_column("restart value" + suffix, arm_b.value, 0, arm_a.value, column,
                                   kValueRows);
        failures += compare_slot("restart state" + suffix, arm_b.state, base_b, arm_a.state,
                                 base_a + column);
    }
    return failures;
}

// ---------------------------------------------------------------------------
// Record path. This is the route MTP and DFlash verification take
// (docs/maintainer/qwen3.6-35b-a3b-model.md section 2.3), and the one invariant 6 of
// docs/maintainer/replayssm-gdn.md section 7 constrains: fold and verify must share the same
// state-store boundary. Fold rebuilds the convolution window out of the BF16 conv_record, so the
// window that a replay of column j starts from is exactly the last three recorded columns. This
// case builds that window on the host and restarts the Op from it.

struct RecordOutputs {
    std::vector<std::uint16_t> query;
    std::vector<std::uint16_t> key;
    std::vector<std::uint16_t> value;
    std::vector<std::uint16_t> record;
};

RecordOutputs run_record_call(DevicePackedWeight& parent,
                              const std::vector<std::uint16_t>& activation_bits,
                              const std::vector<std::uint16_t>& conv_weight_bits,
                              const std::vector<std::uint16_t>& state_bits, std::int32_t slots,
                              std::int32_t tokens, std::int32_t initial_slot) {
    const std::vector<std::int32_t> initial_value{initial_slot};

    DeviceBuffer device_activation  = to_device(activation_bits);
    DeviceBuffer device_conv_weight = to_device(conv_weight_bits);
    DeviceBuffer device_state       = to_device(state_bits);
    DeviceBuffer device_initial     = to_device(initial_value);
    GuardedBf16Tensor record(kChannels, tokens);
    GuardedBf16Tensor query(kQueryRows, tokens);
    GuardedBf16Tensor key(kKeyRows, tokens);
    GuardedBf16Tensor value(kValueRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);

    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens, 1});
    Tensor conv(device_conv_weight.p, DType::BF16, {kChannels, 4});
    Tensor conv_state(device_state.p, DType::BF16, {kChannels, 3, slots});
    Tensor initial(device_initial.p, DType::I32, {1});
    Tensor record_view(record.data(), DType::BF16, {kChannels, tokens, 1});
    Tensor query_view(query.data(), DType::BF16, {kQueryRows, tokens, 1});
    Tensor key_view(key.data(), DType::BF16, {kKeyRows, tokens, 1});
    Tensor value_view(value.data(), DType::BF16, {kValueRows, tokens, 1});
    Tensor z_view(z.data(), DType::BF16, {kZRows, tokens, 1});

    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, 1, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    ops::gdn_input_proj_conv_record(x, parent.view(), conv, conv_state, Tensor{}, initial,
                                    record_view, query_view, key_view, value_view, z_view,
                                    workspace, nullptr);
    cuda_synchronize();

    return RecordOutputs{query.bits(), key.bits(), value.bits(), record.bits()};
}

int run_record_restart_case(DevicePackedWeight& parent, std::int32_t tokens) {
    const std::int32_t slots = tokens + 1;
    const std::size_t stride = static_cast<std::size_t>(kChannels) * 3;

    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 811U + tokens);
    const std::vector<std::uint16_t> activation_bits  = bf16_bits(activation);
    const std::vector<float> conv_weight              = make_conv_weight(kChannels, 821U);
    const std::vector<std::uint16_t> conv_weight_bits = bf16_bits(conv_weight);
    std::vector<std::uint16_t> state_bits = make_state(kChannels, slots, 0, 823U + tokens);

    const RecordOutputs arm_a =
        run_record_call(parent, activation_bits, conv_weight_bits, state_bits, slots, tokens, 0);

    // Slot j holds the window a fold would enter column j with: the last three of
    // [initial columns] ++ [recorded columns 0..j-1], every one of them a recorded BF16 column.
    const auto history = [&](std::int32_t index, std::int32_t row) -> std::uint16_t {
        if (index < 0) {
            return state_bits[static_cast<std::size_t>(3 + index) * kChannels +
                              static_cast<std::size_t>(row)];
        }
        return arm_a
            .record[static_cast<std::size_t>(index) * kChannels + static_cast<std::size_t>(row)];
    };
    for (std::int32_t column = 1; column < tokens; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * stride;
        for (std::int32_t row = 0; row < kChannels; ++row) {
            state_bits[base + static_cast<std::size_t>(row)]             = history(column - 3, row);
            state_bits[base + static_cast<std::size_t>(kChannels + row)] = history(column - 2, row);
            state_bits[base + static_cast<std::size_t>(2 * kChannels + row)] =
                history(column - 1, row);
        }
    }

    int failures = 0;
    for (std::int32_t column = 1; column < tokens; ++column) {
        std::vector<std::uint16_t> shifted(activation_bits.size());
        for (std::int32_t local = 0; local < tokens; ++local) {
            const std::int32_t source = std::min(column + local, tokens - 1);
            std::copy_n(activation_bits.begin() + static_cast<std::ptrdiff_t>(source) * kHidden,
                        kHidden, shifted.begin() + static_cast<std::ptrdiff_t>(local) * kHidden);
        }
        const RecordOutputs arm_b =
            run_record_call(parent, shifted, conv_weight_bits, state_bits, slots, tokens, column);

        const std::string suffix =
            " W8 record T=" + std::to_string(tokens) + " column=" + std::to_string(column);

        const int projection_failures = compare_column(
            "restart record (control)" + suffix, arm_b.record, 0, arm_a.record, column, kChannels);
        failures += projection_failures;
        if (projection_failures != 0) {
            std::cerr << "restart" << suffix
                      << ": recorded column is position dependent, comparison is not conclusive\n";
        }

        failures += compare_column("restart query" + suffix, arm_b.query, 0, arm_a.query, column,
                                   kQueryRows);
        failures +=
            compare_column("restart key" + suffix, arm_b.key, 0, arm_a.key, column, kKeyRows);
        failures += compare_column("restart value" + suffix, arm_b.value, 0, arm_a.value, column,
                                   kValueRows);
    }
    return failures;
}

} // namespace

int main() {
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::W8G32_F16S, 12288, kHidden, 727U));
    int failures = 0;
    // Widths on both sides of the conv route boundary at 16 columns: the fused route carries the
    // tap in registers, the materialized route round-trips it through a BF16 scratch. Both are
    // registered production routes of the same Op and both are checked here.
    for (const std::int32_t tokens : {2, 3, 4, 8, 16, 17}) {
        failures += run_restart_case(parent, tokens);
    }
    // The record form admits T = 2..16 at batch 1 and is fused across that whole domain, so every
    // single-request MTP or DFlash verify window on this target goes through the carry.
    for (const std::int32_t tokens : {2, 3, 4, 8, 16}) {
        failures += run_record_restart_case(parent, tokens);
    }
    failures += parent.verify_preserved("restart parent weight");
    if (failures != 0) {
        std::cerr << "gdn input proj conv restart: " << failures << " failures\n";
        return 1;
    }
    std::cout << "gdn input proj conv restart: OK\n";
    return 0;
}
