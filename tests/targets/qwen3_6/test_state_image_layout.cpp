#include "core/layout.h"
#include <ninfer/targets/qwen3_6/state_image.h>

#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

namespace q36 = ninfer::targets::qwen3_6;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

q36::StateImageDeviceLayout plan(bool dflash) {
    q36::StateImageSpec spec{
        .linear =
            {
                .layers         = 2,
                .conv_channels  = 5,
                .conv_width     = 3,
                .value_heads    = 2,
                .value_head_dim = 4,
                .key_head_dim   = 3,
                .slot_count     = 2,
                .conv_dtype     = ninfer::DType::BF16,
            },
        .hidden = 7,
    };
    if (dflash) {
        spec.dflash_local =
            q36::DFlashLocalStateSpec{.layers = 2, .capacity = 17, .kv_heads = 2, .head_dim = 4};
    }
    ninfer::LayoutBuilder builder;
    return q36::plan_state_image_device_pool(builder, spec);
}

// The real qwen3.6-27b geometry (src/targets/qwen3_6_27b/impl/config.h): 48 gated-delta-net
// layers, convolution_dim = 2*(16*128) + 48*128 = 10240, conv state width 3, 48 value heads of
// 128, hidden 5120. dflash_local is deliberately absent: this variant runs MTP, not DFlash.
q36::StateImageDeviceLayout plan_27b(std::uint32_t slot_count) {
    q36::StateImageSpec spec{
        .linear =
            {
                .layers         = 48,
                .conv_channels  = 10240,
                .conv_width     = 3,
                .value_heads    = 48,
                .value_head_dim = 128,
                .key_head_dim   = 128,
                .slot_count     = slot_count,
                .conv_dtype     = ninfer::DType::BF16,
            },
        .hidden = 5120,
    };
    ninfer::LayoutBuilder builder;
    return q36::plan_state_image_device_pool(builder, spec);
}

// The HOST image, which is ONE slot by construction: HostStatePool multiplies
// host.image_bytes by its capacity, and state_image_transfer_work moves exactly one image.
std::uint64_t host_image_bytes(const q36::StateImageDeviceLayout& layout) {
    return layout.host.linear_conv.bytes + layout.host.linear_recurrent.bytes +
           layout.host.continuation_hidden.bytes;
}

// The DEVICE pool. THIS is the quantity "one more slot costs one image's geometry" is
// about, per the comment below: plan_linear_attention_state_pool folds slot_count into the
// last dimension of both per-layer tensors (core/linear_attention_state.cpp), and the device
// continuation hidden row is added as {hidden, slot_count} (state_image.cpp).
std::uint64_t device_pool_bytes(const q36::StateImageDeviceLayout& layout) {
    std::uint64_t bytes = layout.continuation_hidden.region.bytes;
    for (const ninfer::LayoutRegion& region : layout.linear.conv) { bytes += region.bytes; }
    for (const ninfer::LayoutRegion& region : layout.linear.recurrent) {
        bytes += region.bytes;
    }
    return bytes;
}

} // namespace

int main() {
    const q36::StateImageDeviceLayout common = plan(false);
    const auto common_work                   = q36::state_image_transfer_work(common.host);
    expect(common_work.payload_bytes == common.host.linear_conv.bytes +
                                            common.host.linear_recurrent.bytes +
                                            common.host.continuation_hidden.bytes,
           "common StateImage work payload includes layout padding");
    expect(common_work.copy_operations == 2 * common.host.spec.linear.layers + 1,
           "common StateImage work does not match physical CUDA copies");

    const q36::StateImageDeviceLayout dflash = plan(true);
    const auto full_work                     = q36::state_image_transfer_work(dflash.host);
    const auto local_work                    = q36::dflash_local_transfer_work(dflash.host);
    const std::uint64_t local_bytes =
        2ULL * dflash.host.dflash_local_layer_bytes * dflash.host.spec.dflash_local->layers;
    expect(local_work.payload_bytes == local_bytes &&
               local_work.copy_operations == 2 * dflash.host.spec.dflash_local->layers,
           "DFlash-local work does not match physical K/V layer copies");
    expect(full_work.payload_bytes == common_work.payload_bytes + local_work.payload_bytes &&
               full_work.copy_operations ==
                   common_work.copy_operations + local_work.copy_operations,
           "full DFlash StateImage work is not common plus local state");

    // ONE STATEIMAGE SLOT IS NOT FREE: the device pool is sized `max_concurrency +
    // device_state_slots` (layouts_impl.h) and every linear-attention conv/recurrent tensor plus
    // the continuation hidden row carries `slot_count` as its last dimension, so one more slot
    // costs exactly one image's geometry. At the 27b shape that is
    // 48 x (128*128*48*4 recurrent + 10240*3*2 conv) + 5120*2 hidden = 153,954,304 B
    // = 146.8223 MiB per slot -- the number the engine's own resource ledger reports as the
    // difference between a one-slot and a two-slot pool. Every stride below is already a
    // multiple of 256, so the arena's alignment cannot be hiding a disagreement.
    constexpr std::uint64_t kPerSlot =
        48ULL * (128ULL * 128ULL * 48ULL * 4ULL + 10240ULL * 3ULL * 2ULL) + 5120ULL * 2ULL;
    expect(kPerSlot == 153954304ULL,
           "the analytic per-slot StateImage geometry is no longer 146.8223 MiB");
    const q36::StateImageDeviceLayout one_slot = plan_27b(1);
    const q36::StateImageDeviceLayout two_slot = plan_27b(2);
    expect(two_slot.host.spec.linear.slot_count == 2 && one_slot.host.spec.linear.slot_count == 1,
           "the StateImage spec did not carry the requested slot count");
    expect(device_pool_bytes(two_slot) - device_pool_bytes(one_slot) == kPerSlot,
           "one extra StateImage slot did not cost exactly one image's geometry");
    // The host image must NOT move with the slot count: if it did, HostStatePool would price
    // the slot axis twice. This is the counterpart of the line above, and it is why the
    // accessor changed -- the per-slot contract lives in the DEVICE layout, not the host one.
    expect(host_image_bytes(two_slot) == host_image_bytes(one_slot),
           "the host StateImage image must stay one slot's geometry, not grow with slot_count");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
