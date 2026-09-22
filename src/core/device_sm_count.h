#pragma once

#include <cuda_runtime.h>

#include "core/virtual_device.h"

// The SM count of the device that owns the current CUDA context.
//
// TUNING INPUT, NOT A CAPABILITY TEST. A grid size is a performance decision, and
// nothing here may refuse a device or a request. But a grid size that is
// hardwired to one card model is a hazard in the direction OPPOSITE to a
// refusal: on a GPU with a different SM count the same number silently changes
// how many waves a persistent kernel runs, and a path that picks between two
// kernel templates by grid size would change what it computes. Deriving the
// number from the device removes the whole class of problem instead of
// documenting it: the device is asked, so the answer is right on the card that
// is actually present.
//
// This accessor is deliberately the ONLY place a card-model number is spelled,
// so every such number this build still carries can be found by grepping here.
// Two call sites used to hold an independent `kRtx5090SmCount = 170`
// (ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu,
// ops/linear_attention/gated_delta_net/chunked/output.cu), each of which was a
// bare claim about the hardware rather than a query to it.
namespace ninfer {

struct DeviceSmCount {
    int sm_count = 0;
    bool measured = false; // false => the device could not be asked; the caller decides
    // True when sm_count came from a virtual-device budget rather than from the hardware. The
    // caller must treat that the same way it treats `measured == false`: the number is a
    // tuning input derived from a declared world, not a property of the card in the machine.
    bool virtualized = false;
};

// Never throws: a failure to ask is reported as `measured == false` and the
// caller must choose its own tuning fallback explicitly (there is no silent
// default here -- a silent default is the defect this header removes).
[[nodiscard]] inline DeviceSmCount current_device_sm_count() noexcept {
    DeviceSmCount result;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return result; }
    int sm_count = 0;
    if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        sm_count <= 0) {
        return result;
    }
    // THE SOFT SM BUDGET. This header is the tree's only place a card-model number is spelled
    // (its own doctrine, above), so it is also the only honest place for a SIMULATED card's SM
    // count to arrive. A world of N virtual ranks time-slices the SMs, so each rank's budget is
    // physical_sm_count / N, and a grid sized from it stays inside one rank's share.
    //
    // A refused request leaves `measured == false` and `virtualized == true`, i.e. "I was asked
    // and could not answer", which is this header's existing contract for a failure to ask. The
    // refusal itself is thrown where it belongs: DeviceContext's constructor (core/device.cu),
    // which is the layer that owns the run. Never throwing here is deliberate -- the doctrine
    // above says this accessor is a tuning input and must not refuse a device or a request.
    const multi::VirtualRequest request = multi::virtual_request_from_environment();
    if (request.requested) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess) { return result; }
        const multi::VirtualBinding binding =
            multi::validate_virtual_request(request, device_count, sm_count, 0);
        result.virtualized = true;
        if (!binding.active) { return result; }
        result.sm_count = static_cast<int>(binding.sm_budget);
        result.measured = true;
        return result;
    }

    result.sm_count = sm_count;
    result.measured = true;
    return result;
}

} // namespace ninfer
