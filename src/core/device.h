#pragma once

#include <cuda_runtime.h>

#include "core/shard_plan.h"
#include "core/virtual_device.h"

#include <cstddef>
#include <cstdint>
#include <string>

// INTEGRATE4 (landing order row 2): the three standard headers ExecutionContext's declaration
// needs -- <array> for the two device slots, <optional> for their emptiness, <vector> for the
// constructor's id list. Added additively; nothing above or below this line moved.
#include <array>
#include <optional>
#include <vector>

namespace ninfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

struct DeviceContext {
    int device                   = 0;
    cudaStream_t stream          = nullptr;
    cudaStream_t transfer_stream = nullptr;
    cudaDeviceProp props{};
    // Static YaRN factor-4 rope extension (Text D256/R64 and DFlash D128/R128).
    bool yarn_enabled = false;

    // THE VIRTUAL DEVICE BINDING (core/virtual_device.h). Inactive unless the two-key guard in
    // the environment is set, and inactive means every accessor below returns exactly what it
    // returned before this member existed: rank 0, world 1, the real sm(), the real VRAM.
    //
    // It is a BUDGET and an IDENTITY, never a partition: several logical ranks on one physical
    // card share one HBM and time-slice one SM pool. Nothing measured under it is a multi-GPU
    // measurement, which is why the guard prints a banner that says so.
    multi::VirtualBinding virtual_binding{};

    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();
    DeviceContext(int device_id, const multi::VirtualBinding& binding);

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    void bind_to_current_thread() const;
    void bind_to_current_thread_noexcept() const noexcept;
    int sm() const noexcept;
    std::size_t total_vram() const noexcept;
    void synchronize() const;

    // Reads the two-key guard from the environment and resolves it against the facts of the
    // device just probed. A no-op when nothing was requested; a THROW when a request was made and
    // could not be honoured (see core/virtual_device.h). Declared here rather than hidden in the
    // constructor so the failures it can produce are part of this type's surface.
    void resolve_virtual_device(int physical_count, int device_id);

    // THE SHARD CONTRACT (core/shard_plan.h section 6). Resolves NINFER_WORLD_SIZE /
    // NINFER_RANK against the physical device count just probed and THROWS when the
    // declared world cannot be honoured, naming the piece that is missing. A no-op when
    // nothing was requested, and it prints a one-time notice when the machine has cards
    // this run will not use. Declared here rather than hidden in the constructor for the
    // same reason resolve_virtual_device is: the failures it can produce are part of this
    // type's surface.
    void resolve_shard_world(int physical_count, int device_id);

    // This object's place in the world. (0, 1, none) when no virtual device is active, which is
    // the single-device world every existing call site already is.
    [[nodiscard]] multi::WorldShape world() const noexcept { return virtual_binding.world; }
    [[nodiscard]] std::uint32_t rank() const noexcept { return virtual_binding.world.rank; }
    [[nodiscard]] std::uint32_t world_size() const noexcept {
        return virtual_binding.world.world_size;
    }
    [[nodiscard]] bool virtual_device() const noexcept { return virtual_binding.active; }
    // The soft SM budget of this rank; 0 when the real device's SM count is what applies.
    [[nodiscard]] std::uint32_t virtual_sm_budget() const noexcept {
        return virtual_binding.sm_budget;
    }
};

// INTEGRATE4 (landing order row 2): landed verbatim from the gfx fork
// JCraigWasTaken__ninfer-gfx906/__gfx906-port, src/core/device.h:43-61, because three landed
// consumers need the NAME and two of them need its members: include/ninfer/ops/allreduce.h
// (:104,:121,:199,:224 take `const ExecutionContext&`), src/ops/common/split_launch.h
// (:74,:99,:117 read `ec.tp` and `ec.dev[r]->device`), and src/ops/common/allreduce.cu (:57).
// This header's own line 86 already named the missing type in a comment -- the tree asked for
// this by name, so it lands in the tree's one home for it rather than in a new header.
//
// NOT landed with it, each refusal a measurement: DeviceExecutionView, DeviceContext::load_stream,
// compute_capability(), multiprocessor_count(), execution_view() -- 0 consumers in this tree name
// any of them (git grep --untracked). The constructor body IS landed, in src/core/device.cu: the
// type is useless to a caller that cannot construct it, and that half-landing is the exact defect
// this pass exists to close.
//
// One process, up to two devices. dev[0..tp-1] hold constructed DeviceContext instances;
// the remaining slots stay empty. tp == 1 unless the caller opts into `--tp 2`, which runs the
// tensor-parallel program across both devices.
struct ExecutionContext {
    std::array<std::optional<DeviceContext>, 2> dev;
    int tp = 1;

    // device_ids.size() must be 1 or 2 and becomes tp. Every id is validated to exist by
    // DeviceContext's own constructor; when two ids are given they must additionally share the
    // same compute capability (major.minor), since nothing downstream can reconcile mismatched
    // architectures.
    explicit ExecutionContext(const std::vector<int>& device_ids);

    [[nodiscard]] DeviceContext& primary() noexcept { return *dev[0]; }
    [[nodiscard]] const DeviceContext& primary() const noexcept { return *dev[0]; }
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

// Reusable non-timing event for worker-driven asynchronous control transactions. The owning
// component records it after enqueueing one transfer batch and polls it from later boundaries.
class CudaCompletionEvent {
public:
    explicit CudaCompletionEvent(const DeviceContext& ctx);
    ~CudaCompletionEvent();

    CudaCompletionEvent(const CudaCompletionEvent&)            = delete;
    CudaCompletionEvent& operator=(const CudaCompletionEvent&) = delete;
    CudaCompletionEvent(CudaCompletionEvent&& other) noexcept;
    CudaCompletionEvent& operator=(CudaCompletionEvent&& other) noexcept;

    void record(cudaStream_t stream);
    void wait(cudaStream_t stream) const;
    [[nodiscard]] bool ready() const;
    void synchronize() const;

private:
    int device_        = 0;
    cudaEvent_t event_ = nullptr;
};

} // namespace ninfer
