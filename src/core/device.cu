#include "core/device.h"

#include "core/virtual_device.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {
namespace {

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_stream(cudaStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

} // namespace

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", file, line, expr,
                 cudaGetErrorName(err), cudaGetErrorString(err));
    std::abort();
}

DeviceContext::DeviceContext(int device_id) : device(device_id) {
    int count       = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no CUDA devices available"); }
    if (device_id < 0 || device_id >= count) { throw std::runtime_error("invalid CUDA device id"); }

    bind_to_current_thread();

    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceProperties failed", err));
    }

    // The virtual-device guard, resolved from the environment and validated against the facts
    // just read. It throws when the request cannot be honoured, and does nothing at all when
    // nothing was requested -- see core/virtual_device.h for why a half-given request is a
    // refusal rather than a quiet single-device run.
    resolve_virtual_device(count, device_id);

    // THE SHARD CONTRACT (core/shard_plan.h section 6). A no-op when nothing was
    // requested, which is what keeps this call behaviour-preserving on every existing
    // run: with NINFER_WORLD_SIZE unset, not one line of behaviour below changes. What it
    // replaces is the SILENT SINGLE-CARD run -- the section header carries the file:line
    // for the device-0 binding this machine used to get without being told.
    resolve_shard_world(count, device_id);

    cudaStream_t compute = nullptr;
    cudaStream_t load    = nullptr;
    err                  = cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(stream) failed", err));
    }

    err = cudaStreamCreateWithFlags(&load, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        destroy_stream(compute);
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(transfer_stream) failed", err));
    }

    stream          = compute;
    transfer_stream = load;
}

DeviceContext::DeviceContext(int device_id, const multi::VirtualBinding& binding)
    : DeviceContext(device_id) {
    // The explicit form, for a caller that built the world itself (a test or a launcher that
    // wants several rank objects in one process). The environment form above is the
    // operator-facing one; this one is the programmatic one, and it takes the SAME validated
    // binding type so there is exactly one notion of "what a virtual device is".
    if (!binding.active) {
        throw std::invalid_argument("DeviceContext: an explicit virtual binding must be active");
    }
    // TWO WAYS TO ASK AT ONCE IS A REFUSAL, not a precedence rule. If the environment also
    // declares a world, this object's rank has already been decided twice and one of the two
    // answers is being discarded silently -- which is the class of mistake this guard exists to
    // make impossible.
    if (virtual_binding.active) {
        throw std::invalid_argument(
            "virtual device refused [bad-axis]: this DeviceContext was given an explicit binding "
            "for rank " + std::to_string(binding.world.rank) + " of world " +
            std::to_string(binding.world.world_size) + ", and " +
            std::string(multi::kVirtualDevicesEnv) +
            " also declares a world for the same object. Unset the environment variable to use "
            "the explicit binding, or use the environment alone.");
    }
    virtual_binding = binding;
}

DeviceContext::~DeviceContext() {
    if (stream != nullptr || transfer_stream != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream);
    destroy_stream(stream);
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device(other.device), stream(other.stream), transfer_stream(other.transfer_stream),
      props(other.props), virtual_binding(std::move(other.virtual_binding)) {
    other.stream          = nullptr;
    other.transfer_stream = nullptr;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    if (stream != nullptr || transfer_stream != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream);
    destroy_stream(stream);

    device          = other.device;
    props           = other.props;
    stream          = other.stream;
    transfer_stream = other.transfer_stream;
    virtual_binding = std::move(other.virtual_binding);

    other.stream          = nullptr;
    other.transfer_stream = nullptr;
    return *this;
}

void DeviceContext::resolve_virtual_device(int physical_count, int device_id) {
    const multi::VirtualRequest request = multi::virtual_request_from_environment();
    if (!request.requested) { return; }

    const multi::VirtualBinding binding = multi::validate_virtual_request(
        request, physical_count, props.multiProcessorCount,
        static_cast<std::uint64_t>(props.totalGlobalMem));

    // FAIL CLOSED. A request that was made and could not be honoured stops the run and names the
    // field that failed. It is never downgraded to "the real device, then", because a run that
    // silently became single-device would be indistinguishable from a successful simulation at
    // the point where the numbers are read.
    if (!binding.active) {
        throw std::invalid_argument(
            "virtual device refused [" +
            std::string(multi::virtual_refusal_name(binding.refusal)) + "]: " +
            binding.refusal_detail);
    }
    // A virtual rank is bound to the ONE physical device (the validator has already refused a
    // binding when more than one CUDA device is present). The RANK is not the device id: rank 1
    // of a 2-rank world still runs on physical device 0, which is what makes several rank objects
    // possible in one process and is the only form of multi-rank this box can honour.
    if (device_id != 0) {
        throw std::invalid_argument(
            "virtual device refused [physical-device-count-not-one]: this DeviceContext was "
            "asked for physical device id " + std::to_string(device_id) +
            ", but a virtual rank is bound to physical device 0; the world is made of ranks, not "
            "of devices");
    }

    virtual_binding = binding;

    // LOUD, ONCE PER PROCESS. Not a debug log and not behind a verbosity flag: a number produced
    // under this guard must not be quotable as a hardware result, so the caveat travels with the
    // run rather than living in whoever read the log.
    static bool banner_printed = false;
    if (!banner_printed) {
        banner_printed = true;
        const std::string text = multi::virtual_banner(
            virtual_binding, std::string(props.name),
            static_cast<std::uint32_t>(props.multiProcessorCount));
        std::fprintf(stderr, "%s", text.c_str());
        std::fflush(stderr);
    }
}


// THE SHARD CONTRACT (core/shard_plan.h section 6). The engine's answer to a question the
// shard ARITHMETIC does not ask: is there anything in this tree that can BIND a second device,
// MOVE a byte between two devices, or READ a per-rank weight stream? Every one of those is
// answered by a fact, the facts of this revision are all false, and a declared world is
// therefore REFUSED BY NAME instead of being served as a silent single-card run.
//
// FAIL CLOSED, and for the reason resolve_virtual_device states one screen up: a run that
// silently became single-device is indistinguishable from a successful world at the point
// where the numbers are read.
void DeviceContext::resolve_shard_world(int physical_count, int device_id) {
    (void)device_id;
    const multi::WorldRequest request = multi::world_request_from_environment();
    const std::string refusal       = multi::cross_device_world_refusal(
        request, multi::kCrossDeviceFactsThisRevision,
        static_cast<std::uint32_t>(physical_count));
    if (!refusal.empty()) {
        throw std::invalid_argument(
            "shard world refused [cross-device-world-unavailable]: " + refusal);
    }
    // The other half: a machine that HAS more cards than this run will use must say so. A
    // notice and not a refusal, because a single-card run on a multi-card box is a legitimate
    // run -- what is not legitimate is not knowing. LOUD, ONCE PER PROCESS, like the virtual
    // banner above and for the same reason.
    const std::string notice = multi::idle_device_notice(
        static_cast<std::uint32_t>(physical_count), world());
    if (!notice.empty()) {
        static bool banner_printed = false;
        if (!banner_printed) {
            banner_printed = true;
            std::fprintf(stderr, "%s", notice.c_str());
            std::fflush(stderr);
        }
    }
}

void DeviceContext::bind_to_current_thread() const {
    const cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
    }
}

void DeviceContext::bind_to_current_thread_noexcept() const noexcept {
    log_cuda_error("cudaSetDevice", cudaSetDevice(device));
}

// THE ONE PLACE THE SIMULATED CAPABILITY ENTERS, AND WHY IT IS SAFE.
//
// `sm()` is the input to the artifact-format gate (src/targets/registry.cpp, the single call
// `caps::require_artifact_formats_supported(device.sm(), ...)`) and to the route selector. Under
// an active virtual binding this returns the VIRTUAL card's number, so the engine asks "may an
// sm_70 card load this artifact?" -- and the answer is whatever the ladder says about sm_70.
// That is the point: the guard changes WHICH QUESTION is asked, never the answer. A virtual
// V100 offered an NVFP4 artifact is refused by the real gate, because sm_70's ladder row does
// not cover kind::mxf4nvf4, exactly as a physical V100 would be.
//
// With no binding this is `props.major * 10 + props.minor`, character for character what it was.
int DeviceContext::sm() const noexcept {
    if (virtual_binding.active) { return virtual_binding.capability_sm; }
    return props.major * 10 + props.minor;
}

// Under a virtual binding this is the rank's memory BUDGET, not a partition: the ranks share one
// physical HBM and the binding says so in its banner. It is reported here because this is where
// the engine asks "how much device memory do I have", so a weight-residency decision taken here
// is the decision a real card of that size would make.
std::size_t DeviceContext::total_vram() const noexcept {
    if (virtual_binding.active) { return virtual_binding.memory_budget_bytes; }
    return props.totalGlobalMem;
}

void DeviceContext::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream)); }

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx) : CudaEventTimer(ctx, ctx.stream) {}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream) : stream_(stream) {
    if (stream == nullptr) { throw std::invalid_argument("CUDA timer stream is null"); }
    ctx.bind_to_current_thread();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    cudaError_t err   = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreate(start) failed", err));
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        destroy_event(start);
        throw std::runtime_error(cuda_error_message("cudaEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

CudaEventTimer::~CudaEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    destroy_event(stop_);
    destroy_event(start_);

    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void CudaEventTimer::start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

void CudaEventTimer::record_stop() { CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

float CudaEventTimer::elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

float CudaEventTimer::stop_ms() {
    record_stop();
    CUDA_CHECK(cudaEventSynchronize(stop_));
    return elapsed_ms();
}

CudaCompletionEvent::CudaCompletionEvent(const DeviceContext& ctx) : device_(ctx.device) {
    ctx.bind_to_current_thread();
    const cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreateWithFlags failed", err));
    }
}

CudaCompletionEvent::~CudaCompletionEvent() { destroy_event(event_); }

CudaCompletionEvent::CudaCompletionEvent(CudaCompletionEvent&& other) noexcept
    : device_(other.device_), event_(std::exchange(other.event_, nullptr)) {}

CudaCompletionEvent& CudaCompletionEvent::operator=(CudaCompletionEvent&& other) noexcept {
    if (this == &other) { return *this; }
    destroy_event(event_);
    device_ = other.device_;
    event_  = std::exchange(other.event_, nullptr);
    return *this;
}

void CudaCompletionEvent::record(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    CUDA_CHECK(cudaEventRecord(event_, stream));
}

void CudaCompletionEvent::wait(cudaStream_t stream) const {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not waitable");
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
}

bool CudaCompletionEvent::ready() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    const cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess) { return true; }
    if (status == cudaErrorNotReady) { return false; }
    CUDA_CHECK(status);
    return false;
}

void CudaCompletionEvent::synchronize() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    CUDA_CHECK(cudaEventSynchronize(event_));
}

// INTEGRATE4 (landing order row 2): the ExecutionContext constructor, landed verbatim from the
// gfx fork JCraigWasTaken__ninfer-gfx906/__gfx906-port, src/core/device.cu:45-84. It is the
// companion of the struct landed in src/core/device.h and is landed WITH it on purpose: a
// declared-but-undefined constructor compiles and then fails to link, which is the silent middle
// state this pass exists to avoid.
//
// The fork's own error text is kept verbatim, including the word "gfx" in the compute-capability
// message -- that is the gfx906 port's wording, and re-wording it here would create a second
// spelling of a message for no benefit.
ExecutionContext::ExecutionContext(const std::vector<int>& device_ids) {
    if (device_ids.empty() || device_ids.size() > dev.size()) {
        throw std::runtime_error("ExecutionContext requires 1 or 2 device ids, got " +
                                 std::to_string(device_ids.size()));
    }
    tp = static_cast<int>(device_ids.size());
    // Distinct ids are a correctness precondition, not a preference: every tensor-parallel op
    // pairs `dev[0]` with `dev[1]` and assumes the two hold DIFFERENT shards on DIFFERENT devices.
    // `--devices 0,0` would build two contexts on one GPU, halve nothing, and make the peer
    // copies alias their own source -- silently wrong rather than slow.
    for (std::size_t i = 0; i < device_ids.size(); ++i) {
        for (std::size_t j = i + 1; j < device_ids.size(); ++j) {
            if (device_ids[i] == device_ids[j]) {
                throw std::runtime_error(
                    "ExecutionContext requires distinct device ids, got device " +
                    std::to_string(device_ids[i]) + " twice");
            }
        }
    }
    for (std::size_t i = 0; i < device_ids.size(); ++i) {
        dev[i].emplace(device_ids[i]); // validates existence internally
    }
    if (tp == 2) {
        const cudaDeviceProp& p0 = dev[0]->props;
        const cudaDeviceProp& p1 = dev[1]->props;
        if (p0.major != p1.major || p0.minor != p1.minor) {
            throw std::runtime_error(
                "ExecutionContext requires all devices to share the same compute capability "
                "(device " +
                std::to_string(dev[0]->device) + " is gfx " + std::to_string(p0.major) +
                std::to_string(p0.minor) + ", device " + std::to_string(dev[1]->device) +
                " is gfx " + std::to_string(p1.major) + std::to_string(p1.minor) + ")");
        }
    }
    // POSTCONDITION: rank 0 is the current device. Constructing the contexts in order leaves the
    // LAST one current, so at tp == 2 an ExecutionContext would hand its caller a thread bound to
    // device 1 -- and every caller that then issues work without naming a device (the whole tp1
    // code path, and rank-0-only steps like sampling) would silently target the wrong GPU.
    cuda_check(cudaSetDevice(dev[0]->device), "cudaSetDevice(dev[0])", __FILE__, __LINE__);
}

} // namespace ninfer
