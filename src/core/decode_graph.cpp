#include "core/decode_graph.h"

#include "core/device.h"
#include "core/nvtx.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_graph_exec(cudaGraphExec_t& exec) noexcept {
    if (exec != nullptr) {
        log_cuda_error("cudaGraphExecDestroy", cudaGraphExecDestroy(exec));
        exec = nullptr;
    }
}

void destroy_graph(cudaGraph_t& graph) noexcept {
    if (graph != nullptr) {
        log_cuda_error("cudaGraphDestroy", cudaGraphDestroy(graph));
        graph = nullptr;
    }
}

void discard_capture(cudaStream_t stream) noexcept {
    cudaGraph_t discard = nullptr;
    log_cuda_error("cudaStreamEndCapture(discard)", cudaStreamEndCapture(stream, &discard));
    destroy_graph(discard);
}

} // namespace

DecodeGraphDefinition::~DecodeGraphDefinition() { reset(); }

DecodeGraphDefinition::DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept
    : graphs_(std::move(other.graphs_)) {}

DecodeGraphDefinition& DecodeGraphDefinition::operator=(DecodeGraphDefinition&& other) noexcept {
    if (this == &other) { return *this; }
    reset();
    graphs_ = std::move(other.graphs_);
    return *this;
}

void DecodeGraphDefinition::capture(cudaStream_t stream, const std::function<void()>& body) {
    capture_segments(stream, {body});
}

void DecodeGraphDefinition::capture_segments(
    cudaStream_t stream, const std::vector<std::function<void()>>& bodies) {
    nvtx::ScopedRange capture_range(nvtx::Name::CudaGraphCapture, nvtx::Category::Graph);
    reset();
    if (bodies.empty()) { return; }

    for (const auto& body : bodies) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        try {
            body();
        } catch (...) {
            discard_capture(stream);
            reset();
            throw;
        }
        cudaGraph_t graph = nullptr;
        cudaError_t err   = cudaStreamEndCapture(stream, &graph);
        if (err != cudaSuccess) {
            destroy_graph(graph);
            reset();
            CUDA_CHECK(err);
        }
        graphs_.push_back(graph);
    }
}

bool DecodeGraphDefinition::ready() const noexcept { return !graphs_.empty(); }

std::size_t DecodeGraphDefinition::segment_count() const noexcept { return graphs_.size(); }

void DecodeGraphDefinition::reset() noexcept {
    for (auto& g : graphs_) { destroy_graph(g); }
    graphs_.clear();
}

DecodeGraphExecutable::~DecodeGraphExecutable() { reset(); }

DecodeGraphExecutable::DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept
    : execs_(std::move(other.execs_)) {}

DecodeGraphExecutable& DecodeGraphExecutable::operator=(DecodeGraphExecutable&& other) noexcept {
    if (this == &other) { return *this; }
    reset();
    execs_ = std::move(other.execs_);
    return *this;
}

void DecodeGraphExecutable::instantiate(const DecodeGraphDefinition& definition) {
    nvtx::ScopedRange instantiate_range(nvtx::Name::CudaGraphInstantiate, nvtx::Category::Graph);
    if (!definition.ready()) {
        throw std::logic_error("cannot instantiate an empty CUDA Graph definition");
    }
    reset();
    execs_.reserve(definition.graphs_.size());
    for (const cudaGraph_t graph : definition.graphs_) {
        cudaGraphExec_t exec  = nullptr;
        const cudaError_t err = cudaGraphInstantiate(&exec, graph, 0);
        if (err != cudaSuccess) {
            destroy_graph_exec(exec);
            reset();
            CUDA_CHECK(err);
        }
        execs_.push_back(exec);
    }
}

void DecodeGraphExecutable::update(const DecodeGraphDefinition& definition) {
    nvtx::ScopedRange update_range(nvtx::Name::CudaGraphUpdate, nvtx::Category::Graph);
    if (!ready() || !definition.ready()) {
        throw std::logic_error("CUDA Graph update requires a definition and executable");
    }
    if (execs_.size() != definition.graphs_.size()) {
        throw std::logic_error("CUDA Graph update requires matching segment counts");
    }
    for (std::size_t i = 0; i < execs_.size(); ++i) {
        cudaGraphExecUpdateResultInfo result{};
        const cudaError_t err = cudaGraphExecUpdate(execs_[i], definition.graphs_[i], &result);
        if (err != cudaSuccess || result.result != cudaGraphExecUpdateSuccess) {
            throw std::runtime_error(
                "CUDA Graph executable update failed: " + std::string(cudaGetErrorName(err)) +
                " (update result " + std::to_string(static_cast<int>(result.result)) + ")");
        }
    }
}

void DecodeGraphExecutable::upload(cudaStream_t stream) {
    nvtx::ScopedRange upload_range(nvtx::Name::CudaGraphUpload, nvtx::Category::Graph);
    if (!ready()) { throw std::logic_error("cannot upload an empty CUDA Graph executable"); }
    for (cudaGraphExec_t exec : execs_) { CUDA_CHECK(cudaGraphUpload(exec, stream)); }
}

void DecodeGraphExecutable::launch(cudaStream_t stream) {
    nvtx::ScopedRange launch_range(nvtx::Name::CudaGraphLaunch, nvtx::Category::Graph);
    if (!ready()) { throw std::logic_error("cannot launch an empty CUDA Graph executable"); }
    for (cudaGraphExec_t exec : execs_) { CUDA_CHECK(cudaGraphLaunch(exec, stream)); }
}

bool DecodeGraphExecutable::ready() const noexcept { return !execs_.empty(); }

void DecodeGraphExecutable::reset() noexcept {
    for (auto& e : execs_) { destroy_graph_exec(e); }
    execs_.clear();
}

} // namespace ninfer
