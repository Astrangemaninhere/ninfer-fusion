#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <vector>

namespace ninfer {

class DecodeGraphDefinition {
public:
    DecodeGraphDefinition() = default;
    ~DecodeGraphDefinition();

    DecodeGraphDefinition(const DecodeGraphDefinition&)            = delete;
    DecodeGraphDefinition& operator=(const DecodeGraphDefinition&) = delete;
    DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept;
    DecodeGraphDefinition& operator=(DecodeGraphDefinition&& other) noexcept;

    void capture(cudaStream_t stream, const std::function<void()>& body);
    // Segment-aware capture: each body is recorded as its own graph. At launch the
    // segments run back-to-back on the same stream, but the driver submits each
    // segment's nodes separately: segment k's submission overlaps segment k-1's
    // execution, hiding the per-node launch cost that otherwise idles the GPU.
    void capture_segments(cudaStream_t stream,
                          const std::vector<std::function<void()>>& bodies);
    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] std::size_t segment_count() const noexcept;
    void reset() noexcept;

private:
    friend class DecodeGraphExecutable;
    std::vector<cudaGraph_t> graphs_;
};

class DecodeGraphExecutable {
public:
    DecodeGraphExecutable() = default;
    ~DecodeGraphExecutable();

    DecodeGraphExecutable(const DecodeGraphExecutable&)            = delete;
    DecodeGraphExecutable& operator=(const DecodeGraphExecutable&) = delete;
    DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept;
    DecodeGraphExecutable& operator=(DecodeGraphExecutable&& other) noexcept;

    void instantiate(const DecodeGraphDefinition& definition);
    void update(const DecodeGraphDefinition& definition);
    void upload(cudaStream_t stream);
    void launch(cudaStream_t stream);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    std::vector<cudaGraphExec_t> execs_;
};

} // namespace ninfer
