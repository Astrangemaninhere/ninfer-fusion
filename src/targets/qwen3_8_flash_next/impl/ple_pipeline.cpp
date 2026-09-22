// Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
#include "targets/qwen3_8_flash_next/impl/ple_pipeline.h"

#include "core/arena.h"
#include "targets/qwen3_8_flash_next/impl/ple_decode_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <future>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

constexpr std::size_t kPleOutputWidth          = 16 * kPleRowWidth;
constexpr std::size_t kPleTokenBytes           = kPleOutputWidth * sizeof(std::uint16_t);
constexpr std::size_t kPleCodesPerRow          = 80;
constexpr std::size_t kPleScalesPerRow         = 20;
constexpr std::size_t kPleCompressedRowBytes   = kPleCodesPerRow + kPleScalesPerRow; // 100
constexpr std::size_t kPleCompressedTokenBytes = 16 * kPleCompressedRowBytes;       // 1600

enum class SlotState : std::uint8_t { Idle, Gathering, Copying };

} // namespace

struct PleGatherPipeline::Slot {
    Slot(DeviceContext& device, std::size_t host_bytes, std::size_t dev_bytes)
        : buffer(host_bytes), device_compressed(dev_bytes), completion(device), ordering(device) {}

    PinnedHostBuffer buffer;
    DeviceBuffer device_compressed;
    CudaCompletionEvent completion;
    // Orders the H2D copy after everything already enqueued on the compute stream: the startup
    // zero-fill of the round tensors and the previous round's consumers of `output`. Without it
    // the transfer-stream copy can land before an in-flight memset (WAW) or before the previous
    // PLE layer has read the buffer (WAR).
    CudaCompletionEvent ordering;
    std::vector<std::future<void>> work;
    std::uint64_t generation = 0;
    SlotState state          = SlotState::Idle;
};

PleGatherPipeline::Ticket::Ticket(PleGatherPipeline* owner, std::size_t slot,
                                  std::uint64_t generation, std::size_t tokens) noexcept
    : owner_(owner), slot_(slot), generation_(generation), tokens_(tokens) {}

PleGatherPipeline::PleGatherPipeline(PleTableView table, DeviceContext& device,
                                     std::size_t max_tokens, std::size_t slot_count,
                                     std::uint32_t worker_threads)
    : table_(std::move(table)), device_(device), max_tokens_(max_tokens),
      workers_(worker_threads, slot_count * max_tokens),
      fixed_host_buffer_(max_tokens * kPleTokenBytes) {
    if (max_tokens == 0 || slot_count == 0) {
        throw std::invalid_argument("PLE gather pipeline capacity must be nonzero");
    }
    slots_.reserve(slot_count);
    for (std::size_t slot = 0; slot < slot_count; ++slot) {
        slots_.push_back(std::make_unique<Slot>(
            device_, max_tokens_ * kPleCompressedTokenBytes,
            max_tokens_ * kPleCompressedTokenBytes));
    }
}

PleGatherPipeline::~PleGatherPipeline() {
    for (const std::unique_ptr<Slot>& slot : slots_) {
        for (std::future<void>& work : slot->work) {
            if (work.valid()) {
                try {
                    work.get();
                } catch (...) {}
            }
        }
        if (slot->state == SlotState::Copying) {
            try {
                slot->completion.synchronize();
            } catch (...) {}
        }
    }
}

std::size_t PleGatherPipeline::acquire_slot() {
    for (std::size_t offset = 0; offset < slots_.size(); ++offset) {
        const std::size_t index = (next_slot_ + offset) % slots_.size();
        Slot& slot              = *slots_[index];
        if (slot.state == SlotState::Copying && slot.completion.ready()) {
            slot.state = SlotState::Idle;
        }
        if (slot.state == SlotState::Idle) {
            next_slot_ = (index + 1) % slots_.size();
            return index;
        }
    }
    throw std::runtime_error("PLE gather pipeline has no reusable slot");
}

PleGatherPipeline::Ticket
PleGatherPipeline::prepare(std::span<const std::array<std::int64_t, 16>> global_rows) {
    if (global_rows.empty() || global_rows.size() > max_tokens_) {
        throw std::invalid_argument("PLE gather batch is outside the startup-fixed capacity");
    }
    const std::size_t slot_index = acquire_slot();
    Slot& slot                   = *slots_[slot_index];
    slot.state                   = SlotState::Gathering;
    ++slot.generation;
    slot.work.clear();

    const std::size_t tokens       = global_rows.size();
    auto* base                     = static_cast<std::byte*>(slot.buffer.data());
    const std::size_t codes_bytes  = tokens * 16 * kPleCodesPerRow;
    const std::size_t scales_bytes = tokens * 16 * kPleScalesPerRow;

    gather_ple_rows_compressed(
        table_, global_rows,
        std::span<std::byte>(base, codes_bytes),
        std::span<std::byte>(base + codes_bytes, scales_bytes));

    return Ticket(this, slot_index, slot.generation, tokens);
}

void PleGatherPipeline::enqueue_copy(Ticket&& ticket, Tensor& output) {
    if (ticket.owner_ != this || ticket.slot_ >= slots_.size()) {
        throw std::invalid_argument("PLE gather ticket belongs to another pipeline");
    }
    Slot& slot = *slots_[ticket.slot_];
    if (slot.state != SlotState::Gathering || slot.generation != ticket.generation_) {
        throw std::invalid_argument("PLE gather ticket is stale");
    }
    if (output.dtype != DType::BF16 || output.ne[0] != static_cast<std::int32_t>(kPleOutputWidth) ||
        output.ne[1] != static_cast<std::int32_t>(ticket.tokens_) || output.ne[2] != 1 ||
        output.ne[3] != 1 || !output.is_contiguous() || output.data == nullptr) {
        throw std::invalid_argument("PLE gather output must be contiguous BF16 [2560,tokens]");
    }

    try {
        for (std::future<void>& work : slot.work) { work.get(); }
    } catch (...) {
        slot.work.clear();
        slot.state    = SlotState::Idle;
        ticket.owner_ = nullptr;
        throw;
    }
    slot.work.clear();
    const std::size_t bytes = ticket.tokens_ * kPleCompressedTokenBytes;
    // Payload H2D of gathered PLE rows; not a host control-flow round-trip.
    slot.ordering.record(device_.stream);
    slot.ordering.wait(device_.transfer_stream);
    CUDA_CHECK(cudaMemcpyAsync(slot.device_compressed.p, slot.buffer.data(), bytes,
                               cudaMemcpyHostToDevice, device_.transfer_stream));
    slot.completion.record(device_.transfer_stream);
    slot.completion.wait(device_.stream);
    flash_next_ple_dequant_launch(slot.device_compressed.p, output,
                                  static_cast<int>(ticket.tokens_), device_.stream);
    slot.state    = SlotState::Copying;
    ticket.owner_ = nullptr;
}

void PleGatherPipeline::gather_pinned(
    std::span<const std::array<std::int64_t, 16>> global_rows) {
    if (global_rows.empty() || global_rows.size() > max_tokens_) {
        throw std::invalid_argument("PLE gather batch is outside the startup-fixed capacity");
    }
    auto* output = static_cast<std::uint16_t*>(fixed_host_buffer_.data());

    // Served decode runs at B=1, and this gather sits in the inter-round window that nsys
    // measured at 0.99 ms/token of GPU idle (~1.2 host gaps >100 us per token). At that batch the
    // HostWorkerPool round-trip is two Windows condvar wake-ups (submit notify, then the future
    // wait) to buy ~2560 scalar dequants of parallelism that has nowhere to go. Below the
    // threshold the calling thread does the work itself.
    // Bitwise identity: same gather_ple_rows_bf16 over the same rows into the same
    // token * kPleOutputWidth offsets, so the pinned bytes are unchanged either way. Only the
    // thread that produces them differs.
    constexpr std::size_t kInlineGatherTokens = 2;
    if (global_rows.size() <= kInlineGatherTokens) {
        for (std::size_t token = 0; token < global_rows.size(); ++token) {
            gather_ple_rows_bf16(
                table_, global_rows[token],
                std::span<std::uint16_t>(output + token * kPleOutputWidth, kPleOutputWidth));
        }
        return;
    }

    std::vector<std::future<void>> work;
    work.reserve(global_rows.size());
    for (std::size_t token = 0; token < global_rows.size(); ++token) {
        const std::array<std::int64_t, 16> rows = global_rows[token];
        work.push_back(workers_.submit([this, output, rows, token] {
            gather_ple_rows_bf16(
                table_, rows,
                std::span<std::uint16_t>(output + token * kPleOutputWidth, kPleOutputWidth));
        }));
    }
    for (std::future<void>& w : work) {
        w.get();
    }
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
