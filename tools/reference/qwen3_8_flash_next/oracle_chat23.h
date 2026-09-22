// Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
#pragma once

#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include "targets/qwen3_8_flash_next/impl/text_executor.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

inline constexpr std::array<std::int32_t, 23> kOracleChat23TokenIds = {
    248045, 846,  198,   20206, 303,  799,   2716,  11316, 25,     1092, 3520, 264,
    1603,   42903, 4560, 30,    248046, 198,  248045, 74455, 198,    248068, 198};

// Logits at position p come from one execute_prefill_chunk of tokens[0..p] on a
// fresh lane. Decode rounds never launch the QSA prefill MMA kernel.
// max_positions exists so a test can prove this function reaches the prefill path
// with a single traced prefill: NINFER_FLASH_NEXT_TRACE_STAGES is latched into a
// function-local static on the first prefill of a process, so tracing cannot be
// switched off again and all 23 positions would otherwise be dumped to disk.
inline void dump_oracle_chat23_logits(FlashNextTextExecutor& executor, DeviceContext& device,
                                      const FlashNextRuntimePlan& plan,
                                      const std::filesystem::path& out_dir,
                                      std::int32_t max_positions = 23) {
    std::filesystem::create_directories(out_dir);
    constexpr std::size_t kVocabSize = 248'320;
    const std::int32_t limit = std::min<std::int32_t>(max_positions, 23);
    std::vector<std::uint16_t> host_logits(kVocabSize);
    for (std::int32_t pos = 0; pos < limit; ++pos) {
        const std::size_t n = static_cast<std::size_t>(pos) + 1;
        std::vector<std::int32_t> ids(kOracleChat23TokenIds.begin(),
                                      kOracleChat23TokenIds.begin() + static_cast<std::ptrdiff_t>(n));
        std::vector<std::array<std::int32_t, 3>> positions(n);
        for (std::int32_t i = 0; i <= pos; ++i) {
            positions[static_cast<std::size_t>(i)] = {i, i, i};
        }
        auto lane  = executor.allocate_lane();
        auto round = executor.execute_prefill_chunk(lane, ids, positions, /*first_token_index=*/0,
                                                    nullptr);
        device.synchronize();
        CUDA_CHECK(cudaMemcpy(host_logits.data(), round.logits().data,
                              kVocabSize * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
        char name[32];
        std::snprintf(name, sizeof(name), "pos%04d_logits.bin", pos);
        const auto path = out_dir / name;
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(host_logits.data()),
                  static_cast<std::streamsize>(host_logits.size() * sizeof(std::uint16_t)));
        if (!out) {
            throw std::runtime_error("failed to write " + path.string());
        }
        std::vector<LaneCommitDecision> accept{{.accept = true}};
        round.commit(accept);
        executor.release_lane(lane);
    }
    std::cout << "Wrote " << limit << " oracle logit dumps to " << out_dir.generic_string()
              << " qsa_prefill_mma=" << (plan.config.use_qsa_prefill_mma ? "on" : "off") << std::endl;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
