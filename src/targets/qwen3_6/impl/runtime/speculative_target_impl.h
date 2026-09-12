#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::GqaExecutionEnvelope envelope) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens, *frame.feature_sink);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens);
    }
    ops::speculative_accept_greedy_drafts(frame.target_tokens, frame.target_logits, frame.drafts,
                                          frame.current_extents, frame.frontiers, frame.anchors,
                                          frame.licensed_tokens, frame.licensed_counts,
                                          frame.accepted_drafts, TextConfig::token_domain,
                                          frame.sampling, frame.draft_candidate_ids,
                                          frame.draft_candidate_probs, execution.work,
                                          execution.device.stream);
    if (std::getenv("NINFER_ACCEPTLOG") != nullptr) {
        static int acceptlog_round = 0;
        const auto dump_i32 = [](const Tensor& view, int count, const char* name) {
            if (view.data == nullptr || count <= 0) {
                std::fprintf(stderr, "    %s: (empty)\n", name);
                return;
            }
            const std::size_t words = static_cast<std::size_t>(view.bytes()) / sizeof(std::int32_t);
            std::vector<std::int32_t> host(words > 0 ? words : 1, 0);
            cudaMemcpy(host.data(), view.data, view.bytes(), cudaMemcpyDeviceToHost);
            const std::size_t row = static_cast<std::size_t>(view.nb[0]) / sizeof(std::int32_t);
            std::fprintf(stderr, "    %-9s:", name);
            for (int i = 0; i < count; ++i) {
                const std::size_t off = static_cast<std::size_t>(i) * row;
                std::fprintf(stderr, " %d", off < host.size() ? host[off] : -1);
            }
            std::fprintf(stderr, "\n");
        };
        const int width = frame.target_tokens.ne[0];
        const int drafts_n = frame.drafts.ne[0];
        std::fprintf(stderr, "[acceptlog] round=%d width=%d k=%d\n", acceptlog_round++, width,
                     drafts_n);
        dump_i32(frame.current_extents, 1, "extent");
        dump_i32(frame.accepted_drafts, 1, "accepted");
        dump_i32(frame.licensed_counts, 1, "licensed_n");
        dump_i32(frame.anchors, 1, "anchor");
        dump_i32(frame.drafts, drafts_n, "drafts");
        dump_i32(frame.target_tokens, width, "targets");
        dump_i32(frame.licensed_tokens, width, "emitted");
    }
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.state_destination_slots, continuation_hidden_store,
                 execution.device.stream);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
