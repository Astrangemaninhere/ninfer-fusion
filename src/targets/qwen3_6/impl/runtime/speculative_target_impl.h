#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"
#include "ops/stream_capture.h"   // F881: the one capture-predicate reader

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

namespace {

// ============================================================================================
// acceptlog: capture-safe dump plumbing (ACCEPTLOG-CAPTURE-FIX).
//
// `NINFER_ACCEPTLOG` prints one detail block per MTP/DFlash verify round. Producing that block
// needs the round's device tensors on the HOST: a `cudaStreamSynchronize` of the round's stream
// followed by the legacy-default-stream `cudaMemcpy` of the seven tensors. BOTH are illegal while
// that stream is being captured (`cudaStreamBeginCapture`, src/core/decode_graph.cpp:65): CUDA
// rejects them AND invalidates the capture, and the sticky error then aborts the process at the
// next CUDA_CHECK -- measured at src/ops/launcher/speculative_round.cu:125 as
// `cudaErrorStreamCaptureInvalidated`, rc=134 (SIGABRT). A process with NINFER_ACCEPTLOG set
// therefore died on its FIRST capture, which made the instrument and CUDA graphs mutually
// exclusive.
//
// A capture body is the graph DEFINITION: it runs once, and every later round is only a
// `cudaGraphLaunch` from schedule::run_prepared (graph_impl.h:11-21). The dump is therefore split:
//   * acceptlog_record() -- runs in EVERY round body, captured or not, and issues NO CUDA call. It
//                           photographs the seven tensors (pointer / stride / count) plus the
//                           stream the round is launched on.
//   * acceptlog_emit()   -- the dump itself (sync + legacy D2H copy + printf), unchanged from the
//                           pre-fix body, and only ever called with no capture in flight.
//   * acceptlog_replay_dump(width) -- called by the graph-round entry points (mtp_impl.h,
//                           dflash_impl.h, dflash2_impl.h) right after their launch, i.e. on the
//                           host, outside capture, after that round's kernels: the round's own
//                           post-round read.
// With graphs OFF nothing changes at all: no capture is ever in flight, so the eager path below
// calls exactly the dump it always called.
// ============================================================================================
struct AcceptlogField {
    const void* data  = nullptr;
    std::size_t bytes = 0;
    std::size_t row   = 0;
    int count         = 0;
};

struct AcceptlogSlots {
    bool recorded       = false;
    int width           = 0;      // columns: frame.target_tokens.ne[0]
    int window          = 0;      // drafts:  frame.drafts.ne[0]
    cudaStream_t stream = nullptr;
    AcceptlogField field[7];
};

// The ladder is adaptive, so ONE process captures several widths and a replay must print the width
// ITS rung captured => the photograph is keyed by the column count. k + 1 <=
// kMtpDecodeMaximumDrafts + 1 = 16 in the widest configuration; 64 is slack, not a reachable limit.
constexpr int kAcceptlogWidthLimit = 64;

AcceptlogSlots* acceptlog_slot_table() {
    static AcceptlogSlots slots[kAcceptlogWidthLimit];
    return slots;
}

bool acceptlog_enabled() { return std::getenv("NINFER_ACCEPTLOG") != nullptr; }

// LOG-SYNC env gate (added on top of FIX-SMALL's repair) -- Default ON: the ordering the comment
// block inside target_verify_accept describes IS the fix. `NINFER_ACCEPTLOG_STREAM_SYNC=0` restores
// the pre-fix behaviour (the unordered legacy-stream cudaMemcpy), so the defect and its repair can
// be compared on ONE binary, in ONE window, with this variable as the only difference. Inert unless
// NINFER_ACCEPTLOG is set. Unchanged by this fix.
bool acceptlog_order_dump() {
    const char* const env = std::getenv("NINFER_ACCEPTLOG_STREAM_SYNC");
    return !(env != nullptr && env[0] == '0' && env[1] == '\0');
}

// Is a capture running on `stream` RIGHT NOW?
// F881 -- the predicate is NOT owned here. It is owned by ONE reader, `ops/stream_capture.h`,
// which the `[accmask]` and FreeToken repairs also call; this function is kept as the name the
// rest of this header already calls (`:161`, `:216`) so those call sites did not have to move.
// REPAIRING here and not there, or the other way round, is how one rule becomes two spellings.
bool acceptlog_capturing(cudaStream_t stream) { return ninfer::ops::stream_is_capturing(stream); }

void acceptlog_record(const Tensor& current_extents, const Tensor& accepted_drafts,
                      const Tensor& licensed_counts, const Tensor& anchors, const Tensor& drafts,
                      const Tensor& target_tokens, const Tensor& licensed_tokens, int width,
                      int window, cudaStream_t stream) {
    if (width < 1 || width >= kAcceptlogWidthLimit) { return; }
    const Tensor* const views[7] = {&current_extents, &accepted_drafts, &licensed_counts,
                                    &anchors,         &drafts,          &target_tokens,
                                    &licensed_tokens};
    AcceptlogSlots& slots = acceptlog_slot_table()[width];
    slots                 = AcceptlogSlots{};
    slots.recorded        = true;
    slots.width           = width;
    slots.window          = window;
    slots.stream          = stream;
    for (int i = 0; i < 7; ++i) {
        // The counts are the ones the pre-fix dump passed: one row for each of the four scalars,
        // the whole ingress window for the drafts, the round's own column count for the targets and
        // the emitted tokens.
        const int count = i == 4 ? window : i >= 5 ? width : 1;
        if (views[i]->data == nullptr) { continue; }   // prints "(empty)", exactly as before
        slots.field[i] = AcceptlogField{views[i]->data, views[i]->bytes(),
                                        static_cast<std::size_t>(views[i]->nb[0]), count};
    }
}

void acceptlog_dump_field(const AcceptlogField& field, const char* name) {
    if (field.data == nullptr || field.count <= 0) {
        std::fprintf(stderr, "    %s: (empty)\n", name);
        return;
    }
    const std::size_t words = static_cast<std::size_t>(field.bytes) / sizeof(std::int32_t);
    std::vector<std::int32_t> host(words > 0 ? words : 1, 0);
    cudaMemcpy(host.data(), field.data, field.bytes, cudaMemcpyDeviceToHost);
    const std::size_t row = field.row / sizeof(std::int32_t);
    std::fprintf(stderr, "    %-9s:", name);
    for (int i = 0; i < field.count; ++i) {
        const std::size_t off = static_cast<std::size_t>(i) * row;
        std::fprintf(stderr, " %d", off < host.size() ? host[off] : -1);
    }
    std::fprintf(stderr, "\n");
}

const char* const kAcceptlogNames[7] = {"extent", "accepted", "licensed_n", "anchor",
                                        "drafts", "targets",  "emitted"};

// The dump. NEVER called with a capture in flight -- that is the whole point of the split.
void acceptlog_emit(const AcceptlogSlots& slots) {
    if (acceptlog_order_dump()) { cudaStreamSynchronize(slots.stream); }
    static int acceptlog_round = 0;
    std::fprintf(stderr, "[acceptlog] round=%d width=%d k=%d\n", acceptlog_round++, slots.width,
                 slots.window);
    for (int i = 0; i < 7; ++i) { acceptlog_dump_field(slots.field[i], kAcceptlogNames[i]); }
}

void acceptlog_capture_notice() {
    static bool notice_printed = false;
    if (notice_printed) { return; }
    notice_printed = true;
    std::fprintf(stderr,
                 "[acceptlog] stream capture in progress: no sync and no cudaMemcpy may be issued "
                 "while capturing, so the block of the round being captured now is deferred to the "
                 "per-round replay dump\n");
}

} // namespace

// Called by every graph-round entry point right after its cudaGraphLaunch: the captured body did
// not run for this round, so the photograph taken at capture time is the only way to publish the
// round's block. Inert unless NINFER_ACCEPTLOG is set.
void acceptlog_replay_dump(int width) {
    if (!acceptlog_enabled()) { return; }
    if (width < 1 || width >= kAcceptlogWidthLimit) { return; }
    const AcceptlogSlots& slots = acceptlog_slot_table()[width];
    if (!slots.recorded) { return; }
    if (acceptlog_capturing(slots.stream)) { return; }
    acceptlog_emit(slots);
}

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::GqaExecutionEnvelope envelope) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.column_masks, frame.kv_table_rows,
                                 frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens, *frame.feature_sink);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.column_masks, frame.kv_table_rows,
                                 frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens);
    }
    ops::speculative_accept_greedy_drafts(frame.target_tokens, frame.target_logits, frame.drafts,
                                          frame.current_extents, frame.column_masks,
                                          frame.frontiers, frame.anchors, frame.licensed_tokens,
                                          frame.licensed_counts, frame.accepted_drafts,
                                          frame.accepted_columns, TextConfig::token_domain,
                                          frame.sampling, frame.draft_candidate_ids,
                                          frame.draft_candidate_probs, execution.work,
                                          execution.device.stream);
    if (std::getenv("NINFER_ACCEPTLOG") != nullptr) {
        // ORDER THE DUMP AFTER THIS ROUND'S KERNELS. `execution.device.stream` is created with
        // cudaStreamNonBlocking (src/core/device.cu:64), so the legacy-default-stream blocking
        // cudaMemcpy below does NOT order against the accept launch at :33-40 and would read the
        // previous round's contents of accepted_drafts / licensed_counts / licensed_tokens --
        // measured as an exact one-round lag against the published token stream on 5/5 arms
        // (chain_1 / chain_2 / chain_14 / tree_2_1 / tree_2_7), and the source of the apparent
        // `accepted > extent` rows (chain_14 round 41: dump shows acc=7 ext=2). `extent` is the
        // ingress view (round_state.cpp:256-257), decided on the host before the launch, which is
        // why it was never the stale one. The host-side consumer of the same egress does this
        // correctly: program_impl.h:13974-13979 synchronizes before reading at :13992-14023.
        // ---- LOG-SYNC env gate (added on top of FIX-SMALL's repair) --------------------
        // Default ON: the ordering above IS the fix. `NINFER_ACCEPTLOG_STREAM_SYNC=0` restores
        // the pre-fix behaviour (the unordered legacy-stream cudaMemcpy below), so the defect and
        // its repair can be compared on ONE binary, in ONE window, with this variable as the only
        // difference. Inert unless NINFER_ACCEPTLOG is set.
        const int width    = frame.target_tokens.ne[0];
        const int drafts_n = frame.drafts.ne[0];
        // Photograph the round (no CUDA call) BEFORE deciding how to publish it: with a capture in
        // flight the publication has to wait for the round's graph launch (acceptlog_replay_dump).
        acceptlog_record(frame.current_extents, frame.accepted_drafts, frame.licensed_counts,
                         frame.anchors, frame.drafts, frame.target_tokens, frame.licensed_tokens,
                         width, drafts_n, execution.device.stream);
        if (acceptlog_capturing(execution.device.stream)) {
            // CAPTURE-SAFE branch: this body IS the graph definition being captured right now
            // (src/core/decode_graph.cpp:65). The `cudaStreamSynchronize` and the legacy-stream
            // `cudaMemcpy` the dump needs are ILLEGAL while a stream is capturing: they invalidate
            // the capture and the sticky error then aborts the process at the next CUDA_CHECK
            // (src/ops/launcher/speculative_round.cu:125, cudaErrorStreamCaptureInvalidated,
            // SIGABRT rc=134). So NOTHING is issued here and the block is deferred to
            // acceptlog_replay_dump(), which every graph-round entry point calls after its launch.
            acceptlog_capture_notice();
        } else {
            const AcceptlogSlots& acceptlog_slots = acceptlog_slot_table()[width];
            if (acceptlog_slots.recorded) { acceptlog_emit(acceptlog_slots); }
        }
    }
    // The continuation state is taken at the ACCEPTED COLUMN. `accepted_columns` is written by the
    // greedy accept in both modes and equals `accepted` for a chain, so the tree gate below is the
    // only thing that changes which tensor is read -- and a --draft-tokens round reads the very
    // same one it has always read. (The sampling and penalty routes do not publish a column, and no
    // tree round can reach them: the runtime refuses that combination before the round.)
    const Tensor& continuation_column =
        frame.column_masks.data != nullptr ? frame.accepted_columns : frame.accepted_drafts;
    ops::speculative_select_accepted_hidden(frame.target_hidden, continuation_column,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.state_destination_slots, continuation_hidden_store,
                 execution.device.stream);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
