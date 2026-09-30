#pragma once

// ============================================================================================
// stream_capture -- ONE READER for the question "is this stream being captured RIGHT NOW?"
//
// THE RULE, AND WHY IT NEEDS A GUARD RATHER THAN A COMMENT.
// `cudaStreamSynchronize` and a HOST-destination `cudaMemcpyAsync` are ILLEGAL while the stream
// is under `cudaStreamBeginCapture` (src/core/decode_graph.cpp). CUDA rejects them AND invalidates
// the capture, and the sticky error then aborts the process at the NEXT `CUDA_CHECK` as
// `cudaErrorStreamCaptureInvalidated`, rc=134 (SIGABRT). That is an API-state fault, not a memory
// fault, and it is a HOST abort: it leaves no device footprint.
//
// THREE diagnostics in this tree photograph device tensors for the HOST with exactly those two
// calls. Each one documented the incompatibility in its own comment, and none of them guarded
// itself -- so each one turned a diagnostic into a process kill the moment CUDA graphs (the
// shipped default, `use_cuda_graph = true`) were on:
//
//   * `targets/qwen3_6/impl/runtime/speculative_target_impl.h` -- the `NINFER_ACCEPTLOG` dump.
//     This one was ALREADY REPAIRED, by the `acceptlog_capturing()` guard of ACCEPTLOG-CAPTURE-FIX;
//     that function's body IS this predicate, and it now DELEGATES here, so the repair below and
//     that repair cannot drift into two spellings of one rule.
//   * `ops/wrapper/speculative_round.cpp` -- the `[accmask]` ACCMASK-ONE-SHOT readback. This is
//     the one that kills a `--draft-tree L>1` round under capture (measured at
//     `ops/launcher/speculative_round.cu:97`), and it is REPAIRED HERE.
//   * `ops/common/ft_stats.h` -- the FreeToken `observe()` sample, which kills `--ft-stats on`
//     under capture (measured at `ops/launcher/gqa_attention_decode_smallt.cu:307`). REPAIRED HERE.
//
// WHY A RUNTIME QUERY AND NOT A FLAG, A THREADED PARAMETER, OR A BUILD OPTION.
// `cudaStreamIsCapturing` is legal DURING capture -- it is the query CUDA provides for exactly this
// branch -- and it answers for the stream the round is actually launched on. A build-time flag, or
// a parameter threaded down from the caller, would be WRONG the moment one graph is captured and
// another is not: a capture body is the graph DEFINITION and runs ONCE, while every later round of
// the same width is only a `cudaGraphLaunch` (`src/core/graph_impl.h`). The caller cannot know
// which kind of round it is issuing; the stream can be asked.
//
// WHAT A GUARD HERE DOES NOT DO. It does not weaken a check, remove a refusal, or change any
// numeric path. It suppresses an OBSERVATION, and every site that uses it must NAME the
// observation it suppressed -- a skipped readback that prints nothing is an instrument that has
// silently become a no-op, which this record forbids.
// ============================================================================================

#include <cuda_runtime.h>

namespace ninfer::ops {

// Is a capture running on `stream` RIGHT NOW?
// Returns FALSE when the query itself fails, which is the safe direction: the guarded
// (unconditional) path is the one that aborts the process, so an unanswerable query must fall back
// to issuing the CUDA call, exactly as the pre-guard code did.
inline bool stream_is_capturing(cudaStream_t stream) noexcept {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    const cudaError_t err          = cudaStreamIsCapturing(stream, &status);
    return err == cudaSuccess && status != cudaStreamCaptureStatusNone;
}

} // namespace ninfer::ops
