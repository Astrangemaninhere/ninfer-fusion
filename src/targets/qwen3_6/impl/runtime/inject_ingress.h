#pragma once

// src/targets/qwen3_6/impl/runtime/inject_ingress.h -- THE ENGINE SIDE OF THE INGRESS, i.e. the
// place a declaration from src/spec/inject_channel.h becomes a write into the forward pass.
//
// WHY THIS IS A SEPARATE HEADER AND NOT PART OF THE CHANNEL. spec/inject_channel.h is host-only and
// std-only on purpose (its own comment says why): the whole DECLARATION surface -- every field,
// every refusal name, the digest, the dtype arithmetic -- is then testable with plain g++ and no
// device. What cannot be tested that way is the APPLICATION: one cudaMemcpy into the
// input-embedding matrix at the right offset, made once per prefill chunk. So the application
// lives here, next to the runtime it writes into, and it does three things and no others:
//
//   1. it reads the spec ONCE per process and prints the admission or the refusal BY NAME;
//   2. per prefill chunk, it copies the declared columns of the payload into the columns of the
//      input-embedding matrix (ingest) or out of them (egress) -- ONE contiguous cudaMemcpy, since
//      the matrix is token-major (core/tensor.cpp:51-56) and the declared range is contiguous;
//   3. it refuses, by name and before consuming anything, any chunk whose vision scatter would
//      collide with the declared columns, and it reports at the end whether the declared range was
//      actually covered.
//
// WHY AN ENVIRONMENT VARIABLE AND NOT A REQUEST FIELD. The consumption point is inside a device
// schedule that takes no per-request argument of its own (`prefill_impl`'s signature is the
// prompt and a tap), and the tree already carries exactly this idiom for exactly this kind of
// data: `kvdump_dir()` here, and NINFER_HS_DUMP_DIR in text_prefill_impl.h, both cache a getenv
// and both move NON-TOKEN bytes (bf16 hidden states) across the host/device boundary. The CLI
// spelling `--inject-spec` exists (apps/cli/options.cpp) and is the operator's surface; it
// commits the same variable at parse time, which is the precedent `--ft-stats` sets
// ("the flag wins and is committed to that variable at parse time, because ft::enabled() reads it
// exactly once").
//
// IT IS NOT A BACK DOOR: a run with NINFER_INJECT_SPEC unset has `configured == false` and
// `apply()` returns on its first line, so the cost on every existing path is one branch per
// prefill chunk and the bytes written are none. That property is what makes "大动土木" safe here.

#include "spec/inject_channel.h"

#include "core/device.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::inject_ingress {

// The environment spelling of the declaration. One variable, whose VALUE is the path of the spec
// file -- so the declaration itself has a grammar and a refusal table rather than being a
// comma-separated string in an environment variable, which is how a field stops being checkable.
inline constexpr const char* kSpecVariable = "NINFER_INJECT_SPEC";

// Process-wide, because the declaration is a property of the RUN and not of one prefill chunk.
struct Runtime {
    bool configured = false;   // configure() has run (whether or not it admitted)
    bool admitted   = false;   // an admission exists and may be applied
    bool announced  = false;   // the ADMITTED line has been printed (once, not per chunk)
    spec::inject::Declaration declaration;
    spec::inject::PayloadSettlement payload;
    // Coverage, as a BITMAP rather than a counter, so that a schedule that records and then
    // executes the same prefill twice -- which TextContext does, being a stack object per
    // recording/execution -- covers each declared column once for the purposes of the
    // completeness check instead of reporting twice as many columns as were declared. The bitmap
    // itself lives in the channel header so that its verdict is host-testable.
    spec::inject::Coverage coverage;
    std::int32_t chunks_touched = 0;
    // Egress accumulation: the declared range's bytes, in the payload's own token-major order.
    std::vector<std::uint16_t> drained;
    bool dumped     = false;
    bool summarised = false;
};

[[nodiscard]] inline Runtime& runtime() noexcept {
    static Runtime one;
    return one;
}

// The spec, read once. `limits` is what the ENGINE can consume and is supplied here rather than
// declared by the caller, which is the whole point of the split between Declaration and
// ModelLimits in the channel header.
inline void configure(const spec::inject::ModelLimits& limits) {
    Runtime& state = runtime();
    if (state.configured) { return; }
    state.configured = true;

    const char* spec_path = std::getenv(kSpecVariable);
    if (spec_path == nullptr || *spec_path == '\0') { return; }

    spec::inject::ParseResult parsed = spec::inject::parse_spec_file(spec_path);
    if (!parsed.ok()) {
        std::fprintf(stderr, "%s (spec %s)\n",
                     spec::inject::render_refusal(parsed.settlement.refusal,
                                                  parsed.settlement.field,
                                                  parsed.settlement.detail)
                         .c_str(),
                     spec_path);
        std::fflush(stderr);
        throw std::runtime_error(std::string("inject channel: ") +
                                 spec::inject::refusal_name(parsed.settlement.refusal) +
                                 " in spec " + spec_path);
    }
    spec::inject::Settlement settlement =
        spec::inject::admit_against_engine(parsed.declaration, limits);
    if (!settlement.admitted()) {
        std::fprintf(stderr, "%s (spec %s)\n",
                     spec::inject::render_refusal(settlement.refusal, settlement.field,
                                                  settlement.detail)
                         .c_str(),
                     spec_path);
        std::fflush(stderr);
        throw std::runtime_error(std::string("inject channel: ") +
                                 spec::inject::refusal_name(settlement.refusal) + " (field " +
                                 settlement.field + ") in spec " + spec_path);
    }
    // F745 injectbind: THE PAYLOAD IS AN OUTPUT WHEN direction=egress, and `bake()` READS it.
    // Called unconditionally, this made an egress declaration require a correctly sized, finite file
    // to exist BEFORE the run that writes it -- so the direction that PRODUCES the bytes could not
    // be used to produce them (it refused `refused-file-missing`, and the `flush_dump()` below was
    // unreachable). Measured by line injectbind by calling the three functions this function calls,
    // in this order, on an egress declaration whose path does not exist. The settlement stays for
    // ingest, which is the direction it was written for; egress settles to the EMPTY settlement --
    // refusal None, no bytes read, both digests 0 -- and the ADMITTED line it renders therefore
    // reports bytes=0, which is the truth about an egress declaration at admission time. The egress
    // run's own numbers are the DRAINED line's `written_digest`, printed from the bytes it wrote.
    if (parsed.declaration.direction == spec::inject::Direction::Ingest) {
        state.payload = spec::inject::bake(parsed.declaration);
    }
    if (!state.payload.settled()) {
        std::fprintf(stderr, "%s (spec %s)\n",
                     spec::inject::render_refusal(state.payload.refusal, state.payload.field,
                                                  state.payload.detail)
                         .c_str(),
                     spec_path);
        std::fflush(stderr);
        throw std::runtime_error(std::string("inject channel: ") +
                                 spec::inject::refusal_name(state.payload.refusal) + " (field " +
                                 state.payload.field + ") in spec " + spec_path);
    }
    state.declaration = parsed.declaration;
    state.admitted    = true;
    state.coverage.reset(parsed.declaration.cols);
    if (parsed.declaration.direction == spec::inject::Direction::Egress) {
        state.drained.assign(
            static_cast<std::size_t>(parsed.declaration.cols) *
                static_cast<std::size_t>(parsed.declaration.rows),
            0U);
    }
    // THE OBSERVABILITY REQUIREMENT, met here and only here: the admitted line carries every
    // declared field plus the two measurements that are not in the declaration -- the byte count
    // consumed and the digest of the bytes consumed.
    std::fprintf(stderr, "%s\n",
                 spec::inject::render_admission(state.declaration, state.payload).c_str());
    std::fflush(stderr);
    state.announced = true;
}

// Write the drained columns out, in one file, one digest. Called after every chunk rather than
// only at the end: a single-chunk prompt (the shape every acceptance arm uses) is then complete
// after its first chunk, and a run that never reaches its final chunk still leaves the bytes it
// actually produced rather than nothing.
inline void flush_dump() {
    Runtime& state = runtime();
    if (!state.admitted || state.declaration.direction != spec::inject::Direction::Egress) {
        return;
    }
    std::ofstream out(state.declaration.path, std::ios::binary | std::ios::trunc);
    if (!out) {
        const std::string detail = "'" + state.declaration.path + "' could not be opened for writing";
        std::fprintf(stderr, "%s\n",
                     spec::inject::render_refusal(spec::inject::Refusal::RefusedFileMissing, "path",
                                                  detail)
                         .c_str());
        std::fflush(stderr);
        throw std::runtime_error("inject channel: " + std::string(
                                     spec::inject::refusal_name(spec::inject::Refusal::RefusedFileMissing)));
    }
    out.write(reinterpret_cast<const char*>(state.drained.data()),
              static_cast<std::streamsize>(state.drained.size() * sizeof(std::uint16_t)));
    out.close();
    const std::uint64_t digest = spec::inject::fnv1a64(
        reinterpret_cast<const std::uint8_t*>(state.drained.data()),
        state.drained.size() * sizeof(std::uint16_t));
    if (!state.dumped) {
        char rendered[80];
        std::snprintf(rendered, sizeof(rendered), "fnv1a64:0x%016llx",
                      static_cast<unsigned long long>(digest));
        std::fprintf(stderr,
                     "[inject] DRAINED direction=egress positions=[%d..%d] columns=%d of %d "
                     "bytes=%zu written_digest=%s path=%s\n",
                     state.declaration.position0, state.declaration.last_position(),
                     state.coverage.count, state.declaration.cols,
                     state.drained.size() * sizeof(std::uint16_t), rendered,
                     state.declaration.path.c_str());
        std::fflush(stderr);
        state.dumped = true;
    }
}

// THE APPLICATION. One call per prefill chunk, immediately after `ops::embedding` has filled `x`
// (and before the vision scatter, so that a collision is refused rather than overwritten).
//
//   x               BF16 [hidden, columns] contiguous, token-major, column c holding absolute
//                   position (position_base + c)
//   position_base   the absolute position of x's column 0
//   vision_columns  the vision path's destination columns for THIS chunk, in the same local
//                   column space as x; empty when the chunk carries no vision item
[[nodiscard]] inline bool apply(Tensor& x, std::int32_t position_base,
                                std::span<const std::int32_t> vision_columns,
                                cudaStream_t stream) {
    Runtime& state = runtime();
    if (!state.admitted) { return false; }
    if (x.dtype != DType::BF16 || x.data == nullptr || !x.is_contiguous() ||
        x.ne[2] != 1 || x.ne[3] != 1) {
        // A guard on the ENGINE's own buffer, not on the caller's declaration: if the matrix this
        // channel is about to write into is not the tensor the whole contract is written against,
        // then the contract does not hold and the write must not happen.
        const std::string detail =
            "the input-embedding matrix is not a contiguous BF16 [hidden, columns] tensor";
        std::fprintf(stderr, "%s\n",
                     spec::inject::render_refusal(spec::inject::Refusal::RefusedRowsMismatch,
                                                  "rows", detail)
                         .c_str());
        std::fflush(stderr);
        throw std::runtime_error("inject channel: " + std::string(
                                     spec::inject::refusal_name(spec::inject::Refusal::RefusedRowsMismatch)));
    }
    const std::int32_t hidden = x.ne[0];
    const std::int32_t columns = x.ne[1];
    if (hidden != state.declaration.rows) {
        const std::string detail = "the matrix has " + std::to_string(hidden) +
                                   " rows, the declaration named " +
                                   std::to_string(state.declaration.rows);
        std::fprintf(stderr, "%s\n",
                     spec::inject::render_refusal(spec::inject::Refusal::RefusedRowsMismatch,
                                                  "rows", detail)
                         .c_str());
        std::fflush(stderr);
        throw std::runtime_error("inject channel: " + std::string(
                                     spec::inject::refusal_name(spec::inject::Refusal::RefusedRowsMismatch)));
    }

    // The intersection of [position0, last_position] with this chunk's [position_base,
    // position_base + columns). Everything below is in x's own column space.
    const std::int32_t p0 = state.declaration.position0;
    const std::int32_t p1 = state.declaration.last_position();
    const std::optional<spec::inject::Overlap> overlap =
        spec::inject::overlap_with_chunk(state.declaration, position_base, columns);
    if (!overlap.has_value()) { return false; }
    const std::int32_t first = overlap->first_column;
    const std::int32_t count = overlap->count;

    // THE COLLISION GUARD, and it refuses rather than resolving a precedence: the vision path
    // writes whole columns of the same matrix, and a declared column that is also a vision
    // destination would depend on the order of two writes for its value -- which is not a
    // precedence rule anybody stated. Refused by name, before either write.
    const std::int32_t collision =
        spec::inject::first_collision(*overlap, vision_columns.data(), vision_columns.size());
    if (collision >= 0) {
        const std::string detail = "declared column at absolute position " +
                                   std::to_string(position_base + collision) +
                                   " is also a vision scatter destination in this chunk";
        std::fprintf(stderr, "%s\n",
                     spec::inject::render_refusal(
                         spec::inject::Refusal::RefusedOverlapsVisionScatter, "position0", detail)
                         .c_str());
        std::fflush(stderr);
        throw std::runtime_error(
            "inject channel: " +
            std::string(spec::inject::refusal_name(
                spec::inject::Refusal::RefusedOverlapsVisionScatter)));
    }

    // The payload is token-major over the DECLARED columns, so the element offset of absolute
    // position p is (p - position0) * hidden. One contiguous copy, because consecutive columns of
    // x are consecutive in memory (ne[0] is the fastest-varying dimension).
    const std::size_t element_offset = overlap->payload_element_offset;
    const std::size_t bytes =
        static_cast<std::size_t>(count) * static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
    auto* matrix = static_cast<std::uint8_t*>(x.data) +
                   static_cast<std::size_t>(first) * static_cast<std::size_t>(hidden) *
                       sizeof(std::uint16_t);

    if (state.declaration.direction == spec::inject::Direction::Ingest) {
        CUDA_CHECK(cudaMemcpyAsync(matrix,
                                   reinterpret_cast<const std::uint8_t*>(state.payload.payload.data()) +
                                       element_offset * sizeof(std::uint16_t),
                                   bytes, cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<std::uint8_t*>(state.drained.data()) +
                element_offset * sizeof(std::uint16_t),
            matrix, bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    for (std::int32_t i = 0; i < count; ++i) {
        state.coverage.cover((position_base + first + i) - p0);
    }
    ++state.chunks_touched;
    if (state.declaration.direction == spec::inject::Direction::Egress) { flush_dump(); }

    std::fprintf(stderr,
                 "[inject] APPLIED direction=%s chunk=[%d..%d) columns=[%d..%d) of the declared "
                 "[%d..%d] %d column(s) this chunk, %d of %d so far\n",
                 state.declaration.direction == spec::inject::Direction::Ingest ? "ingest" : "egress",
                 position_base, position_base + columns, position_base + first,
                 position_base + first + count - 1, p0, p1, count, state.coverage.count,
                 state.declaration.cols);
    std::fflush(stderr);
    return true;
}

// THE COMPLETENESS VERDICT, printed once, when the run says its prefill is final. A declaration
// that was admitted and then only partly covered is NOT an admission: it is a payload that
// half-entered the context, which is the silent-corruption shape this record keeps finding, so it
// is refused by name and the run fails rather than continuing with a partial injection.
inline void finish() {
    Runtime& state = runtime();
    if (!state.admitted || state.summarised) { return; }
    state.summarised = true;
    if (!state.coverage.complete()) {
        const std::string detail =
            std::to_string(state.coverage.count) + " of " +
            std::to_string(state.declaration.cols) + " declared column(s) were covered by " +
            std::to_string(state.chunks_touched) + " chunk(s); the declared range [" +
            std::to_string(state.declaration.position0) + ".." +
            std::to_string(state.declaration.last_position()) +
            "] was not fully reached by this run's prefill";
        std::fprintf(stderr, "%s\n",
                     spec::inject::render_refusal(
                         spec::inject::Refusal::RefusedRangeNotFullyApplied, "position0", detail)
                         .c_str());
        std::fflush(stderr);
        throw std::runtime_error(
            "inject channel: " +
            std::string(spec::inject::refusal_name(
                spec::inject::Refusal::RefusedRangeNotFullyApplied)));
    }
    std::fprintf(stderr,
                 "[inject] COMPLETE direction=%s positions=[%d..%d] %d of %d column(s) covered by "
                 "%d chunk(s) ingested_digest=fnv1a64:0x%016llx\n",
                 state.declaration.direction == spec::inject::Direction::Ingest ? "ingest" : "egress",
                 state.declaration.position0, state.declaration.last_position(),
                 state.coverage.count, state.declaration.cols, state.chunks_touched,
                 static_cast<unsigned long long>(state.payload.baked_digest));
    std::fflush(stderr);
}

} // namespace ninfer::targets::qwen3_6::detail::inject_ingress
