// Flash-Next (qwen3.8-flash-next) port: REFUSING STUB, not an implementation.
//
// WHY THIS FILE EXISTS
// --------------------
// The copied target `src/targets/qwen3_8_flash_next` calls into the donor's structured-output
// constraint machinery (runtime::OutputConstraintState / OutputConstraintCompiler /
// CompiledOutputConstraint / validate_structured_output).  The donor implements those in
// `src/runtime/contract/structured_output.cpp`, which was NOT taken with the copy-in, so the
// target archive has undefined references to every one of them.
//
// That donor file CANNOT be taken as-is: it includes <xgrammar/xgrammar.h> and the donor vendors
// the library as `third_party/xgrammar` (donor CMakeLists.txt:129 `add_subdirectory(
// third_party/xgrammar)`, donor src/CMakeLists.txt:41-42 `target_sources(ninfer_core PRIVATE
// runtime/contract/structured_output.cpp)` + `target_link_libraries(ninfer_core PRIVATE
// ninfer_xgrammar)`).  THIS TREE DOES NOT HAVE XGRAMMAR -- `third_party/` holds only
// cpp-httplib, nlohmann and utf8proc, and no system copy exists.  Taking the donor file would
// therefore just move the failure from "undefined symbols" to "cannot find xgrammar".
//
// WHAT THIS STUB DOES, AND WHAT IT DELIBERATELY DOES NOT DO
// --------------------------------------------------------
// It is *not* a silent no-op that pretends structured output works.  It refuses:
//   * validate_structured_output() and OutputConstraintCompiler::compile() accept only
//     StructuredOutputKind::Text and throw std::invalid_argument for JsonObject/JsonSchema,
//     naming the missing dependency;
//   * compile(Text) returns nullptr, i.e. "no constraint", which is the correct meaning of
//     Text -- and it is also the only value this tree can ever produce;
//   * every OutputConstraintState method throws std::logic_error, because a state can only be
//     constructed from a non-null compiled constraint and this stub never produces one.
//
// Consequence for the target: a Text request on a Flash-Next artifact behaves exactly as the
// donor's Text path (no mask, no constraint), while a request that asks for JSON output FAILS
// LOUDLY instead of being served unconstrained.  Nothing else in this tree references any of
// these symbols, so no existing behaviour changes.
//
// TO REPLACE THIS PROPERLY: take the donor's `src/runtime/contract/structured_output.cpp`
// (igorls/ninfer @ 5e4a66d) AND the `third_party/xgrammar` it needs, then delete this file.  That
// is a third-party import of its own, not a symbol fix.
#include "runtime/contract/structured_output.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::runtime {
namespace {

[[noreturn]] void throw_missing_xgrammar(const char* what) {
    throw std::invalid_argument(
        std::string("structured output is not available in this build (" ) + what +
        "): the donor's xgrammar-backed implementation lives in third_party/xgrammar, which this "
        "tree does not vendor. See src/runtime/contract/structured_output.cpp for the full note. "
        "Only StructuredOutputKind::Text is accepted.");
}

} // namespace

// The donor's opaque compiled matcher.  Never constructed by this stub; the type only has to be
// complete enough for shared_ptr<const Impl>.
class CompiledOutputConstraint::Impl {
public:
    Impl() = default;
};

CompiledOutputConstraint::CompiledOutputConstraint(std::shared_ptr<const Impl> impl)
    : impl(std::move(impl)) {}

void validate_structured_output(const StructuredOutputOptions& options) {
    if (options.kind != StructuredOutputKind::Text) { throw_missing_xgrammar("validate"); }
}

class OutputConstraintCompiler::Impl {
public:
    Impl() = default;
};

OutputConstraintCompiler::OutputConstraintCompiler(std::vector<std::string> /*decoded_vocab*/,
                                                   std::vector<int> /*stop_tokens*/,
                                                   int /*end_thinking_token*/)
    : impl_(std::make_unique<Impl>()) {}

OutputConstraintCompiler::~OutputConstraintCompiler() = default;

std::shared_ptr<const CompiledOutputConstraint>
OutputConstraintCompiler::compile(const StructuredOutputOptions& options) {
    if (options.kind != StructuredOutputKind::Text) { throw_missing_xgrammar("compile"); }
    // Text means "no constraint at all".  Returning nullptr is that statement, and it is also
    // what makes every OutputConstraintState below unreachable.
    return nullptr;
}

class OutputConstraintState::Impl {
public:
    Impl() = default;
};

OutputConstraintState::OutputConstraintState(std::shared_ptr<const CompiledOutputConstraint> compiled,
                                             bool /*reasoning*/)
    : impl_() {
    if (compiled == nullptr) {
        throw std::logic_error(
            "OutputConstraintState: no compiled constraint (this build accepts Text only)");
    }
    throw_missing_xgrammar("OutputConstraintState");
}

OutputConstraintState::~OutputConstraintState() = default;

OutputConstraintState::OutputConstraintState(OutputConstraintState&&) noexcept = default;

OutputConstraintState& OutputConstraintState::operator=(OutputConstraintState&&) noexcept = default;

OutputConstraintState::OutputConstraintState(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

std::span<const std::int32_t> OutputConstraintState::next_mask() {
    throw_missing_xgrammar("next_mask");
}

void OutputConstraintState::accept(std::span<const TokenId> /*tokens*/) {
    throw_missing_xgrammar("accept");
}

OutputConstraintState OutputConstraintState::fork() const {
    throw_missing_xgrammar("fork");
}

bool OutputConstraintState::try_accept(TokenId /*token*/) {
    throw_missing_xgrammar("try_accept");
}

bool OutputConstraintState::terminated() const { throw_missing_xgrammar("terminated"); }

} // namespace ninfer::runtime
