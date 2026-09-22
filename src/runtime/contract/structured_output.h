// Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
#pragma once

#include "ninfer/types.h"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::runtime {

// Immutable compiled vocabulary/schema; every active request creates its own matcher.
class CompiledOutputConstraint {
public:
    class Impl;
    explicit CompiledOutputConstraint(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl;
};

// Throws invalid_argument for schema constraints that the compiler cannot enforce.
void validate_structured_output(const StructuredOutputOptions& options);

class OutputConstraintCompiler {
public:
    OutputConstraintCompiler(std::vector<std::string> decoded_vocab, std::vector<int> stop_tokens,
                             int end_thinking_token);
    ~OutputConstraintCompiler();
    std::shared_ptr<const CompiledOutputConstraint> compile(const StructuredOutputOptions& options);
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class OutputConstraintState {
public:
    OutputConstraintState(std::shared_ptr<const CompiledOutputConstraint> compiled, bool reasoning);
    ~OutputConstraintState();
    OutputConstraintState(OutputConstraintState&&) noexcept;
    OutputConstraintState& operator=(OutputConstraintState&&) noexcept;
    [[nodiscard]] std::span<const std::int32_t> next_mask();
    void accept(std::span<const TokenId> tokens);
    [[nodiscard]] OutputConstraintState fork() const;
    [[nodiscard]] bool try_accept(TokenId token);
    [[nodiscard]] bool terminated() const;
private:
    class Impl;
    explicit OutputConstraintState(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::runtime
