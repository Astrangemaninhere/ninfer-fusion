#pragma once

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {});

std::optional<int> optional_int(const nlohmann::json& object, const char* key);
std::optional<double> optional_number(const nlohmann::json& object, const char* key);
bool optional_bool(const nlohmann::json& object, const char* key, bool fallback);

[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept;

// def3: refuse an output budget above the server's own ceiling (RequestLimits::max_context)
// BY NAME, with both numbers in the message. A budget at or below the ceiling is untouched.
// mtplogfix: a ceiling of 0 is no longer "nothing to check". A caller that presents a budget
// against RequestLimits::max_context left at 0 gets a named refusal (500, code
// output_ceiling_unnamed) instead of a silent pass, because the engine's clamp would
// otherwise be the only answer. A request with no budget never reaches this check.
void validate_output_budget(int requested, std::uint32_t ceiling, const char* param);

} // namespace ninfer::serve
