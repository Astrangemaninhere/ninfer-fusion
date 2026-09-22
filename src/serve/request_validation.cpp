#include "serve/request_validation.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param, std::string code) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

std::optional<int> optional_int(const nlohmann::json& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    const nlohmann::json& value = object.at(key);
    if (!value.is_number_integer()) { bad_request(std::string(key) + " must be an integer", key); }
    if (value.is_number_unsigned()) {
        const std::uint64_t converted = value.get<std::uint64_t>();
        if (converted > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            bad_request(std::string(key) + " is out of range", key);
        }
        return static_cast<int>(converted);
    }
    const std::int64_t converted = value.get<std::int64_t>();
    if (converted < std::numeric_limits<int>::min() ||
        converted > std::numeric_limits<int>::max()) {
        bad_request(std::string(key) + " is out of range", key);
    }
    return static_cast<int>(converted);
}

std::optional<double> optional_number(const nlohmann::json& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_number()) { bad_request(std::string(key) + " must be a number", key); }
    const double value = object.at(key).get<double>();
    if (!std::isfinite(value)) { bad_request(std::string(key) + " must be finite", key); }
    return value;
}

bool optional_bool(const nlohmann::json& object, const char* key, bool fallback) {
    if (!object.contains(key) || object.at(key).is_null()) { return fallback; }
    if (!object.at(key).is_boolean()) { bad_request(std::string(key) + " must be a boolean", key); }
    return object.at(key).get<bool>();
}

void validate_output_budget(int requested, std::uint32_t ceiling, const char* param) {
    // The ONE clamp in the engine is `min(requested, max_context - prompt_tokens + 1)`
    // (targets/qwen3_6/impl/runtime/request_plan_impl.h:207-213; engine_core.h:256-257). A
    // request above --max-context is above the whole window, so no prompt length can make it
    // fit and the clamp is the only thing that would answer -- which is why this is refused
    // here, at parse time, before a prompt exists. A value at or below the ceiling is left
    // alone: the prompt-dependent part of the clamp is already NAMED downstream as
    // FinishReason::ContextCapacity ("length" on the wire, openai_chat_response.cpp:25-29).
    // A negative budget is refused by the callers themselves (openai_chat_request.cpp:873,
    // anthropic_messages_request.cpp:1054, openai_responses_request.cpp:1258), so this early
    // return cannot be the place where an unnamed ceiling hides.
    if (requested < 0) { return; }
    if (ceiling == 0) {
        // mtplogfix: the 0 sentinel is a NAMED REFUSAL, not a pass. A caller that assembled
        // RequestLimits without naming a ceiling leaves this function unable to tell a budget
        // that fits from one above the whole window, so the engine's silent clamp above would
        // be the only answer -- a silent failure, which this project counts as a defect.
        // No HTTP front end can reach this: all four copy ServeOptions::max_context in
        // (openai_chat_http.cpp:29, anthropic_messages_http.cpp:43,
        // openai_responses_http.cpp:248 and :394), so reaching here means the omission is on
        // the SERVER side -- hence 500/server_error, the shape this project already uses for
        // a server-side fault, rather than a 400 that would blame the request.
        ApiError error;
        error.status  = 500;
        error.type    = "server_error";
        error.param   = param;
        error.code    = "output_ceiling_unnamed";
        error.message = std::string(param) + "=" + std::to_string(requested) +
                        " cannot be checked against this server's window: the request was "
                        "assembled with RequestLimits::max_context left at 0, so no ceiling "
                        "is named and the engine's own clamp would be the only answer. "
                        "Every HTTP front end must carry ServeOptions::max_context into "
                        "RequestLimits.";
        throw ApiException(std::move(error));
    }
    if (static_cast<std::uint64_t>(requested) <= static_cast<std::uint64_t>(ceiling)) { return; }
    bad_request(std::string(param) + "=" + std::to_string(requested) +
                    " exceeds this server's output ceiling (--max-context=" +
                    std::to_string(ceiling) +
                    "): the engine would silently replace it with what the window has left, so "
                    "it is refused here instead. Give a value in [0, " +
                    std::to_string(ceiling) + "].",
                param);
}

bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept {
    if (name.empty() || name.size() > maximum_length) { return false; }
    for (const unsigned char character : name) {
        if (std::isalnum(character) == 0 && character != '_' && character != '-') { return false; }
    }
    return true;
}

} // namespace ninfer::serve
