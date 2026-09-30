#pragma once

// ninfer::targets::qwen3_6::runtime -- THE ATTENTION-OUTPUT GATE DECISION, BY DECLARED IDENTITY.
//
// Line `accfix`, marker F910. The shape is ops/launcher/gqa_attention_geometry_route.h, which
// dl/musefix F-897 landed for the GQA GEOMETRY axis; this is the same defect on the NUMERICS axis,
// and it is the instance dl/importaudit F-902 named as the worst one it found (its arm A7/A10/A11
// refusals) without fixing it.
//
// (A) THE DEFECT, MEASURED IN THE PREIMAGE RATHER THAN ASSERTED.
//     `ModelConfig::headwise_gate_impl` (text_context.h, PRE :227-233) answered the question "is
//     this arch's attention-output gate headwise?" with
//         if constexpr (requires { TC::headwise_attn_output_gate_enabled(); }) return ...;
//         else return false;                       <-- A SILENT DEFAULT
//     and the tree's own guard against the consequence
//         static_assert(!ModelConfig::headwise_gate() || gate_rows() != query_size, ...)
//     (text_context.h PRE :303-307) is ARMED ONLY WHEN THE ARCH DECLARES headwise. So the guard's
//     arming condition IS the very accessor that may be missing: an arch whose gate IS headwise but
//     which OMITS the accessor reads `false`, the guard is satisfied VACUOUSLY, and the pair falls
//     through to the PER-ELEMENT route of ops::sigmoid_mul (src/ops/wrapper/sigmoid_mul.cpp, the
//     `headwise_gate_shape` test and the loop below it), which ACCEPTS the pair and multiplies `a`
//     by gate rows the target never wrote. rc=0 and a WRONG NUMBER, with nothing printed.
//
// (B) WHY THE DECISION IS NOT THE SHAPE. sigmoid_mul picks its route BY SHAPE -- `x.ne[3] == 1 &&
//     gate.ne[2] == 1 && gate.ne[3] == 1 && gate.ne[0] == x.ne[1] && gate.ne[1] == x.ne[2]` -- so the
//     shape is a CONSEQUENCE of the decision, not a source of it. The tree already spells the two
//     legal shapes at one call site (text_context_impl.h, `ModelConfig::headwise_gate() ? ...`).
//     What was missing is that the DECISION be readable on its own, without a GPU and without the
//     ops library, so a host test can drive the reader's own answer -- which is what this file is.
//
// (C) THE IDENTITY IS THE DECLARED GATE TRIPLE, NOT THE GEOMETRY. (query_heads, kv_heads, head_dim)
//     is NOT an identity here: qwen3_5_9b (16/4/256) and spark_x2_5_4b (16/4/256) are the same
//     geometry with DIFFERENT gates, and muse_glimmer_30b (32/2/128) and qwen3_6_35b_a3b (16/2/256)
//     are different geometries with the SAME declared gate triple. The triple this reader keys on is
//     (headwise_attn_output_gate_enabled, attention_gate_rows, query_size) AS DECLARED -- including
//     whether each of the first two is declared AT ALL, which is a fact about the arch and not a
//     detail of this file. Two arches that declare the same triple take the same route, which is why
//     they may share a row; an arch whose declarations do not match any row is REFUSED BY NAME.
//
// (D) HOW A NEW MODEL ADAPTS, WITHOUT EDITING SHARED CODE. It declares its own gate facts in its own
//     `impl/config.h` -- that is where the fact belongs, and it is the arch's own evidence. If its
//     declared triple is new, one row is added HERE. Declaring a gate row count the per-element
//     route cannot accept while omitting the headwise accessor is NO LONGER COMPILABLE: the
//     inverted guard in text_context.h is armed by the gate's own declaration, so the omission can
//     no longer disarm it.
//
// THE READER IS HOST-ONLY ON PURPOSE: no CUDA, no ops, no engine. dl/accfix's host differential
// drives it against fixtures whose expectations were declared before the run, and against the
// arches' own declarations read out of their config.h files.

#include <cstdint>
#include <string>

namespace ninfer::targets::qwen3_6::detail::attn_output_gate_route {

enum class GateRoute : std::uint8_t {
    // One sigmoid per (head, token), broadcast over head_dim: the gate must reach
    // ops::sigmoid_mul over 2-D ({gate_rows, T}) or it will NOT take this route.
    Headwise = 0,
    // The gate is query_size wide: the per-element route is the right one and the
    // {head_dim, n_q, T} view is the shape that reaches it.
    PerElement,
    // No registered row states this declared triple. REFUSED BY NAME, never defaulted.
    Unregistered,
};

// A REGISTERED ROW IS A DECLARATION ABOUT AN ARCH, COPIED FROM THAT ARCH'S OWN `impl/config.h`.
// `headwise_declared` / `rows_declared` say whether the accessor is declared AT ALL, because
// "the arch says false" and "the arch says nothing" are DIFFERENT FACTS and the whole defect was
// that they were spelled the same. `models` names every arch verified to declare this triple.
struct RegisteredGate {
    const char* models;
    bool headwise_declared;
    bool headwise;
    bool rows_declared;
    std::int32_t gate_rows;
    std::int32_t query_size;
};

// ONE ROW PER DECLARED TRIPLE. Every number here was READ from the arch's own config.h, and the
// reading is recorded in dl/accfix/logs/ with the file and the line.
inline constexpr RegisteredGate kRegisteredAttnOutputGates[] = {
    // spark_x2_5_4b/impl/config.h:178 `attention_gate_rows = query_heads` (= 16) and
    // :271 `headwise_attn_output_gate_enabled() { return true; }` -- the ONLY arch that declares
    // headwise, and the one the PRE guard armed for.
    {"spark_x2_5_4b (16 q / 4 kv / 256)", true, true, true, 16, 4096},
    // The arches that declare NEITHER fact. A row is a DECLARATION that the per-channel form is
    // this arch's truth -- evidenced by its `query_projection_rows == 2 * query_size` (a gate is
    // projected) and by its NOT declaring a headwise accessor anywhere. It is no longer a default:
    // if such an arch starts declaring attention_gate_rows, the drift check refuses by name.
    {"qwen3_5_9b (16 q / 4 kv / 256)", false, false, false, 4096, 4096},
    {"qwen3_6_35b_a3b (16 q / 2 kv / 256)", false, false, false, 4096, 4096},
    {"muse_glimmer_30b (32 q / 2 kv / 128)", false, false, false, 4096, 4096},
    {"qwen3_6_27b (24 q / 4 kv / 256)", false, false, false, 6144, 6144},
};

// THE ONE READER. `query_size` is `ModelConfig::gate_rows()`'s own fallback (`TC::query_size`),
// passed in rather than recomputed here, so this reader and the call site cannot disagree about
// what the fallback is -- they agree by construction because the call site passes its own value.
[[nodiscard]] constexpr GateRoute gate_route_for(bool headwise_declared, bool headwise,
                                                bool rows_declared, std::int32_t gate_rows,
                                                std::int32_t query_size) noexcept {
    for (const RegisteredGate& row : kRegisteredAttnOutputGates) {
        if (row.query_size != query_size) { continue; }
        // FIELD BY FIELD: a field the arch declares and the row does not (or the reverse) is DRIFT.
        if (row.headwise_declared != headwise_declared || row.rows_declared != rows_declared) {
            continue;
        }
        if (headwise_declared && row.headwise != headwise) { continue; }
        if (rows_declared && row.gate_rows != gate_rows) { continue; }
        return row.headwise ? GateRoute::Headwise : GateRoute::PerElement;
    }
    return GateRoute::Unregistered;
}

[[nodiscard]] inline const char* attn_output_gate_route_name(GateRoute route) noexcept {
    switch (route) {
    case GateRoute::Headwise:
        return "Headwise";
    case GateRoute::PerElement:
        return "PerElement";
    case GateRoute::Unregistered:
        break;
    }
    return "Unregistered";
}

// "headwise=<absent>|true|false rows=<absent>|N query_size=N" -- the DECLARATION FINGERPRINT, so a
// refusal and a log line state the facts rather than a bool that cannot distinguish absence.
[[nodiscard]] inline std::string attn_output_gate_declaration_fingerprint(
    bool headwise_declared, bool headwise, bool rows_declared, std::int32_t gate_rows,
    std::int32_t query_size) {
    std::string out = "headwise=";
    if (!headwise_declared) {
        out += "<absent>";
    } else {
        out += headwise ? "true" : "false";
    }
    out += " rows=";
    if (!rows_declared) {
        out += "<absent>";
    } else {
        out += std::to_string(gate_rows);
    }
    out += " query_size=" + std::to_string(query_size);
    return out;
}

// The registered triples, built from the table so a refusal can never list a row the reader does
// not actually consult.
[[nodiscard]] inline std::string attn_output_gate_registered_list() {
    std::string out;
    for (const RegisteredGate& row : kRegisteredAttnOutputGates) {
        if (!out.empty()) { out += "; "; }
        out += attn_output_gate_declaration_fingerprint(row.headwise_declared, row.headwise,
                                                        row.rows_declared, row.gate_rows,
                                                        row.query_size) +
               " [" + row.models + "]";
    }
    return out;
}

// THE REFUSAL, BY NAME, WITH THE FINGERPRINT AND WITH WHAT TO DO ABOUT IT. It replaces a silent
// `false` that produced rc=0 and a wrong number.
[[nodiscard]] inline std::string attn_output_gate_refusal(bool headwise_declared, bool headwise,
                                                          bool rows_declared,
                                                          std::int32_t gate_rows,
                                                          std::int32_t query_size) {
    std::string out =
        "attn output gate: this arch declares " +
        attn_output_gate_declaration_fingerprint(headwise_declared, headwise, rows_declared,
                                                 gate_rows, query_size) +
        ", which matches NO registered row. Registered declared triples: " +
        attn_output_gate_registered_list() +
        ". The identity is the DECLARED GATE TRIPLE, not the geometry: (16 q / 4 kv / 256) is both "
        "spark_x2_5_4b (headwise) and qwen3_5_9b (per-channel). Before this reader an absent "
        "headwise accessor took a SILENT DEFAULT of false, the compile-time guard was disarmed by "
        "the very omission it exists for, and the pair took the PER-ELEMENT route of "
        "ops::sigmoid_mul with rc=0 and a different answer. If this arch's gate is the per-channel "
        "form, add one row to kRegisteredAttnOutputGates stating that as a DECLARATION; if its gate "
        "is headwise, declare headwise_attn_output_gate_enabled() and attention_gate_rows in its own "
        "impl/config.h -- that is where the fact belongs -- and add the row.";
    return out;
}

} // namespace ninfer::targets::qwen3_6::detail::attn_output_gate_route
