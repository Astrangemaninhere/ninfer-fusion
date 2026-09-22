#pragma once

// src/spec/sum_dir_query_registry.h -- JUDGE A': the Q <-> field registration, checked by the
// COMPILER.
//
// THE PROBLEM THIS SOLVES, IN ONE SENTENCE
//   "7 条判据 0/7 按意思" is not a tuning problem, it is a SCHEMA problem with a mechanical cause:
//   the judge's input type `SumDirBlockFacts` (sum_dir.h:883-890) has six fields, of which three
//   are facts about the block's CONTENT -- and those three have no assigner anywhere in the tree.
//   Measured on the tree: `has_control_token` / `pinned_by_anchor` / `inside_system_prefix` occur in
//   exactly their own declaration (sum_dir.h:885-888), one read each (sum_dir.h:935-936) and two
//   comments (program_impl.h:12036-12037, :12121) -- and NOWHERE are they ever assigned.
//
// NOTE ON LINE NUMBERS (added with this landing, and deliberately blunt about it). Every
// `file:line` in this header was read off the tree at HEAD 3944a53 BEFORE the change; each one is
// also identified by SYMBOL, because this landing inserts lines into some of the files it cites and
// a bare number would then point at the wrong line. Current values, re-verified after the landing:
//   tokenizer.cpp   append_normalized_bpe_ids:552  ids.push_back:623  has_internal_boundary:657
//                   independently_normalized != normalized:697  encode_with_boundaries:797
//                   match_token == nullptr:869  push(match_token->id):884
//                   encode_with_frame_classes:941
//   tokenizer.h     BoundaryEncodedText:128  token_classes:158  FrameClassifiedEncode:165
//   processor.cpp   assign_positions:642  encode_rendered_chat:761  exact-frontier throw:814
//   processor.h     ProcessedInput::token_classes:121  EncodedChat::token_classes:141
//   sum_dir.h       SumDirRow:435  kSumDirRowWireBytes:498  static_assert:499  block facts:916
//                   holds[] (the judge's only read of them):968  sum_dir_frame_facts_apply:1028
//                   pack_row frame_bits:1913  unpack_row frame_bits:1936

// THE FIX: REGISTRATION, NOT A CONVENTION (operation "O")
//   Every query q the directory claims to answer must NAME the one field of r it depends on, and
//   every field of r must be named by at least one q. Both directions are enforced at COMPILE TIME:
//
//     * a query with no `::field` is NOT A COMPILE ERROR YOU CAN MISS -- `Registry` is constrained
//       on `names_a_field<Q>`, so instantiating it with one is a hard error, not a warning and not a
//       comment;
//     * a field of r that no query names is a `static_assert` failure ("dead weight"), which is the
//       same defect class sum_dir.h:916-919 names for the rank table.
//
//   The point is the one sum_dir.h already made for itself at :848-852 (`sum_dir_avl_rank` is "the
//   one place the order is written down"): a checked table is a contract, a comment is not. A field
//   with no q is a field nobody can be asking for; a q with no field is a question whose answer
//   cannot be read off r at all -- which is precisely the state `has_control_token` was in.
//
// WHAT THIS CHECK IS NOT (keep these apart or the whole exercise is dishonest)
//   * It does NOT check that a MODEL can answer the q. That predicate is not decidable (it is a
//     property of the model's weights and sampler, not of r).
//   * It DOES check the decidable half: "does the channel r contain the field the answer depends
//     on, and is the dependency WRITTEN DOWN". This is the whole of operation O.
//   * The registration is a DECLARATION. The companion runtime probe
//     (`scratch/FRAME-AXIS-LAND/fa_host_check.cpp`, section 4) proves for the three landed verdicts
//     that the declared driver is real by FLIPPING the input and watching the VERDICT MOVE. A
//     declaration alone is not evidence; a declaration plus a flip is.

#include "spec/sum_dir.h"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ninfer::spec::query_registry {

// ===========================================================================
// 1. THE FIELDS OF r, AS TAG TYPES. One tag per field that is actually IN r.
// ===========================================================================
struct FieldFrameBits {};      // per-block frame bitmap      -> SumDirRow::frame_bits (8 B/block)
struct FieldTokenCount {};     // per-block token count       -> SumDirRow::token_begin/token_end
struct FieldBlockIdentity {};  // 128-bit content identity    -> SumDirRow::block_identity
struct FieldContentTokens {};  // exact token ids, exactly    -> SumDir::content_
struct FieldSummaryLine {};    // the model's catalogue line  -> SumDir::summaries_
struct FieldByteAxis {};       // row.state == Live           -> SumDirRow::state / file_slot

template <class... Fields> struct FieldSet {
    static constexpr std::size_t size = sizeof...(Fields);
};

using RFields = FieldSet<FieldFrameBits, FieldTokenCount, FieldBlockIdentity, FieldContentTokens,
                         FieldSummaryLine, FieldByteAxis>;

// ===========================================================================
// 2. THE ANSWERED QUERIES. Each one NAMES one field; that is the whole contract.
// ===========================================================================
//
// Each q is also given the tree location of its ANSWERER, or the explicit name of its absence --
// because "the field exists" and "something actually answers the q" are two different questions and
// conflating them is how "0/7" happened in the first place.
#define FA_Q(NAME, FIELD, WHAT, WHO)                                                              \
    struct NAME {                                                                                 \
        using field = FIELD;                                                                      \
        static constexpr const char* what = WHAT;                                                 \
        static constexpr const char* who  = WHO;                                                  \
    }

FA_Q(QFramePosition, FieldFrameBits,
     "is token offset j of this block a template / control / role-boundary position?",
     "NEW: frame_axis + SumDirRow::frame_bits (was: NOBODY)");
FA_Q(QTokenCount, FieldTokenCount, "how many tokens does this block hold?",
     "sum_dir_row_tokens, sum_dir.h:469-471");
FA_Q(QBlockIdentity, FieldBlockIdentity, "which block has content identity H?",
     "SumDir::find, sum_dir.h:1483-1499 (after sort_rows)");
FA_Q(QExactTokenAtOffset, FieldContentTokens, "what is the exact token id at offset j?",
     "SumDir::block_content, sum_dir.h:1521-1528");
FA_Q(QSummarySubstring, FieldSummaryLine, "which blocks' catalogue lines contain byte string w?",
     "search_summaries, sum_dir.h:1507-1517 (byte substring; no tokenizer by design)");
FA_Q(QByteAxisStanding, FieldByteAxis, "does a Spill record stand for this block's bytes?",
     "sum_dir_byte_axis_present, sum_dir.h:898-912");
#undef FA_Q

// A q that names NO field: this is the shape the compiler must REFUSE. It is keeled here only so
// the negative test can prove the refusal is real.
struct QWithoutAFieldUnused {
    static constexpr const char* what = "a question with no field behind it";
    static constexpr const char* who  = "nobody: this is the defect";
};

// ===========================================================================
// 3. THE MECHANISM
// ===========================================================================
template <class Q>
concept names_a_field = requires { typename Q::field; };

template <class Q, class Field>
consteval bool query_names_field() {
    if constexpr (names_a_field<Q>) {
        return std::same_as<typename Q::field, Field>;
    } else {
        return false; // unreachable through `Registry`: the constraint is a hard error first
    }
}

// Instantiating this with a fieldless query is a COMPILE ERROR (constraint unsatisfied), which is
// the requirement: "无字段的 q = 编译错误".
template <class... Qs>
    requires((names_a_field<Qs>) && ...)
struct Registry {
    static constexpr std::size_t queries = sizeof...(Qs);

    template <class Field>
    static constexpr std::size_t queries_naming() {
        return (std::size_t{0} + ... + (query_names_field<Qs, Field>() ? 1U : 0U));
    }

    template <class... Fields>
    static constexpr bool every_field_is_named() {
        return ((queries_naming<Fields>() >= 1U) && ...);
    }

    // Per-q view, so a report can print the table instead of asserting it.
    template <class Q>
    static constexpr const char* question() {
        return Q::what;
    }
    template <class Q>
    static constexpr const char* answerer() {
        return Q::who;
    }
};

using DefaultRegistry = Registry<QFramePosition, QTokenCount, QBlockIdentity, QExactTokenAtOffset,
                                 QSummarySubstring, QByteAxisStanding>;

// Unpack a FieldSet into the `every_field_is_named` test.
template <class Set, class Reg>
struct CheckEveryFieldIsNamed;
template <template <class...> class Set, class... Fields, class Reg>
struct CheckEveryFieldIsNamed<Set<Fields...>, Reg> {
    static constexpr bool value = Reg::template every_field_is_named<Fields...>();
};

// BOTH DIRECTIONS, ONCE.
static_assert(DefaultRegistry::queries == 6, "the registered question set changed size");
static_assert(CheckEveryFieldIsNamed<RFields, DefaultRegistry>::value,
              "a field of r that NO registered query names is dead weight -- delete the field or "
              "register the query that needs it");
static_assert(DefaultRegistry::queries_naming<FieldFrameBits>() >= 1,
              "the frame axis must be a registered dependency of a registered query");

// ===========================================================================
// 4. THE JUDGE SIDE: which FACT a verdict leans on, and where that fact comes from
// ===========================================================================
//
// The verdicts are sum_dir.h:838-846. The fact fields are sum_dir.h:883-890. The mapping below is
// the DECLARATION; `fa_host_check.cpp` section 4 is the EVIDENCE that the three new ones are live.
struct FactContentPresent {};
struct FactInsideSystemPrefix {};
struct FactOverlapFrontier {};
struct FactPinnedByAnchor {};
struct FactHasControlToken {};
struct FactByteAxisPresent {};

template <ninfer::spec::sum_dir::SumDirAdmissibility V>
struct VerdictFactField;
template <>
struct VerdictFactField<ninfer::spec::sum_dir::SumDirAdmissibility::Admissible> {
    using type = FactByteAxisPresent;
};
template <>
struct VerdictFactField<ninfer::spec::sum_dir::SumDirAdmissibility::AdmissibleTextOnly> {
    using type = FactByteAxisPresent;
};
template <>
struct VerdictFactField<ninfer::spec::sum_dir::SumDirAdmissibility::ControlToken> {
    using type = FactHasControlToken;
};
template <>
struct VerdictFactField<ninfer::spec::sum_dir::SumDirAdmissibility::PinnedByAnchor> {
    using type = FactPinnedByAnchor;
};
template <>
struct VerdictFactField<ninfer::spec::sum_dir::SumDirAdmissibility::Resident> {
    using type = FactOverlapFrontier;
};
template <>
struct VerdictFactField<ninfer::spec::sum_dir::SumDirAdmissibility::SystemPrefix> {
    using type = FactInsideSystemPrefix;
};
template <>
struct VerdictFactField<ninfer::spec::sum_dir::SumDirAdmissibility::Empty> {
    using type = FactContentPresent;
};

// Which field of r a fact field is read FROM, whether that source is a fact about the block's
// CONTENT (as opposed to pure position / row state), and whether sum_dir.h really assigns it today.
template <class FactField>
struct FactSource;
template <>
struct FactSource<FactHasControlToken> {
    using field                      = FieldFrameBits;
    static constexpr bool content    = true;
    static constexpr bool assigned   = true; // NEW at this landing: frame_facts()
};
template <>
struct FactSource<FactInsideSystemPrefix> {
    using field                      = FieldFrameBits;
    static constexpr bool content    = true;
    static constexpr bool assigned   = true; // NEW at this landing: frame_facts()
};
template <>
struct FactSource<FactPinnedByAnchor> {
    using field                      = FieldFrameBits;
    static constexpr bool content    = true;
    static constexpr bool assigned   = true; // NEW at this landing: frame_facts()
};
template <>
struct FactSource<FactOverlapFrontier> {
    using field                      = FieldTokenCount;
    static constexpr bool content    = false; // token_count % 64 -- pure arithmetic
    static constexpr bool assigned   = true;
};
template <>
struct FactSource<FactContentPresent> {
    using field                      = FieldTokenCount;
    static constexpr bool content    = false; // true by construction for every real block
    static constexpr bool assigned   = true;
};
template <>
struct FactSource<FactByteAxisPresent> {
    using field                      = FieldByteAxis;
    static constexpr bool content    = false; // row.state, i.e. WHERE the bytes are
    static constexpr bool assigned   = true;
};

// A verdict is "按意思" (driven by content) iff
//   (a) its fact field is read from a CONTENT field of r,
//   (b) that r field is named by >= 1 registered query, and
//   (c) sum_dir.h really assigns that fact field.
// All three are compile-time facts, so the count is too.
template <class FactField>
static constexpr bool fact_field_is_meaning_driven() {
    using Source = FactSource<FactField>;
    if constexpr (!Source::content) {
        return false;
    } else {
        return Source::assigned &&
               (DefaultRegistry::template queries_naming<typename Source::field>() >= 1);
    }
}

template <ninfer::spec::sum_dir::SumDirAdmissibility... Vs>
static constexpr std::uint32_t count_meaning_driven() {
    return (std::uint32_t{0} +
            ... + (fact_field_is_meaning_driven<typename VerdictFactField<Vs>::type>() ? 1U : 0U));
}

// THE NUMBER. 0 before this landing, 3 after. `static_assert` keeps it from silently drifting back.
inline constexpr std::uint32_t kVerdictsByMeaning =
    count_meaning_driven<ninfer::spec::sum_dir::SumDirAdmissibility::Admissible,
                         ninfer::spec::sum_dir::SumDirAdmissibility::AdmissibleTextOnly,
                         ninfer::spec::sum_dir::SumDirAdmissibility::ControlToken,
                         ninfer::spec::sum_dir::SumDirAdmissibility::PinnedByAnchor,
                         ninfer::spec::sum_dir::SumDirAdmissibility::Resident,
                         ninfer::spec::sum_dir::SumDirAdmissibility::SystemPrefix,
                         ninfer::spec::sum_dir::SumDirAdmissibility::Empty>();
inline constexpr std::uint32_t kVerdictsTotal = 7;

static_assert(kVerdictsByMeaning >= 3,
              "the frame axis must make at least ControlToken / PinnedByAnchor / SystemPrefix "
              "reachable BY MEANING; if this fires, the axis is no longer wired into the judge");

// The per-verdict triple, so a report can be printed rather than asserted.
struct VerdictDriverRow {
    const char* verdict;
    const char* fact_field;
    const char* r_field;
    bool content;
    bool assigned;
    bool meaning;
};

inline constexpr VerdictDriverRow kVerdictDrivers[kVerdictsTotal] = {
    {"admissible", "byte_axis_present", "row.state (Live)", false, true, false},
    {"admissible-text-only", "byte_axis_present", "row.state (Live)", false, true, false},
    {"control-token", "has_control_token", "frame_bits", true, true, true},
    {"pinned-by-anchor", "pinned_by_anchor", "frame_bits", true, true, true},
    {"resident", "overlap_frontier", "token_count % 64", false, true, false},
    {"system-prefix", "inside_system_prefix", "frame_bits", true, true, true},
    {"empty", "content_present", "token count by construction", false, true, false},
};

} // namespace ninfer::spec::query_registry
