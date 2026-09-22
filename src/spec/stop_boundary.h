#pragma once

// src/spec/stop_boundary.h -- WHERE a recall may interrupt generation.
//
// Independent feature. EXPERIMENTAL. Host-only, std-only: no CUDA, no engine headers, no
// artifact, so it compiles and unit-tests under plain `g++` (tests/test_stop_boundary.cpp).
//
// THE REQUIREMENT THIS FILE IMPLEMENTS, VERBATIM (103-3)
//
//   "停顿绝不能落在'代码输出一半'的地方 ... 停顿只允许发生在语法安全边界（换行/闭合大括号/分号/
//    围栏结束等完整语句边界），绝不在字符串、表达式或缩进块内部。"
//   "实现上是可判定的：decode 本就逐 token 产出，只需给'可停顿'设一个 token 类门禁（末 token 属于
//    边界类才允许停）。若趋势晚发现、当前又不在安全边界 ⇒ 顺延到下一个安全边界。"
//   "等待必须以带外方式告知（API/UX 状态），绝不作为 token 注入流里。"
//
//   The token class gate is `BoundaryScanner::observe_token`, the deferral is the counter it
//   carries, and the out-of-band channel is `PauseNotice` (which carries no token by
//   construction -- see kPauseNoticeEmitsTokens).
//
// ONE READING OF THE LIST, MADE EXPLICIT BECAUSE IT IS A JUDGEMENT CALL
//
//   "换行" (a newline) is listed beside "分号" and "闭合大括号", but the same sentence also
//   forbids stopping "在表达式内部", and a bare newline can sit inside an expression:
//
//       x = foo(
//           a);
//
//   The line ending after `foo(` is a newline AND it is inside an expression. This file reads
//   the list as "a newline that TERMINATES A UNIT", and implements the consequence: a line end
//   counts as a boundary when the line's last significant character closes a statement or a
//   block AND no `(` or `[` is still open. Brace nesting is deliberately NOT a disqualifier: a
//   statement inside a function body is a complete statement, and forbidding it would make the
//   pause unreachable inside any function (the user's own estimate -- "代码块通常几十 token 内
//   就有边界" -- assumes indented statements count).
//
// THE CLASSES
//
//   CodeStatement  the line's last significant character is ';'
//   CodeBlockClose the line's last significant character is '}'
//   FenceClose     a line that closes a ``` / ~~~ fence
//   Paragraph      a blank line, no '(' or '[' open, not inside a string or a block comment
//   ProseSentence  the line's last significant character is a sentence terminator ('.', '!',
//                  '?', or the full-width 。！？). Optional, see StopPolicy: it is the one class
//                  with a real false-positive story ("e.g." / "3.14." at a line end), and prose
//                  is the category where a wrong stop is cheapest.
//
// WHAT IS DELIBERATELY NOT A BOUNDARY
//
//   ')' -- because `if (cond)` / `for (...)` / `else if (...)` are the classic "the statement
//          continues on the next line" shapes. ')' at a line end is the shape, not the end.
//   '{', ',', any operator, any opening delimiter -- mid-unit by construction.
//   A line end inside a string, a triple-quoted string, an escaped-newline continuation, or a
//   block comment -- the shapes the requirement names.
//   A comment-only line -- the scanner does not read comments as content, so it refuses to claim
//   a boundary it has not read.
//   A line that is only a string literal -- `x = "abc"` ends with '"', which terminates nothing.
//
// THE ORDERING CONTRACT (103: the pause must land BEFORE the token that depends on the recall)
//
//   This file cannot enforce when the caller asks; it makes the contract explicit and checkable.
//   The contract is CHECK-THEN-COMMIT: `observe_token` is called only for tokens ALREADY
//   committed to the output, and the pause decision is taken on the committed frontier before
//   the next token is produced. The invariant that makes "before the dependent token" true is
//   `recall_pause_precedes_dependency()`: the frontier at which the pause is taken must not be
//   past the frontier of the first token whose generation would depend on the recalled text.
//   Because a recall is a text re-prefill that lands at the CURRENT position, the operative rule
//   is the strict one: pause at the current frontier, never at a later one.
//
// WHAT THIS FILE DOES NOT DO
//
//   It does not recall, does not plan, does not emit, and does not decide that a pause is
//   warranted. It answers exactly one question: "if a pause were requested right now, is the
//   current output position a legal place for it?"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace ninfer::spec::stop_boundary {

// ---------------------------------------------------------------------------
// the verdict
// ---------------------------------------------------------------------------

enum class StopClass : std::uint8_t {
    None = 0,        // not a boundary: the position is inside a unit
    CodeStatement,   // ';' terminated line
    CodeBlockClose,  // '}' terminated line
    FenceClose,      // a fence delimiter line that closes the outermost fence
    Paragraph,       // blank line outside a string/comment with no open '(' or '['
    ProseSentence,   // sentence terminator at a line end
};

[[nodiscard]] inline const char* stop_class_name(StopClass value) noexcept {
    switch (value) {
    case StopClass::CodeStatement: return "code-statement";
    case StopClass::CodeBlockClose: return "code-block-close";
    case StopClass::FenceClose: return "fence-close";
    case StopClass::Paragraph: return "paragraph";
    case StopClass::ProseSentence: return "prose-sentence";
    case StopClass::None: break;
    }
    return "none";
}

[[nodiscard]] inline bool stop_class_is_boundary(StopClass value) noexcept {
    return value != StopClass::None;
}

// The deferral budget. It is a REPORTING bound, not a licence: when it is exceeded the pause is
// still refused and the caller is told, because a pause inside a string is worse than a late
// pause (103-3). The value is the user's own estimate -- "代码块通常几十 token 内就有边界".
inline constexpr std::uint32_t kStopDeferralReportTokens = 64U;

struct StopPolicy {
    bool prose_sentence_is_boundary = true;
    bool paragraph_is_boundary      = true;
    std::uint32_t deferral_report_tokens = kStopDeferralReportTokens;
};

inline constexpr StopPolicy kDefaultStopPolicy{};

struct StopDecision {
    bool safe      = false;               // true iff `boundary != StopClass::None`
    StopClass boundary = StopClass::None;
    std::uint64_t tokens_since_boundary = 0; // committed tokens since the last legal stop
    std::uint64_t bytes_since_boundary  = 0;
    bool deferral_overran = false;        // tokens_since_boundary > the policy's bound
    // Diagnostics, so a caller can log WHY a pause was refused instead of only that it was.
    bool in_string        = false;
    bool in_block_comment = false;
    bool in_fence         = false;
    std::int32_t paren_depth = 0; // '(' and '[' nesting only; braces are block scope
    std::int32_t brace_depth = 0;
    std::uint64_t tokens_seen = 0;
};

// ---------------------------------------------------------------------------
// the scanner
// ---------------------------------------------------------------------------

// Feed it the decoded surface text of each committed token, in order, and nothing else.
//
// Why the SURFACE text and not the token id: this tree's tokenizer makes no promise that a token
// is a whole lexeme (byte-level BPE splits `";` from `x` and `\n` from `}`), so the only stream
// whose shape is well defined is the decoded text. The scanner therefore keeps the small amount
// of per-line state it needs -- the last significant character, the leading run (for fences), the
// string/comment state, the '(' / '[' depth -- and never the whole output.
class BoundaryScanner {
public:
    void reset() noexcept { *this = BoundaryScanner{}; }

    // One committed token's decoded text. An empty surface changes nothing, including an existing
    // boundary verdict, so a tokenizer that emits empty pieces cannot erase a legal stop.
    void observe_token(std::string_view surface) noexcept {
        if (surface.empty()) { return; }
        ++tokens_seen_;
        bytes_seen_ += surface.size();
        // The position is legal only while the output ENDS at a line end, so any token that is not
        // a pure newline run clears the verdict before its characters are read; a trailing newline
        // inside the token re-establishes it.
        pending_ = StopClass::None;
        for (const char c : surface) { observe_char(c); }
        if (pending_ == StopClass::None) {
            ++tokens_since_boundary_;
            bytes_since_boundary_ += surface.size();
        } else {
            tokens_since_boundary_ = 0;
            bytes_since_boundary_  = 0;
        }
    }

    [[nodiscard]] StopDecision decision(const StopPolicy& policy = kDefaultStopPolicy) const noexcept {
        StopDecision out;
        StopClass boundary = pending_;
        if (boundary == StopClass::Paragraph && !policy.paragraph_is_boundary) {
            boundary = StopClass::None;
        }
        if (boundary == StopClass::ProseSentence && !policy.prose_sentence_is_boundary) {
            boundary = StopClass::None;
        }
        out.boundary              = boundary;
        out.safe                  = boundary != StopClass::None;
        out.tokens_since_boundary = tokens_since_boundary_;
        out.bytes_since_boundary  = bytes_since_boundary_;
        out.deferral_overran      = tokens_since_boundary_ > policy.deferral_report_tokens;
        out.in_string             = string_ != StringState::None;
        out.in_block_comment      = block_comment_;
        out.in_fence              = in_fence_;
        out.paren_depth           = paren_depth_;
        out.brace_depth           = brace_depth_;
        out.tokens_seen           = tokens_seen_;
        return out;
    }

    // The class of the line that just ended, whether or not a policy would honour it. Exposed so a
    // log (and the test) can see the raw classification.
    [[nodiscard]] StopClass last_line_class() const noexcept { return last_line_class_; }
    [[nodiscard]] std::uint64_t tokens_seen() const noexcept { return tokens_seen_; }
    [[nodiscard]] std::uint64_t bytes_seen() const noexcept { return bytes_seen_; }
    [[nodiscard]] bool in_fence() const noexcept { return in_fence_; }

private:
    enum class StringState : std::uint8_t {
        None,
        Single,       // '...' on one line
        Double,       // "..." on one line
        TripleSingle, // ''' may span lines
        TripleDouble, // """ may span lines
    };

    // The bytes of the current character plus the two before it, for the full-width terminators
    // (。 = e3 80 82, ！ = ef bc 81, ？ = ef bc 9f): checked as bytes so no Unicode table is
    // needed here.
    void push_tail(char c) noexcept {
        tail0_ = tail1_;
        tail1_ = tail2_;
        tail2_ = c;
    }

    [[nodiscard]] bool tail_is_full_width_terminator() const noexcept {
        const auto b0 = static_cast<std::uint8_t>(tail0_);
        const auto b1 = static_cast<std::uint8_t>(tail1_);
        const auto b2 = static_cast<std::uint8_t>(tail2_);
        if (b1 == 0x80 && b2 == 0x82 && b0 == 0xE3) { return true; } // 。
        if (b1 == 0xBC && b0 == 0xEF && (b2 == 0x81 || b2 == 0x9F)) { return true; } // ！ ？
        return false;
    }

    void observe_char(char c) noexcept {
        if (c == '\n') {
            end_line();
            return;
        }
        pending_ = StopClass::None;
        push_tail(c);
        // "The line had content" means any non-space character, INSIDE a string or a comment as
        // well: it is what separates a blank line (a paragraph) from a line that merely failed to
        // leave a significant character behind (a comment-only line, or nothing but a literal).
        if (c != ' ' && c != '\t' && c != '\r') { line_had_content_ = true; }
        update_lead_run(c);

        if (block_comment_) {
            // The closer is `*/`; the `/` of the opener is two characters back and is handled in
            // open_comment_maybe() below.
            if (c == '/' && prev_ == '*') { block_comment_ = false; }
            prev_ = c;
            return;
        }
        if (line_comment_) {
            prev_ = c;
            return;
        }
        if (string_ != StringState::None) {
            handle_string_char(c);
            return;
        }
        if (c == '"' || c == '\'') {
            if (pending_quote_ != 0 && pending_quote_char_ != c) { resolve_pending_quotes(); }
            pending_quote_char_ = c;
            ++pending_quote_;
            line_had_content_ = true;
            prev_              = c;
            return;
        }
        // Any other character fixes a pending quote run, and the CURRENT character may already be
        // string content (the char right after the opening quote).
        resolve_pending_quotes();
        if (string_ != StringState::None) {
            handle_string_char(c);
            return;
        }
        if (open_comment_maybe(c)) { return; }

        switch (c) {
        case '(':
        case '[': ++paren_depth_; break;
        case ')':
        case ']':
            if (paren_depth_ > 0) { --paren_depth_; }
            break;
        case '{': ++brace_depth_; break;
        case '}':
            if (brace_depth_ > 0) { --brace_depth_; }
            break;
        default: break;
        }

        if (c != ' ' && c != '\t' && c != '\r') {
            // A character that is part of the LEADING run is the fence marker, not content: that
            // is what lets a pure `````` line be recognised as a delimiter.
            if (!in_lead_run_) {
                line_has_significant_ = true;
                prev_significant_     = last_significant_;
                last_significant_     = c;
            }
        }
        prev_ = c;
    }

    void update_lead_run(char c) noexcept {
        in_lead_run_ = false;
        if (lead_run_done_) { return; }
        if (c == ' ') {
            if (lead_run_len_ == 0) { ++lead_indent_; }
            return;
        }
        if (c == '`' || c == '~') {
            if (lead_run_len_ == 0) { lead_run_char_ = c; }
            if (c == lead_run_char_) {
                ++lead_run_len_;
                in_lead_run_ = true;
                if (lead_run_len_ >= 3) { fence_line_ = true; }
                return;
            }
        }
        lead_run_done_ = true;
    }

    // `//`, `#`, `/*`. Opening a comment RETRACTS the significance of the character that would
    // otherwise have been left as the line's last significant character (`x = 1; // note` must
    // classify off the ';', not off the '/').
    bool open_comment_maybe(char c) noexcept {
        if (c == '/' && prev_ == '/') {
            retract_significance();
            line_comment_ = true;
            prev_         = c;
            return true;
        }
        if (c == '*' && prev_ == '/') {
            retract_significance();
            block_comment_ = true;
            prev_          = c;
            return true;
        }
        if (c == '#') {
            retract_significance();
            line_comment_ = true;
            prev_         = c;
            return true;
        }
        return false;
    }

    void retract_significance() noexcept {
        last_significant_     = prev_significant_;
        line_has_significant_ = prev_significant_ != '\0';
    }

    void handle_string_char(char c) noexcept {
        if (escape_) {
            escape_ = false;
            prev_   = c;
            return;
        }
        if (c == '\\') {
            escape_ = true;
            prev_   = c;
            return;
        }
        const bool single = string_ == StringState::Single || string_ == StringState::TripleSingle;
        const char quote  = single ? '\'' : '"';
        if (c == quote) {
            ++quote_run_;
            const bool one_line = string_ == StringState::Single || string_ == StringState::Double;
            if (quote_run_ >= (one_line ? 1U : 3U)) {
                string_    = StringState::None;
                quote_run_ = 0;
            }
        } else {
            quote_run_ = 0;
        }
        // Nothing inside a string is significant and nothing inside a string moves a bracket
        // depth: that is the whole reason the string state exists.
        prev_ = c;
    }

    void resolve_pending_quotes() noexcept {
        if (pending_quote_ == 0) { return; }
        const std::uint32_t run = pending_quote_;
        const char quote        = pending_quote_char_;
        pending_quote_          = 0;
        line_had_content_       = true;
        if (run == 1) {
            string_    = quote == '\'' ? StringState::Single : StringState::Double;
            quote_run_ = 1;
            return;
        }
        if (run == 2) {
            // "" / '' is an empty literal: opened and closed on the same line.
            string_    = StringState::None;
            quote_run_ = 0;
            return;
        }
        // Three or more in a row: a triple-quoted string opening. A longer run cannot be resolved
        // from a streaming view, and "inside a triple" is the conservative choice -- it can only
        // add refusals, never a wrong stop.
        string_    = quote == '\'' ? StringState::TripleSingle : StringState::TripleDouble;
        quote_run_ = 3;
    }

    void end_line() noexcept {
        const bool one_line_string =
            string_ == StringState::Single || string_ == StringState::Double;
        const bool continuation = escape_ && one_line_string;

        StopClass cls = StopClass::None;
        if (string_ != StringState::None) {
            if (continuation) {
                // A backslash right before the newline continues a single-line string, so the
                // string survives into the next line and this line is not a boundary.
                escape_ = false;
            } else if (one_line_string) {
                // An unterminated single-line string. C-like strings cannot span a line any other
                // way, and in prose an odd apostrophe is far more likely than a multi-line string,
                // so the state is dropped here. The line itself is still refused: the scanner saw
                // a quote it could not close.
                string_    = StringState::None;
                quote_run_ = 0;
            }
        } else if (block_comment_) {
            cls = StopClass::None;
        } else if (fence_line_) {
            cls = end_fence_line();
        } else if (!line_had_content_) {
            cls = paren_depth_ == 0 ? StopClass::Paragraph : StopClass::None;
        } else if (!line_has_significant_) {
            cls = StopClass::None; // comment-only, or nothing but a string literal
        } else if (paren_depth_ != 0) {
            cls = StopClass::None; // "在表达式内部"
        } else if (last_significant_ == ';') {
            cls = StopClass::CodeStatement;
        } else if (last_significant_ == '}') {
            cls = StopClass::CodeBlockClose;
        } else if (last_significant_ == '.' || last_significant_ == '!' || last_significant_ == '?' ||
                   tail_is_full_width_terminator()) {
            cls = StopClass::ProseSentence;
        }

        last_line_class_ = cls;
        pending_         = cls;
        settle_pending_quotes_at_line_end();
        reset_line();
    }

    // A fence delimiter line has just ended: it closes the outermost fence when it is a pure run
    // of the opener's character at least as long as the opener's run, and opens one when it is the
    // first such run outside a fence (an info string after the run is allowed, which is why the
    // opener test is on the RUN and not on the whole line).
    [[nodiscard]] StopClass end_fence_line() noexcept {
        const bool pure_run = !line_has_significant_;
        if (in_fence_) {
            if (pure_run && lead_run_char_ == fence_char_ && lead_run_len_ >= fence_len_ &&
                lead_indent_ <= 3) {
                in_fence_ = false;
                return StopClass::FenceClose;
            }
            return StopClass::None;
        }
        if (lead_run_len_ >= 3 && lead_indent_ <= 3 &&
            (lead_run_char_ == '`' || lead_run_char_ == '~')) {
            in_fence_   = true;
            fence_char_ = lead_run_char_;
            fence_len_  = lead_run_len_;
        }
        return StopClass::None;
    }

    void settle_pending_quotes_at_line_end() noexcept {
        // Nothing pending: whatever string state the line end left behind STANDS. This is the case
        // that carries a backslash continuation and a triple-quoted string across the newline, so an
        // unconditional reset here would silently un-string the rest of the output.
        if (pending_quote_ == 0) { return; }
        if (pending_quote_ >= 3) {
            string_    = pending_quote_char_ == '\'' ? StringState::TripleSingle
                                                     : StringState::TripleDouble;
            quote_run_ = 3;
        } else {
            // A lone quote opened a string that never closed, and `""` closed itself.
            string_    = StringState::None;
            quote_run_ = 0;
        }
        pending_quote_ = 0;
    }

    void reset_line() noexcept {
        line_had_content_     = false;
        line_has_significant_ = false;
        line_comment_         = false;
        fence_line_           = false;
        last_significant_     = '\0';
        prev_significant_     = '\0';
        lead_indent_          = 0;
        lead_run_char_        = '\0';
        lead_run_len_         = 0;
        lead_run_done_        = false;
        in_lead_run_          = false;
        prev_                 = '\0';
        tail0_ = tail1_ = tail2_ = '\0';
    }

    // whole-output state
    StringState string_          = StringState::None;
    bool escape_                 = false;
    std::uint32_t quote_run_     = 0;
    std::uint32_t pending_quote_ = 0;
    char pending_quote_char_     = '\0';
    bool block_comment_          = false;
    std::int32_t paren_depth_    = 0;
    std::int32_t brace_depth_    = 0;
    bool in_fence_               = false;
    char fence_char_             = '\0';
    std::uint32_t fence_len_     = 0;

    // current-line state
    char prev_                     = '\0';
    char last_significant_         = '\0';
    char prev_significant_         = '\0';
    char tail0_                    = '\0';
    char tail1_                    = '\0';
    char tail2_                    = '\0';
    bool line_had_content_         = false;
    bool line_has_significant_     = false;
    bool line_comment_             = false;
    bool fence_line_               = false;
    std::uint32_t lead_indent_     = 0;
    char lead_run_char_            = '\0';
    std::uint32_t lead_run_len_    = 0;
    bool lead_run_done_            = false;
    bool in_lead_run_              = false;

    // verdict + counters
    StopClass pending_                   = StopClass::None;
    StopClass last_line_class_           = StopClass::None;
    std::uint64_t tokens_seen_           = 0;
    std::uint64_t bytes_seen_            = 0;
    std::uint64_t tokens_since_boundary_ = 0;
    std::uint64_t bytes_since_boundary_  = 0;
};

// ---------------------------------------------------------------------------
// the out-of-band channel (103-3: never a token)
// ---------------------------------------------------------------------------

// The pause must be told to the client through the API/UX channel and NOT by emitting a token
// such as "waiting for the KV cache". This is the shape of that channel, kept here so the rule
// has a name and so "it emits no token" is a compile-time fact rather than a convention.
inline constexpr bool kPauseNoticeEmitsTokens = false;
static_assert(!kPauseNoticeEmitsTokens,
              "a pause notice is an out-of-band state, never a token in the output stream");

enum class PauseState : std::uint8_t {
    Running = 0, // no pause owed
    Deferred,    // a pause is owed but the position is not a legal stop yet
    Paused,      // the pause is in force; the client may render a status line
    Resumed,     // the recall landed and generation continued
};

[[nodiscard]] inline const char* pause_state_name(PauseState state) noexcept {
    switch (state) {
    case PauseState::Running: return "running";
    case PauseState::Deferred: return "deferred";
    case PauseState::Paused: return "paused";
    case PauseState::Resumed: return "resumed";
    }
    return "?";
}

struct PauseNotice {
    PauseState state              = PauseState::Running;
    StopClass boundary            = StopClass::None; // the class the pause was taken at
    std::uint32_t ledger_frontier = 0;               // tokens committed when the pause was taken
    std::uint64_t deferred_tokens = 0;               // how long the pause was owed before it landed
    std::string_view reason{};                       // e.g. "recall-trend:svip-entropy"
    bool emits_token              = kPauseNoticeEmitsTokens;
};

// The pause/recall ordering contract, as a predicate so both sides of it can be checked against
// each other. `pause_frontier` is the committed-token frontier at which the pause is taken;
// `first_dependent_frontier` is the frontier of the first token whose generation would depend on
// the recalled text. The recall is a text re-prefill that lands at the CURRENT position, so a
// pause taken after that token is already mis-placed.
[[nodiscard]] constexpr bool
recall_pause_precedes_dependency(std::uint32_t pause_frontier,
                                 std::uint32_t first_dependent_frontier) noexcept {
    return pause_frontier <= first_dependent_frontier;
}

// The one decision a caller needs: may a pause be taken right now?
[[nodiscard]] inline bool stop_is_legal(const StopDecision& decision) noexcept {
    return decision.safe;
}

} // namespace ninfer::spec::stop_boundary
