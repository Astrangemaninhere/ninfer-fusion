// tests/test_stop_boundary.cpp -- host-only self-test for src/spec/stop_boundary.h
//
// Same shape as the other host tests beside it: no CUDA, no artifact, no fixture. Plain g++,
// matching the include roots ninfer_add_test sets up (tests/CMakeLists.txt:12-17):
//
//   g++ -std=c++20 -I src -I include -I third_party tests/test_stop_boundary.cpp -o /tmp/t
//
// WHAT THIS FILE IS FOR
//
//   Section by section it pins the four boundary classes the requirement names (103-3: 换行 /
//   闭合大括号 / 分号 / 围栏结束) and, more importantly, the shapes it forbids (a string, an
//   expression, a triple-quoted string, an escaped continuation) plus the comment cases. Section 8
//   is the one that matters operationally: over a realistically tokenized code stream a legal stop
//   exists within the deferral bound the header advertises.
//
//   Several checks are negative controls and are labelled as such: they assert that a boundary does
//   NOT appear where a naive classifier ("any newline is a boundary") would put one, so a scanner
//   that regressed would fail here rather than pass silently.

#include "spec/stop_boundary.h"

#include <cstdint>
#include <cctype>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace sb = ninfer::spec::stop_boundary;

int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++g_failures;
    }
}

void check_class(sb::StopClass got, sb::StopClass want, const char* what) {
    if (got != want) {
        std::fprintf(stderr, "FAIL %s (got %s, want %s)\n", what, sb::stop_class_name(got),
                     sb::stop_class_name(want));
        ++g_failures;
    }
}

void feed(sb::BoundaryScanner& scanner, const std::vector<std::string_view>& tokens) {
    for (const std::string_view token : tokens) { scanner.observe_token(token); }
}

// Split a literal at '|' into one token per piece, skipping empty pieces so that a case reads as
// the text it is testing. The pieces keep their '\n', so a case that means "and then a line end"
// writes one.
std::vector<std::string_view> pieces(std::string_view text) {
    std::vector<std::string_view> out;
    std::size_t begin = 0;
    for (std::size_t i = 0; i <= text.size(); ++i) {
        if (i == text.size() || text[i] == '|') {
            if (i > begin) { out.push_back(text.substr(begin, i - begin)); }
            begin = i + 1;
        }
    }
    return out;
}

sb::StopDecision scan(std::string_view text) {
    sb::BoundaryScanner scanner;
    feed(scanner, pieces(text));
    return scanner.decision();
}

// Feed a whole document split into lines, each line keeping its '\n'.
void feed_lines(sb::BoundaryScanner& scanner, std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t end = text.find('\n', begin);
        if (end == std::string_view::npos) {
            scanner.observe_token(text.substr(begin));
            break;
        }
        scanner.observe_token(text.substr(begin, end - begin + 1));
        begin = end + 1;
    }
}

// A deliberately simple but realistic tokenizer: a word (letters, digits, underscore) with any
// spaces directly before it is one token, every other non-newline character is its own token, and
// a newline is folded into the token before it. That is the shape byte-level BPE produces and it
// keeps the token count of section 8 honest rather than inflated by one-token-per-space.
std::vector<std::string_view> tokenize(std::string_view text) {
    const auto is_word = [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
    };
    std::vector<std::string_view> out;
    std::size_t i = 0;
    while (i < text.size()) {
        const std::size_t start = i;
        if (text[i] == ' ') {
            while (i < text.size() && text[i] == ' ') { ++i; }
            if (i < text.size() && is_word(text[i])) {
                while (i < text.size() && is_word(text[i])) { ++i; }
            }
        } else if (is_word(text[i])) {
            while (i < text.size() && is_word(text[i])) { ++i; }
        } else {
            ++i;
        }
        if (i < text.size() && text[i] == '\n') { ++i; }
        out.push_back(text.substr(start, i - start));
    }
    return out;
}

void section_1_the_four_named_classes() {
    // 分号
    {
        const auto decision = scan("int x = 1;\n");
        check(decision.safe, "a line ending in ';' is a legal stop");
        check_class(decision.boundary, sb::StopClass::CodeStatement, "class is code-statement");
    }
    // 闭合大括号
    {
        const auto decision = scan("  }\n");
        check(decision.safe, "a line ending in '}' is a legal stop");
        check_class(decision.boundary, sb::StopClass::CodeBlockClose, "class is code-block-close");
    }
    // 围栏结束
    {
        sb::BoundaryScanner scanner;
        feed_lines(scanner, "```python\nx = 1\n```\n");
        check(scanner.decision().safe, "the closing fence is a legal stop");
        check_class(scanner.decision().boundary, sb::StopClass::FenceClose, "class is fence-close");
        check(!scanner.in_fence(), "the fence is closed again");
    }
    // 换行 -- read as "a newline that TERMINATES A UNIT", which is what the prose case needs.
    {
        const auto decision = scan("The answer is 42.\n\n");
        check(decision.safe, "a blank line ends a prose unit");
        check_class(decision.boundary, sb::StopClass::Paragraph, "class is paragraph");
    }
}

void section_2_what_must_not_be_a_boundary() {
    // NEGATIVE CONTROL: the newline is real, the unit is not finished.
    {
        const auto decision = scan("total = foo(\n");
        check(!decision.safe, "a line ending with '(' is inside an expression");
        check(decision.paren_depth > 0, "the open paren is visible in the diagnostics");
    }
    // ')' is the "the statement continues" shape, not an end.
    {
        const auto decision = scan("if (ready)\n");
        check(!decision.safe, "a line ending with ')' is the if/for/else-if shape");
    }
    check(!scan("struct S {\n").safe, "an open brace is not an end");
    check(!scan("items.push_back(x),\n").safe, "a trailing comma is not an end");
    check(!scan("value = a +\n").safe, "a trailing operator is not an end");
    check(!scan("return\n").safe, "a bare keyword is not an end");
    check(!scan("x = \"abc\"\n").safe, "a string literal line terminates nothing without ';'");
    check(!scan("// just a note\n").safe, "a comment-only line is refused rather than claimed");
    check(!scan("/* unfinished\n").safe, "a line ending inside a block comment is refused");
    check(!scan("if (a) {\n").safe, "an open brace line is not an end");

    // The OPEN-PAREN rule, in the two shapes where it is the only thing standing between a naive
    // classifier and a wrong stop: a ';' that is inside an unclosed '(', and a blank line that is
    // inside one. Both are load-bearing (a scanner that dropped the rule passes everything above
    // and fails exactly these).
    {
        sb::BoundaryScanner scanner;
        feed_lines(scanner, "for (int i = 0; i < n;\n");
        check(!scanner.decision().safe,
              "a ';' inside an unclosed '(' is an expression, not a statement end");
        check(scanner.decision().paren_depth == 1, "the open paren is visible in the diagnostics");
        feed_lines(scanner, "     ++i) {\n");
        check(!scanner.decision().safe, "and the block open on the next line is not an end either");
        feed_lines(scanner, "    tick(i);\n");
        check(scanner.decision().safe, "the loop body's statement is a legal stop");
    }
    {
        sb::BoundaryScanner scanner;
        feed_lines(scanner, "call(\n");
        feed_lines(scanner, "    first,\n");
        feed_lines(scanner, "\n");
        check(!scanner.decision().safe, "a blank line inside an unclosed '(' is not a paragraph");
        feed_lines(scanner, "    second);\n");
        check(scanner.decision().safe, "once the call closes, the statement is a legal stop");
    }
}

void section_3_strings_are_never_split() {
    // NEGATIVE CONTROL: a ';' inside a string is not a terminator.
    {
        const auto decision = scan("note = \"a;b\n");
        check(!decision.safe, "a ';' inside an unterminated string ends no statement");
    }
    // A ';' AFTER a closed string on the same line IS a terminator, and the string's contents must
    // not have been read as significant.
    {
        const auto decision = scan("note = \"a;b\";\n");
        check(decision.safe, "a ';' after a closed string still terminates the statement");
        check_class(decision.boundary, sb::StopClass::CodeStatement, "class is code-statement");
    }
    // A brace inside a string must not move the bracket depth.
    {
        sb::BoundaryScanner scanner;
        feed(scanner, pieces("s = \"}\"\n"));
        check(!scanner.decision().safe, "a '}' inside a string is not a block close");
        check(scanner.decision().brace_depth == 0, "a brace inside a string does not count");
    }
    // An escaped newline continues a single-line string, so the line end is refused and the NEXT
    // line is still inside the string.
    {
        sb::BoundaryScanner scanner;
        feed(scanner, pieces("s = \"line one\\\n"));
        check(!scanner.decision().safe, "an escaped newline does not end the unit");
        feed(scanner, pieces("line two\";\n"));
        check(scanner.decision().safe, "the terminated string plus ';' is a legal stop");
    }
    // A triple-quoted docstring: every interior line is refused, including the line that would look
    // like a statement.
    {
        sb::BoundaryScanner scanner;
        feed_lines(scanner, "def f():\n");
        check(!scanner.decision().safe, "a def line ending in ':' is not an end");
        feed_lines(scanner, "    \"\"\"doc\n");
        check(!scanner.decision().safe, "the docstring opener line is refused");
        feed_lines(scanner, "    x = 1;\n");
        check(!scanner.decision().safe, "a ';' inside a docstring is refused");
        feed_lines(scanner, "    \"\"\"\n");
        check(!scanner.decision().safe, "the docstring closer line is refused");
        feed_lines(scanner, "    return x;\n");
        check(scanner.decision().safe, "after the docstring a statement is a legal stop");
    }
    // An apostrophe in prose must not poison the rest of the output: the line is refused, and the
    // next line is classified normally.
    {
        sb::BoundaryScanner scanner;
        feed_lines(scanner, "don't worry, it is fine\n");
        check(!scanner.decision().safe, "the apostrophe line is refused, not mis-read");
        feed_lines(scanner, "\n");
        check(scanner.decision().safe, "the paragraph after it is still a legal stop");
    }
}

void section_4_fences_open_and_close() {
    {
        sb::BoundaryScanner scanner;
        feed_lines(scanner, "Here is how:\n");
        check(!scanner.decision().safe, "a prose line ending in ':' is not an end");
        feed_lines(scanner, "```cpp\n");
        check(!scanner.decision().safe, "an opening fence is not a stop");
        check(scanner.in_fence(), "the fence is open");
        feed_lines(scanner, "int x = 1;\n");
        check(scanner.decision().safe, "a ';' inside a fence is a stop");
        feed_lines(scanner, "\n");
        check(scanner.decision().safe, "a blank line inside a fence is a stop");
        feed_lines(scanner, "```\n");
        check(scanner.decision().safe, "the closer is a stop");
        check_class(scanner.decision().boundary, sb::StopClass::FenceClose, "class is fence-close");
    }
    // A run SHORTER than the opener does not close the fence.
    {
        sb::BoundaryScanner scanner;
        feed_lines(scanner, "````\n");
        feed_lines(scanner, "```\n");
        check(scanner.in_fence(), "a shorter run does not close the fence");
    }
    // Prose after a closed fence is prose again.
    {
        sb::BoundaryScanner scanner;
        feed_lines(scanner, "```\nx;\n```\n\n");
        check(scanner.decision().safe, "the paragraph after the fence is a stop");
        check_class(scanner.decision().boundary, sb::StopClass::Paragraph, "class is paragraph");
    }
}

void section_5_prose_and_the_full_width_terminator() {
    check_class(scan("That is the end.\n").boundary, sb::StopClass::ProseSentence,
                "a '.' at a line end is a sentence boundary");
    check_class(scan("Are you sure?\n").boundary, sb::StopClass::ProseSentence,
                "a '?' at a line end is a sentence boundary");
    // Full-width marks, as bytes: 。 = e3 80 82, ！ = ef bc 81, ？ = ef bc 9f.
    check_class(scan("\xe5\xae\x8c\xe4\xba\x86\xe3\x80\x82\n").boundary, sb::StopClass::ProseSentence,
                "。 is a sentence boundary");
    check_class(scan("\xe5\xa5\xbd\xef\xbc\x81\n").boundary, sb::StopClass::ProseSentence,
                "！ is a sentence boundary");
    check_class(scan("\xe6\x98\xaf\xe7\x9a\x84\xef\xbc\x9f\n").boundary, sb::StopClass::ProseSentence,
                "？ is a sentence boundary");
    // The policy can take the sentence class away; the default keeps it.
    {
        sb::BoundaryScanner scanner;
        feed(scanner, pieces("That is the end.\n"));
        sb::StopPolicy policy;
        policy.prose_sentence_is_boundary = false;
        check(!scanner.decision(policy).safe, "the sentence class can be switched off");
        check(scanner.decision().safe, "and it is on by default");
    }
    // A "." that is not at a line end is not a boundary at all.
    check(!scan("see section 3. and then\n").safe, "a mid-line '.' is not a boundary");
}

void section_6_the_boundary_dies_when_more_text_arrives() {
    sb::BoundaryScanner scanner;
    feed(scanner, pieces("int x = 1;\n"));
    check(scanner.decision().safe, "safe after the ';'");
    feed(scanner, pieces("int"));
    check(!scanner.decision().safe, "no longer safe after more text arrives");
    check(scanner.decision().tokens_since_boundary >= 1, "the deferral counter counts from the stop");
    // An empty token changes nothing: a tokenizer that emits empty pieces cannot erase a stop.
    scanner.reset();
    feed(scanner, pieces("int x = 1;\n"));
    scanner.observe_token(std::string_view{});
    check(scanner.decision().safe, "an empty token does not erase a legal stop");
    // The raw class is exposed even when a policy declines to honour it.
    check_class(scanner.last_line_class(), sb::StopClass::CodeStatement,
                "the raw class of the last line is visible");
}

void section_7_the_pause_notice_is_out_of_band() {
    sb::PauseNotice notice;
    notice.state           = sb::PauseState::Paused;
    notice.boundary        = sb::StopClass::CodeStatement;
    notice.ledger_frontier = 4096;
    notice.deferred_tokens = 17;
    notice.reason          = "recall-trend:svip-entropy";
    check(!notice.emits_token, "a pause notice carries no token");
    check(!sb::kPauseNoticeEmitsTokens, "the constant and the field agree");
    check(std::string(sb::pause_state_name(sb::PauseState::Deferred)) == "deferred",
          "the pause states are named");

    // The ordering contract: a pause is only correctly placed at or before the frontier of the
    // first token that depends on the recalled text.
    check(sb::recall_pause_precedes_dependency(100, 100), "a pause at the dependent token is legal");
    check(sb::recall_pause_precedes_dependency(99, 100), "a pause before it is legal");
    check(!sb::recall_pause_precedes_dependency(101, 100), "a pause after it is too late");
}

void section_8_a_boundary_arrives_within_the_advertised_bound() {
    // The operational claim (103-3): "代码块通常几十 token 内就有边界". Reproduced on a realistic
    // tokenization so the count is honest.
    const std::string code =
        "static int recall_page(const KvPage& page, int slot) {\n"
        "    if (page.slot != slot) {\n"
        "        return -1;\n"
        "    }\n"
        "    return page.tokens;\n"
        "}\n";
    const std::vector<std::string_view> stream = tokenize(code);
    check(stream.size() >= 25, "the sample stream is long enough to be a real test");
    check(stream.size() <= 64, "and short enough that the bound is not trivially met");

    sb::BoundaryScanner scanner;
    std::uint64_t first_stop = 0;
    std::uint64_t stops      = 0;
    for (const std::string_view token : stream) {
        scanner.observe_token(token);
        if (scanner.decision().safe) {
            if (first_stop == 0) { first_stop = scanner.tokens_seen(); }
            ++stops;
        }
    }
    check(stops >= 3, "a code block this size offers several legal stops");
    check(first_stop != 0 && first_stop <= sb::kStopDeferralReportTokens,
          "the FIRST legal stop arrives inside the advertised deferral bound");
    check(scanner.decision().safe, "and the sample ends on a legal stop");

    // The refusal side of the same bound: while the position is illegal the caller can SEE why,
    // which is what makes "defer to the next boundary" implementable rather than a guess.
    {
        const auto decision = scan("value = compute(\n");
        check(!decision.safe && decision.paren_depth == 1,
              "an illegal position reports why it is illegal");
        check(!decision.deferral_overran, "and it is not yet past the reporting bound");
    }
    {
        // A long run with no boundary at all crosses the reporting bound and says so, while still
        // refusing -- the bound is a signal, never a licence.
        sb::BoundaryScanner long_run;
        for (int i = 0; i < 80; ++i) { long_run.observe_token("item"); }
        const auto decision = long_run.decision();
        check(!decision.safe, "a long run with no boundary still refuses");
        check(decision.deferral_overran, "exceeding the reporting bound is reported");
    }
}

} // namespace

int main() {
    section_1_the_four_named_classes();
    section_2_what_must_not_be_a_boundary();
    section_3_strings_are_never_split();
    section_4_fences_open_and_close();
    section_5_prose_and_the_full_width_terminator();
    section_6_the_boundary_dies_when_more_text_arrives();
    section_7_the_pause_notice_is_out_of_band();
    section_8_a_boundary_arrives_within_the_advertised_bound();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_stop_boundary: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_stop_boundary: ok\n");
    return 0;
}
