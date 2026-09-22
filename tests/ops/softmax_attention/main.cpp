#include <iostream>

int run_softmax_attention_causal_cache_tests();
int run_softmax_attention_plain_and_packed_tests();
int run_softmax_attention_context_tests();

namespace {
bool is_red(int code) { return code != 0 && code != 77; }
} // namespace

int main() {
    // All three runners are ALWAYS executed, and a red result is never short-circuited by a
    // sibling's skip: each runner settles its host half before its device gate, so a red host half
    // is a real finding even on a box where the other two runners have nothing to run. 77 is
    // returned only when NO runner is red and at least one skipped.
    const int causal           = run_softmax_attention_causal_cache_tests();
    const int plain_and_packed = run_softmax_attention_plain_and_packed_tests();
    const int context          = run_softmax_attention_context_tests();

    const int red     = (is_red(causal) ? 1 : 0) + (is_red(plain_and_packed) ? 1 : 0) +
                        (is_red(context) ? 1 : 0);
    const int skipped = (causal == 77 ? 1 : 0) + (plain_and_packed == 77 ? 1 : 0) +
                        (context == 77 ? 1 : 0);

    if (red != 0) {
        std::cout << "softmax_attention: FAIL (causal=" << causal
                  << " plain_and_packed=" << plain_and_packed << " context=" << context << ")\n";
        return 1;
    }
    if (skipped != 0) {
        std::cout << "softmax_attention: SKIP (" << skipped << " of 3 runners skipped)\n";
        return 77;
    }
    std::cout << "softmax_attention: PASS\n";
    return 0;
}
