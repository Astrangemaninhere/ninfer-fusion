#pragma once

// ---------------------------------------------------------------------------
// announce_once.h -- ONE keyed, add-only announce set, shared by every keyspace
// that needs one.
//
// WHY THIS FILE EXISTS, AND WHY IT IS NOT A SECOND IMPLEMENTATION.
// ---------------------------------------------------------------
// dl/cufree built `cufree_class_announce_once()` (dl/cufree/mirror/ninfer-fusion/src/core/
// cufree_report.h, sha16 `22183` B file / function at :257-267) and named it the shape to copy:
// a mutex-guarded, keyed set that can only ADD announcements and can never clear one. It exists
// because the FIRST form of the same idea -- `static bool` -- announced the first value it saw
// and then went silent, so a SECOND, DIFFERENT value was announced to nobody.
//
// ⚠ AND THE TREE HAS *TWO* LANDED FORMS OF THAT WEAKER IDEA, WITH TWO DIFFERENT SHAPES:
//   * src/core/arch_sim.h:324      `static bool announced = false;`            -- NOT synchronized
//     (that file's include list, :96-100, has no <atomic>), and it is a DATA RACE, not merely a
//     one-shot: `arch_view_for_device` is `[[nodiscard]] inline` with process-wide state and it
//     has TWO production callers (src/ops/qpn/qpn_arch_route.cpp:61, src/targets/registry.cpp:513).
//   * src/core/arch_caps.h:1675    `static std::atomic<bool> arch_warn_announced{false};` -- yes
//     synchronized, but reached only when `!arch_warn_silenced`, where `arch_warn_silenced` reads
//     NINFER_ARCH_WARN and accepts 0|off|false|no. There, "nothing to announce" and "the
//     announcement was switched off" are indistinguishable -- the defect
//     src/ops/kernel/gqa_attention_simt_ffma.cuh names in its own words.
//   * src/core/virtual_device.h:493  documented "once per process (the caller owns the once)".
// ⇒ The one-shot is the HOUSE STYLE on three axes. This header is the correction, and the third
//   axis (the vendor axis, src/core/vendor_sim.h) is the first consumer.
//
// This header is dl/cufree's function BODY, unchanged in mechanism, with TWO edits:
//   (1) the key type is a template parameter instead of the single enum `CufreeCudaClass`. That
//       edit is what lets one implementation serve several keyspaces -- which is the whole point,
//       because "two tables answering one question must not drift" is this tree's own rule
//       (src/core/arch_caps.h's kPtxFamilyAmdStatus / kPtxSites static_assert pair), and an
//       announce set is exactly a small table answering one question.
//   (2) a READ-ONLY SIZE ACCESSOR, `announce_once_seen_count<Key>()`, added because
//       tests/test_sim_no_support.cpp:568 currently pins the "was the simulator consulted at all"
//       discriminator as a TEXT MATCH on the declaration `"static bool announced = false;"`.
//       A text match on a declaration is a guard that cannot fail on behaviour: it goes red when
//       the code is REFORMATTED and stays green when the behaviour is wrong. A keyed set makes
//       "consulted at all" a VALUE, so this accessor is what lets the replacement assertion be
//       about behaviour. ⚠ IT IS AN OBSERVABILITY SURFACE, NOT A KNOB: it cannot add, remove,
//       clear or suppress an announcement, and it reads no environment variable.
//
// THE MECHANISM, restated so it can be re-derived from here alone:
//   * a function-local `static std::mutex` and a `static std::vector<underlying_type_t<Key>>`;
//   * checked-then-inserted under the lock, so the answer is exact under concurrency;
//   * the vector is APPENDED TO and never erased, scrubbed, reordered or cleared;
//   * the return value of announce_once_keyed() is "this key had not been announced before".
//
// WHY A FUNCTION-LOCAL STATIC, AND NOT A NAMESPACE-SCOPE ONE. `inline` on a function template
// gives the ODR's single instance per instantiation across every translation unit, and the
// state costs NO NEW SYMBOL IN ANY LIBRARY -- the technique
// src/ops/kernel/gqa_attention_simt_ffma.cuh:259-263 names for exactly this reason ("the
// launcher objects must not grow a link dependency on device_probe.o just to ask a route
// question"). A header-only front door can therefore hold per-key state and still link nothing.
//
// AND THE PROPERTY dl/cufree CHOSE DELIBERATELY, KEPT HERE: THERE IS NO SILENCING KNOB. This
// header reads NO environment variable at all, so what was announced is exactly what was seen.
// A caller that wants a knob must add it, and will have to explain it here -- and
// ⚠ NINFER_ARCH_WARN's shape is the counter-example, not the precedent: do not add one "for
// consistency".
//
// ⚠ THE ONE THING THIS HEADER DOES NOT DO: it does not decide WHEN to call. A caller that prints
// on a `true` return and discards a `false` one has the right behaviour; a caller that prints on
// every call has a log flood; a caller that ignores the return value has a silent path. The
// contract is one line: PRINT ON TRUE, AND NOTHING ELSE.
//
// ⚠ AND THE PROPERTY IT CANNOT HAVE: it is per-PROCESS state in an inline function, so it is
// shared by every thread of one process and shared by NOTHING across processes. A test that
// asserts "a second, different key is announced" must therefore use ONE process, and a test that
// asserts "the first key is not re-announced" must not depend on another key having been seen.
// Both properties are exercised in tests/test_vendor_sim.cpp, V2.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <mutex>
#include <type_traits>
#include <vector>

namespace ninfer::detail {

// The key type must be an enum, because the whole point is a CLOSED keyspace: an open key
// (an int, a string, a float) is how a keyspace becomes unbounded and the set grows without a
// bound. The static_assert is not decoration -- it makes the constraint a compile error with a
// reason rather than a template diagnostic a reader has to decode.
template <typename Key>
struct AnnounceSet {
    static_assert(std::is_enum_v<Key>,
                  "announce_once.h keys on a CLOSED enum on purpose: an open key space (an int, "
                  "a string, a float) would let the announced set grow without a bound.");
    using key_type = std::underlying_type_t<Key>;

    std::mutex          mutex;
    std::vector<key_type> seen;
};

// The ODR's single instance per (function, instantiation): an `inline` function template's
// function-local static. Returned by reference so the two operations below share it without
// either one owning it.
template <typename Key>
[[nodiscard]] inline AnnounceSet<Key>& announce_set() noexcept {
    static AnnounceSet<Key> set;
    return set;
}

// "This key had not been announced before." THE ONLY WRITER, AND IT CAN ONLY ADD.
template <typename Key>
[[nodiscard]] inline bool announce_once_keyed(Key key) {
    static_assert(std::is_enum_v<Key>,
                  "announce_once_keyed() keys on a CLOSED enum on purpose: an open key space "
                  "(an int, a string, a float) would let the announced set grow without a bound "
                  "(src/core/announce_once.h).");
    using key_type = std::underlying_type_t<Key>;

    AnnounceSet<Key>&       set  = announce_set<Key>();
    const key_type          value = static_cast<key_type>(key);
    const std::lock_guard<std::mutex> lock(set.mutex);
    for (const key_type seen : set.seen) {
        if (seen == value) { return false; }
    }
    set.seen.push_back(value);
    return true;
}

// How many DISTINCT keys have been announced in this process. Read-only; cannot add, clear,
// suppress or reorder anything. See the header: this exists so that "was this mechanism
// consulted at all" is a value an assertion can read instead of a declaration it can grep.
template <typename Key>
[[nodiscard]] inline std::size_t announce_once_seen_count() {
    static_assert(std::is_enum_v<Key>);
    AnnounceSet<Key>& set = announce_set<Key>();
    const std::lock_guard<std::mutex> lock(set.mutex);
    return set.seen.size();
}

} // namespace ninfer::detail
