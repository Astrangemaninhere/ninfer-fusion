// ---------------------------------------------------------------------------
// ninfer-cufree -- the CUDA-FREE front door (dl/cufree).
//
// WHAT IT IS FOR. This is the one executable in the tree that can print a refusal on a
// machine where the other three front doors cannot START. Its own target links NO ninfer
// library at all (apps/CMakeLists.txt), so `readelf -d` on it shows no CUDA entry, and it
// therefore has no soname that ld.so can fail on before main(). Everything it prints comes
// from src/core/cufree_report.h, which is std-only and header-only, so it costs no link
// edge -- see that header for why the text could not live beside
// render_build_capability_surface() in src/core/arch_caps.h, and for the two measured
// limits on what it may claim.
//
// WHAT IT DOES NOT DO. It calls NO CUDA function: it does not call cudaGetDeviceCount, it
// does not count devices, and it does not name a card vendor. It measures two things the
// loader and the filesystem can answer without CUDA -- whether the two DT_NEEDED sonames
// are findable, and which accelerator kernel interfaces exist as paths -- and it says
// exactly that much.
//
// THE TOOLING NOTE THAT COST FOUR LINES, kept here because this file is where a reader
// would try it: a PATH-only probe for the CUDA toolchain does not error, it reports
// "0 device entries, 0 SASS" -- a broken needle that looks like a clean answer. The same
// discipline applies to this file's witnesses: every row prints the PATH THAT WAS TESTED,
// so "absent" here means "this exact path was tested and did not exist", never "the probe
// found nothing".

#include "core/cufree_report.h"

// dl/amdsafe: the HIP/ROCm half of the same startup report. SAME SHAPE as the header above
// -- std-only, header-only, every function `inline`, no out-of-line symbol -- so including
// it costs no link edge and adds NO SONAME this binary could die on before main(). That is
// the whole reason the sentence can be printed on a machine with no HIP stack at all.
#include "core/amdsafe_hip_stack.h"

#include <dlfcn.h>

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace {

// The sonames, spelled EXACTLY as `readelf -d build/apps/ninfer` spells them in DT_NEEDED.
// They are the two entries that make ld.so abort before main() on a machine without them.
constexpr std::string_view kCudartSoname = "libcudart.so.13";
constexpr std::string_view kCudaDriverSoname = "libcuda.so.1";

// MEASURED BY THE LOADER, not by the filesystem: dlopen() answers the question the dynamic
// linker would ask, in the same search order, and it does NOT execute any CUDA code -- the
// library is loaded and immediately unloaded, and no symbol is ever resolved out of it.
bool soname_is_findable(std::string_view soname) {
    const std::string name(soname);
    void* handle = ::dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) { return false; }
    ::dlclose(handle);
    return true;
}

void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "ninfer-cufree: the CUDA-free front door. Prints the startup report this "
                 "binary\n  can produce where the CUDA-linked front doors cannot start.\n"
                 "  usage: %s [--help]\n"
                 "  No options: this binary has nothing to configure. It takes no model, no "
                 "prompt\n  and no artifact, and it never opens a device.\n"
                 "  exit: 0 = the CUDA runtime and driver interface both load here;\n"
                 "        3 = a refusal (see the report on stdout).\n",
                 argv0);
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i] != nullptr ? argv[i] : "";
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0] != nullptr ? argv[0] : "ninfer-cufree");
            return 0;
        }
        std::fprintf(stderr, "ninfer-cufree: unrecognised argument '%s'\n",
                     std::string(arg).c_str());
        print_usage(argv[0] != nullptr ? argv[0] : "ninfer-cufree");
        return 2;
    }

    ninfer::caps::CufreeObservation obs;
    obs.sonames.push_back({kCudartSoname, soname_is_findable(kCudartSoname)});
    obs.sonames.push_back({kCudaDriverSoname, soname_is_findable(kCudaDriverSoname)});
    obs.witnesses = ninfer::caps::observe_cufree_witnesses();

    // Both sonames are pushed above, so this is measured; the guard is here so that a future
    // edit that drops one cannot silently turn "not measured" into a claimed class.
    const bool measured = ninfer::caps::cufree_sonames_measured(obs);

    // The report always goes out: a process that asks for it is never given silence.
    const std::string report = ninfer::caps::render_cufree_startup_report(obs);
    std::fputs(report.c_str(), stdout);

    // ---- dl/amdsafe: the HIP/ROCm stack, measured the same way and printed the same way. ----
    // The sonames are probed with the SAME dlopen() helper, and the witness table is read under
    // `/` because that is the root this process actually lives in.
    ninfer::caps::AmdsafeHipObservation hip_obs;
    for (ninfer::caps::AmdsafeHipSonameProbe probe : ninfer::caps::amdsafe_hip_soname_table()) {
        probe.findable = soname_is_findable(probe.soname);
        hip_obs.sonames.push_back(probe);
    }
    hip_obs.witnesses = ninfer::caps::observe_amdsafe_hip_witnesses();

    const std::string hip_report = ninfer::caps::render_amdsafe_hip_report(hip_obs);
    std::fputs(hip_report.c_str(), stdout);

    if (!ninfer::caps::amdsafe_hip_sonames_measured(hip_obs)) {
        std::fprintf(stderr,
                     "ninfer-cufree: internal error -- the HIP soname probe did not run\n");
        return 4;
    }
    const ninfer::caps::AmdsafeHipClass hip_cls = ninfer::caps::amdsafe_hip_classify(hip_obs);
    if (ninfer::caps::amdsafe_hip_class_announce_once(hip_cls)) {
        std::fprintf(stderr, "ninfer-cufree: measured class %s\n",
                     ninfer::caps::amdsafe_hip_class_token(hip_cls));
    }
    // THE EXIT CODE IS DELIBERATELY UNCHANGED. `0` still means "the CUDA stack is findable" and
    // `3` still means "it is not", so no existing caller of this binary changes behaviour; the
    // HIP class is announced on stderr and rendered on stdout. A new exit code here would be a
    // silent behavioural change for every script that already reads this one.

    if (!measured) {
        std::fprintf(stderr, "ninfer-cufree: internal error -- the soname probe did not run\n");
        return 3;
    }

    const ninfer::caps::CufreeCudaClass cls = ninfer::caps::cufree_classify(obs);

    // The ANNOUNCEMENT is the part that is once-per-distinct-class rather than
    // once-per-process. This binary is one-shot, so the two coincide here; the property
    // exists for the long-lived callers, where a second, different class must not be silent
    // -- the defect dl/vendorseam measured in the landed arch_caps.h form.
    if (ninfer::caps::cufree_class_announce_once(cls)) {
        std::fprintf(stderr, "ninfer-cufree: measured class %s\n",
                     ninfer::caps::cufree_cuda_class_token(cls));
    }

    return cls == ninfer::caps::CufreeCudaClass::CudaRuntimeAndDriverFindable ? 0 : 3;
}
