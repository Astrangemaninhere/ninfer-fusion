#ifndef NINFER_CORE_CUFREE_REPORT_H
#define NINFER_CORE_CUFREE_REPORT_H

// ---------------------------------------------------------------------------
// THE CUDA-FREE STARTUP REPORT (dl/cufree).
//
// WHY THIS HEADER EXISTS, IN ONE MEASURED SENTENCE. The three operator front doors
// (apps/cli/main.cpp -> ninfer, apps/serve/main.cpp -> ninfer-serve,
// apps/perplexity/main.cpp -> ninfer-perplexity) all link CUDA DYNAMICALLY:
// `readelf -d build/apps/ninfer` shows `NEEDED libcudart.so.13` and `NEEDED libcuda.so.1`,
// and `nm -D` shows `cudaGetDeviceCount` as an UNDEFINED IMPORT. On a machine whose
// loader cannot find either soname, ld.so therefore aborts BEFORE `main()` runs, and the
// message is the loader's, not this tree's. NO C++ REFUSAL IN THIS TREE CAN BE PRINTED
// THERE -- not the render_amd_format_refusal() refusal, not unknown_arch_warning(), not
// render_build_capability_surface(), none of them. The earliest point inside the code
// (apps/cli/main.cpp:328, the device-free --capability-report surface) genuinely precedes
// every device call, and it still cannot start.
//
// SO THE REFUSAL HAS TO LIVE SOMEWHERE THAT LINKS NOTHING. This header is that place:
// std-only, header-only, every function `inline`, NO out-of-line symbol, and therefore NO
// link edge into any library. That property is not a preference -- it is measured. The
// obvious alternative was to put this text beside render_build_capability_surface() in
// src/core/arch_caps.h, which is where the tree's other operator-facing refusal text
// lives (apps/cli/main.cpp:324-326: "so the arch line both print has ONE home and the two
// cannot disagree"). That alternative was TESTED and it does not work: a TU that includes
// core/arch_caps.h and calls render_build_capability_surface() does not link without
// ninfer_artifact, because `ninfer::artifact::format_name(NumericFormat)` is declared
// there and DEFINED out of line. Linking ninfer_artifact pulls ninfer_core, which carries
// `target_link_libraries(ninfer_core PUBLIC CUDA::cudart ...)` (src/CMakeLists.txt:69) --
// so the front door would be CUDA-free only for as long as the toolchain keeps spelling
// `--as-needed`, which is exactly the fragile shape this header exists to avoid.
// See dl/cufree/REPORT.md section 2 for the two-link measurement.
//
// WHAT THIS FILE MAY AND MAY NOT SAY. Read the renderer below. It names the CLASS OF WHAT
// WAS MEASURED and it does not name a card vendor, and that restraint is deliberate and
// tested:
//   * "no CUDA at all" and "CUDA present but no recognised device" are DIFFERENT
//     sentences, and both are produced here as different named classes
//     (CufreeCudaClass::CudaStackAbsent vs CufreeCudaClass::CudaRuntimeNoDriver) --
//     dl/vendorseam refused a single cross-vendor clause that covered both, because one
//     clause over two situations is a false attribution.
//   * NO VENDOR IS ATTRIBUTED TO ANY ACCELERATOR. The witness table names the KERNEL
//     INTERFACE a path proves, and its `attributed_vendor` field is empty for every row,
//     unchecked. That is the SAME shape src/core/arch_caps.h already uses for kAmdLadder,
//     whose `probe_evidence` is "" on all six rows by design and whose struct comment says
//     making it non-empty is the edit that turns the table into support.
//   * The reason for that restraint is MEASURED ON THIS BOX, not stylistic:
//     /sys/bus/pci/devices/ here holds only vendor 0x1af4 (virtio) and 0x1414 (Microsoft,
//     class 0x030200) -- THERE IS NO 0x10DE PCI DEVICE, /proc/driver/nvidia does not exist,
//     /dev/nvidia* does not exist, and /sys/module holds no nvidia module -- while an
//     RTX 5090 D is present and the CUDA stack loads. A detector that named a vendor from
//     PCI IDs or from /dev/nvidia* would report "no NVIDIA accelerator" on a machine that
//     is running one. That is the plausible-sounding false attribution the fleet's rule
//     forbids, and it is why the only accelerator evidence this file reports is
//     "this path exists", printed with the path.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::caps {

// ---------------------------------------------------------------------------
// The class of what was MEASURED about the CUDA stack. Each value names a measurement,
// not a conclusion: nothing here says which card is installed.
// ---------------------------------------------------------------------------
enum class CufreeCudaClass : std::uint8_t {
    // Both sonames the CUDA-linked front doors need were found by the loader. On a machine
    // in this class those front doors WOULD start, so this binary has nothing to rescue and
    // says so rather than implying a fault.
    CudaRuntimeAndDriverFindable = 0,
    // libcudart findable, libcuda.so.1 NOT. A CUDA runtime is installed with no driver
    // interface behind it. This is NOT the "no CUDA" class.
    CudaRuntimeNoDriver = 1,
    // libcuda.so.1 findable, libcudart NOT. A driver interface is installed with no runtime
    // beside it -- a CUDA-linked front door dies here on libcudart.so.13 alone.
    CudaDriverNoRuntime = 2,
    // NEITHER soname findable. THE CLASS THE THREE CUDA-LINKED FRONT DOORS CANNOT REPORT AT
    // ALL, because ld.so kills them before main(). It is also the class every non-CUDA
    // accelerator belongs to: an AMD, Intel, domestic-Chinese or NPU machine that has no
    // CUDA stack installed lands exactly here, and the honest statement is exactly that.
    CudaStackAbsent = 3,
};

[[nodiscard]] inline const char* cufree_cuda_class_token(CufreeCudaClass cls) {
    switch (cls) {
    case CufreeCudaClass::CudaRuntimeAndDriverFindable: return "cuda-stack-findable";
    case CufreeCudaClass::CudaRuntimeNoDriver:          return "cuda-runtime-no-driver";
    case CufreeCudaClass::CudaDriverNoRuntime:          return "cuda-driver-no-runtime";
    case CufreeCudaClass::CudaStackAbsent:              return "cuda-stack-absent";
    }
    return "?";
}

// One sentence per class, stating only what the probe answered.
[[nodiscard]] inline const char* cufree_cuda_class_text(CufreeCudaClass cls) {
    switch (cls) {
    case CufreeCudaClass::CudaRuntimeAndDriverFindable:
        return "the CUDA runtime and the CUDA driver interface BOTH load here, so the "
               "CUDA-linked front doors of this tree would also have started";
    case CufreeCudaClass::CudaRuntimeNoDriver:
        return "a CUDA runtime loads here but libcuda.so.1 does NOT, so the CUDA stack is "
               "installed without a driver interface behind it";
    case CufreeCudaClass::CudaDriverNoRuntime:
        return "libcuda.so.1 loads here but the CUDA runtime does NOT, so a driver "
               "interface is installed with no runtime beside it";
    case CufreeCudaClass::CudaStackAbsent:
        return "NEITHER the CUDA runtime NOR the CUDA driver interface loads here: this "
               "machine has no CUDA stack, which is the honest and complete statement";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// The inputs. EVERY ONE IS A PARAMETER, so the decision and the text below can be
// exercised on a host with no GPU and no CUDA -- which is the only form of evidence this
// box can produce for them. That is the tree's own rule for host-only evidence
// (tests/CMakeLists.txt:758-760: "host-only and take every input as a PARAMETER").
// ---------------------------------------------------------------------------

// A soname, spelled EXACTLY as `readelf -d` spells it in DT_NEEDED. `findable` is the
// caller's measurement; this header never calls dlopen() itself, so that the decision is a
// pure function of data.
struct CufreeSonameProbe {
    std::string_view soname;   // e.g. "libcudart.so.13" -- the DT_NEEDED string, verbatim
    bool             findable = false;
};

// A kernel interface, named by the witness path that proves it.
//
// `interface_name` is the INTERFACE's own name. `attributed_vendor` is deliberately a
// separate, EMPTY-UNLESS-MEASURED field: naming a card vendor from a device-node name is
// an inference, not a measurement, and this box cannot measure any of them. It stays empty
// for every row and the renderer prints that it is empty.
struct CufreeWitness {
    std::string_view path;              // the path that was tested, e.g. "/dev/kfd"
    std::string_view interface_name;    // the interface's own name -- NOT a card vendor
    std::string_view attributed_vendor; // "" = NOT ATTRIBUTED. Do not fill this without a card.
    bool             present = false;
};

struct CufreeObservation {
    std::vector<CufreeSonameProbe> sonames;
    std::vector<CufreeWitness>     witnesses;
};

// ---------------------------------------------------------------------------
// THE WITNESS TABLE. Every row names the KERNEL INTERFACE its path proves. Rows are added
// only for interfaces whose path is a stable, documented kernel ABI surface; a row whose
// path is a vendor's own choice is still reported by its path, which is the witness.
//
// `attributed_vendor` is "" (NOT ATTRIBUTED) on every row and that is the point -- see the
// header comment for the measured counterexample on this box.
//
// The list is NOT a census of accelerators and does not claim to be complete: a path
// absent from this table is simply not tested, and the renderer prints the paths it did
// test so the report cannot be read as "nothing else exists".
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::vector<CufreeWitness> cufree_witness_table() {
    return {
        // WSL2's GPU paravirtualization node. PRESENT ON THE MACHINE THIS TABLE WAS WRITTEN
        // ON, which is an NVIDIA box with no NVIDIA PCI device and no /dev/nvidia* -- see the
        // header comment. It proves a GPU is reachable and does NOT name its vendor.
        {"/dev/dxg", "WSL GPU paravirtualization node (DirectX-Graphics kernel driver)", "", false},
        // The CUDA-on-Linux character devices. On this machine these are ABSENT and the CUDA
        // stack still loads through /dev/dxg, which is why their absence proves nothing here.
        {"/dev/nvidiactl", "NVIDIA character device control node", "", false},
        {"/proc/driver/nvidia", "NVIDIA kernel driver procfs entry", "", false},
        // ROCm's kernel compute interface. Named as the INTERFACE: ROCm documents KFD as its
        // compute driver, and no card of that class exists on this machine, so no vendor is
        // attributed above it.
        {"/dev/kfd", "ROCm kernel compute interface (KFD)", "", false},
        // The kernel's DRM render-node and accelerator class directories. These exist for
        // many vendors, so their presence names no vendor at all -- which is exactly why they
        // are rows here: a detector that read a vendor out of them would be guessing.
        {"/dev/dri", "DRM render-node directory", "", false},
        {"/dev/accel", "kernel accel-class device directory", "", false},
        {"/sys/class/accel", "kernel accel-class sysfs directory", "", false},
    };
}

// THE OBSERVER. Filesystem-only, std-only, and IT TAKES THE ROOT AS A PARAMETER -- the root
// defaults to "/" for the front door, and a test passes a synthetic directory tree instead.
// That is the tree's own rule for host-only evidence (tests/CMakeLists.txt:758-760), and it
// is the only way the refusal path can be exercised on a box that has a working CUDA stack.
[[nodiscard]] inline std::vector<CufreeWitness> observe_cufree_witnesses(
    const std::string& root = std::string("/")) {
    std::vector<CufreeWitness> observed = cufree_witness_table();
    for (CufreeWitness& wit : observed) {
        std::string full;
        full.reserve(root.size() + wit.path.size());
        full.append(root);
        full.append(wit.path);
        std::error_code ec;
        wit.present = std::filesystem::exists(full, ec) && !ec;
    }
    return observed;
}

// ---------------------------------------------------------------------------
// The classifier. Pure. The order matters and is the whole point: the two sonames are
// examined SEPARATELY, so "no CUDA at all" can never be reported for a machine that has
// half of it, and no accelerator evidence ever enters this decision.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool cufree_soname_findable(const CufreeObservation& obs,
                                                std::string_view soname) {
    for (const CufreeSonameProbe& probe : obs.sonames) {
        if (probe.soname == soname) { return probe.findable; }
    }
    // A soname the observer did not report is NOT measured, and an unmeasured soname must
    // not be read as an absent one: not-probed is not a negative result.
    return false;
}

[[nodiscard]] inline bool cufree_sonames_measured(const CufreeObservation& obs) {
    return obs.sonames.size() >= 2;
}

[[nodiscard]] inline CufreeCudaClass cufree_classify(const CufreeObservation& obs) {
    const bool runtime = cufree_soname_findable(obs, "libcudart.so.13");
    const bool driver  = cufree_soname_findable(obs, "libcuda.so.1");
    if (runtime && driver) { return CufreeCudaClass::CudaRuntimeAndDriverFindable; }
    if (runtime)           { return CufreeCudaClass::CudaRuntimeNoDriver; }
    if (driver)            { return CufreeCudaClass::CudaDriverNoRuntime; }
    return CufreeCudaClass::CudaStackAbsent;
}

// ---------------------------------------------------------------------------
// ONCE PER DISTINCT CLASS, NOT ONCE PER PROCESS.
//
// This is the shape dl/vendorseam's P1 patch took for arch_caps.h's UnknownArch branch,
// and it is taken here for the same measured reason. The FIRST form of that branch used
// `static std::atomic<bool>` and announced the first unlisted number and then went silent,
// so a second, DIFFERENT value was announced to nobody. A `bool` here would have the same
// hole: a process that sees class A and later class B reports only A.
//
// The set below can only ADD announcements. It removes none.
//
// AND ONE PROPERTY THIS SURFACE HAS THAT THE arch_caps.h ONE DOES NOT, kept deliberately:
// THERE IS NO SILENCING KNOB. arch_caps.h's branch reads NINFER_ARCH_WARN and accepts
// 0|off|false|no, so there "nothing to announce" and "the announcement was switched off"
// are indistinguishable -- the FFMA file's own named defect, reproduced
// (src/ops/kernel/gqa_attention_simt_ffma.cuh: "a report that says 'the probe answered X'
// while the route is X's opposite is the same lie in a smaller font"). This header reads
// no environment variable at all, so that hole cannot open here: what was announced is
// exactly what was seen.
//
// The state is a function-local static inside an `inline` function, which is the ODR's
// single instance across every TU and costs NO NEW SYMBOL IN ANY LIBRARY -- the technique
// gqa_attention_simt_ffma.cuh:259-263 names for exactly this reason ("the launcher objects
// must not grow a link dependency on device_probe.o just to ask a route question").
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool cufree_class_announce_once(CufreeCudaClass cls) {
    static std::mutex            cufree_announced_mutex;
    static std::vector<uint8_t>  cufree_announced;
    const std::uint8_t           value = static_cast<std::uint8_t>(cls);
    const std::lock_guard<std::mutex> lock(cufree_announced_mutex);
    for (const std::uint8_t seen : cufree_announced) {
        if (seen == value) { return false; }
    }
    cufree_announced.push_back(value);
    return true;
}

// ---------------------------------------------------------------------------
// GAP-1 CLOSED BY NAME: THE NEXT STEP IS PRINTED WHERE IT IS NEEDED, NOT ONLY WHERE IT IS NOT.
// ---------------------------------------------------------------------------
// MEASURED 2026-09-22 (dl/amdland, REPORT.md section 2(a-3)): the block that names the tree's
// own model-free build surface (`ninfer --capability-report`, apps/cli/main.cpp:328) was printed
// for CudaRuntimeAndDriverFindable ONLY -- the one class that does not need it, because on such a
// machine the other three front doors start too. The two renderings differed by exactly 239
// bytes, and cuda-stack-absent -- THE class this binary exists for -- was told what was missing
// and NOT told what to do next.
//
// AND THE COMMAND HAD TO BE CHOSEN, NOT COPIED. On a machine in the absent class
// `ninfer --capability-report` CANNOT RUN: it is one of the three front doors that link
// libcudart.so.13 / libcuda.so.1, and ld.so stops it before main() -- that is the whole reason
// this binary exists. Offering it as THIS machine's next step would be advice that cannot be
// followed, which is the same defect as a refusal that lies about its own remedy. So the
// command is NAMED WITH THE CONDITION IT NEEDS, and the line that runs HERE is the binary that
// is already running.
//
// ONE CALL SITE, ONE HOME. The class-dependent branching lives here, and the renderer calls this
// once for every class, so a class added later cannot be silently left without a next step.
[[nodiscard]] inline std::string cufree_next_step_text(CufreeCudaClass cls) {
    if (cls == CufreeCudaClass::CudaRuntimeAndDriverFindable) {
        // Nothing to add: that class's own block names the build surface already, and saying it
        // twice would be two homes for one sentence.
        return {};
    }
    std::string out;
    out += "\n  WHAT TO DO NEXT. The first line is a command that runs on THIS machine:\n";
    out += "    * run this binary again for this same report: `ninfer-cufree`. It takes no "
           "options,\n";
    out += "      opens no device, and re-measures both sonames on every run, so a second run "
           "after\n";
    out += "      you change the machine is a second measurement and NOT a cached answer.\n";
    out += "    * NO OTHER BINARY IN THIS TREE CAN BE YOUR NEXT STEP ON THIS MACHINE. The device "
           "count\n";
    out += "      and every format floor are decided when an artifact loads "
           "(src/targets/registry.cpp,\n";
    out += "      construct_target), and every binary that can decide them links "
           "libcudart.so.13 and\n";
    out += "      libcuda.so.1 (readelf -d), so ld.so stops it before main() here. The tree's "
           "own\n";
    out += "      model-free build surface is `ninfer --capability-report` "
           "(apps/cli/main.cpp:328): it is\n";
    out += "      named here for the machine where the CUDA stack IS present (see the class\n";
    out += "      cuda-stack-findable), and it is deliberately NOT offered as this machine's "
           "next\n";
    out += "      step, because advice that cannot run is not advice.\n";
    out += "    * IF THE ACCELERATOR ON THIS MACHINE IS NOT CUDA, no command in this build helps "
           "it\n";
    out += "      and none is offered: this build holds no backend and no kernel for it. That "
           "is the\n";
    out += "      build fact below -- WHAT IS MISSING IS A BACKEND, NOT A DRIVER.\n";
    return out;
}

// ---------------------------------------------------------------------------
// The renderer. Pure function of CufreeObservation; prints nothing, throws nothing, and
// never suppresses -- a caller that asks for the report always gets it.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string render_cufree_startup_report(const CufreeObservation& obs) {
    const CufreeCudaClass cls = cufree_classify(obs);
    std::string out;

    out += "ninfer: this binary STARTED on a machine the other front doors cannot start on.\n";
    out += "  binary      : ninfer-cufree -- an operator front door with NO CUDA IN ITS "
           "DT_NEEDED.\n";
    out += "  why it exists : ninfer, ninfer-serve and ninfer-perplexity all link "
           "libcudart.so.13\n";
    out += "                and libcuda.so.1 DYNAMICALLY (readelf -d). On a machine whose "
           "loader\n";
    out += "                cannot find either, ld.so aborts BEFORE main(), so no message "
           "from\n";
    out += "                this tree can be printed there at all. This binary links "
           "neither, so\n";
    out += "                it is the only front door that can say the following.\n";

    out += "\n  MEASURED -- what the loader was asked for, one soname at a time:\n";
    if (obs.sonames.empty()) {
        out += "    <NOT MEASURED: the observer reported no soname probe at all. This is not "
               "a\n     negative result and no class is claimed for it.>\n";
    }
    for (const CufreeSonameProbe& probe : obs.sonames) {
        out += "    ";
        out += probe.findable ? "findable    " : "NOT findable";
        out += " : ";
        out.append(probe.soname);
        out += "\n";
    }

    if (!cufree_sonames_measured(obs)) {
        // NOT-PROBED IS NOT A NEGATIVE RESULT. cufree_classify() would answer CudaStackAbsent
        // for an empty observation, and printing that would present an unprobed run as a
        // measured one -- the FFMA file's named defect. So the class is NOT claimed at all
        // here, and the renderer says which measurement is missing.
        out += "\n  CLASS       : NOT MEASURED. The observer reported fewer than the two "
               "DT_NEEDED\n";
        out += "                sonames this report is about, so NO class is claimed and no "
               "refusal\n";
        out += "                is made below. Not-probed is not a negative result, and this "
               "report\n";
        out += "                does not present an unprobed run as a measured one.\n";
    } else {
        out += "\n  CLASS       : ";
        out += cufree_cuda_class_token(cls);
        out += "\n                ";
        out += cufree_cuda_class_text(cls);
        out += ".\n";

        if (cls == CufreeCudaClass::CudaStackAbsent) {
            out += "\n  This is a COMPLETE statement and it is not a diagnosis of your card. "
                   "What is\n";
            out += "  absent is the CUDA stack -- not necessarily your accelerator.\n";
        } else if (cls == CufreeCudaClass::CudaRuntimeAndDriverFindable) {
            out += "\n  Nothing is wrong here that this binary can detect. Which device, if "
                   "any, is\n";
            out += "  served to this process is NOT answerable from this binary, and this "
                   "binary does\n";
            out += "  not guess it: the device count and every format floor are decided when "
                   "an\n";
            out += "  artifact loads (src/targets/registry.cpp, construct_target), and the "
                   "model-free\n";
            out += "  build surface is `ninfer --capability-report` (apps/cli/main.cpp:328).\n";
        }

        // GAP-1: ONE CALL FOR EVERY CLASS, INSIDE THE MEASURED BRANCH ONLY. An unmeasured
        // observation makes no refusal and therefore offers no next step either -- the same
        // not-probed-is-not-a-negative rule the class claim above obeys.
        out += cufree_next_step_text(cls);
    }

    out += "\n  ACCELERATOR INTERFACES OBSERVED. Each row names the INTERFACE a path proves,\n";
    out += "  printed with the path, so every clause below is falsifiable by `ls`. NO ROW "
           "NAMES A\n";
    out += "  CARD VENDOR: that would be an inference from a device-node name, and no card "
           "of any\n";
    out += "  of these classes is on the machine these rows were written on. The "
           "attribution field\n";
    out += "  is therefore empty for every row and is printed empty.\n";
    if (obs.witnesses.empty()) {
        out += "    <no witness path was tested>\n";
    }
    for (const CufreeWitness& wit : obs.witnesses) {
        out += "    ";
        out += wit.present ? "present" : "absent ";
        out += " : ";
        out.append(wit.path);
        out += "  (";
        out.append(wit.interface_name);
        out += ")\n";
        out += "               vendor attribution : ";
        out += wit.attributed_vendor.empty() ? std::string_view{"NOT ATTRIBUTED"}
                                             : wit.attributed_vendor;
        out += "\n";
    }

    out += "\n  WHAT THIS BUILD CARRIES. This is a BUILD FACT, not a detection, and it is "
           "the same\n";
    out += "  fact on every machine: the project is declared `project(ninfer LANGUAGES C CXX "
           "CUDA)`\n";
    out += "  (CMakeLists.txt:235) and links no HIP/ROCm, no Level-Zero/SYCL and no NPU "
           "runtime in\n";
    out += "  any target. An AMD, Intel, domestic-Chinese or NPU accelerator therefore has "
           "no\n";
    out += "  backend and no kernel in this binary: WHAT IS MISSING IS A BACKEND, NOT A "
           "DRIVER.\n";

    out += "\n  NOT CLAIMED BY THIS REPORT, and named so it cannot be read in:\n";
    out += "    * no device count, and no statement about any device's compute capability --\n";
    out += "      this binary calls no CUDA function at all;\n";
    out += "    * no vendor attributed to any accelerator interface (see above);\n";
    out += "    * no support claim for AMD, Intel, any domestic accelerator or any NPU, and\n";
    out += "      none for the CUDA device this machine does or does not have: this binary\n";
    out += "      has no probe, and a build fact is not a probe result;\n";
    out += "    * no format floor and no arch rung: those live in src/core/arch_caps.h and "
           "are\n";
    out += "      reported where they are decided, not duplicated here.\n";

    return out;
}

} // namespace ninfer::caps

#endif // NINFER_CORE_CUFREE_REPORT_H
