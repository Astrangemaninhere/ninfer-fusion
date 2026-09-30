#ifndef NINFER_CORE_AMDSAFE_HIP_STACK_H
#define NINFER_CORE_AMDSAFE_HIP_STACK_H

// ---------------------------------------------------------------------------
// THE HIP / ROCm STACK REPORT (dl/amdsafe, deliverable 1).
//
// WHY THIS HEADER EXISTS, IN ONE MEASURED SENTENCE. src/core/cufree_report.h closes exactly
// one hole: a machine where neither libcudart.so.13 nor libcuda.so.1 is findable, so ld.so
// kills the three CUDA-linked front doors (apps/cli, apps/serve, apps/perplexity) BEFORE
// main() runs and no C++ refusal in this tree can be printed there. It closes that hole for
// the CUDA stack and for nothing else. The SAME hole exists on the other side and was
// MEASURED OPEN: cufree_report.h's own witness table probes /dev/kfd
// (cufree_report.h:176, "ROCm kernel compute interface (KFD)") and then has no sentence to
// say about it -- there is no `hip-*` named class anywhere in this tree. So an operator who
//   * has an AMD accelerator,
//   * has the kernel interface for it (/dev/kfd, /dev/dri),
//   * and does NOT have the HIP runtime / ROCm userspace stack installed
// gets, today, the loader's own message from whatever HIP-linked binary they ran, and no
// sentence from this tree. THAT is the report the operator's complaint is about: "I do not
// want the user to come back and shout at me after seeing an error", and the loader's
// `cannot open shared object file` is exactly that error.
//
// SO THE REPORT HAS TO LIVE WHERE THE LOADER CANNOT KILL IT. This header is std-only,
// header-only, every function `inline`, NO out-of-line symbol, and therefore NO link edge
// into any library -- the property cufree_report.h:19-32 measures and this file copies
// rather than re-derives. A front door that includes it and links nothing else has neither
// `libcudart` nor `libamdhip64` in its DT_NEEDED (checked with `readelf -d`, NOT asserted),
// so on a machine with no ROCm at all it still reaches main() and still prints. The measured
// contrast is in the line below the report: a binary that DOES record DT_NEEDED
// libamdhip64.so.7 dies before main() on the same machine, and its last words are the
// loader's. Both readings are in dl/amdsafe/logs/.
//
// WHAT THIS FILE MAY AND MAY NOT SAY. It is written under the same restraint as
// cufree_report.h, for the same measured reason, and the restraint is not stylistic:
//   * NO VENDOR IS ATTRIBUTED TO ANY ACCELERATOR. `attributed_vendor` is empty on every
//     witness row and the renderer prints that it is empty. Naming a card vendor from a
//     device-node name is an inference, and cufree_report.h:47-55 records the counterexample
//     measured on this very box: /sys/bus/pci/devices holds no 0x10DE device, /dev/nvidia*
//     does not exist, and an RTX 5090 D is nonetheless running -- so a detector reading a
//     vendor out of PCI IDs or device nodes reports "no such accelerator" on a machine that
//     is running one.
//   * A `hip-stack-absent` CLASS IS NOT A STATEMENT ABOUT THE CARD. It says the HIP runtime
//     and the ROCm userspace driver interface do not load HERE. The card may be present, the
//     kernel interface may be present, and the class is still this one -- the class names the
//     missing STACK, and the renderer prints that sentence in the class's own block rather
//     than leaving the reader to infer it.
//   * THE SONAME VERSIONS ARE NOT MEASURED HERE. This box has no ROCm at all (no /opt/rocm,
//     no hipcc, no libamdhip64.so anywhere in the loader's path). The three runtime sonames
//     and the two driver-interface sonames below are a RANGE covering the releases a HIP
//     build records in DT_NEEDED, and every candidate is printed with its OWN verdict, so a
//     version this list does not carry is visible as "tested and not findable" rather than
//     folded into a single answer. Whoever runs this on a machine with ROCm installed owes
//     the reading that says which candidate was the live one; nothing here claims to know.

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::caps {

// ---------------------------------------------------------------------------
// The class of what was MEASURED about the HIP/ROCm stack. Each value names a measurement,
// not a conclusion: nothing here says which card is installed.
//
// The four values are the one-to-one pair of cufree_report.h's CufreeCudaClass, and the
// pairing is deliberate -- two vocabularies for one situation is how a tree starts
// disagreeing with itself, and one clause over two situations is a false attribution
// (cufree_report.h:37-41).
// ---------------------------------------------------------------------------
enum class AmdsafeHipClass : std::uint8_t {
    // A HIP runtime AND a ROCm userspace driver interface were both found by the loader. On a
    // machine in this class a HIP-linked front door would also have started, so this binary
    // has nothing to rescue and says so rather than implying a fault.
    HipRuntimeAndDriverFindable = 0,
    // A HIP runtime loads, the ROCm userspace driver interface (ROCr) does NOT. The HIP API
    // surface is installed with nothing behind it that can reach /dev/kfd. NOT the "no HIP"
    // class: half the stack is present and that half is what the operator would find when
    // they go looking.
    HipRuntimeNoDriver = 1,
    // The ROCm userspace driver interface loads, the HIP runtime does NOT. A driver interface
    // is installed with no HIP runtime beside it.
    HipDriverNoRuntime = 2,
    // NEITHER findable. A HIP-linked front door dies here on the runtime soname alone, before
    // main(), so a tree-side sentence cannot be printed by THAT binary. It is also the class
    // that has an accelerator of this kind present and the stack absent, which is the case the
    // class text names explicitly so it cannot be read as a verdict on the card.
    HipStackAbsent = 3,
};

[[nodiscard]] inline const char* amdsafe_hip_class_token(AmdsafeHipClass cls) {
    switch (cls) {
    case AmdsafeHipClass::HipRuntimeAndDriverFindable: return "hip-stack-findable";
    case AmdsafeHipClass::HipRuntimeNoDriver:          return "hip-runtime-no-driver";
    case AmdsafeHipClass::HipDriverNoRuntime:          return "hip-driver-no-runtime";
    case AmdsafeHipClass::HipStackAbsent:              return "hip-stack-absent";
    }
    return "?";
}

// One sentence per class, stating only what the probe answered.
[[nodiscard]] inline const char* amdsafe_hip_class_text(AmdsafeHipClass cls) {
    switch (cls) {
    case AmdsafeHipClass::HipRuntimeAndDriverFindable:
        return "the HIP runtime and the ROCm userspace driver interface BOTH load here, so a "
               "HIP-linked front door of this tree would also have started";
    case AmdsafeHipClass::HipRuntimeNoDriver:
        return "a HIP runtime loads here but the ROCm userspace driver interface does NOT, so "
               "the HIP API surface is installed with nothing behind it that can reach the "
               "kernel compute interface";
    case AmdsafeHipClass::HipDriverNoRuntime:
        return "the ROCm userspace driver interface loads here but the HIP runtime does NOT, so "
               "a driver interface is installed with no HIP runtime beside it";
    case AmdsafeHipClass::HipStackAbsent:
        return "NEITHER the HIP runtime NOR the ROCm userspace driver interface loads here: the "
               "HIP/ROCm STACK is absent, which is the honest and complete statement";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// The inputs. EVERY ONE IS A PARAMETER, so the decision and the text below can be exercised
// on a host with no GPU, no ROCm and no AMD silicon -- which is the only form of evidence
// this box can produce for them. That is the tree's own rule for host-only evidence
// (tests/CMakeLists.txt:758-760: "host-only and take every input as a PARAMETER"), and
// cufree_report.h:118-123 states it for the CUDA side.
// ---------------------------------------------------------------------------

// A soname, spelled EXACTLY as `readelf -d` spells it in DT_NEEDED. `findable` is the
// caller's measurement; this header never calls dlopen() itself, so that the decision is a
// pure function of data.
struct AmdsafeHipSonameProbe {
    std::string_view soname;    // e.g. "libamdhip64.so.7" -- the DT_NEEDED string, verbatim
    std::string_view role;      // "HIP runtime" | "ROCm userspace driver interface"
    bool             findable = false;
};

// A kernel interface or a stack install path, named by the witness path that proves it.
//
// `interface_name` is the INTERFACE's own name. `attributed_vendor` is deliberately a
// separate, EMPTY-UNLESS-MEASURED field -- see the header comment and cufree_report.h:133-144.
struct AmdsafeHipWitness {
    std::string_view path;              // the path that was tested, e.g. "/dev/kfd"
    std::string_view interface_name;    // the interface's own name -- NOT a card vendor
    std::string_view kind;              // "kernel ABI surface" | "stack install path"
    std::string_view attributed_vendor; // "" = NOT ATTRIBUTED. Do not fill this without a card.
    bool             present = false;
};

struct AmdsafeHipObservation {
    std::vector<AmdsafeHipSonameProbe> sonames;
    std::vector<AmdsafeHipWitness>     witnesses;
    // The filesystem root the witnesses were looked up under. Printed verbatim, because a
    // witness row is only falsifiable by `ls` if the reader knows WHICH tree was `ls`-ed --
    // and because a namespace run and a bare-metal run must not look the same on paper.
    std::string                        witness_root = std::string("/");
};

// ---------------------------------------------------------------------------
// THE SONAME CANDIDATE TABLES. Split by ROLE, because the whole point of the four classes is
// that the two roles are measured SEPARATELY -- one clause over two situations is a false
// attribution, and "half the stack is installed" must not be spelled the same as "none is".
//
// A RANGE, NOT A PIN, and the reason is stated rather than hidden: this box has no ROCm, so
// no version string here has been measured against a real install. Every candidate is probed
// and every candidate is printed with its own verdict, so printing "not findable" for a
// version that exists in the world but not on this machine is a true reading of a bounded
// question, not a claim that the version does not exist.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::vector<AmdsafeHipSonameProbe> amdsafe_hip_runtime_sonames() {
    return {
        {"libamdhip64.so.7", "HIP runtime", false},
        {"libamdhip64.so.6", "HIP runtime", false},
        {"libamdhip64.so.5", "HIP runtime", false},
        {"libamdhip64.so",   "HIP runtime", false},
    };
}

[[nodiscard]] inline std::vector<AmdsafeHipSonameProbe> amdsafe_hip_driver_sonames() {
    return {
        // ROCr: the userspace half of the ROCm driver stack. It is what opens /dev/kfd.
        {"libhsa-runtime64.so.1", "ROCm userspace driver interface (ROCr)", false},
        {"libhsa-runtime64.so",   "ROCm userspace driver interface (ROCr)", false},
    };
}

[[nodiscard]] inline std::vector<AmdsafeHipSonameProbe> amdsafe_hip_soname_table() {
    std::vector<AmdsafeHipSonameProbe> all = amdsafe_hip_runtime_sonames();
    const std::vector<AmdsafeHipSonameProbe> driver = amdsafe_hip_driver_sonames();
    all.insert(all.end(), driver.begin(), driver.end());
    return all;
}

// ---------------------------------------------------------------------------
// THE WITNESS TABLE. Every row names the KERNEL INTERFACE its path proves, or -- for the two
// install-root rows -- says in its own `kind` field that it is a stack install path and NOT a
// kernel ABI surface. Splitting that out is the same discipline as cufree_report.h's
// attributed_vendor field: a row that cannot prove what the other rows prove is labelled, not
// silently averaged in.
//
// The list is NOT a census and does not claim to be complete: a path absent from this table is
// simply not tested, and the renderer prints the paths it did test so the report cannot be read
// as "nothing else exists".
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::vector<AmdsafeHipWitness> amdsafe_hip_witness_table() {
    return {
        // ROCm's kernel compute interface. This is the row cufree_report.h:176 already carries;
        // it is repeated here because this report is the one that has a sentence for it.
        {"/dev/kfd", "ROCm kernel compute interface (KFD)", "kernel ABI surface", "", false},
        // The kernel's DRM render-node directory. It exists for many vendors, so its presence
        // names no vendor at all -- which is exactly why it is a row here rather than a probe.
        {"/dev/dri", "DRM render-node directory", "kernel ABI surface", "", false},
        // The kernel module behind those two. A module NAME is a kernel fact, not a card fact.
        {"/sys/module/amdgpu", "amdgpu kernel module", "kernel ABI surface", "", false},
        // Stack install paths. These are a vendor's own layout choice, NOT a kernel ABI
        // surface, and they are labelled as such so no reader treats their absence as a
        // hardware measurement.
        {"/opt/rocm", "ROCm userspace install root", "stack install path", "", false},
        {"/opt/rocm/bin/hipcc", "HIP compiler driver", "stack install path", "", false},
        {"/etc/ld.so.conf.d/rocm.conf", "ROCm loader path drop-in", "stack install path", "",
         false},
    };
}

// THE OBSERVER. Filesystem-only, std-only, and IT TAKES THE ROOT AS A PARAMETER -- the root
// defaults to "/" for the front door, and a test or a namespace run passes the tree it built.
// It reports the root it used, so two different roots cannot be confused for one another in a
// log.
[[nodiscard]] inline std::vector<AmdsafeHipWitness> observe_amdsafe_hip_witnesses(
    const std::string& root = std::string("/")) {
    std::vector<AmdsafeHipWitness> observed = amdsafe_hip_witness_table();
    for (AmdsafeHipWitness& wit : observed) {
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
// The classifier. Pure. The two ROLES are examined separately -- runtime candidates and driver
// candidates each get their own `any-of` -- so "no HIP at all" can never be reported for a
// machine that has half of it, and no accelerator evidence ever enters this decision.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool amdsafe_hip_role_findable(const AmdsafeHipObservation& obs,
                                                    std::string_view role) {
    for (const AmdsafeHipSonameProbe& probe : obs.sonames) {
        if (probe.role == role && probe.findable) { return true; }
    }
    // No candidate of this role was findable. If the observer reported NO candidate of this
    // role at all that is NOT a negative result, and amdsafe_hip_sonames_measured() below is
    // what stops the class from being claimed in that case.
    return false;
}

// Both roles must have been REPORTED (whatever the answer) before any class is claimed. The
// threshold is counted per role rather than as a total, so a future edit that drops the whole
// driver list cannot pass by having enough runtime rows.
[[nodiscard]] inline bool amdsafe_hip_sonames_measured(const AmdsafeHipObservation& obs) {
    std::size_t runtime_rows = 0;
    std::size_t driver_rows = 0;
    for (const AmdsafeHipSonameProbe& probe : obs.sonames) {
        if (probe.role == std::string_view("HIP runtime")) { ++runtime_rows; }
        if (probe.role == std::string_view("ROCm userspace driver interface (ROCr)")) {
            ++driver_rows;
        }
    }
    return runtime_rows >= 1 && driver_rows >= 1;
}

[[nodiscard]] inline AmdsafeHipClass amdsafe_hip_classify(const AmdsafeHipObservation& obs) {
    const bool runtime = amdsafe_hip_role_findable(obs, "HIP runtime");
    const bool driver = amdsafe_hip_role_findable(obs, "ROCm userspace driver interface (ROCr)");
    if (runtime && driver) { return AmdsafeHipClass::HipRuntimeAndDriverFindable; }
    if (runtime)           { return AmdsafeHipClass::HipRuntimeNoDriver; }
    if (driver)            { return AmdsafeHipClass::HipDriverNoRuntime; }
    return AmdsafeHipClass::HipStackAbsent;
}

// ---------------------------------------------------------------------------
// ONCE PER DISTINCT CLASS, NOT ONCE PER PROCESS -- the shape cufree_report.h:232-267 takes,
// with the same measured reason (dl/vendorseam's `static std::atomic<bool>` announced the
// first unlisted number and then went silent, so a second, DIFFERENT value was announced to
// nobody). The set below can only ADD announcements; it removes none.
//
// AND THE SAME PROPERTY THIS SURFACE KEEPS: THERE IS NO SILENCING KNOB. This header reads no
// environment variable at all, so "nothing to announce" and "the announcement was switched
// off" cannot be confused -- the defect dl/cufree names
// (src/ops/kernel/gqa_attention_simt_ffma.cuh: "a report that says 'the probe answered X'
// while the route is X's opposite is the same lie in a smaller font").
//
// The state is a function-local static inside an `inline` function, which is the ODR's single
// instance across every TU and costs NO NEW SYMBOL IN ANY LIBRARY.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool amdsafe_hip_class_announce_once(AmdsafeHipClass cls) {
    static std::mutex           amdsafe_announced_mutex;
    static std::vector<uint8_t> amdsafe_announced;
    const std::uint8_t          value = static_cast<std::uint8_t>(cls);
    const std::lock_guard<std::mutex> lock(amdsafe_announced_mutex);
    for (const std::uint8_t seen : amdsafe_announced) {
        if (seen == value) { return false; }
    }
    amdsafe_announced.push_back(value);
    return true;
}

// ---------------------------------------------------------------------------
// WHAT IS MISSING IS ... -- one line per class, and it names the STACK, never the card.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string amdsafe_hip_missing_text(AmdsafeHipClass cls) {
    switch (cls) {
    case AmdsafeHipClass::HipRuntimeAndDriverFindable:
        // Nothing is missing that this binary can detect. Saying "WHAT IS MISSING IS ..." here
        // would invent a fault, so the line is not printed for this class at all.
        return {};
    case AmdsafeHipClass::HipRuntimeNoDriver:
        return "WHAT IS MISSING IS THE ROCm USERSPACE DRIVER INTERFACE BEHIND AN INSTALLED HIP "
               "RUNTIME, NOT A CARD.";
    case AmdsafeHipClass::HipDriverNoRuntime:
        return "WHAT IS MISSING IS A HIP RUNTIME BESIDE AN INSTALLED ROCm DRIVER INTERFACE, NOT "
               "A CARD.";
    case AmdsafeHipClass::HipStackAbsent:
        return "WHAT IS MISSING IS THE HIP/ROCm STACK -- NOT NECESSARILY YOUR ACCELERATOR.";
    }
    return {};
}

// ---------------------------------------------------------------------------
// WHAT TO DO NEXT. Printed for every class EXCEPT the one that needs nothing, because
// cufree_report.h:269-288 records the measured gap where the ONE class that did not need the
// next step was the only one that got it, in a rendering 239 bytes longer than the one that
// did: "advice that cannot run is not advice".
//
// The first item is always the command that runs on THIS machine: this binary, again. It takes
// no options, opens no device, and re-measures every soname on every run, so a second run after
// changing the machine is a second measurement and not a cached answer.
//
// NO COMMAND IN THIS BUILD IS OFFERED AS THE FIX, and that is a BUILD FACT rather than a
// shrug: the project is declared `project(ninfer LANGUAGES C CXX CUDA)` (CMakeLists.txt:235)
// and links no HIP/ROCm runtime in any target, so no binary in this build can execute a HIP
// kernel even on a machine where the stack IS installed. WHAT IS MISSING IS A BACKEND, NOT
// ONLY A RUNTIME; the runtime is what an operator can install, the backend is not in this
// tree. Both halves are said, in that order, so neither is read as the other.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string amdsafe_hip_next_step_text(AmdsafeHipClass cls,
                                                           const AmdsafeHipObservation& obs) {
    if (cls == AmdsafeHipClass::HipRuntimeAndDriverFindable) {
        // That class's own block already says nothing here can be detected as missing, and
        // saying it twice would be two homes for one sentence.
        return {};
    }
    std::string out;
    out += "\n  WHAT TO DO NEXT. The first two lines are commands that run on THIS machine:\n";
    out += "    * run this binary again for this same report. It takes no options, opens no "
           "device, and\n";
    out += "      re-measures every soname on every run, so a second run after you change the "
           "machine is a\n";
    out += "      second measurement and NOT a cached answer.\n";
    out += "    * `ls -l` the paths in the table above, at the root printed with it. Every "
           "present/absent\n";
    out += "      clause in this report is that one command, which is why none of them is a "
           "diagnosis.\n";
    out += "    * NO BINARY IN THIS BUILD CAN BE YOUR NEXT STEP EVEN ON A MACHINE WHERE THE "
           "STACK IS\n";
    out += "      INSTALLED. This project is declared `project(ninfer LANGUAGES C CXX CUDA)` "
           "(CMakeLists.txt:235)\n";
    out += "      and links no HIP/ROCm runtime in any target, so it holds no HIP kernel and no "
           "AMD backend.\n";
    out += "      WHAT IS MISSING IS A BACKEND, AND A BACKEND IS NOT INSTALLED -- IT IS BUILT. "
           "Installing\n";
    out += "      ROCm on this machine changes the sonames in this table and does not add one "
           "line of AMD\n";
    out += "      device code to this tree.\n";
    out += "    * if the accelerator on this machine is NOT HIP-capable at all, the two lines "
           "above are the\n";
    out += "      whole answer and no further step is offered, because none exists here.\n";
    if (!obs.witness_root.empty() && obs.witness_root != std::string("/")) {
        out += "    * the witness table above was read under the root `";
        out.append(obs.witness_root);
        out += "`, NOT under `/`.\n";
        out += "      A `ls` at `/` will therefore disagree with it and the table is the one "
               "that is right.\n";
    }
    return out;
}

// ---------------------------------------------------------------------------
// The renderer. Pure function of AmdsafeHipObservation; prints nothing, throws nothing, and
// never suppresses -- a caller that asks for the report always gets it.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string render_amdsafe_hip_report(const AmdsafeHipObservation& obs) {
    const AmdsafeHipClass cls = amdsafe_hip_classify(obs);
    std::string out;

    out += "\nninfer: HIP/ROCm STACK REPORT (dl/amdsafe). The CUDA half of this startup report "
           "is\n";
    out += "  src/core/cufree_report.h; this is the other side of the same hole, and it exists "
           "because\n";
    out += "  that header probes /dev/kfd (cufree_report.h:176) and then has no sentence to say "
           "about\n";
    out += "  it.\n";
    out += "  why it must be here : a front door that records DT_NEEDED libamdhip64.so.7 is "
           "stopped by\n";
    out += "                ld.so BEFORE main() on a machine without it, so the message the "
           "operator sees\n";
    out += "                is the loader's ('cannot open shared object file') and no C++ "
           "sentence in\n";
    out += "                this tree can be printed there. The binary printing THIS text links "
           "neither\n";
    out += "                libamdhip64 nor libcudart (apps/cufree's own DT_NEEDED is in the "
           "log), which\n";
    out += "                is the only reason the sentence below exists at all.\n";

    out += "\n  MEASURED -- what the loader was asked for, one soname at a time. Each row is a "
           "separate\n";
    out += "  dlopen() of that EXACT string, in the loader's own search order, immediately "
           "unloaded, with\n";
    out += "  no symbol ever resolved out of it:\n";
    if (obs.sonames.empty()) {
        out += "    <NOT MEASURED: the observer reported no soname probe at all. This is not a\n"
               "     negative result and no class is claimed for it.>\n";
    }
    for (const AmdsafeHipSonameProbe& probe : obs.sonames) {
        out += "    ";
        out += probe.findable ? "findable    " : "NOT findable";
        out += " : ";
        out.append(probe.soname);
        out += "  (";
        out.append(probe.role);
        out += ")\n";
    }

    if (!amdsafe_hip_sonames_measured(obs)) {
        // NOT-PROBED IS NOT A NEGATIVE RESULT. amdsafe_hip_classify() would answer
        // HipStackAbsent for an observation with no rows, and printing that would present an
        // unprobed run as a measured one.
        out += "\n  CLASS       : NOT MEASURED. The observer did not report at least one "
               "candidate of\n";
        out += "                EACH role (HIP runtime, ROCm userspace driver interface), so NO "
               "class is\n";
        out += "                claimed and no refusal is made below. Not-probed is not a "
               "negative result.\n";
        out += "\n  ACCELERATOR INTERFACES OBSERVED (witness root `";
        out.append(obs.witness_root);
        out += "`). Each row names\n  the INTERFACE a path proves, printed with the path, so "
               "every clause below is\n  falsifiable by `ls`. NO ROW NAMES A CARD VENDOR: that "
               "would be an inference from a\n  device-node or module name. The attribution "
               "field is empty for every row and is\n  printed empty.\n";
        if (obs.witnesses.empty()) {
            out += "    <no witness path was tested>\n";
        }
        for (const AmdsafeHipWitness& wit : obs.witnesses) {
            out += "    ";
            out += wit.present ? "present" : "absent ";
            out += " : ";
            out.append(wit.path);
            out += "  (";
            out.append(wit.interface_name);
            out += ") [";
            out.append(wit.kind);
            out += "]\n";
            out += "               vendor attribution : ";
            out += wit.attributed_vendor.empty() ? std::string_view{"NOT ATTRIBUTED"}
                                                 : wit.attributed_vendor;
            out += "\n";
        }
        return out;
    }

    out += "\n  CLASS       : ";
    out += amdsafe_hip_class_token(cls);
    out += "\n                ";
    out += amdsafe_hip_class_text(cls);
    out += ".\n";

    const std::string missing = amdsafe_hip_missing_text(cls);
    if (!missing.empty()) {
        out += "\n  ";
        out += missing;
        out += "\n";
    }
    if (cls == AmdsafeHipClass::HipStackAbsent) {
        out += "  This is a COMPLETE statement and it is NOT a diagnosis of your card. What is "
               "absent is\n";
        out += "  the HIP/ROCm STACK -- not necessarily your accelerator. A card of this kind "
               "can be present,\n";
        out += "  with its kernel interface in the table below, and the class is still this "
               "one.\n";
    } else if (cls == AmdsafeHipClass::HipRuntimeNoDriver ||
               cls == AmdsafeHipClass::HipDriverNoRuntime) {
        out += "  HALF OF THE STACK IS PRESENT. That is a different situation from `no HIP at "
               "all`, and it is\n";
        out += "  spelled differently on purpose: one clause over two situations is a false "
               "attribution, and\n";
        out += "  an operator who goes looking will find the half the table below reports as "
               "present.\n";
    } else {
        out += "\n  Nothing is wrong here that this binary can detect. Which device, if any, is "
               "served to\n";
        out += "  this process is NOT answerable from this binary, and this binary does not "
               "guess it: the\n";
        out += "  device count and every format floor are decided when an artifact loads "
               "(src/targets/registry.cpp,\n";
        out += "  construct_target).\n";
    }

    out += amdsafe_hip_next_step_text(cls, obs);

    out += "\n  ACCELERATOR INTERFACES OBSERVED (witness root `";
    out.append(obs.witness_root);
    out += "`). Each row names the\n  INTERFACE a path proves, printed with the path, so every "
           "clause below is falsifiable by\n  `ls`. NO ROW NAMES A CARD VENDOR, and the rows are "
           "split by what they can prove: a\n  `kernel ABI surface` row is a kernel fact, while "
           "a `stack install path` row is that\n  vendor's own layout choice and its absence is "
           "NOT a hardware measurement. The attribution\n  field is empty for every row and is "
           "printed empty.\n";
    if (obs.witnesses.empty()) {
        out += "    <no witness path was tested>\n";
    }
    for (const AmdsafeHipWitness& wit : obs.witnesses) {
        out += "    ";
        out += wit.present ? "present" : "absent ";
        out += " : ";
        out.append(wit.path);
        out += "  (";
        out.append(wit.interface_name);
        out += ") [";
        out.append(wit.kind);
        out += "]\n";
        out += "               vendor attribution : ";
        out += wit.attributed_vendor.empty() ? std::string_view{"NOT ATTRIBUTED"}
                                             : wit.attributed_vendor;
        out += "\n";
    }

    out += "\n  WHAT THIS BUILD CARRIES. This is a BUILD FACT, not a detection, and it is the "
           "same fact on\n  every machine: the project is declared `project(ninfer LANGUAGES C "
           "CXX CUDA)`\n  (CMakeLists.txt:235) and links no HIP/ROCm runtime in any target. An "
           "accelerator reachable only\n  through HIP therefore has NO BACKEND AND NO KERNEL IN "
           "THIS TREE, independently of\n  whether the runtime above loads.\n";

    out += "\n  NOT CLAIMED BY THIS REPORT, and named so it cannot be read in:\n";
    out += "    * no device count, and no statement about any device's capability -- this "
           "binary calls no\n";
    out += "      ROCm and no CUDA function at all;\n";
    out += "    * no vendor attributed to any accelerator interface or kernel module (see "
           "above);\n";
    out += "    * no claim that any particular ROCm RELEASE exists or does not exist. Only the "
           "sonames listed\n";
    out += "      above were tested, each one printed with its own verdict;\n";
    out += "    * no support claim for AMD, Intel, any domestic accelerator or any NPU, and "
           "none for the HIP\n";
    out += "      device this machine does or does not have: this binary has no probe, and a "
           "build fact is\n";
    out += "      not a probe result;\n";
    out += "    * no statement about whether a CPU path exists in this build. Whether an "
           "artifact can run\n";
    out += "      WITHOUT any GPU is a property of the EXECUTION TARGET, not of the stack, and "
           "it is\n";
    out += "      decided where a target is loaded -- not here.\n";

    return out;
}

} // namespace ninfer::caps

#endif // NINFER_CORE_AMDSAFE_HIP_STACK_H
