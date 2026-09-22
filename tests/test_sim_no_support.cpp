// ---------------------------------------------------------------------------
// test_sim_no_support.cpp -- A SIMULATED VERDICT MAY NOT ENTER THE SUPPORT LIST
//
// HOST-ONLY, and runnable with plain g++ because every subject is host-only:
//
//   g++ -std=c++20 -O1 -Wall -Wextra -I <repo>/src -I <repo>/include -I <repo>/tests
//       -I <repo>/third_party -DNINFER_HAVE_QPN=1 -DNINFER_SOURCE_DIR='"<repo>"'
//       <repo>/tests/test_sim_no_support.cpp
//       <repo>/build/src/libninfer_artifact.a <repo>/build/src/libninfer_core.a
//       -o <out>   &&   <out>
//
// (The two archives are on the line for ONE symbol -- ninfer::artifact::format_name -- which
// src/core/format_probe.h and src/core/kernel_route.h call instead of keeping a second copy of
// the format-name table. No CUDA, no device, no GPU.)
//
// WHAT IT PINS, and why each one is load-bearing rather than a style check
// -----------------------------------------------------------------------
// The measured history this file exists because of: the route layer had ZERO callers in the
// engine, six arms under NINFER_SIM_ARCH produced byte-identical ids BECAUSE the simulated rung
// never ran, and later a simulated `none` verdict turned out to be indistinguishable in a summary
// from "the run never reached a route decision" (`grep -c '[route]'` = 0 on an sm_89 request that
// died at the artifact gate). "compiles + gate admits" is not "the engine takes this path", and a
// simulated verdict is not a measurement.
//
//   S1  the laundering shape `supported=yes ... simulated=yes` is refused AT PARSE
//   S2  a simulated row (the honest form: supported=no + sim_rung) parses, and the GATE refuses it
//       BY NAME -- and the refusal carries SIMULATED, the leg, the RUNG, UNMEASURED/UNVERIFIED,
//       the two-prohibition note, and the literal command that would measure the real hardware
//   S3  a simulated row that did not record its rung is refused too, and the refusal says the rung
//       is MISSING instead of naming a rung it does not have
//   S4  the T7 shape (supported=yes leg=route_simulated) still refuses, naming the leg
//   S5  sim_rung must be a ladder rung: a fabricated rung is a parse error
//   S6  an unclassifiable `meta source=` is refused
//   S7  provenance/verdict contradiction in BOTH directions is refused
//   S8  the driver (tools/archkit/probe_formats.sh) cannot write a real_gpu list under the sim:
//       it refuses with rc=10 and an on-ramp, and its `source` field is computed, not declared
//   S9  render()->parse() round-trips the simulated row, so the marker is not decoration
//   S10 THE REAL LIST still parses, still admits a measured row, and contains NO simulated row
//   S11 THE TRAP: the simulator's `none` verdict and the table's `none` verdict produce the SAME
//       route string, and the ONLY discriminator on the line is the [SIMULATED] marker; and the
//       line is emitted for a `none` verdict at all (so "no line + banner present" means the run
//       never reached a route decision)
//   S12 text pins over the tree, so the discriminators S11 relies on cannot be removed quietly
#include "core/format_probe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

std::string source_dir() {
#ifdef NINFER_SOURCE_DIR
    return std::string(NINFER_SOURCE_DIR);
#else
    return std::string("/home/user/ninfer-fusion");
#endif
}

std::string write_temp(const std::string& name, const std::string& text) {
    const std::string dir = "/tmp/ninfer_sim_no_support";
    if (std::system(("mkdir -p " + dir).c_str()) != 0) { std::cerr << "cannot mkdir " << dir << '\n'; }
    const std::string path = dir + "/" + name;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
    out.close();
    return path;
}

// The measured capability vector used as the fixture key, which must equal what the caller's
// measured capabilities render to -- otherwise the gate would refuse for ABSENCE and never reach
// the simulated branch, and the test would pass while proving nothing.
constexpr const char* kKeyAll = "v1;ki=1;bf16=1;fp16=1;i8=1;f8f6f4=1;mxf4=1;smr=1";

ninfer::caps::MeasuredCapabilities measured_all_true() {
    ninfer::caps::MeasuredCapabilities m;
    m.kernel_image = true;
    m.bf16_mma = true;
    m.fp16_mma = true;
    m.int8_mma = true;
    m.fp8_kind_f8f6f4 = true;
    m.nvfp4_mma_block_scale = true;
    m.setmaxnreg = true;
    return m;
}

ninfer::caps::DeclaredIdentity declared_5090() {
    ninfer::caps::DeclaredIdentity d;
    d.name = "NVIDIA GeForce RTX 5090 D";
    d.cc_major = 12;
    d.cc_minor = 0;
    d.build_arch = "120a";
    return d;
}

// A list whose provenance says the run was a SIMULATED one. This is the shape the driver
// (tools/archkit/probe_formats.sh) writes when it is run under the simulator with
// NINFER_PROBE_ALLOW_SIM_ROUTE=1, and it is the only shape in which a simulated row is legal.
std::string sim_provenance(const std::string& key, const std::string& extra_entry) {
    return std::string("# simulated-route list (test fixture)\n") +
           "meta probe_tool=\"tools/archkit/probe_formats.sh\"\n"
           "meta criterion_id=\"linear-reduction-v1\"\n"
           "meta criterion_src=\"tests/ops/linear/linear_test_common.cpp:38-48\"\n"
           "meta reference=\"cpu_linear_gemm_fp64 (tests/ops/linear/linear_test_common.cpp:246)\"\n"
           "meta key=\"" + key + "\"\n"
           "meta date=\"2026-09-18\"\n"
           "meta binary_sha16=\"aaaaaaaaaaaaaaaa\"\n"
           "meta source=\"simulated_route\"\n"
           "meta sim_run_state=\"active:70\"\n"
           "meta sim_run_rung=\"70\"\n" +
           extra_entry;
}

// The REAL provenance shape (what the driver writes on a real card).
std::string real_provenance(const std::string& key, const std::string& extra_entry) {
    return std::string("# real-gpu list (test fixture)\n") +
           "meta probe_tool=\"tools/archkit/probe_formats.sh\"\n"
           "meta criterion_id=\"linear-reduction-v1\"\n"
           "meta criterion_src=\"tests/ops/linear/linear_test_common.cpp:38-48\"\n"
           "meta reference=\"cpu_linear_gemm_fp64 (tests/ops/linear/linear_test_common.cpp:246)\"\n"
           "meta key=\"" + key + "\"\n"
           "meta date=\"2026-09-18\"\n"
           "meta binary_sha16=\"aaaaaaaaaaaaaaaa\"\n"
           "meta source=\"real_gpu\"\n" +
           extra_entry;
}

std::string sim_entry(const char* key, bool supported, const char* leg, const char* rung) {
    std::ostringstream out;
    out << "entry key=\"" << key << "\" format=NVFP4 path=A4 band=17..open supported="
        << (supported ? "yes" : "no") << " leg=" << leg << " cases=1 geometries=1 arm_rc=0"
        << " ratio=0.5 case=\"simulated sm_70 route selection\" simulated=yes";
    if (rung != nullptr && rung[0] != '\0') { out << " sim_rung=" << rung; }
    out << " probe=\"build/tests sha16=aaaaaaaaaaaaaaaa\"\n";
    return out.str();
}

std::string real_entry(const char* key, const char* leg) {
    std::ostringstream out;
    out << "entry key=\"" << key << "\" format=NVFP4 path=A4 band=17..open supported=yes leg="
        << leg << " cases=3 geometries=1 arm_rc=0 ratio=0.5 case=\"NVFP4_A4 T=17\""
        << " probe=\"build/tests sha16=aaaaaaaaaaaaaaaa\"\n";
    return out.str();
}

// ---------------------------------------------------------------------------
// S1  THE LAUNDERING SHAPE. A row that admits its route was simulated and still claims support.
// ---------------------------------------------------------------------------
void s1_supported_and_simulated_is_refused_at_parse() {
    const auto list = ninfer::caps::FormatSupportList::parse(
        sim_provenance(kKeyAll, sim_entry(kKeyAll, true, "reduction_l2", "70")));
    check(!list.valid(), "S1: supported=yes + simulated=yes must not be a usable list");
    const std::string err = list.parse_error();
    check(err.find("simulated=yes") != std::string::npos,
          "S1: the parse error names the simulated marker");
    check(err.find("supported=yes") != std::string::npos,
          "S1: the parse error names the support claim it contradicts");
    check(err.find("ROUTE") != std::string::npos,
          "S1: the parse error says WHY a simulated verdict is not support (it is a route result)");
    check(list.entries().empty(), "S1: the refused row is not kept for a lookup");
}

// ---------------------------------------------------------------------------
// S2  THE NAMED REFUSAL. The honest simulated row parses; the gate refuses it, and the refusal
//     carries everything a reader needs to obtain the measurement.
// ---------------------------------------------------------------------------
void s2_gate_refuses_simulated_by_name_with_the_on_ramp() {
    const std::string path = write_temp(
        "s2.list", sim_provenance(kKeyAll, sim_entry(kKeyAll, false, "route_simulated", "70")));
    setenv("NINFER_FORMAT_PROBE_LIST", path.c_str(), 1);
    const auto verdict = ninfer::caps::format_probe_gate(
        declared_5090(), measured_all_true(), ninfer::artifact::NumericFormat::NVFP4, 1024,
        "test/artifact", "aaaaaaaaaaaaaaaa");
    check((verdict.enabled && !verdict.allowed) || !verdict.enabled,
          "S2: a simulated row is never an admission");
    if (!verdict.enabled) {
        std::cerr << "NOTE S2: the simulated fixture did not parse; see the parse error\n";
        unsetenv("NINFER_FORMAT_PROBE_LIST");
        return;
    }
    std::printf("--- S2 measured refusal text ---\n%s\n--- end ---\n", verdict.why.c_str());
    // (a) NAMED as simulated, and named FIRST.
    check(verdict.why.rfind("SIMULATED", 0) == 0, "S2: the refusal LEADS with SIMULATED");
    check(verdict.why.find("REFUSED AS SUPPORT") != std::string::npos,
          "S2: the refusal says what it is refusing, not just 'unsupported'");
    // (b) the LEG that produced it.
    check(verdict.why.find("route_simulated") != std::string::npos,
          "S2: the refusal names the leg that produced the verdict");
    // (c) the RUNG, both as a bare number and as a capability.
    check(verdict.why.find("sim_rung=70") != std::string::npos,
          "S2: the refusal quotes the recorded sim_rung");
    check(verdict.why.find("sm_70") != std::string::npos,
          "S2: the refusal names the simulated rung as sm_70");
    // (d) it cannot be read as a capability verdict.
    check(verdict.why.find("UNMEASURED") != std::string::npos,
          "S2: the refusal says the SUPPORT is UNMEASURED");
    check(verdict.why.find("UNVERIFIED") != std::string::npos,
          "S2: the refusal says the ARITHMETIC is UNVERIFIED");
    check(verdict.why.find("arithmetic") != std::string::npos,
          "S2: the refusal explains the missing thing in words (arithmetic)");
    // (e) the ON-RAMP: the literal command that would measure the real hardware.
    check(verdict.why.find("NINFER_PROBE_KEY=") != std::string::npos,
          "S2: the refusal carries the literal probe command (the on-ramp)");
    check(verdict.why.find("tools/archkit/probe_formats.sh") != std::string::npos,
          "S2: the on-ramp names the driver by path");
    // (f) the two prohibitions are distinguished, so nobody trades the support rule for the
    //     performance rule (src/core/arch_sim.h:59-70 forbids the latter and is untouched).
    check(verdict.why.find("PERFORMANCE") != std::string::npos,
          "S2: the refusal notes the separate PROHIBITION on performance claims from a sim run");
    // (g) and it is refused, not silently admitted.
    check(!verdict.allowed, "S2: the verdict is refused");
    unsetenv("NINFER_FORMAT_PROBE_LIST");
}

// ---------------------------------------------------------------------------
// S3  A simulated row that did not record its rung is still refused, and the refusal REPORTS the
//     missing rung instead of inventing one.
// ---------------------------------------------------------------------------
void s3_missing_rung_is_reported_not_invented() {
    const std::string path = write_temp(
        "s3.list", sim_provenance(kKeyAll, sim_entry(kKeyAll, false, "route_simulated", "")));
    setenv("NINFER_FORMAT_PROBE_LIST", path.c_str(), 1);
    const auto verdict = ninfer::caps::format_probe_gate(
        declared_5090(), measured_all_true(), ninfer::artifact::NumericFormat::NVFP4, 1024,
        "test/artifact", "aaaaaaaaaaaaaaaa");
    check(verdict.enabled && !verdict.allowed, "S3: a simulated row without a rung is still refused");
    check(verdict.why.find("DID NOT RECORD") != std::string::npos,
          "S3: the refusal says the row did not record a rung");
    check(verdict.why.find("defect of the row") != std::string::npos,
          "S3: and names that as a defect rather than as a neutral omission");
    unsetenv("NINFER_FORMAT_PROBE_LIST");
}

// ---------------------------------------------------------------------------
// S4  The T7 shape (supported=yes + leg=route_simulated) still refuses and still names the leg.
//     This is a REGRESSION guard on the refusal text test_format_probe.cpp asserts on.
// ---------------------------------------------------------------------------
void s4_t7_shape_still_names_the_leg() {
    const std::string path = write_temp(
        "s4.list", sim_provenance(kKeyAll, sim_entry(kKeyAll, true, "route_simulated", "70")));
    setenv("NINFER_FORMAT_PROBE_LIST", path.c_str(), 1);
    const auto verdict = ninfer::caps::format_probe_gate(
        declared_5090(), measured_all_true(), ninfer::artifact::NumericFormat::NVFP4, 1024,
        "test/artifact", "aaaaaaaaaaaaaaaa");
    // NOTE: this row is also refused AT PARSE by S1's rule, so an unparseable list is a legitimate
    // outcome here -- what must not happen is an admission.
    check(!verdict.allowed, "S4: the T7 shape is refused (as an admission) either way");
    check(verdict.why.find("route_simulated") != std::string::npos,
          "S4: the leg is named in the refusal, whichever lock refused it");
    check(verdict.why.find("simulator") != std::string::npos ||
              verdict.why.find("SIMULATED") != std::string::npos,
          "S4: and the refusal says the verdict came from the architecture simulator");
    std::printf("--- S4 why (may be a parse refusal) ---\n%s\n--- end ---\n", verdict.why.c_str());
    unsetenv("NINFER_FORMAT_PROBE_LIST");
}

// ---------------------------------------------------------------------------
// S5/S6/S7  The three other ways a list can be self-contradictory about simulation.
// ---------------------------------------------------------------------------
void s5_sim_rung_must_be_a_ladder_rung() {
    const auto fabricated = ninfer::caps::FormatSupportList::parse(
        sim_provenance(kKeyAll, sim_entry(kKeyAll, false, "route_simulated", "999")));
    check(!fabricated.valid(), "S5: sim_rung=999 has no kArchLadder row and is refused");
    check(fabricated.parse_error().find("kArchLadder") != std::string::npos,
          "S5: the parse error says the rung must exist in the ladder");

    const auto not_a_number = ninfer::caps::FormatSupportList::parse(
        sim_provenance(kKeyAll, sim_entry(kKeyAll, false, "route_simulated", "seventy")));
    check(!not_a_number.valid(), "S5: a non-numeric sim_rung is refused");

    const auto real_rung = ninfer::caps::FormatSupportList::parse(
        sim_provenance(kKeyAll, sim_entry(kKeyAll, false, "route_simulated", "70")));
    check(real_rung.valid(), "S5: sim_rung=70 is a real rung and parses");
    const auto* entry = real_rung.find(kKeyAll, ninfer::artifact::NumericFormat::NVFP4, 1024, "A4");
    check(entry != nullptr, "S5: the simulated entry is findable");
    if (entry != nullptr) {
        check(entry->sim_rung == "70", "S5: the rung is retained on the entry");
        check(entry->is_simulated(), "S5: leg=route_simulated is normalised to a simulated row");
        check(!entry->supported, "S5: and it does not claim support");
    }
}

void s6_unclassifiable_source_is_refused() {
    std::string text = sim_provenance(kKeyAll, sim_entry(kKeyAll, false, "route_simulated", "70"));
    const std::size_t at = text.find("source=\"simulated_route\"");
    text.replace(at, std::strlen("source=\"simulated_route\""), "source=\"looks_fine_to_me\"");
    const auto list = ninfer::caps::FormatSupportList::parse(text);
    check(!list.valid(), "S6: an unclassifiable provenance is refused");
    check(list.parse_error().find("real_gpu") != std::string::npos &&
              list.parse_error().find("simulated_route") != std::string::npos,
          "S6: the parse error names the closed set of provenances");
}

void s7_provenance_and_verdict_must_agree_both_ways() {
    // (a) real_gpu provenance, simulated row.
    const std::string path_a = write_temp(
        "s7a.list", real_provenance(kKeyAll, sim_entry(kKeyAll, false, "route_simulated", "70")));
    const auto list_a = ninfer::caps::FormatSupportList::parse(read_file(path_a));
    check(!list_a.valid(), "S7a: a simulated row in a real_gpu list is refused");
    check(list_a.parse_error().find("real_gpu") != std::string::npos,
          "S7a: the parse error names the contradicting provenance");

    // (b) simulated_route provenance, a row that does not say it was simulated.
    const std::string path_b = write_temp(
        "s7b.list", sim_provenance(kKeyAll, real_entry(kKeyAll, "reduction_l2")));
    const auto list_b = ninfer::caps::FormatSupportList::parse(read_file(path_b));
    check(!list_b.valid(), "S7b: a non-simulated row in a simulated_route list is refused");
    check(list_b.parse_error().find("simulated") != std::string::npos,
          "S7b: the parse error names the missing marker");
}

// ---------------------------------------------------------------------------
// S8  THE DRIVER. A probe run under the simulator may not write a real_gpu list: it must refuse,
//     or mark. Checked as text on the driver (the runtime arms need the GPU and a lock, which this
//     host-only test must not take) -- and the text checks are assertions about the MECHANISM, not
//     about a comment: an `exit 10` in the guard, a computed source, and an on-ramp in the refusal.
// ---------------------------------------------------------------------------
void s8_driver_refuses_or_marks_a_simulated_run() {
    const std::string text = read_file(source_dir() + "/tools/archkit/probe_formats.sh");
    if (text.empty()) {
        check(false, "S8: the driver could not be read at " + source_dir());
        return;
    }
    check(text.find("exit 10") != std::string::npos,
          "S8: the driver refuses a simulated run with its own exit code (rc=10)");
    check(text.find("NINFER_PROBE_ALLOW_SIM_ROUTE") != std::string::npos,
          "S8: the refusal names the explicit opt-in (mark-and-refuse, not removal)");
    check(text.find("SOURCE=\"simulated_route\"") != std::string::npos,
          "S8: the list's source is COMPUTED from the sim state, not declared");
    check(text.find("printf 'meta source=\"real_gpu\"") == std::string::npos,
          "S8: the hardcoded real_gpu provenance is gone");
    check(text.find("printf 'meta source=\"%s\"") != std::string::npos,
          "S8: and it is written from the computed value");
    check(text.find("' simulated=yes'") != std::string::npos,
          "S8: an allowed simulated run marks every row simulated=yes");
    check(text.find("leg=\"route_simulated\"") != std::string::npos,
          "S8: and records leg=route_simulated on every row of such a run");
    check(text.find("probe_sim=") != std::string::npos,
          "S8: the sim state is recorded in the RAW log, so --derive-only cannot lose it");
    // The refusal must be an ON-RAMP: it prints the command that would produce a real row.
    check(text.find("unset NINFER_SIM_ARCH NINFER_SIM_ARCH_ACK") != std::string::npos,
          "S8: the refusal says how to run a real probe (unset the simulator)");
    check(text.find("NINFER_PROBE_KEY='${PROBE_KEY}'") != std::string::npos,
          "S8: and carries the literal probe command");
    check(text.find("--accept-stale") == std::string::npos,
          "S8: the driver no longer claims a --accept-stale flag it does not implement");
}

// ---------------------------------------------------------------------------
// S9  ROUND-TRIP. If the marker did not survive render()->parse(), a list written by the driver
//     could lose it on one re-derivation and become admissible.
// ---------------------------------------------------------------------------
void s9_round_trip_keeps_the_simulated_marker() {
    const auto list = ninfer::caps::FormatSupportList::parse(
        sim_provenance(kKeyAll, sim_entry(kKeyAll, false, "route_simulated", "70")));
    check(list.valid(), "S9: the fixture parses");
    const auto again = ninfer::caps::FormatSupportList::parse(list.render());
    check(again.valid(), "S9: render() output parses");
    const auto* entry = again.find(kKeyAll, ninfer::artifact::NumericFormat::NVFP4, 1024, "A4");
    check(entry != nullptr, "S9: the entry survived the round-trip");
    if (entry != nullptr) {
        check(entry->is_simulated(), "S9: the simulated marker survived the round-trip");
        check(entry->sim_rung == "70", "S9: the rung survived the round-trip");
        check(!entry->supported, "S9: the row still does not claim support");
    }
    check(again.provenance().source == "simulated_route",
          "S9: the provenance's source survived the round-trip");
}

// ---------------------------------------------------------------------------
// S9b THE RE-RENDER PATH. A tool that rewrites a list is a real candidate launderer: render() then
//     parse() then gate. The marks must survive it (S9) AND the gate must still refuse (here), or
//     "an automated rewrite" would be a way in.
// ---------------------------------------------------------------------------
void s9b_rerendered_simulated_list_is_still_refused() {
    const auto list = ninfer::caps::FormatSupportList::parse(
        sim_provenance(kKeyAll, sim_entry(kKeyAll, false, "route_simulated", "70")));
    check(list.valid(), "S9b: the fixture parses");
    const std::string path = write_temp("s9b_rerendered.list", list.render());
    const auto again = ninfer::caps::FormatSupportList::load_file(path);
    check(again.valid(), "S9b: the re-rendered list parses");
    setenv("NINFER_FORMAT_PROBE_LIST", path.c_str(), 1);
    const auto verdict = ninfer::caps::format_probe_gate(
        declared_5090(), measured_all_true(), ninfer::artifact::NumericFormat::NVFP4, 1024,
        "test/artifact", "aaaaaaaaaaaaaaaa");
    check(verdict.enabled && !verdict.allowed,
          "S9b: a RE-RENDERED simulated list is still refused as support");
    if (verdict.enabled) {
        check(verdict.why.rfind("SIMULATED", 0) == 0,
              "S9b: and it is still refused BY NAME as simulated");
        check(verdict.why.find("NINFER_PROBE_KEY=") != std::string::npos,
              "S9b: with the on-ramp intact after the rewrite");
    }
    unsetenv("NINFER_FORMAT_PROBE_LIST");
}

// ---------------------------------------------------------------------------
// S10 THE REAL LIST. The rules above must not have made the box's own measured list unusable --
//     and that list must contain NO simulated row (a real run cannot have one).
// ---------------------------------------------------------------------------
void s10_the_real_list_still_admits_and_has_no_simulated_row() {
    const std::string path = source_dir() + "/tools/archkit/probe_results/format_support-120.list";
    const auto list = ninfer::caps::FormatSupportList::load_file(path);
    if (!list.valid()) {
        check(false, "S10: the real list no longer parses: " + list.parse_error() +
                         " missing=" + list.provenance().missing);
        return;
    }
    check(list.provenance().source == "real_gpu", "S10: the real list's provenance is real_gpu");
    int simulated = 0;
    for (const auto& entry : list.entries()) {
        if (entry.is_simulated()) { ++simulated; }
    }
    check(simulated == 0, "S10: the box's own measured list contains no simulated row");
    check(list.size() > 0, "S10: the real list has entries");

    setenv("NINFER_FORMAT_PROBE_LIST", path.c_str(), 1);
    const auto verdict = ninfer::caps::format_probe_gate(
        declared_5090(), measured_all_true(), ninfer::artifact::NumericFormat::NVFP4, 1024,
        "test/artifact", list.provenance().binary_sha16);
    check(verdict.enabled, "S10: the real list enables the gate");
    check(verdict.allowed, "S10: a measured NVFP4 17..open row is still admitted after these edits");
    unsetenv("NINFER_FORMAT_PROBE_LIST");
}

bool sim_lines_emitted_for_none(const ninfer::caps::RouteChoice& choice) {
    // The engine's own emit condition (src/ops/linear/qpn/qpn_arch_route.cpp): the line is printed
    // when the answer is simulated or refused, NOT when a route was selected. So a simulated
    // `none` -- which selects nothing -- still prints.
    return choice.simulated || choice.warns();
}

// ---------------------------------------------------------------------------
// S11 THE TRAP. A simulated `none` and a table `none` are the SAME route string; the ONLY
//     discriminator on the line is the marker. And a simulated `none` DOES produce a line -- which
//     is what makes "no line + banner present" mean "the run never reached a route decision"
//     rather than "the simulator said none".
// ---------------------------------------------------------------------------
void s11_none_verdict_is_marked_and_is_emitted() {
    using namespace ninfer::caps;
    const ProblemShape shape{1024, 5120, 5120};

    // Find a (rung, format) pair whose verdict is `none` on a rung below the physical card, so the
    // demonstration does not depend on which rungs a later edit leaves routeless.
    const int kRungs[] = {50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 89, 90, 100, 103};
    const ninfer::artifact::NumericFormat kFormats[] = {
        ninfer::artifact::NumericFormat::NVFP4, ninfer::artifact::NumericFormat::Q4G64_F16S,
        ninfer::artifact::NumericFormat::W8G32_F16S};
    int ruled_none = 0;
    int rung_found = 0;
    ninfer::artifact::NumericFormat format_found = ninfer::artifact::NumericFormat::NVFP4;
    for (const ninfer::artifact::NumericFormat format : kFormats) {
        for (const int rung : kRungs) {
            // Only a rung the LADDER contains can be simulated at all (arch_sim.h refuses an
            // unlisted rung rather than inventing a capability set for it), so the scan is
            // restricted to the ladder -- otherwise it could pick a rung the simulator itself
            // refuses and the demonstration would be of nothing.
            if (arch_rung(rung) == nullptr) { continue; }
            if (select_route(rung, format, shape, false).route == KernelRoute::None) {
                ruled_none = rung;
                format_found = format;
                rung_found = rung;
                break;
            }
        }
        if (ruled_none != 0) { break; }
    }
    std::fprintf(stderr, "[S11] `none` demonstration uses rung sm_%d\n", ruled_none);
    check(ruled_none != 0,
          "S11: some rung in the ladder has a `none` route for some format (the trap needs one)");
    if (ruled_none == 0) { return; }

    // (1) the TABLE's `none`, no simulator involved: the plain overload.
    const RouteChoice plain = select_route(rung_found, format_found, shape, false);
    const RouteLogLine plain_line = route_log_line(rung_found, format_found, shape, plain);
    // (2) the SIMULATOR's `none`: the same table, asked through an ArchView.
    const ArchView view = arch_view_for_device_impl(
        120, true, std::to_string(rung_found), std::string(kSimAckPhrase));
    check(view.simulated(), "S11: the ArchView for a lower rung is Active");
    const RouteChoice simulated_choice = select_route(view, format_found, shape, false);
    const RouteLogLine sim_line = route_log_line(rung_found, format_found, shape, simulated_choice);

    const std::string format_label(ninfer::artifact::format_name(format_found));
    std::printf("--- S11 measured lines (rung sm_%d, format %s) ---\n%s\n%s\n--- end ---\n",
                rung_found, format_label.c_str(), plain_line.text.c_str(), sim_line.text.c_str());

    check(plain.route == KernelRoute::None && simulated_choice.route == KernelRoute::None,
          "S11: both verdicts are `none`");
    // THE TRAP: the route VALUE is identical, so the value alone cannot tell a reader which fact
    // they are looking at.
    check(plain_line.text.find("-> none") != std::string::npos &&
              sim_line.text.find("-> none") != std::string::npos,
          "S11: both lines carry the identical `-> none` value");
    check(plain_line.text.rfind("[route] ", 0) == 0 &&
              plain_line.text.find("[route][SIMULATED]") == std::string::npos,
          "S11: the table's `none` line has NO simulated marker");
    check(sim_line.text.rfind("[route][SIMULATED] ", 0) == 0,
          "S11: the simulator's `none` line LEADS with the marker -- the only discriminator");
    // AND the simulator's `none` IS emitted, which is the fact that makes "no line" mean
    // "never reached a route decision". The emit condition in the engine caller is
    // `(r.simulated || r.refused)`, checked as text in S12.
    check(simulated_choice.simulated, "S11: the simulated choice is flagged simulated");
    check(sim_lines_emitted_for_none(simulated_choice), "S11: a simulated `none` yields a line");

    // The banner is the SECOND discriminator: it says the simulator was consulted at all.
    const std::string banner = sim_banner(view);
    check(!banner.empty(), "S11: the Active view prints a banner");
    check(banner.find("SIMULATED") != std::string::npos, "S11: the banner is marked SIMULATED");
    check(banner.find("sm_" + std::to_string(rung_found)) != std::string::npos,
          "S11: the banner names the simulated rung");
    const ArchView off = arch_view_for_device_impl(120, true, "", "");
    check(sim_banner(off).empty(), "S11: the Disabled view prints NO banner");
    // The three-observation table the report states, asserted as code facts:
    //   env set + no banner  -> the process never reached the arch view
    //   banner + no [route]  -> the simulator was consulted, no route decision was reached
    //   banner + [route]     -> a route decision exists (and its value may be `none`)
    check(sim_banner(off).empty(),
          "S11: Disabled -> no banner -> there is nothing to conclude from the run at all");
}

// ---------------------------------------------------------------------------
// S12 THE DISCRIMINATORS, PINNED AS TEXT. Each of these is a fact S11's conclusion depends on; a
//     silent removal of any one of them would turn "the run never reached a route decision" back
//     into "the simulator said none" in every future summary.
// ---------------------------------------------------------------------------
void s12_discriminators_are_still_in_the_tree() {
    const std::string route_cpp =
        read_file(source_dir() + "/src/ops/linear/qpn/qpn_arch_route.cpp");
    const std::string route_h = read_file(source_dir() + "/src/core/kernel_route.h");
    const std::string sim_h = read_file(source_dir() + "/src/core/arch_sim.h");
    const std::string probe_h = read_file(source_dir() + "/src/core/format_probe.h");
    if (route_cpp.empty() || route_h.empty() || sim_h.empty() || probe_h.empty()) {
        check(false, "S12: a source file could not be read from " + source_dir());
        return;
    }
    // (a) the engine emits the line for a simulated or refused answer, NOT for a selected route.
    check(route_cpp.find("if (emit_log_line && (r.simulated || r.refused))") != std::string::npos,
          "S12: the emit condition is the SIM marker, not use_qpn (so `none` is still printed)");
    // (b) the marker is on the route line itself, at the FRONT.
    check(route_h.find("choice.simulated ? std::string(\"[route][SIMULATED] \")") !=
              std::string::npos,
          "S12: route_log_line puts the SIMULATED marker at the front of the line");
    // (c) the banner is printed once per process by the arch view -- the "was the simulator
    //     consulted at all" discriminator.
    check(sim_h.find("static bool announced = false;") != std::string::npos &&
              sim_h.find("sim_banner(view).c_str()") != std::string::npos,
          "S12: arch_view_for_device prints the banner once (the consulted-at-all discriminator)");
    // (d) THE PERFORMANCE PROHIBITION IS UNTOUCHED (task requirement 4: do not weaken it).
    check(sim_h.find("No performance claim of any kind may be sourced from a simulated run.") !=
              std::string::npos,
          "S12: arch_sim.h still forbids sourcing a performance claim from a simulated run");
    // (e) and the SUPPORT prohibition is a DIFFERENT rule, stated as such where a reader looks.
    check(probe_h.find("SIMULATED VERDICT -- REFUSED AS SUPPORT") != std::string::npos,
          "S12: format_probe.h carries the named support refusal");
    check(probe_h.find("FUNCTIONAL support verdict") != std::string::npos ||
              probe_h.find("FUNCTIONAL/support claim") != std::string::npos,
          "S12: and states that the support prohibition is functional, unlike the speed one");
    // (f) the parse-time rule that closes the laundering hole is still there.
    check(probe_h.find("claims supported=yes while marked simulated=yes") != std::string::npos,
          "S12: the parse-time rule against supported=yes + simulated=yes is still present");
    // (g) the two spellings are normalised through ONE predicate.
    check(probe_h.find("bool is_simulated() const noexcept") != std::string::npos,
          "S12: one predicate covers both spellings of 'this came from the simulator'");
    // (h) THE ORDER OF THE TWO ENGINE CALL SITES, which is what lets a reader conclude "the
    //     simulator was consulted and no route decision was reached" from banner-without-line.
    //     This is a STATIC fact about the tree (the call sites and which TU they live in); it is
    //     not a runtime measurement, and the report says so.
    const std::string registry = read_file(source_dir() + "/src/targets/registry.cpp");
    const std::string dispatch = read_file(source_dir() + "/src/ops/linear/nvfp4/nvfp4_dispatch.cpp");
    check(registry.find("caps::arch_view_for_device(device.sm())") != std::string::npos,
          "S12: the registry capability gate calls the arch view (so the banner precedes any op)");
    check(dispatch.find("qpn::select_qpn_arch_route(qpn::current_device_sm()") != std::string::npos,
          "S12: the route decision is asked from INSIDE the op, i.e. only after the gate admitted");
}

// ---------------------------------------------------------------------------
// EXTERNAL MODE. With a path argument, load that list and report whether the gate REFUSES it as
// support; exit 0 iff refused. This is how a list PRODUCED BY THE DRIVER (under the simulator) is
// checked end to end without a rebuild -- the driver's own output must be refused by the gate, or
// the mark-and-refuse contract is broken at the seam between the two files.
// ---------------------------------------------------------------------------
int external_list_refusal_check(const std::string& path) {
    const auto list = ninfer::caps::FormatSupportList::load_file(path);
    std::printf("external check: %s\n", path.c_str());
    std::printf("  parses=%d valid=%d entries=%zu source=%s\n", static_cast<int>(list.parsed()),
                static_cast<int>(list.valid()), list.size(), list.provenance().source.c_str());
    if (!list.valid()) {
        std::printf("  parse_error=%s\n", list.parse_error().c_str());
        std::printf("  VERDICT: REFUSED (not usable as a list at all)\n");
        return 0;
    }
    int simulated = 0;
    for (const auto& entry : list.entries()) {
        if (entry.is_simulated()) { ++simulated; }
    }
    std::printf("  simulated_rows=%d/%zu\n", simulated, list.size());
    setenv("NINFER_FORMAT_PROBE_LIST", path.c_str(), 1);
    const auto verdict = ninfer::caps::format_probe_gate(
        declared_5090(), measured_all_true(), ninfer::artifact::NumericFormat::NVFP4, 1024,
        "external/artifact", list.provenance().binary_sha16);
    std::printf("  enabled=%d allowed=%d\n  why=%s\n", static_cast<int>(verdict.enabled),
                static_cast<int>(verdict.allowed), verdict.why.c_str());
    std::printf("  VERDICT: %s\n",
                verdict.allowed ? "ADMITTED -- A FINDING, THIS MUST NOT HAPPEN" : "REFUSED AS SUPPORT");
    return verdict.allowed ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1) { return external_list_refusal_check(std::string(argv[1])); }
    std::printf("test_sim_no_support: a simulated verdict may not enter the support list\n");
    // PROGRESS GOES TO STDERR, UNBUFFERED. stdout is block-buffered when redirected, so a test
    // that dies on a signal would otherwise report nothing and leave the crash unlocalisable --
    // which is the same "the summary does not say which fact you are looking at" defect this file
    // exists to refuse one level up.
    setvbuf(stderr, nullptr, _IONBF, 0);
    const auto run = [](const char* name, void (*fn)()) {
        std::fprintf(stderr, "[sim-no-support] %s ...\n", name);
        fn();
    };
    run("S1", s1_supported_and_simulated_is_refused_at_parse);
    run("S2", s2_gate_refuses_simulated_by_name_with_the_on_ramp);
    run("S3", s3_missing_rung_is_reported_not_invented);
    run("S4", s4_t7_shape_still_names_the_leg);
    run("S5", s5_sim_rung_must_be_a_ladder_rung);
    run("S6", s6_unclassifiable_source_is_refused);
    run("S7", s7_provenance_and_verdict_must_agree_both_ways);
    run("S8", s8_driver_refuses_or_marks_a_simulated_run);
    run("S9", s9_round_trip_keeps_the_simulated_marker);
    run("S9b", s9b_rerendered_simulated_list_is_still_refused);
    run("S10", s10_the_real_list_still_admits_and_has_no_simulated_row);
    run("S11", s11_none_verdict_is_marked_and_is_emitted);
    run("S12", s12_discriminators_are_still_in_the_tree);
    std::printf("%s test_sim_no_support: %d checks, %d failure(s)\n",
                g_failures == 0 ? "OK " : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
