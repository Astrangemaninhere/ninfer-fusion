// ---------------------------------------------------------------------------
// test_format_probe.cpp -- the contract of src/core/format_probe.h
//
// HOST-ONLY, and RUNNABLE WITH PLAIN g++ BECAUSE THE SUBJECT IS HOST-ONLY:
//
//   g++ -std=c++20 -O1 -I <repo>/src -I <repo>/include -I <repo>/tests
//       -I <repo>/third_party -DNINFER_HAVE_QPN=1 -DNINFER_SOURCE_DIR='"<repo>"'
//       <repo>/tests/test_format_probe.cpp
//       <repo>/build/src/libninfer_artifact.a <repo>/build/src/libninfer_core.a
//       -o /tmp/test_format_probe   &&   /tmp/test_format_probe
//
// (the two archives are on the line for ONE symbol, ninfer::artifact::format_name, which
// src/core/format_probe.h calls rather than keeping a second copy of the format-name table.
// No CUDA, no device, no GPU is needed to run it.)
//
// It links no engine library: the probe list, its parser and the gate are header-only and
// take their inputs as parameters or as a list FILE, which is why every branch below is
// exercisable on a machine with no card in it. (That is also the reason this file can be
// run before it is registered in tests/CMakeLists.txt -- see the note at the bottom.)
//
// WHAT IT PINS, and each of these is a property a later edit could quietly break:
//   T1  the criterion constants still match the file they are QUOTED from
//   T2  the band edges still match the ROUTER's own edges
//   T3  the gate is OFF, with no behaviour change, when the env names no list
//   T4  a measured entry admits its own (arch, format, band)
//   T5  an ABSENT entry REFUSES, and the refusal names arch + format + band + what to probe
//   T6  supported=yes with leg=unprobed is refused AT PARSE (a support verdict must come
//       from a measurement -- this is the "absence of error is not support" rule)
//   T7  a RouteSimulated entry is refused as arithmetic support (the sim is a legitimate
//       ROUTE instrument and must never be laundered into a numbers claim)
//   T8  a stale binary sha16 refuses by default and is admitted only on an explicit opt-in
//   T9  an unreadable / provenance-incomplete list REFUSES rather than default-allowing
//   T10 m == 0 has no band and therefore no entry: it refuses instead of picking the first
//   T11 render() -> parse() round-trips
//   T12 THE STATUS-QUO FINDING: in the list this box actually produced, the NVFP4 W4A4
//       M9..16 row (the 让步带 the record names) has cases=0 and refuses -- pinned so that
//       a later "fix" cannot silently turn an unmeasured band into an admitted one.
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

template <typename T>
void check_eq(const T& got, const T& want, const std::string& what) {
    ++g_checks;
    if (!(got == want)) {
        ++g_failures;
        std::cerr << "FAIL: " << what << "\n  got  = " << got << "\n  want = " << want << '\n';
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
    const std::string path = std::string("/tmp/ninfer_format_probe_test/") + name;
    std::string dir = "/tmp/ninfer_format_probe_test";
    std::string mk = "mkdir -p " + dir;
    if (std::system(mk.c_str()) != 0) { std::cerr << "cannot create " << dir << '\n'; }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
    out.close();
    return path;
}

// The fixture key. It is a MEASURED capability vector, and its shape is asserted in T15.
constexpr const char* kKeyA = "v1;ki=1;bf16=1;fp16=1;i8=1;f8f6f4=1;mxf4=1;smr=1";
constexpr const char* kKeyB = "v1;ki=1;bf16=1;fp16=1;i8=1;f8f6f4=0;mxf4=0;smr=0";

ninfer::caps::MeasuredCapabilities measured_a() {
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
ninfer::caps::MeasuredCapabilities measured_b() {
    ninfer::caps::MeasuredCapabilities m = measured_a();
    m.fp8_kind_f8f6f4 = false;
    m.nvfp4_mma_block_scale = false;
    m.setmaxnreg = false;
    return m;
}
ninfer::caps::DeclaredIdentity declared_v100() {
    ninfer::caps::DeclaredIdentity d;
    d.name = "NVIDIA Tesla V100-SXM2-16GB";
    d.cc_major = 7;
    d.cc_minor = 0;
    d.build_arch = "70";
    return d;
}
ninfer::caps::DeclaredIdentity declared_a() {
    ninfer::caps::DeclaredIdentity d;
    d.name = "NVIDIA GeForce RTX 5090 D";
    d.cc_major = 12;
    d.cc_minor = 0;
    d.build_arch = "120a";
    return d;
}

// The provenance a valid list must carry, so each scenario below can vary ONE thing.
std::string good_header(const std::string& binary_sha16) {
    return "# test list\n"
           "meta probe_tool=\"tools/archkit/probe_formats.sh\"\n"
           "meta criterion_id=\"linear-reduction-v1\"\n"
           "meta criterion_src=\"tests/ops/linear/linear_test_common.cpp:38-48\"\n"
           "meta reference=\"cpu_linear_gemm_fp64 (linear_test_common.cpp:246)\"\n"
           "meta stats_env=\"NINFER_OP_REPORT_STATS=1\"\n"
           "meta key=\"" + std::string(kKeyA) + "\"\n"
           "meta declared_name=\"NVIDIA GeForce RTX 5090 D\"\n"
           "meta declared_cc=\"12.0\"\n"
           "meta declared_build=\"120a\"\n"
           "meta date=\"2026-09-18\"\n"
           "meta host_gpu=\"NVIDIA GeForce RTX 5090 D, 12.0\"\n"
           "meta binary_path=\"build/tests/ninfer_linear_nvfp4_a4_test\"\n"
           "meta binary_sha16=\"" +
           binary_sha16 +
           "\"\n"
           "meta binary_mtime=\"2026-09-18 14:00:41\"\n"
           "meta library_path=\"build/src/libninfer_ops.a\"\n"
           "meta library_sha16=\"b8036d7b4eba9beb\"\n"
           "meta library_mtime=\"2026-09-18 13:59:49\"\n"
           "meta source=\"real_gpu\"\n";
}

std::string entry_line(const char* key, const char* format, const char* path, const char* band,
                       bool supported, const char* leg, int cases, double ratio,
                       const char* case_label) {
    std::ostringstream out;
    const char* decl_name = (std::string(key) == kKeyB) ? "NVIDIA Tesla V100-SXM2-16GB"
                                                       : "NVIDIA GeForce RTX 5090 D";
    const char* decl_cc = (std::string(key) == kKeyB) ? "7.0" : "12.0";
    out << "entry key=\"" << key << "\" declared_name=\"" << decl_name << "\" declared_cc=\""
        << decl_cc << "\" format=" << format << " path=" << path << " band=" << band
        << " supported=" << (supported ? "yes" : "no") << " leg=" << leg << " cases=" << cases
        << " ratio=" << ratio << " case=\"" << case_label << "\"\n";
    return out.str();
}

// ---------------------------------------------------------------------------
// T1  The criterion constants still match the file they are quoted from.
//
// THIS IS THE MECHANISM, NOT A FORMALITY. src/core/format_probe.h quotes
// tests/ops/linear/linear_test_common.cpp:38-48 rather than owning the numbers, and a
// quoted number that nobody re-checks is a number that drifts. So the file is read AS
// TEXT and the three constants plus the three allowance names are required to be present
// with the values this header uses. If the suite ever widens a tolerance, this fails
// before the widened tolerance can be inherited silently by a support verdict.
// ---------------------------------------------------------------------------
void test_criterion_matches_its_source() {
    const std::string path = source_dir() + "/tests/ops/linear/linear_test_common.cpp";
    const std::string text = read_file(path);
    if (text.empty()) {
        std::cout << "SKIP T1: cannot read " << path << " (criterion source)\n";
        return;
    }
    check(text.find("kBf16UnitRoundoff        = 1.0 / 256.0") != std::string::npos ||
              text.find("kBf16UnitRoundoff = 1.0 / 256.0") != std::string::npos,
          "T1: the source still declares kBf16UnitRoundoff = 1.0/256.0");
    check(text.find("kA8QuantizationAllowance = 0.04") != std::string::npos,
          "T1: the source still declares kA8QuantizationAllowance = 0.04");
    check(text.find("kA4QuantizationAllowance = 0.16") != std::string::npos,
          "T1: the source still declares kA4QuantizationAllowance = 0.16");
    check(text.find("kBf16UnitRoundoff, kBf16UnitRoundoff, 2.0 * kBf16UnitRoundoff") !=
              std::string::npos,
          "T1: the A16 ReductionCriterion is still {1/256, 1/256, 2*(1/256)}");
    check(text.find("kA8QuantizationAllowance, kBf16UnitRoundoff, 1.5 * kA8QuantizationAllowance") !=
              std::string::npos,
          "T1: the A8 ReductionCriterion is still {0.04, 1/256, 1.5*0.04}");
    check(text.find("kA4QuantizationAllowance, kBf16UnitRoundoff, kA4QuantizationAllowance") !=
              std::string::npos,
          "T1: the A4 ReductionCriterion is still {0.16, 1/256, 0.16}");

    // And the header's own numbers equal the source's, so the two cannot disagree.
    check_eq(ninfer::caps::kProbeBf16UnitRoundoff, 1.0 / 256.0, "T1: header bf16 roundoff");
    check_eq(ninfer::caps::kProbeCriterionA8.relative_l2, 0.04, "T1: header A8 rel_l2");
    check_eq(ninfer::caps::kProbeCriterionA4.relative_l2, 0.16, "T1: header A4 rel_l2");
    check_eq(ninfer::caps::kProbeCriterionA16.gross_relative_to_max_reference, 2.0 / 256.0,
             "T1: header A16 gross relative");

    // The oracle is named in the header, so the oracle's existence is checked too.
    check(text.find("cpu_linear_gemm_fp64") != std::string::npos,
          "T1: the FP64 oracle is still in the file the header names");
}

// ---------------------------------------------------------------------------
// T2  The band edges are the ROUTER's edges, not a second set invented here.
// ---------------------------------------------------------------------------
void test_bands_match_the_router() {
    const std::string path = source_dir() + "/src/core/kernel_route.h";
    const std::string text = read_file(path);
    if (text.empty()) {
        std::cout << "SKIP T2: cannot read " << path << "\n";
        return;
    }
    // The four bands are justified in kernel_route.h by these tokens. If the router's
    // split moves (a new band, a moved edge), this fails and the band table has to be
    // re-derived from the router rather than carried over.
    check(text.find("m > 16U") != std::string::npos,
          "T2: kernel_route.h still splits at M > 16");
    check(text.find("m <= 3U") != std::string::npos,
          "T2: kernel_route.h still splits at M <= 3");
    check(text.find("m <= 8U") != std::string::npos,
          "T2: kernel_route.h still splits at M <= 8");

    check_eq(ninfer::caps::kProbeBandCount, std::size_t{4}, "T2: four bands");
    check_eq(ninfer::caps::probe_band_name(1, 3), std::string("1..3"), "T2: band 1..3 name");
    check_eq(ninfer::caps::probe_band_name(17, ninfer::caps::kProbeBandOpenEnd),
             std::string("17..open"), "T2: open band name");
    check(ninfer::caps::probe_band_of(0) == nullptr, "T2: m=0 has no band");
    check(ninfer::caps::probe_band_of(4) != nullptr, "T2: m=4 has a band");
    check_eq(ninfer::caps::probe_band_of(16)->m_hi, 16u, "T2: m=16 is in the 9..16 band");
    check_eq(ninfer::caps::probe_band_of(17)->m_lo, 17u, "T2: m=17 opens the last band");
    check_eq(ninfer::caps::probe_band_of(4096)->m_hi, ninfer::caps::kProbeBandOpenEnd,
             "T2: the last band is open-ended");
}

// ---------------------------------------------------------------------------
// T3..T11  the gate
// ---------------------------------------------------------------------------
void test_gate_disabled_by_default() {
    unsetenv("NINFER_FORMAT_PROBE_LIST");
    unsetenv("NINFER_FORMAT_PROBE_ACCEPT_STALE");
    const auto verdict = ninfer::caps::format_probe_gate(
        declared_a(), measured_a(), ninfer::artifact::NumericFormat::NVFP4, 1024, "test/artifact", "deadbeefdeadbeef");
    check(!verdict.enabled, "T3: the gate is disabled when the env names no list");
    check(!verdict.allowed, "T3: disabled means nothing is admitted BY this gate");
    check(!verdict.why.empty(), "T3: the disabled answer still explains itself");
    check(!verdict.refused_for_absent,
          "T3: disabled is not a refusal -- the shipping path is untouched");
}

void test_absent_refuses_and_names_what_to_probe() {
    const std::string list =
        good_header("aaaaaaaaaaaaaaaa") +
        entry_line(kKeyA, "NVFP4", "A4", "1..3", false, "unprobed", 0, 0.0, "") +
        entry_line(kKeyA, "NVFP4", "A4", "17..open", true, "reduction_l2", 3, 0.01,
                   "NVFP4_A4 [14336,5120] T=17");
    const std::string path = write_temp("absent.list", list);
    setenv("NINFER_FORMAT_PROBE_LIST", path.c_str(), 1);
    unsetenv("NINFER_FORMAT_PROBE_ACCEPT_STALE");

    // Admitted: a measured, fresh entry for its own band.
    const auto ok = ninfer::caps::format_probe_gate(
        declared_a(), measured_a(), ninfer::artifact::NumericFormat::NVFP4, 32, "test/artifact", "aaaaaaaaaaaaaaaa");
    check(ok.enabled && ok.allowed, "T4: a measured entry for its own band is admitted");
    check(ok.why.find("MEASURED SUPPORTED") != std::string::npos,
          "T4: the admission says it was measured");
    check(ok.why.find("linear-reduction-v1") != std::string::npos,
          "T4: the admission names the criterion");
    check(ok.why.find("fresh") != std::string::npos, "T4: the admission names the freshness");

    // REFUSED: no entry for the 9..16 band at all.
    const auto absent = ninfer::caps::format_probe_gate(
        declared_a(), measured_a(), ninfer::artifact::NumericFormat::NVFP4, 12, "test/artifact", "aaaaaaaaaaaaaaaa");
    check(absent.enabled && !absent.allowed, "T5: an unprobed band refuses");
    check(absent.refused_for_absent, "T5: the refusal says the entry is ABSENT");
    check(absent.why.find(std::string(kKeyA)) != std::string::npos,
          "T5: refusal names the MEASURED key it looked up");
    check(absent.why.find("NVFP4") != std::string::npos, "T5: refusal names the format");
    check(absent.why.find("9..16") != std::string::npos, "T5: refusal names the band");
    check(absent.why.find("probe_formats.sh") != std::string::npos,
          "T5: refusal names the tool that would change the answer");
    check(absent.why.find("NINFER_OP_REPORT_STATS=1") != std::string::npos,
          "T5: refusal names the stats env the arm needs");
    check(absent.why.find("M 9..16 has NO case") != std::string::npos,
          "T5: refusal says the band may have no case, not only that it was not run");

    // REFUSED: a DIFFERENT format in a probed band -- absence is per (format, band).
    const auto other_format = ninfer::caps::format_probe_gate(
        declared_a(), measured_a(), ninfer::artifact::NumericFormat::Q4G64_F16S, 32, "test/artifact",
        "aaaaaaaaaaaaaaaa");
    check(other_format.enabled && !other_format.allowed,
          "T5: a probed band for one format does not cover another format");
    check(other_format.why.find("Q4G64_F16S") != std::string::npos,
          "T5: that refusal names the format that was not probed");

    // REFUSED: a DIFFERENT measured capability vector in a probed band -- absence is per
    // (measured key, format, band), which is what "no name in the decision" means here.
    const auto other_arch = ninfer::caps::format_probe_gate(
        declared_v100(), measured_b(), ninfer::artifact::NumericFormat::NVFP4, 32, "test/artifact",
        "aaaaaaaaaaaaaaaa");
    check(other_arch.enabled && !other_arch.allowed, "T5: absence is per key too");
    check(other_arch.why.find(std::string(kKeyB)) != std::string::npos,
          "T5: that refusal names the key that was not probed");

    // REFUSED: m = 0, which has no band, so there is no entry that could admit it.
    const auto zero = ninfer::caps::format_probe_gate(
        declared_a(), measured_a(), ninfer::artifact::NumericFormat::NVFP4, 0, "test/artifact", "aaaaaaaaaaaaaaaa");
    check(zero.enabled && !zero.allowed, "T10: m=0 refuses");
    check(zero.refused_for_absent, "T10: m=0 is an absent entry, not a wrong one");

    unsetenv("NINFER_FORMAT_PROBE_LIST");
}

void test_a_support_claim_needs_a_measurement() {
    // T6: supported=yes with leg=unprobed is the laundered claim, refused at parse time.
    const std::string laundered =
        good_header("aaaaaaaaaaaaaaaa") +
        entry_line(kKeyA, "NVFP4", "A4", "1..3", true, "unprobed", 0, 0.0, "");
    const std::string path = write_temp("laundered.list", laundered);
    const auto list = ninfer::caps::FormatSupportList::load_file(path);
    check(!list.parsed() || !list.valid(), "T6: a supported=yes/leg=unprobed list is not valid");
    check(list.parse_error().find("unprobed") != std::string::npos,
          "T6: and the parse error explains which rule was broken");
    setenv("NINFER_FORMAT_PROBE_LIST", path.c_str(), 1);
    const auto verdict = ninfer::caps::format_probe_gate(
        declared_a(), measured_a(), ninfer::artifact::NumericFormat::NVFP4, 2, "test/artifact", "aaaaaaaaaaaaaaaa");
    check(verdict.enabled && !verdict.allowed,
          "T6: the gate refuses when the list's only support claim is unmeasured");
    unsetenv("NINFER_FORMAT_PROBE_LIST");
}

void test_route_verdict_is_not_arithmetic_support() {
    // T7: a verdict from the arch simulator answers "does this rung take this path".
    const std::string list =
        std::string("# simulated-only list\n") +
        "meta probe_tool=\"tools/archkit/probe_formats.sh\"\n"
        "meta criterion_id=\"route-only\"\n"
        "meta criterion_src=\"src/core/arch_sim.h:59-70\"\n"
        "meta reference=\"none: no arithmetic was measured\"\n"
        "meta key=\"" + std::string(kKeyB) + "\"\n"
        "meta declared_name=\"NVIDIA Tesla V100-SXM2-16GB\"\n"
        "meta declared_cc=\"7.0\"\n"
        "meta date=\"2026-09-18\"\n"
        "meta binary_sha16=\"aaaaaaaaaaaaaaaa\"\n"
        "meta source=\"simulated_route\"\n" +
        entry_line(kKeyB, "NVFP4", "A4", "17..open", true, "route_simulated", 1, 0.0,
                   "simulated sm_70 route selection");
    const std::string path = write_temp("simulated.list", list);
    setenv("NINFER_FORMAT_PROBE_LIST", path.c_str(), 1);
    const auto verdict = ninfer::caps::format_probe_gate(
        declared_v100(), measured_b(), ninfer::artifact::NumericFormat::NVFP4, 1024, "test/artifact", "aaaaaaaaaaaaaaaa");
    check(verdict.enabled && !verdict.allowed, "T7: a simulated ROUTE verdict is not support");
    check(verdict.why.find("route_simulated") != std::string::npos,
          "T7: the refusal names the leg that produced it");
    check(verdict.why.find("arithmetic") != std::string::npos,
          "T7: the refusal says the missing thing is arithmetic");
    unsetenv("NINFER_FORMAT_PROBE_LIST");
}

void test_staleness_is_refused_unless_accepted() {
    const std::string list =
        good_header("bbbbbbbbbbbbbbbb") +
        entry_line(kKeyA, "NVFP4", "A4", "17..open", true, "reduction_l2", 3, 0.01, "NVFP4_A4 T=17");
    const std::string path = write_temp("stale.list", list);
    setenv("NINFER_FORMAT_PROBE_LIST", path.c_str(), 1);
    unsetenv("NINFER_FORMAT_PROBE_ACCEPT_STALE");

    const auto stale = ninfer::caps::format_probe_gate(
        declared_a(), measured_a(), ninfer::artifact::NumericFormat::NVFP4, 1024, "test/artifact", "cccccccccccccccc");
    check(stale.enabled && !stale.allowed, "T8: a stale binary sha16 refuses by default");
    check(stale.why.find("bbbbbbbbbbbbbbbb") != std::string::npos,
          "T8: the refusal quotes the sha16 the list recorded");
    check(stale.why.find("cccccccccccccccc") != std::string::npos,
          "T8: the refusal quotes the caller's sha16");
    check(stale.why.find("NINFER_FORMAT_PROBE_ACCEPT_STALE") != std::string::npos,
          "T8: the refusal names the opt-in that would accept it");

    setenv("NINFER_FORMAT_PROBE_ACCEPT_STALE", "1", 1);
    const auto accepted = ninfer::caps::format_probe_gate(
        declared_a(), measured_a(), ninfer::artifact::NumericFormat::NVFP4, 1024, "test/artifact", "cccccccccccccccc");
    check(accepted.enabled && accepted.allowed,
          "T8: the explicit opt-in admits a verdict about a different binary");
    check(accepted.why.find("STALE") != std::string::npos,
          "T8: and the admission still says the verdict is stale");
    unsetenv("NINFER_FORMAT_PROBE_ACCEPT_STALE");
    unsetenv("NINFER_FORMAT_PROBE_LIST");
}

void test_unreadable_and_incomplete_lists_refuse() {
    // T9a: a path that does not exist.
    setenv("NINFER_FORMAT_PROBE_LIST", "/tmp/ninfer_format_probe_test/does-not-exist.list", 1);
    const auto missing = ninfer::caps::format_probe_gate(
        declared_a(), measured_a(), ninfer::artifact::NumericFormat::NVFP4, 1024, "test/artifact", "aaaaaaaaaaaaaaaa");
    check(missing.enabled && !missing.allowed,
          "T9: a list that cannot be opened REFUSES rather than default-allowing");
    check(missing.why.find("could not be used") != std::string::npos,
          "T9: the refusal says the list could not be used");

    // T9b: provenance-incomplete (no criterion, no sha16, no date).
    const std::string thin =
        "meta probe_tool=\"tools/archkit/probe_formats.sh\"\n"
        "meta key=\"" + std::string(kKeyA) + "\"\n" +
        entry_line(kKeyA, "NVFP4", "A4", "17..open", true, "reduction_l2", 3, 0.01, "NVFP4_A4 T=17");
    const std::string thin_path = write_temp("thin.list", thin);
    setenv("NINFER_FORMAT_PROBE_LIST", thin_path.c_str(), 1);
    const auto incomplete = ninfer::caps::format_probe_gate(
        declared_a(), measured_a(), ninfer::artifact::NumericFormat::NVFP4, 1024, "test/artifact", "aaaaaaaaaaaaaaaa");
    check(incomplete.enabled && !incomplete.allowed,
          "T9: an incomplete provenance REFUSES");
    check(incomplete.why.find("provenance incomplete") != std::string::npos,
          "T9: the refusal says the provenance is incomplete");
    check(incomplete.why.find("binary_sha16") != std::string::npos,
          "T9: and names a field that is missing");
    unsetenv("NINFER_FORMAT_PROBE_LIST");
}

void test_render_round_trip() {
    const std::string list =
        good_header("aaaaaaaaaaaaaaaa") +
        entry_line(kKeyA, "NVFP4", "A4", "1..3", false, "reduction_l2", 2, 0.9, "NVFP4_A4 T=1") +
        entry_line(kKeyA, "NVFP4", "A4", "17..open", true, "reduction_l2", 3, 0.01, "NVFP4_A4 T=17") +
        entry_line(kKeyA, "BF16", "A16", "1..3", true, "reduction_l2", 4, 0.001, "BF16_A16 T=1");
    const auto first = ninfer::caps::FormatSupportList::parse(list);
    check(first.valid(), "T11: the fixture list parses as valid");
    const std::string rendered = first.render();
    const auto second = ninfer::caps::FormatSupportList::parse(rendered);
    check(second.valid(), "T11: the rendered list parses back as valid");
    check_eq(second.size(), first.size(), "T11: round trip keeps the entry count");
    check_eq(second.provenance().criterion_id, first.provenance().criterion_id,
             "T11: round trip keeps the criterion id");
    check_eq(second.provenance().binary_sha16, first.provenance().binary_sha16,
             "T11: round trip keeps the binary sha16 (the staleness key)");
    const auto entries_1024 = second.find_all(kKeyA, ninfer::artifact::NumericFormat::NVFP4, 1024);
    const auto* entry = entries_1024.empty() ? nullptr : entries_1024.front();
    check(entry != nullptr, "T11: the round-tripped list still finds its entry");
    if (entry != nullptr) {
        check(entry->supported, "T11: round trip keeps supported");
        check_eq(entry->cases, 3, "T11: round trip keeps the case count");
        check(entry->case_example == "NVFP4_A4 T=17",
              "T11: round trip keeps a quoted value containing '=' and spaces");
    }
    // An entry with supported=no is recorded, not omitted: a negative result is a result.
    const auto entries_2 = second.find_all(kKeyA, ninfer::artifact::NumericFormat::NVFP4, 2);
    const auto* negative = entries_2.empty() ? nullptr : entries_2.front();
    check(negative != nullptr, "T11: the negative entry survived the round trip");
    if (negative != nullptr) { check(!negative->supported, "T11: and it is still negative"); }
}

// ---------------------------------------------------------------------------
// T12  THE STATUS-QUO FINDING, pinned.
//
// The list this box actually produced has an NVFP4 W4A4 row for the 9..16 band with
// cases=0 and leg=unprobed, because the suite's NVFP4 A4 arm has no case in that band --
// and 9..16 is precisely the band tools/archkit/_GPU_MATRIX.md "增补 2" calls the NVFP4
// 让步带. The gate must REFUSE it. If a later edit makes that band admitted, this fails.
// ---------------------------------------------------------------------------
void test_status_quo_nvfp4_a4_concession_band_refuses() {
    const std::string path = std::string(source_dir()) +
                             "/tools/archkit/probe_results/format_support-120.list";
    const auto list = ninfer::caps::FormatSupportList::load_file(path);
    if (!list.parsed()) {
        std::cout << "SKIP T12: no probe list at " << path
                  << " -- run tools/archkit/probe_formats.sh first\n";
        return;
    }
    check(list.valid(), "T12: the produced list is valid (provenance complete)");
    if (!list.valid()) {
        std::cerr << "  parse_error=" << list.parse_error()
                  << " missing=" << list.provenance().missing << '\n';
        return;
    }
    check_eq(list.provenance().key, std::string(kKeyA),
             "T12: the list records the MEASURED capability key it was taken under");
    check(list.provenance().source == "real_gpu", "T12: the list records its source");

    const auto all = list.find_all(kKeyA, ninfer::artifact::NumericFormat::NVFP4, 12);
    check(!all.empty(), "T12: the NVFP4 9..16 rows are present (a band with no case must "
                        "be VISIBLE, one row per measured activation path)");
    const auto* concession = list.find(kKeyA, ninfer::artifact::NumericFormat::NVFP4, 12, "A4");
    check(concession != nullptr,
          "T12: the NVFP4 W4A4 9..16 row exists -- this is the 让步带 the record names");
    if (concession != nullptr) {
        check_eq(concession->cases, 0, "T12: the A4 9..16 band has zero measured cases");
        check(concession->leg == ninfer::caps::ProbeLeg::Unprobed,
              "T12: and is recorded as unprobed, not as a pass");
        check(!concession->supported, "T12: and is therefore not supported");
    }
    // T14: the fail-closed rule the A16/A4 collision exposed. The A16 path of the SAME
    // (format, band) is fully measured and admitted; the A4 path is unprobed. The gate must
    // refuse the COMBINATION, because which path a run takes is a dispatch decision this
    // header cannot make, and answering the easier half is how an unmeasured rung gets
    // laundered into support.
    const auto* a16 = list.find(kKeyA, ninfer::artifact::NumericFormat::NVFP4, 12, "A16");
    check(a16 != nullptr && a16->supported,
          "T14: the NVFP4 A16 9..16 path IS measured and admitted on its own");
    check(concession != nullptr && !concession->supported,
          "T14: while the NVFP4 A4 9..16 path is not");
    setenv("NINFER_FORMAT_PROBE_LIST", path.c_str(), 1);
    const auto verdict = ninfer::caps::format_probe_gate(
        declared_a(), measured_a(), ninfer::artifact::NumericFormat::NVFP4, 12, "test/artifact",
        list.provenance().binary_sha16.c_str());
    check(verdict.enabled, "T12: the gate is enabled by the produced list");
    check(!verdict.allowed, "T12/T14: the NVFP4 9..16 combination is REFUSED by the gate");
    check(verdict.why.find("path=A4") != std::string::npos,
          "T14: and the refusal names the activation path that was not admitted");
    unsetenv("NINFER_FORMAT_PROBE_LIST");
}

// ---------------------------------------------------------------------------
// T13  THE PROBE'S OWN NEGATIVE FINDINGS, pinned.
//
// The first real run of tools/archkit/probe_formats.sh on this box (2026-09-18, arch 120a)
// found two defects that the declared row table in tools/archkit/_GPU_MATRIX.md does not
// record -- its 86/80 row says the groupwise-int profile is already there:
//
//   * ninfer_linear_q4_a16_test exits rc=1 with cudaErrorIllegalAddress at
//     Q4_A16 [24576,4096] T=1, while other cases in the same arm pass;
//   * ninfer_linear_q5_a16_test FAILS the criterion at Q5_A16 [4096,4096] for T=2,4,5,16,
//     17,128 (e.g. actual=-12.1875 vs reference=14.3661).
//
// Both bands must therefore be REFUSED, and the refusal must carry the failing case, not
// just "unsupported". If a later edit turns either into an admitted band, this fails.
// ---------------------------------------------------------------------------
void test_probe_negative_findings_refuse() {
    const std::string path = std::string(source_dir()) +
                             "/tools/archkit/probe_results/format_support-120.list";
    const auto list = ninfer::caps::FormatSupportList::load_file(path);
    if (!list.valid()) {
        std::cout << "SKIP T13: no usable probe list at " << path << "\n";
        return;
    }

    const auto* q4 = list.find(kKeyA, ninfer::artifact::NumericFormat::Q4G64_F16S, 2, "A16");
    check(q4 != nullptr, "T13: the Q4G64_F16S 1..3 row is present");
    if (q4 != nullptr) {
        check(!q4->supported, "T13: Q4G64_F16S is NOT supported after the probe");
        check(q4->arm_rc != 0,
              "T13: and the entry records the arm was not green (measured: "
              "cudaErrorIllegalAddress at [24576,4096] T=1)");
        check(q4->cases > 0, "T13: the band DOES have measured cases -- this is a real "
                             "failure, not an unprobed band");
    }
    setenv("NINFER_FORMAT_PROBE_LIST", path.c_str(), 1);
    const auto q4v = ninfer::caps::format_probe_gate(
        declared_a(), measured_a(), ninfer::artifact::NumericFormat::Q4G64_F16S, 2, "test/artifact",
        list.provenance().binary_sha16.c_str());
    check(q4v.enabled && !q4v.allowed, "T13: the gate refuses Q4G64_F16S");
    check(q4v.why.find("MEASURED UNSUPPORTED") != std::string::npos,
          "T13: and says it was MEASURED unsupported rather than absent");

    const auto* q5 = list.find(kKeyA, ninfer::artifact::NumericFormat::Q5G64_F16S, 2, "A16");
    check(q5 != nullptr, "T13: the Q5G64_F16S 1..3 row is present");
    if (q5 != nullptr) {
        check(!q5->supported, "T13: Q5G64_F16S is NOT supported after the probe");
        check(q5->cases > 0, "T13: with measured cases in the band");
        check(!q5->first_failure.empty(),
              "T13: and a recorded failing case (measured: Q5_A16 [4096,4096] T=2)");
    }
    const auto q5v = ninfer::caps::format_probe_gate(
        declared_a(), measured_a(), ninfer::artifact::NumericFormat::Q5G64_F16S, 2, "test/artifact",
        list.provenance().binary_sha16.c_str());
    check(q5v.enabled && !q5v.allowed, "T13: the gate refuses Q5G64_F16S");
    check(q5v.why.find("FIRST FAILING CASE") != std::string::npos,
          "T13: and the refusal names the first failing case, so the reason is diagnosable");
    unsetenv("NINFER_FORMAT_PROBE_LIST");

    // The counterpart, so T13 is not just "everything is refused": the arms that WERE green
    // must come back admitted. BF16 is the control.
    const auto* bf16 = list.find(kKeyA, ninfer::artifact::NumericFormat::BF16, 2, "A16");
    check(bf16 != nullptr && bf16->supported && bf16->arm_rc == 0,
          "T13: BF16 (a green arm) is recorded supported and its band is admitted");
}

// ---------------------------------------------------------------------------
// T15  THE KEY IS THE MEASURED CAPABILITY VECTOR, AND A NAME CANNOT BE ONE.
//
// The user's requirement: "甚至依我之见必须完全放弃名字探查这种，因为有改 vbios 的卡" -- name
// probing abandoned entirely, because a modified VBIOS can make a card present a listed name
// over different silicon. So: the key is the vector, two vectors that agree render the same
// string, the vector's version is IN the key (a change of vector stops old rows matching),
// and a list written by the name/cc-keyed generation is REFUSED rather than reinterpreted.
// ---------------------------------------------------------------------------
void test_key_is_measured_not_declared() {
    using ninfer::caps::MeasuredCapabilities;
    using ninfer::caps::measured_capability_key;
    check_eq(measured_capability_key(measured_a()), std::string(kKeyA),
             "T15: the rendered key matches the fixture (field order and version are fixed)");
    check_eq(measured_capability_key(measured_b()), std::string(kKeyB),
             "T15: a different measured vector renders a different key");
    check(measured_capability_key(measured_a()) != measured_capability_key(measured_b()),
          "T15: and the two are distinguishable -- the key is a fact, not a label");
    check(measured_capability_key(measured_a()).find("name") == std::string::npos &&
              measured_capability_key(measured_a()).find("5090") == std::string::npos,
          "T15: the key contains no device name at all");
    check(measured_capability_key(measured_a()).compare(0, 2, "v1") == 0,
          "T15: the vector version is part of the key, so a changed vector stops matching");

    // A name/cc-keyed list from the previous generation must be REFUSED, not read under a key
    // whose meaning has changed.
    const std::string legacy =
        "meta probe_tool=\"tools/archkit/probe_formats.sh\"\n"
        "meta criterion_id=\"linear-reduction-v1\"\n"
        "meta criterion_src=\"tests/ops/linear/linear_test_common.cpp:38-48\"\n"
        "meta reference=\"cpu_linear_gemm_fp64\"\n"
        "meta arch=\"120\"\n"
        "meta arch_suffix=\"a\"\n"
        "meta date=\"2026-09-18\"\n"
        "meta binary_sha16=\"aaaaaaaaaaaaaaaa\"\n"
        "meta source=\"real_gpu\"\n" +
        std::string("entry arch=120 format=NVFP4 path=A4 band=17..open supported=yes "
                    "leg=reduction_l2 cases=3\n");
    const auto legacy_path = write_temp("legacy_arch_keyed.list", legacy);
    const auto legacy_list = ninfer::caps::FormatSupportList::load_file(legacy_path);
    check(!legacy_list.valid(), "T15: a name/cc-keyed list is not valid");
    check(legacy_list.parse_error().find("declaration") != std::string::npos,
          "T15: and the parse error names the reason (a declaration is not a measurement)");
    setenv("NINFER_FORMAT_PROBE_LIST", legacy_path.c_str(), 1);
    const auto verdict = ninfer::caps::format_probe_gate(
        declared_a(), measured_a(), ninfer::artifact::NumericFormat::NVFP4, 1024, "test/artifact",
        "aaaaaaaaaaaaaaaa");
    check(verdict.enabled && !verdict.allowed, "T15: the gate refuses a legacy-keyed list");
    unsetenv("NINFER_FORMAT_PROBE_LIST");
}

// ---------------------------------------------------------------------------
// T16  A DECLARATION THAT DISAGREES WITH ITS OWN MEASUREMENT IS REPORTED, NOT ABSORBED.
//
// This is the modded-VBIOS signature: the card declares an identity for which the list HAS a
// row, but this device's own known-answer probes measure a different capability vector. The
// declaration is not the verdict, so the row is invalidated for this device, the run refuses,
// and the disagreement is a named field rather than a line in a log nobody reads.
// ---------------------------------------------------------------------------
void test_declaration_disagreement_is_reported() {
    // A list whose rows were measured under the 5090-D declaration with key A.
    const std::string list =
        good_header("aaaaaaaaaaaaaaaa") +
        entry_line(kKeyA, "NVFP4", "A4", "17..open", true, "reduction_l2", 3, 0.01,
                   "NVFP4_A4 [14336,5120] T=17");
    const std::string path = write_temp("disagree.list", list);
    setenv("NINFER_FORMAT_PROBE_LIST", path.c_str(), 1);
    unsetenv("NINFER_FORMAT_PROBE_ACCEPT_STALE");

    // THE MODDED CARD: it declares the 5090 D (whose row exists), and its measurement is B.
    const auto verdict = ninfer::caps::format_probe_gate(
        declared_a(), measured_b(), ninfer::artifact::NumericFormat::NVFP4, 1024, "test/artifact",
        "aaaaaaaaaaaaaaaa");
    check(verdict.enabled && !verdict.allowed,
          "T16: a declaration that disagrees with the measurement REFUSES");
    check(verdict.declaration_disagrees,
          "T16: and is reported as a declaration/measurement disagreement, not as a miss");
    check(verdict.why.find("DECLARATION AND MEASUREMENT DISAGREE") != std::string::npos,
          "T16: the message says so in words");
    check(verdict.why.find(std::string(kKeyA)) != std::string::npos &&
              verdict.why.find(std::string(kKeyB)) != std::string::npos,
          "T16: and prints BOTH keys, so the disagreement is visible without a debugger");
    check(verdict.why.find("reflashed") != std::string::npos ||
              verdict.why.find("mismatched") != std::string::npos,
          "T16: and names the condition it is the signature of");

    // THE HONEST CARD: same declaration, measurement A -- admitted.
    const auto honest = ninfer::caps::format_probe_gate(
        declared_a(), measured_a(), ninfer::artifact::NumericFormat::NVFP4, 1024, "test/artifact",
        "aaaaaaaaaaaaaaaa");
    check(honest.enabled && honest.allowed && !honest.declaration_disagrees,
          "T16: the same declaration WITH a matching measurement is admitted");
    unsetenv("NINFER_FORMAT_PROBE_LIST");
}

// ---------------------------------------------------------------------------
// T17  THERE IS NO DEFAULT-ALLOW PATH, AND THE DISABLED GATE STILL REPORTS.
//
// The counterexample to design against: an engine that used a generic model for an unlisted
// capability and said so only in a field nobody reads as "this is unmeasured". So: an
// unmeasured key NEVER becomes allowed by any input; and when the gate is disabled the reason
// text still states the measured key and the declaration, and says plainly that a run relying
// on the declaration is relying on something a modified VBIOS can falsify.
// ---------------------------------------------------------------------------
void test_no_default_allow_path() {
    const std::string list =
        good_header("aaaaaaaaaaaaaaaa") +
        entry_line(kKeyA, "NVFP4", "A4", "17..open", true, "reduction_l2", 3, 0.01, "c");
    const std::string path = write_temp("defaults.list", list);
    setenv("NINFER_FORMAT_PROBE_LIST", path.c_str(), 1);
    setenv("NINFER_FORMAT_PROBE_ACCEPT_STALE", "1", 1); // even the staleness opt-in...

    // ...does not turn an unmeasured key into an admitted one.
    const auto unmeasured = ninfer::caps::format_probe_gate(
        declared_a(), measured_b(), ninfer::artifact::NumericFormat::NVFP4, 1024, "test/artifact",
        "aaaaaaaaaaaaaaaa");
    check(unmeasured.enabled && !unmeasured.allowed,
          "T17: an unmeasured key is never admitted, even with the staleness opt-in set");

    // A format with no row at all on a measured key: refused, and it is an ON-RAMP.
    const auto missing_row = ninfer::caps::format_probe_gate(
        declared_a(), measured_a(), ninfer::artifact::NumericFormat::Q5G64_F16S, 1024,
        "test/artifact", "aaaaaaaaaaaaaaaa");
    check(missing_row.enabled && !missing_row.allowed && missing_row.refused_for_absent,
          "T17: an unprobed (key, format, band) refuses");
    check(missing_row.why.find("ON-RAMP") != std::string::npos,
          "T17: and the refusal is an ON-RAMP, not a dead end");
    check(missing_row.why.find("probe_formats.sh") != std::string::npos,
          "T17: naming the command that would add the row");
    check(missing_row.why.find("NINFER_PROBE_KEY=") != std::string::npos,
          "T17: and carrying the observed key, so the person with the card can run it");
    check(missing_row.why.find("/mnt/g/cuda12/tk") != std::string::npos,
          "T17: and naming the toolkit that reaches the low rungs");
    unsetenv("NINFER_FORMAT_PROBE_ACCEPT_STALE");
    unsetenv("NINFER_FORMAT_PROBE_LIST");

    // DISABLED: nothing is admitted, and the reason still refuses to let a declaration pass
    // for a measurement.
    const auto off = ninfer::caps::format_probe_gate(
        declared_a(), measured_a(), ninfer::artifact::NumericFormat::NVFP4, 1024, "test/artifact",
        "aaaaaaaaaaaaaaaa");
    check(!off.enabled && !off.allowed, "T17: disabled means nothing is admitted by this gate");
    check(off.why.find(std::string(kKeyA)) != std::string::npos,
          "T17: the disabled answer reports the MEASURED key it observed");
    check(off.why.find("VBIOS") != std::string::npos,
          "T17: and says that relying on the declaration is relying on something a modified "
          "VBIOS can falsify");
}

} // namespace

int main() {
    test_criterion_matches_its_source();
    test_bands_match_the_router();
    test_gate_disabled_by_default();
    test_absent_refuses_and_names_what_to_probe();
    test_a_support_claim_needs_a_measurement();
    test_route_verdict_is_not_arithmetic_support();
    test_staleness_is_refused_unless_accepted();
    test_unreadable_and_incomplete_lists_refuse();
    test_render_round_trip();
    test_status_quo_nvfp4_a4_concession_band_refuses();
    test_probe_negative_findings_refuse();
    test_key_is_measured_not_declared();
    test_declaration_disagreement_is_reported();
    test_no_default_allow_path();

    std::cout << (g_failures == 0 ? "OK  " : "FAIL ") << "test_format_probe: " << g_checks
              << " checks, " << g_failures << " failure(s)\n";
    return g_failures == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// REGISTRATION -- the one line that needs a SHARED file, and is therefore reported rather
// than applied. tests/CMakeLists.txt carries 734 added / 393 deleted uncommitted lines and
// is owned by another line in flight, so this file is NOT registered there. It links no
// ninfer library (the subject is header-only and host-only), so it registers exactly like
// the other contract tests:
//
//   add_executable(ninfer_format_probe_test test_format_probe.cpp)
//   target_include_directories(ninfer_format_probe_test PRIVATE
//     ${CMAKE_CURRENT_SOURCE_DIR}
//     ${PROJECT_SOURCE_DIR}/include
//     ${PROJECT_SOURCE_DIR}/src
//     ${PROJECT_SOURCE_DIR}/third_party)
//   target_compile_definitions(ninfer_format_probe_test PRIVATE
//     NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
//   add_test(NAME ninfer_format_probe_test COMMAND ninfer_format_probe_test)
//
// Until then it runs standalone; the invocation is in this file's header comment.
// ---------------------------------------------------------------------------
