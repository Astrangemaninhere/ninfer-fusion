// The three KV component switches and their TIER DOMAINS
// (src/product/kv_component_switch.h).
//
// Host-only and std-only: the predicate is a pure function over the per-layer
// dtype table, so the whole question -- "would --kv-rotation / --kv-row-scale /
// --kv-v-codec reach a kernel on THIS configuration?" -- is answerable without
// CUDA, a GPU or an artifact. Same shape as tests/test_kv_tier_formats.cpp and
// tests/test_kv_cold_tier_budget.cpp.
//
// WHAT THIS PINS. Each switch is honoured by a subset of the KV tiers, so a
// switch set outside its domain used to parse, validate and change nothing --
// the repository's definition of fake completeness. The two properties that
// matter are therefore:
//   * a switch that cannot take effect is REFUSED, with the missing part named;
//   * a switch that CAN take effect is never refused, including through the
//     BF16-inherits-the-global-dtype rule plan_cache() uses.

#include "product/kv_component_switch.h"
#include "product/kv_options.h"
#include "product/kv_storage_dtype.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>

namespace {

namespace p = ninfer::product;

using ninfer::DType;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << what << '\n';
}

[[nodiscard]] std::string refusal(p::KvComponentSwitch which, DType kv_dtype,
                                  std::span<const DType> layer_dtypes) {
    return p::kv_component_switch_domain_error(which, kv_dtype, layer_dtypes);
}

// ---- the inheritance rule the whole predicate rests on ----------------------

void test_layer_dtype_resolution() {
    // plan_cache(): selected = (override == BF16) ? global : override -- FOR A
    // SLOT THE SPEC DID NOT WRITE. That qualifier IS the change: this rule used
    // to be stated unconditionally and asserted as if it were the desirable
    // behaviour, which is exactly why no per-layer BF16 baseline could be written
    // at all. The qualifier itself is pinned in
    // test_slot_explicitness_is_the_bf16_bypass() below.
    check(p::kv_resolved_layer_dtype(DType::BF16, DType::BF16) == DType::BF16,
          "BF16 override on a BF16 cache stays BF16");
    check(p::kv_resolved_layer_dtype(DType::NVFP4, DType::BF16) == DType::NVFP4,
          "an UNSET BF16 slot inherits a non-BF16 global dtype (unchanged, and now the only "
          "case that rule applies to)");
    check(p::kv_resolved_layer_dtype(DType::NVFP4, DType::ISO3) == DType::ISO3,
          "a non-BF16 override wins over the global dtype");
    check(p::kv_resolved_layer_dtype(DType::BF16, DType::I8) == DType::I8,
          "an I8 override is I8 even on a BF16 global");
}

// ---- the bypass: an EXPLICIT bf16 slot is BF16, an unset one inherits -------

void test_slot_explicitness_is_the_bf16_bypass() {
    // The case that had no spelling before. With the inheritance rule stated
    // unconditionally, `--kv-layer-storage 0-11:bf16` under `--kv-dtype nvfp4`
    // was indistinguishable from "leave 0-11 alone": the operator's dtype was
    // silently ignored, with no diagnostic, and the only workaround was to make
    // the WHOLE stack BF16 with the global flag (product/kv_options.h said so).
    check(p::kv_resolve_slot_dtype(DType::NVFP4, DType::BF16, /*slot_explicit=*/false) ==
              DType::NVFP4,
          "an UNSET bf16 slot inherits a non-BF16 global dtype");
    check(p::kv_resolve_slot_dtype(DType::NVFP4, DType::BF16, /*slot_explicit=*/true) ==
              DType::BF16,
          "an EXPLICIT bf16 slot stays BF16 under a non-BF16 global dtype (the bypass)");
    // Every other storage is itself either way: the mask must not have become a
    // second way to say "inherit".
    for (const DType tier :
         {DType::I8, DType::FP8_E4M3FN, DType::NVFP4, DType::ISO3, DType::E8Kv}) {
        check(p::kv_resolve_slot_dtype(DType::BF16, tier, false) == tier &&
                  p::kv_resolve_slot_dtype(DType::BF16, tier, true) == tier,
              "a non-BF16 override wins over the global dtype, mask or no mask");
    }
    // The domain predicate must see the SAME resolution the pool does, or the
    // two halves of one guard disagree about the same layers -- the failure that
    // let the V-codec guard walk past its own case.
    const std::array<DType, 2> all_bf16{DType::BF16, DType::BF16};
    const std::array<bool, 2> both_written{true, true};
    const std::array<bool, 2> neither_written{false, false};
    check(p::kv_component_switch_domain_active(p::KvComponentSwitch::Rotation, DType::NVFP4,
                                               all_bf16, both_written) == false,
          "written bf16 slots under a quantized global ARE BF16 layers, so the SO(4) gate is "
          "outside the domain");
    check(p::kv_component_switch_domain_active(p::KvComponentSwitch::Rotation, DType::NVFP4,
                                               all_bf16, neither_written) == true,
          "the same table with NO written slots inherits NVFP4 and IS in the domain");
    const std::array<DType, 2> mixed{DType::BF16, DType::NVFP4};
    check(p::kv_component_switch_domain_active(p::KvComponentSwitch::Rotation, DType::NVFP4,
                                               mixed, both_written) == true,
          "one explicitly-written nvfp4 layer keeps the switch reachable");
}

// ---- an unknown / unmapped dtype must ERROR, never become BF16 --------------

void test_unknown_dtype_names_and_codes_are_refused() {
    // (a) a misspelled NAME inside a spec: refused, and the refusal names the
    //     offending text and the layers it was written for. Before, the message
    //     was "kv-layer-storage has an invalid type" with no text at all, so the
    //     operator had to guess which of five comma-separated entries was wrong.
    bool threw = false;
    try {
        (void)ninfer::product::parse_kv_layer_storage("0-3:int9");
    } catch (const std::invalid_argument& e) {
        threw = true;
        const std::string what = e.what();
        check(what.find("int9") != std::string::npos,
              "the refusal names the misspelled dtype text");
        check(what.find("0-3") != std::string::npos,
              "the refusal names the layer range the typo was written for");
    }
    check(threw, "a misspelled dtype name is refused instead of resolving to BF16");
    //     Positive control: the correct spelling of the same entry still parses.
    check(ninfer::product::parse_kv_storage("int8").has_value(),
          "CONTROL: the correctly spelled name still parses");

    // (b) one slot written TWICE. This used to be silently accepted whenever the
    //     first write was bf16, because the duplicate test read the TABLE and
    //     bf16 is the table's default value: "0:bf16,0:int8" quietly became int8
    //     and the operator's bf16 vanished with no diagnostic. The "was set" mask
    //     is what can see it.
    threw = false;
    try {
        (void)ninfer::product::parse_kv_layer_storage("0:bf16,0:int8");
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw,
          "a slot written twice is refused (it used to be silently won by the second write)");
    //     Controls for the mask itself.
    const auto spec = ninfer::product::parse_kv_layer_storage_spec("0:bf16,1:int8");
    check(spec.set[0] && spec.set[1] &&
              spec.table[0] == ninfer::KvCacheStorage::BFloat16 &&
              spec.table[1] == ninfer::KvCacheStorage::Int8Group64,
          "two entries on different slots are accepted and BOTH marked written");
    check(!spec.set[2], "a slot the spec never mentioned is not marked written");
    check(!ninfer::product::parse_kv_layer_storage_spec("").set[0],
          "an empty spec marks nothing written");

    // (c) the storage->DType mapping, swept over the whole byte range. Every
    //     enumerator maps to its resident codec; every code no enumerator names
    //     is REFUSED. This is the silent downgrade itself: the old mapping ended
    //     in `: DType::BF16`, and because BF16 is also the "inherit the global
    //     --kv-dtype" sentinel an unmapped code did not merely lose its codec, it
    //     silently meant "ignore this layer's dtype".
    const std::array<std::pair<ninfer::KvCacheStorage, DType>, 7> total{{
        {ninfer::KvCacheStorage::BFloat16, DType::BF16},
        {ninfer::KvCacheStorage::Int8Group64, DType::I8},
        {ninfer::KvCacheStorage::Fp8E4M3Row256, DType::FP8_E4M3FN},
        {ninfer::KvCacheStorage::Fp8Group16, DType::FP8_E4M3FN},
        {ninfer::KvCacheStorage::Nvfp4Group16, DType::NVFP4},
        {ninfer::KvCacheStorage::Iso3Group16, DType::ISO3},
        {ninfer::KvCacheStorage::E8Group64, DType::E8Kv},
    }};
    for (const auto& entry : total) {
        check(ninfer::product::kv_dtype_for_storage(entry.first, "test") == entry.second,
              "every KvCacheStorage enumerator maps to its resident codec");
    }
    unsigned refused = 0, silently_bf16 = 0;
    for (unsigned code = 0; code < 256; ++code) {
        const auto storage = static_cast<ninfer::KvCacheStorage>(code);
        try {
            const DType got = ninfer::product::kv_dtype_for_storage(storage, "test");
            if (code > 6 && got == DType::BF16) { ++silently_bf16; }
        } catch (const std::invalid_argument&) { ++refused; }
    }
    std::cout << "  [mapping sweep] codes=256 mapped=7 refused=" << refused
              << " silently_bf16=" << silently_bf16 << '\n';
    check(silently_bf16 == 0,
          "no unmapped storage code resolves to BF16 (the silent downgrade is gone)");
    check(refused == 249U,
          "exactly the 249 codes no enumerator names are refused by the mapping");
}

// ---- Rotation: refused unless some layer is on {NVFP4, FP8, ISO4E} ----------

void test_rotation_domain() {
    const std::array<DType, 4> all_bf16{DType::BF16, DType::BF16, DType::BF16, DType::BF16};
    const std::string off =
        refusal(p::KvComponentSwitch::Rotation, DType::BF16, all_bf16);
    check(!off.empty(),
          "all-BF16: --kv-rotation off is refused (it could not take effect)");
    check(off.find("BF16") != std::string::npos,
          "the rotation refusal names BF16 as the reason");
    check(off.find("--kv-dtype") != std::string::npos,
          "the rotation refusal names the flag that would make it reachable");

    // The refusal must name the tiers that CANNOT honour the switch, because
    // "int8|rk4v4" used to be advertised as reachable in the refusal text.
    check(off.find("Rk4v4") != std::string::npos || off.find("rk4v4") != std::string::npos,
          "the rotation refusal names Rk4v4 and the Hadamard that actually rotates it");
    check(off.find("gqa_kv_hadamard64") != std::string::npos,
          "the rotation refusal names the Rk4v4 rotation mechanism by symbol");
    check(off.find("gqa_isoquant_rot_block4") != std::string::npos,
          "the rotation refusal names the gate function that is the domain test");

    // One packable layer anywhere in the stack makes the switch reachable.
    const std::array<DType, 4> one_nvfp4{DType::BF16, DType::NVFP4, DType::BF16, DType::BF16};
    check(refusal(p::KvComponentSwitch::Rotation, DType::BF16, one_nvfp4).empty(),
          "one NVFP4 layer: --kv-rotation off is accepted");
    const std::array<DType, 2> one_iso4e{DType::BF16, DType::ISO3};
    check(refusal(p::KvComponentSwitch::Rotation, DType::BF16, one_iso4e).empty(),
          "one ISO4E layer: --kv-rotation off is accepted");
    const std::array<DType, 2> one_fp8{DType::BF16, DType::FP8_E4M3FN};
    check(refusal(p::KvComponentSwitch::Rotation, DType::BF16, one_fp8).empty(),
          "one FP8 layer: --kv-rotation off is accepted");
    // I8 and Rk4v4 are BOTH outside the domain: no I8/Rk4v4 kernel reads the gate.
    // I8 rotates on the legacy D256 stack (normalized_hadamard_d256_inplace) and
    // not on the GQA stack; Rk4v4 rotates on both, through gqa_kv_hadamard64().
    const std::array<DType, 2> one_i8{DType::BF16, DType::I8};
    const std::string i8 = refusal(p::KvComponentSwitch::Rotation, DType::BF16, one_i8);
    check(!i8.empty(),
          "one I8 layer: --kv-rotation off is refused (no I8 kernel reads this gate, "
          "and the legacy I8 path rotates through its own Hadamard)");
    check(i8.find("gqa_kv_hadamard64") != std::string::npos,
          "the I8 refusal names Rk4v4's Hadamard, so the reader can tell the two apart");
    check(i8.find("normalized_hadamard_d256_inplace") != std::string::npos,
          "the I8 refusal names the legacy D256 rotation as the I8 mechanism");
    check(i8.find("kernel.cuh:234") != std::string::npos,
          "the I8 refusal cites the legacy rotate site, not just a file name");
    const std::array<DType, 2> one_rk4v4{DType::BF16, DType::E8Kv};
    const std::string rk4v4 = refusal(p::KvComponentSwitch::Rotation, DType::BF16, one_rk4v4);
    check(!rk4v4.empty(),
          "one Rk4v4 layer: --kv-rotation off is refused (Rk4v4 rotates through an "
          "unconditional Hadamard, so 'off' would be a lie)");

    // The global dtype alone can lift the refusal: an all-BF16 override table on
    // a quantized cache has every layer quantized.
    check(refusal(p::KvComponentSwitch::Rotation, DType::NVFP4, all_bf16).empty(),
          "global NVFP4 with an all-BF16 override table is a quantized stack");
    check(refusal(p::KvComponentSwitch::Rotation, DType::ISO3, all_bf16).empty(),
          "global ISO4E with an all-BF16 override table is a quantized stack");
    // The global dtype alone cannot lift the refusal either when it is I8/Rk4v4.
    check(!refusal(p::KvComponentSwitch::Rotation, DType::I8, all_bf16).empty(),
          "global int8 with an all-BF16 override table is still outside the domain");
    check(!refusal(p::KvComponentSwitch::Rotation, DType::E8Kv, all_bf16).empty(),
          "global rk4v4 with an all-BF16 override table is still outside the domain");
    // ... and a single packable layer among I8/Rk4v4 ones is enough to accept.
    const std::array<DType, 4> mostly_i8{DType::I8, DType::I8, DType::ISO3, DType::I8};
    check(refusal(p::KvComponentSwitch::Rotation, DType::I8, mostly_i8).empty(),
          "one ISO4E layer among I8 layers makes --kv-rotation off reachable");
    const std::array<DType, 4> i8_and_rk4v4{DType::I8, DType::E8Kv, DType::I8, DType::E8Kv};
    check(!refusal(p::KvComponentSwitch::Rotation, DType::I8, i8_and_rk4v4).empty(),
          "a stack of only I8 and Rk4v4 layers is refused");

    // Empty override table == inherit wholesale.
    check(!refusal(p::KvComponentSwitch::Rotation, DType::BF16, {}).empty(),
          "empty table + BF16 global: refused");
    check(refusal(p::KvComponentSwitch::Rotation, DType::FP8_E4M3FN, {}).empty(),
          "empty table + FP8 global: accepted");
}

// ---- RowScale / VCodec: the domain is exactly {NVFP4} ----------------------

void test_nvfp4_only_domain() {
    const std::array<DType, 3> all_bf16{DType::BF16, DType::BF16, DType::BF16};
    const std::array<DType, 3> all_iso4e{DType::ISO3, DType::ISO3, DType::ISO3};
    const std::array<DType, 3> all_i8{DType::I8, DType::I8, DType::I8};
    const std::array<DType, 3> mixed{DType::BF16, DType::NVFP4, DType::ISO3};

    for (const p::KvComponentSwitch which :
         {p::KvComponentSwitch::RowScale, p::KvComponentSwitch::VCodec}) {
        const bool rowscale = which == p::KvComponentSwitch::RowScale;
        const std::string name = rowscale ? "row-scale" : "v-codec";

        const std::string bf16 = refusal(which, DType::BF16, all_bf16);
        check(!bf16.empty(), name + ": all-BF16 is refused");
        check(bf16.find("NVFP4") != std::string::npos,
              name + ": the refusal names NVFP4 as the missing tier");

        check(!refusal(which, DType::ISO3, all_iso4e).empty(),
              name + ": an all-ISO4E stack is refused (ISO4E is not the NVFP4 tier)");
        check(!refusal(which, DType::I8, all_i8).empty(),
              name + ": an all-I8 stack is refused");
        check(refusal(which, DType::BF16, mixed).empty(),
              name + ": a stack with one NVFP4 layer is accepted");
        check(refusal(which, DType::NVFP4, all_bf16).empty(),
              name + ": a global NVFP4 dtype with BF16 overrides is accepted");
        check(!refusal(which, DType::BF16, {}).empty(),
              name + ": empty table + BF16 global is refused");
        check(refusal(which, DType::NVFP4, {}).empty(),
              name + ": empty table + NVFP4 global is accepted");
    }

    // The two domains are the same predicate, so they must agree everywhere.
    for (const DType global : {DType::BF16, DType::I8, DType::ISO3, DType::NVFP4}) {
        check(refusal(p::KvComponentSwitch::RowScale, global, mixed) ==
                  refusal(p::KvComponentSwitch::VCodec, global, mixed),
              "row-scale and v-codec share one domain predicate");
    }
}

// ---- the "default run" case the guard exists for ---------------------------

void test_default_configuration_is_the_noop_case() {
    // The engine's default KV dtype is BF16 and the default layer table is
    // all-BF16, so every component switch is inert unless a quantized tier is
    // requested. This is exactly the configuration the guard must catch.
    const std::array<DType, 16> defaults{};
    check(defaults[0] == DType::BF16, "the default layer table is BF16 (DType 0)");
    for (const p::KvComponentSwitch which :
         {p::KvComponentSwitch::Rotation, p::KvComponentSwitch::RowScale,
          p::KvComponentSwitch::VCodec}) {
        check(!refusal(which, DType::BF16, defaults).empty(),
              "every component switch is refused on the default BF16 configuration");
    }
}

} // namespace

int main() {
    test_layer_dtype_resolution();
    test_slot_explicitness_is_the_bf16_bypass();
    test_unknown_dtype_names_and_codes_are_refused();
    test_rotation_domain();
    test_nvfp4_only_domain();
    test_default_configuration_is_the_noop_case();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "kv component switch: all checks passed\n";
    return 0;
}
