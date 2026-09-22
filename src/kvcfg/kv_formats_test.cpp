// kv_formats_test — CPU 对拍 kv_formats.h 解析与 kv_tiers.py 用例集一致。
//
// SECOND SECTION (added): the deprecated spellings and the codec line. The header's
// own static_asserts pin these at compile time, but a static_assert that only ever
// fires on an edit nobody makes proves nothing to a reader; these cases print the
// resolution, so the alias table is visible in a run as well as checked by the
// compiler. `iso3` and `iso4` are the pair a predecessor read as each other, so the
// cases state BOTH that iso3==iso4e and that iso4 is NOT iso4e.
#include "kvcfg/kv_formats.h"

#include <cstdio>
#include <string>
#include <string_view>

using ninfer::kvcfg::KvFormat;
using ninfer::kvcfg::Nvfp4Mode;
using ninfer::kvcfg::format_from_name;
using ninfer::kvcfg::has_resident_codec;
using ninfer::kvcfg::is_readable_codec;
using ninfer::kvcfg::name_of;
using ninfer::kvcfg::parse_tier_formats;

int main() {
    int ok = 0;
    int total = 8;  // the 6 case rows below, plus the 2 default-value checks
    struct Case { const char* spec; Nvfp4Mode mode; bool expect_ok; };
    const Case cases[] = {
        {"hot=bf16,tail=fp16,cold=iso4e", Nvfp4Mode::Fusion, true},
        {"cold=iso4e", Nvfp4Mode::Pure, false},
        {"cold=int4", Nvfp4Mode::Pure, true},
        {"cold=rk4v4", Nvfp4Mode::Fusion, true},
        {"hot=int4", Nvfp4Mode::Fusion, false},
        {"hot=int8,tail=int4", Nvfp4Mode::Fusion, false},
    };
    for (const auto& c : cases) {
        std::string err;
        auto r = parse_tier_formats(c.spec, c.mode, &err);
        const bool got = r.has_value();
        const bool pass = got == c.expect_ok;
        ok += pass;
        std::printf("%s %-28s mode=%s -> %s\n", pass ? "PASS" : "FAIL", c.spec,
                    c.mode == Nvfp4Mode::Pure ? "pure" : "fusion",
                    got ? "ok" : err.c_str());
    }
    // 默认值: fusion cold=iso4e; pure cold=int8
    auto f = parse_tier_formats("", Nvfp4Mode::Fusion);
    auto p = parse_tier_formats("", Nvfp4Mode::Pure);
    std::printf("fusion default cold=%s  pure default cold=%s\n",
                ninfer::kvcfg::name_of(f->cold), ninfer::kvcfg::name_of(p->cold));
    ok += f->cold == ninfer::kvcfg::KvFormat::Iso4e;
    ok += p->cold == ninfer::kvcfg::KvFormat::Int8;

    // ---------------------------------------------------------------------
    // SECTION 2: the deprecated spellings, and the row they are NOT.
    // ---------------------------------------------------------------------
    struct AliasCase { const char* spelling; KvFormat want; };
    const AliasCase aliases[] = {
        {"iso3", KvFormat::Iso4e},   // NOT Iso4 -- this is the pair that was confused
        {"e8",   KvFormat::Rk4v4},
        {"e8k3", KvFormat::Rk3v4},
        {"e8k2", KvFormat::Rk2v4},
    };
    for (const auto& a : aliases) {
        ++total;
        const auto got = format_from_name(a.spelling);
        const bool pass = got.has_value() && *got == a.want;
        ok += pass;
        std::printf("%s alias %-6s -> %s (emitted name \"%s\")\n", pass ? "PASS" : "FAIL",
                    a.spelling, got.has_value() ? name_of(*got) : "<unparsed>",
                    got.has_value() ? name_of(*got) : "-");
    }
    // The distinction, as a runtime-visible fact and not only a static_assert.
    ++total;
    {
        const auto iso4  = format_from_name("iso4");
        const auto iso3  = format_from_name("iso3");
        const bool pass = iso4.has_value() && iso3.has_value() && *iso4 == KvFormat::Iso4 &&
                          *iso3 == KvFormat::Iso4e && *iso4 != *iso3;
        ok += pass;
        std::printf("%s iso4 != iso3  (iso4=%s iso4e=%s)\n", pass ? "PASS" : "FAIL",
                    iso4.has_value() ? name_of(*iso4) : "<unparsed>",
                    iso3.has_value() ? name_of(*iso3) : "<unparsed>");
    }
    // The codec line. iso3==iso4e is deployable; iso4/fp16/int4 are vocabulary-only;
    // rk3v4/rk2v4 have a codec but no reader.
    struct CodecCase { KvFormat f; bool resident; bool readable; };
    const CodecCase codecs[] = {
        {KvFormat::Bf16,    true,  true},
        {KvFormat::Int8,    true,  true},
        {KvFormat::Iso4e,   true,  true},   // == the nvfp4 family in this vocabulary
        {KvFormat::Rk4v4,   true,  true},
        {KvFormat::Rk3v4,   true,  false},  // writable, not readable
        {KvFormat::Rk2v4,   true,  false},
        {KvFormat::Auto,    false, false},  // "inherit", not a format
        {KvFormat::Fp16,    false, false},
        {KvFormat::Int4,    false, false},
        {KvFormat::Iso4,    false, false},  // the row iso3 does NOT resolve to
    };
    for (const auto& c : codecs) {
        ++total;
        const bool pass = has_resident_codec(c.f) == c.resident &&
                          is_readable_codec(c.f) == c.readable;
        ok += pass;
        std::printf("%s codec %-6s resident=%d readable=%d\n", pass ? "PASS" : "FAIL",
                    name_of(c.f), static_cast<int>(has_resident_codec(c.f)),
                    static_cast<int>(is_readable_codec(c.f)));
    }
    // The two engine tiers this vocabulary CANNOT name. If either ever parses, the
    // header's assert has been weakened and the codec mapping is missing.
    {
        const char* absent[] = {"nvfp4", "fp8"};
        for (const char* n : absent) {
            ++total;
            const bool pass = !format_from_name(n).has_value();
            ok += pass;
            std::printf("%s vocabulary cannot spell \"%s\" (it is a --kv-dtype token)\n",
                        pass ? "PASS" : "FAIL", n);
        }
    }

    std::printf("== %d/%d\n", ok, total);
    return ok == total ? 0 : 1;
}
