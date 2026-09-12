// kv_formats_test — CPU 对拍 kv_formats.h 解析与 kv_tiers.py 用例集一致。
#include "kvcfg/kv_formats.h"

#include <cstdio>
#include <string>

using ninfer::kvcfg::Nvfp4Mode;
using ninfer::kvcfg::parse_tier_formats;

int main() {
    int ok = 0;
    const int total = 6;
    struct Case { const char* spec; Nvfp4Mode mode; bool expect_ok; };
    const Case cases[] = {
        {"hot=bf16,tail=fp16,cold=iso3", Nvfp4Mode::Fusion, true},
        {"cold=iso3", Nvfp4Mode::Pure, false},
        {"cold=int4", Nvfp4Mode::Pure, true},
        {"cold=e8", Nvfp4Mode::Fusion, true},
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
    // 默认值: fusion cold=iso3; pure cold=int8
    auto f = parse_tier_formats("", Nvfp4Mode::Fusion);
    auto p = parse_tier_formats("", Nvfp4Mode::Pure);
    std::printf("fusion default cold=%s  pure default cold=%s\n",
                ninfer::kvcfg::name_of(f->cold), ninfer::kvcfg::name_of(p->cold));
    ok += f->cold == ninfer::kvcfg::KvFormat::Iso3;
    ok += p->cold == ninfer::kvcfg::KvFormat::Int8;
    std::printf("== %d/%d\n", ok, total + 2);
    return ok == total + 2 ? 0 : 1;
}
