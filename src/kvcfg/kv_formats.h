#pragma once
// kv_formats.h — KV 分层精度自由组合的引擎侧契约 (C++ 孪生, 与
// tools/gui/kv_tiers.py 同语法同规则; 改动必须两边同步 + 跑对拍测试)。
//
//   语法:  --kv-tier-formats hot=bf16,tail=fp16,cold=iso3
//   模式:  nvfp4-mode fusion|pure  (pure 只允许经典格式, 对照归因用)
//   规则:  热层 >= int8; 尾层 >= 热层; pure 禁 iso/e8 系
//   语义:  热=活动页, 尾=近期高精度窗 (kv-tail-tokens), 冷=老化出窗
//          (cold-keep-tokens / 冷池), 组合不固化。
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::kvcfg {

enum class KvFormat : std::uint8_t {
    Auto = 0,
    Bf16,
    Fp16,
    Int8,
    Int4,
    Iso4,   // iso 4bit: 组内 16 元素共享 fp16 scale, 每元素 4bit 有符号码
    Iso3,   // iso 3bit: 符号+2bit 幅值, 8 元素/3 字节打包
    E8,     // E8 格点 4bit (fusion; 3/2bit 变体后续)
};

enum class Nvfp4Mode : std::uint8_t { Fusion = 0, Pure };

struct KvTierFormats {
    KvFormat hot  = KvFormat::Auto;   // 活动页 (默认随权重档 KV dtype)
    KvFormat tail = KvFormat::Auto;   // 近期高精度尾窗 (默认同 hot)
    KvFormat cold = KvFormat::Iso3;   // 老化出窗 (fusion 默认 iso3)
    Nvfp4Mode mode = Nvfp4Mode::Fusion;
};

inline constexpr int bits_of(KvFormat f) noexcept {
    switch (f) {
        case KvFormat::Bf16: return 16;
        case KvFormat::Fp16: return 16;
        case KvFormat::Int8: return 8;
        case KvFormat::Int4: return 4;
        case KvFormat::Iso4: return 4;
        case KvFormat::Iso3: return 3;
        case KvFormat::E8:   return 4;
        case KvFormat::Auto: return 16;   // 语义: 跟随默认 (按 16 位比较)
    }
    return 16;
}

inline const char* name_of(KvFormat f) noexcept {
    switch (f) {
        case KvFormat::Bf16: return "bf16";
        case KvFormat::Fp16: return "fp16";
        case KvFormat::Int8: return "int8";
        case KvFormat::Int4: return "int4";
        case KvFormat::Iso4: return "iso4";
        case KvFormat::Iso3: return "iso3";
        case KvFormat::E8:   return "e8";
        case KvFormat::Auto: return "auto";
    }
    return "?";
}

inline std::optional<KvFormat> format_from_name(std::string_view s) {
    if (s == "auto") return KvFormat::Auto;
    if (s == "bf16") return KvFormat::Bf16;
    if (s == "fp16") return KvFormat::Fp16;
    if (s == "int8") return KvFormat::Int8;
    if (s == "int4") return KvFormat::Int4;
    if (s == "iso4") return KvFormat::Iso4;
    if (s == "iso3") return KvFormat::Iso3;
    if (s == "e8")   return KvFormat::E8;
    return std::nullopt;
}

// 解析 "hot=bf16,tail=fp16,cold=iso3" (可省略层); 空串 = 全默认。
// 失败返回错误文案 (err), 成功返回配置。与 kv_tiers.py::parse 同规则。
inline std::optional<KvTierFormats> parse_tier_formats(std::string_view spec,
                                                       Nvfp4Mode mode,
                                                       std::string* err = nullptr) {
    const auto fail = [&](const char* why) -> std::optional<KvTierFormats> {
        if (err != nullptr) { *err = why; }
        return std::nullopt;
    };
    KvTierFormats out;
    out.mode = mode;
    if (spec.empty()) {
        if (mode == Nvfp4Mode::Pure) { out.cold = KvFormat::Int8; }
        return out;
    }
    std::size_t pos = 0;
    while (pos <= spec.size()) {
        const std::size_t comma = spec.find(',', pos);
        const std::string_view item =
            spec.substr(pos, comma == std::string_view::npos ? spec.size() - pos
                                                             : comma - pos);
        pos = comma == std::string_view::npos ? spec.size() + 1 : comma + 1;
        if (item.empty()) { continue; }
        const std::size_t eq = item.find('=');
        if (eq == std::string_view::npos) {
            return fail("bad --kv-tier-formats item (want tier=format)");
        }
        const std::string_view tier = item.substr(0, eq);
        const std::string_view fmt_s = item.substr(eq + 1);
        const auto fmt = format_from_name(fmt_s);
        if (!fmt.has_value()) { return fail("unknown kv format"); }
        const bool pure = mode == Nvfp4Mode::Pure;
        if (pure && (*fmt == KvFormat::Iso4 || *fmt == KvFormat::Iso3 ||
                     *fmt == KvFormat::E8)) {
            return fail("nvfp4-mode=pure forbids iso/e8 formats (fusion-only)");
        }
        if (tier == "hot") {
            if (*fmt == KvFormat::Int4 || *fmt == KvFormat::Iso4 ||
                *fmt == KvFormat::Iso3 || *fmt == KvFormat::E8) {
                return fail("hot tier cannot use sub-int8 formats (decode-hot)");
            }
            out.hot = *fmt;
        } else if (tier == "tail") {
            if (*fmt != KvFormat::Auto && bits_of(*fmt) < bits_of(out.hot)) {
                return fail("tail tier precision below hot tier");
            }
            out.tail = *fmt;
        } else if (tier == "cold") {
            out.cold = *fmt;
        } else {
            return fail("unknown tier (want hot|tail|cold)");
        }
    }
    return out;
}

} // namespace ninfer::kvcfg
