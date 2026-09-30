// ===========================================================================================
// kvfix (F893) -- THIS FILE IS A STALE MIRROR, AND IT IS TRACKED.  READ THE LIVE ONE.
// ===========================================================================================
// `$D/` was created by an unexpanded shell variable (the literal name `$D`) and it is IN THE
// GIT INDEX -- `git ls-files -- '$D'` lists these three files.  They are copies of the live KV
// surfaces taken 2026-09-20 and NEVER UPDATED, so they still carry the PRE-repair spellings:
//   * `$D/kv_bit_budget.h` keeps the `--kv-bit-budget 0-7:8,8-63:4.5` example, which the engine
//     refuses by name ("ranges must tile layers 0..15 in order"); the LIVE file at
//     src/product/kv_bit_budget.h:1163 carries the corrected, layer-count-agnostic
//     `0-7:8,8-15:4.5`.
//   * `$D/kv_formats.h` keeps the `--kv-tier-formats hot=bf16,tail=fp16,cold=iso4e` example,
//     which the engine refuses by name ("this engine has no tail tier"); the LIVE file at
//     src/kvcfg/kv_formats.h:5 carries `hot=bf16,cold=iso4e` and names the gap in place.
// MEASURED CONSEQUENCE: a whole-tree grep for either spelling still finds it HERE, so a reader
// concludes the tree ships an example the engine refuses.  Two lines of this record did exactly
// that (dl/namedmech's F-855 triage B1/E3, which counted "NINE places"; dl/kvadv F-892 carried
// it forward).  The live count is ZERO.
// THE FIX, AND WHY IT IS ONLY THIS BANNER: the tree's own convention is set out at
// src/kvcfg/kv_formats.h:38 -- "就地删名会把这个缺口藏起来, 故此处只标注事实" (deleting the name in
// place would HIDE the gap, so only the fact is annotated here).  Nothing is deleted, renamed or
// rewritten; the mirror is named at the top of its own files so the next grep's reader sees it.
// The live file for this mirror is: src/kvcfg/kv_formats.h
// ===========================================================================================

#pragma once
// kv_formats.h — KV 分层精度自由组合的引擎侧契约 (C++ 孪生, 与
// tools/gui/kv_tiers.py 同语法同规则; 改动必须两边同步 + 跑对拍测试)。
//
//   语法:  --kv-tier-formats hot=bf16,tail=fp16,cold=iso4e
//   模式:  nvfp4-mode fusion|pure  (pure 只允许经典格式, 对照归因用)
//   规则:  热层 >= int8; 尾层 >= 热层; pure 禁 iso/rk4v4 系
//   语义:  热=活动页, 尾=近期高精度窗 (kv-tail-tokens), 冷=老化出窗
//          (cold-keep-tokens / 冷池), 组合不固化。
//
//   ⚠ MEASURED 2026-09-20 -- 上面括号里的两个窗口旋钮, 只有一个是活的:
//     `--cold-keep-tokens` 有 parser 臂 (apps/cli/options.cpp) 也有 usage 行。
//     `--kv-tail-tokens` 是**悬空的名字**: apps/cli/options.cpp 里 0 命中,
//     `NINFER_KV_TAIL_TOKENS` 全树 0 命中, 而且**没有东西可供它定尺寸** —— 尾层只在
//     与热层同值时被接受, usage 文本原话是 the engine has no recent-window tier yet。
//     所以本行括号里的那个名字是**引用**, 不是活契约, 与下面 Iso4e 枚举上被就地标注的
//     `gqa_iso4e_codec.cuh` 路径同一性质。同名的第二处引用在 tools/gui/kv_tiers.py:14。
//     就地删名会把这个缺口藏起来, 故此处只标注事实; 这条标注本身就是"实现它, 或者两处
//     一起退役"的提示。
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
    // iso4e: 引擎实际落地的编解码是 **4bit 符号-幅值 nibble**（bit3 符号 +
    // 幅值 0..7，组 16，E4M3 scale = amax/7，两码/字节）= 即 ISO4 的量化器换
    // 了码字映射；见 src/ops/kernel/gqa_iso3_codec.cuh 与
    // /home/user/scratch/iso4e/report.md。（原文此处引 gqa_iso4e_codec.cuh：
    // 该路径在本树中不存在 —— `gqa_iso4e_*` 是尚未落地的目标名，实际文件是
    // ops/kernel/gqa_iso3_codec.cuh。）kv_iso_ref.py/iso_codec.h 里那个
    // "符号+2bit 幅值、8 元素/3 字节" 的 3bit ISO4E **没有实现**，故 bits_of
    // 按实际描述 4（码宽）——原来的 3 是拿参考契约当实现，会误导 --kv-tier-formats
    // 的精度排序判断。
    Iso4e,
    Rk4v4,     // Rk4v4 格点 4bit (fusion; 3/2bit 变体后续)
    // rk4v4 族的 **K 平面码宽** 变体 (原注释里的 "3/2bit 变体")。这不是"更小的 rk4v4 码本":
    // 我们的 rk4v4 是「每元素一个有符号整数码 + 每 64 通道 fp16 scale 平面」，这两档就是
    // 同一套平面把 K 码位压到 3bit / 2bit（V 仍是 i4，不动 —— 生态也只压 K）。
    // 几何与成本由 product/kv_e8_width.h 单点推导并被 static_assert 钉住:
    //   K 平面 8704 -> 6656 -> 4608 B/head-page；层成本 (K+V 平均) 4.25 -> 3.75 -> 3.25。
    // 该头文件里的 w=4 断言必须复现出厂行 425/8704，这是"推导 == 出厂"的检验。
    // Rk3v4 无生态先例 (BenWu/ninfer 只发了 rk4v4-e8 与 rk2v4-e8；52 个 fork 的清单里
    // 没有任何 rk3 / 3-bit token)。
    // Rk2v4 的 **几何** 与生态 rk2v4-e8 的 K 平面逐字节相同，**编解码不是**：rk2v4-e8 的
    // K 是每 8 维 2 字节的 Rk4v4 根圆柱码 (240 根索引 + 半径)，本行是共享 scale 的逐元素码。
    // 两档的「格点地板」结论在 2026-09-18 被重建（不是删除）：src/ops/kernel/e8_lattice.cuh:125-137
    // 实测的 +1 bit/el 需求约束的是「每坐标一个整数码」的那个消费者，对这个消费者结论不变
    // —— 但同一个注释的第二个出口（"where the consumer reconstructs the lattice point
    // rather than an integer code"）已经落地为 src/ops/kv/e8_lattice_plane_codec.cuh：
    // 它的码字是 E8 ± 1/4 的 8 维**格点**，2bit 16 位 / 3bit 24 位，正好是 8 元素 2 / 3 字节，
    // 与本行的平面几何逐字节相同。故 e8_kv_plane_codec_of_record() 对 Rk3v4 / Rk2v4 返回
    // Lattice —— 这两行现在是**真 E8**，不是「同一平面形状里的更窄标量码」。
    // 标量 codec 保留为 W4 的正式 codec 与这两档的对照臂（同一比特、同一 side information）。
    // Rk4v4 仍是标量行：格点码字没有 32 位形式，所以 4bit 上放不下真格点。
    Rk3v4,   // Rk4v4 格点 K 码 3bit (无生态先例; 几何 6656 B; 层 375)
    Rk2v4,   // Rk4v4 格点 K 码 2bit (几何 == rk2v4-e8 的 K 平面 4608 B; 层 325)
};

enum class Nvfp4Mode : std::uint8_t { Fusion = 0, Pure };

struct KvTierFormats {
    KvFormat hot  = KvFormat::Auto;   // 活动页 (默认随权重档 KV dtype)
    KvFormat tail = KvFormat::Auto;   // 近期高精度尾窗 (默认同 hot)
    KvFormat cold = KvFormat::Iso4e;   // 老化出窗 (fusion 默认 iso4e)
    Nvfp4Mode mode = Nvfp4Mode::Fusion;
};

inline constexpr int bits_of(KvFormat f) noexcept {
    switch (f) {
        case KvFormat::Bf16: return 16;
        case KvFormat::Fp16: return 16;
        case KvFormat::Int8: return 8;
        case KvFormat::Int4: return 4;
        case KvFormat::Iso4: return 4;
        // 见上面 Iso4e 的说明: 落地实现是 4bit 码, 不是参考契约的 3bit。
        // (排序守卫的效果: hot=int8,tail=iso4e 仍然拒绝 (4<8); 只有
        //  hot=nvfp4,tail=iso4e 这类请求从"被 bits_of 拒绝"变成"放行到
        //  tail==hot 的第二道规则", 该规则本来也会拒。)
        case KvFormat::Iso4e: return 4;
        case KvFormat::Rk4v4:   return 4;
        // rk4v4 族的 3bit/2bit 档: 宽度换来的成本由平面几何推导（K+V 平均 = 50*w + 225，
        // w=4 时正好复现上面的 425）。见 product/kv_e8_width.h。
        case KvFormat::Rk3v4: return 3;
        case KvFormat::Rk2v4: return 2;
        case KvFormat::Auto: return 16;   // 语义: 跟随默认 (按 16 位比较)
    }
    return 16;
}

// constexpr so the spelling table below can be asserted at COMPILE time rather
// than described in a comment. `constexpr` implies `inline`, so this is a
// qualification change only: no caller, no emitted name and no ODR fact moves.
inline constexpr const char* name_of(KvFormat f) noexcept {
    switch (f) {
        case KvFormat::Bf16: return "bf16";
        case KvFormat::Fp16: return "fp16";
        case KvFormat::Int8: return "int8";
        case KvFormat::Int4: return "int4";
        case KvFormat::Iso4: return "iso4";
        case KvFormat::Iso4e: return "iso4e";
        case KvFormat::Rk4v4:   return "rk4v4";
        // rk<N>v4 -- rotated K at N bits, V at 4 -- which is ALSO what the ecosystem calls
        // these rows (rk4v4-e8 / rk2v4-e8). The older comment here said e8k3/e8k2 avoided
        // confusion with the ecosystem's K-bits x V-bits rotation axis. That was wrong:
        // our V is i4 on every row, so our K width IS the ecosystem's N and the two
        // spellings are two names for ONE axis on the K plane, not two axes
        // (product/kv_e8_width.h:21-28, with the arithmetic at :35-41 that
        // product/kv_tier_formats.h:486 re-states as a static_assert). What is genuinely
        // different is the CODEC: those rows are per-element codes over a shared scale
        // plane, while rk<N>v4-e8's K is a 240-root E8 root-cylinder codebook
        // (product/kv_e8_width.h:42-46). A row that is not the lattice must not be named
        // for it -- which is why the tier vacates `e8` entirely.
        // W3 does not divide a byte: 8 elements pack into 24 bits (3 bytes), the tree's own
        // documented-but-unimplemented ISO3 reference layout (kv_formats.h:27-29).
        case KvFormat::Rk3v4: return "rk3v4";
        case KvFormat::Rk2v4: return "rk2v4";
        case KvFormat::Auto: return "auto";
    }
    return "?";
}

// constexpr for the same reason as name_of above: an alias that is only
// described cannot fail a build, and this one has been misread before.
inline constexpr std::optional<KvFormat> format_from_name(std::string_view s) {
    if (s == "auto") return KvFormat::Auto;
    if (s == "bf16") return KvFormat::Bf16;
    if (s == "fp16") return KvFormat::Fp16;
    if (s == "int8") return KvFormat::Int8;
    if (s == "int4") return KvFormat::Int4;
    if (s == "iso4") return KvFormat::Iso4;
    if (s == "iso4e") return KvFormat::Iso4e;
    if (s == "rk4v4")   return KvFormat::Rk4v4;
    if (s == "rk3v4") return KvFormat::Rk3v4;
    if (s == "rk2v4") return KvFormat::Rk2v4;
    // Deprecated spellings, accepted for one release. See
    // product/kv_bit_budget.h detail::canonical_tier_name for the full rationale: every
    // EMITTED name is canonical (name_of()), these four exist only so that the deferred
    // front ends (apps/cli/*) and existing command lines keep parsing.
    if (s == "iso3") return KvFormat::Iso4e;
    if (s == "e8")   return KvFormat::Rk4v4;
    if (s == "e8k3") return KvFormat::Rk3v4;
    if (s == "e8k2") return KvFormat::Rk2v4;
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// THE SPELLINGS ARE PINNED HERE, BECAUSE THEY HAVE BEEN READ AS EACH OTHER.
// ---------------------------------------------------------------------------
// `iso3` and `iso4` are NOT two spellings of one row. Both statements below were
// only comments before this block, and a reader who takes one row for the other
// was correctable by nothing in the tree:
//
//   `iso3` is a DEPRECATED SPELLING of `Iso4e`. It was renamed because the row was
//   never 3 bits: the landed codec is a 4-bit sign-magnitude nibble over per-16
//   E4M3FN scales -- i.e. 4.00 + 8/16 = 4.50 b/el == 9216 B/head-page. The same
//   rename is implemented a second time in product/kv_bit_budget.h:572-578
//   (detail::canonical_tier_name, whose comment says it in the same words: "the
//   shipped iso row was called `iso3` (never 3 bits)") and a third time in
//   src/serve/serve_options.cpp:403-409, which prints a deprecation notice.
//
//   `iso4` is a DIFFERENT row -- `KvFormat::Iso4`, the reference contract's 4-bit
//   ISO code over a per-16 FP16 scale (see the comment on its enumerator). It is a
//   real vocabulary row with a real bits_of; what it has is NO resident engine
//   codec. product/kv_tier_formats.h:1147-1152 already names it in a refusal as one
//   of the "16-bit or i4/iso4/rk4v4" names that "has no slot codec and no decoder
//   branch", and KvCacheStorage carries no enumerator for it. See
//   has_resident_codec() below.
//
// Nothing tied the three implementations together, so this block pins THIS file's
// copy and adds the one fact the other two cannot state from where they sit: that
// the deprecated spelling does NOT resolve to the row it is named almost like.
static_assert(format_from_name("iso3") == KvFormat::Iso4e,
              "the deprecated spelling `iso3` must keep resolving to iso4e (= the tier the "
              "engine prints as iso4e-g16, KvCacheStorage::Iso3Group16, DType::ISO3). It never "
              "meant 3 bits and it must never be re-pointed at KvFormat::Iso4, which is a "
              "DIFFERENT row with no resident codec.");
static_assert(format_from_name("e8") == KvFormat::Rk4v4,
              "the deprecated spelling `e8` resolves to rk4v4. It was never the E8 lattice: it "
              "is one signed integer code per element over a shared scale plane.");
static_assert(format_from_name("e8k3") == KvFormat::Rk3v4 &&
                  format_from_name("e8k2") == KvFormat::Rk2v4,
              "`e8k3`/`e8k2` resolve to rk3v4/rk2v4");
static_assert(*format_from_name("iso4") == KvFormat::Iso4,
              "the canonical spelling `iso4` must resolve to KvFormat::Iso4 -- the row "
              "distinct from iso4e, NOT an alias of it");
// The distinction itself, stated as a value comparison so that collapsing the two
// rows (the exact edit a future reader might make "to simplify the alias table")
// fails the build instead of silently re-pointing a deprecated name.
static_assert(*format_from_name("iso4") != *format_from_name("iso3"),
              "`iso4` and `iso3` are two DIFFERENT rows: iso4 is KvFormat::Iso4 (fp16-scale "
              "reference contract, no engine codec) and iso3 is the deprecated spelling of "
              "KvFormat::Iso4e (E4M3-scale landed codec). Merging them is not a rename.");
// Every EMITTED name is canonical and every canonical name parses back. The four
// deprecated spellings are therefore unreachable through name_of(), which is what
// makes "accepted for one release" a direction rather than a two-way alias.
static_assert(name_of(KvFormat::Iso4e) == std::string_view{"iso4e"} &&
                  name_of(KvFormat::Iso4) == std::string_view{"iso4"},
              "iso4e/iso4 must EMIT their own canonical names");
static_assert(name_of(KvFormat::Rk4v4) == std::string_view{"rk4v4"} &&
                  name_of(KvFormat::Rk3v4) == std::string_view{"rk3v4"} &&
                  name_of(KvFormat::Rk2v4) == std::string_view{"rk2v4"},
              "the rk4v4 family must EMIT rk4v4/rk3v4/rk2v4, never e8/e8k3/e8k2");
static_assert(*format_from_name(name_of(KvFormat::Iso4e)) == KvFormat::Iso4e &&
                  *format_from_name(name_of(KvFormat::Rk2v4)) == KvFormat::Rk2v4,
              "a canonical name must parse back to the row that emitted it");

// ---------------------------------------------------------------------------
// WHICH OF THESE NAMES THE ENGINE CAN ACTUALLY BUILD -- AND THE TWO IT CANNOT NAME.
// ---------------------------------------------------------------------------
// Two questions with two different answers, because the tree answers them
// differently and both answers are load-bearing.
//
//   has_resident_codec -- is there a KvCacheStorage + DType pair for this row?
//     YES: bf16, int8, iso4e, rk4v4, rk3v4, rk2v4
//          (product/kv_storage_dtype.h:44-63 maps them; :210-211 states the set
//          from the other side: "A per-layer slot must be one of bf16, int8, fp8,
//          nvfp4, iso4e, rk4v4, rk3v4, rk2v4").
//     NO : Auto -- it means "inherit the global --kv-dtype", it is not a format;
//          fp16, int4 and **iso4** -- no KvCacheStorage enumerator, no DType.
//          product/kv_tier_formats.h:1052-1056 states the same set from the hot
//          side: "this engine has no FP16 KV tier ... KvCacheStorage and DType
//          carry bf16, int8, fp8, nvfp4, iso4e and rk4v4 KV codecs only".
//
//   is_readable_codec -- has_resident_codec AND a decoder branch. rk3v4 and rk2v4
//     pass the first test and fail this one: product/kv_storage_dtype.h:106-122
//     refuses them BY NAME because the reader that decodes a 3-bit/2-bit K plate
//     "has NO CALLER outside its own definition and the forced-instantiation TU",
//     so the tier is WRITABLE AND NOT READABLE. Its refusal text closes with the
//     deployable set: "Use rk4v4, int8, nvfp4 or iso4e."
//
// ⭐⭐ `nvfp4` AND `fp8` ARE NOT ROWS OF THIS ENUM, AND THAT IS THE FACT A READER
// MOST OFTEN GETS WRONG ABOUT IT. This enum is the --kv-tier-formats VOCABULARY,
// not the engine's KV codec set; the two overlap and neither contains the other:
//   * The engine deploys TWO tiers this vocabulary cannot even spell. `nvfp4` and
//     `fp8` live in a DIFFERENT parser -- product/kv_options.h:48-51, the
//     --kv-dtype / --kv-layer-storage token table ("nvfp4" -> Nvfp4Group16,
//     "fp8" -> Fp8Group16) -- and this file contains no `nvfp4` and no `fp8`
//     token at all. The engine's own help text admits the narrower vocabulary at
//     apps/cli/options.cpp:224: the documented hot set is "auto|bf16|int8".
//   * Conversely three spellable rows (fp16, int4, iso4) name no engine codec.
//   Neither direction is a defect -- one axis is the tier FORMAT, the other is the
//   resident DTYPE -- but a reader who takes this enum for "the tiers the engine
//   has" gets both halves wrong, which is why the two questions below are
//   predicates and not a paragraph.
//
// ⭐⭐⭐ AND THE BRIDGE BETWEEN THE TWO AXES IS THAT `iso4e` IS THIS VOCABULARY'S
// NAME FOR THE NVFP4 FAMILY -- not a coincidentally similar row beside it:
//   product/kv_tier_formats.h:227-231  kv_cold_format_of(KvLayerClass::Nvfp4Fusion)
//                                      == kvcfg::KvFormat::Iso4e
//   product/kv_tier_formats.h:706-712  kv_cold_codec_of_format(Iso4e)
//                                      == ColdCodec::Nvfp4Rans
//   product/kv_tier_formats.h:579      "iso4e == nvfp4 planes" (9216 B/head-page,
//                                      4.50 b/el, both shared with nvfp4)
// So a request written `cold=iso4e` is a request about the NVFP4 family's cold
// slot, and the pair "iso4e with nvfp4" is expressed in this vocabulary by the ROW
// IDENTITY Iso4e == the nvfp4 family -- there is no separate nvfp4 row for it to
// be fused with. The kernel-level fusion of the two codecs is a different layer
// again (one DType per layer, K and V planes on different codecs): it is
// decoder_state.cpp kv_layer_v_dtype() and product/kv_kv_bits.h:36-59.
[[nodiscard]] constexpr bool has_resident_codec(KvFormat f) noexcept {
    switch (f) {
    case KvFormat::Bf16:
    case KvFormat::Int8:
    case KvFormat::Iso4e:
    case KvFormat::Rk4v4:
    case KvFormat::Rk3v4:
    case KvFormat::Rk2v4:
        return true;
    case KvFormat::Auto:
    case KvFormat::Fp16:
    case KvFormat::Int4:
    case KvFormat::Iso4:
        return false;
    }
    return false;
}

[[nodiscard]] constexpr bool is_readable_codec(KvFormat f) noexcept {
    if (!has_resident_codec(f)) { return false; }
    return f != KvFormat::Rk3v4 && f != KvFormat::Rk2v4;
}

static_assert(!has_resident_codec(KvFormat::Auto) && !has_resident_codec(KvFormat::Fp16) &&
                  !has_resident_codec(KvFormat::Int4),
              "Auto means 'inherit', and fp16/int4 have no KvCacheStorage and no DType: none of "
              "the three is a resident KV codec");
// ⭐ THE ASSERT THAT WOULD HAVE CAUGHT THE PREDECESSOR'S ERROR. The deprecated
// spelling and the row it is named almost like sit on OPPOSITE SIDES of the codec
// line. If a future edit makes iso3 resolve to Iso4, this fires -- and so does the
// `iso4 != iso3` assert above.
static_assert(has_resident_codec(*format_from_name("iso3")) &&
                  !has_resident_codec(*format_from_name("iso4")),
              "iso3==iso4e HAS a resident codec (DType::ISO3); iso4 has NONE. The two rows must "
              "not be collapsed: one is deployable and the other is not.");
static_assert(is_readable_codec(KvFormat::Bf16) && is_readable_codec(KvFormat::Int8) &&
                  is_readable_codec(KvFormat::Iso4e) && is_readable_codec(KvFormat::Rk4v4),
              "the four VOCABULARY rows the engine can deploy: bf16, int8, iso4e, rk4v4");
// The two tiers the engine deploys and THIS ENUM CANNOT NAME. Asserted rather than
// left to the comment above, because the asymmetry is the thing a reader gets
// wrong in both directions: these two names do not parse here, they are not
// typos, and the parser that takes them is product/kv_options.h:48-51.
static_assert(!format_from_name("nvfp4").has_value() && !format_from_name("fp8").has_value(),
              "`nvfp4` and `fp8` are NOT rows of this vocabulary: they are --kv-dtype / "
              "--kv-layer-storage tokens (product/kv_options.h:48-51). `iso4e` is this "
              "vocabulary's handle on the nvfp4 family (kv_cold_format_of(Nvfp4Fusion) == "
              "Iso4e). If either spelling is ever ADDED here it needs its own codec mapping, so "
              "this assert firing is the prompt to write one, not to delete this assert.");
static_assert(has_resident_codec(*format_from_name("iso3")) &&
                  *format_from_name("iso3") == KvFormat::Iso4e,
              "the bridge, restated where the absence of `nvfp4` is argued: the deprecated "
              "spelling `iso3` reaches the nvfp4 family through Iso4e, which is the only row in "
              "this enum that names it");
static_assert(has_resident_codec(KvFormat::Rk3v4) && !is_readable_codec(KvFormat::Rk3v4) &&
                  has_resident_codec(KvFormat::Rk2v4) && !is_readable_codec(KvFormat::Rk2v4),
              "rk3v4/rk2v4 are WRITABLE AND NOT READABLE: their codec, K plate layout and "
              "append arm all exist, and no attention arm reads a 3-bit or 2-bit K plate");

// 解析 "hot=bf16,tail=fp16,cold=iso4e" (可省略层); 空串 = 全默认。
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
        if (pure && (*fmt == KvFormat::Iso4 || *fmt == KvFormat::Iso4e ||
                     *fmt == KvFormat::Rk4v4 || *fmt == KvFormat::Rk3v4 ||
                     *fmt == KvFormat::Rk2v4)) {
            return fail("nvfp4-mode=pure forbids iso/rk4v4 formats (fusion-only)");
        }
        if (tier == "hot") {
            if (*fmt == KvFormat::Int4 || *fmt == KvFormat::Iso4 ||
                *fmt == KvFormat::Iso4e || *fmt == KvFormat::Rk4v4 ||
                *fmt == KvFormat::Rk3v4 || *fmt == KvFormat::Rk2v4) {
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
