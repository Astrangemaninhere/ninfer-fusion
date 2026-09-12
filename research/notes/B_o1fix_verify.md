# B_o1fix_verify.md — round 6 (S15): O1 修复 (config.h 门禁) 的对抗性复验

日期 2026-09-09 · 验证者 B · 对象: `ninfer-fusion-repo\tools\archkit\adapt.py` 三处改动
(1) emit_config_header 未知层型→'Unknown', gdn_layers() 只数显式 linear/gdn (行 186–199);
(2) main() 门禁: 任一 gap tier==new_op → 写 `config.h.BLOCKED` 并 unlink 旧 `config.h` (行 311–318);
(3) `#error` 兜底 (行 237–240), 命名具体缺口。

**总裁决: PASS — 4/4 指定检查通过, 0 新发现泄漏。附 1 条非阻塞观察 (O2: 清洁路径不删旧 .BLOCKED)。**
作者三条自跑未复跑; 用等价合成路径独立确认 (b-neg-a=lfm2 路径, b-neg-b=falcon 路径, MiniCPM5=dense)。
方法: 复用 round-5 合成 config (MiniCPM5 形状, 3 层): neg_a=`["conv","conv","full_attention"]`,
neg_b=无 layer_types+4 个 mamba_* 键, ctrl_2=全 full_attention, flip/flip_dense=同 id 先堵后清。
pre-round-6 基线 = `sed -f old.sed` 回退三处改动 (187d / 193,199c / 202,209d / 237,240d / 314d /
315,318c / 326,328d), py_compile 通过后运行。

## (a) 删除测试 → PASS
```
echo '#pragma once  // FAKE STALE HEADER planted by B verify' > out/b-neg-a/config.h
echo 'struct TextConfig { static constexpr int gdn_layers() { return 2; } };' >> out/b-neg-a/config.h
md5sum out/b-neg-a/config.h.BLOCKED > md5_blocked_before.txt
python3 adapt.py <_b_tmp>/neg_a --model-id b-neg-a
---LS-AFTER---  →  config.h.BLOCKED  engine_hook.patch  manifest.json
---TEST-CONFIGH---  →  CONFIG_H_GONE
md5sum -c md5_blocked_before.txt  →  out/b-neg-a/config.h.BLOCKED: OK   (.BLOCKED 未变)
```
手植的可编译假 config.h 被删, 无可用 config.h 残留。

## (b) dense 逐字节回归 → PASS
```
sed -f old.sed adapt.py > adapt_old_BVER.py && python3 -m py_compile adapt_old_BVER.py  → OLDCOPY_COMPILES
diff ctrl2_new.h  ctrl2_old.h   → 无输出, CTRL2_CONFIGH_IDENTICAL
diff ctrl2_new_manifest.json ctrl2_old_manifest.json → 无输出, CTRL2_MANIFEST_IDENTICAL
diff mp5_new.h    mp5_old.h     → 无输出, MP5_CONFIGH_IDENTICAL
```
(ctrl_2 与 MiniCPM5-1B 均以同 model-id 先新后旧各跑一次再 diff; 结束前用现版重跑 MiniCPM5 留盘。)
dense 路径 config.h 与 manifest 与改前逐字节一致 — 门禁对干净模型零扰动。

## (c) 其他工件泄漏检查 → PASS (无新发现)
```
grep -rn Gdn out/b-neg-a/  → NO_GDN_IN_NEGA ;  out/b-neg-b/ → NO_GDN_IN_NEGB
ls out/b-neg-a/ | grep -c 'leaves\|bindings'  → 0      (leaves/bindings.stub 本版根本不产出)
cat out/b-neg-a/engine_hook.patch  →  仅一行头 '--- engine hook patch (auto) ---'
manifest.json: "layer_kind_order": ["conv","conv","full_attention"]   (原始保留, 未折叠)
```
manifest 中唯一含 'gdn' 的行 (行 75) 是 semantics_v5 的通用 guidance 文案
`"识别时: 层层按 layer_types(full/swa/gdn)..."` — 列举已知词表, 不是把 conv 折成 Gdn。

## (d) conv 注入 gdn_layers() → PASS
`out/b-neg-a/config.h.BLOCKED` 第 21 行: `static constexpr int gdn_layers() { return 0; }`
(round-5 同输入为 2); 且第 3 行:
`#error "auto-adapt: unmodelled layer kinds (conv) have no engine leaf / unresolved new_op gaps (new_op:layer_kinds(conv x2))"`
与作者宣称的 lfm2 格式逐字同构。b-neg-b 独立复现 falcon 路径:
`#error "auto-adapt: unresolved new_op gaps (new_op:state_space(mamba_d_state,mamba_n_heads,mamba_d_conv,mamba_expand))"`
dense ctrl_2 / MiniCPM5 的 config.h 无 `#error`、文件名即 config.h。stdout 门禁行:
`[blocked] config.h withheld as config.h.BLOCKED (unresolved: ...)`。

## 额外对抗 (非指定项)
- **O2 (观察, 非阻塞)**: 同 model-id 先 blocked 后 clean (b-flip: conv config → dense config 重跑):
  out/b-flip 同时存在 `config.h` 与**旧 `config.h.BLOCKED`** — 清洁路径不回收旧 .BLOCKED。
  安全性无影响 (.BLOCKED 自带 #error, 永不可静默编译), 但陈旧 .BLOCKED 会误导审计。
  建议: else 分支同样 `(out_dir/'config.h.BLOCKED').unlink(missing_ok=True)`。
- 子串不对称 (核查过, 安全): catalog 用 7 个已知 kind 精确匹配, emitter 用子串 ('full'/'sliding'/
  'linear'/'gdn')。如 'gdn_like' 会被 emitter 归 Gdn 但 catalog 判 unknown → blocked → #error 在场,
  文件不可编译, 两个方向都关死。
- neg_b 的 .BLOCKED 里 full_attention_layers()==3 (空 kinds 的全 Full 回退) — 即注释所称 "lie",
  但 #error 同文件点名 state_space 缺口, 自描述成立。

## 清理
`_collab/_b_tmp/`、`out/b-neg-a b-neg-b b-ctrl-2 b-flip`、`specs/b-*_spec.json`、
`adapt_old_BVER.py(+pyc)` 已删; `out/minicpm5-1b` 以现版输出留存。未构建、未碰 GPU。
(控制台 `cat` 中文会 GBK 乱码; 报告引用自文件直读, 均为正确 UTF-8。)
