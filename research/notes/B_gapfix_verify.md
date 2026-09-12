# B_gapfix_verify.md — round 5 (S13): 对抗性验证 adapt.py 缺口报告修复

日期 2026-09-09 · 验证者 B · 对象: `ninfer-fusion-repo\tools\archkit\adapt.py`
修复点: `catalog_gaps` 新增 (a) 未归类层型 → `new_op:layer_kinds(<kind> xN)` (行 103–113);
(b) `hybrid_hint` SSM/Mamba 键 且 layer_types 缺失/为空 → `new_op:state_space(...)` (行 114–120,
hybrid_hint 采集在 `extract_spec` 行 60–63)。
原则 (_AUTOADAPT.md S4): new_op → 精确失败, 不假装成功。

**总裁决: PASS (8/8 检查通过)。未发现假阴性或假阳性。附 2 条边界观察 (非本修复缺陷, 见 §观察)。**

方法: 全部用自制最小 config (镜像 `models/MiniCPM5-1B/config.json` 形状, num_hidden_layers=3,
几何字段原样保留, 仅注入待测键), 不重跑作者的 LFM2/Falcon 两条命令。日志经 WSL 重定向到文件后按
UTF-8 读取 (控制台里 wsl.exe 横幅是 GBK 乱码, 文件内容干净, 无需 iconv)。

## 原始命令 (单行, 无 $() )

```
wsl.exe -e bash -c "cd /mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit && python3 adapt.py /mnt/c/Users/User/Documents/ziqinzhang/_collab/_b_tmp/neg_a --model-id b-neg-a > /mnt/c/Users/User/Documents/ziqinzhang/_collab/_b_tmp/log_neg_a.txt 2>&1; ..."
```
(7 个变体逐条同型: neg_a neg_a2 neg_b neg_b2 neg_b3 ctrl_2 ctrl_3; 迁移完已清理)

回归 (pre-fix 模拟 = `sed '108,113d;117,120d' adapt.py > adapt_prefix_BVER.py` 删掉两个新增块,
py_compile 通过后运行):
```
wsl.exe -e bash -c "cd .../tools/archkit && sed '108,113d;117,120d' adapt.py > adapt_prefix_BVER.py && python3 -m py_compile adapt_prefix_BVER.py && python3 adapt_prefix_BVER.py <models>/MiniCPM5-1B --model-id minicpm5-1b > log_reg_pre.txt 2>&1; cp out/minicpm5-1b/manifest.json manifest_reg_pre.json; python3 adapt.py <models>/MiniCPM5-1B --model-id minicpm5-1b > log_reg_post.txt 2>&1; cp out/minicpm5-1b/manifest.json manifest_reg_post.json; diff manifest_reg_pre.json manifest_reg_post.json && echo MANIFEST_IDENTICAL; grep -c 'new_op:layer_kinds\|new_op:state_space' log_reg_post.txt"
```

## 检查与原始输出

### CHK-1a NEG 注入: layer_types=["conv","conv","full_attention"] → PASS
```
adapted b-neg-a -> .../out/b-neg-a
  [covered] attention:gqa_full                             v3 gen
  [new_op] new_op:layer_kinds(conv x2)                    未归类层型: 短卷积可复用 causal_conv1d_silu 叶子, Mamba/SSD 类需新算子
  [covered] head:tied=false                                独立 lm_head 对象
  [hook] token_domain:vocab=130560!=family 248077       FrontendOptions.token_domain + official_specials 覆盖
  [post] quant_geometry                                 转换后校验: fp8/nvfp4 形状对照几何注册表 (A16 起步)
```
精确命中 `conv x2`。

### CHK-1b NEG 注入: 两个未归类层型 ["conv","mamba","full_attention"] → PASS
```
  [new_op] new_op:layer_kinds(conv x1,mamba x1)           未归类层型: ...
```
多键按字典序排序、x1 计数格式正确。

### CHK-2a NEG 注入: 无 layer_types + 4 个 SSM 键 → PASS
config 增 `mamba_d_state/mamba_n_heads/mamba_d_conv/mamba_expand`:
```
  [new_op] new_op:state_space(mamba_d_state,mamba_n_heads,mamba_d_conv,mamba_expand) 检测到 SSM/Mamba 配置键但无 layer_types: 层混合未建模, 需按 hybrid pattern 展开
```
键序 = 代码元组序 (非字典序), 与 Falcon-H1R 证据一致。

### CHK-2b NEG 注入: layer_types=[] (空列表) + ssm_cfg={} → PASS ("空"半边)
```
  [new_op] new_op:state_space(ssm_cfg)                    检测到 SSM/Mamba 配置键但无 layer_types: ...
```
空列表同样触发, 覆盖声明中 "EMPTY or missing" 的 EMPTY 分支。

### CHK-2c NEG 边界: SSM 键为 null (`"ssm_cfg": null, "mamba_d_state": null`) → PASS (按设计静默)
输出无 state_space 行 (仅 tied/token_domain/quant_geometry 三行)。`tc[k] is not None` 守卫把
null 视为"该架构无此能力"(HF 惯例), 不算假阴性。

### CHK-3a CTRL: MiniCPM5-1B 原样 (无 layer_types, 无 SSM 键) → PASS
```
adapted minicpm5-1b -> .../out/minicpm5-1b
  [covered] head:tied=false                                独立 lm_head 对象
  [hook] token_domain:vocab=130560!=family 248077       FrontendOptions.token_domain + official_specials 覆盖
  [hook] layers:24>16                                   cold_slots 类 per-layer 数组容量核对 (家族已扩 64, 新家族须审计)
  [post] quant_geometry                                 转换后校验: fp8/nvfp4 形状对照几何注册表 (A16 起步)
```
`grep -c 'new_op:layer_kinds\|new_op:state_space'` = **0**。无误报。

### CHK-3b CTRL: 显式 layer_types=["full_attention"×3], 无 SSM 键 → PASS
输出无 layer_kinds / state_space 行 (gqa_full covered 正常出现)。

### CHK-4 REG 回归: pre-fix vs post-fix (MiniCPM5-1B) → PASS
`diff manifest_reg_pre.json manifest_reg_post.json` → 无输出 + `MANIFEST_IDENTICAL`
(manifest 全文含 gaps 与 spec 字节级一致)。干净模型缺口清单零变化。

## 观察 (非 FAIL, 供作者/coordinator 参考)

- **O1 (潜在隐患, 建议跟进)**: `main()` 无条件写 `out/<id>/config.h`, 即使存在 new_op 缺口。
  `emit_config_header` (行 184–237) 的 kind_map 把不含 'full'/'sliding' 的层型一律归 **Gdn**:
  b-neg-a 的 config.h 实测 `gdn_layers() { return 2; }` — 两个 conv 层被 S4 工件静默计为 Gdn。
  管线按 _AUTOADAPT 应在 S3 (new_op) 停, 不消费该工件; 但工件已落盘, 若下游忽略 manifest.gaps
  直接吃 config.h, "不假装成功"会被绕过。建议: 存在 new_op 时跳过工件生成或打标。
- **O2 (边界)**: layer_types 已声明全 full_attention 且另有 SSM 键 (ctrl_3: +mamba_d_state) →
  state_space 按 `not kinds` 守卫保持静默。符合本修复的规格 (层混合已建模), 但此类矛盾 config
  会漏报 SSM 能力; 真实世界未见此形态, 记录备查。
- 附注: spec 直通路径 (adapt.py 行 280–284) 不重算 hybrid_hint; 手工旧 spec 无该键会静默跳过
  state_space。manifest 序列化的 spec 自带 hybrid_hint, 正常回灌不受影响。

## 清理
`_collab/_b_tmp/`、`out/b-*`、`specs/b-*_spec.json`、`adapt_prefix_BVER.py(+pyc)` 已删除;
`out/minicpm5-1b` 为真实模型产物, 保留 (内容与 09:32 版一致, CHK-4 已证等价)。
