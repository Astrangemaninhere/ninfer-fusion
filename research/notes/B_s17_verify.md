# B_s17_verify.md — round 7 (S19): C 的 S17 beam-L DDTree 建树器对抗复验

日期 2026-09-09 · 验证者 B · 对象: `ninfer-fusion-repo\tools\archkit\dflash2_tree.py` +
`test_dflash2_tree.py` + `eval_ddtree.py --tree beam:L`（备份 `_eval_ddtree_pre_s17.py.bak`）。

**总裁决: 4/4 检查 PASS（含两项事实纠偏：测试实为 27/27 而非 26/26；"beam≥chain" 边界实锤并给出最小复现）。
代码本身无缺陷；边界是设计使然（docstring 已声明"不假设嵌套"），附 1 条低成本加固建议。**

## (a) 测试复跑 → PASS（计数纠偏：27/27，非 26/26）
```
wsl.exe -e bash -c "cd .../tools/archkit && python3 test_dflash2_tree.py"   # exit 0
grep -c ' PASS ' _b_s17_test.log        → 28   (27 个用例行 + 1 行 'ALL PASS' 横幅)
grep -c ' FAIL ' _b_s17_test.log        → 0
grep ' PASS ' _b_s17_test.log | grep -vc '^ALL'  → 27    ← 精确用例数
```
实际 27 个用例 = (a) 预算 4l×4mult=16 + (b) 分支 6 + (c) l=1 退化 2 + 布局 3，全部 PASS，exit 0。
S17 看板行写 "26/26 PASS" — 少计 1（无害，纯计数笔误）。宣称的关键断言逐一在日志确认：
(b) `EAL beam=4 chain=0`（实数断言）、`chain==(0,0,0,0)`、`beam 树恰含两分支`、`nodes==8`、
`budget=4 → L=1 退化为链`；(c) 64 随机表 top path == 单链 + tie→最低 rank；
布局: step-0 毒化行(p>0=+1e9) 与真 -inf 行产出相同树、budget<l 与坏 mode 均 ValueError。

## (b) "beam-L ≥ 单链" 边界 → 实锤（设计使然，非 bug；附最小复现）
自建探针 `_collab/_b_tmp/bnd_s17.py`（已清理，核心构造如下），l=4、TOP_K=16、L=2（budget 8）：
```
TIE : C 的分支格, truth=链路径      → tree=[(0,0,0,0),(1,1,2,3)]  EAL beam=4 == chain=4
LOSE: 链全局弱格: step0 行 c0=1.0/c1=0.9/c2=0.8; 链的后续行(表[1..3][0])全 -10;
      分支行 表[1..3][1] c1=5.0, 表[1..3][2] c2=4.0:
      chain=(0,0,0,0)  tree=[(1,1,1,1),(1,1,1,0)] nodes=5
      chain path in beam? False
      EAL chain(truth=chain)=4   beam(truth=chain)=0      ← beam 整块输光
PART: 同格 truth=(1,1,9,9)       → beam EAL=2 (前缀条件制), chain=0
L1EQ: build_tree(tab,7,7)==[single_chain_walk(tab,7)] 于 50/50 随机 l=7 表; budget=l 同退化为链
```
**边界结论**: 该命题当且仅当贪心链的累计分保持在 top-L 累计前缀之内（或 L=1, 此时二者恒等）时成立；
一旦链的累计分跌出 top-L 且 truth 骑在链上，beam 严格输（上例 0 vs 4）。建树器**有意不强制包含链**
（docstring: "do not assume nesting"），且 EAL 是前缀条件制——beam 路径与 truth 只在后段相同不吃分。
这是"builder 覆盖"声明的适用边界：它保证每步保住 top-L 累计前缀，不保证 ≥ 贪心链的 EAL。
**加固建议（一行改动级）**: 把 `single_chain_walk` 的路径保底并入输出（或作为第 L+1 条 beam），
至多 +l 个节点即可让 `EAL_tree ≥ EAL_chain` 成为定理（预算充足时零代价）。

## (c) eval_ddtree diff + flag 校验 → PASS
```
diff _eval_ddtree_pre_s17.py.bak eval_ddtree.py | grep -c '^<'  → 0
diff _eval_ddtree_pre_s17.py.bak eval_ddtree.py | grep -c '^>'  → 60
```
与宣称 **0 删/60 增** 逐字吻合。60 行新增全部为: docstring 说明、argparse 注册、
`import sys`(顶层 stdlib)、3 行纯计数器初始化(L239-241)、以及全部行为逻辑均带门禁
(`if args.tree:` L165-174 含惰性 import dflash2_tree; `if tree_L:` L328-346 与 L409-416)。
OFF 路径结构级未动 ✓（运行级 OFF diff 仍按 C 所言缓办——本机 WSL torch 为 CPU-only,
`beam:16` 合法值过闸后在环境处停: `AssertionError: Torch not compiled with CUDA enabled`,
证明闸门先于重活、且完整运行确需 GPU 窗口）。
flag 矩阵（`--ckpt /nonexistent.pt --tree <f>`，脚本 `_b_tmp/flagcheck_s17.sh`）:
```
FLAG[beam:0]   exit=2 traceback=0  last: eval_ddtree.py: error: --tree expects beam:L with 1 <= L <= 16 (e.g. beam:4)
FLAG[beam:17]  exit=2 traceback=0  (同上一行错误)
FLAG[beam:x]   exit=2 traceback=0  (同)
FLAG[nonsense] exit=2 traceback=0  (同)
FLAG[beam]     exit=2 traceback=0  (同)   ← 无冒号
FLAG[beam:4x]  exit=2 traceback=0  (同)   ← 后缀垃圾
FLAG[beam:-1]  exit=2 traceback=0  (同)   ← 负数(含 '-1'.isdigit()=False 路径)
FLAG[beam: 4]  exit=2 traceback=0  (同)   ← 前导空格
FLAG[beam:16]  exit=1  → 过闸, 之后死于 CPU-only torch (环境限制, 非 --tree 问题)
```
8/8 非法值全部干净失败：统一可读单行错误、exit 2、零 traceback、无静默回退 OFF。

## (d) stdlib 纯度 → PASS
```
grep -n 'import' dflash2_tree.py      → 零匹配 (整模块无任何 import)
grep -n '^[fF]rom\|^import' test_dflash2_tree.py → import os / random / sys + from dflash2_tree import ...
python3 -c "import sys; sys.path.insert(0,'.../archkit'); import dflash2_tree, test_dflash2_tree;
            print('torch' in sys.modules, 'numpy' in sys.modules)"   → False False
```
两模块导入后 sys.modules 无 torch/numpy — GPU 窗口期无 torch/numpy 亦可运行。

## 附记
- 本轮中 WSL 服务短暂无响应 3 次（0x8007274c，约 25 秒自愈）；未重启 WSL（会杀 window D 的活）。
- 清理: `_collab/_b_tmp/`、`_b_s17_test.log`、`_b_s17_flags.log` 已删；未动 `dflash2_tree.py`/
  `test_dflash2_tree.py`/`eval_ddtree.py`；无构建、无 GPU。
