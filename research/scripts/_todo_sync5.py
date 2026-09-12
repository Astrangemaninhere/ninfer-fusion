import pathlib, datetime

p = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\_TODO.md")
old = p.read_text(encoding="utf-8", errors="replace")
block = """
## 2026-09-12 凌晨 · dflash2 接受率：逐步取证与根因（重大）

### 一、装了仪表并跑出逐步归因
- 在 `src/ops/kernel/dflash2_selector.cuh` 的 `[df2sel]` 探针旁新增 `[df2cand]`：每步打印 top-K 候选全貌（备份 `/home/user/df2cand_bak/`，含 `--dry` 前 md5 `92026e7092da`）。
- 新增 `NINFER_DF2FEAT`：dump `features / projected / context_full`（`dflash2_impl.h`，备份 `/home/user/df2feat_bak/`）。
  要点：**必须先 D2H 拷贝再落盘**（直接 fwrite 设备指针会得到 0 字节）；**图捕获期间不能 cudaStreamSynchronize**，
  所以特征 dump 必须配 `--no-cuda-graph`（否则 abort，rc=134）。
- 逐列归因（候选为真词表 id，无需跨域映射；只统计落在已接受前缀内的干净列）：
  **head_miss 15/15 = 100%**（干净列）；放宽到全部 290 步为 head_miss 84.8% / walk_error 10.7% / verify_flip 4.1%。
  自纠：曾一度误判候选是"草稿域 id"并二次映射，导致 100% 假 miss；也曾在第二版漏掉干净列过滤。

### 二、根因（实测，非推断）
- **`features` 只有第 0 段（tap0）有数据，tap1..4 整块为 0**：`--no-cuda-graph` 下 call=5..8 的 tap 范数
  依次为 `[278.4,0,0,0,0]`、`[271.3,0,0,0,0]`、`[275.9,0,0,0,0]`、`[272.9,0,0,0,0]`。
  ⇒ 喂给 `fc → context_norm → 5 层草稿网 → selector → 提案头` 的特征 **80% 是空的**。
- 下游算术已排除：`features → fc → rmsnorm` 与离线 numpy 复算**逐位吻合**（相对误差 0.00000）。
- 这一条解释了今晚所有"零差异"实验：换 5 层网 / 换选择器 / 清零 `text/draft_head` / 切 `--lm-head-draft`
  都不改变"候选是否命中"，因为缺料在更上游。

### 三、被实测否定的假设（清单）
| 假设 | 实验 | 结果 |
|---|---|---|
| 4 比特提案头是天花板 | 切 `--lm-head-draft`（draft_head 域） | 4.81% → 4.54%，候选集 0% 相同 |
| 草稿头混血（新层+08-26 老选择器） | 整体换 incoai 自洽头 | 逐项完全相同 |
| 选择器不参与 | 清零两张码本 | 4.81% → 4.01% |
| 5 层草稿网有问题 | 换整套 incoai 5 层网 | 零差异 |
| 提案头不在路径上 | 清零 `text/draft_head` 码区 | 候选集变了（0.9% 相同）但接受率不变 |
| tap 层号错位 | 读 config 对照契约 | `config.h:150 {5,19,33,47,61}` 与 HF `target_layer_ids` 完全一致 |
| KV 精度造成分歧 | 两侧 KV 都设 bfloat16 | 前 7–14 token 逐字一致 |

### 四、契约侧澄清（重要）
- `docs/maintainer/qwen3.8-27b-artifact.md`：`text/draft_head` + `draft_head_token_ids` 是
  **"optimized MTP draft head"**（第 i 行 ↔ 全词表行 `draft_head_token_ids[i]`）⇒ 它属于 **MTP** 路径，
  不是 dflash2 的提案头 —— 与"清零它不影响 dflash2 接受率"的实测一致。
- `_collab/A5_dspark_rootcause.md` 曾"代码级排除" tap 层号错（抓 post-MLP residual、按升序写 `feat[index·hidden]`），
  但本次实测 tap1..4 为 0 ⇒ 那次读码排除漏了某种情况（待三路子代理对照后定因）。

### 五、进行中
- 三路子代理（同方法各做一遍）已后台派出，目标：定位"只有槽 0 被写入"的根因并给最小补丁；
- 修复后由主代理统一编译落地，并用 `_verify_fix.sh` 验收：**tap1..4 范数须由 0 变非零**，
  且接受率相对基线 `4.81% / 20,3,0,0,0,0,0` 有可测改善。
"""
p.write_text(old + block, encoding="utf-8")
print("_TODO.md ->", len(p.read_text(encoding="utf-8", errors="replace").splitlines()), "lines")
