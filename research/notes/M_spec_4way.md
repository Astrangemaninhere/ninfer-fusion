# 四档投机对比 (serve 实测, 2026-09-10 15:36, max_tokens=192, --no-thinking)

| 配置 | decode | gen | speculative |
|---|---|---|---|
| plain_off | decode=54.8tok/s | gen=98 | speculative=off |
| plain_mtp | SERVE_FAILED |  |  |
| dspark | decode=44.7tok/s | gen=120 | speculative=dflash |
| dflash2 | decode=33.2tok/s | gen=2 | speculative=dflash2 |

注: 引擎尚无接受率计数 (E2/S44 补丁在做), 故本表只有速度与合法性;
    MTU/DFlash/DFlash2 的差异需要计数到位后才能归因 (草稿宽度/接受率/草稿成本)。

# 四档投机对比 v2 (serve 实测 2026-09-10 15:39, max_tokens=192, --no-thinking)

| 配置 | backend | 中文 prompt (gen/decode) | 计数 prompt (gen/decode) |
|---|---|---|---|
| plain_off | speculative=off | zh:gen=98/decode=54.8tok/s | num:gen=27/decode=54.3tok/s |
| plain_mtp | speculative=mtp | zh:gen=98/decode=94.0tok/s | num:gen=27/decode=119.4tok/s |
| dspark | speculative=dflash | zh:gen=120/decode=46.1tok/s | num:gen=34/decode=70.1tok/s |
| dflash2 | speculative=dflash2 | zh:gen=2/decode=31.7tok/s | num:gen=42/decode=108.0tok/s |

读法: 中文 prompt 上 dflash2 已知有退化停止缺陷 (gen=2) ⇒ 该列不可用于速度比较;
      计数 prompt 上各配置都能生成满 192 token ⇒ **速度只比这一列**。
      接受率仍需 E2/S44 的计数器落地后才能给出。

## CLI per-position [cli_mtp3] 17:49
summary     mtp acceptance rate       33.33%
summary     mtp acceptance length     2.00 tok/round
summary     mtp accepted by pos       23,12,8

## CLI per-position [cli_dspark] 17:49
summary     dflash acceptance rate    11.22%
summary     dflash acceptance length  1.39 tok/round
summary     dflash accepted by pos    15,5,2,0,0,0,0

## CLI per-position [cli_dflash2] 17:49
summary     dflash2 acceptance rate   10.03%
summary     dflash2 acceptance length 1.69 tok/round
summary     dflash2 accepted by pos   23,11,3,1,0,0,0

## 长上下文 needle 32 发 (2026-09-10 17:56)

| 配置 | ctx | hits | backend |
|---|---|---|---|
| lc_base | 16384 | 13/32 | speculative=off |
| lc_base | 32768 | 16/32 | speculative=off |
| lc_mtp3 | 16384 | 14/32 | speculative=mtp |
| lc_mtp3 | 32768 | 14/32 | speculative=mtp |
| lc_dflash2 | 16384 | 20/32 | speculative=dflash2 |
| lc_dflash2 | 32768 | 13/32 | speculative=dflash2 |

判读: 投机档命中数必须与基线相同; 低于基线即质量回归 (掉针=检索失败)。
# 四档投机对比 (serve 实测, 2026-09-10 17:56, max_tokens=192, --no-thinking)

| 配置 | decode | gen | speculative |
|---|---|---|---|
| plain_off | decode=34.0tok/s | gen=88 | speculative=off |
| plain_mtp | SERVE_FAILED |  |  |
| dspark | decode=31.8tok/s | gen=68 | speculative=dflash |
| dflash2 | decode=74.6tok/s | gen=76 | speculative=dflash2 |

注: 引擎尚无接受率计数 (E2/S44 补丁在做), 故本表只有速度与合法性;
    MTU/DFlash/DFlash2 的差异需要计数到位后才能归因 (草稿宽度/接受率/草稿成本)。
