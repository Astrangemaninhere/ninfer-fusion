
# draft-tokens sweep (2026-09-10 15:42, counting prompt, max_tokens=192)

| run | backend | gen | decode tok/s |
|---|---|---|---|
| mtp_d1 | mtp | 92 | 106.2tok/s |
| mtp_d3 | mtp | 69 | 151.5tok/s |
| mtp_d7 | SERVE_FAILED | [2026-09-10 15:40:34.897] [error] ninfer-serve: --spec mtp requires -- |  |
| dflash_d1 | dflash | 69 | 65.4tok/s |
| dflash_d3 | dflash | 92 | 80.5tok/s |
| dflash_d7 | dflash | 81 | 70.0tok/s |
| dflash2_d1 | SERVE_FAILED | [2026-09-10 15:41:47.362] [error] ninfer-serve: --spec dflash2 uses th |  |
| dflash2_d3 | SERVE_FAILED | [2026-09-10 15:41:52.439] [error] ninfer-serve: --spec dflash2 uses th |  |
| dflash2_d7 | dflash2 | 2 | 31.7tok/s |

baseline (no spec) on this prompt: ~54 tok/s.
draft=1 is the sharpest probe: minimal overhead, so a backend below baseline
there has a pathological draft step rather than an acceptance/width problem.
