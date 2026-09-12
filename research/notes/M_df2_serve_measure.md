# dflash2 serve-path measurement (2026-09-10 15:20)

| run | decode tok/s (per request) | wall | speculative field |
|---|---|---|---|
| baseline | decode=54.8tok/s | wall=2.47s | speculative=off |
| baseline | decode=54.4tok/s | wall=3.36s | speculative=off |

acceptance fields found: 

note: if the engine reports no acceptance counter, the acceptance rate must be
derived from drafted-vs-accepted token counts (next step) — say so, do not guess.

## corrected run (2026-09-10 15:24)

```
[dflash2_auto]
[2026-09-10 15:23:34.730] [info] ninfer-serve: [req 1] done finish=stop_token prompt=62 gen=129 cache=0 reuse=root ttft=133ms prefill=730.3tok/s decode=55.1tok/s wall=2.46s host=411.78ms decode-host=3003.4us/round wait=15159.6us/round speculative=off
[2026-09-10 15:23:38.081] [info] ninfer-serve: [req 2] done finish=stop_token prompt=122 gen=175 cache=0 reuse=root ttft=166ms prefill=1601.0tok/s decode=54.9tok/s wall=3.34s host=507.97ms decode-host=2835.4us/round wait=15411.3us/round speculative=off
  acceptance-ish fields: 
[baseline]
[2026-09-10 15:24:05.712] [info] ninfer-serve: [req 1] done finish=stop_token prompt=62 gen=129 cache=0 reuse=root ttft=134ms prefill=725.0tok/s decode=54.4tok/s wall=2.49s host=445.87ms decode-host=3291.8us/round wait=15124.5us/round speculative=off
[2026-09-10 15:24:09.105] [info] ninfer-serve: [req 2] done finish=stop_token prompt=122 gen=175 cache=0 reuse=root ttft=168ms prefill=1582.5tok/s decode=54.2tok/s wall=3.38s host=589.68ms decode-host=3251.2us/round wait=15216.1us/round speculative=off
  acceptance-ish fields: 
```

## final run (2026-09-10 15:26) — auto vs explicit vs baseline

```
[dflash2_auto] speculative=dflash2 
[2026-09-10 15:25:13.901] [info] ninfer-serve: [req 1] done finish=stop_token prompt=62 gen=2 cache=0 reuse=root ttft=143ms prefill=649.2tok/s decode=31.4tok/s wall=0.17s host=60.96ms decode-host=31515.1us/round wait=326.7us/round speculative=dflash2 1.00tok/round (0.0%)
[2026-09-10 15:25:15.428] [info] ninfer-serve: [req 2] done finish=output_limit prompt=122 gen=192 cache=0 reuse=root ttft=168ms prefill=1553.6tok/s decode=141.6tok/s wall=1.52s host=1355.21ms decode-host=31700.6us/round wait=444.5us/round speculative=dflash2 4.55tok/round (50.9%)

[dflash2_explicit] speculative=dflash2 
[2026-09-10 15:25:37.419] [info] ninfer-serve: [req 1] done finish=stop_token prompt=62 gen=2 cache=0 reuse=root ttft=144ms prefill=644.1tok/s decode=30.0tok/s wall=0.18s host=58.26ms decode-host=32484.6us/round wait=837.3us/round speculative=dflash2 1.00tok/round (0.0%)
[2026-09-10 15:25:38.957] [info] ninfer-serve: [req 2] done finish=output_limit prompt=122 gen=192 cache=0 reuse=root ttft=168ms prefill=1555.0tok/s decode=141.2tok/s wall=1.52s host=1347.26ms decode-host=31757.3us/round wait=476.5us/round speculative=dflash2 4.55tok/round (50.9%)

[baseline] speculative=off 
[2026-09-10 15:26:00.280] [info] ninfer-serve: [req 1] done finish=stop_token prompt=62 gen=129 cache=0 reuse=root ttft=135ms prefill=707.1tok/s decode=55.0tok/s wall=2.47s host=407.16ms decode-host=3024.9us/round wait=15177.1us/round speculative=off
[2026-09-10 15:26:03.696] [info] ninfer-serve: [req 2] done finish=stop_token prompt=122 gen=175 cache=0 reuse=root ttft=167ms prefill=1565.0tok/s decode=53.8tok/s wall=3.41s host=629.51ms decode-host=3518.6us/round wait=15094.6us/round speculative=off

```
acceptance counters printed by the engine: 
(if empty: the engine does not report acceptance — it must be derived from
drafted-vs-accepted counts, which needs a counter added; do not guess it.)
