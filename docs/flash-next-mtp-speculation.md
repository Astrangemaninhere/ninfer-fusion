<!-- Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion). -->
# Finish MTP speculative decoding for Qwen3.8-Flash-Next

**Status:** not implemented. The draft path exists and runs, but produces useless
drafts because it skips the sparse-attention indexer.

**Owner:** unassigned. Written 2026-09-06 from a live production diagnosis.

**Do not start by adding the missing call.** The state it needs does not exist
yet — see "What is actually missing".

---

## Why this matters

Flash-Next carries a full 512-expert MTP draft head — 2,516,582,400 parameters,
**4.69 GiB resident in BF16** — and currently gets nothing from it. On a card
where the model leaves ~7.4 GiB free, that is the single largest piece of dead
weight in the deployment.

NVIDIA ships and deliberately quantizes this head (FP8, 128×128 block-scaled) in
`nvidia/Qwen3.8-Flash-Next-NVFP4`, which is good evidence the head is trained and
useful. The problem is on our side.

For comparison, the same engine running MTP on qwen3.8-27b sustains **66.5%**
draft acceptance under 32k of context.

## Measured symptom

Enabling `--spec mtp --draft-tokens 4` on Flash-Next in production:

```
1158 tokens / 28.36 s = 40.8 tok/s
accepted 150 / 4028 drafted = 4%
accepted_per_position [149, 1, 0, 0]
```

Three findings, all from one run:

1. **Acceptance is 4%**, against 66.5% for the 27B on the same engine.
2. **Decode got 31% SLOWER** than with speculation off (40.8 vs 58.9 tok/s under
   comparable load). Verification overhead is real and the drafts are discarded,
   so speculation is a pure loss.
3. **The engine went unreachable mid-generation** and the supervisor restarted it.
   Two further requests in that batch never completed. Not root-caused; it may be
   a second bug or a consequence of the first.

`accepted_per_position [149, 1, 0, 0]` is the diagnostic that matters. A merely
weak draft head decays smoothly — the 27B shows `[85%, 70%, 57%, 49%]`. A cliff
after position 1 means position 1 is being carried by something OTHER than the
draft head's own attention, and positions 2+ have nothing.

## Root cause

`flash_next_mtp_step` never runs the QSA indexer, so the MTP attention reads
uninitialised block selections.

**The main decode path** (`src/targets/qwen3_8_flash_next/impl/text_decode.cpp:275`)
does this, in order:

```cpp
flash_next_qsa_indexer_decode(
    round_ws.block_input, model.full_attention[qsa_idx], token_indices, mrope_positions,
    table_rows, source_slots, destination_slots, state.qsa_indexer_caches[qsa_idx],
    maximum_blocks, active_blocks, workspace,
    round_ws.selected_blocks, round_ws.selected_counts,   // <-- OUTPUTS
    stream, aliased_recurrent_scan);

flash_next_qsa_attention_decode(
    round_ws.block_input, model.full_attention[qsa_idx], token_indices, mrope_positions,
    table_rows, round_ws.selected_blocks, round_ws.selected_counts,   // <-- consumed
    state.qsa_attention_caches[qsa_idx], workspace, round_ws.block_output, stream);
```

The indexer decides which sparse KV blocks to attend to and writes
`selected_blocks` / `selected_counts` (`qsa_indexer.cpp:54`, both `Tensor&`
outputs).

**The MTP path** (`src/targets/qwen3_8_flash_next/impl/mtp_forward.cpp:165`) goes
straight from hyper-prepare to attention:

```cpp
// 4. Attention hyper prepare -> attn_in
flash_next_hyper_prepare(...);
// 5. QSA Attention decode
flash_next_qsa_attention_decode(attn_in, mtp.attention, token_indices, mrope_positions,
                                table_rows, selected_blocks, selected_counts, mtp_cache,
                                workspace, attn_out, stream);
```

`mtp_forward.cpp` contains **zero** references to the indexer. The
`selected_blocks` / `selected_counts` it passes are allocated in
`text_executor.cpp:891-892` from `mtp_selected_blocks_` / `mtp_selected_counts_`
and **never written**. The draft head attends to whatever those device buffers
happen to contain.

Corroborating fingerprint: `text_executor.cpp:889` computes

```cpp
const auto active_blocks = std::min(max_blocks, (token_index + 1) / 4);
```

inside `draft_mtp_tokens`, and **never uses it**. `active_blocks` is an indexer
parameter. Someone worked out the value and the code meant to consume it was
never written.

This explains the per-position cliff exactly. The MTP stem mixes in the real
backbone hidden state (`mtp_forward.cpp:146-158`), which alone predicts the next
token often enough to explain position 1's ~15%. From position 2 the draft
depends on MTP attention output, which is reading arbitrary blocks, so it is
never right.

## What is actually missing

**There is no MTP indexer cache.** This is the reason this is not a one-line fix.

- `text_executor.h:254` holds `std::optional<QsaAttentionCacheView> mtp_cache_` —
  attention only.
- The main path keeps two parallel arrays: `state.qsa_indexer_caches[i]` and
  `state.qsa_attention_caches[i]`, one pair per QSA layer.
- Grep for an MTP indexer cache returns nothing.

So the work is: give the MTP layer its own indexer cache, keep it populated across
prefill and decode the way the 12 QSA layers are, and then call the indexer.

## Implementation

1. **Add an MTP indexer cache.** Mirror how `qsa_indexer_caches` are declared,
   sized, allocated and bound for the QSA layers. The MTP layer is one additional
   QSA-shaped layer for this purpose. Follow whatever the existing per-layer code
   does rather than inventing a parallel mechanism.

2. **Populate it wherever the attention cache is populated.** Find every site that
   writes `mtp_cache_` / the MTP attention cache — prefill, decode, and any
   context-cache restore or checkpoint path — and give the indexer cache the same
   treatment. If MTP state is captured in CUDA graphs or in state images, the new
   cache has to travel with them.

3. **Thread `active_blocks` and `maximum_blocks` into `flash_next_mtp_step`.**
   Extend its signature (`mtp_forward.h:18-24`). `active_blocks` is already
   computed at `text_executor.cpp:889` but **must be recomputed per draft position
   k**, not once before the loop — blocks turn over every 4 tokens, so a 4-token
   draft window can cross a boundary.

4. **Call the indexer in `flash_next_mtp_step`, between step 4 and step 5**, with
   the same argument shape as `text_decode.cpp:275`.

5. **Size the workspace.** `mtp_workspace_` was never sized for the indexer. The
   main path reserves indexer scratch plus sort temp via
   `allocate_flash_next_qsa_indexer_workspace` and
   `flash_next_qsa_indexer_sort_temp_bytes` (`text_decode.cpp:137-148, 182-185`).
   `mtp_workspace_->reset()` is called per draft position
   (`text_executor.cpp:900`), so the arena must hold one position's worth.

6. **Re-check the wedge.** The engine going unreachable mid-generation was
   observed once and is not explained by this analysis. Reproduce it under the
   fixed code before declaring the feature done.

## Open questions for the implementer

These were not resolved during diagnosis and change the shape of the work:

- **`source_state_slots` / `destination_state_slots`.** The main path passes
  `source_slots` / `destination_slots`. The draft path has no equivalent. What do
  these mean for a speculative position that may be discarded? Getting this wrong
  risks corrupting real KV state with draft state — the highest-risk part of the
  change.
- **`aliased_recurrent_scan`.** The main path passes a flag; the correct value for
  a draft step is not obvious.
- **VRAM cost.** An indexer cache is not free and Flash-Next currently leaves only
  ~7.4 GiB free, on a machine that failed to start five times today over a 1.21 GiB
  shortfall. Size the cache before allocating it, and check startup against a
  populated desktop (browser, other GPU apps) rather than an idle one.
- **Does the draft path need per-lane caches?** The QSA caches are per-layer and
  indexed by lane; MTP drafting runs per lane too.

## Verification

**Do not accept "it runs" as done.** The pre-existing behaviour also ran.

Enable in the model catalog entry for `qwen3.8-flash-next` in the supervisor
config (this is `supervisor.prod.json` on the production box; add to the entry's
`args`, and regenerate `engine.args` if that model is active):

```
"--spec", "mtp", "--draft-tokens", "4"
```

Then generate ~1200 tokens and read the engine's own numbers out of the request
log (`supervisor-logs/prod.jsonl`, `request_done` events, `speculative` block):

| Metric | Broken (today) | Target |
|---|---|---|
| `accepted_tokens / drafted_tokens` | 4% | comparable to the 27B's 66.5% under 32k |
| `accepted_per_position` | `[149, 1, 0, 0]` | smooth decay, e.g. `[85%, 70%, 57%, 49%]` |
| decode tok/s | 40.8 (vs 58.9 with spec off) | materially above the spec-off baseline |

The per-position curve is the real acceptance test. A high aggregate with a cliff
after position 1 means the indexer is still not feeding attention.

Measure the spec-off baseline on the same prompt in the same session — it moves
with context length (58.9 tok/s under long-context agent traffic, 82.6 tok/s on a
short-context probe), so a number from another run proves nothing.

## Constraints on the production box

- The engine binary is held open while it runs; a normal build fails with LNK1104.
  Stop the supervisor first (it owns the engine through a job object, so the engine
  exits with it), or build to a scratch `OutDir` to compile-check only.
- Building with `/p:OutDir` under the temp directory breaks MSBuild's up-to-date
  check (warning MSB8029) and will **relink stale objects while reporting success**.
  Touch a source file to force the compile, and verify the binary timestamp rather
  than trusting exit code 0.
- Editing `supervisor.prod.json` while the supervisor is running trips its
  config-stamp guard by design. Edit it while stopped.
- Flash-Next takes ~60 s to load. Budget that into each test cycle.

## Related, deliberately out of scope

- **Quantizing the MTP experts.** Once speculation works, those 4.69 GiB become
  worth compressing: FP8 frees 2.34 GiB, NVFP4 frees 3.52 GiB. The draft head is
  the most forgiving place in the model to quantize hard, because the verifier
  catches every bad draft — the cost is acceptance rate, never correctness. Do
  this only after there is an acceptance baseline to regress against.
- **`--lm-head-draft`.** Flash-Next supports the optimized proposal head
  (`load/materialized.cpp:338`) but it materializes an extra weight payload.
  Leave it off until the memory envelope is understood.
