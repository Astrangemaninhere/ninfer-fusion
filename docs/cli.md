# NInfer CLI

`build/apps/ninfer` runs one request against one registered `.ninfer` artifact. Build NInfer and
download an artifact using the [project README](../README.md) before following this guide.

The examples use Qwen3.8-27B NVFP4 with FP8 KV storage.

## Text input

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Exactly one of `--prompt` and `--messages` is required. The CLI normally omits `--kv-capacity`, so
the shared Main Text KV pool follows the example's 32,768-token `--max-context`.

Answer content is streamed to stdout. Reasoning, model loading (including the registered target and
canonical `weights_id`), timings, throughput, GPU memory, and speculative-decoding statistics are
written to stderr, so stdout can be redirected independently:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Return one sentence." \
  --max-context 4096 \
  --max-new 64 \
  --kv-dtype fp8 \
  > answer.txt 2> run.log
```

Thinking is enabled by default. If the chat template embedded in the loaded artifact exposes
reasoning effort, `--reasoning-effort low|medium|xhigh` selects it; omitting the option uses the
template's default. An artifact whose template does not expose effort rejects the option. Add
`--no-thinking` for direct-response prompt rendering; it cannot be combined with
`--reasoning-effort`. `--greedy` selects exact argmax decoding independently.

`--thinking-budget N` places a positive upper bound on accepted model-origin tokens while the
new-turn Qwen thinking block remains open. If the model has not emitted `</think>` at that exact
boundary, Engine appends [Qwen's canonical early-close guidance](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md)
and `</think>` to the same resident sequence without sampling, publishes the guidance through the
reasoning stream, then resumes ordinary generation from the updated context. A natural thinking
close, stop condition, cancellation, or total output/context limit at the boundary takes priority
and suppresses this insertion. The option cannot be combined with `--no-thinking`, but it can be
combined with `--reasoning-effort`.

`--max-new` counts every committed generated token, including internally inserted control tokens.
When the effective output capacity extends beyond the thinking budget, it must have room for the
complete tokenizer-derived control suffix plus one post-close model token; an undersized request is
rejected rather than truncating the suffix. Normal output sends the inserted guidance to stderr as
reasoning. `--print-token-ids` includes the inserted IDs, while `--raw-output` preserves the raw
control representation.

For example, this allows at most 512 model-origin thinking tokens while retaining enough total
output capacity for the inserted suffix and the answer:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain speculative decoding, then give a concise conclusion." \
  --max-context 4096 \
  --max-new 1024 \
  --thinking-budget 512 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

## Startup memory profile

GPU residency is frozen when the Engine starts:

- no `--spec` omits MTP/DFlash weights and state and the optimized proposal head;
- `--spec mtp` loads only MTP, while `--spec dflash` loads only the 35B-A3B text-only DFlash
  backend;
- a speculative backend with the full proposal head omits the optimized proposal head;
- Vision is disabled by default, omitting its weights and Vision-specific unified-workspace extent;
- `--vision` loads the weights, expands the one Program workspace for Vision encode/handoff, and
  enables image/video input.
- the one-request CLI uses root-only context mode, so it does not reserve an extra Device
  checkpoint StateImage or capture a continuation that no later request could consume.

The complete `.ninfer` inventory is still validated. These choices are not lazy loading: a
text-only Engine rejects media and cannot enable Vision later. DFlash and Vision are mutually
exclusive. The default speculative and Vision settings produce the smallest resident profile.

## Structured messages

`--messages` accepts either a non-empty JSON message array or an object containing `messages`
and an optional `tools` array.

```json
[
  {
    "role": "system",
    "content": "Answer concisely."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "examples/cli/media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```

Run message files from the repository root when they contain repository-relative media paths:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --kv-dtype fp8 \
  --vision \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`.
System and developer messages retain their array positions; the Qwen family frontend renders both
as system-class ChatML turns rather than moving later instructions to the beginning.

Message content may be a string or an ordered array containing:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

`image_url` and `video_url` may be strings or objects containing a string `url`. Assistant
history may include `reasoning_content` and `tool_calls`; a tool result uses role `tool` and
`tool_call_id`.

See [`examples/cli/`](../examples/cli/) for committed text, image, video, mixed-media, thinking,
long-decode, and long-context inputs.

## Speculative decoding

Speculative decoding is disabled by default. Select MTP with one to five draft positions, or the
35B-A3B text-only DFlash backend with one to fifteen. `--lm-head-draft` selects the optimized
proposal head and requires a selected backend:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For DFlash:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --kv-dtype fp8 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

MTP and DFlash cannot be enabled together. The published [performance results](performance.md)
use MTP with three draft tokens and DFlash with seven draft tokens (block length eight), both with
the optimized proposal head. DFlash accepts one to fifteen draft tokens; seven forms the measured
block length eight, while fifteen uses the full native block.

## Common options

The table lists executable defaults. The examples above select FP8 KV and MTP3.

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | per-sequence logical context ceiling | `2048` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `2048` |
| `--prefill-chunk N` | positive text-prefill chunk, in multiples of 128 | `3072` |
| `--prefill-chunk-mode dynamic\|manual` | which of the two modes owns the prefill unit: `dynamic` lets the bandwidth governor re-derive it while decode is contended, `manual` pins it to `--prefill-chunk` for the whole run | `dynamic` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--kv-dtype bf16\|int8\|fp8\|nvfp4\|iso4e\|iso3\|rk4v4\|e8` | KV-cache storage; `fp8` selects the fp8 KV tier -- E4M3 code planes over per-16 E4M3 group scales, the packed-16 form the fp8 attention kernels read (8.50 bits/element in the bit ladder). Both fp8 spellings are that one tier (`all:fp8` in `--kv-layer-storage` names the same one); the storage enumerator keeps its legacy name `Fp8E4M3Row256`, i.e. the `fp8-e4m3-r256` key in request logs -- a name, not a second codec. `iso3`/`e8` are the deprecated aliases of `iso4e`/`rk4v4`, and `nvfp4` is the weight tier the shipped `qwen3_8_27b_nvfp4_modelopt` artifact records for itself (`weights_id: nvfp4-modelopt`) | `bf16` |
| `--spec auto\|off\|mtp\|dflash\|dflash2\|none` | speculative backend; unset means `auto`, which uses the artifact's own draft backend | `auto` |
| `--draft-tokens N` | MTP `0..15` (`0` = adaptive ladder); DFlash `1..15` | unset |
| `--draft-tree L,d` | MTP tree verify shape: `L` rank-paths per depth, depth `d`, node budget `L*d <= 15`; replaces `--draft-tokens` (`L,d` = `1,d` must reproduce `--draft-tokens d`) | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--vision` | enable image/video input and load Vision GPU allocations | off |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-thinking` | disable thinking in prompt rendering | thinking on |
| `--thinking-budget N` | positive model-origin thinking-token cap; omitted means unlimited | unset |
| `--reasoning-effort low\|medium\|xhigh` | select an effort exposed by the loaded chat template | template default |
| `--greedy` | exact argmax decoding | off |
| `--temperature F` | sampling temperature override | registered model/mode default |
| `--top-p F` | nucleus-threshold override | registered model/mode default |
| `--top-k N` | top-k-threshold override (`0..20`; zero selects the top-20 cap) | registered model/mode default |
| `--min-p F` | min-p-threshold override | registered model/mode default |
| `--presence-penalty F` | presence-penalty override | registered model/mode default |
| `--frequency-penalty F` | frequency-penalty override | registered model/mode default (`0`) |
| `--seed N` | sampling seed | `0` |

`--prefill-chunk` is the unit's **ceiling** and `--prefill-chunk-mode` decides who owns it. `manual`
pins the unit to whatever `--prefill-chunk` asked for, for the whole run, with the governor constructed
disabled so nothing re-derives it; `dynamic` (the default) lets the governor shrink the unit while decode
is contended. The value is **case-sensitive** and an unknown one is refused with
`prefill chunk mode must be dynamic|manual, got '<value>': dynamic lets the bandwidth governor shrink
the prefill unit while decode is contended, manual pins it to --prefill-chunk for the whole run`. The
resolution order is `CLI > environment > default`: an explicit flag wins over `NINFER_FT_BW_GOV` (the
switch behind the mode, `=0` the kill switch and `=1` forcing it on), and an unset flag defers to that
variable. `--prefill-chunk-mode manual` therefore reaches exactly the same state as
`NINFER_FT_BW_GOV=0`, and with `NINFER_FT_BW_TRACE=1` the trace line names which mode produced the unit it
printed — a mode that could not be told apart from the other from outside the process would not be a
mode. Both `ninfer` and `ninfer-serve` accept both values; the perplexity front end accepts only
`manual`, because it must not re-derive its own prefill unit (see
[Perplexity](perplexity.md)).

When a sampling flag is omitted, Engine selects the official general-task preset registered for
the loaded model and the rendered prompt mode. The current presets are:

| Model | Prompt mode | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.6-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.8-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.8-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | thinking | `1.0` | `0.95` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |

Frequency penalty is `0` in every registered preset. Task-specific profiles such as Qwen's
precise-coding profile use explicit sampling overrides.

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use
`--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include
generated token IDs in diagnostics.

Run `./build/apps/ninfer --help` for the exact option contract.

## Context and memory

The registered model IDs have a native context limit of 262,144 tokens. The practical allocation
on one RTX 5090 depends on the selected artifact, media workload, output budget, and KV-cache type.
The compact large-context profile uses `--kv-dtype fp8`, which selects the fp8 KV tier: E4M3 code
planes over per-16 E4M3 group scales, the same tier `all:fp8` names in `--kv-layer-storage` (8.50
bits/element in the bit ladder; the storage enumerator's legacy name for it is `Fp8E4M3Row256`).
INT8 group-64 and BF16 are also available. Artifact identity selects the weight profile;
`--kv-dtype` selects runtime KV storage. The prepared prompt must fit
`--max-context`; generation stops at the remaining context capacity when necessary.
`--kv-capacity N` controls the shared physical Main Text KV pool independently and is rounded up to
the 64-token page size. `--kv-capacity auto` loads the selected weights, measures the remaining GPU
memory, and directly chooses the largest legal page capacity for the complete enabled runtime
layout. This includes the selected speculative backend, fixed sequence state, unified workspace,
and CUDA Graph allowance, while leaving the default 1 GiB automatic headroom
unallocated. It does not probe allocations or resize the pool at request time. The single-request
CLI normally leaves the option omitted so it follows
`--max-context`; the distinction matters primarily to a concurrent Engine or server.

At Engine startup NInfer reserves model weights, persistent sequence state, one phase-reused
Program workspace, and a separate CUDA Graph driver allowance. With Vision enabled, that one
workspace contains a general execution prefix and a fixed item-output handoff region. Vision encode
may reuse the full backing before producing the output; Text/MTP/decode work remains inside the
general prefix while the handoff is live. The capacity is therefore the maximum legal simultaneous
extent, not the sum of Text, Vision scratch, and Vision output allocations. Text prefill uses
`min(--prefill-chunk,--max-context)`; Vision keeps the existing 32,768-token aggregate prompt budget
but plans Device execution for the registered 16,384-token maximum single item. Requests perform no
project-owned device allocation or growth. Context-cache capacity controls are intentionally absent
from this one-request interface; the persistent Engine and server routes own cross-request reuse and
optional Host backing.

All weight, sequence, workspace, and graph allocations are released when the Engine is destroyed.

## KV strategy and cold tier

The options below select **how much of each layer's KV is kept, and in what precision**. They are
independent of `--kv-dtype`, which pins one global tier for every layer. The two are alternative
ways to fill the same per-layer table, so the pair is refused rather than merged:

| Option | Meaning | Default (unset = absent) |
|---|---|---:|
| `--kv-bits F` | one overall per-layer bit ceiling, solved into the per-layer table | unset |
| `--kv-k-bits F` / `--kv-v-bits F` | per-plane bit ceilings (K and V solved separately) | unset |
| `--kv-bit-budget F` | plane-agnostic spelling of the same ceiling | unset |
| `--kv-bits-mode split\|joint\|ceiling` | which reading of a per-plane request runs | `split` |
| `--kv-quality-weight F` | per-tier penalty weight inside the fit: `0` = fastest, `1` = most accurate | unset |
| `--kv-codec-preference CODEC[,CODEC...]` | WHICH codec the joint fit picks among candidates that cost the SAME bits — an order over `rk4v4,bf16,int8,fp8,nvfp4,iso4e` | unset |
| `--kv-tier-scores PATH` | measured two-column table (`quality`, `speed`) for the fit | unset |
| `--kv-k-tier-scores PATH` / `--kv-v-tier-scores PATH` | per-plane variants of the same table | unset |
| `--kv-layer-storage SPEC` | explicit per-layer storage table, e.g. `0-9:rk4v4-g64 10-15:nvfp4-g16` | unset |
| `--kv-tier-formats SPEC` | tier/format vocabulary for a per-layer spec | unset |
| `--kv-rotation` / `--kv-row-scale` | KV rotation and row-scale bake controls | unset |
| `--kv-v-codec` | V-plane codec selection | unset |
| `--nvfp4-mode` | NVFP4 storage variant | unset |
| `--cold-policy` | cold-tier policy: `none`/`off`, `window`, `host`, `disk`, or `host-then-disk` (whose equivalent spelling `host+disk` is also accepted) | unset |
| `--max-cold-pages N` | cold pool page ceiling | unset |
| `--cold-keep-tokens N` | tokens kept hot in front of the cold window | unset |
| `--cold-disk-path PATH` | cold backing file path | unset |
| `--cold-disk-bytes N` / `--cold-host-bytes N` | cold backing size in bytes | unset |

Every option in this table has an **environment spelling** (`NINFER_KV_BITS`, `NINFER_KV_K_BITS`,
`NINFER_KV_V_BITS`, `NINFER_KV_BITS_MODE`, `NINFER_KV_QUALITY_WEIGHT`, `NINFER_KV_ROTATION`,
`NINFER_KV_ROWSCALE`, `NINFER_KV_CALIB_DIR`, `NINFER_KV_CALIB_MAX_TOKENS`) — **except**
`--kv-codec-preference`. The environment layer of this family is
`kv_bits_request_from_env` (`src/product/kv_kv_bits.h`), which fills *scalars* and leaves an unset
field unset; a candidate **order** is not a scalar, and giving it a second grammar on the
environment would be a new vocabulary rather than another spelling. So the preference is
flag-only, and naming it without a flag fails loudly rather than being half-supported. An unset
flag leaves the operator's environment variable alone.

#### `--kv-codec-preference` — the same-bit axis of the slider

`--kv-bits` says **how many bits**; `--kv-codec-preference` says **which codec** at that same bit
count. It is an ordered, comma-separated list over the candidate grammar
`rk4v4,bf16,int8,fp8,nvfp4,iso4e` (most wanted first), e.g. `--kv-codec-preference iso4e` or
`--kv-codec-preference iso4e,nvfp4`. It changes nothing about the bit arithmetic: the solver still
maximises bits under the ceiling and then minimises its own penalty.

That last point is the whole contract, and it is why the flag can refuse:

* A preference is **not a penalty override**. It only breaks a tie its own columns already call
  equal. Under the shipped single-penalty ladder the `iso4e` row carries a deliberate policy pin
  (`penalty 200` against `nvfp4`'s `30`), so
  `--kv-bits 4.5 --kv-codec-preference iso4e` returns `0-15:nvfp4` **by design** — and because
  "accepted and read by nothing" is this project's worst outcome, that outcome is a **refusal**
  naming the cause and the fix, not a silent no-op.
* Make the two candidates equal on the columns the fit minimises and the preference decides:
  `--kv-quality-weight 0` leaves the default table's speed column, which rates `nvfp4` and `iso4e`
  identically (`114`), so at weight `0` they tie and the order breaks the tie. At 16 layers and
  `--kv-bits 4.5` that is the difference between `0-15:nvfp4` and `0-15:iso4e` — same `4.5000`
  bits, same `176.25 MiB` of device KV, different codec.
* An unknown name, an empty element, a repeat, a request with no `--kv-bits`, a
  `--kv-layer-storage` or `--kv-bit-budget` request, a request that resolves to a per-plane
  reading, and a preference the fit could not honour are each **refused by name** with the
  accepted list attached. Nothing is accepted and ignored.

### Exactly one way to say the ceiling

Every combination below is a **contradiction, not a merge**, and both front ends refuse it with the
reason named, before any model work starts. The rule in every case is the same: a ceiling is
resolved into exactly the per-layer table the other option provides, so one of the two would be
accepted and read by nothing.

- `--kv-dtype` with any ceiling — `--kv-dtype` pins the global tier and a ceiling resolves into the
  per-layer table that would replace it. (Measured before this was refused:
  `--kv-bit-budget 4.5 --kv-dtype int8` and `--kv-bit-budget 4.5` produced the identical dtype *and*
  the identical payload — the `int8` was accepted and dropped — while `--kv-dtype int8` alone is a
  different payload.)
- `--kv-bit-budget` with `--kv-bits` / `--kv-k-bits` / `--kv-v-bits` — two spellings of the same
  ceiling set.
- `--kv-bit-budget` or the ceiling family with `--kv-layer-storage`.
- `--kv-bits` with `--kv-k-bits` / `--kv-v-bits` — `--kv-bits` is the ONE overall ceiling.
- `--kv-bits-mode` with `--kv-bits` — a single ceiling has no second reading for the mode to select.
- `--kv-bits-mode` with no ceiling at all.
- `--kv-quality-weight` with no ceiling to act on — it is read only inside the bit-budget fit.
- `--kv-tier-scores` / `--kv-k-tier-scores` / `--kv-v-tier-scores` with no
  `--kv-quality-weight`: a table's two columns are combined only by the weight
  (`penalty = w*quality + (1-w)*speed`), so without one the shipped single-penalty ladder runs and
  the table would be read by nothing.
- `--kv-k-tier-scores` / `--kv-v-tier-scores` on a request that resolves to the **joint** reading —
  one ceiling, one ladder, so a per-plane table has nothing to fit.
- `--kv-codec-preference` with no `--kv-bits` — the preference travels in the joint fit's request,
  and only the `--kv-bits` resolution carries one. `--kv-bit-budget` and `--kv-layer-storage` fill
  the per-layer table directly, so on those the preference would be read by nothing.
- `--kv-codec-preference` on a request that resolves to a **per-plane** reading
  (`--kv-k-bits`/`--kv-v-bits` with `--kv-bits-mode split|ceiling`) — each plane is fitted in its
  own budget and then reconciled per layer, so a preference would be carried into a fit that can
  still refuse it. The mirror image of the per-plane-table rule above, refused for the same reason.

### Weight offload

| Option | Meaning | Default (unset = absent) |
|---|---|---:|
| `--weight-host-bytes N` | Host-resident weight budget | `0` |
| `--weight-device-arena-bytes N` | Device arena for the offloaded set | `0` |
| `--weight-span-floor-bytes N` | minimum span worth offloading | `0` |
| `--weight-prefetch-layers N` | layers prefetched ahead of use | unset |

Startup contradictions are refused here; the runtime strides are the consumer's to report.
`--weight-device-arena-bytes` / `--weight-span-floor-bytes` need a positive `--weight-host-bytes`,
and `--weight-device-arena-bytes` must be strictly smaller than `--weight-host-bytes`, or the offload
would free no device memory.

### Draft trees

`--draft-tree L,d` selects an MTP **tree** verify shape (see the table in
[Speculative decoding](#speculative-decoding)). A tree round's mask-driven accept reads the
verifier's own argmax, which only `--greedy` produces, so a tree round on a sampling or penalty
route is refused on the host before the round begins rather than verified with the masks silently
ignored. `--draft-tree` is not accepted by `ninfer-serve`; see
[HTTP serving](serving.md#mtp-draft-trees-are-not-expressible-over-http).

### Cold tier

`NINFER_KV_DROP_LAYERS` names layers whose KV planes are **discarded** entirely — the layer's KV
storage reports as `dropped`, and anything that asks such a layer for a storage tier or a `DType` is
**refused with a message naming the discard**, rather than being classified as `bf16`. That
distinction matters because `bf16` is also the per-layer "inherit the global `--kv-dtype`" sentinel,
so reporting a discarded layer as `bf16` would simultaneously claim the layer is bf16 and instruct
the engine to ignore `--kv-dtype` on it. `NINFER_KVDUMP_DIR` selects the directory KV dumps are
written to. `NINFER_KV_CALIB_DIR` and `NINFER_KV_CALIB_MAX_TOKENS` control KV calibration input and
its token budget.

### FreeToken observation

`--ft-stats on|off` enables the per-layer attention-energy observation and commits the explicit flag
to `NINFER_FT_STATS`. An unset flag leaves the operator's `NINFER_FT_STATS` alone, which is what
keeps "environment only" working. Related environment knobs: `NINFER_FT_BW_GOV`,
`NINFER_FT_BW_TRACE`, `NINFER_FT_RELOAD_SECS`, `NINFER_FT_FULL_ATTN_LAYERS`,
`NINFER_FT_DEEP_FRAC`, `NINFER_FT_VRAM_AXIS`, `NINFER_FT_COLD_MODE`, `NINFER_FT_COLD_PAGES`.

### Adaptivity

`NINFER_MTP_ADAPTIVE` is the environment spelling of the adaptive MTP ladder that
`--draft-tokens 0` selects. An explicit `--draft-tokens` value wins over it, per the
`CLI > environment > default` order above: with the flag absent the environment selects the ladder,
with `--draft-tokens 0` the flag selects it, and with `--draft-tokens N` (`N > 0`) the flag pins width
`N` and the environment's ladder request is overridden. A backend left unset (or written `auto`) is
promoted to `mtp` by that variable; an explicit `--spec none|dflash|dflash2` is never rewritten.

The ladder decides the width per round rather than once per request. Each round's accept record is
folded into the criterion in `src/targets/qwen3_6/impl/runtime/mtp_window_cut.h`, which re-prices the
cut, and the captured width is re-decided every `kMtpWidthRedecisionRounds` rounds -- so a request
shorter than that cadence makes its first decision its only one. Before the criterion has any
measurement the source stages an initial rung: the widest captured rung not above
`kMtpWindowInitialRung`. It is deliberately **not** the ladder top; that header's own note on
`kMtpWindowInitialRung` records the measured top-vs-start cost, the staged-top observations and the
open questions behind the choice. That staging is a **source landing whose effect has not been
measured yet**: no run has been made on a binary carrying it, and a binary built before the change
stages the ladder top on round 1. `NINFER_MTP_WINDOW_TRACE=1` prints the chosen rung whenever it is
re-decided or changes (`=2` every round) together with the raw cut, the survival, the hit rate and the
threshold behind it -- that trace is the instrument for measuring the effect, and this document makes
no speed claim for it.

### Building and inspecting the tier surface

`NINFER_WARN_SHADOW_SWITCH=off|report|error` (CMake cache variable, default `off`) and
`NINFER_WARN_ERROR_CHECKS` (default `switch`) turn the switch-coverage gate on. `report` adds
`-Wswitch -Wswitch-enum` for CXX only and is safe on this tree; `error` additionally applies
`-Werror` for each named check and is a hard gate. Do **not** name `switch-enum` in
`NINFER_WARN_ERROR_CHECKS`: this tree has 767 `-Wswitch-enum` warnings over 19 files, most of them
switches that name part of their enumeration on purpose. See
[KV storage names and the switch-coverage gate](maintainer/kv-storage-names-and-switch-gates.md).

`NINFER_W13_STATS` reports weight-residency statistics. `NINFER_BUILD_APPS`, `NINFER_BUILD_SERVE`,
`NINFER_BUILD_PROMPT_INPUT`, `NINFER_BUILD_MEDIA_ACQUIRE` and `NINFER_BUILD_BENCHMARKS` are the
CMake cache variables that select which front ends a build produces.

### The build's arch list, and the unknown-arch warning

Two names around the capability gate are easy to misread, so both are stated here.

`NINFER_BUILD_CUDA_ARCHES` is **not an environment variable**. It is a **preprocessor macro** the build
exports from `CMAKE_CUDA_ARCHITECTURES` (`set(... CACHE STRING "CUDA architectures to build")`, default
`120a`), and `#ifdef NINFER_BUILD_CUDA_ARCHS` in `src/core/arch_caps.h` -- the spelling the build
actually writes; `..._ARCHES` is still accepted as a second spelling -- is the only way to see it.
Setting the environment variable of that name does nothing. It exists because a compile-time `-a` suffix
such as `sm_120a` is **not observable through `cudaDeviceProp`**, which reports only `12.0`; so the arch
list the binary was built for is the only way a run can name the arch-accelerated target of the binary
it is in. When the build does not export it, the capability text says so and names
`CMAKE_CUDA_ARCHITECTURES` as the thing to set, rather than printing nothing.

⚠️ **The name in the text above is not the name the build defines.** `src/CMakeLists.txt` exports
`NINFER_BUILD_CUDA_ARCHS` (**no `E`**), and scopes it to one translation unit:

```cmake
string(REPLACE ";" "," NINFER_CUDA_ARCHS_TEXT "${CMAKE_CUDA_ARCHITECTURES}")
set_source_files_properties(core/device_probe.cu PROPERTIES
  COMPILE_DEFINITIONS "NINFER_BUILD_CUDA_ARCHS=\"${NINFER_CUDA_ARCHS_TEXT}\"")
```

`src/core/arch_caps.h` reads `NINFER_BUILD_CUDA_ARCHES` (**with `E`**). In a bounded probe of the eight
files that own this name, the no-`E` spelling occurs only in `src/CMakeLists.txt` (the definition) and in
`src/core/device_capabilities.h` (which uses it for its own `build_architectures`), while the with-`E`
spelling occurs only in `src/core/arch_caps.h` (four times) and once as a string literal in
`tests/test_arch_caps.cpp`. So `#ifdef NINFER_BUILD_CUDA_ARCHES` cannot be true in a build this tree
produces, `build_arch_list()` returned its empty branch. **This is fixed.** The reader looks for the
name the build writes (`NINFER_BUILD_CUDA_ARCHS`), so it returns the build's own list wherever the
macro reaches the translation unit; the CMake side gains an `INTERFACE` target,
`ninfer_build_cuda_arches`, on the same blast-radius rule the tree's other build facts use, and a
target that links it gets the fact. Two consequences of the old state are worth keeping, because
neither was known when this paragraph was written:

* **the prescribed remedy was a no-op.** The sentence that told a reader to "rebuild with
  `CMAKE_CUDA_ARCHITECTURES` exported as `NINFER_BUILD_CUDA_ARCHES`" named a macro this build cannot
  define (it is defined on **0** of the 573 recorded compile commands), so an operator who followed
  the instruction rebuilt and read the **same line**, and nothing anywhere reported that the remedy
  had failed. The message now names what to change and says that a rebuild alone does not change it;
* **two surfaces contradicted each other in one run**: the capability probe printed the real list
  (`120a`) while this report printed `<unreported>`. They now read the same value.

The capability probe itself was **never** affected — it uses the no-`E` spelling and is the
translation unit the CMake definition reaches — so the arch list does reach the engine through
`src/core/device_capabilities.h`; only the second, doc-facing spelling is dead.

`NINFER_ARCH_WARN` **is** a real environment variable, and it is the only `getenv` in
`src/core/arch_caps.h`. By default a compute capability with no row in the ladder produces **no output
at all**: the gate proceeds on the conservative route and refuses only the shapes it has no kernel for,
so an unknown card degrades instead of stopping. Set `NINFER_ARCH_WARN` (any value) and the same verdict
is written to **stderr** at the moment an artifact's formats are checked. Unlike the KV options, no flag
leaves this one alone — there is no flag for it.

### Multi-device

`--device N` selects **one** CUDA device index, and that is the whole of the current operator surface for
device placement: there is no rank, no world size, no collective and no shard option. The multi-device
arithmetic (the rank/world shape, the tensor-vs-pipeline split rules, the per-rank persistence naming,
and the one predicate the KV policy needs before a page's residency bit can survive a split) has landed
as a host-only, single-GPU-testable contract with **no engine call path yet**, alongside the
per-architecture GEMM route selector that replaced the old `sm() != 120` throw. See
[Multi-device sharding, the arch route table, and the capability gate](maintainer/multi-device-and-shard-plan.md);
that document names what is missing before two cards can run, and which of the three host tests around it
are still unregistered.
