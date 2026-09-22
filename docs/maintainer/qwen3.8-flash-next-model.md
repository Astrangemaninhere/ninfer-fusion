<!-- Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion). -->
# Qwen3.8-Flash-Next model semantics

This reference defines the exact Text mathematics, persistent state, PLE indexing, MTP, and Vision
topology for NInfer's `qwen3.8-flash-next/mixed-nvfp4-fp8-ple-int4` target. The checkpoint is an
early preview of the Qwen4 architecture and uses the Transformers model type `qwen4_exp`; it is not
a Qwen3.6-family Variant.

The mathematical oracle is the Qwen/Hugging Face implementation generated from
[`modular_qwen4_exp.py`](https://github.com/huggingface/transformers/blob/281dd533060988a1de8d063c4c1ea72b304a2bb8/src/transformers/models/qwen4_exp/modular_qwen4_exp.py)
and its generated
[`modeling_qwen4_exp.py`](https://github.com/huggingface/transformers/blob/281dd533060988a1de8d063c4c1ea72b304a2bb8/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py),
evaluated with the revision-pinned source `config.json`. NInfer may fuse operations and choose
different internal arithmetic, but each execution leaf is qualified against these represented
semantics.

## 1. Fixed dimensions and topology

| Fact | Value |
|---|---:|
| vocabulary / Text width | 248320 / 2560 |
| hyper-connection streams / concatenated width | 4 / 10240 |
| Text layers | 48 |
| full-attention layers | `3,7,11,...,47` (12) |
| Gated DeltaNet layers | all other layers (36) |
| routed experts / selected experts | 512 / 10 |
| routed and shared intermediate width | 640 |
| full-attention query / KV heads / head width | 24 / 2 / 256 |
| rotary fraction / rotary width | 0.25 / 64 |
| indexer query / KV heads / head width | 4 / 1 / 128 |
| indexer block width / token budget | 4 / 2048 |
| GDN key heads x width | 16 x 128 = 2048 |
| GDN value heads x width | 48 x 128 = 6144 |
| GDN convolution channels / taps | 10240 / 4 |
| maximum positions / RoPE theta | 262144 / 10000000 |
| PLE injection layer | decoder layer 1 (the second layer) |
| PLE n-gram order / heads per order | bigram and trigram / 8 |
| PLE row width / logical rows | 160 / 320001536 stored rows |
| MTP | one full-attention MoE layer |
| Vision depth / width / intermediate / heads | 27 / 1152 / 4304 / 16 |

Text embeddings are repeated four times to form the initial `[4,2560]` hyper state. After the 48
decoder layers, the final hyper mixer reduces those streams to one 2560-wide state before the
independent output head.

## 2. Normalization and hyper-connections

The Text RMSNorm is one-centered. For each group `x` of the configured group size,

```text
rms(x) = x / sqrt(mean(x^2) + 1e-6)
norm(x,w) = rms(x) * (1 + w)
```

For a concatenated four-stream state `H`, the gated residual computes:

```text
N = group_rmsnorm(H), with four independent groups of width 2560
M = sigmoid(W_up(silu(W_down(N) / 4))) reshaped to [4,2560]
block_input = mean_stream(M * reshape(N,[4,2560]))
injection_weight = 2 * sigmoid(W_inject(N) / 4)
H' = H + flatten(block_output[:,None,:] * injection_weight[:,:,None])
```

Attention and MoE each own one such transition. The target and MTP final mixers omit
`W_inject` and return only `block_input`.

## 3. Gated DeltaNet layers

The input projection produces `[q,k,v,z] = [2048,2048,6144,6144]`; a depthwise causal
four-tap convolution followed by SiLU applies to `[q,k,v]`. The 16 key heads are repeated three
times to align with 48 value heads. `q` and `k` are independently L2-normalized. Per value head,

```text
beta = sigmoid(Bx)
g = -exp(A_log) * softplus(Ax + dt_bias)
S_t = exp(g_t) * S_(t-1)
delta_t = (v_t - k_t^T S_t) * beta_t
S_t = S_t + k_t delta_t^T
o_t = q_t^T S_t
```

The recurrent matrix state is FP32 `[48,128,128]` per sequence and layer. The convolution state is
the previous three `[q,k,v]` values `[10240,3]`. The output applies RMSNorm to each 128-wide value
head, multiplies by its direct ones-initialized gated-norm weight (this weight does not use the
ordinary Text norm's unit offset), multiplies by `sigmoid(z)` (`output_gate_type=sigmoid` in the checkpoint config; Qwen3-Next used `silu`), flattens to 6144, and projects to 2560.
Prefill may use an equivalent chunked delta-rule evaluation; decode must preserve the same state
transition.

## 4. QSA full-attention layers

The main projection stores each query head as `[query_256,output_gate_256]`, followed by 512 key
rows and 512 value rows. Query and key heads use one-centered RMSNorm and 64 rotary dimensions.
After causal sparse attention, each query head output is multiplied by `sigmoid(output_gate)` before
the 6144-to-2560 output projection.

The QSA indexer projects four 128-wide queries and one raw 128-wide key per token. For every visible
prefix, it partitions the complete visible tokens into consecutive four-token blocks and leaves an
incomplete tail uncompressed. Each complete block key is the FP32 mean of its four raw keys,
one-centered RMS-normalized, and rotated at the block's first position. The score for a block is:

```text
score = sum_head(relu(q_head dot pooled_block_key)) / sqrt(128)
```

The indexer selects up to `2048 / 4 = 512` blocks, expands them back to tokens, appends the visible
tail, and intersects that mask with the causal mask. Raw indexer keys, main keys, main values, and
full MRoPE positions are persistent per-sequence state.

NInfer stores each completed block's mean-cast, normalized, MRoPE-rotated key once in a paged
block-key plane, plus the four raw keys and positions of the block currently being formed. This is
the same represented result because a completed consecutive block is immutable. Selection uses
FP32 scores and stable descending order; an exact score tie retains the lower block ID.

Decode evaluates the selected-block attention through the `selected_block_attention` Op, shared
by target decode and MTP. Each query head partitions its visible tokens across 16 thread blocks;
four warps per block form FP32 online-softmax statistics and value numerators using coalesced
feature loads. A second kernel combines the partitions by their maxima and denominators before
the BF16 output cast. Empty selections with no causal tail produce exact zero. The temporary
partial buffer is owned by the caller's workspace, included in both decode and MTP capacity
planning, and retains stable addresses for CUDA Graph replay. This changes the arithmetic
association, not the selected token set or persistent cache state.

The default prefill route uses the same Op with one page-table row shared across its query tokens.
Each thread block processes the 12 query heads sharing a KV head. Its 64-token K tile is dead
after the tensor-core query/key products, so V staging reuses that shared allocation after a
block-wide barrier. A second barrier completes V staging before value accumulation; the final
tile barrier protects the next K overwrite. Query/key products, softmax, and weighted-value
accumulation retain FP32 accumulators, with BF16 input staging and final output storage. The
physical storage reuse reduces static shared memory from 77,824 to 45,056 bytes without changing
the attention formula or introducing another device workspace.

## 5. Mixture of experts

The router computes a 512-way FP32 softmax, selects the top 10 probabilities and renormalizes them
to sum to one (`norm_topk_prob=true`, the transformers Qwen4ExpConfig default; the official
config.json omits the key). For selected expert `e`,

```text
expert_e(x) = down_e(silu(gate_e(x)) * up_e(x))
routed(x) = sum_selected(prob_e * expert_e(x))
shared(x) = sigmoid(shared_gate(x)) * shared_down(silu(shared_gate_proj(x)) * shared_up(x))
moe(x) = routed(x) + shared(x)
```

The production expert-bank kernel may group tokens and use native NVFP4 W4A4, but expert selection,
probability multiplication, and the FP32 softmax remain part of the semantic contract.

## 6. Per-Layer Embedding

PLE has eight bigram and eight trigram heads. The table is 128 physical shards of 2500012 rows;
each logical head selects a global row from its own prime-sized subrange. Token multiplication and
XOR use wrapping signed int64 semantics. For current token `t`, previous segment tokens `p1,p2`,
stored multipliers `m0,m1,m2`, prime vocabulary `v_h`, and offset `o_h`:

```text
b = wrap64(t*m0) XOR wrap64(p1*m1)
g = b XOR wrap64(p2*m2)
row_h = o_h + floor_mod(h < 8 ? b : g, v_h)
```

History begins with `[eos,eos]`; committing EOS resets both history positions. PLE lookup therefore
participates in speculative accept/rollback rather than mutating history during proposal.

The compressed PLE codes and scales remain in read-only host mappings (32,000,153,600 bytes).
After device weights are materialized, startup reads ahead and touches these mappings before
publishing Engine readiness. This moves cold random page faults out of request prefill without
another copy or GPU allocation. These pages remain reclaimable by the operating system.

The 16 selected 160-wide rows concatenate to a 2560-wide embedding `E`. The layer computes a
10240-wide key, a 2560-wide value, and normalized four-stream query. Per stream:

```text
raw_gate = dot(norm(key_stream), norm(query_stream)) / sqrt(2560)
gate = sign(raw_gate) * sqrt(max(abs(raw_gate),1e-6))
V_stream = sigmoid(gate) * value(E)
PLE(H) = V + silu(depthwise_dilated_conv(group_rmsnorm(V)))
```

The convolution has four taps and dilation 3, so its persistent history length is nine. PLE is added
to the four-stream hidden state before layer 1's attention hyper-connection.

## 7. MTP and Vision

MTP pairs target hidden `H[t]` with the text embedding of token `t+1`, at position `t`.
Its stem applies unit-offset RMSNorm to the embedding (width 2560) and to the complete
four-stream hidden (width 10240), then applies the same 2560-by-2560 hidden projection
independently to each stream. The projected embedding is added to each projected stream:

```text
E = fc_embedding(norm(embedding(token[t+1]), embedding_norm))
N = reshape(norm(H[t], hidden_norm), [4,2560])
MTP_input[s] = fc_hidden(N[s]) + E
```

The four streams remain distinct. There is no stem mixer. One full-attention QSA + MoE
layer follows; the sole MTP mixer runs at the end to produce the head input. Later draft
positions carry the MTP layer's post-injection, pre-mixer four-stream hidden, the newly
drafted token, and incremented positions. For image/video placeholders the drafter uses
the text embedding table; visual information reaches it through the target hidden.

The MTP attention and indexer have their own cache planes under the lane's shared physical
page ownership. Target prefill and verification seed these planes with teacher-forced pairs.
The final target hidden and all three MRoPE coordinates are kept per state slot until the
next input token is known. Prompt seeding needs only the stem, attention hyper-connection,
and KV/indexer projections, not the MTP attention output, MoE, or head.

Proposal and commit transactionally include main KV, both indexers' raw keys/positions,
GDN state, PLE token/convolution history, and the saved target hidden/positions. Verification
commits the consumed inputs (anchor plus accepted drafts); its correction/bonus token is
consumed in the next round. The draft window is bounded by context and output budget.
Sequential verification has separate captured graphs from independent-lane decode.

Draft metadata for the whole window is uploaded once to fixed, aligned device views.
The first draft uses the saved target hidden and MRoPE position; subsequent drafts consume
the preceding device token and carried hidden without CPU readback. Each draft position has
startup-captured graphs for the decode context buckets. Bucket selection occurs per step,
including windows that cross the QSA identity/scoring boundary. All draft tokens return in
one pinned transfer after the chain. Draft completion waits are included in the device-wait
timing bucket, alongside verification waits.

With draft window `K`, each active lane owns `K+1` recurrent-state slots (two when MTP is off).
Continuation checkpoint slots begin after the complete active ring. A private owner consists of
its endpoint and optional TurnClosure rewrite. Admission matches only the exact checkpoint offered
by the Engine. Consuming either checkpoint transfers its state to the active lane and releases
the complete old owner; sibling-session retention preserves and protects both checkpoints.
Admission includes physical checkpoint capacity as well as KV capacity. Each admitted request
that publishes a continuation reserves its endpoint and, when planned and the pool has at least
two slots, its TurnClosure rewrite, after planned eviction and source consumption. The pressure
planner accounts for both reservations, so a full pool cannot silently suppress the rewrite needed
by thinking-enabled follow-up requests. Idle lanes reserve none. Capture uses the lane's reserved
rewrite slot; skipping capture releases it. Finish publishes the endpoint and releases unused
reservations, while cancellation returns both reservations. A one-slot pool retains endpoint-only
behavior.

With `preserve_thinking=false`, a new real user message removes reasoning from earlier
assistant messages in the preceding tool-call sequence. Both the latest rolling rewrite and
the endpoint can then fail prefix identity. Flash-Next currently publishes no inherited
turn-start anchor, so this case can require root prefill even when the old owner remains
resident. Keeping reasoning preserves that prefix when the client replays it consistently.

Vision is the checkpoint's 27-block 1152-wide tower: 3D patch projection, learned position
embedding, non-causal 16-head attention, GELU MLP, and a 2x2 patch merger from 4608 to Text width
2560. Text uses interleaved three-axis MRoPE with sections `[11,11,10]`; the QSA indexer additionally
retains full positions for cached raw keys.
