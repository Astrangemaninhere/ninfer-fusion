<!-- Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion). -->
# Qwen3.8-Flash-Next Oracle & State Diagnostic Harness

This directory provides the authoritative CPU FP32 reference forward pass and state divergence analysis harness for Qwen3.8-Flash-Next.

## 1. Environment Setup

Run the setup script using Python 3.14 to create the isolated virtual environment:

```powershell
.\setup_env.ps1
```

This creates the isolated venv at `E:\NInfer\venv-qwen4exp` with PyTorch (CPU), safetensors, numpy, and transformers.

## 2. Generating State Dumps

### Deliverable A (NInfer C++ Engine Dump)
Generate raw stage tensor dumps for single token or prompt execution using the reference tool:

- **Single Token Execution (e.g. `<|im_start|>` token 248045):**
  ```powershell
  $env:PATH = "P:\third_party\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;" + $env:PATH
  .\build-win\tools\reference\qwen3_8_flash_next\ninfer_qwen3_8_flash_next_reference.exe `
    -m E:\NInfer\qwen3_8_flash_next.ninfer `
    --execute-token --token-id 248045 `
    --dump-states P:\dumps\ninfer_tok0
  ```

- **Chat Diagnostic Prompt Prefill:**
  ```powershell
  .\build-win\tools\reference\qwen3_8_flash_next\ninfer_qwen3_8_flash_next_reference.exe `
    -m E:\NInfer\qwen3_8_flash_next.ninfer `
    --chat-diagnostic --prompt "Hello" `
    --dump-states P:\dumps\ninfer_chat
  ```

### Deliverable B (Python CPU Oracle Dump)
Generate reference tensors for the same token(s):

```powershell
E:\NInfer\venv-qwen4exp\Scripts\python.exe run_oracle.py `
  --token-id 248045 `
  --dump-states P:\dumps\oracle_tok0
```

For multi-token sequence:
```powershell
E:\NInfer\venv-qwen4exp\Scripts\python.exe run_oracle.py `
  --ids "248045,846,198,20206" `
  --dump-states P:\dumps\oracle_seq
```

### MTP reference qualification

`mtp_reference.py` implements the MTP stem and wraps one Transformers decoder layer,
following the [vLLM MTP implementation](https://github.com/vllm-project/vllm/blob/main/vllm/models/qwen4_exp/nvidia/mtp.py).
The hidden normalization covers all 10,240 features, its shared projection preserves
four distinct streams, and the sole mixer follows the decoder layer. The first draft
pairs the last target hidden with the target's next token; later drafts carry MTP hidden.

Run the focused comparison from the repository root with the environment above:

```powershell
E:\NInfer\venv-qwen4exp\Scripts\python.exe tests/targets/qwen3_8_flash_next/test_mtp_oracle_parity.py `
  --ninfer-exe build-win/tests/Release/ninfer_qwen3_8_flash_next_mtp_test.exe `
  --dump-dir profiles/bench/flash-next-mtp/oracle-parity
```

This compares the five stem stages against the FP32 reference under shared synthetic
inputs and weights, and checks the remaining C++ stages for finite, nonzero values.
It does not compare decoder-layer or head values: their synthetic weights differ.
Zero indexer counts are legitimate diagnostics. `--stem-only` runs the pure tensor
semantics check; full mode fails explicitly if Qwen4Exp Transformers is unavailable.

The September 6 C++ comparison passed with maximum stem relative L2 error 0.002428.
The synthetic Transformers MTP forward also ran successfully. `run_oracle.py --mtp-real`
supports real-weight teacher seeding and chained drafts, but full real-checkpoint CPU
MTP parity has not been run and is not implied by those focused checks.

## 3. Comparing States and Finding First Divergence

Run `compare_states.py` to compare stage tensors across positions:

```powershell
E:\NInfer\venv-qwen4exp\Scripts\python.exe compare_states.py `
  P:\dumps\ninfer_tok0 `
  P:\dumps\oracle_tok0 `
  --threshold 0.05
```

The script reports $\max |d|$, $\text{rel-L2}$, and $\text{cosine}$ similarity per stage and immediately isolates the exact layer and operator where the numerical divergence starts.
