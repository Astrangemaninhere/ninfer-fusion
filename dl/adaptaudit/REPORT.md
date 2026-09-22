# ADAPT-AUDIT — adaptation coverage for pre-Ada NVIDIA / AMD / Intel / domestic Chinese cards / NPUs

**STATUS: IN PROGRESS**

Audit line: `dl/adaptaudit/`. Repo root established as `/home/user/ninfer-fusion` (WSL2 Ubuntu 2,
reached from the Windows host shell via `wsl.exe -d Ubuntu -- bash -lc "..."`). All paths below are
WSL-side absolute paths.

Read-only w.r.t. the tree: the only writes are under `/home/user/ninfer-fusion/dl/adaptaudit/`.
No build, no lock, no GPU, no engine run.

---

## 1. THE QUESTION

For each required adaptation target —
(1) pre-Ada NVIDIA (sm_52 and up), (2) AMD, (3) Intel, (4) domestic Chinese cards, (5) NPUs —
what does the tree do **today** when it meets one, classified as exactly one of:

- **(i) NAMED refusal** — `file:line` + quoted message.
- **(ii) SILENT degradation** — compiles and runs, fast path skipped / fallback taken, no message names it.
- **(iii) UNREACHABLE** — no code path at all.

## 2. THE LOAD-BEARING SUB-QUESTION

Every place the tree decides *what GPU this is*, classified as
(a) name/probe-based, (b) capability-based, (c) table-driven by name.
Then: which would misclassify a card whose **name is wrong but capabilities are real** (modified
VBIOS, rebadged engineering sample).

## 3. GAP LIST

One row per target: minimum change that moves (ii)/silent → (i)/named, or "already (i)".

---

## 4. PRIOR LINES (cited, not re-derived)

⚠️ **LOCATION CORRECTION.** The brief says the prior lines live under `dl/`. They are **not** under
`/home/user/ninfer-fusion/dl/` — that directory does not exist (probed: `ls -d /home/user/dl
/home/user/ninfer-fusion/dl` → both "No such file or directory"). They live on the Windows side:
`C:\Users\User\Documents\ziqinzhang\dl\{simtprobe,isoname,routeprobe,eolcensus,portablepin,refusals}\`.
This report is written to **both** `/home/user/ninfer-fusion/dl/adaptaudit/REPORT.md` (per the brief's
"write only under `dl/adaptaudit/`", read from WSL) and
`C:\Users\User\Documents\ziqinzhang\dl\adaptaudit\REPORT.md` (co-located with the cited siblings).

All six share: HEAD `3944a53eda1aac439a566a1cf46ea741f0415fdc`, frozen binary `build/apps/ninfer` sha256
→ `b6ddb5eeebbdad70`, and the same owner's-goal paragraph (adaptation across pre-Ada NVIDIA + AMD +
Intel + domestic Chinese cards + NPUs).

- **`dl/simtprobe/`** — *"replacing the `__CUDA_ARCH_LIST__` identity decision with a measured probe"*.
  Lead: the replacement is a 5-hunk/3-file patch turning `gqa_attention_simt_ffma_selected()` from a
  reader of `__CUDA_ARCH_LIST__` into a reader of a byte published by a real launch of
  `ops::gqa_simt_ffma_row_pass` on the bound device; +1.83 s on `src/core/device_probe.cu` at sm_120a
  (6.83 s → 8.66 s, measured); closure 6 launcher objects + the probe TU, with a firing positive control.
  **This is the capability-based replacement for a name-based decision — cite it as the precedent shape.**
- **`dl/routeprobe/`** — *"replacing the route decision with a measurement (measurement closure)"*.
  Lead: a measured capability set that can only NARROW a ladder row, read from the probe report the engine
  already caches; it GATES every Cap bit a probe can `Refute` (so a lying `sm` number cannot make a failed
  probe pass), RANKS and NAMES every bit no probe can answer, and on a never-probed device falls back to
  the build's default row **and names it**. Also owns the **one-letter defect**:
  `NINFER_BUILD_CUDA_ARCHS` (`src/CMakeLists.txt:49-51`, 1 of 573 recorded command lines) vs
  `NINFER_BUILD_CUDA_ARCHES` (`arch_caps.h` reads it, 0 of 573) ⇒ `build_arch_list()` **has always
  returned `""`**; proof is behavioural (control `rc=0 / REFUSALS=0`, four mutants that all compile and
  are caught by the run).
- **`dl/isoname/`** — the `iso3` → `iso4e` rename list. Not about device identity; **not load-bearing here**,
  cited only because `stage1_host.patch` touches `device_capabilities.h` — a file this line also reads.
- **`dl/eolcensus/`** — EOL census + trap register + applier contract. **Not load-bearing here**; cited
  because it establishes the *evidence standard* this fleet uses (anchor-local, byte-preserving, single
  occurrence asserted). It also states the tree is **not** at HEAD in the worktree.
- **`dl/portablepin/`** — a portable `-Wswitch` replacement. Two findings this line cites:
  (1) it **confirms and sharpens to "write-only"** the `NINFER_BUILD_CUDA_ARCHS` defect —
  `device_probe.cu` **does not include `arch_caps.h`** (single-file grep, 0 hits), so the one TU receiving
  the definition is the one TU that cannot read it; (2) site **A**
  `src/core/device_capabilities.h:268` `requirements_for_kv_storage()` falls through to
  `return CapabilityNeeds{kKernelImageBaseline}` — *"a silent POSITIVE CLAIM ('this tier needs no codec
  capability')"*. **That is a (ii)-shaped silent degradation already measured — cite it.**
- **`dl/refusals/`** — audit of the new refusal set *as one set*. Establishes this fleet's refusal
  doctrine verbatim: *"如果确实做不到，必须明确报错"* — **if it truly cannot be done, it must report a CLEAR
  ERROR** — and *"A refusal that does not tell the operator what to do is not a refusal, it is a wall."*
  Its message table scores every refusal on three axes: **names the knob / carries the numbers / names the
  remedy**. **This line adopts that three-axis test as its scoring rubric for (i) vs (ii).**

### What THIS line adds that they do not

None of the six lines answers *"what happens when the engine meets an AMD / Intel / domestic / NPU
device"*. `simtprobe` and `routeprobe` replace **one** name-based decision each with a capability-based
one and measure the closure; they do not enumerate **all** identity decisions, and neither classifies by
vendor. `portablepin` measures coverage of a *KV-storage enumeration*, not of vendors. **This line's
addition: the full census of GPU-identity decision points, the name-vs-capability classification of each,
the five-target behaviour classification, and the gap list.**

## 5. FINDINGS

(none yet)

## 6. MEASUREMENT / READING / NOT CLAIMED

(none yet)
