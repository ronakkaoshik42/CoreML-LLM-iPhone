# Qwen3.5-4B iPhone Optimization Handoff

Updated: 2026-07-08

## Objective

Run Qwen3.5-4B multimodal inference locally on an iPhone 16 Pro Max, preserving
image quality and practical MMMU capability while reducing sustained latency,
memory pressure, and thermal throttling.

The target workload is **100 unique images using the same prompt**. Model-load
time is normally paid once per app process. Visual-embedding caching does not
help this workload because every image is unique.

## Non-negotiable project rules

Read `AGENTS.md`, `CLI_HANDOFF.md`, `BENCHMARKING.md`, and
`BENCHMARK_AUTOMATION.md` before editing.

- Never commit model blobs, GGUF, mmproj, `.mlpackage`, `.mlmodelc`, or `.bin`.
- Do not change the production app bundle identifier.
- Do not delete, move, or replace model files on the Mac or iPhone.
- Preserve existing 4B/8B scripts and bundle-ID alignment.
- Inspect the worktree before edits and stage explicit paths only.
- Change one performance variable at a time.
- Treat wrapper-level/second `[RESULT]` as source of truth where applicable.
- Validate 4B text before expanding production benchmark behavior.
- Do not resume the paused 8B work unless explicitly requested.

## Repository state

- Branch: `main`
- Public repository: <https://github.com/ronakkaoshik42/CoreML-LLM-iPhone>
- Public remote name: `public`
- Private working mirror remote: `origin`
- Upstream remote: `upstream`
- Latest public commit: `e48c67a` (`Document Qwen3.5 iPhone proof setup`)
- Local `main` is two commits ahead of the private `origin/main`; those commits
  are already pushed to `public/main`.

Expected dirty file, unrelated to current Qwen3.5 work:

```text
 M conversion/build_qwen3_vl_2b_stateful_multifunction.py
```

This is paused 8B work. Preserve it and do not stage or revert it.

## Hardware and validated software

- iPhone 16 Pro Max
- Physical device, not Simulator
- Qwen3.5-4B Q4_K_M GGUF
- Qwen3.5 F16 mmproj baseline
- llama.cpp commit `bec4772f6a2527d371557b5d2032641e5ff7619c`
- Xcode 26.6
- Python 3.12.13
- coremltools 9.0
- PyTorch 2.7.0
- Transformers 5.13.0.dev0

Keep charging state and ambient conditions consistent across comparisons. Every
accepted sustained comparison below started at thermal state 0 (nominal) and
ended at state 2 (serious).

## Current architecture

### Language model

- llama.cpp iOS XCFramework
- Qwen3.5-4B Q4_K_M
- Metal offload (`n_gpu_layers = 999`)
- Context 2048
- Logical batch 512
- Physical microbatch 256
- Flash Attention left on llama.cpp default/automatic mode
- Six CPU threads for context and batch work

### Original vision path

- llama.cpp `mtmd`
- F16 mmproj
- Metal vision encoding
- Aspect-preserving dynamic image grid

### Core ML proof path

- Official Qwen3.5 vision weights converted from the shard containing
  `model.visual.*`
- Core ML CPU+GPU execution; do not use the full tower on ANE
- Aspect-preserving 768x576 bucket for the validated 4:3 image
- 432 final visual embeddings of width 2560
- External embeddings passed to llama.cpp using
  `mtmd_helper_decode_image_chunk`
- Multimodal RoPE positions still come from mtmd metadata

The current bridge temporarily loads both the Core ML vision tower and F16
mmproj. This is correct but duplicates roughly 672 MB of vision weights.

## Tracked proof source

- `Experiments/Qwen35iOSProof/README.md`
- `Experiments/Qwen35iOSProof/CoreMLVisionBench.swift`
- `Experiments/Qwen35iOSProof/VisionProof.swift`
- `Experiments/Qwen35iOSProof/BridgeRunner.swift`
- `conversion/build_qwen35_vision.py`
- `conversion/qwen35_vision_parity.py`
- `conversion/probe_qwen35_vision_ane.py`

The README contains fixed-image reproduction, conversion, Xcode, installation,
launch, and collection steps.

## Local-only proof app and artifacts

These are ignored by Git and must never be staged:

```text
output/qwen35-mac-proof/ios-proof/
output/qwen35-mac-proof/models/Qwen3.5-4B-Q4_K_M.gguf
output/qwen35-mac-proof/models/mmproj-F16.gguf
output/qwen35-coreml-hf/
output/qwen35-coreml-vision/
```

Important compiled/proof names:

```text
qwen35_vision_768.mlmodelc
qwen35_vision_768x576.mlmodelc
qwen35_vision_768x576_input_f16.bin
Qwen3.5-4B-Q4_K_M.gguf
mmproj-F16.gguf
```

The isolated proof app uses the bundle identifier already configured in its
local Xcode project. Do not apply that identifier to the production app.

## Verified performance

### Text-only smoke test

One recorded Qwen3.5 text run:

```text
load_sec=11.618 ttft_sec=1.062 total_sec=1.858 tokens=12 tokps=15.07
```

Text decode has also been observed around 17 tok/s depending on prompt and
thermal state. Do not use this as the image-workload throughput estimate.

### Original 100-image llama.cpp run

This older run used 100 unique COCO images with varied prompts:

| Metric | Result |
| --- | ---: |
| Success | 100/100 |
| Wall time | 789.6 s |
| Prefill total | 529.9 s |
| Generation total | 254.8 s |
| Sustained generation | 10.16 tok/s |
| Working-memory range | approximately 3.6-4.5 GB |

The workload reached serious thermal state. Prefill was approximately 67% of
wall time and generation approximately 32%.

### Core ML bridge, fixed 4:3 image

Validated result:

```text
[QWEN35_COREML_BRIDGE_RESULT]
coreml_load_sec=3.235
vision_sec=1.678
llama_load_sec=4.437
prefill_sec=2.611
generation_sec=0.329
tokens=2
output=candy
```

Original combined image encoding + language prefill was 6.571 s. Core ML
vision plus external-embedding language prefill was 4.289 s, a 34.7% reduction
for this fixed-image proof. This is not yet a live multi-aspect COCO pipeline.

### Core ML numerical parity

| Path | Result | Decision |
| --- | ---: | --- |
| Full tower, CPU+GPU | cosine 0.999927 | Valid |
| Patch embedding, ANE | cosine 0.999972 | Valid |
| First transformer block, ANE | NaN | Invalid |
| Full tower, ANE | cosine 0.050240 | Invalid |

The full-resolution ANE failure begins inside the first transformer block,
not preprocessing or patch embedding. Layer chunking alone cannot fix it.

### Aspect ratio finding

- Forced square 768: 576 visual tokens and distorted non-square images.
- Aspect-preserving 768x576: 432 visual tokens and correct output.
- Keep the 768-pixel long edge; use aspect buckets rather than reducing image
  resolution or forcing a square.

## Completed language-prefill experiment group

All runs below used:

- Same first 20 unique COCO images
- Identical fixed prompt: `Describe this image in one concise sentence.`
- Identical output limit
- Exactly 543 generated tokens
- Nominal starting thermal state
- Serious ending thermal state

| Configuration | Prefill | Generation | Wall | Peak Metal | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Auto Flash, u256 | 65.910 s | 43.453 s | 113.023 s | 3.560 GB | Keep |
| Forced Flash, u256 | 72.855 s | 46.088 s | 120.169 s | 3.560 GB | Revert |
| Auto Flash, u384 | 72.169 s | 46.112 s | 120.839 s | 3.681 GB | Revert |
| Auto Flash, u512 | 92.025 s | 55.793 s | 154.917 s | 3.802 GB | Revert |

Conclusions:

- Leave Flash Attention on automatic; forcing it increased wall time by 6.3%.
- Keep `n_ubatch=256`.
- u384 increased wall time by 6.9%.
- u512 increased wall time by 37.1%.
- Larger microbatches consume more Metal memory and worsen sustained thermal
  behavior on this device.

The ignored local proof source has been restored to automatic Flash and u256,
but the app currently installed on the iPhone may still be the last u512 build.
Rebuild and reinstall before the next measurement.

Relevant local results:

```text
output/qwen35-coreml-vision/perf20_auto_u256_fixed_complete.jsonl
output/qwen35-coreml-vision/perf20_enabled_u256_fixed_complete.jsonl
output/qwen35-coreml-vision/perf20_auto_u384_fixed_retry_complete.jsonl
output/qwen35-coreml-vision/perf20_auto_u512_fixed_complete.jsonl
```

## Rejected experiments and traps

### `n_batch/n_ubatch = 1024/1024`

Short single-image latency looked much better, but the 100-image run was worse:

| Metric | Original | 1024/1024 |
| --- | ---: | ---: |
| Wall | 789.6 s | 979.331 s |
| Prefill | 529.9 s | 637.827 s |
| Generation | 254.8 s | 336.462 s |

It was reverted. Note that charging state differed across the two long runs, so
the exact percentage is not a clean A/B; the sustained regression was still
large and the controlled 20-image microbatch sweep independently confirms that
wider microbatches are worse.

### Visual-embedding caching

Useful for repeated prompts on the same image, but irrelevant for 100 unique
images. Identical prompt tokenization saves only milliseconds. Tokens after the
image cannot reuse KV state because they attend to different image embeddings.

### Reducing image resolution

Not authorized. It risks MMMU and OCR/detail performance. Use long-edge 768 and
aspect-preserving buckets.

### Full ANE tower

Do not integrate it. Compiler placement succeeded but numerical parity failed.

### Limiting output tokens

Only do this if the real task requires short answers. Do not shorten outputs to
artificially improve a benchmark.

## Logical next priorities

### 1. Restore and verify the winning language settings

Before any new benchmark:

1. Confirm ignored `VisionProof.swift` has automatic Flash and u256.
2. Rebuild and reinstall the isolated proof app because the installed build may
   still use u512.
3. Run 4B text smoke validation.
4. Run a short image smoke test and verify correct output.

### 2. Implement live Core ML image preprocessing with aspect buckets

Current Core ML proof consumes a Mac-generated raw tensor. Build a real iOS
preprocessor that matches `conversion/qwen35_vision_parity.py` exactly:

- EXIF orientation
- Bicubic resize
- Preserve aspect ratio
- Normalize RGB from `[0,255]` to `[-1,1]`
- Duplicate temporal frame because temporal patch size is 2
- Patch size 16
- Spatial merge size 2
- Exact merged-patch row ordering

Start with common buckets sharing long edge 768, for example:

- 768x768
- 768x576 and 576x768
- 768x512 and 512x768

Do not ship multiple 636 MB copies blindly. Investigate a multi-function or
enumerated-shape Core ML package with shared weights. Validate every bucket
against PyTorch and llama.cpp token count before timing.

### 3. Measure Core ML vision separately on unique images

For each bucket:

- One model warmup
- At least 20 unique images
- Report mean, median, p95, thermal state, Metal memory, and process footprint
- Compare CPU+GPU only
- Verify output parity before performance

The fixed 768x576 tower measured 1.678 s for one image, but sustained behavior
is not yet known.

### 4. Remove duplicate mmproj residency

The current bridge uses mmproj only to create multimodal chunk/grid metadata.
For a lightweight implementation, add a narrow isolated llama.cpp helper that
accepts external embeddings plus explicit grid dimensions/M-RoPE positions.
Then avoid loading mmproj in the Core ML path.

Expected benefit: lower load time, disk footprint, and roughly 672 MB less
vision-weight residency. This is primarily a memory/enabler improvement, not a
direct 1.678 s encoding reduction.

### 5. Pipeline across unique images

After duplicate mmproj memory is removed, overlap:

- Core ML vision encoding for image N+1
- llama prefill/generation for image N

Use bounded depth 2 first. Measure thermals and memory. Do not assume concurrency
helps; GPU contention may serialize or throttle both stages.

### 6. Only then evaluate multi-image batching

- Core ML batch size 2 within the same aspect bucket
- llama multi-sequence prefill only if memory allows
- Compare sustained wall time, not cold burst latency
- Stop immediately on memory growth, serious thermal acceleration, or output
  differences

## Theoretical workload interpretation

With the fixed-image Core ML proof, per-image work before long-output effects is
approximately:

```text
vision encode     1.678 s
language prefill  2.611 s
generation        workload-dependent; old 100-image mean about 2.55 s
```

For unique images, prompt reuse does not remove the first two stages. Pipeline
overlap can hide some vision time, but reaching less than 100 seconds for 100
full-resolution images is not realistic without substantial batching, a
smaller model, visual-token reduction, or output/task changes. Preserve quality
and report that limit honestly.

## Immediate commands for the next session

Inspect state:

```bash
git status --short --branch
git log -3 --oneline
git remote -v
```

Verify winning proof settings:

```bash
rg -n "flash_attn|n_ubatch|n_batch" \
  output/qwen35-mac-proof/ios-proof/llama.cpp.swift/VisionProof.swift
```

Expected:

```text
n_batch = 512
n_ubatch = 256
no forced flash_attn_type assignment
```

Build the isolated app using the existing local project and signing values, then
install it without changing its bundle identifier. Keep the iPhone unlocked.

## Suggested opening prompt for a new session

```text
Read AGENTS.md, CLI_HANDOFF.md, BENCHMARKING.md,
BENCHMARK_AUTOMATION.md, and QWEN35_IPHONE_OPTIMIZATION_HANDOFF.md.
Inspect git status before editing. Continue Qwen3.5-4B iPhone optimization from
the handoff. Preserve the dirty paused 8B converter file. Do not touch
production bundle IDs or model files. First rebuild/reinstall the isolated proof
with auto Flash and n_ubatch=256, validate 4B text/image, then implement live
Core ML aspect-preserving preprocessing and bucket parity. Change one variable
at a time and report keep/revert.
```
