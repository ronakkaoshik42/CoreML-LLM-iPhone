# Qwen3.5-4B iPhone vision proof

This isolated experiment connects a Qwen3.5 Core ML vision tower to the
Qwen3.5-4B Q4_K_M llama.cpp language model using llama.cpp's public external
image-embedding decode helper. It does not alter the production app.

## Validated device

- iPhone 16 Pro Max
- Qwen3.5-4B Q4_K_M GGUF
- F16 multimodal projector for the llama.cpp baseline
- 768-pixel image long edge

## Results

The original llama.cpp path took 6.571 seconds for combined image encoding and
language prefill on the single-image proof. A fixed square 768 Core ML tower was
numerically correct on GPU but produced 576 tokens and distorted non-square
images. Matching llama.cpp's aspect-preserving 768x576 grid produced 432 visual
tokens and the correct answer (`candy`):

| Stage | Seconds |
| --- | ---: |
| Core ML model load | 3.235 |
| Core ML vision encode | 1.678 |
| llama.cpp model load | 4.437 |
| External-embedding language prefill | 2.611 |
| Generation | 0.329 |

Vision plus language prefill was 4.289 seconds, 34.7% below the original
6.571-second path. The proof temporarily loads both Core ML and the original
mmproj to obtain tokenizer/grid metadata. Removing that duplicate projector is
the next memory optimization.

The full monolithic tower is not safe on ANE at this resolution: GPU output
matched PyTorch (`0.999927` cosine), while ANE failed inside the first
transformer block. Patch embedding alone matched (`0.999972`), isolating the
failure to full-resolution transformer attention.

A wider llama.cpp batch (`n_batch/n_ubatch = 1024/1024`) improved short cold
tests but was slower over 100 unique images due to sustained thermal
throttling. It was reverted to `512/256`.

## Source files

- `CoreMLVisionBench.swift` loads a compiled Core ML vision model and converts
  its FP16 output to external llama.cpp embeddings.
- `VisionProof.swift` tokenizes the multimodal prompt and passes external image
  embeddings through `mtmd_helper_decode_image_chunk`, preserving multimodal
  RoPE positions.
- `conversion/build_qwen35_vision.py` builds square or rectangular Core ML
  vision buckets.
- `conversion/qwen35_vision_parity.py` validates PyTorch/Core ML parity and
  creates matching pre-patchified inputs.
- `conversion/probe_qwen35_vision_ane.py` isolates ANE numerical drift.

## Excluded artifacts

GGUF, mmproj, `.mlpackage`, `.mlmodelc`, `.bin`, device logs, benchmark output,
DerivedData, signing material, and app containers are intentionally excluded.
