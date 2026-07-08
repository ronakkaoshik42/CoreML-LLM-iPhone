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
- `BridgeRunner.swift` runs the validated fixed-image bridge and formats its
  result line.
- `conversion/build_qwen35_vision.py` builds square or rectangular Core ML
  vision buckets.
- `conversion/qwen35_vision_parity.py` validates PyTorch/Core ML parity and
  creates matching pre-patchified inputs.
- `conversion/probe_qwen35_vision_ane.py` isolates ANE numerical drift.

## Excluded artifacts

GGUF, mmproj, `.mlpackage`, `.mlmodelc`, `.bin`, device logs, benchmark output,
DerivedData, signing material, and app containers are intentionally excluded.

## Reproduce the iPhone proof

This section reproduces the validated **fixed-image engineering proof**. It is
not yet a general image-picker app: the Core ML input tensor is prepared on the
Mac for one 768x576 image. Live preprocessing and multiple aspect-ratio buckets
are subsequent integration work.

### 1. Requirements

- Apple-silicon Mac with Xcode and command-line tools
- Physical iPhone running iOS 18 or newer, with Developer Mode enabled
- iPhone paired, connected, unlocked, and trusted by the Mac
- Python 3.12 virtual environment
- CMake and Git LFS
- At least 15 GB of free Mac storage
- Hugging Face access for the selected model repositories

The validated environment used Xcode 26.6, Python 3.12.13, coremltools 9.0,
PyTorch 2.7.0, Transformers 5.13.0.dev0, and llama.cpp commit
`bec4772f6a2527d371557b5d2032641e5ff7619c`.

### 2. Clone and install Python dependencies

```bash
git clone https://github.com/ronakkaoshik42/CoreML-LLM-iPhone.git
cd CoreML-LLM-iPhone

python3.12 -m venv .venv
.venv/bin/pip install coremltools==9.0 torch==2.7.0 \
  transformers safetensors pillow numpy huggingface_hub
```

### 3. Download model inputs locally

Do not add these files to Git.

```bash
mkdir -p output/qwen35-coreml-hf output/qwen35-models

.venv/bin/hf download Qwen/Qwen3.5-4B \
  config.json model.safetensors.index.json preprocessor_config.json \
  model.safetensors-00002-of-00002.safetensors \
  --local-dir output/qwen35-coreml-hf

.venv/bin/hf download unsloth/Qwen3.5-4B-GGUF \
  Qwen3.5-4B-Q4_K_M.gguf mmproj-F16.gguf \
  --local-dir output/qwen35-models
```

The converter intentionally loads only the safetensors shard containing
`model.visual.*`. Verify the index if the upstream checkpoint layout changes.

### 4. Build the aspect-preserving Core ML vision model

For the validated 4:3 image, keep the long edge at 768 and use 768x576:

```bash
.venv/bin/python conversion/build_qwen35_vision.py \
  --model-dir output/qwen35-coreml-hf \
  --out-dir output/qwen35-coreml-vision \
  --image-width 768 --image-height 576

mkdir -p output/qwen35-coreml-vision/compiled-768x576
xcrun coremlcompiler compile \
  output/qwen35-coreml-vision/qwen35_vision_768x576.mlpackage \
  output/qwen35-coreml-vision/compiled-768x576
```

The converter's ANE placement audit does not establish numerical correctness.
Run parity using the same source image before deploying:

```bash
.venv/bin/python conversion/qwen35_vision_parity.py \
  --model-dir output/qwen35-coreml-hf \
  --coreml output/qwen35-coreml-vision/qwen35_vision_768x576.mlpackage \
  --image /absolute/path/to/candy.png \
  --image-width 768 --image-height 576
```

Use the CPU+GPU result. Do not use the monolithic ANE result unless it passes
parity on your toolchain and device.

### 5. Create the pre-patchified proof input

The Swift proof expects raw little-endian FP16 data named
`qwen35_vision_768x576_input_f16.bin`:

```bash
.venv/bin/python -c "import sys; from pathlib import Path; \
sys.path.insert(0, 'conversion'); \
from qwen35_vision_parity import patchify; \
patchify(Path('/absolute/path/to/candy.png'), 576, 768).tofile(\
'output/qwen35-coreml-vision/qwen35_vision_768x576_input_f16.bin')"
```

Also retain the same source image as `candy-768-valid.png`; llama.cpp uses it
only to create matching multimodal chunk/grid metadata in the current proof.

### 6. Build llama.cpp for iOS

```bash
mkdir -p third_party
git clone https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
git -C third_party/llama.cpp checkout \
  bec4772f6a2527d371557b5d2032641e5ff7619c

cd third_party/llama.cpp
./build-xcframework.sh
cd ../..
```

Start from `third_party/llama.cpp/examples/llama.swiftui`. Add
`CoreMLVisionBench.swift`, `VisionProof.swift`, and `BridgeRunner.swift` to its
app target, and link/embed `third_party/llama.cpp/build-apple/llama.xcframework`.
Set your own signing team and a unique bundle identifier; do not copy the
author's signing values.

In the app state's initializer, add a launch hook equivalent to:

```swift
if ProcessInfo.processInfo.arguments.contains("--run-qwen35-coreml-bridge-proof") {
    Task {
        let documents = FileManager.default.urls(
            for: .documentDirectory, in: .userDomainMask)[0]
        let result = await Qwen35BridgeRunner.run(documents: documents)
        print(result)
        try? (result + "\n").write(
            to: documents.appendingPathComponent("qwen35_proof_results.log"),
            atomically: true, encoding: .utf8)
    }
}
```

Build and install the app from Xcode once to establish signing and its data
container.

### 7. Copy local artifacts to the app container

Find the device identifier:

```bash
xcrun devicectl list devices
```

Set local shell variables, replacing both placeholders:

```bash
DEVICE_ID='YOUR-DEVICE-UUID'
BUNDLE_ID='your.unique.bundle.identifier'
```

Copy the five required local files. These commands do not belong in Git:

```bash
xcrun devicectl device copy to --device "$DEVICE_ID" \
  --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
  --source output/qwen35-models/Qwen3.5-4B-Q4_K_M.gguf \
  --destination Documents/Qwen3.5-4B-Q4_K_M.gguf

xcrun devicectl device copy to --device "$DEVICE_ID" \
  --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
  --source output/qwen35-models/mmproj-F16.gguf \
  --destination Documents/mmproj-F16.gguf

xcrun devicectl device copy to --device "$DEVICE_ID" \
  --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
  --source output/qwen35-coreml-vision/compiled-768x576/qwen35_vision_768x576.mlmodelc \
  --destination Documents/qwen35_vision_768x576.mlmodelc

xcrun devicectl device copy to --device "$DEVICE_ID" \
  --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
  --source output/qwen35-coreml-vision/qwen35_vision_768x576_input_f16.bin \
  --destination Documents/qwen35_vision_768x576_input_f16.bin

xcrun devicectl device copy to --device "$DEVICE_ID" \
  --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
  --source /absolute/path/to/candy-768-valid.png \
  --destination Documents/candy-768-valid.png
```

Do not reinstall with a different bundle identifier after copying files; that
creates a different app data container.

### 8. Run and collect the result

Keep the phone unlocked. For comparable thermal measurements, keep charging
state and ambient conditions consistent across runs.

```bash
xcrun devicectl device process launch --device "$DEVICE_ID" \
  --terminate-existing "$BUNDLE_ID" \
  --run-qwen35-coreml-bridge-proof

xcrun devicectl device copy from --device "$DEVICE_ID" \
  --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
  --source Documents/qwen35_proof_results.log \
  --destination qwen35_proof_results.log

cat qwen35_proof_results.log
```

A successful result starts with `[QWEN35_COREML_BRIDGE_RESULT]`. Decode the
answer if needed:

```bash
sed -E 's/.*output_b64=//' qwen35_proof_results.log | base64 --decode
```

### Known limitations

- The proof uses one precomputed 768x576 image tensor.
- Other aspect ratios need matching Core ML buckets and preprocessing.
- The F16 mmproj is still loaded for metadata, duplicating vision weights.
- Core ML model load is included separately from warm vision latency.
- The full vision transformer currently fails numerical parity on ANE.
- Long continuous runs require thermal and power-state controls.
