# Qwen3-VL iPhone Benchmarking

Use a physical iPhone and run the app in Release configuration. In Xcode, choose
**Product → Scheme → Edit Scheme → Run → Build Configuration → Release**. Record
whether each result was collected with the debugger attached or by stopping
Xcode and opening the installed app directly.

Keep the same device, iOS version, image, charger state, and fresh/warm app state
when comparing 4B with 8B. The 8B model appears sensitive to charger and power
state, so begin its tests plugged in from a fresh app launch and do not load 4B
first.

## Test matrix

For text-only runs, do not attach an image. Use:

```text
Say hello in exactly 20 words.
```

For image runs, reuse the same image and use:

```text
Describe this image in one sentence.
```

For scratch inspection, attach the same test image and use:

```text
Inspect for small physical scratches, scuffs, smudges, or surface defects.

Only localized irregular marks count. Ignore clean continuous lines, borders, printed graphics, display edges, shadows, or UI/label elements.

Return exactly:
Classification: [No clear scratches visible / Possible minor scratch/mark visible / Clear scratch visible]
Location:
Evidence:
Uncertainty:

Max 60 words.
```

Run in this order:

1. Fresh launch, load 4B, run text without an image.
2. With 4B still loaded, run image and scratch prompts.
3. Terminate and freshly launch the app, plug in the charger, load 8B, and run text.
4. Only if 8B text succeeds, run its image and scratch prompts.
5. Repeat 8B from a fresh launch without the charger to test power sensitivity.

The current app generation settings are 120 maximum new tokens, temperature
0.0, and repetition penalty 1.2. Do not change them between comparison runs.

## Console output

- `[PERF]` lines show stage timing, physical footprint, available memory, TTFT,
  decode throughput, and peak-memory samples.
- `[RESULT_LOAD]` is the copyable model-load summary. It covers the shared
  generator's embeddings and Core ML chunks. Tokenizer and external vision
  encoder loading occur in the app wrapper; ANE prewarm is timed separately.
- `[RESULT]` is the copyable generation summary for `text` or `image` mode.
  Scratch inspection is reported as image mode; identify it by the prompt in
  your notes.

Result lines are also saved with timestamps in
`Documents/benchmark_results.log`. In the app, open **Bench → Benchmark
Results** to see the latest load and generation entries, then tap **Copy
Results** to copy the full history.

Copy the result lines into this table:

| Date | Device | iOS | Model | Mode | Charger | Fresh app | Load sec | TTFT sec | Tok/s | Peak GB | Debugger | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| | | | 4B | text | yes | yes | | | | | | |
| | | | 4B | image | yes | no | | | | | | |
| | | | 8B | text | yes | yes | | | | | | |
| | | | 8B | image | yes | no | | | | | | |

For cold-versus-warm comparisons, fully terminate the app before a cold load.
For a standalone measurement, install once from Xcode, stop the Xcode run, then
launch the app from the iPhone and capture its logs separately.
