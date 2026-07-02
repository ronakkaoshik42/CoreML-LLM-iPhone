# Codex CLI Handoff — iPhone Qwen3-VL Benchmarks

## Project summary

CoreML-LLM now includes a repeatable iPhone benchmark harness for the stateful
Core ML Qwen3-VL 4B and 8B models. It measures model load time, time to first
token (TTFT), total generation latency, decode tokens/second, available memory,
and process physical footprint. Results are printed, persisted in the app
container, shown in the app, and collectable from a Mac.

## Current working setup

- The `CoreMLLLMChat` iOS app builds, installs, and runs on the physical iPhone.
- Qwen3-VL 4B and 8B model files are sideloaded in the app container.
- Manual and launch-argument benchmark flows work for both models.
- Mac scripts successfully build, install, launch, and collect results through
  `xcrun devicectl`.
- Results persist at `Documents/benchmark_results.log`.
- 8B works more reliably when the iPhone is plugged in. Keep it plugged in and
  unlocked for 8B runs.
- Wrapper-level `[RESULT]` is the source of truth. Successful runs currently
  also emit a nearly identical shared-generator `[RESULT]` immediately before
  it; use the second line.
- The reported `peak_gb=0.40` is process physical footprint and clearly omits
  substantial Core ML model residency. Use Instruments/Memory Report for total
  memory ground truth.

## Important files

- `Sources/CoreMLLLM/CoreMLPerfStats.swift` — timestamps, process/available
  memory, result persistence, and stored-result reading.
- `Sources/CoreMLLLM/Qwen3VL2BStatefulGenerator.swift` — shared 4B/8B load,
  prewarm, generation, TTFT, throughput, and memory instrumentation.
- `Examples/CoreMLLLMChat/CoreMLLLMChat/LLMRunner.swift` — active 4B/8B app
  wrappers and wrapper-level benchmark results.
- `Examples/CoreMLLLMChat/CoreMLLLMChat/BenchmarkSuite.swift` — one-shot 4B/8B
  text/image suite plus launch-argument parsing.
- `Examples/CoreMLLLMChat/CoreMLLLMChat/ChatView.swift` — Bench menu, visible
  status, launch auto-run, results sheet, and Copy Results.
- `scripts/run_iphone_benchmark.sh` — Release build, install, and argument-based
  launch on a connected iPhone.
- `scripts/collect_benchmark_results.sh` — app-container log extraction.
- `BENCHMARKING.md` — prompts, result fields, and manual matrix.
- `BENCHMARK_AUTOMATION.md` — manual, launch-argument, Mac, and collection flows.
- `PLAN.md` — original phased instrumentation and benchmark plan.

## Validated results

| Run | Load | TTFT | Total | Decode | Tokens | Result |
|---|---:|---:|---:|---:|---:|---|
| 4B run 1 | 34.75s | 1.87s | 5.33s | 9.83 tok/s | 34 | success |
| 4B automated | 34.02s | 1.87s | 4.53s | 9.76 tok/s | 26 | success |
| 8B automated, charger connected | 56.98s | 4.46s | 13.76s | 3.76 tok/s | 35 | success |

The automated pipeline was validated end to end: Release build, device install,
argument launch, `[BENCH_DONE] success=true`, and container-log collection.

## Current commands

```bash
bash scripts/run_iphone_benchmark.sh --model 4B --mode text
bash scripts/run_iphone_benchmark.sh --model 8B --mode text
bash scripts/collect_benchmark_results.sh
```

Pass `--device DEVICE_ID` to either script if automatic device selection is
ambiguous. Keep the phone unlocked. Plug it in before running 8B.

Manual fallback:

```text
Bench → Benchmark Results → Copy Results
```

## Next phase

Implement small, reviewable additions in this order:

1. Repeated benchmark runs with configurable count and cold/warm labeling.
2. Run tags containing device condition, charger state, model, mode, and run ID.
3. A summarizer that selects wrapper-level results and reports median/range.
4. A safe clear-results helper that removes only the benchmark log after an
   explicit user action.
5. Controlled optimization experiments, one variable at a time, only after
   baseline repetitions are captured.

Image launch automation still has no default image source. A fresh automated
image run correctly emits `RESULT_ERROR reason=no_image_available`.

## First prompt for Codex CLI

```text
Read AGENTS.md, CLI_HANDOFF.md, BENCHMARKING.md, and BENCHMARK_AUTOMATION.md.
Inspect git status before editing. Continue the iPhone Qwen3-VL benchmark
harness with the next phase only: add repeat-count support and run tags to the
existing BenchmarkSuite and Mac launch script, then add a non-destructive
summarizer that treats the wrapper-level (second) [RESULT] line as source of
truth. Preserve all current 4B/8B behavior and benchmark defaults. Make small,
reviewable changes; do not optimize model execution yet. Validate 4B text first
and stop with exact iPhone commands and a diff summary.
```

## Safety constraints for CLI

- Do not change the bundle ID or development signing setup without approval.
- Do not delete, move, stage, or commit model files.
- Never commit `.mlpackage`, `.mlmodelc`, `.bin`, DerivedData, `.xcresult`, app
  containers, device logs, provisioning profiles, certificates, or secrets.
- Do not run destructive Git or filesystem commands (`git reset`, `git clean`,
  `rm -rf`, model deletion) without explicit approval.
- Preserve the working sideload scripts and model locations.
- Keep edits small and reviewable; use explicit staging paths.
- Measure before optimizing and change one optimization variable at a time.
- Use wrapper-level `[RESULT]` as the benchmark source of truth.
- Remind the user to keep the iPhone plugged in and unlocked for 8B.
