# iPhone Benchmark Automation

The benchmark suite runs the existing Qwen3-VL 4B or 8B load and generation
paths. It does not change normal chat generation settings. Suite runs default to
64 maximum new tokens and append timestamped markers and results to
`Documents/benchmark_results.log`.

## Manual app flow

In the app, open **Bench** and choose one of:

- **Run 4B Text Benchmark**
- **Run 8B Text Benchmark**
- **Run 4B Image Benchmark**
- **Run 8B Image Benchmark**

Image runs use the image currently selected in chat. Without one, the suite
records `RESULT_ERROR reason=no_image_available` and finishes unsuccessfully.
The orange status strip shows when a suite is running and when it finishes.

Open **Bench → Benchmark Results → Copy Results** to copy the full persisted
history. A successful run contains `BENCH_START`, `RESULT_LOAD` when a model
load was needed, one or more `RESULT` lines, and `BENCH_DONE success=true`.

## Launch arguments

An automated run starts only when `--run-benchmark-suite` is present. Supported
arguments are:

```text
--run-benchmark-suite
--benchmark-model=4B
--benchmark-mode=text
--benchmark-max-new-tokens=64
--benchmark-repeat-count=1
--benchmark-run-tag=device-condition_charger-state
--benchmark-fresh-state-each-run
```

Change `4B` to `8B` or `text` to `image` as needed. A normal launch without the
first flag never auto-runs. Launch-argument image mode currently has no default
image source, so a fresh automated image launch records `RESULT_ERROR`.

To test from Xcode, add the four arguments under **Product → Scheme → Edit
Scheme → Run → Arguments Passed On Launch**, then run on the physical iPhone.

## Mac command flow

From the repository root:

```bash
bash scripts/run_iphone_benchmark.sh --model 4B --mode text
```

Use `--repeat-count N` for repeated generations in one app launch. Runs that
load the model are labeled `state=cold`; runs reusing it are labeled
`state=warm`. Use `--run-tag` to record the device condition and charger state;
model, mode, and run ID are appended by the suite. Defaults remain one run and
an unspecified charger state.

For correctness isolation, `--fresh-state-each-run` resets persisted KV state
between repetitions while keeping the model loaded. It is diagnostic and off
by default.

For 8B:

```bash
bash scripts/run_iphone_benchmark.sh --model 8B --mode text
```

The script detects the bundle ID from the Xcode project, selects a connected
iPhone, builds Release, installs it, and launches it with benchmark arguments.
Use `--device DEVICE_ID` or set `DEVICE_ID` if automatic selection is ambiguous.
Set `DERIVED_DATA_PATH` to override the temporary build location.

Keep the phone unlocked and watch the orange in-app status. The launch command
returns before inference completes.

## Collecting results

After the benchmark finishes:

```bash
bash scripts/collect_benchmark_results.sh
```

Optional arguments:

```bash
bash scripts/collect_benchmark_results.sh \
  --device DEVICE_ID \
  --output benchmark_results/my-run.log
```

The script requests `Documents/benchmark_results.log` from the app data
container with `devicectl`, saves it under the ignored `benchmark_results/`
folder by default, and prints it. If container extraction fails, use **Bench →
Benchmark Results → Copy Results**.

Summarize one or more collected logs without changing them:

```bash
python3 scripts/summarize_benchmark_results.py benchmark_results/my-run.log
```

The summarizer uses the second `[RESULT]` inside each successful benchmark run,
which is the wrapper-level result, and reports medians and ranges by tag, model,
mode, and cold/warm state.

Text runs also record `[BENCH_TEXT]` with the word count, validation status, and
base64-encoded generated text. The fixed text benchmark succeeds only when the
response contains exactly 20 whitespace-delimited words.

Useful diagnostics:

```bash
xcrun devicectl list devices
xcrun devicectl device info apps --device DEVICE_ID --bundle-id BUNDLE_ID
```

## Limitations

- The iPhone must be connected, unlocked, paired, and trusted.
- Keep the charger plugged in for 8B; it is sensitive to power state.
- First launch, developer trust, and signing may require manual interaction.
- Model files must already be installed in the app container.
- Automated image mode needs an available/default image; fresh launches do not
  currently provide one and therefore emit `RESULT_ERROR`.
- `devicectl` app-container extraction varies by Xcode/iOS version. Copy Results
  remains the fallback.
