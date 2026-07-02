# PLAN.md — CoreML-LLM Qwen3-VL iPhone Inference Benchmark + Optimization

## Goal

We already have CoreML-LLM building and Qwen3-VL 4B / 8B model files sideloaded to the iPhone.

Now the goal is **not** prompt tuning. The goal is to turn this project into a repeatable benchmark harness for core iPhone inference:

- model load time
- TTFT / time to first token
- decode tokens/sec
- total generation latency
- peak memory / physical footprint
- available memory
- crash/memory-pressure signals
- 4B vs 8B
- text-only vs image mode
- cold vs warm runs
- charger vs non-charger condition if possible

The final output should make it easy to answer:

> Is 8B viable on iPhone?
> What is the actual TTFT and tok/s?
> Where does time go?
> Is the bottleneck model load, image prefill, or decode?
> What is peak memory before 8B crashes or succeeds?

---

## How to use this with Codex

Open the repo on the Mac:

```bash
cd ~/Developer/CoreML-LLM
```

Then start Codex in this repo and paste this entire file as the task prompt.

Suggested first Codex instruction:

```text
Read PLAN.md. Execute it step by step. Make small, reviewable changes only. Do not delete model files, do not change bundle ID, and do not commit model blobs. After each phase, summarize what changed and what I should run on the iPhone.
```

Optional: ask Codex to create `AGENTS.md` for persistent project rules, or run `/init` in Codex if available.

---

## Environment assumptions

Project root:

```bash
~/Developer/CoreML-LLM
```

Xcode project:

```bash
Examples/CoreMLLLMChat/CoreMLLLMChat.xcodeproj
```

Primary app target:

```text
CoreMLLLMChat
```

Current model setup:

```text
Qwen3-VL 4B stateful Core ML: works
Qwen3-VL 8B stateful Core ML: works when iPhone is plugged in / clean state
```

Generation settings already patched or should be patched:

```swift
opts.maxNewTokens = 120
opts.temperature = 0.0
opts.repetitionPenalty = 1.2
```

Do not undo these without asking.

---

## Non-negotiable safety rules

1. **Do not commit model files.**
2. **Do not commit `.mlpackage`, `.mlmodelc`, `.bin`, app containers, Xcode DerivedData, signing files, or logs with device identifiers.**
3. **Do not change bundle ID unless necessary.**
4. **Do not remove existing working 4B/8B sideload scripts.**
5. **Do not delete model files from the iPhone unless explicitly asked.**
6. **Do not rewrite the project architecture. Add small, reviewable instrumentation first.**
7. **Prefer logging to Xcode console before building complex UI.**

---

## Step 1 — Check current repo status

Run:

```bash
cd ~/Developer/CoreML-LLM
git status --short
git diff -- Examples/CoreMLLLMChat/CoreMLLLMChat/LLMRunner.swift
```

Expected known changes may include:

```swift
opts.maxNewTokens = 120
opts.temperature = 0.0
opts.repetitionPenalty = 1.2
```

If there are unexpected large changes, pause and report them.

---

## Step 2 — Add `.gitignore` for local model/build artifacts

Create or update `.gitignore` in repo root:

```gitignore
# Xcode
DerivedData/
*.xcuserstate
*.xcworkspace/xcuserdata/
*.xcodeproj/xcuserdata/
build/
*.ipa

# Core ML model artifacts
*.mlpackage/
*.mlmodelc/
*.bin

# Local model folders
Models/
qwen3_vl_*_stateful_chunks/
qwen3_vl_*_vision/
tmp/

# Local logs/bench output
benchmark_results/
*.xcresult
```

Acceptance criteria:

- `.gitignore` exists.
- No model blobs appear in `git status`.

---

## Step 3 — Add `PerfStats.swift`

Create:

```text
Examples/CoreMLLLMChat/CoreMLLLMChat/PerfStats.swift
```

Paste:

```swift
import Foundation
import MachO
import os

enum PerfStats {
    static func physFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(TASK_VM_INFO),
                    $0,
                    &count
                )
            }
        }

        if result == KERN_SUCCESS {
            return info.phys_footprint
        }

        return 0
    }

    static func availableMemoryBytes() -> UInt64 {
        if #available(iOS 13.0, *) {
            return os_proc_available_memory()
        }
        return 0
    }

    static func gb(_ bytes: UInt64) -> String {
        String(format: "%.2f GB", Double(bytes) / 1024.0 / 1024.0 / 1024.0)
    }

    static func mb(_ bytes: UInt64) -> String {
        String(format: "%.0f MB", Double(bytes) / 1024.0 / 1024.0)
    }

    static func now() -> CFAbsoluteTime {
        CFAbsoluteTimeGetCurrent()
    }

    static func log(_ label: String) {
        let used = physFootprintBytes()
        let avail = availableMemoryBytes()
        print("[PERF] \(label) | used=\(gb(used)) | available=\(gb(avail))")
    }

    static func logInterval(_ label: String, start: CFAbsoluteTime, end: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()) {
        print("[PERF] \(label)=\(String(format: "%.3f", end - start)) sec")
    }
}
```

Acceptance criteria:

- Project compiles.
- `PerfStats.log("test")` can be called from app code without import errors.

---

## Step 4 — Find the 4B/8B load paths

Run:

```bash
grep -R "qwen3-vl-4b\|qwen3-vl-8b\|Qwen3VL4B\|Qwen3VL8B\|StatefulGenerator" -n Examples/CoreMLLLMChat/CoreMLLLMChat Sources | head -200

grep -R "MLModelConfiguration\|computeUnits\|MLModel" -n Examples/CoreMLLLMChat/CoreMLLLMChat Sources | head -200
```

Identify:

- file containing 4B load
- file containing 8B load
- generator class/function
- load button action
- generate function
- token loop / callback if available
- where `maxNewTokens` is passed

Do not change behavior yet. Summarize what files/functions are involved.

Likely relevant files may include:

```text
Examples/CoreMLLLMChat/CoreMLLLMChat/Qwen3VL4BStatefulGeneratorView.swift
Examples/CoreMLLLMChat/CoreMLLLMChat/Qwen3VL8BStatefulGeneratorView.swift
Examples/CoreMLLLMChat/CoreMLLLMChat/LLMRunner.swift
Sources/CoreMLLLM/...
```

---

## Step 5 — Add load-time logging around 4B and 8B

Where the user presses **Load Model** or where the generator/model is initialized, add logs.

For 4B load path:

```swift
let loadStart = PerfStats.now()
PerfStats.log("4B load start")

// existing load code

PerfStats.log("4B load end")
PerfStats.logInterval("4B load_time_sec", start: loadStart)
```

For 8B load path:

```swift
let loadStart = PerfStats.now()
PerfStats.log("8B load start")

// existing load code

PerfStats.log("8B load end")
PerfStats.logInterval("8B load_time_sec", start: loadStart)
```

If chunks are loaded in a visible sequence, add per-chunk logs:

```swift
PerfStats.log("8B before chunk_0")
PerfStats.log("8B after chunk_0")
PerfStats.log("8B before chunk_1")
PerfStats.log("8B after chunk_1")
```

Do this only where it is straightforward. Do not refactor chunk loading heavily.

Acceptance criteria:

- Pressing Load Model prints:
  - `[PERF] 4B load start`
  - `[PERF] 4B load end`
  - `[PERF] 4B load_time_sec=...`
- Same for 8B.
- Logs include used and available memory.

---

## Step 6 — Add generation timing logs

Find the core generation function for 4B/8B stateful.

We need:

- `genStart`
- `firstTokenTime`
- `genEnd`
- `tokenCount`
- `TTFT = firstTokenTime - genStart`
- `decodeTokps = tokenCount / (genEnd - firstTokenTime)`

Add logic around the generation call or streaming token loop:

```swift
let genStart = PerfStats.now()
var firstTokenTime: CFAbsoluteTime?
var generatedTokenCount = 0

PerfStats.log("generation start")
```

Inside token emission / callback / loop:

```swift
generatedTokenCount += 1

if firstTokenTime == nil {
    firstTokenTime = PerfStats.now()
    let ttft = firstTokenTime! - genStart
    PerfStats.log("first token")
    print("[PERF] ttft_sec=\(String(format: "%.3f", ttft))")
}
```

At generation end:

```swift
let genEnd = PerfStats.now()
let first = firstTokenTime ?? genEnd
let decodeSec = max(genEnd - first, 0.001)
let totalSec = genEnd - genStart
let tokps = Double(generatedTokenCount) / decodeSec

PerfStats.log("generation end")
print("[PERF] tokens=\(generatedTokenCount) total_sec=\(String(format: "%.3f", totalSec)) decode_sec=\(String(format: "%.3f", decodeSec)) tokps=\(String(format: "%.2f", tokps))")
```

If the existing generator returns the full output only after completion and does not expose per-token streaming, use the closest possible metric and print:

```swift
print("[PERF] no_stream_ttft_unavailable total_sec=...")
```

Acceptance criteria:

- Every generation run prints total time.
- Streaming path prints TTFT and tok/s.
- If streaming is unavailable in a specific code path, clearly log that TTFT is unavailable.

---

## Step 7 — Add peak memory tracking during generation

During generation, update a local peak memory value:

```swift
var peakFootprint: UInt64 = PerfStats.physFootprintBytes()

func updatePeakMemory(_ label: String) {
    let current = PerfStats.physFootprintBytes()
    if current > peakFootprint {
        peakFootprint = current
        print("[PERF] peak_update label=\(label) peak=\(PerfStats.gb(peakFootprint))")
    }
}
```

Call:

```swift
updatePeakMemory("generation start")
updatePeakMemory("first token")
updatePeakMemory("generation end")
```

If inside token loop, call every 8 tokens to reduce spam:

```swift
if generatedTokenCount % 8 == 0 {
    updatePeakMemory("token \(generatedTokenCount)")
}
```

At end:

```swift
print("[PERF] peak_memory=\(PerfStats.gb(peakFootprint))")
```

Acceptance criteria:

- Each benchmark run prints a peak memory estimate.

---

## Step 8 — Add a compact `[RESULT]` line

In addition to verbose logs, print a single machine-readable line:

```swift
print("[RESULT] model=\(modelName) mode=\(modeName) tokens=\(generatedTokenCount) ttft_sec=\(ttftString) total_sec=\(totalString) tokps=\(tokpsString) peak_gb=\(peakGbString)")
```

Use model names:

```text
4B
8B
```

Use mode names:

```text
text
image
scratch
```

Acceptance criteria:

- A benchmark run produces one `[RESULT]` line that can be copied into a spreadsheet.

---

## Step 9 — Run Release build for perf

In Xcode:

```text
Product → Scheme → Edit Scheme → Run → Build Configuration → Release
```

Also test once without the Xcode debugger attached:

```text
Build/run app → stop Xcode → open app directly on iPhone → run benchmark
```

Acceptance criteria:

- `BENCHMARKING.md` documents whether results are debug-attached or standalone release.

---

## Step 10 — Manual benchmark matrix

Run these manually first. Do not automate until logging works.

### A. 4B text-only

Condition:

```text
fresh app launch
charger plugged if possible
load 4B
no image
```

Prompt:

```text
Say hello in exactly 20 words.
```

Settings:

```text
maxNewTokens = 64
temperature = 0.0
repetitionPenalty = 1.2
```

Record `[RESULT]`.

### B. 4B image

Condition:

```text
same app session after 4B loaded
same test image every run
```

Prompt:

```text
Describe this image in one sentence.
```

Record `[RESULT]`.

### C. 4B scratch

Prompt:

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

Record `[RESULT]`.

### D. 8B text-only

Condition:

```text
charger plugged
fresh app launch
do not load 4B first
load 8B
no image
```

Prompt:

```text
Say hello in exactly 20 words.
```

Settings:

```text
maxNewTokens = 64
temperature = 0.0
repetitionPenalty = 1.2
```

Record `[RESULT]`.

### E. 8B image

Only if 8B text-only works.

Prompt:

```text
Describe this image in one sentence.
```

Record `[RESULT]`.

### F. 8B scratch

Use same scratch prompt as 4B.

Record `[RESULT]`.

---

## Step 11 — Power-state experiments

Because 8B became stable after plugging in the charger, collect condition data.

Test matrix:

| Model | Mode | Charger | Fresh app | Notes |
|---|---|---:|---:|---|
| 4B | text | yes | yes | baseline |
| 4B | image | yes | no | warm |
| 8B | text | yes | yes | must pass |
| 8B | image | yes | no | likely viable |
| 8B | text | no | yes | see if load crashes |
| 8B | image | no | no | only if text works |

If easy, log low power mode:

```swift
let powerNote = ProcessInfo.processInfo.isLowPowerModeEnabled ? "low_power=1" : "low_power=0"
print("[PERF] \(powerNote)")
```

Optional battery state:

```swift
UIDevice.current.isBatteryMonitoringEnabled = true
let state = UIDevice.current.batteryState
let level = UIDevice.current.batteryLevel
print("[PERF] battery_state=\(state.rawValue) battery_level=\(level)")
```

---

## Step 12 — Inspect Core ML compute units

Search:

```bash
grep -R "computeUnits\|MLModelConfiguration" -n Examples/CoreMLLLMChat/CoreMLLLMChat Sources | head -120
```

Log current compute unit setting at load time:

```swift
print("[PERF] compute_units=...")
```

Do not change defaults until baseline numbers are collected.

After baseline, optionally compare:

```text
.cpuAndNeuralEngine
.all
.cpuAndGPU
```

Benchmark; do not guess.

---

## Step 13 — Use Xcode Instruments for ground truth

Run app on physical iPhone:

```text
Xcode → Product → Profile
```

Use:

```text
Core ML
Time Profiler
Allocations
Memory Report
```

For 8B crash/hangs, also check:

```text
Window → Devices and Simulators → select iPhone → View Device Logs
```

Look for:

```text
jetsam
EXC_RESOURCE
memory pressure
per-process-limit
phys_footprint
```

If those appear, record them.

---

## Step 14 — Expected numbers / interpretation

These are rough target ranges, not truth.

### 4B text-only

```text
TTFT target: < 1–2 sec after warm
Decode target: 10–15 tok/s
Peak memory: ideally < 5.5–6.0 GB
```

### 4B image

```text
TTFT target: 2–6 sec
Decode target: 8–12 tok/s
Peak memory: ideally < 6.5 GB
```

### 8B text-only

```text
First goal: load without crash
Decode target if works: 3–8 tok/s
Peak memory: likely high
```

### 8B image

```text
First goal: survive load + first answer
Likely slower / memory-sensitive
```

If 8B only works while plugged in, document:

```text
8B viable only under charger/fresh-state conditions
```

---

## Step 15 — Deliverables

At the end of this task, produce:

1. Code changes:
   - `PerfStats.swift`
   - instrumentation in 4B and 8B load/generate paths
   - optional `.gitignore`
   - optional small BenchmarkView only if simple

2. `BENCHMARKING.md` with:
   - how to run
   - prompts to use
   - what logs mean
   - how to collect results
   - known 8B charger/memory caveat

3. A sample result table template:

```markdown
| Date | Device | iOS | Model | Mode | Charger | Load sec | TTFT sec | Tok/s | Peak GB | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| | iPhone 16 Pro Max | | 4B | text | yes | | | | | |
| | iPhone 16 Pro Max | | 4B | image | yes | | | | | |
| | iPhone 16 Pro Max | | 8B | text | yes | | | | | |
| | iPhone 16 Pro Max | | 8B | image | yes | | | | | |
```

---

## Step 16 — Validation checklist

Before finishing, confirm:

```text
[ ] App builds successfully.
[ ] 4B still loads.
[ ] 4B generation still works.
[ ] 8B still loads when plugged in.
[ ] Load logs print memory before and after.
[ ] Generation logs print TTFT, token count, total time, tok/s.
[ ] Peak memory is printed.
[ ] One-line [RESULT] logs appear.
[ ] No model blobs appear in git status.
[ ] No bundle ID was changed accidentally.
```

---

## Step 17 — Do not optimize before measuring

After this plan is complete, do not make speed optimizations until baseline numbers are collected.

The optimization levers to test later are:

```text
Release vs Debug
debugger attached vs standalone
prewarm model chunks
keep model resident
shorter maxNewTokens
stop after structured fields
image resize/crop
computeUnits .cpuAndNeuralEngine vs .all
fixed context/image buckets
charger / low-power mode
```

But first: measure.
