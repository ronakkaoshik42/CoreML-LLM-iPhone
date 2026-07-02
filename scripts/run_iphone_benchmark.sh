#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$ROOT/Examples/CoreMLLLMChat/CoreMLLLMChat.xcodeproj"
PBXPROJ="$PROJECT/project.pbxproj"
SCHEME="CoreMLLLMChat"
MODEL=""
MODE=""
MAX_NEW_TOKENS=64
REPEAT_COUNT=1
RUN_TAG=""
FRESH_STATE_EACH_RUN=0
DEVICE_ID="${DEVICE_ID:-}"

usage() {
    echo "Usage: $0 --model 4B|8B --mode text|image [--repeat-count N] [--run-tag DEVICE_CONDITION] [--fresh-state-each-run] [--device DEVICE_ID]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="${2:-}"; shift 2 ;;
        --mode) MODE="${2:-}"; shift 2 ;;
        --repeat-count) REPEAT_COUNT="${2:-}"; shift 2 ;;
        --run-tag) RUN_TAG="${2:-}"; shift 2 ;;
        --fresh-state-each-run) FRESH_STATE_EACH_RUN=1; shift ;;
        --device) DEVICE_ID="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ "$MODEL" != "4B" && "$MODEL" != "8B" ]]; then
    echo "--model must be 4B or 8B" >&2
    exit 2
fi
if [[ "$MODE" != "text" && "$MODE" != "image" ]]; then
    echo "--mode must be text or image" >&2
    exit 2
fi
if [[ ! "$REPEAT_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "--repeat-count must be a positive integer" >&2
    exit 2
fi

BUNDLE_ID="$(sed -n 's/^[[:space:]]*PRODUCT_BUNDLE_IDENTIFIER = \([^;]*\);/\1/p' "$PBXPROJ" | head -1 | tr -d '"')"
if [[ -z "$BUNDLE_ID" ]]; then
    echo "Could not detect PRODUCT_BUNDLE_IDENTIFIER from $PBXPROJ" >&2
    exit 1
fi

if [[ -z "$DEVICE_ID" ]]; then
    DEVICE_ID="$({ xcrun devicectl list devices \
            --timeout 10 \
            --filter "name CONTAINS[c] 'iPhone'" \
            --columns identifier --hide-default-columns --hide-headers 2>/dev/null \
            || true; } | awk 'NF { print $NF; exit }')"
fi

if [[ -z "$DEVICE_ID" ]]; then
    echo "No connected iPhone found. Unlock/trust it, then run:" >&2
    echo "  xcrun devicectl list devices" >&2
    echo "Or set DEVICE_ID / pass --device." >&2
    exit 1
fi
if [[ -z "$RUN_TAG" ]]; then
    RUN_TAG="device-${DEVICE_ID}_charger-unknown"
fi

if [[ "$MODEL" == "8B" ]]; then
    echo "Warning: keep the iPhone plugged into its charger for the 8B benchmark."
fi
if [[ "$MODE" == "image" ]]; then
    echo "Warning: launch automation has no selected image and will record RESULT_ERROR unless one is already available in the app session."
fi

DERIVED_DATA="${DERIVED_DATA_PATH:-${TMPDIR:-/tmp}/CoreMLLLMChatBenchmarkDerivedData}"
APP_PATH="$DERIVED_DATA/Build/Products/Release-iphoneos/CoreMLLLMChat.app"

echo "Bundle ID: $BUNDLE_ID"
echo "Device:    $DEVICE_ID"
echo "Run tag:   $RUN_TAG"
echo "Repeats:   $REPEAT_COUNT"
echo "Fresh KV:  $FRESH_STATE_EACH_RUN"
echo "Building Release app..."
xcodebuild \
    -project "$PROJECT" \
    -scheme "$SCHEME" \
    -configuration Release \
    -destination "id=$DEVICE_ID" \
    -derivedDataPath "$DERIVED_DATA" \
    build

if [[ ! -d "$APP_PATH" ]]; then
    echo "Build succeeded but app was not found at $APP_PATH" >&2
    exit 1
fi

echo "Installing app..."
xcrun devicectl device install app --device "$DEVICE_ID" --timeout 120 "$APP_PATH"

echo "Launching automated benchmark..."
LAUNCH_ARGS=(
    --run-benchmark-suite
    "--benchmark-model=$MODEL"
    "--benchmark-mode=$MODE"
    "--benchmark-max-new-tokens=$MAX_NEW_TOKENS"
    "--benchmark-repeat-count=$REPEAT_COUNT"
    "--benchmark-run-tag=$RUN_TAG"
)
if [[ "$FRESH_STATE_EACH_RUN" == "1" ]]; then
    LAUNCH_ARGS+=(--benchmark-fresh-state-each-run)
fi
if ! xcrun devicectl device process launch \
    --device "$DEVICE_ID" \
    --timeout 30 \
    --terminate-existing \
    "$BUNDLE_ID" \
    "${LAUNCH_ARGS[@]}"; then
    echo "Automatic launch failed. In Xcode, add these Run arguments and launch on the iPhone:" >&2
    echo "  --run-benchmark-suite" >&2
    echo "  --benchmark-model=$MODEL" >&2
    echo "  --benchmark-mode=$MODE" >&2
    echo "  --benchmark-max-new-tokens=$MAX_NEW_TOKENS" >&2
    echo "  --benchmark-repeat-count=$REPEAT_COUNT" >&2
    echo "  --benchmark-run-tag=$RUN_TAG" >&2
    if [[ "$FRESH_STATE_EACH_RUN" == "1" ]]; then
        echo "  --benchmark-fresh-state-each-run" >&2
    fi
    exit 1
fi

echo "Benchmark launched. Keep the iPhone unlocked until the visible status says done."
echo "Then collect results with:"
echo "  bash scripts/collect_benchmark_results.sh --device '$DEVICE_ID'"
