#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PBXPROJ="$ROOT/Examples/CoreMLLLMChat/CoreMLLLMChat.xcodeproj/project.pbxproj"
DEVICE_ID="${DEVICE_ID:-}"
OUTPUT=""

usage() {
    echo "Usage: $0 [--device DEVICE_ID] [--output PATH]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE_ID="${2:-}"; shift 2 ;;
        --output) OUTPUT="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

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
    echo "No connected iPhone found. Run: xcrun devicectl list devices" >&2
    exit 1
fi

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="$ROOT/benchmark_results/benchmark_results-$(date +%Y%m%d-%H%M%S).log"
fi
mkdir -p "$(dirname "$OUTPUT")"

echo "Bundle ID: $BUNDLE_ID"
echo "Device:    $DEVICE_ID"
echo "Copying Documents/benchmark_results.log..."

if xcrun devicectl device copy from \
    --device "$DEVICE_ID" \
    --timeout 30 \
    --domain-type appDataContainer \
    --domain-identifier "$BUNDLE_ID" \
    --source "Documents/benchmark_results.log" \
    --destination "$OUTPUT"; then
    echo "Results saved to: $OUTPUT"
    echo
    cat "$OUTPUT"
    exit 0
fi

echo "Could not copy the app-container log with devicectl." >&2
echo "Fallback: Loaded app → Bench → Benchmark Results → Copy Results" >&2
echo "Useful commands:" >&2
echo "  xcrun devicectl list devices" >&2
echo "  xcrun devicectl device info apps --device '$DEVICE_ID' --bundle-id '$BUNDLE_ID'" >&2
exit 1
