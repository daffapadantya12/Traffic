#!/usr/bin/env bash
# Assemble the minimal build context for the twin image in $1 (no datasets, reports or OSM downloads).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:?usage: stage.sh <directory>}"
mkdir -p "$OUT/data/fatmawati"
cp "$ROOT/deploy/gcp/Dockerfile" "$ROOT/deploy/gcp/requirements-twin.txt" "$OUT/"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ROOT/traffic_twin" "$OUT/"
cp -R "$ROOT/data/fatmawati/net" "$ROOT/data/fatmawati/viewer" "$OUT/data/fatmawati/"
echo "$OUT"
