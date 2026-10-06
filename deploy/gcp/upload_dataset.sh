#!/usr/bin/env bash
# Copy a generated CSV release (e.g. data/fatmawati_v3) to the project bucket:  deploy/gcp/upload_dataset.sh DIR
set -euo pipefail
: "${PROJECT_ID:?set PROJECT_ID}"
DIR="${1:?usage: upload_dataset.sh <release directory>}"
BUCKET="${BUCKET:-${PROJECT_ID}-fatmawati-twin}"
(cd "${DIR}" && shasum -a 256 -c SHA256SUMS.txt >/dev/null) || { echo "checksums do not match"; exit 1; }
read -r -p "Upload ${DIR} to gs://${BUCKET}/datasets/$(basename "${DIR}")/ ? [y/N] " a
[[ "$a" == "y" || "$a" == "Y" ]] && gcloud storage cp -r "${DIR}" "gs://${BUCKET}/datasets/"
