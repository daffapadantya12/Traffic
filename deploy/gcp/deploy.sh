#!/usr/bin/env bash
# Deploy the Fatmawati 3D twin to Cloud Run (private) with a Cloud Storage bucket for the CSV releases and the
# TomTom request counter.  Run it yourself after creating the project and billing (see README.md); every step
# that creates or changes something asks first.
#
#   PROJECT_ID=my-project REGION=asia-southeast2 deploy/gcp/deploy.sh
set -euo pipefail
: "${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-asia-southeast2}"                       # Jakarta
SERVICE="${SERVICE:-fatmawati-twin}"
BUCKET="${BUCKET:-${PROJECT_ID}-fatmawati-twin}"
REPO="${REPO:-twin}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:$(date +%Y%m%d-%H%M%S)"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

ask() { read -r -p "$1 [y/N] " a; [[ "$a" == "y" || "$a" == "Y" ]]; }

echo "Project ${PROJECT_ID}, region ${REGION}, service ${SERVICE}, bucket gs://${BUCKET}"
gcloud config set project "${PROJECT_ID}" >/dev/null

if ask "Enable Cloud Run, Cloud Build, Artifact Registry, Secret Manager and Cloud Storage APIs?"; then
  gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
    secretmanager.googleapis.com storage.googleapis.com
fi
if ! gcloud artifacts repositories describe "${REPO}" --location "${REGION}" >/dev/null 2>&1; then
  ask "Create Artifact Registry repository '${REPO}' in ${REGION}?" && \
    gcloud artifacts repositories create "${REPO}" --repository-format docker --location "${REGION}"
fi
if ! gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
  ask "Create bucket gs://${BUCKET} (CSV releases, TomTom request counter)?" && \
    gcloud storage buckets create "gs://${BUCKET}" --location "${REGION}" --uniform-bucket-level-access
fi

SECRET_FLAG=()
if gcloud secrets describe tomtom-api-key >/dev/null 2>&1; then
  SECRET_FLAG=(--set-secrets "TOMTOM_API_KEY=tomtom-api-key:latest")
elif ask "Store a TomTom API key in Secret Manager now (typed here, never written to a file)?"; then
  read -r -s -p "TomTom API key: " KEY; echo
  printf %s "${KEY}" | gcloud secrets create tomtom-api-key --data-file=-
  unset KEY
  SECRET_FLAG=(--set-secrets "TOMTOM_API_KEY=tomtom-api-key:latest")
fi

SA="$(gcloud projects describe "${PROJECT_ID}" --format 'value(projectNumber)')-compute@developer.gserviceaccount.com"
if ask "Let the service account ${SA} read the TomTom secret and write to the bucket?"; then
  gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --member "serviceAccount:${SA}" --role roles/storage.objectAdmin >/dev/null
  if [[ ${#SECRET_FLAG[@]} -gt 0 ]]; then
    gcloud secrets add-iam-policy-binding tomtom-api-key --member "serviceAccount:${SA}" --role roles/secretmanager.secretAccessor >/dev/null
  fi
fi

STAGE="$(mktemp -d)"
"${ROOT}/deploy/gcp/stage.sh" "${STAGE}" >/dev/null
if ask "Build the image with Cloud Build (${IMAGE})?"; then
  gcloud builds submit "${STAGE}" --tag "${IMAGE}" --region "${REGION}"
fi
rm -rf "${STAGE}"

# One instance runs one simulation; it scales to zero when nobody watches (the simulation then restarts with a
# warm-up on the next visit).  Private: open it with `gcloud run services proxy` (see README.md).
if ask "Deploy Cloud Run service '${SERVICE}' (private, max 1 instance, 2 vCPU, 2 GiB)?"; then
  gcloud run deploy "${SERVICE}" --image "${IMAGE}" --region "${REGION}" --no-allow-unauthenticated \
    --execution-environment gen2 --cpu 2 --memory 2Gi --no-cpu-throttling --min-instances 0 --max-instances 1 \
    --concurrency 10 --timeout 3600 --port 8080 \
    --add-volume "name=state,type=cloud-storage,bucket=${BUCKET}" --add-volume-mount "volume=state,mount-path=/state" \
    --set-env-vars "TOMTOM_USAGE_FILE=/state/tomtom_usage.json" "${SECRET_FLAG[@]}"
  echo "Open it with:  gcloud run services proxy ${SERVICE} --region ${REGION} --port 8770   then http://localhost:8770"
fi
