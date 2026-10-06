# Running the Fatmawati twin on Google Cloud

What goes where:

- **Cloud Run**, a private service in `asia-southeast2` (Jakarta), runs the live 3D twin (`traffic_twin.fatmawati.webapp3d`).
  - One instance runs one simulation.
  - It scales to zero when nobody is watching. On the next visit the simulation restarts with a 10-minute warm-up.
- **Cloud Storage** holds two things:
  - the CSV sensor releases (`gs://<bucket>/datasets/`);
  - the TomTom request counter (`tomtom_usage.json`: counts only, never TomTom data).
- **Secret Manager** holds the TomTom API key. The key never reaches the browser, a file or the logs.

No TomTom data is stored anywhere. The live panel shows TomTom's speeds as they arrive (see `traffic_twin/fatmawati/tomtom.py`).

## What you need to do yourself

These steps need your own accounts, so Claude cannot do them.

1. **Google Cloud:** create a project and attach a billing account. Install the gcloud CLI and run `gcloud auth login`.
2. **TomTom (optional):**
   - Create an account in MyTomTom; the old developer portal closed after 15 Sep 2026.
   - Create an API key and keep it private.
   - The free tier allows 20,000 Flow Segment Data requests a month. The twin polls 8 points every 60 s, and only while a viewer is open. That is 480 requests an hour, so it stops itself at 18,000 requests.
3. **Deploy:** run the script below. It asks before each step.

```bash
PROJECT_ID=your-project-id REGION=asia-southeast2 deploy/gcp/deploy.sh
```

## Opening the twin

The service is private (`--no-allow-unauthenticated`). The free TomTom tier covers evaluation use, so the twin is not published as a public website. Open it through an authenticated proxy:

```bash
gcloud run services proxy fatmawati-twin --region asia-southeast2 --port 8770
```

Then go to http://localhost:8770. Live TomTom speeds go to the first open viewer only (TomTom T&C 11.4).

## Uploading a dataset release

```bash
PROJECT_ID=your-project-id deploy/gcp/upload_dataset.sh data/fatmawati_v3
```

The script checks `SHA256SUMS.txt` before uploading. The data is shared under the ODbL with "© OpenStreetMap contributors".

## Cost

- **Cloud Run:** billed only while the instance runs (2 vCPU, 2 GiB, CPU always allocated), that is while someone watches plus about 15 idle minutes. Expect very roughly USD 0.2–0.3 per hour; check the current Cloud Run price list for asia-southeast2.
- **Other services:** storage, Artifact Registry and the occasional Cloud Build cost cents.

## Removing everything

```bash
gcloud run services delete fatmawati-twin --region asia-southeast2
gcloud storage rm -r gs://your-project-id-fatmawati-twin
gcloud secrets delete tomtom-api-key
gcloud artifacts repositories delete twin --location asia-southeast2
```

## Testing the image locally

```bash
d=$(mktemp -d) && deploy/gcp/stage.sh "$d" && docker build --platform linux/amd64 -t fatmawati-twin "$d"
```

```bash
docker run --rm -p 8770:8080 fatmawati-twin
```
