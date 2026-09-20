# Deployment Guide

This guide deploys the standalone `shot-eval` package. It does not require the original video-generation repository.

## 1. Prerequisites

- Linux/macOS/Windows with Python 3.11+;
- a Google Cloud project with Vertex AI access;
- Application Default Credentials (ADC) or a service account;
- input bundles in the [README input layout](README.md#input-directory-convention).

## 2. Install

```bash
git clone https://github.com/<your-org>/shot-eval.git
cd shot-eval
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[optimizer]"
```

For evaluation and reporting only:

```bash
pip install -e .
```

Verify:

```bash
shot-eval --help
shot-eval-serve --help
shot-eval-report --help
```

## 3. Authenticate to Vertex AI

### Local developer ADC

```bash
gcloud auth login
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
export GOOGLE_CLOUD_LOCATION=global
```

### Service account

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/secure/path/service-account.json
export GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
export GOOGLE_CLOUD_LOCATION=global
```

Use a secret manager, workload identity, or CI secret store for service account material. Do not place the JSON file in this repository.

## 4. Evaluate a run

```bash
shot-eval /data/runs/example --dry-run

shot-eval /data/runs/example \
  --repeats 3 \
  --judge-model gemini-3.8-flash \
  --out /data/runs/example/eval/judge-38flash
```

For reproducible comparisons, keep these unchanged between runs:

```text
judge model
prompt version tag
beat source
style source
render source
design source
```

## 5. Serve reports

For local use:

```bash
shot-eval-serve /data/runs --host 127.0.0.1 --port 8010
```

Browse:

```text
http://127.0.0.1:8010/
```

The built-in server has no authentication. Do not bind it directly to a public interface without an authenticated reverse proxy.

Example systemd service:

```ini
[Unit]
Description=shot-eval report server
After=network.target

[Service]
Type=simple
User=shot-eval
WorkingDirectory=/srv/shot-eval
Environment=GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
Environment=GOOGLE_CLOUD_LOCATION=global
ExecStart=/srv/shot-eval/.venv/bin/shot-eval-serve /srv/video-runs --host 127.0.0.1 --port 8010
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Put Nginx, Caddy, Cloud IAP, VPN, or equivalent authentication in front of the service before exposing it to other users.

## 6. Compare reports

The report service discovers every `shots-*.json` under its configured roots. Open:

```text
http://127.0.0.1:8010/compare
```

Select left and right reports. The page displays every paired shot with both videos, original narration, full video prompt, scores, Render result, assessment evidence and grounded optimization data.

## 7. Use the optional ADK optimization agent

The optional optimization agent reads an existing evaluation report and writes a separate optimized bundle. It does not generate or overwrite video.

```bash
shot-eval-optimize \
  --run /data/runs/example \
  --eval /data/runs/example/eval/judge-38flash/shots-example-<timestamp>.json \
  --project <your-gcp-project-id> \
  --location global
```

Dry-run before writing files:

```bash
shot-eval-optimize \
  --run /data/runs/example \
  --eval /data/runs/example/eval/judge-38flash/shots-example-<timestamp>.json \
  --dry-run
```

## 8. Production safety checklist

Before deploying:

- [ ] `GOOGLE_CLOUD_PROJECT` comes from deployment configuration, not source code.
- [ ] No `.env`, service-account JSON, video assets or run outputs are committed.
- [ ] The report server binds to loopback or sits behind authentication.
- [ ] The selected Judge model and prompt contract are recorded for every report.
- [ ] Video generation remains a separate, explicitly approved action.
- [ ] A secret scan and the test suite pass before each release.

## 9. Troubleshooting

### `缺少 GCP project`

Set `GOOGLE_CLOUD_PROJECT` or `VERTEX_PROJECT`.

### `Reauthentication is needed`

Refresh ADC:

```bash
gcloud auth application-default login
```

### `Address already in use`

Select another port or stop the prior server:

```bash
shot-eval-serve /data/runs --port 8011
```

### Report has no generation model tag

The source run must retain `video_prompts.json` with a `meta` block. Historical reports can be displayed without it, but generation metadata will be `unknown`.
