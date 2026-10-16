<<<<<<< HEAD
# Booth Counter - Windows setup

This machine is ready to run. Python 3.12 and the dependencies are installed locally under `.python` and `.venv`. Keep both folders in place. No virtual-environment activation or system Python installation is required.

## Process a video

Open PowerShell:

```powershell
cd C:\Code\Human-Counting-
.\run.cmd --source "C:\Videos\booth.mp4" --show --csv visitors.csv
```

Replace the video path with your actual video. Press Q in the preview window to stop early. The CSV is saved when processing finishes or stops. Without `--show`, processing runs without a preview window.

## Webcam

```powershell
.\run.cmd --list-cameras
.\run.cmd --source 0 --show --duration 1
```

`--duration` is in minutes of elapsed execution time. Omit it to run until stopped.

## Optional face matching

Both face model files are installed. Add `--faces` when you want to enable face matching:

```powershell
.\run.cmd --source "C:\Videos\booth.mp4" --show --faces
```

Face matching is off by default. The supplied code keeps face fingerprints in memory during a run.

## Optional annotated video

```powershell
.\run.cmd --source "C:\Videos\booth.mp4" --show --save-screen annotated.mp4
```

## Installed models and sources

- `yolo26n-pose.pt`: [Ultralytics official release](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt).
- `osnet_x0_25_msmt17.pt`: the pretrained MSMT17 combine-all checkpoint from the author's [OSNet model zoo](https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md), renamed to match this program's default filename. [Original download](https://drive.google.com/file/d/1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e/view).
- `models/face_detection_yunet_2023mar.onnx`: [OpenCV model](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
- `models/face_recognition_sface_2021dec.onnx`: [OpenCV model](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface).

`model-checksums.json` records SHA256 hashes of the installed model files. `requirements-lock.txt` records the exact installed package versions. CPU PyTorch is installed; this setup does not require an NVIDIA GPU or CUDA.

## Repeat setup

`setup.ps1` uses the existing `uv` installation on this machine. It creates project-local Python and a virtual environment if missing, installs dependencies, downloads missing weights, and checks model loading and dependency compatibility.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

The command's execution-policy setting applies only to that PowerShell process. On another machine, install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. The setup needs internet access. The installed program can use its local weights afterward.

## Validation performed

All four models loaded successfully. The complete CLI processed a seven-second test clip made by repeating the official Ultralytics bus sample image, with face matching enabled. It exported two visitor rows and an annotated video. Test data is under `work/`; these counts are synthetic validation results, not real booth measurements.

The program exports CSV and can now upload a JSON batch directly to the website API after processing.

## Upload to the website / GCP

The API upload integration is now available. Example:

```powershell
$env:BOOTH_API_KEY = 'YOUR_SERVER_UPLOAD_KEY'
.\run.cmd --source "C:\Videos\booth.mp4" --recorded-at "2026-10-12T09:00:00+08:00" --camera-id booth-1 --campaign-id "YOUR_CAMPAIGN_ID" --api-url "https://YOUR_DOMAIN"
```

The recording timestamp must describe when the original video started. A timezone is mandatory. Campaign/evidence uploads currently accept recorded video files. Local webcam counting remains available without upload options. Uploading saves a replayable JSON file before sending.
Retry with `.\upload.cmd visitors.json --api-url "https://YOUR_DOMAIN"`; the same IDs are not counted twice.
Missing demographics remain Unknown. Use `--json-out visitors.json` without an API URL to export only.
Detailed GCP instructions are in the website repository's `GCP-DEPLOYMENT.md`.

## Campaign trends, staff and evidence

See [CAMPAIGN-WORKFLOW.md](CAMPAIGN-WORKFLOW.md) for campaign setup, staff enrollment, recorded-video snapshots, uploads and dashboard review.

## A full day split into 24 clips

Use [BATCH-PROCESSING.md](BATCH-PROCESSING.md) and `batch.cmd` to keep tracking across consecutive 30-minute clips, rather than starting 24 independent counters.


