# Process 24 half-hour CCTV clips as one 12-hour day

Use `batch.cmd` for this workflow. It loads the models once, keeps the tracker/person galleries between continuous
clips, captures snapshots and uploads each completed clip. It does not run 24 independent counters.

## 1. Collect and check the clips

Put this camera's 24 complete clips in one folder, for example `C:\Videos\EventDay`.
Use filenames whose natural order matches recording order (`clip_01.mp4` through `clip_24.mp4`, or timestamp names).
The folder input assumes all clips are 30 minutes and have no intentional missing slots. Files must come from
one camera at the same resolution. MP4, AVI, MKV, MOV, M4V and TS files are recognized; the installed decoder must be able to read them.

Before processing, generate a timestamp playlist:

```powershell
cd C:\Code\Human-Counting-
.\batch.cmd --input-dir "C:\Videos\EventDay" --recorded-at "2026-10-12T09:00:00+08:00" --prepare-only --output-dir "batch-output\day12"
```

Replace the date/time with the ORIGINAL start time of the FIRST recording, including timezone.
For a 09:00–21:00 day this generates 24 starts: 09:00, 09:30, …, 20:30.
It checks that all 24 files exist, are readable and approximately 30 minutes, and writes
`batch-output\day12\playlist.csv`. Open that CSV and confirm each filename corresponds to the real CCTV start time.
No models are loaded and no video analysis is done with `--prepare-only`.

If clips are short, missing, or start at irregular times, create/edit an explicit CSV with these columns:

```csv
file,recorded_at
C:\Videos\EventDay\0900.mp4,2026-10-12T09:00:00+08:00
C:\Videos\EventDay\0930.mp4,2026-10-12T09:30:00+08:00
```

Rows must already be chronological. Relative filenames are resolved relative to the CSV file.
Overlapping recordings are rejected. A real gap resets motion tracking while retaining appearance/staff galleries;
the missing time is not included in dwell totals. Video time currently uses decoded frame index and frame rate,
so export constant-frame-rate footage for this workflow.

## 2. Create one campaign and enroll staff

Start the updated website with `npm run dev` locally, or deploy it to your VM. Create ONE campaign for the day/event
and copy its campaign ID from the filters. Use one stable camera name, such as `booth-1`.

Enroll staff once from a clear moment in one of the day's clips:

```powershell
.\enroll.cmd --source "C:\Videos\EventDay\clip_01.mp4" --at 30 --name "Staff-A" --out staff_profiles.json
```

Click the staff person's box and press S. Repeat with distinct names for other staff. Add extra viewpoints for the
same name when useful. Reuse this profile file across all 24 clips. For face-assisted recognition, add `--faces`
both during enrollment and batch processing. Recognition still requires checking against your real CCTV footage.

## 3. Run the full day

For your GCP server, set the operator upload key in this PowerShell session:

```powershell
$env:BOOTH_API_KEY = 'YOUR_VM_UPLOAD_KEY'
.\batch.cmd --playlist "batch-output\day12\playlist.csv" --campaign-id "YOUR_CAMPAIGN_ID" --camera-id booth-1 --staff-profiles staff_profiles.json --snapshot-interval 300 --api-url "https://YOUR_DOMAIN" --output-dir "batch-output\day12"
```

Replace the campaign ID, domain, and key. The updated backend must be deployed: batch mode checks upload authorization
and the campaign ID BEFORE loading models or processing footage.
For local testing, use `--api-url "http://localhost:3001"`; a local upload key is optional unless configured on the server.
Omit `--api-url` to process everything offline, then upload later using step 4.

Use `--snapshot-interval 300` for five-minute images: about 144 periodic snapshots over 12 hours.
Use 60 for one-minute images: about 720 periodic snapshots. Person-specific images are additional.
Add `--show` for a preview. Progress is printed as clips and video minutes are processed.

Processing 12 hours of footage can take many hours on CPU; speed depends on footage complexity and hardware.
`--skip 2` analyses every second frame to reduce processing work, while periodic snapshots still use the complete
decoded timeline. Compare accuracy on a sample before choosing frame skipping for production.

## What is preserved across clips

- The same tracker and re-ID counter remain active across continuous, same-camera clips.
- Visitor IDs are namespaced to the whole ordered recording set rather than restarting for each file.
- The five-second qualification threshold carries over. A person visible for two seconds at the end of one clip
  and three seconds in the next can qualify across the boundary.
- Observed duration is divided between the actual clip/day/shift fragments; it is not duplicated for each file.
- Daily/shift reports group fragments by tracked visitor ID before counting and compute duration from summed observations.
- Staff profiles remain active across all files; no fixed deduction is applied when staff are absent.
- Staff/demographic review corrections apply to all fragments of the same tracked identity, including future uploads.

The observation table retains clip fragments for evidence and can include Passing rows that have not reached the
visitor threshold. Daily totals count tracked visitors once per day, and overall totals sum these daily counts.
Staff totals refer to tracked identities; fragmented tracking may still assign multiple IDs to one employee.

Counts remain estimates: occlusion, similar clothing, long absences and late re-identification can split or merge identities.
The ordinary visitor re-ID memory is ten minutes; staff enrollment has no such expiry. Batch processing preserves
continuity at file boundaries, but it does not guarantee true all-day identity recognition across long departures.
Photos for a clicked person can include their evidence across other clips, labeled by source file and actual time.

## 4. Retry uploads without analysing video again

Each completed clip saves a JSON, CSV and JPEG folder under the chosen output directory. If uploading fails,
analysis continues through the remaining files. A final summary lists pending uploads.

```powershell
.\batch.cmd --retry-uploads --output-dir "batch-output\day12" --api-url "https://YOUR_DOMAIN"
```

Keep `BOOTH_API_KEY` set. This reads the existing `clip-001.json` through `clip-024.json` and retries recordings,
records, images and completion checks. It does not load the video models. Already-delivered records/images are not duplicated.
Keep the saved JSON and its evidence folders in their original relative layout.

## Important workflow rules

- Use batch mode for this recording set; do not import the same clips separately through `run.cmd` as another analysis.
  The server checks the analysis group and processing fingerprint and rejects incompatible reuse of a recording.
- Do not change filenames/order, timestamps, staff profiles, model weights or processing options partway through a job.
- If you stop/crash during analysis, completed clip outputs remain safe. You can retry their uploads immediately.
  To continue ANALYSIS with the same identity history, replay the same playlist from the beginning; no resumable tracker
  checkpoint is claimed. Changed derived results are rejected instead of overwriting saved/imported records.
- A clip's payload is finalized when any pending identification from its last frames has resolved in the following
  clip, so upload may occur a short time after the next file starts. The last file is finalized at EOF.
- Gender and Adult/Kid remain Unknown automatically; use verified manual Review when appropriate.
- Rebuild/deploy the website backend along with this Python update before remote batch uploads.

## Checks performed

Tested natural ordering and timestamps for a 24-file plan; clip-boundary duration allocation; shift splitting;
and exclusion of recording gaps. An end-to-end test processed two actual four-second clips with one shared tracker.
The same visitor ID appeared in both clips, first-clip observations below five seconds qualified after the second clip,
the daily report counted ONE visitor and ONE tracked staff identity, and visitor duration totalled 7.9 seconds.
Both clips' evidence/completion uploads and replay without duplicate counts passed. A staff correction affected
both linked fragments. Website/API and Python regression tests passed. A full 12-hour run has not been benchmarked.
