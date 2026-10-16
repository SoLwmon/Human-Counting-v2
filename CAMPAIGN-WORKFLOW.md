# Campaign dashboard, staff enrollment and recording evidence

## What is implemented

- Client, campaign, location, monitored space (m²), and date-range filters.
- A daily trend graph with visitor sessions, average visible duration, Gender, Adult vs Kid, and staff records. Days with no uploaded observations or completed recording are gaps, rather than invented zeros.
- Gender and Adult/Kid groups include Unknown; staff are excluded from visitor/demographic totals.
- Paginated visitor/staff records. Clicking a row opens associated evidence; Review changes staff/visitor classification.
- A recording browser for periodic CCTV snapshots, including recordings with no qualifying visitors.
- Recorded-video staff enrollment with body samples and optional face samples, retained locally across program restarts.
- Periodic snapshots every 60 or 300 video seconds, plus highlighted person evidence.
- Authenticated, retry-safe recording, session, and image uploads. Images require a viewer key on the hosted server.

## 1. Start the website and create a campaign

```powershell
cd C:\Code\booth-visitor-counter-website
npm ci
npm run dev
```

Open `http://localhost:5173`, click **New campaign**, and enter client, campaign name, location, monitored area,
and campaign start/end dates. On the hosted site, also enter your operator upload key.
After creation, the selected campaign's ID is shown under the filters. Copy it for the processing command.

Existing records are retained under an **Existing records / Unassigned** campaign. Database migrations preserve
old data and create backups. Earlier demo records and guessed demographics cannot be retrospectively identified.

## 2. Enroll staff from the recording

Choose a clear moment where a staff member is visible. `--at` is a position in the VIDEO, in seconds.

```powershell
cd C:\Code\Human-Counting-
.\enroll.cmd --source "C:\Videos\booth.mp4" --at 30 --name "Staff-A" --out "staff_profiles.json"
```

In the enrollment window, click that staff member's detected box. Selected boxes turn green.
Press **S** to save; **Q** or Escape cancels. For several staff, enroll each with a distinct name such as
Staff-A, Staff-B. You can add samples for the SAME named staff member from another clear timestamp:

```powershell
.\enroll.cmd --source "C:\Videos\booth.mp4" --at 45 --name "Staff-A" --out "staff_profiles.json"
```

Select only Staff-A again. Samples are appended to that person's profile (up to 20), rather than creating another identity.
For optional face-assisted recognition, add `--faces` during enrollment AND video processing. Face mode is useful
only when a face is sufficiently clear and large; body matching remains the fallback.

Profile files contain appearance vectors, stay on your processing machine, and are excluded from Git by the default
`staff_profiles*.json` pattern. They are not uploaded in the JSON payload. Re-enroll if staff appearance changes substantially,
or if the OSNet weights change. Enrollment and processing should use the same `--no-enhance` setting.

The matching gallery does not expire after ten minutes. A returning staff member can therefore be classified even
after a long lunch break or when ByteTrack assigns a new ID. Exclusions apply to recorded observations, not a fixed
deduction when staff are absent. Recognition remains approximate: multiple viewpoints and real-footage testing are needed.
Similar uniforms, occlusion and lighting can cause false or missed matches. Ambiguous matches are left as visitors
for review instead of being automatically excluded. Default body and face thresholds are 0.85 and 0.45, configurable
with `--staff-body-match` and `--staff-face-match`.

## 3. Process the full video with snapshots

For local testing:

```powershell
.\run.cmd --source "C:\Videos\booth.mp4" --recorded-at "2026-10-12T09:00:00+08:00" --campaign-id "YOUR_CAMPAIGN_ID" --camera-id "booth-1" --staff-profiles "staff_profiles.json" --snapshot-interval 60 --api-url "http://localhost:3001" --csv "visitors.csv"
```

For your GCP website:

```powershell
$env:BOOTH_API_KEY = 'YOUR_OPERATOR_UPLOAD_KEY'
.\run.cmd --source "C:\Videos\booth.mp4" --recorded-at "2026-10-12T09:00:00+08:00" --campaign-id "YOUR_CAMPAIGN_ID" --camera-id "booth-1" --staff-profiles "staff_profiles.json" --snapshot-interval 300 --api-url "https://YOUR_DOMAIN" --csv "visitors.csv"
```

Replace the recording time with when the ORIGINAL video began, including its timezone.
Add `--show` for a preview. Without `--staff-profiles`, all qualifying people remain visitors until reviewed.
`--snapshot-interval 60` captures at the start and every minute; 300 captures every five minutes.
The schedule follows decoded video timestamps, not how fast the computer processes the recording. Periodic images are
taken before frame-skipping. Person evidence is a camera frame with that tracked person's box highlighted.

Files are saved locally as CSV, a JSON payload, and JPEGs under `captures/<recording ID>/` by default.
Use `--snapshots-dir` to select another folder. Keep the JPEG folder with the saved JSON for retries.
Automatic upload occurs only when processing reaches the end of the recording. Stopping early saves partial results
locally without automatically publishing them. Live camera campaign/evidence uploads are not part of this recorded-video workflow.

If you want to inspect the results before sending, replace `--api-url ...` with `--json-out visitors.json`.

## 4. Retry delivery without processing again

```powershell
.\upload.cmd "visitors.json" --api-url "https://YOUR_DOMAIN"
```

The uploader registers the recording, sends the session batch, then uploads JPEG evidence individually.
Already-saved recordings, sessions and images are not duplicated. If the image upload fails after sessions were saved,
retry the same JSON and image folder: it finishes delivery without inflating counts. The server marks a recording Complete only after validating its expected session and image IDs; interrupted delivery remains marked Upload in progress. A reused ID with changed data is
rejected rather than silently overwriting it. Staff review overrides are preserved when automatic results are replayed.

## 5. Review in the dashboard

- Select client/campaign/location/space and the reporting dates. All graphs, record rows and recordings follow those filters.
- Change the graph's **Show** selector to compare visitor counts, duration, Gender, Adult vs Kid, or staff records across days.
- Click an observation row to open associated person evidence and periodic frames within its observed time range.
- Use **Recording snapshots** to browse the full periodic camera timeline, with additional image pages loaded as needed.
- Enter the evidence viewer key in the image panel. This is kept in the panel's memory, not in image URLs or saved browser storage.
- Click **Review** on a row to mark it Staff or Visitor, or restore the automatic result. You can also supply a verified Gender and Adult/Kid classification, or restore the original Unknown value. Enter the operator upload key and an optional note.
  The graph and totals refresh after the correction. The backend records an audit entry.

Staff observations remain in the database and review table. The dashboard does not pretend to know a fixed staff count
or their presence when they are not visible. A staff label can appear in multiple records if tracking fragments; staff
record counts are not a count of unique employees.

## GCP image storage and access

The existing Compute Engine deployment still runs the dashboard/API together with persistent SQLite at `/data`.
By default JPEGs are stored at `/data/evidence`, on the same persistent VM mount. Videos stay on your Windows machine.

Set a separate `EVIDENCE_VIEW_KEY` in `deploy/.env` if viewers should have image access without permission to upload/correct data.
Otherwise evidence access uses the operator upload key. Public analytics stay readable; image metadata and JPEG routes require a key.

Optional private Cloud Storage:

1. Create a private bucket and keep public access disabled.
2. Set `EVIDENCE_BUCKET=YOUR_BUCKET_NAME` in `deploy/.env`.
3. Attach a service account to the VM with bucket-scoped Object Creator and Object Viewer permissions and an appropriate
   cloud-platform access scope. The official Google Storage SDK uses Application Default Credentials from that VM account.
4. Rebuild/restart the Compose deployment. The server uploads new JPEGs to the bucket and reads them through protected API routes;
   the Windows program does not need GCP credentials.

Keep that bucket configured to read historical GCS images. Existing disk images remain readable if new uploads switch to GCS.
Actual GCP resources/bucket permissions must be configured in your account; no cloud deployment or real bucket upload has been run here.
For local GCS testing, configure ADC using the official Google instructions. Do not commit a service-account key.

See `GCP-DEPLOYMENT.md` in the website repository for VM, domain, HTTPS and Docker setup. To rebuild on the VM:

```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d --build
```

## Validation and limits

Verified with isolated test databases and sample footage:

- Campaign metadata/filtering, legacy schema migration, unknown demographics and duplicate-safe batches.
- Staff exclusion and audited manual corrections, including preserving corrections during retries.
- One-minute/five-minute video-clock snapshot schedules and merged tracking-ID evidence association.
- Staff recognition independent of expiring tracking IDs, plus conservative ambiguous-match handling.
- Actual enrollment from a sample video, recognition of one enrolled person, upload of two records and three JPEGs,
  evidence access and replay without duplicate records/images.
- Dashboard production build and API/uploader regression tests.
- Browser checks of campaign filters, metric switching, image viewing, classification corrections, campaign creation and mobile layout, with no browser runtime errors.
- Production dependency audit: zero reported vulnerabilities. Some inherited development/build-tool audit findings remain and require major tooling upgrades; these dependencies are excluded from the deployed runtime image.

Gender and Adult/Kid are supported in reports. Automatic counter results remain Unknown because it does not estimate demographics; verified manual classifications can be entered in Review and are protected from being overwritten by retries.
Recognition accuracy has not been measured against your real CCTV footage. The single-video command combines observations within that file. For a day split into 24 clips, use BATCH-PROCESSING.md: batch mode splits observed time by clip/day/shift and groups the same tracked visitor across fragments.
Separate overlapping recordings can still represent the same real person more than once.

The Docker/GCP deployment and GCS credentials require testing on your VM. Existing development/build dependencies still have
audit findings that require major tooling upgrades; production serves the built frontend and does not expose Vite.

## Official Google references

- [VM service accounts and access scopes](https://docs.cloud.google.com/compute/docs/access/service-accounts)
- [Cloud Storage authentication / ADC](https://docs.cloud.google.com/storage/docs/authentication)
- [Node.js object upload example](https://docs.cloud.google.com/storage/docs/samples/storage-upload-file)

## 24 half-hour clips

Use BATCH-PROCESSING.md for the full-day batch command, timestamp playlist, continuous tracking and upload retries. Daily/shift counts now group fragments of the same tracked visitor instead of counting every clip row separately.
