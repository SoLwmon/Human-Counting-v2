import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from session_upload import parse_recorded_at, recording_id, build_payload, save_payload, api_endpoint
import json
from io import BytesIO
from unittest.mock import patch
from session_upload import upload_payload


class UploadTests(unittest.TestCase):
    def test_requires_recording_timezone(self):
        for value in ('', '2026-10-12T09:00:00', '2026-02-30T09:00:00+08:00'):
            with self.assertRaises(ValueError):
                parse_recorded_at(value)
        self.assertEqual(parse_recorded_at('2026-10-12T09:00:00+08:00'), datetime(2026, 10, 12, 1, tzinfo=timezone.utc))

    def test_stable_ids_and_visible_duration(self):
        start = parse_recorded_at('2026-10-12T09:00:00+08:00')
        with TemporaryDirectory() as directory:
            video = Path(directory) / 'video.mp4'
            video.write_bytes(b'same-video-content')
            run = recording_id(video, start, 'camera1')
            self.assertEqual(run, recording_id(video, parse_recorded_at('2026-10-12T01:00:00Z'), 'camera1'))
            self.assertNotEqual(run, recording_id(video, start, 'camera2'))
            people = [SimpleNamespace(pid=1, visitor=True, first_seen=10, last_seen=35, stay_time=20), SimpleNamespace(pid=2, visitor=False)]
            payload = build_payload(people, start, run, 'camera1')
            self.assertEqual(len(payload['sessions']), 1)
            record = payload['sessions'][0]
            self.assertEqual(record['start_time'], '2026-10-12T01:00:10.000+00:00')
            self.assertEqual(record['end_time'], '2026-10-12T01:00:35.000+00:00')
            self.assertEqual(record['dwell_time'], 20)
            self.assertEqual(record['gender'], 'Unknown')
            self.assertEqual(record['age_group'], 'Unknown')
            self.assertEqual(record['session_id'], build_payload(people, start, run, 'camera1')['sessions'][0]['session_id'])
            saved = Path(directory) / 'out.json'
            save_payload(payload, saved)
            self.assertEqual(json.loads(saved.read_text()), payload)

    def test_upload_acknowledgement_and_malformed_payload(self):
        with self.assertRaises(ValueError):
            upload_payload({'wrong_field': []}, 'http://localhost:3001')
        payload = {'sessions': [{'session_id': 'example'}]}
        with patch('urllib.request.build_opener') as opener:
            opener.return_value.open.return_value = BytesIO(b'{"success":true,"received":1,"count":0,"duplicates":1}')
            self.assertEqual(upload_payload(payload, 'http://localhost:3001')['duplicates'], 1)
        with patch('urllib.request.build_opener') as opener:
            opener.return_value.open.return_value = BytesIO(b'{"success":true,"received":1,"count":0,"duplicates":0}')
            with self.assertRaises(RuntimeError):
                upload_payload(payload, 'http://localhost:3001')

    def test_remote_uploads_require_https(self):
        self.assertEqual(api_endpoint('https://booth.example.com'), 'https://booth.example.com/api/sessions/batch')
        self.assertEqual(api_endpoint('http://localhost:3001'), 'http://localhost:3001/api/sessions/batch')
        for url in ('http://booth.example.com', 'https://name:password@booth.example.com', 'https://booth.example.com/path'):
            with self.assertRaises(ValueError):
                api_endpoint(url)


if __name__ == '__main__':
    unittest.main()
