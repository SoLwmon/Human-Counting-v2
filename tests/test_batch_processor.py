import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from datetime import timedelta
from batch_processor import make_plan,parse_recorded_at,FragmentLedger,inspect_plan


class BatchTests(unittest.TestCase):
    def test_24_clip_plan_uses_numeric_order_and_half_hour_start_times(self):
        with TemporaryDirectory() as directory:
            for index in range(1,25):(Path(directory)/f'clip_{index}.mp4').write_bytes(b'placeholder')
            plan=make_plan(directory,parse_recorded_at('2026-10-12T09:00:00+08:00'))
            self.assertEqual(len(plan),24)
            self.assertEqual(Path(plan[1]['file']).name,'clip_2.mp4')
            self.assertEqual(plan[-1]['recorded_at'],'2026-10-12T20:30:00+08:00')

    def test_small_overlap_requires_opt_in_and_trims_only_previous_clip(self):
        import cv2
        with TemporaryDirectory() as directory:
            clips=[Path(directory)/'a.mkv',Path(directory)/'b.mkv']
            for path in clips:path.touch()
            origin=parse_recorded_at('2026-10-16T12:00:00+08:00')
            rows=[{'file':str(path),'recorded_at':(origin+timedelta(seconds=offset)).isoformat()} for path,offset in zip(clips,[0,3.8])]
            def capture(_):
                return SimpleNamespace(isOpened=lambda:True,release=lambda:None,get=lambda key:{cv2.CAP_PROP_FPS:15,cv2.CAP_PROP_FRAME_COUNT:60,cv2.CAP_PROP_FRAME_WIDTH:640,cv2.CAP_PROP_FRAME_HEIGHT:480}[key])
            with patch.object(cv2,'VideoCapture',side_effect=capture):
                with self.assertRaisesRegex(ValueError,'overlaps'):inspect_plan(rows)
                result=inspect_plan(rows,trim_overlaps=True)
                self.assertEqual(result[0]['frames'],57)
                self.assertAlmostEqual(result[0]['duration_s'],3.8)
                self.assertEqual(result[1]['frames'],60)
                self.assertEqual([path.stat().st_size for path in clips],[0,0])
                rows[1]['recorded_at']=(origin+timedelta(seconds=1)).isoformat()
                with self.assertRaisesRegex(ValueError,'overlaps'):inspect_plan(rows,trim_overlaps=True)

    def test_continuing_person_has_one_identity_and_precise_clip_fragments(self):
        origin=parse_recorded_at('2026-10-12T09:00:00+08:00')
        clips=[{'index':0,'id':'clip1','start':origin,'duration_s':4},{'index':1,'id':'clip2','start':origin+timedelta(seconds=4),'duration_s':4}]
        ledger=FragmentLedger(clips,origin)
        person=SimpleNamespace(pid=1,first_seen=0,stay_time=3.9,visitor=False,detected_role='visitor',staff_label=None,staff_match_score=None)
        ledger.observe([person],3.9);person.stay_time=4;ledger.observe([person],4)
        counter=SimpleNamespace(people={1:person})
        first=ledger.sessions(clips[0],counter,'day','camera','campaign')
        self.assertEqual(first[0]['dwell_time'],4)
        self.assertFalse(first[0]['qualifies_visitor'])
        person.stay_time=7.9;person.visitor=True;ledger.observe([person],7.9)
        second=ledger.sessions(clips[1],counter,'day','camera','campaign')
        self.assertEqual(first[0]['visitor_id'],second[0]['visitor_id'])
        self.assertNotEqual(first[0]['session_id'],second[0]['session_id'])
        self.assertAlmostEqual(second[0]['dwell_time'],3.9)
        self.assertTrue(second[0]['qualifies_visitor'])

    def test_observed_time_is_split_at_shift_boundary_and_gaps_are_not_filled(self):
        origin=parse_recorded_at('2026-10-12T14:59:59+08:00')
        clips=[{'index':0,'id':'clip','start':origin,'duration_s':2}]
        ledger=FragmentLedger(clips,origin)
        person=SimpleNamespace(pid=1,first_seen=0,stay_time=1.5,visitor=False,detected_role='visitor',staff_label=None,staff_match_score=None)
        ledger.observe([person],1.5)
        rows=ledger.sessions(clips[0],SimpleNamespace(people={1:person}),'day','camera','campaign')
        self.assertEqual([row['dwell_time'] for row in rows],[1,0.5])
        gapclips=[{'index':0,'id':'a','start':origin,'duration_s':2},{'index':1,'id':'b','start':origin+timedelta(seconds=10),'duration_s':2}]
        gap=FragmentLedger(gapclips,origin);person.stay_time=2;gap.observe([person],2);person.stay_time=2.5;gap.observe([person],10.5)
        self.assertAlmostEqual(sum(row['seconds'] for row in gap.segments.values()),2.5)


if __name__=='__main__':unittest.main()
