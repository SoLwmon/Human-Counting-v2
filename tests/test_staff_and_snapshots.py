import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from datetime import datetime,timezone
import numpy as np
from staff_identity import StaffMatcher,weights_hash
from snapshot_capture import SnapshotCapture
from session_upload import build_payload


class FeatureTests(unittest.TestCase):
    def test_staff_identity_survives_new_tracks_and_ambiguous_matches_are_not_excluded(self):
        with TemporaryDirectory() as directory:
            weights=Path(directory)/'weights.pt';weights.write_bytes(b'body model')
            a=np.zeros(512,np.float32);a[0]=1;b=np.zeros(512,np.float32);b[1]=1
            profile={'version':1,'weights_sha256':weights_hash(weights),'profiles':[{'name':'Team-1','bodies':[a.tolist()],'faces':[]}]}
            path=Path(directory)/'staff.json';path.write_text(json.dumps(profile))
            matcher=StaffMatcher(path,weights)
            person=SimpleNamespace(pid=1,gallery=[np.r_[a,np.zeros(128)]]*3,face_gallery=[],detected_role='visitor',first_seen=0,last_seen=10,stay_time=10,visitor=True)
            matcher.classify(person);self.assertEqual(person.detected_role,'staff')
            # A fresh tracker ID much later still matches permanent enrollment.
            returning=SimpleNamespace(pid=77,gallery=[np.r_[a,np.zeros(128)]]*3,face_gallery=[],detected_role='visitor')
            matcher.classify(returning);self.assertEqual(returning.detected_role,'staff')
            visitor=SimpleNamespace(pid=2,gallery=[np.r_[b,np.zeros(128)]]*3,face_gallery=[],detected_role='visitor')
            matcher.classify(visitor);self.assertEqual(visitor.detected_role,'visitor')
            payload=build_payload([person],datetime(2026,10,12,tzinfo=timezone.utc),'run','camera')
            self.assertEqual(payload['sessions'][0]['person_role'],'staff')
            profile['profiles'].append({'name':'Team-2','bodies':[a.tolist()],'faces':[]});path.write_text(json.dumps(profile))
            ambiguous=StaffMatcher(path,weights);visitor.gallery=[np.r_[a,np.zeros(128)]]*3
            ambiguous.classify(visitor);self.assertEqual(visitor.detected_role,'visitor')

    def test_clear_faces_override_similar_body_appearance(self):
        with TemporaryDirectory() as directory:
            weights=Path(directory)/'weights.pt';weights.write_bytes(b'body model')
            body=np.zeros(512,np.float32);body[0]=1
            face=np.zeros(128,np.float32);face[0]=1
            different=np.zeros(128,np.float32);different[1]=1
            path=Path(directory)/'staff.json';path.write_text(json.dumps({'version':1,'weights_sha256':weights_hash(weights),'profiles':[{'name':'Team-1','bodies':[body.tolist()],'faces':[face.tolist()]}]}))
            matcher=StaffMatcher(path,weights)
            person=SimpleNamespace(pid=1,gallery=[np.r_[body,np.zeros(128)]]*3,face_gallery=[different]*3,detected_role='visitor')
            matcher.classify(person);self.assertEqual(person.detected_role,'visitor')
            person.face_gallery=[face]*3
            matcher.classify(person);self.assertEqual(person.detected_role,'staff')

    def test_periodic_snapshots_follow_video_clock_and_person_evidence_follows_merges(self):
        with TemporaryDirectory() as directory:
            captures=SnapshotCapture(directory,'run',60);frame=np.zeros((100,200,3),np.uint8)
            for seconds in [0,1,59.9,60,61,120]:captures.periodic(frame,seconds)
            self.assertEqual([item['time_s'] for item in captures.items],[0,60,120])
            person=SimpleNamespace(pid=3);captures.people(frame,[(person,np.array([10,10,50,90]))],10)
            captures.people(frame,[],11,{3:1})
            import hashlib
            session=hashlib.sha256(b'run:1').hexdigest();manifest=captures.manifest([{'session_id':session}],'run')
            self.assertEqual(next(item for item in manifest if item['kind']=='person')['session_id'],session)
            self.assertTrue(all((Path(directory)/item['file']).is_file() for item in manifest))
            other=SnapshotCapture(Path(directory)/'five','run2',300)
            for seconds in [0,60,299,300,600]:other.periodic(frame,seconds)
            self.assertEqual([item['time_s'] for item in other.items],[0,300,600])
            original=(Path(directory)/captures.items[0]['file']).read_bytes()
            replay=SnapshotCapture(directory,'run',60)
            with self.assertRaises(RuntimeError):replay.periodic(np.full_like(frame,255),0)
            self.assertEqual((Path(directory)/captures.items[0]['file']).read_bytes(),original)


if __name__=='__main__':unittest.main()
