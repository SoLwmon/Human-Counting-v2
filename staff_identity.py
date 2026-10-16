"""Persistent, operator-enrolled staff identities, independent of the tracker's 10-minute memory."""
import hashlib
import json
from pathlib import Path
import numpy as np


def weights_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StaffMatcher:
    def __init__(self, path, weights, body_threshold=0.85, face_threshold=0.45, margin=0.08):
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('version')!=1 or data.get('weights_sha256')!=weights_hash(weights):
            raise ValueError('Staff profiles were created with different body weights. Re-enroll using the current --reid weights.')
        self.profiles=data.get('profiles',[])
        if not self.profiles: raise ValueError('No staff profiles found. Run enroll_staff.py first.')
        self.body_threshold=body_threshold; self.face_threshold=face_threshold; self.margin=margin
        for profile in self.profiles:
            if not isinstance(profile.get('name'),str) or not profile['name'].strip(): raise ValueError('Staff profile must have a name.')
            for key,size in [('bodies',512),('faces',128)]:
                values=np.asarray(profile.get(key,[]),dtype=np.float32)
                if values.size and (values.ndim!=2 or values.shape[1]!=size or not np.isfinite(values).all()): raise ValueError(f'Invalid {key} samples in staff profile.')
                profile[key]=values
            if not profile['bodies'].size: raise ValueError('Staff profile needs body samples.')

    @staticmethod
    def similarity(query,gallery):
        scores=np.asarray(query,dtype=np.float32)@gallery.T
        # Require support from multiple observed samples rather than a single frame.
        best=scores.max(axis=1)
        return float(np.sort(best)[-min(3,len(best)):].mean())

    def classify(self,person):
        if len(person.gallery)<3: return
        faces=getattr(person,'face_gallery',[])
        scores=[]
        for profile in self.profiles:
            use_face=len(faces)>=3 and profile['faces'].size>0
            query=faces[-5:] if use_face else [np.asarray(body)[:512] for body in person.gallery[-5:]]
            score=self.similarity(query,profile['faces'] if use_face else profile['bodies'])
            threshold=self.face_threshold if use_face else self.body_threshold
            scores.append((score-threshold,score,profile['name']))
        scores.sort(reverse=True)
        if not scores or scores[0][0]<0 or (len(scores)>1 and scores[0][0]-scores[1][0]<self.margin): return
        person.detected_role='staff';person.staff_label=scores[0][2];person.staff_match_score=round(scores[0][1],6)
