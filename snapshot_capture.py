"""Periodic camera snapshots and per-person evidence using VIDEO timestamps."""
import hashlib
from pathlib import Path
import cv2


class SnapshotCapture:
    def __init__(self,directory,recording_id,interval=60):
        if interval not in (60,300):raise ValueError('Snapshot interval must be 60 or 300 seconds.')
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.recording_id=recording_id;self.interval=interval;self.next_at=0.0;self.items=[];self.captured_people=set()

    def _save(self,frame,time_s,kind,pid=None,box=None):
        # Use source timestamp and kind for deterministic evidence IDs across retries.
        evidence_id=hashlib.sha256(f'{self.recording_id}:{kind}:{time_s:.3f}:{pid}'.encode()).hexdigest()
        view=frame.copy()
        if box is not None:
            x1,y1,x2,y2=map(int,box);cv2.rectangle(view,(x1,y1),(x2,y2),(0,220,220),3)
            cv2.putText(view,f'Tracking ID {pid}',(x1,max(25,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,220,220),2)
        h,w=view.shape[:2]
        if w>1280:view=cv2.resize(view,(1280,round(h*1280/w)))
        name=evidence_id+'.jpg'
        ok,encoded=cv2.imencode('.jpg',view,[cv2.IMWRITE_JPEG_QUALITY,85])
        if not ok:raise RuntimeError('Could not encode snapshot.')
        destination=self.directory/name;data=encoded.tobytes()
        if destination.exists():
            if destination.read_bytes()!=data:raise RuntimeError('This snapshot differs from existing saved evidence. Original image was retained; retry the saved JSON or use a separate analysis folder.')
        else:
            temporary=destination.with_suffix('.jpg.part');temporary.write_bytes(data);temporary.replace(destination)
        self.items.append({'id':evidence_id,'time_s':round(time_s,3),'kind':kind,'file':name,'person_pid':pid})

    def periodic(self,frame,time_s):
        if time_s+1e-6>=self.next_at:
            self._save(frame,time_s,'periodic')
            # A discontinuity creates one real frame, not duplicate images pretending to fill a gap.
            self.next_at=(int(time_s//self.interval)+1)*self.interval

    def people(self,frame,tracked,time_s,remap=None):
        for old,new in (remap or {}).items():
            for item in self.items:
                if item['person_pid']==old:item['person_pid']=new
            if old in self.captured_people:self.captured_people.add(new)
        for person,box,*_ in tracked:
            if person.pid not in self.captured_people:
                self._save(frame,time_s,'person',person.pid,box);self.captured_people.add(person.pid)

    def manifest(self,sessions,run_id):
        result=[];valid={session['session_id'] for session in sessions}
        for item in self.items:
            entry=dict(item);pid=entry.pop('person_pid')
            if pid is not None:
                session=hashlib.sha256(f'{run_id}:{pid}'.encode()).hexdigest()
                if session not in valid:continue
                entry['session_id']=session
            result.append(entry)
        return result
