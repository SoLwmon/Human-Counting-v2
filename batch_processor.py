"""Process a CCTV day as ordered clips, sharing one tracker and one re-ID counter."""
import argparse
import csv
from datetime import datetime,timedelta,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from session_upload import parse_recorded_at,recording_id,save_payload,upload_payload,api_endpoint

MYT=timezone(timedelta(hours=8))
VIDEO_SUFFIXES={'.mp4','.avi','.mkv','.mov','.m4v','.ts'}


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)',path.name)]


def make_plan(directory,start,minutes=30,expected=24):
    root=Path(directory).resolve()
    files=sorted([file for file in root.iterdir() if file.is_file() and file.suffix.lower() in VIDEO_SUFFIXES],key=natural_key)
    if len(files)!=expected:raise ValueError(f'Expected {expected} clips, found {len(files)}. Use a folder containing this camera/day only, or supply an explicit --playlist CSV.')
    if not math.isfinite(minutes) or minutes<=0:raise ValueError('--clip-minutes must be positive.')
    return [{'file':str(file),'recorded_at':(start+timedelta(minutes=index*minutes)).astimezone(MYT).isoformat()} for index,file in enumerate(files)]


def read_plan(path):
    path=Path(path).resolve()
    with path.open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        if not reader.fieldnames or not {'file','recorded_at'}.issubset(reader.fieldnames):raise ValueError('Playlist needs file and recorded_at columns.')
        rows=[]
        for row in reader:
            if not row.get('file') or not row.get('recorded_at'):raise ValueError('Every playlist row needs a file and recording timestamp.')
            file=Path(row['file']);file=file if file.is_absolute() else path.parent/file
            rows.append({'file':str(file.resolve()),'recorded_at':parse_recorded_at(row['recorded_at']).isoformat()})
    if not rows:raise ValueError('Playlist is empty.')
    return rows


def write_plan(rows,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['file','recorded_at']);writer.writeheader();writer.writerows(rows)


def inspect_plan(rows,require_duration=None,skip=1,trim_overlaps=False):
    import cv2
    result=[];seen=set();shape=None;previous=None
    for index,row in enumerate(rows):
        path=Path(row['file']).resolve();start=parse_recorded_at(row['recorded_at'])
        if not path.is_file():raise ValueError(f'Missing clip {index+1}: {path}')
        if str(path).casefold() in seen:raise ValueError('The same file is listed more than once.')
        seen.add(str(path).casefold())
        cap=cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():raise ValueError(f'Cannot open {path.name}. Export the CCTV clip to a supported video format.')
            fps=float(cap.get(cv2.CAP_PROP_FPS));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        finally:cap.release()
        if not math.isfinite(fps) or fps<=0 or frames<=0 or min(size)<=0:raise ValueError(f'{path.name}: invalid FPS/frame metadata. Re-export this recording.')
        duration=frames/fps
        if skip/fps>1:raise ValueError('--skip would leave more than a second between analysed frames and break duration measurements. Reduce --skip.')
        if require_duration is not None and abs(duration-require_duration)>max(2,2/fps):raise ValueError(f'{path.name} is {duration:.2f}s, not the declared {require_duration:.2f}s. Use an explicit timestamp playlist for short/missing clips.')
        if shape is not None and shape!=size:raise ValueError('All clips must be from the same camera and resolution.')
        shape=size
        if previous is not None:
            gap=(start-previous['start']).total_seconds()-previous['duration_s']
            tolerance=max(1/fps,1/previous['fps'])+0.001
            if gap < -tolerance:
                if not trim_overlaps or gap < -2:
                    raise ValueError(f'{path.name} overlaps the previous clip by {-gap:.3f}s. Correct the timestamps, or use --trim-overlaps for overlaps of at most 2 seconds.')
                frames_to_keep=math.floor((start-previous['start']).total_seconds()*previous['fps']+0.000001)
                if frames_to_keep<1:raise ValueError('Overlap would remove the entire previous clip. Correct the recording timestamps/order.')
                previous['frames']=frames_to_keep
                previous['duration_s']=frames_to_keep/previous['fps']
                print(f"Trimming {Path(previous['file']).name}: ignoring {-gap:.3f}s at its end to avoid overlapping {path.name}. Original file is unchanged.")
                gap=(start-previous['start']).total_seconds()-previous['duration_s']
            if abs(gap)>tolerance:print(f'Coverage gap before {path.name}: {gap:.2f}s. Missing time will not be counted.')
        entry={'file':str(path),'start':start,'fps':fps,'frames':frames,'duration_s':duration,'size':size,'index':index}
        result.append(entry);previous=entry
    return result


def period_of(value):
    local=value.astimezone(MYT)
    shift=1 if 9<=local.hour<15 else 2 if 15<=local.hour<21 else 0
    return local.date().isoformat(),shift


def next_boundary(value):
    local=value.astimezone(MYT);midnight=local.replace(hour=0,minute=0,second=0,microsecond=0)
    for hour in (9,15,21,24):
        boundary=midnight+timedelta(hours=hour)
        if boundary>local:return boundary.astimezone(timezone.utc)
    raise AssertionError('No time boundary')


class FragmentLedger:
    """Split actual observed-time increments by clip and Malaysian day/shift."""
    def __init__(self,clips,origin):
        self.clips=clips;self.origin=origin;self.segments={};self.totals={}
        for clip in clips:clip['offset_s']=(clip['start']-origin).total_seconds()

    def remap(self,changes):
        for old,new in changes.items():
            self.totals[new]=self.totals.get(new,0)+self.totals.pop(old,0)
            for key in list(self.segments):
                if key[1]!=old:continue
                segment=self.segments.pop(key);target=(key[0],new,key[2],key[3]);existing=self.segments.get(target)
                if existing:
                    existing['first']=min(existing['first'],segment['first']);existing['last']=max(existing['last'],segment['last']);existing['seconds']+=segment['seconds']
                else:self.segments[target]=segment

    def _add(self,index,pid,start,end,seconds):
        stamp=self.origin+timedelta(seconds=start);day,shift=period_of(stamp)
        key=(index,pid,day,shift);segment=self.segments.setdefault(key,{'first':start,'last':end,'seconds':0.0})
        segment['first']=min(segment['first'],start);segment['last']=max(segment['last'],end);segment['seconds']+=seconds

    def observe(self,people,t):
        for person in {person.pid:person for person in people}.values():
            old=self.totals.get(person.pid,0.0);delta=max(0.0,person.stay_time-old);self.totals[person.pid]=person.stay_time
            start=max(person.first_seen,t-delta)
            if delta>0:
                for clip in self.clips:
                    left=max(start,clip['offset_s']);right=min(t,clip['offset_s']+clip['duration_s'])
                    while left<right-1e-8:
                        boundary=(next_boundary(self.origin+timedelta(seconds=left))-self.origin).total_seconds()
                        end=min(right,boundary);self._add(clip['index'],person.pid,left,end,end-left);left=end
            # Preserve an actual observation at the first frame of the next clip, even if its time increment
            # belongs to the previous interval. Later frames add duration to this fragment.
            for clip in self.clips:
                if clip['offset_s']-1e-7<=t<clip['offset_s']+clip['duration_s']-1e-7:
                    self._add(clip['index'],person.pid,t,t,0);break

    def sessions(self,clip,counter,group_id,camera_id,campaign_id):
        result=[]
        for (index,pid,day,shift),segment in sorted(self.segments.items()):
            if index!=clip['index'] or pid not in counter.people:continue
            person=counter.people[pid]
            sid=hashlib.sha256(f"fragment-v1:{clip['id']}:{pid}:{day}:{shift}".encode()).hexdigest()
            result.append({'session_id':sid,'visitor_id':f'{camera_id}:{group_id}:person{pid}',
                'start_time':(self.origin+timedelta(seconds=segment['first'])).isoformat(timespec='milliseconds'),
                'end_time':(self.origin+timedelta(seconds=segment['last'])).isoformat(timespec='milliseconds'),
                'dwell_time':round(segment['seconds'],3),'qualifies_visitor':bool(person.visitor),
                'gender':'Unknown','age_group':'Unknown','person_role':person.detected_role,
                'staff_label':person.staff_label,'staff_match_score':person.staff_match_score,
                'campaign_id':campaign_id,'recording_id':clip['id'],
                '_pid':pid,'_first_s':segment['first']-clip['offset_s'],'_last_s':segment['last']-clip['offset_s']})
        return result


def clip_evidence(captures,sessions):
    result=[]
    for item in captures.items:
        entry=dict(item);pid=entry.pop('person_pid')
        if pid is not None:
            matches=[session for session in sessions if session['_pid']==pid and session['_first_s']-0.002<=item['time_s']<=session['_last_s']+0.002]
            if not matches:continue
            entry['session_id']=matches[0]['session_id']
        result.append(entry)
    return result


def reset_after_gap(model,counter):
    for tracker in getattr(getattr(model,'predictor',None),'trackers',[]):tracker.reset()
    if counter is None:return
    counter.track_to_pid.clear();counter.pending.clear();counter.frame_pids.clear();counter.last_check.clear();counter.mismatches.clear();counter.dropped.clear()
    for person in counter.people.values():
        person.last_t=None;person.hip_track=[];person.still_since=None;person.looking=False


def retry_uploads(output,url):
    files=sorted(Path(output).glob('clip-*.json'))
    if not files:raise ValueError('No saved clip payloads found in the output directory.')
    failed=[]
    for path in files:
        try:upload_payload(json.loads(path.read_text(encoding='utf-8')),url,payload_path=path)
        except (ValueError,RuntimeError,OSError) as error:print(f'{path.name}: {error}');failed.append(path.name)
    if failed:raise RuntimeError(f'{len(failed)} clip uploads still failed. Results remain saved; retry later.')
    print(f'All {len(files)} saved clips delivered. No video processing was needed.')


def run(args):
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    if args.api_url:
        endpoint=api_endpoint(args.api_url)
        from urllib.parse import urlsplit
        if urlsplit(endpoint).hostname not in ('localhost','127.0.0.1','::1') and not os.environ.get('BOOTH_API_KEY'):raise ValueError('Set BOOTH_API_KEY before processing for remote uploads, or omit --api-url for offline processing.')
    if args.retry_uploads:
        if not args.api_url:raise ValueError('--retry-uploads requires --api-url.')
        retry_uploads(output,args.api_url);return
    if args.playlist:rows=read_plan(args.playlist);expected=None
    elif args.input_dir:
        rows=make_plan(args.input_dir,parse_recorded_at(args.recorded_at),args.clip_minutes,args.expected_clips);expected=args.clip_minutes*60
    else:raise ValueError('Supply --playlist or --input-dir with --recorded-at.')
    write_plan(rows,output/'playlist.csv')
    clips=inspect_plan(rows,expected,args.skip,args.trim_overlaps)
    print(f'Validated {len(clips)} clips. Timeline: {clips[0]["start"].astimezone(MYT).isoformat()} to {(clips[-1]["start"]+timedelta(seconds=clips[-1]["duration_s"])).astimezone(MYT).isoformat()}')
    if args.prepare_only:print(f'Review {output / "playlist.csv"} before running this playlist.');return
    if not args.campaign_id or not args.camera_id or len(args.camera_id)>120:raise ValueError('Supply a campaign ID and a stable camera name (up to 120 characters).')
    if args.api_url:
        from urllib.request import Request,build_opener,HTTPRedirectHandler
        from urllib.error import HTTPError,URLError
        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self,*values,**options):return None
        key=os.environ.get('BOOTH_API_KEY');headers={'Content-Type':'application/json',**({'Authorization':f'Bearer {key}'} if key else {})}
        check=Request(api_endpoint(args.api_url).removesuffix('/api/sessions/batch')+'/api/uploads/check',data=json.dumps({'campaign_id':args.campaign_id}).encode(),headers=headers,method='POST')
        try:
            with build_opener(NoRedirect).open(check,timeout=15) as response:result=json.load(response)
            if result.get('success') is not True:raise ValueError('API upload check failed.')
        except (HTTPError,URLError,OSError,ValueError) as error:raise ValueError('Upload preflight failed. Check the URL, upload key and campaign ID, and deploy the updated website backend. Omit --api-url to process offline.') from error
    import cv2
    from ultralytics import YOLO
    from importlib.metadata import version
    from booth_counter import Counter,enhance,draw,DET_CONF
    from osnet import ReID
    from faces import FaceID
    from staff_identity import StaffMatcher,weights_hash
    from snapshot_capture import SnapshotCapture
    zone=tuple(float(value) for value in args.zone.split(','))
    if len(zone)!=4 or not (0<=zone[0]<zone[2]<=1 and 0<=zone[1]<zone[3]<=1):raise ValueError('Invalid counting zone.')
    print('Fingerprinting clips for repeatable recording and visitor IDs...')
    for clip in clips:
        identity=recording_id(clip['file'],clip['start'],args.camera_id)
        clip['id']=hashlib.sha256(f'{identity}:{args.campaign_id}'.encode()).hexdigest() if args.campaign_id!='legacy' else identity
    group_id=hashlib.sha256(json.dumps([args.campaign_id,args.camera_id,[clip['id'] for clip in clips]],separators=(',',':')).encode()).hexdigest()
    config={'trim_overlaps':args.trim_overlaps,'clip_frames':[clip['frames'] for clip in clips],'model':weights_hash(args.model),'reid':weights_hash(args.reid),'staff':hashlib.sha256(Path(args.staff_profiles).read_bytes()).hexdigest() if args.staff_profiles else '',
        'faces':args.faces,'enhance':not args.no_enhance,'zone':zone,'imgsz':args.imgsz,'skip':args.skip,'snapshot_interval':args.snapshot_interval,'ultralytics':version('ultralytics'),'staff_body_match':args.staff_body_match,'staff_face_match':args.staff_face_match,'torch':version('torch'),'opencv':version('opencv-python'),'software':hashlib.sha256(b''.join((Path(__file__).parent/name).read_bytes() for name in ['batch_processor.py','booth_counter.py','staff_identity.py','snapshot_capture.py'])).hexdigest(),'batch_version':1}
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    job={'analysis_group_id':group_id,'processing_fingerprint':fingerprint,'clip_ids':[clip['id'] for clip in clips]}
    state_path=output/'job.json'
    if state_path.exists() and json.loads(state_path.read_text(encoding='utf-8'))!=job:raise ValueError('This output directory belongs to a different playlist/configuration. Use a new directory. Do not upload another analysis of already-imported footage without an explicit correction plan.')
    state_path.write_text(json.dumps(job,indent=2),encoding='utf-8')
    model=YOLO(args.model,task='pose');reid=ReID(args.reid);faces=FaceID() if args.faces else None
    staff=StaffMatcher(args.staff_profiles,args.reid,args.staff_body_match,args.staff_face_match) if args.staff_profiles else None
    counter=None;origin=clips[0]['start'];ledger=FragmentLedger(clips,origin);captures={};queue=[];failures=[];finished=[];previous=None;global_frame=0
    def export(index):
        clip=clips[index];sessions=ledger.sessions(clip,counter,group_id,args.camera_id,args.campaign_id)
        images=clip_evidence(captures[index],sessions)
        clean=[{key:value for key,value in row.items() if not key.startswith('_')} for row in sessions]
        payload={'recording_id':clip['id'],'analysis_group_id':group_id,'recorded_at':clip['start'].isoformat(),'camera_id':args.camera_id,'campaign_id':args.campaign_id,
            'recording':{'id':clip['id'],'campaign_id':args.campaign_id,'camera_id':args.camera_id,'recorded_at':clip['start'].isoformat(),'source_name':Path(clip['file']).name,
                'duration_s':round(clip['duration_s'],3),'snapshot_interval_s':args.snapshot_interval,'analysis_group_id':group_id,'processing_fingerprint':fingerprint},
            'sessions':clean,'evidence':images,'evidence_directory':os.path.relpath(captures[index].directory,output)}
        path=output/f'clip-{index+1:03d}.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=payload:raise RuntimeError(f'{path.name} differs from the saved analysis. Existing results were retained; use --retry-uploads to deliver the saved results.')
        save_payload(payload,path)
        with (output/f'clip-{index+1:03d}.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            fields=['session_id','visitor_id','start_time','end_time','dwell_time','person_role','staff_label','qualifies_visitor']
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(clean)
        finished.append(path.name)
        if args.api_url:
            try:upload_payload(payload,args.api_url,payload_path=path)
            except (ValueError,RuntimeError,OSError) as error:print(f'Upload pending for clip {index+1}: {error}');failures.append(path.name)
        print(f'Clip {index+1}/{len(clips)} saved: {len(clean)} observation fragments, {len(images)} images.')
    try:
        for clip in clips:
            print(f'Processing clip {clip["index"]+1}/{len(clips)}: {Path(clip["file"]).name} ({clip["start"].astimezone(MYT).isoformat()})')
            if previous is not None:
                gap=clip['offset_s']-(previous['offset_s']+previous['duration_s'])
                if abs(gap)>max(1/clip['fps'],1/previous['fps'])+0.001:
                    reset_after_gap(model,counter)
                    while queue:export(queue.pop(0))
            cap=cv2.VideoCapture(clip['file']);captures[clip['index']]=SnapshotCapture(output/'evidence'/clip['id'],clip['id'],args.snapshot_interval)
            frame_index=0;last_progress=time.monotonic()
            try:
                while frame_index<clip['frames']:
                    ok,frame=cap.read()
                    if not ok:break
                    local_t=frame_index/clip['fps'];t=clip['offset_s']+local_t;frame_index+=1;global_frame+=1
                    captures[clip['index']].periodic(frame,local_t)
                    if global_frame%args.skip:continue
                    if counter is None:counter=Counter(frame.shape,reid,faces,staff);counter.set_zone(zone)
                    analysis=frame if args.no_enhance else enhance(frame)
                    result=model.track(analysis,persist=True,tracker='bytetrack.yaml',imgsz=args.imgsz,classes=[0],conf=DET_CONF,verbose=False)[0]
                    tracked=counter.update(analysis,result,t);ledger.remap(counter.remap)
                    # Keep evidence associations in unpublished clips in step with late identity merges.
                    for manager in captures.values():manager.people(frame,[],local_t,counter.remap)
                    captures[clip['index']].people(frame,tracked,local_t)
                    ledger.observe([person for person,*_ in tracked],t)
                    while queue and not any(pending['since']<clips[queue[0]]['offset_s']+clips[queue[0]]['duration_s'] for pending in counter.pending.values()):export(queue.pop(0))
                    if args.show:
                        view=draw(frame.copy(),counter,tracked);cv2.putText(view,f'Clip {clip["index"]+1}/{len(clips)} - {local_t/60:.1f} min',(12,frame.shape[0]-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2);cv2.imshow('CCTV batch',view)
                        if cv2.waitKey(1)&0xff in (ord('q'),27):raise KeyboardInterrupt
                    if time.monotonic()-last_progress>=30:
                        print(f'Clip {clip["index"]+1}: video minute {local_t/60:.1f}/{clip["duration_s"]/60:.1f}; {len(counter.people)} tracked identities');last_progress=time.monotonic()
            finally:cap.release()
            if frame_index<clip['frames']-max(2,args.skip):raise RuntimeError(f'{Path(clip["file"]).name} ended early ({frame_index}/{clip["frames"]} frames). No completed payload was created for this clip; check the recording.')
            queue.append(clip['index']);previous=clip
        if counter is None:raise RuntimeError('No frames were processed.')
        # At the final EOF there are no future confirmed observations to resolve pending tracks.
        counter.pending.clear()
        while queue:export(queue.pop(0))
    except KeyboardInterrupt:
        raise RuntimeError('Batch stopped. Completed clip JSON/CSV/images are retained. Retry uploads without processing using --retry-uploads. Continuing analysis requires replaying the playlist from the start to rebuild tracking state.')
    finally:
        if args.show:cv2.destroyAllWindows()
        (output/'batch-summary.json').write_text(json.dumps({'analysis_group_id':group_id,'completed_clips':finished,'pending_uploads':failures},indent=2),encoding='utf-8')
    print(f'Processed {len(finished)} clips. Saved outputs: {output}')
    if failures:raise RuntimeError(f'{len(failures)} uploads remain pending. Processing continued; use --retry-uploads later.')


def main():
    parser=argparse.ArgumentParser(description='Process ordered clips from one camera with continuous tracking and automatic uploads.')
    source=parser.add_mutually_exclusive_group();source.add_argument('--input-dir');source.add_argument('--playlist')
    parser.add_argument('--trim-overlaps',action='store_true',help='Ignore up to 2 seconds at the end of a clip if the following timestamp overlaps it; original videos are unchanged')
    parser.add_argument('--recorded-at',default='');parser.add_argument('--clip-minutes',type=float,default=30);parser.add_argument('--expected-clips',type=int,default=24)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--retry-uploads',action='store_true');parser.add_argument('--output-dir',default='batch-output')
    parser.add_argument('--campaign-id',default='');parser.add_argument('--camera-id',default='booth-1');parser.add_argument('--api-url',default='');parser.add_argument('--staff-profiles',default='')
    parser.add_argument('--model',default='yolo26n-pose.pt');parser.add_argument('--reid',default='osnet_x0_25_msmt17.pt');parser.add_argument('--faces',action='store_true');parser.add_argument('--no-enhance',action='store_true')
    parser.add_argument('--imgsz',type=int,default=320);parser.add_argument('--skip',type=int,default=1);parser.add_argument('--zone',default='0.15,0.10,0.85,0.95');parser.add_argument('--show',action='store_true')
    parser.add_argument('--snapshot-interval',type=int,choices=[60,300],default=300);parser.add_argument('--staff-body-match',type=float,default=0.85);parser.add_argument('--staff-face-match',type=float,default=0.45)
    args=parser.parse_args()
    if args.skip<1:parser.error('--skip must be at least 1.')
    if args.expected_clips<1:parser.error('--expected-clips must be positive.')
    try:run(args)
    except (ValueError,RuntimeError,OSError) as error:parser.exit(1,f'{error}\n')


if __name__=='__main__':main()
