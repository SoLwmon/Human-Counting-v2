"""Build a replayable batch payload and upload it to the dashboard API."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request


def parse_recorded_at(value):
    if not value or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})', value):
        raise ValueError('Supply --recorded-at as ISO 8601 with timezone, e.g. 2026-10-12T09:00:00+08:00.')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return result.astimezone(timezone.utc)


def api_endpoint(url):
    parts = urlsplit(url)
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('The API URL must not contain credentials, a query string, or a fragment.')
    if parts.scheme not in ('http', 'https') or not parts.hostname:
        raise ValueError('Supply an HTTP(S) website URL.')
    if parts.scheme != 'https' and parts.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Use HTTPS for a remote server. HTTP is permitted only for localhost testing.')
    if parts.path.rstrip('/') not in ('', '/api/sessions/batch'):
        raise ValueError('Use the website root URL or its /api/sessions/batch endpoint.')
    return url.rstrip('/') if parts.path.rstrip('/') else url.rstrip('/') + '/api/sessions/batch'


def recording_id(source, recorded_at, camera_id, live_run_id=None):
    digest = hashlib.sha256()
    if live_run_id is None:
        print('Calculating video fingerprint for repeatable upload IDs...')
        with open(source, 'rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
    else:
        digest.update(live_run_id.encode())
    digest.update(json.dumps([recorded_at.isoformat(), camera_id], separators=(',', ':')).encode())
    return digest.hexdigest()


def build_payload(people, recorded_at, run_id, camera_id):
    records = []
    for person in sorted(people, key=lambda p: p.pid):
        role = getattr(person,'detected_role','visitor')
        if not person.visitor and role != 'staff':
            continue
        start = recorded_at + timedelta(seconds=person.first_seen)
        end = recorded_at + timedelta(seconds=person.last_seen)
        records.append({
            'session_id': hashlib.sha256(f'{run_id}:{person.pid}'.encode()).hexdigest(),
            'visitor_id': f'{camera_id}:{run_id}:person{person.pid}',
            'start_time': start.isoformat(timespec='milliseconds'),
            'end_time': end.isoformat(timespec='milliseconds'),
            'dwell_time': round(person.stay_time),
            'gender': 'Unknown', 'age_group': 'Unknown',
            'person_role': role, 'staff_label': getattr(person,'staff_label',None), 'staff_match_score': getattr(person,'staff_match_score',None),
        })
    # The API derives Kuala Lumpur shifts. This metadata is useful when reviewing/retrying a file.
    return {'recorded_at': recorded_at.isoformat(), 'recording_id': run_id, 'camera_id': camera_id, 'sessions': records}


def save_payload(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)
    print(f'Saved upload payload: {path}')



def upload_payload(payload, url, api_key=None, payload_path=None):
    endpoint = api_endpoint(url)
    if not isinstance(payload, dict) or not isinstance(payload.get('sessions'), list):
        raise ValueError('The JSON file must contain a sessions array.')
    key = api_key or os.environ.get('BOOTH_API_KEY')
    if urlsplit(endpoint).hostname not in ('localhost', '127.0.0.1', '::1') and not key:
        raise ValueError('Set BOOTH_API_KEY to the server upload key before uploading remotely.')
    headers = {'Authorization': f'Bearer {key}'} if key else {}
    from urllib.request import HTTPRedirectHandler, build_opener
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs): return None
    opener=build_opener(NoRedirect)
    root=endpoint.removesuffix('/api/sessions/batch')
    def send(address,body,content_type='application/json'):
        request=Request(address,data=body,headers={**headers,'Content-Type':content_type},method='POST')
        try:
            with opener.open(request,timeout=60) as response:result=json.load(response)
        except HTTPError as error:
            message=error.read(4096).decode('utf-8',errors='replace')
            raise RuntimeError(f'Upload rejected (HTTP {error.code}): {message}') from error
        except (URLError,TimeoutError,OSError) as error:
            raise RuntimeError('Upload could not be confirmed. Retry the saved JSON; stable IDs protect completed uploads.') from error
        if not isinstance(result,dict) or result.get('success') is not True:raise RuntimeError('Unexpected upload acknowledgement. Retry the same saved JSON.')
        return result
    evidence=payload.get('evidence',[])
    if not isinstance(evidence,list):raise ValueError('Evidence must be an array.')
    paths=[]
    if evidence:
        if not payload_path or not payload.get('evidence_directory'):raise ValueError('Evidence upload needs the saved JSON path and its local image directory.')
        directory=(Path(payload_path).resolve().parent/payload['evidence_directory']).resolve()
        for item in evidence:
            filename=item.get('file','')
            if not re.fullmatch(r'[a-f0-9]{64}\.jpg',filename):raise ValueError('Invalid evidence filename.')
            image=(directory/filename).resolve()
            if image.parent!=directory or not image.is_file():raise ValueError(f'Evidence file missing or outside its folder: {filename}')
            if image.stat().st_size>4*1024*1024:raise ValueError('An evidence image exceeds 4 MB.')
            paths.append(image)
    if payload.get('recording'):
        send(root+'/api/recordings',json.dumps(payload['recording']).encode())
    if payload['sessions']:
        body=json.dumps({'sessions':payload['sessions']}).encode('utf-8')
        if len(body)>10*1024*1024 or len(payload['sessions'])>20000:raise ValueError('Batch exceeds 10 MB or 20,000 sessions. Split the saved sessions into smaller batches preserving IDs.')
        result=send(endpoint,body)
        if result.get('received')!=len(payload['sessions']) or any(type(result.get(field)) is not int or result[field]<0 for field in ('count','duplicates')) or result['count']+result['duplicates']!=result['received']:
            raise RuntimeError('Upload acknowledgement does not match the batch. Retry the same saved JSON.')
        print(f"Upload confirmed: {result['count']} new sessions, {result['duplicates']} already present.")
    else:
        result={'count':0,'duplicates':0};print('No qualifying visitor/staff records. Uploading camera snapshots only.')
    from urllib.parse import urlencode,quote
    for index,(item,image) in enumerate(zip(evidence,paths),1):
        query={'time_s':item['time_s'],'kind':item['kind']}
        if item.get('session_id'):query['session_id']=item['session_id']
        address=root+f"/api/recordings/{quote(payload['recording']['id'],safe='')}/evidence/{quote(item['id'],safe='')}?"+urlencode(query)
        send(address,image.read_bytes(),'image/jpeg')
        if index==len(evidence) or index%50==0:print(f'Evidence uploaded: {index}/{len(evidence)} images')
    if payload.get('recording'):
        completion={'session_ids':[item['session_id'] for item in payload['sessions']],'evidence_ids':[item['id'] for item in evidence]}
        send(root+'/api/recordings/'+quote(payload['recording']['id'],safe='')+'/complete',json.dumps(completion).encode())
        print('Recording upload complete.')
    return result


def main():
    parser = argparse.ArgumentParser(description='Retry a saved JSON upload without processing the video again.')
    parser.add_argument('payload', help='Saved JSON file from booth_counter.py')
    parser.add_argument('--api-url', required=True, help='HTTPS website root URL (localhost HTTP allowed for testing)')
    args = parser.parse_args()
    try:
        payload = json.loads(Path(args.payload).read_text(encoding='utf-8'))
        upload_payload(payload, args.api_url, payload_path=args.payload)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
