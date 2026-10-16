"""Select staff from a recorded video and save local appearance profiles."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from osnet import ReID
from faces import FaceID
from staff_identity import weights_hash


def main():
    parser=argparse.ArgumentParser(description='Enroll staff from a clear point in a recorded video. Select boxes, then S to save.')
    parser.add_argument('--source',required=True)
    parser.add_argument('--at',type=float,default=0,help='Video time in seconds where staff are clearly visible')
    parser.add_argument('--out',default='staff_profiles.json')
    parser.add_argument('--model',default='yolo26n-pose.pt');parser.add_argument('--reid',default='osnet_x0_25_msmt17.pt')
    parser.add_argument('--no-enhance',action='store_true');parser.add_argument('--faces',action='store_true');parser.add_argument('--name',default='Staff',help='Staff label; multi-selection adds a number')
    parser.add_argument('--select',default='',help='For repeatable enrollment without a window: comma-separated detected person numbers, starting at 1')
    args=parser.parse_args()
    if args.at<0:parser.error('--at must be non-negative.')
    cap=cv2.VideoCapture(args.source)
    if not cap.isOpened():parser.error('Could not open this recording.')
    cap.set(cv2.CAP_PROP_POS_MSEC,args.at*1000);ok,frame=cap.read();cap.release()
    if not ok:parser.error('No frame at this timestamp. Select a time inside the recording.')
    from booth_counter import enhance
    analysis=frame if args.no_enhance else enhance(frame)
    model=YOLO(args.model,task='pose');result=model.predict(analysis,imgsz=640,classes=[0],conf=0.5,verbose=False)[0]
    boxes=result.boxes.xyxy.cpu().numpy();keys=result.keypoints.xy.cpu().numpy();confs=result.keypoints.conf.cpu().numpy()
    if not len(boxes):parser.error('No clear people detected at this timestamp. Choose another --at value.')
    selected=set()
    if args.select:
        try:selected={int(value)-1 for value in args.select.split(',')}
        except ValueError:parser.error('--select must contain comma-separated person numbers.')
        if any(index<0 or index>=len(boxes) for index in selected):parser.error('Selected person number was not detected.')
    else:
        window='Enroll staff - click boxes, S save, Q cancel'
        cv2.namedWindow(window,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
        def mouse(event,x,y,flags,param):
            if event!=cv2.EVENT_LBUTTONDOWN:return
            matches=[(float((b[2]-b[0])*(b[3]-b[1])),i) for i,b in enumerate(boxes) if b[0]<=x<=b[2] and b[1]<=y<=b[3]]
            if matches:
                index=min(matches)[1]
                if index in selected:selected.remove(index)
                else:selected.add(index)
        cv2.setMouseCallback(window,mouse)
        while True:
            view=frame.copy()
            for index,box in enumerate(boxes):
                x1,y1,x2,y2=map(int,box);color=(0,220,0) if index in selected else (150,150,150)
                cv2.rectangle(view,(x1,y1),(x2,y2),color,2);cv2.putText(view,f'{index+1}'+(' STAFF' if index in selected else ''),(x1,max(y1-8,22)),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
            cv2.imshow(window,view);key=cv2.waitKey(20)&0xff
            if key in (ord('q'),27):cv2.destroyAllWindows();parser.exit(1,'Enrollment cancelled.\n')
            if key==ord('s') and selected:break
            if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:parser.exit(1,'Enrollment window closed.\n')
        cv2.destroyAllWindows()
    reid=ReID(args.reid);faces=FaceID() if args.faces else None
    body=reid.embed(analysis,boxes);face=faces.embed(analysis,boxes,keys,confs) if faces else [None]*len(boxes)
    path=Path(args.out);data={'version':1,'weights_sha256':weights_hash(args.reid),'profiles':[]}
    if path.exists():
        data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('weights_sha256')!=weights_hash(args.reid):parser.error('Existing profiles use different weights. Choose a new --out file.')
    for index in sorted(selected):
        name=args.name if len(selected)==1 else f'{args.name}-{index+1}'
        profile=next((p for p in data['profiles'] if p['name']==name),None)
        if profile is None:profile={'name':name,'bodies':[],'faces':[]};data['profiles'].append(profile)
        profile['bodies']=(profile['bodies']+[body[index].tolist()])[-20:]
        if face[index] is not None:profile['faces']=(profile['faces']+[face[index].tolist()])[-20:]
        print(f'Enrolled detected person {index+1} as {name}')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(f'Saved {path}. Add samples for the same staff label at other clear timestamps to improve coverage.')


if __name__=='__main__':main()
