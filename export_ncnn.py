"""Export the pose model to NCNN (fastest CPU format on Raspberry Pi)."""
from ultralytics import YOLO

YOLO("yolo26n-pose.pt").export(format="ncnn", imgsz=320)
# Then run:  python booth_counter.py --model yolo26n-pose_ncnn_model --skip 2
