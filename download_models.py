"""Download the pretrained weights used by booth_counter.py."""
from pathlib import Path
import urllib.request
import gdown
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def download(url, path):
    if path.exists() and path.stat().st_size > 10000:
        print(f"Already present: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    if temporary.stat().st_size < 10000:
        raise RuntimeError(f"Download is too small: {path.name}")
    temporary.replace(path)

if __name__ == "__main__":
    import os
    os.chdir(ROOT)
    YOLO("yolo26n-pose.pt", task="pose")
    body = ROOT / "osnet_x0_25_msmt17.pt"
    if not body.exists():
        temporary = str(body) + ".part"
        result = gdown.download(id="1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e", output=temporary, quiet=False, use_cookies=False)
        if result is None:
            raise RuntimeError("OSNet download failed")
        Path(temporary).replace(body)
    base = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    download(base + "face_detection_yunet/face_detection_yunet_2023mar.onnx", ROOT / "models/face_detection_yunet_2023mar.onnx")
    download(base + "face_recognition_sface/face_recognition_sface_2021dec.onnx", ROOT / "models/face_recognition_sface_2021dec.onnx")
    from osnet import ReID
    from faces import FaceID
    ReID(str(body))
    FaceID()
    print("All model weights downloaded and loaded successfully.")

