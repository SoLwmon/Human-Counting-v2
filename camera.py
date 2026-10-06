"""Live camera input: USB webcams (by number) and network streams (phone apps, CCTV / IP cameras)."""
import os
import threading
import time

import cv2


def is_live(source):
    """True for a camera number or a network stream address; False for a video file."""
    return isinstance(source, int) or str(source).lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))


def list_cameras(max_index=6):
    """Print the camera numbers this computer can open (built-in webcam, USB webcams, phone-as-webcam apps)."""
    found = False
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok:
            found = True
            print(f"  camera {i}: working, {frame.shape[1]}x{frame.shape[0]}   ->  --source {i}")
    if not found:
        print("  no cameras found - check the cable, or that no other app is using the camera")


class LiveCamera:
    """Reads a live camera in the background and always hands out the newest picture.

    Network cameras queue up pictures if the program is slower than the camera, so without this the
    video on screen falls further and further behind real life. It also reconnects by itself if the
    connection drops (common with Wi-Fi and CCTV streams).
    """

    def __init__(self, source):
        self.source = source
        if str(source).lower().startswith("rtsp"):
            # CCTV streams are much more stable over TCP than OpenCV's default (must be set before opening)
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self.cap = cv2.VideoCapture(source)
        self.opened = self.cap.isOpened()
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1 <= fps <= 60 else 25.0   # network cameras often report nonsense here
        self.frame, self.count, self.stopped = None, 0, False
        self.lock = threading.Lock()
        if self.opened:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.count = frame, self.count + 1
                continue
            if self.stopped:
                break
            print("Camera connection lost - trying to reconnect in 2 seconds ...")
            self.cap.release()
            time.sleep(2)
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                print("Camera reconnected")

    def read(self, last_count, timeout=1.0):
        """Wait for a picture newer than `last_count`. Returns (count, frame), or (last_count, None) if none came."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if self.count != last_count:
                    return self.count, self.frame
            time.sleep(0.002)
        return last_count, None

    def release(self):
        self.stopped = True
        time.sleep(0.05)
        self.cap.release()
