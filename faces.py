"""Optional face matching (turn on with --faces).

Uses OpenCV's built-in YuNet face detector and SFace face recogniser (models in ./models).
Face fingerprints are kept in memory only while the program runs - nothing is written to disk.
Check consent / notice rules (e.g. Malaysia PDPA) with the event organiser before using this.
"""
import cv2
import numpy as np

NOSE, L_EYE, R_EYE = 0, 1, 2
MIN_FACE_PX = 40         # faces smaller than this (far away) are too unreliable to use


class FaceID:
    def __init__(self, detector="models/face_detection_yunet_2023mar.onnx",
                 recogniser="models/face_recognition_sface_2021dec.onnx"):
        self.det = cv2.FaceDetectorYN.create(detector, "", (320, 320), 0.8, 0.3, 50)
        self.rec = cv2.FaceRecognizerSF.create(recogniser, "")

    def embed(self, frame, boxes, kxys, kconfs):
        """One unit-length face fingerprint per person, or None where no clear face is visible."""
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        _, found = self.det.detect(frame)
        found = [] if found is None else found
        out = []
        for box, kxy, kconf in zip(boxes, kxys, kconfs):
            # The person's face = a detected face containing their nose/eye keypoints, inside their body box
            anchor = kxy[NOSE] if kconf[NOSE] > 0.3 else (kxy[L_EYE] + kxy[R_EYE]) / 2
            best = None
            for f in found:
                fx, fy, fw, fh = f[:4]
                inside = fx <= anchor[0] <= fx + fw and fy <= anchor[1] <= fy + fh
                in_box = box[0] - 10 <= fx + fw / 2 <= box[2] + 10
                if inside and in_box and fw >= MIN_FACE_PX and (best is None or f[14] > best[14]):
                    best = f
            if best is None:
                out.append(None)
                continue
            feat = self.rec.feature(self.rec.alignCrop(frame, best)).flatten()
            out.append((feat / max(np.linalg.norm(feat), 1e-6)).astype(np.float32))
        return out
