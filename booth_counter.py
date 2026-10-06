"""Smart booth visitor counter.

Pipeline per frame:  pose detection -> ByteTrack IDs -> re-ID (face if --faces, else body) -> stop/look rules -> CSV.
Run:  python booth_counter.py --source video.mp4 --show
Sources: a camera number (0, 1, ...), a video file, or a network camera address (rtsp://... or http://...).
"""
import argparse
import csv
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

from camera import LiveCamera, is_live, list_cameras
from faces import FaceID
from osnet import ReID

# ---- Tunable thresholds (tune these on real booth footage) ----
KP_CONF = 0.5            # min keypoint confidence for nose/eyes
EAR_CONF = 0.3           # ears are often low-confidence; lower bar
STOP_SPEED = 0.05        # hip movement < 5% of box height per second = stopped
STOP_MIN_S = 2.0         # must be still this long before counting as stopped
LOOK_DEBOUNCE_S = 0.5    # new looking/not-looking state must hold this long
VISITOR_MIN_S = 5.0      # total time on camera to become a "visitor" (shorter = just passing through)
REID_MEMORY_S = 600.0    # remember people who left for this long (seconds)
REID_MATCH = 0.70        # body similarity (0-1) needed to reuse an old ID; raise if different people get merged
FACE_MATCH = 0.36        # face similarity needed to reuse an old ID (--faces only); OpenCV's recommended value
REID_MARGIN = 0.05       # if two *different-looking* old people score this close, treat the newcomer as new
COLOUR_WEIGHT = 0.0      # share of the body score from shirt colour (rest from OSNet); >0 merged look-alike shirts
CONFIRM_SAMPLES = 5      # appearance samples to collect before deciding new vs returning person
CONFIRM_MIN_S = 0.3      # a track must last this long to count (filters flicker/false detections)
CONFIRM_MAX_S = 1.5      # decide after this long even without enough samples
GALLERY_SIZE = 20        # appearance samples remembered per person
CHECK_EVERY_S = 0.5      # re-check that a tracked person is still the same person this often
SWITCH_CHECKS = 3        # re-checks in a row scoring below REID_MATCH = the tracker swapped people
DET_CONF = 0.5           # ignore person detections below this confidence
BLUR_MIN = 250           # samples less sharp than this (motion blur) are skipped for identifying people
BLUR_GIVE_UP_S = 4.0     # ...unless a new person has been blurry for this long, then use what we have
FACE_MERGE = 0.41        # two IDs whose faces match this well (never on screen together) get merged into one
# Counting zone: a rectangle inside the picture, as fractions of its width/height (left, top, right, bottom).
# Tracking and identification only start once the centre of a person's body is inside it, so people who
# just walk past outside it are ignored. Change with --zone, or drag a new one with the mouse in the window.
ZONE = (0.15, 0.10, 0.85, 0.95)

# COCO keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12


@dataclass
class Person:
    pid: int
    first_seen: float
    last_seen: float
    gallery: list = field(default_factory=list)     # body appearance fingerprints
    face_gallery: list = field(default_factory=list)  # face fingerprints (only with --faces; memory only)
    last_gallery_t: float = 0.0
    hip_track: list = field(default_factory=list)   # [(t, x, y, box_h)]
    looking: bool = False
    pending_look: bool = False
    pending_since: float = 0.0
    still_since: float | None = None
    look_time: float = 0.0   # time spent standing still and facing the camera (kept as extra info)
    stay_time: float = 0.0   # total time on camera since being picked up in the counting zone
    last_t: float | None = None

    @property
    def visitor(self) -> bool:
        return self.stay_time >= VISITOR_MIN_S


def at_edge(box, frame_w):
    """Person is cut off by the left/right edge of the screen (walking in/out) - a poor sample."""
    return box[0] < 0.02 * frame_w or box[2] > 0.98 * frame_w


def shirt_colour(frame, kxy, kconf, box):
    """Colour fingerprint of the shirt: unit-length sqrt(hue x saturation histogram).

    Region is between the shoulders and hips (or ~1.2 shoulder-widths down when hips are out of
    view, e.g. close to a webcam); side-on people fall back to the upper-middle of their box.
    """
    ok = lambda *idx: all(kconf[i] > 0.4 for i in idx)
    if ok(L_SHO, R_SHO) and abs(kxy[L_SHO, 0] - kxy[R_SHO, 0]) > 10:
        sx1, sx2 = sorted([kxy[L_SHO, 0], kxy[R_SHO, 0]])
        sy = (kxy[L_SHO, 1] + kxy[R_SHO, 1]) / 2
        width = sx2 - sx1
        bottom = (kxy[L_HIP, 1] + kxy[R_HIP, 1]) / 2 if ok(L_HIP, R_HIP) else min(sy + 1.2 * width, box[3])
        x1, y1, x2, y2 = sx1 + 0.15 * width, sy + 0.15 * (bottom - sy), sx2 - 0.15 * width, bottom
    else:
        bw, bh = box[2] - box[0], box[3] - box[1]
        x1, y1, x2, y2 = box[0] + 0.3 * bw, box[1] + 0.3 * bh, box[2] - 0.3 * bw, box[1] + 0.6 * bh
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(max(x1, 0)), int(max(y1, 0)), int(min(x2, w)), int(min(y2, h))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return np.full(128, 1 / np.sqrt(128), np.float32)  # no usable shirt area: neutral
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = np.sqrt(cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten())
    return (hist / max(np.linalg.norm(hist), 1e-6)).astype(np.float32)


def fingerprint(reid, frame, boxes, kxys, kconfs):
    """OSNet + shirt colour, weighted so a dot product = (1-w)*OSNet similarity + w*colour similarity."""
    colour = np.stack([shirt_colour(frame, k, c, b) for b, k, c in zip(boxes, kxys, kconfs)])
    return np.concatenate([np.sqrt(1 - COLOUR_WEIGHT) * reid.embed(frame, boxes),
                           np.sqrt(COLOUR_WEIGHT) * colour], axis=1)


_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def enhance(frame):
    """Even out the lighting (local contrast on brightness only) so people standing in front of a
    bright window are not just dark silhouettes. Used for analysis; the picture shown is unchanged."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[..., 0] = _CLAHE.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sharpness(frame, box):
    """Variance of the Laplacian on the person crop - low means motion blur."""
    x1, y1, x2, y2 = (int(v) for v in box)
    crop = frame[max(y1, 0):max(y2, y1 + 2), max(x1, 0):max(x2, x1 + 2)]
    grey = cv2.cvtColor(cv2.resize(crop, (64, 128)), cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(grey, cv2.CV_64F).var()


def similarity(feats, gallery):
    """How alike a new track is to a stored person: mean of the 3 closest gallery matches."""
    sims = (np.asarray(feats) @ np.asarray(gallery).T).max(axis=0)
    return float(np.sort(sims)[-3:].mean())


def is_looking(kxy, kconf):
    """Facing the camera: nose + both eyes visible, nose horizontally between ears (or shoulders)."""
    if min(kconf[NOSE], kconf[L_EYE], kconf[R_EYE]) < KP_CONF:
        return False
    if kconf[L_EAR] > EAR_CONF and kconf[R_EAR] > EAR_CONF:
        a, b = kxy[L_EAR, 0], kxy[R_EAR, 0]
    elif kconf[L_SHO] > EAR_CONF and kconf[R_SHO] > EAR_CONF:
        a, b = kxy[L_SHO, 0], kxy[R_SHO, 0]
    else:
        return False
    return min(a, b) <= kxy[NOSE, 0] <= max(a, b)


class Counter:
    def __init__(self, frame_shape, reid, faces=None):
        h, w = frame_shape[:2]
        self.frame_w = w
        self.reid = reid
        self.faces = faces                       # FaceID or None
        self.frame_h = h
        self.set_zone(ZONE)
        self.shown: list[int] = []               # detections in this frame that are being tracked (for drawing)
        self.people: dict[int, Person] = {}      # pid -> Person
        self.track_to_pid: dict[int, int] = {}   # ByteTrack id -> our stable pid (kept for the whole run)
        self.pending: dict[int, dict] = {}       # new tracks still collecting appearance samples
        self.next_pid = 1
        self.frame_pids: set[int] = set()        # people visible in the current frame
        self.last_check: dict[int, float] = {}   # tracker id -> time of last identity re-check
        self.mismatches: dict[int, int] = {}     # tracker id -> failed re-checks in a row
        self.log: list[dict] = []                # every new-vs-returning decision, for debugging
        self.together: set[tuple] = set()        # pairs of IDs seen on screen at the same time (can't be one person)
        self.new_faces: set[int] = set()         # IDs that got new face samples this frame (check for merges)
        self.dropped: dict[int, int] = {}        # tracker id -> ID it was unlinked from by a failed re-check
        self.remap: dict[int, int] = {}          # IDs merged away during this frame -> the ID they became

    def set_zone(self, zone):
        """zone = (left, top, right, bottom) as fractions of the picture."""
        x1, y1, x2, y2 = zone
        self.zone = (int(x1 * self.frame_w), int(y1 * self.frame_h), int(x2 * self.frame_w), int(y2 * self.frame_h))

    def _in_zone(self, box):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return self.zone[0] <= cx <= self.zone[2] and self.zone[1] <= cy <= self.zone[3]

    @staticmethod
    def _score(p, bodies, faces):
        """How far above the 'same person' bar a person scores (>= 0 means match).

        A visible face on both sides decides on its own; otherwise body appearance is used.
        """
        if faces and p.face_gallery:
            return similarity(faces, p.face_gallery) - FACE_MATCH, "face"
        return similarity(bodies, p.gallery) - REID_MATCH, "body"

    def _match_returning(self, bodies, faces, t):
        """Compare a new track's appearance with everyone not visible in this frame.

        This covers both people who walked away and came back, and people who never left
        but whose tracker number changed (common when someone stands close to the camera).
        """
        scores = sorted(((*self._score(p, bodies, faces), p.pid) for p in self.people.values()
                         if p.pid not in self.frame_pids and p.gallery and t - p.last_seen <= REID_MEMORY_S),
                        reverse=True)
        self._last_scores = scores[:3]
        if not scores or scores[0][0] < 0:
            return None
        if len(scores) > 1 and scores[1][0] >= 0 and scores[0][0] - scores[1][0] < REID_MARGIN:
            # Two old IDs both match. If those two look alike they're probably the same person
            # split earlier, so take the best; if they look different, it's too close to call.
            a, b = self.people[scores[0][2]], self.people[scores[1][2]]
            if self._score(a, b.gallery, b.face_gallery)[0] < 0:
                return None
        return scores[0][2]

    def _needs_sample(self, tid, box, t):
        if tid in self.track_to_pid:
            return t - self.last_check.get(tid, -1e9) >= CHECK_EVERY_S and not at_edge(box, self.frame_w)
        pend = self.pending.get(tid)
        waited = 0.0 if pend is None else t - pend["since"]
        # Skip samples while someone is still half-way in at the screen edge, unless we've waited a while
        return not at_edge(box, self.frame_w) or waited >= CONFIRM_MAX_S / 2

    def _blurry_too_long(self, tid, t):
        pend = self.pending.get(tid)
        return pend is not None and t - pend["since"] >= BLUR_GIVE_UP_S

    def _resolve_pid(self, tid, sample, t):
        """Return the stable pid for a track, or None while it is still being confirmed."""
        if tid in self.track_to_pid:
            return self.track_to_pid[tid]
        # New track: collect a few appearance samples before deciding who this is
        pend = self.pending.setdefault(tid, {"since": t, "bodies": [], "faces": []})
        pend["last"] = t
        if sample is not None:
            pend["bodies"].append(sample[0])
            if sample[1] is not None:
                pend["faces"].append(sample[1])
        waited = t - pend["since"]
        if waited < CONFIRM_MIN_S or not pend["bodies"] or (len(pend["bodies"]) < CONFIRM_SAMPLES and waited < CONFIRM_MAX_S):
            return None
        del self.pending[tid]
        self._last_scores = []
        pid = self._match_returning(pend["bodies"], pend["faces"], t) if pend["bodies"] else None
        matched = pid is not None
        if pid is None:
            pid = self.next_pid
            self.next_pid += 1
            self.people[pid] = Person(pid, pend["since"], t)
        top = "; ".join(f"ID {q} {kind}={sc + (FACE_MATCH if kind == 'face' else REID_MATCH):.2f}"
                        for sc, kind, q in self._last_scores) or "nobody to compare"
        self.log.append({"time_s": f"{t:.1f}", "tracker_id": int(tid), "samples": len(pend["bodies"]),
                         "faces_seen": len(pend["faces"]),
                         "decision": f"returning ID {pid}" if matched else f"NEW ID {pid}",
                         "visible_now": " ".join(map(str, sorted(self.frame_pids))), "closest": top})
        print(f"[{t:6.1f}s] tracker #{tid} -> {self.log[-1]['decision']}   (closest: {top})")
        p = self.people[pid]
        p.gallery = (p.gallery + pend["bodies"])[-GALLERY_SIZE:]
        p.face_gallery = (p.face_gallery + pend["faces"])[-GALLERY_SIZE:]
        if pend["faces"]:
            self.new_faces.add(pid)
        p.last_gallery_t = t
        # Count the short wait while they were being identified, without double-counting time already counted
        p.stay_time += t - max(pend["since"], p.last_t if p.last_t is not None else pend["since"])
        p.last_t = t
        self.track_to_pid[tid] = pid
        return self._absorb_ghost(tid, pid, t)

    def _absorb_ghost(self, tid, pid, t):
        """A tracked person sometimes gets a short-lived ID before their face is visible (back turned,
        or dark against a window) and is then recognised properly a moment later. The short-lived ID
        is the same tracked person, so fold it into the real one instead of counting it separately.

        Only with --faces, and only if the left-behind ID never showed a face, never became a visitor,
        belongs to no other track and was never on screen together with the real one - otherwise it
        may be a different person who took over the track (e.g. sat down in the same spot)."""
        old = self.dropped.pop(tid, None)
        ghost = self.people.get(old)
        if ghost is None or old == pid or self.faces is None or not self.people[pid].face_gallery:
            return pid
        pair = (min(old, pid), max(old, pid))
        if ghost.face_gallery or ghost.visitor or pair in self.together or old in self.track_to_pid.values():
            return pid
        self._merge(self.people[pair[0]], self.people[pair[1]], t, "same tracked person, seen before their face was visible")
        return pair[0]

    def update(self, frame, result, t):
        self.shown = []
        self.remap = {}
        if result.boxes.id is None or result.keypoints is None:
            self._drop_pending(t)
            return []
        tids = result.boxes.id.int().cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        kxys = result.keypoints.xy.cpu().numpy()
        kconfs = result.keypoints.conf.cpu().numpy()
        # Anyone already visible in this frame can't also be the "new" track
        self.frame_pids = {self.track_to_pid[tid] for tid in tids if tid in self.track_to_pid}
        # Take appearance samples only for tracks that need one (new people, ~2/s for known people)
        inside = [self._in_zone(box) for box in boxes]
        # Tracking starts only inside the counting zone; someone who already has an ID keeps it if they step out
        self.shown = [i for i, tid in enumerate(tids) if inside[i] or tid in self.track_to_pid]
        need = [i for i in self.shown if self._needs_sample(tids[i], boxes[i], t)
                and (sharpness(frame, boxes[i]) >= BLUR_MIN or self._blurry_too_long(tids[i], t))]
        samples = {}
        if need:
            bodies = fingerprint(self.reid, frame, boxes[need], kxys[need], kconfs[need])
            faces = self.faces.embed(frame, boxes[need], kxys[need], kconfs[need]) if self.faces else [None] * len(need)
            samples = dict(zip(need, zip(bodies, faces)))
        out = []
        for i in self.shown:
            tid, box, kxy, kconf = tids[i], boxes[i], kxys[i], kconfs[i]
            sample = samples.get(i)
            known = tid in self.track_to_pid
            pid = self._resolve_pid(tid, sample, t)
            if pid is None:
                continue  # still confirming who this is
            self.frame_pids.add(pid)
            p = self.people[pid]
            if known and sample is not None:
                self.last_check[tid] = t
                body, face = sample
                if self._score(p, [body], [face] if face is not None else [])[0] >= 0:
                    self.mismatches[tid] = 0
                    if t - p.last_gallery_t >= 1.0:
                        # Keep a rolling set of samples (different poses/lighting) - only ones that match
                        p.gallery = (p.gallery + [body])[-GALLERY_SIZE:]
                        if face is not None:
                            p.face_gallery = (p.face_gallery + [face])[-GALLERY_SIZE:]
                            self.new_faces.add(pid)
                        p.last_gallery_t = t
                else:
                    self.mismatches[tid] = self.mismatches.get(tid, 0) + 1
                    if self.mismatches[tid] >= SWITCH_CHECKS:
                        # The tracker kept the same number but it's now someone else (e.g. one person
                        # left and another sat down in the same spot). Forget the link and decide afresh.
                        print(f"[{t:6.1f}s] tracker #{tid} no longer looks like ID {pid} - re-checking who this is")
                        del self.track_to_pid[tid]
                        self.dropped[tid] = pid
                        self.mismatches[tid] = 0
                        self.frame_pids.discard(pid)
                        self._resolve_pid(tid, sample, t)
                        continue
            dt = 0.0 if p.last_t is None or t - p.last_t > 1.0 else t - p.last_t
            p.last_seen = p.last_t = t
            p.stay_time += dt

            box_h = max(box[3] - box[1], 1.0)
            hips_ok = kconf[L_HIP] > 0.3 and kconf[R_HIP] > 0.3
            hip = (kxy[L_HIP] + kxy[R_HIP]) / 2 if hips_ok else np.array([(box[0] + box[2]) / 2, box[3]])

            # Stopped: hip speed over the last ~1 s below threshold, sustained STOP_MIN_S
            p.hip_track = [h for h in p.hip_track if t - h[0] <= 1.0] + [(t, *hip, box_h)]
            t0, x0, y0, _ = p.hip_track[0]
            span = t - t0
            speed = np.hypot(hip[0] - x0, hip[1] - y0) / span / box_h if span > 0.3 else 0.0
            if speed < STOP_SPEED:
                p.still_since = p.still_since or t
            else:
                p.still_since = None
            stopped = p.still_since is not None and t - p.still_since >= STOP_MIN_S

            # Looking, debounced
            raw = is_looking(kxy, kconf)
            if raw != p.pending_look:
                p.pending_look, p.pending_since = raw, t
            if raw != p.looking and t - p.pending_since >= LOOK_DEBOUNCE_S:
                p.looking = raw

            if inside[i] and stopped and p.looking:
                p.look_time += dt
            out.append((p, box, kxy, kconf))
        self._drop_pending(t)
        visible = sorted(q.pid for q, *_ in out)
        self.together |= {(a, b) for i, a in enumerate(visible) for b in visible[i + 1:]}
        self._merge_by_face(t)

        def current(pid):  # follow any merges made during this frame
            while pid in self.remap:
                pid = self.remap[pid]
            return pid

        return [(self.people[current(q.pid)], *rest) for q, *rest in out]

    def _merge_by_face(self, t):
        """Late fix-up: a face seen now may prove two IDs are the same person (e.g. first seen blurry
        or side-on with no face, so they got a new ID). Merge them if never on screen together."""
        for pid in sorted(self.new_faces):
            p = self.people.get(pid)
            if p is None or not p.face_gallery:
                continue
            for q in list(self.people.values()):
                pair = (min(pid, q.pid), max(pid, q.pid))
                if q.pid == pid or not q.face_gallery or pair in self.together:
                    continue
                if similarity(p.face_gallery, q.face_gallery) >= FACE_MERGE:
                    keep, drop = self.people[pair[0]], self.people[pair[1]]
                    self._merge(keep, drop, t, "same face")
                    break
        self.new_faces.clear()

    def _merge(self, keep, drop, t, reason):
        print(f"[{t:6.1f}s] ID {drop.pid} merged into ID {keep.pid} ({reason})")
        self.log.append({"time_s": f"{t:.1f}", "tracker_id": "", "samples": "", "faces_seen": "",
                         "decision": f"MERGED ID {drop.pid} into ID {keep.pid}", "visible_now": "", "closest": reason})
        self.remap[drop.pid] = keep.pid
        if drop.pid in self.frame_pids:
            self.frame_pids = (self.frame_pids - {drop.pid}) | {keep.pid}
        keep.first_seen = min(keep.first_seen, drop.first_seen)
        keep.last_seen = max(keep.last_seen, drop.last_seen)
        keep.look_time += drop.look_time
        keep.stay_time += drop.stay_time
        keep.gallery = (keep.gallery + drop.gallery)[-GALLERY_SIZE:]
        keep.face_gallery = (keep.face_gallery + drop.face_gallery)[-GALLERY_SIZE:]
        for tid, pid in self.track_to_pid.items():
            if pid == drop.pid:
                self.track_to_pid[tid] = keep.pid
        self.together = {tuple(sorted(keep.pid if x == drop.pid else x for x in pair)) for pair in self.together}
        del self.people[drop.pid]

    def _drop_pending(self, t):
        # Tracks that vanished before being confirmed were flicker/false detections
        for tid, pend in list(self.pending.items()):
            if t - pend["last"] > 1.0:
                del self.pending[tid]


HEADER_H = 38  # height of the black strip holding the counts


def draw(frame, counter, tracked):
    x1, y1, x2, y2 = counter.zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
    cv2.putText(frame, "Counting zone", (x1 + 6, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
    h, w = frame.shape[:2]
    for p, box, kxy, kconf in tracked:
        colour = (0, 200, 0) if p.visitor else (200, 200, 200)
        label = f"ID {p.pid}  {p.stay_time:.1f}s"
        # Label sits above the person's head on a dark background, kept inside the frame
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = int(np.clip((box[0] + box[2]) / 2 - tw / 2, 0, w - tw - 8))
        y = int(np.clip(box[1] - 10, HEADER_H + th + 10, h - 4))
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y + 4), (0, 0, 0), -1)
        cv2.putText(frame, label, (x + 4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    total = len(counter.people)
    visitors = sum(p.visitor for p in counter.people.values())
    cv2.rectangle(frame, (0, 0), (w, HEADER_H), (0, 0, 0), -1)
    cv2.putText(frame, f"Unique people: {total}   Visitors: {visitors}", (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def write_csv(path, counter):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "first_seen_s", "last_seen_s", "time_on_camera_s", "visitor", "look_time_s"])
        for p in sorted(counter.people.values(), key=lambda p: p.pid):
            w.writerow([p.pid, f"{p.first_seen:.1f}", f"{p.last_seen:.1f}",
                        f"{p.stay_time:.1f}", "yes" if p.visitor else "no", f"{p.look_time:.1f}"])
    ppl = counter.people.values()
    print(f"Saved {path}: {len(ppl)} unique people, {sum(p.visitor for p in ppl)} visitors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="camera number (0, 1, ...), a video file, or a network camera address (rtsp://... or http://...)")
    ap.add_argument("--list-cameras", action="store_true", help="show which camera numbers work on this computer, then exit")
    ap.add_argument("--duration", type=float, default=0,
                    help="stop by itself and save after this many minutes (0 = run until you press q or Ctrl+C)")
    ap.add_argument("--model", default="yolo26n-pose.pt", help="e.g. yolo26n-pose_ncnn_model on the Pi")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--skip", type=int, default=1, help="process every Nth frame (2 on a Pi 4)")
    ap.add_argument("--reid", default="osnet_x0_25_msmt17.pt", help="OSNet re-ID weights")
    ap.add_argument("--faces", action="store_true",
                    help="also match by face (more accurate up close; check privacy/consent rules first)")
    ap.add_argument("--no-enhance", action="store_true",
                    help="turn off the automatic lighting correction used for analysis")
    ap.add_argument("--zone", default=",".join(map(str, ZONE)),
                    help="counting zone as left,top,right,bottom fractions of the picture, e.g. 0.2,0.1,0.8,0.95")
    ap.add_argument("--csv", default="visitors.csv")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--record", default="", help="also save the raw camera video to this file, e.g. session.mp4")
    ap.add_argument("--save-screen", default="",
                    help="save what the window shows (skeletons, IDs, zone, counts) to this file, e.g. screen.mp4")
    args = ap.parse_args()
    if args.list_cameras:
        print("Cameras this computer can open:")
        list_cameras()
        return

    src = int(args.source) if args.source.isdigit() else args.source
    is_file = not is_live(src)
    cap = cv2.VideoCapture(src) if is_file else LiveCamera(src)
    if not (cap.isOpened() if is_file else cap.opened):
        kind = "video file" if is_file else "camera"
        print(f"Could not open the {kind}. Check the number/address, that the device is on the same network, "
              "and that no other app is using it. Run with --list-cameras to see working camera numbers.")
        return
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else cap.fps
    seen = 0
    model = YOLO(args.model, task="pose")
    reid = ReID(args.reid)
    faces = FaceID() if args.faces else None
    counter, frame_idx, start, tick = None, 0, time.time(), time.time()
    writer, raw_frames, screen_writer, screen_frames = None, 0, None, 0
    zone = {"rect": tuple(float(v) for v in args.zone.split(",")), "drag": None, "size": (1, 1)}
    if args.show:
        # Resizable window; the video stretches to fill it but keeps its shape
        cv2.namedWindow("Booth Counter", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        def on_mouse(event, x, y, flags, _):
            """Drag with the left mouse button to draw a new counting zone."""
            if event == cv2.EVENT_LBUTTONDOWN:
                zone["drag"] = (x, y, x, y)
            elif event == cv2.EVENT_MOUSEMOVE and zone["drag"]:
                zone["drag"] = (*zone["drag"][:2], x, y)
            elif event == cv2.EVENT_LBUTTONUP and zone["drag"]:
                ax, ay, bx, by = zone["drag"]
                zone["drag"] = None
                w, h = zone["size"]
                if abs(bx - ax) > 0.1 * w and abs(by - ay) > 0.1 * h:  # ignore accidental clicks
                    rect = (min(ax, bx) / w, min(ay, by) / h, max(ax, bx) / w, max(ay, by) / h)
                    zone["rect"] = tuple(round(float(np.clip(v, 0, 1)), 2) for v in rect)
                    print("New counting zone. To start with it next time add:  --zone "
                          + ",".join(map(str, zone["rect"])))

        cv2.setMouseCallback("Booth Counter", on_mouse)

    try:
        while True:
            if args.duration and time.time() - start >= args.duration * 60:
                print(f"Reached {args.duration:g} minutes - stopping")
                break
            if is_file:
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                seen, frame = cap.read(seen)
                if frame is None:  # no new picture yet (slow or reconnecting camera) - keep the window responsive
                    if args.show and cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
            if args.record:
                if writer is None:
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (frame.shape[1], frame.shape[0]))
                # Live cameras: repeat frames as needed so the saved video plays at real speed
                target = raw_frames + 1 if is_file else max(int((time.time() - start) * fps), raw_frames + 1)
                while raw_frames < target:
                    writer.write(frame)
                    raw_frames += 1
            frame_idx += 1
            if frame_idx % args.skip:
                continue
            # Video files use video time so tests match reality; cameras use wall-clock time
            t = frame_idx / fps if is_file else time.time() - start
            counter = counter or Counter(frame.shape, reid, faces)
            zone["size"] = (frame.shape[1], frame.shape[0])
            counter.set_zone(zone["rect"])
            seen_frame = frame if args.no_enhance else enhance(frame)   # what the models look at
            result = model.track(seen_frame, persist=True, tracker="bytetrack.yaml",
                                 imgsz=args.imgsz, classes=[0], conf=DET_CONF, verbose=False)[0]
            tracked = counter.update(seen_frame, result, t)
            if args.show or args.save_screen:
                # Skeletons are only drawn for people being tracked (inside the zone, or already given an ID)
                vis = result[counter.shown].plot(boxes=False, labels=False, img=frame.copy()) if counter.shown else frame.copy()
                vis = draw(vis, counter, tracked)
                if zone["drag"]:
                    ax, ay, bx, by = zone["drag"]
                    cv2.rectangle(vis, (ax, ay), (bx, by), (255, 255, 255), 1)
                now = time.time()
                cv2.putText(vis, f"{1 / max(now - tick, 1e-6):.0f} FPS", (vis.shape[1] - 85, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
                tick = now
            if args.save_screen:
                if screen_writer is None:
                    screen_writer = cv2.VideoWriter(args.save_screen, cv2.VideoWriter_fourcc(*"mp4v"),
                                                    fps if not is_file else fps / args.skip,
                                                    (vis.shape[1], vis.shape[0]))
                # A live camera rarely delivers exactly its advertised frame rate, so repeat frames as
                # needed to keep the saved video playing at real speed
                target = screen_frames + 1 if is_file else max(int(t * fps), screen_frames + 1)
                while screen_frames < target:
                    screen_writer.write(vis)
                    screen_frames += 1
            if args.show:
                cv2.imshow("Booth Counter", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved video {args.record}")
        if screen_writer is not None:
            screen_writer.release()
            print(f"Saved screen recording {args.save_screen}")
        cv2.destroyAllWindows()
        if counter:
            write_csv(args.csv, counter)
            if counter.log:
                with open("reid_log.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(counter.log[0]))
                    w.writeheader()
                    w.writerows(counter.log)
                print("Saved reid_log.csv")


if __name__ == "__main__":
    main()
