from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
import cv2
import numpy as np
import math
from ultralytics import YOLO
import torch #

DEVICE = 0 if torch.cuda.is_available() else "cpu"

MODEL = "yolo26l.pt"
BALL_CLASS = 32         # predefined class identifier value for BALL in YOLO
CONF = 0.2             
IMGSZ_CAP = 1920        # upper limit on inference size; raise for 4K if speed allows: 1080p
STRIDE = 32             # imgsz should be a multiple of this

# gate plausible position of ball during inerpolation: needs tuning
GATE_BASE = 40          # base gate (px)
GATE_GROWTH = 20        # growwth per frame gate (px)
GATE_MAX = 300          # max gate (px)

MAX_FILL_SEC = 0.5      # longest gap to interpolate (s) 
MAX_LOST_SEC = 1.5      # longest time to keep tracking before giving up and dropping the old track 

VEL_DECAY = 0.9         # velocity kept per lost frame (1.0 = old constant-velocity behavior)

REF_FPS = 30            # frame rate the GATE_* pixel values were tuned for

TEST_MODE = True
TEST_GATE = True     
TEST_DIR = Path("test_videos")
TEST_VIDEO = "ball_tracking.mp4"
TEST_OUT = "ball_tracking_out.mp4"

GATE_COLOR = (255, 255, 0)   # BGR


@dataclass
class BallRecord:
    frame: int
    t: float          # seconds, includes chunk start time
    x: float          # NaN when missing
    y: float
    conf: float
    status: str       # "detected" | "interpolated" | "missing"

class Candidate(NamedTuple):
    x: float
    y: float
    conf: float

class Gate(NamedTuple):
    cx: float         # gate centre = predicted ball position (px)
    cy: float
    radius: float     # px

FrameCandidates = list[Candidate]          # all candidates in one frame
VideoCandidates = list[FrameCandidates]    # one entry per frame
Track = list[Candidate | None]             # chosen candidate per frame, None if none
Gates = list[Gate | None]                  # gate per frame, None when no active track
BallRecords = list[BallRecord]             # list of one record per frame, with interpolated positions


def video_info(video_path: Path | str) -> tuple[float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, w, h

def pick_imgsz(w: int, h: int) -> int:
    # long side, rounded up to a multiple of STRIDE, capped
    return min(math.ceil(max(w, h) / STRIDE) * STRIDE, IMGSZ_CAP)

# detect_candidates: detect ball canditates per frame using YOLO with
#                    given model, for given video file.
def detect_candidates(model: YOLO, video_path: Path | str, imgsz: int) -> VideoCandidates:
    # list of lists of (x, y, conf) candidates per frame
    cands: VideoCandidates = [] 

    # r is result per frame    
    for r in model.predict(
            source = str(video_path),   # path to video file
            classes=[BALL_CLASS],       # what classes to detect
            conf=CONF,                  # confidence threshold
            imgsz=imgsz,                # normalize to this size for YOLO model
            stream=True,                # per frame results, not batch results
            verbose=True,               # if to use logging
            save=False,
            device=DEVICE):

        xywh = r.boxes.xywh.cpu().numpy() # [center_x, center_y, width, height] per detection box
        conf = r.boxes.conf.cpu().numpy() # confidence per detection box

        # convert to list of (x, y, conf) per detection box
        cands.append([
            Candidate(float(b[0]), float(b[1]), float(c)) 
            for b, c in zip(xywh, conf)])
    return cands


def link_detections(cands: VideoCandidates, fps: float,
                    record_gates: bool = False) -> tuple[Track, Gates | None]:
    # Greedy gated linking: pick one candidate per frame (or None)
    # Upgrade to Kalman + RTS smoother or Viterbi later
    max_lost = round(MAX_LOST_SEC * fps)
    scale = REF_FPS / fps

    track: Track = []
    gates: Gates = []
    last = None
    vel = np.zeros(2)
    lost = 0

    for frame_c in cands:
        best = None
        pred = None
        radius = 0.0

        # gate geometry is needed for the selection below, so always compute it
        if last is not None:
            # distance travelled in "velocity units": 1 on the first frame, then a
            # geometric series that levels off at 1 / (1 - VEL_DECAY) = 10
            if VEL_DECAY < 1.0:
                steps = (1 - VEL_DECAY ** (lost + 1)) / (1 - VEL_DECAY)
            else:
                steps = lost + 1   
            pred = last + vel * steps
            radius = min((GATE_BASE + GATE_GROWTH * lost) * scale, GATE_MAX)

        if frame_c:
            if pred is None:
                best = max(frame_c, key=lambda c: c.conf)
            else:
                in_gate = [c for c in frame_c
                           if np.hypot(c.x - pred[0], c.y - pred[1]) <= radius]
                if in_gate:
                    best = max(in_gate, key=lambda c: c.conf)

        if best is not None:
            pos = np.array([best.x, best.y])
            if last is not None:
                vel = (pos - last) / (lost + 1)
            last = pos
            lost = 0
        elif last is not None:
            lost += 1
            if lost > max_lost:
                last = None
                vel = np.zeros(2)
                lost = 0

        track.append(best)

        # only store it when asked (debug video)
        if record_gates:
            gates.append(None if pred is None
                         else Gate(float(pred[0]), float(pred[1]), float(radius)))

    return track, (gates if record_gates else None)


def fill_gaps(track: Track, fps: float, start_time: float) -> list[BallRecord]:
    max_fill = round(MAX_FILL_SEC * fps)   # seconds -> frames

    # Initialize every frame as missing
    records: BallRecords = [
        BallRecord(i, start_time + i / fps, np.nan, np.nan, 0.0, "missing")
        for i in range(len(track))]
    
    # then overwrite the frames that have detections with the detected position
    detected = []
    for i, d in enumerate(track):
        if d is not None:
            records[i] = BallRecord(i, records[i].t, d.x, d.y, d.conf, "detected")
            detected.append(i)
    
    # then interpolate the frames that are missing but are between two detected frames.
    # Short gaps between two detections are linearly interpolated;
    # longer gaps stay "missing".
    for a, b in zip(detected, detected[1:]):
        gap = b - a - 1                    # number of missing frames between a and b
        if 0 < gap <= max_fill:
            for k in range(1, gap + 1):
                f = k / (gap + 1)          # fraction of the way from a to b
                records[a + k] = BallRecord(
                    a + k, 
                    records[a + k].t,
                    records[a].x + f * (records[b].x - records[a].x),
                    records[a].y + f * (records[b].y - records[a].y),
                    0.0, "interpolated")
    return records


def run_pipeline(video_path, start_time=0.0, model=None) -> tuple[BallRecords, Gates | None]:
    print(f"device: {DEVICE}")

    fps, w, h = video_info(video_path)
    imgsz = pick_imgsz(w, h)
    print(f"video: {w}x{h} @ {fps:.2f} fps, imgsz: {imgsz}")
    model = model or YOLO(MODEL)

    candidates = detect_candidates(model, video_path, imgsz)
    track, gates = link_detections(candidates, fps, record_gates=TEST_GATE)
    records = fill_gaps(track, fps, start_time)
    return records, gates


def track_ball(video_path, start_time=0.0, model=None) -> BallRecords:
    # public interface of the block: records only
    records, _ = run_pipeline(video_path, start_time, model)
    return records


def render_debug(video_path, records, out_path, trail = 30, gates: Gates | None = None):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    colors = {"detected": (0, 255, 0),       # green
              "interpolated": (0, 255, 255), # yellow
              "missing": (0, 0, 255)}        # red
    
    for record in records:
        ok, frame = cap.read()

        # if the video is shorter than the records, stop early
        if not ok: 
            break

        # gate circle around the predicted ball position (test only)
        if gates is not None:
            g = gates[record.frame]
            if g is not None:
                centre = (int(g.cx), int(g.cy))
                cv2.circle(frame, centre, int(g.radius), GATE_COLOR, 1)
                cv2.drawMarker(frame, centre, GATE_COLOR, cv2.MARKER_CROSS, 10, 1)

        # draw the trail of the ball for the last `trail` frames
        for p in records[max(0, record.frame - trail):record.frame]:
            if p.status != "missing":
                cv2.circle(frame, (int(p.x), int(p.y)), 2, colors[p.status], -1)

        # draw the current ball position and status 
        if record.status == "missing":
            cv2.putText(frame, "ball: lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[record.status], 2)
        else:
            cv2.circle(frame, (int(record.x), int(record.y)), 12, colors[record.status], 2)
            cv2.putText(frame, f"{record.status} {record.conf:.2f}", (int(record.x) + 15, int(record.y) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[record.status], 2)
        out.write(frame)
    cap.release()
    out.release()

if __name__ == "__main__":
    if TEST_MODE:
        video = TEST_DIR / TEST_VIDEO
        video_out = TEST_DIR / TEST_OUT
        records, gates = run_pipeline(video)
        render_debug(video, records, video_out, gates=gates)
        found = sum(r.status != "missing" for r in records)
        print(f"{found}/{len(records)} frames with a ball position")
