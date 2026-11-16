from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
import cv2
import numpy as np
from ultralytics import YOLO

MODEL = "yolo26s.pt"
BALL_CLASS = 32         # predefined class identifier value for BALL in YOLO
CONF = 0.05             # false positives easier to filter out than false negatives 
IMGSZ = 1280

# gate plausible position of ball during inerpolation: needs tuning
GATE_BASE = 60          # base gate (px)
GATE_GROWTH = 40        # growwth per frame gate (px)
GATE_MAX = 500          # max gate (px)

MAX_FILL_SEC = 0.5      # longest gap to interpolate (s) 
MAX_LOST_SEC = 1.5      # longest time to keep tracking before giving up and dropping the old track 

REF_FPS = 30            # frame rate the GATE_* pixel values were tuned for

TEST_MODE = True
TEST_DIR = Path("test_videos")
TEST_VIDEO = "ball_tracking.mp4"
TEST_OUT = "ball_tracking_out.mp4"


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

FrameCandidates = list[Candidate]          # all candidates in one frame
VideoCandidates = list[FrameCandidates]    # one entry per frame
Track = list[Candidate | None]             # chosen candidate per frame, None if none
BallRecords = list[BallRecord]             # list of one record per frame, with interpolated positions

# detect_candidates: detect ball canditates per frame using YOLO with
#                    given model, for given video file.
def detect_candidates(model: YOLO, video_path: Path | str) -> VideoCandidates:
    # list of lists of (x, y, conf) candidates per frame
    cands: VideoCandidates = [] 

    # r is result per frame    
    for r in model.predict(
            source = str(video_path),   # path to video file
            classes=[BALL_CLASS],       # what classes to detect
            conf=CONF,                  # confidence threshold
            imgsz=IMGSZ,                # normalize to this size for YOLO model
            stream=True,                # per frame results, not batch results
            verbose=True,               # if to use logging
            save=False):
        
        xywh = r.boxes.xywh.cpu().numpy() # [center_x, center_y, width, height] per detection box
        conf = r.boxes.conf.cpu().numpy() # confidence per detection box

        # convert to list of (x, y, conf) per detection box
        cands.append([
            Candidate(float(b[0]), float(b[1]), float(c)) 
            for b, c in zip(xywh, conf)])
    return cands


def link_detections(cands: VideoCandidates, fps: float) -> Track:
    # Greedy gated linking: pick one candidate per frame (or None)
    # Upgrade to Kalman + RTS smoother or Viterbi later

    max_lost = round(MAX_LOST_SEC * fps)   # seconds -> frames
    scale = REF_FPS / fps                  # tune gate size to actual frame rate

    track: Track = []
    last = None                            # last accepted position (np array) or None
    vel = np.zeros(2)                      # px per frame. Initially zero
    lost = 0                               # consecutive frames without a pick

    # per frame, pick the best candidate (or None) and update the track
    for frame_c in cands:
        best = None

        if frame_c:
            if last is None:
                # no track yet: take the most confident candidate
                best = max(frame_c, key=lambda c: c.conf)
            else:
                # predict next position and gate candidates and pick the most confident one
                pred = last + vel * (lost + 1)
                gate = min((GATE_BASE + GATE_GROWTH * lost) * scale, GATE_MAX)
                in_gate = [c for c in frame_c
                           if np.hypot(c.x - pred[0], c.y - pred[1]) <= gate]
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
            if lost > max_lost:            # give up on the old track
                last = None
                vel = np.zeros(2)
                lost = 0

        track.append(best)

    return track


def fill_gaps(track: Track, fps: float, start_time: float) -> list[BallRecord]:
    max_fill = round(MAX_FILL_SEC * fps)   # seconds -> frames
    records: BallRecords = []

    # Initialize every frame as missing
    records = [BallRecord(i, start_time + i / fps, np.nan, np.nan, 0.0, "missing")
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


def track_ball(video_path, start_time=0.0, model=None):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    model = model or YOLO(MODEL)      # load once, reuse across chunks

    candidates = detect_candidates(model, video_path) # dections
    tracks = link_detections(candidates, fps)         # filtered detections
    records = fill_gaps(tracks, fps, start_time)      # interpolated detections with formatted result 

    return records


def render_debug(video_path, records, out_path, trail = 30):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    colors = {"detected": (0, 255, 0),       # green
              "interpolated": (0, 255, 255), # yellow
              "missing": (255, 0, 0)}        # red
    
    for record in records:
        ok, frame = cap.read()

        # if the video is shorter than the records, stop early
        if not ok: 
            break

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
        records = track_ball(video)
        render_debug(video, records, video_out)
        found = sum(r.status != "missing" for r in records)
        print(f"{found}/{len(records)} frames with a ball position")
