#!/usr/bin/env python3
"""Track people/balls and estimate the visible grass region in a soccer video.

Run: python track_soccer.py OSUshortclip.mp4
Requires: pip install -U ultralytics lap
The grass outline is an image-space estimate, not a calibrated pitch model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from ultralytics import YOLO

from ball_tracker import BallSelector, BallTracker


def field_region(frame):
    """Find the largest grass region at reduced resolution; return a full-size mask."""
    height, width = frame.shape[:2]
    small = cv2.resize(frame, (640, max(1, round(height * 640 / width))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # Includes the yellow-green turf in the supplied OSU footage.
    grass = cv2.inRange(hsv, (20, 35, 30), (95, 255, 255))
    # Remove narrow connections before filling small gaps. Aggressive closing
    # can join the pitch to landscaping behind the stands when zoom changes.
    grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((height, width), dtype=np.uint8)
    if not contours:
        return mask, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < grass.size * 0.10:
        return mask, None
    hull = cv2.convexHull(largest).astype(np.float32)
    hull[:, :, 0] *= width / small.shape[1]
    hull[:, :, 1] *= height / small.shape[0]
    hull = cv2.approxPolyDP(hull, 0.006 * cv2.arcLength(hull, True), True)
    hull = np.rint(hull).astype(np.int32)
    cv2.fillPoly(mask, [hull], 255)
    return mask, hull


class FieldTracker:
    """Smooth grass estimates after compensating for camera pan and zoom.

    Only blend history when feature matches support a reliable camera transform.
    Fresh segmentation always supplies newly exposed parts of the image.
    """

    def __init__(self):
        self.gray = None
        self.probability = None
        self.last_transform = None

    def camera_motion(self, gray):
        if self.gray is None or self.gray.shape != gray.shape or self.probability is None:
            return None
        region = (self.probability > 0.5).astype(np.uint8) * 255
        region = cv2.erode(region, np.ones((9, 9), np.uint8))
        points = cv2.goodFeaturesToTrack(self.gray, maxCorners=400, qualityLevel=0.01, minDistance=8, mask=region)
        if points is None or len(points) < 15:
            return None
        following, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, points, None)
        if following is None or status is None:
            return None
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, following, None)
        if backward is None or back_status is None:
            return None
        valid = (status.ravel() == 1) & (back_status.ravel() == 1)
        valid &= np.linalg.norm(points - backward, axis=2).ravel() < 1.5
        if valid.sum() < 15:
            return None
        source = points[valid].reshape(-1, 2)
        target = following[valid].reshape(-1, 2)
        transform, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if transform is None or inliers is None or not np.isfinite(transform).all():
            return None
        accepted = inliers.ravel().astype(bool)
        if accepted.sum() < 12 or accepted.mean() < 0.6:
            return None
        # Local player motion must not determine the camera transform.
        spread = np.ptp(source[accepted], axis=0)
        if spread[0] < gray.shape[1] * 0.20 or spread[1] < gray.shape[0] * 0.12:
            return None
        scale = np.hypot(transform[0, 0], transform[1, 0])
        if not 0.85 < scale < 1.18 or np.linalg.norm(transform[:, 2]) > gray.shape[1] * 0.25:
            return None
        warped = cv2.warpAffine(self.gray, transform, (gray.shape[1], gray.shape[0]))
        overlap = cv2.warpAffine(region, transform, (gray.shape[1], gray.shape[0])) > 0
        if not overlap.any() or np.median(cv2.absdiff(warped, gray)[overlap]) > 30:
            return None
        return transform

    def update(self, frame):
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (640, max(1, round(height * 640 / width))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        fresh, polygon = field_region(small)
        transform = self.camera_motion(gray)
        self.last_transform = transform
        probability = fresh.astype(np.float32) / 255.0
        if polygon is not None and transform is not None:
            size = (small.shape[1], small.shape[0])
            previous = cv2.warpAffine(self.probability, transform, size)
            supported = cv2.warpAffine(np.ones_like(self.probability), transform, size) > 0.99
            # A large segmentation change signals a cut or unreliable grass estimate.
            difference = np.mean((previous > 0.5) != (fresh > 0))
            if difference < 0.18:
                probability[supported] = 0.65 * previous[supported] + 0.35 * probability[supported]
        # Drop history on failed segmentation instead of retaining a stale outline.
        self.gray = gray
        self.probability = probability if polygon is not None else None
        binary = (probability > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros((height, width), dtype=np.uint8)
        if polygon is None or not contours:
            return mask, None
        hull = cv2.convexHull(max(contours, key=cv2.contourArea)).astype(np.float32)
        hull = cv2.approxPolyDP(hull, 0.003 * cv2.arcLength(hull, True), True)
        hull[:, :, 0] *= width / small.shape[1]
        hull[:, :, 1] *= height / small.shape[0]
        hull = np.rint(hull).astype(np.int32)
        hull[:, :, 0] = np.clip(hull[:, :, 0], 0, width - 1)
        hull[:, :, 1] = np.clip(hull[:, :, 1], 0, height - 1)
        cv2.fillPoly(mask, [hull], 255)
        return mask, hull


def draw_field(frame, mask, polygon):
    if polygon is None:
        return
    tint = np.zeros_like(frame)
    tint[:] = (40, 180, 40)
    blended = cv2.addWeighted(frame, 0.92, tint, 0.08, 0)
    frame[mask > 0] = blended[mask > 0]
    height, width = frame.shape[:2]
    margin = max(4, round(width / 160))
    points = polygon.reshape(-1, 2)
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        # Image borders are clipping edges, not observed field boundaries.
        on_border = (
            (start[0] <= margin and end[0] <= margin)
            or (start[0] >= width - margin and end[0] >= width - margin)
            or (start[1] <= margin and end[1] <= margin)
            or (start[1] >= height - margin and end[1] >= height - margin)
        )
        if not on_border:
            cv2.line(frame, tuple(start), tuple(end), (70, 230, 70), 2)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", type=Path, default=Path("OSUshortclip.mp4"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/soccer"))
    parser.add_argument("--model", default="yolo26s.pt", help="YOLO weights; downloaded on first use")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference size; larger helps small balls")
    parser.add_argument("--player-conf", type=float, default=0.25)
    parser.add_argument("--ball-conf", type=float, default=0.05)
    parser.add_argument("--ball-start", nargs=2, type=float, metavar=("X", "Y"), help="Initial ball center in original-video pixels")
    parser.add_argument("--all-ball-candidates", action="store_true", help="Debug: display every ball candidate instead of selecting one")
    parser.add_argument("--device", default=None, help="Examples: cpu, mps (Apple GPU), 0 (CUDA)")
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes the entire clip")
    parser.add_argument("--show", action="store_true", help="Preview while processing; Q stops")
    parser.add_argument("--include-off-field", action="store_true", help="Also include people outside the grass estimate")
    args = parser.parse_args()
    if args.imgsz <= 0 or args.max_frames < 0:
        parser.error("--imgsz must be positive and --max-frames cannot be negative")
    if not all(0 < value <= 1 for value in (args.player_conf, args.ball_conf)):
        parser.error("Confidence thresholds must be greater than 0 and at most 1")
    return args


def main():
    args = arguments()
    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")
    output = args.output_dir.resolve()
    video_path = output / f"{args.video.stem}_tracked.mp4"
    if args.video.resolve() == video_path:
        raise SystemExit("Output would overwrite the input; choose another --output-dir.")
    # Existing results are kept intact; choose another directory to rerun.
    paths = (video_path, output / "tracks.csv", output / "field.csv")
    if any(path.exists() for path in paths):
        raise SystemExit(f"Results already exist in {output}. Choose another --output-dir.")
    model = YOLO(args.model)
    names = {int(key): str(value).lower() for key, value in model.names.items()}
    person_ids = [key for key, value in names.items() if value in {"person", "player", "goalkeeper", "referee"}]
    ball_ids = [key for key, value in names.items() if value in {"sports ball", "ball", "soccer ball", "football"}]
    if not person_ids or not ball_ids:
        raise SystemExit(f"Model must include person/player and ball classes. Found: {names}")
    raw_balls = np.empty((0, 6), dtype=np.float32)
    raw_people = np.empty((0, 4), dtype=np.float32)

    def remember_raw_balls(predictor):
        # Registered BEFORE Ultralytics' tracking callback: preserve detections
        # that ByteTrack would otherwise hide while waiting for confirmation.
        nonlocal raw_balls, raw_people
        boxes = predictor.results[0].boxes
        if boxes is None:
            raw_balls = np.empty((0, 6), dtype=np.float32)
            raw_people = np.empty((0, 4), dtype=np.float32)
        else:
            data = boxes.cpu().numpy()
            raw_balls = data.data[np.isin(data.cls, ball_ids)].copy()
            raw_people = data.xyxy[np.isin(data.cls, person_ids) & (data.conf >= args.player_conf)].copy()

    model.add_callback("on_predict_postprocess_end", remember_raw_balls)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    writer = None
    processed = 0
    try:
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise RuntimeError("Video has invalid dimensions")
        output.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Cannot create MP4 output")
        histories = {}
        field_tracker = FieldTracker()
        ball_tracker = BallTracker()
        ball_selector = BallSelector(args.ball_start)
        # Lower tracker thresholds allow small, low-confidence balls to start tracks.
        low = min(args.player_conf, args.ball_conf, 0.10)
        with TemporaryDirectory(prefix="soccer_tracker_") as temp, paths[1].open("w", newline="") as track_file, paths[2].open("w", newline="") as field_file:
            tracker_path = Path(temp) / "bytetrack.yaml"
            tracker_path.write_text(
                f"tracker_type: bytetrack\ntrack_high_thresh: {low}\n"
                f"track_low_thresh: {low / 2}\nnew_track_thresh: {low}\n"
                "track_buffer: 30\nmatch_thresh: 0.8\nfuse_score: true\n"
            )
            tracks = csv.writer(track_file)
            tracks.writerow(["frame", "time_s", "kind", "track_id", "confidence", "x1", "y1", "x2", "y2", "anchor_x", "anchor_y"])
            fields = csv.writer(field_file)
            fields.writerow(["frame", "time_s", "visible_grass_polygon_xy"])
            print(f"Processing {args.video}: {width}x{height}, {fps:g} FPS, {total} frames", flush=True)
            while args.max_frames == 0 or processed < args.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                mask, polygon = field_tracker.update(frame)
                fields.writerow([processed, round(processed / fps, 4), json.dumps(polygon.reshape(-1, 2).tolist() if polygon is not None else [])])
                result = model.track(
                    frame, persist=True, tracker=str(tracker_path),
                    classes=person_ids + ball_ids, conf=low / 2,
                    imgsz=args.imgsz, device=args.device, verbose=False,
                )[0]
                draw_field(frame, mask, polygon)
                people_count, ball_count = 0, 0
                # Keep ByteTrack for people. Balls use raw detections plus a
                # center-motion tracker, so fast movement and weak scores do not
                # suppress visible detections through IoU/confirmation gates.
                person_rows = np.empty((0, 7), dtype=np.float32)
                if result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()
                    selected = np.isin(boxes.cls, person_ids)
                    ids = boxes.id if boxes.id is not None else np.full(len(boxes), -1)
                    person_rows = np.column_stack((boxes.xyxy, ids, boxes.conf, boxes.cls))[selected]
                ball_rows = ball_tracker.update(raw_balls[raw_balls[:, 4] >= args.ball_conf], processed)
                if not args.all_ball_candidates:
                    motion = field_tracker.last_transform
                    if motion is not None:
                        # FieldTracker estimates motion on its 640-pixel image.
                        small_height = max(1, round(height * 640 / width))
                        scale = np.diag([width / 640, height / small_height])
                        motion = np.column_stack((scale @ motion[:, :2] @ np.linalg.inv(scale), scale @ motion[:, 2]))
                    ball_rows = ball_selector.update(ball_rows, raw_people, motion)
                detections = np.concatenate((person_rows, ball_rows))
                if len(detections):
                    coordinates = detections[:, :4]
                    classes = detections[:, 6].astype(int)
                    confidences = detections[:, 5]
                    ids = detections[:, 4].astype(int)
                    for box, class_id, confidence, track_id in zip(coordinates, classes, confidences, ids):
                        is_ball = class_id in ball_ids
                        if confidence < (args.ball_conf if is_ball else args.player_conf):
                            continue
                        x1, y1, x2, y2 = map(float, box)
                        anchor = (int((x1 + x2) / 2), int((y1 + y2) / 2) if is_ball else int(y2))
                        ax, ay = np.clip(anchor[0], 0, width - 1), np.clip(anchor[1], 0, height - 1)
                        if not is_ball and not args.include_off_field and polygon is not None and mask[ay, ax] == 0:
                            continue
                        kind = "ball" if is_ball else "player/person"
                        color = (0, 180, 255) if is_ball else (255, 200, 50)
                        people_count += int(not is_ball)
                        ball_count += int(is_ball)
                        key = (is_ball, int(track_id))
                        if track_id >= 0:
                            if key not in histories or processed - histories[key][1] > 1:
                                histories[key] = (deque(maxlen=30), processed)
                            trail = histories[key][0]
                            trail.append(anchor)
                            histories[key] = (trail, processed)
                            if len(trail) > 1:
                                cv2.polylines(frame, [np.array(trail, dtype=np.int32)], False, color, 2)
                        if is_ball:
                            cv2.circle(frame, anchor, max(9, int(max(x2 - x1, y2 - y1) / 2) + 3), color, 2)
                        else:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"{'Ball' if is_ball else 'Person'} #{track_id if track_id >= 0 else '?'} {confidence:.2f}"
                        cv2.putText(frame, label, (max(0, int(x1)), max(18, int(y1) - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        tracks.writerow([processed, round(processed / fps, 4), kind, int(track_id), round(float(confidence), 4), *[round(v, 2) for v in box], *anchor])
                histories = {key: value for key, value in histories.items() if processed - value[1] < 60}
                cv2.rectangle(frame, (8, 8), (800, 72), (25, 25, 25), -1)
                cv2.putText(frame, f"People: {people_count} | Balls: {ball_count} | Time: {processed / fps:.1f}s", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Green outline: estimated visible grass | coordinates in pixels", (20, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 240, 100), 1)
                writer.write(frame)
                processed += 1
                if processed == 1 or processed % 30 == 0:
                    print(f"Processed {processed}/{total} frames | people={people_count}, balls={ball_count}", flush=True)
                if args.show:
                    preview = cv2.resize(frame, (min(width, 1280), round(height * min(width, 1280) / width)))
                    cv2.imshow("Soccer tracking - Q to stop", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    if processed == 0:
        raise SystemExit("No video frames decoded; no usable output was produced.")
    print(f"Saved {processed} frames to {video_path}\nTracks: {paths[1]}\nField: {paths[2]}")


if __name__ == "__main__":
    main()
