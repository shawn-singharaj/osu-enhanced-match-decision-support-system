"""Associate ball detections by center motion without requiring box overlap."""

import numpy as np


def overlap(first, second):
    intersection = np.maximum(0, np.minimum(first[2:4], second[2:4]) - np.maximum(first[:2], second[:2])).prod()
    areas = [np.maximum(0, box[2:4] - box[:2]).prod() for box in (first, second)]
    return float(intersection / max(areas[0] + areas[1] - intersection, 1e-6))


class BallTracker:
    """Return every current detection, assigning IDs using short motion history.

    Input rows: x1, y1, x2, y2, confidence, class.
    Output rows: x1, y1, x2, y2, track_id, confidence, class.
    IDs are independent of player IDs. No locations are invented in missing frames.
    """

    def __init__(self, max_missed=8):
        self.max_missed = max_missed
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, frame_index):
        detections = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        # End-to-end detectors can return duplicate ball candidates at low thresholds.
        kept = []
        for index in np.argsort(-detections[:, 4]):
            box = detections[index]
            if not any(overlap(box, other) > 0.5 for other in kept):
                kept.append(box)
        detections = np.asarray(kept, dtype=np.float32).reshape(-1, 6)
        self.tracks = {key: value for key, value in self.tracks.items() if frame_index - value['frame'] <= self.max_missed + 1}
        centers = (detections[:, :2] + detections[:, 2:4]) / 2
        candidates = []
        for track_id, track in self.tracks.items():
            gap = frame_index - track['frame']
            predicted = track['center'] + track['velocity'] * gap
            for index, center in enumerate(centers):
                diameter = max(detections[index, 2:4] - detections[index, :2])
                gate = max(40.0, 6.0 * diameter) * (1 + 0.25 * max(0, gap - 1))
                distance = float(np.linalg.norm(center - predicted))
                if distance <= gate:
                    candidates.append((distance / gate, track_id, index))
        assigned = {}
        used_tracks = set()
        for _, track_id, index in sorted(candidates):
            if index not in assigned and track_id not in used_tracks:
                assigned[index] = track_id
                used_tracks.add(track_id)
        output = []
        for index, detection in enumerate(detections):
            track_id = assigned.get(index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                velocity = np.zeros(2, dtype=np.float32)
            else:
                previous = self.tracks[track_id]
                measured = (centers[index] - previous['center']) / (frame_index - previous['frame'])
                velocity = 0.5 * previous['velocity'] + 0.5 * measured
            self.tracks[track_id] = {'center': centers[index], 'velocity': velocity, 'frame': frame_index}
            output.append([*detection[:4], track_id, *detection[4:]])
        return np.asarray(output, dtype=np.float32).reshape(-1, 7)


class BallSelector:
    """Select one plausible match ball; this is a heuristic, not a classifier.

    A starting position is optional and uses original-video pixel coordinates.
    Player-foot overlap reduces ranking rather than banning all nearby balls.
    Unselected stationary marks stay hidden without rejecting a stopped ball.
    """

    def __init__(self, start=None):
        self.position = None if start is None else np.asarray(start, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.selected_id = None
        self.missing = 0
        self.diameter = 10.0
        self.seeded = start is not None

    def update(self, detections, people=(), camera_transform=None):
        rows = np.asarray(detections, dtype=np.float32).reshape(-1, 7)
        people = np.asarray(people, dtype=np.float32).reshape(-1, 4)
        previous = None if self.position is None else self.position.copy()
        if previous is not None and camera_transform is not None:
            previous = camera_transform[:, :2] @ previous + camera_transform[:, 2]
            self.velocity = camera_transform[:, :2] @ self.velocity
        predicted = None if previous is None else previous + self.velocity
        ranked = []
        for row in rows:
            center = (row[:2] + row[2:4]) / 2
            size = np.maximum(row[2:4] - row[:2], 1)
            aspect = float(max(size) / min(size))
            if aspect > 3.0:
                continue
            near_foot = any(
                p[0] <= center[0] <= p[2]
                and p[3] - 0.28 * (p[3] - p[1]) <= center[1] <= p[3] + 2
                for p in people
            )
            score = float(row[5]) - 0.30 * near_foot - 0.05 * max(0, aspect - 1.5)
            if predicted is None:
                # Weak candidates can sustain a trajectory, but cannot start one.
                if row[5] < 0.15:
                    continue
            else:
                gate = max(45.0, 6 * self.diameter) * (1 + 0.20 * min(self.missing, 8))
                distance = float(np.linalg.norm(center - predicted))
                if distance > gate:
                    continue
                score += 0.15 * (1 - distance / gate)
                score += 0.08 * (int(row[4]) == self.selected_id)
                if self.seeded and self.selected_id is None:
                    score += 0.30 * (1 - distance / gate)
            if score >= 0.12:
                ranked.append((score, row, center, float(max(size))))
        if not ranked:
            self.position = predicted
            self.missing += 1
            # Stop expanding the search around a stale trajectory. A manually
            # seeded run waits for a plausible nearby detection rather than
            # switching to a distant penalty spot.
            if self.missing > 15 and not self.seeded:
                self.position = None
                self.velocity[:] = 0
                self.selected_id = None
            return np.empty((0, 7), dtype=np.float32)
        _, row, center, diameter = max(ranked, key=lambda entry: entry[0])
        if previous is not None and self.selected_id is not None:
            self.velocity = 0.5 * self.velocity + 0.5 * (center - previous)
        self.position = center.copy()
        self.selected_id = int(row[4])
        self.diameter = diameter
        self.missing = 0
        return row.reshape(1, 7).copy()
