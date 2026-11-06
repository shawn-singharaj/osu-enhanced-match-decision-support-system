import sys
import time
import cv2
import numpy as np

from collections import deque
from ultralytics import YOLO

VIDEO = "OSUshortclip.mp4"
OUTPUT = "soccer_tracker_out.mp4"
MODEL = "yolo26s.pt"  # weights download automatically on first run

PERSON, BALL = 0, 32  # COCO class ids ("person", "sports ball")

model = YOLO(MODEL)

results = model.predict(
    source=VIDEO,
    classes=[PERSON, BALL],
    conf=0.2,
    imgsz=1280,
    save=True,
    stream=True,
)

ball_count = 0
ball_path = deque(maxlen=30) # keep last 30 ball positions

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()  # only opened it to read these values

out = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

start_time = time.perf_counter()
for i, r in enumerate(results):
    frame_counted = False
    if i >= 120:
        break
    balls = [(conf, box) for cls, conf, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())
             if int(cls) == BALL]
    for conf, box in balls:
        print(f"frame {i}: ball {conf:.2f} at x={box[0]:.0f}, y={box[1]:.0f}")

    keep_ball = max(balls, key=lambda b: b[0]) if balls else None
    frame = r.plot()
    if keep_ball:
        ball_count += 1
        conf, box = keep_ball
        center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
        ball_path.append(center)
    for p1, p2 in zip(ball_path, list(ball_path)[1:]):
        cv2.line(frame, p1, p2, (0, 140, 255), 2)  # orange trail
    out.write(frame)

out.release()
end_time = time.perf_counter()
print(f"Final ball count out of 120 frames: {ball_count}")
elapsed_time = end_time - start_time
print(f"Processing time: {elapsed_time:.4f} seconds")