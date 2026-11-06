import time
import numpy as np
import cv2

VIDEO = "FieldBorders.mp4"
OUTPUT = "field_mapping_out.mp4"

MAX_ANGLE = 15      # degrees from horizontal; sidelines ~0-10, goal lines ~20+, halfway line ~80
MAX_MISSING = 15    # frames to keep showing a line after it stops being detected

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# how close (px) a line must be to the grass edge to count as a sideline. The near side is
# closer to the camera, so the grass strip beyond the line looks wider there.
FAR_TOL = 0.035 * h
NEAR_TOL = 0.10 * h


def filter_grass(hsv):
    mask = cv2.inRange(hsv, (25, 40, 40), (90, 255, 255))  # filter by green (this grass is hue ~33)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))  # fill lines/players
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    # keep biggest, aka field
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    field = np.where(labels == biggest, 255, 0).astype(np.uint8)
    # fill any holes left inside (players, shadows) by filling the outer contour
    contours, _ = cv2.findContours(field, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(field)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def white_on_grass(frame, hsv, field):
    # lines are thin and dim from this distance, so look for pixels brighter than
    # their surroundings (top-hat) instead of a fixed "pure white" threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    bright = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)[1]
    low_sat = cv2.inRange(hsv, (0, 0, 0), (180, 90, 255))  # whitish, not orange/colored
    lines = cv2.bitwise_and(bright, low_sat)
    return cv2.bitwise_and(lines, lines, mask=field)


def find_sidelines(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    field = filter_grass(hsv)
    white = white_on_grass(frame, hsv, field)
    segs = cv2.HoughLinesP(white, 1, np.pi / 180, threshold=50,
                           minLineLength=w // 12, maxLineGap=60)
    if segs is None:
        return None, None

    # grass top/bottom across the middle of the frame (median is robust to a player standing on the edge)
    center_cols = field[:, int(w * 0.4):int(w * 0.6)] > 0
    has_grass = center_cols.any(axis=0)
    if not has_grass.any():
        return None, None
    center_top = np.median(center_cols.argmax(axis=0)[has_grass])
    center_bot = np.median((h - 1 - center_cols[::-1].argmax(axis=0))[has_grass])

    far_cands, near_cands = [], []
    for x1, y1, x2, y2 in segs.reshape(-1, 4):  # OpenCV 5 returns (N, 4)
        # keep near-horizontal segments only (drops halfway line / goal lines)
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if MAX_ANGLE < angle < 180 - MAX_ANGLE:
            continue
        m = (y2 - y1) / (x2 - x1 + 1e-6)
        b = y1 - m * x1
        length = np.hypot(x2 - x1, y2 - y1)

        # a sideline sits right at the edge of the grass; interior lines (penalty box) don't
        x_mid = min(max((x1 + x2) // 2, 0), w - 1)
        y_mid = m * x_mid + b
        grass_rows = np.where(field[:, x_mid] > 0)[0]
        if len(grass_rows) == 0:
            continue
        # ...and extended to the frame center it must still be at the edge. This rejects
        # penalty-box lines near the goal and lines on neighbouring fields.
        y_center = m * (w / 2) + b
        if abs(y_mid - grass_rows.min()) < FAR_TOL and abs(y_center - center_top) < FAR_TOL:
            far_cands.append((length, m, b))
        elif abs(y_mid - grass_rows.max()) < NEAR_TOL and abs(y_center - center_bot) < NEAR_TOL:
            near_cands.append((length, m, b))

    # longest segment wins
    far = max(far_cands)[1:] if far_cands else None
    near = max(near_cands)[1:] if near_cands else None
    return far, near


def y_at_center(line):
    m, b = line
    return m * (w / 2) + b


def smooth(prev, new, alpha=0.3, max_jump=0.08 * h):
    if new is None:
        return prev
    if prev is None:
        return new
    if abs(y_at_center(new) - y_at_center(prev)) > max_jump:
        return prev  # sudden jump = probably a wrong line this frame; ignore it
    return tuple(alpha * n + (1 - alpha) * p for p, n in zip(prev, new))


def draw_line(frame, line, color):
    if line is None:
        return
    m, b = line
    cv2.line(frame, (0, int(b)), (w, int(m * w + b)), color, 3)  # extend across the frame


far_line = near_line = None
far_missing = near_missing = 0
start_time = time.perf_counter()
i = 0
while i < 500:
    ok, frame = cap.read()
    if not ok:
        break
    far, near = find_sidelines(frame)

    # count frames without a detection; drop the line if it's been gone too long
    far_missing = 0 if far is not None else far_missing + 1
    near_missing = 0 if near is not None else near_missing + 1
    far_line = None if far_missing > MAX_MISSING else smooth(far_line, far)
    near_line = None if near_missing > MAX_MISSING else smooth(near_line, near)

    draw_line(frame, far_line, (0, 0, 255))   # red = far sideline
    draw_line(frame, near_line, (255, 0, 0))  # blue = near sideline
    out.write(frame)
    i += 1

cap.release()
out.release()
print(f"Processing time: {time.perf_counter() - start_time:.4f} s")
