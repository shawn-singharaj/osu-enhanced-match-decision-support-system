"""
Goal: Homography
Limitations: Not all marked lines are always in frame, need to figure out how to "estimate it", eventually create our own grid
Claude says we use homography.
"""

import time
import numpy as np
import cv2

VIDEO = "FieldBorders.mp4"
OUTPUT = "field_mapping_out.mp4"

MAX_HORIZONTAL_ANGLE = 15      # degrees from horizontal; sidelines ~0-10, goal lines ~20+, halfway line ~80
MAX_MISSING = 15    # frames to keep showing a line after it stops being detected
L_GOALLINE_MIN_ANGLE = 15 # steeper than sidelines
L_GOALLINE_MAX_ANGLE = 45 # flatter than halfway line
R_GOALLINE_MIN_ANGLE = 135
R_GOALLINE_MAX_ANGLE = 165

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# how close (px) a line must be to the grass edge to count as a sideline. The near side is
# closer to the camera, so the grass strip beyond the line looks wider there.
FAR_TOL = 0.035 * h
NEAR_TOL = 0.10 * h
TOUCH_TOL = 0.02 * h        # how close (20 px) goal line's top must be to the sidelines
GOALLINE_JUMP = 0.08 * w    # for smoothing, measured horizontally
SIDELINE_JUMP = 0.08 * h

MIN_GOALLINE_LENGTH = w // 6

goal_line_drawn = 0

# Returns solid white silhouette of pitch
def filter_grass(hsv):
    mask = cv2.inRange(hsv, (25, 40, 40), (90, 255, 255))  # filter by green (this grass is hue ~33)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))  # gaps smaller than 25px get filled
    # number of labels, image same size as mask where each pixel holds its blob number, one row per blob/pixel count
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    # keep biggest blob in stats, aka field
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    field = np.where(labels == biggest, 255, 0).astype(np.uint8)
    # trace outer outline of blob, compress straight runs of the outline to their endpoints
    contours, _ = cv2.findContours(field, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(field)
    # paints inside of outline solid onto blank image of all contours
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled

# Returns black image with painted lines in white + noise
def white_on_grass(frame, hsv, field):
    # convert to brightness only
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # make 9 x 9 kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # compute image - open(image) to erase bright things narrower than kernel, so get the plain grass
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    # pixels at least 15 brighter than their surroundings become 255, everything else 0, to be compared to field
    bright = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)[1]
    low_sat = cv2.inRange(hsv, (0, 0, 0), (180, 90, 255))  # keep white/grey
    # keep a pixel only where both masks are 255
    lines = cv2.bitwise_and(bright, low_sat)
    return cv2.bitwise_and(lines, lines, mask=field)

def detect_segments(frame):
    # convert BGR to HSV to mask colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    field = filter_grass(hsv)
    white = white_on_grass(frame, hsv, field)
    # hough transform has every white pixel vote for all the lines that could pass thru it
    # return segments with endpoints
    # color, rho (distance resolution in pixel), theta (angle resolution), threshold (min votes), discard short segs, fill gaps up to 60 px
    segs = cv2.HoughLinesP(white, 1, np.pi / 180, threshold=50,
                           minLineLength=w // 12, maxLineGap=60)
    return field, segs

# Returns far and near sidelines
def find_sidelines(field, segs):
    if segs is None:
        return None, None

    # Begin filtering segments:
    # grass top/bottom across the middle of the frame (median is robust to a player standing on the edge)
    center_cols = field[:, int(w * 0.4):int(w * 0.6)] > 0 # take middle 20% of columns as true/false array
    has_grass = center_cols.any(axis=0)
    if not has_grass.any():
        return None, None
    center_top = np.median(center_cols.argmax(axis=0)[has_grass])
    center_bot = np.median((h - 1 - center_cols[::-1].argmax(axis=0))[has_grass])

    far_cands, near_cands = [], []
    for x1, y1, x2, y2 in segs.reshape(-1, 4):  # resize for OpenCV
        # keep near-horizontal segments only (drops halfway line / goal lines)
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if MAX_HORIZONTAL_ANGLE < angle < 180 - MAX_HORIZONTAL_ANGLE:
            continue
        # line equation y = mx + b, 1e-6 to avoid dividing by 0
        m = (y2 - y1) / (x2 - x1 + 1e-6)
        b = y1 - m * x1
        length = np.hypot(x2 - x1, y2 - y1)

        # a sideline sits right at the edge of the grass; interior lines (penalty box) don't
        x_mid = min(max((x1 + x2) // 2, 0), w - 1) # get midpoint of segment
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

def find_goalines(far_line, segs):
    # take segments, filter by vertical lines, create line equation x = my + b, see if line touches both sidelines, return that, handle if one goaline too
    if segs is None or far_line is None:
        return None, None

    right_cands, left_cands = [], []
    for x1, y1, x2, y2 in segs.reshape(-1, 4):  # resize for OpenCV
        angle =  abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if not is_goaline(angle):
            continue
        # x = my + b for vertical lines
        m = (x2 - x1) / (y2 - y1 + 1e-6)
        b = x1 -  m * y1

        # minimum segment length
        if np.hypot(x2 - x1, y2 - y1) < MIN_GOALLINE_LENGTH:
            continue

        # get the upper endpoint of the line
        x_top, y_top = (x1, y1) if y1 < y2 else (x2, y2)

        mf, bf = far_line
        y_sideline = mf * x_top + bf
        touches_sideline = abs(y_top - y_sideline) < TOUCH_TOL
        # if our endpoint touches the far sideline
        if not touches_sideline:
            continue

        # right or left goal line?
        if m > 0:
            right_cands.append((x_top, m, b)) 
        elif m < 0:
            left_cands.append((x_top, m, b))

    right = max(right_cands)[1:] if right_cands else None
    left = min(left_cands)[1:] if left_cands else None
    return right, left

def is_goaline(angle):
    if L_GOALLINE_MIN_ANGLE <= angle <= L_GOALLINE_MAX_ANGLE:
        return True 

    elif R_GOALLINE_MIN_ANGLE <= angle <= R_GOALLINE_MAX_ANGLE:
        return True

    return False

# Where line crosses the middle column, single number that summarizes the line's position
def y_at_center(line):
    m, b = line
    return m * (w / 2) + b

# goal line counterpart to y_at_center
def x_at_middle_row(line):        
    m, b = line
    return m * (h / 2) + b

# Return if there is a new or changed detection of line when comparing to past line
def smooth(prev, new, max_jump, pos_fn, alpha=0.3):
    if new is None:
        return prev
    if prev is None:
        return new
    if abs(pos_fn(new) - pos_fn(prev)) > max_jump:
        return prev  # sudden jump = probably a wrong line this frame; ignore it
    return tuple(alpha * n + (1 - alpha) * p for p, n in zip(prev, new))

# Draws the line on the output video
def draw_sideline(frame, line, color):
    if line is None:
        return
    m, b = line
    cv2.line(frame, (0, int(b)), (w, int(m * w + b)), color, 3)  # extend across the frame

def draw_goalline(frame, line, far_line, color):
    if line is None:
        return
    mg, bg = line  # goal line: x = mg*y + bg

    # start at the corner flag: where the goal line meets the far sideline (y = mf*x + bf).
    # substitute x into the sideline equation: y = mf*(mg*y + bg) + bf  ->  solve for y
    y_start = 0
    if far_line is not None:
        mf, bf = far_line
        denom = 1 - mf * mg
        if abs(denom) > 1e-6:  # 0 would mean the two lines are parallel (never meet)
            y_start = (mf * bg + bf) / denom
    x_start = mg * y_start + bg

    # end at the bottom of the frame; cv2.line clips anything off-screen
    y_end = h
    x_end = mg * y_end + bg
    
    cv2.line(frame, (int(x_start), int(y_start)), (int(x_end), int(y_end)), color, 3)
    global goal_line_drawn
    goal_line_drawn += 1

far_line = near_line = right_line = left_line = None
far_missing = near_missing = right_missing = left_missing = 0
start_time = time.perf_counter()
i = 0
while i < 1000:
    ok, frame = cap.read()
    if not ok:
        break

    field, segs = detect_segments(frame)
    far, near = find_sidelines(field, segs)

    # count frames without a detection; drop the line if it's been gone too long
    far_missing = 0 if far is not None else far_missing + 1
    near_missing = 0 if near is not None else near_missing + 1
    far_line = None if far_missing > MAX_MISSING else smooth(far_line, far, SIDELINE_JUMP, y_at_center)
    near_line = None if near_missing > MAX_MISSING else smooth(near_line, near, SIDELINE_JUMP, y_at_center)

    right, left = find_goalines(far_line, segs)

    right_missing = 0 if right is not None else right_missing + 1
    left_missing = 0 if left is not None else left_missing + 1
    right_line = None if right_missing > MAX_MISSING else smooth(right_line, right, GOALLINE_JUMP, x_at_middle_row)
    left_line = None if left_missing > MAX_MISSING else smooth(left_line, left, GOALLINE_JUMP, x_at_middle_row)

    draw_sideline(frame, far_line, (0, 0, 255))   # red = far sideline
    draw_sideline(frame, near_line, (255, 0, 0))  # blue = near sideline
    draw_goalline(frame, right_line, far_line, (0, 255, 255))  # yellow
    draw_goalline(frame, left_line, far_line, (255, 0, 255))

    out.write(frame)
    i += 1

cap.release()
out.release()
print(f"Processing time: {time.perf_counter() - start_time:.4f} s")
print(f"Frames that goal line was drawn: {goal_line_drawn}")
