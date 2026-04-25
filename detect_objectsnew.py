import os
import cv2
import numpy as np

# =========================================================
# INPUT / OUTPUT
# =========================================================
SCAN_DIR = "scan_images"
RESULTS_DIR = "results"
DEBUG_DIR = "debug_objects"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

HEADINGS = ["front", "right", "back", "left"]

IMAGE_PATHS = {
    "front": os.path.join(SCAN_DIR, "front.jpg"),
    "right": os.path.join(SCAN_DIR, "right.jpg"),
    "back":  os.path.join(SCAN_DIR, "back.jpg"),
    "left":  os.path.join(SCAN_DIR, "left.jpg"),
}

# =========================================================
# ROI / SLOT SETTINGS
# =========================================================
ROI_TOP_FRAC = 0.62
ROI_BOT_FRAC = 0.93

SLOT_PAD_X_FRAC = 0.02
SLOT_PAD_Y_FRAC = 0.06

# =========================================================
# STRICT WHITE BOX DETECTION
# =========================================================
WHITE_THRESH = 135
MIN_WHITE_AREA = 900
MIN_WHITE_FILL = 0.35

MIN_BOX_W = 30
MIN_BOX_H = 30
MIN_ASPECT = 0.65
MAX_ASPECT = 1.35

# =========================================================
# TARGET = WHITE BOX + BLACK X
# =========================================================
BLACK_THRESH = 90
MIN_BLACK_DIAG_SCORE = 0.12
MIN_BLACK_COMBINED = 0.28

# =========================================================
# OBSTACLE = WHITE BOX + RED X
# =========================================================
RED_MIN_R = 110
RED_MARGIN = 35
MIN_RED_DIAG_SCORE = 0.10
MIN_RED_COMBINED = 0.24

# =========================================================
# EXTRA RELIABILITY SETTINGS
# =========================================================
INNER_CROP_FRAC = 0.12
DIAG_BAND_FRAC = 0.08

EMPTY = "EMPTY"
TARGET = "TARGET"
OBSTACLE = "OBSTACLE"
AGENT = "AGENT"

# Local 3x3 object placement around agent
HEADING_TO_POSITIONS = {
    "front": [(-1, +1), (0, +1), (+1, +1)],
    "right": [(+1, +1), (+1, 0), (+1, -1)],
    "back":  [(+1, -1), (0, -1), (-1, -1)],
    "left":  [(-1, -1), (-1, 0), (-1, +1)],
}


def state_to_char(state):
    if state == TARGET:
        return "T"
    if state == OBSTACLE:
        return "X"
    if state == AGENT:
        return "A"
    return "E"


def blank_local_grid():
    grid = {}
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            grid[(dx, dy)] = EMPTY
    grid[(0, 0)] = AGENT
    return grid


def pretty_matrix(grid):
    rows = []
    for dy in [1, 0, -1]:
        row = []
        for dx in [-1, 0, 1]:
            row.append(state_to_char(grid[(dx, dy)]))
        rows.append(row)
    return rows


def print_matrix(grid):
    print("\nFinal 3x3 object matrix:")
    for row in pretty_matrix(grid):
        print(" ".join(row))


def make_slot_boxes(w, h):
    y0 = int(h * ROI_TOP_FRAC)
    y1 = int(h * ROI_BOT_FRAC)

    roi_h = y1 - y0
    slot_w = w // 3

    boxes = []
    for i in range(3):
        x0 = i * slot_w
        x1 = (i + 1) * slot_w

        pad_x = int(slot_w * SLOT_PAD_X_FRAC)
        pad_y = int(roi_h * SLOT_PAD_Y_FRAC)

        sx0 = max(0, x0 + pad_x)
        sx1 = min(w, x1 - pad_x)
        sy0 = max(0, y0 + pad_y)
        sy1 = min(h, y1 - pad_y)

        boxes.append((sx0, sy0, sx1, sy1))

    return boxes


def diagonal_band_masks(h, w):
    yy, xx = np.indices((h, w))
    band = max(2, int(min(h, w) * DIAG_BAND_FRAC))

    d1 = np.abs(yy - ((h - 1) / max(1, w - 1)) * xx) <= band
    d2 = np.abs(yy - ((h - 1) - ((h - 1) / max(1, w - 1)) * xx)) <= band

    return d1, d2


def inner_crop(arr):
    h, w = arr.shape[:2]

    mx = int(w * INNER_CROP_FRAC)
    my = int(h * INNER_CROP_FRAC)

    x0 = mx
    x1 = w - mx
    y0 = my
    y1 = h - my

    if x1 <= x0 or y1 <= y0:
        return arr

    return arr[y0:y1, x0:x1]


WHITE_THRESH = 135      # instead of 190
MIN_WHITE_AREA = 900    # instead of 1500
MIN_WHITE_FILL = 0.35   # instead of 0.60

Also make the white detection use HSV instead of grayscale only. Replace get_strict_white_candidates() with this version:

def get_strict_white_candidates(slot_bgr):
    """
    Detect white box using HSV + brightness.
    More reliable than gray threshold only under Pi camera lighting.
    """
    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # white = bright enough and low saturation
    white_mask = ((v >= WHITE_THRESH) & (s <= 90)).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        white_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_WHITE_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w < MIN_BOX_W or h < MIN_BOX_H:
            continue

        aspect = w / float(h)
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        roi_white = white_mask[y:y+h, x:x+w]
        if roi_white.size == 0:
            continue

        white_fill = np.mean(roi_white > 0)

        if white_fill < MIN_WHITE_FILL:
            continue

        candidates.append((x, y, w, h, white_fill))

    return candidates, white_mask


def black_x_scores(inner_bgr):
    gray = cv2.cvtColor(inner_bgr, cv2.COLOR_BGR2GRAY)
    black_mask = gray < BLACK_THRESH

    h, w = gray.shape
    if h < 20 or w < 20:
        return 0.0, 0.0, black_mask

    d1, d2 = diagonal_band_masks(h, w)

    s1 = float(black_mask[d1].mean()) if np.any(d1) else 0.0
    s2 = float(black_mask[d2].mean()) if np.any(d2) else 0.0

    return s1, s2, black_mask


def red_x_scores(inner_bgr):
    b = inner_bgr[:, :, 0].astype(np.int16)
    g = inner_bgr[:, :, 1].astype(np.int16)
    r = inner_bgr[:, :, 2].astype(np.int16)

    red_mask = (
        (r >= RED_MIN_R) &
        ((r - g) >= RED_MARGIN) &
        ((r - b) >= RED_MARGIN)
    )

    h, w = red_mask.shape
    if h < 20 or w < 20:
        return 0.0, 0.0, red_mask

    d1, d2 = diagonal_band_masks(h, w)

    s1 = float(red_mask[d1].mean()) if np.any(d1) else 0.0
    s2 = float(red_mask[d2].mean()) if np.any(d2) else 0.0

    return s1, s2, red_mask


def classify_marker_in_slot(slot_bgr):
    """
    Strict logic:
    1. Find real white square.
    2. Crop inside that square.
    3. Check if black X or red X exists.
    4. If uncertain, return EMPTY.
    """
    candidates, white_mask = get_strict_white_candidates(slot_bgr)

    best_state = EMPTY
    best_rect = None
    best_info = {
        "reason": "no_strong_white_box",
        "black_combined": 0.0,
        "red_combined": 0.0,
        "white_fill": 0.0,
    }

    best_score = -1.0

    for x, y, w, h, white_fill in candidates:
        box_roi = slot_bgr[y:y+h, x:x+w]
        if box_roi.size == 0:
            continue

        inner = inner_crop(box_roi)
        if inner.size == 0:
            continue

        b1, b2, _ = black_x_scores(inner)
        r1, r2, _ = red_x_scores(inner)

        black_combined = b1 + b2
        red_combined = r1 + r2

        # -------------------------------
        # Hard reject weak detections
        # -------------------------------
        if black_combined < MIN_BLACK_COMBINED and red_combined < MIN_RED_COMBINED:
            continue

        is_target = (
            b1 >= MIN_BLACK_DIAG_SCORE and
            b2 >= MIN_BLACK_DIAG_SCORE and
            black_combined >= MIN_BLACK_COMBINED
        )

        is_obstacle = (
            r1 >= MIN_RED_DIAG_SCORE and
            r2 >= MIN_RED_DIAG_SCORE and
            red_combined >= MIN_RED_COMBINED
        )

        # Extra safety: weak partial response is ignored
        if black_combined < MIN_BLACK_COMBINED:
            is_target = False

        if red_combined < MIN_RED_COMBINED:
            is_obstacle = False

        state = EMPTY
        score = -1.0

        target_score = black_combined + 0.25 * white_fill
        obstacle_score = red_combined + 0.25 * white_fill

        if is_target and not is_obstacle:
            state = TARGET
            score = target_score

        elif is_obstacle and not is_target:
            state = OBSTACLE
            score = obstacle_score

        elif is_target and is_obstacle:
            # If both trigger, choose only if one is clearly stronger.
            if target_score > obstacle_score + 0.10:
                state = TARGET
                score = target_score
            elif obstacle_score > target_score + 0.10:
                state = OBSTACLE
                score = obstacle_score
            else:
                state = EMPTY
                score = -1.0

        if state != EMPTY and score > best_score:
            best_score = score
            best_state = state
            best_rect = (x, y, w, h)
            best_info = {
                "reason": "valid_white_box_and_x",
                "white_fill": white_fill,
                "black_d1": b1,
                "black_d2": b2,
                "black_combined": black_combined,
                "red_d1": r1,
                "red_d2": r2,
                "red_combined": red_combined,
                "score": score,
            }

    return best_state, best_rect, best_info


def color_for_state(state):
    if state == TARGET:
        return (0, 255, 0)
    if state == OBSTACLE:
        return (0, 0, 255)
    return (180, 180, 180)


def draw_text(img, text, x, y, color=(255, 255, 255), scale=0.5, thick=1):
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thick,
        cv2.LINE_AA
    )


def process_heading_image(heading, img):
    view = img.copy()
    h, w = img.shape[:2]
    slots = make_slot_boxes(w, h)

    states = []
    infos = []

    for i, (x0, y0, x1, y1) in enumerate(slots):
        slot = img[y0:y1, x0:x1]

        state, rect, info = classify_marker_in_slot(slot)
        states.append(state)
        infos.append(info)

        color = color_for_state(state)

        cv2.rectangle(view, (x0, y0), (x1, y1), (255, 255, 0), 2)
        draw_text(view, f"slot {i}: {state}", x0 + 5, y0 + 20, color, 0.55, 2)

        if rect is not None and state != EMPTY:
            rx, ry, rw, rh = rect
            cv2.rectangle(
                view,
                (x0 + rx, y0 + ry),
                (x0 + rx + rw, y0 + ry + rh),
                color,
                2
            )

        b = info.get("black_combined", 0.0)
        r = info.get("red_combined", 0.0)
        wf = info.get("white_fill", 0.0)

        draw_text(view, f"B:{b:.2f} R:{r:.2f} W:{wf:.2f}", x0 + 5, y1 - 10, color)

    row_chars = [state_to_char(s) for s in states]

    draw_text(view, f"heading: {heading}", 10, 25, (255, 255, 255), 0.65, 2)
    draw_text(view, f"row: {','.join(row_chars)}", 180, 25, (255, 255, 255), 0.65, 2)

    debug_path = os.path.join(DEBUG_DIR, f"{heading}_objects_debug.jpg")
    cv2.imwrite(debug_path, view)

    return states, infos, debug_path


def apply_heading_to_grid(local_grid, heading, states):
    positions = HEADING_TO_POSITIONS[heading]

    for pos, state in zip(positions, states):
        if pos == (0, 0):
            continue

        local_grid[pos] = state


def save_matrix_txt(local_grid):
    mat = pretty_matrix(local_grid)

    out_path = os.path.join(RESULTS_DIR, "object_3x3.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        for row in mat:
            f.write(",".join(row) + "\n")

    return out_path


def main():
    print("=== Strict object detection from saved scan images ===")
    print("Target   = white box + thick black X")
    print("Obstacle = white box + thick red X")
    print("Bias     = EMPTY if uncertain")

    local_grid = blank_local_grid()

    for heading in HEADINGS:
        path = IMAGE_PATHS[heading]

        if not os.path.exists(path):
            print(f"ERROR: Missing image for {heading}: {path}")
            return

        img = cv2.imread(path)
        if img is None:
            print(f"ERROR: Could not read image: {path}")
            return

        states, infos, debug_path = process_heading_image(heading, img)
        apply_heading_to_grid(local_grid, heading, states)

        print(f"\nHeading: {heading}")
        print(f"image: {path}")
        print(f"states: {[state_to_char(s) for s in states]}")
        print(f"debug: {debug_path}")

        for i, info in enumerate(infos):
            print(
                f"  slot {i}: "
                f"W={info.get('white_fill', 0.0):.2f}, "
                f"B={info.get('black_combined', 0.0):.2f}, "
                f"R={info.get('red_combined', 0.0):.2f}, "
                f"reason={info.get('reason', '')}"
            )

    print_matrix(local_grid)

    out_path = save_matrix_txt(local_grid)

    print(f"\nSaved object matrix to: {out_path}")
    print(f"Saved debug images to: {DEBUG_DIR}")


if __name__ == "__main__":
    main()
