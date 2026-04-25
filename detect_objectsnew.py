import os
import cv2
import json
import numpy as np

# =========================================================
# CONFIG
# =========================================================

SCAN_DIR = "scan_images"
DEBUG_DIR = "debug_objects"
RESULTS_DIR = "results"

HEADINGS = ["front", "right", "back", "left"]

# Image ROI used for the 3 front slots
ROI_TOP_FRAC = 0.34
ROI_BOT_FRAC = 0.94

SLOT_PAD_X_FRAC = 0.03
SLOT_PAD_Y_FRAC = 0.06

# =========================================================
# HSV THRESHOLDS
# =========================================================

# White box detection
WHITE_S_MAX = 80
WHITE_V_MIN = 150

# Black X detection
BLACK_V_MAX = 80

# Red X detection
RED_S_MIN = 90
RED_V_MIN = 90

# Candidate box filtering
MIN_WHITE_AREA_FRAC = 0.025
MAX_WHITE_AREA_FRAC = 0.40

MIN_BOX_W_FRAC = 0.18
MIN_BOX_H_FRAC = 0.18

# X classification thresholds
MIN_BLACK_X_FRAC = 0.025
MIN_RED_X_FRAC = 0.025

# Strong bias toward empty if unsure
REQUIRE_WHITE_BOX = True

# =========================================================
# OUTPUT
# =========================================================

os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def clean_mask(mask, ksize=5, iterations=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return mask


def get_front_slots(img):
    h, w = img.shape[:2]

    y1 = int(h * ROI_TOP_FRAC)
    y2 = int(h * ROI_BOT_FRAC)

    roi_h = y2 - y1
    slot_w = w // 3

    slots = []

    for i in range(3):
        x1 = i * slot_w
        x2 = (i + 1) * slot_w if i < 2 else w

        px = int((x2 - x1) * SLOT_PAD_X_FRAC)
        py = int(roi_h * SLOT_PAD_Y_FRAC)

        sx1 = max(0, x1 + px)
        sx2 = min(w, x2 - px)
        sy1 = max(0, y1 + py)
        sy2 = min(h, y2 - py)

        slots.append((sx1, sy1, sx2, sy2))

    return slots


def detect_white_box(slot):
    hsv = cv2.cvtColor(slot, cv2.COLOR_BGR2HSV)
    h, w = slot.shape[:2]
    slot_area = h * w

    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, WHITE_V_MIN]),
        np.array([179, WHITE_S_MAX, 255])
    )

    white_mask = clean_mask(white_mask, ksize=5, iterations=1)

    contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_frac = area / slot_area

        if area_frac < MIN_WHITE_AREA_FRAC or area_frac > MAX_WHITE_AREA_FRAC:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw < MIN_BOX_W_FRAC * w or bh < MIN_BOX_H_FRAC * h:
            continue

        aspect = bw / float(bh)
        if aspect < 0.45 or aspect > 2.2:
            continue

        if area > best_area:
            best_area = area
            best = (x, y, bw, bh, white_mask)

    return best


def classify_x(slot, box):
    x, y, bw, bh, white_mask = box

    # Focus only inside the detected white box
    pad_x = int(0.08 * bw)
    pad_y = int(0.08 * bh)

    x1 = max(0, x + pad_x)
    y1 = max(0, y + pad_y)
    x2 = min(slot.shape[1], x + bw - pad_x)
    y2 = min(slot.shape[0], y + bh - pad_y)

    crop = slot[y1:y2, x1:x2]

    if crop.size == 0:
        return "E", 0.0, 0.0, None, None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Black X mask
    black_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 0]),
        np.array([179, 255, BLACK_V_MAX])
    )

    # Red X mask: red wraps around HSV hue
    red_mask1 = cv2.inRange(
        hsv,
        np.array([0, RED_S_MIN, RED_V_MIN]),
        np.array([10, 255, 255])
    )

    red_mask2 = cv2.inRange(
        hsv,
        np.array([170, RED_S_MIN, RED_V_MIN]),
        np.array([179, 255, 255])
    )

    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    black_mask = clean_mask(black_mask, ksize=3, iterations=1)
    red_mask = clean_mask(red_mask, ksize=3, iterations=1)

    crop_area = crop.shape[0] * crop.shape[1]

    black_frac = cv2.countNonZero(black_mask) / crop_area
    red_frac = cv2.countNonZero(red_mask) / crop_area

    # Classification rule
    if red_frac >= MIN_RED_X_FRAC and red_frac > black_frac:
        return "O", black_frac, red_frac, black_mask, red_mask

    if black_frac >= MIN_BLACK_X_FRAC and black_frac >= red_frac:
        return "T", black_frac, red_frac, black_mask, red_mask

    return "E", black_frac, red_frac, black_mask, red_mask


def process_heading(heading):
    img_path = os.path.join(SCAN_DIR, f"{heading}.jpg")

    if not os.path.exists(img_path):
        print(f"[WARN] Missing image: {img_path}")
        return ["E", "E", "E"]

    img = cv2.imread(img_path)

    if img is None:
        print(f"[WARN] Could not read image: {img_path}")
        return ["E", "E", "E"]

    debug = img.copy()
    slots = get_front_slots(img)

    result = []

    for idx, (x1, y1, x2, y2) in enumerate(slots):
        slot = img[y1:y2, x1:x2]

        box = detect_white_box(slot)

        if box is None:
            label = "E"
            black_frac = 0.0
            red_frac = 0.0
        else:
            label, black_frac, red_frac, black_mask, red_mask = classify_x(slot, box)

            bx, by, bw, bh, _ = box
            cv2.rectangle(
                debug,
                (x1 + bx, y1 + by),
                (x1 + bx + bw, y1 + by + bh),
                (255, 255, 255),
                2
            )

        result.append(label)

        # Draw slot
        color = (0, 255, 0)
        if label == "T":
            color = (0, 0, 0)
        elif label == "O":
            color = (0, 0, 255)

        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            debug,
            f"{label} B:{black_frac:.3f} R:{red_frac:.3f}",
            (x1 + 10, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

    out_path = os.path.join(DEBUG_DIR, f"{heading}_object_debug.jpg")
    cv2.imwrite(out_path, debug)

    return result


def build_object_3x3(front, right, back, left):
    """
    Local 3x3 object grid.

    Center is agent A.

    This keeps the same scan logic:
    front heading fills top row.
    right heading fills right column.
    back heading fills bottom row.
    left heading fills left column.
    """

    grid = [
        ["E", "E", "E"],
        ["E", "A", "E"],
        ["E", "E", "E"]
    ]

    # front image: left, center, right
    grid[0][0] = front[0]
    grid[0][1] = front[1]
    grid[0][2] = front[2]

    # right image
    grid[0][2] = strongest(grid[0][2], right[0])
    grid[1][2] = right[1]
    grid[2][2] = right[2]

    # back image
    grid[2][2] = strongest(grid[2][2], back[0])
    grid[2][1] = back[1]
    grid[2][0] = back[2]

    # left image
    grid[2][0] = strongest(grid[2][0], left[0])
    grid[1][0] = left[1]
    grid[0][0] = strongest(grid[0][0], left[2])

    return grid


def strongest(a, b):
    """
    Resolve duplicated corner observations.
    Object wins over empty.
    Obstacle wins over target if conflict, because obstacle is safety-critical.
    """
    if a == "O" or b == "O":
        return "O"
    if a == "T" or b == "T":
        return "T"
    return "E"


def save_grid(grid):
    txt_path = os.path.join(RESULTS_DIR, "object_3x3.txt")
    json_path = os.path.join(RESULTS_DIR, "object_3x3.json")

    with open(txt_path, "w") as f:
        for row in grid:
            f.write(" ".join(row) + "\n")

    with open(json_path, "w") as f:
        json.dump({"object_grid": grid}, f, indent=2)

    print(f"[OK] Saved: {txt_path}")
    print(f"[OK] Saved: {json_path}")


def main():
    print("[INFO] Starting object detection...")

    heading_results = {}

    for heading in HEADINGS:
        row = process_heading(heading)
        heading_results[heading] = row
        print(f"{heading}: {row}")

    object_grid = build_object_3x3(
        heading_results["front"],
        heading_results["right"],
        heading_results["back"],
        heading_results["left"]
    )

    print("\nFinal object 3x3:")
    for row in object_grid:
        print(row)

    save_grid(object_grid)

    print(f"[OK] Debug images saved in: {DEBUG_DIR}")


if __name__ == "__main__":
    main()
