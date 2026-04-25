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

# Lower ROI: focus more on first/front row only
ROI_TOP_FRAC = 0.48
ROI_BOT_FRAC = 0.98

SLOT_PAD_X_FRAC = 0.03
SLOT_PAD_Y_FRAC = 0.02

# Inside each slot, inspect lower/front part
FRONT_ROW_TOP_FRAC = 0.25

# =========================================================
# HSV THRESHOLDS
# =========================================================

# White object box
WHITE_S_MAX = 140
WHITE_V_MIN = 110

# Black X target
BLACK_V_MAX = 95

# Red X obstacle
RED_S_MIN = 100
RED_V_MIN = 90

# Detection thresholds
MIN_WHITE_FRAC = 0.025
MIN_BLACK_FRAC = 0.020
MIN_RED_FRAC = 0.025

# Object-zone dilation controls how much area around white box is allowed
OBJECT_ZONE_KERNEL = 17
OBJECT_ZONE_ITERATIONS = 2

# =========================================================
# SETUP
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


def classify_slot_front_row_only(slot):
    h, w = slot.shape[:2]

    y_start = int(h * FRONT_ROW_TOP_FRAC)
    crop = slot[y_start:h, :]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # White box mask
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, WHITE_V_MIN]),
        np.array([179, WHITE_S_MAX, 255])
    )

    # Black X mask
    black_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 0]),
        np.array([179, 255, BLACK_V_MAX])
    )

    # Red X mask
    red1 = cv2.inRange(
        hsv,
        np.array([0, RED_S_MIN, RED_V_MIN]),
        np.array([10, 255, 255])
    )

    red2 = cv2.inRange(
        hsv,
        np.array([170, RED_S_MIN, RED_V_MIN]),
        np.array([179, 255, 255])
    )

    red_mask = cv2.bitwise_or(red1, red2)

    # Clean masks
    white_mask = clean_mask(white_mask, 5, 1)
    black_mask = clean_mask(black_mask, 3, 1)
    red_mask = clean_mask(red_mask, 3, 1)

    # =====================================================
    # IMPORTANT UPDATE:
    # Only count black/red pixels close to the white object.
    # This blocks red/pink tile background from becoming O.
    # =====================================================
    kernel_big = np.ones((OBJECT_ZONE_KERNEL, OBJECT_ZONE_KERNEL), np.uint8)
    object_zone = cv2.dilate(white_mask, kernel_big, iterations=OBJECT_ZONE_ITERATIONS)

    black_mask = cv2.bitwise_and(black_mask, object_zone)
    red_mask = cv2.bitwise_and(red_mask, object_zone)

    area = crop.shape[0] * crop.shape[1]

    white_frac = cv2.countNonZero(white_mask) / area
    black_frac = cv2.countNonZero(black_mask) / area
    red_frac = cv2.countNonZero(red_mask) / area

    # No white box/object zone means empty
    if white_frac < MIN_WHITE_FRAC:
        return "E", white_frac, black_frac, red_frac

    # Black X target gets priority if strong
    if black_frac >= MIN_BLACK_FRAC and black_frac >= red_frac * 0.6:
        return "T", white_frac, black_frac, red_frac

    # Red X obstacle must be much stronger than black
    if red_frac >= MIN_RED_FRAC and red_frac > black_frac * 2.0:
        return "O", white_frac, black_frac, red_frac

    return "E", white_frac, black_frac, red_frac


def process_heading(heading):
    img_path = os.path.join(SCAN_DIR, f"{heading}.jpg")

    if not os.path.exists(img_path):
        print(f"[WARN] Missing {img_path}")
        return ["E", "E", "E"]

    img = cv2.imread(img_path)

    if img is None:
        print(f"[WARN] Could not read {img_path}")
        return ["E", "E", "E"]

    debug = img.copy()
    slots = get_front_slots(img)

    result = []

    for idx, (x1, y1, x2, y2) in enumerate(slots):
        slot = img[y1:y2, x1:x2]

        label, white_frac, black_frac, red_frac = classify_slot_front_row_only(slot)

        result.append(label)

        color = (0, 255, 0)
        if label == "T":
            color = (0, 0, 0)
        elif label == "O":
            color = (0, 0, 255)

        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            debug,
            f"{label} W:{white_frac:.2f} B:{black_frac:.2f} R:{red_frac:.2f}",
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


def strongest(a, b):
    if a == "O" or b == "O":
        return "O"
    if a == "T" or b == "T":
        return "T"
    return "E"


def build_object_3x3(front, right, back, left):
    grid = [
        ["E", "E", "E"],
        ["E", "A", "E"],
        ["E", "E", "E"]
    ]

    # Front scan fills top row
    grid[0][0] = front[0]
    grid[0][1] = front[1]
    grid[0][2] = front[2]

    # Right scan fills right column
    grid[0][2] = strongest(grid[0][2], right[0])
    grid[1][2] = right[1]
    grid[2][2] = right[2]

    # Back scan fills bottom row
    grid[2][2] = strongest(grid[2][2], back[0])
    grid[2][1] = back[1]
    grid[2][0] = back[2]

    # Left scan fills left column
    grid[2][0] = strongest(grid[2][0], left[0])
    grid[1][0] = left[1]
    grid[0][0] = strongest(grid[0][0], left[2])

    return grid


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
    print("[INFO] Object detection with front-row ROI + object-zone masking")

    results = {}

    for heading in HEADINGS:
        res = process_heading(heading)
        results[heading] = res
        print(f"{heading}: {res}")

    grid = build_object_3x3(
        results["front"],
        results["right"],
        results["back"],
        results["left"]
    )

    print("\nFinal object grid:")
    for row in grid:
        print(row)

    save_grid(grid)
    print(f"[OK] Debug images saved in: {DEBUG_DIR}")


if __name__ == "__main__":
    main()
