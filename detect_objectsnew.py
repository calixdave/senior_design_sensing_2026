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

# ROI for front slots
ROI_TOP_FRAC = 0.34
ROI_BOT_FRAC = 0.94

SLOT_PAD_X_FRAC = 0.03
SLOT_PAD_Y_FRAC = 0.06

# Only analyze the front-most part of each slot
FRONT_ROW_TOP_FRAC = 0.45

# =========================================================
# HSV THRESHOLDS (tuned for your setup)
# =========================================================

WHITE_S_MAX = 140
WHITE_V_MIN = 110

BLACK_V_MAX = 95

RED_S_MIN = 80
RED_V_MIN = 80

# Detection thresholds
MIN_WHITE_FRAC = 0.035
MIN_BLACK_FRAC = 0.020
MIN_RED_FRAC = 0.020

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


# =========================================================
# SLOT EXTRACTION
# =========================================================

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


# =========================================================
# CORE DETECTION (FRONT ROW ONLY)
# =========================================================

def classify_slot_front_row_only(slot):
    h, w = slot.shape[:2]

    # Only look at front-most area
    y_start = int(h * FRONT_ROW_TOP_FRAC)
    crop = slot[y_start:h, :]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # White box
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, WHITE_V_MIN]),
        np.array([179, WHITE_S_MAX, 255])
    )

    # Black X
    black_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 0]),
        np.array([179, 255, BLACK_V_MAX])
    )

    # Red X
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

    area = crop.shape[0] * crop.shape[1]

    white_frac = cv2.countNonZero(white_mask) / area
    black_frac = cv2.countNonZero(black_mask) / area
    red_frac = cv2.countNonZero(red_mask) / area

    # --- Decision ---
    if white_frac < MIN_WHITE_FRAC:
        return "E", white_frac, black_frac, red_frac

    if red_frac >= MIN_RED_FRAC and red_frac > black_frac:
        return "O", white_frac, black_frac, red_frac

    if black_frac >= MIN_BLACK_FRAC:
        return "T", white_frac, black_frac, red_frac

    return "E", white_frac, black_frac, red_frac


# =========================================================
# PROCESS EACH HEADING
# =========================================================

def process_heading(heading):
    img_path = os.path.join(SCAN_DIR, f"{heading}.jpg")

    if not os.path.exists(img_path):
        print(f"[WARN] Missing {img_path}")
        return ["E", "E", "E"]

    img = cv2.imread(img_path)
    debug = img.copy()

    slots = get_front_slots(img)
    result = []

    for (x1, y1, x2, y2) in slots:
        slot = img[y1:y2, x1:x2]

        label, white_frac, black_frac, red_frac = classify_slot_front_row_only(slot)

        result.append(label)

        # Color for debug
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


# =========================================================
# BUILD 3x3 GRID
# =========================================================

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

    grid[0] = front

    grid[0][2] = strongest(grid[0][2], right[0])
    grid[1][2] = right[1]
    grid[2][2] = right[2]

    grid[2][2] = strongest(grid[2][2], back[0])
    grid[2][1] = back[1]
    grid[2][0] = back[2]

    grid[2][0] = strongest(grid[2][0], left[0])
    grid[1][0] = left[1]
    grid[0][0] = strongest(grid[0][0], left[2])

    return grid


# =========================================================
# SAVE
# =========================================================

def save_grid(grid):
    txt_path = os.path.join(RESULTS_DIR, "object_3x3.txt")

    with open(txt_path, "w") as f:
        for row in grid:
            f.write(" ".join(row) + "\n")

    print(f"[OK] Saved: {txt_path}")


# =========================================================
# MAIN
# =========================================================

def main():
    print("[INFO] Object detection (front-row mode)")

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


if __name__ == "__main__":
    main()
