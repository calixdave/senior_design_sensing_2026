import os
import json
import cv2
import numpy as np

# =========================================================
# CONFIG
# =========================================================

SCAN_DIR = "scan_images"
DEBUG_DIR = "debug_objects"
RESULTS_DIR = "results"

HEADINGS = ["front", "right", "back", "left"]

# ---------------------------------------------------------
# Parameters copied from your live tuner baseline
# ---------------------------------------------------------

ROI_TOP_FRAC = 0.34
ROI_BOT_FRAC = 0.94

SLOT_PAD_X_FRAC = 0.03
SLOT_PAD_Y_FRAC = 0.06

WHITE_MIN = 126
BLACK_MAX = 95
RED_S_MIN = 95
RED_V_MIN = 120

WHITE_RATIO_TH = 0.35
RED_RATIO_TH = 0.18
BLACK_RATIO_TH = 0.14

EMPTY_WHITE_TH = 0.80
EMPTY_RED_TH = 0.80
EMPTY_BLACK_TH = 0.90

CANNY1 = 40
CANNY2 = 120
HOUGH_TH = 30
MIN_LINE = 8
MAX_GAP = 40

BLUR_ODD = 1

# Object meaning:
# O = obstacle   (white box with thick red X)
# T = target     (white box with thick black X)
# E = empty
# ? = unknown

HEADING_TO_POSITIONS = {
    "front": [(-1, +1), (0, +1), (+1, +1)],
    "right": [(+1, +1), (+1, 0), (+1, -1)],
    "back":  [(+1, -1), (0, -1), (-1, -1)],
    "left":  [(-1, -1), (-1, 0), (-1, +1)],
}


# =========================================================
# ROI / SLOT HELPERS
# =========================================================

def get_three_slot_rois(img):
    h, w = img.shape[:2]

    y0 = int(ROI_TOP_FRAC * h)
    y1 = int(ROI_BOT_FRAC * h)

    if y1 <= y0:
        return []

    band = img[y0:y1, :]
    bh, bw = band.shape[:2]

    slots = []

    for i in range(3):
        sx0 = int(i * bw / 3)
        sx1 = int((i + 1) * bw / 3)

        pad_x = int(SLOT_PAD_X_FRAC * (sx1 - sx0))
        pad_y = int(SLOT_PAD_Y_FRAC * bh)

        cx0 = max(0, sx0 + pad_x)
        cx1 = min(bw, sx1 - pad_x)
        cy0 = max(0, pad_y)
        cy1 = min(bh, bh - pad_y)

        crop = band[cy0:cy1, cx0:cx1]
        slots.append(crop)

    return slots


# =========================================================
# OBJECT DETECTION
# =========================================================

def detect_one_object_slot(slot_bgr):
    if slot_bgr is None or slot_bgr.size == 0:
        return "?", {}

    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2GRAY)

    if BLUR_ODD > 1:
        k = BLUR_ODD if BLUR_ODD % 2 == 1 else BLUR_ODD + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # -------------------------------
    # red X detection
    # -------------------------------
    lower_red1 = np.array([0, RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red1, red2)
    red_ratio = float(np.count_nonzero(red_mask)) / red_mask.size

    # -------------------------------
    # white object area
    # -------------------------------
    white_mask = cv2.inRange(gray, WHITE_MIN, 255)
    white_ratio = float(np.count_nonzero(white_mask)) / white_mask.size

    # -------------------------------
    # black X detection
    # -------------------------------
    black_mask = cv2.inRange(gray, 0, BLACK_MAX)
    black_ratio = float(np.count_nonzero(black_mask)) / black_mask.size

    # -------------------------------
    # diagonal line detection
    # -------------------------------
    edges = cv2.Canny(gray, CANNY1, CANNY2)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(1, HOUGH_TH),
        minLineLength=max(1, MIN_LINE),
        maxLineGap=max(0, MAX_GAP)
    )

    diag_pos = 0
    diag_neg = 0

    if lines is not None:
        for ln in lines[:, 0]:
            x1, y1, x2, y2 = ln
            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                continue

            ang = np.degrees(np.arctan2(dy, dx))

            if 25 <= ang <= 65:
                diag_pos += 1
            elif -65 <= ang <= -25:
                diag_neg += 1

    has_x_shape = (diag_pos >= 1 and diag_neg >= 1)

    metrics = {
        "red_ratio": round(red_ratio, 4),
        "white_ratio": round(white_ratio, 4),
        "black_ratio": round(black_ratio, 4),
        "diag_pos": int(diag_pos),
        "diag_neg": int(diag_neg),
        "has_x_shape": bool(has_x_shape),
    }

    # obstacle = white box + red X
    if white_ratio > WHITE_RATIO_TH and red_ratio > RED_RATIO_TH and has_x_shape:
        return "O", metrics

    # target = white box + black X
    if white_ratio > WHITE_RATIO_TH and black_ratio > BLACK_RATIO_TH and has_x_shape:
        return "T", metrics

    # empty
    if white_ratio < EMPTY_WHITE_TH and red_ratio < EMPTY_RED_TH and black_ratio < EMPTY_BLACK_TH:
        return "E", metrics

    return "?", metrics


# =========================================================
# MATRIX HELPERS
# =========================================================

def matrix_rows_from_grid(final_grid):
    rows = []
    for row in [1, 0, -1]:
        vals = []
        for col in [-1, 0, 1]:
            vals.append(final_grid.get((col, row), "?"))
        rows.append(vals)
    return rows


def pretty_print_matrix(final_grid):
    rows = matrix_rows_from_grid(final_grid)
    for row in rows:
        print(" ".join(row))


def save_matrix_txt(path, final_grid):
    rows = matrix_rows_from_grid(final_grid)
    with open(path, "w") as f:
        for row in rows:
            f.write(" ".join(row) + "\n")


# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    final_grid = {
        (-1, +1): "?",
        ( 0, +1): "?",
        (+1, +1): "?",
        (-1,  0): "?",
        ( 0,  0): "A",
        (+1,  0): "?",
        (-1, -1): "?",
        ( 0, -1): "?",
        (+1, -1): "?",
    }

    detailed = {}

    for heading in HEADINGS:
        path = os.path.join(SCAN_DIR, f"{heading}.jpg")

        if not os.path.exists(path):
            print(f"ERROR: Missing image: {path}")
            return

        img = cv2.imread(path)
        if img is None:
            print(f"ERROR: Could not read image: {path}")
            return

        slots = get_three_slot_rois(img)
        if len(slots) != 3:
            print(f"ERROR: Could not build 3 slots for heading: {heading}")
            return

        heading_info = []
        print(f"\nHeading: {heading}")

        for i, tile in enumerate(slots):
            dbg_name = os.path.join(DEBUG_DIR, f"{heading}_slot{i}.jpg")
            cv2.imwrite(dbg_name, tile)

            obj_char, metrics = detect_one_object_slot(tile)
            pos = HEADING_TO_POSITIONS[heading][i]
            final_grid[pos] = obj_char

            print(f"  slot {i}: object={obj_char}, saved={dbg_name}")
            print(f"    metrics: {metrics}")

            heading_info.append({
                "slot_index": i,
                "pos": [pos[0], pos[1]],
                "object": obj_char,
                "debug_crop": dbg_name,
                "metrics": metrics
            })

        detailed[heading] = heading_info

    print("\nFinal 3x3 object matrix:")
    pretty_print_matrix(final_grid)

    out = {
        "center": [0, 0],
        "agent": "A",
        "grid_objects": {
            f"{c},{r}": final_grid[(c, r)]
            for (c, r) in final_grid
        },
        "per_heading": detailed
    }

    json_path = os.path.join(RESULTS_DIR, "object_results.json")
    txt_path = os.path.join(RESULTS_DIR, "local_object_3x3.txt")

    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    save_matrix_txt(txt_path, final_grid)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved debug crops in: {DEBUG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
