import os
import json
import cv2
import numpy as np

# =========================================================
# detect_objects.py
#
# Detects objects only in the FRONT ROW of each scan image.
#
# Target   = white box + black X
# Obstacle = white box + red X
# Empty    = no valid white box
#
# Main method:
#   1. Split front-row ROI into 3 slots.
#   2. Find white mask.
#   3. Keep largest white blob.
#   4. Validate blob:
#        - large enough
#        - dominant enough
#        - not touching slot border
#   5. Dilate blob to object zone.
#   6. Count red/black pixels only inside that zone.
# =========================================================

SCAN_DIR = "scan_images"
DEBUG_DIR = "debug_objects"
RESULTS_DIR = "results"

HEADINGS = ["front", "right", "back", "left"]

os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================================
# FRONT-ROW ROI SETTINGS
# =========================================================

ROI_TOP_FRAC = 0.52
ROI_BOT_FRAC = 0.94

SLOT_PAD_X_FRAC = 0.03
SLOT_PAD_Y_FRAC = 0.05

# =========================================================
# WHITE BOX SETTINGS
# =========================================================

WHITE_S_MAX = 90
WHITE_V_MIN = 145

MIN_WHITE_FRAC = 0.08

# Real object box should create one strong blob
MIN_BLOB_FRAC = 0.12
MAX_BLOB_FRAC = 0.70

# Largest blob must contain most of the white pixels
MIN_BLOB_DOMINANCE = 0.60

# Reject blob if it touches the slot border
REJECT_BORDER_TOUCHING_BLOB = True

# =========================================================
# RED / BLACK X SETTINGS
# =========================================================

RED_S_MIN = 140
RED_V_MIN = 80

BLACK_V_MAX = 85
BLACK_S_MAX = 180

MIN_RED_FRAC_IN_ZONE = 0.030
MIN_BLACK_FRAC_IN_ZONE = 0.035

RED_DOMINANCE_RATIO = 1.20
BLACK_DOMINANCE_RATIO = 1.10

# =========================================================
# DILATED OBJECT ZONE SETTINGS
# =========================================================

DILATE_KERNEL_SIZE = 13
DILATE_ITERATIONS = 2

# =========================================================
# DRAW COLORS
# =========================================================

COLOR_EMPTY = (0, 255, 0)
COLOR_TARGET = (0, 0, 0)
COLOR_OBSTACLE = (0, 0, 255)
COLOR_SLOT = (0, 255, 0)


def front_row_slots(img):
    h, w = img.shape[:2]

    roi_y1 = int(h * ROI_TOP_FRAC)
    roi_y2 = int(h * ROI_BOT_FRAC)

    roi_h = roi_y2 - roi_y1
    slot_w = w // 3

    pad_x = int(w * SLOT_PAD_X_FRAC)
    pad_y = int(roi_h * SLOT_PAD_Y_FRAC)

    slots = []

    for i in range(3):
        x1 = i * slot_w
        x2 = (i + 1) * slot_w if i < 2 else w

        sx1 = max(0, x1 + pad_x)
        sx2 = min(w, x2 - pad_x)

        sy1 = max(0, roi_y1 + pad_y)
        sy2 = min(h, roi_y2 - pad_y)

        slots.append((sx1, sy1, sx2, sy2))

    return slots


def make_white_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    white = ((s <= WHITE_S_MAX) & (v >= WHITE_V_MIN)).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)

    return white


def largest_white_blob_mask(white_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        white_mask,
        connectivity=8
    )

    if num_labels <= 1:
        return None, 0

    largest_label = None
    largest_area = 0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area > largest_area:
            largest_area = area
            largest_label = label

    if largest_label is None:
        return None, 0

    blob = np.zeros_like(white_mask)
    blob[labels == largest_label] = 255

    return blob, largest_area


def blob_touches_border(blob_mask):
    if blob_mask is None:
        return False

    top = np.any(blob_mask[0, :] > 0)
    bottom = np.any(blob_mask[-1, :] > 0)
    left = np.any(blob_mask[:, 0] > 0)
    right = np.any(blob_mask[:, -1] > 0)

    return top or bottom or left or right


def dilate_blob(blob_mask):
    kernel = np.ones((DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), np.uint8)
    return cv2.dilate(blob_mask, kernel, iterations=DILATE_ITERATIONS)


def make_red_black_masks(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    red1 = h <= 10
    red2 = h >= 170

    red = ((red1 | red2) & (s >= RED_S_MIN) & (v >= RED_V_MIN)).astype(np.uint8) * 255

    black = ((v <= BLACK_V_MAX) & (s <= BLACK_S_MAX)).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)

    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)

    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, kernel)
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel)

    return red, black


def classify_slot(slot_bgr):
    slot_h, slot_w = slot_bgr.shape[:2]
    slot_area = slot_h * slot_w

    white_mask = make_white_mask(slot_bgr)
    total_white_pixels = cv2.countNonZero(white_mask)
    white_frac_total = total_white_pixels / float(slot_area)

    blob_mask, blob_area = largest_white_blob_mask(white_mask)

    info = {
        "white_frac_total": float(white_frac_total),
        "largest_blob_frac": 0.0,
        "blob_dominance": 0.0,
        "blob_touches_border": False,
        "red_frac_in_zone": 0.0,
        "black_frac_in_zone": 0.0,
        "red_pixels": 0,
        "black_pixels": 0,
        "zone_pixels": 0,
        "label": "E"
    }

    masks = {
        "white_mask": white_mask,
        "blob_mask": np.zeros_like(white_mask),
        "zone_mask": np.zeros_like(white_mask),
        "red_in_zone": np.zeros_like(white_mask),
        "black_in_zone": np.zeros_like(white_mask)
    }

    if blob_mask is None or total_white_pixels <= 0:
        return "E", info, masks

    blob_frac = blob_area / float(slot_area)
    blob_dominance = blob_area / float(total_white_pixels)
    touches_border = blob_touches_border(blob_mask)

    info["largest_blob_frac"] = float(blob_frac)
    info["blob_dominance"] = float(blob_dominance)
    info["blob_touches_border"] = bool(touches_border)

    masks["blob_mask"] = blob_mask

    # -----------------------------------------------------
    # Strict white-box validation
    # -----------------------------------------------------

    if white_frac_total < MIN_WHITE_FRAC:
        return "E", info, masks

    if blob_frac < MIN_BLOB_FRAC:
        return "E", info, masks

    if blob_frac > MAX_BLOB_FRAC:
        return "E", info, masks

    if blob_dominance < MIN_BLOB_DOMINANCE:
        return "E", info, masks

    if REJECT_BORDER_TOUCHING_BLOB and touches_border:
        return "E", info, masks

    # -----------------------------------------------------
    # Only now create object zone and inspect X color
    # -----------------------------------------------------

    zone_mask = dilate_blob(blob_mask)
    masks["zone_mask"] = zone_mask

    zone_pixels = cv2.countNonZero(zone_mask)
    info["zone_pixels"] = int(zone_pixels)

    if zone_pixels <= 0:
        return "E", info, masks

    red_mask, black_mask = make_red_black_masks(slot_bgr)

    red_in_zone = cv2.bitwise_and(red_mask, zone_mask)
    black_in_zone = cv2.bitwise_and(black_mask, zone_mask)

    red_pixels = cv2.countNonZero(red_in_zone)
    black_pixels = cv2.countNonZero(black_in_zone)

    red_frac = red_pixels / float(zone_pixels)
    black_frac = black_pixels / float(zone_pixels)

    info["red_pixels"] = int(red_pixels)
    info["black_pixels"] = int(black_pixels)
    info["red_frac_in_zone"] = float(red_frac)
    info["black_frac_in_zone"] = float(black_frac)

    masks["red_in_zone"] = red_in_zone
    masks["black_in_zone"] = black_in_zone

    label = "E"

    if red_frac >= MIN_RED_FRAC_IN_ZONE and red_frac > black_frac * RED_DOMINANCE_RATIO:
        label = "O"

    elif black_frac >= MIN_BLACK_FRAC_IN_ZONE and black_frac > red_frac * BLACK_DOMINANCE_RATIO:
        label = "T"

    info["label"] = label

    return label, info, masks


def make_debug_canvas(img, slots, labels, infos):
    debug = img.copy()

    for i, (x1, y1, x2, y2) in enumerate(slots):
        label = labels[i]
        info = infos[i]

        if label == "O":
            color = COLOR_OBSTACLE
            text = "O obstacle"
        elif label == "T":
            color = COLOR_TARGET
            text = "T target"
        else:
            color = COLOR_EMPTY
            text = "E empty"

        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            debug,
            text,
            (x1 + 8, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA
        )

        line1 = (
            f"W={info['white_frac_total']:.2f} "
            f"B={info['largest_blob_frac']:.2f} "
            f"D={info['blob_dominance']:.2f}"
        )

        line2 = (
            f"R={info['red_frac_in_zone']:.3f} "
            f"K={info['black_frac_in_zone']:.3f} "
            f"BT={int(info['blob_touches_border'])}"
        )

        cv2.putText(
            debug,
            line1,
            (x1 + 8, y1 + 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            debug,
            line2,
            (x1 + 8, y1 + 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

    return debug


def save_mask(path, mask):
    cv2.imwrite(path, mask)


def process_heading(heading):
    jpg_path = os.path.join(SCAN_DIR, f"{heading}.jpg")
    png_path = os.path.join(SCAN_DIR, f"{heading}.png")

    if os.path.exists(jpg_path):
        img_path = jpg_path
    elif os.path.exists(png_path):
        img_path = png_path
    else:
        print(f"[WARN] Missing image for {heading}")
        return ["E", "E", "E"], []

    img = cv2.imread(img_path)

    if img is None:
        print(f"[WARN] Could not read image: {img_path}")
        return ["E", "E", "E"], []

    slots = front_row_slots(img)

    labels = []
    infos = []

    for idx, (x1, y1, x2, y2) in enumerate(slots):
        slot = img[y1:y2, x1:x2]

        label, info, masks = classify_slot(slot)

        labels.append(label)
        infos.append(info)

        prefix = os.path.join(DEBUG_DIR, f"{heading}_slot{idx}")

        save_mask(prefix + "_white_mask.png", masks["white_mask"])
        save_mask(prefix + "_largest_white_blob.png", masks["blob_mask"])
        save_mask(prefix + "_dilated_object_zone.png", masks["zone_mask"])
        save_mask(prefix + "_red_inside_zone.png", masks["red_in_zone"])
        save_mask(prefix + "_black_inside_zone.png", masks["black_in_zone"])

    debug_img = make_debug_canvas(img, slots, labels, infos)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{heading}_debug_objects.png"), debug_img)

    return labels, infos


def main():
    all_results = {}
    all_debug = {}

    for heading in HEADINGS:
        labels, infos = process_heading(heading)

        all_results[heading] = labels
        all_debug[heading] = infos

        print(f"{heading}: {labels}")

    json_path = os.path.join(RESULTS_DIR, "object_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)

    debug_json_path = os.path.join(RESULTS_DIR, "object_debug_values.json")
    with open(debug_json_path, "w") as f:
        json.dump(all_debug, f, indent=4)

    txt_path = os.path.join(RESULTS_DIR, "object_results.txt")
    with open(txt_path, "w") as f:
        for heading in HEADINGS:
            row = all_results.get(heading, ["E", "E", "E"])
            f.write(f"{heading}: {' '.join(row)}\n")

    print("\nSaved:")
    print(f"  {json_path}")
    print(f"  {debug_json_path}")
    print(f"  {txt_path}")
    print(f"  {DEBUG_DIR}/")


if __name__ == "__main__":
    main()
