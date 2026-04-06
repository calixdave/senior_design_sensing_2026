import os

# =========================================================
# CONFIG
# =========================================================

COLOR_FILE = "results/local_color_3x3.txt"
OBJECT_FILE = "results/local_object_3x3.txt"

RESULTS_DIR = "results"
MAP_RESULT_FILE = os.path.join(RESULTS_DIR, "map_result.txt")
COMPACT_RESULT_FILE = os.path.join(RESULTS_DIR, "compact_map_result.txt")

# 6x6 unique test matrix
BIG_GRID = [
    ['G', 'R', 'P', 'Y', 'P', 'P'],
    ['P', 'Y', 'B', 'R', 'M', 'G'],
    ['P', 'P', 'Y', 'R', 'R', 'B'],
    ['M', 'G', 'G', 'M', 'Y', 'Y'],
    ['B', 'M', 'Y', 'M', 'M', 'B'],
    ['R', 'G', 'G', 'B', 'R', 'B'],
]

MIN_KNOWN_NEIGHBORS = 5
MAX_MISMATCHES = 4

# Scan settings
SCAN_START_LOCAL = "FRONT"
SCAN_SWEEP = "cw"
NUM_VIEWS = 4


# =========================================================
# HELPERS
# =========================================================

def read_local_3x3(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 3:
                raise ValueError(f"Each row must have 3 entries. Bad row: {line}")
            rows.append([p.upper() for p in parts])

    if len(rows) != 3:
        raise ValueError(f"Expected 3 rows in local 3x3 file, found {len(rows)}")

    return rows


def read_local_object_3x3(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 3:
                raise ValueError(f"Each row must have 3 entries. Bad row: {line}")
            rows.append([p.upper() for p in parts])

    if len(rows) != 3:
        raise ValueError(f"Expected 3 rows in local object 3x3 file, found {len(rows)}")

    return rows


def pretty_matrix(mat):
    return "\n".join(" ".join(row) for row in mat)


def rotate_3x3_ccw(mat):
    return [
        [mat[0][2], mat[1][2], mat[2][2]],
        [mat[0][1], mat[1][1], mat[2][1]],
        [mat[0][0], mat[1][0], mat[2][0]],
    ]


def rotate_n_ccw(mat, n):
    out = [row[:] for row in mat]
    for _ in range(n % 4):
        out = rotate_3x3_ccw(out)
    return out


def get_window_3x3(grid, center_r, center_c):
    rows = len(grid)
    cols = len(grid[0])

    if center_r - 1 < 0 or center_r + 1 >= rows:
        return None
    if center_c - 1 < 0 or center_c + 1 >= cols:
        return None

    return [
        [grid[center_r - 1][center_c - 1], grid[center_r - 1][center_c], grid[center_r - 1][center_c + 1]],
        [grid[center_r][center_c - 1],     'A',                          grid[center_r][center_c + 1]],
        [grid[center_r + 1][center_c - 1], grid[center_r + 1][center_c], grid[center_r + 1][center_c + 1]],
    ]


def score_match(local_3x3, window_3x3):
    known = 0
    matches = 0
    mismatches = 0

    for r in range(3):
        for c in range(3):
            lv = local_3x3[r][c]
            wv = window_3x3[r][c]

            if lv == 'A':
                continue
            if lv == '?':
                continue

            known += 1
            if lv == wv:
                matches += 1
            else:
                mismatches += 1

    return {
        "known": known,
        "matches": matches,
        "mismatches": mismatches,
        "score": matches
    }


def rotation_to_facing(rotation_ccw_deg):
    mapping = {
        0: "UP",
        90: "RIGHT",
        180: "DOWN",
        270: "LEFT",
    }
    return mapping[rotation_ccw_deg]


def rotate_direction(direction, steps_ccw):
    dirs = ["UP", "LEFT", "DOWN", "RIGHT"]
    idx = dirs.index(direction)
    return dirs[(idx + steps_ccw) % 4]


def get_scan_order(scan_start_local="FRONT", scan_sweep="cw", num_views=4):
    scan_start_local = scan_start_local.upper()
    scan_sweep = scan_sweep.lower()

    if scan_sweep == "cw":
        base_order = ["FRONT", "RIGHT", "BACK", "LEFT"]
    elif scan_sweep == "ccw":
        base_order = ["FRONT", "LEFT", "BACK", "RIGHT"]
    else:
        raise ValueError(f"scan_sweep must be 'cw' or 'ccw', got: {scan_sweep}")

    if scan_start_local not in base_order:
        raise ValueError(f"scan_start_local must be one of {base_order}, got: {scan_start_local}")

    start_idx = base_order.index(scan_start_local)
    ordered = base_order[start_idx:] + base_order[:start_idx]
    return ordered[:num_views]


def local_heading_to_map_direction(start_map_direction, local_heading):
    local_heading = local_heading.upper()

    local_steps_ccw = {
        "FRONT": 0,
        "LEFT": 1,
        "BACK": 2,
        "RIGHT": 3,
    }

    return rotate_direction(start_map_direction, local_steps_ccw[local_heading])


def get_final_camera_direction_after_scan(start_map_direction, scan_start_local="FRONT", scan_sweep="cw", num_views=4):
    order = get_scan_order(scan_start_local, scan_sweep, num_views)
    final_local_heading = order[-1]
    final_map_direction = local_heading_to_map_direction(start_map_direction, final_local_heading)
    return final_local_heading, final_map_direction


def direction_to_char(direction):
    mapping = {
        "UP": "U",
        "RIGHT": "R",
        "DOWN": "D",
        "LEFT": "L",
    }
    return mapping[direction]


def build_compact_17char(color_3x3, object_3x3, final_direction):
    out = []

    for r in range(3):
        for c in range(3):
            if r == 1 and c == 1:
                continue

            color_char = str(color_3x3[r][c]).strip().upper()[:1]
            obj_char = str(object_3x3[r][c]).strip().upper()[:1]

            out.append(color_char + obj_char)

    out.append(direction_to_char(final_direction))
    return "".join(out)


# =========================================================
# MAIN MATCHING
# =========================================================

def find_best_match(local_3x3, big_grid):
    rows = len(big_grid)
    cols = len(big_grid[0])

    candidates = []

    for rot_steps in range(4):
        rotated = rotate_n_ccw(local_3x3, rot_steps)
        rotation_ccw_deg = rot_steps * 90

        for center_r in range(1, rows - 1):
            for center_c in range(1, cols - 1):
                window = get_window_3x3(big_grid, center_r, center_c)
                if window is None:
                    continue

                s = score_match(rotated, window)

                if s["known"] < MIN_KNOWN_NEIGHBORS:
                    continue
                if s["mismatches"] > MAX_MISMATCHES:
                    continue

                candidates.append({
                    "center_row": center_r,
                    "center_col": center_c,
                    "rotation_ccw_deg": rotation_ccw_deg,
                    "facing": rotation_to_facing(rotation_ccw_deg),
                    "known": s["known"],
                    "matches": s["matches"],
                    "mismatches": s["mismatches"],
                    "score": s["score"],
                    "rotated_local": rotated,
                    "window": window,
                })

    if not candidates:
        return None, []

    candidates.sort(
        key=lambda x: (x["score"], -x["mismatches"], x["known"]),
        reverse=True
    )

    best = candidates[0]
    return best, candidates


# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    local_color_3x3 = read_local_3x3(COLOR_FILE)
    local_object_3x3 = read_local_object_3x3(OBJECT_FILE)

    print("\nLOCAL COLOR 3x3:")
    print(pretty_matrix(local_color_3x3))

    print("\nLOCAL OBJECT 3x3:")
    print(pretty_matrix(local_object_3x3))

    best, candidates = find_best_match(local_color_3x3, BIG_GRID)

    if best is None:
        msg = (
            "No valid match found.\n"
            "Try:\n"
            "- lowering MIN_KNOWN_NEIGHBORS\n"
            "- increasing MAX_MISMATCHES\n"
            "- improving color detection\n"
        )
        print("\n" + msg)

        with open(MAP_RESULT_FILE, "w") as f:
            f.write(msg)

        print(f"Saved: {MAP_RESULT_FILE}")
        return

    camera_direction_before_scan = best["facing"]

    final_local_heading_after_scan, camera_direction_after_scan = get_final_camera_direction_after_scan(
        start_map_direction=camera_direction_before_scan,
        scan_start_local=SCAN_START_LOCAL,
        scan_sweep=SCAN_SWEEP,
        num_views=NUM_VIEWS
    )

    compact_17char = build_compact_17char(
        best["window"],
        local_object_3x3,
        camera_direction_after_scan
    )

    print("\nBEST MATCH FOUND")
    print("-------------------------")
    print(f"center_row                     = {best['center_row']}")
    print(f"center_col                     = {best['center_col']}")
    print(f"rotation_ccw_deg               = {best['rotation_ccw_deg']}")
    print(f"facing_in_big_grid             = {best['facing']}")
    print(f"camera_direction_before_scan   = {camera_direction_before_scan}")
    print(f"scan_start_local               = {SCAN_START_LOCAL}")
    print(f"scan_sweep                     = {SCAN_SWEEP}")
    print(f"final_local_heading_after_scan = {final_local_heading_after_scan}")
    print(f"camera_direction_after_scan    = {camera_direction_after_scan}")
    print(f"compact_17char                 = {compact_17char}")
    print("\nMATCHED WINDOW USED FOR COMPACT COLOR:")
    print(pretty_matrix(best["window"]))
    print(f"known_neighbors                = {best['known']}")
    print(f"matches                        = {best['matches']}")
    print(f"mismatches                     = {best['mismatches']}")
    print(f"score                          = {best['score']}/{best['known']}")

    print("\nROTATED LOCAL 3x3 USED FOR MATCH:")
    print(pretty_matrix(best["rotated_local"]))

    print("\nMATCHED WINDOW IN BIG_GRID:")
    print(pretty_matrix(best["window"]))

    print(f"\nTotal valid candidates: {len(candidates)}")

    if len(candidates) > 1:
        print("\nTop candidates:")
        for i, c in enumerate(candidates[:5], start=1):
            print(
                f"{i}. center=({c['center_row']},{c['center_col']}), "
                f"rot={c['rotation_ccw_deg']} deg, "
                f"facing={c['facing']}, "
                f"score={c['score']}/{c['known']}, "
                f"mismatches={c['mismatches']}"
            )

    result_lines = [
        "BEST MATCH FOUND",
        "-------------------------",
        f"center_row                     = {best['center_row']}",
        f"center_col                     = {best['center_col']}",
        f"rotation_ccw_deg               = {best['rotation_ccw_deg']}",
        f"facing_in_big_grid             = {best['facing']}",
        f"camera_direction_before_scan   = {camera_direction_before_scan}",
        f"scan_start_local               = {SCAN_START_LOCAL}",
        f"scan_sweep                     = {SCAN_SWEEP}",
        f"final_local_heading_after_scan = {final_local_heading_after_scan}",
        f"camera_direction_after_scan    = {camera_direction_after_scan}",
        f"compact_17char                 = {compact_17char}",
        f"known_neighbors                = {best['known']}",
        f"matches                        = {best['matches']}",
        f"mismatches                     = {best['mismatches']}",
        f"score                          = {best['score']}/{best['known']}",
        "",
        "LOCAL COLOR 3x3:",
        pretty_matrix(local_color_3x3),
        "",
        "LOCAL OBJECT 3x3:",
        pretty_matrix(local_object_3x3),
        "",
        "ROTATED LOCAL 3x3 USED FOR MATCH:",
        pretty_matrix(best["rotated_local"]),
        "",
        "MATCHED WINDOW IN BIG_GRID:",
        pretty_matrix(best["window"]),
        "",
        f"Total valid candidates: {len(candidates)}"
    ]

    if len(candidates) > 1:
        result_lines.append("")
        result_lines.append("Top candidates:")
        for i, c in enumerate(candidates[:5], start=1):
            result_lines.append(
                f"{i}. center=({c['center_row']},{c['center_col']}), "
                f"rot={c['rotation_ccw_deg']} deg, "
                f"facing={c['facing']}, "
                f"score={c['score']}/{c['known']}, "
                f"mismatches={c['mismatches']}"
            )

    with open(MAP_RESULT_FILE, "w") as f:
        f.write("\n".join(result_lines) + "\n")

    with open(COMPACT_RESULT_FILE, "w") as f:
        f.write(compact_17char + "\n")

    print(f"\nSaved: {MAP_RESULT_FILE}")
    print(f"Saved: {COMPACT_RESULT_FILE}")


if __name__ == "__main__":
    main()
