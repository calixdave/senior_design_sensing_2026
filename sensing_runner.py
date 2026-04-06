import os
import sys
import subprocess

# =========================================================
# CONFIG
# =========================================================

SCAN_SCRIPT = "capture_scan.py"      # change if your scan script has a different name
COLOR_SCRIPT = "detect_colors.py"
OBJECT_SCRIPT = "detect_objects.py"
MAP_SCRIPT = "map_location.py"

PYTHON_EXE = sys.executable

# optional output files to display at the end
COLOR_MATRIX_FILE = "results/local_color_3x3.txt"
OBJECT_MATRIX_FILE = "results/local_object_3x3.txt"   # only if your object script saves it
MAP_OUTPUT_FILE = "results/map_result.txt"            # only if your map script saves it


# =========================================================
# HELPERS
# =========================================================

def run_script(script_name, title):
    print("\n" + "=" * 70)
    print(f"{title}: {script_name}")
    print("=" * 70)

    if not os.path.exists(script_name):
        print(f"ERROR: File not found: {script_name}")
        return False

    result = subprocess.run([PYTHON_EXE, script_name])

    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed with code {result.returncode}")
        return False

    print(f"\nDONE: {script_name}")
    return True


def show_file(path, label):
    if os.path.exists(path):
        print("\n" + "-" * 70)
        print(label)
        print("-" * 70)
        with open(path, "r") as f:
            print(f.read().strip())
    else:
        print(f"\n[INFO] {label} file not found: {path}")


# =========================================================
# MAIN
# =========================================================

def main():
    pipeline = [
        (SCAN_SCRIPT, "STEP 1 - SCANNING"),
        (COLOR_SCRIPT, "STEP 2 - COLOR DETECTION"),
        (OBJECT_SCRIPT, "STEP 3 - OBJECT DETECTION"),
        (MAP_SCRIPT, "STEP 4 - LOCATION MAPPING"),
    ]

    print("\nStarting full pipeline...\n")

    for script_name, title in pipeline:
        ok = run_script(script_name, title)
        if not ok:
            print("\nPipeline stopped.")
            return

    print("\n" + "=" * 70)
    print("PIPELINE FINISHED")
    print("=" * 70)

    # show final saved outputs if they exist
    show_file(COLOR_MATRIX_FILE, "FINAL COLOR 3x3")
    show_file(OBJECT_MATRIX_FILE, "FINAL OBJECT 3x3")
    show_file(MAP_OUTPUT_FILE, "FINAL MAP RESULT")


if __name__ == "__main__":
    main()
