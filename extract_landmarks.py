"""
extract_landmarks.py - turn a folder of labelled chord images into a feature table.

CRISP-DM stage: data preparation.

This is the bridge between images and classical machine learning. For every
image it runs MediaPipe Hands in static_image_mode, converts the 21 landmarks
into the 85 engineered features from src/features.py, and appends one row to a
CSV. train.py then reads only that CSV - it never touches the images again,
which is what keeps this a landmark-feature project rather than a pixel one.

Expected input layout (one subfolder per chord):
    <root>/A/*.jpg   <root>/B/*.jpg   ...   <root>/G/*.jpg

Usage
-----
    # the public dataset
    python extract_landmarks.py --root data/images --source public

    # my own webcam captures, appended to the same CSV
    python extract_landmarks.py --root data/recordings --source own

    # explicit output path
    python extract_landmarks.py --root data/images --source public --out data/landmarks.csv

Output columns:
    filepath, label, source, group, <85 feature columns>

The "group" column and why it matters
-------------------------------------
Rows that come from almost the same photograph must never be split across
training and test sets: the model would then be tested on data it has
effectively already seen, and the reported accuracy would be fiction. The
group column labels which rows are "the same underlying shot", so train.py can
keep whole groups together with GroupShuffleSplit / GroupKFold.

    own recordings : group = the session id from the filename
                     (<label>_<sessionid>_<timestamp>.jpg), because frames from
                     one sitting are near-duplicates of each other.
    public images   : group = the original filename with any augmentation
                     suffix stripped. Public datasets (Roboflow in particular)
                     ship several augmented copies of one photo, named like
                     C_photo12_jpg.rf.4f1c....jpg - everything from ".rf." on is
                     the augmentation id, so stripping it makes all copies of
                     one photo share a group.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp

from src import config
from src.features import FEATURE_NAMES, extract_features, landmarks_to_array

mp_hands = mp.solutions.hands

CSV_HEADER = ["filepath", "label", "source", "group"] + FEATURE_NAMES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand-landmark features from labelled chord images."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="folder containing one subfolder per chord label (A..G)",
    )
    parser.add_argument(
        "--source",
        required=True,
        help='name recorded in the "source" column, e.g. public or own',
    )
    parser.add_argument(
        "--out",
        default=str(config.LANDMARKS_CSV),
        help="output CSV path, appended to if it exists (default: %(default)s)",
    )
    return parser.parse_args()


def resolve_path(value):
    """Interpret a command-line path as absolute, or relative to the project root."""
    given = Path(value)
    return given.resolve() if given.is_absolute() else (config.PROJECT_ROOT / given).resolve()


def group_for(path, source):
    """
    Decide which group an image belongs to. See the module docstring.

    Own recordings are named <label>_<sessionid>_<timestamp>.jpg, so the session
    id is the second underscore-separated field. Public images use the original
    photo name with the augmentation suffix removed.
    """
    stem = path.stem

    if source == "own":
        parts = stem.split("_")
        if len(parts) >= 3:
            return parts[1]  # the session id
        # Fall back to the whole stem rather than guessing, so nothing is
        # silently grouped together that should not be.
        return stem

    # Public images: drop the Roboflow-style augmentation suffix.
    if ".rf." in path.name:
        return path.name.split(".rf.")[0]
    return stem


def iter_images(root):
    """Yield (image_path, label) for every image in <root>/<label>/."""
    for label in config.CHORD_LABELS:
        folder = root / label
        if not folder.exists():
            print(f"  note: {folder} does not exist, skipping label {label}")
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS:
                yield path, label


def main():
    args = parse_args()

    # Accept both absolute paths and paths relative to the project root, so the
    # usage examples above work from any working directory.
    root = resolve_path(args.root)
    if not root.exists():
        sys.exit(f"Root folder does not exist: {root}")

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Append mode: the public dataset and my own recordings land in one CSV, so
    # the header is only written when the file is new.
    write_header = not out_path.exists() or out_path.stat().st_size == 0

    detected = {label: 0 for label in config.CHORD_LABELS}
    failed = {label: 0 for label in config.CHORD_LABELS}
    failed_paths = []

    print(f"root   : {root}")
    print(f"source : {args.source}")
    print(f"output : {out_path} ({'new file' if write_header else 'appending'})")
    print()

    with open(out_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(CSV_HEADER)

        # static_image_mode: run the palm detector on every image instead of
        # tracking from a previous frame, which is correct for unrelated stills.
        with mp_hands.Hands(**config.STATIC_MODE_SETTINGS) as hands:
            for image_path, label in iter_images(root):
                image = cv2.imread(str(image_path))
                if image is None:
                    failed[label] += 1
                    failed_paths.append((image_path, "unreadable file"))
                    continue

                height, width = image.shape[:2]
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if not results.multi_hand_landmarks:
                    # No hand found. Common for B and F, where the barre hides
                    # the knuckles. Logged rather than silently dropped.
                    failed[label] += 1
                    failed_paths.append((image_path, "no hand detected"))
                    continue

                # max_num_hands is 1, so there is exactly one entry here.
                landmarks = landmarks_to_array(results.multi_hand_landmarks[0])
                features = extract_features(landmarks, aspect_ratio=width / height)

                relative = image_path.relative_to(config.PROJECT_ROOT) \
                    if image_path.is_relative_to(config.PROJECT_ROOT) else image_path
                writer.writerow(
                    [relative.as_posix(), label, args.source, group_for(image_path, args.source)]
                    + [f"{value:.6f}" for value in features]
                )
                detected[label] += 1

    # ---------------------------------------------------------------- summary
    print("per-chord detection summary")
    print(f"  {'chord':<6}{'detected':>10}{'failed':>8}{'total':>8}{'rate':>8}")
    total_detected = total_failed = 0
    for label in config.CHORD_LABELS:
        count_ok, count_bad = detected[label], failed[label]
        total = count_ok + count_bad
        rate = f"{100 * count_ok / total:.1f}%" if total else "-"
        print(f"  {label:<6}{count_ok:>10}{count_bad:>8}{total:>8}{rate:>8}")
        total_detected += count_ok
        total_failed += count_bad

    overall = total_detected + total_failed
    rate = f"{100 * total_detected / overall:.1f}%" if overall else "-"
    print(f"  {'ALL':<6}{total_detected:>10}{total_failed:>8}{overall:>8}{rate:>8}")

    if failed_paths:
        print(f"\n{len(failed_paths)} image(s) skipped:")
        for path, reason in failed_paths[:20]:
            print(f"  {reason}: {path.name}")
        if len(failed_paths) > 20:
            print(f"  ... and {len(failed_paths) - 20} more")

    print(f"\nwrote {total_detected} row(s) to {out_path}")


if __name__ == "__main__":
    main()
