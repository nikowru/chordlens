"""
record_data.py - collect my own webcam images of guitar chord shapes.

CRISP-DM stage: data understanding / data collection.

Why this script exists
----------------------
The public dataset images are shot with other guitars, other hands and other
lighting. A model trained on them alone tends to do noticeably worse on my own
webcam, so this script lets me record my own labelled examples under exactly
the conditions the final demo will run in.

How to use it
-------------
    python record_data.py

    keys a-g   choose the chord you are currently playing (A to G)
    SPACE      save the current frame
    t          toggle auto-capture on/off (saves every Nth frame)
    q          quit

Images are written to data/recordings/<label>/ as
    <label>_<sessionid>_<timestamp>.jpg
The session id is generated once per run of this script. Every image recorded
in one sitting therefore shares a session id, and train.py uses that id as the
grouping key so that near-identical consecutive frames cannot be split across
the train and test sets (that would leak, and inflate accuracy).

The on-screen per-chord counts are there so I can keep the classes balanced
while recording, rather than discovering an imbalance at training time.
"""

import argparse
import sys
import uuid
from datetime import datetime

import cv2
import mediapipe as mp

from src import config

# MediaPipe helper modules for drawing the skeleton on the preview.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_INDEX,
        help="webcam index (default: %(default)s)",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=config.AUTO_CAPTURE_EVERY,
        help="in auto-capture mode, save every Nth frame (default: %(default)s)",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help=(
            "flip the frame left-right before showing and saving it. Mirroring "
            "turns a left hand into a right hand as far as MediaPipe is "
            "concerned, so use the same setting for every recording session."
        ),
    )
    return parser.parse_args()


def count_existing_images():
    """Count images already on disk per chord, so the display shows totals."""
    counts = {}
    for label in config.CHORD_LABELS:
        folder = config.RECORDINGS_DIR / label
        if not folder.exists():
            counts[label] = 0
            continue
        counts[label] = sum(
            1
            for path in folder.glob("*")
            if path.suffix.lower() in config.IMAGE_EXTENSIONS
        )
    return counts


def save_frame(frame, label, session_id):
    """Write one clean frame (no drawn landmarks) and return its path."""
    folder = config.RECORDINGS_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    # Millisecond precision, so rapid auto-capture cannot collide.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = folder / f"{label}_{session_id}_{timestamp}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def draw_overlay(frame, label, counts, session_counts, session_id, auto_on, every, hand_found):
    """Draw the status panel: current label, counts, and key reminders."""
    green, red, white, yellow = (0, 200, 0), (0, 0, 255), (255, 255, 255), (0, 220, 220)

    cv2.putText(frame, f"chord: {label}  ({config.CHORD_VOICINGS[label]})",
                (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, green, 2)

    cv2.putText(frame, "hand detected" if hand_found else "NO HAND",
                (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, green if hand_found else red, 2)

    auto_text = f"AUTO every {every}f" if auto_on else "auto off  (t)"
    cv2.putText(frame, auto_text, (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                yellow if auto_on else white, 2)

    # Per-chord totals on disk, with this session's contribution in brackets.
    line = "  ".join(
        f"{chord}:{counts[chord]}(+{session_counts[chord]})"
        for chord in config.CHORD_LABELS
    )
    cv2.putText(frame, line, (12, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1)

    height = frame.shape[0]
    cv2.putText(frame, f"session {session_id}", (12, height - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
    cv2.putText(frame, "a-g label   SPACE save   t auto   q quit",
                (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1)


def main():
    args = parse_args()

    # One session id per run of the script; used in every filename below.
    session_id = uuid.uuid4().hex[:8]

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # CAP_DSHOW: faster on Windows
    if not camera.isOpened():
        sys.exit(f"Could not open camera {args.camera}. Is another app using it?")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    counts = count_existing_images()
    session_counts = {label: 0 for label in config.CHORD_LABELS}
    current_label = config.CHORD_LABELS[0]
    auto_capture = False
    frame_index = 0

    print(f"session id: {session_id}")
    print("keys: a-g label | SPACE save | t auto-capture | q quit")

    # Live video settings: detect once, then track between frames.
    with mp_hands.Hands(**config.VIDEO_MODE_SETTINGS) as hands:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Dropped a frame from the camera, retrying...")
                continue

            if args.mirror:
                frame = cv2.flip(frame, 1)

            # Keep a clean copy: saved images must not contain the drawn
            # skeleton, or extract_landmarks.py would be reading its own output.
            clean_frame = frame.copy()

            # MediaPipe expects RGB; OpenCV gives BGR.
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hand_found = bool(results.multi_hand_landmarks)

            if hand_found:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Auto-capture only keeps frames that actually contain a hand, so a
            # toggle left on while I reposition does not fill the folder with
            # unusable images.
            frame_index += 1
            if auto_capture and hand_found and frame_index % args.every == 0:
                save_frame(clean_frame, current_label, session_id)
                counts[current_label] += 1
                session_counts[current_label] += 1

            draw_overlay(frame, current_label, counts, session_counts, session_id,
                         auto_capture, args.every, hand_found)
            cv2.imshow("record chord data  -  q to quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                # Manual capture always saves: I am the one looking at the frame.
                path = save_frame(clean_frame, current_label, session_id)
                counts[current_label] += 1
                session_counts[current_label] += 1
                note = "" if hand_found else "  (warning: no hand detected)"
                print(f"saved {path.name}{note}")
            elif key == ord("t"):
                auto_capture = not auto_capture
                print(f"auto-capture {'ON' if auto_capture else 'OFF'}")
            elif key != 255 and chr(key) in config.LABEL_KEYS:  # 255 == no key pressed
                current_label = config.LABEL_KEYS[chr(key)]
                print(f"label -> {current_label} ({config.CHORD_VOICINGS[current_label]})")

    camera.release()
    cv2.destroyAllWindows()

    print(f"\nsession {session_id} finished. Saved this session:")
    for label in config.CHORD_LABELS:
        print(f"  {label}: {session_counts[label]:4d}   (total on disk: {counts[label]})")


if __name__ == "__main__":
    main()
