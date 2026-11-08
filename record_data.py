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

    keys a-g   choose the root note you are currently playing (A to G)
    n          toggle major/minor - affects the next root key you press
               (e.g. press n, then d, to record Dm; press n again for D major)
    SPACE      save the current frame
    t          toggle auto-capture on/off (saves one frame every 0.7s, by default)
    l          show/hide the neck alignment guide
    q          quit

Only the chords we are still collecting can be selected (config.RECORDABLE_LABELS
- currently A, B, C, D, Dm, E, Em, F, G and Bm). That list is the nine the model
trains on plus any chord still under evaluation: Bm is not in the model today,
but more of it is exactly what we need in order to decide whether it can be.

Am, Cm, Fm and Gm are dropped for good and cannot be selected. Each is
geometrically near-identical to another chord once the features drop position
(see src/config.py), so more of them would only be filtered back out. A key
combination that would select one is refused with a message instead of saving.

A key toggle rather than Shift+letter is used for major/minor, because
Shift's effect on cv2.waitKey() can vary with keyboard layout - a plain key
works the same on every machine.

The alignment guide is a band drawn on the preview showing roughly where the
guitar neck should sit. It helps several people recording in several sittings
keep a similar camera angle. It is drawn on the preview only - it never
reaches the saved image, and no feature uses it.

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
import math
import sys
import time
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
        "--interval",
        type=float,
        default=config.AUTO_CAPTURE_INTERVAL_SECONDS,
        help="in auto-capture mode, seconds between saved frames (default: %(default)s)",
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
    """Count images already on disk per chord, so the display shows totals.

    Only over RECORDABLE_LABELS: the dropped chords (Am, Cm, Fm, Gm) cannot be
    selected any more, so there is nothing new to count for them here. Their
    existing images are untouched on disk - extract_landmarks.py still reads
    every folder regardless of what this script tracks.
    """
    counts = {}
    for label in config.RECORDABLE_LABELS:
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


def draw_alignment_guide(frame):
    """
    Draw the neck-alignment band on the preview.

    Purely a framing aid. It is drawn on the display frame only - never on the
    clean frame that gets saved - and nothing in src/features.py looks at it,
    so it cannot influence what the model learns. Keeping the guitar roughly
    inside this band from session to session is what keeps the camera angle
    consistent across several people and several sittings.
    """
    height, width = frame.shape[:2]
    centre_x = config.GUIDE_CENTRE[0] * width
    centre_y = config.GUIDE_CENTRE[1] * height

    # Unit vector along the neck axis, and the perpendicular used to offset
    # the two edges of the band.
    angle = math.radians(config.GUIDE_ANGLE_DEGREES)
    along_x, along_y = math.cos(angle), math.sin(angle)
    across_x, across_y = -along_y, along_x

    half_band = config.GUIDE_BAND_HEIGHT * height / 2.0
    span = max(width, height)  # long enough to run past both frame edges

    def axis_points(offset):
        """The two endpoints of a line parallel to the axis, offset sideways."""
        base_x = centre_x + across_x * offset
        base_y = centre_y + across_y * offset
        return (
            (int(base_x - along_x * span), int(base_y - along_y * span)),
            (int(base_x + along_x * span), int(base_y + along_y * span)),
        )

    # Band edges in a muted grey, centre axis slightly brighter: visible
    # enough to line the neck up against, dim enough not to hide the hand.
    for offset in (-half_band, half_band):
        start, end = axis_points(offset)
        cv2.line(frame, start, end, (120, 120, 120), 1, cv2.LINE_AA)
    start, end = axis_points(0.0)
    cv2.line(frame, start, end, (170, 170, 170), 1, cv2.LINE_AA)

    cv2.putText(frame, "align neck to this band  (l to hide)",
                (12, int(centre_y + half_band) + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)


def draw_overlay(frame, label, minor_mode, counts, session_counts, session_id,
                  auto_on, interval, hand_found):
    """Draw the status panel: current label, counts, and key reminders."""
    green, red, white, yellow = (0, 200, 0), (0, 0, 255), (255, 255, 255), (0, 220, 220)

    mode_text = "MINOR (n)" if minor_mode else "major (n for minor)"
    cv2.putText(frame, f"chord: {label}  ({config.CHORD_VOICINGS[label]})  [{mode_text}]",
                (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, yellow if minor_mode else green, 2)

    cv2.putText(frame, "hand detected" if hand_found else "NO HAND",
                (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, green if hand_found else red, 2)

    auto_text = f"AUTO every {interval}s" if auto_on else "auto off  (t)"
    cv2.putText(frame, auto_text, (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                yellow if auto_on else white, 2)

    # Per-chord totals on disk, with this session's contribution in brackets.
    # Split across two rows (majors, minors) - 14 labels on one line would run
    # off the edge of the frame.
    majors = [chord for chord in config.RECORDABLE_LABELS if not chord.endswith("m")]
    minors = [chord for chord in config.RECORDABLE_LABELS if chord.endswith("m")]
    for row, chords in enumerate((majors, minors)):
        line = "  ".join(
            f"{chord}:{counts[chord]}(+{session_counts[chord]})" for chord in chords
        )
        cv2.putText(frame, line, (12, 126 + row * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)

    height = frame.shape[0]
    cv2.putText(frame, f"session {session_id}", (12, height - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
    cv2.putText(frame, "a-g root   n major/minor   SPACE save   t auto   l guide   q quit",
                (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)


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
    session_counts = {label: 0 for label in config.RECORDABLE_LABELS}
    current_root = config.ROOT_NOTES[0]
    minor_mode = False
    auto_capture = False
    show_guide = config.SHOW_ALIGNMENT_GUIDE
    last_auto_capture_time = 0.0  # time.monotonic() of the last auto-saved frame

    print(f"session id: {session_id}")
    print(f"recording {len(config.RECORDABLE_LABELS)} chords: "
          f"{', '.join(config.RECORDABLE_LABELS)}")
    print("keys: a-g root | n major/minor | SPACE save | t auto-capture | "
          "l guide | q quit")

    # The odd dropped frame is normal, but a camera that has been unplugged or
    # grabbed by another app fails every read. Without a limit the loop spins
    # forever printing, and never reaches the key handling below - so not even
    # 'q' would get you out.
    dropped_in_a_row = 0

    # Live video settings: detect once, then track between frames.
    with mp_hands.Hands(**config.VIDEO_MODE_SETTINGS) as hands:
        while True:
            ok, frame = camera.read()
            if not ok:
                dropped_in_a_row += 1
                if dropped_in_a_row >= config.MAX_DROPPED_FRAMES:
                    print(
                        f"\nThe camera stopped returning frames "
                        f"({dropped_in_a_row} in a row). Common causes:\n"
                        "  - another app has the camera (Messenger, Zoom, Teams, "
                        "a browser tab, the Camera app)\n"
                        "  - the webcam was unplugged or went to sleep\n"
                        "  - Windows camera privacy settings are blocking it\n"
                        f"  - wrong camera: try 'python record_data.py --camera 1'\n"
                        f"\nNothing was lost; images saved this session are on disk."
                    )
                    break
                print("Dropped a frame from the camera, retrying...")
                continue
            dropped_in_a_row = 0

            if args.mirror:
                frame = cv2.flip(frame, 1)

            # Keep a clean copy: saved images must not contain the drawn
            # skeleton, or extract_landmarks.py would be reading its own output.
            clean_frame = frame.copy()

            current_label = current_root + ("m" if minor_mode else "")

            # Drawn first, so the hand skeleton and status text sit on top of
            # it. Note this touches `frame` (shown) and not `clean_frame`
            # (saved) - the guide must never end up in the dataset.
            if show_guide:
                draw_alignment_guide(frame)

            # MediaPipe expects RGB; OpenCV gives BGR. Always run detection on
            # the clean frame, never the one we have drawn on: that way the
            # preview agrees with what extract_landmarks.py will later compute
            # from the saved image.
            results = hands.process(cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB))
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
            # unusable images. Timed by the clock rather than a frame count, so
            # the save rate stays the same regardless of the camera's FPS.
            now = time.monotonic()
            if auto_capture and hand_found and (now - last_auto_capture_time) >= args.interval:
                save_frame(clean_frame, current_label, session_id)
                counts[current_label] += 1
                session_counts[current_label] += 1
                last_auto_capture_time = now

            draw_overlay(frame, current_label, minor_mode, counts, session_counts,
                         session_id, auto_capture, args.interval, hand_found)
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
            elif key == ord("n"):
                candidate_minor_mode = not minor_mode
                candidate = current_root + ("m" if candidate_minor_mode else "")
                if candidate not in config.RECORDABLE_LABELS:
                    print(f"{candidate} is not being collected (see "
                          f"config.py) - staying on {current_label}")
                else:
                    minor_mode = candidate_minor_mode
                    print(f"mode -> {'minor' if minor_mode else 'major'}")
            elif key == ord("l"):
                show_guide = not show_guide
            elif key != 255 and chr(key) in config.LABEL_KEYS:  # 255 == no key pressed
                candidate_root = config.LABEL_KEYS[chr(key)]
                candidate = candidate_root + ("m" if minor_mode else "")
                if candidate not in config.RECORDABLE_LABELS:
                    print(f"{candidate} is not being collected (see "
                          f"config.py) - staying on {current_label}")
                else:
                    current_root = candidate_root
                    print(f"label -> {candidate} ({config.CHORD_VOICINGS[candidate]})")

    camera.release()
    cv2.destroyAllWindows()

    print(f"\nsession {session_id} finished. Saved this session:")
    for label in config.RECORDABLE_LABELS:
        print(f"  {label}: {session_counts[label]:4d}   (total on disk: {counts[label]})")


if __name__ == "__main__":
    main()
