"""
realtime.py - live chord detection from a webcam.

CRISP-DM stage: deployment.

This is the demo. It reads frames from the webcam, finds the fretting hand with
MediaPipe, converts the 21 landmarks into the same 85 features used in
training, and asks the saved model which chord it is. No image ever reaches the
classifier - only geometry - which is the same guarantee that holds in train.py.

Why the prediction is smoothed
------------------------------
A single frame is a weak piece of evidence: a finger blurs, the hand turns, and
one frame's prediction flips. But a player holds a chord for *seconds*, so the
frames either side of a bad one are usually right. Taking the majority vote over
a short window of recent frames turns that redundancy into accuracy, and was
measured on this dataset as worth up to +9.5 points. It costs nothing but a
fraction of a second of lag.

The display shows both numbers on purpose: the raw per-frame guess and the
smoothed one. Watching the raw label flicker while the smoothed label stays
steady is the clearest way to show a panel what the smoothing is doing.

What this model can and cannot say
----------------------------------
It predicts the chords in config.ACTIVE_LABELS - ten of the fourteen. Bm, Cm,
Fm and Gm are excluded because they are geometrically identical to another
chord under these features; see the comment in src/config.py. All seven letters
A-G are still covered.

Usage
-----
    python realtime.py
    python realtime.py --camera 1 --window 11
    python realtime.py --min-confidence 0.5    # show nothing when unsure

Keys
----
    q    quit
    l    show/hide the hand skeleton
    h    show/hide the help text
"""

import argparse
import json
import sys
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np

from src import config
from src.features import FEATURE_NAMES, extract_features, landmarks_to_array

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

MODEL_PATH = config.MODELS_DIR / "chord_model.joblib"
META_PATH = config.MODELS_DIR / "chord_model_meta.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        help="webcam index (default: %(default)s)")
    parser.add_argument("--window", type=int, default=config.SMOOTHING_WINDOW,
                        help="frames of history in the majority vote (default: %(default)s)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="hide the prediction below this probability (default: off)")
    parser.add_argument("--mirror", action="store_true",
                        help=(
                            "flip the image left-right. OFF by default and it should "
                            "stay off: the model was trained on unmirrored frames, and "
                            "mirroring turns the fretting hand into the other hand as "
                            "far as MediaPipe is concerned, which invalidates every feature."
                        ))
    return parser.parse_args()


def load_model():
    """
    Load the saved pipeline and check it still matches this code.

    The metadata records the exact feature order train.py used. If
    src/features.py has changed since the model was fitted, the columns would
    silently shift and every prediction would be quietly wrong - the worst kind
    of bug, because the demo still looks like it works. So we refuse to start.
    """
    if not MODEL_PATH.exists():
        sys.exit(
            f"No trained model at {MODEL_PATH}\n"
            "Train one first:\n"
            "    python train.py"
        )

    model = joblib.load(MODEL_PATH)

    if not META_PATH.exists():
        print(f"warning: {META_PATH} is missing, cannot verify feature order")
        return model, config.ACTIVE_LABELS, {}

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    saved_features = meta.get("feature_names", [])
    if saved_features and saved_features != FEATURE_NAMES:
        sys.exit(
            "The saved model does not match src/features.py.\n"
            f"  model was trained on {len(saved_features)} features\n"
            f"  this code produces  {len(FEATURE_NAMES)} features\n"
            "features.py has changed since the model was fitted. Re-run:\n"
            "    python extract_landmarks.py --root data/recordings --source own\n"
            "    python train.py"
        )

    return model, meta.get("labels", config.ACTIVE_LABELS), meta


def draw_overlay(frame, voted, raw, confidence, votes, window, labels, show_help):
    """Draw the prediction panel. Nothing here is ever saved or fed to the model."""
    height, width = frame.shape[:2]

    # Dim strip behind the text so it stays readable over any background.
    panel = frame[0:120, 0:width].copy()
    frame[0:120, 0:width] = cv2.addWeighted(
        panel, 0.35, np.zeros_like(panel), 0.65, 0
    )

    if voted is None:
        cv2.putText(frame, "no hand", (16, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (120, 120, 120), 3)
    else:
        cv2.putText(frame, voted, (16, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (80, 230, 120), 4)

        # Confidence bar, drawn next to the chord name.
        bar_x, bar_y, bar_w = 190, 40, 220
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (70, 70, 70), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + int(bar_w * confidence), bar_y + 20), (80, 230, 120), -1)
        cv2.putText(frame, f"{100 * confidence:.0f}%", (bar_x + bar_w + 10, bar_y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)

        # The raw per-frame guess, so the smoothing is visible rather than hidden.
        cv2.putText(frame, f"this frame: {raw}", (bar_x, bar_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # What the vote window currently holds, most common first.
    if votes:
        summary = "  ".join(f"{name} {count}" for name, count in votes.most_common(3))
        cv2.putText(frame, f"vote({window}): {summary}", (16, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.putText(frame, f"{len(labels)} chords: {', '.join(labels)}",
                (16, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    if show_help:
        for i, line in enumerate(["q quit", "l skeleton", "h help"]):
            cv2.putText(frame, line, (width - 130, 30 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)


def main():
    args = parse_args()
    model, labels, meta = load_model()

    print(f"model   : {MODEL_PATH}")
    if meta:
        print(f"type    : {meta.get('model', '?').upper()}, "
              f"{meta.get('n_rows', '?')} rows, {meta.get('n_sessions', '?')} sessions")
        accuracy = meta.get("leave_one_session_out_accuracy")
        if accuracy is not None:
            print(f"accuracy: {100 * accuracy:.1f}% on unseen sessions")
    print(f"chords  : {', '.join(labels)}")
    print(f"smoothing: majority vote over {args.window} frames")
    print("\npress q in the preview window to quit\n")

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        sys.exit(
            f"Could not open camera {args.camera}. Another app may have it "
            f"(Messenger, Zoom, Teams, a browser tab), or try --camera 1."
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    history = deque(maxlen=max(1, args.window))
    show_skeleton = True
    show_help = True
    dropped_in_a_row = 0

    with mp_hands.Hands(**config.VIDEO_MODE_SETTINGS) as hands:
        while True:
            ok, frame = camera.read()
            if not ok:
                # Same guard as record_data.py: a camera grabbed by another app
                # fails every read, and without a limit this loop spins forever
                # without ever reaching the key handling below.
                dropped_in_a_row += 1
                if dropped_in_a_row >= config.MAX_DROPPED_FRAMES:
                    print(f"\nThe camera stopped returning frames "
                          f"({dropped_in_a_row} in a row). Another app may have "
                          f"taken it, or try --camera 1.")
                    break
                continue
            dropped_in_a_row = 0

            if args.mirror:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            raw_label = None
            confidence = 0.0

            if results.multi_hand_landmarks:
                landmark_list = results.multi_hand_landmarks[0]

                # Exactly the same two calls train.py's data went through.
                landmarks = landmarks_to_array(landmark_list)
                features = extract_features(landmarks, aspect_ratio=width / height)

                probabilities = model.predict_proba([features])[0]
                best = int(np.argmax(probabilities))
                raw_label = model.classes_[best]
                confidence = float(probabilities[best])

                history.append(raw_label)

                if show_skeleton:
                    mp_drawing.draw_landmarks(
                        frame, landmark_list, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
            else:
                # No hand: let the window forget rather than hold a stale chord.
                if history:
                    history.popleft()

            # ---- the smoothing itself -------------------------------------
            votes = Counter(history)
            voted_label = None
            if votes:
                voted_label, vote_count = votes.most_common(1)[0]
                # Report the confidence of the label we are actually showing,
                # measured as its share of the window, not the last frame's
                # probability - otherwise one lucky frame inflates it.
                confidence = vote_count / len(history)
                if confidence < args.min_confidence:
                    voted_label = None

            draw_overlay(frame, voted_label, raw_label, confidence,
                         votes, args.window, labels, show_help)
            cv2.imshow("chord detection  -  q to quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                show_skeleton = not show_skeleton
            elif key == ord("h"):
                show_help = not show_help

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
