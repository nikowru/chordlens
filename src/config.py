"""
Central configuration for the guitar chord detection project.

Everything that more than one script needs to agree on lives here: the list of
chord labels, where files are read from and written to, and the MediaPipe Hands
settings. Keeping it in one place means the training script and the real-time
script cannot silently disagree (for example about label order), which is a
common source of bugs in this kind of pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

# The seven major chords we classify. The order is fixed and used everywhere
# (CSV label column, confusion matrix axes, class order in reports) so that
# figures produced by different scripts are directly comparable.
CHORD_LABELS = ["A", "B", "C", "D", "E", "F", "G"]

# Keyboard keys used by record_data.py to select the current chord.
# "a" -> "A", "b" -> "B", ... built from CHORD_LABELS so the two never drift.
LABEL_KEYS = {label.lower(): label for label in CHORD_LABELS}

# One standard voicing per chord. Mixing voicings for the same label would mean
# teaching the model two different hand shapes under one class name, so the
# dataset must stick to these. Strings are listed low E to high e;
# "x" = not played, 0 = open string, digits = fret number.
CHORD_VOICINGS = {
    "A": "x02220",   # open A major
    "B": "x24442",   # B major barre, index barre at the 2nd fret
    "C": "x32010",   # open C major
    "D": "xx0232",   # open D major
    "E": "022100",   # open E major
    "F": "133211",   # F major full barre at the 1st fret
    "G": "320003",   # open G major
}

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, so the scripts work from anywhere)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"          # public dataset, one subfolder per chord
RECORDINGS_DIR = DATA_DIR / "recordings"  # own webcam captures, same structure
LANDMARKS_CSV = DATA_DIR / "landmarks.csv"  # extracted feature table

MODELS_DIR = PROJECT_ROOT / "models"      # joblib-serialised trained models
REPORTS_DIR = PROJECT_ROOT / "reports"    # confusion matrices and other figures

# File types extract_landmarks.py will pick up.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ---------------------------------------------------------------------------
# MediaPipe Hands settings
# ---------------------------------------------------------------------------

# One hand only: the fretting hand. A second detected hand (strumming hand or a
# bystander) would produce feature vectors that do not describe the chord shape.
MAX_NUM_HANDS = 1

# 0 = fast/light, 1 = slower/more accurate. Barre chords (B and F) press the
# fingers flat against the fretboard and occlude the knuckles, so the accurate
# model is worth the extra milliseconds.
MODEL_COMPLEXITY = 1

# Settings for reading still images (extract_landmarks.py). static_image_mode
# runs the palm detector on every image instead of tracking between frames.
STATIC_MODE_SETTINGS = {
    "static_image_mode": True,
    "max_num_hands": MAX_NUM_HANDS,
    "model_complexity": MODEL_COMPLEXITY,
    "min_detection_confidence": 0.5,
}

# Settings for live video (record_data.py, realtime.py). Here MediaPipe detects
# once and then tracks, which is faster and steadier frame to frame.
VIDEO_MODE_SETTINGS = {
    "static_image_mode": False,
    "max_num_hands": MAX_NUM_HANDS,
    "model_complexity": MODEL_COMPLEXITY,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

# ---------------------------------------------------------------------------
# Recording / real-time defaults
# ---------------------------------------------------------------------------

CAMERA_INDEX = 0          # change if you have more than one webcam
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
AUTO_CAPTURE_EVERY = 5    # in auto-capture mode, save every Nth frame
SMOOTHING_WINDOW = 7      # frames of prediction history smoothed in realtime.py
