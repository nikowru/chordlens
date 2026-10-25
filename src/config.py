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

# The seven natural-note roots. A chord label is either a bare root (major,
# e.g. "A") or a root with a trailing "m" (minor, e.g. "Am") - standard chord
# notation. record_data.py builds the current label from one of these roots
# plus a major/minor toggle, so the two can never drift apart.
ROOT_NOTES = ["A", "B", "C", "D", "E", "F", "G"]

# All 14 chords we classify: each root's major chord followed by its minor.
# Grouping by root (rather than all majors then all minors) keeps same-root
# pairs next to each other on a confusion matrix, which is exactly where a
# major/minor mix-up would show up.
# The order is fixed and used everywhere (CSV label column, confusion matrix
# axes, class order in reports) so figures from different scripts line up.
CHORD_LABELS = [label for root in ROOT_NOTES for label in (root, root + "m")]

# Keyboard keys used by record_data.py to select the current root chord.
# "a" -> "A", "b" -> "B", ... record_data.py appends "m" itself when minor
# mode is toggled on, so this only needs to map a key to a root note.
LABEL_KEYS = {root.lower(): root for root in ROOT_NOTES}

# One standard voicing per chord. Mixing voicings for the same label would mean
# teaching the model two different hand shapes under one class name, so the
# dataset must stick to these. Strings are listed low E to high e;
# "x" = not played, 0 = open string, digits = fret number.
#
# Majors use the open shape where the root allows it, and a barre chord
# otherwise (B, F). Minors follow the same pattern using the minor-shape
# equivalents: open Am/Dm/Em shapes where available, and barre chords (the
# "Am shape" for Bm/Cm, the "Em shape" for Fm/Gm) where there is no simple
# open minor voicing - this matches how these chords are actually played.
CHORD_VOICINGS = {
    "A": "x02220",    # open A major
    "Am": "x02210",   # open A minor
    "B": "x24442",    # B major barre (A-shape), index barre at the 2nd fret
    "Bm": "x24432",   # B minor barre (Am-shape), index barre at the 2nd fret
    "C": "x32010",    # open C major
    "Cm": "x35543",   # C minor barre (Am-shape), index barre at the 3rd fret
    "D": "xx0232",    # open D major
    "Dm": "xx0231",   # open D minor
    "E": "022100",    # open E major
    "Em": "022000",   # open E minor
    "F": "133211",    # F major barre (E-shape), full barre at the 1st fret
    "Fm": "133111",   # F minor barre (Em-shape), full barre at the 1st fret
    "G": "320003",    # open G major
    "Gm": "355333",   # G minor barre (Em-shape), full barre at the 3rd fret
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

# 0 = fast/light, 1 = slower/more accurate. Barre chords (B, F, Bm, Cm, Fm, Gm)
# press the fingers flat against the fretboard and occlude the knuckles, so
# the accurate model is worth the extra milliseconds.
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
AUTO_CAPTURE_INTERVAL_SECONDS = 0.7  # in auto-capture mode, save one frame this often
SMOOTHING_WINDOW = 7      # frames of prediction history smoothed in realtime.py

# ---------------------------------------------------------------------------
# Recording alignment guide
# ---------------------------------------------------------------------------
#
# An on-screen band showing roughly where the guitar neck should sit in the
# frame while recording. It is a framing aid and nothing more:
#
#   - it is drawn on the preview only, never on the saved image
#   - no feature in src/features.py uses it, and the model never sees it
#
# Its only job is to help several people, recording in several sittings, hold
# a similar camera angle - which keeps the dataset consistent without
# constraining what the model is allowed to learn. Toggle it live with "l";
# tune the numbers here if your setup needs a different angle.
SHOW_ALIGNMENT_GUIDE = True
GUIDE_CENTRE = (0.5, 0.55)    # the neck axis passes through here (fractions of width, height)
GUIDE_ANGLE_DEGREES = -20.0   # neck tilt in the frame; negative rises to the right
GUIDE_BAND_HEIGHT = 0.20      # thickness of the band, as a fraction of frame height
