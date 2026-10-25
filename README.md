# Guitar Chord Detection from Hand Position

Real-time classification of 14 guitar chords — the major and minor triad for
each of the seven natural-note roots (A, B, C, D, E, F, G) — from a webcam,
using **MediaPipe Hands** landmarks and **classical machine learning** (SVM
and MLP).

The model never sees raw pixels. MediaPipe reduces each frame to 21 hand
landmarks, and those landmarks are turned into 85 engineered geometric features
(see [src/features.py](src/features.py)). The classifier learns from those
features alone, which is what keeps the dataset requirement small and makes
every input the model uses explainable.

The project is organised along **CRISP-DM**, and so is the rest of this README.

---

## Requirements and setup

**Python 3.12 is required.** MediaPipe's legacy `mp.solutions.hands` API — the
one this project uses — ships Windows wheels for CPython 3.9–3.12 only, up to
`mediapipe==0.10.21`. Newer MediaPipe releases (0.10.35 and 1.x) removed that
API in favour of the Tasks API, which needs a separately downloaded
`hand_landmarker.task` model file. The pin is deliberate; see the comments in
[requirements.txt](requirements.txt).

```powershell
# from the project folder, using the Windows py launcher
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the install:

```powershell
python -c "import mediapipe as mp; print(mp.__version__); print(hasattr(mp.solutions, 'hands'))"
# expected: 0.10.21  /  True
```

## Project layout

```
chord-detection/
  data/
    images/          public dataset images, one subfolder per chord: A/ Am/ B/ Bm/ ... G/ Gm/
    recordings/      my own webcam captures, same per-chord structure
    landmarks.csv    extracted features (one row per image)
  models/            trained models saved with joblib
  reports/           figures: confusion matrix, etc.
  src/
    config.py        labels, paths, MediaPipe settings
    features.py      landmarks -> 85-element feature vector (pure function)
  record_data.py     webcam capture tool for building my own dataset
  extract_landmarks.py   images -> data/landmarks.csv
  train.py           Phase 2 (not written yet)
  realtime.py        Phase 3 (not written yet)
```

---

## 1. Business understanding

A beginner guitarist cannot easily tell whether the shape their hand is making
is the chord they intended. The goal is a tool that watches the fretting hand
through a webcam and names the chord in real time, accurately enough to be
useful as practice feedback, and fast enough to feel instant on an ordinary
laptop CPU.

Success criteria: correct chord on held, well-lit shapes, with per-class
precision and recall reported honestly on data the model has never seen.

## 2. Data understanding

Two sources, both labelled by chord and both recorded in the `source` column of
the CSV so their contributions can be compared:

| source   | folder             | what it is                                           |
| -------- | ------------------ | ---------------------------------------------------- |
| `public` | `data/images/`     | public dataset photos, other players and other guitars |
| `own`    | `data/recordings/` | my own webcam captures, same camera as the live demo  |

Place the public images in `data/images/<CHORD>/`, one folder per chord.

### Voicings used

One standard voicing per chord. Mixing voicings under one label would mean
teaching the model two different hand shapes with the same name, so every image
in the dataset uses the shape below. Strings are written low E to high e;
`x` = string not played, `0` = open string, digits = fret number.

| Root | Major | Shape | Minor | Shape |
| ---- | ----- | ----- | ----- | ----- |
| A | `x02220` | open | `x02210` | open |
| B | `x24442` | **barre**, index at 2nd fret | `x24432` | **barre**, index at 2nd fret |
| C | `x32010` | open | `x35543` | **barre** (Am-shape, index at 3rd fret) |
| D | `xx0232` | open | `xx0231` | open |
| E | `022100` | open | `022000` | open |
| F | `133211` | **barre**, full barre at 1st fret | `133111` | **barre**, full barre at 1st fret |
| G | `320003` | open | `355333` | **barre** (Em-shape, full barre at 3rd fret) |

**6 of the 14 chords are barre chords: B, Bm, C minor, F, Fm, G minor.** The
fingers lie flat along the fretboard on these, which hides the knuckles from
the camera, so MediaPipe detects a hand less reliably on them.
`extract_landmarks.py` therefore reports detection and failure counts per
chord rather than an overall number — a low detection rate on the barre
chords is expected and is something to discuss, not hide.

### Collecting my own data

```powershell
python record_data.py
```

| key     | action                                                      |
| ------- | ------------------------------------------------------------ |
| `a`–`g` | set the current chord root (defaults to major)                |
| `n`     | toggle major/minor — applies to the next root key you press  |
| `SPACE` | save the current frame                                       |
| `t`     | toggle auto-capture (saves one frame every 0.7s, by default)  |
| `q`     | quit                                                          |

A key toggle is used for major/minor rather than Shift+letter, since Shift's
effect on `cv2.waitKey()` can vary by keyboard layout.

The live feed shows the detected hand skeleton, whether a hand is currently
found, and per-chord image counts, so the classes can be kept balanced while
recording. Images are saved as
`data/recordings/<label>/<label>_<sessionid>_<timestamp>.jpg`, where the session
id is generated once per run.

Useful options: `--camera N` to pick a different webcam, `--interval N` to
change the auto-capture interval in seconds (default: `src/config.py`'s
`AUTO_CAPTURE_INTERVAL_SECONDS`), `--mirror` to flip the image. Mirroring turns
a left hand into a right hand as far as MediaPipe is concerned, so use the same
`--mirror` setting for every session.

Saved frames are always clean camera frames — the drawn skeleton is never baked
into the image.

## 3. Data preparation

```powershell
# public dataset
python extract_landmarks.py --root data/images --source public

# my own recordings, appended to the same CSV
python extract_landmarks.py --root data/recordings --source own
```

This walks `<root>/<label>/*.jpg|png`, runs MediaPipe Hands in
`static_image_mode`, computes the features and appends one row per image to
`data/landmarks.csv`:

```
filepath, label, source, group, <85 feature columns>
```

Images with no detected hand are skipped, logged, and counted in a per-chord
detected-vs-failed summary printed at the end. The CSV is appended to if it
already exists, so the two sources accumulate in one table.

### The 85 features

| group | count | what it is | why |
| ----- | ----- | ---------- | --- |
| normalised coordinates | 60 | all 21 landmarks translated so the wrist is the origin, then divided by the wrist-to-middle-finger-MCP distance; the wrist itself is dropped as it is then always `(0,0,0)` | makes features **translation- and scale-invariant**: where the hand sits in the frame, and how far it is from the camera, stop mattering |
| fingertip distances | 10 | the distance between every pair of the five fingertips (`C(5,2)`) | states directly how spread or bunched the fingers are — fanned open C vs. a stacked barre |
| joint angles | 15 | three joint angles per finger, each the angle between the two bones meeting at that joint | invariant to translation, scale **and rotation**, so a tilted guitar neck does not change them; captures how far each finger is curled |

`extract_features()` is a pure function of the 21 landmarks, so the same code
path produces training rows and live predictions — no train/serve skew.

### The `group` column and data leakage

Rows that come from nearly the same photograph must never be split across the
training and test sets. If they are, the model is tested on data it has
effectively already seen and the reported accuracy is fiction. The `group`
column marks which rows are "the same underlying shot":

- **own recordings** — `group` is the session id from the filename. Frames from
  one sitting are near-duplicates of each other.
- **public images** — `group` is the original filename with the augmentation
  suffix stripped (everything from `.rf.` onwards), so all augmented copies of
  one photo share a group.

Phase 2 splits on this column with `GroupShuffleSplit` / `GroupKFold`.

## 4. Modeling — Phase 2, not written yet

`train.py` will read `data/landmarks.csv` and:

- split roughly 60% train / 20% validation / 20% test, **grouped** so no group
  appears in more than one split;
- train an SVM (linear and RBF kernels) and an MLP (tuned hidden layer sizes),
  using the validation set for tuning;
- re-check the choice with grouped k-fold cross-validation;
- save the best model to `models/` with joblib.

## 5. Evaluation — Phase 2, not written yet

Reported on the **test set only**: overall accuracy, per-class precision,
recall and F1, and a confusion matrix saved to `reports/`.

## 6. Deployment — Phase 3, not written yet

`realtime.py` will open the webcam, extract the same 85 features per frame,
predict with the saved model, and display the chord and confidence on screen,
smoothed over the last few frames to stop the label flickering.
