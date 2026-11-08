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
  train.py           Phase 2: trains and evaluates, writes models/ and reports/
  realtime.py        Phase 3: live webcam demo using the saved model
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
| F | `xx3211` | "mini F", top 4 strings, index barres B and e | `133111` | **barre**, full barre at 1st fret |
| G | `320003` | open | `355333` | **barre** (Em-shape, full barre at 3rd fret) |

**5 of the 14 chords are full barre chords: B, Bm, Cm, Fm, Gm.** The fingers
lie flat along the fretboard on these, which hides the knuckles from the
camera, so MediaPipe detects a hand less reliably on them. (F uses the "mini F"
shape instead of the full barre — see the note below the table.)
`extract_landmarks.py` therefore reports detection and failure counts per
chord rather than an overall number — a low detection rate on the barre
chords is expected and is something to discuss, not hide.

**Why F uses "mini F" rather than the full barre.** The full barre `133211`
differs from Fm `133111` by one finger on one fret, so the two shapes would sit
almost on top of each other in feature space. Measured on our own recordings,
the mini F shape sits 3.88 from Fm against a typical inter-chord distance of
4.50 — comfortably separable. The trade is mild proximity to Dm (1.67), which
is also a compact four-string shape. This avoids designing in a collision
between a chord and its own minor.

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

## 4. Modeling

```powershell
python train.py
```

`train.py` reads `data/landmarks.csv` and never touches an image. It trains two
candidates and picks the better one:

| Model | Settings |
| --- | --- |
| SVM | RBF kernel, `C=10`, `gamma="scale"` |
| MLP | hidden layers `(128, 64)`, `max_iter=800` |

Both are wrapped in a `StandardScaler` **inside** a pipeline. This matters: the
85 features are on very different scales (normalised coordinates sit near 0,
joint angles run to pi), and putting the scaler in the pipeline means it is
refitted on each training fold, so the test fold never influences the scaling.
Scaling once up front would leak information and inflate the score.

### Which chords are learned

Nine of the fourteen. `Am`, `Bm`, `Cm`, `Fm` and `Gm` are recorded and
extracted but excluded at training time, because each is geometrically
identical to another chord under these features:

- `Bm` (`x24432`) and `Cm` (`x35543`) are the same A-shape minor barre, two frets apart
- `Fm` (`133111`) and `Gm` (`355333`) are the same E-shape minor barre, two frets apart
- `Am` (`x02210`) and `E` (`022100`) are the same three-finger shape moved one string

`Am` was excluded last, on 2026-11-05, after live testing showed the demo
confusing it with `E` constantly. `E` was kept rather than `Am` because every
letter A-G must be covered and A major already covers the letter A. The effect
was large and one-sided: `E` went from 0.540 to **0.834** f1.

The features are translation- and scale-invariant but deliberately ignore
*where on the neck* the hand sits, so no amount of data separates these pairs.
Keeping them also costs accuracy elsewhere, because `B` (`x24442`) is one finger
from `Bm` and gets absorbed by it. Measured over 16 sessions and 8,744 rows:

| Scheme | Classes | Unseen session | Unseen person |
| --- | --- | --- | --- |
| all 14 | 14 | 63.0% | 51.7% |
| drop those four | 10 | 71.6% | 71.8% (6 cls) |
| **also drop `Am`** | **9** | **74.6%** | **70.8%** (5 cls) |
| merge the pairs instead | 12 | 72.7% | 54.7% |

All seven letters survive as majors (A-G) plus `Am`, `Dm` and `Em`, so every
chord A through G is still detected. The list lives in `src/config.py` as
`EXCLUDED_LABELS`, so re-including them is a one-line change, and
`python train.py --all-labels` reproduces the 14-class row above.

## 5. Evaluation

### Leave-one-session-out, not a random split

Frames from one recording sitting are near-duplicates of each other. Splitting
them randomly would test the model on data it had effectively memorised, and
the reported accuracy would be fiction - on this dataset, roughly 20-30 points
too high.

So `train.py` trains on every session but one, tests on the session held out,
and repeats for all 16. Every number below is accuracy on a sitting the model
has never seen.

| Model | Accuracy | sd | Worst session | Best session |
| --- | --- | --- | --- | --- |
| SVM | 73.4% | 10.1 | 55.3% | 88.9% |
| **MLP** | **74.6%** | 10.1 | 56.0% | 90.4% |

Per chord (MLP, f1-score): `F` 0.948, `B` 0.879, `E` 0.834, `A` 0.748, `D` 0.729,
`Dm` 0.657, `G` 0.635, `C` 0.630, `Em` 0.386.

**A caution on comparing the two cross-person numbers.** Excluding a chord also
removes it from the held-out person's test set, so the 10-class model was tested
on 6 classes and the 9-class model on 5. Measured against chance the cross-person
result did *not* improve (4.3x vs 3.5x); the gain from excluding `Am` is in
same-person accuracy, and the write-up should say so.

### The harder test: an unseen person

Two of the 16 sessions were recorded by a different person. Holding them out
entirely answers the question a demo actually faces:

```powershell
python train.py --holdout-sessions 1189027a e587eb5b
```

### Remaining confusions, and why they are structural

| Pair | Errors | Cause |
| --- | --- | --- |
| A <-> D | 304 | `x02220` vs `xx0232`, both compact three-finger shapes |
| E <-> Am | 274 | the same shape on different strings |
| C <-> Dm | 216 | `x32010` vs `xx0231` |

These follow directly from the feature design: a position-blind feature set
cannot distinguish shapes that differ only by which strings they sit on. This
is a documented limitation of the approach, not a bug to be fixed with more
data, and published work meets the same wall.

### Outputs

| File | Contents |
| --- | --- |
| `reports/confusion_matrix.png` | row-normalised matrix with counts |
| `reports/classification_report.txt` | per-class precision, recall, f1 |
| `reports/per_session_scores.csv` | accuracy for each held-out session |
| `models/chord_model.joblib` | the fitted pipeline |
| `models/chord_model_meta.json` | labels, feature order, scores, provenance |

## 6. Deployment

```powershell
python realtime.py
```

`realtime.py` opens the webcam, runs MediaPipe in video mode, converts the 21
landmarks into the same 85 features, and predicts with the saved model.

**Prediction smoothing.** A single frame is weak evidence - a finger blurs, the
hand turns, the label flips. But a chord is *held* for seconds, so the frames
either side of a bad one are usually right. `realtime.py` takes the majority
vote over the last `config.SMOOTHING_WINDOW` frames, which was measured on this
dataset as worth up to +9.5 points. The display shows both the raw per-frame
guess and the smoothed one, so the effect is visible rather than hidden.

**Feature-order safety.** The model's metadata records the exact feature order
used at training time. If `src/features.py` changes afterwards, the columns
would silently shift and every prediction would be quietly wrong - the worst
kind of bug, because the demo still appears to work. `realtime.py` compares the
two lists at startup and refuses to run on a mismatch.

| Key | Action |
| --- | --- |
| `q` | quit |
| `l` | show/hide the hand skeleton |
| `h` | show/hide the help text |

Useful flags: `--camera 1` for a second webcam, `--window 11` for a longer vote,
`--min-confidence 0.5` to show nothing when the model is unsure.

**Do not use `--mirror`.** The model was trained on unmirrored frames, and
mirroring turns the fretting hand into the other hand as far as MediaPipe is
concerned, which invalidates every feature.
