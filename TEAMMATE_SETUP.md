# Getting started — for teammates

This is a step-by-step guide to get this project running on your own machine.
Read it in order; each step depends on the one before it.

If you get stuck, the error message is usually the fastest way to figure out
what's wrong — read it before asking, and paste it when you do ask.

---

## 0. What this project is

Real-time detection of guitar major chords (A–G) from a webcam. It uses
MediaPipe to find hand landmarks, turns those landmarks into engineered
geometric features (not raw pixels), and will classify them with an SVM/MLP.
See [README.md](README.md) for the full project write-up — this file is just
about getting your machine set up.

---

## 1. Get access to the repo

The repo is **private**, so you can't clone it until you're added as a
collaborator.

1. Give the project owner your GitHub username.
2. You'll get an email or a GitHub notification inviting you to the repo.
3. Click **Accept invitation** before moving on — cloning will fail with a
   "Repository not found" error until you do.

---

## 2. Check your Python version

**You need Python 3.12.** Not 3.13, not 3.11 — specifically 3.12.

This is not a style preference. The MediaPipe version this project pins
(`mediapipe==0.10.21`) only publishes Windows installers for Python 3.9
through 3.12. On 3.13 or newer, `pip install` will fail to find a matching
package.

Check what you have:

```powershell
py -0p
```

This lists every Python version installed on your machine and its path. Look
for a line with `3.12` in it. If you don't see one, download Python 3.12 from
[python.org/downloads](https://www.python.org/downloads/) and install it
before continuing (during install, tick "Add python.exe to PATH").

---

## 3. Clone the repo

Pick a folder on your machine where you keep projects, open a terminal there,
and run:

```powershell
git clone https://github.com/nikowru/chordlens.git
cd chordlens
```

The first time you do this, Git may open a browser window asking you to sign
in to GitHub — that's normal, sign in and it'll continue.

---

## 4. Create your own virtual environment

A virtual environment keeps this project's Python packages separate from
everything else on your machine. **Nobody shares a venv** — everyone creates
their own locally, and it is never committed to the repo.

```powershell
py -3.12 -m venv .venv
```

This creates a `.venv` folder inside the project. If `py -3.12` doesn't work,
go back to step 2 — it means 3.12 isn't installed or isn't registered with the
`py` launcher.

---

## 5. Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

You'll know it worked because your prompt will now start with `(.venv)`.

If PowerShell refuses with a script-execution error, run this once (it only
needs to be done the first time, per machine) and then retry step 5:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**You must activate the venv every time you open a new terminal to work on
this project.** If your prompt doesn't say `(.venv)`, you're not in it.

---

## 6. Install the dependencies

With the venv active (prompt shows `(.venv)`):

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This takes a few minutes — MediaPipe and its dependencies are a sizeable
download. Let it finish.

---

## 7. Verify everything works

Run this to confirm MediaPipe, OpenCV, and the rest installed correctly with
the right versions:

```powershell
python -c "import mediapipe as mp, cv2, sklearn; print('mediapipe', mp.__version__); print('legacy Hands API present:', hasattr(mp.solutions, 'hands')); print('cv2', cv2.__version__); print('sklearn', sklearn.__version__)"
```

Expected output:

```
mediapipe 0.10.21
legacy Hands API present: True
cv2 4.11.0
sklearn ...
```

If `mediapipe` fails to import, or the version is wrong, something went
sideways in steps 2–6 — double check you're using Python 3.12 inside an
activated venv.

---

## 8. Understand the project layout

```
chordlens/
  data/
    images/          public dataset images, one subfolder per chord: A/ B/ ... G/
    recordings/      webcam captures, same per-chord structure
    landmarks.csv    extracted features (generated, not written by hand)
  models/            trained models (not created yet — later phase)
  reports/           figures like the confusion matrix (not created yet)
  src/
    config.py        shared settings: chord labels, paths, MediaPipe settings
    features.py       the 21 landmarks -> 85 feature vector, with full docstrings
  record_data.py     webcam tool to record your own chord images
  extract_landmarks.py   turns a folder of images into data/landmarks.csv
  requirements.txt   exact Python packages this project needs
  README.md          the full project write-up (read this next)
```

The project follows **CRISP-DM**: data understanding → data preparation →
modeling → evaluation → deployment. The README is organized the same way.

---

## 9. Try running something

Record a few images of yourself holding a chord shape:

```powershell
python record_data.py
```

Keys: `a`–`g` pick the chord label, `SPACE` saves a frame, `t` toggles
auto-capture, `q` quits. It'll show the detected hand skeleton live and
per-chord counts on screen.

Then turn a folder of images into a feature CSV:

```powershell
python extract_landmarks.py --root data/recordings --source own
```

If both of those run without errors, your setup is good.

---

## 10. Day-to-day workflow (Git basics)

Before you start working, make sure you have the latest changes:

```powershell
git pull
```

After you make changes you want to share:

```powershell
git status          # see what changed
git add .
git commit -m "describe what you changed, e.g. 'add F chord recordings'"
git push
```

A few rules to avoid headaches on a shared repo:

- **Pull before you push.** If someone else pushed in the meantime, `git push`
  will be rejected — run `git pull` first, resolve anything that conflicts,
  then push again.
- **Don't commit your `.venv` folder.** It's already in `.gitignore`, so this
  should happen automatically — but if `git status` ever shows `.venv/` as a
  new/changed folder, stop and ask before committing.
- **Check `git status` before `git add .`.** It's the easiest way to catch
  accidentally adding something you didn't mean to (a stray screenshot, a
  half-downloaded file, etc.).
- If you're unsure what a command will do, ask in the group chat before
  running it — some git commands (`reset --hard`, `checkout --`, `clean`) can
  discard work permanently.

---

## Common problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `git clone` says "Repository not found" | You haven't accepted the collaborator invite yet | Check your email/GitHub notifications, accept it |
| `py -3.12` not recognized | Python 3.12 isn't installed | Install it from python.org (step 2) |
| `pip install -r requirements.txt` can't find mediapipe | Wrong Python version active | Confirm `(.venv)` is showing and it was created with `py -3.12` |
| Prompt doesn't show `(.venv)` | Venv isn't activated | Re-run step 5 in this terminal |
| `Activate.ps1` blocked by PowerShell | Execution policy | Run the `Set-ExecutionPolicy` command in step 5, once per machine |
| Webcam window doesn't open / camera error | Another app is using the webcam, or wrong camera index | Close other apps using the camera (Zoom, Teams, etc.); try `python record_data.py --camera 1` |
| `git push` rejected | Someone else pushed first | `git pull`, then `git push` again |

If none of these match what you're seeing, copy the exact error message and
ask in the group chat rather than guessing.
