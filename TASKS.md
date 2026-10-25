# Team tasks — data collection

This assigns the image-collection work for Phase 1/2. Read
[TEAMMATE_SETUP.md](TEAMMATE_SETUP.md) first if you haven't set up the project
yet — this file assumes your environment already works.

The coding (feature extraction, training, real-time app) is already handled.
**Your job is producing good webcam images of chord shapes**, using the
`record_data.py` tool that's already in the repo.

---

## Who does what

**Update:** the professor's feedback means minor chords are now in scope too
— not just the 7 majors, all 14 (every root, major **and** minor). Each
teammate keeps the same root letters as before, just adding the minor version
of each.

| Teammate | Chords | Count | Includes barre chord |
| --- | --- | --- | --- |
| **Teammate 1** | A, Am, B, Bm, C, Cm, D, Dm | 8 | B, Bm, Cm (barre) |
| **Teammate 2** | E, Em, F, Fm, G, Gm | 6 | F, Fm, Gm (barre) |

Each person is responsible only for their own roots — don't record the other
person's letters, so we don't end up with duplicate or conflicting data.

### Target: 200 images per chord

That's 200 × 8 = **1,600 images total for Teammate 1**, and 200 × 6 =
**1,200 total for Teammate 2** — 2,800 combined. Yes, this is roughly double
the original plan, because minor chords were added to scope.

This sounds like a lot, but `record_data.py` auto-captures one frame every
0.7 seconds while you hold a shape, so 200 images is only about **2.5 minutes
of actual holding time** per chord. The real time cost isn't holding the
pose — it's repeating that across **several separate sessions**, which matters
more than raw count. See the next section for why.

---

## Important: record in several separate sessions, not one sitting

Every time you run `record_data.py`, it generates one session ID, and every
image saved in that run shares it. Later, when the model is trained, images
that share a session ID are kept together on the same side of the
train/test split (so near-identical frames can't leak between them and
fake a good score).

**If you record all 200 images for a chord in one continuous 2.5-minute
burst, they'll all belong to one session** — which gives the training script
less to work with and makes your images less useful for teaching the model
what a chord looks like in different conditions.

Instead, spread each chord's 200 images across **at least 4–5 separate runs**
of the script, changing something each time:

- different time of day / lighting
- move the webcam or your chair (different distance, different angle)
- different background behind your hand
- a short break in between (don't just toggle auto-capture off and on in the
  same sitting — actually quit with `q` and restart later)

Roughly 40–50 images per session, across 4–5 sessions, per chord.

---

## How to record

1. Make sure you're using **your assigned voicing** for each chord (see
   table below) — mixing voicings under one label teaches the model two
   different hand shapes with the same name, which breaks everything
   downstream.
2. Run:
   ```powershell
   python record_data.py
   ```
3. Press the letter key for the root you're about to play (e.g. `a` for A).
   This defaults to **major**.
4. For the **minor** version of that root, press `n` first to switch to minor
   mode, then the letter (e.g. `n` then `a` records Am). Press `n` again to
   switch back to major. The on-screen title always shows which one is
   currently selected — check it before you start holding the shape.
5. Hold the chord shape steady, in frame, with your hand clearly lit.
6. Press `t` to turn auto-capture on. It'll save a frame every 0.7s as long
   as a hand is detected. Press `t` again to pause it when you reposition or
   switch chords.
7. Watch the on-screen counter for your chord — stop a session once you've
   added roughly 40–50 for that sitting.
8. Press `q` to quit. Repeat in a new session later (see above).

You can also press `SPACE` any time to save a single manual frame — useful
for a specific angle you want to make sure is captured.

### Voicings (use exactly these — see `src/config.py`)

| Root | Major | Minor | Notes |
| --- | --- | --- | --- |
| A | `x02220` | `x02210` | both open |
| B | `x24442` | `x24432` | **both barre**, index at 2nd fret |
| C | `x32010` | `x35543` | major open, **minor barre** (index at 3rd fret) |
| D | `xx0232` | `xx0231` | both open |
| E | `022100` | `022000` | both open |
| F | `133211` | `133111` | **both barre**, full barre at 1st fret |
| G | `320003` | `355333` | major open, **minor barre** (full barre at 3rd fret) |

Columns are low E to high e strings; `x` = not played, `0` = open string,
digits = fret number.

For the barre chords (B, Bm, C minor, F, Fm, G minor): MediaPipe sometimes
can't find a hand because the fingers lie flat and hide the knuckles. That's
expected — if the on-screen status keeps saying "NO HAND" for several
seconds, adjust your camera angle or lighting rather than forcing it; a shot
MediaPipe can't read isn't usable anyway.

---

## Checking your own work before you push

Before committing, check:

- [ ] The on-screen per-chord counts (visible on the live feed) roughly
      match what you expect — you haven't accidentally recorded under the
      wrong label.
- [ ] You used the correct voicing for every chord (no mixing).
- [ ] Spread across multiple sessions, not one long sitting (see above).
- [ ] No obviously unusable shots (hand out of frame, totally dark, a
      photo of something else by accident) — delete those from
      `data/recordings/<your letter>/` before committing.

---

## Pushing your images to the repo

Images save automatically to `data/recordings/<LABEL>/`. Once you're happy
with a session's results:

```powershell
git pull
git add data/recordings
git status
git commit -m "Add recordings for chord <LABEL>, session N"
git push
```

Check `git status` before committing — it should only show new files under
`data/recordings/<your letters>/`, nothing else. If you see files you don't
recognize, stop and ask before committing.

Push after every session or two rather than waiting until you have all 200 —
smaller, frequent commits are easier for everyone to review and far easier
to recover from if something goes wrong.

---

## Definition of done

- [ ] 200 images recorded for each of your assigned chords (majors **and**
      minors)
- [ ] Spread across at least 4–5 separate sessions per chord
- [ ] All pushed to the repo under `data/recordings/<LABEL>/`
- [ ] You've let the team lead know you're finished, so the data can be
      merged into `data/landmarks.csv` with `extract_landmarks.py`

If you get stuck on setup or git, check the troubleshooting table at the
bottom of [TEAMMATE_SETUP.md](TEAMMATE_SETUP.md) first, then ask.
