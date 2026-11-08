# Recording tracker

My own checklist for finishing data collection. Tick things off as I go.

Status as of **2026-11-05**. Recording continues but is **targeted** — see below.
Kept as the measurement log for the write-up and the defence.

---

## ✅ 16 sessions, two people, model trained. Recording is now targeted.

9,528 images, 8,744 usable rows, 16 sessions, two people. Teammate 1's A-D
recordings arrived 2 Nov. `train.py` and `realtime.py` exist.

### Recording is now TARGETED, not finished

A generic extra session is no longer worth the evening - the learning curve has
flattened. But a **per-chord** curve shows three chords still climbing hard, so
recording those specifically still pays:

| Chord | Voicing | Recall at 3 -> 13 sessions | Gain over last step |
| --- | --- | --- | --- |
| **Dm** | `xx0231` | 14.4% -> 53.8% | **+10.2** |
| **D** | `xx0232` | 27.6% -> 66.3% | **+8.7** |
| **E** | `022100` | 47.3% -> 67.2% | **+6.5** |

Everything else is flat: `F` +0.0 (already 100%), `G` +0.0, `B` -7.3, `C` +2.0,
`Em` +2.0. `Dm` is also the weakest chord on a new person at 37.1%, so it is the
single best use of recording time. See the brief below for how to record them
without breaking the evaluation.

Overall sessions, for the record:

| Training sessions | Accuracy | Gain per session |
| --- | --- | --- |
| 1 | 24.2% | |
| 4 | 52.1% | +6.8 |
| 8 | 60.6% | +5.8 |
| 10 | 64.7% | +2.1 |
| 13 | 66.0% | **+0.4** |

### STEP 1 BRIEF — the targeted round (do this before the paper)

Three sessions. Each one covers **all nine active chords**, but spends double the
frames on `D`, `Dm` and `E`.

**Why all nine and not just the three.** `train.py` scores the model with
leave-one-**session**-out: it holds one session back, trains on the other
fifteen, and averages. A session containing only three labels makes a fold whose
test set has 3 classes while every other fold has 9, so the per-session scores
stop being comparable and the averaged headline is no longer a clean number. The
weighting gets the same targeted data with no footnote on the metric.

**Frame budget per session** (`--interval 0.7`, so seconds ≈ frames × 0.7):

| Chord | Frames this round | Why |
| --- | --- | --- |
| `E` | **80** | thinnest class in the dataset: 408 rows vs A's 1066 |
| `D` | 70 | still climbing, +8.7 on the last curve step |
| `Dm` | 70 | still climbing, and worst chord on a new person (37.1%) |
| `A` `B` `C` `Em` `F` `G` | 35 each | flat — just keep the session balanced |

That is about 500 frames, roughly 6 minutes of capture plus setup.

**Three separate launches, not one long one.** Quit `record_data.py` and relaunch
between sessions — each launch gets a new session id, and effective sample size
scales with (chords × **sessions**), not with image count. Three 500-frame
sessions beat one 1,500-frame session.

**Change something structural between the three sessions.** Camera height, your
distance from it, which chair, sleeves vs bare forearm, daylight vs lamp. This is
what makes the sessions independent rather than three copies.

**Within each take, keep moving** — the measured +7.8-point finding. Re-grip,
lift and replace fingers, rock the wrist, let the neck angle drift a few degrees.
Do not hold a statue pose. Expect MediaPipe to detect slightly fewer frames
(~83% vs ~87% for static bursts); that is the known cost and it is worth paying.

**Run it as:**

```
python record_data.py --interval 0.7
```

Keys: `a`-`g` selects the root, `n` toggles minor mode, **`t` starts/stops
auto-capture**, space saves one frame by hand, `l` toggles the neck guide, `q`
quits. (Space is a single manual shot, not the auto toggle — easy to mix up.)

**After all three sessions, run these two and tell me:**

```
python extract_landmarks.py --root data/recordings --source own
python train.py
```

Then I write the paper against whatever those print. Nothing else in the repo
needs touching.

### Em is the open question

`Em` (`022000`) sits at **0.386 f1**, the worst of the nine, and it got *worse*
when `Am` was excluded. It is one finger from `E` (`022100`) and flat on the
learning curve at 13.2%, so more recording will not fix it. Excluding it would
give 8 classes, still cover all seven letters, and likely push the headline past
78% - at the cost of leaving `Dm` as the only minor chord. Not decided.

| Final result | |
| --- | --- |
| Unseen session (MLP, leave-one-session-out, 16 sessions) | **74.6%** |
| Unseen **person** (5 classes, chance 20%) | **70.8%** |
| SVM, same evaluation | 73.4% |
| Chords classified | 9 — every letter A-G as majors, plus Dm and Em |

### Five chords are excluded, and it was the single biggest gain

`Am`, `Bm`, `Cm`, `Fm`, `Gm` are recorded but excluded from training. Each is
geometrically identical to another chord once the features ignore position:

- `Bm` (`x24432`) and `Cm` (`x35543`) — same A-shape minor barre, two frets apart
- `Fm` (`133111`) and `Gm` (`355333`) — same E-shape minor barre, two frets apart
- `Am` (`x02210`) and `E` (`022100`) — same three-finger shape, moved one string

`Am` was excluded on 5 Nov after live testing: the demo confused it with `E`
constantly, and the confusion matrix agreed (274 errors). Excluding it took `E`
from **0.540 to 0.834** f1 and the headline from 71.6% to 74.6%. Note honestly
that the *cross-person* result did not improve once you account for the smaller
test set (4.3x chance before, 3.5x after) — the gain is in same-person accuracy.

| Scheme | Classes | Unseen session | Unseen person |
| --- | --- | --- | --- |
| all 14 | 14 | 63.0% | 51.7% |
| drop those four | 10 | 71.6% | 71.8% (6 cls) |
| **also drop `Am`** | **9** | **74.6%** | **70.8%** (5 cls) |
| merge the pairs instead | 12 | 72.7% | 54.7% |
| merge B/Bm/Cm and Fm/Gm | 11 | 75.0% | 67.3% |

**Why dropping beats merging**, which is worth being able to explain: `B`
(`x24442`) is one finger from `Bm`. Merging `Bm` with `Cm` creates one large
class that swallows `B` — on a new person `B` scored **6.1%** under the merge
and **96.9%** once `Bm` was dropped. The 11-class merge scores better on my own
unseen sessions (75.0%) but worse on a new person (67.3%), so it is written up
as a measured alternative, not the chosen one.

### What is still wrong, and cannot be fixed with data

| Pair | Errors | Cause |
| --- | --- | --- |
| A <-> D | 304 | `x02220` vs `xx0232`, both compact three-finger shapes |
| E <-> Am | 274 | the same shape on different strings |
| C <-> Dm | 216 | `x32010` vs `xx0231` |

The same pairs dominate the errors on the unseen person, which is the good kind
of bad news: the model fails *consistently*, not randomly.

---

## How the data was collected (kept for the record)

The procedure below was validated on 1 Nov: at equal session count, sessions
recorded with varied grip beat sessions recorded holding still by **+7.8**
points. That finding stands and is worth stating in the paper.

## How to run a round — the new way (~25 min)

**1.** Start it:

```powershell
python record_data.py
```

**2. Majors pass** — check the overlay says `[major (n for minor)]`, then:

`a` → `b` → `c` → `d` → `e` → `f` → `g`

**3. Switch to minors** — press `n` **once**. Confirm the overlay reads `[MINOR (n)]`.

**4. Minors pass** — same keys again.

**5.** Press `q` and check the summary.

### For each individual chord — this is the part that changed

1. Press the letter
2. **Read the overlay** — confirm it shows the chord you're about to play
3. Check `hand detected` is **green**
4. Press `t` to start
5. **Hold ~25 seconds, and keep changing things the whole time:**
   - **Every ~3 seconds, take your fingers off the strings and put them back on
     the same chord.** Keep recording, keep the same chord, leave the camera alone.
     This is the single most important one. Your fingers never land in exactly the
     same spot, so each lift is a genuinely new sample; staying put is not.
   - Shift your hand a little along the neck, then come back
   - Roll your wrist slightly, change the angle of your palm
   - Lean in and out, so your hand is nearer and further from the camera
   - Tilt the guitar neck up and down a bit
6. Press `t` to stop

**Target ~35 frames per chord with ~8 hand lifts** — not 50 frames of the same thing.

### Barre chords: short bursts, not one long hold

Measured on the 2026-11-01 batch: the two chords that *lost* accuracy were the two
varied least, and both are hard to hold. `Fm` (`133111`, full barre at the 1st fret)
had the lowest spread of any chord, 2.59, and lost 18.2 points of recall.

So for **Fm, Gm, Cm, Bm**: press `t` off after ~8 seconds, rest 3 seconds, press `t`
back on. Three short bursts give the same frame count, real hand lifts between them,
and no forearm cramp. One 25-second barre hold is how Fm ended up static.

Also prefer `t` over holding the spacebar. Auto-capture takes averaged 6.39 spread
against 5.14 for held-spacebar bursts — a small free gain.

Three levels, easy to mix up:

| Level | What changes | How often |
| --- | --- | --- |
| **Round** | Camera, lighting, guitar | once per sitting |
| **Take** (one chord) | nothing — same chord throughout | 14 per round, ~25s each |
| **Hand lift** | fingers off the strings and back on | ~8 per take |

Still keep the chord *correct* the whole time. Vary the viewpoint and the grip,
never the notes.

---

## Where I am

| Session | Date | Images | Detection | Note |
| --- | --- | --- | --- | --- |
| `9140eed9` | 10-24 | 354 | 100% | first session |
| `65579446` | 10-25 | 665 | 100% | 13 chords |
| `9e2c59ad` | 10-25 | 449 | 100% | 11 chords |
| `53993a1d` | 10-26 | 739 | 93% | all 14 |
| `22e4ece6` | 10-28 | 728 | 83% | |
| `45b37cbd` | 10-28 | 729 | 89% | |
| `47b25588` | 10-28 | 702 | 87% | |
| `8fa2a578` | 11-01 | 505 | 79% | ← new method |
| `ce5a6445` | 11-01 | 507 | 79% | ← new method, too dark |
| `2cd8232e` | 11-01 | 506 | 79% | ← new method |
| `fb4e4b98` | 11-01 | 510 | 94% | ← new method, best |

**11 sessions, 6,394 images, 5,701 usable rows (89.2%).**

The ~79% detection in three new sessions is mostly the *price of movement*, not a
fault: held-spacebar takes detect at 87% and varied takes at 83%, because a moving
hand passes through poses MediaPipe cannot read. One genuine fix - `ce5a6445` is
dark (mean brightness 59.5, against 99-119 in the older rounds). Add a lamp.

### The four new rounds were all one evening

They ran 18:35 to 20:58 on the same night, and it shows: distance between the four
new session centroids is **6.93**, against **9.22** among the older seven. Brightness
did vary between them (59-93), which helped. Not worth re-recording - just put the
next round on a different day, in a different room.

### How many more sessions? Still worth recording.

Train on *k* sessions, always test on a session held out and fixed (MLP, no
augmentation, averaged over every choice of held-out session):

| Training sessions | Accuracy | Spread | Gain |
| --- | --- | --- | --- |
| 1 | 23.5% | 12.5 | |
| 2 | 29.7% | 12.5 | +6.2 |
| 3 | 37.7% | 11.3 | +8.0 |
| 4 | 43.7% | 8.4 | +6.0 |
| 5 | 47.4% | 8.9 | +3.7 |
| 6 | 51.6% | 6.9 | +4.3 |

**No plateau** — still gaining ~4 points per session at six, so more rounds are
worth the time. The spread column matters too: it falls from 12.5 to 6.9, so more
sessions make the result *trustworthy*, not just higher. That's what to say if the
panel asks how confident we are.

Rough projection: three more sessions ≈ 63%, plus ~8 from rotation augmentation
≈ low 70s, before temporal voting in the live demo. Indicative only — the curve
must flatten somewhere and we cannot see where yet.

Every session in that table was recorded the **old static way**, and the projection
above was written before the new batch existed. What actually happened: 7 -> 11
sessions took the MLP from 51.6% to 60.1%, **+8.5 points**, and the new-vs-old test
at the top of this file shows the method change is worth +7.8 of its own. Still no
plateau.

---

## Per-chord status

All 11 sessions. Recall is leave-one-session-out with the MLP; "old" is the same
measure over the 7 pre-11-01 sessions, so the change column isolates the new batch.
Spread is the mean within-take spread in the four new sessions - the variety measure.

| Chord | Rows | Detection | Spread | Recall (old) | Recall (now) | Change |
| --- | --- | --- | --- | --- | --- | --- |
| A | 697 | 92% | 6.46 | 61.4% | 61.8% | +0.4 |
| Am | 371 | 93% | 4.88 | 32.7% | 61.2% | **+28.5** |
| B | 349 | 88% | 7.03 | 62.9% | 68.8% | +5.9 |
| Bm | 369 | 88% | 7.11 | 23.8% | 42.0% | **+18.2** |
| C | 294 | **73%** | 5.78 | 33.3% | 67.0% | **+33.7** |
| Cm | 431 | 97% | 7.21 | 46.3% | 46.6% | +0.4 |
| D | 440 | 93% | 6.36 | 63.5% | 72.0% | +8.5 |
| Dm | 398 | 95% | **4.04** | 75.0% | 67.1% | **−7.9** |
| E | 327 | 78% | 5.26 | 39.5% | 51.1% | +11.6 |
| Em | 373 | 83% | 6.38 | 16.7% | 43.4% | **+26.7** |
| F | 424 | 89% | 4.64 | 70.8% | 87.7% | +16.9 |
| Fm | 405 | 90% | **2.59** | 78.0% | 59.8% | **−18.2** |
| G | 412 | 93% | 7.09 | 32.3% | 54.1% | **+21.8** |
| Gm | 411 | 94% | 3.24 | 46.5% | 54.7% | +8.3 |

All five chords flagged as priorities in the old version of this file improved:
C +33.7, Am +28.5, Em +26.7, G +21.8, E +11.6. Em was 6.7% and is now mid-pack.

### Variety is what did it

Sort the table by spread and the pattern is plain:

| | Chords | Mean recall change |
| --- | --- | --- |
| Bottom 4 by spread | Fm, Gm, Dm, F | **−0.2** |
| Top 4 by spread | B, G, Bm, Cm | **+11.6** |

Correlation between spread and recall change: **+0.40**. The chords I moved improved;
the chords I held still did not.

### Redo list for next round

1. **Fm** - lowest spread (2.59) and the only big regression (−18.2). Use the
   short-burst method above. Note `8fa2a578` Fm gave only **8 usable rows from 35
   images** (23% detection) - effectively a lost take.
2. **Dm** - second-lowest spread (4.04), −7.9. Same treatment.
3. **Gm** - spread 3.24. It gained +8.3, but partly by trading against Fm rather
   than improving outright, so it still needs real variety.
4. **C** - 73% detection, the lowest of any chord. Framing, not variety: its recall
   is fine now (+33.7).

Everything else is in good shape. **Do not re-record the bulk of the new batch.**

## Notes — things that went wrong before

**🔴 Watch the `hand detected` indicator.** If it flickers red, stop, fix framing or
lighting, restart. One session lost 130 images this way.

**🔴 Keep the chord correct while you vary the viewpoint.** The new method changes
*how you're seen*, not *what you play*. F once got recorded with two different
shapes in a take and had to be redone.

**🔴 Only press `n` once per round.** Toggling between every chord is how Am got
recorded as A major. Majors first, one toggle, then minors.

**🔴 F is mini F (`xx3211`)** — top four strings. **Not** the full barre. Deliberate:
the full barre is one finger from Fm and the model can't separate them.

**🔴 Change the setup between rounds too.** Move the camera, change the lighting,
swap guitars. Varying inside the take and between rounds both matter.

**🟢 Don't delete anything yourself** unless you're sure — send me the session and
I'll check detection rates and shape consistency first.

---

## Voicings (one shape per chord, always)

| Root | Major | Minor |
| --- | --- | --- |
| A | `x02220` | `x02210` |
| B | `x24442` | `x24432` |
| C | `x32010` | `x35543` |
| D | `xx0232` | `xx0231` |
| E | `022100` | `022000` |
| F | `xx3211` ← mini F | `133111` |
| G | `320003` | `355333` |

Low E to high e. `x` = don't play, `0` = open, digits = fret.

---

## Known issues, already decided

**Structural collisions — cannot be fixed with data.** Bm↔Cm and Fm↔Gm are the same
hand shape at different frets; E↔Am is the same shape on different strings. Our
features deliberately ignore position, so these pairs will always collide. Counted
over all 11 sessions:

| Confusion | Errors |
| --- | --- |
| Bm ↔ Cm | 350 |
| A ↔ D | 245 |
| Fm ↔ Gm | 228 |
| E → Am | 131 |
| G → A | 102 |

**A↔D is new to the top of this list** and belongs in the report alongside the
others: `x02220` and `xx0232` are both compact three-finger shapes, and
position-blind features cannot separate them by where they sit on the neck. Published work hits the same wall. This goes in the report
as a documented limitation, not a bug.

**Public datasets — tested, mostly rejected.** Two were measured:

- `dduka/guitar-chords` — 6–19% detection. Rejected.
- `hlcv/guitar-chords-eu59l` — 14 chords, 60–65% detection, 1,365 usable rows after
  filtering to the fretting hand and mirror-correcting one session. Effect on
  accuracy: **+1.7 SVM, −5.3 MLP.** Not used by default. The hand is too small in
  frame for MediaPipe to resolve fingers; it scores only 18.4% classifying itself.

The screening lesson: **hand size in frame decides everything.** Any dataset where
the whole player is in shot will fail the same way, however many images it has.

---

## Carried forward to later phases

Measured on 2026-10-31 over 7 sessions, leave-one-session-out. The absolute
numbers are now low - the baseline is 60.1% over 11 sessions - but the rankings
stand until re-measured. To be applied when we get there.

**1. Rotation augmentation in `train.py` (Phase 2) — biggest free win.**

Rebuild the (21,3) landmarks from the normalised coordinates, rotate by a random
angle on each axis, recompute the 85 features. Augment the **training** sessions
only; the held-out session must stay untouched or the score is meaningless.

| Setting | MLP |
| --- | --- |
| none | 51.6% |
| **rot ±30° × 3 copies** | **59.4%** ← use this |
| rot ±45° × 3 | 58.6% |
| rot ±30° × 5 | 58.2% |
| rot ±60° × 5 | 57.6% |

More rotation and more copies both make it worse. **Do not add coordinate jitter** —
it dropped the MLP from 58.6% to 54.5%.

Per chord, 11 of 14 improve: C +22.4, F +21.8, Em +16.7, A +14.4. Three regress:
**Fm −12.0**, D −3.3, G −3.0. The Fm loss is the price of orientation-blindness —
Gm gained +6.3 over the same run, so the Fm/Gm pair rebalanced rather than both
improving. E did not move at all (39.5% → 39.5%); augmentation cannot fix E.

**2. Majority-vote the prediction over ~1–2 seconds** in `realtime.py` (Phase 3).
Worth up to +9.5 points (46.9% → 56.3% over 30 frames). A chord is held for seconds;
classifying a single frame throws that away. Measured on frames 0.7s apart, so
expect less at 30fps where consecutive frames are more alike.

**3. Prefer the MLP over the SVM** (Phase 2). 51.6% vs 46.9% on identical splits.

**4. Tune hyperparameters** (Phase 2). `C=10, gamma=scale` has never been optimised.

Don't expect 1–3 to simply add up — augmentation and voting absorb some of the same
variance.
