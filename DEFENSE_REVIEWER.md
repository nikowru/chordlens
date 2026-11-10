# ChordLens — Defense Reviewer

Study notes for the thesis defense. Everything here is measured from the code and
data in this repository, not estimated.

**Last updated:** 2026-11-10. Final dataset (22 sessions, 2 players), final model
(multinomial logistic regression), all numbers re-measured.

---

## Table of contents

1. [The 60-second pitch](#1-the-60-second-pitch)
2. [Numbers cheat sheet](#2-numbers-cheat-sheet)
3. [What each file does](#3-what-each-file-does)
4. [The 85 features](#4-the-85-features)
5. [Why landmarks instead of raw pixels](#5-why-landmarks-instead-of-raw-pixels)
6. [How we evaluate, and why it's harder than normal](#6-how-we-evaluate-and-why-its-harder-than-normal)
7. [Algorithm comparison](#7-algorithm-comparison)
8. [Why 5 chords are excluded](#8-why-5-chords-are-excluded)
9. [The Bm story](#9-the-bm-story)
10. [Known weaknesses — say these before they ask](#10-known-weaknesses--say-these-before-they-ask)
11. [Likely panel questions](#11-likely-panel-questions)
12. [Textbook alignment (Maheshwari)](#12-textbook-alignment-maheshwari)
13. [Glossary](#13-glossary)

---

## 1. The 60-second pitch

ChordLens recognises which guitar chord a player is fretting, in real time, from a
normal webcam — with no audio and no special hardware.

A camera frame goes to **MediaPipe Hands**, which returns 21 points on the fretting
hand. Those points are converted into **85 engineered geometric features**
(normalised positions, fingertip distances, joint angles). A **classical machine
learning model** classifies those 85 numbers into one of **9 chords**.

The model never sees a photograph. It only ever sees 85 numbers describing hand
geometry.

**Covered chords:** A, B, C, D, Dm, E, Em, F, G — every letter A through G, as
required.

---

## 2. Numbers cheat sheet

### Dataset

| | |
|---|---|
| images captured | 13,369 |
| hands successfully detected | 12,302 (**92.0%**) |
| recording sessions | 22 |
| people | 2 |
| rows used for the 9 modelled chords | 9,592 |

### Rows per chord

| chord | rows | sessions |
|---|---|---|
| A | 1,420 | 16 |
| D | 1,171 | 16 |
| Dm | 1,171 | 16 |
| B | 1,057 | 15 |
| G | 982 | 17 |
| E | 961 | 18 |
| Em | 961 | 18 |
| F | 956 | 19 |
| C | 913 | 13 |

### Accuracy — the two headline numbers

**Final model: multinomial logistic regression, 9 chords.**

| test | result |
|---|---|
| **unseen session** (leave-one-session-out, 22 sessions) | **79.4%** (sd 7.8) |
| **unseen person**, all 9 chords | **68.9%** = **6.20x chance** |
| unseen person, 5 chords only | 85.2% = 4.26x chance |
| worst single session | 60.1% |
| best single session | 91.9% |

> **Know the difference between those two cross-person rows — a panel can probe it.**
> The 85.2% comes from sessions that contain only 5 of the 9 chords, so `train.py`
> trains a 5-class model for that test. It is a real number but it is **not your
> deployed system**. The 69.1% is the actual 9-chord model tested on a whole unseen
> player. Quote **68.9% / 6.20x chance** as your cross-person result. In
> chance-normalised terms it is the stronger of the two anyway (6.20x vs 4.26x).

79.4% clears the 70% usability threshold given in the course textbook.

### How we got here — two improvements, same day

> **Where these figures come from.** The script that produced this table
> originally was not kept, and the two stored copies of it had drifted apart. The
> numbers here are a re-derivation, not a recovery: `tools_diversity_check.py`
> rebuilds the 18-session feature table from commit 8d8aae9 and re-runs the
> protocol. Rows 2 and 3 reproduced the stored values exactly; row 1 did not, and
> has been replaced with what the re-run gives. Say so plainly if asked.

(cross-person row uses the 5-chord protocol throughout, so the three are comparable)

| | 1 person, SVM | 2 people, SVM | 2 people, logistic regression |
|---|---|---|---|
| unseen session | 74.8% | 76.4% | **78.9%** |
| **unseen person** | **66.7%** | **80.9%** | **85.2%** |

**Cross-person accuracy rose 18.5 points in total**, from two separate causes:

1. **+14.2 from a second person's recordings.** This is the most important result in
   the project. The session number barely moved (74.8 → 76.4): a second contributor
   transformed accuracy on an unseen person and left accuracy on the original player
   almost untouched, which is the signature of a model that had been fitting one
   person's hands. A more varied dataset is also a harder, more honest test.
2. **+4.3 from switching to a linear model.** See section 7.

**Then a third improvement, which raised consistency rather than the mean.** The
second player later contributed four more sessions covering E, Em, F and G under
better lighting:

| | before | after |
|---|---|---|
| unseen session | 78.9% | **79.4%** |
| **standard deviation** | 9.3 | **7.8** |
| **worst session** | 54.9% | **60.1%** |
| Em f1 | .576 | **.649** |
| G f1 | .723 | **.787** |

**Say this one out loud in the defense.** The mean barely moved, but the floor came
up five points and the spread narrowed. For a live demo, a steadier model matters
more than half a point of average accuracy. And it was measured on a *harder* test
than before - 22 sessions instead of 17.

Biggest per-chord gains on an unseen person from the second recorder: **C +20.0**
(57.5 → 77.5), **Dm +21.5** (52.2 → 73.7). Those were previously the two weakest
chords — the classic signature of a model that had been memorising one person's
hands.

### Per-chord performance (unseen sessions, f1-score)

| strong | | medium | | weak | |
|---|---|---|---|---|---|
| B | .906 | G | .787 | A | .716 |
| F | .900 | Dm | .745 | C | .676 |
| E | .818 | D | .815 | Em | .649 |

### Per-chord recall on the held-out player (all 5 of his sessions, 9 chords)

| | | | | | |
|---|---|---|---|---|---|
| Dm | 85.8% | D | 80.4% | B | 66.9% |
| A | 81.4% | G | 78.1% | Em | 55.5% |
| | | E | 73.1% | F | 54.4% |
| | | | | C | 46.1% |

68.9% overall against 11.1% chance = **6.20x chance**.

### Per-chord recall on a completely unseen person (all 9 chords)

| | | | | | |
|---|---|---|---|---|---|
| Dm | 96.9% | D | 80.7% | G | 58.9% |
| A | 92.7% | F | 76.6% | C | 52.4% |
| | | B | 65.7% | Em | 43.1% |
| | | E | 64.9% | | |

69.8% overall against an 11.1% chance level = **6.29× chance**.

---

## 3. What each file does

| file | CRISP-DM stage | what it does |
|---|---|---|
| `src/config.py` | — | Single source of truth: chord labels, voicings, paths, MediaPipe settings. Prevents training and real-time from silently disagreeing. |
| `src/features.py` | Data Preparation | Turns 21 raw landmarks into the 85 features. |
| `record_data.py` | Data Understanding | Webcam capture tool. Saves labelled frames with a session id in the filename. |
| `extract_landmarks.py` | Data Preparation | Runs MediaPipe over every saved image, writes `data/landmarks.csv`. |
| `train.py` | Modeling + Evaluation | Trains candidate models, runs leave-one-session-out, writes reports and the final model. |
| `realtime.py` | Deployment | Live webcam prediction with smoothing. |

**Key design point to state out loud:** `train.py` reads only the CSV. It never
touches an image. That is what makes this a landmark-feature project rather than a
computer-vision-on-pixels project.

---

## 4. The 85 features

| group | count | what it is |
|---|---|---|
| normalised coordinates | 60 | 20 landmarks × (x, y, z). The wrist is dropped because after centring it is always (0,0,0) and carries no information. |
| fingertip pair distances | 10 | every pair among the 5 fingertips |
| joint angles | 15 | the bend angle at each finger joint |

### The two normalisation steps

1. **Translation invariance** — every point is measured relative to the wrist, so it
   doesn't matter where the hand sits in the frame.
2. **Scale invariance** — every distance is divided by the wrist→middle-knuckle
   distance, so it doesn't matter how big the hand is or how close the camera is.

### What we deliberately did NOT make invariant

- **Rotation** — the angle of the hand is real information about the chord.
- **Mirroring** — a left hand and a right hand making the same shape are genuinely
  different, so we keep that difference.
- **Position along the neck** — *this is the single most important limitation in the
  project.* See sections 8 and 10.

---

## 5. Why landmarks instead of raw pixels

Be ready for "why not just use a CNN?"

1. **Dataset size.** We have ~12,000 frames from 2 people. A CNN on raw pixels needs
   far more, and would learn our rooms, our shirts and our guitars.
2. **It runs on a laptop.** No GPU. Training takes seconds to minutes; the live demo
   runs on a normal webcam. Our whole feature table is about 12,000 × 85 numbers.
3. **It is explainable.** We can point at a joint angle and say why two chords get
   confused. A CNN gives a number with no reason attached.
4. **MediaPipe already solved hand detection.** Re-learning "where is a hand" from
   500 photos would be worse than using a model trained on millions.

---

## 6. How we evaluate, and why it's harder than normal

### The trap we avoided

Our recorder saves a frame every 0.7 seconds. Consecutive frames are nearly
identical. A normal random train/test split would put near-duplicate photos on
**both** sides — the model would be tested on images it had effectively already
seen, and the accuracy would be **fiction**, probably 95%+.

### What we do instead

**1. Leave-one-session-out.** Every frame carries a `group` column = the recording
session id. We train on all sessions but one, test on the held-out session, repeat
for all 18. A whole sitting is always entirely in training or entirely in test.

**2. Held-out person.** We completely remove one person's sessions, train without
them, and test on them. This answers the real question: *does it work on someone we
have never recorded?*

### Why we report "× chance"

Schemes with different numbers of chords aren't directly comparable — guessing among
5 chords is easier than among 10. Dividing accuracy by the chance level makes them
comparable. 9 chords → 11.1% chance; 69.8% accuracy = 6.29× chance.

> **If asked "isn't 79% low?"** — Say: a random split would have given us 95% and it
> would have been meaningless. We chose the harder, honest measurement. Our textbook
> puts the usability threshold at 70%, and we clear it on both tests.

---

## 7. Algorithm comparison

All 11 run on identical data with identical evaluation.

| algorithm | unseen session | unseen person |
|---|---|---|
| **Linear SVM** | **79.2%** (sd 8.9) | 80.2% |
| **Logistic Regression** ← shipped | 78.9% (sd 9.2) | **80.6%** |
| SVM (RBF) | 76.4% (sd 10.3) | 77.8% |
| Neural network (MLP) | 76.2% (sd 9.7) | 77.9% |
| LDA | 75.5% (sd 8.4) | 79.1% |
| HistGradientBoosting | 69.7% | 72.4% |
| Random Forest | 64.7% | 70.7% |
| Extra Trees | 61.0% | 67.2% |
| k-NN | 54.9% | 60.2% |
| Decision Tree | 51.3% | 47.5% |
| Naïve Bayes | 36.8% | 50.8% |

### The finding worth defending

**The two simplest models won.** Linear SVM and Logistic Regression beat the RBF
SVM and the neural network, and they were *more consistent* across sessions
(standard deviation 8.9 vs 10.3).

**Why:** the 85 features already do the hard work. By the time a model sees them,
the chords are close to linearly separable. The more powerful models have spare
capacity and spend it memorising session-specific quirks — one person's camera
angle, one room's lighting — instead of learning chord shape. The linear models
cannot overfit that way, so they generalise better.

**This is the argument for the whole project design:** good feature engineering made
the complex models unnecessary.

The tree models collapsing fits the same story — axis-aligned splits are a poor
match for continuous geometric quantities like joint angles.

### Why Logistic Regression and not Linear SVM

Linear SVM scored 0.3 points higher on unseen sessions, but:

1. **It has no `predict_proba`.** [`realtime.py`](realtime.py) needs probabilities for
   its live confidence display — shipping LinearSVC would break the demo.
2. **It was worse on an unseen person** (80.2% vs 80.6%), which is the measure that
   matters more.

So `train.py` now tries **logistic regression, RBF SVM and MLP**, and picks the best
by leave-one-session-out. On the final dataset logistic regression won at **78.9%**,
and **69.1%** on a held-out player across all nine chords.

> **Note on the three different cross-person numbers you will see:**
> - **69.1%** - the deployed 9-chord model on a whole unseen player. **Quote this one.**
> - **85.2%** - a 5-class model on sessions containing only 5 chords. Easier task.
> - **80.6%** - the figure in the table above, produced by the comparison script,
>   which held all 11 algorithms to one identical protocol so the *ranking* is fair.
>
> All three are real; they answer different questions. The table ranks algorithms,
> it does not state the system's accuracy.

---

## 8. Why 5 chords are excluded

Excluded: **Am, Bm, Cm, Fm, Gm**. Kept: A, B, C, D, Dm, E, Em, F, G.

### The reason is structural, not a data shortage

Our features deliberately ignore **where on the neck** the hand is. That buys
robustness to camera position, but it costs the ability to tell apart chords that
are *the same shape at a different fret*:

| pair | voicings | relationship |
|---|---|---|
| Bm / Cm | x24432 / x35543 | identical A-shape minor barre, 2 frets apart |
| Fm / Gm | 133111 / 355333 | identical E-shape minor barre, 2 frets apart |
| Am / E | x02210 / 022100 | same 3-finger shape, moved one string |

Position-blind features make these literally indistinguishable. **No algorithm can
recover information that was never in the data.**

### The evidence

Re-measured on the final two-person dataset (the old figures came from the
single-player era and did not survive):

| scheme | classes | unseen session | unseen person | x chance |
|---|---|---|---|---|
| all 14 | 14 | 68.1% | 57.1% | 5.71x |
| merge the identical pairs | 11 | **80.5%** | 67.8% | **6.78x** |
| the 9 plus Bm | 10 | 78.4% | 65.0% | 6.50x |
| **drop these 5 (adopted)** | **9** | 79.1% | **69.1%** | 6.22x |

> **Be ready for this one.** The merged scheme now *beats* our adopted set on two of
> three measures. We still chose 9, for two reasons: it has the **highest raw
> cross-person accuracy** (69.1 vs 67.8), and its answers are **unambiguous** — a
> merged class can only say "Bm or Cm" without saying which, which is useless to a
> learner. Say this before they find it; the merged scheme is a legitimate
> alternative and pretending otherwise looks worse than admitting it.

### Why this costs the project nothing

Every letter A–G is still covered as a major, plus Dm and Em. The scope was "detect
chords A through G" and that is fully met.

### Also worth knowing

The closest pair in the current 9-chord set is **C–Dm at 2.08** in scaled feature
space — closer than B–Bm (3.44). That is very likely why C has the weakest f1
(.519). This was found late and is honest limitations material.

---

## 9. The Bm story

This is the best "we did real science" story in the project. Tell it.

**Act 1 — Bm looked impossible.** Every attempt to include Bm crashed cross-person
accuracy to 54–57%, against 69.8% without it. Worse, B major collapsed with it —
recall fell to 12–19%.

**Act 2 — we found the real cause.** It wasn't the chord. A groupmate had recorded
Bm with the *wrong voicing*. We were teaching the model two different hand shapes
under one label. A mislabeled voicing produces near-0% recall — that signature is
what gave it away. The bad frames were deleted.

**Act 3 — a second player recorded it correctly.** With a correct x24432 barre from
a second person:

| | 9 chords | 10 chords (+Bm) |
|---|---|---|
| unseen session | 76.8% | **78.1%** |
| unseen person | **69.8%** | 66.9% |
| unseen person, × chance | 6.28× | **6.69×** |
| **B major recall** | 66% | **72%** |

**The collapse is gone.** B major now gets *better* when Bm is added. The original
failure was our data bug, not a property of the chord.

**Act 4 — we still ship 9, and here's why.** Bm's own recall is only **36.1%**. Its
errors go to B (23.9%), D (20.0%) and A (13.3%) — all shapes it genuinely resembles
once neck position is removed. A chord that fails two times in three is not
demo-safe, and we don't need it: all seven letters are already covered.

**Would more Bm photos fix it? No — measured:**

| Bm frames in training | Bm recall |
|---|---|
| 122 | 29.4% |
| 245 | 27.6% |
| 367 | 29.4% |
| 490 | 33.3% |

4× the data buys ~4 points. The curve is flat → **structural limit, not a data
shortage.** (Caveat: this varied frames from the *same* people. A third person's Bm
is a different question and remains untested.)

---

## 10. Known weaknesses — say these before they ask

Volunteering these makes you look rigorous. Hiding them and getting caught does the
opposite.

1. **Only 2 people.** The single biggest limitation. We measured that going from 1
   to 2 people gained 9 points cross-person — which strongly implies person 3 would
   help again. This is the top future-work item.
   *A third contributor did attempt to record, but on a **ukulele** (4 strings) rather
   than a guitar (6). Every chord voicing in the dataset is a six-string shape, so
   those 199 frames were rejected rather than merged — see weakness 10.*
2. **Position-blind features.** The root cause of all 5 exclusions and of Bm's 36%.
   **The fix is known:** add a feature encoding hand position along the neck. That
   would break all these collisions at once. It is the highest-value next step.
3. **Em is the weakest chord** — f1 .649, and 55.5% on an unseen player. But do NOT
   call this structural: we thought it was, re-recorded it properly, and it gained
   **7.3 points** (.576 to .649), the largest single-chord gain in the project. The
   ceiling was higher than the first measurement implied. It is still the weakest,
   and more good data would likely help again.
4. **C is weak** — f1 .676, and only **46.1%** on an unseen player, the worst of any
   chord there. Caused by the C-Dm proximity (2.08, the closest pair in the set).
   Unlike Em, this one *is* structural - it comes from the features, not the sample.
5. **Sensitivity to a change in recording setup.** The second player's E major scores
   **83.1%** on his original session but only about **47%** on his four newer ones,
   even though those are better lit and better captured. Same player, same chord,
   different camera setups. This is the clearest evidence in the project that the
   model is still partly fitting recording conditions rather than pure hand geometry.
   Volunteer it - it is an honest limitation and it points straight at the fix (more
   varied setups, not more frames).

6. **F major's voicing is the wrong trade-off now — and you found it live.**
   F is recorded as mini F (`xx3211`), not the full barre (`133211`). That was
   chosen to avoid colliding with **Fm**, which differs by one finger and sat
   within 1.0 in feature space. **But Fm is excluded from the model.** The
   collision we paid to avoid no longer exists; the cost still does — mini F is a
   compact 4-string shape and so is D (`xx0232`), so they get confused.
   Evidence: F is f1 **.900** overall but only **54.4%** recall on an unseen
   player, and **F read as D is 113 frames** in the held-out matrix, the third
   largest error cell. Live testing on 2026-11-10 reproduced it.
   **Why we didn't fix it:** changing a voicing invalidates all 956 existing F
   frames across 19 sessions, and re-recording them could not be done in the
   remaining time without creating a thin single-person class — the exact failure
   that broke Bm. Documented as future work instead.
   **If asked "why not just use a real F?"** — that is the right question, and the
   answer is in Section VII of the paper. A voicing choice is only valid relative
   to the chord set it sits in; ours changed underneath it.

7. **Lighting sensitivity.** One contributor's recordings averaged 35.8 brightness
   against our 87.2 — 2.4× darker. His F detected at only 63.4% where his other
   chords hit 95–100%. **MediaPipe needs to see the hand.** Honest framing: this is a
   data-collection discipline issue, and it is also a real deployment constraint.
8. **Some frames are near-duplicates.** 1,248 frames were captured at ~6 per second
   instead of the intended 0.7s interval — effectively one frozen pose. Removing them
   *improved* accuracy slightly (76.8% → 77.2%), confirming they add nothing.
9. **No audio.** We classify hand shape, not sound. We cannot tell whether the chord
   was actually strummed, or strummed cleanly.
10. **One voicing per chord.** Real players use several. We standardised on one each
   (listed in `config.py`) because mixing voicings under one label is exactly the bug
   that broke Bm.
11. **Repository size.** ~1.9 GB because the image dataset is version-controlled.
   Workable, but not how you'd ship this.
12. **Label integrity is the project's recurring failure mode.** Twice, data arrived
    that was labelled correctly but was not the thing the label described: once a Bm
    played with the wrong voicing (section 9), once an entire batch recorded on a
    ukulele instead of a guitar. Both were caught, neither reached the final model —
    but both cost time, and the second was only caught by visual inspection, not by
    any automated check.
    **The lesson, and a genuine future-work item:** a dataset built by several
    contributors needs a validation step at intake. A simple one would work — check
    the instrument has six strings, and flag any chord whose features sit
    implausibly far from that chord's existing centroid. Say this before the panel
    finds it; it shows you understand *why* the errors happened, not just that they
    did.

---

## 11. Likely panel questions

**"Why not use sound? Audio chord recognition already exists."**
> It does, and it's a different problem. Audio tells you what was played; we tell you
> what the hand is doing. That matters for a learner who is forming a shape before
> strumming, and it still works in a noisy room or on an unplugged electric.

**"Why only 9 chords when there are 14?"**
> Five are the same hand shape at a different fret. Our features are deliberately
> blind to fret position, so they are structurally indistinguishable — we measured
> 68.1% with all 14 versus 79.1% with 9. All seven letters A–G are still covered.

**"79% doesn't sound very accurate."**
> Against an 11.1% chance level that's nearly 7× chance, and our textbook puts the
> usability threshold at 70%. More importantly, we could have reported 95% by using a
> random split — but consecutive frames are near-identical, so that number would have
> been fiction. We chose the honest measurement.

**"How do you know it isn't just memorising you?"**
> We removed one person's recordings entirely, trained without them, and tested on
> them: 69.1% across all nine chords, which is 6.22x chance. We also measured that
> adding a second person raised our cross-person accuracy by 9
> points, which is exactly what you'd expect if the earlier version *had* been partly
> memorising.

**"Which algorithm did you use, and why?"**
> We compared 11. Linear models won — Linear SVM at 79.2%, logistic regression at
> 78.9% — and both beat the neural network. We ship logistic regression. That's because the 85 engineered features
> already make the classes nearly linearly separable, so the complex models only had
> spare capacity to overfit with.

**"Your demo confused F with D. Why?"**
> Because our F uses a reduced four-string voicing that is geometrically similar
> to D. We chose it to avoid a collision with F minor — but F minor was later
> excluded from the model, so that justification no longer holds while its cost
> does. It is measured: F reads as D in 113 frames of the held-out matrix. The
> fix is the full barre voicing, which spans six strings and is distinct from D.
> We did not make the change because it invalidates all 956 existing F frames and
> re-recording them in the time available would have produced a single-person
> class — the same flaw that forced us to exclude B minor.

**"What would you do with more time?"**
> Three things, in order: more people; a feature encoding position along the neck,
> which would unlock all five excluded chords at once; and standardised lighting
> guidance for contributors.

**"Is this real-time?"**
> Yes. MediaPipe runs per frame, the feature computation is 85 arithmetic operations,
> and the classifier is a single matrix multiply. We smooth predictions over a 7-frame
> window to stop flicker.

---

## 12. Textbook alignment (Maheshwari)

*Data Analytics Made Accessible* (2020), the course reference.

**Chapter 4 defines CRISP-DM's six steps** — our README and code follow them in
order. Two lines are worth quoting:

> *"A host of modeling tools and algorithms should be used."* (Step 4, Modeling)

That is the textbook **requiring** the 11-algorithm comparison in section 7.

> *"It is better to triangulate the analysis by applying multiple data mining
> techniques, and conducting many what-if scenarios."* (Step 5, Evaluation)

That is the Bm scheme comparison in section 9.

> *"In practice, predictive models with more than 70% accuracy can be considered
> usable in business domains."* (Chapter 4)

**We clear this on both tests.** Cite it when asked whether 76–80% is good.

### Every classification chapter in the book is covered

| chapter | our result |
|---|---|
| Ch 7 — Regression / Logistic Regression | 78.9% |
| Ch 13 — Support Vector Machines (incl. the Kernel Method) | 79.2% linear / 76.4% RBF |
| Ch 8 — Artificial Neural Networks | 76.2% |
| Ch 6 — Decision Trees | 51.3% |
| Ch 12 — Naïve Bayes | 36.8% |

The remaining chapters (Clustering, Association Rules, Text Mining, Web Mining,
Social Network Analysis) are unsupervised or not applicable to classification.

### Where we exceed the book — be ready for this

The book never mentions **cross-validation**; its evaluation is a simple train/test
split. We use grouped leave-one-session-out plus a held-out person. If asked why:

> A random split would put near-identical frames from one sitting on both sides, and
> the accuracy would be fiction. Grouping by session prevents that.

⚠️ **Terminology:** the book defines **confidence interval** in Ch. 18, which is a
different thing from predictive accuracy. Say **accuracy**, not "confidence level."

---

## 13. Glossary

| term | meaning |
|---|---|
| **landmark** | one of 21 points MediaPipe returns on a hand |
| **feature** | one of the 85 numbers we compute from those points |
| **session / group** | one sitting at the camera; one run of `record_data.py` |
| **leave-one-session-out** | train on all sessions but one, test on that one, repeat |
| **held-out person** | one person's sessions removed entirely before training |
| **recall** | of all the real X frames, what share did we catch? |
| **precision** | of everything we *called* X, what share really was X? |
| **f1-score** | the balance of precision and recall, 0 to 1 |
| **× chance** | accuracy ÷ random-guess rate; lets you compare different class counts |
| **chance level** | 1 ÷ number of classes. 9 chords → 11.1% |
| **confusion matrix** | grid of what got predicted as what; the diagonal is correct |
| **structural collision** | two chords our features genuinely cannot distinguish |
| **voicing** | the specific fret/finger pattern for a chord, e.g. B = x24442 |
| **overfitting** | learning quirks of the training data instead of the real pattern |
| **standard deviation (sd)** | how much accuracy varies between sessions; lower is steadier |

---

## Chord voicings reference

Low E to high e. `x` = not played, `0` = open, digits = fret.

| chord | voicing | | chord | voicing |
|---|---|---|---|---|
| A | x02220 | | E | 022100 |
| B | x24442 | | Em | 022000 |
| C | x32010 | | F | xx3211 (mini F) |
| D | xx0232 | | G | 320003 |
| Dm | xx0231 | | | |

**Excluded:** Am x02210 · Bm x24432 · Cm x35543 · Fm 133111 · Gm 355333

> **Why mini F and not the full barre?** The full barre (133211) differs from Fm
> (133111) by a single finger on a single fret, putting them within 1.0 of each other
> in feature space. The mini F sits 3.88 from Fm instead.
