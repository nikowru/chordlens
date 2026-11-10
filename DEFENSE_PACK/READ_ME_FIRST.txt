CHORDLENS — DEFENSE PACK
Built 10 November 2026, 19:05. Table III re-derived, all three rows. Defense: Wednesday 12 November 2026.

Everything in this folder is current. If a ChordLens file exists anywhere
else on your machine, it is older than these. Use only what is here.


WHAT TO BRING
-------------

  ChordLens_Defense.pptx   The deck. 16 slides. Speaker notes on every slide.
  ChordLens_Paper.docx     The paper. Four tables, five figures, two columns.

  The speaking script is a live doc, not a file:
  https://claude.ai/artifact/Md5NUxNncrnoXxruFchUyY

  Open it on a phone or second screen. It updates by itself, so there is
  nothing to download and no stale copy to worry about.


WHO SPEAKS WHEN
---------------

  Nicolas   slides 1-4, then 15-16      4 min + 2 min
  Jamir     slides 5-9                  5 min
  Ryu       slides 10-14                5 min

  About 15 minutes total. Leave 10-15 for questions.


THE TWO NUMBERS YOU QUOTE
-------------------------

  79.4%   on an unseen recording session (22 folds, sd 7.8)
  68.9%   on a player held out entirely, all 9 chords = 6.20x chance

  Nothing else. Other cross-player figures appear in the paper and answer
  different questions; the script doc has the wording if one comes up.


TABLE III WAS RE-DERIVED -- SAY FOURTEEN, NOT NINE
--------------------------------------------------

  All three rows of Table III are now one measurement on the 18-session
  dataset Section V-C is written against:

                             unseen session   unseen player
    One player, SVM                   74.8%           66.7%
    Two players, SVM                  76.4%           80.9%
    Two players, log. regr.           78.9%           85.2%

  Rows 2 and 3 reproduced the stored values exactly. Row 1 had been
  carried over from an older state of the data and has been replaced.

  Two things changed in what you say, and both are already in the deck
  and the paper:

    - The gain from a second player is +14.2 points, not nine.
    - Within-player accuracy barely moved: 74.8 to 76.4, under two
      points. The paper used to say it "declined marginally". Gone.

  THE ASYMMETRY IS THE FINDING. A second contributor transformed
  accuracy on a person the model had never seen, and left accuracy on
  the original player almost untouched. That is exactly what you would
  expect if the model had been fitting one person's hands rather than
  chord geometry. It is a better result than the old wording claimed.

  On provenance, if asked: the script that produced this table
  originally was not kept, and the two stored copies had drifted apart.
  These numbers are a RE-DERIVATION, not a recovery -- the protocol
  re-run on the archived feature table (tools_diversity_check.py). Say
  that plainly. It is a stronger answer than implying more certainty
  than you have.

  Full detail is in the script doc, first section.


IF YOU RUN THE LIVE DEMO
------------------------

  python realtime.py

  Both the raw per-frame guess and the smoothed label show on screen. Say
  that out loud -- it lets the panel watch the vote window working instead
  of taking your word for it.

  Press q to quit. Good light on the fretting hand matters; MediaPipe has
  to see it before anything else can happen.


REBUILDING, IF A NUMBER EVER CHANGES
------------------------------------

  Both documents generate from paper_numbers.json in the project root:

    node tools_paper_build.js paper_numbers.json ChordLens_Paper.docx
    node tools_slides_build.js paper_numbers.json ChordLens_Defense.pptx

  Never edit the .docx or .pptx by hand -- the next build overwrites it.
