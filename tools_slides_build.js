// Builds the thesis defense deck, ChordLens_Defense.pptx.
//
//   npm install pptxgenjs
//   .venv/Scripts/python.exe tools_slides_bg.py        # chrome.png
//   .venv/Scripts/python.exe tools_make_figures.py     # paper figures
//   node tools_slides_build.js paper_numbers.json ChordLens_Defense.pptx
//
// Design follows the Canva template the team picked: maroon panel on white,
// maroon sidebar, concentric arc motif. That chrome is one background image
// (see tools_slides_bg.py); everything else is real PowerPoint shapes, so the
// deck stays editable.
//
// Every figure in the deck is read from paper_numbers.json, the same file the
// paper builds from, so a slide can never disagree with the manuscript.
const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");

const N = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const OUT = process.argv[3] || "ChordLens_Defense.pptx";

// ------------------------------------------------------------------ theme
const THEME = {
  name: "ChordLens",
  headFontFace: "Arial Narrow",   // condensed, stands in for the template's
  bodyFontFace: "Calibri",        // Big Shoulders; both ship with Office
  colors: {
    dk1: "2B0E0C", lt1: "FFFFFF", dk2: "531B18", lt2: "F4F0F0",
    accent1: "531B18",  // deep maroon, the dominant colour
    accent2: "8C4A43",  // mid maroon
    accent3: "C19B96",  // dusty rose, the second chart series
    accent4: "6E332E",  // sidebar maroon
    accent5: "F4F0F0",
    accent6: "9B7B78",
    hlink: "8C4A43", folHlink: "9B7B78",
  },
};

const MAROON = "531B18";
const MID = "8C4A43";
const ROSE = "C19B96";
const INK = "2B0E0C";
const W = "FFFFFF";

// ------------------------------------------------------- measured geometry
const CX = 0.78;          // content left edge, inside the maroon panel
const CW = 9.42;          // content width
const TITLE_Y = 1.70;
const KICK_Y = 2.84;
const BODY_Y = 3.28;
const BODY_H = 3.50;

const CHROME = path.join(__dirname, "reports", "slides", "chrome.png");
const FIG = (f) => path.join(__dirname, "reports", f);

const NAV = ["INTRODUCTION", "METHOD", "RESULTS", "CONCLUSION"];
const NAV_X = [0.85, 3.05, 4.95, 6.85];

const pres = new PptxGenJS();
pres.layout = "LAYOUT_WIDE";            // 13.333 x 7.5 — set before any slide
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "ARBOLEDA, ANDAM, NARAWA";
pres.company = "Technological University of the Philippines - Manila";
pres.title = "ChordLens: Real-Time Guitar Chord Recognition from Hand Landmarks";

const C = pres.SchemeColor;

// ----------------------------------------------------------------- layouts
// One layout per nav section, so the active tab lights up by itself.
function navObjects(active) {
  return NAV.map((label, i) => ({
    text: {
      text: label,
      options: {
        x: NAV_X[i], y: 0.54, w: 2.1, h: 0.42, margin: 0,
        fontFace: THEME.bodyFontFace, fontSize: 12,
        bold: i === active, color: i === active ? W : ROSE,
        charSpacing: 1.2, valign: "middle", align: "left",
      },
    },
  }));
}

function sidebarObjects() {
  return [
    {
      text: {
        text: "TUP MANILA",
        options: {
          x: 10.66, y: 1.30, w: 2.34, h: 0.34, margin: 0,
          fontFace: THEME.headFontFace, fontSize: 24, bold: true,
          color: W, align: "center", valign: "middle", charSpacing: 0.6,
        },
      },
    },
    {
      text: {
        text: "College of Science\nComputer Studies",
        options: {
          x: 10.66, y: 1.66, w: 2.34, h: 0.52, margin: 0,
          fontFace: THEME.bodyFontFace, fontSize: 10.5,
          color: ROSE, align: "center", valign: "top", lineSpacing: 13,
        },
      },
    },
    {
      text: {
        text: "CHORDLENS",
        options: {
          x: 10.66, y: 6.55, w: 2.34, h: 0.26, margin: 0,
          fontFace: THEME.bodyFontFace, fontSize: 9, bold: true,
          color: ROSE, align: "center", valign: "middle", charSpacing: 1.6,
        },
      },
    },
  ];
}

function defineLayout(name, active) {
  pres.defineSlideMaster({
    title: name,
    background: { color: W },
    objects: [
      { image: { x: 0, y: 0, w: 13.333, h: 7.5, path: CHROME } },
      ...navObjects(active),
      ...sidebarObjects(),
      {
        placeholder: {
          options: {
            name: "title", type: "title",
            x: CX, y: TITLE_Y, w: CW, h: 1.10, margin: 0,
            fontFace: THEME.headFontFace, fontSize: 66, bold: true,
            color: W, align: "left", valign: "middle",
          },
          text: "",
        },
      },
      {
        placeholder: {
          options: {
            name: "kicker", type: "body",
            x: CX, y: KICK_Y, w: CW, h: 0.30, margin: 0,
            fontFace: THEME.bodyFontFace, fontSize: 13,
            color: ROSE, align: "left", valign: "middle", charSpacing: 1.1,
          },
          text: "",
        },
      },
    ],
    slideNumber: {
      x: 10.66, y: 6.90, w: 2.34, h: 0.28,
      fontFace: THEME.bodyFontFace, fontSize: 11, bold: true,
      color: W, align: "center",
    },
  });
}

defineLayout("INTRO", 0);
defineLayout("METHOD", 1);
defineLayout("RESULTS", 2);
defineLayout("CONCLUSION", 3);

// the opening and closing slides carry the chrome but no title placeholder
pres.defineSlideMaster({
  title: "COVER",
  background: { color: W },
  objects: [
    { image: { x: 0, y: 0, w: 13.333, h: 7.5, path: CHROME } },
    ...navObjects(0),
    ...sidebarObjects(),
  ],
});

// ----------------------------------------------------------------- helpers
const num = (s) => parseFloat(String(s).replace(/[^0-9.]/g, ""));

function head(slide, title, kicker) {
  slide.addText(title, { placeholder: "title" });
  slide.addText(kicker, { placeholder: "kicker" });
}

// A white card inside the maroon panel — the template's own device for
// holding images and anything that needs to read as a separate block.
function card(slide, x, y, w, h, opt = {}) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.06,
    fill: { color: opt.fill || W },
    line: { color: opt.fill || W, width: 0 },
    objectName: opt.name || "card",
  });
}

function body(slide, text, opt = {}) {
  slide.addText(text, {
    x: opt.x === undefined ? CX : opt.x,
    y: opt.y === undefined ? BODY_Y : opt.y,
    w: opt.w === undefined ? CW : opt.w,
    h: opt.h === undefined ? 0.9 : opt.h,
    margin: 0, isTextBox: true,
    fontFace: THEME.bodyFontFace, fontSize: opt.size || 15.5,
    color: opt.color || W, align: "left", valign: opt.valign || "top",
    lineSpacing: opt.lineSpacing || (opt.size ? opt.size * 1.42 : 22),
    bold: !!opt.bold, italic: !!opt.italic,
    objectName: opt.name,
  });
}

// label over value, the deck's stat callout
function stat(slide, x, y, w, value, label, opt = {}) {
  slide.addText(value, {
    x, y, w, h: 0.92, margin: 0, isTextBox: true,
    fontFace: THEME.headFontFace, fontSize: opt.size || 54, bold: true,
    color: opt.color || W, align: "left", valign: "middle",
    objectName: `stat-${label}`,
  });
  slide.addText(label, {
    x, y: y + 0.88, w, h: 0.6, margin: 0, isTextBox: true,
    fontFace: THEME.bodyFontFace, fontSize: 12, color: opt.labelColor || ROSE,
    align: "left", valign: "top", lineSpacing: 15,
  });
}

// Calibri runs at roughly 0.47 em per character in mixed-case prose. Good
// enough to size a text box from its content instead of guessing, which is
// what kept blocks from colliding with whatever sits under them.
function estLines(text, wIn, pt) {
  const charW = (pt * 0.47) / 72;
  const perLine = Math.max(6, wIn / charW) * 0.93;   // slack for word wrap
  return text.split("\n").reduce(
    (n, para) => n + Math.max(1, Math.ceil(para.length / perLine)), 0);
}
function estH(text, wIn, pt, lineSpacing) {
  return estLines(text, wIn, pt) * (lineSpacing || pt * 1.3) / 72 + 0.06;
}

// A titled block of copy. Returns how far below `y` it reaches, so callers
// can stack blocks with a cursor rather than hard-coded offsets.
function blockCol(slide, x, y, w, title, text, opt = {}) {
  const size = opt.size || 13.5;
  const ls = opt.lineSpacing || 19;
  const h = opt.h || estH(text, w, size, ls);
  slide.addText(title, {
    x, y, w, h: 0.34, margin: 0, isTextBox: true,
    fontFace: THEME.bodyFontFace, fontSize: 14.5, bold: true,
    color: opt.titleColor || W, align: "left", valign: "middle",
    objectName: `blocktitle-${title}`,
  });
  slide.addText(text, {
    x, y: y + 0.40, w, h, margin: 0, isTextBox: true,
    fontFace: THEME.bodyFontFace, fontSize: size,
    color: opt.color || ROSE, align: "left", valign: "top",
    lineSpacing: ls,
  });
  return 0.40 + h;
}

const slides = [];
const add = (master, section) => {
  const s = pres.addSlide({ masterName: master, sectionTitle: section });
  slides.push(s);
  return s;
};

// ================================================================ 1. cover
pres.addSection({ title: "Opening" });
{
  const s = pres.addSlide({ masterName: "COVER", sectionTitle: "Opening" });
  slides.push(s);

  s.addText("THESIS DEFENSE", {
    x: CX, y: 1.72, w: CW, h: 0.3, margin: 0, isTextBox: true,
    fontFace: THEME.bodyFontFace, fontSize: 13, color: ROSE,
    charSpacing: 3, valign: "middle",
  });
  s.addText("CHORDLENS", {
    x: CX, y: 2.02, w: CW, h: 1.55, margin: 0, isTextBox: true,
    fontFace: THEME.headFontFace, fontSize: 100, bold: true,
    color: W, valign: "middle", objectName: "cover-title",
  });
  s.addText(
    "Real-Time Guitar Chord Recognition from Hand Landmarks\nUsing Classical Machine Learning",
    {
      x: CX, y: 3.62, w: 8.6, h: 0.78, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 17, italic: true,
      color: W, valign: "top", lineSpacing: 23,
    },
  );
  s.addShape(pres.ShapeType.rect, {
    x: CX, y: 4.58, w: 9.0, h: 0.015,
    fill: { color: MID }, line: { color: MID, width: 0 },
    objectName: "cover-rule",
  });
  s.addText("PRESENTED BY", {
    x: CX, y: 4.76, w: 3.0, h: 0.26, margin: 0, isTextBox: true,
    fontFace: THEME.bodyFontFace, fontSize: 11, color: ROSE, charSpacing: 2,
  });

  const authors = [
    ["ARBOLEDA, Nicolas Angelo T.", "Introduction · Conclusion"],
    ["ANDAM, Jamir Dwayne C.", "Methodology"],
    ["NARAWA, Ryu A.", "Results · Limitations"],
  ];
  authors.forEach(([name, role], i) => {
    const x = CX + i * 3.08;
    s.addText(name, {
      x, y: 5.10, w: 2.95, h: 0.3, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 13.5, bold: true, color: W,
    });
    s.addText(role, {
      x, y: 5.38, w: 2.95, h: 0.28, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 11, color: ROSE,
    });
  });

  s.addText("BSCS 3-A  ·  Data Analytics  ·  12 November 2026", {
    x: CX, y: 6.02, w: 9.0, h: 0.3, margin: 0, isTextBox: true,
    fontFace: THEME.bodyFontFace, fontSize: 11.5, color: ROSE, charSpacing: 0.8,
  });

  s.addNotes(
    "Nicolas opens. Title, the three of us, and what the system does in one line: " +
    "it watches the fretting hand through a normal webcam and names the chord, with no audio.",
  );
}

// ============================================================== 2. problem
pres.addSection({ title: "Introduction" });
{
  const s = add("INTRO", "Introduction");
  head(s, "THE PROBLEM", "WHAT A BEGINNER CANNOT CHECK ALONE");
  body(s,
    "A chord is a precise arrangement of fingers across six strings. Put one finger on the wrong " +
    "string and the guitar tells you something is wrong — but not what.",
    { h: 0.8, w: 9.3 });

  const cols = [
    ["Sound is ambiguous",
      "A wrong shape, too little pressure and a sloppy strum all sound much the same to someone still learning."],
    ["Audio arrives too late",
      "Existing recognition systems listen. They stay silent while the hand is still forming the shape — which is when help is needed."],
    ["The usual fix is a teacher",
      "Which is precisely the resource a self-taught learner does not have."],
  ];
  cols.forEach(([t, d], i) => blockCol(s, CX + i * 3.19, 4.30, 2.92, t, d, { h: 1.9 }));

  s.addNotes(
    "Nicolas. Frame the problem as verification, not transcription. The learner's question is not " +
    "'what chord was that' — it is 'is my hand right'. Nobody answers that without a teacher.",
  );
}

// =========================================================== 3. objectives
{
  const s = add("INTRO", "Introduction");
  head(s, "OBJECTIVES", "WHAT WE SET OUT TO BUILD");

  blockCol(s, CX, BODY_Y, 4.3, "General",
    "Determine, from a single webcam frame and in real time, which guitar chord the fretting " +
    "hand is forming — without audio and without specialised hardware.",
    { h: 1.5, size: 14.5, lineSpacing: 20 });

  s.addText("Specific", {
    x: 5.5, y: BODY_Y, w: 4.7, h: 0.34, margin: 0, isTextBox: true,
    fontFace: THEME.bodyFontFace, fontSize: 14.5, bold: true, color: W,
  });
  s.addText([
    { text: "Represent the fretting hand as geometry, never as pixels.", options: { bullet: true, breakLine: true } },
    { text: "Cover all seven natural-note roots, A through G.", options: { bullet: true, breakLine: true } },
    { text: "Compare classical classifiers under an evaluation that resists near-duplicate leakage.", options: { bullet: true, breakLine: true } },
    { text: "Run in real time on a laptop CPU, with no GPU.", options: { bullet: true } },
  ], {
    x: 5.5, y: BODY_Y + 0.40, w: 4.7, h: 2.5, margin: 0, isTextBox: true,
    fontFace: THEME.bodyFontFace, fontSize: 13.5, color: ROSE,
    paraSpaceAfter: 8, lineSpacing: 19,
  });

  s.addNotes(
    "Nicolas. The coverage requirement — every letter A to G — is the one Prof. Lee set, and we meet it. " +
    "Flag the last bullet: real time on a CPU is a constraint, and it drove the modelling choice.",
  );
}

// =============================================================== 4. why vision
{
  const s = add("INTRO", "Introduction");
  head(s, "WHY VISION", "TWO DECISIONS THAT SHAPED EVERYTHING");

  blockCol(s, CX, BODY_Y, 4.4, "Why not audio?",
    "Audio chord recognition is mature, but it answers what was played, not whether the hand is " +
    "right. It is silent until the strings are struck, and it degrades in exactly the noisy rooms " +
    "where people practise.",
    { h: 1.7 });
  s.addText("We watch the hand instead.", {
    x: CX, y: 5.52, w: 4.4, h: 0.3, margin: 0, isTextBox: true,
    fontFace: THEME.bodyFontFace, fontSize: 14, bold: true, italic: true, color: W,
  });

  blockCol(s, 5.80, BODY_Y, 4.4, "Why not raw pixels?",
    "A convolutional network trained on twelve thousand frames from two people would learn our " +
    "rooms, our shirts and our guitars. MediaPipe already solved hand detection on millions of " +
    "images; we add 85 numbers on top of it.",
    { h: 1.7 });
  s.addText("Smaller data, explainable errors, no GPU.", {
    x: 5.80, y: 5.52, w: 4.4, h: 0.3, margin: 0, isTextBox: true,
    fontFace: THEME.bodyFontFace, fontSize: 14, bold: true, italic: true, color: W,
  });

  s.addNotes(
    "Nicolas. This is the 'why not a CNN' question answered before it is asked. Three reasons: " +
    "our dataset is far too small for pixels, the errors stay explainable, and it runs on a laptop. " +
    "Hand off to Jamir after this slide.",
  );
}

// ============================================================= 5. pipeline
pres.addSection({ title: "Methodology" });
{
  const s = add("METHOD", "Methodology");
  head(s, "THE PIPELINE", "FROM FRAME TO CHORD IN FIVE STAGES");

  const stages = [
    ["Webcam frame", "1280 x 720"],
    ["MediaPipe Hands", "21 landmarks"],
    ["Feature extraction", "85 features"],
    ["Logistic regression", `${N.perchord_rows.length} probabilities`],
    ["Majority vote", "7-frame window"],
  ];
  const bw = 1.70, gap = 0.23, by = 3.55, bh = 1.45;
  stages.forEach(([t, d], i) => {
    const x = CX + i * (bw + gap);
    const lit = i >= 2;   // everything past MediaPipe sees only geometry
    s.addShape(pres.ShapeType.roundRect, {
      x, y: by, w: bw, h: bh, rectRadius: 0.08,
      fill: { color: lit ? W : MID },
      line: { color: lit ? W : ROSE, width: 1 },
      objectName: `stage-${i + 1}`,
    });
    s.addText(t, {
      x: x + 0.09, y: by + 0.26, w: bw - 0.18, h: 0.62, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 13, bold: true,
      color: lit ? INK : W, align: "center", valign: "middle", lineSpacing: 16,
    });
    s.addText(d, {
      x: x + 0.09, y: by + 0.86, w: bw - 0.18, h: 0.34, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 11,
      color: lit ? MID : ROSE, align: "center", valign: "top",
    });
    if (i < stages.length - 1) {
      s.addText(">", {
        x: x + bw, y: by + 0.56, w: gap, h: 0.34, margin: 0, isTextBox: true,
        fontFace: THEME.bodyFontFace, fontSize: 15, bold: true,
        color: ROSE, align: "center", valign: "middle",
      });
    }
  });

  s.addText(
    "No pixel data passes beyond MediaPipe. From the third box onward the model sees 85 numbers and nothing else.",
    {
      x: CX, y: 5.26, w: CW, h: 0.34, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 14, italic: true, color: W,
      align: "left", valign: "middle",
    });
  body(s,
    "train.py reads only the landmark CSV. It never opens an image — which is what makes this a " +
    "feature-engineering project rather than computer vision on pixels.",
    { y: 5.76, h: 0.8, size: 13.5, color: ROSE });

  s.addNotes(
    "Jamir takes over here. Walk the five boxes left to right. The sentence that matters is the one " +
    "in italics: the classifier never sees a photograph. If asked why the vote window is seven — " +
    "frames arrive about every 0.7 s, and seven smooths the transition between shapes without lag.",
  );
}

// ============================================================= 6. features
{
  const s = add("METHOD", "Methodology");
  head(s, "THE FEATURES", "85 NUMBERS THAT DESCRIBE A HAND");

  card(s, CX, BODY_Y, 4.55, 3.32, { name: "landmark-card" });
  s.addImage({
    path: FIG("fig2_landmarks.png"),
    x: CX + 0.12, y: BODY_Y + 0.12, w: 4.31, h: 3.08,
  });

  const groups = [
    ["60", "Normalised coordinates", "20 landmarks in x, y and z. The wrist is dropped — after centring it is always zero."],
    ["10", "Fingertip distances", "Every pair among the five fingertips."],
    ["15", "Joint angles", "The flexion angle at each finger joint."],
  ];
  groups.forEach(([n, t, d], i) => {
    const y = BODY_Y + i * 0.86;
    s.addText(n, {
      x: 5.70, y, w: 0.72, h: 0.46, margin: 0, isTextBox: true,
      fontFace: THEME.headFontFace, fontSize: 34, bold: true,
      color: W, align: "left", valign: "middle",
    });
    s.addText(t, {
      x: 6.46, y, w: 3.74, h: 0.3, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 14, bold: true, color: W, valign: "middle",
    });
    s.addText(d, {
      x: 6.46, y: y + 0.29, w: 3.74, h: 0.5, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 12, color: ROSE, lineSpacing: 15,
    });
  });

  s.addText(
    "Invariant to where the hand sits in the frame and to how big it is. Deliberately NOT invariant " +
    "to rotation, to mirroring — or to position along the neck.",
    {
      x: 5.70, y: 5.96, w: 4.5, h: 0.66, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 13, italic: true, color: W, lineSpacing: 18,
    });

  s.addNotes(
    "Jamir. Two normalisations: everything measured from the wrist (so position in frame stops " +
    "mattering), everything divided by the wrist-to-knuckle distance (so hand size and camera " +
    "distance stop mattering). The last line is the one to land — leaving out neck position is a " +
    "choice, and slide 13 is the bill for it.",
  );
}

// ============================================================== 7. dataset
{
  const s = add("METHOD", "Methodology");
  head(s, "THE DATASET", "RECORDED BY US, ON ORDINARY LAPTOP WEBCAMS");

  const stats = [
    [N.images_total, "frames captured"],
    [N.sessions, "recording sessions"],
    ["2", "players"],
    [N.detection_rate.replace(" per cent", "%"), "hand detection rate"],
  ];
  stats.forEach(([v, l], i) => stat(s, CX + i * 2.35, BODY_Y, 2.30, v, l, { size: 48 }));

  body(s,
    `MediaPipe found a hand in ${N.rows_total} of them. Every frame carries the id of the session ` +
    "it came from — one continuous sitting at the camera. That single column is what makes the " +
    "evaluation on the next slide possible.",
    { y: 5.00, h: 0.9, w: 9.3 });

  s.addText(
    "Captured under uncontrolled domestic lighting, which is the condition the system has to survive.",
    {
      x: CX, y: 5.96, w: 9.3, h: 0.34, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 13, italic: true, color: ROSE, valign: "middle",
    });

  s.addNotes(
    "Jamir. If asked about the gap between frames captured and rows used: MediaPipe has to actually " +
    "find a hand, and in poor light it sometimes does not. We report the detection rate rather than " +
    "hide it. Two players is our biggest limitation and we say so on slide 14.",
  );
}

// =========================================================== 8. evaluation
{
  const s = add("METHOD", "Methodology");
  head(s, "EVALUATION", "WHY WE DID NOT REPORT 95 PER CENT");

  card(s, CX, BODY_Y, 9.42, 1.14, { fill: MID, name: "trap" });
  s.addText("The trap we avoided", {
    x: CX + 0.26, y: BODY_Y + 0.14, w: 9.0, h: 0.3, margin: 0, isTextBox: true,
    fontFace: THEME.bodyFontFace, fontSize: 14, bold: true, color: W,
  });
  s.addText(
    "Our recorder saves a frame every 0.7 seconds, so consecutive frames are near-identical. A random " +
    "train-test split puts near-duplicates on both sides and the model is scored on what it has already " +
    "seen. That reads around 95 per cent and means nothing.",
    {
      x: CX + 0.26, y: BODY_Y + 0.44, w: 8.9, h: 0.6, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 12.5, color: W, lineSpacing: 16,
    });

  const pa = blockCol(s, CX, 4.76, 4.4, "Leave-one-session-out",
    `A whole sitting is always entirely in training or entirely in test, never split. ` +
    `${N.sessions} folds, one per session.`);
  const pb = blockCol(s, 5.80, 4.76, 4.4, "Held-out player",
    "One player's sessions are removed before any training happens. This answers the question that " +
    "actually matters: does it work on someone we have never recorded?");
  const protoEnd = 4.76 + Math.max(pa, pb);

  s.addText(
    `Accuracies are also reported as a multiple of chance, so schemes with different numbers of chords stay comparable.`,
    {
      x: CX, y: protoEnd + 0.14, w: 9.42, h: 0.34, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 13, italic: true, color: ROSE, valign: "middle",
    });

  s.addNotes(
    "Jamir — this is the methodological heart of the talk; do not rush it. If a panellist says 79 per " +
    "cent sounds low, the answer is here: we could have shown them 95 and it would have been fiction. " +
    "We chose the harder, honest measurement, and the textbook's usability threshold is 70.",
  );
}

// ========================================================== 9. algorithms
{
  const s = add("METHOD", "Methodology");
  head(s, "ALGORITHMS", "ELEVEN COMPARED — THE TWO SIMPLEST WON");

  // PowerPoint plots a horizontal bar chart bottom-up, so reverse the ranking
  // to put the best performer at the top of the axis.
  const rows = N.algo_rows
    .map((r) => [r[0].replace("*", ""), num(r[1]), num(r[2])])
    .reverse();
  card(s, CX, BODY_Y, 5.75, 3.34, { name: "algo-card" });
  s.addChart(pres.ChartType.bar, [
    { name: "Unseen session", labels: rows.map((r) => r[0]), values: rows.map((r) => r[1]) },
    { name: "Unseen player", labels: rows.map((r) => r[0]), values: rows.map((r) => r[2]) },
  ], {
    x: CX + 0.06, y: BODY_Y + 0.08, w: 5.63, h: 3.18,
    barDir: "bar", barGapWidthPct: 40,
    chartColors: [MAROON, ROSE],
    catAxisLabelColor: INK, catAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelColor: INK, valAxisLabelFontSize: 10, valAxisLabelFontFace: "+mn-lt",
    valAxisMinVal: 0, valAxisMaxVal: 90,
    valGridLine: { color: "E3DCDB", size: 1 },
    catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendColor: INK,
    legendFontSize: 10, legendFontFace: "+mn-lt",
    showTitle: false, showValue: false,
  });

  let cur = BODY_Y + 0.02;
  cur += blockCol(s, 6.80, cur, 3.4, "The two linear models led",
    "Logistic regression and the linear SVM both beat the RBF support vector machine and the " +
    "multilayer perceptron — and varied less from session to session.",
    { size: 12.5, lineSpacing: 17 }) + 0.18;
  blockCol(s, 6.80, cur, 3.4, "Why the simple ones win",
    "The 85 features have already done the hard work; by then the classes are nearly linearly " +
    "separable. Spare capacity goes into memorising one room's light and one person's camera angle.",
    { size: 12.5, lineSpacing: 17 });

  s.addNotes(
    "Jamir. This table ranks algorithms under one identical protocol — the numbers here are not the " +
    "deployed system's accuracy, which is on the next slide. We ship logistic regression rather than " +
    "the linear SVM because it gives calibrated probabilities, which the live demo needs, and it was " +
    "better on the held-out player. Hand off to Ryu.",
  );
}

// ============================================================= 10. results
pres.addSection({ title: "Results" });
{
  const s = add("RESULTS", "Results");
  head(s, "RESULTS", "THE TWO NUMBERS THAT MATTER");

  stat(s, CX, BODY_Y, 3.1, N.loso.replace(" per cent", "%"),
    "on an unseen recording session\nleave-one-session-out, " + N.sessions + " folds, sd 7.8",
    { size: 62 });
  stat(s, 4.15, BODY_Y, 3.1, N.person_all9.replace(" per cent", "%"),
    `on a player held out entirely\nall ${N.perchord_rows.length} chords, nothing of theirs in training`,
    { size: 62 });
  stat(s, 7.30, BODY_Y, 2.9, N.person_x.replace(" times", "x"),
    `the ${N.chance.replace(" per cent", "%")} chance level\nhow the two compare across schemes`,
    { size: 62, color: ROSE });

  card(s, CX, 5.42, 9.42, 1.2, { fill: MID, name: "context" });
  s.addText(
    "Worst session 60.1%. Best 91.9%. The course textbook puts practical usability at 70 per cent — " +
    "we clear it on both tests, and the floor sits only ten points under the mean.",
    {
      x: CX + 0.26, y: 5.58, w: 8.9, h: 0.9, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 14, color: W, lineSpacing: 19,
    });

  s.addNotes(
    "Ryu opens the results. Quote these two numbers and no others for overall accuracy. If asked about " +
    "the 85.2 per cent that appears in the paper: that is a five-class model on sessions that only " +
    "contained five chords — a real number answering an easier question. The deployed nine-chord " +
    "figure is 68.9. This is also the natural point to run the live demo.",
  );
}

// =========================================================== 11. per chord
{
  const s = add("RESULTS", "Results");
  head(s, "CHORD BY CHORD", "WHERE IT IS STRONG, AND WHERE IT IS NOT");

  const pc = N.perchord_rows;
  card(s, CX, BODY_Y, 5.95, 3.34, { name: "perchord-card" });
  s.addChart(pres.ChartType.bar, [
    { name: "Precision", labels: pc.map((r) => r[0]), values: pc.map((r) => +(num(r[1]) * 100).toFixed(1)) },
    { name: "Recall", labels: pc.map((r) => r[0]), values: pc.map((r) => +(num(r[2]) * 100).toFixed(1)) },
  ], {
    x: CX + 0.06, y: BODY_Y + 0.08, w: 5.83, h: 3.18,
    barDir: "col", barGapWidthPct: 45,
    chartColors: [MAROON, ROSE],
    catAxisLabelColor: INK, catAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelColor: INK, valAxisLabelFontSize: 10, valAxisLabelFontFace: "+mn-lt",
    valAxisMinVal: 0, valAxisMaxVal: 100,
    valGridLine: { color: "E3DCDB", size: 1 },
    catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendColor: INK,
    legendFontSize: 10, legendFontFace: "+mn-lt",
    showTitle: false, showValue: false,
  });

  let cur = BODY_Y + 0.02;
  cur += blockCol(s, 7.00, cur, 3.2, "Strong",
    "B (.906) and F (.900) are recognised reliably; E follows at .818.",
    { size: 12.5, lineSpacing: 17 }) + 0.16;
  cur += blockCol(s, 7.00, cur, 3.2, "Weak",
    "Em (.649) and C (.676) are the two that cost us.",
    { size: 12.5, lineSpacing: 17 }) + 0.16;
  blockCol(s, 7.00, cur, 3.2, "What it confuses",
    `On the held-out player Em reads as A ${N.holdout_em_as_a} times, C as Dm ` +
    `${N.holdout_c_as_dm}. C and Dm are the closest pair in the whole feature set.`,
    { size: 12.5, lineSpacing: 17 });

  s.addNotes(
    "Ryu. Em was the honest surprise. It was our worst chord at .576 and we assumed that was " +
    "structural — then we re-recorded it under better light and it gained seven points, the biggest " +
    "single-chord jump in the project. The limit was our data, not the representation. C is different: " +
    "C and Dm sit 2.08 apart in standardised feature space, closer than B and B minor, and that one " +
    "really is structural.",
  );
}

// =========================================================== 12. diversity
{
  const s = add("RESULTS", "Results");
  head(s, "DIVERSITY WINS", "ONE MORE PERSON BEAT FOUR TIMES THE DATA");

  const d = N.diversity_rows;
  const headers = ["Configuration", "Unseen session", "Unseen player"];
  const rows = [
    headers.map((t) => ({
      text: t,
      options: { bold: true, color: W, fill: { color: MID }, fontSize: 12.5 },
    })),
    ...d.map((r) => r.map((cell, i) => ({
      text: String(cell).replace("*", ""),
      options: {
        bold: String(r[0]).startsWith("*"),
        color: String(r[0]).startsWith("*") ? W : ROSE,
        fontSize: 12.5, align: i === 0 ? "left" : "center",
      },
    }))),
  ];
  s.addTable(rows, {
    x: CX, y: BODY_Y, w: 5.9, colW: [2.9, 1.5, 1.5],
    rowH: 0.36, margin: [4, 8, 4, 8],
    fontFace: THEME.bodyFontFace,
    border: { type: "solid", color: MID, pt: 0.75 },
    fill: { color: MAROON },
    objectName: "diversity-table",
  });
  s.addText(
    "Cross-player column uses the five chords those sessions contain, so chance there is 20 per cent.",
    {
      x: CX, y: BODY_Y + 1.58, w: 5.9, h: 0.40, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 11, italic: true, color: ROSE, lineSpacing: 14,
    });

  stat(s, CX, 5.42, 2.8, "+14.2", "points of cross-player accuracy\nfrom one additional player", { size: 46 });
  stat(s, 3.60, 5.42, 2.8, "+4.0", "points from quadrupling the frames\nof a player already recorded",
    { size: 46, color: ROSE });

  const consEnd = BODY_Y + 0.02 + blockCol(s, 7.00, BODY_Y + 0.02, 3.2,
    "Then consistency, not mean",
    "Four later sessions moved the average barely at all — 78.9 to 79.4 — but lifted the worst " +
    "session from 54.9 to 60.1 and narrowed the spread from 9.3 to 7.8.",
    { size: 12.5, lineSpacing: 17 });
  s.addText(
    "For a live demo, a steadier model matters more than half a point of average accuracy.",
    {
      x: 7.00, y: consEnd + 0.18, w: 3.2, h: 0.8, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 13, bold: true, italic: true,
      color: W, lineSpacing: 18,
    });

  s.addNotes(
    "Ryu. The most useful finding in the project for anyone repeating the work. Point at the " +
    "cross-player column: 66.7 to 80.9, fourteen points from one extra person. A more varied " +
    "dataset is a harder test, not an easier one, so that gain is real. The chords that gained most " +
    "were the two that had been weakest, C and Dm, about twenty points each: the signature of a " +
    "model that had been partly memorising one person's hands.",
  );
}

// =========================================================== 13. exclusions
{
  const s = add("RESULTS", "Results");
  head(s, "NINE CHORDS", "FIVE EXCLUSIONS WE CAN DEFEND");

  const pairs = [
    ["Bm / Cm", "x24432 / x35543", "same A-shape barre, two frets apart"],
    ["Fm / Gm", "133111 / 355333", "same E-shape barre, two frets apart"],
    ["Am / E", "x02210 / 022100", "same three-finger shape, one string over"],
  ];
  pairs.forEach(([p, v, d], i) => {
    const y = BODY_Y + i * 0.62;
    s.addText(p, {
      x: CX, y, w: 1.25, h: 0.3, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 14, bold: true, color: W, valign: "middle",
    });
    s.addText(v, {
      x: CX + 1.25, y, w: 1.75, h: 0.3, margin: 0, isTextBox: true,
      fontFace: "Consolas", fontSize: 11.5, color: ROSE, valign: "middle",
    });
    s.addText(d, {
      x: CX + 2.95, y, w: 2.85, h: 0.44, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 12.5, color: ROSE, valign: "middle",
    });
  });

  s.addText(
    "Our features are deliberately blind to fret position. These pairs are not difficult — they are " +
    "indistinguishable. No algorithm recovers information that was never in the data.",
    {
      x: CX, y: 5.22, w: 5.8, h: 0.86, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 13.5, italic: true, color: W, lineSpacing: 18,
    });

  const ex = N.exclusion_rows;
  const rows = [
    ["Scheme", "Cls", "Session", "Player"].map((t) => ({
      text: t, options: { bold: true, color: W, fill: { color: MID }, fontSize: 11.5 },
    })),
    ...ex.map((r) => r.map((cell, i) => ({
      text: String(cell).replace("*", ""),
      options: {
        bold: String(r[0]).startsWith("*"),
        color: String(r[0]).startsWith("*") ? W : ROSE,
        fontSize: 11.5, align: i === 0 ? "left" : "center",
      },
    }))),
  ];
  s.addTable(rows, {
    x: 6.70, y: BODY_Y, w: 3.5, colW: [1.52, 0.46, 0.76, 0.76],
    rowH: 0.32, margin: [3, 6, 3, 6],
    fontFace: THEME.bodyFontFace,
    border: { type: "solid", color: MID, pt: 0.75 },
    fill: { color: MAROON },
    objectName: "exclusion-table",
  });
  s.addText(
    "Merging the identical pairs beats us on unseen sessions. We still chose nine: it has the highest " +
    "cross-player accuracy, and its answers are unambiguous. “B minor or C minor” is no use to a learner.",
    {
      x: 6.70, y: BODY_Y + 1.82, w: 3.5, h: 1.3, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 12, color: ROSE, lineSpacing: 16,
    });

  s.addText("All seven letters A to G are still covered, plus D minor and E minor.", {
    x: CX, y: 6.24, w: 5.8, h: 0.34, margin: 0, isTextBox: true,
    fontFace: THEME.bodyFontFace, fontSize: 13.5, bold: true, color: W, valign: "middle",
  });

  s.addNotes(
    "Ryu. Volunteer the awkward row before the panel finds it: the merged scheme now beats our " +
    "adopted set on unseen sessions and on multiple of chance. It is a legitimate alternative and " +
    "pretending otherwise would look worse. We chose nine for cross-player accuracy and for " +
    "unambiguous output. If asked about B minor specifically, that is the story worth telling — " +
    "it looked impossible until we found a groupmate had recorded it with the wrong voicing.",
  );
}

// ========================================================== 14. limitations
{
  const s = add("RESULTS", "Results");
  head(s, "LIMITATIONS", "WHAT WE WOULD TELL YOU BEFORE YOU ASK");

  const items = [
    ["Only two players", "Going from one player to two gained fourteen points cross-player. A third would very likely help again."],
    ["Position-blind features", "The root cause of every exclusion. The fix is known: encode position along the neck."],
    ["F uses a reduced voicing", "xx3211 avoids colliding with F minor — which we later excluded. F now reads as D in 113 frames."],
    ["Still partly fitting the setup", "The same player's E scored 83% on one session and about 47% on four newer ones."],
    ["No audio", "We classify hand shape, not sound. We cannot say whether the chord was actually strummed."],
  ];
  items.forEach(([t, d], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = CX + col * 4.86, y = BODY_Y - 0.02 + row * 1.09;
    s.addText(`0${i + 1}`, {
      x, y, w: 0.48, h: 0.3, margin: 0, isTextBox: true,
      fontFace: THEME.headFontFace, fontSize: 19, bold: true, color: ROSE, valign: "middle",
    });
    s.addText(t, {
      x: x + 0.48, y, w: 4.0, h: 0.3, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 14, bold: true, color: W, valign: "middle",
    });
    s.addText(d, {
      x: x + 0.48, y: y + 0.29, w: 4.0, h: 0.70, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 12, color: ROSE, lineSpacing: 15,
    });
  });

  s.addText(
    "Twice, data arrived labelled correctly but showing something else — a wrong voicing, then a " +
    "ukulele. Both were caught by eye. An automatic intake check would have caught both.",
    {
      x: CX, y: 6.50, w: 9.42, h: 0.44, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 12.5, italic: true, color: W, lineSpacing: 17,
    });

  s.addNotes(
    "Ryu. Say all of these out loud. Volunteering them reads as rigour; being caught hiding one reads " +
    "as the opposite. Number three is the one we found during live testing two days ago — it is a real " +
    "mistake, honestly reported, and the reasoning behind it is in Section VII of the paper. Hand back " +
    "to Nicolas.",
  );
}

// =========================================================== 15. conclusion
pres.addSection({ title: "Conclusion" });
{
  const s = add("CONCLUSION", "Conclusion");
  head(s, "CONCLUSION", "WHAT WE SHOWED, AND WHAT COMES NEXT");

  const findings = [
    ["Simple beat complex", "Linear classifiers outperformed every non-linear alternative. Good features can make extra capacity unnecessary — or harmful."],
    ["Diversity beat volume", "One additional player was worth fourteen points cross-player; four times the frames for an existing player was worth about four."],
    ["Limits are by design", "Every failure traces to one explicit decision — omitting neck position — confirmed by measurement, not assumed."],
  ];
  findings.forEach(([t, d], i) => blockCol(s, CX + i * 3.19, BODY_Y, 2.92, t, d, { h: 1.7 }));

  card(s, CX, 5.26, 9.42, 1.36, { fill: MID, name: "future" });
  s.addText("Next", {
    x: CX + 0.26, y: 5.38, w: 1.0, h: 0.28, margin: 0, isTextBox: true,
    fontFace: THEME.bodyFontFace, fontSize: 13, bold: true, color: ROSE, charSpacing: 1.4,
  });
  s.addText([
    { text: "Encode position along the neck — it breaks every collision at once and could restore all five chords.", options: { bullet: true, breakLine: true } },
    { text: "Recruit more contributors, and validate contributed data at intake.", options: { bullet: true, breakLine: true } },
    { text: "Revisit F major's voicing now that F minor is gone.", options: { bullet: true } },
  ], {
    x: CX + 0.26, y: 5.64, w: 8.9, h: 0.9, margin: 0, isTextBox: true,
    fontFace: THEME.bodyFontFace, fontSize: 12.5, color: W,
    paraSpaceAfter: 2, lineSpacing: 16,
  });

  s.addNotes(
    "Nicolas closes. Three findings, three next steps. The honest headline: guitar chord recognition " +
    "from hand position works, on a laptop, without a GPU, and we know exactly why it fails where it does.",
  );
}

// ================================================================ 16. thanks
{
  const s = pres.addSlide({ masterName: "COVER", sectionTitle: "Conclusion" });
  slides.push(s);

  s.addText("THANK", {
    x: CX, y: 2.06, w: CW, h: 1.28, margin: 0, isTextBox: true,
    fontFace: THEME.headFontFace, fontSize: 100, bold: true, color: W, valign: "middle",
  });
  s.addText("YOU", {
    x: CX, y: 3.34, w: CW, h: 1.28, margin: 0, isTextBox: true,
    fontFace: THEME.headFontFace, fontSize: 100, bold: true, color: W, valign: "middle",
  });
  s.addText("Questions welcome.", {
    x: CX, y: 4.72, w: 5.0, h: 0.4, margin: 0, isTextBox: true,
    fontFace: THEME.bodyFontFace, fontSize: 17, italic: true, color: ROSE, valign: "middle",
  });
  s.addText(
    `${N.loso.replace(" per cent", "%")} on an unseen session  ·  ` +
    `${N.person_all9.replace(" per cent", "%")} on an unseen player  ·  ` +
    `${N.perchord_rows.length} chords  ·  ${N.person_x.replace(" times", "x")} chance`,
    {
      x: CX, y: 5.70, w: 9.42, h: 0.4, margin: 0, isTextBox: true,
      fontFace: THEME.bodyFontFace, fontSize: 13.5, bold: true, color: W,
      valign: "middle", charSpacing: 0.4,
    });
  s.addText("ARBOLEDA  ·  ANDAM  ·  NARAWA      BSCS 3-A      12 November 2026", {
    x: CX, y: 6.14, w: 9.42, h: 0.34, margin: 0, isTextBox: true,
    fontFace: THEME.bodyFontFace, fontSize: 11.5, color: ROSE, valign: "middle",
  });

  s.addNotes(
    "Leave this up for questions. The strip of numbers is there so the panel can see the headline " +
    "figures while they ask. The defense script doc has the prepared answers.",
  );
}

// ------------------------------------------------------------------ write
pres.writeFile({ fileName: OUT }).then(() => {
  const kb = (fs.statSync(OUT).size / 1024).toFixed(0);
  console.log(`wrote ${OUT} (${kb} KB, ${slides.length} slides)`);
});
