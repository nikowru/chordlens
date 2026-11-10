// Builds the IEEE-style conference paper as a .docx.
// Format modelled on the two TUP Manila reference papers (layout only).
const fs = require("fs");
const path = require("path");
const d = require("docx");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  SectionType, PageOrientation, ImageRun,
} = d;

// ---------------------------------------------------------------- numbers
const N = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));

const COL = 4326;                 // usable width of one column, DXA
const FONT = "Times New Roman";

// ---------------------------------------------------------------- helpers
const p = (text, opt = {}) => new Paragraph({
  alignment: opt.align || AlignmentType.JUSTIFIED,
  spacing: { after: opt.after === undefined ? 120 : opt.after, line: 240 },
  indent: opt.indent === undefined ? { firstLine: 283 } : opt.indent,
  children: [new TextRun({ text, font: FONT, size: opt.size || 20, bold: !!opt.bold, italics: !!opt.italics })],
});

const rich = (runs, opt = {}) => new Paragraph({
  alignment: opt.align || AlignmentType.JUSTIFIED,
  spacing: { after: opt.after === undefined ? 120 : opt.after, line: 240 },
  indent: opt.indent === undefined ? { firstLine: 283 } : opt.indent,
  children: runs.map(r => new TextRun({
    text: r.t, font: FONT, size: opt.size || 20, bold: !!r.b, italics: !!r.i,
  })),
});

const h1 = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 240, after: 120 },
  keepNext: true,
  children: [new TextRun({ text, font: FONT, size: 20, bold: true, allCaps: true })],
});

const h2 = (text) => new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { before: 120, after: 60 },
  keepNext: true,
  indent: { firstLine: 283 },
  children: [new TextRun({ text, font: FONT, size: 20, italics: true })],
});

const caption = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 60, after: 180 },
  children: [new TextRun({ text, font: FONT, size: 16 })],
});

// ---------------------------------------------------------------- figures
// A figure is a centred image paragraph tied by keepNext to the caption below
// it, so the two never split across a column break.
//
// COL is 4326 DXA = 216.3 pt = 288 px at the 96 dpi that docx-js assumes for
// ImageRun transformations. Anything wider overflows the column.
const FIG_W = 286;

// PNG intrinsic size, read from the IHDR chunk, so an aspect ratio is measured
// from the file rather than guessed and no figure is ever squashed.
function pngSize(file) {
  const b = fs.readFileSync(file);
  if (b.readUInt32BE(0) !== 0x89504e47) throw new Error(`${file} is not a PNG`);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function figure(file, capText, width = FIG_W) {
  const abs = path.join(__dirname, file);
  const { w, h } = pngSize(abs);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 0 },
      keepNext: true,
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(abs),
        transformation: { width, height: Math.round((width * h) / w) },
      })],
    }),
    caption(capText),
  ];
}

// a compact table that fits inside one column
function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = COL / total;
  const w = widths.map(x => Math.round(x * scale));
  const cell = (text, i, opt = {}) => new TableCell({
    width: { size: w[i], type: WidthType.DXA },
    shading: opt.head ? { type: ShadingType.CLEAR, fill: "E8E8E8" } : undefined,
    margins: { top: 40, bottom: 40, left: 60, right: 60 },
    children: [new Paragraph({
      alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      spacing: { after: 0, line: 220 },
      children: [new TextRun({ text: String(text), font: FONT, size: 16, bold: !!opt.head || !!opt.bold })],
    })],
  });
  return new Table({
    columnWidths: w,
    width: { size: COL, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4 }, bottom: { style: BorderStyle.SINGLE, size: 4 },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2 }, insideVertical: { style: BorderStyle.NONE },
    },
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((t, i) => cell(t, i, { head: true })) }),
      ...rows.map(r => new TableRow({ children: r.map((t, i) => cell(t, i, { bold: String(t).startsWith("*") })) })),
    ],
  });
}

// ---------------------------------------------------------------- title block
const AUTHORS = [
  ["ARBOLEDA, Nicolas Angelo T.", "nicolasarboleda222@gmail.com"],
  ["ANDAM, Jamir Dwayne C.", "andamjamirdwayne@gmail.com"],
  ["NARAWA, Ryu A.", N.ryu_email],
];

const titleBlock = [
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [new TextRun({ text: "ChordLens: Real-Time Guitar Chord Recognition from Hand", font: FONT, size: 36, bold: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: "Landmarks Using Classical Machine Learning", font: FONT, size: 36, bold: true })],
  }),
];

const authorTable = new Table({
  columnWidths: [3120, 3120, 3120],
  width: { size: 9360, type: WidthType.DXA },
  borders: {
    top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
    left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
    insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE },
  },
  rows: [new TableRow({
    children: AUTHORS.map(([name, email]) => new TableCell({
      width: { size: 3120, type: WidthType.DXA },
      children: [
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: name, font: FONT, size: 20, bold: true })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: "Technological University of the Philippines – Manila", font: FONT, size: 18 })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: "College of Science – Computer Studies", font: FONT, size: 18 })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: "Manila, Philippines", font: FONT, size: 18 })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: email, font: FONT, size: 18 })] }),
      ],
    })),
  })],
});

// ---------------------------------------------------------------- body
const body = [];
const S = (x) => body.push(x);
const SFIG = (...args) => figure(...args).forEach(S);

const N_CHORDS = N.perchord_rows.length;
const WORD = ["zero", "one", "two", "three", "four", "five", "six", "seven",
  "eight", "nine", "ten"];

// --- Abstract
S(rich([
  { t: "Abstract—", b: true },
  { t: `Learning guitar requires a beginner to verify that a chord shape is correct, which normally depends on an instructor. This study presents ChordLens, a real-time system that recognises which guitar chord a player is fretting using only a consumer webcam, with no audio input and no specialised hardware. Rather than classifying raw pixels, the system uses MediaPipe Hands to reduce each frame to twenty-one hand landmarks, from which eighty-five translation- and scale-invariant geometric features are engineered. A total of ${N.images_total} frames were captured across ${N.sessions} recording sessions by two players, yielding ${N.rows_total} usable feature vectors at a ${N.detection_rate} hand-detection rate. Eleven classical classifiers were compared under leave-one-session-out cross-validation. Multinomial logistic regression achieved the highest accuracy at ${N.loso}, outperforming both a radial-basis-function support vector machine (${N.svm_loso}) and a multilayer perceptron (${N.mlp_loso}); on a player entirely withheld from training it reached ${N.person_all9} across all nine chords, or ${N.person_x} the ${N.chance} chance level. Notably, the two linear models outperformed every non-linear alternative, indicating that the engineered feature set renders the classes close to linearly separable and that additional model capacity is spent on session-specific artefacts rather than chord structure. Five chords were deliberately excluded after measurement showed them to be structurally indistinguishable under position-invariant features. The system retains full coverage of all seven natural-note roots and runs in real time on a laptop CPU without a GPU.` },
], { indent: { firstLine: 283 } }));

S(rich([
  { t: "Keywords—", b: true, i: true },
  { t: "hand landmark detection, guitar chord recognition, MediaPipe, feature engineering, logistic regression, CRISP-DM, computer vision" },
], { indent: { firstLine: 283 } }));

// --- I. Introduction
S(h1("I. Introduction"));
S(p("Learning to play the guitar presents a verification problem for the beginner. A chord is a specific arrangement of fingers across strings and frets, and a learner who has placed a finger on the wrong string frequently cannot tell from sound alone whether the error lies in the shape, in the pressure applied, or in the strumming hand. Resolving this ambiguity usually requires an instructor, which is precisely the resource a self-taught learner lacks."));
S(p("Existing automatic chord recognition systems operate almost exclusively on audio. These systems answer the question of what was played, but they cannot answer the question a learner actually asks, which is whether the hand is in the right position. An audio system is silent until the strings are struck, offers no guidance while a shape is being formed, and degrades in the noisy environments where practice typically occurs."));
S(p("This study takes the complementary approach of observing the fretting hand directly. The central design decision is that the classifier never sees a photograph. Each frame is reduced by MediaPipe Hands to twenty-one three-dimensional landmarks, and those landmarks are converted into eighty-five engineered geometric features. This decision has three consequences that shape the entire project. First, the data requirement falls dramatically, because the system does not need to learn what a hand looks like under varying illumination, skin tone, or background. Second, every input is interpretable, so a confusion between two chords can be traced to a specific geometric similarity rather than attributed to an opaque representation. Third, the resulting model is small enough to run in real time on a laptop CPU."));
S(p("The contributions of this paper are as follows. An eighty-five-feature geometric representation of the fretting hand is defined and justified. Eleven classical classifiers are compared under a grouped cross-validation protocol designed to prevent the near-duplicate leakage that video-derived datasets invite. The finding that linear models outperform non-linear ones on this representation is reported and explained. Finally, a measured, rather than assumed, account is given of which chords this representation can and cannot separate."));

// --- II. Background
S(h1("II. Background"));
S(h2("A. Hand Landmark Detection"));
S(p("MediaPipe Hands [1], [2] is a machine learning pipeline that localises a hand within an image and regresses twenty-one keypoints corresponding to the wrist, the knuckles, and the finger joints. Each landmark carries normalised image coordinates together with a relative depth estimate. The pipeline was trained on a large and diverse corpus, and in this study it is treated as a fixed upstream component rather than something to be retrained, which is what allows a dataset of twelve thousand frames to be sufficient."));
S(p("MediaPipe additionally reports a handedness label with an associated confidence. This study exploits that label in an unexpected way: because the label is a property of the camera viewpoint rather than of the physical hand, a systematic disagreement with the expected label identifies a mirrored image, which permits automatic correction of datasets captured from the opposite viewpoint."));

S(h2("B. Chord Voicings and Structural Ambiguity"));
S(p("A guitar chord is realised as a voicing, a specific assignment of fingers to frets across the six strings. Two voicings may be geometrically identical yet denote different chords when played at different positions along the neck. The A-shape minor barre, for instance, produces B minor at the second fret and C minor at the third; the hand shape is the same and only its position differs. This property is central to the results reported in Section V."));

S(h2("C. The CRISP-DM Process"));
S(p("The Cross-Industry Standard Process for Data Mining [3], [4] organises a data mining project into business understanding, data understanding, data preparation, modeling, evaluation, and deployment. The present work follows this structure, and Section IV is organised accordingly. The standard explicitly directs that a range of modeling tools and algorithms be applied and that analysis be triangulated across multiple techniques; the eleven-algorithm comparison in Section V is a direct response to that guidance."));

// --- III. Review of Related Studies
S(h1("III. Review of Related Studies"));
S(p("Automatic chord recognition from audio is a mature field. Humphrey and Bello [5] applied convolutional neural networks to chord recognition from spectrogram representations, and McFee and Bello [6] subsequently addressed the large-vocabulary case through structured training. These systems achieve strong performance on recorded music but share the limitation identified in Section I: they describe the sound produced rather than the hand that produced it, and therefore cannot assist a learner before a chord is struck."));
S(p("Vision-based approaches to instrument interaction have more commonly targeted fingering transcription from high-resolution video of the fretboard, typically requiring a fixed camera, controlled lighting, and often fiducial markers placed on the instrument. Such arrangements are effective in a laboratory but impractical for a learner at home."));
S(p("Hand landmark representations have been applied with considerable success to sign language recognition and general gesture classification, where the same insight applies: an upstream pose estimator trained on large corpora reduces a visually complex problem to a low-dimensional geometric one, after which classical classifiers suffice. The present work applies that strategy to guitar chords, a domain in which the hand shapes are more constrained than in sign language but, as Section V demonstrates, more systematically confusable."));

// --- IV. Methodology
S(h1("IV. Methodology"));
S(p("The project follows the six phases of CRISP-DM. Each subsection below corresponds to one phase and to the specific scripts implementing it."));
SFIG(
  "reports/fig1_pipeline.png",
  "Fig. 1. The recognition pipeline. MediaPipe Hands is the last stage that sees pixels; everything after it operates on geometry alone.",
  240,
);

S(h2("A. Business Understanding"));
S(p("The objective is to determine, from a single webcam frame and in real time, which of a defined set of guitar chords the fretting hand is forming. The requirement set at the outset was coverage of all seven natural-note roots. Audio is explicitly out of scope, as the aim is to assess hand position rather than acoustic output."));

S(h2("B. Data Understanding"));
S(p(`All data were collected by the authors. A custom capture tool writes each frame with a label and a session identifier encoded in the filename, where a session denotes one continuous sitting at the camera. This identifier is essential to the evaluation protocol described in Section IV-E. In total, ${N.images_total} frames were captured across ${N.sessions} sessions by two players using ordinary laptop webcams under uncontrolled domestic lighting. MediaPipe successfully located a hand in ${N.rows_total} of these, a detection rate of ${N.detection_rate}.`));
S(p("Two data quality issues emerged during collection and are reported here because both materially affected the results. In one session the mean frame brightness was approximately 2.4 times lower than in the remainder of the corpus, and the detection rate for one chord in that session fell to 63.4 per cent against 95 to 100 per cent for the same contributor's other chords. Separately, a subset of frames was captured by holding the manual shutter rather than using the timed interval, producing approximately six frames per second; four chords thereby accumulated 350 frames each within sixty seconds, which represents a single held pose rather than the intended range of positions. Removing these frames altered accuracy by less than one point in either direction, confirming that they contributed no information. Both problems were subsequently addressed by re-recording the four affected chords across four separate sessions under improved illumination, which raised mean frame brightness from 35.8 to 65.5 and lifted hand detection for those chords to 100 per cent. The effect on accuracy is reported in Section V-B."));

S(h2("C. Data Preparation"));
S(p("Each detected hand yields twenty-one landmarks, which are converted into eighty-five features in three groups. Sixty features are normalised landmark coordinates: every point is expressed relative to the wrist and divided by the distance from the wrist to the middle-finger knuckle, after which the wrist itself is discarded as it is identically zero. Ten features are the pairwise Euclidean distances among the five fingertips. Fifteen features are the flexion angles at the finger joints."));
SFIG(
  "reports/fig2_landmarks.png",
  "Fig. 2. The twenty-one MediaPipe landmarks located on the fretting hand for four chords, each panel a single frame from the corpus. The tablature above each panel gives the voicing recorded.",
);
S(p("This construction makes the representation invariant to where the hand appears in the frame and to its apparent size, so that camera distance and hand size do not affect the result. Three invariances are deliberately withheld. Rotation is retained because the inclination of the hand carries genuine information about the chord. Mirror symmetry is retained because a left and a right hand forming the same shape are distinct configurations. Most consequentially, position along the neck is not encoded, as the representation is purely intrinsic to the hand. This last choice confers robustness to camera placement at the cost described in Section V-C."));

S(h2("D. Modeling"));
S(p("Eleven classifiers were evaluated: multinomial logistic regression, linear support vector machine, radial-basis-function support vector machine [7], multilayer perceptron, linear discriminant analysis, histogram gradient boosting, random forest, extremely randomised trees, k-nearest neighbours, decision tree, and Gaussian naive Bayes. All were implemented with scikit-learn [8]."));
S(p("Every classifier is wrapped in a pipeline whose first stage standardises the features. Placing the scaler inside the pipeline rather than applying it once to the whole dataset is methodologically necessary: it is refitted on each training fold, so no information from the test fold influences the scaling."));

S(h2("E. Evaluation"));
S(p("The evaluation protocol is the most consequential methodological choice in this study. The capture tool records a frame approximately every 0.7 seconds, so consecutive frames within a session are near-duplicates. Under a conventional random train-test split, near-identical frames would be distributed across both partitions, the model would be assessed on images it had effectively already seen, and the resulting accuracy would be substantially inflated and without meaning."));
S(p("Two grouped protocols are therefore used. Under leave-one-session-out cross-validation, the session identifier defines the group, and each session is held out in turn while the model trains on all others, so a sitting is never split. Under the held-out-player protocol, all sessions belonging to one player are removed before any training occurs, and the trained model is evaluated on that player alone. The latter answers the question of practical interest, namely whether the system functions for a person it has never observed."));
S(p("Because schemes with differing numbers of classes are not directly comparable, accuracies are additionally reported as a multiple of the chance level."));

S(h2("F. Deployment"));
S(p("The selected model is serialised together with its label ordering and applied to a live webcam stream. Predictions are smoothed over a seven-frame window to suppress flicker during transitions between shapes. The complete pipeline runs on a laptop CPU without GPU acceleration."));

// --- V. Results
S(h1("V. Results and Discussion"));

S(h2("A. Comparison of Classifiers"));
S(p("Table I reports all eleven classifiers under leave-one-session-out cross-validation and on the held-out player."));
S(caption("TABLE I. CLASSIFIER COMPARISON"));
S(table(
  ["Classifier", "Session", "Player"],
  N.algo_rows,
  [2100, 1100, 1100],
));
S(p(`Multinomial logistic regression attained the highest accuracy at ${N.loso}, with a linear support vector machine 0.3 points behind. Both linear models outperformed the radial-basis-function support vector machine and the multilayer perceptron, and both exhibited lower variance across sessions, indicating steadier behaviour on unseen recording conditions.`));
S(p("This ordering is the principal finding of the modeling phase and merits explanation. The eighty-five features are not raw measurements but engineered geometric quantities that have already normalised away translation and scale. By the point at which a classifier receives them, the classes are close to linearly separable. The additional capacity of a kernel machine or a neural network therefore has no genuine structure left to exploit and is instead expended on artefacts particular to individual sessions, such as one room's illumination or one participant's habitual camera angle. The linear models cannot represent such artefacts and consequently generalise better. The poor performance of the tree-based methods is consistent with this account, as axis-aligned splits are a poor match for continuous geometric quantities such as joint angles."));
S(p("Logistic regression rather than the linear support vector machine was selected for deployment, despite the latter's marginally higher cross-validation score, because it provides calibrated class probabilities required by the real-time interface and because it performed better on the held-out player."));

S(h2("B. Final Model Performance"));
S(p(`The deployed model attains ${N.loso} under leave-one-session-out cross-validation, exceeding the seventy per cent threshold that Maheshwari [3] identifies as the point of practical usability. On a player withheld entirely from training it attains ${N.person_all9} across all nine chords against a ${N.chance} chance level, a factor of ${N.person_x}. The latter figure warrants a note on protocol. Where the withheld sessions contain only a subset of the chord vocabulary, restricting the classifier to the classes actually present yields a higher figure, 85.2 per cent over five chords; that number describes a five-class model rather than the deployed nine-class one, and is reported here only for comparison with Table III. The ${N.person_all9} figure describes the system as deployed and is the one carried forward. Per-chord results appear in Table II.`));
SFIG(
  "reports/confusion_matrix.png",
  `Fig. 3. Confusion matrix under leave-one-session-out cross-validation over ${N.sessions} sessions (${N.loso}). Each cell is shaded by the fraction of that chord's frames it accounts for, so the diagonal reads as recall.`,
);
S(caption("TABLE II. PER-CHORD PERFORMANCE"));
S(table(
  ["Chord", "Prec.", "Rec.", "F1"],
  N.perchord_rows,
  [900, 800, 800, 800],
));
S(p("Performance is uneven across chords. B major, F major and E major are recognised reliably, with F-measures of 0.906, 0.900 and 0.818 respectively. E minor remains weakest at 0.649, followed by C major at 0.676 and A major at 0.716. Analysis of the feature space offers a partial explanation: C major and D minor form the closest pair in the adopted chord set, separated by a distance of 2.08 in standardised feature space, which is smaller than the separation between B major and B minor despite the latter differing by only a single finger. This proximity was not anticipated during design and accounts for a substantial share of the error affecting C major."));
SFIG(
  "reports/fig4_per_chord.png",
  "Fig. 4. Per-chord precision and recall under leave-one-session-out cross-validation, the values tabulated in Table II. B major and F major are the most reliably recognised; E minor is the weakest.",
);
S(p("E minor warrants separate comment, because an initial reading of the evidence proved wrong. It recorded the weakest F-measure in the corpus at 0.576, and part of its sample belonged to the rapidly captured subset described in Section IV-B, which left two competing explanations: that E minor is intrinsically difficult under this representation, or that its sample was simply of poorer quality. Rather than adopt the former, the four affected chords were re-recorded across four fresh sessions under improved illumination. E minor rose to 0.649 and G major from 0.723 to 0.787, the two largest single-chord gains observed. The limitation was therefore one of data quality and not of representation. E minor remains the weakest chord, but its ceiling is evidently higher than the first measurement implied."));

S(h2("C. The Effect of Player Diversity"));
S(p("The dataset was initially collected by a single player. Introducing a second player produced the largest single improvement observed in this study, as shown in Table III."));
S(caption("TABLE III. EFFECT OF A SECOND PLAYER. The held-out-player column uses the five chords present in those sessions; chance is therefore 20 per cent."));
S(table(
  ["Configuration", "Session", "Player"],
  N.diversity_rows,
  [2100, 1100, 1100],
));
S(p("Cross-player accuracy rose by nine points, while accuracy on unseen sessions of the same player declined marginally. A further four sessions contributed later by the same second player reduced the standard deviation across sessions from 9.3 to 7.8 and raised the worst observed session from 54.9 to 60.1 per cent, improving the consistency of the system rather than its mean. This pattern is characteristic and should not be read as degradation: a more heterogeneous dataset constitutes a harder and more honest test, and a single-player corpus flatters itself. The chords that improved most were precisely those that had been weakest, C major and D minor gaining approximately twenty points each, which is the expected signature of a model that had partially memorised one individual's hands and was compelled to learn chord geometry instead."));
SFIG(
  "reports/confusion_matrix_holdout.png",
  `Fig. 5. Confusion matrix on a player held out of training entirely (${N.person_all9} across all ${WORD[N_CHORDS]} chords). The two largest off-diagonal cells are E minor read as A major (${N.holdout_em_as_a} frames) and C major read as D minor (${N.holdout_c_as_dm} frames), both discussed in Section V-D.`,
);
S(p("The implication for future work is direct. Volume alone does not help: quadrupling the frames available for one chord from a player already represented moved its recall by approximately four points. Diversity does: a single additional player moved overall cross-player accuracy by nine."));

S(h2("D. Structurally Indistinguishable Chords"));
S(p("Five chords were excluded from the final system following measurement. The cause is not insufficient data but the deliberate omission of neck position from the feature set. B minor and C minor are the identical A-shape minor barre two frets apart; F minor and G minor are the identical E-shape minor barre two frets apart; and A minor and E major are the same three-finger configuration displaced by one string. Under a position-invariant representation these pairs are not merely difficult but formally indistinguishable."));
S(caption("TABLE IV. CHORD SET SELECTION. All schemes evaluated under one protocol over all 18 sessions for comparability; the deployed evaluation reserves two further sessions, hence the small difference from Table II."));
S(table(
  ["Scheme", "Cls", "Session", "Player"],
  N.exclusion_rows,
  [1700, 500, 1000, 1000],
));
S(p("Retaining all fourteen chords is clearly inferior, falling to 68.1 per cent on unseen sessions and 57.1 per cent on an unseen player. The remaining three schemes are closer than anticipated, and the comparison does not favour the adopted set unambiguously. Merging each indistinguishable pair into a single combined class, rather than discarding both members, attains the highest accuracy on unseen sessions at 80.5 per cent and the highest multiple of chance at 6.78, against 79.1 per cent and 6.22 for the nine-chord set. The nine-chord set is nonetheless adopted for two reasons. It achieves the highest raw accuracy on an unseen player, 69.1 against 67.8 per cent, which is the measure most representative of deployment; and its outputs are unambiguous, whereas a merged class can report only that the hand is forming either B minor or C minor without resolving which. For an application whose purpose is to tell a learner what they are playing, an ambiguous answer has limited value. The merged scheme nevertheless remains a defensible alternative and would be preferable if coverage were weighted above specificity."));
S(p("It is worth noting that this comparison differs from an earlier measurement taken when the corpus contained a single player, under which the merged scheme scored markedly worse. The reversal is itself evidence for the finding reported in Section V-C: conclusions drawn from single-player data did not survive the addition of a second contributor."));
S(p("A learning curve constructed for B minor confirms that the limitation is structural rather than a matter of sample size: increasing its training frames fourfold improved its recall by approximately four points, remaining essentially flat. When B minor is included, its own recall reaches only 36.1 per cent, with errors distributed principally to B major, D major and A major, all configurations it genuinely resembles once position is discarded."));
S(p("A noteworthy intermediate result arose during this investigation. Early experiments showed that including B minor caused B major to collapse to between 12 and 19 per cent recall. Inspection revealed that part of the B minor data had been recorded with an incorrect voicing, so that two distinct hand shapes had been trained under a single label. After the affected frames were removed and the chord was re-recorded correctly, the inclusion of B minor raised B major from 66 to 72 per cent. The original failure was therefore an artefact of label contamination rather than a property of the chord. B minor is nevertheless excluded from the deployed system on the grounds that 36.1 per cent recall is insufficient for practical use."));
S(p("The adopted nine-chord set retains all seven natural-note roots as major chords together with D minor and E minor, satisfying the coverage requirement established in Section IV-A."));

// --- VI. Conclusion
S(h1("VI. Conclusion"));
S(p(`This study demonstrates that guitar chord recognition from hand position is achievable using hand landmarks and classical machine learning, without raw-pixel training, specialised hardware, or GPU acceleration. The deployed model recognises nine chords at ${N.loso} on unseen recording sessions and ${N.person_all9} on a player never observed during training, a factor of ${N.person_x} chance.`));
S(p("Three findings are of wider interest. First, linear classifiers outperformed every non-linear alternative evaluated, indicating that well-constructed geometric features can render additional model capacity unnecessary or actively harmful. Second, player diversity proved far more valuable than data volume, with one additional contributor yielding nine points of cross-player accuracy where a fourfold increase in frames for an existing contributor yielded approximately four points for a single chord. Third, the limits of the system are traceable to an explicit and deliberate design decision rather than to opaque model behaviour: because the representation omits position along the neck, chords that share a hand shape at different positions cannot be separated, and this was confirmed by measurement rather than assumed."));

// --- VII. Recommendation
S(h1("VII. Recommendation"));
S(p("Four directions follow from these results. The most consequential is the introduction of a feature encoding hand position along the neck, which would address every structural collision identified in Section V-D simultaneously and could restore all five excluded chords. Such a feature would require a reference for neck geometry, whether through fretboard detection or a fiducial marker, and would partially sacrifice the camera-placement robustness that the present representation provides."));
S(p("Second, additional contributors should be recruited, as the evidence indicates this is the most efficient available improvement. Third, intake validation for contributed data is warranted: on two occasions during this project, data arrived correctly labelled but did not depict what the label described, once through an incorrect voicing and once through recording on a four-string instrument. Both were detected by manual inspection, and an automated check comparing submitted frames against the existing centroid for that chord would have identified both immediately. Fourth, contributors should be given explicit guidance on illumination and framing, as the brightness disparity documented in Section IV-B demonstrably reduced detection."));

// --- VIII. References
S(h1("VIII. References"));
const refs = [
  "Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., ... & Grundmann, M. (2019). MediaPipe: A framework for building perception pipelines. arXiv preprint arXiv:1906.08172.",
  "Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C. L., & Grundmann, M. (2020). MediaPipe Hands: On-device real-time hand tracking. arXiv preprint arXiv:2006.10214.",
  "Maheshwari, A. (2020). Data Analytics Made Accessible (2020 ed.).",
  "Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. Proceedings of the 4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining, 29–39.",
  "Humphrey, E. J., & Bello, J. P. (2012). Rethinking automatic chord recognition with convolutional neural networks. Proceedings of the 11th International Conference on Machine Learning and Applications, 357–362.",
  "McFee, B., & Bello, J. P. (2017). Structured training for large-vocabulary chord recognition. Proceedings of the 18th International Society for Music Information Retrieval Conference, 188–194.",
  "Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273–297.",
  "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
  "Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools, 25(11), 120–125.",
];
refs.forEach((r, i) => S(new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { after: 100, line: 220 },
  indent: { left: 283, hanging: 283 },
  children: [new TextRun({ text: `[${i + 1}] ${r}`, font: FONT, size: 16 })],
})));

// ---------------------------------------------------------------- document
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: FONT, size: 20 }, paragraph: { spacing: { line: 240 } } },
    },
  },
  sections: [
    {
      properties: {
        page: { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } },
      },
      children: [...titleBlock, authorTable, new Paragraph({ text: "", spacing: { after: 120 } })],
    },
    {
      properties: {
        type: SectionType.CONTINUOUS,
        page: { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } },
        column: { count: 2, space: 420, equalWidth: true },
      },
      children: body,
    },
  ],
});

const out = process.argv[3];
Packer.toBuffer(doc).then(b => { fs.writeFileSync(out, b); console.log("wrote " + out + " (" + (b.length / 1024).toFixed(0) + " KB)"); });
