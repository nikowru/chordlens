// features.js — JavaScript port of src/features.py.
//
// This MUST stay numerically identical to the Python version, because the model
// was trained on the Python output. tools_verify_port.js checks that claim
// against webapp/fixture.json; run it after any change here.
//
// 85 features: 60 normalised coordinates (wrist dropped) + 10 fingertip pair
// distances + 15 joint angles, in exactly that order.

export const WRIST = 0;
export const MIDDLE_FINGER_MCP = 9;
const EPSILON = 1e-8;

const FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"];
const FINGER_CHAINS = {
  thumb:  [0, 1, 2, 3, 4],
  index:  [0, 5, 6, 7, 8],
  middle: [0, 9, 10, 11, 12],
  ring:   [0, 13, 14, 15, 16],
  pinky:  [0, 17, 18, 19, 20],
};
const FINGERTIPS = {};
for (const f of FINGER_NAMES) FINGERTIPS[f] = FINGER_CHAINS[f][4];

// itertools.combinations(FINGER_NAMES, 2) — same order as Python.
const TIP_PAIRS = [];
for (let i = 0; i < FINGER_NAMES.length; i++)
  for (let j = i + 1; j < FINGER_NAMES.length; j++)
    TIP_PAIRS.push([FINGER_NAMES[i], FINGER_NAMES[j]]);

const norm = (v) => Math.hypot(v[0], v[1], v[2]);
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/**
 * Wrist-centred, hand-size-normalised landmarks.
 * @param {number[][]} landmarks 21 x [x,y,z]
 * @param {number} aspectRatio source width / height
 */
export function normaliseLandmarks(landmarks, aspectRatio = 1.0) {
  if (landmarks.length !== 21) throw new Error("expected 21 landmarks");

  // Undo MediaPipe's per-axis normalisation so x, y and z share a unit.
  const pts = landmarks.map((p) => [p[0] * aspectRatio, p[1], p[2]]);

  // Translation invariance: the wrist becomes the origin.
  const w = pts[WRIST];
  for (let i = 0; i < pts.length; i++) pts[i] = sub(pts[i], w);

  // Scale invariance: one hand size becomes one unit.
  const handSize = Math.max(norm(pts[MIDDLE_FINGER_MCP]), EPSILON);
  return pts.map((p) => [p[0] / handSize, p[1] / handSize, p[2] / handSize]);
}

/** Angle in radians at b, between bones b->a and b->c. Range [0, pi]. */
function jointAngle(a, b, c) {
  const one = sub(a, b);
  const two = sub(c, b);
  const norms = norm(one) * norm(two);
  if (norms < EPSILON) return 0.0;
  let cosine = dot(one, two) / norms;
  cosine = Math.min(1.0, Math.max(-1.0, cosine)); // guard float drift
  return Math.acos(cosine);
}

/**
 * 21 landmarks -> the 85-element feature vector.
 * @returns {number[]} length 85, ordered as FEATURE_NAMES in features.py
 */
export function extractFeatures(landmarks, aspectRatio = 1.0) {
  const pts = normaliseLandmarks(landmarks, aspectRatio);
  const out = [];

  // Group 1: normalised coordinates, wrist dropped (it is all zeros).
  for (let i = 0; i < 21; i++) {
    if (i === WRIST) continue;
    out.push(pts[i][0], pts[i][1], pts[i][2]);
  }

  // Group 2: fingertip-to-fingertip distances.
  for (const [a, b] of TIP_PAIRS) {
    out.push(norm(sub(pts[FINGERTIPS[a]], pts[FINGERTIPS[b]])));
  }

  // Group 3: three interior joint angles per finger.
  for (const finger of FINGER_NAMES) {
    const chain = FINGER_CHAINS[finger];
    for (let p = 1; p <= 3; p++) {
      out.push(jointAngle(pts[chain[p - 1]], pts[chain[p]], pts[chain[p + 1]]));
    }
  }

  if (out.length !== 85) throw new Error(`expected 85 features, got ${out.length}`);
  return out;
}

/** Scale, multiply, softmax — the exact forward pass of the sklearn pipeline. */
export function predictProba(features, model) {
  const z = features.map((v, i) => (v - model.mean[i]) / model.scale[i]);
  const logits = model.coef.map((row, k) => {
    let s = model.intercept[k];
    for (let i = 0; i < row.length; i++) s += row[i] * z[i];
    return s;
  });
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / total);
}
