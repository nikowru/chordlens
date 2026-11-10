// Verifies the JavaScript feature port against Python ground truth.
// Run:  node webapp/tools_verify_port.js
// Fails loudly if the JS and Python feature vectors or probabilities diverge.

import { readFileSync } from "fs";
import { extractFeatures, predictProba } from "./public/features.js";

const model = JSON.parse(readFileSync(new URL("./public/model.json", import.meta.url)));
const cases = JSON.parse(readFileSync(new URL("./fixture.json", import.meta.url)));

let worstFeature = 0;
let worstProba = 0;
let failures = 0;

console.log("case   chord  max|dF|       max|dP|       python -> js");
console.log("-".repeat(62));

for (const c of cases) {
  const feats = extractFeatures(c.landmarks, c.aspect);
  const proba = predictProba(feats, model);

  const dF = Math.max(...feats.map((v, i) => Math.abs(v - c.features[i])));
  const dP = Math.max(...proba.map((v, i) => Math.abs(v - c.proba[i])));
  worstFeature = Math.max(worstFeature, dF);
  worstProba = Math.max(worstProba, dP);

  const pyTop = model.classes[c.proba.indexOf(Math.max(...c.proba))];
  const jsTop = model.classes[proba.indexOf(Math.max(...proba))];
  if (pyTop !== jsTop) failures++;

  console.log(
    `${c.label.padEnd(6)} ${c.label.padEnd(6)} ${dF.toExponential(2).padEnd(12)} ` +
    `${dP.toExponential(2).padEnd(12)} ${pyTop} -> ${jsTop}${pyTop === jsTop ? "" : "  MISMATCH"}`
  );
}

console.log("-".repeat(62));
console.log(`worst feature difference    : ${worstFeature.toExponential(3)}`);
console.log(`worst probability difference: ${worstProba.toExponential(3)}`);
console.log(`label mismatches            : ${failures}`);

const ok = worstFeature < 1e-9 && worstProba < 1e-9 && failures === 0;
console.log(ok ? "\nPORT VERIFIED - JavaScript matches Python" : "\nPORT BROKEN - do not ship");
process.exit(ok ? 0 : 1);
