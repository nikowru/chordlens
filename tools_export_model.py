"""
tools_export_model.py - export the trained model for the browser demo.

CRISP-DM stage: deployment.

Writes two files:

    webapp/public/model.json   the scaler, the logistic-regression weights, the
                               class order and each chord's voicing - everything
                               the browser needs to reproduce predict_proba()
    webapp/fixture.json        five real frames with their Python-computed
                               features and probabilities, used by
                               webapp/tools_verify_port.mjs to prove the
                               JavaScript port still matches this code

Why this is safe to port
------------------------
The pipeline is StandardScaler followed by LogisticRegression, so inference is
arithmetic a browser can do exactly:

    z      = (x - mean) / scale
    logits = z @ coef.T + intercept
    proba  = softmax(logits)

This script asserts that the hand-rolled forward pass above reproduces
sklearn's own predict_proba before writing anything. If sklearn ever changes
in a way that breaks the equivalence, this fails here rather than silently in
the browser.

Run it after any retrain:

    .\\.venv\\Scripts\\python.exe tools_export_model.py
    node webapp\\tools_verify_port.mjs
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.features import extract_features, landmarks_to_array

WEBAPP = Path(__file__).resolve().parent / "webapp"
SAMPLE_CHORDS = ["A", "C", "F", "G", "Em"]


def load_pipeline():
    saved = joblib.load(config.MODELS_DIR / "chord_model.joblib")
    return saved["model"] if isinstance(saved, dict) else saved


def export_model(pipeline) -> dict:
    """Serialise the pipeline, after proving the browser's arithmetic matches."""
    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["logisticregression"]

    # Prove the port is exact before trusting it.
    probe = np.random.RandomState(0).randn(8, len(scaler.mean_))
    reference = pipeline.predict_proba(probe)
    z = (probe - scaler.mean_) / scaler.scale_
    logits = z @ classifier.coef_.T + classifier.intercept_
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    manual = exponent / exponent.sum(axis=1, keepdims=True)
    difference = float(np.abs(reference - manual).max())
    assert difference < 1e-10, f"forward pass does not match sklearn ({difference})"
    print(f"  forward pass matches sklearn to {difference:.1e}")

    return {
        "classes": [str(c) for c in classifier.classes_],
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
        "coef": [[float(v) for v in row] for row in classifier.coef_],
        "intercept": [float(v) for v in classifier.intercept_],
        "voicings": {
            name: voicing
            for name, voicing in config.CHORD_VOICINGS.items()
            if name in classifier.classes_
        },
    }


def export_fixture(pipeline) -> list:
    """Real frames with Python's features and probabilities, for the JS check."""
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(**config.STATIC_MODE_SETTINGS)
    cases = []
    for chord in SAMPLE_CHORDS:
        paths = sorted(glob.glob(str(config.RECORDINGS_DIR / chord / "*.jpg")))
        if not paths:
            continue
        image = cv2.imread(paths[len(paths) // 2])
        if image is None:
            continue
        height, width = image.shape[:2]
        result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            continue
        landmarks = landmarks_to_array(result.multi_hand_landmarks[0])
        features = extract_features(landmarks, aspect_ratio=width / height)
        cases.append({
            "label": chord,
            "aspect": width / height,
            "landmarks": [[float(v) for v in point] for point in landmarks],
            "features": [float(v) for v in features],
            "proba": [float(v) for v in pipeline.predict_proba([features])[0]],
        })
    return cases


def main():
    pipeline = load_pipeline()
    print("exporting model")
    model = export_model(pipeline)

    (WEBAPP / "public").mkdir(parents=True, exist_ok=True)
    model_path = WEBAPP / "public" / "model.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")
    print(f"  wrote {model_path}  "
          f"({len(model['classes'])} classes, {len(model['mean'])} features)")

    print("exporting verification fixture")
    cases = export_fixture(pipeline)
    fixture_path = WEBAPP / "fixture.json"
    fixture_path.write_text(json.dumps(cases), encoding="utf-8")
    print(f"  wrote {fixture_path}  ({len(cases)} cases)")

    print("\nNow run:  node webapp\\tools_verify_port.mjs")


if __name__ == "__main__":
    main()
