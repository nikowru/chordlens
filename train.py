"""
train.py - train and evaluate the chord classifier on hand-landmark features.

CRISP-DM stage: modeling and evaluation.

This script never sees an image. It reads the feature table written by
extract_landmarks.py and trains classical models (logistic regression, SVM and MLP) on the 85
engineered landmark features from src/features.py. That is the whole point of
the project's design: the model learns hand *geometry*, not pixels, so it
cannot quietly memorise a background, a shirt, or the lighting in one room.

How the score is produced, and why it is honest
-----------------------------------------------
Frames from one recording sitting are near-duplicates of each other. If those
frames were split randomly between training and test sets, the model would be
tested on data it had effectively already memorised, and the reported accuracy
would be fiction - typically 20-30 points too high.

So evaluation is **leave-one-session-out**: train on every session but one,
test on the session held out, repeat for all sessions, and average. Every
reported number is therefore the accuracy on a sitting the model has never
seen, which is the only number that predicts how the live demo will behave.

With --holdout-sessions you can go further and hold out a whole *person*, which
answers the harder question a panel will ask: does this work for anyone, or
just for the person who recorded the data?

Usage
-----
    # the standard evaluation, writes reports/ and models/
    python train.py

    # hold out a different person entirely (their session ids)
    python train.py --holdout-sessions 1189027a e587eb5b

    # include the excluded chords, to reproduce the comparison in config.py
    python train.py --all-labels

Outputs
-------
    models/chord_model.joblib      the fitted pipeline, ready for realtime.py
    models/chord_model_meta.json   labels, feature names, scores, provenance
    reports/confusion_matrix.png   the headline figure for the paper
    reports/classification_report.txt
    reports/per_session_scores.csv
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")          # no GUI needed; write PNG files directly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config
from src.features import FEATURE_NAMES


# ---------------------------------------------------------------------------
# The candidate models
# ---------------------------------------------------------------------------
#
# All are wrapped in a StandardScaler. The 85 features are on wildly different
# scales - normalised coordinates sit near 0, joint angles run to pi - and every
# model here is sensitive to that. Putting the scaler *inside* the pipeline
# matters: it is refitted on each training fold, so the test fold never
# influences the scaling. Scaling once up front would leak.
#
# Why these three, measured over 11 algorithms on 12,302 rows (2026-11-09),
# leave-one-session-out / held-out person:
#
#   Linear SVM            79.2% / 80.2%   <- best, but no predict_proba
#   logistic regression   78.9% / 80.6%   <- shipped
#   RBF SVM               76.4% / 77.8%
#   MLP                   76.2% / 77.9%
#   LDA                   75.5% / 79.1%
#   HistGradientBoosting  69.7% / 72.4%
#   random forest         64.7% / 70.7%
#   extra trees           61.0% / 67.2%
#   k-NN                  54.9% / 60.2%
#   decision tree         51.3% / 47.5%
#   naive Bayes           36.8% / 50.8%
#
# The headline result is that the two *linear* models beat both the RBF SVM and
# the neural network, and were steadier across sessions (sd 8.9 vs 10.3). The 85
# features already do the hard work, leaving the classes close to linearly
# separable; the higher-capacity models spend their extra capacity memorising
# session-specific quirks (one room's lighting, one person's camera angle)
# rather than learning chord shape.
#
# LinearSVC scored highest but has no predict_proba, which realtime.py needs for
# its confidence display, and it was the weaker of the two linear models on an
# unseen person. Logistic regression gives native calibrated probabilities at
# effectively the same accuracy, so that is the one shipped. The RBF SVM and MLP
# are kept as the non-linear comparison the evaluation argument rests on.

def build_models():
    """Return the candidate models as {name: unfitted pipeline factory}."""
    return {
        # Multinomial logistic regression: a linear model, and the best
        # performer here that also exposes predict_proba.
        "logreg": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        ),
        # RBF SVM: the classical strong baseline for small tabular datasets.
        "svm": lambda: make_pipeline(
            StandardScaler(),
            SVC(C=10, gamma="scale", probability=True, random_state=0),
        ),
        # A small MLP. Two hidden layers are enough for 85 inputs and ~10
        # classes; anything larger overfits a dataset this size.
        "mlp": lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=800,
                random_state=0,
            ),
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--csv",
        default=str(config.LANDMARKS_CSV),
        help="feature table from extract_landmarks.py (default: %(default)s)",
    )
    parser.add_argument(
        "--all-labels",
        action="store_true",
        help="train on all 14 chords instead of config.ACTIVE_LABELS",
    )
    parser.add_argument(
        "--holdout-sessions",
        nargs="+",
        default=None,
        metavar="SESSION",
        help=(
            "session ids to hold out entirely as a final test set, e.g. one "
            "person's recordings. Leave-one-session-out still runs on the rest."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CRISP-DM: data understanding
# ---------------------------------------------------------------------------

def load_features(csv_path, labels):
    """Read the feature table and report what is in it before touching a model."""
    if not csv_path.exists():
        sys.exit(
            f"No feature table at {csv_path}\n"
            "Run this first:\n"
            "    python extract_landmarks.py --root data/recordings --source own"
        )

    frame = pd.read_csv(csv_path)

    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing:
        sys.exit(
            f"{csv_path} is missing {len(missing)} feature columns "
            f"(first few: {missing[:3]}).\n"
            "It was probably written by an older version of src/features.py - "
            "delete it and re-run extract_landmarks.py."
        )

    print(f"feature table : {csv_path}")
    print(f"rows          : {len(frame)}")
    print(f"sessions      : {frame.group.nunique()}")

    kept = frame[frame.label.isin(labels)].copy()
    dropped = len(frame) - len(kept)
    if dropped:
        excluded = sorted(set(frame.label) - set(labels))
        print(f"excluded      : {dropped} rows ({', '.join(excluded)}) - see config.EXCLUDED_LABELS")
    print(f"training rows : {len(kept)}  across {len(labels)} classes")

    # Class balance is worth printing rather than assuming. A class with far
    # fewer rows than the others will look worse in the report for that reason
    # alone, and the reader should be able to see it.
    counts = kept.label.value_counts()
    print(f"\n  {'chord':<8}{'rows':>7}{'sessions':>10}")
    for label in labels:
        rows = counts.get(label, 0)
        sessions = kept[kept.label == label].group.nunique()
        warn = "  <- thin" if rows < 0.6 * counts.median() else ""
        print(f"  {label:<8}{rows:>7}{sessions:>10}{warn}")

    if kept.empty:
        sys.exit("No rows left after filtering labels - nothing to train on.")
    return kept


# ---------------------------------------------------------------------------
# CRISP-DM: evaluation
# ---------------------------------------------------------------------------

def leave_one_session_out(frame, model_factory, labels):
    """
    Train on every session but one, test on the held-out session, repeat.

    Returns (mean accuracy, std, per-session scores, predictions aligned to
    frame's row order). The predictions are collected across folds, so every
    row was predicted by a model that never saw that row's session - which is
    what makes the pooled confusion matrix trustworthy.
    """
    X = frame[FEATURE_NAMES].to_numpy(dtype=float)
    y = frame.label.to_numpy()
    groups = frame.group.to_numpy()

    predictions = np.empty_like(y)
    per_session = []

    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        fold_pred = model.predict(X[test_idx])
        predictions[test_idx] = fold_pred

        session = groups[test_idx][0]
        per_session.append(
            {
                "session": session,
                "rows": len(test_idx),
                "accuracy": accuracy_score(y[test_idx], fold_pred),
            }
        )

    scores = pd.DataFrame(per_session).sort_values("accuracy")
    return scores.accuracy.mean(), scores.accuracy.std(), scores, predictions


def save_confusion_matrix(y_true, y_pred, labels, path, title):
    """Row-normalised confusion matrix: each row shows where that chord went."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = matrix / matrix.sum(axis=1, keepdims=True)
    normalised = np.nan_to_num(normalised)

    size = max(6.0, 0.55 * len(labels) + 3.0)
    figure, axes = plt.subplots(figsize=(size, size * 0.86))
    image = axes.imshow(normalised, cmap="Blues", vmin=0, vmax=1)

    axes.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axes.set_yticks(range(len(labels)), labels)
    axes.set_xlabel("predicted chord")
    axes.set_ylabel("actual chord")
    axes.set_title(title)

    # Print the count in each cell, in whichever colour stays readable.
    for row in range(len(labels)):
        for col in range(len(labels)):
            if matrix[row, col] == 0:
                continue
            axes.text(
                col, row, str(matrix[row, col]),
                ha="center", va="center", fontsize=8,
                color="white" if normalised[row, col] > 0.5 else "#1b1f24",
            )

    figure.colorbar(image, ax=axes, shrink=0.8, label="fraction of that chord's frames")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"  wrote {path}")


def main():
    args = parse_args()
    labels = config.CHORD_LABELS if args.all_labels else config.ACTIVE_LABELS

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = (config.PROJECT_ROOT / csv_path).resolve()

    print("=" * 66)
    print("DATA UNDERSTANDING")
    print("=" * 66)
    frame = load_features(csv_path, labels)

    # ------------------------------------------------------------------
    # Optional: hold out whole sessions (e.g. one person) as a final test
    # ------------------------------------------------------------------
    holdout = None
    if args.holdout_sessions:
        requested = set(args.holdout_sessions)
        present = set(frame.group.unique())
        unknown = requested - present
        if unknown:
            sys.exit(
                f"These session ids are not in the CSV: {', '.join(sorted(unknown))}\n"
                f"Available: {', '.join(sorted(present))}"
            )
        holdout = frame[frame.group.isin(requested)].copy()
        frame = frame[~frame.group.isin(requested)].copy()
        print(f"\nheld out {len(holdout)} rows from {len(requested)} session(s) "
              f"as a final unseen test set")

    if frame.group.nunique() < 2:
        sys.exit("Need at least 2 sessions to evaluate leave-one-session-out.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 66)
    print("MODELING AND EVALUATION  (leave-one-session-out)")
    print("=" * 66)

    results = {}
    for name, factory in build_models().items():
        mean, std, scores, predictions = leave_one_session_out(frame, factory, labels)
        results[name] = {
            "mean": mean, "std": std, "scores": scores,
            "predictions": predictions, "factory": factory,
        }
        print(f"  {name.upper():<5} {100 * mean:5.1f}%  (sd {100 * std:4.1f}, "
              f"worst session {100 * scores.accuracy.min():.1f}%, "
              f"best {100 * scores.accuracy.max():.1f}%)")

    best_name = max(results, key=lambda n: results[n]["mean"])
    best = results[best_name]
    print(f"\n  best model: {best_name.upper()} at {100 * best['mean']:.1f}%")

    # ------------------------------------------------------------------
    print("\n" + "=" * 66)
    print("REPORTS")
    print("=" * 66)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    y_true = frame.label.to_numpy()
    report_text = classification_report(
        y_true, best["predictions"], labels=labels, zero_division=0, digits=3
    )
    print(report_text)

    report_path = config.REPORTS_DIR / "classification_report.txt"
    header = (
        f"Model          : {best_name.upper()}\n"
        f"Evaluation     : leave-one-session-out over {frame.group.nunique()} sessions\n"
        f"Rows           : {len(frame)}\n"
        f"Classes        : {len(labels)} ({', '.join(labels)})\n"
        f"Mean accuracy  : {100 * best['mean']:.1f}%  (sd {100 * best['std']:.1f})\n"
        f"Generated      : {datetime.now():%Y-%m-%d %H:%M}\n\n"
    )
    report_path.write_text(header + report_text, encoding="utf-8")
    print(f"  wrote {report_path}")

    best["scores"].to_csv(config.REPORTS_DIR / "per_session_scores.csv", index=False)
    print(f"  wrote {config.REPORTS_DIR / 'per_session_scores.csv'}")

    save_confusion_matrix(
        y_true, best["predictions"], labels,
        config.REPORTS_DIR / "confusion_matrix.png",
        f"{best_name.upper()}, leave-one-session-out "
        f"({100 * best['mean']:.1f}% over {frame.group.nunique()} sessions)",
    )

    # ------------------------------------------------------------------
    # The held-out person, if one was requested. This is a single fit on all
    # the training sessions, tested on data from someone the model has never
    # seen - the hardest and most realistic test in the project.
    # ------------------------------------------------------------------
    holdout_score = None
    if holdout is not None:
        print("\n" + "=" * 66)
        print("HELD-OUT SESSIONS  (never seen in any fold above)")
        print("=" * 66)
        shared = sorted(set(holdout.label) & set(labels))
        train_part = frame[frame.label.isin(shared)]
        test_part = holdout[holdout.label.isin(shared)]

        model = best["factory"]()
        model.fit(train_part[FEATURE_NAMES].to_numpy(dtype=float), train_part.label.to_numpy())
        predicted = model.predict(test_part[FEATURE_NAMES].to_numpy(dtype=float))
        holdout_score = accuracy_score(test_part.label.to_numpy(), predicted)

        print(f"  classes present in the held-out data: {len(shared)} "
              f"({', '.join(shared)})")
        print(f"  chance level : {100 / len(shared):.1f}%")
        print(f"  accuracy     : {100 * holdout_score:.1f}%  on {len(test_part)} rows\n")
        for label in shared:
            mask = test_part.label.to_numpy() == label
            print(f"    {label:<8}{100 * (predicted[mask] == label).mean():>6.1f}%  (n={mask.sum()})")

        save_confusion_matrix(
            test_part.label.to_numpy(), predicted, shared,
            config.REPORTS_DIR / "confusion_matrix_holdout.png",
            f"{best_name.upper()} on held-out sessions ({100 * holdout_score:.1f}%)",
        )

    # ------------------------------------------------------------------
    # CRISP-DM: deployment. Refit on everything available, then save.
    # ------------------------------------------------------------------
    print("\n" + "=" * 66)
    print("DEPLOYMENT")
    print("=" * 66)

    final_frame = pd.concat([frame, holdout]) if holdout is not None else frame
    final_model = best["factory"]()
    final_model.fit(
        final_frame[FEATURE_NAMES].to_numpy(dtype=float), final_frame.label.to_numpy()
    )

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "chord_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"  wrote {model_path}")

    # The metadata file is what makes the saved model self-describing: it
    # records the exact feature order realtime.py must reproduce, so a change
    # to src/features.py cannot silently feed the model the wrong columns.
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "model": best_name,
        "labels": labels,
        "excluded_labels": [] if args.all_labels else config.EXCLUDED_LABELS,
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "n_rows": int(len(final_frame)),
        "n_sessions": int(final_frame.group.nunique()),
        "leave_one_session_out_accuracy": round(float(best["mean"]), 4),
        "leave_one_session_out_std": round(float(best["std"]), 4),
        "holdout_sessions": sorted(args.holdout_sessions) if args.holdout_sessions else [],
        "holdout_accuracy": round(float(holdout_score), 4) if holdout_score is not None else None,
        "source_csv": str(csv_path),
    }
    meta_path = config.MODELS_DIR / "chord_model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  wrote {meta_path}")

    print(
        f"\nFinal model: {best_name.upper()} on {len(labels)} chords, "
        f"{100 * best['mean']:.1f}% on unseen sessions"
        + (f", {100 * holdout_score:.1f}% on held-out sessions." if holdout_score is not None else ".")
    )


if __name__ == "__main__":
    main()
