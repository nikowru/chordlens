"""Re-measures Table III of the paper (the effect of a second player).

Run:  .venv/Scripts/python.exe tools_diversity_check.py

Why this exists
---------------
Table III was measured once, by hand, on the 18-session dataset as it stood at
commit 8d8aae9, and the script that produced it was not kept. The stored copies
then drifted apart:

    DEFENSE_REVIEWER.md   unseen session: 77.1 / 76.4 / 78.9
    paper_numbers.json    unseen session: 77.1 / 78.9 / 79.6

Only one can be right, and the paper's own Section V-C prose ("accuracy on
unseen sessions ... declined marginally") only holds for one of them. This
script rebuilds that measurement from the era's feature table so the question
is settled by running it rather than by reading it.

Protocol, recovered from the data
---------------------------------
The held-out player is sessions 1189027a and e587eb5b. They contain exactly
five of the nine modelled chords (A, B, C, D, Dm), which is what the paper's
Table III note means by "the five chords present in those sessions"; chance is
therefore 20 per cent, not 11.1.

The second player is tatsu-06, session 79ef64ac. "One player" is the training
pool without it; "two players" is the training pool with it. The held-out
player is the same in both, so the two rows differ only in whether a second
person's recordings were available to train on — which is exactly the claim
Table III makes.

    --era      measure on the 18-session table at 8d8aae9 (reproduces Table III)
    --current  measure the same protocol on today's 22-session table

SVC is built with probability=False here. train.py sets it True because
realtime.py needs predict_proba, but that only fits an extra Platt calibration;
predict() uses the decision function either way, so accuracy is identical and
this runs minutes faster.
"""

import argparse
import pathlib
import subprocess
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.features import FEATURE_NAMES  # noqa: E402
from src import config  # noqa: E402

ERA_COMMIT = "8d8aae9"
HELD_OUT_PLAYER = ["1189027a", "e587eb5b"]
# tatsu-06. Only 79ef64ac existed when Table III was measured; the other four
# arrived on 9 November. Listing all five is correct for --current and harmless
# for --era, where the later four are simply not in the table.
SECOND_PLAYER = ["79ef64ac", "d19eac0c", "962cbb8e", "063e82a8", "caf21d07"]

MODELS = {
    "SVM (RBF)": lambda: make_pipeline(
        StandardScaler(), SVC(C=10, gamma="scale", random_state=0)),
    "Logistic regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000)),
}


def era_table():
    """The feature table as it stood when Table III was measured."""
    out = subprocess.run(
        ["git", "show", f"{ERA_COMMIT}:data/landmarks.csv"],
        cwd=ROOT, capture_output=True, check=True,
    )
    cache = ROOT / "reports" / "slides" / f"landmarks_{ERA_COMMIT}.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(out.stdout)
    return pd.read_csv(cache)


def loso(frame, factory):
    X = frame[FEATURE_NAMES].to_numpy(dtype=float)
    y = frame.label.to_numpy()
    groups = frame.group.to_numpy()
    scores = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        model = factory()
        model.fit(X[train_idx], y[train_idx])
        scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
    s = pd.Series(scores)
    return s.mean(), s.std()


def cross_player(train_pool, holdout, factory, labels):
    """train.py's protocol: restrict both sides to the chords they share."""
    shared = sorted(set(holdout.label) & set(labels))
    tr = train_pool[train_pool.label.isin(shared)]
    te = holdout[holdout.label.isin(shared)]
    model = factory()
    model.fit(tr[FEATURE_NAMES].to_numpy(dtype=float), tr.label.to_numpy())
    pred = model.predict(te[FEATURE_NAMES].to_numpy(dtype=float))
    return accuracy_score(te.label.to_numpy(), pred), shared


def run(frame, labels, title):
    frame = frame[frame.label.isin(labels)].copy()
    holdout = frame[frame.group.isin(HELD_OUT_PLAYER)].copy()
    pool = frame[~frame.group.isin(HELD_OUT_PLAYER)].copy()
    one = pool[~pool.group.isin(SECOND_PLAYER)].copy()

    print(f"\n{title}")
    print("=" * 74)
    print(f"  sessions in table      : {frame.group.nunique()}")
    print(f"  held-out player        : {', '.join(HELD_OUT_PLAYER)} "
          f"({len(holdout)} rows)")
    print(f"  training pool, 1 player: {one.group.nunique()} sessions, {len(one)} rows")
    print(f"  training pool, 2 players: {pool.group.nunique()} sessions, {len(pool)} rows")

    configs = [
        ("One player, SVM", one, MODELS["SVM (RBF)"]),
        ("Two players, SVM", pool, MODELS["SVM (RBF)"]),
        ("Two players, log. regr.", pool, MODELS["Logistic regression"]),
    ]

    print(f"\n  {'configuration':<26}{'unseen session':>16}{'unseen player':>16}")
    print("  " + "-" * 56)
    rows = []
    for name, train_pool, factory in configs:
        mean, sd = loso(train_pool, factory)
        acc, shared = cross_player(train_pool, holdout, factory, labels)
        rows.append((name, mean, sd, acc))
        print(f"  {name:<26}{100 * mean:>14.1f}%{100 * acc:>15.1f}%"
              f"   (sd {100 * sd:.1f})")
    print(f"\n  held-out player chords : {len(shared)} "
          f"({', '.join(shared)})  chance {100 / len(shared):.1f}%")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--era", action="store_true",
                    help=f"measure on the feature table at {ERA_COMMIT} (default)")
    ap.add_argument("--current", action="store_true",
                    help="also measure on today's feature table")
    args = ap.parse_args()
    labels = config.ACTIVE_LABELS

    if not args.current or args.era:
        rows = run(era_table(), labels,
                   f"AS MEASURED FOR TABLE III  (feature table at {ERA_COMMIT})")
        print("\n  stored copies of the unseen-session column:")
        print("    DEFENSE_REVIEWER.md   77.1 / 76.4 / 78.9")
        print("    paper_numbers.json    77.1 / 78.9 / 79.6")
        print("    this run              "
              + " / ".join(f"{100 * r[1]:.1f}" for r in rows))

    if args.current:
        run(pd.read_csv(config.LANDMARKS_CSV), labels,
            "SAME PROTOCOL ON TODAY'S FEATURE TABLE")


if __name__ == "__main__":
    main()
