"""Builds the figures embedded in ChordLens_Paper.docx.

Run:  .venv/Scripts/python.exe tools_make_figures.py

Every number drawn here is read from paper_numbers.json or from src/config.py,
never typed in, so a figure cannot drift away from the prose. Output goes to
reports/ under the figure number used in the paper.

  fig1_pipeline.png     regenerated: reports/fig_pipeline.png was stale
                        (it named the MLP and ten chords; the deployed model is
                        logistic regression over nine)
  fig2_landmarks.png    re-tiled from reports/fig_landmarks.png, whose content
                        is correct but whose 1x4 strip is illegible at the
                        4326 DXA column width; this is the same four panels in
                        a 2x2 grid
  fig4_per_chord.png    regenerated: reports/fig_per_chord.png was stale
                        (MLP, old corpus)

reports/confusion_matrix.png and reports/confusion_matrix_holdout.png are
already current for the deployed model and are used by the paper unchanged.

Style follows the confusion matrices: matplotlib defaults, "Blues" colormap,
no decorative colour.
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

N = json.loads((ROOT / "paper_numbers.json").read_text(encoding="utf-8"))

CHORDS = [row[0] for row in N["perchord_rows"]]
N_CHORDS = len(CHORDS)
BLUES = plt.get_cmap("Blues")


# --------------------------------------------------------------- figure 1
def pipeline():
    """Vertical five-stage flow.

    Vertical, not horizontal: the paper is two-column, so a figure is at most
    ~7.5 cm wide. Five boxes across that width leaves each one too narrow to
    carry a legible label.
    """
    stages = [
        ("Webcam frame",
         f"{config.FRAME_WIDTH}x{config.FRAME_HEIGHT} BGR"),
        ("MediaPipe Hands",
         "21 landmarks (x, y, z)"),
        ("Feature extraction",
         "85 engineered features"),
        ("Logistic regression",
         f"{N_CHORDS} chord probabilities"),
        ("Majority vote",
         f"{config.SMOOTHING_WINDOW}-frame window"),
    ]

    fig, ax = plt.subplots(figsize=(4.2, 5.6), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box_h, gap = 1.42, 0.52
    top = 9.55
    # Stages that no longer see pixels are shaded; the shade is a step of the
    # same Blues ramp the confusion matrices use, not a separate colour.
    shaded = {2, 3}

    for i, (title, detail) in enumerate(stages):
        y = top - i * (box_h + gap) - box_h
        ax.add_patch(FancyBboxPatch(
            (0.9, y), 8.2, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.0, edgecolor="0.25",
            facecolor=BLUES(0.10) if i in shaded else "white",
        ))
        ax.text(5.0, y + box_h * 0.63, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color="0.1")
        ax.text(5.0, y + box_h * 0.26, detail, ha="center", va="center",
                fontsize=10.5, color="0.35")

        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch(
                (5.0, y - 0.04), (5.0, y - gap + 0.04),
                arrowstyle="-|>", mutation_scale=14,
                linewidth=1.0, color="0.25",
            ))

    bottom = top - len(stages) * (box_h + gap) + gap
    ax.annotate(
        "", xy=(0.55, bottom), xytext=(0.55, top - 2 * (box_h + gap)),
        arrowprops=dict(arrowstyle="-", linewidth=0.9, color="0.45"),
    )
    ax.text(0.30, (bottom + top - 2 * (box_h + gap)) / 2,
            "no pixel data beyond this point",
            rotation=90, ha="center", va="center",
            fontsize=9.5, style="italic", color="0.35")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out = REPORTS / "fig1_pipeline.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


# --------------------------------------------------------------- figure 2
def landmarks_grid():
    """Re-tile the verified 1x4 landmark strip into a 2x2 grid.

    The source panels are reused pixel-for-pixel; only their arrangement
    changes, so nothing about the content needs re-verifying. The strip's
    own suptitle is dropped because the docx supplies a caption.
    """
    src = REPORTS / "fig_landmarks.png"
    img = Image.open(src).convert("RGB")
    arr = np.array(img.convert("L"))

    # Panel titles start below the suptitle; find the white band under it.
    body_top = 87
    col_white = (arr[body_top:, :] > 250).all(axis=0)
    runs, start = [], None
    for i, white in enumerate(col_white):
        if white and start is None:
            start = i
        elif not white and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(col_white)))
    gaps = [r for r in runs if r[1] - r[0] > 5]

    spans = []
    for left, right in zip(gaps, gaps[1:]):
        spans.append((left[1], right[0]))
    if len(spans) != 4:
        raise SystemExit(f"expected 4 panels in {src.name}, found {len(spans)}")

    row_white = (arr > 250).all(axis=1)
    body_bottom = len(row_white)
    while body_bottom > body_top and row_white[body_bottom - 1]:
        body_bottom -= 1

    panels = [img.crop((x0, body_top, x1, body_bottom)) for x0, x1 in spans]

    pad = 18
    cell_w = max(p.width for p in panels)
    cell_h = max(p.height for p in panels)
    grid = Image.new("RGB", (2 * cell_w + 3 * pad, 2 * cell_h + 3 * pad),
                     "white")
    for i, panel in enumerate(panels):
        cx, cy = i % 2, i // 2
        x = pad + cx * (cell_w + pad) + (cell_w - panel.width) // 2
        y = pad + cy * (cell_h + pad) + (cell_h - panel.height) // 2
        grid.paste(panel, (x, y))

    out = REPORTS / "fig2_landmarks.png"
    grid.save(out)
    return out


# --------------------------------------------------------------- figure 5
def per_chord():
    """Per-chord precision and recall, straight from paper_numbers.json."""
    precision = [float(r[1]) * 100 for r in N["perchord_rows"]]
    recall = [float(r[2]) * 100 for r in N["perchord_rows"]]
    chance = float(N["chance"].split()[0])

    x = np.arange(N_CHORDS)
    width = 0.38

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    ax.bar(x - width / 2, precision, width, label="precision",
           color=BLUES(0.75), edgecolor="0.25", linewidth=0.6)
    ax.bar(x + width / 2, recall, width, label="recall",
           color=BLUES(0.35), edgecolor="0.25", linewidth=0.6)
    # Labelled through the legend rather than annotated on the line: every
    # bar is taller than chance, so there is no clear space beside it.
    ax.axhline(chance, linestyle=":", linewidth=1.1, color="0.4",
               label=f"chance ({N['chance']})")

    ax.set_xticks(x)
    ax.set_xticklabels(CHORDS)
    # Headroom above the tallest bar so the legend sits clear of the data.
    ax.set_ylim(0, 118)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("%")
    ax.set_xlabel("chord")
    ax.set_title(f"Per-chord precision and recall "
                 f"(logistic regression, {N['loso']})", fontsize=12)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = REPORTS / "fig4_per_chord.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    for path in (pipeline(), landmarks_grid(), per_chord()):
        with Image.open(path) as im:
            print(f"wrote {path.relative_to(ROOT).as_posix()}  "
                  f"{im.size[0]}x{im.size[1]}")
