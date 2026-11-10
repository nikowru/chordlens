"""Draws the slide chrome for ChordLens_Defense.pptx as a single background PNG.

Run:  .venv/Scripts/python.exe tools_slides_bg.py

The deck's look is taken from the Canva template the team chose (maroon panel on
white, maroon sidebar, concentric arc motif, dotted grids). Geometry below was
measured off the template's rendered pages, in inches on a 13.333 x 7.5 slide.

It is baked into one image rather than built from PowerPoint shapes because
pptxgenjs slide layouts accept only rect/line/text/image objects — no ovals — so
the arcs could not live on the layout any other way. One image also keeps every
slide to a handful of real shapes.

Output: reports/slides/chrome.png
"""

import pathlib

from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "reports" / "slides"
OUT.mkdir(parents=True, exist_ok=True)

DPI = 150
W_IN, H_IN = 13.333, 7.5
W, H = int(W_IN * DPI), int(H_IN * DPI)

MAROON = (0x53, 0x1B, 0x18)
SIDEBAR = (0x6E, 0x33, 0x2E)
WHITE = (0xFF, 0xFF, 0xFF)

# --- measured from the template -------------------------------------------
NAV = (0.00, 0.51, 10.56, 0.99)        # x0, y0, x1, y1
NAV_TAB = (13.11, 0.51, 13.33, 0.99)
PANEL = (0.38, 1.44, 10.56, 6.99)
BAR = (10.56, 0.00, 13.10, 7.50)       # right sidebar
RULE = (11.82, 2.45, 6.52)             # x, y0, y1 of the sidebar's white rule


def px(v):
    return int(round(v * DPI))


def box(b):
    return [px(b[0]), px(b[1]), px(b[2]), px(b[3])]


def rings(draw, cx, cy, r0, r1, step, colour, width=2):
    """Concentric circles — the template's one repeated motif."""
    r = r0
    while r <= r1:
        draw.ellipse(
            [px(cx - r), px(cy - r), px(cx + r), px(cy + r)],
            outline=colour, width=width,
        )
        r += step


def dots(draw, x0, y0, cols, rows, gap, colour, r=0.018):
    for i in range(cols):
        for j in range(rows):
            cx, cy = x0 + i * gap, y0 + j * gap
            draw.ellipse([px(cx - r), px(cy - r), px(cx + r), px(cy + r)],
                         fill=colour)


def main():
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)

    # nav bar and the small tab that closes it on the right
    d.rectangle(box(NAV), fill=MAROON)
    d.rectangle(box(NAV_TAB), fill=MAROON)

    # dotted grid above the panel, and the arcs that run behind it
    dots(d, 0.72, 1.16, 11, 3, 0.235, MAROON)
    rings(d, 8.90, 0.95, 0.34, 2.25, 0.135, MAROON)
    rings(d, 0.18, 6.88, 0.22, 1.35, 0.135, MAROON)

    # the panel covers the middle of both motifs
    d.rectangle(box(PANEL), fill=MAROON)

    # arcs continue across the panel a shade lighter, as in the template
    over = Image.new("RGB", (W, H), MAROON)
    od = ImageDraw.Draw(over)
    rings(od, 8.90, 0.95, 0.34, 2.25, 0.135, SIDEBAR)
    rings(od, 0.18, 6.88, 0.22, 1.35, 0.135, SIDEBAR)
    panel = box(PANEL)
    img.paste(over.crop(panel), (panel[0], panel[1]))

    # right sidebar, drawn last so it sits over the panel and the arcs
    d.rectangle(box(BAR), fill=SIDEBAR)
    d.line([px(RULE[0]), px(RULE[1]), px(RULE[0]), px(RULE[2])],
           fill=WHITE, width=2)

    # dotted grid below the panel
    dots(d, 3.82, 7.06, 11, 2, 0.235, MAROON)

    path = OUT / "chrome.png"
    img.save(path, dpi=(DPI, DPI))
    print(f"wrote {path.relative_to(ROOT).as_posix()}  {W}x{H}")


if __name__ == "__main__":
    main()
