#!/usr/bin/env python3
"""Batch render translated pages using Saber's native renderer.

Layout safety net (added 2026-10-22, modelled on Saber's own layout
math in src/core/rendering.py): Saber's vertical typesetter anchors at
the box's right edge and marches columns leftward; when the text block
is wider than the box it does NOT shrink or clip — columns spill past
the left edge and glyphs drawn off-page silently vanish (this clipped
a line on Kuro-chan p019). Saber's calculate_auto_font_size is only a
rough estimator (avg char width == font size), so its "fits" verdict
can be wrong for narrow/edge boxes. Before committing each bubble we
therefore trial-render it alone on a transparent scratch page, measure
the actual ink bbox against the bubble box, and step the font size
down until the spill is within tolerance. Residual spills are logged
as OVERFLOW warnings for manual fixing (e.g. widening the box).
"""
import sys, json, os
from pathlib import Path
# saber-src/ is vendored in this repo (Saber GPL-3.0, see saber-src/LICENSE).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'saber-src'))
from types import SimpleNamespace
from PIL import Image
from src.core.rendering import render_bubbles_unified, calculate_auto_font_size

FONT = os.environ.get("MANGA_FONT", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

# --- style configuration ------------------------------------------------
# Current production values (all books shipped so far used these).
# Saber factory defaults differ: stroke ON (white, 3px) and
# inline/block align 'start'/'start'. Flip only after a visual A/B.
STROKE_ENABLED = False
STROKE_WIDTH = 3
STROKE_COLOR = "#FFFFFF"
BLOCK_ALIGN = "center"   # Saber factory: "start"
INLINE_ALIGN = "center"  # Saber factory: "start"
LINE_SPACING = 1.0
MIN_FONT = 12
MAX_FONT = 48

# --- overflow fitting ----------------------------------------------------
# Tolerated ink spill beyond the box before we shrink the font: small
# overhangs are normal in Saber's typesetting (and in the JP original).
SPILL_TOL_PX = 8
SPILL_TOL_RATIO = 0.15   # of box width (vertical spill: of box height)
FONT_FLOOR = 10          # shrink limit; below this we warn instead

def _bubble_state(text, box, font_size, direction, color="#000000"):
    return SimpleNamespace(
        translated_text=text,
        coords=(box['x1'], box['y1'], box['x2'], box['y2']),
        font_size=font_size,
        font_family=FONT,
        text_direction=direction,
        text_color=color,
        position_offset={"x": 0, "y": 0},
        rotation_angle=0,
        stroke_enabled=STROKE_ENABLED,
        stroke_width=STROKE_WIDTH if STROKE_ENABLED else 0,
        stroke_color=STROKE_COLOR,
        block_align=BLOCK_ALIGN,
        inline_align=INLINE_ALIGN,
        line_spacing=LINE_SPACING,
    )

def _ink_spill(text, box, font_size, direction, page_size):
    """Trial-render one bubble on a scratch page. Returns
    (spill, clipped): spill is how far the ink extends beyond the box
    as (left, top, right, bottom) px; clipped is True when the ink
    reaches a page edge the box does not — glyphs were cut off there,
    which is actual data loss and never tolerable."""
    scratch = Image.new('RGBA', page_size, (0, 0, 0, 0))
    render_bubbles_unified(scratch, [_bubble_state(text, box, font_size, direction)])
    ink = scratch.getchannel('A').getbbox()
    if ink is None:
        return (0, 0, 0, 0), False
    w, h = page_size
    spill = (max(0, box['x1'] - ink[0]), max(0, box['y1'] - ink[1]),
             max(0, ink[2] - box['x2']), max(0, ink[3] - box['y2']))
    clipped = ((ink[0] <= 0 and box['x1'] > 4) or (ink[1] <= 0 and box['y1'] > 4) or
               (ink[2] >= w and box['x2'] < w - 4) or (ink[3] >= h and box['y2'] < h - 4))
    return spill, clipped

def _spill_exceeds(spill, clipped, bw, bh):
    tol_x = max(SPILL_TOL_PX, SPILL_TOL_RATIO * bw)
    tol_y = max(SPILL_TOL_PX, SPILL_TOL_RATIO * bh)
    return (clipped or spill[0] > tol_x or spill[1] > tol_y or
            spill[2] > tol_x or spill[3] > tol_y)

def fit_font_size(text, box, direction, page_size, start_size):
    """Step the font size down from start_size until the trial ink
    fits the box within tolerance and nothing is clipped at a page
    edge. Returns (font_size, spill_tuple, clipped_bool)."""
    bw, bh = box['x2'] - box['x1'], box['y2'] - box['y1']
    fs = start_size
    spill, clipped = _ink_spill(text, box, fs, direction, page_size)
    while _spill_exceeds(spill, clipped, bw, bh) and fs > FONT_FLOOR:
        fs -= 2
        spill, clipped = _ink_spill(text, box, fs, direction, page_size)
    return fs, spill, clipped

def pick_text_color(img, b):
    """Black-filled bubbles (demons, narration) need white text.
    Judge by the median luminance of the box region on the inpainted
    (text-erased) page: dark fill -> white text, light fill -> black."""
    import numpy as np
    x1, y1, x2, y2 = int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])
    region = np.asarray(img.crop((x1, y1, x2, y2)).convert('L'), dtype=np.float32)
    if region.size == 0:
        return "#000000"
    return "#FFFFFF" if float(np.median(region)) < 110 else "#000000"

def is_horizontal_text(text):
    """Dates, numbers, and short Latin text should be horizontal.

    NOTE: no box-aspect rule. Saber's real auto-direction logic
    (src/core/detection.py: angle_to_direction) reads the TEXT
    LINES' angle — |angle| <= 45 deg -> horizontal, else vertical —
    it never guesses from the bounding rect's aspect, because a wide
    rect can hold vertical columns (shout bands, nameplates). A
    rect-aspect heuristic was tried 2026-10-22 and misfired on 2/183
    real entries, so it was reverted the same day."""
    import re
    t = text.replace('\n', '')
    if re.match(r'^[\d年月日\s\-/\.]+$', t):
        return True
    if len(t) <= 6 and re.match(r'^[A-Za-z0-9\s\.\!]+$', t):
        return True
    return False

def pick_direction(text, box):
    """Direction for one bubble.

    Primary: the API's per-box `angle` (change request #6, live
    2026-10-22) via Saber's angle_to_direction口径: |angle| <= 45
    -> horizontal, else vertical. This is the fitted text-line
    orientation, NOT the axis-aligned rect's aspect — on c002's
    first 19 pages the angles cluster at +-90 (vertical columns)
    with a few true slants (25-33 deg), so the signal is real.
    Content rules (numbers/dates/short Latin) still win first since
    they describe the glyphs themselves; boxes without angle data
    fall back to the old default (vertical unless content says
    horizontal)."""
    if is_horizontal_text(text):
        return 'horizontal'
    a = box.get('angle')
    if a is not None:
        return 'horizontal' if abs(a) <= 45 else 'vertical'
    return 'vertical'

def render_chapter(work_dir, chapter):
    ch_dir = os.path.join(work_dir, chapter)
    out_dir = os.path.join(ch_dir, 'out')
    render_dir = os.path.join(ch_dir, 'rendered')
    os.makedirs(render_dir, exist_ok=True)

    trans = json.load(open(os.path.join(ch_dir, 'translations.json')))
    pages = sorted(trans.keys())

    print(f"Rendering {len(pages)} pages for {chapter}", flush=True)
    for pg in pages:
        boxes_data = json.load(open(os.path.join(out_dir, f'{pg}.boxes.json')))
        img = Image.open(os.path.join(out_dir, f'{pg}.inpainted.png')).convert('RGB')

        bubbles = []
        for t in trans[pg]:
            b = boxes_data['boxes'][t['box']]
            text = t['zh']
            direction = pick_direction(text, b)
            bw, bh = b['x2'] - b['x1'], b['y2'] - b['y1']
            # auto-fit font size to bubble (was fixed 24, caused overflow)
            try:
                fs = calculate_auto_font_size(text, bw, bh, direction, FONT,
                                              min_size=MIN_FONT, max_size=MAX_FONT,
                                              padding_ratio=0.85)
            except Exception:
                fs = 24
            # Saber-estimator safety net: shrink until the real ink fits
            fs, spill, clipped = fit_font_size(text, b, direction, img.size, fs)
            if _spill_exceeds(spill, clipped, bw, bh):
                print(f"  OVERFLOW {pg} box{t['box']}: spill L{spill[0]} T{spill[1]} "
                      f"R{spill[2]} B{spill[3]} clipped={clipped} at font {fs} "
                      f"— check manually", flush=True)
            bubbles.append(_bubble_state(text, b, fs, direction,
                                         color=pick_text_color(img, b)))

        if bubbles:
            result = render_bubbles_unified(img, bubbles)
        else:
            result = img
        result.save(os.path.join(render_dir, f'{pg}.png'))
        print(f"  {pg}: {len(bubbles)} bubbles", flush=True)

    print(f"Done {chapter}", flush=True)

if __name__ == '__main__':
    render_chapter(sys.argv[1], sys.argv[2])
