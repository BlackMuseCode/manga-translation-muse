#!/usr/bin/env python3
"""Plate QA after inpainting, before translation/rendering.

Two checks per page, using the final repair mask saved by
batch_inpaint.py ({page}.mask.png, white = inpainted):

1. Outside-mask integrity: the API composites only mask pixels, so
   plate == original everywhere outside the mask. Any change there
   means the response was not what it claimed (wrong image, silent
   resize, ...). Counted with a small per-channel tolerance.

2. Per-box ink coverage (adapted from the rough screen in the API's
   own validation record): for boxes on a near-flat light or dark
   background, take the high-contrast "candidate ink" pixels of the
   original and measure what fraction the final mask covers. Boxes
   under 99% are flagged — their glyphs may survive as residue in
   the plate. Mid-tone / non-flat boxes are skipped (the heuristic
   cannot judge them), exactly as in the API validation.
   Two refinements keep the screen usable on real books (first run
   on Kuro-chan c001 flagged 89/251, nearly all false positives):
   - only boxes that carry a translation are screened when
     translations.json exists (residue matters where text is typeset;
     junk detections on artwork are not plate defects);
   - ink components touching the crop edge are excluded — those are
     panel borders and art lines running through the box, the same
     exclusion the API's stroke completion uses.

Usage: check_plates.py <work_dir> <chapter>
Read-only; prints a report, always exits 0.
"""
import sys, os, json, glob
import numpy as np
from PIL import Image

COVERAGE_FLOOR = 0.99
MIN_INK_PIXELS = 20

def find_original(ch_dir, pg):
    for ext in ('jpg', 'jpeg', 'png'):
        p = os.path.join(ch_dir, f'{pg}.{ext}')
        if os.path.exists(p):
            return p
    return None

def _drop_edge_components(ink):
    """Remove ink components touching the crop border (art lines /
    panel borders), keeping interior components (glyph strokes).
    Uses scipy (cv2 is not installed on this VM — do NOT reintroduce
    a cv2 dependency here; a silent fallback once disabled this
    filter and flooded the report with false positives).
    Calibrated on Kuro-chan c001: edge filter ON, no erosion ->
    4 flags / 182 boxes, worst case verified visually as a hairline
    remnant only. Eroding the ink first forgives everything (0
    flags) and is too lax; without the edge filter, tone/border
    pixels produce 56 false flags."""
    from scipy import ndimage
    labels, n = ndimage.label(ink, structure=np.ones((3, 3), int))
    if n <= 1:
        return ink
    edge = set(np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))) - {0}
    if not edge:
        return ink
    return ink & ~np.isin(labels, list(edge))

def check_page(ch_dir, out_dir, pg, only_boxes=None):
    boxes = json.load(open(os.path.join(out_dir, f'{pg}.boxes.json')))['boxes']
    orig_path = find_original(ch_dir, pg)
    if orig_path is None:
        return [f'{pg}: original image not found'], 0, 0
    orig = Image.open(orig_path).convert('RGB')
    plate = Image.open(os.path.join(out_dir, f'{pg}.inpainted.png')).convert('RGB')
    mask_path = os.path.join(out_dir, f'{pg}.mask.png')
    if not os.path.exists(mask_path):
        return [f'{pg}: no .mask.png (plate predates mask saving) — skipped'], 0, 0
    mask = np.asarray(Image.open(mask_path).convert('L'))
    if mask.shape != (orig.size[1], orig.size[0]):
        return [f'{pg}: mask size {mask.shape} != page {orig.size} — skipped'], 0, 0

    notes = []
    # 1. outside-mask integrity
    o = np.asarray(orig, dtype=np.int16)
    p = np.asarray(plate, dtype=np.int16)
    outside = mask < 128
    changed = int((np.abs(o - p).max(axis=2)[outside] > 2).sum())
    if changed:
        notes.append(f'{pg}: {changed} px changed OUTSIDE the mask')

    # 2. per-box ink coverage on flat backgrounds
    gray = np.asarray(orig.convert('L'), dtype=np.float32)
    screened = flagged = 0
    for i, b in enumerate(boxes):
        if only_boxes is not None and i not in only_boxes:
            continue
        crop = gray[b['y1']:b['y2'], b['x1']:b['x2']]
        if crop.size == 0:
            continue
        median = float(np.median(crop))
        if median >= 200:
            ink = crop < median - 50
        elif median <= 55:
            ink = crop > median + 50
        else:
            continue  # mid-tone background: cannot judge
        if float(np.mean(np.abs(crop - median) <= 12)) < 0.5:
            continue  # not near-flat (artwork / gradient): cannot judge
        ink = _drop_edge_components(ink)
        n_ink = int(ink.sum())
        if n_ink < MIN_INK_PIXELS:
            continue
        screened += 1
        covered = float(np.mean(mask[b['y1']:b['y2'], b['x1']:b['x2']][ink] >= 128))
        if covered < COVERAGE_FLOOR:
            flagged += 1
            notes.append(f'{pg} box{i}: ink coverage {covered * 100:.2f}% '
                         f'({n_ink} ink px) — possible residue')
    return notes, screened, flagged

def main():
    work_dir, chapter = sys.argv[1], sys.argv[2]
    ch_dir = os.path.join(work_dir, chapter)
    out_dir = os.path.join(ch_dir, 'out')
    pages = sorted(os.path.basename(p)[:-len('.boxes.json')]
                   for p in glob.glob(os.path.join(out_dir, '*.boxes.json')))
    # Screen only translated boxes when translations exist (see header).
    trans_path = os.path.join(ch_dir, 'translations.json')
    translated = None
    if os.path.exists(trans_path):
        trans = json.load(open(trans_path))
        translated = {pg: {e['box'] for e in entries}
                      for pg, entries in trans.items()}
    all_notes, screened, flagged = [], 0, 0
    for pg in pages:
        only = translated.get(pg) if translated is not None else None
        notes, s, f = check_page(ch_dir, out_dir, pg, only_boxes=only)
        all_notes += notes
        screened += s
        flagged += f
    for n in all_notes:
        print(' ', n, flush=True)
    print(f'Plate QA {chapter}: {len(pages)} pages, {screened} boxes screened, '
          f'{flagged} flagged', flush=True)

if __name__ == '__main__':
    main()
