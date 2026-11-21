#!/usr/bin/env python3
"""Per-box crop sheets for transcription, modelled on Saber's
AI-vision OCR flow (src/core/ocr.py): crop each detected box from the
ORIGINAL page, upscale small crops, and let the reader (here: the
translating agent) work box by box so text is bound to its box by
construction — no eyeball mapping against the full page.

Differences from Saber's version: it sends one crop per API call to a
VLM whose minimum input is ~32 px; here crops are composited into
labelled sheets for an agent to read, so the upscale floor is higher
(min side 320 px) and each cell carries a "page#box" label. The
full-page numbered overlay remains for context / missed-box checks.

Usage: make_crops.py <work_dir> <chapter>
Output: <chapter>/crops/sheet-NN.png and crops/crops.json manifest.
"""
import sys, os, json, math
from PIL import Image, ImageDraw, ImageFont

MIN_SIDE = 320        # upscale floor for the crop's shorter side
MAX_SCALE = 8.0
COLS, ROWS = 3, 4     # crops per sheet
CELL_W, CELL_H = 520, 500
LABEL_H = 30
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

def load_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()

def main():
    work_dir, chapter = sys.argv[1], sys.argv[2]
    ch_dir = os.path.join(work_dir, chapter)
    out_dir = os.path.join(ch_dir, 'out')
    crops_dir = os.path.join(ch_dir, 'crops')
    os.makedirs(crops_dir, exist_ok=True)
    font = load_font(22)

    cells = []      # (label, crop_image)
    manifest = []   # (label, page, box_index)
    pages = sorted(f[:-11] for f in os.listdir(out_dir) if f.endswith('.boxes.json'))
    for pg in pages:
        boxes = json.load(open(os.path.join(out_dir, f'{pg}.boxes.json')))['boxes']
        src = None
        for ext in ('jpg', 'jpeg', 'png'):
            p = os.path.join(ch_dir, f'{pg}.{ext}')
            if os.path.exists(p):
                src = Image.open(p).convert('RGB')
                break
        if src is None:
            print(f'  {pg}: original not found, skipped', flush=True)
            continue
        for i, b in enumerate(boxes):
            crop = src.crop((b['x1'], b['y1'], b['x2'], b['y2']))
            w, h = crop.size
            if w == 0 or h == 0:
                continue
            scale = min(MAX_SCALE, max(1.0, MIN_SIDE / min(w, h)))
            if scale > 1.0:
                crop = crop.resize((round(w * scale), round(h * scale)),
                                   Image.LANCZOS)
            label = f'{pg}#{i}'
            cells.append((label, crop))
            manifest.append({'label': label, 'page': pg, 'box': i})
        src.close()

    per_sheet = COLS * ROWS
    n_sheets = math.ceil(len(cells) / per_sheet) if cells else 0
    for s in range(n_sheets):
        sheet = Image.new('RGB', (COLS * CELL_W, ROWS * CELL_H), 'white')
        draw = ImageDraw.Draw(sheet)
        for k, (label, crop) in enumerate(cells[s * per_sheet:(s + 1) * per_sheet]):
            cx = (k % COLS) * CELL_W
            cy = (k // COLS) * CELL_H
            draw.rectangle([cx, cy, cx + CELL_W - 1, cy + CELL_H - 1],
                           outline=(180, 180, 180))
            draw.text((cx + 8, cy + 4), label, fill=(200, 0, 0), font=font)
            avail_w, avail_h = CELL_W - 16, CELL_H - LABEL_H - 12
            scale = min(avail_w / crop.size[0], avail_h / crop.size[1], 1.0)
            disp = crop if scale >= 1.0 else crop.resize(
                (round(crop.size[0] * scale), round(crop.size[1] * scale)),
                Image.LANCZOS)
            ox = cx + (CELL_W - disp.size[0]) // 2
            oy = cy + LABEL_H + (avail_h - disp.size[1]) // 2
            sheet.paste(disp, (ox, oy))
            draw.rectangle([ox - 1, oy - 1, ox + disp.size[0], oy + disp.size[1]],
                           outline=(255, 0, 0))
        name = f'sheet-{s + 1:02d}.png'
        sheet.save(os.path.join(crops_dir, name))
        for m in manifest[s * per_sheet:(s + 1) * per_sheet]:
            m['sheet'] = name
        print(f'  {name}: {min(per_sheet, len(cells) - s * per_sheet)} crops',
              flush=True)
    json.dump(manifest, open(os.path.join(crops_dir, 'crops.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f'Done {chapter}: {len(cells)} crops in {n_sheets} sheets -> {crops_dir}',
          flush=True)

if __name__ == '__main__':
    main()
