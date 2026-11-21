#!/usr/bin/env python3
"""QC review helper for the manga translation pipeline.

Manual process this codifies (AGENTS.md 2026-10-22): after machine
translation, before final packaging,
  1. draw a numbered overlay on each page's ORIGINAL so translations
     can be re-mapped against the actual Japanese in each detected box;
  2. crop every box that has NO translation, enlarged, for a second pass
     (catches skipped dialogue, sound-effect boxes that contain speech,
     and swapped/misassigned bubbles).

Conventions (same as batch_render.py):
  chapter dir/
    001.jpg  002.jpg ...        original pages
    out/001.boxes.json         {"width","height","boxes":[{"x1","y1","x2","y2","score"}]}
    translations.json          {"001":[{"box": idx, "zh": text}, ...]}

Output goes to chapter dir/qc/:
    ov_001.png                 original + numbered boxes
                               red   = has a translation
                               orange (thicker) = untranslated
    untranslated_001.png        contact sheet of 2x crops of untranslated boxes
    untranslated_index.txt     page -> box index list for the remap pass

Usage:  python3 qc_review.py <chapter_dir>
"""
import sys, os, json, math
from PIL import Image, ImageDraw, ImageFont

RED = (255, 0, 0)
ORANGE = (255, 140, 0)
WHITE = (255, 255, 255)


def label_font(scale=1):
    for path, size in [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(22 * scale)),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", int(22 * scale)),
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def draw_numbered_overlay(orig, boxes, translated_idx):
    img = orig.copy()
    dr = ImageDraw.Draw(img)
    font = label_font()
    w, h = img.size
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = (int(b[k]) for k in ("x1", "y1", "x2", "y2"))
        has = i in translated_idx
        color = RED if has else ORANGE
        width = 2 if has else 4
        dr.rectangle([x1, y1, x2, y2], outline=color, width=width)
        label = f"#{i}"
        tb = dr.textbbox((0, 0), label, font=font)
        lw, lh = tb[2] - tb[0], tb[3] - tb[1]
        lx = clamp(x1, 0, w - lw - 4)
        ly = clamp(y1 - lh - 4, 0, h - lh - 4)
        dr.rectangle([lx, ly, lx + lw + 4, ly + lh + 4], fill=color)
        dr.text((lx + 2, ly + 2), label, font=font, fill=WHITE)
    return img


def crop_sheet(orig, boxes, idxs, page_no, sheet_w=1200, scale=2, pad=24):
    """Contact sheet of enlarged crops so small boxes are readable."""
    w, h = orig.size
    crops = []
    for i in idxs:
        b = boxes[i]
        x1, y1, x2, y2 = (int(b[k]) for k in ("x1", "y1", "x2", "y2"))
        bw, bh = x2 - x1, y2 - y1
        cx1 = clamp(x1 - pad, 0, w)
        cy1 = clamp(y1 - pad, 0, h)
        cx2 = clamp(x2 + pad, 0, w)
        cy2 = clamp(y2 + pad, 0, h)
        crop = orig.crop((cx1, cy1, cx2, cy2))
        crop = crop.resize(((cx2 - cx1) * scale, (cy2 - cy1) * scale), Image.LANCZOS)
        crops.append((i, crop, x1, y1, bw, bh))
    if not crops:
        return None
    font = label_font()
    cell_pad = 10
    cols = 3
    rows = math.ceil(len(crops) / cols)
    cell_w = sheet_w // cols
    cell_h = max(c[1].size[1] for c in crops) + 40
    sheet = Image.new("RGB", (sheet_w, cell_h * rows), WHITE)
    dr = ImageDraw.Draw(sheet)
    for n, (i, crop, x1, y1, bw, bh) in enumerate(crops):
        r, c = divmod(n, cols)
        ox, oy = c * cell_w, r * cell_h
        px = ox + (cell_w - crop.size[0]) // 2
        dr.text((ox + cell_pad, oy + 4),
                f"p{page_no} #{i}  ({x1},{y1} {bw}x{bh})",
                font=font, fill=(0, 0, 0))
        sheet.paste(crop, (px, oy + 34))
        dr.rectangle([px, oy + 34, px + crop.size[0], oy + 34 + crop.size[1]],
                     outline=ORANGE, width=3)
    return sheet


def main():
    if len(sys.argv) != 2:
        print("usage: python3 qc_review.py <chapter_dir>", file=sys.stderr)
        sys.exit(1)
    ch_dir = sys.argv[1].rstrip("/")
    out_dir = os.path.join(ch_dir, "out")
    trans = json.load(open(os.path.join(ch_dir, "translations.json")))
    qc_dir = os.path.join(ch_dir, "qc")
    os.makedirs(qc_dir, exist_ok=True)

    index_lines = []
    total_untr = 0
    for pg in sorted(trans.keys()):
        orig = Image.open(os.path.join(ch_dir, f"{pg}.jpg")).convert("RGB")
        boxes_data = json.load(open(os.path.join(out_dir, f"{pg}.boxes.json")))
        boxes = boxes_data["boxes"]
        translated_idx = {t["box"] for t in trans[pg]}
        untr = [i for i in range(len(boxes)) if i not in translated_idx]

        overlay = draw_numbered_overlay(orig, boxes, translated_idx)
        overlay.save(os.path.join(qc_dir, f"ov_{pg}.png"))

        if untr:
            sheet = crop_sheet(orig, boxes, untr, pg)
            if sheet is not None:
                sheet.save(os.path.join(qc_dir, f"untranslated_{pg}.png"))
        total_untr += len(untr)
        index_lines.append(f"{pg}: {len(boxes)} boxes, {len(untr)} untranslated"
                           + (f" -> {','.join('#'+str(i) for i in untr)}" if untr else ""))

    with open(os.path.join(qc_dir, "untranslated_index.txt"), "w") as f:
        f.write("\n".join(index_lines) + "\n")
    print(f"qc done: {len(trans)} pages, {total_untr} untranslated boxes -> {qc_dir}", flush=True)


if __name__ == "__main__":
    main()
