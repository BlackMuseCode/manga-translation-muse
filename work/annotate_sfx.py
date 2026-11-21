#!/usr/bin/env python3
"""Best-effort Chinese glosses for untranslated SFX that sit INSIDE balloons
(readers default to reading balloon contents as dialogue, so these get a
small gloss in blank space beside the balloon; loose in-panel SFX stay
untouched by design).

Input:  <chapter>/sfx_glosses.json  {"002": [{"box": 8, "zh": "哇——！"},
                                             {"box": 9, "zh": "…", "pos": [x, y]},
                                             {"box": 10, "zh": "…", "skip": true}]}
        <chapter>/out/{page}.boxes.json, <chapter>/rendered/{page}.png
Output: <chapter>/final/{page}.png for every rendered page (glossed pages
        annotated, others copied verbatim). Placement is a grid search on
        the rendered page's ink map: zero-ink slot, no overlap with any
        other detected box or earlier gloss, smallest gap to the balloon
        wins (below/right slightly preferred, page-edge slots penalized).
        Anything with no slot within MAX_GAP is skipped and reported —
        best effort, never forced onto art."""
import json, os, shutil, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
SIZE = 23
MAX_GAP = 90

def find_spot(ink, boxes, self_idx, tw, th, W, H, drawn):
    b = boxes[self_idx]
    others = [(o["x1"] - 3, o["y1"] - 3, o["x2"] + 3, o["y2"] + 3)
              for j, o in enumerate(boxes) if j != self_idx] + drawn
    best = None
    for y in range(max(6, b["y1"] - 190), min(H - th - 6, b["y2"] + 190), 6):
        for x in range(max(6, b["x1"] - 230), min(W - tw - 6, b["x2"] + 230), 6):
            gx = max(b["x1"] - (x + tw), x - b["x2"], 0)
            gy = max(b["y1"] - (y + th), y - b["y2"], 0)
            gap = (gx * gx + gy * gy) ** 0.5
            if gap > MAX_GAP:
                continue
            eff = gap
            if y >= b["y2"]:            # below the balloon: preferred
                eff -= 6
            elif x >= b["x2"]:          # right of it: also natural
                eff -= 3
            if x < 24 or x + tw > W - 24:   # page-edge slots look stray
                eff += 18
            if best and eff >= best[0]:
                continue
            rect = (x, y, x + tw, y + th)
            if any(not (rect[2] < r[0] or rect[0] > r[2] or rect[3] < r[1] or rect[1] > r[3]) for r in others):
                continue
            if ink[y:y + th, x:x + tw].mean() > 0.003:
                continue
            best = (eff, rect)
    return best[1] if best else None

def main(chapter):
    glosses = json.load(open(os.path.join(chapter, "sfx_glosses.json")))
    font = ImageFont.truetype(FONT, SIZE)
    out_dir = os.path.join(chapter, "final")
    os.makedirs(out_dir, exist_ok=True)
    rendered = os.path.join(chapter, "rendered")
    for name in sorted(os.listdir(rendered)):
        if not name.endswith(".png"):
            continue
        pg = name[:-4]
        src = os.path.join(rendered, name)
        dst = os.path.join(out_dir, name)
        items = glosses.get(pg)
        if not items:
            shutil.copyfile(src, dst)
            continue
        boxes = json.load(open(os.path.join(chapter, "out", f"{pg}.boxes.json")))["boxes"]
        img = Image.open(src).convert("RGB")
        W, H = img.size
        ink = ndimage.binary_dilation(np.asarray(img.convert("L")) < 180, iterations=2)
        d = ImageDraw.Draw(img)
        drawn = []
        for it in items:
            bi, zh = it["box"], it["zh"]
            if it.get("skip"):
                print(f"{pg}#{bi} {zh} -> FORCED SKIP")
                continue
            tw = int(d.textlength(zh, font=font))
            th = SIZE + 6
            if "pos" in it:
                x, y = it["pos"]
                rect = (x, y, x + tw, y + th)
                warn = " (over ink!)" if ink[y:y + th, x:x + tw].mean() > 0.003 else ""
                print(f"{pg}#{bi} {zh} -> MANUAL {rect}{warn}")
            else:
                rect = find_spot(ink, boxes, bi, tw, th, W, H, drawn)
                if rect is None:
                    print(f"{pg}#{bi} {zh} -> SKIPPED (no blank spot)")
                    continue
                print(f"{pg}#{bi} {zh} -> {rect}")
            d.text((rect[0], rect[1] + 2), zh, font=font, fill=(0, 0, 0))
            drawn.append(rect)
        img.save(dst)
    print("final pages in", out_dir)

if __name__ == "__main__":
    main(sys.argv[1])
