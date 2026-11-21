"""Small adapter to the installed Saber models; no substitute inference code."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
import sys


def complete_segmented_glyphs(image, precise_mask, rectangles, expand_ratio):
    """Complete high-contrast ink components anchored by model segmentation.

    Segmentation sometimes leaves holes inside very bold punctuation. Complete
    only seeded ink components on near-flat light/dark backgrounds; never fill
    rectangles or components running into the crop edge (often bubble borders).
    Native Saber mask cropping/dilation still runs after this step.
    """
    import cv2
    import numpy as np
    gray = np.asarray(image.convert("L"))
    result = np.asarray(precise_mask, dtype=np.uint8).copy()
    height, width = gray.shape
    for x1, y1, x2, y2 in rectangles:
        dx, dy = (x2-x1)*expand_ratio/200, (y2-y1)*expand_ratio/200
        left, top = max(0, int(x1-dx)), max(0, int(y1-dy))
        right, bottom = min(width, int(x2+dx)+1), min(height, int(y2+dy)+1)
        crop = gray[top:bottom, left:right]
        median = float(np.median(crop))
        if median >= 200:
            ink = crop < median-50
        elif median <= 55:
            ink = crop > median+50
        else:
            continue
        if np.mean(np.abs(crop.astype(float)-median) <= 12) < .5:
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
        local = result[top:bottom, left:right]
        seeds = np.asarray(precise_mask)[top:bottom, left:right] >= 128
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area < 4 or x == 0 or y == 0 or x+w == crop.shape[1] or y+h == crop.shape[0]:
                continue
            component = labels[y:y+h, x:x+w] == label
            if np.count_nonzero(seeds[y:y+h, x:x+w] & component) >= area*.25:
                local[y:y+h, x:x+w][component] = 255
    return result


def fill_mask_holes(mask):
    """Fill enclosed gaps left inside segmented strokes, without growing edges."""
    import cv2
    import numpy as np
    padded = np.pad(mask, 1, constant_values=0)
    exterior = padded.copy()
    cv2.floodFill(exterior, None, (0, 0), 255)
    return np.maximum(padded, 255-exterior)[1:-1, 1:-1]


class SaberEngine:
    def __init__(self, saber_root):
        self.root = Path(saber_root)
        self.detector = None
        self.mask_detector = None
        self.ctd_detector = None
        self.inpainter = None

    def load(self):
        if not (self.root / "src/core/repair_mask.py").is_file():
            raise FileNotFoundError("Saber v3.5.17 or later is required")
        for model in ("saber_yolo/saber_yolo.pt", "default/detect-20241225.ckpt",
                      "ctd/comictextdetector.pt",
                      "lama-manga/lama-manga.safetensors"):
            if not (self.root / "models" / model).is_file():
                raise FileNotFoundError(f"Required local model is missing: {model}")
        os.chdir(self.root)
        sys.path.insert(0, str(self.root))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; GPU is unavailable")
        from src.core.detector.backends.saber_yolo_backend import SaberYoloBackend
        from src.core.detector.backends.default_backend import DefaultBackend
        from src.core.detector.backends.ctd_backend import CTDBackend
        from src.interfaces.lama_interface import get_lama_manga_inpainter
        self.detector = SaberYoloBackend(device="cuda", conf_thresh=0.3)
        self.mask_detector = DefaultBackend(device="cuda")
        self.ctd_detector = CTDBackend(device="cuda")
        self.inpainter = get_lama_manga_inpainter()
        self.inpainter.load(device="cuda:0")
        if any(not str(model.device).startswith("cuda") for model in
               (self.detector, self.mask_detector, self.ctd_detector)):
            raise RuntimeError("Saber detectors did not select CUDA")

    def process(self, image, mask_dilate_size=10, mask_box_expand_ratio=20.0):
        import torch
        import numpy as np
        from PIL import Image
        from src.core.repair_mask import build_repair_mask
        from src.interfaces.lama_interface import clean_image_with_lama
        width, height = image.size
        with torch.inference_mode():
            detection = self.detector.detect(image, merge_lines=False, sort_method="none",
                                             enable_aux_yolo_detection=False)
            boxes = []
            for line in detection.raw_lines:
                if line.confidence < 0.3:
                    continue
                x1, y1, x2, y2 = line.xyxy
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append(dict(x1=x1, y1=y1, x2=x2, y2=y2, score=float(line.confidence)))
            boxes.sort(key=lambda box: box["score"], reverse=True)
            rectangles = [(box["x1"], box["y1"], box["x2"], box["y2"]) for box in boxes]
            repair_mask = np.zeros((height, width), dtype=np.uint8)
            if rectangles:
                # The original YOLO boxes above are never merged/replaced by
                # the independent DBNet/CTD segmentation passes. CTD covers
                # large punctuation that DBNet can miss; only masks are combined.
                segmented = self.mask_detector.detect(image, merge_lines=False, sort_method="none",
                                                       enable_aux_yolo_detection=False)
                if segmented.mask is None:
                    raise RuntimeError("Saber DBNet returned no precise text mask")
                supplemental = self.ctd_detector.detect(image, merge_lines=False, sort_method="none",
                                                        enable_aux_yolo_detection=False)
                if supplemental.mask is None:
                    raise RuntimeError("Saber CTD returned no precise text mask")
                precise_mask = np.maximum(segmented.mask, supplemental.mask)
                precise_mask = complete_segmented_glyphs(
                    image, precise_mask, rectangles, mask_box_expand_ratio)
                repair_mask = build_repair_mask(
                    (height, width), rectangles, precise_mask=precise_mask,
                    mask_dilate_size=mask_dilate_size,
                    mask_box_expand_ratio=mask_box_expand_ratio,
                    method="lama", regional_inpainting=True)
                repair_mask = fill_mask_holes(repair_mask)
            if np.any(repair_mask):
                with Image.fromarray(255 - repair_mask) as native_mask:
                    cleaned = clean_image_with_lama(
                        image, native_mask, lama_model="lama_manga",
                        disable_resize=True, regional_inpainting=True)
            else:
                cleaned = image.copy()
            try:
                if cleaned.size != image.size:
                    raise RuntimeError("Saber returned an unexpected image size")
                output = io.BytesIO()
                cleaned.save(output, format="PNG")
                mask_output = io.BytesIO()
                with Image.fromarray(repair_mask) as final_mask:
                    final_mask.save(mask_output, format="PNG")
                return dict(width=width, height=height, boxes=boxes,
                            mask_b64=base64.b64encode(mask_output.getvalue()).decode("ascii"),
                            inpainted_image_b64=base64.b64encode(output.getvalue()).decode("ascii"))
            finally:
                cleaned.close()

    def close(self):
        self.detector = None
        self.mask_detector = None
        self.ctd_detector = None
        if self.inpainter is not None:
            self.inpainter.unload()
            self.inpainter = None
