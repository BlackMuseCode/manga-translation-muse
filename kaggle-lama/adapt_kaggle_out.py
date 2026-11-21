#!/usr/bin/env python3
"""Adapt a fetched Kaggle LaMA job into the existing out/ layout.

Reads <fetch-dir>/job-result.json + results.zip (already verified by
client.py fetch) and writes, per page:
  <out-dir>/{stem}.boxes.json      {"width","height","boxes":[{x1,y1,x2,y2,score,polygon,angle}]}
  <out-dir>/{stem}.inpainted.png   (from pNNNNNN/cleaned.png)
  <out-dir>/{stem}.mask.png        (from pNNNNNN/mask.png)
  <out-dir>/{stem}.precise-mask.png (from pNNNNNN/precise-mask.png)
Per-file sha256/size are re-checked against the page record.
Exits non-zero and lists the failed pages if any page status != ok.
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

FILE_MAP = [
    ("inpainted_image_b64", "inpainted.png"),
    ("mask_b64", "mask.png"),
    ("precise_mask_b64", "precise-mask.png"),
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    fetch_dir = Path(args.fetch_dir)
    out_dir = Path(args.out_dir)
    result = json.loads((fetch_dir / "job-result.json").read_text())
    if result.get("status") != "completed":
        print(f"job status is {result.get('status')!r}, not completed; refusing to adapt",
              file=sys.stderr)
        return 2
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        print(f"{out_dir} is not empty; pass --force to overwrite", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    failed, adapted, boxes_total = [], 0, 0
    with zipfile.ZipFile(fetch_dir / "results.zip") as archive:
        for page in result["pages"]:
            stem = Path(page["source_name"]).stem
            if page.get("status") != "ok":
                failed.append((stem, page.get("status"), page.get("error", "")))
                continue
            boxes = page["boxes"]
            (out_dir / f"{stem}.boxes.json").write_text(
                json.dumps({"width": page["width"], "height": page["height"],
                            "boxes": boxes}, ensure_ascii=False), encoding="utf-8")
            for key, suffix in FILE_MAP:
                meta = page["files"][key]
                data = archive.read(meta["path"])
                if len(data) != meta["bytes"] or digest(data) != meta["sha256"]:
                    print(f"checksum mismatch on {stem} {suffix}", file=sys.stderr)
                    return 3
                (out_dir / f"{stem}.{suffix}").write_bytes(data)
            adapted += 1
            boxes_total += len(boxes)
    print(f"adapted {adapted} pages, {boxes_total} boxes -> {out_dir}")
    if failed:
        print(f"FAILED pages ({len(failed)}):")
        for stem, status, err in failed:
            print(f"  {stem}: {status} {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
