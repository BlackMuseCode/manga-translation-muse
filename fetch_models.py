#!/usr/bin/env python3
"""Download the four inference models and verify them by SHA-256.

The model weights are NOT shipped in this repository (see README for why).
This script fetches each file from its upstream source and checks it
against the pinned hashes in kaggle-lama/runtime-manifest.json:

  saber_yolo/saber_yolo.pt            56 MB  Saber models-storage release pack
  default/detect-20241225.ckpt       308 MB  Saber models-storage release pack
  ctd/comictextdetector.pt            80 MB  manga-image-translator release
  lama-manga/lama-manga.safetensors  204 MB  Hugging Face mayocream/lama-manga

The Saber pack is a ~3.3 GB split archive (models.z01 + models.zip); the
script downloads both parts and extracts only the two files this
pipeline needs with 7z (Python's zipfile refuses split archives even
after joining). The other two files are direct downloads.

Output layout matches what kaggle-lama/build_runtime.py expects:
  <models-dir>/saber_yolo/saber_yolo.pt
  <models-dir>/default/detect-20241225.ckpt
  <models-dir>/ctd/comictextdetector.pt
  <models-dir>/lama-manga/lama-manga.safetensors

Usage:
  python fetch_models.py [--models-dir models] [--only saber_yolo,ctd]
  Files that already exist with matching size + SHA-256 are skipped.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

UA = {"User-Agent": "manga-translation-muse/1.0 (model fetch)"}

# (relative path, size in bytes, sha256, kind, source)
MODELS = [
    ("saber_yolo/saber_yolo.pt", 56455253,
     "a34bcb05eb3ca75941cfdaa59883edf19f6b209bcc9cc1fb139a2f5cf0bc6324",
     "pack", "Saber models-storage release"),
    ("default/detect-20241225.ckpt", 308380176,
     "67ce1c4ed4793860f038c71189ba9630a7756f7683b1ee5afb69ca0687dc502e",
     "pack", "Saber models-storage release"),
    ("ctd/comictextdetector.pt", 79948869,
     "1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb",
     "direct",
     "https://github.com/zyddnys/manga-image-translator/releases/download/"
     "beta-0.3/comictextdetector.pt"),
    ("lama-manga/lama-manga.safetensors", 204332996,
     "a790515e9da839b8d89af7d565ceb110d908b7d6fbdb991f2acb2ec7d9b08bdb",
     "direct",
     "https://huggingface.co/mayocream/lama-manga/resolve/main/"
     "lama-manga.safetensors"),
]

PACK_BASE = ("https://github.com/MashiroSaber03/Saber-Translator"
             "/releases/download/models-storage/")
PACK_PARTS = ["models.z01", "models.zip"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ok(path, size, sha):
    p = Path(path)
    return p.is_file() and p.stat().st_size == size and sha256_file(p) == sha


def download(url, dest, expect_size=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    done = 0
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done % (50 << 20) < (1 << 20):
                print(f"  {dest.name}: {done >> 20} MB", flush=True)
    if expect_size is not None and dest.stat().st_size != expect_size:
        raise SystemExit(f"size mismatch for {dest.name}: got "
                         f"{dest.stat().st_size}, expected {expect_size}")
    print(f"  {dest.name}: done ({done >> 20} MB)")


def find_7z():
    """Locate a 7z binary (needed to unpack the split Saber pack)."""
    for name in ("7z", "7zz", "7za"):
        path = shutil.which(name)
        if path:
            return path
    return None


def fetch_pack(models_dir, keep_pack):
    """Download the Saber split archive, extract the two needed files.

    The pack is a split ZIP (models.z01 + models.zip); Python's zipfile
    refuses multi-disk archives even after joining, so extraction uses
    7z — the same tool Saber's own CI uses for this release.
    """
    seven_zip = find_7z()
    if not seven_zip:
        raise SystemExit(
            "7z not found: install it first, then re-run.\n"
            "  Ubuntu/Debian: sudo apt install p7zip-full\n"
            "  macOS:         brew install p7zip\n"
            "  Windows:       https://www.7-zip.org/download.html")
    # NOTE: never use /tmp here — it is a 512 MB tmpfs on many machines and
    # the pack is ~3.3 GB. Keep the temp dir next to the models dir.
    free = shutil.disk_usage(models_dir.parent).free
    if free < 5 << 30:
        raise SystemExit(f"not enough disk space: {free >> 30} GB free, "
                         f"need ~5 GB (pack 3.3 GB + extracted models)")
    tmp = Path(tempfile.mkdtemp(prefix=".saber-models-pack-",
                                dir=models_dir.parent))
    try:
        for name in PACK_PARTS:
            print(f"downloading Saber models pack part {name} (~3.3 GB total)...")
            download(PACK_BASE + name, tmp / name)
        extract = tmp / "extracted"
        members = ["models/" + rel for rel, _s, _h, kind, _src in MODELS
                   if kind == "pack"]
        print("extracting needed files with 7z...")
        import subprocess
        res = subprocess.run(
            [seven_zip, "x", str(tmp / "models.zip"),
             f"-o{extract}", "-y", *members],
            capture_output=True, text=True, timeout=1800)
        if res.returncode != 0:
            raise SystemExit(f"7z failed:\n{res.stdout}\n{res.stderr}")
        for rel, size, sha, kind, _src in MODELS:
            if kind != "pack":
                continue
            src = extract / "models" / rel
            dest = models_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            if not ok(dest, size, sha):
                raise SystemExit(f"SHA-256 mismatch after extraction: {rel}")
            print(f"  {rel}: OK")
        if keep_pack:
            keep = models_dir / "models-storage-pack"
            keep.mkdir(parents=True, exist_ok=True)
            for name in PACK_PARTS:
                shutil.move(str(tmp / name), keep / name)
            print(f"pack parts kept in {keep}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir", default="models",
                    help="where to put the models (default: ./models)")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset, e.g. saber_yolo,ctd")
    ap.add_argument("--keep-pack", action="store_true",
                    help="keep the downloaded ~3.3 GB Saber pack parts")
    args = ap.parse_args(argv)
    models_dir = Path(args.models_dir)
    wanted = None
    if args.only:
        wanted = set(args.only.split(","))
    need_pack = False
    for rel, size, sha, kind, src in MODELS:
        key = rel.split("/")[0]
        if wanted and key not in wanted and rel not in wanted:
            continue
        dest = models_dir / rel
        if ok(dest, size, sha):
            print(f"{rel}: already present, verified OK")
            continue
        if kind == "direct":
            print(f"downloading {rel} ({size >> 20} MB) from {src} ...")
            download(src, dest, expect_size=size)
            if not ok(dest, size, sha):
                raise SystemExit(f"SHA-256 mismatch: {rel} "
                                 f"(source file may have changed; "
                                 f"expected {sha[:16]}...)")
            print(f"  {rel}: OK")
        else:
            need_pack = True
    if need_pack:
        fetch_pack(models_dir, args.keep_pack)
    print("all requested models verified.")
    print("next: python kaggle-lama/build_runtime.py "
          "--saber-root <Saber-Translator@13491d83> "
          f"--engine engine/lama_engine.py --output .work/new-runtime")


if __name__ == "__main__":
    main()
