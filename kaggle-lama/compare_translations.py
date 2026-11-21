#!/usr/bin/env python3
"""Compare two translations.json files box by box.

Usage: compare_translations.py <new> <old>
Prints identical / different / only-in-new / only-in-old counts and the
full diff list (label, old zh, new zh). Exits 0 even when diffs exist.
"""
import json
import sys
from pathlib import Path


def load(path):
    data = json.loads(Path(path).read_text())
    out = {}
    for page, entries in data.items():
        for e in entries:
            out[f"{page}#{e['box']}"] = e["zh"]
    return out


def main():
    new = load(sys.argv[1])
    old = load(sys.argv[2])
    keys = sorted(set(new) | set(old), key=lambda k: (k.split("#")[0], int(k.split("#")[1])))
    same = diff = only_new = only_old = 0
    diffs = []
    for k in keys:
        if k in new and k in old:
            if new[k] == old[k]:
                same += 1
            else:
                diff += 1
                diffs.append({"label": k, "old": old[k], "new": new[k]})
        elif k in new:
            only_new += 1
            diffs.append({"label": k, "old": None, "new": new[k]})
        else:
            only_old += 1
            diffs.append({"label": k, "old": old[k], "new": None})
    total = len(keys)
    print(f"total boxes compared: {total}")
    print(f"identical: {same} ({same / total * 100:.1f}%)" if total else "identical: 0")
    print(f"different: {diff}")
    print(f"only in new: {only_new}")
    print(f"only in old: {only_old}")
    for d in diffs:
        print(f"\n{d['label']}:")
        print(f"  old: {d['old']!r}")
        print(f"  new: {d['new']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
