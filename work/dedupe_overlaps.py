#!/usr/bin/env python3
"""Detect overlapping detection boxes and merge them automatically.
For each pair with IoU > threshold (or containment), keep the larger box
and append the smaller box's translation into it, then drop the smaller.
Run after translation, before rendering.

GUARDRAIL (2026-10-22): merges are destructive and NOT always right —
adjacent column boxes of one split bubble legitimately overlap, and a
manual fix pass may deliberately split a merged bubble back into two
entries (Kuro-chan c001: re-running this after the fix pass re-merged
023 #5->#8 and 014 #6->#9 and duplicated a line). Default mode is
therefore DRY-RUN: it only reports what would merge. Pass --yes to
actually write. Only run on a fresh first-pass translation.
"""
import sys, json, os, itertools

def iou(b1, b2):
    x1 = max(b1['x1'], b2['x1']); y1 = max(b1['y1'], b2['y1'])
    x2 = min(b1['x2'], b2['x2']); y2 = min(b1['y2'], b2['y2'])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1['x2'] - b1['x1']) * (b1['y2'] - b1['y1'])
    a2 = (b2['x2'] - b2['x1']) * (b2['y2'] - b2['y1'])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

def area(b):
    return (b['x2'] - b['x1']) * (b['y2'] - b['y1'])

def containment(smaller, larger):
    """What fraction of smaller is inside larger."""
    x1 = max(smaller['x1'], larger['x1']); y1 = max(smaller['y1'], larger['y1'])
    x2 = min(smaller['x2'], larger['x2']); y2 = min(smaller['y2'], larger['y2'])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a = area(smaller)
    return inter / a if a > 0 else 0

def dedupe_chapter(ch_dir, iou_thresh=0.3, contain_thresh=0.8, apply=True):
    trans_path = os.path.join(ch_dir, 'translations.json')
    trans = json.load(open(trans_path))
    merged = 0
    for pg, entries in trans.items():
        if len(entries) < 2:
            continue
        boxes = json.load(open(os.path.join(ch_dir, 'out', f'{pg}.boxes.json')))['boxes']
        idxs = [e['box'] for e in entries]
        # find overlapping pairs
        to_remove = set()
        merges = {}  # keep_box -> list of (drop_box, text)
        for i, j in itertools.combinations(idxs, 2):
            if i in to_remove or j in to_remove:
                continue
            b1, b2 = boxes[i], boxes[j]
            v = iou(b1, b2)
            c1 = containment(b1, b2)  # b1 inside b2?
            c2 = containment(b2, b1)  # b2 inside b1?
            overlap = v > iou_thresh or c1 > contain_thresh or c2 > contain_thresh
            if not overlap:
                continue
            # keep larger, drop smaller
            if area(b1) >= area(b2):
                keep, drop = i, j
            else:
                keep, drop = j, i
            to_remove.add(drop)
            e_drop = next(e for e in entries if e['box'] == drop)
            merges.setdefault(keep, []).append((drop, e_drop['zh']))
            print(f'  {pg}: box{drop} {"merged" if apply else "would merge"} '
                  f'into box{keep} (IoU={v:.2f}, contain={max(c1,c2):.2f})')
        if not to_remove:
            continue
        # apply: append dropped texts to keeper (smaller box text goes after)
        new_entries = []
        for e in entries:
            if e['box'] in to_remove:
                continue
            if e['box'] in merges:
                extra = '\n'.join(t for _, t in merges[e['box']])
                e['zh'] = e['zh'] + '\n' + extra
                merged += len(merges[e['box']])
            new_entries.append(e)
        trans[pg] = new_entries
    if apply:
        json.dump(trans, open(trans_path, 'w'), ensure_ascii=False, indent=1)
        print(f'Done {ch_dir}: merged {merged} overlapping boxes')
    else:
        print(f'DRY-RUN {ch_dir}: {merged} boxes WOULD merge — nothing written. '
              f'Re-run with --yes to apply (fresh first-pass translations only).')
    return merged

if __name__ == '__main__':
    # usage: dedupe_overlaps.py <work_dir> <chapter> [--yes]
    args = [a for a in sys.argv[1:] if a != '--yes']
    work_dir, chapter = args[0], args[1]
    dedupe_chapter(os.path.join(work_dir, chapter), apply='--yes' in sys.argv)
