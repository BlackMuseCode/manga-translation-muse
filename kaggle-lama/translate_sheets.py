#!/usr/bin/env python3
"""Translate manga crops sheets with a Model Proxy vision model.

One call per crops sheet (image + box labels -> JSON translations).
Defaults match the pilot: Claude Sonnet 5, default thinking (no
reasoning_effort), no response_format (the proxy 400s on json_object
for Claude; the prompt alone yields clean JSON). The proxy token is
minted in-process and never printed or written to disk.

Progress is saved after every sheet to <work>/translations.json.partial
so a crash or a failed sheet does not lose completed sheets; the final
translations.json is written only when every sheet succeeded.
Non-dialogue boxes (empty zh) and uncertain boxes are omitted from
translations.json, matching the existing pipeline convention; both are
listed in the run report.
"""
import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from model_proxy import get_token

DEFAULT_SERIES_INTRO = (
    "作品是《片喰と黄金》（满天星与黄金），"
    "背景是 1849 年加州淘金热，爱尔兰大饥荒中失去一切的主仆艾美瑞亚与科纳横渡大西洋去加州。"
)

PROMPT_TEMPLATE = (
    "你是漫画翻译，把图里的日文对白翻成简体中文。{series_intro}\n\n"
    "术语表（人名必须按此译）：\n{terms}\n\n"
    "规则：气泡对白全翻；拟声词、招牌、装饰文字是画面一部分，不强翻（对应 zh 留空）——"
    "这条是硬规则：纯声音词一律不翻，哪怕你认得这些字也不要翻译，"
    "如猫叫（ニャー/喵）、呼吸声（ハァ/呼）、笑声（フフ/呵呵）、喊声（あああ/啊啊啊）、"
    "噗（プッ）这类词，以及招牌、标题页、作者名、系列 logo，都把 zh 留空；"
    "竖排长句用 \\n 分段；"
    "看不清、拿不准的框不要猜，把它的 label 列进 uncertain。\n\n"
    "只输出一个 JSON 对象，不要解释："
    '{"translations": [{"label": "001#0", "zh": "译文"}...], "uncertain": ["002#1", ...]}，'
    "每个给出的框都要有一条。"
)
# NOTE: build the prompt with replace(), not str.format() — the JSON
# example above contains literal braces that .format() would choke on.


def post(base, key, body, timeout=300):
    req = urllib.request.Request(
        base + "/openapi/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def strip_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--model", default="anthropic/claude-sonnet-5@default")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--terms-file", default=None,
                    help="series TERMS.md (default: <work>/../../TERMS.md)")
    ap.add_argument("--series-intro", default=DEFAULT_SERIES_INTRO,
                    help="one-line series description for the prompt")
    ap.add_argument("--key-file", default=None,
                    help="Kaggle token file (raw KGAT text or kaggle.json); "
                         "or set KAGGLE_API_TOKEN")
    ap.add_argument("--sheets", nargs="*", help="only these sheets, e.g. sheet-03.png")
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()

    work = Path(args.work_dir)
    crops_dir = work / "crops"
    crops = json.loads((crops_dir / "crops.json").read_text())
    terms_path = Path(args.terms_file) if args.terms_file else work.parent.parent / "TERMS.md"
    terms = terms_path.read_text()
    system = PROMPT_TEMPLATE.replace("{series_intro}", args.series_intro).replace("{terms}", terms)

    sheets = {}
    for entry in crops:
        sheets.setdefault(entry["sheet"], []).append(entry["label"])
    sheet_names = sorted(sheets)
    if args.sheets:
        sheet_names = [s for s in sheet_names if s in set(args.sheets)]

    base, key = get_token(args.key_file)
    translations = {}
    partial_path = work / "translations.json.partial"
    if partial_path.exists():
        # Resume/merge: a previous run (or a --sheets rerun) already
        # translated some sheets; keep them and only fill the gaps.
        translations = json.loads(partial_path.read_text())
        print(f"loaded {sum(len(v) for v in translations.values())} boxes "
              f"from existing {partial_path.name}")
    report = {"model": args.model, "sheets": {}, "total_cost_usd": 0.0,
              "failed_sheets": [], "uncertain": []}

    for sheet in sheet_names:
        labels = sheets[sheet]
        img_b64 = base64.b64encode((crops_dir / sheet).read_bytes()).decode()
        body = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": "data:image/png;base64," + img_b64}},
                    {"type": "text", "text":
                     "图里是 %d 个裁出来的对话气泡，按页#框编号。逐个翻译：\n%s"
                     % (len(labels), "、".join(labels))},
                ]},
            ],
            "max_tokens": args.max_tokens,
        }
        last_err = None
        payload = None
        content = ""
        for attempt in range(1, args.max_attempts + 1):
            t0 = time.time()
            try:
                payload = post(base, key, body)
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code} {e.read().decode()[:200]}"
                if e.code == 401:  # token expired; re-mint once and retry
                    base, key = get_token(args.key_file)
                payload = None
            except Exception as e:  # timeout / connection reset
                last_err = f"{type(e).__name__}: {e}"
                payload = None
            if payload is None:
                if attempt < args.max_attempts:
                    time.sleep(5 * attempt)
                continue
            content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            try:
                parsed = json.loads(strip_fences(content))
                _ = parsed["translations"]
                break  # valid JSON: proceed to merge
            except Exception as e:
                last_err = f"bad JSON: {e}"
                (work / f"failed-{sheet}-attempt{attempt}.txt").write_text(
                    content, encoding="utf-8")
                payload = None
                if attempt < args.max_attempts:
                    time.sleep(5 * attempt)
        if payload is None:
            report["failed_sheets"].append({"sheet": sheet, "error": last_err})
            print(f"{sheet}: FAILED {last_err}", flush=True)
            continue
        usage = payload.get("usage") or {}
        cost = usage.get("cost") or {}
        ci = cost.get("input_tokens_cost_nanodollars") or 0
        co = cost.get("output_tokens_cost_nanodollars") or 0
        sheet_cost = (ci + co) / 1e9
        report["total_cost_usd"] += sheet_cost
        parsed = json.loads(strip_fences(content))
        items = parsed["translations"]
        got = set()
        for item in items:
            label = item.get("label")
            zh = (item.get("zh") or "").strip()
            got.add(label)
            if not zh:
                continue
            page, box = label.split("#")
            translations.setdefault(page, []).append({"box": int(box), "zh": zh})
        missing = [lb for lb in labels if lb not in got]
        uncertain = parsed.get("uncertain") or []
        report["uncertain"].extend(uncertain)
        report["sheets"][sheet] = {
            "seconds": round(time.time() - t0, 1),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cost_usd": round(sheet_cost, 6),
            "boxes": len(items), "missing_labels": missing,
            "uncertain": uncertain,
        }
        print(f"{sheet}: {len(items)} boxes, ${sheet_cost:.5f}, "
              f"{report['sheets'][sheet]['seconds']}s"
              + (f" uncertain={uncertain}" if uncertain else "")
              + (f" missing={missing}" if missing else ""), flush=True)
        # incremental progress
        ordered = {p: sorted(v, key=lambda e: e["box"])
                   for p, v in sorted(translations.items())}
        partial_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=1),
                                encoding="utf-8")

    report_path = work / "translation-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["failed_sheets"]:
        print(f"{len(report['failed_sheets'])} sheet(s) failed; "
              f"partial translations in {partial_path}; report in {report_path}")
        return 1
    ordered = {p: sorted(v, key=lambda e: e["box"])
               for p, v in sorted(translations.items())}
    (work / "translations.json").write_text(
        json.dumps(ordered, ensure_ascii=False, indent=1), encoding="utf-8")
    total_boxes = sum(len(v) for v in ordered.values())
    print(f"done: {len(ordered)} pages, {total_boxes} boxes, "
          f"total cost ${report['total_cost_usd']:.4f}, "
          f"uncertain {len(report['uncertain'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
