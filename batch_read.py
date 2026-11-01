#!/usr/bin/env python3
"""Read the same set of described fields from many label images.

  python batch_read.py labels/ fields.example.json -o results.csv
  python batch_read.py labels/ fields.example.json -o results.csv --expected expected.csv

``fields.json`` maps a column name to a description in words:
  {"tracking_number": "the tracking number printed under the large barcode", ...}

``--expected`` (optional) is a CSV with an ``image`` column and one column per
field holding the correct text; an accuracy summary is then printed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from parcel_ocr.reader import DEFAULT_MODEL, MODES, LabelReader, load_image, save_debug

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def normalise(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().upper()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", help="folder of label images (or a single image)")
    p.add_argument("fields", help="JSON file: {column_name: description}")
    p.add_argument("-o", "--output", default="results.csv")
    p.add_argument("--mode", choices=MODES, default="crop")
    p.add_argument("--pick", choices=("union", "first", "largest"), default="union")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-pixels", type=int, default=None)
    p.add_argument("--debug-dir", help="save one overlay PNG per image here")
    p.add_argument("--expected", help="CSV with ground truth (image + field columns)")
    args = p.parse_args()

    fields: dict[str, str] = json.loads(Path(args.fields).read_text(encoding="utf-8"))
    src = Path(args.images)
    paths = [src] if src.is_file() else sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXT)
    reader = LabelReader(args.model, backend=args.backend, device=args.device,
                         max_pixels=args.max_pixels, pick=args.pick)

    rows, details = [], []
    for path in paths:
        image = load_image(path)
        results = [reader.read(image, desc, args.mode) for desc in fields.values()]
        row = {"image": path.name}
        for name, res in zip(fields, results):
            row[name] = res.text or ""
        rows.append(row)
        details.append({"image": path.name, "results": {n: r.to_dict() for n, r in zip(fields, results)}})
        print(path.name, json.dumps({k: v for k, v in row.items() if k != "image"}, ensure_ascii=False))
        if args.debug_dir:
            save_debug(image, results, Path(args.debug_dir) / f"{path.stem}_debug.png")

    out = Path(args.output)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", *fields])
        writer.writeheader()
        writer.writerows(rows)
    out.with_suffix(".jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in details), encoding="utf-8"
    )
    print(f"\nwrote {out} and {out.with_suffix('.jsonl')}")

    if args.expected:
        with open(args.expected, newline="", encoding="utf-8") as fh:
            truth = {r["image"]: r for r in csv.DictReader(fh)}
        print("\nexact-match accuracy (case/whitespace-insensitive):")
        for name in fields:
            pairs = [(row[name], truth[row["image"]][name]) for row in rows
                     if row["image"] in truth and name in truth[row["image"]]]
            hits = sum(normalise(a) == normalise(b) for a, b in pairs)
            print(f"  {name:<24} {hits}/{len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
