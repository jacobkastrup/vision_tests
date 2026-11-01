#!/usr/bin/env python3
"""Read the same described fields from many label images, optionally comparing engines.

  python batch_read.py labels/ fields.example.json -o outputs/results.csv
  python batch_read.py labels/ fields.example.json -o outputs/results.csv \\
      --mode crop,filter,full --engine both --expected expected.csv

``fields.json`` maps a column name to a description in words:
  {"tracking_number": "the tracking number printed under the large barcode", ...}

The CSV has one row per (image, mode/engine) and one column per field.
``--expected`` is a CSV with an ``image`` column and one column per field holding
the correct text; an accuracy + latency table per mode/engine is then printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parcel_ocr.batch import list_images, read_fields, score, to_record, write_outputs
from parcel_ocr.cli_args import add_common, reader_from_args, runs_from_args
from parcel_ocr.reader import load_image, save_debug


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", help="folder of label images (or a single image)")
    p.add_argument("fields", help="JSON file: {column_name: description}")
    p.add_argument("-o", "--output", default="outputs/results.csv")
    add_common(p)
    p.add_argument("--debug-dir", help="save one overlay PNG per image here")
    p.add_argument("--expected", help="CSV with ground truth (image + field columns)")
    args = p.parse_args()

    fields: dict[str, str] = json.loads(Path(args.fields).read_text(encoding="utf-8"))
    runs = runs_from_args(args)
    reader = reader_from_args(args)
    reader.warm_up(runs)

    records = []
    for path in list_images(args.images):
        image = load_image(path)
        results = read_fields(reader, image, fields, runs)
        records.append(to_record(path.name, fields, results))
        for r in results:
            print(f"{path.name}  {r.run:<16} {r.description[:40]:<40} -> {(r.text or '<not found>')!r}")
        if args.debug_dir:
            save_debug(image, [r for r in results if r.mode not in ("ask", "full")],
                       Path(args.debug_dir) / f"{path.stem}_debug.png")

    rows = write_outputs(records, fields, args.output)
    print(f"\nwrote {args.output} and {Path(args.output).with_suffix('.jsonl')}")
    if args.expected:
        print("\n" + score(rows, fields, args.expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
