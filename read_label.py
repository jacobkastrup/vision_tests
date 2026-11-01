#!/usr/bin/env python3
"""Read the part(s) of a parcel label you describe in words.

Examples
--------
  python read_label.py label.jpg "the tracking number"
  python read_label.py label.jpg "the recipient address block" "the sender's postcode"
  python read_label.py label.jpg "the weight" --mode all --engine both --debug out/dbg.png
  python read_label.py label.jpg --mode full --engine ppocr     # CPU only, no description needed
"""

from __future__ import annotations

import argparse
import json
import sys

from parcel_ocr.batch import read_fields
from parcel_ocr.cli_args import add_common, reader_from_args, runs_from_args
from parcel_ocr.reader import load_image, save_debug


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="photo / scan of the parcel label")
    p.add_argument("descriptions", nargs="*", help='what to read, e.g. "the tracking number below the barcode"')
    add_common(p)
    p.add_argument("--rotate", type=int, default=0, help="rotate image clockwise by N degrees first")
    p.add_argument("--debug", metavar="PNG", help="save an overlay of located regions and words")
    p.add_argument("--json", action="store_true", help="print full JSON (boxes, raw responses, latency)")
    args = p.parse_args()

    runs = runs_from_args(args)
    if not args.descriptions and any(m != "full" for m, _ in runs):
        p.error("give at least one description (or use --mode full)")
    image = load_image(args.image, args.rotate)
    reader = reader_from_args(args)
    fields = {f"field_{i}": d for i, d in enumerate(args.descriptions, 1)}

    def show(r):  # print each result as soon as it is ready
        print(f"== {r.description}  [{r.run}, {r.latency_s:.2f}s]")
        print(r.text if r.text is not None else "<not found>")
        print(flush=True)

    results = read_fields(reader, image, fields, runs, on_result=None if args.json else show)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
    if args.debug:
        save_debug(image, [r for r in results if r.mode not in ("ask", "full")] or results, args.debug)
        print(f"debug overlay: {args.debug}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
