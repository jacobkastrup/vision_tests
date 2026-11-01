#!/usr/bin/env python3
"""Read the part(s) of a parcel label you describe in words.

Examples
--------
  python read_label.py label.jpg "the tracking number"
  python read_label.py label.jpg "the recipient address block" "the sender's postcode"
  python read_label.py label.jpg "the weight" --mode all --debug out/label_debug.png
"""

from __future__ import annotations

import argparse
import json
import sys

from parcel_ocr.reader import DEFAULT_MODEL, MODES, LabelReader, load_image, save_debug


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="photo / scan of the parcel label")
    p.add_argument("descriptions", nargs="+", help='what to read, e.g. "the tracking number below the barcode"')
    p.add_argument("--mode", choices=(*MODES, "all"), default="crop", help="reading strategy (default: crop)")
    p.add_argument("--pick", choices=("union", "first", "largest"), default="union",
                   help="if the description matches several boxes: merge them (default), or take one")
    p.add_argument("--rotate", type=int, default=0, help="rotate image clockwise by N degrees first")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF id or local path (env WEVISONE_MODEL)")
    p.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    p.add_argument("--device", default="auto", help="auto, cuda:0, cpu, ...")
    p.add_argument("--max-pixels", type=int, default=None, help="cap on image pixels fed to the model")
    p.add_argument("--debug", metavar="PNG", help="save an overlay of located regions and words")
    p.add_argument("--json", action="store_true", help="print full JSON (boxes, raw responses, latency)")
    args = p.parse_args()

    image = load_image(args.image, args.rotate)
    reader = LabelReader(args.model, backend=args.backend, device=args.device,
                         max_pixels=args.max_pixels, pick=args.pick)
    modes = MODES if args.mode == "all" else (args.mode,)

    results = [reader.read(image, d, m) for d in args.descriptions for m in modes]

    if args.json:
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
    else:
        for r in results:
            print(f"== {r.description}  [{r.mode}, {r.latency_s:.2f}s]")
            print(r.text if r.text is not None else "<not found>")
            print()
    if args.debug:
        save_debug(image, [r for r in results if r.mode != "ask"] or results, args.debug)
        print(f"debug overlay: {args.debug}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
