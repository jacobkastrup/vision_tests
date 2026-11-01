"""Shared batch logic: run fields x (mode, engine) over images, write CSV/JSONL, score."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image

from .reader import LabelReader, ReadResult

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(src: str | Path) -> list[Path]:
    src = Path(src)
    if src.is_file():
        return [src]
    return sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXT)


def read_fields(
    reader: LabelReader,
    image: Image.Image,
    fields: dict[str, str],
    runs: Iterable[tuple[str, str]],
    on_result: Callable[[ReadResult], None] | None = None,
) -> list[ReadResult]:
    """All (field, mode, engine) results for one image. ``full`` runs once, not per field.

    ``on_result`` is called as soon as each result is ready (progress output)."""
    results = []
    for mode, engine in runs:
        descs = ["(whole label)"] if mode == "full" else list(fields.values())
        for desc in descs:
            _log(f"{mode}/{engine}: {desc[:50]} ...")
            result = reader.read(image, desc, mode, engine)
            results.append(result)
            if on_result:
                on_result(result)
    return results


def _log(message: str) -> None:
    print(f"[parcel_ocr] {message}", file=sys.stderr, flush=True)


def to_record(image_name: str, fields: dict[str, str], results: list[ReadResult]) -> dict[str, Any]:
    by_desc = {v: k for k, v in fields.items()}
    return {
        "image": image_name,
        "results": [dict(r.to_dict(), field=by_desc.get(r.description, "_full")) for r in results],
    }


def write_outputs(records: list[dict[str, Any]], fields: dict[str, str], output: str | Path) -> list[dict[str, str]]:
    """CSV: one row per (image, mode/engine), one column per field. JSONL: everything."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for rec in records:
        for r in rec["results"]:
            run = f"{r['mode']}/{r['engine']}"
            row = rows.setdefault((rec["image"], run), {"image": rec["image"], "run": run})
            row[r["field"]] = r["text"] or ""
            row[f"_s_{r['field']}"] = row.get(f"_s_{r['field']}", 0) + r["latency_s"]
    columns = ["image", "run", *fields]
    if any(r.get("_full") for r in rows.values()):
        columns.append("_full")
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows.values())
    output.with_suffix(".jsonl").write_text(
        "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records), encoding="utf-8"
    )
    return list(rows.values())


def normalise(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().upper()


def score(rows: list[dict[str, str]], fields: dict[str, str], expected_csv: str | Path) -> str:
    """Exact-match accuracy (case/whitespace-insensitive) per run x field, plus a
    'contained in full-label text' rate for ``full`` runs."""
    with open(expected_csv, newline="", encoding="utf-8") as fh:
        truth = {r["image"]: r for r in csv.DictReader(fh)}
    runs = sorted({r["run"] for r in rows})
    width = max([len(f) for f in fields] + [10])
    lines = ["accuracy (exact match, case/whitespace-insensitive; full = expected text found in output)",
             " " * (width + 2) + "".join(f"{run:>18}" for run in runs)]
    for name in fields:
        cells = []
        for run in runs:
            hits = total = 0
            for row in rows:
                gt = truth.get(row["image"], {}).get(name)
                if row["run"] != run or not gt:
                    continue
                total += 1
                if run.startswith("full/"):
                    hits += normalise(gt) in normalise(row.get("_full"))
                else:
                    hits += normalise(row.get(name)) == normalise(gt)
            cells.append(f"{hits}/{total}" if total else "-")
        lines.append(f"  {name:<{width}}" + "".join(f"{c:>18}" for c in cells))
    latency = {}
    for row in rows:
        latency.setdefault(row["run"], []).append(sum(v for k, v in row.items() if k.startswith("_s_")))
    lines.append(f"  {'sec/image':<{width}}" + "".join(
        f"{sum(latency[r]) / len(latency[r]):>18.2f}" for r in runs))
    return "\n".join(lines)
