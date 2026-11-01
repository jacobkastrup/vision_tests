"""Run the label reader on a cloud GPU with Modal (https://modal.com).

Your laptop only uploads images and receives text; the models run on a rented GPU
billed per second (an L4 is roughly $0.80/hour; check modal.com/pricing).

One-time setup:
    pip install modal
    modal setup                      # opens a browser to log in / create an account

Single image:
    modal run cloud/modal_app.py --images label.jpg --describe "the tracking number;the recipient address block"

Batch + engine comparison + accuracy:
    modal run cloud/modal_app.py --images labels/ --fields fields.example.json \
        --mode crop,filter,full --engine both --expected expected.csv --out outputs/cloud

Options: --gpu (env MODAL_GPU: L4 default, A10G, A100-40GB, H100), --model tencent/WeVisOne-VL-2B,
--ppocr-tier medium|small|tiny, --pick union|first|largest, --debug (save overlays locally).

The first run builds the container image (~10 min) and downloads the weights (~9 GB)
into a persistent Modal Volume; later runs start in well under a minute.
"""

# (no `from __future__ import annotations`: Modal reads parameter type annotations)

import io
import json
import os
from pathlib import Path

import sys

import modal

REPO = Path(__file__).resolve().parent.parent
if modal.is_local():
    sys.path.insert(0, str(REPO))  # so the local entrypoint can import parcel_ocr
GPU = os.environ.get("MODAL_GPU", "L4")
WEVISONE_GIT = "https://github.com/Tencent/WeVisOne-VL.git"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1")
    .run_commands(
        f"git clone --depth 1 {WEVISONE_GIT} /opt/WeVisOne-VL",
        # vllm/gradio are only for their fast backend / web demo; skipped to keep the image lean
        "grep -v -E '^(vllm|gradio)' /opt/WeVisOne-VL/requirements.txt > /tmp/req.txt",
        "pip install -r /tmp/req.txt",
        "pip install -e /opt/WeVisOne-VL",
    )
    # PP-OCRv6 on CPU (fast enough, and avoids a second CUDA stack next to torch)
    .pip_install("paddlepaddle==3.3.1", "paddleocr==3.7.0")
    .env({"HF_HOME": "/cache/hf", "PADDLE_PDX_CACHE_HOME": "/cache/paddlex", "PYTHONPATH": "/root"})
    .add_local_dir(REPO / "parcel_ocr", "/root/parcel_ocr")
)

cache = modal.Volume.from_name("parcel-ocr-cache", create_if_missing=True)
app = modal.App("parcel-label-ocr")


@app.cls(image=image, gpu=GPU, volumes={"/cache": cache}, timeout=60 * 60, startup_timeout=30 * 60, scaledown_window=120)
class Reader:
    model: str = modal.parameter(default="tencent/WeVisOne-VL-4B")
    ppocr_tier: str = modal.parameter(default="medium")
    pick: str = modal.parameter(default="union")
    runs_json: str = modal.parameter(default='[["crop", "wevisone"]]')

    @modal.enter()
    def load(self) -> None:
        from parcel_ocr.reader import LabelReader

        self.runs = [tuple(r) for r in json.loads(self.runs_json)]
        self.reader = LabelReader(self.model, pick=self.pick, ppocr_tier=self.ppocr_tier, ppocr_device="cpu")
        self.reader.warm_up(self.runs)
        cache.commit()  # persist downloaded weights for the next run

    @modal.method()
    def read(self, name: str, data: bytes, fields: dict[str, str]) -> dict:
        from PIL import Image

        from parcel_ocr.batch import read_fields, to_record
        from parcel_ocr.reader import prepare_image

        img = prepare_image(Image.open(io.BytesIO(data)))
        return to_record(name, fields, read_fields(self.reader, img, fields, self.runs))


@app.local_entrypoint()
def main(
    images: str,
    describe: str = "",
    fields: str = "",
    mode: str = "crop",
    engine: str = "wevisone",
    model: str = "tencent/WeVisOne-VL-4B",
    ppocr_tier: str = "medium",
    pick: str = "union",
    out: str = "outputs/cloud",
    expected: str = "",
    debug: bool = False,
) -> None:
    from types import SimpleNamespace

    from PIL import Image

    from parcel_ocr.batch import list_images, score, write_outputs
    from parcel_ocr.cli_args import runs_from_args
    from parcel_ocr.reader import ReadResult, prepare_image, save_debug

    if fields:
        field_map = json.loads(Path(fields).read_text(encoding="utf-8"))
    else:
        field_map = {f"field_{i}": d.strip() for i, d in enumerate(describe.split(";"), 1) if d.strip()}
    runs = runs_from_args(SimpleNamespace(mode=mode, engine=engine))
    if not field_map and any(m != "full" for m, _ in runs):
        raise SystemExit('give --describe "a;b" or --fields fields.json (or use --mode full)')

    paths = list_images(images)
    reader = Reader(model=model, ppocr_tier=ppocr_tier, pick=pick, runs_json=json.dumps(runs))
    records = []
    for path in paths:  # sequential: one warm GPU container serves all images
        rec = reader.read.remote(path.name, path.read_bytes(), field_map)
        records.append(rec)
        for r in rec["results"]:
            print(f"{path.name}  {r['mode']}/{r['engine']:<9} {r['description'][:40]:<40} -> {r['text']!r}")
        if debug:
            results = [ReadResult(**{k: v for k, v in r.items() if k != "field"}) for r in rec["results"]]
            save_debug(prepare_image(Image.open(path)), [r for r in results if r.mode not in ("ask", "full")],
                       Path(out) / "debug" / f"{path.stem}_debug.png")

    rows = write_outputs(records, field_map, Path(out) / "results.csv")
    print(f"\nwrote {Path(out) / 'results.csv'} and results.jsonl")
    if expected:
        print("\n" + score(rows, field_map, expected))
