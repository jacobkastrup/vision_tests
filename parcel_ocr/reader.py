"""Read a described part of a parcel label with WeVisOne-VL.

Three strategies (``mode``):

* ``crop``   (default) 1) ``referring`` task finds the region you described,
             2) the region is cropped (padded + upscaled) and 3) the ``ocr``
             task reads every word in the crop, joined in reading order.
             Best for small print: the model sees the region at high resolution.
* ``filter`` ``referring`` finds the region, ``ocr`` runs once on the whole
             label, and only words whose centre falls in the region are kept.
* ``ask``    A single free-form question ("Read the ... Reply with only the
             text"). Experimental: the checkpoint is tuned for structured
             outputs, so this tests how much general VQA ability remains.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from .parsing import (
    Region,
    Word,
    is_none_answer,
    pad_box,
    parse_regions,
    parse_words,
    union_box,
    words_in_box,
    words_to_text,
)

MODES = ("crop", "filter", "ask")
DEFAULT_MODEL = os.environ.get("WEVISONE_MODEL", "tencent/WeVisOne-VL-4B")

ASK_TEMPLATE = (
    "This is a photo of a parcel / shipping label. Read the {description}. "
    "Reply with only the exact text as printed, keeping line breaks. "
    "If it is not on the label, reply None."
)


@dataclass
class ReadResult:
    description: str
    mode: str
    text: str | None  # None = region not found / nothing read
    region_box: list[int] | None = None  # pixel x1,y1,x2,y2 on the (rotated) input image
    words: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, str] = field(default_factory=dict)  # prompt -> raw model response
    latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_image(path: str | Path, rotate: int = 0) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if rotate % 360:
        image = image.rotate(-rotate, expand=True)  # positive = clockwise
    return image


def _upscale(image: Image.Image, min_side: int, max_side: int) -> Image.Image:
    w, h = image.size
    scale = min_side / max(1, min(w, h))
    scale = min(scale, max_side / max(w, h))
    if scale <= 1.0:
        return image
    return image.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)


class LabelReader:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        backend: str = "transformers",
        device: str = "auto",
        dtype: str = "bfloat16",
        max_pixels: int | None = None,
        max_new_tokens: int = 2048,
        pick: str = "union",
        crop_min_side: int = 448,
        crop_max_side: int = 1600,
    ) -> None:
        from wevisone_vl import WeVisOneVL  # heavy import, keep lazy

        kwargs: dict[str, Any] = {"backend": backend, "device": device, "dtype": dtype}
        if max_pixels:
            kwargs["max_pixels"] = int(max_pixels)
        self.model = WeVisOneVL(model, **kwargs)
        self.max_new_tokens = max_new_tokens
        if pick not in {"union", "first", "largest"}:
            raise ValueError("pick must be union, first or largest")
        self.pick = pick
        self.crop_min_side = crop_min_side
        self.crop_max_side = crop_max_side

    # -- model calls -----------------------------------------------------------------
    def _run(self, image: Image.Image, task: str, query: str = "", prompt: str | None = None):
        return self.model.predict(
            image,
            task=task,
            query=query,
            prompt=prompt,
            decode_matting=False,
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
        )

    def locate(self, image: Image.Image, description: str) -> tuple[list[Region], Any]:
        result = self._run(image, "referring", description)
        return parse_regions(result.response, *image.size), result

    def ocr(self, image: Image.Image) -> tuple[list[Word], Any]:
        result = self._run(image, "ocr")
        return parse_words(result.response, *image.size), result

    def _pick_box(self, regions: list[Region]):
        boxes = [r.box for r in regions]
        if self.pick == "first":
            return boxes[0]
        if self.pick == "largest":
            return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return union_box(boxes)

    # -- public API ------------------------------------------------------------------
    def read(self, image: Image.Image, description: str, mode: str = "crop") -> ReadResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        started = time.perf_counter()
        out = ReadResult(description=description, mode=mode, text=None)

        if mode == "ask":
            prompt = ASK_TEMPLATE.format(description=description.strip().rstrip("."))
            result = self._run(image, "freeform", prompt=prompt)
            out.raw[result.prompt] = result.response
            answer = result.response.split("<|im_end|>", 1)[0].strip()
            out.text = None if is_none_answer(answer) else answer
            out.latency_s = time.perf_counter() - started
            return out

        regions, located = self.locate(image, description)
        out.raw[located.prompt] = located.response
        if not regions:  # model answered "None": region not on the label
            out.latency_s = time.perf_counter() - started
            return out
        x1, y1, x2, y2 = pad_box(self._pick_box(regions), *image.size)
        out.region_box = [x1, y1, x2, y2]

        if mode == "crop":
            crop = _upscale(image.crop((x1, y1, x2, y2)), self.crop_min_side, self.crop_max_side)
            words, ocr_result = self.ocr(crop)
            sx, sy = (x2 - x1) / crop.width, (y2 - y1) / crop.height
            words = [  # map back to full-image pixel coordinates
                Word(w.text, (x1 + w.box[0] * sx, y1 + w.box[1] * sy, x1 + w.box[2] * sx, y1 + w.box[3] * sy))
                for w in words
            ]
        else:  # filter
            all_words, ocr_result = self.ocr(image)
            words = words_in_box(all_words, (x1, y1, x2, y2))
        out.raw["[ocr] " + ocr_result.prompt] = ocr_result.response

        out.words = [{"text": w.text, "box": [round(v, 1) for v in w.box]} for w in words]
        out.text = words_to_text(words) or None
        out.latency_s = time.perf_counter() - started
        return out


def save_debug(image: Image.Image, results: list[ReadResult], path: str | Path) -> None:
    """Draw located regions (thick) and recognised words (thin) on the label."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    colors = ["#ff3b5c", "#2fbf71", "#3b82f6", "#f59e0b", "#a855f7", "#06b6d4"]
    lw = max(2, round(max(image.size) / 400))
    for i, res in enumerate(results):
        color = colors[i % len(colors)]
        if res.region_box:
            draw.rectangle(res.region_box, outline=color, width=lw * 2)
            draw.text((res.region_box[0] + 4, max(0, res.region_box[1] - 14)), f"{i + 1}: {res.description}", fill=color)
        for w in res.words:
            draw.rectangle(w["box"], outline=color, width=1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
