"""Read a described part of a parcel label.

Modes (how the region is found and read):

* ``crop``   (default) WeVisOne-VL ``referring`` finds the region you described,
             it is cropped (padded + upscaled) and an OCR engine reads the crop.
* ``filter`` ``referring`` finds the region, the OCR engine reads the whole
             label, and only words/lines whose centre is in the region are kept.
* ``ask``    One free-form question to WeVisOne-VL (experimental; no OCR engine).
* ``full``   No region: OCR the whole label (a baseline; ignores the description).

OCR engines (``engine``): ``wevisone`` (WeVisOne-VL ``ocr`` task) or ``ppocr``
(PaddleOCR PP-OCRv6). Region finding always uses WeVisOne-VL, since PP-OCR cannot
understand a description; ``full`` + ``ppocr`` needs no GPU at all.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageOps

from .engines import ENGINES, PPOCRv6, WeVisOneOCR
from .parsing import Region, Word, is_none_answer, pad_box, parse_regions, union_box, words_in_box, words_to_text

MODES = ("crop", "filter", "ask", "full")
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
    engine: str
    text: str | None  # None = region not found / nothing read
    region_box: list[int] | None = None  # pixel x1,y1,x2,y2 on the (rotated) input image
    words: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, str] = field(default_factory=dict)  # prompt -> raw model response
    latency_s: float = 0.0

    @property
    def run(self) -> str:
        return f"{self.mode}/{self.engine}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_runs(modes: Iterable[str], engines: Iterable[str]) -> list[tuple[str, str]]:
    """Valid (mode, engine) combinations. ``ask`` only exists for WeVisOne-VL."""
    engines = list(engines)
    runs = []
    for mode in modes:
        if mode == "ask":
            runs.append(("ask", "wevisone"))
        else:
            runs.extend((mode, engine) for engine in engines)
    return list(dict.fromkeys(runs))


def needs_vlm(runs: Iterable[tuple[str, str]]) -> bool:
    return any(mode != "full" or engine == "wevisone" for mode, engine in runs)


def load_image(path: str | Path, rotate: int = 0) -> Image.Image:
    return prepare_image(Image.open(path), rotate)


def prepare_image(image: Image.Image, rotate: int = 0) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if rotate % 360:
        image = image.rotate(-rotate, expand=True)  # positive = clockwise
    return image


def _upscale(image: Image.Image, min_side: int, max_side: int) -> Image.Image:
    w, h = image.size
    scale = min(min_side / max(1, min(w, h)), max_side / max(w, h))
    if scale <= 1.0:
        return image
    return image.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)


class LabelReader:
    """Models are loaded lazily, so e.g. ``full``+``ppocr`` never loads WeVisOne-VL."""

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
        ppocr_tier: str = "medium",
        ppocr_device: str | None = "cpu",
        crop_min_side: int = 448,
        crop_max_side: int = 1600,
    ) -> None:
        if pick not in {"union", "first", "largest"}:
            raise ValueError("pick must be union, first or largest")
        self._vlm_kwargs: dict[str, Any] = {"backend": backend, "device": device, "dtype": dtype}
        if max_pixels:
            self._vlm_kwargs["max_pixels"] = int(max_pixels)
        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self.pick = pick
        self.ppocr_tier = ppocr_tier
        self.ppocr_device = ppocr_device
        self.crop_min_side = crop_min_side
        self.crop_max_side = crop_max_side
        self._vlm = None
        self._engines: dict[str, Any] = {}

    # -- lazy model loading ---------------------------------------------------------
    @property
    def vlm(self):
        if self._vlm is None:
            from wevisone_vl import WeVisOneVL

            self._vlm = WeVisOneVL(self.model_name, **self._vlm_kwargs)
        return self._vlm

    def engine(self, name: str):
        if name not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}")
        if name not in self._engines:
            if name == "wevisone":
                self._engines[name] = WeVisOneOCR(self.vlm, self.max_new_tokens)
            else:
                self._engines[name] = PPOCRv6(self.ppocr_tier, self.ppocr_device)
        return self._engines[name]

    def warm_up(self, runs: Iterable[tuple[str, str]]) -> None:
        """Load every model the given (mode, engine) runs need, up front."""
        for mode, engine in runs:
            if mode != "full":
                _ = self.vlm  # region finding / ask
            if mode != "ask":
                self.engine(engine)

    # -- steps ------------------------------------------------------------------------
    def _run_vlm(self, image: Image.Image, task: str, query: str = "", prompt: str | None = None):
        return self.vlm.predict(
            image, task=task, query=query, prompt=prompt, decode_matting=False,
            max_new_tokens=self.max_new_tokens, temperature=0.0,
        )

    def locate(self, image: Image.Image, description: str) -> tuple[list[Region], Any]:
        result = self._run_vlm(image, "referring", description)
        return parse_regions(result.response, *image.size), result

    def _pick_box(self, regions: list[Region]):
        boxes = [r.box for r in regions]
        if self.pick == "first":
            return boxes[0]
        if self.pick == "largest":
            return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return union_box(boxes)

    # -- public API ------------------------------------------------------------------
    def read(self, image: Image.Image, description: str, mode: str = "crop", engine: str = "wevisone") -> ReadResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "ask":
            engine = "wevisone"
        started = time.perf_counter()
        out = ReadResult(description=description, mode=mode, engine=engine, text=None)

        def done() -> ReadResult:
            out.latency_s = time.perf_counter() - started
            return out

        if mode == "ask":
            prompt = ASK_TEMPLATE.format(description=description.strip().rstrip("."))
            result = self._run_vlm(image, "freeform", prompt=prompt)
            out.raw[result.prompt] = result.response
            answer = result.response.split("<|im_end|>", 1)[0].strip()
            out.text = None if is_none_answer(answer) else answer
            return done()

        ocr_engine = self.engine(engine)
        if mode == "full":
            words, prompt, raw = ocr_engine.ocr(image)
        else:
            regions, located = self.locate(image, description)
            out.raw[located.prompt] = located.response
            if not regions:  # model answered "None": region not on the label
                return done()
            x1, y1, x2, y2 = pad_box(self._pick_box(regions), *image.size)
            out.region_box = [x1, y1, x2, y2]
            if mode == "crop":
                crop = _upscale(image.crop((x1, y1, x2, y2)), self.crop_min_side, self.crop_max_side)
                words, prompt, raw = ocr_engine.ocr(crop)
                sx, sy = (x2 - x1) / crop.width, (y2 - y1) / crop.height
                words = [  # map back to full-image pixel coordinates
                    Word(w.text, (x1 + w.box[0] * sx, y1 + w.box[1] * sy, x1 + w.box[2] * sx, y1 + w.box[3] * sy))
                    for w in words
                ]
            else:  # filter
                all_words, prompt, raw = ocr_engine.ocr(image)
                words = words_in_box(all_words, (x1, y1, x2, y2))
        out.raw[f"[{engine}] {prompt}"] = raw
        out.words = [{"text": w.text, "box": [round(v, 1) for v in w.box]} for w in words]
        out.text = words_to_text(words) or None
        return done()


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
            draw.text((res.region_box[0] + 4, max(0, res.region_box[1] - 14)),
                      f"{i + 1}: {res.description} [{res.run}]", fill=color)
        for w in res.words:
            draw.rectangle(w["box"], outline=color, width=1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
