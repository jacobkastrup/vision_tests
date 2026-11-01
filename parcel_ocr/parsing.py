"""Pure-Python parsing of WeVisOne-VL responses (no torch / model needed).

WeVisOne-VL writes results as structured text, with box coordinates
normalised to 0..1000:

    referring: <ref>the tracking number</ref><box><120><400><880><460></box>
    ocr:       <ref>COFFEE</ref><box><450><33><850><250></box>, <ref>REAL</ref><box>...</box>
    absent:    None
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_REF_BOX_RE = re.compile(
    r"<ref>(.*?)</ref>\s*<box>(.*?)(?:</box>|(?=<ref>)|$)", re.DOTALL | re.IGNORECASE
)
_COORD_RE = re.compile(r"<(\d{1,4})>")
_END_TOKENS = ("<|im_end|>", "<|endoftext|>")

Box = tuple[float, float, float, float]  # pixel-space x1, y1, x2, y2


@dataclass
class Word:
    text: str
    box: Box

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


@dataclass
class Region:
    label: str
    box: Box


def _clean(text: str) -> str:
    text = text or ""
    for token in _END_TOKENS:
        text = text.split(token, 1)[0]
    return text


def _boxes(body: str, width: int, height: int) -> list[Box]:
    boxes: list[Box] = []
    for segment in body.split(","):
        values = [int(v) for v in _COORD_RE.findall(segment)]
        for i in range(0, len(values) - 3, 4):
            x1, y1, x2, y2 = values[i : i + 4]
            if not all(0 <= v <= 1000 for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height))
    return boxes


def parse_regions(response: str, width: int, height: int) -> list[Region]:
    """All boxes in a referring/detection response, in pixel coordinates."""
    regions: list[Region] = []
    for label, body in _REF_BOX_RE.findall(_clean(response)):
        regions.extend(Region(label.strip(), box) for box in _boxes(body, width, height))
    return regions


def parse_words(response: str, width: int, height: int) -> list[Word]:
    """Words from an ``ocr`` task response, in pixel coordinates."""
    words: list[Word] = []
    for text, body in _REF_BOX_RE.findall(_clean(response)):
        boxes = _boxes(body, width, height)
        if boxes and text.strip():
            words.append(Word(text.strip(), boxes[0]))
    return words


def words_to_text(words: list[Word], line_overlap: float = 0.5) -> str:
    """Join words in reading order: group into lines by vertical overlap, then left->right."""
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda w: w.cy):
        for line in lines:
            top = min(w.box[1] for w in line)
            bottom = max(w.box[3] for w in line)
            overlap = min(bottom, word.box[3]) - max(top, word.box[1])
            if overlap >= line_overlap * min(word.height, bottom - top):
                line.append(word)
                break
        else:
            lines.append([word])
    lines.sort(key=lambda line: sum(w.cy for w in line) / len(line))
    return "\n".join(" ".join(w.text for w in sorted(line, key=lambda w: w.cx)) for line in lines)


def union_box(boxes: list[Box]) -> Box:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def pad_box(box: Box, width: int, height: int, pad_frac: float = 0.08, pad_px: int = 6) -> tuple[int, int, int, int]:
    """Grow a box by a fraction of its size (+ a few pixels) and clamp it to the image."""
    x1, y1, x2, y2 = box
    px = (x2 - x1) * pad_frac + pad_px
    py = (y2 - y1) * pad_frac + pad_px
    return (
        max(0, int(x1 - px)),
        max(0, int(y1 - py)),
        min(width, int(round(x2 + px))),
        min(height, int(round(y2 + py))),
    )


def words_in_box(words: list[Word], box: Box) -> list[Word]:
    """Words whose centre lies inside ``box``."""
    x1, y1, x2, y2 = box
    return [w for w in words if x1 <= w.cx <= x2 and y1 <= w.cy <= y2]


def is_none_answer(response: str) -> bool:
    text = _clean(response).strip().strip(".").strip().lower()
    return text in {"none", "no", "not found", "n/a", ""}
