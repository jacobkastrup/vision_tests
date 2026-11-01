"""Full-image OCR engines. Each returns words/lines with pixel boxes.

* ``wevisone`` - WeVisOne-VL ``ocr`` task (word level, GPU recommended).
* ``ppocr``    - PaddleOCR PP-OCRv6 det+rec pipeline (line level, fast on CPU).
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from .parsing import Word, parse_words

ENGINES = ("wevisone", "ppocr")


class WeVisOneOCR:
    name = "wevisone"

    def __init__(self, vlm: Any, max_new_tokens: int = 2048) -> None:
        self.vlm = vlm
        self.max_new_tokens = max_new_tokens

    def ocr(self, image: Image.Image) -> tuple[list[Word], str, str]:
        result = self.vlm.predict(
            image, task="ocr", decode_matting=False, max_new_tokens=self.max_new_tokens, temperature=0.0
        )
        return parse_words(result.response, *image.size), result.prompt, result.response


class PPOCRv6:
    """PaddleOCR 3.x with the PP-OCRv6 models (``tier``: medium, small or tiny)."""

    name = "ppocr"

    def __init__(
        self, tier: str = "medium", device: str | None = "cpu", min_score: float = 0.5, mkldnn: bool = False
    ) -> None:
        from paddleocr import PaddleOCR  # heavy import, keep lazy

        kwargs: dict[str, Any] = dict(
            text_detection_model_name=f"PP-OCRv6_{tier}_det",
            text_recognition_model_name=f"PP-OCRv6_{tier}_rec",
            # labels are flat; enable these for crumpled / rotated photos
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # oneDNN (MKLDNN) CPU acceleration crashes PP-OCRv6 under paddlepaddle 3.3.x:
            # "ConvertPirAttribute2RuntimeAttribute not support pir::ArrayAttribute<pir::DoubleAttribute>"
            enable_mkldnn=mkldnn,
        )
        if device:
            kwargs["device"] = device
        self.pipeline = PaddleOCR(**kwargs)
        self.tier = tier
        self.min_score = min_score

    def ocr(self, image: Image.Image) -> tuple[list[Word], str, str]:
        import numpy as np

        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])  # PaddleX expects BGR
        res = self.pipeline.predict(bgr)[0]
        return ppocr_to_words(res, self.min_score), f"PP-OCRv6_{self.tier}", _ppocr_raw(res)


def ppocr_to_words(res: Any, min_score: float = 0.0) -> list[Word]:
    """Convert a PaddleOCR result (``rec_texts`` / ``rec_scores`` / ``rec_boxes``) to Words."""
    words = []
    for text, score, box in zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"]):
        if text.strip() and float(score) >= min_score:
            x1, y1, x2, y2 = (float(v) for v in box)
            words.append(Word(text.strip(), (x1, y1, x2, y2)))
    return words


def _ppocr_raw(res: Any) -> str:
    return "\n".join(
        f"{float(s):.3f}\t{[int(v) for v in b]}\t{t}"
        for t, s, b in zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"])
    )
