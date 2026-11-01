"""Full-image OCR engines. Each returns words/lines with pixel boxes.

* ``wevisone`` - WeVisOne-VL ``ocr`` task (word level, GPU recommended).
* ``ppocr``    - PaddleOCR PP-OCRv6 det+rec pipeline (line level, fast on CPU).
"""

from __future__ import annotations

import os
import sys
import time
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
    """PaddleOCR 3.x with the PP-OCRv6 models (``tier``: medium, small or tiny).

    Paddle runs in a separate *worker process*: loading PaddlePaddle and PyTorch in one
    process makes their OpenMP runtimes fight (hangs / extreme slowness on small VMs
    such as Colab). The worker only imports Paddle, and every call has a timeout.
    """

    name = "ppocr"

    def __init__(
        self,
        tier: str = "medium",
        device: str | None = "cpu",
        min_score: float = 0.5,
        mkldnn: bool = False,
        cpu_threads: int | None = None,
        timeout_s: float = 300.0,
        load_timeout_s: float = 900.0,
    ) -> None:
        import multiprocessing as mp

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
            # PaddleOCR defaults to 10 threads; on a 2-core VM that oversubscribes badly
            cpu_threads=cpu_threads or max(1, os.cpu_count() or 1),
        )
        if device:
            kwargs["device"] = device
        self.tier = tier
        self.min_score = min_score
        self.timeout_s = timeout_s

        ctx = mp.get_context("spawn")  # fresh interpreter: no torch, no inherited threads
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_ppocr_worker, args=(child, kwargs), daemon=True)
        self._proc.start()
        child.close()
        started = time.perf_counter()
        status, payload = self._recv(load_timeout_s, "loading PP-OCRv6")
        if status != "ready":
            raise RuntimeError(f"PP-OCRv6 worker failed to start:\n{payload}")
        _log(f"PP-OCRv6_{tier} ready in worker pid {self._proc.pid} "
             f"({kwargs['cpu_threads']} CPU threads, {time.perf_counter() - started:.1f}s)")

    def _recv(self, timeout: float, what: str):
        if not self._conn.poll(timeout):
            self.close()
            raise TimeoutError(
                f"PP-OCRv6 timed out after {timeout:.0f}s while {what}. Try --ppocr-tier small, "
                "a machine with more CPU cores, or --ppocr-device gpu:0 with paddlepaddle-gpu."
            )
        try:
            return self._conn.recv()
        except EOFError:
            raise RuntimeError(f"PP-OCRv6 worker died while {what} (exit code {self._proc.exitcode})") from None

    def ocr(self, image: Image.Image) -> tuple[list[Word], str, str]:
        import numpy as np

        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])  # PaddleX expects BGR
        started = time.perf_counter()
        self._conn.send(bgr)
        status, payload = self._recv(self.timeout_s, f"reading a {image.width}x{image.height} image")
        if status != "ok":
            raise RuntimeError(f"PP-OCRv6 failed:\n{payload}")
        _log(f"PP-OCRv6 {image.width}x{image.height}: {time.perf_counter() - started:.1f}s")
        return ppocr_to_words(payload, self.min_score), f"PP-OCRv6_{self.tier}", _ppocr_raw(payload)

    def close(self) -> None:
        if self._proc.is_alive():
            self._proc.terminate()


def _ppocr_worker(conn: Any, kwargs: dict[str, Any]) -> None:
    """Runs in the child process: load PaddleOCR once, then serve images until EOF."""
    import traceback

    try:
        from paddleocr import PaddleOCR

        pipeline = PaddleOCR(**kwargs)
        conn.send(("ready", None))
    except Exception:
        conn.send(("error", traceback.format_exc()))
        return
    while True:
        try:
            bgr = conn.recv()
        except EOFError:
            return
        try:
            res = pipeline.predict(bgr)[0]
            conn.send(("ok", {
                "rec_texts": list(res["rec_texts"]),
                "rec_scores": [float(v) for v in res["rec_scores"]],
                "rec_boxes": [[float(v) for v in box] for box in res["rec_boxes"]],
            }))
        except Exception:
            conn.send(("error", traceback.format_exc()))


def _log(message: str) -> None:
    print(f"[parcel_ocr] {message}", file=sys.stderr, flush=True)


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
