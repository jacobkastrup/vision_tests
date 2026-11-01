import numpy as np

from parcel_ocr.engines import ppocr_to_words
from parcel_ocr.parsing import words_to_text
from parcel_ocr.reader import needs_vlm, plan_runs


def test_ppocr_result_conversion():
    # shape of a PaddleOCR 3.x OCRResult (dict-like)
    res = {
        "rec_texts": ["1Z 999 AA1", "01 2345 6784", "", "noise"],
        "rec_scores": [0.99, 0.97, 0.9, 0.2],
        "rec_boxes": np.array([[10, 100, 200, 130], [210, 102, 400, 131], [0, 0, 5, 5], [0, 300, 50, 320]]),
    }
    words = ppocr_to_words(res, min_score=0.5)
    assert [w.text for w in words] == ["1Z 999 AA1", "01 2345 6784"]
    assert words[0].box == (10.0, 100.0, 200.0, 130.0)
    assert words_to_text(words) == "1Z 999 AA1 01 2345 6784"


def test_plan_runs():
    assert plan_runs(["crop", "ask"], ["wevisone", "ppocr"]) == [
        ("crop", "wevisone"), ("crop", "ppocr"), ("ask", "wevisone")]
    assert plan_runs(["ask"], ["ppocr"]) == [("ask", "wevisone")]


def test_ppocr_full_needs_no_gpu_model():
    assert not needs_vlm([("full", "ppocr")])
    assert needs_vlm([("crop", "ppocr")])
    assert needs_vlm([("full", "wevisone")])
