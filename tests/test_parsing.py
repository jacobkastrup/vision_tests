"""Parsing tests built from real WeVisOne-VL responses (from the official tutorials)."""

from parcel_ocr.parsing import (
    is_none_answer,
    pad_box,
    parse_regions,
    parse_words,
    union_box,
    words_in_box,
    words_to_text,
)

# coffee_sign.jpg, 424x640, task="ocr"
OCR_RESPONSE = (
    "<ref>COFFEE</ref><box><450><33><850><250></box>, <ref>REAL</ref><box><94><62><406><245></box>, "
    "<ref>&</ref><box><443><116><540><179></box>, <ref>Tea</ref><box><308><184><452><242></box>, "
    "<ref>Café</ref><box><470><187><666><252></box>, <ref>10</ref><box><107><289><193><342></box>, "
    "<ref>METERS</ref><box><251><291><751><357></box>, <ref>AHEAD</ref><box><296><363><647><423></box>, "
    "<ref>TURN</ref><box><135><422><410><487></box>, <ref>RIGHT</ref><box><457><429><775><491></box>, "
    "<ref>IN</ref><box><424><492><515><547></box>, <ref>SMALL</ref><box><129><544><480><606></box>, "
    "<ref>ALLEY</ref><box><512><548><799><609></box><|im_end|>"
)
W, H = 424, 640


def test_parse_words_count_and_scale():
    words = parse_words(OCR_RESPONSE, W, H)
    assert len(words) == 13
    assert words[0].text == "COFFEE"
    assert words[0].box == (0.45 * W, 0.033 * H, 0.85 * W, 0.25 * H)


def test_reading_order():
    text = words_to_text(parse_words(OCR_RESPONSE, W, H))
    lines = text.splitlines()
    assert lines[-1] == "SMALL ALLEY"
    assert "TURN RIGHT" in lines
    assert "10 METERS" in lines


def test_words_in_region():
    words = parse_words(OCR_RESPONSE, W, H)
    # referring box for "SMALL ALLEY" from the tutorial
    region = parse_regions("<ref>SMALL ALLEY</ref><box><122><536><804><614></box>", W, H)[0]
    assert words_to_text(words_in_box(words, region.box)) == "SMALL ALLEY"


def test_multi_box_referring_and_union():
    resp = "<ref>dog</ref><box><13><274><399><949>,<379><222><620><954></box>"
    regions = parse_regions(resp, 1000, 1000)
    assert len(regions) == 2
    assert union_box([r.box for r in regions]) == (13, 222, 620, 954)


def test_none_and_invalid():
    assert parse_regions("None", 100, 100) == []
    assert is_none_answer("None.")
    assert is_none_answer("None<|im_end|>")
    assert not is_none_answer("1Z999AA10123456784")
    # degenerate box (x2 <= x1) is dropped
    assert parse_regions("<ref>x</ref><box><500><10><400><20></box>", 100, 100) == []


def test_pad_box_clamps():
    assert pad_box((0, 0, 100, 100), 100, 100) == (0, 0, 100, 100)
    x1, y1, x2, y2 = pad_box((40, 40, 60, 60), 100, 100)
    assert x1 < 40 and y1 < 40 and x2 > 60 and y2 > 60
