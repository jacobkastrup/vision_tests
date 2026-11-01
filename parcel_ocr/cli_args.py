"""Command-line options shared by read_label.py and batch_read.py."""

from __future__ import annotations

import argparse

from .engines import ENGINES
from .reader import DEFAULT_MODEL, MODES, LabelReader, plan_runs


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", default="crop",
                   help=f"comma list of {', '.join(MODES)}, or 'all' (= crop,filter,ask). Default: crop")
    p.add_argument("--engine", default="wevisone",
                   help="OCR engine for crop/filter/full: wevisone, ppocr, or both. Default: wevisone")
    p.add_argument("--pick", choices=("union", "first", "largest"), default="union",
                   help="if the description matches several boxes: merge them (default) or take one")
    p.add_argument("--model", default=DEFAULT_MODEL, help="WeVisOne-VL HF id or local path (env WEVISONE_MODEL)")
    p.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    p.add_argument("--device", default="auto", help="WeVisOne-VL device: auto, cuda:0, cpu, ...")
    p.add_argument("--dtype", default="bfloat16", help="bfloat16 (A10/L4/A100/H100) or float16 (T4, V100)")
    p.add_argument("--max-pixels", type=int, default=None, help="cap on image pixels fed to WeVisOne-VL")
    p.add_argument("--ppocr-tier", choices=("medium", "small", "tiny"), default="medium")
    p.add_argument("--ppocr-device", default="cpu", help="PP-OCRv6 device: cpu (default) or gpu:0")


def runs_from_args(args: argparse.Namespace) -> list[tuple[str, str]]:
    modes = ["crop", "filter", "ask"] if args.mode == "all" else [m.strip() for m in args.mode.split(",")]
    engines = list(ENGINES) if args.engine == "both" else [e.strip() for e in args.engine.split(",")]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"unknown mode {m!r}; choose from {MODES} or 'all'")
    for e in engines:
        if e not in ENGINES:
            raise SystemExit(f"unknown engine {e!r}; choose from {ENGINES} or 'both'")
    return plan_runs(modes, engines)


def reader_from_args(args: argparse.Namespace) -> LabelReader:
    return LabelReader(
        args.model, backend=args.backend, device=args.device, dtype=args.dtype,
        max_pixels=args.max_pixels, pick=args.pick,
        ppocr_tier=args.ppocr_tier, ppocr_device=args.ppocr_device,
    )
