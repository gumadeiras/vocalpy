#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the mouse pipeline on shipped example audio and compare against baselines."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


DEFAULT_FILES = ("mouse_1", "mouse_2")


@dataclass
class ComparisonResult:
    name: str
    baseline_count: int
    current_count: int
    matched_count: int
    extra_rows: pd.DataFrame
    missing_rows: pd.DataFrame
    top1_mismatches: int
    top2_mismatches: int
    max_start_delta_ms: float
    max_end_delta_ms: float


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=Path("examples/audios"))
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--tolerance-ms", type=float, default=3.0)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--fail-on-drift", action="store_true")
    parser.add_argument("files", nargs="*", default=list(DEFAULT_FILES))
    return parser.parse_args(argv)


def run_pipeline(audio_dir: Path, work_dir: Path, names: list[str]) -> None:
    for name in names:
        source_audio = audio_dir / f"{name}.wav"
        run_dir = work_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        run_audio = run_dir / source_audio.name
        shutil.copy2(source_audio, run_audio)
        subprocess.run(
            [sys.executable, "-m", "vocalpy.cli", "-a", "mouse", "-p", str(run_audio)],
            check=True,
        )


def align_rows(base: pd.DataFrame, current: pd.DataFrame, tolerance_seconds: float):
    matches = []
    used_current = set()
    for base_index, base_row in base.iterrows():
        matched_index = None
        for current_index, current_row in current.iterrows():
            if current_index in used_current:
                continue
            if (
                abs(base_row["start(s)"] - current_row["start(s)"]) <= tolerance_seconds
                and abs(base_row["end(s)"] - current_row["end(s)"]) <= tolerance_seconds
            ):
                matched_index = current_index
                break
        if matched_index is not None:
            used_current.add(matched_index)
            matches.append((base_index, matched_index))

    missing = base.drop(index=[base_index for base_index, _ in matches])
    extra = current.drop(index=list(used_current))
    return matches, missing, extra


def compare_pair(name: str, audio_dir: Path, work_dir: Path, tolerance_ms: float) -> ComparisonResult:
    baseline = pd.read_csv(audio_dir / f"{name}_outputs" / f"{name}_stats.csv")
    current = pd.read_csv(work_dir / name / f"{name}_outputs" / f"{name}_stats.csv")
    matches, missing, extra = align_rows(baseline, current, tolerance_ms / 1000.0)

    if matches:
        matched_baseline = baseline.loc[[base_index for base_index, _ in matches]].reset_index(drop=True)
        matched_current = current.loc[[current_index for _, current_index in matches]].reset_index(drop=True)
        max_start_delta_ms = float((matched_current["start(s)"] - matched_baseline["start(s)"]).abs().max() * 1000)
        max_end_delta_ms = float((matched_current["end(s)"] - matched_baseline["end(s)"]).abs().max() * 1000)
        top1_mismatches = int((matched_current["class_top1"] != matched_baseline["class_top1"]).sum())
        top2_mismatches = int((matched_current["class_top2"] != matched_baseline["class_top2"]).sum())
    else:
        max_start_delta_ms = 0.0
        max_end_delta_ms = 0.0
        top1_mismatches = 0
        top2_mismatches = 0

    return ComparisonResult(
        name=name,
        baseline_count=len(baseline),
        current_count=len(current),
        matched_count=len(matches),
        extra_rows=extra,
        missing_rows=missing,
        top1_mismatches=top1_mismatches,
        top2_mismatches=top2_mismatches,
        max_start_delta_ms=max_start_delta_ms,
        max_end_delta_ms=max_end_delta_ms,
    )


def print_result(result: ComparisonResult) -> None:
    print(f"== {result.name} ==")
    print(
        f"baseline={result.baseline_count} current={result.current_count} "
        f"matched={result.matched_count} extra={len(result.extra_rows)} missing={len(result.missing_rows)}"
    )
    print(
        f"max_start_delta_ms={result.max_start_delta_ms:.3f} "
        f"max_end_delta_ms={result.max_end_delta_ms:.3f} "
        f"class_top1_mismatches={result.top1_mismatches} "
        f"class_top2_mismatches={result.top2_mismatches}"
    )
    if not result.extra_rows.empty:
        cols = ["start(s)", "end(s)", "duration(ms)", "avg_intensity", "bg_intensity", "class_top1", "class_top2"]
        print("extra rows:")
        print(result.extra_rows[cols].to_string(index=False))
    if not result.missing_rows.empty:
        cols = ["start(s)", "end(s)", "duration(ms)", "avg_intensity", "bg_intensity", "class_top1", "class_top2"]
        print("missing rows:")
        print(result.missing_rows[cols].to_string(index=False))
    print()


def main(argv=None):
    args = parse_args(argv)
    temp_dir = None
    work_dir = args.work_dir
    if work_dir is None:
        temp_dir = TemporaryDirectory(prefix="vocalpy-mouse-compare-")
        work_dir = Path(temp_dir.name)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_pipeline(args.audio_dir, work_dir, args.files)
        results = [compare_pair(name, args.audio_dir, work_dir, args.tolerance_ms) for name in args.files]
        drift_found = False
        for result in results:
            print_result(result)
            if result.extra_rows.shape[0] or result.missing_rows.shape[0] or result.top1_mismatches or result.top2_mismatches:
                drift_found = True
        print(f"work_dir={work_dir}")
        return 1 if (args.fail_on_drift and drift_found) else 0
    finally:
        if temp_dir is not None and not args.keep_work_dir:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
