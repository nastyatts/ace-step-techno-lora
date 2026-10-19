"""Reassign existing metadata splits without repeating audio/LLM processing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

if __package__:
    from .track_split import assign_track_splits, split_summary
else:
    from track_split import assign_track_splits, split_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--report_json", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_train", type=float, default=0.90)
    parser.add_argument("--split_val", type=float, default=0.05)
    args = parser.parse_args()
    if args.input_csv.resolve() == args.output_csv.resolve():
        parser.error("Use a different output path to preserve the original experiment metadata.")
    if args.report_json and args.report_json.resolve() in {
        args.input_csv.resolve(), args.output_csv.resolve()
    }:
        parser.error("The JSON report must have a different path from the CSV files.")

    # Preserve every existing field as text; only the split column is changed.
    frame = pd.read_csv(args.input_csv, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    before = split_summary(frame) if "split" in frame else None
    frame["split"] = assign_track_splits(frame, args.seed, args.split_train, args.split_val)
    after = split_summary(frame)
    if after["cross_split_track_groups"]:
        raise RuntimeError("Source tracks overlap across splits; no output was written.")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    report = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "seed": args.seed,
        "requested_track_ratios": {
            "train": args.split_train,
            "val": args.split_val,
            "test": max(0.0, 1.0 - args.split_train - args.split_val),
        },
        "before": before,
        "after": after,
        "note": "Only split labels are reassigned. Existing model weights are not changed.",
    }
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
