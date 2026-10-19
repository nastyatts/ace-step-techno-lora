"""Split audio fragments by source track, keeping linked identifiers together."""
from __future__ import annotations

import math
import posixpath
import re

import numpy as np
import pandas as pd


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _path(value: object) -> str:
    text = _text(value).replace("\\", "/")
    return posixpath.normpath(text) if text else ""


def track_groups(frame: pd.DataFrame) -> pd.Series:
    """Group shared trackUri/source_file values, with chunk-name fallback.

    Linking identifiers also handles missing trackUri in some rows and copies
    of one track stored under different source paths. No audio is loaded.
    """
    parent: dict[str, str] = {}

    def root(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    row_keys: list[str] = []
    columns = [c for c in ("trackUri", "source_file", "filepath") if c in frame]
    for position, values in enumerate(frame[columns].itertuples(index=False, name=None)):
        row = dict(zip(columns, values))
        keys: list[str] = []
        uri = _text(row.get("trackUri", ""))
        source = _path(row.get("source_file", ""))
        filepath = _path(row.get("filepath", ""))
        if uri:
            keys.append("uri:" + uri)
        if source:
            keys.append("source:" + source)
        match = re.fullmatch(r"(.+)_chunk_\d+\.[^/.]+", filepath, flags=re.IGNORECASE)
        if match:
            keys.append("chunk_source:" + match.group(1))
        if not keys:
            raise ValueError(
                f"Cannot identify source track at row position {position}. "
                "Supply trackUri/source_file, or a filepath ending in _chunk_NNNN.ext."
            )
        for key in keys:
            left, right = root(keys[0]), root(key)
            if left != right:
                parent[max(left, right)] = min(left, right)
        row_keys.append(keys[0])

    return pd.Series([root(key) for key in row_keys], index=frame.index, dtype="object")


def assign_track_splits(
    frame: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float
) -> pd.Series:
    """Assign approximate ratios of tracks (not fragments), reproducibly."""
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (train_ratio, val_ratio)):
        raise ValueError("Split ratios must be finite numbers between 0 and 1.")
    total = train_ratio + val_ratio
    if total > 1 and not math.isclose(total, 1, abs_tol=1e-12, rel_tol=0):
        raise ValueError("train_ratio + val_ratio must not exceed 1.")
    ratios = np.array([train_ratio, val_ratio, max(0.0, 1.0 - total)])
    ratios /= ratios.sum()
    groups = track_groups(frame)
    unique = np.array(sorted(groups.unique()), dtype="object")
    if not len(unique):
        return pd.Series(index=frame.index, dtype="object")
    if len(unique) < np.count_nonzero(ratios):
        raise ValueError("Too few source tracks for the requested non-empty splits.")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    raw_counts = len(unique) * ratios
    counts = np.floor(raw_counts).astype(int)
    remainder_order = np.argsort(-(raw_counts - counts), kind="stable")
    for index in remainder_order[: len(unique) - int(counts.sum())]:
        counts[index] += 1
    for index in np.flatnonzero((ratios > 0) & (counts == 0)):
        donor = int(np.argmax(counts))
        counts[donor] -= 1
        counts[index] += 1

    train_end, val_end = int(counts[0]), int(counts[0] + counts[1])
    mapping = {key: "train" for key in unique[:train_end]}
    mapping.update({key: "val" for key in unique[train_end:val_end]})
    mapping.update({key: "test" for key in unique[val_end:]})
    return groups.map(mapping)


def split_summary(frame: pd.DataFrame) -> dict[str, object]:
    groups = track_groups(frame)
    labels = frame["split"].reset_index(drop=True)
    if labels.isna().any() or not labels.isin(["train", "val", "test"]).all():
        raise ValueError("Every row must have a train/val/test split label.")
    grouped = pd.DataFrame({"group": groups.to_numpy(), "split": labels.to_numpy()})
    return {
        "fragments": len(frame),
        "source_track_groups": int(groups.nunique()),
        "fragments_per_split": {str(k): int(v) for k, v in labels.value_counts().items()},
        "track_groups_per_split": {
            str(k): int(v) for k, v in grouped.groupby("split")["group"].nunique().items()
        },
        "cross_split_track_groups": int((grouped.groupby("group")["split"].nunique() > 1).sum()),
    }
