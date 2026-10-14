"""Shared dataset paths and input validation for the minimum-parameter model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = (0.6, 0.7, 0.8, 0.9, 1.0)


def resolve_dataset(name=None):
    name = name or "sift100m"
    for candidate in (Path(name), Path(__file__).resolve().parent / name):
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Dataset directory not found: {name}")


def model_directory(dataset):
    return resolve_dataset(dataset) / "output"


def parameter_bounds(model_dir):
    metrics = json.loads((Path(model_dir) / "metrics.json").read_text(encoding="utf-8"))
    return metrics["data"]["learn"]["SearchParameter_grid"]


def read_lids(path):
    frame = pd.read_csv(path, header=None, skip_blank_lines=False, dtype=np.float32)
    if frame.shape[1] != 1 or frame.empty or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path}: expected one finite numeric LID per row")
    return frame.iloc[:, 0].to_numpy()
