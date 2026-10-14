"""Measure saved LID models without retraining or overwriting predictions.

Usage: python benchmark_inference.py --dataset sift100m
       python benchmark_inference.py --all
Times feature construction and model prediction for each query.
Excludes file loading, distance retrieval and load-time preprocessing.
"""

import argparse
import gc
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import lightgbm as lgb
import numpy as np

import gbdt


def summarize(values):
    spread = stdev(values)
    return {
        "mean_ms_per_prediction": mean(values),
        "round_std_ms_per_prediction": spread,
        "mean_standard_error_ms": spread / len(values) ** 0.5,
        "round_means_ms_per_prediction": values,
    }


def benchmark(dataset, rounds=15, queries=100, warmup=100):
    folder = Path(dataset).resolve()
    print(f"[{folder.name}] Loading model and query distances...", flush=True)
    model = lgb.Booster(model_file=str(folder / "lgbm_model.txt"))
    distances, vectors, labels = gbdt.load_dataset(
        str(folder / "query_head.csv"), str(folder / "query_lid.csv"),
        str(folder / "query_list.csv"), folder.name,
    )
    if not len(distances) or len(distances) != len(vectors) or len(distances) != len(labels):
        raise ValueError("Empty or mismatched query data")
    rng = np.random.default_rng(42)
    indices = rng.choice(len(distances), min(queries, len(distances)), replace=False)

    def infer_one(i):
        # Feature construction and Python/LightGBM invocation are BOTH timed.
        features, _ = gbdt.build_feature_matrix(
            distances[i:i + 1], [vectors[i]], save_csv=False, verbose=False)
        features = np.ascontiguousarray(features, dtype=np.float64)
        return model.predict(features, num_threads=1)[0]

    # Verify per-query features against the original matrix implementation.
    check_indices = indices[:min(8, len(indices))]
    reference, names = gbdt.build_feature_matrix(
        distances[check_indices], [vectors[i] for i in check_indices],
        save_csv=False, verbose=False)
    if reference.shape[1] != model.num_feature():
        raise ValueError("Saved model does not match current feature extraction")
    for row, i in zip(reference, check_indices):
        single, _ = gbdt.build_feature_matrix(
            distances[i:i + 1], [vectors[i]], save_csv=False, verbose=False)
        np.testing.assert_allclose(single[0], row, rtol=1e-12, atol=1e-12)
        if not np.isfinite(infer_one(i)):
            raise ValueError("Non-finite prediction")

    print(f"[{folder.name}] {model.num_trees()} trees; warmup {warmup} queries; "
          f"{rounds} rounds x {len(indices)} queries; features + prediction, 1 thread.", flush=True)
    for j in range(warmup):
        infer_one(indices[j % len(indices)])
    values = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for round_id in range(rounds):
            print(f"[{folder.name}] Round {round_id + 1}/{rounds}...", flush=True)
            start = time.perf_counter_ns()
            for i in indices:
                infer_one(i)
            elapsed = time.perf_counter_ns() - start
            values.append(elapsed / 1e6 / len(indices))
            print(f"  features + prediction = {values[-1]:.8f} ms/query", flush=True)
    finally:
        if gc_enabled:
            gc.enable()
    result = {
        "dataset": folder.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "threads": 1, "rounds": rounds, "warmup_single_calls": warmup,
        "single_rows_per_round": len(indices),
        "scope": "Per-query feature construction plus Python LightGBM prediction; excludes model loading, file I/O, distance retrieval, and load_dataset distance preprocessing",
        "note": "Fixed random query subset; round variability does not capture query sampling error.",
        "feature_names": names, "model_trees": model.num_trees(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "lightgbm": lgb.__version__,
                        "numpy": np.__version__},
        "single_query_features_and_prediction": summarize(values),
    }
    destination = folder / "inference_timing.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    timing = result["single_query_features_and_prediction"]
    print(f"{folder.name}: features + prediction mean = {timing['mean_ms_per_prediction']:.8f} ms; "
          f"round std = {timing['round_std_ms_per_prediction']:.8f} ms", flush=True)
    print(f"Saved {destination}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--queries", type=int, default=100,
                        help="Single-query samples per round (default: 100)")
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    if args.rounds < 2 or args.queries < 1 or args.warmup < 1:
        parser.error("Require rounds >= 2, queries >= 1, warmup >= 1")
    datasets = [args.dataset] if args.dataset else [
        str(Path(__file__).resolve().parent / name)
        for name in ("sift100m", "deep100m", "gist1m", "text2img", "msmarco")
    ]
    for dataset in datasets:
        benchmark(dataset, args.rounds, args.queries, args.warmup)


if __name__ == "__main__":
    main()
