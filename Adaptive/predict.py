"""Batch prediction from query_lid.csv; each output row is SearchParameter,latency(ms).

Usage:
    python predict.py sift100m

Outputs: <dataset>/output/recall_<target>.csv
Input order and duplicate LIDs are preserved. Outputs have no header.
"""

import argparse
import gc
import json
import time
from statistics import mean, stdev
import lightgbm as lgb
import numpy as np
from model_common import TARGETS, resolve_dataset, model_directory, parameter_bounds, read_lids

# 两个模型使用相同线程数；计时不包含加载、特征准备、取整和文件读写。
INFERENCE_THREADS = 1
TIMING_WARMUP = 100
TIMING_ROUNDS = 15
TIMING_QUERIES = 10000


def benchmark_inference(models, features):
    """Time single-row Python calls and batch throughput separately.

    Prebuilt inputs and untimed warmup reduce setup effects. Each timing window
    covers many predictions; model order alternates to reduce ordering bias.
    Round-to-round spread describes noise, not a guaranteed measurement error.
    """
    rng = np.random.default_rng(42)
    indices = rng.choice(len(features), min(TIMING_QUERIES, len(features)), replace=False)
    rows = [features[i:i+1] for i in indices]
    if not rows:
        raise ValueError("Cannot benchmark empty input")
    for model in models.values():
        for i in range(TIMING_WARMUP):
            model.predict(rows[i % len(rows)], num_threads=INFERENCE_THREADS)
        model.predict(features, num_threads=INFERENCE_THREADS)

    timings = {name: {"single": [], "batch": []} for name in models}
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for round_id in range(TIMING_ROUNDS):
            names = list(models)
            if round_id % 2:
                names.reverse()
            for name in names:
                predict = models[name].predict
                start = time.perf_counter_ns()
                for row in rows:
                    predict(row, num_threads=INFERENCE_THREADS)
                elapsed = time.perf_counter_ns() - start
                timings[name]["single"].append(elapsed / 1e6 / len(rows))
                start = time.perf_counter_ns()
                predict(features, num_threads=INFERENCE_THREADS)
                elapsed = time.perf_counter_ns() - start
                timings[name]["batch"].append(elapsed / 1e6 / len(features))
    finally:
        if gc_enabled:
            gc.enable()

    def summarize(values):
        return {"mean_ms_per_prediction": mean(values),
                "round_std_ms_per_prediction": stdev(values),
                "mean_standard_error_ms": stdev(values) / len(values) ** 0.5,
                "round_means_ms_per_prediction": values}

    results = {"threads": INFERENCE_THREADS, "rounds": TIMING_ROUNDS,
               "single_rows_per_round": len(rows), "batch_rows_per_round": len(features),
               "scope": "Python LightGBM predict only; excludes loading, feature preparation, rounding, clipping, file I/O",
               "models": {name: {"single_query": summarize(t["single"]),
                                  "batch_amortized": summarize(t["batch"])}
                          for name, t in timings.items()}}
    for name, result in results["models"].items():
        single = result["single_query"]
        batch = result["batch_amortized"]
        print(f"{name}: single-query mean = {single['mean_ms_per_prediction']:.8f} ms "
              f"(round std = {single['round_std_ms_per_prediction']:.8f} ms); "
              f"batch amortized = {batch['mean_ms_per_prediction']:.8f} ms", flush=True)
    return results


def predict_dataset(dataset):
    data_dir = resolve_dataset(dataset)
    model_dir = model_directory(data_dir)
    source = data_dir / "query_lid.csv"
    lids = read_lids(source)
    pmin, pmax = parameter_bounds(model_dir)
    parameter_model = lgb.Booster(model_file=str(model_dir / "model_SearchParameter.txt"))
    latency_model = lgb.Booster(model_file=str(model_dir / "model_latency.txt"))

    # Predict all targets in a batch; round search budgets upward, then clip.
    features = np.empty((len(TARGETS), len(lids), 2), dtype=np.float32)
    features[:, :, 0] = lids
    features[:, :, 1] = np.array(TARGETS, dtype=np.float32)[:, None]
    flat = features.reshape(-1, 2)
    parameters = np.clip(np.ceil(parameter_model.predict(flat, num_threads=INFERENCE_THREADS)), pmin, pmax)
    latencies = latency_model.predict(flat, num_threads=INFERENCE_THREADS)
    if not (np.isfinite(parameters).all() and np.isfinite(latencies).all()):
        raise ValueError(f"{data_dir.name}: non-finite model predictions")
    parameters = parameters.reshape(len(TARGETS), len(lids))
    latencies = latencies.reshape(len(TARGETS), len(lids))

    output_dir = model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, target in enumerate(TARGETS):
        destination = output_dir / f"recall_{target:.1f}.csv"
        np.savetxt(destination, np.column_stack([parameters[i], latencies[i]]),
                   delimiter=",", fmt=["%d", "%.10g"])
        print(f"{data_dir.name}: recall={target:.1f}, {len(lids)} queries -> {destination}")

    timing = benchmark_inference({"SearchParameter": parameter_model, "latency": latency_model}, flat)
    (output_dir / "inference_timing.json").write_text(
        json.dumps(timing, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset name or directory")
    args = parser.parse_args()
    predict_dataset(args.dataset)


if __name__ == "__main__":
    main()
