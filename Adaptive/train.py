"""Train minimum attained parameters with a configurable label margin.

python train_min_margin.py sift100m

Training evaluates and saves models, metrics, a heatmap, and two slice figures
in output/. Batch query prediction remains a separate script. Margin is applied
to training labels only.
Raw CSVs are scanned in chunks; query IDs are reconstructed from verified row
order across parameter blocks, never from LID uniqueness.
"""

# 标签余量：0.20 表示最小达标参数增加 20%；改为 0.0 即不加余量。
MARGIN_RATIO = 0.3
# 只对这些目标 recall 增加余量；例如 (0.8, 0.9, 1.0)。
MARGIN_TARGETS = (0.9,1.0)

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from model_common import TARGETS as PREDICTION_TARGETS, resolve_dataset, model_directory, read_lids

TARGETS = np.arange(1, 11, dtype=np.float32) / 10
EPS = 1e-6


def blocks(path):
    pending = []
    previous = None
    for chunk in pd.read_csv(path, header=None, dtype=np.float32, chunksize=250000):
        if chunk.shape[1] != 4 or not np.isfinite(chunk.to_numpy()).all():
            raise ValueError(f"{path}: expected four finite numeric columns")
        values = chunk.to_numpy()
        p = values[:, 2]
        if not np.equal(p, np.rint(p)).all():
            raise ValueError(f"{path}: non-integer search parameter")
        cuts = np.r_[0, np.flatnonzero(p[1:] != p[:-1]) + 1, len(p)]
        for start, end in zip(cuts[:-1], cuts[1:]):
            parameter = int(p[start])
            if previous is not None and parameter != previous:
                yield previous, np.concatenate(pending)
                pending = []
            pending.append(values[start:end].copy())
            previous = parameter
    if pending:
        yield previous, np.concatenate(pending)


def load_profile(folder, name):
    iterator = iter(blocks(folder / f"{name}.csv"))
    first_p, first = next(iterator)
    lids = first[:, 0].copy()
    n = len(lids)
    lid_file = folder / f"{name}_lid.csv"
    offset = None
    if lid_file.exists():
        original = pd.read_csv(lid_file, header=None, dtype=np.float32,
                               skip_blank_lines=False).to_numpy()
        if original.shape[1] != 1 or not np.isfinite(original).all():
            raise ValueError(f"{lid_file}: invalid LID input")
        offset = len(original) - n
        if offset < 0 or not np.array_equal(original[offset:, 0], lids):
            raise ValueError(f"{name}: first parameter block does not match LID file suffix")
    parameters, recalls, latencies = [], [], []
    duplicates = {}

    def add(p, values):
        if len(values) % n:
            raise ValueError(f"{name}, parameter={p}: incomplete query block")
        count = len(values) // n
        for repeat in range(count):
            if not np.array_equal(values[repeat*n:(repeat+1)*n, 0], lids):
                raise ValueError(f"{name}, parameter={p}: query order changed")
        if p in parameters:
            raise ValueError(f"{name}: non-consecutive repeated parameter {p}")
        if count > 1:
            # Keep one original paired measurement per query, avoiding extra
            # weighting or synthetic recall values from repeated sweep blocks.
            duplicates[str(p)] = count - 1
        parameters.append(p)
        recalls.append(values[:n, 1].copy())
        latencies.append(values[:n, 3].copy())

    add(first_p, first)
    for p, values in iterator:
        add(p, values)
    order = np.argsort(parameters)
    grid = np.array(parameters, dtype=np.int32)[order]
    R = np.stack(recalls, axis=1)[:, order]
    L = np.stack(latencies, axis=1)[:, order]
    if (R < 0).any() or (R > 1).any() or (L < 0).any():
        raise ValueError(f"{name}: recall or latency out of range")
    info = {"queries": n, "LID_file_prefix_not_profiled": offset,
            "discarded_repeat_blocks": duplicates,
            "repeat_policy": "first paired measurement; every repeat order verified",
            "SearchParameter_grid": [int(grid[0]), int(grid[-1])]}
    print(f"{folder.name}/{name}: {n} query IDs, {len(grid)} parameters, "
          f"unprofiled prefix={offset}, repeats={duplicates}", flush=True)
    return lids, grid, R, L, info


def make_labels(lids, grid, R, L):
    query_ids, features, parameters, latencies = [], [], [], []
    stats = {}
    for target in TARGETS:
        hit = R >= target - EPS
        attainable = hit.any(axis=1)
        ids = np.flatnonzero(attainable)
        chosen = np.argmax(hit[ids], axis=1)
        ratio = MARGIN_RATIO if any(abs(float(target)-r) < EPS for r in MARGIN_TARGETS) else 0.0
        minimum = grid[chosen]
        # Adding ceil(ratio * minimum) avoids rounding an integer budget up
        # twice due to floating-point representation of 1 + ratio.
        desired = np.minimum(grid[-1], minimum + np.ceil(ratio * minimum))
        eligible = hit[ids] & (grid[None, :] >= desired[:, None])
        available = eligible.any(axis=1)
        chosen = np.argmax(eligible, axis=1)
        # If measured recall regresses above the margin, retain the largest
        # qualifying measured setting and count this exception explicitly.
        chosen[~available] = len(grid)-1-np.argmax(hit[ids][~available, ::-1], axis=1)
        query_ids.append(ids)
        features.append(np.column_stack([lids[ids], np.full(len(ids), target)]))
        parameters.append(grid[chosen])
        latencies.append(L[ids, chosen])
        stats[f"{target:.1f}"] = {"attainable": int(len(ids)),
            "unattainable": int((~attainable).sum()),
            "margin_ratio": ratio,
            "margin_fallback": int((~available).sum())}
    return (np.concatenate(query_ids), np.concatenate(features).astype(np.float32),
            np.concatenate(parameters), np.concatenate(latencies), stats)


def evaluate(mp, ml, lids, grid, R, L, mask):
    ids = np.flatnonzero(mask)
    result = {}
    for target in PREDICTION_TARGETS:
        X = np.column_stack([lids[ids], np.full(len(ids), target)]).astype(np.float32)
        p = np.clip(np.ceil(mp.predict(X)), grid[0], grid[-1])
        indices = np.searchsorted(grid, p)
        observed = R[ids, indices]
        true_latency = L[ids, indices]
        eligible = R[ids] >= target-EPS
        attainable = eligible.any(axis=1)
        minimum = np.argmax(eligible, axis=1)
        hit = observed >= target-EPS
        result[f"{target:.1f}"] = {
            "queries": len(ids), "attainable_queries": int(attainable.sum()),
            "hit_rate_all": float(hit.mean()),
            "hit_rate_attainable": float(hit[attainable].mean()) if attainable.any() else None,
            "mean_recall": float(observed.mean()),
            "mean_SearchParameter": float(p.mean()),
            "median_SearchParameter": float(np.median(p)),
            "median_margin_over_min_attainable": float(np.median(
                p[attainable]-grid[minimum[attainable]])) if attainable.any() else None,
            "mean_latency_ms": float(true_latency.mean()),
            "latency_MAE_at_executed_parameter_ms": float(np.abs(ml.predict(X)-true_latency).mean())}
    return result


def plot_heatmap(model, lid_all, pmin, pmax, out, ds_name):
    """Square heatmap with recall 0.6..1.0 and an aligned colorbar."""
    import matplotlib.pyplot as plt

    lid_lo, lid_hi = np.quantile(lid_all, [0.005, 0.995])
    # ---------- 2. 网格 ----------
    N_LID, N_REC = 400, 9                        # LID 400 点, recall 0.6~1 共 9 档 (0.05 步长)
    lid_grid = np.linspace(lid_lo, lid_hi, N_LID)
    rec_grid = np.linspace(0.6, 1.0, N_REC)
    LID, REC = np.meshgrid(lid_grid, rec_grid)   # 形状 (N_REC, N_LID)
    X = np.column_stack([LID.ravel(), REC.ravel()]).astype(np.float32)

    # ---------- 3. 模型推理 ----------
    P = model.predict(X).reshape(LID.shape)
    P = np.clip(P, pmin, pmax)

    # ---------- 4. 绘图 ----------
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.set_box_aspect(1)
    pcm = ax.pcolormesh(LID, REC, P, cmap="viridis", shading="auto",
                        vmin=pmin, vmax=pmax)
    cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label("predicted SearchParameter", fontsize=11)

    ax.set_xlabel("LID", fontsize=12)
    ax.set_ylabel("recall", fontsize=12)
    ax.set_title(f"SearchParameter predicted by (LID, recall)\n"
                 f"{ds_name}  |  LightGBM  |  minimum-based labels", fontsize=12)
    ax.set_yticks(np.arange(0.6, 1.01, 0.1))
    ax.set_ylim(0.6, 1.0)

    fig.tight_layout()
    dst = out / "heatmap_SearchParameter.png"
    fig.savefig(dst)
    plt.close(fig)
    print(f"saved: {dst}")



def plot_paper_figures(model, lid_all, pmin, pmax, out):
    """Recall and LID slices, exported as PNG and vector PDF."""
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    LID_LO, LID_HI = np.quantile(lid_all, [0.005, 0.995])

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"[saved] {name}", flush=True)

    # ============================================================
    # 图 1  固定 recall 的 LID 切片 (模型曲线, x=LID)
    # ============================================================
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    lid_grid = np.linspace(LID_LO, LID_HI, 400)

    for r, c in zip([0.6, 0.7, 0.8, 0.9, 1.0], COLORS):
        Xg = np.column_stack([lid_grid, np.full_like(lid_grid, r)]).astype(np.float32)
        y = np.clip(model.predict(Xg), pmin, pmax)
        ax.plot(lid_grid, y, lw=1.4, color=c, label=f"r = {r}")

    ax.axvline(np.quantile(lid_all, 0.95), color="gray", lw=0.7, ls=":", alpha=0.9)
    ax.text(np.quantile(lid_all, 0.95), pmin + 2, " 95th pct", color="gray", fontsize=7,
            rotation=90, va="bottom", ha="right")
    ax.set_xlim(LID_LO, LID_HI)
    ax.set_ylim(pmin - 0.02 * (pmax - pmin), pmax + 0.02 * (pmax - pmin))
    ax.set_xlabel("LID")
    ax.set_ylabel("SearchParameter")
    ax.set_title("SearchParameter vs. LID at fixed recall")
    ax.legend(loc="upper left")
    save(fig, "fig_lid_slices")

    # ============================================================
    # 图 2  固定 LID 的 recall-SearchParameter 切片 (模型曲线, x=SearchParameter, 轻度平滑)
    # ============================================================

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    rec_grid = np.round(np.arange(0.1, 1.001, 0.02), 3)
    qs = [0.05, 0.10, 0.30, 0.50, 0.70, 0.90]   # 5th 分位起, 避开低 LID 稀疏区外推噪声
    sigma = 3                                     # 平滑窗口 ~ 3*2*sigma+1 个采样点

    for q, c in zip(qs, COLORS):
        v = float(np.quantile(lid_all, q))
        Xg = np.column_stack([np.full_like(rec_grid, v), rec_grid]).astype(np.float32)
        y = np.clip(model.predict(Xg), pmin, pmax)
        y_s = gaussian_filter1d(y, sigma, mode="nearest")
        ax.plot(y_s, rec_grid, lw=1.4, color=c, label=f"LID = {v:.1f} ({q * 100:g}th)")

    ax.set_ylim(0.58, 1.03)
    ax.set_xlim(pmin - 0.02 * (pmax - pmin), pmax + 0.02 * (pmax - pmin))
    ax.set_xlabel("SearchParameter")
    ax.set_ylabel("recall")
    ax.set_title("recall vs. SearchParameter at fixed LID (smoothed)")
    ax.legend(loc="lower right")
    save(fig, "fig_recall_slices")



def plot_training_figures(folder, model, grid):
    """Use the trained in-memory model and read the small LID file once."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = folder / "query_lid.csv"
    if not source.exists():
        source = folder / "learn_lid.csv"
    lid_all = read_lids(source)
    out = model_directory(folder)
    with plt.rc_context({"font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
                         "axes.unicode_minus": False}):
        plot_heatmap(model, lid_all, grid[0], grid[-1], out, folder.name)
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
                         "pdf.fonttype": 42, "savefig.dpi": 300}):
        plot_paper_figures(model, lid_all, grid[0], grid[-1], out)


def report_feature_importance(model, name):
    """Print and return normalized gain for features in training column order."""
    gain = model.feature_importance(importance_type="gain")
    total = float(gain.sum())
    importance = {feature: float(value / total) if total > 0 else 0.0
                  for feature, value in zip(("LID", "recall"), gain)}
    print(f"{name} feature importance (normalized gain): "
          f"LID={importance['LID']:.2%}, recall={importance['recall']:.2%}", flush=True)
    return importance


def train_dataset(dataset, threads):
    folder = Path(resolve_dataset(dataset))
    lids, grid, R, L, info = load_profile(folder, "learn")
    if not np.all(np.diff(grid) == 1):
        raise ValueError("Current integer-budget deployment requires a continuous grid")
    ids, X, yp, yl, stats = make_labels(lids, grid, R, L)
    shuffled = np.random.default_rng(42).permutation(len(lids))
    assignment = np.full(len(lids), 2)
    assignment[shuffled[:int(.8*len(lids))]] = 0
    assignment[shuffled[int(.8*len(lids)):int(.9*len(lids))]] = 1
    tr, va = assignment[ids] == 0, assignment[ids] == 1
    out = model_directory(folder)
    out.mkdir(parents=True, exist_ok=True)
    params = {"objective": "regression_l2", "metric": "l1", "learning_rate": .1,
              "num_leaves": 63, "max_depth": 10, "min_data_in_leaf": 100,
              "bagging_fraction": .8, "bagging_freq": 1, "lambda_l2": 1.,
              "seed": 42, "verbosity": -1, "num_threads": threads}

    def fit(y, name):
        print(f"{folder.name}: training {name}, {tr.sum()} train / {va.sum()} valid labels", flush=True)
        model = lgb.train(params, lgb.Dataset(X[tr], label=y[tr]), num_boost_round=2000,
                          valid_sets=[lgb.Dataset(X[va], label=y[va])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        model.save_model(str(out / f"model_{name}.txt"))
        print(f"{name}: {model.best_iteration} trees", flush=True)
        return model

    mp, ml = fit(yp, "SearchParameter"), fit(yl, "latency")
    importance = {"SearchParameter": report_feature_importance(mp, "SearchParameter"),
                  "latency": report_feature_importance(ml, "latency")}
    metrics = {"data": {"dataset": folder.name, "learn": info,
               "target": "minimum attained parameter plus target-specific proportional label margin; paired measured latency",
               "margin_ratio": MARGIN_RATIO, "margin_targets": list(MARGIN_TARGETS),
               "training_targets": TARGETS.tolist(),
               "split": "query ID 80/10/10, seed 42",
               "unattainable_policy": "excluded from regression; counted in evaluation",
               "label_stats": stats},
               "feature_importance_normalized_gain": importance,
               "learn_internal_test": evaluate(mp, ml, lids, grid, R, L, assignment == 2)}
    del R, L
    qlids, qgrid, QR, QL, qinfo = load_profile(folder, "query")
    if not np.array_equal(grid, qgrid):
        raise ValueError("Learn/query grids differ")
    metrics["data"]["query"] = qinfo
    mask = np.ones(len(qlids), dtype=bool)
    metrics["query_external_test"] = evaluate(mp, ml, qlids, grid, QR, QL, mask)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    for target, result in metrics["query_external_test"].items():
        print(f"{folder.name} r={target}: mean parameter {result['mean_SearchParameter']:.2f}; "
              f"hit rate {result['hit_rate_all']:.3f}", flush=True)

    plot_training_figures(folder, mp, grid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset name or directory")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if not np.isfinite(MARGIN_RATIO) or MARGIN_RATIO < 0:
        parser.error("MARGIN_RATIO must be finite and nonnegative")
    train_dataset(args.dataset, args.threads)


if __name__ == "__main__":
    main()
