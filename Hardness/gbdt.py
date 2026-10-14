"""
LID (Local Intrinsic Dimension) Prediction Model
================================================

基于 LightGBM 的局部内在维度(LID)预测模型。

数据来源：
- 质心距离：最近质心的距离序列（已升序排列）
- 向量距离：初步搜索到的最近邻向量距离（已升序排列）

特征工程：
- Top 10 重要特征，包括 MLE 估计、距离比率、变异系数等

模型配置：
- 默认 MSE 回归；DEEP 使用 Huber 稳健回归
- LightGBM GBDT 框架
- 早停机制防止过拟合

输出文件（统一存放在 DATA_DIR 指定的目录下）：
- features.csv: 特征矩阵
- lgbm_model.txt: 训练好的模型
- lgbm_model.txt.metadata.json: 特征版本、训练参数与模型校验值
- predictions.csv: 测试集预测结果
- train_predictions.csv: 训练集预测结果（拟合效果）
- lgbm_plot_train.png: 训练集拟合散点图（x/y 轴 0~LID_PLOT_MAX，正方形）
- lgbm_plot_test.png: 测试集拟合散点图（x/y 轴 0~LID_PLOT_MAX，正方形）
- lgbm_plot_importance.png: 特征重要性柱状图
- lgbm_plot_rmse.png: RMSE 训练曲线
- lgbm_plot_lid_dist.png: LID 值分布直方图（左训练 / 右测试）
- rmse_history.csv: 每轮迭代的训练集与验证集 RMSE 值
- metrics.txt: 性能指标汇总
- feature_importance.csv: 特征重要性（含占总增益的百分比）
- centroid_feature_importance.csv: 质心相关 5 特征重要性（组内归一化，和为 100%）

作者: AI Assistant
日期: 2026-08-06
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional


# ============================================================================
# 配置参数
# ============================================================================

import os
import argparse
import hashlib
import json

# ===== 数据目录（输入，通过 --dataset 必填参数传入）=====
DATA_DIR = 'spacev'  # 模块级默认值，运行时会被 --dataset 参数覆盖

# 数据文件路径（在 _setup_paths() 中赋值）
TRAIN_CENTROID_FILE = ''
TRAIN_VECTOR_FILE = ''
TRAIN_LID_FILE = ''
TEST_CENTROID_FILE = ''
TEST_VECTOR_FILE = ''
TEST_LID_FILE = ''
OUTPUT_FEATURES = ''
OUTPUT_MODEL = ''
OUTPUT_PREDICTIONS = ''
OUTPUT_TRAIN_PREDICTIONS = ''
OUTPUT_PLOT_TRAIN = ''
OUTPUT_PLOT_TEST = ''
OUTPUT_PLOT_IMPORTANCE = ''
OUTPUT_PLOT_RMSE = ''
OUTPUT_PLOT_LID_DIST = ''
OUTPUT_METRICS = ''
OUTPUT_IMPORTANCE = ''
OUTPUT_RMSE_HISTORY = ''
OUTPUT_CENTROID_IMPORTANCE = ''


def _setup_paths(data_dir: str) -> None:
    """根据数据目录（重新）初始化所有文件路径。"""
    global DATA_DIR
    global TRAIN_CENTROID_FILE, TRAIN_VECTOR_FILE, TRAIN_LID_FILE
    global TEST_CENTROID_FILE, TEST_VECTOR_FILE, TEST_LID_FILE
    global OUTPUT_FEATURES, OUTPUT_MODEL, OUTPUT_PREDICTIONS, OUTPUT_TRAIN_PREDICTIONS
    global OUTPUT_PLOT_TRAIN, OUTPUT_PLOT_TEST, OUTPUT_PLOT_IMPORTANCE, OUTPUT_PLOT_RMSE
    global OUTPUT_PLOT_LID_DIST
    global OUTPUT_METRICS, OUTPUT_IMPORTANCE, OUTPUT_RMSE_HISTORY
    global OUTPUT_CENTROID_IMPORTANCE

    DATA_DIR = data_dir

    # 数据文件路径（拼接在 DATA_DIR 下）
    TRAIN_CENTROID_FILE = os.path.join(data_dir, 'learn_head.csv')
    TRAIN_VECTOR_FILE = os.path.join(data_dir, 'learn_list.csv')
    TRAIN_LID_FILE = os.path.join(data_dir, 'learn_lid.csv')

    TEST_CENTROID_FILE = os.path.join(data_dir, 'query_head.csv')
    TEST_VECTOR_FILE = os.path.join(data_dir, 'query_list.csv')
    TEST_LID_FILE = os.path.join(data_dir, 'query_lid.csv')

    # 输出文件路径（全部拼接在 DATA_DIR 下）
    OUTPUT_FEATURES = os.path.join(data_dir, 'features.csv')
    OUTPUT_MODEL = os.path.join(data_dir, 'lgbm_model.txt')
    OUTPUT_PREDICTIONS = os.path.join(data_dir, 'predictions.csv')
    OUTPUT_TRAIN_PREDICTIONS = os.path.join(data_dir, 'train_predictions.csv')
    OUTPUT_PLOT_TRAIN = os.path.join(data_dir, 'lgbm_plot_train.png')
    OUTPUT_PLOT_TEST = os.path.join(data_dir, 'lgbm_plot_test.png')
    OUTPUT_PLOT_IMPORTANCE = os.path.join(data_dir, 'lgbm_plot_importance.png')
    OUTPUT_PLOT_RMSE = os.path.join(data_dir, 'lgbm_plot_rmse.png')
    OUTPUT_PLOT_LID_DIST = os.path.join(data_dir, 'lgbm_plot_lid_dist.png')
    OUTPUT_METRICS = os.path.join(data_dir, 'metrics.txt')
    OUTPUT_IMPORTANCE = os.path.join(data_dir, 'feature_importance.csv')
    OUTPUT_RMSE_HISTORY = os.path.join(data_dir, 'rmse_history.csv')
    OUTPUT_CENTROID_IMPORTANCE = os.path.join(data_dir, 'centroid_feature_importance.csv')


# 用默认值初始化（保证模块被 import 时也有合理值）
_setup_paths(DATA_DIR)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='基于 LightGBM 的局部内在维度(LID)预测模型'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='数据目录路径（必填，如 spacev / sift1m / gist1m）'
    )
    parser.add_argument(
        '--lid-max',
        type=float,
        default=None,
        help='LID 统计上限：拟合散点图与测试集指标（MSE/MAE/RMSE/Pearson/R²）均只统计'
             '真实值与预测值均在 [0, 该值] 范围内的点，坐标轴也限定在该范围'
             '（默认: 不过滤，统计全部样本）'
    )
    return parser.parse_args()

MODEL_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.01,
    'min_data_in_leaf': 200,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'max_depth': 8,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'extra_trees': False,
    'verbose': -1,
    'seed': 42,
    # 固定训练线程，复现本轮参数对比；推理基准单独使用 1 线程。
    'num_threads': 6
}

# 训练参数
# 默认配置；各数据集可覆盖学习率、轮数上限和最小改善量。
NUM_BOOST_ROUND = 10000
EARLY_STOPPING_ROUNDS = 200
LOG_EVALUATION_PERIOD = 200
DATASET_TRAINING_OVERRIDES = {
    # 固定留出验证筛选、完整训练和单线程推理计时后的折中配置。
    # 各数据集的候选与原始测量见对应 tuning / tuning_balance 目录。
    'sift100m': (0.3, 10000, 0.005),
    # DEEP 的 min_delta 用于 Pearson，而非 RMSE。
    'deep100m': (0.05, 6000, 0.0001),
    'gist1m': (0.05, 10000, 0.001),
    'text2img': (0.15, 10000, 0.001),
    'msmarco': (0.2, 10000, 0.001),
}
DATASET_MODEL_OVERRIDES = {
    'deep100m': {
        'objective': 'huber', 'alpha': 0.9,
        'num_leaves': 63, 'max_depth': 8, 'min_data_in_leaf': 50,
        'feature_fraction': 1.0, 'bagging_fraction': 1.0, 'metric': 'None',
    },
}
DATASET_PEARSON_EARLY_STOPPING = {'deep100m': 300}


def get_training_config(data_dir: str):
    """按数据目录名选择学习率、轮数上限和早停最小改善量。"""
    dataset = os.path.basename(os.path.normpath(data_dir)).lower()
    params = MODEL_PARAMS.copy()
    learning_rate, rounds, min_delta = DATASET_TRAINING_OVERRIDES.get(
        dataset, (params['learning_rate'], NUM_BOOST_ROUND, 0.0)
    )
    params['learning_rate'] = learning_rate
    params.update(DATASET_MODEL_OVERRIDES.get(dataset, {}))
    return params, rounds, min_delta


def make_pearson_evaluation(lid_max: Optional[float]):
    """在固定真实 LID 范围内监控 Pearson，不按预测值删点。"""
    def evaluate(predictions, dataset):
        labels = dataset.get_label()
        mask = np.isfinite(labels)
        if lid_max is not None:
            mask &= (labels >= 0) & (labels <= lid_max)
        selected_labels, selected_predictions = labels[mask], predictions[mask]
        if len(selected_labels) < 2 or np.std(selected_labels) == 0:
            raise ValueError('Pearson early stopping needs at least two nonconstant labels')
        if np.std(selected_predictions) == 0:
            correlation = 0.0
        else:
            correlation = float(pearsonr(selected_labels, selected_predictions)[0])
        rmse = float(np.sqrt(np.mean((predictions - labels) ** 2)))
        return [('pearson_fixed', correlation, True), ('rmse', rmse, False)]
    return evaluate


def save_model_metadata(model_path, feature_names, params, lid_max, stopping_metric):
    """记录特征语义与模型校验值，区分修复前后的同维度模型。"""
    model_path = os.fspath(model_path)
    with open(model_path, 'rb') as model_file:
        model_hash = hashlib.sha256(model_file.read()).hexdigest()
    metadata = {
        'feature_schema': 'centroid_distance_columns_v2',
        'feature_names': feature_names,
        'model_sha256': model_hash,
        'training_params': params,
        'lid_max': lid_max,
        'early_stopping_metric': stopping_metric,
    }
    with open(model_path + '.metadata.json', 'w', encoding='utf-8') as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

# 评估阈值
HIGH_LID_THRESHOLD = 20
EXTREME_LID_THRESHOLD = 30

# LID 统计上限（同时作用于拟合散点图与测试集评估指标）：
# 只统计真实值与预测值均在 [0, LID_PLOT_MAX] 内的样本，
# 散点图中的 Pearson、坐标轴范围以及 MSE/MAE/RMSE/Pearson/R² 等指标
# 均基于该范围内的点计算。
# 可通过命令行参数 --lid-max 覆盖；None 表示不过滤（统计全部样本）。
LID_PLOT_MAX = None

# 是否剔除 LID 值等于某个固定值的异常样本
# 设为 None 表示不剔除；设为具体数值（如 100）则剔除 y == 该值的训练样本
EXCLUDE_LID_VALUE = None

# 数据维度（用于 LID MLE 的上限 cap 和退化 fallback）
# 设为 0 表示不指定，此时用 k（邻居数）作为 fallback，与 C++ 中 --dim 未指定时行为一致
DATA_DIM = 0

# 质心相关特征（仅由质心距离矩阵 D 计算得到，不含向量距离信息）
# 用于单独统计这组特征的重要性：组内归一化后百分比之和为 100%
CENTROID_FEATURES = ['centroid_mle', 'tail_front_mean_ratio', 'tail_cv', 'cv', 'std_log']


# ============================================================================
# 数据加载模块
# ============================================================================

def load_vector_data(filename: str) -> List[np.ndarray]:
    """
    加载向量距离数据（变长行格式）。
    
    文件格式：每行为 "query_id,d1,d2,d3,..."，距离已升序排列。
    
    Args:
        filename: 数据文件路径
        
    Returns:
        按 query_id 排序的距离列表，每个元素是一维 numpy 数组
    """
    rows = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            query_id = int(parts[0])
            distances = np.array([float(x) for x in parts[1:]])
            rows.append((query_id, distances))
    
    # 按 query_id 排序确保一致性
    rows.sort(key=lambda x: x[0])
    return [distances for _, distances in rows]


def load_dataset(centroid_file: str, 
                 lid_file: str, 
                 vector_file: str,
                 dataset_name: str) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    加载完整数据集（质心距离 + 向量距离 + LID 标签）。
    
    质心文件支持两种格式："query_id,d1,d2,..." 或
    "query_id,d1,list_size1,d2,list_size2,..."。
    旧格式的列表大小不参与距离特征；距离必须非负、有限且升序。
    
    注意：如果第一个质心距离为 0，则从第二个质心开始使用。
    
    Args:
        centroid_file: 质心距离文件路径
        lid_file: LID 标签文件路径
        vector_file: 向量距离文件路径
        dataset_name: 数据集名称（用于日志输出）
        
    Returns:
        (D, V, y): 质心距离矩阵、向量距离列表、LID 标签数组
    """
    # 加载质心数据
    df = pd.read_csv(centroid_file, header=None)
    raw = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    if raw.shape[1] == 0 or not np.all(np.isfinite(raw)) or np.any(raw < 0):
        raise ValueError(f"Invalid centroid distances in {centroid_file}")
    distance_only = np.all(np.diff(raw, axis=1) >= 0)
    paired = (raw.shape[1] % 2 == 0
              and np.all(raw[:, 1::2] == np.floor(raw[:, 1::2]))
              and np.all(np.diff(raw[:, ::2], axis=1) >= 0))
    if distance_only:
        D = raw
    elif paired:
        D = raw[:, ::2]
        print(f"[{dataset_name}] Legacy head format: {D.shape[1]} distances; "
              "list-size columns excluded from distance features")
    else:
        raise ValueError(f"Unsorted or malformed centroid file: {centroid_file}")
    
    # 过滤掉距离为 0 的质心（从前往后找第一个非零距离）
    # 如果第一个质心距离为 0，则从第二个质心开始使用
    n_samples = D.shape[0]
    D_filtered = []
    for i in range(n_samples):
        # 找到第一个非零距离的索引
        non_zero_idx = np.where(D[i] > 0)[0]
        if len(non_zero_idx) > 0:
            start_idx = non_zero_idx[0]
            D_filtered.append(D[i, start_idx:])
        else:
            # 所有距离都为 0（极端情况），保留原始数据
            D_filtered.append(D[i])
    
    # 统计过滤情况
    n_filtered = sum(1 for i in range(n_samples) if D[i, 0] == 0)
    if n_filtered > 0:
        print(f"[{dataset_name}] {n_samples} samples, {n_filtered} samples have zero first centroid distance")
        n_centroids_after = [len(d) for d in D_filtered]
        min_centroids = min(n_centroids_after)
        max_centroids = max(n_centroids_after)
        print(f"  Centroids after filtering: min={min_centroids}, max={max_centroids}")
    
    # 为了统一处理，使用最小的质心数
    min_centroids = min(len(d) for d in D_filtered)
    D_uniform = np.zeros((n_samples, min_centroids))
    for i in range(n_samples):
        D_uniform[i] = D_filtered[i][:min_centroids]
    
    D = D_uniform
    
    # 加载 LID 标签
    y = pd.read_csv(lid_file, header=None).iloc[:, 0].values
    
    # 加载向量距离
    V = load_vector_data(vector_file)
    
    print(f"[{dataset_name}] {len(y)} samples, {D.shape[1]} centroids (after filtering), {len(V)} vectors")
    return D, V, y


# ============================================================================
# 特征工程模块
# ============================================================================

def compute_mle(distances: np.ndarray, k: int, dim: int = 0) -> float:
    """
    计算 LID 的 MLE 估计量，与 C++ compute_lid_from_dists() 完全对齐。

    公式: LID = -actual_k / Σ log(d_i / d_k)
    其中 d_i 遍历 k 个邻居距离，d_k 是第 k 近（最远）距离。
    actual_k 是非零距离的个数。

    边界处理（与 C++ 一致）：
    - k == 0 或最远距离 == 0 → 返回 0.0
    - 跳过 d_i == 0 的项，actual_k 只计非零项
    - sum == 0（所有距离相等）→ 返回 dim（或 k 作为 fallback）
    - 结果上限为 dim（或 k 作为 fallback）
    - dim == 0 时用 k 作为 fallback，等价于 C++ 中 --dim 未指定

    Args:
        distances: 已排序的距离序列（升序）
        k: 使用的邻居数量
        dim: 数据维度上限，0 表示用 k 作为 fallback

    Returns:
        LID 估计值
    """
    limit = min(len(distances), k)
    if limit == 0:
        return 0.0

    maxdistk = distances[limit - 1]
    if maxdistk == 0.0:
        return 0.0

    # dim fallback：未指定时用 k（与 C++ --dim 未指定时 dim_fallback = K 一致）
    dim_fallback = dim if dim > 0 else limit

    total = 0.0
    actual_k = 0
    for i in range(limit):
        if distances[i] == 0.0:
            continue
        total += np.log(distances[i] / maxdistk)
        actual_k += 1

    if actual_k == 0:
        return 0.0

    if total == 0.0:
        return float(dim_fallback)

    return float(min(dim_fallback, -actual_k / total))


def compute_centroid_mle_batch(D: np.ndarray) -> np.ndarray:
    """
    批量计算质心距离的 MLE。

    使用全部质心计算，这是 Top 1 重要特征。

    Args:
        D: 质心距离矩阵 (n_samples, n_centroids)，已按列排序

    Returns:
        MLE 估计值数组
    """
    n_samples = D.shape[0]
    k = D.shape[1]  # 使用全部质心
    mle = np.zeros(n_samples)

    for i in range(n_samples):
        mle[i] = compute_mle(D[i], k, DATA_DIM)

    return mle


def compute_vector_features_batch(V: List[np.ndarray], 
                                   D_centroid: np.ndarray) -> Dict[str, np.ndarray]:
    """
    批量计算向量距离特征。
    
    提取的特征（Top 10 中的向量特征）：
    - vec_d1: 第1近邻距离
    - vec_cv: 变异系数（标准差/均值）
    - vec_vs_centroid_d1: 向量d1与质心d1的比值
    - vec_d1_d10_ratio: 第1近邻与第10近邻的距离比
    - vec_mle: 向量距离的 MLE（用于后续计算 vec_mle_ratio）
    
    Args:
        V: 向量距离列表，每个元素是已排序的距离数组
        D_centroid: 质心距离矩阵，用于对比计算
        
    Returns:
        特征字典，键为特征名，值为特征数组
    """
    n = len(V)
    feats = {}
    
    # 提取第1和第10近邻距离
    vec_d1 = np.array([v[0] if len(v) > 0 else np.nan for v in V])
    vec_d10 = np.array([v[9] if len(v) >= 10 else np.nan for v in V])
    
    # 特征 1: vec_d1
    feats['vec_d1'] = vec_d1
    
    # 特征 2: vec_d1_d10_ratio
    feats['vec_d1_d10_ratio'] = vec_d1 / np.maximum(vec_d10, 1e-6)
    
    # 特征 3: vec_cv（变异系数）
    vec_cv = np.zeros(n)
    for i in range(n):
        v = V[i]
        if len(v) >= 2:
            vec_cv[i] = np.std(v) / max(np.mean(v), 1e-6)
    feats['vec_cv'] = vec_cv
    
    # 特征 4: vec_vs_centroid_d1
    centroid_d1 = D_centroid[:, 0]
    feats['vec_vs_centroid_d1'] = vec_d1 / np.maximum(centroid_d1, 1e-6)
    
    # 特征 5: vec_mle（用于后续计算 vec_mle_ratio）
    vec_mle = np.ones(n)
    for i in range(n):
        v = V[i]
        if len(v) >= 5:
            vec_mle[i] = compute_mle(v, len(v), DATA_DIM)
    feats['vec_mle'] = vec_mle
    
    return feats


def build_feature_matrix(D: np.ndarray, 
                          V: List[np.ndarray],
                          y: Optional[np.ndarray] = None,
                          save_csv: bool = True,
                          verbose: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    构建特征矩阵。
    
    Top 10 特征：
    1. centroid_mle - 质心距离的 MLE（最重要）
    2. vec_d1 - 向量第1近邻距离
    3. tail_front_mean_ratio - 尾段与前段均值比
    4. vec_cv - 向量距离变异系数
    5. vec_vs_centroid_d1 - 向量d1与质心d1比值
    6. cv - 质心距离变异系数
    7. vec_d1_d10_ratio - 向量d1与d10比值
    8. tail_cv - 尾段变异系数
    9. std_log - 对数距离标准差
    10. vec_mle_ratio - 向量MLE与质心MLE比值
    
    Args:
        D: 质心距离矩阵
        V: 向量距离列表
        y: LID 标签（可选，用于保存到 CSV）
        save_csv: 是否保存特征到 CSV 文件
        
    Returns:
        (X, feat_names): 特征矩阵和特征名列表
    """
    log = print if verbose else lambda *args, **kwargs: None
    n_samples = D.shape[0]
    n_centroids = D.shape[1]
    feats = {}
    
    # ===== 1. centroid_mle（Top 1）=====
    log("  [1/10] Computing centroid_mle...")
    feats['centroid_mle'] = compute_centroid_mle_batch(D)
    
    # ===== 向量特征（Top 2, 4, 5, 7, 10）=====
    if V is not None:
        log(f"  Computing vector features ({len(V)} samples)...")
        vec_feats = compute_vector_features_batch(V, D)
        feats.update(vec_feats)
        
        # ===== 10. vec_mle_ratio =====
        vec_mle = feats.get('vec_mle', np.ones(n_samples))
        feats['vec_mle_ratio'] = vec_mle / np.maximum(feats['centroid_mle'], 1e-6)
    
    # ===== 3. tail_front_mean_ratio =====
    log("  [3/10] Computing tail_front_mean_ratio...")
    front_n = min(10, n_centroids // 4)
    tail_n = min(10, n_centroids // 4)
    front_mean = np.mean(D[:, :front_n], axis=1)
    tail_mean = np.mean(D[:, -tail_n:], axis=1)
    feats['tail_front_mean_ratio'] = tail_mean / np.maximum(front_mean, 1e-6)
    
    # ===== 8. tail_cv =====
    log("  [8/10] Computing tail_cv...")
    mid_point = n_centroids // 2
    tail_segment = D[:, mid_point:]
    tail_mean = np.mean(tail_segment, axis=1)
    tail_std = np.std(tail_segment, axis=1)
    feats['tail_cv'] = tail_std / np.maximum(tail_mean, 1e-6)
    
    # ===== 6. cv =====
    log("  [6/10] Computing cv...")
    mean_d = np.nanmean(D, axis=1)
    std_d = np.nanstd(D, axis=1)
    feats['cv'] = std_d / np.maximum(mean_d, 1e-6)
    
    # ===== 9. std_log =====
    log("  [9/10] Computing std_log...")
    log_D = np.log(np.clip(D, 1e-6, None))
    feats['std_log'] = np.nanstd(log_D, axis=1)
    
    # 构建特征矩阵
    feat_names = list(feats.keys())
    X = np.column_stack([feats[name] for name in feat_names])
    
    # 处理缺失值和无穷值
    X = np.where(np.isfinite(X), X, 0.0)
    
    log(f"  Feature matrix shape: {X.shape}")
    
    # 保存到 CSV
    if save_csv:
        feat_df = pd.DataFrame(X, columns=feat_names)
        if y is not None:
            feat_df['LID'] = y
        feat_df.to_csv(OUTPUT_FEATURES, index=True, index_label='sample_id')
        log(f"  ✓ Features saved to {OUTPUT_FEATURES}")
    
    return X, feat_names


# ============================================================================
# 数据清洗模块
# ============================================================================

def filter_by_lid_value(D: np.ndarray,
                        V: List[np.ndarray],
                        y: np.ndarray,
                        exclude_value: float) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    剔除 LID 标签等于指定固定值的异常样本。

    某些数据集中存在一批"固定值"异常点（例如 LID 恒为 100），
    这些样本通常是无意义或计算失败的产物，会干扰模型训练。

    Args:
        D: 质心距离矩阵 (n_samples, n_centroids)
        V: 向量距离列表
        y: LID 标签数组
        exclude_value: 要剔除的 LID 固定值（精确匹配）

    Returns:
        过滤后的 (D, V, y)
    """
    mask = y != exclude_value
    n_removed = (~mask).sum()
    print(f"  [Filter] 剔除 LID == {exclude_value} 的样本: {n_removed} / {len(y)} 条 "
          f"(剩余 {mask.sum()} 条)")

    return D[mask], [V[i] for i in range(len(V)) if mask[i]], y[mask]


# ============================================================================
# 模型训练与评估模块
# ============================================================================

def evaluate_predictions(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         threshold_high: float = HIGH_LID_THRESHOLD,
                         threshold_extreme: float = EXTREME_LID_THRESHOLD,
                         lid_max: Optional[float] = None) -> Dict:
    """
    计算评估指标。
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        threshold_high: 高 LID 阈值
        threshold_extreme: 极端 LID 阈值
        lid_max: LID 统计上限。给定时只统计真实值与预测值均在 [0, lid_max]
                 内的样本（与 --lid-max 拟合散点图的过滤口径一致）；
                 None 表示不过滤，统计全部样本。
        
    Returns:
        指标字典
    """
    # 按范围过滤样本（真实值与预测值都需落在 [0, lid_max] 内）
    n_total = len(y_true)
    if lid_max is not None:
        range_mask = ((y_true >= 0) & (y_true <= lid_max)
                      & (y_pred >= 0) & (y_pred <= lid_max))
        n_dropped = int((~range_mask).sum())
        y_true = y_true[range_mask]
        y_pred = y_pred[range_mask]
        print(f"[Metrics] 统计范围 LID ∈ [0, {lid_max:g}]: "
              f"{len(y_true)} / {n_total} points (dropped {n_dropped})")
    n_used = len(y_true)
    
    # 整体指标
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    corr = pearsonr(y_true, y_pred)[0] if n_used >= 2 else np.nan
    # R²（决定系数）: 1 - SS_res / SS_tot，标准回归评估定义
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'pearson': corr,
        'r2': r2,
        'lid_max': lid_max,
        'n_used': n_used,
        'n_total': n_total,
    }
    
    # 高 LID 指标
    high_mask = y_true >= threshold_high
    if high_mask.sum() > 0:
        y_true_high = y_true[high_mask]
        y_pred_high = y_pred[high_mask]
        metrics['high_n'] = high_mask.sum()
        metrics['high_mae'] = np.mean(np.abs(y_true_high - y_pred_high))
        metrics['high_rmse'] = np.sqrt(np.mean((y_true_high - y_pred_high) ** 2))
        metrics['high_corr'] = pearsonr(y_true_high, y_pred_high)[0] if len(y_true_high) > 1 else np.nan
        metrics['high_bias'] = np.mean(y_pred_high - y_true_high)
    
    # 极端 LID 指标
    extreme_mask = y_true >= threshold_extreme
    if extreme_mask.sum() > 0:
        y_true_ext = y_true[extreme_mask]
        y_pred_ext = y_pred[extreme_mask]
        metrics['extreme_n'] = extreme_mask.sum()
        metrics['extreme_mae'] = np.mean(np.abs(y_true_ext - y_pred_ext))
        metrics['extreme_bias'] = np.mean(y_pred_ext - y_true_ext)
    
    return metrics


def print_metrics(metrics: Dict) -> None:
    """打印评估指标。"""
    print(f"\n{'='*50}")
    print("Evaluation Results")
    print(f"{'='*50}")
    if metrics.get('lid_max') is not None:
        print(f"统计范围: LID ∈ [0, {metrics['lid_max']:g}] "
              f"({metrics['n_used']} / {metrics['n_total']} points)")
    print(f"RMSE:    {metrics['rmse']:.4f}")
    print(f"MSE:     {metrics['mse']:.4f}")
    print(f"MAE:     {metrics['mae']:.4f}")
    print(f"Pearson: {metrics['pearson']:.4f}")
    print(f"R²:      {metrics['r2']:.4f}")
    
    if 'high_n' in metrics:
        print(f"\nHigh LID (≥{HIGH_LID_THRESHOLD}):")
        print(f"  n:     {metrics['high_n']}")
        print(f"  MAE:   {metrics['high_mae']:.3f}")
        print(f"  RMSE:  {metrics['high_rmse']:.3f}")
        print(f"  Corr:  {metrics['high_corr']:.3f}")
        print(f"  Bias:  {metrics['high_bias']:+.3f}")
    
    if 'extreme_n' in metrics:
        print(f"\nExtreme LID (≥{EXTREME_LID_THRESHOLD}):")
        print(f"  n:     {metrics['extreme_n']}")
        print(f"  MAE:   {metrics['extreme_mae']:.3f}")
        print(f"  Bias:  {metrics['extreme_bias']:+.3f}")


def save_metrics_file(metrics: Dict) -> None:
    """保存指标到文件。"""
    with open(OUTPUT_METRICS, 'w', encoding='utf-8') as f:
        f.write("LID Prediction Model - Test Results\n")
        f.write("="*50 + "\n\n")
        if metrics.get('lid_max') is not None:
            f.write(f"统计范围: LID ∈ [0, {metrics['lid_max']:g}] "
                    f"({metrics['n_used']} / {metrics['n_total']} points)\n\n")
        f.write(f"Test RMSE:    {metrics['rmse']:.4f}\n")
        f.write(f"Test MSE:     {metrics['mse']:.4f}\n")
        f.write(f"Test MAE:     {metrics['mae']:.4f}\n")
        f.write(f"Test Pearson: {metrics['pearson']:.4f}\n")
        f.write(f"Test R²:      {metrics['r2']:.4f}\n\n")
        
        if 'high_n' in metrics:
            f.write(f"High LID (≥{HIGH_LID_THRESHOLD}):\n")
            f.write(f"  n:     {metrics['high_n']}\n")
            f.write(f"  MAE:   {metrics['high_mae']:.3f}\n")
            f.write(f"  RMSE:  {metrics['high_rmse']:.3f}\n")
            f.write(f"  Corr:  {metrics['high_corr']:.3f}\n")
            f.write(f"  Bias:  {metrics['high_bias']:+.3f}\n\n")
        
        if 'extreme_n' in metrics:
            f.write(f"Extreme LID (≥{EXTREME_LID_THRESHOLD}):\n")
            f.write(f"  n:     {metrics['extreme_n']}\n")
            f.write(f"  MAE:   {metrics['extreme_mae']:.3f}\n")
            f.write(f"  Bias:  {metrics['extreme_bias']:+.3f}\n")
    
    print(f"✓ Metrics saved to {OUTPUT_METRICS}")


def visualize_results(y_true_test: np.ndarray, 
                      y_pred_test: np.ndarray,
                      y_true_train: np.ndarray,
                      y_pred_train: np.ndarray,
                      feat_importance: pd.DataFrame,
                      rmse_history: Optional[Dict] = None,
                      lid_max: Optional[float] = None) -> None:
    """
    生成评估可视化图表（4 个独立文件）。
    
    1. 训练集拟合散点图（x/y 轴范围 0~lid_max，正方形图）
    2. 测试集拟合散点图（x/y 轴范围 0~lid_max，正方形图）
    3. 特征重要性柱状图
    4. RMSE 训练曲线
    
    Args:
        y_true_test: 测试集真实值
        y_pred_test: 测试集预测值
        y_true_train: 训练集真实值
        y_pred_train: 训练集预测值
        feat_importance: 特征重要性 DataFrame
        rmse_history: LightGBM 训练过程记录的指标字典
        lid_max: 拟合散点图的 LID 统计上限。只统计真实值与预测值均在
                 [0, lid_max] 内的样本点，Pearson 与坐标轴同样限定在该范围。
                 None 时使用全局配置 LID_PLOT_MAX；若仍为 None 则不过滤
                 （统计全部点，坐标轴默认 0~100）。
    """
    from scipy.stats import pearsonr

    if lid_max is None:
        lid_max = LID_PLOT_MAX
    # 未指定范围（None）时不做过滤，坐标轴使用默认范围，与旧版行为一致
    use_filter = lid_max is not None
    AXIS_MAX = lid_max if use_filter else 100  # 拟合图 x/y 轴最大范围
    range_label = f'LID ∈ [0, {AXIS_MAX:g}], ' if use_filter else ''

    def _in_range(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """真实值与预测值都落在 [0, AXIS_MAX] 内的样本掩码。"""
        if not use_filter:
            return np.ones(len(y_true), dtype=bool)
        return ((y_true >= 0) & (y_true <= AXIS_MAX)
                & (y_pred >= 0) & (y_pred <= AXIS_MAX))

    # ---- 1) 训练集拟合散点图 ----
    mask_train = _in_range(y_true_train, y_pred_train)
    yt_train, yp_train = y_true_train[mask_train], y_pred_train[mask_train]
    n_train_dropped = (~mask_train).sum()
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    ax1.scatter(yt_train, yp_train, alpha=0.4, s=8, edgecolors='none', c='#4C78A8')
    ax1.plot([0, AXIS_MAX], [0, AXIS_MAX], 'r--', lw=1.5, label='Perfect prediction')
    corr_train, _ = pearsonr(yt_train, yp_train)
    ax1.set_xlim(0, AXIS_MAX)
    ax1.set_ylim(0, AXIS_MAX)
    ax1.set_xlabel('Actual LID', fontsize=11)
    ax1.set_ylabel('Predicted LID', fontsize=11)
    ax1.set_title(f'Train: Prediction vs Actual ({range_label}'
                  f'n={mask_train.sum()}/{len(y_true_train)})\n'
                  f'Pearson={corr_train:.4f}', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_TRAIN, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    if use_filter:
        print(f"✓ Train scatter plot saved to {OUTPUT_PLOT_TRAIN} "
              f"(LID ∈ [0, {AXIS_MAX:g}], dropped {n_train_dropped} points)")
    else:
        print(f"✓ Train scatter plot saved to {OUTPUT_PLOT_TRAIN}")

    # ---- 2) 测试集拟合散点图 ----
    mask_test = _in_range(y_true_test, y_pred_test)
    yt_test, yp_test = y_true_test[mask_test], y_pred_test[mask_test]
    n_test_dropped = (~mask_test).sum()
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.scatter(yt_test, yp_test, alpha=0.4, s=8, edgecolors='none', c='#4C78A8')
    ax2.plot([0, AXIS_MAX], [0, AXIS_MAX], 'r--', lw=1.5, label='Perfect prediction')
    corr_test, _ = pearsonr(yt_test, yp_test)
    ax2.set_xlim(0, AXIS_MAX)
    ax2.set_ylim(0, AXIS_MAX)
    ax2.set_xlabel('Actual LID', fontsize=11)
    ax2.set_ylabel('Predicted LID', fontsize=11)
    ax2.set_title(f'Test: Prediction vs Actual ({range_label}'
                  f'n={mask_test.sum()}/{len(y_true_test)})\n'
                  f'Pearson={corr_test:.4f}', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_TEST, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    if use_filter:
        print(f"✓ Test scatter plot saved to {OUTPUT_PLOT_TEST} "
              f"(LID ∈ [0, {AXIS_MAX:g}], dropped {n_test_dropped} points)")
    else:
        print(f"✓ Test scatter plot saved to {OUTPUT_PLOT_TEST}")

    # ---- 3) 特征重要性柱状图 ----
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    top_features = feat_importance.head(10)
    ax3.barh(range(len(top_features)), top_features['importance'], align='center', color='#4C78A8')
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features['feature'])
    ax3.invert_yaxis()
    ax3.set_xlabel('Importance (Gain)', fontsize=11)
    ax3.set_title('Top 10 Features', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_IMPORTANCE, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"✓ Feature importance plot saved to {OUTPUT_PLOT_IMPORTANCE}")

    # ---- 4) RMSE 训练曲线 ----
    if rmse_history is not None and 'train' in rmse_history and 'valid' in rmse_history:
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        iters = np.arange(1, len(rmse_history['train']['rmse']) + 1)
        ax4.plot(iters, rmse_history['train']['rmse'], label='Train RMSE', lw=1.5)
        ax4.plot(iters, rmse_history['valid']['rmse'], label='Valid RMSE', lw=1.5)
        ax4.set_xlabel('Iteration', fontsize=11)
        ax4.set_ylabel('RMSE', fontsize=11)
        ax4.set_title('RMSE Training History', fontsize=12)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT_RMSE, dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print(f"✓ RMSE history plot saved to {OUTPUT_PLOT_RMSE}")
    else:
        print("[INFO] No RMSE history available, skipping RMSE plot.")


def plot_lid_distribution(y_train: np.ndarray, y_test: np.ndarray) -> None:
    """
    绘制 LID 值分布直方图（左训练 / 右测试），保存为单独的 PNG 文件。

    两张子图共享相同的 bin 边界和坐标范围，便于直观对比训练集与测试集
    的 LID 分布是否一致。每张子图上标注了样本数、均值、中位数等统计信息。

    Args:
        y_train: 训练集 LID 标签数组
        y_test:  测试集 LID 标签数组
    """
    # 统一 bin 边界：取两份数据合并后的范围，等宽分箱
    all_vals = np.concatenate([y_train, y_test])
    bin_min = np.floor(all_vals.min())
    bin_max = np.ceil(all_vals.max())
    n_bins = min(60, max(20, int(bin_max - bin_min)))
    bins = np.linspace(bin_min, bin_max, n_bins + 1)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # ---- 左图：训练集（概率密度）----
    ax_l.hist(y_train, bins=bins, density=True, color='#4C78A8', edgecolor='white', alpha=0.85)
    train_mean = np.mean(y_train)
    train_median = np.median(y_train)
    train_std = np.std(y_train)
    ax_l.axvline(train_mean, color='#E45756', ls='--', lw=1.5, label=f'Mean={train_mean:.2f}')
    ax_l.axvline(train_median, color='#F58518', ls=':', lw=1.5, label=f'Median={train_median:.2f}')
    ax_l.set_xlabel('LID', fontsize=11)
    ax_l.set_ylabel('Density', fontsize=11)
    ax_l.set_title(f'Train LID Distribution (n={len(y_train)}, σ={train_std:.2f})', fontsize=12)
    ax_l.legend(loc='upper right', fontsize=9)
    ax_l.grid(True, alpha=0.3, axis='y')

    # ---- 右图：测试集（概率密度）----
    ax_r.hist(y_test, bins=bins, density=True, color='#54A24B', edgecolor='white', alpha=0.85)
    test_mean = np.mean(y_test)
    test_median = np.median(y_test)
    test_std = np.std(y_test)
    ax_r.axvline(test_mean, color='#E45756', ls='--', lw=1.5, label=f'Mean={test_mean:.2f}')
    ax_r.axvline(test_median, color='#F58518', ls=':', lw=1.5, label=f'Median={test_median:.2f}')
    ax_r.set_xlabel('LID', fontsize=11)
    ax_r.set_title(f'Test LID Distribution (n={len(y_test)}, σ={test_std:.2f})', fontsize=12)
    ax_r.legend(loc='upper right', fontsize=9)
    ax_r.grid(True, alpha=0.3, axis='y')

    plt.suptitle('LID Value Distribution: Train vs Test', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_LID_DIST, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ LID distribution plot saved to {OUTPUT_PLOT_LID_DIST}")


def round_to_100(values: np.ndarray, decimals: int = 2) -> np.ndarray:
    """
    百分比舍入修正（最大余额法）：四舍五入到 decimals 位后，
    各项之和严格等于 100，避免逐项 round 产生的 99.99%/100.01% 尾差。

    Args:
        values: 百分比数组（全精度，理论总和应为 100）
        decimals: 保留小数位数

    Returns:
        舍入修正后的数组，总和恰为 100 * 10^decimals / 10^decimals
    """
    factor = 10 ** decimals
    scaled = np.asarray(values, dtype=float) * factor
    floors = np.floor(scaled).astype(int)
    remainder = int(round(100 * factor)) - floors.sum()
    if remainder == 0:
        return floors / factor
    fracs = scaled - floors
    order = np.argsort(-fracs)
    if remainder > 0:
        # 差额为正：给小数部分最大的几项各 +1
        floors[order[:remainder]] += 1
    else:
        # 差额为负：从小数部分最小的几项各 -1
        floors[order[remainder:]] -= 1
    return floors / factor


def train_model() -> lgb.Booster:
    """
    训练 LID 预测模型。
    
    Returns:
        训练好的 LightGBM 模型
    """
    # 加载数据
    print("="*50)
    print("Loading Training Data")
    print("="*50)
    D_train, V_train, y_train = load_dataset(
        TRAIN_CENTROID_FILE, TRAIN_LID_FILE, TRAIN_VECTOR_FILE, "Train"
    )
    
    # 剔除训练集中 LID 值等于固定值的异常样本
    if EXCLUDE_LID_VALUE is not None:
        print(f"\n--- Filtering Training Data (exclude LID == {EXCLUDE_LID_VALUE}) ---")
        D_train, V_train, y_train = filter_by_lid_value(
            D_train, V_train, y_train, EXCLUDE_LID_VALUE
        )
    
    print("\n" + "="*50)
    print("Loading Test Data")
    print("="*50)
    D_test, V_test, y_test = load_dataset(
        TEST_CENTROID_FILE, TEST_LID_FILE, TEST_VECTOR_FILE, "Test"
    )
    
    print(f"\nLID Range:")
    print(f"  Train: [{y_train.min():.1f}, {y_train.max():.1f}]")
    print(f"  Test:  [{y_test.min():.1f}, {y_test.max():.1f}]")

    # 绘制 LID 值分布直方图（训练集 vs 测试集）
    print("\n" + "="*50)
    print("Plotting LID Distribution")
    print("="*50)
    plot_lid_distribution(y_train, y_test)

    # 构建特征
    print("\n" + "="*50)
    print("Building Features")
    print("="*50)
    print("\nTraining set:")
    X_train, feat_names = build_feature_matrix(D_train, V_train, y=y_train, save_csv=True)
    print("\nTest set:")
    X_test, _ = build_feature_matrix(D_test, V_test, y=y_test, save_csv=False)
    
    # 创建 LightGBM 数据集
    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_test, y_test, reference=train_data)
    
    # 训练模型（同时记录训练集与验证集指标）
    print("\n" + "="*50)
    print("Training Model")
    print("="*50)
    model_params, num_boost_round, min_delta = get_training_config(DATA_DIR)
    dataset_name = os.path.basename(os.path.normpath(DATA_DIR)).lower()
    pearson_stopping = dataset_name in DATASET_PEARSON_EARLY_STOPPING
    stopping_rounds = DATASET_PEARSON_EARLY_STOPPING.get(dataset_name, EARLY_STOPPING_ROUNDS)
    print(f"Training config: learning_rate={model_params['learning_rate']}, "
          f"max_rounds={num_boost_round}, early_stopping={stopping_rounds}, "
          f"min_delta={min_delta}, num_leaves={model_params['num_leaves']}, "
          f"max_depth={model_params['max_depth']}, num_threads={model_params['num_threads']}")
    if pearson_stopping:
        print(f"Early stopping metric: Pearson on true LID in [0, {LID_PLOT_MAX}]"
              if LID_PLOT_MAX is not None else "Early stopping metric: Pearson on all labels")
    eval_history: Dict = {}
    model = lgb.train(
        model_params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        feval=make_pearson_evaluation(LID_PLOT_MAX) if pearson_stopping else None,
        callbacks=[
            lgb.log_evaluation(LOG_EVALUATION_PERIOD),
            lgb.early_stopping(stopping_rounds, first_metric_only=pearson_stopping,
                               min_delta=min_delta),
            lgb.record_evaluation(eval_history)
        ]
    )
    
    # 预测（测试集 + 训练集）
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    # 保存 RMSE 训练过程到 CSV
    rmse_history_df = pd.DataFrame({
        'iteration': range(1, len(eval_history['train']['rmse']) + 1),
        'train_rmse': eval_history['train']['rmse'],
        'valid_rmse': eval_history['valid']['rmse']
    })
    rmse_history_df.to_csv(OUTPUT_RMSE_HISTORY, index=False)
    print(f"✓ RMSE history saved to {OUTPUT_RMSE_HISTORY}")
    
    # 评估（与拟合散点图使用相同的 LID 统计范围）
    metrics = evaluate_predictions(y_test, y_pred, lid_max=LID_PLOT_MAX)
    print_metrics(metrics)
    
    # 特征重要性
    importance = model.feature_importance(importance_type='gain')
    total_gain = importance.sum()
    feat_importance = pd.DataFrame({
        'feature': feat_names,
        'importance': importance,
        'percent': importance / total_gain if total_gain > 0 else 0.0
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*50)
    print("Feature Importance (Top 10)")
    print("="*50)
    top10 = feat_importance.head(10)
    top10_pct = round_to_100(top10['percent'].values * 100)
    for (_, row), pct in zip(top10.iterrows(), top10_pct):
        print(f"  {row['feature']:25s} {row['importance']:>10.0f}  ({pct:5.2f}%)")
    
    feat_importance['percent'] = feat_importance['percent'] * 100
    feat_importance.to_csv(OUTPUT_IMPORTANCE, index=False)
    print(f"✓ Feature importance saved to {OUTPUT_IMPORTANCE}")
    
    # 质心相关 5 特征的重要性（组内归一化，百分比之和严格 = 100%）
    centroid_imp = feat_importance[feat_importance['feature'].isin(CENTROID_FEATURES)].copy()
    group_gain = centroid_imp['importance'].sum()
    raw_pct = (centroid_imp['importance'] / group_gain * 100
               if group_gain > 0 else np.zeros(len(centroid_imp)))
    centroid_imp['percent_in_group'] = round_to_100(raw_pct.values)
    centroid_imp = centroid_imp.sort_values('importance', ascending=False)
    centroid_imp.to_csv(OUTPUT_CENTROID_IMPORTANCE, index=False,
                        columns=['feature', 'importance', 'percent_in_group'])
    
    print("\n" + "="*50)
    print("Centroid Features Importance (group-normalized, sums to 100%)")
    print("="*50)
    for _, row in centroid_imp.iterrows():
        print(f"  {row['feature']:25s} {row['percent_in_group']:5.2f}%")
    print(f"  Group sum check: {centroid_imp['percent_in_group'].sum():.2f}%")
    print(f"✓ Centroid feature importance saved to {OUTPUT_CENTROID_IMPORTANCE}")
    
    # 可视化（含训练集拟合效果与RMSE训练曲线）
    visualize_results(y_test, y_pred, y_train, y_train_pred, feat_importance, eval_history)
    
    # 保存模型
    model.save_model(OUTPUT_MODEL)
    save_model_metadata(OUTPUT_MODEL, feat_names, model_params, LID_PLOT_MAX,
                        'pearson_fixed' if pearson_stopping else 'rmse')
    print(f"✓ Model saved to {OUTPUT_MODEL}")
    
    # 保存测试集预测结果
    pd.DataFrame({'id': np.arange(len(y_test)), 'real': y_test, 'predicted': y_pred}).to_csv(
        OUTPUT_PREDICTIONS, index=False
    )
    print(f"✓ Test predictions saved to {OUTPUT_PREDICTIONS}")
    
    # 保存训练集预测结果
    pd.DataFrame({'id': np.arange(len(y_train)), 'real': y_train, 'predicted': y_train_pred}).to_csv(
        OUTPUT_TRAIN_PREDICTIONS, index=False
    )
    print(f"✓ Train predictions saved to {OUTPUT_TRAIN_PREDICTIONS}")
    
    # 保存指标
    save_metrics_file(metrics)
    
    return model


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主入口函数。"""
    global LID_PLOT_MAX

    # 解析命令行参数，根据传入的数据目录重新初始化所有文件路径
    args = parse_args()
    _setup_paths(args.dataset)

    # 应用拟合散点图的 LID 统计上限（visualize_results 默认读取该全局值）
    LID_PLOT_MAX = args.lid_max

    # 确保输出目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"数据目录: {os.path.abspath(DATA_DIR)}")
    print(f"输出目录: {os.path.abspath(DATA_DIR)}")

    print("\n" + "="*70)
    print("LID Prediction Model Training")
    print("="*70 + "\n")
    
    model = train_model()
    
    print("\n" + "="*70)
    print("Training Completed Successfully!")
    print("="*70 + "\n")
    
    return model


if __name__ == '__main__':
    main()
