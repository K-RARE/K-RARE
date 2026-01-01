import os
import json
import random
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

DATA_DIR = "datasets"

RUN_DIR = "results/KNN"

CSV_OUT_DIR = "results/KNN"

TEST_SIZE_INFO = "Using externally pre-split train/test CSV files"
RANDOM_STATE = 42

TOPK = 3

TASKS = [
    {
        "name": "pressure_low",
        "train_csv": os.path.join(DATA_DIR, "pressureLow_train.csv"),
        "test_csv": os.path.join(DATA_DIR, "pressureLow_test.csv"),
        "target_col": "pressure_low",
    },
    {
        "name": "pressure_high",
        "train_csv": os.path.join(DATA_DIR, "pressureHigh_train.csv"),
        "test_csv": os.path.join(DATA_DIR, "pressureHigh_test.csv"),
        "target_col": "pressure_high",
    },
]

MISSING_TOKEN = "NA_MISSING"


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def hamming_distance_row(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


class KNNClassifier:
    def __init__(self, n_neighbors=5, weights="uniform", random_state=42, verbose=False):
        self.n_neighbors = int(n_neighbors)
        self.weights = str(weights)
        self.random_state = random_state
        self.verbose = verbose

        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None
        self.class_prior_ = None

    def _weight_from_distance(self, d: float) -> float:
        if self.weights == "uniform":
            return 1.0
        elif self.weights == "distance":
            return 1.0 / (1.0 + float(d))
        return 1.0

    def _compute_class_prior(self):
        counts = Counter(self.y_train_)
        total = float(len(self.y_train_))
        prior = {}
        for c in self.classes_:
            prior[c] = counts.get(c, 0.0) / total if total > 0 else 0.0
        return prior

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D array.")
        if y_arr.ndim != 1:
            raise ValueError("y must be 1D array.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.X_train_ = X_arr
        self.y_train_ = y_arr
        self.classes_ = np.unique(self.y_train_)
        self.class_prior_ = self._compute_class_prior()
        return self

    def _predict_one_proba(self, x: np.ndarray) -> dict:
        dists = [hamming_distance_row(x, self.X_train_[i]) for i in range(self.X_train_.shape[0])]

        k = min(self.n_neighbors, len(dists))
        if k <= 0:
            return dict(self.class_prior_)

        idx_sorted = np.argsort(dists)[:k]

        weights_per_class = {c: 0.0 for c in self.classes_}
        for idx in idx_sorted:
            d = dists[idx]
            w = self._weight_from_distance(d)
            cls = self.y_train_[idx]
            weights_per_class[cls] += w

        total_w = sum(weights_per_class.values())
        if total_w <= 0:
            return dict(self.class_prior_)

        for c in weights_per_class:
            weights_per_class[c] /= total_w
        return weights_per_class

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        n_samples = X_arr.shape[0]
        proba = np.zeros((n_samples, len(self.classes_)), dtype=float)
        for i in range(n_samples):
            proba_dict = self._predict_one_proba(X_arr[i])
            for j, c in enumerate(self.classes_):
                proba[i, j] = proba_dict.get(c, 0.0)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        best_idx = np.argmax(proba, axis=1)
        return self.classes_[best_idx]

def load_best_params_for_task(task_name: str):
    cfg_path = os.path.join(RUN_DIR, f"{task_name}_knn_best_params.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "best_params" in data:
                best_params = data["best_params"]
            else:
                best_params = data
            print(f"[{task_name}] Loaded best_params from {cfg_path}: {best_params}")
            return best_params
        except Exception as e:
            print(f"[WARN] Failed to read {cfg_path}. Reason: {e}. Using default params.")
    else:
        print(f"[WARN] File not found: {cfg_path}. Using default params.")

    default_params = {"n_neighbors": 5, "weights": "uniform"}
    print(f"[{task_name}] Using default params: {default_params}")
    return default_params

def run_knn_for_task(task: dict):
    name = task["name"]
    train_csv = task["train_csv"]
    test_csv = task["test_csv"]
    target_col = task["target_col"]

    print("\n" + "=" * 80)
    print(f"Task: {name}")
    print(f"Train CSV: {train_csv}")
    print(f"Test  CSV: {test_csv}")
    print("=" * 80)

    train_df = pd.read_csv(train_csv, dtype=str)
    test_df = pd.read_csv(test_csv, dtype=str)

    feature_cols = [c for c in train_df.columns if c not in ["pressure_low", "pressure_high"]]

    X_train = train_df[feature_cols].astype(str).fillna(MISSING_TOKEN).values
    y_train = train_df[target_col].astype(str).fillna(MISSING_TOKEN).values

    for col in feature_cols:
        if col not in test_df.columns:
            raise ValueError(f"Test set is missing feature column: {col}")
    X_test = test_df[feature_cols].astype(str).fillna(MISSING_TOKEN).values
    y_test = test_df[target_col].astype(str).fillna(MISSING_TOKEN).values

    best_params = load_best_params_for_task(name)

    clf = KNNClassifier(
        n_neighbors=best_params.get("n_neighbors", 5),
        weights=best_params.get("weights", "uniform"),
        random_state=RANDOM_STATE,
        verbose=False
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    classes = clf.classes_.tolist()
    k_eff = min(TOPK, len(classes))
    n_samples = X_test.shape[0]

    rows = []
    ranks_full = []
    top1_preds = []

    for i in range(n_samples):
        probs = proba[i]
        truth = y_test[i]

        order = np.argsort(probs)[::-1]
        sorted_classes = [classes[j] for j in order]

        top1 = sorted_classes[0] if len(sorted_classes) > 0 else ""
        top2 = sorted_classes[1] if len(sorted_classes) > 1 else ""
        top3 = sorted_classes[2] if len(sorted_classes) > 2 else ""

        top1_preds.append(top1)

        rank_full = sorted_classes.index(truth) + 1 if truth in sorted_classes else None
        ranks_full.append(rank_full)

        rank_within_top3 = None
        for idx, cls in enumerate(sorted_classes[:k_eff], start=1):
            if cls == truth:
                rank_within_top3 = idx
                break

        rows.append({
            "row_index": i + 1,
            "top1_pred": top1,
            "top2_pred": top2,
            "top3_pred": top3,
            "first_correct_rank_within_top3": rank_within_top3 if rank_within_top3 is not None else ""
        })

    accuracy = accuracy_score(y_test, top1_preds) if n_samples > 0 else 0.0

    print(f"[{name}] Test samples: {n_samples}")
    print(f"[{name}] Top-1 accuracy = {accuracy:.4f}")
    print(f"[{name}] Top-k coverage & average decision steps (samples with rank<=k):")
    print(" k  covered  total  coverage    avg_rank_within_topk")

    avg_rank_per_k = {}
    for k in range(1, TOPK + 1):
        covered_idx = [idx for idx, r in enumerate(ranks_full) if (r is not None and r <= k)]
        covered_cnt = len(covered_idx)
        coverage_k = covered_cnt / n_samples if n_samples > 0 else 0.0
        avg_rank_k = float(np.mean([ranks_full[idx] for idx in covered_idx])) if covered_cnt > 0 else float("nan")
        avg_rank_per_k[k] = avg_rank_k
        print(f"{k:2d}  {covered_cnt:7d}  {n_samples:5d}  {coverage_k:9.6f}      {avg_rank_k:.4f}")

    avg_rank_topK = avg_rank_per_k.get(TOPK, float("nan"))

    summary_row = {
        "row_index": "SUMMARY",
        "top1_pred": f"accuracy={accuracy:.6f}",
        "top2_pred": (
            f"avg_rank_top{TOPK}={avg_rank_topK:.6f}"
            if not np.isnan(avg_rank_topK) else
            f"avg_rank_top{TOPK}=N/A"
        ),
        "top3_pred": "",
        "first_correct_rank_within_top3": ""
    }
    rows.append(summary_row)

    df_out = pd.DataFrame(rows)

    os.makedirs(CSV_OUT_DIR, exist_ok=True)
    if name == "pressure_low":
        out_csv = os.path.join(CSV_OUT_DIR, "knn_topk_low.csv")
    elif name == "pressure_high":
        out_csv = os.path.join(CSV_OUT_DIR, "knn_topk_high.csv")
    else:
        out_csv = os.path.join(CSV_OUT_DIR, f"{name}_knn_top{TOPK}_results.csv")

    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[{name}] Top-{TOPK} prediction results (including SUMMARY) saved to: {out_csv}")


def main():
    set_global_seed(RANDOM_STATE)
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(CSV_OUT_DIR, exist_ok=True)

    print(f"best_params JSON directory: {RUN_DIR}")
    print(f"KNN TopK CSV directory   : {CSV_OUT_DIR}")
    print(f"Data directory           : {DATA_DIR}")
    print(f"Notes                    : {TEST_SIZE_INFO}")
    print("=" * 80)

    for task in TASKS:
        run_knn_for_task(task)


if __name__ == "__main__":
    main()
