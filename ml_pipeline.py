"""ml_pipeline.py - all machine-learning logic for the K-Means customer clustering project.

There is NO Streamlit code in this file, so it can be trained, tested and reused anywhere:

    python train_ml_model.py              # train + evaluate + save the model
    python -m unittest test_ml_model -v   # run the test-suite

Pipeline:  load -> clean -> train/test split -> choose K (cross-validation on TRAIN only)
           -> evaluate on the unseen TEST set -> stability check -> refit on all data -> save.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "kmeans_ml_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "ml_metrics.json"

ID_COL = "CustomerID"
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
CATEGORICAL = ["Gender", "Education", "Marital Status"]
INCOME, SPENDING, AGE = FEATURES[1], FEATURES[2], FEATURES[0]

TEST_SIZE = 0.2
RANDOM_STATE = 42
K_RANGE = range(2, 11)
N_SPLITS = 5
N_BOOTSTRAP = 30


# --------------------------------------------------------------------------- #
# 1. Data loading + cleaning
# --------------------------------------------------------------------------- #
def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in [ID_COL, *FEATURES] if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"M": "Male", "F": "Female"})
    return df


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove duplicates / rows with missing features. Returns (clean_df, report)."""
    report = {
        "rows_before": int(len(df)),
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    df = df.drop_duplicates().dropna(subset=FEATURES).reset_index(drop=True)
    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")
    report["rows_after"] = int(len(df))
    return df, report


def file_hash(path: str | Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def split_indices(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """The (fixed, reproducible) train / test split used everywhere."""
    return train_test_split(np.arange(n_rows), test_size=TEST_SIZE, random_state=RANDOM_STATE)


# --------------------------------------------------------------------------- #
# 2. Model
# --------------------------------------------------------------------------- #
@dataclass
class ClusterModel:
    """StandardScaler + KMeans, with clusters numbered from low to high income."""

    features: list
    scaler: StandardScaler
    kmeans: KMeans
    order: np.ndarray  # order[new_label] = original kmeans label

    @property
    def k(self) -> int:
        return int(self.kmeans.n_clusters)

    def scale(self, X) -> np.ndarray:
        return self.scaler.transform(np.asarray(X, dtype=float))

    def predict(self, X) -> np.ndarray:
        inverse = np.argsort(self.order)
        return inverse[self.kmeans.predict(self.scale(X))]

    def distances(self, X) -> np.ndarray:
        """Distance (in scaled space) from each row to each cluster centre."""
        return self.kmeans.transform(self.scale(X))[:, self.order]

    @property
    def centers_(self) -> pd.DataFrame:
        """Cluster centres in the ORIGINAL units (years, k$, score)."""
        c = self.scaler.inverse_transform(self.kmeans.cluster_centers_[self.order])
        return pd.DataFrame(c, columns=self.features)


def make_model(X, k: int) -> ClusterModel:
    X = np.asarray(X, dtype=float)
    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    km.fit(scaler.transform(X))
    centers = scaler.inverse_transform(km.cluster_centers_)
    order = np.lexsort((centers[:, FEATURES.index(SPENDING)], centers[:, FEATURES.index(INCOME)]))
    return ClusterModel(list(FEATURES), scaler, km, order)


# --------------------------------------------------------------------------- #
# 3. Choosing K (uses TRAIN data only) and evaluation
# --------------------------------------------------------------------------- #
def select_k(X_train, k_range=K_RANGE, n_splits: int = N_SPLITS) -> pd.DataFrame:
    """For every K: cross-validated silhouette (fit on the training folds, score on the
    held-out fold) + inertia and silhouette on the full training set."""
    X_train = np.asarray(X_train, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for k in k_range:
        fold_scores = []
        for tr, va in kf.split(X_train):
            m = make_model(X_train[tr], k)
            labels = m.predict(X_train[va])
            if 1 < len(set(labels)) < len(va):
                fold_scores.append(silhouette_score(m.scale(X_train[va]), labels))
        full = make_model(X_train, k)
        labels = full.predict(X_train)
        rows.append(
            {
                "k": int(k),
                "cv_silhouette": float(np.mean(fold_scores)),
                "cv_std": float(np.std(fold_scores)),
                "train_silhouette": float(silhouette_score(full.scale(X_train), labels)),
                "train_inertia": float(full.kmeans.inertia_),
            }
        )
    return pd.DataFrame(rows)


def evaluate_model(model: ClusterModel, X) -> dict:
    """Clustering quality on any data (train or test)."""
    X = np.asarray(X, dtype=float)
    labels = model.predict(X)
    Xs = model.scale(X)
    ok = 1 < len(set(labels)) < len(X)
    return {
        "n_samples": int(len(X)),
        "silhouette": float(silhouette_score(Xs, labels)) if ok else float("nan"),
        "davies_bouldin": float(davies_bouldin_score(Xs, labels)) if ok else float("nan"),
        "calinski_harabasz": float(calinski_harabasz_score(Xs, labels)) if ok else float("nan"),
        "inertia_per_sample": float((model.distances(X).min(axis=1) ** 2).mean()),
    }


def cluster_share(model: ClusterModel, X) -> list:
    counts = np.bincount(model.predict(X), minlength=model.k)
    return (counts / counts.sum() * 100).round(2).tolist()


# --------------------------------------------------------------------------- #
# 4. Full training run
# --------------------------------------------------------------------------- #
def run_training(df: pd.DataFrame, k_range=K_RANGE, n_boot: int = N_BOOTSTRAP):
    """Train + test everything. Returns (final_model, metrics_dict)."""
    X = df[FEATURES].to_numpy(dtype=float)
    idx_train, idx_test = split_indices(len(X))
    X_train, X_test = X[idx_train], X[idx_test]

    # (a) choose K with cross-validation on the training set only
    k_table = select_k(X_train, k_range)
    best_k = int(k_table.loc[k_table["cv_silhouette"].idxmax(), "k"])

    # (b) fit on TRAIN, evaluate on unseen TEST
    train_model = make_model(X_train, best_k)
    train_scores = evaluate_model(train_model, X_train)
    test_scores = evaluate_model(train_model, X_test)

    # (c) baseline: what silhouette would purely random labels get on the test set?
    rng = np.random.default_rng(RANDOM_STATE)
    random_labels = rng.integers(0, best_k, len(X_test))
    baseline = float(silhouette_score(train_model.scale(X_test), random_labels))

    # (d) final model = refit on ALL data (standard practice once the design is validated)
    final_model = make_model(X, best_k)
    ref_labels = final_model.predict(X)

    # (e) stability: refit on bootstrap resamples and compare with the final clusters
    aris = []
    for _ in range(n_boot):
        sample = rng.integers(0, len(X), len(X))
        m = make_model(X[sample], best_k)
        aris.append(float(adjusted_rand_score(ref_labels, m.predict(X))))

    metrics = {
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "n_rows": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "k_selection": k_table.round(5).to_dict(orient="records"),
        "chosen_k": best_k,
        "train": train_scores,
        "test": test_scores,
        "baseline_random_silhouette": baseline,
        "train_vs_final_ari": float(
            adjusted_rand_score(final_model.predict(X), train_model.predict(X))
        ),
        "stability": {
            "n_bootstrap": n_boot,
            "ari_values": [round(a, 4) for a in aris],
            "mean_ari": float(np.mean(aris)),
            "std_ari": float(np.std(aris)),
        },
        "cluster_share": {
            "train": cluster_share(train_model, X_train),
            "test": cluster_share(train_model, X_test),
        },
    }
    return final_model, metrics


def save_artifacts(model, metrics, data_path=DATA_PATH, model_path=MODEL_PATH, metrics_path=METRICS_PATH):
    model_path, metrics_path = Path(model_path), Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {**metrics, "data_hash": file_hash(data_path)}
    joblib.dump({"model": model, "sklearn_version": sklearn.__version__}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def load_or_train(
    data_path=DATA_PATH, model_path=MODEL_PATH, metrics_path=METRICS_PATH, force: bool = False
):
    """Load the saved model if it is valid; otherwise train a new one automatically.

    Returns (model, metrics, clean_df, cleaning_report, status_message).
    The automatic retrain means the app can never show a 'no trained model' error, even if
    scikit-learn on the server is a different version than the one used to save the file.
    """
    df, cleaning = clean_dataset(load_dataset(data_path))
    reason = "forced retrain"
    if not force:
        try:
            metrics = json.loads(Path(metrics_path).read_text())
            if metrics.get("sklearn_version") != sklearn.__version__:
                raise RuntimeError("scikit-learn version on this server differs from the saved model")
            if metrics.get("data_hash") != file_hash(data_path):
                raise RuntimeError("dataset changed since the model was trained")
            bundle = joblib.load(model_path)
            metrics["cleaning"] = cleaning
            return bundle["model"], metrics, df, cleaning, "Loaded saved model"
        except Exception as exc:  # noqa: BLE001
            reason = str(exc) or "no saved model found"

    model, metrics = run_training(df)
    metrics["cleaning"] = cleaning
    try:
        save_artifacts(model, metrics, data_path, model_path, metrics_path)
    except Exception:  # noqa: BLE001  (read-only disk is fine, we keep it in memory)
        pass
    return model, metrics, df, cleaning, f"Trained on start-up ({reason})"


# --------------------------------------------------------------------------- #
# 5. Prediction helpers
# --------------------------------------------------------------------------- #
def predict_customer(model: ClusterModel, values: dict) -> dict:
    x = np.array([[values[f] for f in model.features]], dtype=float)
    d = model.distances(x)[0]
    w = 1.0 / (d**2 + 1e-9)
    nearest = np.sort(d)
    return {
        "cluster": int(model.predict(x)[0]),
        "distances": d,
        "match": w / w.sum(),
        # 0 = sits exactly between two segments, 1 = right on its own centroid
        "confidence": float((nearest[1] - nearest[0]) / max(nearest[1], 1e-9)),
    }


def nearest_customers(model: ClusterModel, df: pd.DataFrame, values: dict, n: int = 5) -> pd.DataFrame:
    x = model.scale(np.array([[values[f] for f in model.features]], dtype=float))
    Xs = model.scale(df[model.features].to_numpy(dtype=float))
    dist = np.sqrt(((Xs - x) ** 2).sum(axis=1))
    out = df.iloc[np.argsort(dist)[:n]].copy()
    out["Distance"] = np.sort(dist)[:n].round(2)
    return out


# --------------------------------------------------------------------------- #
# 6. Segment naming + marketing advice
# --------------------------------------------------------------------------- #
PERSONAS = {
    ("High", "High"): ("💎 Premium Spenders", "High income and high spending - your most valuable customers.",
                       "Reward loyalty: VIP membership, early access to new arrivals, personalised offers and premium service."),
    ("High", "Average"): ("📈 Affluent Regulars", "High income with moderate spending - they can easily buy more.",
                          "Upsell and cross-sell: bundles, premium ranges and personalised recommendations."),
    ("High", "Low"): ("🧠 Careful Earners", "High income but low spending - capable buyers who are not convinced yet.",
                      "Win them with quality-focused campaigns, exclusive deals, reviews and trust-building content."),
    ("Average", "High"): ("🌟 Enthusiastic Shoppers", "Average income but high spending - they love to shop.",
                          "Loyalty points, seasonal campaigns and referral rewards to keep them engaged."),
    ("Average", "Average"): ("⚖️ Balanced Standard", "Average income and average spending - the stable core of the customer base.",
                             "Regular promotions, combo offers and reminders to lift spending gradually."),
    ("Average", "Low"): ("🪙 Cautious Middle", "Average income but low spending - they hold back.",
                         "Value-for-money messaging, limited-time discounts and easy returns."),
    ("Low", "High"): ("🛍️ Impulsive Shoppers", "Low income but high spending - trend-driven, impulsive buyers.",
                      "Flash sales, instalment plans and trendy low-price items (watch credit risk)."),
    ("Low", "Average"): ("🧾 Value Seekers", "Low income and average spending - they look for good deals.",
                         "Value packs, coupons and bundle discounts."),
    ("Low", "Low"): ("💰 Budget Savers", "Low income and low spending - very price-sensitive.",
                     "Discount coupons and low-cost entry products; keep marketing spend low."),
}


def _level(z: float, hi: float = 0.5, lo: float = -0.5) -> str:
    return "High" if z >= hi else "Low" if z <= lo else "Average"


def describe_segments(df: pd.DataFrame, label_col: str = "Cluster") -> dict:
    """Name every cluster and attach a description + marketing strategy.

    `df` must contain the FEATURES and a cluster-label column."""
    mean, std = df[FEATURES].mean(), df[FEATURES].std(ddof=0).replace(0, 1)
    profiles = df.groupby(label_col)[FEATURES].mean()
    sizes = df.groupby(label_col).size()

    segments = {}
    for cid, row in profiles.iterrows():
        z = (row - mean) / std
        title, desc, strategy = PERSONAS[(_level(z[INCOME]), _level(z[SPENDING]))]
        age_level = _level(z[AGE])
        age_tag = {"High": "Older", "Low": "Younger", "Average": "Mid-age"}[age_level]
        if age_level != "Average":
            desc += f" Typically {age_tag.lower()} (average age {row[AGE]:.0f})."
        segments[int(cid)] = {
            "title": title, "age_tag": age_tag, "description": desc, "strategy": strategy,
            "size": int(sizes[cid]), "pct": float(sizes[cid] / len(df) * 100),
            "means": row.to_dict(),
        }

    # Two clusters can share a persona (e.g. two "Premium Spenders" groups) -> tell them apart
    counts = Counter(s["title"] for s in segments.values())
    for s in segments.values():
        if counts[s["title"]] > 1:
            s["title"] = f"{s['title']} · {s['age_tag']}"
    counts = Counter(s["title"] for s in segments.values())
    for cid, s in segments.items():
        if counts[s["title"]] > 1:
            s["title"] = f"{s['title']} (C{cid})"
    return segments
