"""train_ml_model.py - trains, tests and saves the K-Means customer clustering model.

Run:  python train_ml_model.py
Creates: models/kmeans_ml_model.joblib  and  models/ml_metrics.json
"""
import time

import ml_pipeline as ml


def main() -> None:
    start = time.time()
    print("=" * 62)
    print(" K-Means Customer Clustering ML - training & testing")
    print("=" * 62)

    df, cleaning = ml.clean_dataset(ml.load_dataset())
    print(f"[1/5] Data loaded      : {cleaning['rows_after']} customers "
          f"({cleaning['missing_cells']} missing cells, {cleaning['duplicate_rows']} duplicates)")

    model, metrics = ml.run_training(df)
    print(f"[2/5] Train/Test split : {metrics['n_train']} train / {metrics['n_test']} test "
          f"(test_size={metrics['test_size']})")

    print("[3/5] Choosing K by 5-fold cross-validated silhouette (train set only):")
    for row in metrics["k_selection"]:
        star = "  <-- chosen" if row["k"] == metrics["chosen_k"] else ""
        print(f"        K={row['k']:>2}  cv_silhouette={row['cv_silhouette']:.3f} "
              f"(+/- {row['cv_std']:.3f}){star}")

    tr, te = metrics["train"], metrics["test"]
    print(f"[4/5] Evaluation with K={metrics['chosen_k']}")
    print(f"        {'metric':<22}{'train':>10}{'test':>10}")
    for key in ["silhouette", "davies_bouldin", "calinski_harabasz", "inertia_per_sample"]:
        print(f"        {key:<22}{tr[key]:>10.3f}{te[key]:>10.3f}")
    print(f"        random-label baseline silhouette (test): {metrics['baseline_random_silhouette']:.3f}")
    st = metrics["stability"]
    print(f"        bootstrap stability (ARI): {st['mean_ari']:.3f} +/- {st['std_ari']:.3f}")

    metrics["cleaning"] = cleaning
    ml.save_artifacts(model, metrics)
    print(f"[5/5] Saved {ml.MODEL_PATH.name} and {ml.METRICS_PATH.name} to the models/ folder")
    print(f"Done in {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
