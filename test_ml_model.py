"""test_ml_model.py - automated tests for the K-Means customer clustering ML project.

Run:  python -m unittest test_ml_model -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score

import ml_pipeline as ml


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = ml.load_dataset()
        cls.df, cls.report = ml.clean_dataset(cls.raw)

    def test_required_columns_exist(self):
        for col in [ml.ID_COL, *ml.FEATURES, *ml.CATEGORICAL]:
            self.assertIn(col, self.df.columns)

    def test_no_missing_values_in_features(self):
        self.assertEqual(int(self.df[ml.FEATURES].isnull().sum().sum()), 0)

    def test_cleaning_report_is_consistent(self):
        self.assertEqual(self.report["rows_after"], len(self.df))
        self.assertLessEqual(self.report["rows_after"], self.report["rows_before"])

    def test_feature_ranges_are_sensible(self):
        self.assertTrue(self.df["Age"].between(0, 120).all())
        self.assertTrue((self.df[ml.INCOME] > 0).all())
        self.assertTrue(self.df[ml.SPENDING].between(1, 100).all())

    def test_train_test_split_is_disjoint_and_complete(self):
        tr, te = ml.split_indices(len(self.df))
        self.assertEqual(len(tr) + len(te), len(self.df))
        self.assertEqual(len(set(tr) & set(te)), 0)
        self.assertAlmostEqual(len(te) / len(self.df), ml.TEST_SIZE, delta=0.01)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, _ = ml.clean_dataset(ml.load_dataset())
        cls.X = cls.df[ml.FEATURES].to_numpy(float)
        cls.tr, cls.te = ml.split_indices(len(cls.X))
        cls.model = ml.make_model(cls.X[cls.tr], 6)

    def test_labels_are_valid_cluster_ids(self):
        labels = self.model.predict(self.X)
        self.assertEqual(set(labels), set(range(6)))

    def test_prediction_is_deterministic(self):
        a, b = self.model.predict(self.X), self.model.predict(self.X)
        self.assertTrue((a == b).all())

    def test_training_is_reproducible(self):
        again = ml.make_model(self.X[self.tr], 6)
        self.assertTrue(np.allclose(self.model.centers_.values, again.centers_.values))

    def test_centroid_is_predicted_as_its_own_cluster(self):
        centers = self.model.centers_.to_numpy()
        self.assertEqual(list(self.model.predict(centers)), list(range(6)))

    def test_clusters_are_ordered_by_income(self):
        income = self.model.centers_[ml.INCOME].to_numpy()
        self.assertTrue((np.diff(income) >= -1e-9).all())

    def test_scaler_is_fitted_on_training_data_only(self):
        self.assertTrue(np.allclose(self.model.scaler.mean_, self.X[self.tr].mean(axis=0)))

    def test_generalises_to_unseen_test_data(self):
        test_scores = ml.evaluate_model(self.model, self.X[self.te])
        self.assertGreater(test_scores["silhouette"], 0.25)

    def test_beats_random_labels(self):
        Xte = self.X[self.te]
        real = silhouette_score(self.model.scale(Xte), self.model.predict(Xte))
        rng = np.random.default_rng(0)
        random = silhouette_score(self.model.scale(Xte), rng.integers(0, 6, len(Xte)))
        self.assertGreater(real, random + 0.3)

    def test_far_away_customers_land_in_expected_segments(self):
        rich_big_spender = {"Age": 30, ml.INCOME: 130, ml.SPENDING: 95}
        poor_saver = {"Age": 45, ml.INCOME: 18, ml.SPENDING: 10}
        hi = ml.predict_customer(self.model, rich_big_spender)["cluster"]
        lo = ml.predict_customer(self.model, poor_saver)["cluster"]
        centers = self.model.centers_
        self.assertGreater(centers.loc[hi, ml.INCOME], 70)
        self.assertGreater(centers.loc[hi, ml.SPENDING], 65)
        self.assertLess(centers.loc[lo, ml.INCOME], 40)
        self.assertLess(centers.loc[lo, ml.SPENDING], 40)

    def test_prediction_outputs_are_well_formed(self):
        out = ml.predict_customer(self.model, {"Age": 35, ml.INCOME: 60, ml.SPENDING: 50})
        self.assertAlmostEqual(float(out["match"].sum()), 1.0, places=6)
        self.assertEqual(len(out["distances"]), 6)
        self.assertTrue(0.0 <= out["confidence"] <= 1.0)

    def test_nearest_customers_returns_requested_rows(self):
        near = ml.nearest_customers(self.model, self.df, {"Age": 35, ml.INCOME: 60, ml.SPENDING: 50}, n=5)
        self.assertEqual(len(near), 5)
        self.assertTrue(near["Distance"].is_monotonic_increasing)


class TestFullPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, _ = ml.clean_dataset(ml.load_dataset())
        cls.model, cls.metrics = ml.run_training(cls.df, n_boot=10)

    def test_chosen_k_in_range(self):
        self.assertIn(self.metrics["chosen_k"], list(ml.K_RANGE))
        self.assertEqual(self.model.k, self.metrics["chosen_k"])

    def test_test_metrics_are_reported_and_good(self):
        self.assertEqual(self.metrics["n_train"] + self.metrics["n_test"], self.metrics["n_rows"])
        self.assertGreater(self.metrics["test"]["silhouette"], 0.25)
        self.assertGreater(self.metrics["test"]["silhouette"], self.metrics["baseline_random_silhouette"])

    def test_clusters_are_stable_under_resampling(self):
        self.assertGreater(self.metrics["stability"]["mean_ari"], 0.6)

    def test_segments_are_named_and_unique(self):
        df = self.df.copy()
        df["Cluster"] = self.model.predict(df[ml.FEATURES].to_numpy(float))
        segments = ml.describe_segments(df)
        self.assertEqual(len(segments), self.model.k)
        titles = [s["title"] for s in segments.values()]
        self.assertEqual(len(set(titles)), len(titles))
        self.assertAlmostEqual(sum(s["pct"] for s in segments.values()), 100.0, places=4)


class TestArtifacts(unittest.TestCase):
    def test_saved_model_and_metrics_exist_and_match(self):
        self.assertTrue(ml.MODEL_PATH.exists(), "run train_ml_model.py first")
        self.assertTrue(ml.METRICS_PATH.exists(), "run train_ml_model.py first")
        model, metrics, _, _, status = ml.load_or_train()
        self.assertEqual(model.k, metrics["chosen_k"])
        self.assertTrue(status.startswith(("Loaded", "Trained")))

    def test_app_recovers_when_model_file_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, metrics, _, _, status = ml.load_or_train(
                model_path=Path(tmp) / "m.joblib", metrics_path=Path(tmp) / "m.json"
            )
            self.assertTrue(status.startswith("Trained"))
            self.assertGreaterEqual(model.k, 2)
            # ...and it saved fresh artifacts that load correctly next time
            _, _, _, _, status2 = ml.load_or_train(
                model_path=Path(tmp) / "m.joblib", metrics_path=Path(tmp) / "m.json"
            )
            self.assertEqual(status2, "Loaded saved model")

    def test_app_recovers_from_corrupted_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp, jp = Path(tmp) / "m.joblib", Path(tmp) / "m.json"
            ml.load_or_train(model_path=mp, metrics_path=jp)
            mp.write_bytes(b"this is not a model")
            _, _, _, _, status = ml.load_or_train(model_path=mp, metrics_path=jp)
            self.assertTrue(status.startswith("Trained"))

    def test_metrics_json_is_valid(self):
        data = json.loads(ml.METRICS_PATH.read_text())
        for key in ["chosen_k", "train", "test", "stability", "k_selection", "data_hash"]:
            self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main()
