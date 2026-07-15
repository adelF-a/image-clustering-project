import unittest

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

from algorithms import CustomDBSCAN, CustomKMeans


class CustomKMeansTests(unittest.TestCase):
    def test_random_state_zero_is_reproducible(self):
        X, _ = make_blobs(n_samples=180, centers=3, random_state=12)
        first = CustomKMeans(n_clusters=3, random_state=0).fit(X)
        second = CustomKMeans(n_clusters=3, random_state=0).fit(X)
        np.testing.assert_allclose(first.cluster_centers_, second.cluster_centers_)
        np.testing.assert_array_equal(first.labels_, second.labels_)
        self.assertAlmostEqual(first.inertia_, second.inertia_)

    def test_empty_cluster_does_not_create_nan_centers(self):
        X = np.array([[0.0], [0.0], [10.0], [10.0]])
        model = CustomKMeans(n_clusters=3, random_state=4).fit(X)
        self.assertTrue(np.isfinite(model.cluster_centers_).all())
        self.assertTrue(np.isfinite(model.inertia_))

    def test_predict_uses_fitted_centers(self):
        X = np.array([[0.0], [0.2], [9.8], [10.0]])
        model = CustomKMeans(n_clusters=2, random_state=3).fit(X)
        np.testing.assert_array_equal(model.labels_, model.predict(X))

    def test_recovers_well_separated_blob_partition(self):
        X, expected = make_blobs(
            n_samples=240, centers=3, cluster_std=0.35, random_state=21
        )
        actual = CustomKMeans(n_clusters=3, random_state=7).fit_predict(X)
        self.assertAlmostEqual(adjusted_rand_score(expected, actual), 1.0)


class CustomDBSCANTests(unittest.TestCase):
    def test_density_reachability_expands_beyond_first_neighborhood(self):
        X = np.arange(0, 2.0, 0.4).reshape(-1, 1)
        labels = CustomDBSCAN(eps=0.41, min_samples=2).fit_predict(X)
        self.assertEqual(len(set(labels)), 1)
        self.assertNotIn(CustomDBSCAN.NOISE, labels)

    def test_partition_matches_sklearn_on_moons(self):
        X, _ = make_moons(n_samples=250, noise=0.04, random_state=8)
        expected = DBSCAN(eps=0.18, min_samples=5).fit_predict(X)
        actual = CustomDBSCAN(eps=0.18, min_samples=5).fit_predict(X)
        self.assertAlmostEqual(adjusted_rand_score(expected, actual), 1.0)
        np.testing.assert_array_equal(expected == -1, actual == -1)

    def test_invalid_parameters_are_rejected(self):
        X = np.array([[0.0], [1.0]])
        with self.assertRaises(ValueError):
            CustomDBSCAN(eps=0).fit(X)
        with self.assertRaises(ValueError):
            CustomDBSCAN(min_samples=0).fit(X)


if __name__ == "__main__":
    unittest.main()
