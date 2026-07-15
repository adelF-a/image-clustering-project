"""Educational, scikit-learn-style implementations of K-Means and DBSCAN."""

from collections import deque

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array


class CustomKMeans:
    """K-Means with repeated initialization and deterministic local randomness."""

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        random_state: int | None = None,
        tol: float = 1e-4,
        n_init: int = 5,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.n_init = n_init

    def _validate_parameters(self, n_samples: int) -> None:
        if not isinstance(self.n_clusters, int) or self.n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer")
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed the number of samples")
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(self.n_init, int) or self.n_init < 1:
            raise ValueError("n_init must be a positive integer")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

    @staticmethod
    def compute_squared_distances(
        X: np.ndarray, centers: np.ndarray
    ) -> np.ndarray:
        """Return squared Euclidean distances without allocating a 3-D tensor."""
        distances = (
            np.sum(X * X, axis=1)[:, None]
            + np.sum(centers * centers, axis=1)[None, :]
            - 2 * X @ centers.T
        )
        return np.maximum(distances, 0.0)

    def compute_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Return Euclidean distances for compatibility with the original API."""
        return np.sqrt(self.compute_squared_distances(X, centers))

    def _compute_new_centers(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        squared_distances: np.ndarray,
    ) -> np.ndarray:
        centers = np.empty((self.n_clusters, X.shape[1]), dtype=np.float64)
        nearest_distance = squared_distances[np.arange(len(X)), labels].copy()

        for cluster_id in range(self.n_clusters):
            members = X[labels == cluster_id]
            if len(members):
                centers[cluster_id] = members.mean(axis=0)
            else:
                # Re-seed an empty cluster with the currently worst represented point.
                replacement = int(np.argmax(nearest_distance))
                centers[cluster_id] = X[replacement]
                nearest_distance[replacement] = -np.inf
        return centers

    def compute_new_centers(
        self, X: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Compute centers and safely re-seed empty clusters."""
        X = check_array(X, dtype=np.float64)
        labels = np.asarray(labels)
        if labels.shape != (len(X),):
            raise ValueError("labels must contain one value per sample")
        if np.any((labels < 0) | (labels >= self.n_clusters)):
            raise ValueError("labels contain an invalid cluster index")
        reference = np.zeros((self.n_clusters, X.shape[1]), dtype=np.float64)
        squared_distances = self.compute_squared_distances(X, reference)
        return self._compute_new_centers(X, labels, squared_distances)

    def fit(self, X: np.ndarray, y=None) -> "CustomKMeans":
        """Fit the model and retain the run with the lowest inertia."""
        del y
        X = check_array(X, dtype=np.float64)
        self._validate_parameters(len(X))
        rng = np.random.default_rng(self.random_state)

        best_inertia = np.inf
        best_centers = None
        best_labels = None
        best_iteration = 0

        for _ in range(self.n_init):
            indices = rng.choice(len(X), self.n_clusters, replace=False)
            centers = X[indices].copy()
            previous_labels = None

            for iteration in range(1, self.max_iter + 1):
                squared_distances = self.compute_squared_distances(X, centers)
                labels = np.argmin(squared_distances, axis=1)
                new_centers = self._compute_new_centers(
                    X, labels, squared_distances
                )
                center_shift = float(np.sum((new_centers - centers) ** 2))
                assignments_unchanged = (
                    previous_labels is not None
                    and np.array_equal(labels, previous_labels)
                )
                centers = new_centers
                previous_labels = labels
                if assignments_unchanged or center_shift <= self.tol**2:
                    break

            final_distances = self.compute_squared_distances(X, centers)
            final_labels = np.argmin(final_distances, axis=1)
            inertia = float(final_distances[np.arange(len(X)), final_labels].sum())
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = final_labels.copy()
                best_iteration = iteration

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.cost_ = best_inertia  # Backward-compatible name used by the notebook.
        self.n_iter_ = best_iteration
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign samples to the nearest fitted center."""
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("fit must be called before predict")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data")
        return np.argmin(
            self.compute_squared_distances(X, self.cluster_centers_), axis=1
        )

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit the model and return cluster labels."""
        return self.fit(X, y).labels_


class CustomDBSCAN:
    """DBSCAN using radius-neighbor search and correct density expansion."""

    UNVISITED = -2
    NOISE = -1

    def __init__(
        self, eps: float = 0.5, min_samples: int = 5, metric: str = "euclidean"
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def _validate_parameters(self) -> None:
        if self.eps <= 0:
            raise ValueError("eps must be greater than zero")
        if not isinstance(self.min_samples, int) or self.min_samples < 1:
            raise ValueError("min_samples must be a positive integer")

    def fit(self, X: np.ndarray, y=None) -> "CustomDBSCAN":
        """Cluster samples by recursively expanding density-connected cores."""
        del y
        self._validate_parameters()
        X = check_array(X, dtype=np.float64)
        neighbor_model = NearestNeighbors(radius=self.eps, metric=self.metric)
        neighbor_model.fit(X)
        neighborhoods = neighbor_model.radius_neighbors(X, return_distance=False)
        is_core = np.fromiter(
            (len(neighbors) >= self.min_samples for neighbors in neighborhoods),
            dtype=bool,
            count=len(X),
        )

        labels = np.full(len(X), self.UNVISITED, dtype=int)
        cluster_id = 0
        for point in range(len(X)):
            if labels[point] != self.UNVISITED:
                continue
            if not is_core[point]:
                labels[point] = self.NOISE
                continue

            self._expand_cluster(
                point, cluster_id, labels, neighborhoods, is_core
            )
            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.flatnonzero(is_core)
        self.components_ = X[self.core_sample_indices_].copy()
        self.n_clusters_ = cluster_id
        self.n_features_in_ = X.shape[1]
        return self

    def _expand_cluster(
        self,
        point: int,
        cluster_id: int,
        labels: np.ndarray,
        neighborhoods: np.ndarray,
        is_core: np.ndarray,
    ) -> None:
        labels[point] = cluster_id
        queue = deque(int(index) for index in neighborhoods[point] if index != point)
        queued = set(queue)

        while queue:
            current = queue.popleft()
            if labels[current] == self.NOISE:
                labels[current] = cluster_id
            if labels[current] != self.UNVISITED:
                continue

            labels[current] = cluster_id
            if is_core[current]:
                for neighbor in neighborhoods[current]:
                    neighbor = int(neighbor)
                    if labels[neighbor] == self.UNVISITED and neighbor not in queued:
                        queue.append(neighbor)
                        queued.add(neighbor)

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit the model and return cluster labels."""
        return self.fit(X, y).labels_
