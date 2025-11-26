import numpy as np
from sklearn.utils import check_array
from sklearn.metrics.pairwise import pairwise_distances



class CustomKMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None, tol=1e-4):
        """
        Initializes the CustomKMeans clustering algorithm.

        Parameters:
        - n_clusters (int): Number of clusters to form (default: 8).
        - max_iter (int): Maximum number of iterations for the algorithm (default: 300).
        - random_state (int): Seed for random initialization of cluster centers (default: None).
        - tol (float): Tolerance for cost improvement to determine convergence (default: 1e-4).
        """
        self.n_clusters = n_clusters  # Number of clusters
        self.max_iter = max_iter  # Maximum number of iterations
        self.random_state = random_state  # Random seed for reproducibility
        self.tol = tol  # Tolerance for cost improvement
        self.cluster_centers_ = None  # Stores the final cluster centers
        self.labels_ = None  # Stores the final cluster labels for each data point
        self.cost_ = None  # Stores the final clustering cost (TD2)

    def fit(self, X, y=None):
        """
        Fits the K-Means model to the input data.

        Parameters:
        - X (np.ndarray): Input data of shape (n_samples, n_features).
        - y: Ignored. Present for compatibility with scikit-learn's API.

        Returns:
        - self: The fitted CustomKMeans instance.
        """
        # Validate input data
        X = check_array(X, accept_sparse='csr')

        # Set random seed for reproducibility
        if self.random_state:
            np.random.seed(self.random_state)

        # Step 1: Initialize cluster centers randomly
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices].copy()

        # Initialize previous cost to a large value
        previous_cost = np.inf

        # Iterate until convergence or max_iter is reached
        for iteration in range(self.max_iter):
            # Step 2: Assignment Step - Assign each point to the nearest cluster
            distances = self.compute_distances(X, self.cluster_centers_)
            new_labels = np.argmin(distances, axis=1)

            # Step 3: Compute the current clustering cost (TD2)
            current_cost = np.sum(np.min(distances, axis=1) ** 2)

            # Step 4: Check for convergence (if cost improvement is less than tolerance)
            if np.abs(previous_cost - current_cost) < self.tol:
                print(f"Converged at iteration {iteration}: Cost improvement < tolerance ({self.tol}).")
                break

            # Update previous cost
            previous_cost = current_cost

            # Step 5: Check if cluster assignments have changed
            if np.array_equal(self.labels_, new_labels):
                print(f"Converged at iteration {iteration}: Cluster assignments did not change.")
                break
            self.labels_ = new_labels

            # Step 6: Update Step - Recompute cluster centers
            new_centers = self.compute_new_centers(X, self.labels_)
            if np.allclose(self.cluster_centers_, new_centers):
                print(f"Converged at iteration {iteration}: Cluster centers did not change.")
                break
            self.cluster_centers_ = new_centers.copy()

        # Store the final clustering cost
        self.cost_ = current_cost
        return self

    def fit_predict(self, X, y=None):
        """
        Fits the model to the data and returns the cluster labels.

        Parameters:
        - X (np.ndarray): Input data of shape (n_samples, n_features).
        - y: Ignored. Present for compatibility with scikit-learn's API.

        Returns:
        - labels (np.ndarray): Cluster labels for each data point.
        """
        self.fit(X)
        return self.labels_

    def compute_distances(self, X, centers):
        """
        Computes the Euclidean distance between each data point and each cluster center.

        Parameters:
        - X (np.ndarray): Input data of shape (n_samples, n_features).
        - centers (np.ndarray): Cluster centers of shape (n_clusters, n_features).

        Returns:
        - distances (np.ndarray): Distance matrix of shape (n_samples, n_clusters).
        """
        # Compute the difference between each point and each cluster center
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # Compute the Euclidean distance
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        return distances

    def compute_new_centers(self, X, labels):
        """
        Computes new cluster centers as the mean of all points assigned to each cluster.

        Parameters:
        - X (np.ndarray): Input data of shape (n_samples, n_features).
        - labels (np.ndarray): Cluster labels for each data point of shape (n_samples,).

        Returns:
        - new_centers (np.ndarray): New cluster centers of shape (n_clusters, n_features).
        """
        # Compute the mean of points assigned to each cluster
        new_centers = np.array([np.mean(X[labels == i], axis=0) for i in range(self.n_clusters)])
        return new_centers









class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        """
        Initializes the CustomDBSCAN clustering algorithm.

        Parameters:
        - eps (float): The maximum distance between two samples for them to be considered neighbors.
        - min_samples (int): The minimum number of points required to form a dense region (core point).
        - metric (str or callable): The distance metric to use for computing distances between points.
                                    Default is 'euclidean'. Supports any metric supported by `pairwise_distances`.
        """
        self.eps = eps  # Radius of the neighborhood
        self.min_samples = min_samples  # Minimum number of points in a neighborhood
        self.metric = metric  # Distance metric (e.g., 'euclidean', 'manhattan', 'cosine', or custom)
        self.labels_ = None  # Will store the cluster labels for each point

    def fit(self, X: np.ndarray, y=None):
        """
        Performs DBSCAN clustering on the input data X.

        Parameters:
        - X (np.ndarray): Input data of shape (n_samples, n_features).
        - y: Ignored. Present for compatibility with scikit-learn's API.

        Returns:
        - self: The fitted CustomDBSCAN instance.
        """
        # Validate input data
        X = check_array(X, accept_sparse='csr')

        ############################################################
        # **ALLOWS MULTIPLE DISTANCE METRICS HERE**
        # Compute pairwise distances between all points using the specified metric
        distances = pairwise_distances(X, metric=self.metric)
        ############################################################

        # Initialize labels: -1 means noise (unclassified)
        labels = -1 * np.ones(X.shape[0], dtype=int)

        # Initialize cluster ID counter
        cluster_id = 0

        # Iterate through each point in the dataset
        for idx in range(X.shape[0]):
            # Skip points that are already assigned to a cluster
            if labels[idx] != -1:
                continue

            # Find all neighbors within eps distance
            neighbors = np.where(distances[idx] <= self.eps)[0]

            # If the point has fewer than min_samples neighbors, mark it as noise
            if len(neighbors) < self.min_samples:
                labels[idx] = -1
            else:
                # Start a new cluster and expand it
                self._expand_cluster(idx, neighbors, labels, cluster_id, distances)
                cluster_id += 1  # Increment cluster ID for the next cluster

        # Store the final labels
        self.labels_ = labels
        return self

    def _expand_cluster(self, idx, neighbors, labels, cluster_id, distances):
        """
        Expands a cluster by adding density-reachable points.

        Parameters:
        - idx (int): Index of the core point.
        - neighbors (np.ndarray): Indices of neighboring points.
        - labels (np.ndarray): Current cluster labels for all points.
        - cluster_id (int): ID of the current cluster.
        - distances (np.ndarray): Precomputed pairwise distances between points.
        """
        # Assign the core point to the current cluster
        labels[idx] = cluster_id

        # Use a queue to process all density-reachable points
        queue = list(neighbors)

        while queue:
            # Get the next point from the queue
            current_idx = queue.pop(0)

            # If the point is noise, mark it as a border point of the current cluster
            if labels[current_idx] == -1:
                labels[current_idx] = cluster_id

            # Skip points that are already assigned to a cluster
            if labels[current_idx] != -1:
                continue

            # Assign the point to the current cluster
            labels[current_idx] = cluster_id

            ############################################################
            # **USES THE SAME DISTANCE METRIC HERE**
            # Find neighbors of the current point using the precomputed distances
            current_neighbors = np.where(distances[current_idx] <= self.eps)[0]
            ############################################################

            # If the current point is a core point, add its neighbors to the queue
            if len(current_neighbors) >= self.min_samples:
                queue.extend([n for n in current_neighbors if labels[n] == -1])

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Performs clustering on the input data X and returns the cluster labels.

        Parameters:
        - X (np.ndarray): Input data of shape (n_samples, n_features).
        - y: Ignored. Present for compatibility with scikit-learn's API.

        Returns:
        - labels (np.ndarray): Cluster labels for each point.
        """
        self.fit(X)
        return self.labels_