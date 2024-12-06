import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        """
        Initialize KMeans clustering.
        
        :param n_clusters: Number of clusters
        :param max_iter: Maximum number of iterations
        :param tol: Tolerance for convergence
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, X):
        """
        Fit the KMeans model to the data.
        
        :param X: Data array of shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape

        # Randomly initialize centroids
        rng = np.random.default_rng()
        self.centroids = X[rng.choice(n_samples, self.n_clusters, replace=False)]

        for i in range(self.max_iter):
            # Assign clusters based on closest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            # Calculate new centroids
            new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])

            # Check for convergence
            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) < self.tol):
                break
            self.centroids = new_centroids

    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.
        
        :param X: Data array of shape (n_samples, n_features)
        :return: Cluster labels for each sample
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        """
        Fit the model and return cluster assignments.
        
        :param X: Data array of shape (n_samples, n_features)
        :return: Cluster labels for each sample
        """
        self.fit(X)
        return self.labels

