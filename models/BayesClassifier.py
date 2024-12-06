import numpy as np 

class BayesClassifier:
    __epsilon = 1e-8
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.prior = {}

    def fit(self, X, y):
        """
        Fit the classifier to the training data.
        :param X: ndarray of shape (n_samples, n_features) - Training data
        :param y: ndarray of shape (n_samples,) - Class labels
        """
        self.classes = np.unique(y)
        for cls in self.classes:
            X_cls = X[y == cls]
            self.mean[cls] = np.mean(X_cls, axis=0)
            self.var[cls] = np.var(X_cls, axis=0)+self.__epsilon
            self.prior[cls] = (X_cls.shape[0]+1) / (X.shape[0]+len(self.classes))

    def _gaussian_density(self, x, mean, var):
        """
        Compute the Gaussian probability density function.
        :param x: Value to compute density for
        :param mean: Mean of the Gaussian distribution
        :param var: Variance of the Gaussian distribution
        :return: Probability density
        """
        coeff = 1 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-(x - mean) ** 2 / (2 * var))
        res = coeff * exponent
        res[res <= 10**-320] = 10**-320
        return res #max(coeff * exponent,10**-320)

    def _predict_class(self, x):
        """
        Predict the class for a single sample.
        :param x: ndarray of shape (n_features,) - Single sample
        :return: Predicted class
        """
        posteriors = {}
        for cls in self.classes:
            prior = np.log(self.prior[cls])
            likelihood = np.sum(np.log(self._gaussian_density(x, self.mean[cls], self.var[cls])))
            posteriors[cls] = prior + likelihood
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        """
        Predict class labels for the input data.
        :param X: ndarray of shape (n_samples, n_features) - Test data
        :return: ndarray of shape (n_samples,) - Predicted class labels
        """
        return np.array([self._predict_class(x) for x in X])
