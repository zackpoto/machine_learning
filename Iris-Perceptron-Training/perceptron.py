import numpy as np

class Perceptron():
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta                      #: float - Learning rate (between 0.0 and 1.0)
        self.n_iter = n_iter                #: int - Passes over the training dataset.
        self.random_state = random_state    #: int - Random number generator seed for random weight initialization.

    def fit(self, X, y):
        """Fit training data.

        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        returns -> self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1]) #: 1D-array - Weights after fitting.
        self.errors_ = []                                               #: list - Number of misclassifications (updates) in each epoch.

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
