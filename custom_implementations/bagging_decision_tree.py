import numpy as np
from custom_implementations.decision_tree import DecisionTree
from sklearn.base import BaseEstimator, ClassifierMixin


class BaggingDecisionTree(BaseEstimator, ClassifierMixin):
    """
    Bagging class based on decision trees.

    Parameters:
    -----------
    max_depth: int
        The maximum depth of the tree.
    n_estimators: int
        The number of trees in the bagging.

    Attributes:
    -----------
    estimators: list
        The list of trees in the bagging.
    
    Methods:
    --------
    fit:
        Fits the model to the data.
        Returns: None

    predict_example(x, tree=None):
        Predicts the class of the given example.
        Returns: int (assuming the class labels are integers)
            The predicted class of the example.

    predict(X):
        Predicts the class of the given data.
        Returns: array-like
            The predicted classes of the data.

    score(X, y)
        Returns the accuracy of the model on the given data.
        Returns: float
            The accuracy of the model on the given data.
    """

    def __init__(self, max_depth=None, n_estimators=10):
        """
        Initializes the Bagging object. Sets the maximum depth of the trees and the number of trees. Sets the estimators to an empty list.
        """
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.estimators = []

    def resample(self, X, y):
        """
        Resamples the data.
        """
        n = len(X)
        indices = np.random.choice(n, n, replace=True)
        X_subset = X.iloc[indices]
        y_subset = y.iloc[indices]
        return X_subset, y_subset

    def fit(self, X, y):
        """
        Fits the model to the data.
        """
        for _ in range(self.n_estimators):
            X_subset, y_subset = self.resample(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_subset, y_subset)
            self.estimators.append(tree)

    def predict_example(self, x):
        """
        Predicts the class of the given example.
        """
        predictions = [estimator.predict_example(
            x) for estimator in self.estimators]
        return np.bincount(predictions).argmax()

    def predict(self, X):
        """
        Predicts the class of the given data.
        """
        return [self.predict_example(row) for index, row in X.iterrows()]

    def score(self, X, y):
        """
        Returns the mean accuracy of the predictions.
        """
        return (self.predict(X) == y).mean()