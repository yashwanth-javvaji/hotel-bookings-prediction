import numpy as np
from custom_implementations.decision_tree import DecisionTree
from sklearn.base import BaseEstimator, ClassifierMixin


class BoostingDecisionTree(BaseEstimator, ClassifierMixin):
    """
    Boosting class based on decision tree.

    Parameters:
    -----------
    max_depth: int
        The maximum depth of the tree.

    n_estimators: int
        The number of trees in the forest.

    learning_rate: float
        The learning rate of the algorithm.

    Attributes:
    -----------
    estimators: list
        The list of trees in the forest.

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
    def __init__(self, max_depth=None, n_estimators=10, learning_rate=0.1):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []

    def fit(self, X, y):
        self.estimators = []
        weights = np.ones(len(X)) / len(X)
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, weights=weights)
            tree.fit(X, y)
            y_pred = tree.predict(X)
            error = weights[y_pred != y].sum() / weights.sum()
            alpha = np.log((1 - error) / error) / 2
            weights *= np.exp(-alpha * y * y_pred)
            weights /= weights.sum()
            self.estimators.append((tree, alpha))

    def predict_example(self, x):
        """
        Predicts the class of the given example.
        """
        predictions = {}
        for tree, alpha in self.estimators:
            prediction = tree.predict_example(x)
            predictions[prediction] = predictions.get(prediction, 0) + alpha
        return max(predictions, key=predictions.get)
    
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
