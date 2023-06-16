import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """
    K-Nearest Neighbors classifier.

    Parameters:
    -----------
    k: int
        The number of neighbors to consider.
    
    Methods:
    --------
    fit(X, y)
        Fits the model to the data.
        Returns: None

    manhattan_distance(a, b)
        Calculates the Manhattan distance between two points.
        Returns: float
            The Manhattan distance between the two points.

    euclidean_distance(a, b)
        Calculates the Euclidean distance between two points.
        Returns: float
            The Euclidean distance between the two points.

    predict_example(x)
        Predicts the class of a single example.
        Returns: int (assuming the class labels are integers)
            The predicted class.

    predict(X)
        Predicts the class of a list of examples.
        Returns: array-like
            The predicted classes.

    score(X, y)
        Returns the accuracy of the model.
        Returns: float
            The accuracy of the model.
    """

    def __init__(self, k=3, distance_function='euclidean', weights='uniform'):
        """
        Initializes the KNearestNeighbors object. Sets the number of neighbors to consider.
        """
        self.k = k
        self.X_train = None
        self.y_train = None
        self.distance_function = distance_function
        self.weights = weights

    def manhattan_distance(self, a, b):
        """
        Calculates the Manhattan distance between two points.
        """
        return np.sum(np.abs(a - b))
    
    def euclidean_distance(self, a, b):
        """
        Calculates the Euclidean distance between two points.
        """
        return np.sqrt(np.sum((a - b) ** 2))

    def fit(self, X, y):
        """
        Fits the model to the data.
        """
        self.X_train = X
        self.y_train = y

    def predict_example(self, x):
        """
        Predicts the class of the given example.
        """
        distances = []
        for index, row in self.X_train.iterrows():
            if not np.array_equal(row, x):
                if self.distance_function == 'manhattan':
                    distance = self.manhattan_distance(row, x)
                else:
                    distance = self.euclidean_distance(row, x)
                distances.append((distance, self.y_train[index]))
        distances.sort(key=lambda x: x[0])
        if self.weights == 'uniform':
            votes = [d[1] for d in distances[:self.k]]
        else:
            votes = [d[1] / d[0] for d in distances[:self.k]]
        votes = np.array(votes)
        votes = np.bincount(votes.astype('int'))
        return np.argmax(votes)
        
    def predict(self, X):
        """
        Predicts the class of the given data.
        """
        return [self.predict_example(row) for index, row in X.iterrows()]
    
    def score(self, X, y):
        """
        Returns the accuracy of the model.
        """
        return (self.predict(X) == y).mean()