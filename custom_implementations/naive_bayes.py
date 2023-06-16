import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class NaiveBayes(BaseEstimator, ClassifierMixin):
    """
    Naive Bayes classifier.

    Parameters:
    -----------
    alpha: float
        The additive smoothing parameter.

    Attributes:
    -----------
    classes: array-like
        The unique class labels.
    n_classes: int
        The number of classes.
    n_features: int
        The number of features.
    prior_probability: array-like
        The prior probabilities for each class.
    conditional_probability: array-like
        The conditional probabilities for each feature given each class.

    Methods:
    --------
    partition(X, y)
        Partitions the data into the classes.
        Returns: dict
            The partitions of the data into the classes.

    fit(X, y)
        Fits the model to the data.
        Returns: None

    predict_example(x)
        Predicts the class of the given example.
        Returns: int (assuming the class labels are integers)
            The predicted class of the example.

    predict(X)
        Predicts the class of the given data.
        Returns: array-like
            The predicted classes of the data.

    score(X, y)
        Returns the accuracy of the model on the given data.
        Returns: float
            The accuracy of the model on the given data.
    """

    def __init__(self, alpha=1.0):
        """
        Initializes the NaiveBayes object. Sets the additive smoothing parameter. Sets the attributes to None.
        """
        self.alpha = alpha
        self.classes = None
        self.n_classes = None
        self.n_features = None
        self.prior_probability = {}
        self.conditional_probability = {}
    
    def partition(self, X, y):
        """
        Partitions the data into the classes.
        """
        partitions = {}
        unique_classes = np.unique(y)
        for unique_class in unique_classes:
            partitions[unique_class] = X[y == unique_class]
        return partitions

    def fit(self, X, y):
        """
        Fits the model to the data.
        """
        partitions = self.partition(X, y)
        self.classes = list(partitions.keys())
        self.n_classes = len(self.classes)
        self.features = X.columns
        self.n_features = len(self.features)
        for class_label in self.classes:
            self.conditional_probability[class_label] = {}
            subset = partitions[class_label]
            self.prior_probability[class_label] = subset.shape[0] / X.shape[0]
            feature_counts = subset.sum(axis=0)
            for feature, count in feature_counts.items():
                self.conditional_probability[class_label][feature] = (count + self.alpha) / (subset.shape[0] + self.alpha * self.n_features)
            self.conditional_probability[class_label]["__unseen__"] = self.alpha / (subset.shape[0] + self.alpha * self.n_features)

    def predict_example(self, x):
        """
        Predicts the class of the given example.
        """
        posterior_log_likelihoods = {}
        for class_label in self.prior_probability:
            posterior_log_likelihoods[class_label] = np.log(self.prior_probability[class_label])
            for feature, value in x.items():
                if value == 0:
                    posterior_log_likelihoods[class_label] += np.log(self.conditional_probability[class_label]["__unseen__"])
                else:
                    posterior_log_likelihoods[class_label] += np.log(self.conditional_probability[class_label][feature])
        return max(posterior_log_likelihoods, key=posterior_log_likelihoods.get)
    
    def predict(self, X):
        """
        Predicts the class of the given data.
        """
        predictions = []
        for index, row in X.iterrows():
            row = {key: value for key, value in zip(self.features, row)}
            predictions.append(self.predict_example(row))
        return np.array(predictions)
    
    def score(self, X, y):
        """
        Returns the accuracy of the model on the given data.
        """
        return (self.predict(X) == y).mean()