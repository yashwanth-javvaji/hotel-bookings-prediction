import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class DecisionTree(BaseEstimator, ClassifierMixin):
    """
    Decision Tree Classifier.

    Parameters:
    -----------
    max_depth: int
        The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves
        contain less than min_samples_split samples.

    Attributes:
    -----------
    tree: dict
        The tree structure. It has the following form:
            tree = {
                (attribute_name, attribute_value, True): subtree,
                (attribute_name, attribute_value, False): subtree,
                ...
            }
            ...
        }

    Methods:
    --------
    partition(x):
        Partitions the data into the classes.
        Returns: dict
            The partitions of the data into the classes.

    entropy(y):
        Calculates the entropy of the data set.
        Returns: float
            The entropy of the data set.

    mutual_information(x, y, feature, value):
        Calculates the mutual information of the data set.
        Returns: float
            The mutual information of the data set.

    id3(x, y, attribute_value_pairs=None, depth=0):
        Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
        attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
            1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
            2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
            value of y (majority label)
            3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
        Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
        and partitions the data set based on the values of that attribute before the next recursive call to ID3.
        Returns: dict
            The ID3 decision tree.

    fit(X, y):
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

    def __init__(self, max_depth=None):
        """
        Initializes the DecisionTree object. Sets the maximum depth of the tree. Sets the tree to None.
        """
        self.max_depth = max_depth
        self.tree = {}

    def partition(self, x):
        """
        Partitions the data set by class.
        """
        partitions = {}
        for label in np.unique(x):
            partitions[label] = np.where(x == label)[0]
        return partitions

    def entropy(self, y):
        """
        Calculates the entropy of the data set.
        """
        partitions = self.partition(y)
        entropy = 0
        for partition in partitions.values():
            probability = partition.shape[0] / y.shape[0]
            entropy -= probability * np.log2(probability)
        return entropy
    
    def weighted_entropy(self, y, weights):
        """
        Calculates the weighted entropy of the data set.
        """
        # TODO
        pass

    def mutual_information(self, x, y, feature, value):
        """
        Calculates the mutual information of the data set.
        """
        partitions = self.partition(x[feature] == value)
        mutual_information = self.entropy(y)
        mutual_information -= sum(
            [
                (partition.shape[0] / y.shape[0]) * self.entropy(y.iloc[partition])
                for partition in partitions.values()
            ]
        )
        return mutual_information
    
    def weighted_mutual_information(self, x, y, feature, value, weights):
        """
        Calculates the weighted mutual information of the data set.
        """
        # TODO
        pass

    def id3(self, x, y, attribute_value_pairs=None, depth=0):
        """
        Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
        attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
            1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
            2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
            value of y (majority label)
            3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
        Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
        and partitions the data set based on the values of that attribute before the next recursive call to ID3.

        The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
        to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
        (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
        attributes with their corresponding values:
        [(x1, a),
        (x1, b),
        (x1, c),
        (x2, d),
        (x2, e)]
        If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
        the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

        The tree is stored as a nested dictionary, where each entry is of the form
                        (attribute_index, attribute_value, True/False): subtree
        * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
        indicates that we test if (x4 == 2) at the current node.
        * The subtree itself can be nested dictionary, or a single label (leaf node).
        * Leaf nodes are (majority) class labels

        Returns a decision tree represented as a nested dictionary, for example
        {(4, 1, False):
            {(0, 1, False):
                {(1, 1, False): 1,
                (1, 1, True): 0},
            (0, 1, True):
                {(1, 1, False): 0,
                (1, 1, True): 1}},
        (4, 1, True): 1}
        """
        # If attribute_value_pairs is None, then this is the first call to ID3. Initialize it.
        if attribute_value_pairs is None:
            attribute_value_pairs = []
            for feature in x.columns:
                for value in np.unique(x[feature]):
                    attribute_value_pairs.append((feature, value))

        # Terminate recursion
        label_counts = dict(zip(*np.unique(y, return_counts=True)))

        # If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        if len(label_counts) == 1:
            return list(label_counts.keys())[0]

        # If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
        # value of y (majority label)
        if len(attribute_value_pairs) == 0:
            return max(label_counts, key=label_counts.get)

        # If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
        if self.max_depth is not None and depth == self.max_depth:
            return max(label_counts, key=label_counts.get)

        # Select the next best attribute-value pair using MUTUAL INFORMATION as the splitting criterion
        # and partitions the data set based on the values of that attribute before the next recursive call to ID3.
        mutual_information_values = []
        for (attribute_name, attribute_value) in attribute_value_pairs:
            mutual_information_values.append(
                self.mutual_information(x, y, attribute_name, attribute_value))

        # Get the best attribute-value pair
        best_attribute_value_pair_index = np.argmax(mutual_information_values)

        # If the best attribute-value pair is pure (all y = only 0 or only 1), then return that label
        if mutual_information_values[best_attribute_value_pair_index] == 0:
            return max(label_counts, key=label_counts.get)

        # Otherwise split on the best attribute-value pair
        best_attribute_name, best_attribute_value = attribute_value_pairs[
            best_attribute_value_pair_index]

        # Remove the best attribute-value pair from the list of attribute_value_pairs
        attribute_value_pairs.pop(best_attribute_value_pair_index)

        # Partition the data set based on the values of the best attribute and call ID3 on each partitioned data set
        return {
            (best_attribute_name, best_attribute_value, True):
                self.id3(
                    x[x[best_attribute_name] == best_attribute_value],
                    y[x[best_attribute_name] == best_attribute_value],
                    attribute_value_pairs,
                    depth + 1
                ),
            (best_attribute_name, best_attribute_value, False):
                self.id3(
                    x[x[best_attribute_name] != best_attribute_value],
                    y[x[best_attribute_name] != best_attribute_value],
                    attribute_value_pairs,
                    depth + 1
                )
        }

    def fit(self, X, y):
        """
        Fits the model to the data.
        """
        self.tree = self.id3(X, y)

    def predict_example(self, x, tree=None):
        """
        Predicts the class of the given example.
        """
        if tree is None:
            tree = self.tree

        # If the tree is a leaf node, return the label
        if not isinstance(tree, dict):
            return tree

        # Otherwise, split on the attribute-value pair
        for (attribute_name, attribute_value, value), subtree in tree.items():
            if x[attribute_name] == attribute_value and value:
                return self.predict_example(x, subtree)
            elif x[attribute_name] != attribute_value and not value:
                return self.predict_example(x, subtree)

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