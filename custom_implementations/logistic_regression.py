import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression classifier.

    Parameters:
    -----------
    lr: float
        The learning rate.
    n_iters: int
        The number of iterations.

    Attributes:
    -----------
    weights: array-like
        The weights after fitting.
    bias: float
        The bias after fitting.

    Methods:
    --------
    sigmoid(x)
        Calculates the sigmoid of the given data.
        Returns: array-like
            The sigmoid of the given data.
        
    cost_function(X, y)
        Calculates the cost function.
        Returns: float
            The cost of the given data.

    gradient(X, y)
        Calculates the gradient.
        Returns: array-like
            The gradient of the given data.

    gradient_descent(X, y)
        Performs gradient descent.
        Returns: None

    fit(X, y, recomputed=False)
        Fits the model to the data.
        Returns: None

    predict_probability(X)
        Predicts the probabilities of the given data.
        Returns: array-like
            The predicted probabilities of the given data.

    predict(X)
        Predicts the classes of the given data.
        Returns: array-like
            The predicted classes of the given data.

    score(X, y)
        Calculates the accuracy of the model.
        Returns: float
            The accuracy of the model.
    """

    def __init__(self, lr=0.01, n_iters=1000):
        """
        Initializes the LogisticRegression object. Sets the learning rate and the number of iterations. Sets the weights and the bias to None.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, x):
        """
        Calculates the sigmoid of the given data. 
        """
        return 1 / (1 + np.exp(-x))
    
    def cost_function(self, X, y):
        """
        Calculates the cost function.
        """
        y_predicted = self.predict_probability(X)
        return (-y * np.log(y_predicted) - (1 - y) * np.log(1 - y_predicted)).mean()
    
    def gradient(self, X, y):
        """
        Calculates the gradient.
        """
        y_predicted = self.predict_probability(X)
        return np.dot(X.T, y_predicted - y) / y.shape[0], (y_predicted - y).mean()
    
    def gradient_descent(self, X, y):
        """
        Performs gradient descent.
        """
        for i in range(self.n_iters):
            prev_cost = self.cost_function(X, y)
            grad_w, grad_b = self.gradient(X, y)
            norm_grad_w = np.linalg.norm(grad_w)
            self.weights -= (self.lr / norm_grad_w) * grad_w
            self.bias -= (self.lr / norm_grad_w) * grad_b
            current_cost = self.cost_function(X, y)
            if abs(current_cost - prev_cost) < 1e-5:
                break
    
    def fit(self, X, y, recomputed=False):
        """
        Fits the model to the data.
        """
        if not recomputed:
            self.weights = np.zeros(X.shape[1])
            self.bias = 0
        self.gradient_descent(X, y)

    def predict_probability(self, X):
        """
        Predicts the probabilities of the given data. 
        """
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        """
        Predicts the class of the given data.
        """
        return self.predict_probability(X).round()

    def score(self, X, y):
        """
        Calculates the accuracy.
        """
        return (self.predict(X) == y).mean()