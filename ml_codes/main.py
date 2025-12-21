"""Linear Regression module for single variable"""

import numpy as np
from dataclasses import dataclass

@dataclass
class LinearRegression:
    """Linear Regression Class"""

    X: np.array
    y: np.array


    def cost_function(self, y_hat: np.array, y_true: np.array = y) -> float:
        """cost function for Linear Regression"""

        error = y_hat - y_true
        squared_error = error**2
        mean_squared_error = np.mean(squared_error)/2
        return mean_squared_error


    def train(self, seed: int, learning_rate: float) -> None:
        """train Linear Regression method"""

        np.random.seed(seed=seed)
        initial_weight, initial_bias = tuple(np.random.randint(1, 100, 2))
        y_hat = initial_weight*self.X + initial_bias
        error = self.cost_function(y_hat=y_hat)
        