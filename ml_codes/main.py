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


    def train(self) -> None:
        """train Linear Regression method"""

        pass