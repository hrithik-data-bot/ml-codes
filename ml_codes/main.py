"""Linear Regression module for single variable"""

import numpy as np
from dataclasses import dataclass

@dataclass
class LinearRegression:
    """Linear Regression Class"""

    X: np.array
    y: np.array

    def cost_function(self, y_hat: np.array, y_true: np.array = y):
        """cost function for Linear Regression"""
        pass