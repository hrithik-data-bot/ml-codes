"""all ML metrics"""

import numpy as np


def mean_absolute_error(y_pred: np.array, y_true: np.array) -> float:
    """mean absolute error"""

    return np.mean(y_pred - y_true)
 

def mean_squared_error(y_pred:, y_true: ) -> float:
    """regression mse"""
    