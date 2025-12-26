"""all ML metrics"""

import numpy as np


def mean_absolute_error(y_pred: np.array, y_true: np.array) -> float:
    """mean absolute error for regression"""

    return np.mean(y_pred - y_true)
 

def mean_squared_error(y_pred: np.array, y_true: np.array) -> float:
    """mean squared error for regression"""

    return np.mean((y_pred - y_true)**2)


def root_mean_squared_error(y_pred: np.array, y_true: np.array) -> float:
    """root mean squared error for regression"""

    return np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
