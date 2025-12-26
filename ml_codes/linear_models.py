"""Linear Regression module for single variable"""

from typing import Dict
from dataclasses import dataclass, field
import numpy as np

@dataclass
class LinearRegression:
    """Linear Regression Class"""

    X: np.array
    y: np.array
    alpha: float
    iterations: int
    initial_weight: float
    initial_bias: float
    model_summary = None
    is_multiple: bool = field(init=False)


    def __post_init__(self):
        self.is_multiple = self.X.ndim > 1 and self.X.shape[1] != 1


    def train(self) -> str:
        """train Linear Regression method"""

        errors, weight, bias = [], [], []
        for e, _ in enumerate(range(self.iterations), start=1):
            if self.is_multiple:
                y_pred = np.array([np.dot(self.initial_weight, row) + self.initial_bias for row in self.X])
            else:
                y_pred = self.initial_weight * self.X + self.initial_bias
            error = y_pred - self.y
            squared_error = error ** 2
            mean_squared_error = np.mean(squared_error) / 2
            if self.is_multiple:
                for _ in self.initial_weight:
                    dw = np.mean(error * self.X)
                    self.initial_weight = self.initial_weight - self.alpha * dw
            else:
                dw = np.mean(error * self.X)
                self.initial_weight = self.initial_weight - self.alpha * dw
            db = np.mean(error)
            self.initial_bias = self.initial_bias - self.alpha * db
            errors.append(mean_squared_error)
            weight.append(self.initial_weight)
            bias.append(self.initial_bias)
        self.model_summary = {"MSE": np.array(errors), "Weights": np.array(weight), "Bias": np.array(bias)}
        min_error_idx = errors.index(min(errors))
        self.initial_weight, self.initial_bias = (weight[min_error_idx], bias[min_error_idx],)
        return f"Model Trained"


    @property
    def _coefficient(self) -> float:
        """returns coefficient of model"""

        return self.initial_weight


    @property
    def _intercept(self) -> float:
        """returns intercept of model"""

        return self.initial_bias


    def predict(self, X: np.array) -> np.array:
        """predict method for Linear Regression"""

        if self.is_multiple:
            predictions = np.array([np.dot(self.initial_weight, row)+self.initial_bias for row in self.X])
        else:
            predictions = self._coefficient * X + self._intercept
        return predictions
