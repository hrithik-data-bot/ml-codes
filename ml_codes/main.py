"""Linear Regression module for single variable"""


from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np

@dataclass
class LinearRegression:
    """Linear Regression Class"""

    X: np.array
    y: np.array
    alpha: float
    iterations: int


    def train(self) -> None:
        """train Linear Regression method"""

        errors, weight, bias = [], [], []
        initial_w, initial_b = 1, 100
        for _ in range(self.iterations):
            y_pred = initial_w*self.X + initial_b
            error = y_pred - self.y
            squared_error = error**2
            mean_squared_error = np.mean(squared_error)/2
            dw = np.mean(error*self.X)
            db = np.mean(error)
            initial_w = initial_w - self.alpha*dw
            initial_b = initial_b - self.alpha*db
            errors.append(mean_squared_error); weight.append(initial_w); bias.append(initial_b)
        return np.array(errors), np.array(weight), np.array(bias)


    @property
    def _coefficient(self) -> float:
        """returns coefficient of model"""

        errors, slope, _ = self.train()
        min_error_idx = np.argmin(errors)
        return slope[min_error_idx]


    @property
    def _intercept(self) -> float:
        """returns intercept of model"""

        errors, _, intercept = self.train()
        min_error_idx = np.argmin(errors)
        return intercept[min_error_idx]
        

    def predict(self, X: np.array) -> np.array:
        """predict method for Linear Regression"""

        predictions = self._coefficient*X + self._intercept
        return predictions


if __name__ == "__main__":

    X = np.array([1, 2, 5, 100, 102])
    y = np.array([100, 120, 100, 105, 111])
    lr = LinearRegression(X, y, alpha=0.001, iterations=100)
    print(lr._coefficient)
    print(lr._intercept)
    print(lr.predict(X))