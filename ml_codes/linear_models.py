"""Linear Regression module for single variable"""


from dataclasses import dataclass
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


    def train(self) -> None:
        """train Linear Regression method"""

        errors, weight, bias = [], [], []
        for e, _ in enumerate(range(self.iterations), start=1):
            y_pred = self.initial_weight*self.X + self.initial_bias
            error = y_pred - self.y
            squared_error = error**2
            mean_squared_error = np.mean(squared_error)/2
            dw = np.mean(error*self.X)
            db = np.mean(error)
            self.initial_weight = self.initial_weight - self.alpha*dw
            self.initial_bias = self.initial_bias - self.alpha*db
            errors.append(mean_squared_error); weight.append(self.initial_weight); bias.append(self.initial_bias)
            print(f"Loss at iteration:- {e}; Slope:- {self.initial_weight}; Intercept:- {self.initial_bias}")
        min_error_idx = errors.index(min(errors))
        self.initial_weight, self.initial_bias = weight[min_error_idx], bias[min_error_idx]
        return np.array(errors), np.array(weight), np.array(bias)


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

        predictions = self._coefficient*X + self._intercept
        return predictions
