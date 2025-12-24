"""Linear Regression module for single variable"""

from typing import Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class SingleLinearRegression:
    """Single Linear Regression Class"""

    X: np.array
    y: np.array
    alpha: float
    iterations: int
    initial_weight: float
    initial_bias: float


    def train(self) -> Dict:
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
        return {'MSE': np.array(errors), 'Weights':np.array(weight), 'Bias':np.array(bias)}


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


@dataclass
class MultipleLinearRegression:
    
    X: np.array
    y: np.array
    alpha: float
    iterations: int
    initial_weight: np.array
    initial_bias: float

    def train(self) -> Dict:
        """train multiple linear regression example"""
    
        errors, weight, bias = [], [], []
        for e, _ in enumerate(range(self.iterations), start=1):
            y_pred = np.array([np.dot(self.initial_weight, row) + self.initial_bias for row in self.X])
            error = y_pred - self.y
            squared_error = error**2
            mean_squared_error = np.mean(squared_error)/2
            for _ in self.initial_weight:
                dw = np.mean(error*self.X)
                self.initial_weight = self.initial_weight - self.alpha*dw
            db = np.mean(error)
            self.initial_bias = self.initial_bias - self.alpha*db
            errors.append(mean_squared_error); weight.append(self.initial_weight); bias.append(self.initial_bias)
            print(f"Loss at iteration:- {e}; Slope:- {self.initial_weight}; Intercept:- {self.initial_bias}")
        min_error_idx = errors.index(min(errors))
        self.initial_weight, self.initial_bias = weight[min_error_idx], bias[min_error_idx]
        return {'MSE': np.array(errors), 'Weights':np.array(weight), 'Bias':np.array(bias)}
    
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
                            
            


if __name__ == "__main__":
    X = np.array([[1,2,1],[2,1,2],[1,3,2]])
    y = np.array([5,6,12])
    model = MultipleLinearRegression(X=X, y=y, alpha=0.001, iterations=1000, initial_weight=np.array([1, 2.5, 1.5]), initial_bias=4)
    print(model.train())
    print(model._coefficient)
    print(model._intercept)
            
            