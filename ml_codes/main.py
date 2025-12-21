"""Linear Regression module for single variable"""

import numpy as np
from dataclasses import dataclass

@dataclass
class LinearRegression:
    """Linear Regression Class"""

    X: np.array
    y: np.array


    def train(self, alpha: float, iterations: int) -> None:
        """train Linear Regression method"""

        errors, weight, bias = [], [], []
        initial_w, initial_b = 1, 100
        for _ in range(iterations):
            y_pred = initial_w*self.X + initial_b
            error = y_pred - self.y
            squared_error = error**2
            mean_squared_error = np.mean(squared_error)/2
            dw = np.mean(error*self.X)
            db = np.mean(error)
            initial_w = initial_w - alpha*dw
            initial_b = initial_b - alpha*db
            errors.append(mean_squared_error); weight.append(initial_w); bias.append(initial_b)
        return errors, weight, bias


if __name__ == "__main__":

    X=np.array([1,2,5,100,102]); y=np.array([100,120,100,105,111])
    lr = LinearRegression(X=X, y=y)
    print(lr.train(0.0005, 100))        
        