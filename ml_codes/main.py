"""Linear Regression module for single variable"""

import numpy as np
from dataclasses import dataclass

@dataclass
class LinearRegression:
    """Linear Regression Class"""

    X: np.array
    y: np.array


    def cost_function(self, y_hat: np.array, y_true: np.array) -> float:
        """cost function for Linear Regression"""

        error = y_hat - y_true
        squared_error = error**2
        mean_squared_error = np.mean(squared_error)/2
        return mean_squared_error


    def train(self, alpha: float, iterations: int) -> None:
        """train Linear Regression method"""

        weights, bias, errors = [], [], []
        # np.random.seed(seed=seed)
        initial_weight, initial_bias = 1, 100 # tuple(np.random.randint(1, 100, 2))

        # Gradient Descent
        # dw = np.mean((y_hat - self.y)*self.X)
        # db = np.mean(y_hat - self.y)

        for i in range(iterations):
            y_hat = initial_weight*self.X + initial_bias
            error = self.cost_function(y_hat=y_hat, y_true=self.y)
            # initial_weight = initial_weight - alpha*dw
            # initial_bias = initial_bias - alpha*db
            # weights.append(initial_weight), bias.append(initial_bias),
            errors.append(error)
        return weights, bias, errors

if __name__ == "__main__":

    X=np.array([1,2,5,100,102]); y=np.array([101,102,105,200,202])
    lr = LinearRegression(X=X, y=y)
    print(lr.train(0.01, 1))        
        