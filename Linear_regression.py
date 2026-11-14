import numpy as np 

class LinearRegression:
    def __init__(self):
        self._W = None

    def fit(self, X, y):
        '''
            X: (N, D)
            y: (N, )
        '''
        _, D = X.shape
        self._W = np.zeros((D, ))
        cov = X.T @ X # (D, D)
        inv_cov = np.linalg.inv(cov) # (D, D)
        self._W = inv_cov @ X.T @ y # (D, )

    def predict(self, X):
        '''
            X: (N, D)
        '''
        y_hat = X @ self._W.reshape(-1, 1) # (N, )

        return y_hat

    def evaluate(self, y, y_hat):
        rmse = ((y - y_hat)**2).mean()**0.5
        return rmse