import numpy as np

class LinearRegression(object):
    """docstring for LinearRegression"""
    def __init__(self, method='OLS', learning_rate=0.01, epochs=100):
        self.method = method
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X, y):
        N, D = X.shape
        X_ = np.hstack((np.ones((N,1)),X))

        if self.method in ['GD','SGD']:
            self.w = np.random.randn(1, D+1)
            self.error_log = list()
            for j in range(self.epochs):
                error = y - X_ @ self.w.T
                self.error_log.append((error**2).mean())
                if self.method == 'GD':
                    self.w += self.learning_rate * (error * X_).mean(axis=0)
                else:
                    per = np.random.permutation(N)
                    X_per = X_[per]
                    y_per = y[per]
                    for i in range(N):
                        error_i = y_per[[i]] - X_per[[i]] @ self.w.T
                        self.w += self.learning_rate * error_i * X_per[[i]]
        else:
            self.w = y.T @ np.linalg.pinv(X_).T
        
    def predict(self, X):
        N, _ = X.shape
        X_ = np.hstack((np.ones((N,1)),X))
        return X_ @ self.w.T