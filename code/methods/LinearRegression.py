import numpy as np



class LinearRegression(object):
    """docstring for LinearRegression"""
    def __init__(self, method='OLS', learning_rate=0.01, epochs=100):
        super(LinearRegression, self).__init__()
        self.method = method
    
    def fit(X, y):
        N, D = X.shape
        X_ = np.hstack((np.ones((N,1)),X))

        if self.method in ['GD','SGD']:
            self.w = np.random.randn(D+1,1)
            for j in range(epochs):
                if self.method == 'GD':
                    error = y - X_ @ self.w.T
                    self.w = self.w + self.learning_rate * np.mean(error * X_, axis=0)
                else:
                    np.random.shuffle(X_)
                    for i in range(N):
                        error_i = y[[i]] - self.w.T @ X_[[i]].T
                        self.w = self.w + self.learning_rate * error_i * X_[[i]].T
        else:
            self.w = np.linalg.inv(X_.T @ X_) @ X_.T @ y
        pass
    def predict(X):
        N, _ = X.shape
        X_ = np.hstack((np.ones((N,1)),X))
        return X_ @ self.w.T