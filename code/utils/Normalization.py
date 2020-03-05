import numpy as np

class Normalization(object):
    """docstring for LinearRegression"""
    def __init__(self, method='std'):
        self.method = method
    
    def fit(self, X):
        if self.method == 'std':
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0) 
        else:
            self.X_max =  X.max(axis=0)
            self.X_min =  X.min(axis=0)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def detransform(self, X):
        if self.method == 'std':
            return (X * self.X_std) + self.X_mean
        else:
            return (X * (self.X_max - self.X_min)) + self.X_min

    def transform(self, X):
        if self.method == 'std':
            return (X - self.X_mean) / self.X_std
        else:
            return (X - self.X_min) / (self.X_max - self.X_min)