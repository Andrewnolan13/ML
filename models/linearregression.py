import numpy as np

class LinearRegression:
    def __init__(self,X:np.ndarray,y:np.ndarray):
        '''
        Parameters:
            X:np.ndarray
            y:np.ndarray
        Methods:
            fit -> None
            predict -> np.ndarray
        '''
        self.X = X
        self.y = y
    def fit(self)->None:
        self.beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
    def predict(self,X:np.ndarray)->np.ndarray:
        return X @ self.beta
    