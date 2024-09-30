import numpy as np
from collections import Counter

CLASSIFICATION = 'c'
REGRESSION = 'r'

class KNearestNeighbours:
    """K nearest Neighbours Regression/Classification model"""
    def __init__(self, k, classification:bool = None, regression:bool = None):
        self.k = k
        if classification and regression:
            raise ValueError("Specify at most one of classification or regression")
        self.type = CLASSIFICATION if classification else REGRESSION if  regression else None
    def fit(self,X:np.ndarray,y:np.ndarray)->None:
        """
        expects X to be of shape (n,m) ie n feature vectors of dimension m.
        expects y to be of shape (n,p) ie n target vectors of dimension p.

        if receive y of shape (n,), it will reshape to (n,1)
        """
        self.X = X
        self.y = self._reshape_y(y)
        self.type = self.type or self._infer_type()
        self._validate_dimensions()
    
    def _reshape_y(self,y:np.ndarray)->np.ndarray:
        '''
        reshape y to expected shape
        '''
        if not isinstance(y,np.ndarray):
            raise ValueError("y must be a numpy array.")
        elif len(y.shape) >2:
            raise ValueError("y must be a numpy array with shape (n,p) where p is the dimension of the targets. If y is of shape (n,), it will be reshaped to (n,1)")
        elif len(y.shape) == 1:
            return y.reshape((*y.shape,1))
        else:
            raise ValueError("y.shape not as expected, got {}, expected (n,p,1) or (n,1) where n is number of observations an p is the dimension of the target vectors".format(y.shape))
        
    def _infer_type(self)->str:
        if self.y.dtype.kind in ['U','O']:
            return CLASSIFICATION
        elif self.y.dtype.kind in ['i','f']:
            return REGRESSION
        else:
            raise ValueError("Could not infer type (classification/regression) from y.")
    
    def _validate_dimensions(self)->None:
        """
        Ensures X and y have same number of observations.
        Ensures X is of numeric type.
        Ensures y is of numeric type if regression. 
        """
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have same number of observations.")
        if self.X.dtype.kind not in ['i','f']:
            raise ValueError("X must be of numeric type.")
        if self.type == REGRESSION and self.y.dtype.kind not in ['i','f']:
            raise ValueError("y must be of numeric type for regression.")
    
    def predict(self,X:np.ndarray)->np.ndarray:
        distances = np.array([np.linalg.norm(x - self.X,axis=1)for x in X])
        indicesOfNeighbours = [np.argsort(distance)[:self.k] for distance in distances]
        neighbours = np.array([self.y[indices] for indices in indicesOfNeighbours])

        if self.type == REGRESSION:
            predictions = neighbours.mean(axis = 1)
        else:
            predictions = np.array([self._get_mode(n) for n in neighbours])

        return predictions
    
    def _get_mode(self,arr:np.ndarray)->np.ndarray:
        '''
        given an array of shape (n,p)
        find the most common vector (p,)
        '''
        reshaped_array = arr.reshape(arr.shape[0],arr.shape[1])
        tuple_vectors = [tuple(vector) for vector in reshaped_array] #convert to tuples to make hashable

        return np.array(Counter(tuple_vectors).most_common(1)[0][0])
    
    


