import numpy as np
import pandas as pd
import plotly.graph_objects 

def sigmoid(z:float)->float:
    if z > 100:
        return 1.00
    if z < -100:
        return 0.00
    return 1/(1+np.exp(-z)) 

# vectorise R -> R function, meaning you can call
# sigmoid(array) without it trying to do np.exp(-array)
# it does newarray = [sigmoid(z:float) for z in array]
# but it's somehow quicker
vectorized_sigmoid:np.vectorize = np.vectorize(sigmoid) 

class LogisticClassifier:
    def __init__(self,X:np.ndarray,y:np.ndarray):
        '''
        X should be an array of shape (m,n); m observations of dimension n.
        y should be an array of shape (m,1); m observations of 0 or 1
        '''
        self.X:np.ndarray = self._getX(X)
        self.y:np.ndarray = self._getY(y)
        self.n:int        = self.X.shape[1]
        self.m:int        = self.X.shape[0]

        self.Xprime:np.ndarray = np.c_[self.X,np.ones(self.m)]
        self.theta:np.ndarray = np.random.randn(self.n+1,1)*0.05

        self._ypred:np.ndarray = self._forward(self.Xprime)
        self._training_history:list = list()
        self._iteration:int = 0
    
    def _getX(self,X:np.ndarray)->np.ndarray:
        '''
        does nothing for the moment, but this will check the shape and dtypes of X.
        '''
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        #dtypes should be numeric only
        if not np.issubdtype(X.dtype,np.number):
            raise ValueError('X should have numeric dtypes')
        return X
    
    def _getY(self,y:np.ndarray)->np.ndarray:
        '''
        it will reshape if necessary
        it will rempa strings to 0,1 if necessary
        it will raise errors if necessary
        '''
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
        if not np.issubdtype(y.dtype,np.number):
            if np.issubdtype(y.dtype,np.str):
                #ensure that there are only two unique strings. If there are more, raise an error
                unique_strings = np.unique(y)
                if len(unique_strings) != 2:
                    raise ValueError('y should have only two unique strings')
                #map the strings to 0,1
                y = np.where(y == unique_strings[0],0,1)
            else:
                raise ValueError('y should have numeric dtypes or strings')
        return y
    
    def _forward(self,Xprime:np.ndarray)->np.ndarray:
        '''
        Xprime is the modified version of self.X
        Xprime is of shape (m,n+1)
        Xprime is X, but each vector is given an extra dimension, with entry 1.
        This allows for simpler gradient descent formulae down the line

        this maps R(m,n+1) to R(m,1)

        (m,n+1) dot (n+1,1) -> (m,1)
        Xprime  dot theta   -> ypred
        '''
        z:np.ndarray = np.dot(Xprime,self.theta)
        return vectorized_sigmoid(z)

    def _dL_dtheta(self)->float:
        differences:np.ndarray = self.y - self._ypred
        # dl_dtheta = []
        # for j in range(self.n+1):
        #     dl_dtheta_j = (differences * self.Xprime[:,j].reshape(-1,1)).sum()
        #     dl_dtheta.append(dl_dtheta_j)
        # return np.array(dl_dtheta).reshape(-1,1)
        return self.Xprime.T.dot(differences)
    
    def _backward(self,learning_rate:float,reg_lambda: float)->None:
        self.theta += learning_rate * (self._dL_dtheta() - reg_lambda * self.theta) # L2 regularisation

    def _liklihood(self)->float:
        '''
        can't pass args because it's internal and more efficient.
        '''
        ypred_clipped:np.ndarray = np.clip(self._ypred,1e-8,1-1e-8)
        log_liklihood:np.ndarray = np.multiply(self.y,np.log(ypred_clipped)) + np.multiply(1-self.y,np.log(1-ypred_clipped))
        log_liklihood:float = log_liklihood.sum()/self.m
        return np.exp(log_liklihood) if log_liklihood > -100 else 0.0 # prevent overflow warnings
    
    def fit(self,
            learning_rate:float,
            num_iterations:int,
            tolerance:float = None,
            print_every:int = None,
            training_history:bool = True,
            reg_lambda:float = 0.0
            )->None:
        '''
        learning_rate: float, the step size for gradient descent
        num_iterations: int, the number of iterations to run gradient descent for
        tolerance: float, if 1-liklihood is less than this, stop
        print_every: int, if not None, print the liklihood every print_every iterations
        training_history: bool, if True, record the liklihood at each iteration
        reg_lambda: float, the L2 regularisation parameter

        '''
        get_liklihood:bool = (training_history is not None) or (tolerance is not None) or (print_every is not None) 
        for i in range(num_iterations):
            self._backward(learning_rate,reg_lambda)
            self._ypred = self._forward(self.Xprime)
            liklihood = self._liklihood() if get_liklihood else None # only calculate liklihood if needed
            self._iteration += 1
            
            # record training history if wanted. Probably is memory inefficient, hence the choice 
            if training_history:
                self._training_history.append({'iteration':self._iteration,'liklihood':liklihood})
            
            # stopping conditions
            if tolerance is not None and 1-liklihood < tolerance:
                print('Tolerance reached')
                break
            # printing
            if print_every is not None and i % print_every == 0:
                print(f'Iteration {self._iteration} liklihood: {liklihood}')

    def predict(self,X:np.ndarray,probabilities:bool = False)->np.ndarray:
        '''
        X is an array of shape (m,n)
        returns an array of shape (m,1) of 0s and 1s or probabilities
        '''
        xcopy = X.copy()

        if len(xcopy.shape) == 1:
            xcopy = X.reshape(-1,1)
        if xcopy.shape[1] != self.n:
            raise ValueError(f'X should have {self.n} columns, not {xcopy.shape[1]}')
        
        Xprime = np.c_[xcopy,np.ones(xcopy.shape[0])]
        ypred = self._forward(Xprime)
        return ypred if probabilities else (ypred > 0.5).astype(int)
    
    def show_training_history(self)->plotly.graph_objects.Figure:
        '''
        returns a plotly figure of the training history. If there is no training history, raises an error.
        '''
        if self._training_history:
            df = pd.DataFrame(self._training_history)
            return df.plot(x = 'iteration',
                            y = 'liklihood',
                            backend = 'plotly')
        else:
            raise ValueError('No training history to show')

    

    

if __name__ == '__main__':
    np.random.seed(1)
    X = [[np.random.rand()*0.5 +4 ]for i in range(25)] + [[np.random.rand()*0.5 +10] for i in range(25)]
    y = [[0]]*25 + [[1]]*25
    X = np.array(X)
    y = np.array(y)

    print(X.shape)
    print(y.shape)

    df = pd.DataFrame(data = {'X':X.flatten(),'y':y.flatten()})
    df['color'] = df.y.astype('str')
    df.plot(x = 'X',y='y',backend = 'plotly',kind = 'scatter',color = 'color').show()
    np.random.seed(None)

    classifier = LogisticClassifier(X,y)
    print(classifier.theta)
    # print(classifier.theta)
    # classifier._backward(0.01)
    # print(classifier.theta)
    # print(classifier._dL_dtheta().shape)
    # print(classifier.theta.shape)
    classifier.fit(learning_rate=0.01,num_iterations = 1000)
    classifier.show_training_history().show()

    df['ypred'] = classifier._ypred
    print(df)




    