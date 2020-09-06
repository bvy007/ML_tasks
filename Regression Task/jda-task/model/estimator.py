from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


class ensembleRegressor(RegressorMixin):
    
    """  Custom  designed regressor class in sklearn format 
    Inputs:
        RFG() : Random Forest Regressor
                A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples
                of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
        fit() : Build a forest of trees from the given dataset 
        Input Parameters:
            n_estimators : The number of trees in the forest.
            max_depth: The maximum depth of the tree.
            random_state: It is the seed used by the random number generator
        
    Outputs:
        predict(): Method which returns the predicted values for a given Dataset
        score(): Returns the coefficient of determination R-square of the prediction.
        maescore(): Returns the Mean Absolute Deviation Score
        
        """
    
    def __init__(self, **kwargs):
        """  Initializing the parameters needed for the regressor"""
        self.max_depth = kwargs['max_depth']
        self.random_state = kwargs['random_state']
        self.n_estimators = kwargs['n_estimators']
        self.algo = self.RFG()
        
    def RFG(self):
        """  Initializing the RFG Algorithm  """
        regressor = RandomForestRegressor(max_depth = self.max_depth, 
                                          random_state = self.random_state,  
                                          n_estimators = self.n_estimators)
        return regressor
        
        
    def predict(self, X, y = None):
        """  Predict the regression value for the given X  """
        y_hat = self.algo.predict(X)        
        return y_hat
    
    def fit(self, X, y = None):
        """  Build regression trees over the given X  """
        self.algo.fit(X, y)
        return self
    
    def score(self, X, y):
        """  Return the r-square coefficient value for the given dataset  """
        r2_score = self.algo.score(X, y)
        return r2_score
    
    def maescore(self, y_true, y_hat):
        """  Returns the Mean Absolute Deviations Score for the given actual and Predicted Values  """
        score = mean_absolute_error(y_true, y_hat)
        return score