from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
from functools import reduce

class data_cleaner(TransformerMixin):
    """  Custom Define Transformer for Cleaning the Dataset  """
    
    def __init__(self, **kwargs):
        self.columns_names = kwargs['new_column_names']
        
    def transform(self, X, y = None):
        """  Transforms the given dataset to a required form"""
        del_cols = self.columns_names['del_columns']
        category_cols = self.columns_names['category_columns']

        X = X.drop(list(del_cols.values()), axis = 1)
        X[list(category_cols.values())] = X[list(category_cols.values())].astype('category')
        
        return X
    
    def fit(self, X, y = None):
        return self

    
class one_hot_encoder(TransformerMixin):
    """  Custom Designed Class to convert Categories in columns to one hot vectors using the pandas dummies method  """  
    def __init__(self):
        return
        
    def transform(self, X, y = None):
        X = pd.get_dummies(X, prefix_sep='_', drop_first=True)
        return X
    
    def fit(self, X, y = None):
        return self