from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
from functools import reduce
from nltk import sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


class DFFeatureUnion(TransformerMixin):
    """ Custom defined featureunion class which tries handles the dataframe. It takes each and every used defined transformer 
    in the given pipeline and transforms it according to the needs followed by merging individual dataframes """ 

    def __init__(self, transformer_list):
        
        self.transformer_list = transformer_list

    def fit(self, X, y=None, **kwargs):
        
        for (name, t) in self.transformer_list: 
            t.fit(X, y) 
            
        return self

    def transform(self, X, y=None, **kwargs):
        
        Xts = [t.transform(X) for name, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
             
        return Xunion

class RequestedColumns(TransformerMixin):
    """ Custom defined Transformer Class which takes the columns, data as an input and output the segment of the given datframe
    with given input columns """ 
    
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None, **kwargs):
        
        return self

    def transform(self, X, y = None, **kwargs):
        Xcols = X[self.cols]
        return Xcols


class data_cleaner(TransformerMixin):
    """  Custom Define Transformer for Cleaning the Dataset  """
    
    def __init__(self, **kwargs):
        self.columns_names = kwargs['column_names']
        
    def transform(self, X, y = None, **kwargs):
        """  Transforms the given dataset to a required form """
        
        for col in self.columns_names:
            X.loc[: , col] = X[col].str.lower()
            X.loc[: , col] = X[col].str.normalize('NFKD').str.encode('ascii', errors= 'ignore').str.decode('utf-8')
            X.loc[: , col] = X[col].str.replace('[^\w\s]',' ')
            X.loc[: , col] = X[col].map(lambda x : ' '.join([item for item in x.split()]))
            X.loc[: , col] = X[col].map(lambda x : x.strip("º"))
        return X
    
    def fit(self, X, y = None, **kwargs):
        return self

    
class col_remover(TransformerMixin):
    """  Custom Define Transformer for dropping off the selected columns """
    
    def __init__(self, **kwargs):
        self.columns = kwargs['columns']
        
    def transform(self, X, y = None, **kwargs):
        """  Transforms the given dataset to a required form"""

        X = X.drop(self.columns, axis = 1)
            
        return X
    
    def fit(self, X, y = None, **kwargs):
        return self    
    
    
class word_vectorizer_df(TransformerMixin):
    """  Custom Designed Transformer Class to create word level features out of the given text description using
    Tfidfvectorizer """  
    
    def __init__(self, **kwargs):
        #self.test_text = kwargs['other']
        #self.column = kwargs['column']
        self.vectorizer = self.vectorizer()
    
    def vectorizer(self):
        word_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                          strip_accents='unicode',
                                          analyzer='word',
                                          token_pattern=r'\w{1,}',
                                          stop_words='english',
                                          ngram_range=(1, 1),
                                          max_features=10000)
        return word_vectorizer
        
    def transform(self, X, y = None, **kwargs):
        #print('****** vec start shape**********',X.shape)
        main_text_features = self.vectorizer.transform(X)
        cols_list = self.vectorizer.get_feature_names()
        main_text_features_df = pd.DataFrame(main_text_features.toarray(), columns  = cols_list, index = X.index)
        #print('****** vec end shape**********', main_text_features_df.shape)
        return main_text_features_df
    
    def fit(self, X, y = None, **kwargs):
        self.vectorizer.fit(X)
        return self  
    
class char_vectorizer_df(TransformerMixin):
    """  Custom Designed Transformer Class to create character level features out of the given text description using
    Tfidfvectorizer """ 
    
    def __init__(self, **kwargs):
        #self.test_text = kwargs['other']
        #self.column = kwargs['column']
        self.vectorizer = self.vectorizer()
    
    def vectorizer(self):
        char_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                          strip_accents='unicode',
                                          analyzer='char',
                                          stop_words='english',
                                          ngram_range=(2, 6),
                                          max_features=100000)
        return char_vectorizer
        
    def transform(self, X, y = None, **kwargs):
        #print('****** vec start shape**********',X.shape)
        main_text_features = self.vectorizer.transform(X)
        cols_list = self.vectorizer.get_feature_names()
        main_text_features_df = pd.DataFrame(main_text_features.toarray(), columns  = cols_list, index = X.index)
        #print('****** vec end shape**********', main_text_features_df.shape)
        return main_text_features_df
    
    def fit(self, X, y = None, **kwargs):
        self.vectorizer.fit(X)
        return self 

    
class one_hot_encoder(TransformerMixin):
    """  Custom Designed Class to convert Categories in columns to one hot vectors using OneHotEncoder method  """  
    
    def __init__(self):
        self.encoder = self.ohe()
    
    def ohe(self):
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        return ohe
        
    def transform(self, X, y = None, **kwargs):
        #print('****** ohe start shape**********', X.shape)
        
        X_train_ohe = self.encoder.transform(X)
        cols_list = self.encoder.get_feature_names()
        cat_df = pd.DataFrame(X_train_ohe, columns = cols_list, index = X.index)
        #print('****** ohe end shape**********', X_train_ohe.shape)
        return cat_df
    
    def fit(self, X, y = None, **kwargs):
        self.encoder.fit(X)
        return self