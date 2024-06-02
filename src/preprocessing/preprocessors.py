import numpy as np
import os

from src.config import config

class preprocess_data:
    
    def fit(self,X,y=None):
        
        self.num_rows = X.shape[0]
        self.num_input_features = X.shape[1]
        
        if len(X.shape) == 1:
            self.traget_feature_dim = 1
        else:
            self.traget_feature_dim = y.shape[1]
        
        if len(y.shape) == 1:
            self.traget_feature_dim = 1
        else:
            self.traget_feature_dim = y.shape[1]
        
    def transform(self,X=None,y=None):
        
        self.X = np.array(X).reshape(self.num_rows,self.num_input_features)
        self.y = np.array(y).reshape(self.num_rows,self.traget_feature_dim)
        
        