
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import dataUtil as dataut

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion



'''
This method defines the sklearn Pipeline. It can be modified to change the Pipeline components.
For simplicity, I am just returning a hard-coded value.
'''
def getPipeline():

    path, target = dataut.getVars()
    X = dataut.preProcessData(path)

    return Pipeline([
        ('fadd',FeatureAdd()),
        ('gbr', GradientBoostingRegressor())
        ])


'''
This method specifies the parameter options for sklearn Pipeline. It can be modified to change the parameter options.
For simplicity, I am just returning a hard-coded value.
'''
def getParameters():

   return { 
            "gbr__n_estimators"      : [20,50,70],
            "gbr__max_features"      : ["auto", "sqrt", "log2"],
            "gbr__max_depth" : [3,5,9,12],
            "gbr__learning_rate": [0.1, 0.01]
            }       
'''
This method returns a GridSearchCV object comprising of an sklearn Pipeline and its corresponding parameter set
'''
def getGridSearchCv(pipeline,parameters):
    
    return GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='r2')
    

class FeatureAdd(TransformerMixin):
    def __init__(self):
        pass
        
    def transform(self, X,y=None):
        
        pd.options.mode.chained_assignment = None
        X['priceXing_count'] = X['price']*X['ingredients_count']
        
        constant = 'Recipe_code'
        dependentVar = 'score'
                
        X = X.drop([dependentVar,constant],1)
                
        return X
    
    def fit(self, *_):
        return self