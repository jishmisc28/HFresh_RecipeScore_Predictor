
import re
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import dataUtil as dataut
import modelUtil as modelut

'''
This method creates an optimal sklearn Pipeline as per the best set of
parameters obtained in cookTrain.py
'''

def getPipeline():
    
    path, dependentVar = dataut.getVars()
	# Load best set of parameters
    bestParameters = joblib.load(open(path+"/meta/bestParams.pkl","rb"))
    # Create sklearn Pipeline
    pipe = Pipeline([
    ('fadd',modelut.FeatureAdd()),
    ('gbr', 
            GradientBoostingRegressor({"n_estimators": bestParameters['gbr__n_estimators'],
            "max_features"      :  bestParameters['gbr__max_features'],
            "max_depth" : bestParameters['gbr__max_depth'],
            "learning_rate": bestParameters['gbr__learning_rate']}))
    ])   
    # We create this empty dict as it is required for the syntax of GridSearchCV
    parameters = {}
    # Return sklearn Pipeline and empty dict
    return pipe, parameters


def main():
   
    # Get optimal sklearn Pipeline
    #pipe, parameters = getPipeline()
    # Create gridSearchRegressor from optimal Pipeline
    #gridSearchRegressor = GridSearchCV(pipe,parameters,n_jobs=3, verbose=1, scoring='r2')
    
    path, target = dataut.getVars()
    
    # Load Test Set
    X_test = pd.read_csv(path+'/data/test.csv')
    # Load original merged set
    data = dataut.getData(path)    
    
    loaded_model = joblib.load(path+'/meta/finalized_model.sav')
    
    constant = 'Recipe_code'
    dependentVar = 'score'
    X_test['priceXing_count'] = X_test['price']*X_test['ingredients_count']
    #X_test = X_test.drop([dependentVar,constant],1)
    
    # Predict Score on Test Set    
    #predictions=gridSearchRegressor.predict(X_test)
    predictions=loaded_model.predict(X_test)
    result_df = data.loc[data.Recipe_code.isin(X_test.Recipe_code)]
    # Create new column in Test dataframe
    result_df['predicted_score'] = predictions
    # Save the submission dataframe with the new column
    submission_df = result_df[['Recipe_code','predicted_score']]
    submission_df.to_csv(path+'/out/predict_output.csv')
    
if __name__ == '__main__':
    main()