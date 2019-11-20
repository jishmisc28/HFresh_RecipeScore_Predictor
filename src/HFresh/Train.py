
import re
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

import modelUtil as modelut
import dataUtil as dataut
import feedbackUtil as feedut


def getBestParameters(pipeline,parameters):    
    
    path, dependentVar = dataut.getVars()
        
    X = dataut.preProcessData(path)   
    
    y = X.loc[:,dependentVar]       
        
    # create and fit a GBR model    
    grid = dataut.GridSearchCV(pipeline, parameters)
    grid.fit(X, y)
    
    # summarize the results of the grid search
    print(grid.best_score_)
    print(grid.best_estimator_.get_params())
    bestParameters = grid.best_estimator_.get_params()
    
    #persist to disk
    filename = path+'/meta/finalized_model.sav'
    joblib.dump(grid,filename)
        
    # Display best set of parameters
    #print ('best parameters set:')
    #for paramName in sorted(parameters.keys()):
    #    print ('\t %s: %r' % (paramName, bestParameters[paramName]))        
        
    # Evaluate pegbrormance of gridSearchRegressor on Validation Set
    X_valid = pd.read_csv(path+'/data/validation.csv')
    y_valid = X_valid.loc[:,dependentVar]
    
    constant = 'Recipe_code'
    dependentVar = 'score'
    X_valid['priceXing_count'] = X_valid['price']*X_valid['ingredients_count']
    X_valid = X_valid.drop([dependentVar,constant],1)

    # Make predictions on validation set and calculate best set of parameters  
    bestParameters,predictions=feedut.validate(parameters, grid, X_valid, y_valid)    

    # Initialize DataFrame for feedback loop
    valdf = pd.DataFrame(index = X_valid.index.values)
    # Add ingredients column
    valdf=valdf.join(X_valid)
    # Add correct cuisine
    valdf["cuisine"] = y_valid
    # Add predictions column
    valdf["pred_cuisine"] = predictions
    # Add check column. This column would be false for incorrect predictions
    valdf["check"] = valdf.pred_cuisine==valdf.cuisine
    # Store DataFrame for feedback
    valdf.to_csv(path+"/out/feedback.csv")

    # Create joint DataFrame to incorporate feedback data. As of now, this will only have the ingredients and cuisine columns from the training set
    ultimateTraindf = pd.DataFrame(index = X.index.values)
    ultimateTraindf=ultimateTraindf.join(X)
    ultimateTraindf["cuisine"] = y
    # Calculate best set of parameters after retraining with feedback data. Make predictions on validation set 
    bestParameters,predictions = feedut.feedback(pipeline,parameters,ultimateTraindf)       
    
    """validation_R2 = r2_score(y_valid,grid.fit(X_valid,y_valid).predict(X_valid))
    print("............................................")
    print("............................................")
    print("our validation set R2 score is %.2f%%"%(validation_R2*100))
    
    X_valid['pred_score'] = grid.predict(X_valid)
    X_valid['difference'] = X_valid['score']-X_valid['pred_score']
    X_valid = X_valid[['score','pred_score','difference']]
    X_valid.to_csv(path+'/out/validation_predict.csv')  """  
        
    
    return bestParameters
	
def main():
    
    # Get Pipeline components
    pipeline = modelut.getPipeline()
    
    # Get parameter options for Pipeline components
    parameters = modelut.getParameters()
    
    # Get best set of parameters and evaluate validation set accuracy
    bestParameters = getBestParameters(pipeline,parameters)
    
    path, dependentVar = dataut.getVars()
	
    # Save best parameter set
    res = open(path+"/meta/best_params_model.txt", 'w')
    res.write ('best parameters set:\n')
    for paramName in sorted(parameters.keys()):
        res.write('\t %s: %r\n' % (paramName, bestParameters[paramName]))
    
    joblib.dump(bestParameters,open(path+"/meta/bestParams.pkl","wb"))
    
if __name__ == '__main__':
    main()

