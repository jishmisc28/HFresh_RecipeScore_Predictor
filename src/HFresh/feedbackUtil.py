
import pandas as pd

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score

import modelUtil as modelut
import dataUtil as dataut

'''
This will be used to calculating validation results and implementing feedback loop to improve our model
'''
def feedback(pipeline,parameters,ultimateTraindf):
	# Read feedback data
    fbkdf = pd.read_csv(path+"/out/feedback.csv")
    # Extract incorrect predictions
    fbkdf = fbkdf.loc[fbkdf['check'] == False]
	# Combine with rest of the training data
    fbkdf = fbkdf.append(ultimateTraindf, ignore_index=True)
    # Create matrix for learning and prediction
    X, y = fbkdf.drop(['score'],axis=1), fbkdf['score'].as_matrix()
    # Split further into training and validation sets
    Xtrain, Xvalidate, ytrain, yValidate = train_test_split(X, y, train_size=0.7)
    # Initialize gridSearchRegressorCV Classifier with parameters
    gridSearchRegressor = modelut.getGridSearchCv(pipeline, parameters)
    # Fit the gridSearchRegressor on Training Set
    gridSearchRegressor.fit(Xtrain,ytrain)
    # Calculate best set of parameters and make predictions on validation set
    return validate(parameters, gridSearchRegressor, Xvalidate, yValidate)

'''
In this method I calculate the best of parameters for the sklearn Pipeline. 
Then, I make predictions on the validation set and evaluate metrics and scores
Arguments:
gridSearchRegressor <=> GridSearchCV object fitted with feedback data
parameters <=> Initial parameters for Pipeline components
Xvalidate <=> Learning component of validation set
yValidate <=> Prediction component of validation set
Returns:
bestParameters <=> Best set of parameters for sklearn Pipeline after retraining with feedback data
predictions <=> Predictions on the validation set
'''
def validate(parameters, gridSearchRegressor,Xvalidate, yValidate):
    
    # Calculate best score for gridSearchRegressor
    print ('best score: %0.3f' % gridSearchRegressor.best_score_)
    # Calculate best set of parameters for gridSearchRegressor
    bestParameters = gridSearchRegressor.best_estimator_.get_params()
    # Display best set of parameters
    print ('best parameters set:')
    for paramName in sorted(parameters.keys()):
        print ('\t %s: %r' % (paramName, bestParameters[paramName]))
        
    # Make predictions on validation set and evaluate performance of gridSearchRegressor
    predictions = gridSearchRegressor.predict(Xvalidate)
    print ('R2 Score:', r2_score(yValidate, predictions))

	# Return best set of parameters and predictions
    return bestParameters, predictions