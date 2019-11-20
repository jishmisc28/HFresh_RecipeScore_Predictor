
##Food Recipe Score Prediction

**Refrences link(to_read) : https://www.ritchieng.com/machine-learning-evaluate-linear-regression-model/
**Refrences link : https://lib.ugent.be/fulltxt/RUG01/002/166/653/RUG01-002166653_2014_0001_AC.pdf
**Refrences link : https://github.com/sinclam2/regression-analysis-with-python/blob/master/03-Multiple-Regression-in-Action.ipynb
**Refrences link : https://github.com/abtpst/Kaggle-Whats-Cooking/blob/master/src/cook/cookTrain.py


__Problem Statement__
A recipe score or rating is the comparative ranking that a user gives to a specific recipe. If customers like a specific recipe, we would expect them to rate the recipe high, if the customer did not like a recipe, we would expect them to rate it low. Here we will build a model that predicts the average recipe score using the data that is provided in the input datasets. We will concentrate on below two parts of the solution for this problem:
	1. Problem Modeling
	2. Production-Ready Implementation

__Solution Approach__
    1. Problem Modeling : We will be working with python's sci-kit learn library and do a comprehensive Exploratory Data Analysis on the provided dataset over a Jupyter Notebook. We will preprocess the dataset and perform EDA. Basis which we will create new features and understand feature importances and their respective impact on model R2 score. We will use Gradient Boosted Regressor for modeling which provides good accuracy over different loss function options and works well with varied data.
    
    2. Production-Ready Implementation : We will combine all above functioning components in sklearn GridSearchCV Pipeline. We will use GridSearchCV to store and operate on the best set of parameters. We will create datasets as test set for unseen data, and divide the remaining datat to train, validation. 
    We will write all the code as python modules and save the model and output files. 

### Structure

1. **data** 
   
    This folder has the training, validation and test data.

2. **meta**
   
    This folder is used to store intermediate results such as the best set of parameters determined by  GridSearchCV and our trained model. 

3. **notebooks**

    This folder is used to host the notebook containing EDA steps and model execution steps over notebook.

4. **out**

    This folder contains final prediction output on test(unseen) data.  
 
4. **src**

    This folder contains all of the source code.
    
### Training

Training model `Train.py` script in the **HFresh** package inside **src**.
Here is the flow of events

1.  Create `GridSearchCV` `Pipeline` comprising of custom feature transformer and `GradientBoostingRegressor` regressor
2.  Load training data, perform cleanup and feature transformation. Create training and validation sets.
3.  Fit the training set on the pipeline
4.  Calculate the best set of parameters and make predictions on the validation set
5.  Evaluate metrics and scores for the pipeline's `best_estimator`
6.  Document the prediction results on the validation set and save it to a `pandas` `DataFrame`. 
7.  Store the best set of parameters

### Predict
Please look at the documented `Predict.py` script in the **HFresh** package inside **src**.
Here is the flow of events

1. Create `GridSearchCV` `Pipeline` comprising of custom feature transformer and `GradientBoostingRegressor` regressor. This time we use the best set set of parameters obtained from training.
2.  Load test data and perform cleanup and feature transformation. 
3.  Make predictions on the test data and store prediction output in out folder.