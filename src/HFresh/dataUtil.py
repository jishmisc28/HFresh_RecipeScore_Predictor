

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

'''
This is a data preprocessing module which will merge and create a unified dataset from the given data files.
'''
def preProcessData(path):
    
    ingredients = pd.read_csv(path+'/data/ingredients.csv')
    rating = pd.read_csv(path+'/data/recipe_ratings.csv')
    
    print ("................")
    print ("grouping the ingridients dataset on recipe code and recipe name.. ")
    print ("adding a feature column as ingredients_list and ingredients_count.. ")
    ingredients_grp = ingredients.groupby(['Recipe_code','Recipe_name'])['Ingredient'].apply(list).reset_index(name='ingredients_list')
    ingredients_grp['ing_list2'] = ingredients_grp.ingredients_list.astype('str').str.split(',')
    ingredients_grp['ingredients_count'] = ingredients_grp.ing_list2.str.len()
    ingredients_grp.drop(['ing_list2'], axis=1,inplace=True)
    ingredients_grp.drop(['ingredients_list'], axis=1,inplace=True)

    print ("................")
    print ("unique recipe codes in ingredients.csv: ",ingredients.Recipe_code.nunique())
    print ("unique recipe codes in rating.csv: ",rating.Recipe_code.nunique())
    print ("unique recipe codes in merged and unfied dataset: ",ingredients_grp.Recipe_code.nunique())    
    
    data = pd.merge(ingredients_grp,rating,how='left',on='Recipe_code')    
    print ("created a merged dataset with shape: ",data.shape)
    print ("................")
    print ("unique count in the unified dataset")
    print(data[['Recipe_code', 'Recipe_name', 'score', 'price','ingredients_count']].apply(pd.Series.nunique).sort_values(ascending=False))
    print ("................")
    print ("There are %r records in the unified dataset where target variable (score) are NaNs"%(len(data.loc[data.new==1.0])))    
    
    print ("................")
    print ("applying OHE on the dataset")
    data = pd.get_dummies(data, columns=data[['Recipe_name', 'score', 'price','ingredients_count']].select_dtypes(include=['object']).columns, drop_first=True)
    
    data_pred = data.loc[data.new==1.0]
    data = data.loc[data.new==0.0]
    
    print ("................")
    print ("Creating a train, validation and test set csv files..")
    train_df = data.sample(frac=0.8, replace=False, random_state=1)
    valid_df = data.sample(frac=0.2, replace=False, random_state=1)
    test_df = data_pred
    
    
    train_df.to_csv(path+'/data/train.csv') 
    valid_df.to_csv(path+'/data/validation.csv')
    test_df.to_csv(path+'/data/test.csv')
    
    return train_df
    
    
def getData(path):
    
    ingredients = pd.read_csv(path+'/data/ingredients.csv')
    rating = pd.read_csv(path+'/data/recipe_ratings.csv')
    
    ingredients_grp = ingredients.groupby(['Recipe_code','Recipe_name'])['Ingredient'].apply(list).reset_index(name='ingredients_list')
    ingredients_grp['ing_list2'] = ingredients_grp.ingredients_list.astype('str').str.split(',')
    ingredients_grp['ingredients_count'] = ingredients_grp.ing_list2.str.len()
    ingredients_grp.drop(['ing_list2'], axis=1,inplace=True)
    ingredients_grp.drop(['ingredients_list'], axis=1,inplace=True)

    data = pd.merge(ingredients_grp,rating,how='left',on='Recipe_code')    
    
    return data
    
def oheData(df,columns_list):
    df = df.loc[(df.columns.isin(columns_list))]
    df_encoded = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True)
    return df_encoded

def preModelSteps(df,columns_list):
    df_encoded = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True)
    
    df_encoded = oheData(df,columns_list)
    dependentVar = columns_list[-1]
    
    df_encoded_pred = df_encoded.loc[df_encoded.new==1.0]
    df_encoded = df_encoded.loc[df_encoded.new==0.0]
    
    y = df_encoded.loc[:,dependentVar]
    X = df_encoded.drop(dependentVar,1).values
    
    return X,y    
    
def getVars():
    path  = "C:\\Users\\abhranshu\\Desktop\\Incubating\\Ab_ML_Notebooks\\HFresh_Submission"
    target = 'score'
    return path,target
    

