import os 
import sys 

import numpy as np 
import pandas as pd 

from src.nutrigoal.exception import CustomException
from src.nutrigoal.logger import logging

import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}
        best_params = {}

        for model_name,model in models.items():
            grid = GridSearchCV(model,params[model_name],cv=5,n_jobs=1)
            grid.fit(X_train,y_train)

            #get best Model
            best_model = grid.best_estimator_
            best_params[model_name] = grid.best_params_
            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            #Evaluate
            train_r2 = r2_score(y_train,y_train_pred)
            test_r2 = r2_score(y_test,y_test_pred) 

            report[model_name] = test_r2

        #Retrieve best model name and best_model parameters
        best_model_score = max(report.values())
        best_model_name = [key for key, value in report.items() if value == best_model_score][0]
        best_model_param = best_params[best_model_name]
        return report,best_model_param,best_model
    except Exception as e:
        raise CustomException(e,sys)