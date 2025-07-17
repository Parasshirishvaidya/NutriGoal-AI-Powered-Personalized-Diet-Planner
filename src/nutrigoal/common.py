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

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def calculate_macros(weight,height,goal,diet_type,age,gender,activity_level):

    """
    Calculate daily calorie and macronutrient requirements based on user profile and goal.

    Scientific Basis:
    - BMR (Basal Metabolic Rate) is calculated using the Mifflin-St Jeor Equation.
    - TDEE (Total Daily Energy Expenditure) is derived from BMR and activity level.
    - Calories are adjusted based on fitness goal.
    - Protein and fat requirements are goal-specific (in g/kg body weight).
    - Remaining calories are allotted to carbohydrates.
    - Fiber is fixed to a general healthy intake.

    Parameters:
        weight (float): Weight in kilograms.
        height (float): Height in centimeters.
        goal (str): One of ['gain', 'weight_loss', 'lean_protein', 'maintenance', 'general'].
        diet_type (str): User's diet preference (currently unused, placeholder for future).
        age (int): Age in years.
        gender (str): 'male' or 'female'.
        activity_level (str): One of ['sedentary', 'light', 'moderate', 'active', 'very_active'].

    Returns:
        dict: {
            "calories": Total daily calories,
            "protein": Grams of protein per day,
            "fat": Grams of fat per day,
            "carbs": Grams of carbs per day,
            "fiber": Grams of fiber (fixed at 25g)
        }
    """
    
    if gender.lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age +5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    # Apply activity Multiplier to get TDEE
    activity_multipliers = {
         "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9
    } 
    tdee = bmr * activity_multipliers.get(activity_level.lower(), 1.2)

    # Adjust Calories based on goal
    if goal == "gain":
        calories = tdee + 300
        protein = 1.8 * weight
        fat = 1.0 * weight
    elif goal == "weight_loss":
        calories = tdee - 500
        protein = 2.2 * weight
        fat = 0.6 * weight
    elif goal == "lean_protein":
        calories = tdee + 100
        protein = 2.5 * weight
        fat = 0.6 * weight
    elif goal == "maintenance":
        calories = tdee
        protein = 1.6 * weight
        fat = 0.8 * weight 
    else:
        calories = tdee
        protein = 1.6 * weight
        fat = 0.8 * weight
    
    protein_cals = protein * 4
    fat_cals = fat * 9

    # Remaining Calories go to carbs
    carbs = (calories-(protein_cals+fat_cals))/4

    fiber = 25

    return {
        "calories": round(calories, 2),
        "protein": round(protein, 2),
        "fat": round(fat, 2),
        "carbs": round(carbs, 2),
        "fiber": fiber
    }



def compute_serving(food,target_macro,macro_key):
    if food[macro_key] <= 0:
        return 0
    grams_needed = (target_macro/food[macro_key]) * 100
    return round(grams_needed,1)
