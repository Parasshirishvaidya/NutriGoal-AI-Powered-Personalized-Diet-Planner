import os
import sys
 
from dataclasses import dataclass

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.nutrigoal.logger import logging
from src.nutrigoal.exception import CustomException
from src.nutrigoal.common import save_object
from src.nutrigoal.common import evaluate_models
from sklearn.model_selection import train_test_split

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,X_array,y_array):

        try:
            X_train,X_test,y_train,y_test = train_test_split(X_array,y_array,test_size=0.2,random_state=42)

            models = {
                "Random Forest Regressor": RandomForestRegressor(),
                "Decision Tree Regressor" : DecisionTreeRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Regressor" : KNeighborsRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "CatBoost Regressor" : CatBoostRegressor(verbose=False),
                "AdaBoost Regressor" : AdaBoostRegressor(),
            }
            param_grids = {
                "Random Forest Regressor": {"n_estimators": [100, 200], "max_depth": [10, 20, None]},
                "Decision Tree Regressor": {"max_depth": [10, 20, None], "criterion": ["squared_error", "friedman_mse"]},
                "Linear Regression": {},  # No hyperparameters to tune for LinearRegression
                "K-Neighbors Regressor": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
                "XGBRegressor": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]},
                "CatBoost Regressor": {"depth": [6, 8], "learning_rate": [0.01, 0.1], "iterations": [100, 200]},
                "AdaBoost Regressor": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.5]},
                }
            logging.info("Training Models")
            model_report,best_model_param,best_model = evaluate_models(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                models=models,
                params=param_grids
            )
            #Print values
            best_model_score = max(model_report.values())
            best_model_name = [key for key, value in model_report.items() if value == best_model_score][0]
            print(best_model_name)
            print(f'Best model score {best_model_score}')

            logging.info(f"Saved best model to: {self.model_trainer_config.trained_model_file_path}")
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            print(f"Saved best model to: {self.model_trainer_config.trained_model_file_path}")
        except Exception as e:
            raise CustomException(e,sys)