import os
import sys
import pandas as pd
import numpy as np

from src.nutrigoal.exception import CustomException
from src.nutrigoal.logger import logging
from src.nutrigoal.common import load_object
from src.nutrigoal.common import calculate_macros
from src.nutrigoal.common import compute_serving

class PredictPipeline:
    def __init__(self):
        self.scaler = load_object("artifacts/scaler.pkl")
        self.le_diet = load_object("artifacts/le_diet.pkl")
        self.le_goal = load_object("artifacts/le_goal.pkl")
        self.model = load_object("artifacts/model.pkl")
        self.df = pd.read_csv("src/nutrigoal/datasets/transformed_dataset.csv")

    def predict(self,data:dict):
        try:
            weight = data["weight"]
            height = data["height"]
            age = data["age"]
            gender = data["gender"]
            goal = data["goal"]
            diet_type = data["diet_type"]
            activity_level = data["activity_level"]  

            user_macros = calculate_macros(
                weight=weight,height=height,goal=goal,diet_type=diet_type,age=age,gender=gender,activity_level=activity_level
            )

            num_features = ["carbs","protein","fat","fiber","calories"]
            X_user = pd.DataFrame([user_macros])[num_features]
            X_scaled = self.scaler.transform(X_user)

            goal_encoded = self.le_goal.transform([goal])[0]
            diet_encoded = self.le_diet.transform([diet_type])[0]

            X_final = np.hstack([X_scaled[0],goal_encoded,diet_encoded]).reshape(1,-1)
            
            df_features = self.df[["carbs","protein","fat","fiber","calories","goal_tag","diet_type"]]
            X_all_final = df_features.to_numpy()
            predictions = self.model.predict(X_all_final)
            self.df["predicted_score"] = predictions

            filtered_df = self.df[
                (self.df["goal_tag"] == goal_encoded) &
                (self.df["diet_type"] == diet_encoded)
            ]

            top_carbs_raw = (
                filtered_df.sort_values(["carbs","predicted_score"], ascending=[False, False])
                .head(10)
                .copy()
                )
            carb_scaled_features = top_carbs_raw[["carbs","protein","fat","fiber","calories"]]
            carb_unscaled = self.scaler.inverse_transform(carb_scaled_features)
            carbs_unscaled_df = pd.DataFrame(carb_unscaled,columns=["carbs","protein","fat","fiber","calories"])
            top_carbs = pd.concat(
                [top_carbs_raw[["Description","predicted_score"]].reset_index(drop=True),carbs_unscaled_df[["carbs","calories","protein","fat"]]],axis=1
            )
            

            top_protein_raw = (
                filtered_df.sort_values(["protein","predicted_score"], ascending=[False,False])
                .head(10)
                .copy()
            )
            protein_scaled_features = top_protein_raw[["carbs", "protein", "fat", "fiber", "calories"]]
            protein_unscaled = self.scaler.inverse_transform(protein_scaled_features)
            protein_unscaled_df = pd.DataFrame(protein_unscaled, columns=["carbs", "protein","fat","fiber", "calories"])
            top_protein = pd.concat(
                [top_protein_raw[["Description", "predicted_score"]].reset_index(drop=True), protein_unscaled_df[["protein","calories","carbs","fat"]]],
                axis=1
                )


            top_fat_raw = (
                filtered_df.sort_values(["fat","predicted_score"],ascending=[False,False])
                .head(10)
                .copy()
            )
            fat_scaled_features = top_fat_raw[["carbs", "protein", "fat", "fiber", "calories"]]      
            fat_unscaled = self.scaler.inverse_transform(fat_scaled_features)
            fat_unscaled_df = pd.DataFrame(fat_unscaled, columns=["carbs", "protein", "fat", "fiber", "calories"])
            top_fat = pd.concat(
                [top_fat_raw[["Description", "predicted_score"]].reset_index(drop=True), fat_unscaled_df[["fat","calories","protein","carbs"]]],
                axis=1
                )

            user_prefrences = {
                "Top Carbs Sources" : top_carbs.reset_index(drop=True),
                "Top Protein Sources" : top_protein.reset_index(drop=True),
                "Top Fat Sources" : top_fat.reset_index(drop=True) 
            }

            predictions = self.generate_meal_plan(user_prefrences,user_macros)
            return user_macros["calories"],predictions
        except Exception as e:
            raise CustomException(e,sys)

    def generate_meal_plan(self,user_food:dict,total_macros:dict):
        try:
            meal_macros = {
                "calories" : total_macros["calories"]/4,
                "protein" : total_macros["protein"]/4,
                "carbs" : total_macros["carbs"]/4,
                "fat" : total_macros["fat"]/4
            }

            meal_plan = []

            for i in range(3):
                meal = {
                    "meal_number" : i+1,
                    "items" : []
                }

                carb = user_food["Top Carbs Sources"].loc[i]
                protein = user_food["Top Protein Sources"].loc[i]
                fat = user_food["Top Fat Sources"].loc[i]

                carb_serving = compute_serving(carb,meal_macros["carbs"],"carbs")
                protein_serving = compute_serving(protein,meal_macros["protein"],"protein")
                fat_serving = compute_serving(fat,meal_macros["fat"],"fat")

                meal["items"].append({
                    "Name" : carb["Description"],
                    "Target" : "Carbohydrates",
                    "Serving Size" : float(carb_serving),
                    "per_100g_macros" : {
                        "calories" : round(float(carb["calories"]),2),
                        "carbs" : round(float(carb["carbs"]),2),
                        "protein" : round(float(carb["protein"]),2),
                        "fat" : round(float(carb["fat"]),2)
                    }
                })

                meal["items"].append({
                    "Name" : protein["Description"],
                    "Target" : "Protein",
                    "Serving Size" : float(protein_serving),
                    "per 100gm macros" : {
                        "calories" : round(float(protein["calories"]),2),
                        "carbs" : round(float(protein["carbs"]),2),
                        "protein" : round(float(protein["protein"]),2),
                        "fat" : round(float(protein["fat"]),2)
                    }  
                })

                meal["items"].append({
                    "Name" : fat["Description"],
                    "Target" : "Fat",
                    "Serving Size": float(fat_serving),
                    "per 100gm macros" : {
                        "calories" : round(float(fat["calories"]),2),
                        "carbs" : round(float(fat["carbs"]),2),
                        "protein" : round(float(fat["protein"]),2),
                        "fat" : round(float(fat["fat"]),2)
                    }
                })
                meal_plan.append(meal)
            return meal_plan
        except Exception as e:
            raise CustomException(e,sys)      