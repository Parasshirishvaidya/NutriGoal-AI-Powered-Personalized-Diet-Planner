import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.nutrigoal.exception import CustomException
from src.nutrigoal.logger import logging 
from src.nutrigoal.common import save_object

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler

@dataclass
class DataTransformationConfig:
    transformed_csv_file_path: str = os.path.join("src/nutrigoal/datasets", "transformed_dataset.csv")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self,data_path):
        try:
            df = pd.read_csv(data_path)

            scaler = StandardScaler()
            num_cols = ["carbs","protein","fat","fiber","calories"]
            df[num_cols] = scaler.fit_transform(df[num_cols])
            save_object(file_path="artifacts/scaler.pkl",obj=scaler)
            le_goal = LabelEncoder()
            le_diet = LabelEncoder()
            df["goal_tag"] = le_goal.fit_transform(df["goal_tag"])
            df["diet_type"] = le_diet.fit_transform(df["diet_type"])
            save_object(file_path="artifacts/le_goal.pkl",obj=le_goal)
            save_object(file_path="artifacts/le_diet.pkl",obj=le_diet)
            df["match_score"] = df.apply(simulate_match_score, axis=1)
            X = df[["carbs","protein","fat","fiber","calories","goal_tag","diet_type"]]
            y = df["match_score"]
            df.to_csv(self.data_transformation_config.transformed_csv_file_path,index=False)
            return(
                X.to_numpy(),
                y.to_numpy(),
                self.data_transformation_config.transformed_csv_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)


def simulate_match_score(row,noise_mean=0,noise_std=0.05):

    prot = row["protein"]
    fat = row["fat"]
    carbs = row["carbs"]
    cal = row["calories"]
    fiber = row["fiber"]
    goal = row["goal_tag"]

    if goal == 0:  # gain
        score = 0.3 * prot + 0.3 * carbs + 0.3 * cal - 0.1 * fat
    elif goal == 1:  # general
        score = 0.25 * prot + 0.25 * carbs + 0.25 * cal - 0.25 * fat
    elif goal == 2:  # lean_protein
        score = 0.5 * prot - 0.2 * fat + 0.1 * fiber
    elif goal == 3:  # maintenance
        score = 0.3 * prot + 0.3 * carbs + 0.2 * fat + 0.2 * cal
    elif goal == 4:  # weight_loss
        score = 0.4 * prot - 0.3 * fat - 0.2 * cal + 0.1 * fiber
    else:
        score = 0.0

    noise = np.random.normal(loc=noise_mean, scale=noise_std)
    noisy_score = score + noise

    return np.clip((noisy_score + 1) / 2, 0, 1)  # Normalize between 0 and 1