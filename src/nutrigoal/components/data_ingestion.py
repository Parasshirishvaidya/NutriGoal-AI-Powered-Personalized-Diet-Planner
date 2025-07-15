import os 
import sys

from src.nutrigoal.exception import CustomException
from src.nutrigoal.logger import logging

import pandas as pd
from dataclasses import dataclass
import numpy as np

@dataclass
class DataIngestionConfig:
    data_path: str=os.path.join('src/nutrigoal/datasets','dataset1_filtering_done.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
         
    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion component")
        try:
            df = pd.read_csv("research/dataset0_rawdata.csv")
            logging.info('Read dataset as dataframe')
            
            # Renaming Columns
            df = df.rename(columns={
                 "Data.Protein" : "protein",
                 "Data.Fat.Total Lipid" : "fat",
                 "Data.Carbohydrate" : "carbs",
                 "Data.Fiber" : "fiber"
            })
             
            # Saving only the required columns in the dataset
            df_new = df[["Category","Description","carbs","protein","fat","fiber"]].copy()

            # Dropping irrelevant records
            df_new.drop(df_new[df_new["Description"] == "Milk, human"].index, inplace=True)
            df_new.drop(df_new[df_new["Category"] == "Infant formula"].index, inplace=True)
            indices_to_drop = df_new[df_new["Category"]=="Beef"].index
            df_new.drop(indices_to_drop,inplace=True)
            # Filter out rows where 'beef' is mentioned in either description or category
            df_new = df_new[~df_new['Description'].str.contains('beef', na=False)]
            df_new = df_new[~df_new['Category'].str.contains('beef', na=False)]
            df_new = df_new.reset_index(drop=True)

            #Calculate Calories
            df_new["calories"] = (df_new["protein"] * 4) + (df_new["carbs"] * 4) + (df_new["fat"] * 9)

            #create new goal tag
            df_new["goal_tag"] = df_new.apply(assign_goal,axis=1)
             
            #adding veg or non_veg tag
            df_new["diet_type"] = df_new["Category"].apply(get_food_type)

            #Entering new records
            veg_proteins = [
                {
                    "Category": "Plant Protein",
                    "Description": "Tofu (firm)",
                    "carbs": 2.0,
                    "protein": 17.0,
                    "fat": 8.0,
                    "fiber": 1.5,
                    "calories": (17 * 4) + (2 * 4) + (8 * 9),  # 68 + 8 + 72 = 148
                    "goal_tag": "lean_protein",
                    "diet_type": "veg"
                },
                {
                    "Category": "Plant Protein",
                    "Description": "Tempeh",
                    "carbs": 9.0,
                    "protein": 19.0,
                    "fat": 11.0,
                    "fiber": 1.0,
                    "calories": (19 * 4) + (9 * 4) + (11 * 9),  # 76 + 36 + 99 = 211
                    "goal_tag": "lean_protein",
                    "diet_type": "veg"
                },
                {
                    "Category": "Dairy",
                    "Description": "Low-Fat Paneer",
                    "carbs": 3.0,
                    "protein": 20.0,
                    "fat": 8.0,
                    "fiber": 0.0,
                    "calories": (20 * 4) + (3 * 4) + (8 * 9),  # 80 + 12 + 72 = 164
                    "goal_tag": "lean_protein",
                    "diet_type": "veg"
                },
                {
                    "Category": "Supplements",
                    "Description": "Whey Protein Isolate",
                    "carbs": 2.0,
                    "protein": 85.0,
                    "fat": 1.5,
                    "fiber": 0.0,
                    "calories": (85 * 4) + (2 * 4) + (1.5 * 9),  # 350.5
                    "goal_tag": "lean_protein",
                    "diet_type": "veg"
                }
                ]
            
            df_new = pd.concat([df_new, pd.DataFrame(veg_proteins)], ignore_index=True)

            #Categories and Desription to lower case to remove any discripencies
            df_new["Category"] = df_new["Category"].str.lower()
            df_new["Description"] = df_new["Description"].str.lower()

            #Saving new Dataset 
            os.makedirs(os.path.dirname(self.ingestion_config.data_path), exist_ok=True)
            df_new.to_csv(self.ingestion_config.data_path,index=False,header=True)
            return self.ingestion_config.data_path

        except Exception as e:
            raise CustomException(e,sys)
    
#Assigning Goal Tags
def assign_goal(row):
    cal = row["calories"]
    prot = row["protein"]
    fat = row["fat"]
    carbs = row["carbs"]
    cat = row["Category"]
         
    if prot>=15 and cal >=250:
        return "gain"
    elif prot>=15 and cal<=150 and fat<=5:
        return "lean_protein"
    elif fat<=5 and cal<=120:
        return "weight_loss"
    elif 8<=prot<=20 and 150<=cal<=300 and 5<=fat<=15:
        return "maintenance"
    elif "chicken" in cat.lower() and prot>=20 and cal<200:
        return "lean_protein"
    else:
        return "general"
    
#assiging VEG or NON-VEG type
def get_food_type(desc):
    desc = desc.lower()
    if any(x in desc for x in [
        "chicken","meat","pork","fish","squid","tuna",'whiting',"tilapia",'fish', 'salmon', 'tuna',
    'cod', 'haddock', 'mackerel', 'halibut', 'snapper', 'sardine', 'anchovy',
    'tilapia', 'bass', 'catfish', 'swordfish', 'trout', 'pollock', 'grouper', 'barramundi', 'bluefish',
    'herring', 'flounder', 'sole', 'shrimp', 'prawn', 'lobster', 'crab', 'crayfish', 'clam', 'mussel',
    'oyster', 'scallop', 'cockle', 'whelk', 'abalone', 'squid', 'octopus', 'cuttlefish', 'eel', 'uni',
    'sea urchin', 'sea cucumber', 'jellyfish',"turtle","ocean perch","perch","pike","porgy","mullet",
    "frog legs","goat","ham","bison","moose","turkey","croaker","canadian bacon","frankfurter or hot dog",
    " venison/deer steak","wild pig","cornish game hen","kidney","venison or deer with tomato-based sauce",
    "venison or deer with gravy","venison/deer steak","gizzard"
    ]):
        return 'non_veg'
    
    else:
        return "veg"


