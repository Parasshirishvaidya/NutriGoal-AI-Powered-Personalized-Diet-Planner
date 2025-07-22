from src.nutrigoal.pipeline.model_trainer_pipeline import run_training_pipeline
from src.nutrigoal.pipeline.predict_pipeline import PredictPipeline
from src.nutrigoal.logger import logging
import pprint

def get_user_input():
    
    print("Enter your details for personalized diet recommendations")
    logging.info("Asking for user details")
    weight = float(input("Enter your weight(kg): "))
    height = float(input("Enter your height(cm): "))
    age = int(input("Enter your age: "))
    gender = input("Enter your gender (male/female): ").strip().lower()
    goal = input("Enter your goal (gain/weight_loss/lean_protein/maintenance/general): ").strip().lower()
    diet_type = input("Enter your diet type (veg/non-veg): ").strip().lower()
    activity_level = input("Enter your activity level (sedentary / light / moderate / active / very_active): ").strip().lower()

    return  {
        "weight" : weight,
        "height" : height,
        "age" : age,
        "gender" : gender,
        "goal" : goal,
        "diet_type" : diet_type,
        "activity_level" : activity_level
    }

def get_diet_recommendation(user_data:dict):
    pipeline = PredictPipeline()
    calories,recommendation = pipeline.predict(
        data=user_data
    )
    return calories , recommendation


if __name__ =="__main__":
    #run_training_pipeline()    

    logging.info("Starting Predict Pipeline")
    user_details = get_user_input()

    logging.info("Getting results")
    pipeline = PredictPipeline()
    calories,recommendation = pipeline.predict(
        data=user_details
    )

    print(calories)
    pprint.pprint(recommendation)

    # print("\nRecommended Calorie Intake:", recommendation["User Calories"])
    # print("\nTop Carbs Sources:\n", recommendation["Top Carbs Sources"])
    # print("\nTop Protein Sources:\n", recommendation["Top Protein Sources"])
    # print("\nTop Fat Sources:\n", recommendation["Top Fat Sources"])

