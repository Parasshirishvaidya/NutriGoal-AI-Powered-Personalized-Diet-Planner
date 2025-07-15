from src.nutrigoal.components.data_ingestion import DataIngestion
from src.nutrigoal.components.data_transformation import DataTransformation
from src.nutrigoal.components.data_transformation import DataTransformationConfig

from src.nutrigoal.components.model_trainer import ModelTrainer
from src.nutrigoal.components.model_trainer import ModelTrainerConfig

def run_training_pipeline():
    obj = DataIngestion()
    data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_arr,y_arr,_ = data_transformation.initiate_data_transformation(data_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(X_array=X_arr,y_array=y_arr)