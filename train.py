from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
import pandas as pd
import os

if __name__ == "__main__":
    print("🚀 Starting training pipeline")

    # 1. Load your raw dataset manually here
    # Replace this with your actual data file path
    data_path = os.path.join("notebook", "data", "stud.csv")
    df = pd.read_csv(data_path)
    print("✅ Loaded dataset")

    # 2. Data Transformation
    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(df)

    # 3. Model Training
    trainer = ModelTrainer()
    print("🏋️ Starting model training...")
    print(trainer.initiate_model_trainer(train_arr, test_arr))

    print("✅ Training complete. Artifacts saved to /artifacts")
