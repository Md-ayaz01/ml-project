import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.utils import save_object

class DataTransformation:
    def __init__(self):
        pass

    def get_data_transformer_object(self):
        try:
            numerical_features = ["reading_score", "writing_score"]
            categorical_features = [
                "gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"
            ]

            num_pipeline = Pipeline([
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("one_hot", OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numerical_features),
                ("cat", cat_pipeline, categorical_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, df: pd.DataFrame):
        try:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            X_train = train_df.drop("math_score", axis=1)
            y_train = train_df["math_score"]

            X_test = test_df.drop("math_score", axis=1)
            y_test = test_df["math_score"]

            preprocessor = self.get_data_transformer_object()
            preprocessor.fit(X_train)

            X_train_transformed = preprocessor.transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_transformed, y_train]
            test_arr = np.c_[X_test_transformed, y_test]

            os.makedirs("artifacts", exist_ok=True)
            save_object("artifacts/preprocessor.pkl", preprocessor)

            return train_arr, test_arr, "artifacts/preprocessor.pkl"

        except Exception as e:
            raise CustomException(e, sys)

