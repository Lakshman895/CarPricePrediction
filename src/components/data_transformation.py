import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,FunctionTransformer
from src.exception import CustomException
from src.logger import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import boxcox

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This fuction is responsible for data transformation
        '''
        try:
            numerical_columns = ['Year', 'Mileage', 'Engine_Size', 'Age']
            categorical_columns = ['Make', 'Model', 'Fuel_Type', 'Transmission', 'Body_Type', 'Color', 'Owner_History']
            
            num_pipeline = Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline,categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            logging.info('Entered initiate data transformation')
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Read train and test data completed')
            
            train_df['Price'] = np.log1p(train_df['Price'])
            test_df['Price'] = np.log1p(test_df['Price'])
            
            logging.info(f'Skew of price train data {train_df["Price"].skew()}')
            logging.info(f'Skew of price test data {test_df["Price"].skew()}')
            
            logging.info('Obtaining preprocessing object')
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "Price"
            
            X_train_df = train_df.drop(columns=[target_column_name],axis=1)
            y_train_data = train_df[target_column_name]
            
            X_test_df = test_df.drop(columns=[target_column_name],axis=1)
            y_test_data = test_df[target_column_name]
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            
            processed_train_data = preprocessing_obj.fit_transform(X_train_df)
            processed_test_data = preprocessing_obj.transform(X_test_df)
            
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                
            )
            
            return (
                processed_train_data,
                y_train_data,
                processed_test_data,
                y_test_data,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.error(f'Error occurred: {e}')
            raise CustomException(e,sys)