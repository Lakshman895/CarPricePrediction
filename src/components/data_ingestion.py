import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))

from src.exception import CustomException
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.components.data_transformation import DataTransformation 
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    artifacts_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','artifacts'))
    train_data_path: str = os.path.join(artifacts_dir,'train.csv')
    test_data_path: str = os.path.join(artifacts_dir,'test.csv')
    raw_data_path: str = os.path.join(artifacts_dir,'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()    
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            logging.info("Loading the data set")
            df = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','..','data','cleaned_car_data.csv'))
            
            num_cols = df.select_dtypes(exclude=['object']).columns
            logging.info(f"The skewness of the data is : \n {df[num_cols].skew().sort_values(ascending=False)}")
            
            os.makedirs(os.path.dirname(os.path.abspath(self.ingestion_config.raw_data_path)), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)
            logging.info('Data loaded successfully, about to split into train and test sets')
            
            train_set,test_set = train_test_split(df,test_size=0.2, random_state=42)
            logging.info('Data splitted successfully')
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True)
            logging.info('Train Data splitted successfully')
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Test Data splitted successfully')
            logging.info('Ingestion of the data is completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            raise CustomException(e,sys)        
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    print(train_data_path)
    print(test_data_path)
    
    data_transformation = DataTransformation()
    X_train_data,y_train_data,X_test_data,y_test_data,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(X_train_data,y_train_data,X_test_data,y_test_data))