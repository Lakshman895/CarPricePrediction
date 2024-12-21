import pickle,os,sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as f:
            pickle.dump(obj, f)
            
        logging.info('Dumped the object file into pickle file')
        
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test, models,params):
    try:
        report = {}
        report_data = []
        
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = models[model_name]
            param = params[model_name]
            
            logging.info(f'Performing the gridsearch:{model_name}')
            
            gs = GridSearchCV(model, param, cv=5, scoring='r2')
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            logging.info('Predicting the train and test data')
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            train_model_score = r2_score(y_train,y_pred_train)
            test_model_score = r2_score(y_test,y_pred_test)
            mse = mean_squared_error(y_test,y_pred_test)
            rmse = np.sqrt(mse)
            cross_score = cross_val_score(model, X_train,y_train,cv=5,scoring='r2')
            
            result = {
                'Model': model_name,
                'Train_Score': train_model_score,
                'Test_Score': test_model_score,
                'RMSE': rmse,
                'Cross_val_score': cross_score.mean()
            }
            
            report_data.append(result)
            
            logging.info('Appending the scores')
        
        report_df = pd.DataFrame(report_data)
        logging.info('Saving the result into csv file')
        report_df.to_csv('artifacts/model_result.csv',index=False)
        
        return report_df
        
    except Exception as e:
        logging.error(f'Error occurred: {e}')
        raise CustomException(e,sys)
        
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)