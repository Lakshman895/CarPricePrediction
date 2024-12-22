import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        model_path = os.path.join('artifacts', 'model.pkl')
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        print("Before Loading")
        
        # Loading the model and preprocessor
        model = load_object(file_path=model_path)
        print("The model is : ", model)
        print('Features: ', features)
        
        preprocessor = load_object(file_path=preprocessor_path)
        print("After Loading")
        
        # Transforming the features
        data_scaled = preprocessor.transform(features)
        print("The scaled data is : ", data_scaled.T)
        
        # Making predictions
        preds = model.predict(data_scaled)
        
        return preds
    
class CustomData:
    def __init__(self,
                 Make: str,
                 Model: str,
                 Year: int,
                 Mileage: int,
                 Fuel_Type: str,
                 Engine_Size: float,
                 Transmission: str,
                 Body_Type: str,
                 Color: str,
                 Owner_History: str,
                 Age: int):
        
        self.Make = Make
        self.Model = Model
        self.Year = Year
        self.Mileage = Mileage
        self.Fuel_Type = Fuel_Type
        self.Engine_Size = Engine_Size
        self.Transmission = Transmission
        self.Body_Type = Body_Type
        self.Color = Color
        self.Owner_History = Owner_History
        self.Age = Age
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'Make' : [self.Make],
                'Model' : [self.Model],
                'Year' : [self.Year],
                'Mileage' : [self.Mileage],
                'Fuel_Type' : [self.Fuel_Type],
                'Engine_Size' : [self.Engine_Size],
                'Transmission' : [self.Transmission],
                'Body_Type' : [self.Body_Type],
                'Color' : [self.Color],
                'Owner_History' : [self.Owner_History],
                'Age' : [self.Age]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
