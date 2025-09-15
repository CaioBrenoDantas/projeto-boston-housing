import joblib
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

class house_features(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: float
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: float
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float

try :    
    modelo =joblib.load('../models/trained_model.pkl')
except FileNotFoundError:
    modelo =joblib.load('models/trained_model.pkl')

@app.get('/')
def ola_mundo():
    return 'Olá mundo'

@app.post('/predict')
async def predict_house_price(data:house_features):
    
    input_data = data.model_dump()
    
    feature_list = [[input_data['CRIM'], input_data['ZN'], input_data['INDUS'], input_data['CHAS'],
                    input_data['NOX'], input_data['RM'], input_data['AGE'], input_data['DIS'],
                    input_data['RAD'], input_data['TAX'], input_data['PTRATIO'], input_data['B'],
                    input_data['LSTAT']]]
    
    previsao = modelo.predict(feature_list)[0]
    
    return {'Previsão': previsao}
    

