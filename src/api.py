import joblib
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import pandas as pd
from pydantic import BaseModel
import seaborn as sns
import matplotlib.pyplot as plt
import io

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
    LSTAT: float

try :    
    modelo =joblib.load('../models/trained_model.pkl')
except FileNotFoundError:
    modelo =joblib.load('models/trained_model.pkl')
    
    
try:
    caminho_arquivo = '../data/processed/HousingData.csv'
    df = pd.read_csv(caminho_arquivo)
except FileNotFoundError:
    caminho_arquivo = 'data/processed/HousingData.csv'
    df = pd.read_csv(caminho_arquivo)

@app.get('/')
def ola_mundo():
    return 'Olá mundo'

@app.post('/predict')
async def predict_house_price(data:house_features):
    
    input_data = data.model_dump()
    
    feature_list = [[input_data['CRIM'], input_data['ZN'], input_data['INDUS'], input_data['CHAS'],
                    input_data['NOX'], input_data['RM'], input_data['AGE'], input_data['DIS'],
                    input_data['RAD'], input_data['TAX'], input_data['PTRATIO'],input_data['LSTAT']]]
    
    previsao = modelo.predict(feature_list)[0]
    
    return {'Previsão(milhares de dólares)': round(previsao,5)}

@app.get('/info')
async def get_data_info():
    insights = df.describe().to_dict()
    return insights
    
@app.get('/correlation')
async def get_correlation():
    
    correlacao = df.corr(method="pearson")
    plt.figure(figsize=(10,6))
    sns.heatmap(correlacao,annot=True,cmap='coolwarm',fmt='.2f')
    plt.title('Matriz de correlação de Peasorn dos variáveis do Dataframe')
    
    buffer = io.BytesIO()
    plt.savefig(buffer,format='png')
    buffer.seek(0)
    plt.close()
    
    return StreamingResponse(buffer,media_type='image/png')

