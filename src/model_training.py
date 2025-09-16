import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics    import mean_absolute_error
import joblib

try:
    caminho_arquivo = '../data/raw/HousingData.csv'
    df = pd.read_csv(caminho_arquivo)
except FileNotFoundError:
    caminho_arquivo = 'data/raw/HousingData.csv'
    df = pd.read_csv(caminho_arquivo)
# A variável 'B' foi removida do modelo por questões éticas
df = df.drop('B',axis=1)
print(df.isnull().sum())
df_limpo = df.fillna(df.median(numeric_only=True))
try:
    caminho_arquivo ='../data/processed/HousingData.csv'
    df_limpo.to_csv(caminho_arquivo,index=False)
except:
    caminho_arquivo ='data/processed/HousingData.csv'
    df_limpo.to_csv(caminho_arquivo,index=False)
print(df_limpo.isnull().sum())

correlacao = df_limpo.corr(method='pearson')
plt.figure(figsize=(12,10))
sns.heatmap(correlacao,annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Matriz de correlação das variáveis')
plt.show()

X = df_limpo.drop('MEDV',axis=1)
y = df_limpo['MEDV']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

modelo = LinearRegression()

modelo.fit(X=X_train,y=y_train)

predicao = modelo.predict(X_test)

media_erro =mean_absolute_error(y_true=y_test,y_pred=predicao)
print(media_erro)

try:
    joblib.dump(modelo,'../models/trained_model.pkl')
except FileNotFoundError:
    joblib.dump(modelo,'models/trained_model.pkl')