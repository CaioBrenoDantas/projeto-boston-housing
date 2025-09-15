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

print(df.isnull().sum())
preencher_null = {'CRIM':df['CRIM'].median(),'ZN': df['ZN'].median()
                  ,'INDUS': df['INDUS'].median(),'CHAS': df['CHAS'].median()
                  ,'AGE': df['AGE'].median(),'LSTAT': df['LSTAT'].median()}
df = df.fillna(preencher_null)
print(df.isnull().sum())

correlacao = df.corr(method='pearson')
plt.figure(figsize=(12,10))
sns.heatmap(correlacao,annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Matriz de correlação das variáveis')
plt.show()

X = df.drop('MEDV',axis=1)
y = df['MEDV']
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