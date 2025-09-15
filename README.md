Título: Projeto Boston Housing
Descrição: Este projeto consiste em uma API de machine learning para previsão do preço de casas, 
utilizando um modelo de regressão treinado com o dataset Boston Housing. 
A API foi desenvolvida com FastAPI para permitir que as previsões sejam acessadas e utilizadas por outras aplicações.

Tecnologias:

Python: Linguagem de programação.

pandas: Análise e manipulação de dados.

scikit-learn: Treinamento do modelo de regressão.

seaborn: Carregamento do dataset.

FastAPI: Criação da API.

Uvicorn: Servidor da API.

Conda: Gerenciamento do ambiente.

Estrutura de Pastas
    /projeto-boston-housing
    ├── data/
    │    └── processed
    │    └── raw
    │       └── boston_housing.csv
    ├── src/
    │   ├── model_training.py
    │   └── api.py
    ├── models/
    │   └── trained_model.pkl
    ├── environment.yml
    └── README.md

1.0 - Pré-requisitos: Certifique-se de ter o Conda instalado na sua máquina.

2.0 - Crie e ative o ambiente virtual:
2.1 - Navegue até a pasta do projeto e execute o comando para criar o ambiente a partir do arquivo environment.yml:

conda env create -f environment.yml
conda activate projeto-boston-housing

3.0 - Treine o modelo:
3.1 - Execute o script Python responsável por treinar e salvar o modelo de regressão.

python src/model_training.py

4.0 - Inicie a API:
4.1 - Execute a aplicação FastAPI usando o Uvicorn. O argumento --reload garante que a API reinicie automaticamente ao detectar alterações no código.

uvicorn src.api:app --reload

Afim de facilitar o entendimento há abaixo uma lista que descreve o signficado de cada sigla no Dataframe

- CRIM: Taxa de criminalidade por cidade.

- ZN: Proporção de terrenos residenciais zoneados para lotes com mais de 25.000 pés quadrados.

- INDUS: Proporção de acres de negócios não-varejo por cidade.

- CHAS: Variável dummy do Charles River (1 se o terreno margeia o rio; 0 caso contrário).

- NOX: Concentração de óxidos nítricos (partes por 10 milhões).

- RM: Número médio de quartos por residência.

- AGE: Proporção de unidades ocupadas por proprietários construídas antes de 1940.

- DIS: Distâncias ponderadas até cinco centros de empregos em Boston.

- RAD: Índice de acessibilidade a rodovias radiais.

- TAX: Taxa de imposto sobre a propriedade de valor total por 10.000 dólares.

- PTRATIO: Relação aluno-professor por cidade.

- B: 1000(Bk - 0.63)^2, onde Bk é a proporção de negros por cidade. (Essa é a variável com questões éticas).

- LSTAT: % de status inferior da população.

- MEDV: Valor médio das casas ocupadas pelos proprietários em 1.000 dólares(Var Alvo)
