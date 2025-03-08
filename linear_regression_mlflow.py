import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Configuração do MLflow com URI personalizada para o banco de dados
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Criando e registrando o experimento no MLflow
experiment_name = "mlops_intro"
mlflow.set_experiment(experiment_name)
print(f"Experiment '{experiment_name}' criado ou carregado.")

# Gerando dados sintéticos
print("Gerando dados sintéticos...")
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Variável independente
y = 2.5 * X + np.random.randn(100, 1) * 2  # Variável dependente com ruído

# Exibindo um resumo dos dados
print(f"Primeiros 5 exemplos de X:\n{X[:5]}")
print(f"Primeiros 5 exemplos de y:\n{y[:5]}")

# Separando em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dados divididos em treino e teste.")

# ------------------------------
# ** Experimento 1: Regressão Linear **
# ------------------------------

print("\nIniciando experimento com Regressão Linear...")
with mlflow.start_run() as run:
    # Registrando parâmetros e métrica
    mlflow.log_param("model_type", "LinearRegression")

    # Criando e treinando o modelo
    model = LinearRegression()
    print("Treinando o modelo de Regressão Linear...")
    model.fit(X_train, y_train)

    # Predições
    y_pred = model.predict(X_test)

    # Calculando o erro
    mse = mean_squared_error(y_test, y_pred)
    print(f"Erro Quadrático Médio (MSE) da Regressão Linear: {mse:.4f}")

    # Registrando a métrica no MLflow
    mlflow.log_metric("mse", mse)

    # Registrando o modelo no MLflow
    mlflow.sklearn.log_model(model, "linear_regression_model")
    print("Modelo Linear Regression registrado no MLflow!")
