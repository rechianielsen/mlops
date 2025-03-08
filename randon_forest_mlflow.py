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
# ** Experimento 2: Random Forest **
# ------------------------------

print("\nIniciando experimento com Random Forest...")
with mlflow.start_run() as run:
    # Registrando parâmetros e métrica
    mlflow.log_param("model_type", "RandomForest")

    # Criando e treinando o modelo Random Forest
    model_rf = RandomForestRegressor(n_estimators=50, random_state=42)
    print("Treinando o modelo de Random Forest...")
    model_rf.fit(X_train, y_train.ravel())

    # Predições
    y_pred_rf = model_rf.predict(X_test)

    # Calculando erro
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    print(f"Erro Quadrático Médio (MSE) do Random Forest: {mse_rf:.4f}")

    # Registrando a métrica no MLflow
    mlflow.log_metric("mse", mse_rf)

    # Registrando o modelo no MLflow
    mlflow.sklearn.log_model(model_rf, "random_forest_model")
    print("Modelo Random Forest registrado no MLflow!")