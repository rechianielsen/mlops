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


print("\nIniciando experimento com Random Forest e diferentes hiperparâmetros...")
for n in [10, 50, 100, 200]:
    with mlflow.start_run() as run:
        # Registrando parâmetros
        mlflow.log_param("n_estimators", n)
        print(f"Treinando Random Forest com {n} árvores...")

        # Criando e treinando o modelo
        model_rf_hp = RandomForestRegressor(n_estimators=n, random_state=42)
        model_rf_hp.fit(X_train, y_train.ravel())

        # Predições
        y_pred_rf_hp = model_rf_hp.predict(X_test)

        # Calculando erro
        mse_rf_hp = mean_squared_error(y_test, y_pred_rf_hp)
        print(f"Erro Quadrático Médio (MSE) do Random Forest com {n} árvores: {mse_rf_hp:.4f}")

        # Registrando a métrica no MLflow
        mlflow.log_metric("mse", mse_rf_hp)

        # Registrando o modelo no MLflow
        mlflow.sklearn.log_model(model_rf_hp, f"random_forest_{n}_model")
        print(f"Modelo Random Forest com {n} árvores registrado no MLflow!")

# Resumo
print("\nExperimentos finalizados e registrados no MLflow!")