import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import shap

# Carregar os dados processados
pacientes = pd.read_csv('pacientes_processados.csv')

# Definindo X (features) e y (target)
X = pacientes[['los', 'exame_media', 'exame_desvio', 'exame_maximo']]
y = np.random.randint(0, 2, size=len(X))  # Placeholder para a variável alvo (0: sobreviveu, 1: faleceu)

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de regressão logística
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Avaliar o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Explicabilidade com SHAP
explainer = shap.Explainer(modelo, X_test)
shap_values = explainer(X_test)

# Visualizar a importância das features
shap.plots.beeswarm(shap_values)

# Salvar o modelo
import joblib
joblib.dump(modelo, 'modelo_regressao_logistica_uti.pkl')

print("✅ Modelo treinado e salvo como 'modelo_regressao_logistica_uti.pkl'")