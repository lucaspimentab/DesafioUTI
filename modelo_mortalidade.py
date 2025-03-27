import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
import joblib

# 📌 Carregar dataset
dataset_uti = pd.read_csv("dataset_UTI_completo.csv")

# 📌 Converter variável categórica 'gender' em numérica
label_encoder = LabelEncoder()
dataset_uti['gender'] = label_encoder.fit_transform(dataset_uti['gender'])

# 📌 Agrupar dados por paciente
dataset_uti_static = dataset_uti.groupby('subject_id').mean().reset_index()

# 📌 Definir variável-alvo e variáveis preditoras
y = dataset_uti_static['mortalidade_total']
X = dataset_uti_static.drop(columns=['subject_id', 'mortalidade_total'])

# 📌 Tratar valores ausentes
X.fillna(X.mean(), inplace=True)

# 📌 Normalização
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 📌 Divisão treino-teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 📌 Modelo Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

# 📌 Validação cruzada
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='f1')
print(f"\n🔍 F1-score médio na validação cruzada: {np.mean(cv_scores):.4f}")

# 📌 Treinamento do modelo final
rf_model.fit(X_train, y_train)

# 📌 Avaliação no conjunto de teste
y_pred = rf_model.predict(X_test)
print("\n📊 Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\n📌 Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# 📌 Salvar modelo treinado
joblib.dump(rf_model, "modelo_rf_mortalidade_total.pkl")
print("\n💾 Modelo salvo como 'modelo_rf_mortalidade_total.pkl'")

# 📌 Explicabilidade com SHAP
print("\n🔍 Gerando explicabilidade com SHAP...")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Criar DataFrame com os dados de teste
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# Se shap_values for uma lista (binário), usamos o índice 1 (classe positiva)
if isinstance(shap_values, list):
    shap_values_used = shap_values[1]
else:
    shap_values_used = shap_values

# Se tiver mais de 1 dimensão (às vezes 3D), fazer flatten
if shap_values_used.ndim > 2:
    shap_values_used = shap_values_used[:, :, 0]

# Calcular impacto médio
impacto_medio = np.mean(np.abs(shap_values_used), axis=0)

# Garantir que seja 1D
impacto_medio = impacto_medio.ravel()

# Construir DataFrame de importância
shap_df = pd.DataFrame({
    'variavel': X_test_df.columns,
    'impacto_medio': impacto_medio
}).sort_values(by='impacto_medio', ascending=False)

print("\n📈 Variáveis mais importantes no modelo:")
print(shap_df)

# 📌 Gráfico SHAP
shap.summary_plot(shap_values_used, X_test_df, show=False)
plt.tight_layout()
plt.savefig("shap_importancia_variaveis.png", dpi=300)
plt.close()
print("\n📊 Gráfico SHAP salvo como 'shap_importancia_variaveis.png'")